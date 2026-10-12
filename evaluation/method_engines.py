"""Method-faithful LLaVA adapters sharing SCCD's canonical evaluation contract."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class _Stats:
    generated_token_count: int = 0
    changed_step_count: int = 0
    method_forward_count: int = 0
    runtime_seconds: float = 0.0
    exception_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineEngine:
    """Greedy, VCD, ICD, and RITUAL generation on the common LLaVA checkpoint.

    The adapters preserve each paper's logit combination and plausibility cutoff,
    while using deterministic argmax selection so every method shares SCCD's
    reproducible, append-safe resume semantics.
    """

    def __init__(self, config: dict[str, Any], method: str, components: Any) -> None:
        self.config = config
        self.method = method
        self.method_config = dict(config[method])
        self.tokenizer, self.model, self.image_processor, self.tokenizer_image_token = components

    def answer_pope_sample(self, row: dict[str, Any]):
        prompt = f"USER: <image>\\n{row['question'].strip()}\\nPlease answer yes or no.\\nASSISTANT:"
        return self._generate(row, prompt, 8, "pope")

    def answer_object_existence_sample(self, row: dict[str, Any], benchmark: str):
        prompt = f"USER: <image>\\n{row['question'].strip()}\\nASSISTANT:"
        return self._generate(row, prompt, 8, benchmark)

    def generate_chair_sample(self, row: dict[str, Any], prompt: str, max_new_tokens: int, benchmark: str = "chair"):
        return self._generate(row, prompt, max_new_tokens, benchmark)

    def _generate(self, row: dict[str, Any], prompt: str, max_new_tokens: int, benchmark: str):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX

        started = time.time()
        stats = _Stats()
        sample_id = str(row.get("sample_id", row.get("image_id", "unknown")))
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        image = Image.open(str(row["image_path"])).convert("RGB")
        image_tensor = self._tensor(image)
        aux_tensor, aux_input_ids, aux_meta = self._auxiliary(sample_id, image, image_tensor, prompt)
        generated: list[int] = []
        traces: list[dict[str, Any]] = []

        try:
            with torch.inference_mode():
                for step in range(int(max_new_tokens)):
                    original = self._next_logits(input_ids, image_tensor)
                    stats.method_forward_count += 1
                    selected_before = int(torch.argmax(original).item())
                    adjusted = original
                    if self.method == "vcd":
                        auxiliary = self._next_logits(input_ids, aux_tensor)
                        stats.method_forward_count += 1
                        adjusted = self._contrast(original, auxiliary, float(self.method_config["alpha"]), float(self.method_config["beta"]), sign=-1)
                    elif self.method == "ritual":
                        auxiliary = self._next_logits(input_ids, aux_tensor)
                        stats.method_forward_count += 1
                        adjusted = self._contrast(original, auxiliary, float(self.method_config["alpha_pos"]), float(self.method_config["beta"]), sign=1)
                    elif self.method == "icd":
                        auxiliary = self._next_logits(aux_input_ids, image_tensor)
                        stats.method_forward_count += 1
                        adjusted = self._contrast(original, auxiliary, float(self.method_config["alpha"]), float(self.method_config["beta"]), sign=-1)
                    selected_after = int(torch.argmax(adjusted).item())
                    changed = selected_after != selected_before
                    stats.changed_step_count += int(changed)
                    if self.method != "greedy":
                        traces.append({
                            "sample_id": sample_id,
                            "benchmark": benchmark,
                            "step": step,
                            "action": f"{self.method}_logit_combination",
                            "delta": float((adjusted[selected_after] - original[selected_after]).item()),
                            "selected_token_before": selected_before,
                            "selected_token_after": selected_after,
                            "output_changed": changed,
                            **aux_meta,
                        })
                    generated.append(selected_after)
                    next_token = torch.tensor([[selected_after]], dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    if aux_input_ids is not None:
                        aux_input_ids = torch.cat([aux_input_ids, next_token], dim=1)
                    if selected_after == self.tokenizer.eos_token_id:
                        break
        except Exception:
            stats.exception_count += 1
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stats.generated_token_count = len(generated)
        stats.runtime_seconds = time.time() - started
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), traces, stats.to_dict()

    def _next_logits(self, input_ids: Any, image_tensor: Any):
        return self.model(input_ids=input_ids, images=image_tensor, use_cache=False, return_dict=True).logits[:, -1, :].float().squeeze(0)

    @staticmethod
    def _contrast(original: Any, auxiliary: Any, alpha: float, beta: float, sign: int):
        import torch
        cutoff = math.log(beta) + original.max(dim=-1, keepdim=True).values
        combined = (1.0 + alpha) * original + sign * alpha * auxiliary if sign < 0 else original + alpha * auxiliary
        return combined.masked_fill(original < cutoff, -torch.inf)

    def _tensor(self, image: Image.Image):
        return self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).half().cuda()

    def _auxiliary(self, sample_id: str, image: Image.Image, image_tensor: Any, prompt: str):
        if self.method == "greedy":
            return None, None, {"auxiliary": "none"}
        if self.method == "vcd":
            return self._diffusion_noise(image_tensor, sample_id), None, {
                "auxiliary": "diffusion_noise", "noise_step": int(self.method_config["noise_step"])
            }
        if self.method == "ritual":
            name, transformed = self._ritual_transform(image, sample_id)
            return self._tensor(transformed), None, {"auxiliary": "image_transform", "transform": name}
        if self.method == "icd":
            from llava.constants import IMAGE_TOKEN_INDEX
            disturbance = str(self.method_config["disturbance_instruction"])
            prompt_cd = f"SYSTEM: {disturbance}\\n{prompt}"
            ids = self.tokenizer_image_token(prompt_cd, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            return None, ids, {"auxiliary": "instruction_disturbance", "disturbance_instruction": disturbance}
        raise ValueError(f"Unsupported method: {self.method}")

    def _seed(self, sample_id: str) -> int:
        raw = f"{self.method_config.get('seed', 55)}:{sample_id}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**31 - 1)

    def _diffusion_noise(self, image_tensor: Any, sample_id: str):
        import torch
        steps = 1000
        beta = torch.linspace(-6, 6, steps, device=image_tensor.device)
        beta = torch.sigmoid(beta) * (0.5e-2 - 1e-5) + 1e-5
        alpha_bar = torch.cumprod(1 - beta, dim=0)
        t = int(self.method_config["noise_step"])
        generator = torch.Generator(device=image_tensor.device).manual_seed(self._seed(sample_id))
        noise = torch.randn(image_tensor.shape, generator=generator, device=image_tensor.device, dtype=image_tensor.dtype)
        return alpha_bar[t].sqrt() * image_tensor + (1 - alpha_bar[t]).sqrt() * noise

    def _ritual_transform(self, image: Image.Image, sample_id: str) -> tuple[str, Image.Image]:
        choices = ("horizontal_flip", "vertical_flip", "rotate_180", "color_jitter", "gaussian_blur", "center_crop")
        name = choices[self._seed(sample_id) % len(choices)]
        if name == "horizontal_flip":
            return name, ImageOps.mirror(image)
        if name == "vertical_flip":
            return name, ImageOps.flip(image)
        if name == "rotate_180":
            return name, image.rotate(180)
        if name == "color_jitter":
            return name, ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.15)).enhance(0.85)).enhance(1.1)
        if name == "gaussian_blur":
            return name, image.filter(ImageFilter.GaussianBlur(radius=1.75))
        width, height = image.size
        side = int(min(width, height) * 0.85)
        left, top = (width - side) // 2, (height - side) // 2
        return name, image.crop((left, top, left + side, top + side)).resize(image.size)
