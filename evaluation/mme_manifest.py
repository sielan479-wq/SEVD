"""Build the private MME run manifest from an authorized local benchmark copy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


TASKS = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _task_locations(root: Path, task: str) -> tuple[Path, Path]:
    task_root = root / task
    question_root = task_root / "questions_answers_YN"
    image_root = task_root / "images"
    return (
        question_root if question_root.is_dir() else task_root,
        image_root if image_root.is_dir() else task_root,
    )


def _task_image_index(image_root: Path) -> dict[str, Path]:
    """Index only the current task's images to prevent filename collisions."""

    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing MME task directory: {image_root}")
    return {
        path.stem: path
        for path in sorted(image_root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def build_mme_rows(root: Path) -> list[dict[str, object]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MME benchmark root does not exist: {root}")
    rows: list[dict[str, object]] = []
    for task in TASKS:
        question_root, image_root = _task_locations(root, task)
        images = _task_image_index(image_root)
        question_files = sorted(question_root.glob("*.txt"))
        if not question_files:
            raise FileNotFoundError(f"No MME question files found for {task}: {question_root}")
        sample_index = 0
        for pair_index, question_file in enumerate(question_files):
            image = images.get(question_file.stem)
            if image is None:
                raise FileNotFoundError(
                    f"No MME image matches {question_file.relative_to(root)}"
                )
            lines = [
                line.strip()
                for line in question_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for line in lines:
                try:
                    question, answer = line.rsplit("\t", 1)
                except ValueError as error:
                    raise ValueError(
                        f"Expected tab-separated question and answer in {question_file}: {line!r}"
                    ) from error
                answer = answer.strip().lower()
                if answer not in {"yes", "no"}:
                    raise ValueError(
                        f"Unexpected MME answer in {question_file}: {answer!r}"
                    )
                sample_index += 1
                image_relative = image.relative_to(root).as_posix()
                rows.append(
                    {
                        "sample_id": f"{task}:{sample_index:05d}",
                        "task": task,
                        "image_id": image.name,
                        "image_path": image_relative,
                        "question": question.strip(),
                        "ground_truth": answer,
                        "pair_id": f"{task}:{pair_index:05d}:{image.name}",
                        "official_source": f"eval_tool/Your_Results/{task}.txt",
                        "benchmark": "mme",
                    }
                )
    return rows


def write_mme_manifest(rows: Iterable[dict[str, object]], output: Path) -> int:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
