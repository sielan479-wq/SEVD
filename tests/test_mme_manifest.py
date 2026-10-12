import tempfile
import unittest
from pathlib import Path

from evaluation.mme_manifest import TASKS, build_mme_rows


class MMEManifestTests(unittest.TestCase):
    def test_flat_and_nested_task_layouts_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in TASKS:
                task_root = root / task
                if task in {"posters", "celebrity", "scene", "landmark", "artwork"}:
                    question_root = task_root / "questions_answers_YN"
                    image_root = task_root / "images"
                else:
                    question_root = image_root = task_root
                question_root.mkdir(parents=True, exist_ok=True)
                image_root.mkdir(parents=True, exist_ok=True)
                (question_root / "0001.txt").write_text(
                    "Is it visible?\tYes\nIs it absent?\tNo\n", encoding="utf-8"
                )
                (image_root / "0001.jpg").write_bytes(b"test")

            rows = build_mme_rows(root)
            self.assertEqual(len(rows), 2 * len(TASKS))
            self.assertEqual(rows[0]["sample_id"], "existence:00001")
            self.assertEqual(rows[1]["ground_truth"], "no")
            poster = next(row for row in rows if row["task"] == "posters")
            self.assertEqual(poster["image_path"], "posters/images/0001.jpg")


if __name__ == "__main__":
    unittest.main()
