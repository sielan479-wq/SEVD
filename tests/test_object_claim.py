import tempfile
import unittest
from pathlib import Path

from sccd.object_claim import object_from_existence_question
from sccd.vocabulary import ObjectVocabulary


class ObjectExistenceQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        synonyms = Path(self.temp_dir.name) / "synonyms.txt"
        synonyms.write_text(
            "train,trains\nbaseball bat,baseball bats\nhair drier,hair driers\n",
            encoding="utf-8",
        )
        self.vocab = ObjectVocabulary.from_synonyms_file(synonyms)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parses_pope_template(self) -> None:
        claim = object_from_existence_question(
            "Is there a train in the image?", self.vocab
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim.canonical_object, "train")

    def test_parses_mme_template_without_instruction_leakage(self) -> None:
        claim = object_from_existence_question(
            "Is there a baseball bat in this image? Please answer yes or no.",
            self.vocab,
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim.canonical_object, "baseball bat")

    def test_rejects_non_existence_semantics(self) -> None:
        claim = object_from_existence_question(
            "Is the train on the left side of the picture? Please answer yes or no.",
            self.vocab,
        )
        self.assertIsNone(claim)


if __name__ == "__main__":
    unittest.main()
