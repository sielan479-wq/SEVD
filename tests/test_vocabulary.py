import tempfile
import unittest
from pathlib import Path

from sccd.vocabulary import ObjectVocabulary


class VocabularyTests(unittest.TestCase):
    def test_object_synonym_and_plural_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            synonyms = Path(temp_dir) / "synonyms.txt"
            synonyms.write_text(
                "dog,dogs,puppy\ntraffic light,traffic lights\n",
                encoding="utf-8",
            )
            vocab = ObjectVocabulary.from_synonyms_file(synonyms)
            self.assertEqual(vocab.canonicalize_token(" dogs"), "dog")
            self.assertEqual(
                vocab.canonicalize_phrase("traffic lights"), "traffic light"
            )


if __name__ == "__main__":
    unittest.main()
