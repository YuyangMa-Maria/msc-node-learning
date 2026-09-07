"""Check that every external dataset has reproducible source metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from nlrisk.data.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


class DatasetRegistryTests(unittest.TestCase):
    def test_all_datasets_have_source_and_licence_metadata(self) -> None:
        dataset_root, entries = load_registry(ROOT)
        self.assertEqual(dataset_root, "dataset")
        self.assertEqual(len(entries), 5)
        for entry in entries:
            with self.subTest(dataset=entry.id):
                self.assertTrue(entry.source_url.startswith("https://"))
                self.assertTrue(entry.version)
                self.assertTrue(entry.citation)
                self.assertTrue(entry.access_notes)
                self.assertTrue(entry.licence.identifier)
                self.assertTrue(entry.licence.url.startswith("https://"))
                self.assertTrue(entry.subset_used)
                self.assertTrue(entry.expected_layout)
                self.assertFalse(Path(entry.path).is_absolute())

    def test_all_citation_keys_are_defined(self) -> None:
        _, entries = load_registry(ROOT)
        bib = (ROOT / "docs" / "DATASET_CITATIONS.bib").read_text(encoding="utf-8")
        defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
        requested = {key for entry in entries for key in entry.citation}
        self.assertEqual(requested - defined, set())

    def test_formal_split_sets_are_present(self) -> None:
        split_root = ROOT / "reproducibility" / "splits"
        for name in ("conference_grouped", "codebrim", "sdnet2018", "vbn_orion"):
            with self.subTest(split_set=name):
                files = [path for path in (split_root / name).iterdir() if path.is_file()]
                self.assertTrue(files)


if __name__ == "__main__":
    unittest.main()
