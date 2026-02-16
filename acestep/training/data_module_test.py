"""Tests for path-sanitisation in data_module.

Covers the safe_path integration in PreprocessedTensorDataset and
load_dataset_from_json that guards against path-traversal attacks
(CodeQL: uncontrolled data used in path expression).
"""

import os
import json
import tempfile
import unittest

from acestep.training.path_safety import safe_path, set_safe_root
from acestep.training.data_module import (
    PreprocessedTensorDataset,
    load_dataset_from_json,
)


class SafePathTests(unittest.TestCase):
    """Tests for safe_path from path_safety module."""

    def test_valid_directory(self):
        with tempfile.TemporaryDirectory() as d:
            parent = os.path.dirname(os.path.realpath(d))
            set_safe_root(parent)
            result = safe_path(d)
            self.assertEqual(result, os.path.realpath(d))

    def test_traversal_raises(self):
        with tempfile.TemporaryDirectory() as d:
            set_safe_root(d)
            with self.assertRaises(ValueError):
                safe_path("../../etc/passwd", base=d)

    def test_absolute_path_outside_raises(self):
        with tempfile.TemporaryDirectory() as d:
            set_safe_root(d)
            with self.assertRaises(ValueError):
                safe_path("/etc/passwd", base=d)

    def test_normal_child(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.realpath(d)
            result = safe_path("foo.pt", base=base)
            self.assertEqual(result, os.path.join(base, "foo.pt"))

    def test_absolute_path_inside_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.realpath(d)
            child = os.path.join(base, "sub", "file.pt")
            result = safe_path(child, base=base)
            self.assertEqual(result, child)


class PreprocessedTensorDatasetPathSafetyTests(unittest.TestCase):
    """Tests that PreprocessedTensorDataset rejects traversal paths."""

    def setUp(self):
        # Allow /tmp paths during tests
        set_safe_root(tempfile.gettempdir())

    def test_manifest_traversal_paths_skipped(self):
        """Paths in manifest.json that escape tensor_dir are ignored."""
        with tempfile.TemporaryDirectory() as d:
            # Create a manifest with one good and one bad path
            good_pt = os.path.join(d, "good.pt")
            open(good_pt, "wb").close()  # touch

            manifest = {
                "samples": [
                    "good.pt",
                    "../../etc/passwd",
                ]
            }
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f)

            ds = PreprocessedTensorDataset(d)
            # Only the safe path should survive
            self.assertEqual(len(ds.valid_paths), 1)
            self.assertTrue(ds.valid_paths[0].endswith("good.pt"))

    def test_fallback_scan_only_finds_pt_files(self):
        """Without manifest, only .pt files inside tensor_dir are found."""
        with tempfile.TemporaryDirectory() as d:
            for name in ["a.pt", "b.pt", "c.txt"]:
                open(os.path.join(d, name), "wb").close()

            ds = PreprocessedTensorDataset(d)
            self.assertEqual(len(ds.valid_paths), 2)

    def test_nonexistent_dir_raises(self):
        with self.assertRaises(ValueError):
            PreprocessedTensorDataset("/tmp/nonexistent_xyz_12345")


class LoadDatasetFromJsonTests(unittest.TestCase):
    """Tests for load_dataset_from_json path validation."""

    def setUp(self):
        set_safe_root(tempfile.gettempdir())

    def test_nonexistent_file_raises(self):
        with self.assertRaises(ValueError):
            load_dataset_from_json("/tmp/nonexistent_file.json")

    def test_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metadata": {"v": 1}, "samples": [{"a": 1}]}, f)
            path = f.name
        try:
            samples, meta = load_dataset_from_json(path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(meta["v"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
