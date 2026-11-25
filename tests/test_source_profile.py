"""Typed-source failures must be caught before a long model run."""
import csv
import tempfile
import unittest
from pathlib import Path


class SourceProfileTests(unittest.TestCase):
    def fixture(self, root, rows):
        path = Path(root) / "credit.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])
            writer.writerows(rows)
        return path

    def test_profiles_zero_amount_without_inventing_calendar_time(self):
        from yk_evidence.source_profile import profile_ulb
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root, [[0] + [0] * 28 + [0, 0], [10] + [1] * 28 + [50, 1]])
            result = profile_ulb(path)
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["positive_count"], 1)
            self.assertEqual(result["zero_amount_count"], 1)
            self.assertEqual(result["time_range_seconds"], [0.0, 10.0])
            self.assertEqual(len(result["source_sha256"]), 64)

    def test_rejects_nonfinite_features(self):
        from yk_evidence.source_profile import profile_ulb
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root, [[0] + [float("nan")] + [0] * 27 + [50, 0]])
            with self.assertRaises(ValueError):
                profile_ulb(path)

    def test_rejects_nonbinary_label(self):
        from yk_evidence.source_profile import profile_ulb
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root, [[0] + [0] * 28 + [50, 2]])
            with self.assertRaises(ValueError):
                profile_ulb(path)

    def test_rejects_empty_dataset(self):
        from yk_evidence.source_profile import profile_ulb
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                profile_ulb(self.fixture(root, []))
