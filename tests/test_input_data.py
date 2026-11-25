import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from yk_evidence.input_data import load_synthetic_csv
from yk_evidence.pipeline import contract_run, verify
from yk_evidence.risk_data import generate_transactions


class ImportedInputTest(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path]:
        csv_path = root / "operator_input.csv"
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["transaction_id", "timestamp_utc", "customer_id",
                                                        "merchant_id", "amount", "label"])
            writer.writeheader()
            for index, row in enumerate(generate_transactions(seed=43, count=400)):
                writer.writerow({key: row[key] for key in ("transaction_id", "customer_id", "merchant_id", "amount", "label")} |
                                {"timestamp_utc": (start + timedelta(minutes=index)).isoformat()})
        metadata_path = root / "operator_metadata.json"
        metadata_path.write_text(json.dumps({
            "source_name": "Project-generated importer test fixture", "source_url": "local:test-fixture",
            "dataset_version": "test-1", "license_id": "project-generated-test-only",
            "label_definition": "Every nineteenth row is an injected pattern, for test only",
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "classification": "synthetic_private", "redistribution_permission": "private_local_only",
        }), encoding="utf-8")
        return csv_path, metadata_path

    def test_imported_csv_runs_same_evidence_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            csv_path, metadata_path = self.make_inputs(root)
            bundle = contract_run(root / "runs", "imported", seed=43,
                                  transactions_csv=csv_path, transactions_metadata=metadata_path)
            self.assertEqual([], verify(bundle))
            self.assertEqual("external_synthetic_benchmark", json.loads((bundle / "benchmark.json").read_text())["classification"])
            self.assertEqual([], json.loads((bundle / "validation.json").read_text())["stability_runs"])
            self.assertEqual(400, json.loads((bundle / "import_profile.json").read_text())["row_count"])
            self.assertIn("transaction_source.csv", json.loads((bundle / "bundle_manifest.json").read_text())["artifacts"])
            self.assertEqual("blocked", json.loads((bundle / "release_gate.json").read_text())["status"])

    def test_wrong_hash_and_duplicate_id_fail_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            csv_path, metadata_path = self.make_inputs(root)
            metadata = json.loads(metadata_path.read_text())
            metadata["csv_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "hash"):
                contract_run(root / "runs", "bad-hash", transactions_csv=csv_path, transactions_metadata=metadata_path)
            self.assertFalse((root / "runs" / "bad-hash").exists())
            metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["transaction_id"] = rows[0]["transaction_id"]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata["csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_synthetic_csv(csv_path, metadata_path)


if __name__ == "__main__":
    unittest.main()
