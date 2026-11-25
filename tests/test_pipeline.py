import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yk_evidence import pipeline
from yk_evidence.pipeline import grounded_memo, integrated_run, run, verify


class PipelineTest(unittest.TestCase):
    def test_installed_source_without_git_has_a_reproducible_revision(self):
        with tempfile.TemporaryDirectory() as root:
            original_root = pipeline.ROOT
            try:
                pipeline.ROOT = Path(root)
                bundle = integrated_run(Path(root) / "runs", "no-git-001", seed=42)
            finally:
                pipeline.ROOT = original_root
            record = json.loads((bundle / "run_record.json").read_text())
            self.assertEqual("source-sha256:" + record["code_sha256"], record["code_revision"])
            self.assertEqual([], verify(bundle))

    def test_git_executable_is_not_required_for_an_installed_run(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"PATH": ""}):
            bundle = integrated_run(Path(root), "no-git-command", seed=42)
            record = json.loads((bundle / "run_record.json").read_text())
            self.assertEqual("source-sha256:" + record["code_sha256"], record["code_revision"])
            self.assertEqual([], verify(bundle))

    def test_run_writes_all_layer_outputs_and_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = run(Path(root), "demo-001")
            required = {
                "source_snapshot.txt", "transactions_snapshot.csv", "finance_memo.json",
                "risk_output.json", "run_record.json", "stress_test.json",
                "manifest.json", "summary.md",
            }
            self.assertEqual(required, {p.name for p in bundle.iterdir()})
            self.assertEqual([], verify(bundle))
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual("synthetic_demo", manifest["source_classification"])
            self.assertEqual(required - {"manifest.json"}, set(manifest["artifacts"]))
            memo = json.loads((bundle / "finance_memo.json").read_text())
            self.assertTrue(memo["citations"])
            risk = json.loads((bundle / "risk_output.json").read_text())
            self.assertTrue(any(x["related_entities"] for x in risk["findings"]))

    def test_tamper_and_missing_file_are_detected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = run(Path(root), "demo-001")
            (bundle / "summary.md").write_text("changed")
            (bundle / "risk_output.json").unlink()
            errors = verify(bundle)
            self.assertTrue(any("hash mismatch" in e for e in errors))
            self.assertTrue(any("missing" in e for e in errors))

    def test_manifest_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = run(Path(root), "demo-001")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["../outside"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            self.assertTrue(any("unsafe path" in e for e in verify(bundle)))

    def test_unsupported_finance_question_stays_unanswered(self):
        memo = grounded_memo("Synthetic revenue was 100 units.", "What was profit?")
        self.assertEqual("no_supported_answer", memo["status"])
        self.assertEqual([], memo["citations"])

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            run(Path(root), "demo-001")
            with self.assertRaises(FileExistsError):
                run(Path(root), "demo-001")

    def test_integrated_run_writes_public_and_synthetic_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "integrated-001", seed=42)
            self.assertEqual([], verify(bundle))
            self.assertEqual({
                "financial_snapshot.json", "finance_memo.json", "synthetic_transactions.json",
                "benchmark.json", "validation.json", "stress_replay.json", "governance.json", "run_record.json",
                "data_card.md", "model_card.md", "summary.md", "manifest.json",
            }, {p.name for p in bundle.iterdir()})
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual("0.2", manifest["schema_version"])
            self.assertEqual("public_sec_filing_and_synthetic_benchmark", manifest["source_classification"])
            memo = json.loads((bundle / "finance_memo.json").read_text())
            self.assertEqual(36602000000, memo["increase_usd"])
            benchmark = json.loads((bundle / "benchmark.json").read_text())
            self.assertEqual(4, len(benchmark["models"]))
            self.assertEqual(100, len(benchmark["models"]["rule"]["predictions"]))
            validation = json.loads((bundle / "validation.json").read_text())
            self.assertEqual("synthetic_only", validation["external_validity"])
            self.assertEqual(5, len(validation["stability_runs"]))
            replay = json.loads((bundle / "stress_replay.json").read_text())
            self.assertEqual({"amount_scale_1_2", "ring_link_suppression"}, set(replay["scenarios"]))

    def test_integrated_verifier_detects_tampered_benchmark(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "integrated-001", seed=42)
            (bundle / "benchmark.json").write_text("{}")
            self.assertTrue(any("hash mismatch: benchmark.json" in error for error in verify(bundle)))

    def test_integrated_verifier_rejects_rehashed_run_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "integrated-001", seed=42)
            record_path = bundle / "run_record.json"
            record = json.loads(record_path.read_text())
            record["run_id"] = "a-different-run"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["run_record.json"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("run_id mismatch" in error for error in verify(bundle)))

    def test_integrated_verifier_rejects_rehashed_result_that_breaks_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "integrated-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            benchmark["sample_count"] += 1
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["benchmark.json"] = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("result hash mismatch: benchmark.json" in error for error in verify(bundle)))

    def test_verifier_rejects_unknown_manifest_schema(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "integrated-001", seed=42)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = "unrecognized"
            manifest_path.write_text(json.dumps(manifest))
            self.assertTrue(any("unsupported schema" in error for error in verify(bundle)))


if __name__ == "__main__":
    unittest.main()
