"""Public contract for the complete evidence-bundle export."""

import json
import hashlib
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from yk_evidence import pipeline


class ContractBundleTest(unittest.TestCase):
    def test_rehashed_finance_memo_number_is_recomputed_from_source(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "finance-audit", seed=42)
            target = bundle / "finance_memo.json"
            memo = json.loads(target.read_text())
            memo["increase_usd"] += 1_000_000
            target.write_text(json.dumps(memo), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("finance claim mismatch" in error for error in pipeline.verify(bundle)))

    def test_rehashed_stress_drop_is_recomputed_from_saved_scores(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "stress-audit", seed=42)
            target = bundle / "stress_replay.json"
            replay = json.loads(target.read_text())
            replay["scenarios"]["ring_link_suppression"]["observed_pr_auc_drop"] = 0.0
            target.write_text(json.dumps(replay), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("stress metrics mismatch" in error for error in pipeline.verify(bundle)))

    def test_schema_is_packaged_and_rejects_missing_metadata_field(self):
        from yk_evidence.pipeline import asset

        schema_path = asset("schemas/contract_0_3.json")
        self.assertTrue(schema_path.is_file(), "packaged JSON Schema is missing")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "run_metadata.json"
            metadata = json.loads(target.read_text())
            del metadata["utc_time"]
            target.write_text(json.dumps(metadata), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("schema violation" in error and "utc_time" in error
                                for error in pipeline.verify(bundle)))

    def test_schema_rejects_wrong_input_row_count_type(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "input_manifest.json"
            inputs = json.loads(target.read_text())
            inputs["inputs"][1]["row_count"] = "400"
            target.write_text(json.dumps(inputs), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("schema violation" in error and "row_count" in error
                                for error in pipeline.verify(bundle)))

    def _rehash(self, bundle, name):
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_rehashed_input_manifest_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "input_manifest.json"
            inputs = json.loads(target.read_text())
            inputs["inputs"][0]["sha256"] = "0" * 64
            target.write_text(json.dumps(inputs), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("input manifest mismatch" in error for error in pipeline.verify(bundle)))

    def test_rehashed_retrieval_answer_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "retrieval_evidence.json"
            retrieval = json.loads(target.read_text())
            retrieval["answer"] = "Unsupported claim."
            target.write_text(json.dumps(retrieval), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("retrieval evidence mismatch" in error for error in pipeline.verify(bundle)))

    def test_rehashed_model_metric_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "metrics.json"
            metrics = json.loads(target.read_text())
            model = next(iter(metrics["models"]))
            metrics["models"][model]["pr_auc"] = 0.0
            target.write_text(json.dumps(metrics), encoding="utf-8")
            self._rehash(bundle, target.name)
            self.assertTrue(any("metrics mismatch" in error for error in pipeline.verify(bundle)))

    def test_coordinated_metric_tampering_is_rejected_from_raw_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            benchmark["models"]["rule"]["metrics"]["pr_auc"] = 0.99
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            metrics_path = bundle / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["models"]["rule"]["pr_auc"] = 0.99
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self._rehash(bundle, metrics_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("recomputed metric mismatch" in error for error in errors), errors)

    def test_rehashed_prediction_count_is_checked_against_raw_scores(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "model_metadata.json"
            metadata = json.loads(target.read_text())
            metadata["models"]["rule"]["prediction_count"] = 999
            target.write_text(json.dumps(metadata), encoding="utf-8")
            self._rehash(bundle, target.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("prediction count mismatch" in error for error in errors), errors)

    def test_coordinated_training_id_substitution_is_not_a_holdout_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            source = json.loads((bundle / "synthetic_transactions.json").read_text())
            labels_by_id = {row["transaction_id"]: row["label"] for row in source["rows"]}
            prediction = benchmark["models"]["rule"]["predictions"][0]
            old_id = prediction["transaction_id"]
            training_id = next(transaction_id for transaction_id in benchmark["split"]["train_ids"]
                               if labels_by_id[transaction_id] == prediction["label"])
            prediction["transaction_id"] = training_id
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                if row["model_id"] == "rule" and row["transaction_id"] == old_id:
                    row["transaction_id"] = training_id
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("held-out transaction IDs mismatch" in error for error in errors), errors)

    def test_duplicate_declared_holdout_id_is_rejected_after_coordinated_rehash(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            first_id, second_id = benchmark["split"]["test_ids"][:2]
            benchmark["split"]["test_ids"][1] = first_id
            for result in benchmark["models"].values():
                result["predictions"][1]["transaction_id"] = first_id
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                if row["transaction_id"] == second_id:
                    row["transaction_id"] = first_id
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("held-out transaction IDs mismatch" in error for error in errors), errors)

    def test_declared_holdout_cannot_overlap_training_ids(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            source = json.loads((bundle / "synthetic_transactions.json").read_text())
            labels_by_id = {row["transaction_id"]: row["label"] for row in source["rows"]}
            old_id = benchmark["split"]["test_ids"][0]
            training_id = next(transaction_id for transaction_id in benchmark["split"]["train_ids"]
                               if labels_by_id[transaction_id] == labels_by_id[old_id])
            benchmark["split"]["test_ids"][0] = training_id
            for result in benchmark["models"].values():
                result["predictions"][0]["transaction_id"] = training_id
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                if row["transaction_id"] == old_id:
                    row["transaction_id"] = training_id
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("held-out transaction IDs mismatch" in error for error in errors), errors)

    def test_coordinated_split_rewrite_cannot_override_source_order(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            transactions = json.loads((bundle / "synthetic_transactions.json").read_text())
            source_labels = {row["transaction_id"]: row["label"] for row in transactions["rows"]}
            old_test_id = benchmark["split"]["test_ids"][0]
            train_ids = benchmark["split"]["train_ids"]
            train_index = next(i for i, transaction_id in enumerate(train_ids)
                               if source_labels[transaction_id] == source_labels[old_test_id])
            old_train_id = train_ids[train_index]
            train_ids[train_index] = old_test_id
            benchmark["split"]["test_ids"][0] = old_train_id
            for result in benchmark["models"].values():
                result["predictions"][0]["transaction_id"] = old_train_id
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                if row["transaction_id"] == old_test_id:
                    row["transaction_id"] = old_train_id
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            governance_path = bundle / "governance.json"
            governance = json.loads(governance_path.read_text())
            governance["lineage"]["training_ids"] = train_ids
            governance["lineage"]["holdout_ids"] = benchmark["split"]["test_ids"]
            governance["lineage"]["benchmark_sha256"] = hashlib.sha256(
                json.dumps(benchmark, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
            governance_path.write_text(json.dumps(governance), encoding="utf-8")
            self._rehash(bundle, governance_path.name)

            record_path = bundle / "run_record.json"
            record = json.loads(record_path.read_text())
            record["result_sha256"]["benchmark"] = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
            record["result_sha256"]["governance"] = hashlib.sha256(governance_path.read_bytes()).hexdigest()
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self._rehash(bundle, record_path.name)
            lineage_path = bundle / "lineage.json"
            lineage = json.loads(lineage_path.read_text())
            lineage["split"] = benchmark["split"]
            lineage["governance_lineage"] = governance["lineage"]
            lineage["result_sha256"] = record["result_sha256"]
            lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
            self._rehash(bundle, lineage_path.name)
            old_manifest_path = bundle / "manifest.json"
            old_manifest = json.loads(old_manifest_path.read_text())
            for name in ("benchmark.json", "governance.json", "run_record.json"):
                old_manifest["artifacts"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
            self._rehash(bundle, old_manifest_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("source-order split mismatch" in error for error in errors), errors)

    def test_coordinated_train_threshold_change_recomputes_holdout_rates(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            benchmark["models"]["graph_feature_logistic"]["train_fpr_threshold"] = 0.0
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            metadata_path = bundle / "model_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["models"]["graph_feature_logistic"]["train_fpr_threshold"] = 0.0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            self._rehash(bundle, metadata_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("recomputed metric mismatch" in error and "observed_fpr" in error
                                for error in errors), errors)

    def test_coordinated_f1_tampering_is_rejected_from_raw_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            benchmark["models"]["rule"]["metrics"]["f1_at_0_5"] = 0.99
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            metrics_path = bundle / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["models"]["rule"]["f1_at_0_5"] = 0.99
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self._rehash(bundle, metrics_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("recomputed metric mismatch" in error and "f1_at_0_5" in error
                                for error in errors), errors)

    def test_coordinated_label_tampering_is_rejected_against_input_rows(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            for prediction in benchmark["models"]["rule"]["predictions"]:
                prediction["label"] = 0
            benchmark["models"]["rule"]["metrics"]["pr_auc"] = 0.99
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            metrics_path = bundle / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["models"]["rule"]["pr_auc"] = 0.99
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self._rehash(bundle, metrics_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                if row["model_id"] == "rule":
                    row["label"] = "0"
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("raw transaction label mismatch" in error for error in errors), errors)

    def test_coordinated_undefined_reason_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            benchmark["models"]["rule"]["metrics"]["undefined_reason"] = "holdout has only one class"
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            metrics_path = bundle / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["models"]["rule"]["undefined_reason"] = "holdout has only one class"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self._rehash(bundle, metrics_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("recomputed metric mismatch" in error and "undefined_reason" in error
                                for error in errors), errors)

    def test_single_class_holdout_cannot_keep_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            benchmark_path = bundle / "benchmark.json"
            benchmark = json.loads(benchmark_path.read_text())
            holdout_ids = set(benchmark["split"]["test_ids"])
            for result in benchmark["models"].values():
                for prediction in result["predictions"]:
                    prediction["label"] = 0
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            self._rehash(bundle, benchmark_path.name)

            transactions_path = bundle / "synthetic_transactions.json"
            transactions = json.loads(transactions_path.read_text())
            for row in transactions["rows"]:
                if row["transaction_id"] in holdout_ids:
                    row["label"] = 0
            transactions_path.write_text(json.dumps(transactions), encoding="utf-8")
            self._rehash(bundle, transactions_path.name)
            inputs_path = bundle / "input_manifest.json"
            inputs = json.loads(inputs_path.read_text())
            inputs["inputs"][1]["sha256"] = hashlib.sha256(transactions_path.read_bytes()).hexdigest()
            inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
            self._rehash(bundle, inputs_path.name)

            scores_path = bundle / "risk_scores.csv"
            with scores_path.open(newline="", encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
            for row in score_rows:
                row["label"] = "0"
            with scores_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
                writer.writeheader()
                writer.writerows(score_rows)
            self._rehash(bundle, scores_path.name)

            errors = pipeline.verify(bundle)
            self.assertTrue(any("single-class metric mismatch" in error for error in errors), errors)

    def test_rehashed_release_gate_cannot_turn_synthetic_run_into_pass(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "release_gate.json"
            gate = json.loads(target.read_text())
            gate["status"] = "pass"
            target.write_text(json.dumps(gate), encoding="utf-8")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("release gate mismatch" in error for error in pipeline.verify(bundle)))

    def test_rehashed_risk_score_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "risk_scores.csv"
            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["score"] = "-1"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("risk score mismatch" in error for error in pipeline.verify(bundle)))

    def test_rehashed_run_metadata_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            target = bundle / "run_metadata.json"
            metadata = json.loads(target.read_text())
            metadata["run_id"] = "other-run"
            target.write_text(json.dumps(metadata), encoding="utf-8")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("run_id mismatch" in error for error in pipeline.verify(bundle)))

    def test_cli_contract_run_is_reproducible_entry_point(self):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(
                [sys.executable, "-m", "yk_evidence", "contract-run", "--output", root,
                 "--run-id", "cli-contract-001", "--seed", "42"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual([], pipeline.verify(Path(root) / "cli-contract-001"))

    def test_contract_run_emits_required_artifacts_and_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue(hasattr(pipeline, "contract_run"), "contract_run is missing")
            bundle = pipeline.contract_run(Path(root), "contract-001", seed=42)
            expected = {
                "run_metadata.json", "input_manifest.json", "provenance.json",
                "lineage.json", "retrieval_evidence.json", "risk_scores.csv",
                "model_metadata.json", "metrics.json", "drift_report.json",
                "stress_test_results.csv", "release_gate.json", "review_report.html",
            }
            manifest = json.loads((bundle / "bundle_manifest.json").read_text())
            self.assertEqual("0.3", manifest["schema_version"])
            self.assertEqual("contract-001", manifest["run_id"])
            self.assertTrue(expected.issubset(manifest["artifacts"]))
            self.assertNotIn("bundle_manifest.json", manifest["artifacts"])
            self.assertEqual([], pipeline.verify(bundle))
            gate = json.loads((bundle / "release_gate.json").read_text())
            self.assertEqual("blocked", gate["status"])
            self.assertIn("synthetic_benchmark", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
