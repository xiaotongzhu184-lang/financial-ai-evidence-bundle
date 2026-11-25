"""Public-seam tests for the tabular ULB evaluation contract."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sklearn.metrics import average_precision_score, brier_score_loss

import yk_evidence.evaluation as evaluation_module
from yk_evidence.evaluation import evaluate_dataset, verify_evaluation


HEADERS = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"]
METADATA = {
    "dataset_name": "generated ULB-schema smoke fixture",
    "classification": "synthetic_smoke_fixture",
    "source_url": "local:test_evaluation.py",
    "license_status": "project-generated",
}


def make_csv(path: Path, count: int = 160) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        for index in range(count):
            label = int(index % 8 == 0)
            features = [round((index % (7 + (j % 5))) * 0.1 + label * (j % 3), 3) for j in range(1, 29)]
            writer.writerow([index // 2, *features, 20 + index % 100, label])


def rehash_manifest(folder: Path, name: str) -> None:
    manifest_path = folder / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class EvaluationTests(unittest.TestCase):
    def test_real_models_produce_recheckable_tie_safe_evidence(self) -> None:
        """Catches missing model runs, partition leakage, or unverifiable output."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            output = folder / "evaluation"
            summary = evaluate_dataset(source, METADATA, output, seed=17, load_rows=256)
            self.assertEqual([], verify_evaluation(output))
            self.assertEqual(160, summary["dataset"]["row_count"])
            self.assertEqual(
                {"logistic_regression", "random_forest", "hist_gradient_boosting"},
                set(summary["models"]),
            )
            splits = summary["partitions"]
            self.assertEqual(160, sum(item["row_count"] for item in splits.values()))
            for earlier, later in zip(list(splits.values()), list(splits.values())[1:]):
                self.assertLess(earlier["time_max"], later["time_min"])
            with (output / "evaluation_predictions.csv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(3 * 72, len(records))
            self.assertIn("calibration_input", records[0])
            for model_id in summary["models"]:
                test_rows = [row for row in records if row["model_id"] == model_id and row["partition"] == "test"]
                labels = [int(row["label"]) for row in test_rows]
                raw = [float(row["raw_score"]) for row in test_rows]
                reported = summary["models"][model_id]["raw_metrics"]["test"]
                self.assertAlmostEqual(average_precision_score(labels, raw), reported["average_precision"])
                self.assertAlmostEqual(brier_score_loss(labels, raw), reported["brier"])
                workload = summary["models"][model_id]["metrics"]["test"]["review_workload"]
                alerts = sum(int(row["decision"]) for row in test_rows)
                missed = sum(int(row["label"]) == 1 and int(row["decision"]) == 0 for row in test_rows)
                self.assertEqual(alerts, workload["alert_count"])
                self.assertAlmostEqual(10_000 * alerts / len(test_rows), workload["alerts_per_10000"])
                self.assertEqual(missed, workload["missed_cases"])
                self.assertIs(False, workload["monetary_cost_estimated"])
            self.assertEqual(256, summary["capacity"]["workloads"][-1]["scored_rows"])
            memory = summary["capacity"]["workloads"][-1]
            if memory["process_maxrss_source_unit"] == "unavailable":
                self.assertIsNone(memory["process_maxrss_raw"])
                self.assertIsNone(memory["process_maxrss_bytes"])
            else:
                self.assertIn(memory["process_maxrss_source_unit"], ("bytes", "kibibytes"))
                self.assertIs(type(memory["process_maxrss_raw"]), int)
                self.assertGreaterEqual(memory["process_maxrss_raw"], 0)
                self.assertIs(type(memory["process_maxrss_bytes"]), int)
                self.assertGreaterEqual(memory["process_maxrss_bytes"], 0)
            self.assertGreaterEqual(summary["capacity"]["machine"]["cpu_count"], 1)
            selected = summary["selection"]["model_id"]
            self.assertEqual(summary["models"][selected]["metrics"]["test"], summary["stress"]["baseline_metrics"])
            for scenario in summary["stress"]["scenarios"].values():
                self.assertIn(scenario["status"], ("pass", "fail"))
                self.assertEqual(0.10, scenario["maximum_allowed_average_precision_drop"])

    def test_claimed_external_source_digest_must_match_input(self) -> None:
        """Catches evaluating one file under another dataset's provenance hash."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            wrong_metadata = dict(METADATA, source_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "source.*SHA|source.*hash"):
                evaluate_dataset(source, wrong_metadata, folder / "evaluation", load_rows=16)

    def test_existing_empty_stage_is_accepted_but_nonempty_output_is_not_overwritten(self) -> None:
        """Catches breaking integration's staging directory or overwriting evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            output = folder / "evaluation"
            output.mkdir()
            evaluate_dataset(source, METADATA, output, seed=17, load_rows=16)
            self.assertEqual([], verify_evaluation(output))
            with self.assertRaises(FileExistsError):
                evaluate_dataset(source, METADATA, output, seed=17, load_rows=16)

    def test_rehashed_score_threshold_or_metric_edit_fails_semantic_verification(self) -> None:
        """Catches accepting changed statistics merely because the local hashes were rebuilt."""
        for changed_file, change in (
            ("evaluation_predictions.csv", "score"),
            ("evaluation_summary.json", "threshold"),
            ("evaluation_summary.json", "metric"),
            ("evaluation_summary.json", "raw_metric"),
            ("evaluation_summary.json", "calibration_parameter"),
            ("evaluation_summary.json", "wrong_model_artifact"),
            ("evaluation_summary.json", "review_workload"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                source = folder / "smoke.csv"
                make_csv(source)
                output = folder / "evaluation"
                evaluate_dataset(source, METADATA, output, seed=17, load_rows=16)
                target = output / changed_file
                if change == "score":
                    with target.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    rows[0]["calibrated_score"] = "0.123456"
                    with target.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    summary = json.loads(target.read_text(encoding="utf-8"))
                    model = summary["selection"]["model_id"]
                    if change == "threshold":
                        summary["models"][model]["threshold"] = -1.0
                    elif change == "metric":
                        summary["models"][model]["metrics"]["test"]["average_precision"] = -1.0
                    elif change == "raw_metric":
                        summary["models"][model]["raw_metrics"]["test"]["brier"] = -1.0
                    elif change == "wrong_model_artifact":
                        other = next(name for name in summary["models"] if name != model)
                        summary["models"][model]["model_artifact"] = summary["models"][other]["model_artifact"]
                        summary["models"][model]["model_artifact_sha256"] = summary["models"][other]["model_artifact_sha256"]
                    elif change == "review_workload":
                        summary["models"][model]["metrics"]["test"]["review_workload"]["alert_count"] += 1
                    else:
                        summary["models"][model]["calibration_mapping"]["b"] += 1.0
                    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                rehash_manifest(output, changed_file)
                self.assertTrue(verify_evaluation(output), change)

    def test_invalid_source_rows_fail_without_final_output(self) -> None:
        """Catches treating absent or invalid labels/features as measured evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            output = folder / "evaluation"
            with self.assertRaisesRegex(ValueError, "not found"):
                evaluate_dataset(source, METADATA, output, load_rows=16)
            for column, bad_value in (("Time", "nan"), ("V1", "nan"), ("Class", "2")):
                with self.subTest(column=column):
                    make_csv(source)
                    with source.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    rows[0][column] = bad_value
                    with source.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=HEADERS)
                        writer.writeheader()
                        writer.writerows(rows)
                    with self.assertRaisesRegex(ValueError, "CSV row 2"):
                        evaluate_dataset(source, METADATA, output, load_rows=16)
                    self.assertFalse(output.exists())

    def test_invalid_seed_and_load_controls_fail_before_fitting(self) -> None:
        """Catches unbounded or ambiguous workload configuration."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            for number, options in enumerate(({"seed": -1}, {"seed": True}, {"load_rows": 10_000_001},
                                              {"load_rows": 1.5}, {"fpr_ceiling": float("nan")})):
                output = folder / f"evaluation-{number}"
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        evaluate_dataset(source, METADATA, output, **options)
                    self.assertFalse(output.exists())

    def test_missing_os_peak_memory_api_is_reported_as_unavailable(self) -> None:
        """Catches crashing on Windows or inventing a memory measurement."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            output = folder / "evaluation"
            with patch.object(evaluation_module, "resource", None):
                summary = evaluate_dataset(source, METADATA, output, load_rows=16)
            workload = summary["capacity"]["workloads"][0]
            self.assertIsNone(workload["process_maxrss_bytes"])
            self.assertIsNone(workload["process_maxrss_raw"])
            self.assertEqual("unavailable", workload["process_maxrss_source_unit"])
            self.assertEqual([], verify_evaluation(output))

    def test_coordinated_stress_scenario_deletion_addition_and_count_edits_fail(self) -> None:
        """Catches a self-consistent rehash hiding required stress experiments."""
        for change in ("delete_one", "delete_all", "add_unknown", "wrong_changed_rows", "missing_test_row"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                source = folder / "smoke.csv"
                make_csv(source)
                output = folder / "evaluation"
                evaluate_dataset(source, METADATA, output, load_rows=16)
                stress_file = output / "evaluation_stress.json"
                summary_file = output / "evaluation_summary.json"
                stress = json.loads(stress_file.read_text(encoding="utf-8"))
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                scenarios = stress["scenarios"]
                key = next(iter(scenarios))
                if change == "delete_one":
                    del scenarios[key]
                elif change == "delete_all":
                    scenarios.clear()
                elif change == "add_unknown":
                    scenarios["unapproved_stress"] = dict(scenarios[key])
                elif change == "wrong_changed_rows":
                    scenarios[key]["changed_rows"] = 0
                else:
                    scenarios[key]["predictions"].pop()
                summary["stress"] = stress
                stress_file.write_text(json.dumps(stress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                rehash_manifest(output, stress_file.name)
                rehash_manifest(output, summary_file.name)
                self.assertTrue(verify_evaluation(output), change)

    def test_coordinated_duplicate_capacity_workload_fails(self) -> None:
        """Catches repeated capacity entries passing as two declared workloads."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            output = folder / "evaluation"
            evaluate_dataset(source, METADATA, output, load_rows=100_001)
            capacity_file = output / "evaluation_capacity.json"
            summary_file = output / "evaluation_summary.json"
            capacity = json.loads(capacity_file.read_text(encoding="utf-8"))
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            capacity["workloads"].append(dict(capacity["workloads"][0]))
            summary["capacity"] = capacity
            capacity_file.write_text(json.dumps(capacity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rehash_manifest(output, capacity_file.name)
            rehash_manifest(output, summary_file.name)
            self.assertTrue(verify_evaluation(output))

    def test_coordinated_cross_partition_source_id_substitution_fails(self) -> None:
        """Catches validation rows being relabelled as calibration source rows after all hashes change."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "smoke.csv"
            make_csv(source)
            output = folder / "evaluation"
            evaluate_dataset(source, METADATA, output, load_rows=16)
            predictions_file = output / "evaluation_predictions.csv"
            summary_file = output / "evaluation_summary.json"
            with predictions_file.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            calibration_id = next(row["row_id"] for row in records if row["model_id"] == "logistic_regression"
                                  and row["partition"] == "calibration")
            for model_id in summary["models"]:
                validation = [row for row in records if row["model_id"] == model_id and row["partition"] == "validation"]
                validation[0]["row_id"] = calibration_id
                canonical = [
                    (int(row["row_id"]), float(row["relative_time"]), int(row["label"]),
                     float(row["raw_score"]), float(row["calibration_input"]),
                     float(row["calibrated_score"]), int(row["decision"])) for row in validation
                ]
                summary["models"][model_id]["predictions_sha256"]["validation"] = hashlib.sha256(
                    json.dumps(canonical, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            validation_ids = [int(row["row_id"]) for row in records if row["model_id"] == "logistic_regression"
                              and row["partition"] == "validation"]
            summary["partitions"]["validation"]["row_ids_sha256"] = hashlib.sha256(
                b"".join(struct.pack("<q", row_id) for row_id in validation_ids)).hexdigest()
            with predictions_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0])
                writer.writeheader()
                writer.writerows(records)
            summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rehash_manifest(output, predictions_file.name)
            rehash_manifest(output, summary_file.name)
            self.assertTrue(verify_evaluation(output))
