import unittest

from yk_evidence.benchmark import run_benchmark
from yk_evidence.risk_data import generate_transactions
from yk_evidence.validation import evaluate_stability_and_errors


class ValidationTest(unittest.TestCase):
    def test_stability_runs_distinct_synthetic_seeds_and_records_raw_metrics(self):
        rows = generate_transactions(seed=42, count=400)
        reference = run_benchmark(rows, seed=42)
        report = evaluate_stability_and_errors(reference, seeds=[42, 43, 44])
        self.assertEqual("synthetic_only", report["external_validity"])
        self.assertEqual([42, 43, 44], [item["seed"] for item in report["stability_runs"]])
        self.assertEqual(3, len(report["stability_runs"]))
        for item in report["stability_runs"]:
            self.assertEqual(4, len(item["models"]))
            self.assertTrue(all(0 <= m["pr_auc"] <= 1 for m in item["models"].values()))

    def test_error_counts_are_recomputed_from_saved_holdout_predictions(self):
        rows = generate_transactions(seed=42, count=400)
        reference = run_benchmark(rows, seed=42)
        report = evaluate_stability_and_errors(reference, seeds=[42])
        for name, model in reference["models"].items():
            predictions = model["predictions"]
            observed_fp = sum(row["label"] == 0 and row["decision_at_0_5"] == 1 for row in predictions)
            observed_fn = sum(row["label"] == 1 and row["decision_at_0_5"] == 0 for row in predictions)
            self.assertEqual(observed_fp, report["error_analysis"][name]["false_positives"])
            self.assertEqual(observed_fn, report["error_analysis"][name]["false_negatives"])


if __name__ == "__main__":
    unittest.main()
