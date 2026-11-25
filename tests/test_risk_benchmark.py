import unittest

from yk_evidence.risk_data import generate_transactions
from yk_evidence.benchmark import run_benchmark


class RiskBenchmarkTest(unittest.TestCase):
    def test_generator_is_reproducible_and_contains_both_classes(self):
        first = generate_transactions(seed=42, count=400)
        self.assertEqual(first, generate_transactions(seed=42, count=400))
        self.assertEqual(400, len(first))
        self.assertEqual({0, 1}, {row["label"] for row in first})
        self.assertEqual(400, len({row["transaction_id"] for row in first}))

    def test_benchmark_saves_disjoint_split_and_raw_predictions(self):
        result = run_benchmark(generate_transactions(seed=42, count=400), seed=42)
        train = set(result["split"]["train_ids"])
        test = set(result["split"]["test_ids"])
        self.assertFalse(train & test)
        self.assertEqual(400, len(train | test))
        self.assertEqual({"rule", "tabular_logistic", "graph_feature_logistic", "graph_feature_random_forest"}, set(result["models"]))
        for model in result["models"].values():
            self.assertEqual(test, {x["transaction_id"] for x in model["predictions"]})
            self.assertTrue(all(0 <= x["score"] <= 1 for x in model["predictions"]))
            self.assertEqual({0, 1}, {x["label"] for x in model["predictions"]})

    def test_metric_is_recomputable_from_stored_predictions(self):
        from sklearn.metrics import average_precision_score, f1_score

        result = run_benchmark(generate_transactions(seed=42, count=400), seed=42)
        model = result["models"]["graph_feature_logistic"]
        raw = model["predictions"]
        labels = [x["label"] for x in raw]
        scores = [x["score"] for x in raw]
        self.assertAlmostEqual(average_precision_score(labels, scores), model["metrics"]["pr_auc"])
        self.assertEqual(f1_score(labels, [int(x >= 0.5) for x in scores]), model["metrics"]["f1_at_0_5"])

    def test_random_forest_scores_are_seed_reproducible(self):
        rows = generate_transactions(seed=42, count=400)
        first = run_benchmark(rows, seed=42)["models"]["graph_feature_random_forest"]
        second = run_benchmark(rows, seed=42)["models"]["graph_feature_random_forest"]
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(100, len(first["predictions"]))

    def test_graph_features_do_not_read_labels(self):
        from yk_evidence.risk_data import graph_history_features

        rows = generate_transactions(seed=42, count=200)
        original = graph_history_features(rows)
        changed = [dict(row, label=1 - row["label"]) for row in rows]
        self.assertEqual(original, graph_history_features(changed))

    def test_single_class_holdout_has_explicitly_undefined_metrics(self):
        rows = generate_transactions(seed=42, count=400)
        for row in rows[300:]:
            row["label"] = 0
        result = run_benchmark(rows, seed=42)
        metrics = result["models"]["graph_feature_logistic"]["metrics"]
        self.assertIsNone(metrics["pr_auc"])
        self.assertEqual("holdout has only one class", metrics["undefined_reason"])


if __name__ == "__main__":
    unittest.main()
