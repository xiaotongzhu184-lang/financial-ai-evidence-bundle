import unittest

from yk_evidence.benchmark import run_benchmark
from yk_evidence.risk_data import generate_transactions
from yk_evidence.stress_replay import replay_graph_stress


class StressReplayTest(unittest.TestCase):
    def test_frozen_graph_model_reproduces_baseline_and_rescores_scenarios(self):
        rows = generate_transactions(seed=42, count=400)
        benchmark = run_benchmark(rows, seed=42)
        report = replay_graph_stress(rows, benchmark, seed=42)
        original = [p["score"] for p in benchmark["models"]["graph_feature_logistic"]["predictions"]]
        baseline = [p["score"] for p in report["baseline"]["predictions"]]
        self.assertEqual(len(original), len(baseline))
        for expected, observed in zip(original, baseline):
            self.assertAlmostEqual(expected, observed, places=12)
        self.assertEqual({"amount_scale_1_2", "ring_link_suppression"}, set(report["scenarios"]))
        self.assertTrue(all(item["model_sha256"] == report["model_sha256"] for item in report["scenarios"].values()))
        for item in report["scenarios"].values():
            self.assertGreater(item["changed_rows"], 0)
            self.assertEqual(100, len(item["predictions"]))
            self.assertTrue(any(a["score"] != b["score"] for a, b in zip(report["baseline"]["predictions"], item["predictions"])))

    def test_stress_replay_refuses_mismatched_reference_split(self):
        rows = generate_transactions(seed=42, count=400)
        benchmark = run_benchmark(rows, seed=42)
        benchmark["split"]["test_ids"][0] = "wrong-id"
        with self.assertRaises(ValueError):
            replay_graph_stress(rows, benchmark, seed=42)


if __name__ == "__main__":
    unittest.main()
