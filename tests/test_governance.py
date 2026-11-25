import unittest

from yk_evidence.benchmark import run_benchmark
from yk_evidence.governance import evaluate_governance
from yk_evidence.risk_data import generate_transactions


class GovernanceTest(unittest.TestCase):
    def setUp(self):
        self.rows = generate_transactions(seed=42, count=400)
        self.benchmark = run_benchmark(self.rows, seed=42)

    def test_lineage_drift_and_two_versioned_scenarios(self):
        report = evaluate_governance(self.benchmark, self.rows)
        self.assertEqual(64, len(report["lineage"]["transactions_sha256"]))
        self.assertEqual(64, len(report["lineage"]["benchmark_sha256"]))
        self.assertGreaterEqual(report["drift"]["amount_psi"], 0)
        self.assertEqual({"amount_scale_1_2_v1", "ring_burst_x2_v1"}, {s["scenario_id"] for s in report["scenarios"]})
        self.assertIn(report["gate"]["status"], {"pass", "fail"})

    def test_forced_zero_psi_limit_fails_gate(self):
        report = evaluate_governance(self.benchmark, self.rows, psi_limit=0.0)
        self.assertEqual("fail", report["gate"]["status"])
        self.assertTrue(any("PSI" in reason for reason in report["gate"]["reasons"]))

    def test_missing_benchmark_evidence_is_incomplete(self):
        report = evaluate_governance({}, self.rows)
        self.assertEqual("incomplete", report["gate"]["status"])


if __name__ == "__main__":
    unittest.main()
