"""Decision-impact calculations must stay tied to saved held-out predictions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from yk_evidence.decision_impact import build_decision_impact
from yk_evidence.pipeline import contract_run, verify


class DecisionImpactTest(unittest.TestCase):
    def test_fixed_train_threshold_and_scenario_arithmetic(self):
        benchmark = {"classification": "synthetic_benchmark", "models": {
            "toy": {"train_fpr_threshold": 0.5, "predictions": [
                {"transaction_id": "a", "label": 1, "score": 0.9},
                {"transaction_id": "b", "label": 1, "score": 0.4},
                {"transaction_id": "c", "label": 0, "score": 0.8},
                {"transaction_id": "d", "label": 0, "score": 0.1},
            ]}
        }}
        policy = {"scenario_id": "toy-policy", "assumed_prevalence": 0.1,
                  "transactions_per_period": 1000, "review_cost_units": 1.0,
                  "miss_cost_units": 20.0, "review_capacity_per_period": 200}
        result = build_decision_impact(benchmark, "toy-001", [policy])
        model = result["models"]["toy"]
        self.assertEqual(0.5, model["threshold_from_training"])
        self.assertEqual({"tp": 1, "fp": 1, "tn": 1, "fn": 1}, model["heldout_confusion"])
        self.assertEqual(0.5, model["heldout_tpr"])
        self.assertEqual(0.5, model["heldout_fpr"])
        scenario = model["scenarios"][0]
        self.assertEqual(500.0, scenario["expected_alerts_per_period"])
        self.assertEqual(450.0, scenario["expected_false_alerts_per_period"])
        self.assertEqual(50.0, scenario["expected_missed_positives_per_period"])
        self.assertEqual(1500.0, scenario["illustrative_cost_units"])
        self.assertEqual("capacity_exceeded", scenario["capacity_status"])

    def test_single_class_holdout_does_not_invent_rates(self):
        benchmark = {"classification": "synthetic_benchmark", "models": {
            "toy": {"train_fpr_threshold": 0.5, "predictions": [
                {"transaction_id": "a", "label": 0, "score": 0.9},
                {"transaction_id": "b", "label": 0, "score": 0.1},
            ]}
        }}
        result = build_decision_impact(benchmark, "toy-002")
        model = result["models"]["toy"]
        self.assertIsNone(model["heldout_tpr"])
        self.assertEqual("single_class_holdout", model["availability"])
        self.assertEqual([], model["scenarios"])

    def test_invalid_policy_and_score_are_rejected(self):
        benchmark = {"classification": "synthetic_benchmark", "models": {
            "toy": {"train_fpr_threshold": 0.5, "predictions": [
                {"transaction_id": "a", "label": 1, "score": 1.2},
                {"transaction_id": "b", "label": 0, "score": 0.1},
            ]}
        }}
        with self.assertRaises(ValueError):
            build_decision_impact(benchmark, "toy-003")
        benchmark["models"]["toy"]["predictions"][0]["score"] = 0.9
        with self.assertRaises(ValueError):
            build_decision_impact(benchmark, "toy-003", [{"scenario_id": "invalid", "assumed_prevalence": 1.2,
                "transactions_per_period": 1000, "review_cost_units": 1.0,
                "miss_cost_units": 20.0, "review_capacity_per_period": 200}])

    def test_contract_emits_verifiable_impact_and_rehashed_tamper_fails(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "impact-001", seed=42)
            target = bundle / "decision_impact.json"
            self.assertTrue(target.is_file())
            self.assertEqual([], verify(bundle))
            report = json.loads(target.read_text(encoding="utf-8"))
            model = next(iter(report["models"].values()))
            model["scenarios"][0]["expected_false_alerts_per_period"] = 0
            target.write_text(json.dumps(report), encoding="utf-8")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["decision_impact.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("decision impact mismatch" in error for error in verify(bundle)))

    def test_deleting_impact_and_rewriting_manifest_cannot_hide_it(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "impact-required", seed=42)
            (bundle / "decision_impact.json").unlink()
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"].pop("decision_impact.json")
            manifest["required_artifacts"].remove("decision_impact.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            errors = verify(bundle)
            self.assertTrue(any("missing manifest entry: decision_impact.json" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
