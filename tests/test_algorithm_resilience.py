"""Real four-estimator integration and adverse evidence for schema 0.6."""

import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib

from yk_evidence.algorithm import (REQUIRED, algorithm_smoke, render_algorithm_report,
                                   render_resilience_report, verify_algorithm)
from yk_evidence.fusion import apply_policy, assess_scores
from yk_evidence.research import policy_for


def _write_and_rehash(bundle: Path, name: str, value: dict) -> None:
    path = bundle / name
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class AlgorithmResilienceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.reference = algorithm_smoke(Path(cls.temporary.name), "resilience-smoke",
                                        seed=17, load_rows=32)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.working = tempfile.TemporaryDirectory()
        self.bundle = Path(self.working.name) / "bundle"
        shutil.copytree(self.reference, self.bundle)

    def tearDown(self):
        self.working.cleanup()

    def test_real_smoke_emits_retry_fallback_and_fail_closed_evidence(self):
        # Regression target: a 0.5 runner without actual recovery evidence
        # must fail this test before the 0.6 integration exists.
        bundle = self.bundle
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        self.assertEqual("0.6-algorithm", manifest["schema_version"])
        model_record = json.loads((bundle / "algorithm_models.json").read_text())
        self.assertEqual(1, model_record["models"]["random_forest"]["parameters"]["n_jobs"])
        trained = joblib.load(bundle / "model_random_forest.joblib")
        forest = trained["calibrator"].calibrated_classifiers_[0].estimator.estimator.steps[-1][1]
        self.assertEqual(1, forest.n_jobs)
        gate = json.loads((bundle / "release_gate.json").read_text())
        self.assertIn("algorithm_results.json", gate["stress_interpretation"])
        self.assertIn("resilience", gate["stress_interpretation"])
        self.assertIn("resilience_policy.json", manifest["artifacts"])
        self.assertIn("resilience_evidence.json", manifest["artifacts"])
        self.assertEqual([], verify_algorithm(bundle))
        evidence = json.loads((bundle / "resilience_evidence.json").read_text())
        reports = json.loads((bundle / "algorithm_results.json").read_text())["resilience"]
        measurement = evidence["runtime_measurement"]
        self.assertGreater(measurement["elapsed_seconds"], 0)
        self.assertEqual(sum(sum(item["call_counts"].values()) for item in evidence["scenarios"]),
                         measurement["model_call_count"])
        self.assertEqual(8, len(evidence["scenarios"]))
        self.assertEqual(8, len(reports["scenarios"]))
        by_name = {item["name"]: item for item in evidence["scenarios"]}
        strongest = evidence["strongest_model"]
        self.assertEqual("baseline_fusion", by_name["clean"]["route"])
        self.assertEqual(1, by_name["clean"]["call_counts"][strongest])
        for name in ("strongest_model_dropout", "one_model_row_roll", "eight_bit_quantization", "stale_context"):
            self.assertEqual("ok", by_name[name]["status"])
            self.assertEqual(2, by_name[name]["call_counts"][strongest])
            self.assertEqual("baseline_fusion", by_name[name]["route"])
        self.assertEqual("validation_frozen_fallback", by_name["persistent_strongest_model_dropout"]["route"])
        self.assertEqual("fail_closed", by_name["all_models_unavailable"]["status"])
        self.assertEqual(0, by_name["all_models_unavailable"]["decision_count"])
        self.assertEqual([], by_name["all_models_unavailable"]["scores"])
        self.assertIsNone(by_name["all_models_unavailable"]["queue"])
        self.assertEqual("not_applicable", next(item for item in reports["scenarios"]
                                                  if item["name"] == "all_models_unavailable")["quality_status"])
        page = (bundle / "review_report.html").read_text()
        self.assertIn("Protected scenarios meeting the frozen AP-loss criterion:", page)
        self.assertIn("Individual model delivery attempts", page)
        self.assertLess(page.index("Prediction delivery recovery"),
                        page.index("Unprotected 0.5 control probes (before recovery)"))
        interactive_page = render_resilience_report(bundle, "test-csrf").decode()
        self.assertLess(interactive_page.index("Prediction delivery recovery"),
                        interactive_page.index('<form'))

    def test_rehashed_receipt_policy_status_and_scenario_omission_are_rejected(self):
        cases = (
            ("resilience_evidence.json", lambda o: o["scenarios"][0]["attempts"][0].__setitem__(
                "receipt_sha256", "0" * 64)),
            ("resilience_evidence.json", lambda o: o["scenarios"].pop()),
            ("resilience_policy.json", lambda o: o["subsets"][0].__setitem__(
                "selected_model", "xgboost")),
            ("algorithm_results.json", lambda o: o["resilience"]["scenarios"][0].__setitem__(
                "quality_status", "fail")),
        )
        for index, (name, change) in enumerate(cases):
            with self.subTest(case=index):
                target = Path(self.working.name) / f"tamper-{index}"
                shutil.copytree(self.reference, target)
                value = json.loads((target / name).read_text())
                change(value)
                _write_and_rehash(target, name, value)
                self.assertTrue(verify_algorithm(target))

    def test_rehashed_runtime_call_count_float_is_rejected(self):
        name = "resilience_evidence.json"
        value = json.loads((self.bundle / name).read_text())
        value["runtime_measurement"]["model_call_count"] = float(
            value["runtime_measurement"]["model_call_count"])
        _write_and_rehash(self.bundle, name, value)
        self.assertTrue(verify_algorithm(self.bundle))

    def test_independent_stdlib_audit_recomputes_and_rejects_rehashed_metrics(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_resilience_results.py"
        clean = subprocess.run([sys.executable, str(script), str(self.bundle)],
                               capture_output=True, text=True)
        self.assertEqual(0, clean.returncode, clean.stdout + clean.stderr)
        value = json.loads((self.bundle / "algorithm_results.json").read_text())
        value["resilience"]["scenarios"][0]["protected_metrics"]["average_precision"] = 0.01
        _write_and_rehash(self.bundle, "algorithm_results.json", value)
        changed = subprocess.run([sys.executable, str(script), str(self.bundle)],
                                 capture_output=True, text=True)
        self.assertEqual(1, changed.returncode)
        self.assertIn("protected_metrics.average_precision", changed.stdout)

    def test_independent_audit_recomputes_validation_frozen_fallback_threshold(self):
        bundle = self.bundle
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_resilience_results.py"
        policy = json.loads((bundle / "resilience_policy.json").read_text())
        evidence = json.loads((bundle / "resilience_evidence.json").read_text())
        scores = json.loads((bundle / "algorithm_scores.json").read_text())
        results = json.loads((bundle / "algorithm_results.json").read_text())
        scenario = next(item for item in evidence["scenarios"]
                        if item["name"] == "persistent_strongest_model_dropout")
        subset = next(item for item in policy["subsets"]
                      if item["healthy_models"] == scenario["healthy_models"])
        subset["threshold"] = 0.99
        payload = {key: value for key, value in policy.items() if key != "payload_sha256"}
        policy["payload_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _write_and_rehash(bundle, "resilience_policy.json", policy)
        ids = evidence["row_ids"]
        labels = [row["label"] for row in scores["test"]]
        scenario["threshold"] = subset["threshold"]
        scenario["queue"] = apply_policy(scenario["scores"], ids, subset["threshold"],
                                         policy["budget_fraction"])
        _write_and_rehash(bundle, "resilience_evidence.json", evidence)
        report = next(item for item in results["resilience"]["scenarios"]
                      if item["name"] == scenario["name"])
        report["protected_metrics"] = assess_scores(labels, scenario["scores"], ids,
                                                     subset["threshold"], policy["budget_fraction"])
        _write_and_rehash(bundle, "algorithm_results.json", results)
        self.assertTrue(verify_algorithm(bundle))
        checked = subprocess.run([sys.executable, str(script), str(bundle)],
                                 capture_output=True, text=True)
        self.assertEqual(1, checked.returncode, checked.stdout + checked.stderr)
        self.assertIn("validation-frozen policy", checked.stdout)

    def test_independent_audit_rejects_rehashed_subnanounit_fallback_threshold(self):
        bundle = self.bundle
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_resilience_results.py"
        policy = json.loads((bundle / "resilience_policy.json").read_text())
        evidence = json.loads((bundle / "resilience_evidence.json").read_text())
        scores = json.loads((bundle / "algorithm_scores.json").read_text())
        results = json.loads((bundle / "algorithm_results.json").read_text())
        scenario = next(item for item in evidence["scenarios"]
                        if item["name"] == "persistent_strongest_model_dropout")
        subset = next(item for item in policy["subsets"]
                      if item["healthy_models"] == scenario["healthy_models"])
        subset["threshold"] -= 5e-10
        payload = {key: value for key, value in policy.items() if key != "payload_sha256"}
        policy["payload_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        _write_and_rehash(bundle, "resilience_policy.json", policy)
        ids = evidence["row_ids"]
        labels = [row["label"] for row in scores["test"]]
        scenario["threshold"] = subset["threshold"]
        scenario["queue"] = apply_policy(scenario["scores"], ids, subset["threshold"],
                                         policy["budget_fraction"])
        _write_and_rehash(bundle, "resilience_evidence.json", evidence)
        report = next(item for item in results["resilience"]["scenarios"]
                      if item["name"] == scenario["name"])
        report["protected_metrics"] = assess_scores(labels, scenario["scores"], ids,
                                                     subset["threshold"], policy["budget_fraction"])
        _write_and_rehash(bundle, "algorithm_results.json", results)
        checked = subprocess.run([sys.executable, str(script), str(bundle)],
                                 capture_output=True, text=True)
        self.assertEqual(1, checked.returncode, checked.stdout + checked.stderr)
        self.assertIn("validation-frozen policy", checked.stdout)

    def test_legacy_05_bundle_still_verifies_and_keeps_legacy_html(self):
        bundle = self.bundle
        (bundle / "resilience_policy.json").unlink()
        (bundle / "resilience_evidence.json").unlink()
        results = json.loads((bundle / "algorithm_results.json").read_text())
        results.pop("resilience")
        (bundle / "algorithm_results.json").write_text(json.dumps(results))
        models = json.loads((bundle / "algorithm_models.json").read_text())
        models["schema_version"] = "0.5-algorithm"
        models["models"]["random_forest"]["parameters"]["n_jobs"] = 2
        models.pop("test_feature_batch_sha256")
        (bundle / "algorithm_models.json").write_text(json.dumps(models))
        metadata = json.loads((bundle / "run_metadata.json").read_text())
        metadata["schema_version"] = "0.5-algorithm"
        metadata["project_version"] = "0.5.0rc1"
        (bundle / "run_metadata.json").write_text(json.dumps(metadata))
        source = json.loads((bundle / "source_metadata.json").read_text())
        (bundle / "release_gate.json").write_text(json.dumps(policy_for(source["classification"])))
        (bundle / "review_report.html").write_bytes(render_algorithm_report(bundle))
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        manifest["schema_version"] = "0.5-algorithm"
        manifest["project_version"] = "0.5.0rc1"
        manifest["required_artifacts"] = list(REQUIRED)
        manifest["artifacts"] = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                                 for name in REQUIRED}
        (bundle / "bundle_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual([], verify_algorithm(bundle))
        self.assertEqual((bundle / "review_report.html").read_bytes(),
                         render_algorithm_report(bundle))


if __name__ == "__main__":
    unittest.main()
