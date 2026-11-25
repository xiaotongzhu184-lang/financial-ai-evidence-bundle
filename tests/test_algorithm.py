"""Real estimator integration and adversarial semantic checks for schema 0.5."""

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from yk_evidence.algorithm import algorithm_run, verify_algorithm, MANIFEST, REQUIRED_RESILIENCE


def _fixture(path: Path, rows: int = 600) -> dict:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"])
        for index in range(rows):
            label = int(index % 8 == 0)
            features = [round((index % (7 + j % 5)) * .1 + label * (j % 3), 3) for j in range(1, 29)]
            writer.writerow([index // 2, *features, index % 100, label])
    return {"dataset_name": "Generated algorithm integration fixture",
            "classification": "synthetic_smoke_fixture",
            "source_url": "local:tests.test_algorithm._fixture",
            "license_status": "project-generated synthetic fixture",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _rehash(bundle: Path, name: str) -> None:
    manifest_path = bundle / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


class AlgorithmEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "generated.csv"
        cls.metadata = _fixture(cls.source)
        cls.reference = algorithm_run(cls.root / "runs", "reference", cls.source,
                                      cls.metadata, seed=17, load_rows=256,
                                      budget_fraction=.1)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.working = tempfile.TemporaryDirectory()
        self.bundle = Path(self.working.name) / "bundle"
        shutil.copytree(self.reference, self.bundle)

    def tearDown(self):
        self.working.cleanup()

    def test_real_four_model_bundle_and_semantic_verification(self):
        self.assertEqual([], verify_algorithm(self.bundle))
        self.assertEqual(set(REQUIRED_RESILIENCE) | {MANIFEST}, {p.name for p in self.bundle.iterdir()})
        policy = json.loads((self.bundle / "fusion_policy.json").read_text())
        self.assertEqual(4, len(policy["model_names"]))
        self.assertIn("xgboost", policy["model_names"])
        self.assertEqual(35, len(policy["candidates"]))
        self.assertEqual(1., sum(policy["weights"]))
        results = json.loads((self.bundle / "algorithm_results.json").read_text())
        self.assertEqual(7, len(results["ablations"]))
        self.assertEqual(3, len(results["stress"]["scenarios"]))
        self.assertIn("window_monitoring", results)
        self.assertEqual("independent_retrospective_window_diagnostics",
                         results["window_monitoring"]["classification"])
        self.assertEqual({*policy["model_names"], "equal_weight", "robust_weighted_fusion"},
                         set(results["window_monitoring"]["analyses"]))
        for analysis in results["window_monitoring"]["analyses"].values():
            self.assertEqual(policy["policy_row_ids"], analysis["reference"]["row_ids"])
            windows = analysis["windows"]
            self.assertEqual(3, len(windows))
            self.assertEqual(90, sum(window["row_count"] for window in windows))
            self.assertTrue(all(left["relative_time_max"] < right["relative_time_min"]
                                for left, right in zip(windows, windows[1:])))
        self.assertLessEqual(results["test_queue"]["reviewed_count"], results["test_queue"]["capacity"])
        self.assertEqual("score_space_faults_only", results["stress"]["classification"])
        self.assertEqual("repeated_feature_inference_load_only", results["capacity"]["classification"])
        self.assertGreater(len(results["disagreement"]["sample_rows"]), 0)
        self.assertEqual({"row_id", "stddev"}, set(results["disagreement"]["sample_rows"][0]))
        self.assertFalse((self.bundle / "generated.csv").exists())

    def test_report_exposes_review_and_evidence_navigation_with_queue_errors(self):
        from yk_evidence.algorithm import render_algorithm_report
        page = render_algorithm_report(self.bundle, csrf_token="fake-test-token").decode()
        for expected in ('action="/api/reviews"', 'href="/api/export"',
                         'href="/api/manifest"', 'href="/api/reviews"',
                         'False positives', 'False negatives', 'Threshold alerts',
                         'Deferred', 'Precision', 'Window drift diagnostics',
                         'PSI', 'name="csrf"'):
            self.assertIn(expected, page)
        self.assertIn('name="reviewer"', page)
        self.assertIn('name="decision"', page)
        self.assertIn('name="rationale"', page)

    def test_rehashed_weights_threshold_and_test_label_forgery_rejected(self):
        for name, mutate in (
            ("fusion_policy.json", lambda value: value["weights"].__setitem__(0, .123)),
            ("fusion_policy.json", lambda value: value.__setitem__("threshold", .123)),
            ("algorithm_scores.json", lambda value: value["test"][0].__setitem__("label", 1 - value["test"][0]["label"])),
        ):
            with self.subTest(name=name, mutate=mutate):
                shutil.copyfile(self.reference / name, self.bundle / name)
                obj = json.loads((self.bundle / name).read_text())
                mutate(obj)
                (self.bundle / name).write_text(json.dumps(obj))
                _rehash(self.bundle, name)
                self.assertTrue(verify_algorithm(self.bundle))
                shutil.copyfile(self.reference / name, self.bundle / name)
                _rehash(self.bundle, name)

    def test_rehashed_raw_score_without_decision_change_rejected(self):
        path = self.bundle / "algorithm_scores.json"
        obj = json.loads(path.read_text())
        obj["test"][0]["models"]["logistic_regression"]["raw_score"] = .999
        path.write_text(json.dumps(obj))
        _rehash(self.bundle, path.name)
        self.assertTrue(verify_algorithm(self.bundle))

    def test_rehashed_capacity_stream_and_partition_tie_claim_rejected(self):
        path = self.bundle / "algorithm_results.json"
        obj = json.loads(path.read_text())
        obj["capacity"]["workloads"][0]["score_stream_sha256"] = "0" * 64
        path.write_text(json.dumps(obj))
        _rehash(self.bundle, path.name)
        self.assertTrue(verify_algorithm(self.bundle))
        shutil.copyfile(self.reference / path.name, path)
        _rehash(self.bundle, path.name)
        path = self.bundle / "algorithm_models.json"
        obj = json.loads(path.read_text())
        obj["partitions"]["train"]["time_max"] = obj["partitions"]["calibration"]["time_min"]
        path.write_text(json.dumps(obj))
        _rehash(self.bundle, path.name)
        self.assertTrue(verify_algorithm(self.bundle))

    def test_rehashed_capacity_rate_and_rss_claims_rejected(self):
        path = self.bundle / "algorithm_results.json"
        original = path.read_bytes()
        for key, wrong in (("scores_per_second", 1e30), ("process_maxrss_source_unit", "furlongs"),
                           ("elapsed_seconds", True)):
            with self.subTest(key=key):
                obj = json.loads(original)
                obj["capacity"]["workloads"][0][key] = wrong
                path.write_text(json.dumps(obj))
                _rehash(self.bundle, path.name)
                self.assertTrue(verify_algorithm(self.bundle))
        path.write_bytes(original)
        _rehash(self.bundle, path.name)

    def test_rehashed_fit_timing_claim_rejected(self):
        path = self.bundle / "algorithm_models.json"
        obj = json.loads(path.read_text())
        obj["fit_seconds"]["xgboost"] = -12.0
        path.write_text(json.dumps(obj))
        _rehash(self.bundle, path.name)
        self.assertTrue(verify_algorithm(self.bundle))

    def test_rehashed_window_psi_bins_and_queue_counts_rejected(self):
        path = self.bundle / "algorithm_results.json"
        original = json.loads(path.read_text())
        for field in ("psi", "bin_counts", "reviewed_count"):
            with self.subTest(field=field):
                obj = json.loads(json.dumps(original))
                window = obj["window_monitoring"]["analyses"]["robust_weighted_fusion"]["windows"][0]
                if field == "psi":
                    window["psi"] += .25
                elif field == "bin_counts":
                    window["bin_counts"][0] += 1
                else:
                    window["queue"]["reviewed_count"] += 1
                path.write_text(json.dumps(obj))
                _rehash(self.bundle, path.name)
                self.assertTrue(verify_algorithm(self.bundle))

    def test_test_period_label_change_cannot_select_a_different_policy(self):
        changed = self.root / "changed-test-label.csv"
        with self.source.open(newline="") as handle:
            rows = list(csv.reader(handle))
        rows[-1][-1] = "1" if rows[-1][-1] == "0" else "0"
        with changed.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        source = {**self.metadata, "source_sha256": hashlib.sha256(changed.read_bytes()).hexdigest()}
        other = algorithm_run(self.root / "runs", "label-perturbation", changed, source,
                              seed=17, load_rows=256, budget_fraction=.1)
        frozen = json.loads((self.reference / "fusion_policy.json").read_text())
        changed_policy = json.loads((other / "fusion_policy.json").read_text())
        self.assertEqual(frozen, changed_policy)
        self.assertEqual([], verify_algorithm(other))

    def test_default_smoke_is_generated_and_verified(self):
        from yk_evidence.algorithm import algorithm_smoke
        bundle = algorithm_smoke(self.root / "smokes", "generated", seed=12, load_rows=128)
        self.assertEqual([], verify_algorithm(bundle))
        self.assertEqual("synthetic_smoke_fixture",
                         json.loads((bundle / "source_metadata.json").read_text())["classification"])

    def test_missing_and_extra_files_and_symlink_rejected(self):
        (self.bundle / "extra.txt").write_text("hidden")
        self.assertTrue(verify_algorithm(self.bundle))
        (self.bundle / "extra.txt").unlink()
        target = self.bundle / "source_profile.json"
        target.unlink()
        target.symlink_to(self.reference / "source_profile.json")
        self.assertTrue(verify_algorithm(self.bundle))

    def test_invalid_inputs_cannot_create_completed_bundle(self):
        root = self.root / "invalid"
        with self.assertRaises(ValueError):
            algorithm_run(root, "../escape", self.source, self.metadata, load_rows=100)
        with self.assertRaises(ValueError):
            algorithm_run(root, "bool-budget", self.source, self.metadata,
                          load_rows=100, budget_fraction=True)
        with self.assertRaises(ValueError):
            algorithm_run(root, "wrong-hash", self.source,
                          {**self.metadata, "source_sha256": "0" * 64}, load_rows=100)
        self.assertEqual([], list(root.iterdir()) if root.exists() else [])
        with self.assertRaises(FileExistsError):
            algorithm_run(self.root / "runs", "reference", self.source,
                          self.metadata, load_rows=100)


if __name__ == "__main__":
    unittest.main()
