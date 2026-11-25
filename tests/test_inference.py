"""Trusted batch scoring of a frozen 0.6 algorithm bundle."""

import csv
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yk_evidence.algorithm import algorithm_smoke
from yk_evidence.inference import run_inference, verify_inference


def _batch(path: Path, rows: int = 200, *, header=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["row_id", "Time", *(f"V{i}" for i in range(1, 29)), "Amount"])
        for index in range(rows):
            label = int(index % 8 == 0)
            features = [round((index % (7 + j % 5)) * .1 + label * (j % 3), 3)
                        for j in range(1, 29)]
            writer.writerow([1000 + index, index // 2, *features, index % 100])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.source_bundle = algorithm_smoke(Path(cls.fixture.name) / "source", "smoke", load_rows=256)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv = self.root / "unlabelled.csv"
        _batch(self.csv)
        self.outputs = self.root / "outputs"

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, *, source=None, csv=None, trusted=True, attested=True):
        return run_inference(source or self.source_bundle, csv or self.csv,
                             self.outputs, "new-batch", trusted_model_load=trusted,
                             feature_space_attested=attested)

    def test_scores_real_saved_models_with_frozen_policy_and_sealed_provenance(self):
        # Removing model prediction or using a newly fitted policy must fail here.
        output = self._run()
        self.assertEqual([], verify_inference(output, self.source_bundle))
        self.assertEqual([], verify_inference(output))
        metadata = json.loads((output / "inference_metadata.json").read_text())
        scores = json.loads((output / "inference_scores.json").read_text())
        queue = json.loads((output / "inference_queue.json").read_text())
        source_policy = json.loads((self.source_bundle / "fusion_policy.json").read_text())
        source_manifest = self.source_bundle / "bundle_manifest.json"
        self.assertEqual(hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                         metadata["source_manifest_sha256"])
        self.assertEqual(hashlib.sha256(self.csv.read_bytes()).hexdigest(),
                         metadata["input_sha256"])
        self.assertEqual(source_policy["weights"], metadata["frozen_policy"]["weights"])
        self.assertEqual(source_policy["threshold"], metadata["frozen_policy"]["threshold"])
        self.assertEqual(200, len(scores))
        self.assertEqual(1000, scores[0]["row_id"])
        self.assertEqual(4, len(scores[0]["model_scores"]))
        for score in scores[0]["model_scores"].values():
            self.assertEqual({"raw_score", "calibration_input", "calibrated_score"}, set(score))
            self.assertGreaterEqual(score["raw_score"], 0)
            self.assertLessEqual(score["raw_score"], 1)
        self.assertEqual(1, queue["capacity"])
        self.assertEqual(queue["reviewed_count"], len(queue["reviewed_row_ids"]))
        self.assertFalse((output / self.csv.name).exists())
        self.assertNotIn("accuracy", metadata)

    def test_no_trust_or_feature_attestation_refuses_before_output(self):
        for trust, attested in ((False, True), (True, False)):
            with self.subTest(trust=trust, attested=attested):
                with self.assertRaises(ValueError):
                    self._run(trusted=trust, attested=attested)
                self.assertFalse((self.outputs / "new-batch").exists())

    def test_module_version_accepts_xgboost_cpu_distribution_name(self):
        # Linux pins xgboost-cpu; the import is xgboost, not necessarily its
        # distribution metadata name. A package-name lookup must not block it.
        with patch("importlib.metadata.version",
                   side_effect=importlib.metadata.PackageNotFoundError):
            output = self._run()
        self.assertEqual([], verify_inference(output, self.source_bundle))

    def test_label_extra_column_duplicate_ids_and_bad_numbers_fail_closed(self):
        cases = (
            (["row_id", "Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"],
             "label"),
            (["row_id", "Time", "V2", "V1", *(f"V{i}" for i in range(3, 29)), "Amount"],
             "schema"),
        )
        for header, name in cases:
            with self.subTest(name=name):
                _batch(self.csv, header=header)
                with self.assertRaises(ValueError):
                    self._run()
                self.assertFalse((self.outputs / "new-batch").exists())
        _batch(self.csv)
        lines = self.csv.read_text().splitlines()
        lines[2] = lines[1]
        self.csv.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse((self.outputs / "new-batch").exists())
        _batch(self.csv)
        lines = self.csv.read_text().splitlines()
        fields = lines[1].split(",")
        fields[2] = "NaN"
        lines[1] = ",".join(fields)
        self.csv.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            self._run()
        self.assertFalse((self.outputs / "new-batch").exists())

    def test_changed_source_manifest_or_runtime_versions_refuse_before_model_load(self):
        bad = self.root / "bad-source"
        shutil.copytree(self.source_bundle, bad)
        (bad / "model_logistic_regression.joblib").write_bytes(b"untrusted")
        with patch("joblib.load", side_effect=AssertionError("untrusted model was loaded")):
            with self.assertRaises(ValueError):
                self._run(source=bad)
        self.assertFalse((self.outputs / "new-batch").exists())
        shutil.rmtree(bad)
        shutil.copytree(self.source_bundle, bad)
        models = json.loads((bad / "algorithm_models.json").read_text())
        models["packages"]["scikit-learn"] = "0.0"
        model_path = bad / "algorithm_models.json"
        model_path.write_text(json.dumps(models))
        manifest_path = bad / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][model_path.name] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with patch("joblib.load", side_effect=AssertionError("incompatible model was loaded")):
            with self.assertRaises(ValueError):
                self._run(source=bad)
        self.assertFalse((self.outputs / "new-batch").exists())

    def test_verifier_rejects_rehashed_score_and_queue_forgery_without_pickle(self):
        output = self._run()
        for filename, mutate in (
            ("inference_scores.json", lambda data: data[0].__setitem__("fusion_score", 0.9)),
            ("inference_queue.json", lambda data: data.__setitem__("capacity", 99)),
        ):
            with self.subTest(filename=filename):
                clone = self.root / filename.replace(".json", "")
                shutil.copytree(output, clone)
                path = clone / filename
                data = json.loads(path.read_text())
                mutate(data)
                path.write_text(json.dumps(data))
                manifest_path = clone / "bundle_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["artifacts"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with patch("joblib.load", side_effect=AssertionError("verifier loaded pickle")):
                    self.assertTrue(verify_inference(clone, self.source_bundle))

    def test_rehashed_raw_score_forgery_breaks_calibration_relationship_without_pickle(self):
        output = self._run()
        score_path = output / "inference_scores.json"
        rows = json.loads(score_path.read_text())
        rows[0]["model_scores"]["logistic_regression"]["raw_score"] = 0.99
        score_path.write_text(json.dumps(rows))
        manifest_path = output / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][score_path.name] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with patch("joblib.load", side_effect=AssertionError("verifier loaded pickle")):
            self.assertTrue(verify_inference(output, self.source_bundle))

    def test_rehashed_metadata_cannot_invent_unlabelled_accuracy(self):
        output = self._run()
        metadata_path = output / "inference_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["accuracy"] = 0.99
        metadata_path.write_text(json.dumps(metadata))
        manifest_path = output / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][metadata_path.name] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(verify_inference(output))


if __name__ == "__main__":
    unittest.main()
