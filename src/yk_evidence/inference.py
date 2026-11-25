"""Explicitly trusted, validation-frozen scoring for ULB-coordinate batches.

The input must use the same anonymous V1..V28 transformation as the source
bundle. Column names and numeric ranges cannot prove that provenance: the
operator attests to it. Joblib loading executes pickle code, so only models
from a separately trusted producer may be loaded. JSON hashes detect ordinary
corruption, not a coordinated attacker who can rewrite and rehash evidence.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from .algorithm import MODEL_NAMES, RESILIENCE_SCHEMA, verify_algorithm
from .evaluation import FEATURES, _score
from .fusion import apply_policy
from .research import read, sha, write

INFERENCE_SCHEMA = "0.6-inference"
INPUT_COLUMNS = ["row_id", "Time", *FEATURES]
ARTIFACTS = ("inference_metadata.json", "inference_scores.json", "inference_queue.json")
MANIFEST = "bundle_manifest.json"
METADATA_KEYS = frozenset({
    "schema_version", "run_id", "source_run_id", "source_manifest_sha256",
    "source_policy_sha256", "source_model_sha256", "source_model_packages",
    "source_calibrators", "source_classification", "source_sha256",
    "input_sha256", "input_row_count", "input_row_ids_sha256",
    "input_relative_time_range", "input_feature_columns",
    "feature_coordinate_verification", "feature_coordinate_note",
    "raw_input_included", "frozen_policy",
})
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_COORDINATE_NOTE = (
    "Operator attests that V1..V28 and Amount use the same transformed "
    "feature coordinates as the source bundle; this cannot be proved from "
    "the CSV schema or values alone. Research scoring only."
)


def _read_input(csv_path: Path) -> tuple[np.ndarray, list[int], list[float]]:
    if not csv_path.is_file():
        raise ValueError(f"input CSV not found: {csv_path}")
    ids: list[int] = []
    times: list[float] = []
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != INPUT_COLUMNS:
            raise ValueError("input schema must be row_id,Time,V1..V28,Amount exactly; Class and extra columns are forbidden")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(INPUT_COLUMNS) or not re.fullmatch(r"[0-9]+", row[0]):
                raise ValueError(f"CSV row {line_number}: invalid width or row_id")
            try:
                row_id = int(row[0])
                relative_time = float(row[1])
                values = [float(value) for value in row[2:]]
            except ValueError as exc:
                raise ValueError(f"CSV row {line_number}: nonnumeric feature") from exc
            if not math.isfinite(relative_time) or relative_time < 0 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"CSV row {line_number}: nonfinite feature or negative Time")
            ids.append(row_id)
            times.append(relative_time)
            rows.append(values)
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("input requires at least one row and unique stable row_id values")
    return np.asarray(rows, dtype=np.float64), ids, times


def _source_contract(source_bundle: Path) -> tuple[dict, dict, dict, dict]:
    errors = verify_algorithm(source_bundle)
    if errors:
        raise ValueError("source bundle failed safe verification: " + "; ".join(errors))
    manifest = read(source_bundle / MANIFEST)
    if manifest["schema_version"] != RESILIENCE_SCHEMA:
        raise ValueError("inference requires a 0.6 algorithm bundle")
    model_record = read(source_bundle / "algorithm_models.json")
    policy = read(source_bundle / "fusion_policy.json")
    run_record = read(source_bundle / "run_metadata.json")
    if model_record.get("schema_version") != RESILIENCE_SCHEMA or model_record.get("feature_columns") != FEATURES:
        raise ValueError("source model feature schema differs from ULB V1..V28,Amount")
    if model_record.get("model_names") != list(MODEL_NAMES) or policy.get("model_names") != list(MODEL_NAMES):
        raise ValueError("source model names disagree")
    try:
        import xgboost  # xgboost-cpu has the same import name on Linux.
        versions = {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                    "xgboost": xgboost.__version__}
    except (ImportError, OSError) as exc:
        raise ValueError("required model runtime is unavailable") from exc
    if model_record.get("packages") != versions:
        raise ValueError("saved model runtime versions differ from the installed environment")
    weights = policy.get("weights")
    if (not isinstance(weights, list) or len(weights) != len(MODEL_NAMES)
            or any(isinstance(weight, bool) or not isinstance(weight, (int, float))
                   or not math.isfinite(weight) or weight < 0 for weight in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("invalid frozen weights")
    # apply_policy validates the saved threshold and budget without refitting.
    apply_policy([0.5], [0], policy.get("threshold"), policy.get("budget_fraction"))
    return manifest, model_record, policy, run_record


def _predict(source_bundle: Path, matrix: np.ndarray) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with threadpool_limits(limits=1):
            for name in MODEL_NAMES:
                saved = joblib.load(source_bundle / f"model_{name}.joblib")
                if (not isinstance(saved, dict) or set(saved) != {"estimator", "calibrator"}
                        or not isinstance(saved["calibrator"], CalibratedClassifierCV)
                        or getattr(saved["estimator"], "n_features_in_", None) != len(FEATURES)
                        or getattr(saved["calibrator"], "n_features_in_", None) != len(FEATURES)):
                    raise ValueError(f"saved model {name} has incompatible feature or calibrator schema")
                raw, calibration_input, calibrated = _score(saved["estimator"], saved["calibrator"], matrix)
                if any(np.asarray(values).shape != (len(matrix),) for values in (raw, calibration_input, calibrated)):
                    raise ValueError(f"saved model {name} returned an invalid probability shape")
                entries = [{"raw_score": float(raw[index]),
                            "calibration_input": float(calibration_input[index]),
                            "calibrated_score": float(calibrated[index])}
                           for index in range(len(matrix))]
                if any(not _probability(entry["raw_score"]) or
                       not math.isfinite(entry["calibration_input"]) or
                       not _probability(entry["calibrated_score"])
                       for entry in entries):
                    raise ValueError(f"saved model {name} returned invalid scores")
                result[name] = entries
    return result


def _row_id_digest(ids: list[int]) -> str:
    return hashlib.sha256("\n".join(str(row_id) for row_id in ids).encode("ascii")).hexdigest()


def run_inference(source_bundle: Path, input_csv: Path, output_root: Path, run_id: str, *,
                  trusted_model_load: bool = False, feature_space_attested: bool = False) -> Path:
    """Score an unlabeled batch; publish only after complete JSON verification.

    ``trusted_model_load`` means the caller trusts the producer of every
    joblib file, beyond its self-asserted digest. ``feature_space_attested``
    means the caller knows the input uses the source ULB transformation.
    """
    if trusted_model_load is not True:
        raise ValueError("explicit trusted_model_load=True required before pickle loading")
    if feature_space_attested is not True:
        raise ValueError("explicit same-coordinate feature_space_attested=True required")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 safe filename characters")
    source_bundle, input_csv, output_root = Path(source_bundle), Path(input_csv), Path(output_root)
    if output_root.is_symlink():
        raise ValueError("output root may not be a symbolic link")
    final = output_root / run_id
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"inference run already exists: {final}")
    manifest, models, policy, source_run = _source_contract(source_bundle)
    input_sha256 = sha(input_csv)
    matrix, ids, times = _read_input(input_csv)
    if sha(input_csv) != input_sha256:
        raise ValueError("input changed during CSV validation")
    model_scores = _predict(source_bundle, matrix)
    if sha(input_csv) != input_sha256:
        raise ValueError("input changed during model scoring")
    weights = [float(value) for value in policy["weights"]]
    fused = [sum(weights[index] * model_scores[name][row_index]["calibrated_score"]
                 for index, name in enumerate(MODEL_NAMES)) for row_index in range(len(ids))]
    queue = apply_policy(fused, ids, policy["threshold"], policy["budget_fraction"])
    rows = [{"row_id": row_id, "relative_time": times[index],
             "model_scores": {name: model_scores[name][index] for name in MODEL_NAMES},
             "fusion_score": fused[index],
             "threshold_alerted": queue["threshold_alerted"][index],
             "reviewed": queue["reviewed"][index], "deferred": queue["deferred"][index]}
            for index, row_id in enumerate(ids)]
    summary = {key: queue[key] for key in ("capacity", "threshold_alert_count", "reviewed_count",
                                           "deferred_count", "budget_utilization")}
    summary.update({"threshold_alerted_row_ids": [row_id for index, row_id in enumerate(ids)
                                                  if queue["threshold_alerted"][index]],
                    "reviewed_row_ids": [row_id for index, row_id in enumerate(ids)
                                         if queue["reviewed"][index]],
                    "deferred_row_ids": [row_id for index, row_id in enumerate(ids)
                                         if queue["deferred"][index]]})
    metadata = {
        "schema_version": INFERENCE_SCHEMA, "run_id": run_id,
        "source_run_id": source_run["run_id"],
        "source_manifest_sha256": sha(source_bundle / MANIFEST),
        "source_policy_sha256": manifest["artifacts"]["fusion_policy.json"],
        "source_model_sha256": {name: manifest["artifacts"][f"model_{name}.joblib"] for name in MODEL_NAMES},
        "source_model_packages": models["packages"],
        "source_calibrators": models["calibrators"],
        "source_classification": source_run["source_classification"],
        "source_sha256": source_run["source_sha256"],
        "input_sha256": input_sha256, "input_row_count": len(ids),
        "input_row_ids_sha256": _row_id_digest(ids),
        "input_relative_time_range": [min(times), max(times)],
        "input_feature_columns": list(FEATURES),
        "feature_coordinate_verification": "operator_attestation_only",
        "feature_coordinate_note": _COORDINATE_NOTE,
        "raw_input_included": False,
        "frozen_policy": {"model_names": list(MODEL_NAMES), "weights": weights,
                          "threshold": policy["threshold"],
                          "budget_fraction": policy["budget_fraction"]},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ("." + run_id + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix="." + run_id + "-", dir=output_root))
        write(stage / "inference_metadata.json", metadata)
        write(stage / "inference_scores.json", rows)
        write(stage / "inference_queue.json", summary)
        write(stage / MANIFEST, {"schema_version": INFERENCE_SCHEMA, "run_id": run_id,
                                 "required_artifacts": list(ARTIFACTS),
                                 "artifacts": {name: sha(stage / name) for name in ARTIFACTS}})
        errors = verify_inference(stage, source_bundle)
        if errors:
            raise ValueError("inference evidence failed verification: " + "; ".join(errors))
        if sha(input_csv) != input_sha256:
            raise ValueError("input changed before inference finalization")
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"inference run already exists: {final}")
        stage.rename(final)
        stage = None
        return final
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def _probability(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def verify_inference(bundle: Path, source_bundle: Path | None = None) -> list[str]:
    """Check sealed hashes, fusion arithmetic and queue without loading pickle.

    With ``source_bundle``, also verify its safe JSON evidence and tie the
    recorded source/policy/model digests back to those exact files.
    """
    bundle = Path(bundle)
    errors: list[str] = []
    try:
        if not bundle.is_dir() or bundle.is_symlink():
            return ["unsafe inference bundle"]
        if {path.name for path in bundle.iterdir()} != {*ARTIFACTS, MANIFEST}:
            return ["inference artifact allowlist differs"]
        if any((bundle / name).is_symlink() or not (bundle / name).is_file()
               for name in (*ARTIFACTS, MANIFEST)):
            return ["unsafe inference artifact"]
        manifest = read(bundle / MANIFEST)
        if (manifest.get("schema_version") != INFERENCE_SCHEMA
                or set(manifest) != {"schema_version", "run_id", "required_artifacts", "artifacts"}
                or manifest.get("required_artifacts") != list(ARTIFACTS)
                or set(manifest.get("artifacts", {})) != set(ARTIFACTS)):
            return ["invalid inference manifest contract"]
        for name in ARTIFACTS:
            digest = manifest["artifacts"][name]
            if not isinstance(digest, str) or not _HEX64.fullmatch(digest) or sha(bundle / name) != digest:
                errors.append(f"artifact hash differs: {name}")
        if errors:
            return errors
        metadata = read(bundle / "inference_metadata.json")
        rows = read(bundle / "inference_scores.json")
        summary = read(bundle / "inference_queue.json")
        if (set(metadata) != METADATA_KEYS
                or metadata.get("schema_version") != INFERENCE_SCHEMA or metadata.get("run_id") != manifest.get("run_id")
                or metadata.get("input_feature_columns") != FEATURES or metadata.get("raw_input_included") is not False
                or metadata.get("feature_coordinate_verification") != "operator_attestation_only"
                or metadata.get("feature_coordinate_note") != _COORDINATE_NOTE):
            errors.append("inference metadata contract differs")
        packages = metadata.get("source_model_packages")
        if (not isinstance(packages, dict) or set(packages) != {"scikit-learn", "numpy", "xgboost"}
                or any(not isinstance(value, str) or not value for value in packages.values())):
            errors.append("source package versions invalid")
        if metadata.get("source_classification") not in {"public_ulb_creditcard", "synthetic_smoke_fixture"}:
            errors.append("source classification invalid")
        for field in ("source_manifest_sha256", "source_policy_sha256", "source_sha256",
                      "input_sha256", "input_row_ids_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                errors.append(f"invalid {field}")
        source_hashes = metadata.get("source_model_sha256")
        if (not isinstance(source_hashes, dict) or set(source_hashes) != set(MODEL_NAMES)
                or any(not isinstance(value, str) or not _HEX64.fullmatch(value)
                       for value in source_hashes.values())):
            errors.append("source model hash contract differs")
        calibrators = metadata.get("source_calibrators")
        if (not isinstance(calibrators, dict) or set(calibrators) != set(MODEL_NAMES)
                or any(not isinstance(entry, dict)
                       or set(entry) != {"method", "a", "b", "input_kind"}
                       or entry["method"] != "sigmoid"
                       or entry["input_kind"] not in ("predict_proba", "decision_function")
                       or any(isinstance(entry[key], bool) or not isinstance(entry[key], (int, float))
                              or not math.isfinite(entry[key]) for key in ("a", "b"))
                       for entry in calibrators.values())):
            errors.append("source calibrator contract differs")
        policy = metadata.get("frozen_policy", {})
        weights = policy.get("weights") if isinstance(policy, dict) else None
        if (not isinstance(weights, list) or len(weights) != len(MODEL_NAMES)
                or policy.get("model_names") != list(MODEL_NAMES)
                or any(not _probability(value) for value in weights)
                or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12)):
            errors.append("invalid frozen model weights")
        if not isinstance(rows, list) or not rows or metadata.get("input_row_count") != len(rows):
            errors.append("input row count differs")
        if errors:
            return errors
        ids: list[int] = []
        times: list[float] = []
        fused: list[float] = []
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != {"row_id", "relative_time", "model_scores", "fusion_score",
                                                      "threshold_alerted", "reviewed", "deferred"}):
                errors.append(f"score row {index} contract differs")
                continue
            row_id, relative_time, model_scores = row["row_id"], row["relative_time"], row["model_scores"]
            if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
                errors.append(f"score row {index} ID invalid")
                continue
            if (isinstance(relative_time, bool) or not isinstance(relative_time, (int, float))
                    or not math.isfinite(relative_time) or relative_time < 0):
                errors.append(f"score row {index} Time invalid")
                continue
            if not isinstance(model_scores, dict) or set(model_scores) != set(MODEL_NAMES):
                errors.append(f"score row {index} model scores invalid")
                continue
            for name in MODEL_NAMES:
                entry = model_scores[name]
                if (not isinstance(entry, dict)
                        or set(entry) != {"raw_score", "calibration_input", "calibrated_score"}
                        or not _probability(entry["raw_score"])
                        or not isinstance(entry["calibration_input"], (int, float))
                        or isinstance(entry["calibration_input"], bool)
                        or not math.isfinite(entry["calibration_input"])
                        or not _probability(entry["calibrated_score"])):
                    errors.append(f"score row {index} {name} values invalid")
                    continue
                params = calibrators[name]
                if params["input_kind"] == "predict_proba":
                    if not math.isclose(entry["raw_score"], entry["calibration_input"], abs_tol=1e-12):
                        errors.append(f"score row {index} {name} calibration input differs")
                elif not math.isclose(entry["raw_score"], float(expit(entry["calibration_input"])),
                                      rel_tol=1e-10, abs_tol=1e-10):
                    errors.append(f"score row {index} {name} raw score differs")
                predicted = float(expit(-(params["a"] * entry["calibration_input"] + params["b"])))
                if not math.isclose(predicted, entry["calibrated_score"], rel_tol=1e-10, abs_tol=1e-10):
                    errors.append(f"score row {index} {name} calibrator arithmetic differs")
            if errors:
                continue
            calculated = sum(weights[position] * model_scores[name]["calibrated_score"]
                             for position, name in enumerate(MODEL_NAMES))
            if not _probability(row["fusion_score"]) or row["fusion_score"] != calculated:
                errors.append(f"score row {index} fusion arithmetic differs")
            if any(type(row[key]) is not bool for key in ("threshold_alerted", "reviewed", "deferred")):
                errors.append(f"score row {index} queue flag invalid")
            ids.append(row_id)
            times.append(float(relative_time))
            fused.append(calculated)
        if errors:
            return errors
        if len(set(ids)) != len(ids) or metadata.get("input_row_ids_sha256") != _row_id_digest(ids):
            errors.append("row IDs are duplicated or digest differs")
        if metadata.get("input_relative_time_range") != [min(times), max(times)]:
            errors.append("input Time range differs")
        expected = apply_policy(fused, ids, policy.get("threshold"), policy.get("budget_fraction"))
        for index, row in enumerate(rows):
            for key in ("threshold_alerted", "reviewed", "deferred"):
                if row[key] != expected[key][index]:
                    errors.append(f"score row {index} {key} differs")
        expected_summary = {key: expected[key] for key in ("capacity", "threshold_alert_count", "reviewed_count",
                                                        "deferred_count", "budget_utilization")}
        for key in ("threshold_alerted", "reviewed", "deferred"):
            expected_summary[key + "_row_ids"] = [row_id for index, row_id in enumerate(ids) if expected[key][index]]
        if summary != expected_summary:
            errors.append("review queue differs from frozen policy")
        if source_bundle is not None:
            source_bundle = Path(source_bundle)
            source_errors = verify_algorithm(source_bundle)
            if source_errors:
                errors.append("source algorithm bundle differs: " + "; ".join(source_errors))
            else:
                source_manifest = read(source_bundle / MANIFEST)
                source_models = read(source_bundle / "algorithm_models.json")
                source_policy = read(source_bundle / "fusion_policy.json")
                source_run = read(source_bundle / "run_metadata.json")
                if (metadata.get("source_manifest_sha256") != sha(source_bundle / MANIFEST)
                        or metadata.get("source_policy_sha256") != source_manifest["artifacts"]["fusion_policy.json"]
                        or metadata.get("source_model_sha256") !=
                        {name: source_manifest["artifacts"][f"model_{name}.joblib"] for name in MODEL_NAMES}
                        or metadata.get("source_model_packages") != source_models["packages"]
                        or metadata.get("source_calibrators") != source_models["calibrators"]
                        or metadata.get("source_run_id") != source_run["run_id"]
                        or metadata.get("source_classification") != source_run["source_classification"]
                        or metadata.get("source_sha256") != source_run["source_sha256"]
                        or policy != {"model_names": list(MODEL_NAMES), "weights": source_policy["weights"],
                                      "threshold": source_policy["threshold"],
                                      "budget_fraction": source_policy["budget_fraction"]}):
                    errors.append("inference source or frozen policy provenance differs")
        return errors
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return errors + [f"invalid inference evidence: {exc}"]
