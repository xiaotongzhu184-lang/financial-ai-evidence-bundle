"""Chronological, leakage-controlled tabular evaluation for the ULB CSV schema.

The ULB file has relative ``Time`` and anonymized features. It has no account,
merchant, graph, or absolute timestamp fields. This module never infers them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from threadpoolctl import threadpool_limits

try:
    import resource  # POSIX peak RSS; unavailable in the Windows standard library.
except ImportError:
    resource = None


FEATURES = [*(f"V{i}" for i in range(1, 29)), "Amount"]
HEADERS = ["Time", *FEATURES, "Class"]
PARTITIONS = ("train", "calibration", "validation", "test")
MODELS = ("logistic_regression", "random_forest", "hist_gradient_boosting")
STRESS_SCENARIOS = ("v1_v2_missing_10pct", "v1_v2_noise_10pct", "amount_x1_2")
PREDICTION_FIELDS = ("model_id", "partition", "row_id", "relative_time", "label", "raw_score",
                     "calibration_input", "calibrated_score", "decision")
SCHEMA_VERSION = "0.4-evaluation"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _read_ulb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"source CSV not found: {path}")
    values = array("d")
    times = array("d")
    labels = bytearray()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != HEADERS:
            raise ValueError("expected ULB columns Time,V1..V28,Amount,Class in that order")
        for row_id, row in enumerate(reader, start=2):
            if len(row) != len(HEADERS):
                raise ValueError(f"CSV row {row_id}: expected {len(HEADERS)} columns")
            try:
                when = float(row[0])
                features = [float(value) for value in row[1:-1]]
            except ValueError as exc:
                raise ValueError(f"CSV row {row_id}: nonnumeric time or feature") from exc
            if not math.isfinite(when) or when < 0 or not all(math.isfinite(value) for value in features):
                raise ValueError(f"CSV row {row_id}: nonfinite or negative time/feature")
            if row[-1] not in ("0", "1"):
                raise ValueError(f"CSV row {row_id}: Class must be 0 or 1")
            times.append(when)
            values.extend(features)
            labels.append(int(row[-1]))
    count = len(labels)
    if count < 80:
        raise ValueError("evaluation requires at least 80 source rows")
    x = np.frombuffer(values, dtype=np.float64).reshape(count, len(FEATURES))
    t = np.frombuffer(times, dtype=np.float64)
    y = np.frombuffer(labels, dtype=np.uint8)
    row_ids = np.arange(2, count + 2, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    return x[order], y[order], t[order], row_ids[order]


def _boundaries(times: np.ndarray, labels: np.ndarray) -> dict[str, slice]:
    count = len(times)
    boundaries = [0]
    for proportion in (0.55, 0.70, 0.85):
        cut = max(boundaries[-1] + 1, int(count * proportion))
        while cut < count and times[cut - 1] == times[cut]:
            cut += 1
        boundaries.append(cut)
    boundaries.append(count)
    parts = {name: slice(left, right) for name, left, right in zip(PARTITIONS, boundaries[:-1], boundaries[1:])}
    if any(part.start >= part.stop for part in parts.values()):
        raise ValueError("time ties leave an empty chronological partition")
    for name, part in parts.items():
        if len(np.unique(labels[part])) != 2:
            raise ValueError(f"{name} partition needs both Class values for defined AP and FPR")
        if name == "calibration" and min(np.bincount(labels[part], minlength=2)) < 2:
            raise ValueError("calibration partition needs at least two rows from each class")
    return parts


def _threshold(labels: np.ndarray, scores: np.ndarray, ceiling: float) -> float:
    negatives = np.sort(scores[labels == 0])[::-1]
    permitted = int(math.floor(len(negatives) * ceiling))
    return float(negatives[permitted]) if permitted < len(negatives) else -1.0


def _wilson(numerator: int, denominator: int) -> list[float] | None:
    if denominator == 0:
        return None
    z = 1.959963984540054
    proportion = numerator / denominator
    scale = 1 + z * z / denominator
    centre = (proportion + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(proportion * (1 - proportion) / denominator + z * z / (4 * denominator * denominator)) / scale
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if len(labels) != len(scores) or len(labels) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("metrics require finite aligned scores and labels")
    if np.min(scores) < 0 or np.max(scores) > 1 or len(np.unique(labels)) != 2:
        raise ValueError("scores must be probabilities and labels must include both classes")
    decisions = scores > threshold
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(decisions & truth))
    fp = int(np.count_nonzero(decisions & ~truth))
    fn = int(np.count_nonzero(~decisions & truth))
    tn = int(np.count_nonzero(~decisions & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": len(labels)},
        "review_workload": {
            "alert_count": tp + fp,
            "alerts_per_10000": 10_000 * (tp + fp) / len(labels),
            "missed_cases": fn,
            "monetary_cost_estimated": False,
        },
        "wilson_95": {"recall": _wilson(tp, tp + fn), "fpr": _wilson(fp, fp + tn), "precision": _wilson(tp, tp + fp)},
    }


def _models(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=300, random_state=seed),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=4, max_features="sqrt",
                                   class_weight="balanced_subsample", n_jobs=2, random_state=seed),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, min_samples_leaf=10,
                                           learning_rate=0.08, early_stopping=False, random_state=seed),
        ),
    }


def _score(model: object, calibrated: object, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = model.predict_proba(x)[:, 1]
    calibration_input = model.decision_function(x) if hasattr(model, "decision_function") else raw
    return raw, calibration_input, calibrated.predict_proba(x)[:, 1]


def _prediction_digest(rows: list[dict]) -> str:
    fields = [(int(row["row_id"]), float(row["relative_time"]), int(row["label"]),
               float(row["raw_score"]), float(row["calibration_input"]),
               float(row["calibrated_score"]), int(row["decision"])) for row in rows]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _stress(model: object, calibrated: object, x: np.ndarray, y: np.ndarray, row_ids: np.ndarray,
            seed: int, threshold: float) -> dict:
    rng = np.random.default_rng(seed + 1109)
    affected = np.sort(rng.choice(len(x), size=max(1, int(len(x) * 0.10)), replace=False))
    _, _, baseline_scores = _score(model, calibrated, x)
    baseline_metrics = _metrics(y, baseline_scores, threshold)
    maximum_allowed_drop = 0.10
    variants = {}
    for name in STRESS_SCENARIOS:
        changed = x.copy()
        if name == "v1_v2_missing_10pct":
            changed[affected, :2] = np.nan
            count = len(affected)
        elif name == "v1_v2_noise_10pct":
            scale = np.maximum(np.std(x[:, :2], axis=0), 1e-12)
            changed[affected, :2] += rng.normal(0, 0.1 * scale, size=(len(affected), 2))
            count = len(affected)
        else:
            changed[:, -1] *= 1.2
            count = len(x)
        _, _, scores = _score(model, calibrated, changed)
        metrics = _metrics(y, scores, threshold)
        observed_drop = baseline_metrics["average_precision"] - metrics["average_precision"]
        variants[name] = {
            "changed_rows": int(count), "metrics": metrics,
            "observed_average_precision_drop": observed_drop,
            "maximum_allowed_average_precision_drop": maximum_allowed_drop,
            "status": "pass" if observed_drop <= maximum_allowed_drop else "fail",
            "predictions": [{"row_id": int(row_id), "label": int(label), "score": float(score)}
                            for row_id, label, score in zip(row_ids, y, scores)],
        }
    return {
        "classification": "feature_perturbation_on_fixed_test_labels",
        "assumptions": "Feature changes are controlled probes, not observed future transactions or newly verified fraud labels.",
        "threshold_provenance": "Fixed demonstration AP-drop ceiling 0.10, carried from v0.3 stress replay before the v0.4 final run; not a production limit.",
        "baseline_metrics": baseline_metrics,
        "seed": seed + 1109, "scenarios": variants,
    }


def _capacity(calibrated: object, x: np.ndarray, load_rows: int) -> dict:
    if load_rows < 1:
        raise ValueError("load_rows must be positive")
    targets = sorted(set((min(100_000, load_rows), load_rows)))
    workloads = []
    rss_unit = "unavailable" if resource is None else "kibibytes" if platform.system() == "Linux" else "bytes"
    rss_multiplier = 1024 if rss_unit == "kibibytes" else 1
    for target in targets:
        digest = hashlib.sha256()
        start = time.perf_counter()
        processed = 0
        while processed < target:
            size = min(8192, target - processed)
            indices = np.arange(processed, processed + size) % len(x)
            scores = calibrated.predict_proba(x[indices])[:, 1].astype("<f8", copy=False)
            digest.update(scores.tobytes())
            processed += size
        elapsed = time.perf_counter() - start
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        workloads.append({"scored_rows": target, "batch_size_max": 8192, "elapsed_seconds": elapsed,
                          "scores_per_second": target / elapsed, "score_stream_sha256": digest.hexdigest(),
                          "process_maxrss_raw": raw_rss, "process_maxrss_source_unit": rss_unit,
                          "process_maxrss_bytes": raw_rss * rss_multiplier if raw_rss is not None else None})
    return {"classification": "repeated_feature_inference_load_only", "independent_accuracy_rows": 0,
            "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                        "cpu_count": os.cpu_count(), "python": platform.python_version(),
                        "sklearn": sklearn.__version__, "threadpool_limit": 2},
            "limitations": "Rows are repeated from the held-out feature matrix. Throughput is machine-specific and is not accuracy evidence.",
            "workloads": workloads}


def evaluate_dataset(csv_path: Path, metadata: dict, output_dir: Path, seed: int = 42,
                     fpr_ceiling: float = 0.01, load_rows: int = 1_000_000) -> dict:
    """Fit on train, calibrate separately, select on validation, and seal a test report.

    ``output_dir`` must not exist. The whole run is staged before an atomic
    directory rename, so an interrupted fit is not mistaken for a final run.
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    if (not isinstance(metadata, dict)
            or any(not isinstance(metadata.get(key), str) or not metadata[key].strip()
                   for key in ("classification", "source_url", "license_status"))):
        raise ValueError("metadata requires classification, source_url and license_status")
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            or isinstance(load_rows, bool) or not isinstance(load_rows, int) or not 1 <= load_rows <= 10_000_000
            or isinstance(fpr_ceiling, bool) or not isinstance(fpr_ceiling, (int, float))
            or not math.isfinite(fpr_ceiling) or not 0 <= fpr_ceiling < 1):
        raise ValueError("invalid seed, fpr_ceiling or load_rows")
    existing_empty_dir = output_dir.is_dir() and not output_dir.is_symlink() and not any(output_dir.iterdir())
    if output_dir.exists() and not existing_empty_dir:
        raise FileExistsError(f"evaluation output already exists: {output_dir}")
    x, y, relative_time, row_ids = _read_ulb(csv_path)
    parts = _boundaries(relative_time, y)
    source_hash = _sha_file(csv_path)
    claimed_hash = metadata.get("source_sha256")
    if claimed_hash is not None and claimed_hash != source_hash:
        raise ValueError("source SHA-256 hash does not match metadata claim")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {"metadata": metadata, "source_file": csv_path.name, "source_sha256": source_hash,
                        "row_count": len(y), "feature_columns": FEATURES,
                        "time_semantics": "ULB relative Time; no absolute date or UTC inferred",
                        "row_id_semantics": "one-based CSV physical line number including header"},
            "config": {"seed": seed, "fpr_ceiling": fpr_ceiling, "load_rows": load_rows,
                       "split_targets": [0.55, 0.15, 0.15, 0.15],
                       "split_tie_rule": "move each boundary forward after the last equal relative Time",
                       "selection": "maximum validation average precision; fixed model order breaks ties",
                       "threshold": "validation negative score at index floor(FPR ceiling * negative count), descending; strict score > threshold",
                       "calibration": "sigmoid on separate calibration partition using frozen fitted estimator",
                       "preprocessing": "median imputation fitted on train only; StandardScaler on logistic train only"},
            "partitions": {
                name: {"row_count": part.stop - part.start, "time_min": float(relative_time[part.start]),
                       "time_max": float(relative_time[part.stop - 1]),
                       "row_ids_sha256": hashlib.sha256(row_ids[part].astype("<i8").tobytes()).hexdigest(),
                       "positive_count": int(np.sum(y[part]))}
                for name, part in parts.items()
            },
            "models": {},
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "numpy": np.__version__, "platform": platform.platform(),
                            "threadpool_limit": 2},
            "limitations": ["ULB rows contain no entity links; this is tabular evaluation, not graph validation.",
                            "The source CSV is external; its SHA-256 is recorded but its bytes are not redistributed.",
                            "A self-contained SHA-256 manifest is not an independent digital signature."],
        }
        all_predictions: list[dict] = []
        fitted = {}
        for name, model in _models(seed).items():
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                model.fit(x[parts["train"]], y[parts["train"]])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", cv=2, ensemble=False)
                calibrated.fit(x[parts["calibration"]], y[parts["calibration"]])
            calibration_seconds = time.perf_counter() - started
            with threadpool_limits(limits=2):
                scored = {partition: _score(model, calibrated, x[parts[partition]])
                          for partition in PARTITIONS[1:]}
            threshold = _threshold(y[parts["validation"]], scored["validation"][2], fpr_ceiling)
            model_rows = {}
            model_metrics = {}
            raw_metrics = {}
            for partition in PARTITIONS[1:]:
                part = parts[partition]
                raw, calibration_input, probabilities = scored[partition]
                rows = [
                    {"model_id": name, "partition": partition, "row_id": int(row_id),
                     "relative_time": float(when), "label": int(label), "raw_score": float(raw_score),
                     "calibration_input": float(input_score),
                     "calibrated_score": float(score), "decision": int(score > threshold)}
                    for row_id, when, label, raw_score, input_score, score in zip(
                        row_ids[part], relative_time[part], y[part], raw, calibration_input, probabilities)
                ]
                model_rows[partition] = rows
                all_predictions.extend(rows)
                model_metrics[partition] = _metrics(y[part], probabilities, threshold)
                raw_metrics[partition] = {"average_precision": float(average_precision_score(y[part], raw)),
                                          "brier": float(brier_score_loss(y[part], raw))}
            fitted[name] = (model, calibrated)
            artifact = f"model_{name}.joblib"
            joblib.dump({"estimator": model, "calibrator": calibrated}, staging / artifact, compress=3)
            sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
            summary["models"][name] = {
                "hyperparameters": model.steps[-1][1].get_params(deep=False),
                "fit_seconds": fit_seconds, "calibration_seconds": calibration_seconds,
                "calibration_mapping": {"method": "sigmoid", "input_method":
                                        "decision_function" if hasattr(model, "decision_function") else "predict_proba",
                                        "a": float(sigmoid.a_), "b": float(sigmoid.b_),
                                        "formula": "expit(-(a * calibration_input + b))"},
                "threshold": threshold, "metrics": model_metrics, "raw_metrics": raw_metrics,
                "predictions_sha256": {partition: _prediction_digest(model_rows[partition]) for partition in PARTITIONS[1:]},
                "model_artifact": artifact, "model_artifact_sha256": _sha_file(staging / artifact),
            }
        selection = max(MODELS, key=lambda name: summary["models"][name]["metrics"]["validation"]["average_precision"])
        summary["selection"] = {"model_id": selection, "validation_average_precision":
                                summary["models"][selection]["metrics"]["validation"]["average_precision"]}
        with (staging / "evaluation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(all_predictions)
        selected_model, selected_calibration = fitted[selection]
        selected_threshold = summary["models"][selection]["threshold"]
        with threadpool_limits(limits=2):
            stress = _stress(selected_model, selected_calibration, x[parts["test"]], y[parts["test"]],
                             row_ids[parts["test"]], seed, selected_threshold)
            capacity = _capacity(selected_calibration, x[parts["test"]], load_rows)
        stress["selected_model_id"] = selection
        capacity["selected_model_id"] = selection
        summary["stress"] = stress
        summary["capacity"] = capacity
        _write_json(staging / "evaluation_stress.json", stress)
        _write_json(staging / "evaluation_capacity.json", capacity)
        _write_json(staging / "evaluation_summary.json", summary)
        _write_json(staging / "evaluation_manifest.json", {
            "schema_version": SCHEMA_VERSION, "source_sha256": source_hash,
            "artifacts": {path.name: _sha_file(path) for path in staging.iterdir() if path.is_file()},
            "limitations": "Integrity check within a self-contained bundle; an independent signature is required for adversarial authenticity.",
        })
        if existing_empty_dir:
            output_dir.rmdir()
        os.replace(staging, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_evaluation(output_dir: Path) -> list[str]:
    """Validate hashes and semantic links in an evaluation evidence directory."""
    folder = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((folder / "evaluation_manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(artifacts, dict):
            return ["invalid evaluation manifest schema or artifacts"]
        required = {"evaluation_summary.json", "evaluation_predictions.csv", "evaluation_stress.json",
                    "evaluation_capacity.json", *(f"model_{name}.joblib" for name in MODELS)}
        if set(artifacts) != required:
            errors.append("evaluation manifest artifact set mismatch")
        for name, expected in artifacts.items():
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                errors.append(f"unsafe evaluation artifact path: {name}")
                continue
            path = folder / name
            if not path.is_file() or path.is_symlink() or _sha_file(path) != expected:
                errors.append(f"evaluation artifact hash mismatch or missing: {name}")
        if errors:
            return errors
        summary = json.loads((folder / "evaluation_summary.json").read_text(encoding="utf-8"))
        stress = json.loads((folder / "evaluation_stress.json").read_text(encoding="utf-8"))
        capacity = json.loads((folder / "evaluation_capacity.json").read_text(encoding="utf-8"))
        with (folder / "evaluation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(PREDICTION_FIELDS):
                return ["evaluation prediction columns mismatch"]
            predictions = list(reader)
        if summary.get("schema_version") != SCHEMA_VERSION:
            errors.append("evaluation summary schema mismatch")
        if summary.get("dataset", {}).get("source_sha256") != manifest.get("source_sha256"):
            errors.append("source hash mismatch: manifest vs summary")
        if summary.get("stress") != stress or summary.get("capacity") != capacity:
            errors.append("stress or capacity cross-file mismatch")
        if set(summary.get("models", {})) != set(MODELS):
            errors.append("evaluation model set mismatch")
        if set(summary.get("partitions", {})) != set(PARTITIONS):
            errors.append("evaluation partition set mismatch")
        if errors:
            return errors

        groups: dict[tuple[str, str], list[dict]] = {(name, part): [] for name in MODELS for part in PARTITIONS[1:]}
        for row in predictions:
            key = (row.get("model_id"), row.get("partition"))
            if key not in groups:
                errors.append(f"unknown model or partition in predictions: {key}")
                continue
            try:
                parsed = {"model_id": key[0], "partition": key[1], "row_id": int(row["row_id"]),
                          "relative_time": float(row["relative_time"]), "label": int(row["label"]),
                          "raw_score": float(row["raw_score"]),
                          "calibration_input": float(row["calibration_input"]),
                          "calibrated_score": float(row["calibrated_score"]),
                          "decision": int(row["decision"])}
                if (parsed["row_id"] < 2 or parsed["label"] not in (0, 1)
                        or parsed["decision"] not in (0, 1)
                        or not all(math.isfinite(parsed[field]) and 0 <= parsed[field] <= 1
                                   for field in ("raw_score", "calibrated_score"))
                        or not math.isfinite(parsed["calibration_input"])
                        or not math.isfinite(parsed["relative_time"])
                        or parsed["relative_time"] < 0):
                    raise ValueError("invalid prediction value")
                groups[key].append(parsed)
            except (KeyError, ValueError, TypeError) as exc:
                errors.append(f"invalid evaluation prediction: {exc}")
        if errors:
            return errors

        reference: dict[str, list[tuple[int, float, int]]] = {}
        for name in MODELS:
            item = summary["models"][name]
            mapping = item.get("calibration_mapping", {})
            if (mapping.get("method") != "sigmoid"
                    or mapping.get("input_method") not in ("decision_function", "predict_proba")
                    or not all(math.isfinite(float(mapping[field])) for field in ("a", "b"))):
                errors.append(f"invalid calibration mapping: {name}")
                continue
            validation = groups[(name, "validation")]
            if not validation:
                errors.append(f"missing validation predictions: {name}")
                continue
            y_val = np.array([row["label"] for row in validation], dtype=np.uint8)
            s_val = np.array([row["calibrated_score"] for row in validation], dtype=float)
            try:
                threshold = _threshold(y_val, s_val, float(summary["config"]["fpr_ceiling"]))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"invalid validation threshold inputs: {exc}")
                continue
            if not math.isclose(float(item["threshold"]), threshold, rel_tol=0, abs_tol=1e-15):
                errors.append(f"validation threshold mismatch: {name}")
            for partition in PARTITIONS[1:]:
                rows = groups[(name, partition)]
                if len(rows) != summary["partitions"][partition]["row_count"]:
                    errors.append(f"prediction count mismatch: {name}/{partition}")
                if len({row["row_id"] for row in rows}) != len(rows):
                    errors.append(f"duplicate source row id: {name}/{partition}")
                if any(rows[index]["relative_time"] > rows[index + 1]["relative_time"]
                       for index in range(len(rows) - 1)):
                    errors.append(f"prediction time order mismatch: {name}/{partition}")
                identity = [(row["row_id"], row["relative_time"], row["label"]) for row in rows]
                if partition in reference and reference[partition] != identity:
                    errors.append(f"cross-model source row mismatch: {name}/{partition}")
                reference.setdefault(partition, identity)
                if _prediction_digest(rows) != item["predictions_sha256"][partition]:
                    errors.append(f"prediction score link mismatch: {name}/{partition}")
                for row in rows:
                    raw_expected = (float(expit(row["calibration_input"]))
                                    if mapping["input_method"] == "decision_function"
                                    else row["calibration_input"])
                    calibrated_expected = float(expit(-(mapping["a"] * row["calibration_input"] + mapping["b"])))
                    if (not math.isclose(row["raw_score"], raw_expected, rel_tol=0, abs_tol=1e-10)
                            or not math.isclose(row["calibrated_score"], calibrated_expected, rel_tol=0, abs_tol=1e-10)):
                        errors.append(f"calibration mapping mismatch: {name}/{partition}/row {row['row_id']}")
                        break
                if any(row["decision"] != int(row["calibrated_score"] > threshold) for row in rows):
                    errors.append(f"decision threshold mismatch: {name}/{partition}")
                if rows:
                    labels = np.array([row["label"] for row in rows], dtype=np.uint8)
                    scores = np.array([row["calibrated_score"] for row in rows], dtype=float)
                    try:
                        actual = _metrics(labels, scores, threshold)
                        if not _same_metrics(actual, item["metrics"][partition]):
                            errors.append(f"evaluation metric mismatch: {name}/{partition}")
                        raw_scores = np.array([row["raw_score"] for row in rows], dtype=float)
                        raw_actual = {"average_precision": float(average_precision_score(labels, raw_scores)),
                                      "brier": float(brier_score_loss(labels, raw_scores))}
                        if not _same_metrics(raw_actual, item["raw_metrics"][partition]):
                            errors.append(f"raw evaluation metric mismatch: {name}/{partition}")
                    except (ValueError, KeyError, TypeError) as exc:
                        errors.append(f"invalid evaluation metrics: {name}/{partition}: {exc}")
            if (item.get("model_artifact") != f"model_{name}.joblib"
                    or item.get("model_artifact_sha256") != manifest["artifacts"].get(item.get("model_artifact"))):
                errors.append(f"model artifact link mismatch: {name}")

        if not errors:
            data_count = int(summary["dataset"]["row_count"])
            if sum(summary["partitions"][part]["row_count"] for part in PARTITIONS) != data_count:
                errors.append("partition row counts do not sum to source row count")
            seen_source_rows: dict[int, tuple[str, float, int]] = {}
            for partition in PARTITIONS[1:]:
                for row_id, when, label in reference[partition]:
                    if not 2 <= row_id <= data_count + 1:
                        errors.append(f"source row id outside CSV range: {partition}/{row_id}")
                    previous = seen_source_rows.get(row_id)
                    if previous is not None:
                        errors.append(f"source row id reused across partitions: {previous[0]}/{partition}/{row_id}")
                    else:
                        seen_source_rows[row_id] = (partition, when, label)
            for earlier, later in zip(PARTITIONS, PARTITIONS[1:]):
                if summary["partitions"][earlier]["time_max"] >= summary["partitions"][later]["time_min"]:
                    errors.append("chronological partitions overlap or split tied times")
            for partition in PARTITIONS[1:]:
                identities = reference[partition]
                metadata = summary["partitions"][partition]
                ids = np.array([item[0] for item in identities], dtype="<i8")
                if (metadata["row_ids_sha256"] != hashlib.sha256(ids.tobytes()).hexdigest()
                        or metadata["positive_count"] != sum(item[2] for item in identities)
                        or metadata["time_min"] != min(item[1] for item in identities)
                        or metadata["time_max"] != max(item[1] for item in identities)):
                    errors.append(f"partition source metadata mismatch: {partition}")
            selected = max(MODELS, key=lambda name: summary["models"][name]["metrics"]["validation"]["average_precision"])
            if (summary.get("selection", {}).get("model_id") != selected
                    or not math.isclose(summary["selection"]["validation_average_precision"],
                                        summary["models"][selected]["metrics"]["validation"]["average_precision"],
                                        rel_tol=0, abs_tol=1e-12)):
                errors.append("validation model selection mismatch")
            if stress.get("selected_model_id") != selected or capacity.get("selected_model_id") != selected:
                errors.append("stress or capacity selected model mismatch")
            if capacity.get("independent_accuracy_rows") != 0:
                errors.append("capacity workload incorrectly counted as independent accuracy data")
            load_rows = summary["config"]["load_rows"]
            expected_targets = sorted(set((min(100_000, load_rows), load_rows)))
            workloads = capacity.get("workloads", [])
            if [item.get("scored_rows") for item in workloads] != expected_targets:
                errors.append("capacity workload sizes mismatch")
            for workload in workloads:
                elapsed = workload.get("elapsed_seconds", 0)
                unit = workload.get("process_maxrss_source_unit")
                multiplier = 1024 if unit == "kibibytes" else 1
                raw_rss = workload.get("process_maxrss_raw")
                rss_bytes = workload.get("process_maxrss_bytes")
                valid_rss = ((unit == "unavailable" and raw_rss is None and rss_bytes is None)
                             or (unit in ("bytes", "kibibytes") and isinstance(raw_rss, int)
                                 and raw_rss >= 0 and rss_bytes == raw_rss * multiplier))
                if (not isinstance(elapsed, (int, float)) or elapsed <= 0
                        or not valid_rss
                        or not math.isclose(workload.get("scores_per_second", -1), workload["scored_rows"] / elapsed,
                                            rel_tol=1e-9, abs_tol=0)):
                    errors.append("capacity measurement metadata mismatch")
            test_identity = reference["test"]
            test_threshold = float(summary["models"][selected]["threshold"])
            baseline_metrics = summary["models"][selected]["metrics"]["test"]
            if not _same_metrics(baseline_metrics, stress.get("baseline_metrics")):
                errors.append("stress baseline metrics mismatch with selected test model")
            scenarios = stress.get("scenarios")
            if not isinstance(scenarios, dict) or set(scenarios) != set(STRESS_SCENARIOS):
                errors.append("stress scenario set mismatch")
                scenarios = scenarios if isinstance(scenarios, dict) else {}
            if (stress.get("classification") != "feature_perturbation_on_fixed_test_labels"
                    or stress.get("seed") != summary["config"]["seed"] + 1109):
                errors.append("stress scenario schema or seed mismatch")
            for scenario, result in scenarios.items():
                if scenario not in STRESS_SCENARIOS:
                    continue
                entries = result.get("predictions", [])
                expected_changed = (len(test_identity) if scenario == "amount_x1_2"
                                    else max(1, int(len(test_identity) * 0.10)))
                if result.get("changed_rows") != expected_changed:
                    errors.append(f"stress changed-row count mismatch: {scenario}")
                if [(row["row_id"], row["label"]) for row in entries] != [(row[0], row[2]) for row in test_identity]:
                    errors.append(f"stress row/label mismatch: {scenario}")
                    continue
                scores = np.array([row["score"] for row in entries], dtype=float)
                labels = np.array([row["label"] for row in entries], dtype=np.uint8)
                try:
                    computed = _metrics(labels, scores, test_threshold)
                    if not _same_metrics(computed, result["metrics"]):
                        errors.append(f"stress metric mismatch: {scenario}")
                    observed_drop = baseline_metrics["average_precision"] - computed["average_precision"]
                    ceiling = result["maximum_allowed_average_precision_drop"]
                    if (ceiling != 0.10 or not math.isclose(result["observed_average_precision_drop"], observed_drop,
                                                            rel_tol=0, abs_tol=1e-12)
                            or result["status"] != ("pass" if observed_drop <= ceiling else "fail")):
                        errors.append(f"stress gate mismatch: {scenario}")
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"invalid stress evidence: {scenario}: {exc}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid evaluation evidence: {exc}")
    return errors


def _same_metrics(actual: dict, stored: dict) -> bool:
    """Compare recursively while allowing only rounding noise in JSON floats."""
    if isinstance(actual, dict) and isinstance(stored, dict):
        return set(actual) == set(stored) and all(_same_metrics(actual[key], stored[key]) for key in actual)
    if isinstance(actual, list) and isinstance(stored, list):
        return len(actual) == len(stored) and all(_same_metrics(a, b) for a, b in zip(actual, stored))
    if isinstance(actual, (int, float)) and isinstance(stored, (int, float)):
        return math.isclose(actual, stored, rel_tol=0, abs_tol=1e-12)
    return actual == stored
