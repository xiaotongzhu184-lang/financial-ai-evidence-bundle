"""Four-estimator challenger and validation-frozen review-queue evidence.

The sealed JSON can be checked without executing model serialization. The raw
licensed source is intentionally external; hashes are provenance assertions,
not signatures or independent authentication of the prediction values.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .evaluation import FEATURES, _boundaries, _models, _read_ulb, _score
from .fusion import apply_policy, assess_scores, fit_fusion
from .pipeline import asset
from .research import code_identity, policy_for, read, sha, validate_source_identity, write
from .review import DECISIONS, read_reviews
from .resilience import fit_resilience_policy, reconstruct_resilience, run_resilience_study
from .source_profile import profile_ulb

SCHEMA = "0.5-algorithm"  # Preserve the legacy report and verifier contract.
RESILIENCE_SCHEMA = "0.6-algorithm"
MANIFEST = "bundle_manifest.json"
MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost")
REQUIRED = (
    "source_metadata.json", "source_profile.json", "run_metadata.json",
    "algorithm_models.json", "algorithm_scores.json", "fusion_policy.json",
    "algorithm_results.json", "release_gate.json", "review_report.html",
    *(f"model_{name}.joblib" for name in MODEL_NAMES),
)
REQUIRED_RESILIENCE = (*REQUIRED, "resilience_policy.json", "resilience_evidence.json")
PARTITIONS = ("train", "calibration", "validation", "test")
SCORED_PARTITIONS = ("calibration", "validation", "test")
STRESS_NAMES = ("strongest_model_dropout", "one_model_row_roll", "eight_bit_quantization")


def _resilience_gate(classification: str) -> dict:
    gate = policy_for(classification)
    gate["stress_interpretation"] = (
        "Failed unprotected controls remain valid negative findings; inspect "
        "algorithm_results.json stress and resilience separately.")
    return gate


def _model_spec(seed: int, positive: int, negative: int,
                rf_n_jobs: int = 2) -> dict:
    if positive <= 0 or negative <= 0:
        raise ValueError("train partition requires two classes")
    return {
        "logistic_regression": {"package": "scikit-learn", "estimator": "LogisticRegression",
                                "preprocessing": ["SimpleImputer(median)", "StandardScaler"],
                                "parameters": {"solver": "liblinear", "class_weight": "balanced", "max_iter": 300, "random_state": seed}},
        "random_forest": {"package": "scikit-learn", "estimator": "RandomForestClassifier",
                          "preprocessing": ["SimpleImputer(median)"],
                          "parameters": {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 4,
                                         "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": rf_n_jobs,
                                         "random_state": seed}},
        "hist_gradient_boosting": {"package": "scikit-learn", "estimator": "HistGradientBoostingClassifier",
                                   "preprocessing": ["SimpleImputer(median)"],
                                   "parameters": {"max_iter": 80, "max_leaf_nodes": 15, "min_samples_leaf": 10,
                                                  "learning_rate": .08, "early_stopping": False, "random_state": seed}},
        "xgboost": {"package": "xgboost", "estimator": "XGBClassifier",
                    "preprocessing": ["SimpleImputer(median)"],
                    "parameters": {"tree_method": "hist", "n_estimators": 160, "max_depth": 4,
                                   "learning_rate": .05, "subsample": .8, "colsample_bytree": .8,
                                   "min_child_weight": 5, "reg_lambda": 5, "n_jobs": 2,
                                   "random_state": seed, "scale_pos_weight": math.sqrt(negative / positive)}},
    }


def _check_config(run_id: str, metadata: dict, seed: int, load_rows: int,
                  budget_fraction: float) -> None:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be 1-64 safe filename characters")
    if (not isinstance(metadata, dict) or any(not isinstance(metadata.get(key), str) or not metadata[key].strip()
                                             for key in ("dataset_name", "classification", "source_url", "license_status"))):
        raise ValueError("source metadata requires dataset_name, classification, source_url, license_status")
    if metadata["classification"] not in ("public_ulb_creditcard", "synthetic_smoke_fixture"):
        raise ValueError("unsupported source classification")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if isinstance(load_rows, bool) or not isinstance(load_rows, int) or not 1 <= load_rows <= 10_000_000:
        raise ValueError("load_rows must be between 1 and 10000000")
    if (isinstance(budget_fraction, bool) or not isinstance(budget_fraction, (int, float))
            or not math.isfinite(budget_fraction) or not 0 < budget_fraction < 1):
        raise ValueError("budget_fraction must be finite and strictly between zero and one")


def _sha_ids(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def _sha_labels(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("u1", copy=False).tobytes()).hexdigest()


def _partition_record(times: np.ndarray, labels: np.ndarray, row_ids: np.ndarray, part: slice) -> dict:
    return {"row_count": part.stop - part.start, "positive_count": int(np.sum(labels[part])),
            "time_min": float(times[part.start]), "time_max": float(times[part.stop - 1]),
            "row_ids_sha256": _sha_ids(row_ids[part]), "labels_sha256": _sha_labels(labels[part])}


def _models4(seed: int, specs: dict) -> dict:
    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:
        raise ValueError("XGBoost native runtime unavailable; install libomp (macOS: brew install libomp) or use the scoped bundled runtime") from exc
    models = _models(seed)
    # The legacy evaluator stays at n_jobs=2. Parallel RF probability
    # accumulation can vary in the last bit between calls on the same batch,
    # which invalidates an exact score-delivery receipt in schema 0.6.
    models["random_forest"].set_params(
        randomforestclassifier__n_jobs=specs["random_forest"]["parameters"]["n_jobs"])
    models["xgboost"] = make_pipeline(SimpleImputer(strategy="median"),
                                      XGBClassifier(**specs["xgboost"]["parameters"]))
    return models


def _rows_for_partition(x: np.ndarray, y: np.ndarray, times: np.ndarray, row_ids: np.ndarray,
                        part: slice, fitted: dict, calibration: dict) -> list[dict]:
    by_model = {}
    for name in MODEL_NAMES:
        raw, cal_input, probability = _score(fitted[name], calibration[name], x[part])
        by_model[name] = (raw, cal_input, probability)
    return [{"row_id": int(row_id), "relative_time": float(when), "label": int(label),
             "models": {name: {"raw_score": float(by_model[name][0][index]),
                               "calibration_input": float(by_model[name][1][index]),
                               "calibrated_score": float(by_model[name][2][index])}
                        for name in MODEL_NAMES}}
            for index, (row_id, when, label) in enumerate(zip(row_ids[part], times[part], y[part]))]


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row["models"][name]["calibrated_score"] for name in MODEL_NAMES]
                       for row in rows], dtype=np.float64)


def _row_values(rows: list[dict]) -> tuple[list[int], list[int], list[float]]:
    return ([row["row_id"] for row in rows], [row["label"] for row in rows],
            [row["relative_time"] for row in rows])


def _ablation_specs(policy: dict) -> list[tuple[str, dict]]:
    frozen = policy["ablation_policies"]
    return [*((name, item) for name, item in zip(MODEL_NAMES, frozen["single_models"])),
            ("equal_weight", frozen["equal_weight"]),
            ("validation_selected_best_single", frozen["best_single"]),
            ("robust_weighted_fusion", frozen["robust_weighted"])]


def _window_monitoring(scores: dict, policy: dict) -> dict:
    """Apply one frozen reference per comparator to three later time windows."""
    from .monitoring import monitor_score_windows

    validation, test = scores["validation"], scores["test"]
    validation_matrix, test_matrix = _matrix(validation), _matrix(test)
    reference_positions = {row["row_id"]: index for index, row in enumerate(validation)}
    positions = [reference_positions[row_id] for row_id in policy["policy_row_ids"]]
    test_ids, test_labels, test_times = _row_values(test)
    analyses = {}
    for name, frozen in _ablation_specs(policy):
        if name == "validation_selected_best_single":
            continue  # Already represented by its named single estimator.
        weights = np.asarray(frozen["weights"], dtype=np.float64)
        reference_scores = validation_matrix[positions] @ weights
        later_scores = test_matrix @ weights
        analyses[name] = monitor_score_windows(
            reference_scores.tolist(), policy["policy_row_ids"], later_scores.tolist(),
            test_times, test_ids, test_labels, frozen["threshold"], policy["budget_fraction"])
    return {"classification": "independent_retrospective_window_diagnostics",
            "interpretation": "Each time window has its own diagnostic queue and budget; these are not the whole-test queue or a prospective online policy.",
            "analyses": analyses}


def _results(scores: dict, policy: dict, capacity: dict) -> dict:
    validation = scores["validation"]
    test = scores["test"]
    validation_matrix = _matrix(validation)
    test_matrix = _matrix(test)
    test_ids, test_labels, _ = _row_values(test)
    policy_positions = {row["row_id"]: index for index, row in enumerate(validation)}
    policy_indices = [policy_positions[row_id] for row_id in policy["policy_row_ids"]]
    budget = policy["budget_fraction"]
    robust_scores = test_matrix @ np.asarray(policy["weights"], dtype=np.float64)
    queue = apply_policy(robust_scores.tolist(), test_ids, policy["threshold"], budget)
    metrics = assess_scores(test_labels, robust_scores.tolist(), test_ids, policy["threshold"], budget)
    ablations = []
    for name, frozen in _ablation_specs(policy):
        weights, threshold = frozen["weights"], frozen["threshold"]
        candidate_scores = test_matrix @ np.asarray(weights, dtype=np.float64)
        ablations.append({"name": name, "weights": list(weights), "threshold": threshold,
                          "test_metrics": assess_scores(test_labels, candidate_scores.tolist(), test_ids,
                                                        threshold, budget)})
    strongest = int(np.argmax(np.asarray(policy["weights"])))
    variants = {}
    for name in STRESS_NAMES:
        changed = test_matrix.copy()
        if name == "strongest_model_dropout":
            changed[:, strongest] = 0.0
        elif name == "one_model_row_roll":
            changed[:, strongest] = np.roll(changed[:, strongest], 1)
        else:
            changed = np.rint(changed * 255.0) / 255.0
        changed_scores = changed @ np.asarray(policy["weights"])
        stressed = assess_scores(test_labels, changed_scores.tolist(), test_ids, policy["threshold"], budget)
        drop = metrics["average_precision"] - stressed["average_precision"]
        variants[name] = {"score_values": [float(value) for value in changed_scores],
                          "metrics": stressed, "observed_average_precision_drop": drop,
                          "maximum_allowed_average_precision_drop": .10,
                          "status": "pass" if drop <= .10 else "fail"}
    policy_scores = validation_matrix[policy_indices] @ np.asarray(policy["weights"])
    ks = ks_2samp(policy_scores, robust_scores, method="exact" if len(policy_scores) * len(robust_scores) <= 100_000 else "asymp")
    disagreement = np.std(test_matrix, axis=1)
    return {
        "test_queue": queue, "test_metrics": metrics, "ablations": ablations,
        "stress": {"classification": "score_space_faults_only",
                   "limits": "Controlled score-pipeline faults, not raw financial feature or future-data stress.",
                   "strongest_weight_model": MODEL_NAMES[strongest], "scenarios": variants},
        "drift": {"classification": "descriptive_policy_test_score_distribution_only",
                  "ks_statistic": float(ks.statistic), "ks_pvalue_descriptive_only": float(ks.pvalue),
                  "iid_validity_claim": False},
        "disagreement": {"method": "standard_deviation_of_four_calibrated_probabilities_per_test_row",
                         "mean": float(np.mean(disagreement)), "max": float(np.max(disagreement)),
                         "sample_rows": [{"row_id": row_id, "stddev": float(stddev)}
                                         for row_id, stddev in zip(test_ids[:20], disagreement[:20])]},
        "capacity": capacity,
        "window_monitoring": _window_monitoring(scores, policy),
        "interpretation": "Retrospective exploratory ULB comparison; no prospective or institution-specific effectiveness claim.",
    }


def _resilience_context(profile: dict, model_record: dict, model_dir: Path,
                        test_ids: list[int]) -> dict:
    """Bind a local score delivery to source, ordered features and model files."""
    return {
        "protocol": "local-score-delivery-v1",
        "source_sha256": profile["source_sha256"],
        "ordered_feature_batch_sha256": model_record["test_feature_batch_sha256"],
        "ordered_row_ids_sha256": _sha_ids(np.asarray(test_ids, dtype=np.int64)),
        "feature_shape": [len(test_ids), len(FEATURES)],
        "model_artifact_sha256": {name: sha(model_dir / f"model_{name}.joblib")
                                  for name in MODEL_NAMES},
    }


def _resilience_posthoc(evidence: dict, labels: list[int], policy: dict,
                        baseline_results: dict) -> dict:
    """Assess saved recovery outputs after label-free runtime decisions exist."""
    ids = evidence["row_ids"]
    baseline_ap = baseline_results["test_metrics"]["average_precision"]
    controls = baseline_results["stress"]["scenarios"]
    scenarios = []
    for item in evidence["scenarios"]:
        control = controls.get(item["name"])
        unprotected = ({"average_precision": control["metrics"]["average_precision"],
                        "observed_average_precision_drop": control["observed_average_precision_drop"],
                        "quality_status": control["status"]} if control is not None else None)
        if item["status"] == "fail_closed":
            scenario = {"name": item["name"], "runtime_status": "fail_closed",
                        "quality_status": "not_applicable", "protected_metrics": None,
                        "observed_average_precision_drop": None,
                        "unprotected_control": unprotected,
                        "decision_count": 0, "coverage": 0.0}
        else:
            metrics = assess_scores(labels, item["scores"], ids,
                                    item["threshold"], policy["budget_fraction"])
            loss = baseline_ap - metrics["average_precision"]
            scenario = {"name": item["name"], "runtime_status": "ok",
                        "quality_status": "pass" if loss <= .10 else "fail",
                        "protected_metrics": metrics,
                        "observed_average_precision_drop": loss,
                        "unprotected_control": unprotected,
                        "decision_count": item["decision_count"],
                        "coverage": item["coverage"]}
        scenarios.append(scenario)
    return {"classification": "retrospective_posthoc_quality_not_runtime_input",
            "baseline_average_precision": baseline_ap,
            "maximum_allowed_average_precision_drop": .10,
            "scenarios": scenarios}


def _capacity(calibrated: dict, x: np.ndarray, weights: list[float], load_rows: int,
              xgboost_version: str) -> dict:
    targets = sorted({min(100_000, load_rows), load_rows})
    workloads = []
    try:
        import resource
    except ImportError:
        resource = None
    rss_unit = "unavailable" if resource is None else "kibibytes" if platform.system() == "Linux" else "bytes"
    for target in targets:
        digest = hashlib.sha256()
        start = time.perf_counter()
        processed = 0
        with threadpool_limits(limits=2):
            while processed < target:
                size = min(8192, target - processed)
                indices = np.arange(processed, processed + size) % len(x)
                batch = np.column_stack([calibrated[name].predict_proba(x[indices])[:, 1] for name in MODEL_NAMES])
                fused = (batch @ np.asarray(weights)).astype("<f8", copy=False)
                digest.update(fused.tobytes())
                processed += size
        elapsed = time.perf_counter() - start
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        workloads.append({"scored_rows": target, "batch_size_max": 8192,
                          "elapsed_seconds": elapsed, "scores_per_second": target / elapsed,
                          "score_stream_sha256": digest.hexdigest(), "process_maxrss_raw": raw_rss,
                          "process_maxrss_source_unit": rss_unit,
                          "process_maxrss_bytes": raw_rss * (1024 if rss_unit == "kibibytes" else 1) if raw_rss is not None else None})
    return {"classification": "repeated_feature_inference_load_only", "independent_accuracy_rows": 0,
            "models_scored": list(MODEL_NAMES), "batch_limit": 8192,
            "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                        "cpu_count": os.cpu_count(), "python": platform.python_version(),
                        "sklearn": sklearn.__version__, "xgboost": xgboost_version,
                        "threadpool_limit": 2},
            "limitations": "Repeated held-out features measure inference load only; machine-specific throughput is not accuracy evidence.",
            "workloads": workloads}


def _score_stream_digest(scores: np.ndarray, weights: list[float], target: int) -> str:
    """Reconstruct the measured repeat-load stream from archived test scores."""
    digest = hashlib.sha256()
    processed = 0
    while processed < target:
        size = min(8192, target - processed)
        indices = np.arange(processed, processed + size) % len(scores)
        fused = (scores[indices] @ np.asarray(weights)).astype("<f8", copy=False)
        digest.update(fused.tobytes())
        processed += size
    return digest.hexdigest()


def render_algorithm_report(bundle: Path, csrf_token: str | None = None) -> bytes:
    """Render a small deterministic, escaped review surface for an existing bundle."""
    bundle = Path(bundle)
    metadata = read(bundle / "source_metadata.json")
    policy = read(bundle / "fusion_policy.json")
    results = read(bundle / "algorithm_results.json")
    models = read(bundle / "algorithm_models.json")
    esc = lambda value: html.escape(str(value), quote=True)
    table = "".join(
        "<tr><th scope=\"row\">" + esc(entry["name"]) + "</th>" +
        "<td>" + f'{entry["test_metrics"]["average_precision"]:.6f}' + "</td>" +
        "<td>" + f'{entry["test_metrics"]["precision"]:.6f}' + "</td>" +
        "<td>" + f'{entry["test_metrics"]["recall"]:.6f}' + "</td>" +
        "<td>" + esc(entry["test_metrics"]["counts"]["fp"]) + "</td>" +
        "<td>" + esc(entry["test_metrics"]["counts"]["fn"]) + "</td>" +
        "<td>" + esc(entry["test_metrics"]["threshold_alert_count"]) + "</td>" +
        "<td>" + esc(entry["test_metrics"]["reviewed_count"]) + "</td>" +
        "<td>" + esc(entry["test_metrics"]["deferred_count"]) + "</td></tr>"
        for entry in results["ablations"])
    stress = "".join("<li>" + esc(name) + ": " + esc(value["status"]) +
                     " (AP drop " + f'{value["observed_average_precision_drop"]:.6f}' + ")</li>"
                     for name, value in results["stress"]["scenarios"].items())
    window_rows = []
    for name, analysis in results["window_monitoring"]["analyses"].items():
        for window in analysis["windows"]:
            measured = window["queue_metrics"]
            counts = measured["counts"] if measured is not None else None
            cells = (name, window["window_index"], window["row_count"],
                     f'{window["psi"]:.6f}',
                     counts["tp"] if counts is not None else "n/a",
                     counts["fp"] if counts is not None else "n/a",
                     counts["fn"] if counts is not None else "n/a",
                     window["queue"]["threshold_alert_count"],
                     window["queue"]["reviewed_count"],
                     window["queue"]["deferred_count"],
                     "yes" if window["small_sample"] else "no")
            window_rows.append("<tr>" + "".join("<td>" + esc(value) + "</td>" for value in cells) + "</tr>")
    upstream_url = ("https://github.com/evidentlyai/evidently/blob/"
                    "34771fe41c08ca074d96a52aaf4a45cd640cbaf5/"
                    "src/evidently/legacy/calculations/stattests/psi.py")
    license_url = ("https://github.com/evidentlyai/evidently/blob/"
                   "34771fe41c08ca074d96a52aaf4a45cd640cbaf5/LICENSE")
    form = ""
    if csrf_token is not None:
        options = "".join('<option value="' + esc(decision) + '">' + esc(decision) + "</option>"
                          for decision in sorted(DECISIONS))
        form = ('<form method="post" action="/api/reviews"><input type="hidden" name="csrf" value="' + esc(csrf_token) +
                '"><label>Reviewer<input name="reviewer" required></label><label>Decision<select name="decision">' +
                options + '</select></label><label>Rationale<textarea name="rationale" required></textarea></label>' +
                '<button type="submit">Record local review</button></form>')
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>Algorithm evidence review</title><style>'
                ':root{color-scheme:light}*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;'
                'color:#16324a;background:#f2f7f8;max-width:1100px;margin:auto;padding:1.5rem;overflow-wrap:anywhere}'
                'h1,h2{color:#103454}h1{border-bottom:4px solid #087e83;padding-bottom:.4rem}'
                'a{color:#006c76}nav{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}'
                '.card{background:white;border:1px solid #ccdee3;border-radius:10px;padding:1rem;margin:1rem 0}'
                '.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:850px}'
                'th,td{border-bottom:1px solid #cfdee2;padding:.5rem;text-align:left;white-space:nowrap}'
                'thead{background:#dcecef}label{display:block;margin:1rem 0}input,select,textarea{display:block;'
                'width:100%;max-width:35rem;padding:.4rem}button{background:#087e83;color:white;border:0;'
                'padding:.6rem 1rem;border-radius:6px}</style></head><body><h1>Algorithm evidence review</h1>'
                '<nav aria-label="Evidence files"><a href="/api/manifest">Manifest</a>'
                '<a href="/api/export">Export evidence ZIP</a><a href="/api/reviews">Review journal</a></nav>'
                '<section class="card"><h2>Run and source</h2><p>Dataset: ' + esc(metadata["dataset_name"]) +
                ' (' + esc(metadata["classification"]) + ')</p><p>Schema ' + SCHEMA +
                '; four fitted and sigmoid calibrated models. Source raw CSV excluded.</p><p>Selected weights: ' +
                esc(policy["weights"]) + '; frozen threshold: ' + esc(policy["threshold"]) +
                '; scenario budget: ' + esc(policy["budget_fraction"]) + '</p></section>'
                '<section class="card"><h2>Held-out comparison</h2><div class="table-wrap"><table><thead><tr>'
                '<th>Comparison</th><th>Test AP</th><th>Precision</th><th>Reviewed recall</th>'
                '<th>False positives</th><th>False negatives</th><th>Threshold alerts</th>'
                '<th>Reviewed</th><th>Deferred</th></tr></thead><tbody>' + table +
                '</tbody></table></div></section><section class="card"><h2>Selected queue</h2><p>Threshold alerts: ' +
                esc(results["test_queue"]["threshold_alert_count"]) + '; reviewed: ' +
                esc(results["test_queue"]["reviewed_count"]) + '; deferred: ' +
                esc(results["test_queue"]["deferred_count"]) + '</p></section>'
                '<section class="card"><h2>Score-space fault probes</h2><ul>' + stress +
                '</ul><p>Policy/test score KS statistic: ' + f'{results["drift"]["ks_statistic"]:.6f}' +
                '. Descriptive only.</p></section><section class="card"><h2>Window drift diagnostics</h2>'
                '<p>PSI uses frozen policy-period reference bins. Each of three relative-Time windows has an '
                'independent diagnostic review cap; these counts do not sum to the whole-test queue. Small samples '
                'and single-class windows limit interpretation. PSI is descriptive, not proof of accuracy or fraud-rate change.</p>'
                '<div class="table-wrap"><table><thead><tr><th>Comparator</th><th>Window</th><th>n</th>'
                '<th>PSI</th><th>TP</th><th>FP</th><th>FN</th><th>Threshold alerts</th>'
                '<th>Reviewed</th><th>Deferred</th><th>Small sample</th></tr></thead><tbody>' +
                ''.join(window_rows) + '</tbody></table></div><p>Modified numeric core from '
                '<a href="' + upstream_url + '">Evidently PSI source at fixed commit</a>; '
                '<a href="' + license_url + '">Apache-2.0 license</a>. This is an adapted PSI calculation, '
                'not the full Evidently platform.</p></section><section class="card"><h2>Scope</h2><p>Package versions: sklearn ' +
                esc(models["packages"]["scikit-learn"]) + ', XGBoost ' + esc(models["packages"]["xgboost"]) +
                '. Retrospective exploratory benchmark; no production approval.</p></section>' + form + '</body></html>')
    return document.encode("utf-8")


def render_resilience_report(bundle: Path, csrf_token: str | None = None) -> bytes:
    """Put 0.6 recovery evidence ahead of legacy controls; preserve 0.5 bytes."""
    base = render_algorithm_report(bundle, csrf_token).decode("utf-8")
    evidence = read(Path(bundle) / "resilience_evidence.json")
    results = read(Path(bundle) / "algorithm_results.json")["resilience"]
    assessed = {item["name"]: item for item in results["scenarios"]}
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    attempt_rows = []
    for item in evidence["scenarios"]:
        result = assessed[item["name"]]
        control = result["unprotected_control"]
        measured = result["protected_metrics"]
        before = "n/a" if control is None else f'{control["average_precision"]:.6f} ({esc(control["quality_status"])})'
        after = "no decisions" if measured is None else f'{measured["average_precision"]:.6f} ({esc(result["quality_status"])})'
        counts = ("no decisions" if measured is None else
                  f'{measured["counts"]["tp"]}/{measured["counts"]["fp"]}/{measured["counts"]["fn"]}')
        rows.append('<tr><th scope="row">' + esc(item["name"]) + '</th><td>' + before +
                    '</td><td>' + after + '</td><td>' + esc(item["route"]) +
                    '</td><td>' + esc(", ".join(item["quarantined_models"]) or "none") +
                    '</td><td>' + esc(item["selected_model"] or "baseline") +
                    '</td><td>' + esc(sum(item["call_counts"].values())) +
                    '</td><td>' + f'{item["coverage"]:.1%}' +
                    '</td><td>' + counts + '</td></tr>')
        for attempt in item["attempts"]:
            attempt_rows.append('<tr><td>' + esc(item["name"]) + '</td><td>' +
                                esc(attempt["model"]) + '</td><td>' + esc(attempt["attempt"]) +
                                '</td><td>' + esc(attempt["status"]) + '</td><td>' +
                                esc(attempt["failure_reason"] or "none") + '</td></tr>')
    assessed_quality = [item for item in results["scenarios"] if item["quality_status"] != "not_applicable"]
    quality_passes = sum(item["quality_status"] == "pass" for item in assessed_quality)
    fail_closed = sum(item["runtime_status"] == "fail_closed" for item in results["scenarios"])
    panel = ('<section class="card"><h2>Prediction delivery recovery</h2>'
             '<p>Protected scenarios meeting the frozen AP-loss criterion: ' +
             esc(quality_passes) + '/' + esc(len(assessed_quality)) +
             '. Fail-closed safety outcomes: ' + esc(fail_closed) +
             ' (reported separately, with zero decisions).</p>'
             '<p>Producer receipts bind model, row IDs and context before a controlled score-transport fault. '
             'Rejected deliveries trigger at most one fresh model call. Persistent loss quarantines that model '
             'and uses a validation-frozen fallback; fewer than two healthy models emit no decisions. '
             'Receipts protect against local post-production corruption, not coordinated forgery or errors '
             'before receipt creation.</p>'
             '<p><a href="/api/artifacts/resilience_policy.json">Frozen fallback policy</a> · '
             '<a href="/api/artifacts/resilience_evidence.json">Attempt receipts and delivery transcript</a> · '
             '<a href="/api/artifacts/algorithm_results.json">Post-hoc quality and queue metrics</a></p>'
             '<div class="table-wrap"><table><thead><tr><th>Scenario</th><th>Unprotected AP</th>'
             '<th>Recovered AP</th><th>Route</th><th>Quarantined</th><th>Fallback</th>'
             '<th>Actual model calls</th><th>Coverage</th><th>Reviewed TP/FP/FN</th>'
             '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
             '<details><summary>Individual model delivery attempts</summary><div class="table-wrap"><table>'
             '<thead><tr><th>Scenario</th><th>Model</th><th>Attempt</th><th>Receipt check</th>'
             '<th>Failure reason</th></tr></thead><tbody>' + ''.join(attempt_rows) +
             '</tbody></table></div></details>'
             '<p>The unprotected controls remain in the separate panel below. The recovered AP and '
             'confusion counts use held-out labels only after runtime routing. Fail-closed is a safety outcome, '
             'not an accuracy pass. This same historical period remains retrospective.</p></section>')
    base = base.replace("Schema 0.5-algorithm", "Schema 0.6-algorithm", 1)
    base = base.replace('<h1>Algorithm evidence review</h1>',
                        '<h1>Algorithm evidence review | 0.6 recovery</h1>', 1)
    base = base.replace('<h2>Score-space fault probes</h2>',
                        '<h2>Unprotected 0.5 control probes (before recovery)</h2>', 1)
    return base.replace('<section class="card"><h2>Unprotected 0.5 control probes (before recovery)</h2>',
                        panel + '<section class="card"><h2>Unprotected 0.5 control probes (before recovery)</h2>',
                        1).encode("utf-8")


def algorithm_run(output_root: Path, run_id: str, csv_path: Path, metadata: dict,
                  seed: int = 42, load_rows: int = 100_000,
                  budget_fraction: float = .005) -> Path:
    """Train four estimators, freeze validation policy, and atomically seal evidence."""
    _check_config(run_id, metadata, seed, load_rows, budget_fraction)
    csv_path, output_root = Path(csv_path), Path(output_root)
    if output_root.is_symlink():
        raise ValueError("output root may not be a symbolic link")
    bundle = output_root / run_id
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(f"run already exists: {bundle}")
    profile = profile_ulb(csv_path)
    validate_source_identity(metadata, profile)
    metadata = {**metadata, "csv_sha256": profile["source_sha256"],
                "source_sha256": profile["source_sha256"]}
    x, y, times, row_ids = _read_ulb(csv_path)
    parts = _boundaries(times, y)
    if len(y) != profile["row_count"] or int(np.sum(y)) != profile["positive_count"] or sha(csv_path) != profile["source_sha256"]:
        raise ValueError("source changed between profiling and fitting")
    train = parts["train"]
    positives = int(np.sum(y[train]))
    specs = _model_spec(seed, positives, len(y[train]) - positives, rf_n_jobs=1)
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ("." + run_id + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix="." + run_id + "-", dir=output_root))
        fitted = _models4(seed, specs)
        import xgboost  # Training-only native import; semantic verification never imports it.
        xgboost_version = xgboost.__version__
        calibrated = {}
        calibrator_records = {}
        fit_seconds = {}
        for name in MODEL_NAMES:
            start = time.perf_counter()
            with threadpool_limits(limits=2):
                fitted[name].fit(x[train], y[train])
                calibrated[name] = CalibratedClassifierCV(FrozenEstimator(fitted[name]),
                                                          method="sigmoid", cv=2, ensemble=False)
                calibrated[name].fit(x[parts["calibration"]], y[parts["calibration"]])
            fit_seconds[name] = time.perf_counter() - start
            sigmoid = calibrated[name].calibrated_classifiers_[0].calibrators[0]
            calibrator_records[name] = {"method": "sigmoid", "a": float(sigmoid.a_), "b": float(sigmoid.b_),
                                        "input_kind": "decision_function" if hasattr(fitted[name], "decision_function") else "predict_proba"}
            joblib.dump({"estimator": fitted[name], "calibrator": calibrated[name]},
                        stage / f"model_{name}.joblib", compress=3)
        score_rows = {name: _rows_for_partition(x, y, times, row_ids, parts[name], fitted, calibrated)
                      for name in SCORED_PARTITIONS}
        validation = score_rows["validation"]
        val_ids, val_labels, val_times = _row_values(validation)
        policy = fit_fusion(_matrix(validation), val_labels, val_times, val_ids,
                            MODEL_NAMES, budget_fraction)
        capacity = _capacity(calibrated, x[parts["test"]], policy["weights"], load_rows,
                             xgboost_version)
        results = _results(score_rows, policy, capacity)
        partition_records = {name: _partition_record(times, y, row_ids, parts[name]) for name in PARTITIONS}
        test_features = x[parts["test"]]
        model_record = {"schema_version": RESILIENCE_SCHEMA, "seed": seed, "feature_columns": FEATURES,
                        "split_targets": [0.55, 0.15, 0.15, 0.15],
                        "time_tie_rule": "move each boundary forward after the last equal relative Time",
                        "test_feature_batch_sha256": hashlib.sha256(
                            np.ascontiguousarray(test_features, dtype="<f8").tobytes()).hexdigest(),
                        "partitions": partition_records, "model_names": list(MODEL_NAMES),
                        "models": specs, "calibrators": calibrator_records, "fit_seconds": fit_seconds,
                        "packages": {"scikit-learn": sklearn.__version__,
                                     "xgboost": xgboost_version,
                                     "numpy": np.__version__}}
        resilience_policy = fit_resilience_policy(_matrix(validation), val_labels,
                                                  val_times, val_ids, MODEL_NAMES,
                                                  budget_fraction)
        test_ids, test_labels, _ = _row_values(score_rows["test"])
        context = _resilience_context(profile, model_record, stage, test_ids)
        predictors = {name: (lambda features, selected=name:
                             calibrated[selected].predict_proba(features)[:, 1])
                      for name in MODEL_NAMES}
        recovery_started = time.perf_counter()
        with threadpool_limits(limits=2):
            resilience_evidence = run_resilience_study(
                predictors, test_features, test_ids, context, resilience_policy,
                policy["weights"], policy["threshold"])
        recovery_elapsed = time.perf_counter() - recovery_started
        resilience_evidence["runtime_measurement"] = {
            "classification": "eight_scenario_actual_model_callbacks_not_capacity_probe",
            "elapsed_seconds": recovery_elapsed,
            "model_call_count": sum(sum(item["call_counts"].values())
                                    for item in resilience_evidence["scenarios"]),
            "feature_rows_per_scenario": len(test_ids),
        }
        results["resilience"] = _resilience_posthoc(resilience_evidence,
                                                    test_labels, policy, results)
        write(stage / "source_metadata.json", metadata)
        write(stage / "source_profile.json", profile)
        write(stage / "algorithm_models.json", model_record)
        write(stage / "algorithm_scores.json", score_rows)
        write(stage / "fusion_policy.json", policy)
        write(stage / "resilience_policy.json", resilience_policy)
        write(stage / "resilience_evidence.json", resilience_evidence)
        write(stage / "algorithm_results.json", results)
        write(stage / "release_gate.json", _resilience_gate(metadata["classification"]))
        version = asset("VERSION").read_text().strip()
        write(stage / "run_metadata.json", {"schema_version": RESILIENCE_SCHEMA, "run_id": run_id,
                                             "project_version": version, "completed": True,
                                             "utc_time": datetime.now(timezone.utc).isoformat(),
                                             "seed": seed, "load_rows": load_rows,
                                             "source_classification": metadata["classification"],
                                             "source_sha256": profile["source_sha256"],
                                             "raw_source_included": False,
                                             "distribution_note": "Raw licensed CSV excluded; predictions and models are derived artifacts subject to source rights.",
                                             **code_identity()})
        (stage / "review_report.html").write_bytes(render_resilience_report(stage))
        write(stage / MANIFEST, {"schema_version": RESILIENCE_SCHEMA, "run_id": run_id,
                                "project_version": version,
                                "source_classification": metadata["classification"],
                                "required_artifacts": list(REQUIRED_RESILIENCE),
                                "artifacts": {name: sha(stage / name) for name in REQUIRED_RESILIENCE}})
        errors = verify_algorithm(stage, check_reviews=False)
        if errors:
            raise ValueError("algorithm evidence failed verification: " + "; ".join(errors))
        if bundle.exists() or bundle.is_symlink():
            raise FileExistsError(f"run already exists: {bundle}")
        stage.rename(bundle)
        stage = None
        return bundle
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def algorithm_smoke(output_root: Path, run_id: str, seed: int = 42,
                    load_rows: int = 1000) -> Path:
    """Train the actual operators on generated ULB-shaped rows, clearly labelled."""
    with tempfile.TemporaryDirectory(prefix="yk-algorithm-smoke-") as tmp:
        source = Path(tmp) / "generated.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Time", *(f"V{i}" for i in range(1, 29)), "Amount", "Class"])
            for index in range(400):
                label = int(index % 8 == 0)
                features = [round((index % (7 + j % 5)) * .1 + label * (j % 3), 3) for j in range(1, 29)]
                writer.writerow([index // 2, *features, index % 100, label])
        metadata = {"dataset_name": "Generated ULB-schema algorithm software smoke fixture",
                    "classification": "synthetic_smoke_fixture",
                    "source_url": "local:yk_evidence.algorithm.algorithm_smoke",
                    "license_status": "project-generated synthetic fixture"}
        return algorithm_run(output_root, run_id, source, metadata, seed=seed, load_rows=load_rows)


def _same(expected, actual, path: str, errors: list[str], tolerance: float = 1e-9) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            errors.append(path + " keys differ")
            return
        for key, value in expected.items():
            _same(value, actual[key], path + "." + str(key), errors, tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(path + " length differs")
            return
        for index, value in enumerate(expected):
            _same(value, actual[index], path + f"[{index}]", errors, tolerance)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual)
                or not math.isclose(float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance)):
            errors.append(path + " numeric value differs")
    elif type(expected) is not type(actual) or expected != actual:
        errors.append(path + " differs")


def _verify_scores(scores: dict, model_record: dict, profile: dict, errors: list[str]) -> None:
    if not isinstance(scores, dict) or set(scores) != set(SCORED_PARTITIONS):
        errors.append("score partition allowlist differs")
        return
    total_rows = model_record["partitions"]["train"]["row_count"]
    total_positive = model_record["partitions"]["train"]["positive_count"]
    previous_time = model_record["partitions"]["train"]["time_max"]
    observed_ids = set()
    for name in SCORED_PARTITIONS:
        rows = scores[name]
        expected = model_record["partitions"][name]
        if not isinstance(rows, list) or len(rows) != expected["row_count"]:
            errors.append(name + " score rows differ from partition")
            continue
        ids, labels, times = [], [], []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"row_id", "relative_time", "label", "models"}:
                errors.append(name + " score row shape differs")
                return
            row_id, when, label = row["row_id"], row["relative_time"], row["label"]
            if (isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 2 or row_id > profile["row_count"] + 1
                    or isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(when) or when < 0
                    or isinstance(label, bool) or label not in (0, 1)):
                errors.append(name + " invalid row identity")
                return
            ids.append(row_id); labels.append(label); times.append(when)
            if not isinstance(row["models"], dict) or set(row["models"]) != set(MODEL_NAMES):
                errors.append(name + " model score columns differ")
                return
            for model in MODEL_NAMES:
                entry = row["models"][model]
                if not isinstance(entry, dict) or set(entry) != {"raw_score", "calibration_input", "calibrated_score"}:
                    errors.append(name + " model score shape differs")
                    return
                raw, cal_input, probability = (entry[key] for key in ("raw_score", "calibration_input", "calibrated_score"))
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (raw, cal_input, probability)) or not 0 <= raw <= 1 or not 0 <= probability <= 1:
                    errors.append(name + " nonfinite/nonprobability model score")
                    return
                params = model_record["calibrators"][model]
                if params["input_kind"] == "predict_proba" and not math.isclose(raw, cal_input, abs_tol=1e-12):
                    errors.append(name + " calibration input disagrees with raw probability")
                    return
                if params["input_kind"] == "decision_function" and not math.isclose(
                        raw, float(expit(cal_input)), rel_tol=1e-10, abs_tol=1e-10):
                    errors.append(name + " raw probability disagrees with decision function")
                    return
                calculated = float(expit(-(params["a"] * cal_input + params["b"])))
                if not math.isclose(calculated, probability, abs_tol=1e-10, rel_tol=1e-10):
                    errors.append(name + " sigmoid calibration formula differs")
                    return
        if len(set(ids)) != len(ids) or observed_ids.intersection(ids) or any(times[k] > times[k + 1] for k in range(len(times)-1)):
            errors.append(name + " unsorted or duplicate score identities")
        observed_ids.update(ids)
        if previous_time >= times[0] or times[0] != expected["time_min"] or times[-1] != expected["time_max"]:
            errors.append(name + " chronology differs")
        previous_time = times[-1]
        _same(expected["row_ids_sha256"], _sha_ids(np.asarray(ids)), name + ".row_ids_sha256", errors)
        _same(expected["labels_sha256"], _sha_labels(np.asarray(labels)), name + ".labels_sha256", errors)
        if sum(labels) != expected["positive_count"]:
            errors.append(name + " positive count differs")
        total_rows += len(rows); total_positive += sum(labels)
    if total_rows != profile["row_count"] or total_positive != profile["positive_count"]:
        errors.append("score partitions disagree with source profile totals")


def verify_algorithm(bundle: Path, check_reviews: bool = True) -> list[str]:
    """Reconstruct 0.5 or 0.6 evidence without loading joblib/pickle files."""
    bundle = Path(bundle)
    errors: list[str] = []
    try:
        if not bundle.is_dir() or bundle.is_symlink() or (bundle / MANIFEST).is_symlink():
            return ["unsafe algorithm bundle or manifest"]
        manifest = read(bundle / MANIFEST)
        if not isinstance(manifest, dict) or manifest.get("schema_version") not in (SCHEMA, RESILIENCE_SCHEMA):
            return ["unsupported algorithm schema"]
        schema = manifest["schema_version"]
        required = REQUIRED_RESILIENCE if schema == RESILIENCE_SCHEMA else REQUIRED
        if manifest.get("required_artifacts") != list(required) or not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != set(required):
            return ["algorithm artifact allowlist differs"]
        if {path.name for path in bundle.iterdir()} != set(required) | {MANIFEST}:
            return ["unlisted or missing algorithm files"]
        for name, digest in manifest["artifacts"].items():
            path = bundle / name
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or path.is_symlink() or not path.is_file():
                errors.append("unsafe algorithm artifact or digest: " + name)
            elif sha(path) != digest:
                errors.append("hash mismatch: " + name)
        if errors:
            return errors
        metadata, profile = read(bundle / "source_metadata.json"), read(bundle / "source_profile.json")
        model_record = read(bundle / "algorithm_models.json")
        scores, policy = read(bundle / "algorithm_scores.json"), read(bundle / "fusion_policy.json")
        results, meta = read(bundle / "algorithm_results.json"), read(bundle / "run_metadata.json")
        validate_source_identity(metadata, profile)
        if metadata.get("classification") not in ("public_ulb_creditcard", "synthetic_smoke_fixture"):
            errors.append("unsupported source classification")
        for key in ("run_id", "project_version", "source_classification"):
            if meta.get(key) != manifest.get(key):
                errors.append("manifest/run metadata disagree: " + key)
        if (meta.get("schema_version") != schema or meta.get("completed") is not True
                or meta.get("raw_source_included") is not False
                or meta.get("source_sha256") != profile.get("source_sha256")
                or meta.get("source_classification") != metadata.get("classification")
                or model_record.get("seed") != meta.get("seed")):
            errors.append("run schema/source/seed completion claim differs")
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("code_sha256", ""))) or not meta.get("code_revision"):
            errors.append("missing source code identity")
        if (model_record.get("schema_version") != schema or model_record.get("model_names") != list(MODEL_NAMES)
                or model_record.get("feature_columns") != FEATURES
                or model_record.get("split_targets") != [0.55, 0.15, 0.15, 0.15]
                or set(model_record.get("partitions", {})) != set(PARTITIONS)
                or set(model_record.get("calibrators", {})) != set(MODEL_NAMES)):
            errors.append("model protocol differs")
        partitions = model_record["partitions"]
        if (partitions["train"]["time_min"] != profile.get("time_range_seconds", [None, None])[0]
                or partitions["test"]["time_max"] != profile.get("time_range_seconds", [None, None])[-1]
                or profile.get("invalid_count") != 0
                or any(partitions[left]["time_max"] >= partitions[right]["time_min"]
                       for left, right in zip(PARTITIONS[:-1], PARTITIONS[1:]))):
            errors.append("source profile/partition chronology differs")
        expected_specs = _model_spec(meta["seed"], model_record["partitions"]["train"]["positive_count"],
                                     model_record["partitions"]["train"]["row_count"] - model_record["partitions"]["train"]["positive_count"],
                                     rf_n_jobs=1 if schema == RESILIENCE_SCHEMA else 2)
        _same(expected_specs, model_record["models"], "model specifications", errors)
        fit_seconds = model_record.get("fit_seconds")
        if (not isinstance(fit_seconds, dict) or set(fit_seconds) != set(MODEL_NAMES)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0 for value in fit_seconds.values())):
            errors.append("invalid model fit timing record")
        for name, params in model_record["calibrators"].items():
            expected_kind = "decision_function" if name in ("logistic_regression", "hist_gradient_boosting") else "predict_proba"
            if (params.get("method") != "sigmoid" or params.get("input_kind") != expected_kind
                    or any(isinstance(params.get(key), bool) or not isinstance(params.get(key), (int, float))
                           or not math.isfinite(params[key]) for key in ("a", "b"))):
                errors.append("calibrator record differs: " + name)
        _verify_scores(scores, model_record, profile, errors)
        if errors:
            return errors
        if schema == RESILIENCE_SCHEMA and not re.fullmatch(
                r"[0-9a-f]{64}", str(model_record.get("test_feature_batch_sha256", ""))):
            errors.append("missing ordered feature batch fingerprint")
            return errors
        validation = scores["validation"]
        ids, labels, times = _row_values(validation)
        expected_policy = fit_fusion(_matrix(validation), labels, times, ids, MODEL_NAMES,
                                     policy["budget_fraction"])
        _same(expected_policy, policy, "fusion policy", errors)
        capacity = results["capacity"]
        if (capacity.get("classification") != "repeated_feature_inference_load_only"
                or capacity.get("models_scored") != list(MODEL_NAMES)
                or capacity.get("batch_limit") != 8192 or capacity.get("independent_accuracy_rows") != 0
                or not isinstance(capacity.get("workloads"), list)
                or [item.get("scored_rows") for item in capacity["workloads"]] != sorted({min(100_000, meta["load_rows"]), meta["load_rows"]})):
            errors.append("load protocol differs")
        else:
            test_matrix = _matrix(scores["test"])
            machine = capacity.get("machine")
            if (not isinstance(machine, dict) or machine.get("threadpool_limit") != 2
                    or any(not isinstance(machine.get(key), str) or not machine[key]
                           for key in ("platform", "architecture", "python", "sklearn", "xgboost"))
                    or (machine.get("cpu_count") is not None and
                        (isinstance(machine["cpu_count"], bool) or not isinstance(machine["cpu_count"], int)
                         or machine["cpu_count"] < 1))
                    or machine.get("xgboost") != model_record.get("packages", {}).get("xgboost")
                    or machine.get("sklearn") != model_record.get("packages", {}).get("scikit-learn")):
                errors.append("load machine/package claim differs")
            for item in capacity["workloads"]:
                elapsed = item.get("elapsed_seconds")
                rate = item.get("scores_per_second")
                unit = item.get("process_maxrss_source_unit")
                raw_rss = item.get("process_maxrss_raw")
                bytes_rss = item.get("process_maxrss_bytes")
                if (item.get("batch_size_max") != 8192 or isinstance(elapsed, bool)
                        or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0
                        or isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate)
                        or rate <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("score_stream_sha256", "")))):
                    errors.append("invalid measured load record")
                elif item["score_stream_sha256"] != _score_stream_digest(test_matrix, policy["weights"], item["scored_rows"]):
                    errors.append("load score stream differs from archived test probabilities")
                elif not math.isclose(rate, item["scored_rows"] / elapsed,
                                      rel_tol=1e-9, abs_tol=1e-9):
                    errors.append("load throughput differs from elapsed time")
                if unit not in ("bytes", "kibibytes", "unavailable"):
                    errors.append("load RSS unit differs")
                elif unit == "unavailable":
                    if raw_rss is not None or bytes_rss is not None:
                        errors.append("unavailable load RSS has values")
                elif (isinstance(raw_rss, bool) or not isinstance(raw_rss, (int, float))
                      or not math.isfinite(raw_rss) or raw_rss < 0
                      or isinstance(bytes_rss, bool) or not isinstance(bytes_rss, (int, float))
                      or not math.isfinite(bytes_rss) or bytes_rss != raw_rss * (1024 if unit == "kibibytes" else 1)):
                    errors.append("load RSS conversion differs")
        if errors:
            return errors
        expected_results = _results(scores, policy, capacity)
        if schema == RESILIENCE_SCHEMA:
            resilience_policy = read(bundle / "resilience_policy.json")
            expected_resilience_policy = fit_resilience_policy(
                _matrix(validation), labels, times, ids, MODEL_NAMES,
                policy["budget_fraction"])
            _same(expected_resilience_policy, resilience_policy, "resilience policy", errors)
            test_ids, test_labels, _ = _row_values(scores["test"])
            expected_context = _resilience_context(profile, model_record, bundle, test_ids)
            resilience_evidence = read(bundle / "resilience_evidence.json")
            expected_evidence = reconstruct_resilience(
                _matrix(scores["test"]), test_ids, expected_context,
                expected_resilience_policy, policy["weights"], policy["threshold"])
            measurement = resilience_evidence.get("runtime_measurement")
            if (not isinstance(measurement, dict) or set(measurement) != {
                    "classification", "elapsed_seconds", "model_call_count",
                    "feature_rows_per_scenario"}
                    or measurement["classification"] != "eight_scenario_actual_model_callbacks_not_capacity_probe"
                    or isinstance(measurement["elapsed_seconds"], bool)
                    or not isinstance(measurement["elapsed_seconds"], (int, float))
                    or not math.isfinite(measurement["elapsed_seconds"])
                    or measurement["elapsed_seconds"] <= 0
                    or type(measurement["model_call_count"]) is not int
                    or measurement["model_call_count"] != sum(
                        sum(item["call_counts"].values()) for item in expected_evidence["scenarios"])
                    or type(measurement["feature_rows_per_scenario"]) is not int
                    or measurement["feature_rows_per_scenario"] != len(test_ids)):
                errors.append("resilience runtime measurement differs")
            transcript = {key: value for key, value in resilience_evidence.items()
                          if key != "runtime_measurement"}
            _same(expected_evidence, transcript, "resilience evidence", errors)
            expected_results["resilience"] = _resilience_posthoc(
                expected_evidence, test_labels, policy, expected_results)
        _same(expected_results, results, "algorithm results", errors)
        expected_gate = (_resilience_gate(metadata["classification"])
                         if schema == RESILIENCE_SCHEMA else policy_for(metadata["classification"]))
        _same(expected_gate, read(bundle / "release_gate.json"), "release gate", errors)
        renderer = render_resilience_report if schema == RESILIENCE_SCHEMA else render_algorithm_report
        if (bundle / "review_report.html").read_bytes() != renderer(bundle):
            errors.append("algorithm HTML report differs from evidence")
        if check_reviews:
            read_reviews(bundle)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
        errors.append("algorithm evidence malformed: " + str(exc))
    return errors
