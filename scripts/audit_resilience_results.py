#!/usr/bin/env python3
"""Independently audit 0.6 recovery results using only Python's standard library.

Usage: python scripts/audit_resilience_results.py BUNDLE [BUNDLE ...]
This checks sealed artifact hashes, deterministic routes and score streams,
retry receipts, review queues, average precision, and before/after outcomes.
It does not authenticate the external dataset or execute serialized models.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path


SCENARIOS = (
    "clean", "strongest_model_dropout", "one_model_row_roll",
    "eight_bit_quantization", "persistent_strongest_model_dropout",
    "reordered_records", "stale_context", "all_models_unavailable",
)
MODEL_NAMES = ("logistic_regression", "random_forest",
               "hist_gradient_boosting", "xgboost")
TRANSIENT = {"strongest_model_dropout", "one_model_row_roll",
             "eight_bit_quantization", "stale_context"}


def fail(message: str) -> None:
    raise ValueError(message)


def read_json(path: Path):
    def reject_constant(value):
        fail("nonfinite JSON constant: " + value)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def close(actual, expected, path="value"):
    """Compare parsed evidence including all fields, with numeric roundoff only."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            fail(path + " keys differ")
        for key, value in expected.items():
            close(actual[key], value, path + "." + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            fail(path + " list differs")
        for index, value in enumerate(expected):
            close(actual[index], value, path + f"[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isclose(actual, expected,
                        rel_tol=1e-9, abs_tol=1e-9)):
            fail(path + " number differs")
    elif type(actual) is not type(expected) or actual != expected:
        fail(path + " differs")


def exact(actual, expected, path="value"):
    """Compare policy fields without a tolerance that could move a queue cut."""
    if type(actual) is not type(expected):
        fail(path + " type differs")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            fail(path + " keys differ")
        for key, value in expected.items():
            exact(actual[key], value, path + "." + str(key))
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            fail(path + " length differs")
        for index, value in enumerate(expected):
            exact(actual[index], value, path + f"[{index}]")
    elif actual != expected:
        fail(path + " differs")


def average_precision(labels, scores):
    """Precision at each distinct score, weighted by positives in that tie."""
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        fail("AP requires aligned binary labels with both classes")
    ordering = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_positive = sum(labels)
    seen = true_positive = 0
    weighted_precision = 0.0
    cursor = 0
    while cursor < len(ordering):
        score = scores[ordering[cursor]]
        group_positive = group_size = 0
        while cursor < len(ordering) and scores[ordering[cursor]] == score:
            group_positive += labels[ordering[cursor]]
            group_size += 1
            cursor += 1
        seen += group_size
        true_positive += group_positive
        weighted_precision += group_positive * true_positive / seen
    return weighted_precision / total_positive


def validation_frozen_policy(rows, names, budget):
    """Refit all 15 fallback policies from archived validation rows only."""
    if (not isinstance(rows, list) or not rows or list(names) != list(MODEL_NAMES)
            or type(budget) not in (int, float) or not math.isfinite(budget)
            or not 0 < budget < 1):
        fail("invalid validation policy inputs")
    ids = [row["row_id"] for row in rows]
    labels = [row["label"] for row in rows]
    times = [row["relative_time"] for row in rows]
    if (len(set(ids)) != len(ids) or any(type(item) is not int or item < 0 for item in ids)
            or any(type(item) is not int or item not in (0, 1) for item in labels)
            or any(type(item) not in (int, float) or not math.isfinite(item) or item < 0
                   for item in times)):
        fail("invalid validation identities, labels, or time")
    columns = {name: [row["models"][name]["calibrated_score"] for row in rows]
               for name in names}
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for values in columns.values() for value in values):
        fail("invalid validation probability")
    chronological = sorted(range(len(rows)), key=lambda i: (times[i], ids[i]))
    boundary = len(rows) // 2
    while 0 < boundary < len(rows) and times[chronological[boundary - 1]] == times[chronological[boundary]]:
        boundary += 1
    if not 0 < boundary < len(rows):
        fail("validation time ties make frozen period empty")
    tuning, freeze = chronological[:boundary], chronological[boundary:]
    tuning_labels = [labels[i] for i in tuning]
    if set(tuning_labels) != {0, 1}:
        fail("validation tuning period lacks both classes")
    selection_metrics = {}
    capacity = math.floor(budget * len(tuning))
    for name in names:
        tuning_scores = [columns[name][i] for i in tuning]
        top = sorted(tuning, key=lambda i: (-columns[name][i], ids[i]))[:capacity]
        selection_metrics[name] = {
            "average_precision": average_precision(tuning_labels, tuning_scores),
            "budget_recall": sum(labels[i] for i in top) / sum(tuning_labels),
        }
    subsets = []
    for size in range(1, 5):
        for indices in combinations(range(4), size):
            selected = min(indices, key=lambda j: (
                -selection_metrics[names[j]]["average_precision"],
                -selection_metrics[names[j]]["budget_recall"], names[j]))
            threshold_rank = math.floor(budget * len(freeze))
            threshold = sorted((columns[names[selected]][i] for i in freeze),
                               reverse=True)[threshold_rank]
            subsets.append({"healthy_models": [names[j] for j in indices],
                            "selected_model": names[selected],
                            "weights": [float(j == selected) for j in range(4)],
                            "threshold": float(threshold)})
    payload = {"schema_version": "0.6-resilience-policy", "model_names": list(names),
               "budget_fraction": budget, "tuning_row_ids": [ids[i] for i in tuning],
               "policy_row_ids": [ids[i] for i in freeze],
               "model_selection_metrics": selection_metrics, "subsets": subsets}
    return {**payload, "payload_sha256": digest(payload)}


def queue_and_metrics(labels, scores, row_ids, threshold, budget):
    n = len(scores)
    if (not n or len(labels) != n or len(row_ids) != n or len(set(row_ids)) != n
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in scores)
            or type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1 or not 0 < budget < 1):
        fail("malformed queue input")
    capacity = math.floor(n * budget)
    alert = [value > threshold for value in scores]
    ranked = sorted((i for i, flag in enumerate(alert) if flag),
                    key=lambda i: (-scores[i], row_ids[i]))
    chosen = set(ranked[:capacity])
    reviewed = [i in chosen for i in range(n)]
    deferred = [flag and not reviewed[i] for i, flag in enumerate(alert)]
    def counts(mask):
        tp = sum(label == 1 and mask[i] for i, label in enumerate(labels))
        fp = sum(label == 0 and mask[i] for i, label in enumerate(labels))
        fn = sum(label == 1 and not mask[i] for i, label in enumerate(labels))
        tn = n - tp - fp - fn
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": n}
    reviewed_counts = counts(reviewed)
    result_queue = {"threshold_alerted": alert, "reviewed": reviewed,
                    "deferred": deferred, "capacity": capacity,
                    "threshold_alert_count": len(ranked), "reviewed_count": len(chosen),
                    "deferred_count": len(ranked) - len(chosen),
                    "budget_utilization": len(chosen) / capacity if capacity else 0.0}
    result_metrics = {
        "average_precision": average_precision(labels, scores),
        "brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / n,
        "counts": reviewed_counts, "threshold_counts": counts(alert),
        "precision": reviewed_counts["tp"] / (reviewed_counts["tp"] + reviewed_counts["fp"])
        if reviewed_counts["tp"] + reviewed_counts["fp"] else 0.0,
        "recall": reviewed_counts["tp"] / (reviewed_counts["tp"] + reviewed_counts["fn"]),
        "fpr": reviewed_counts["fp"] / (reviewed_counts["fp"] + reviewed_counts["tn"]),
        **{key: result_queue[key] for key in (
            "capacity", "threshold_alert_count", "reviewed_count",
            "deferred_count", "budget_utilization")},
    }
    return result_queue, result_metrics


def score_delivery(original, row_ids, context, model, name, strongest, attempt):
    """Implement the published eight fault cases separately from package code."""
    delivery = {"context": context, "model": model, "row_ids": list(row_ids),
                "scores": list(original)}
    if name == "reordered_records":
        delivery["row_ids"].reverse()
        delivery["scores"].reverse()
    elif model == strongest and (attempt == 1 or name == "persistent_strongest_model_dropout"):
        if name in {"strongest_model_dropout", "persistent_strongest_model_dropout"}:
            delivery["scores"] = [0.0] * len(original)
        elif name == "one_model_row_roll":
            delivery["scores"] = [original[-1], *original[:-1]]
    if name == "eight_bit_quantization" and attempt == 1:
        delivery["scores"] = [round(value * 255) / 255 for value in original]
    if name == "stale_context" and model == strongest and attempt == 1:
        delivery["context"] = {"stale_context": context}
    return delivery


def audit_bundle(bundle: Path):
    manifest = read_json(bundle / "bundle_manifest.json")
    if manifest.get("schema_version") != "0.6-algorithm":
        fail("not a 0.6 algorithm bundle")
    required = set(manifest["required_artifacts"])
    if required != set(manifest["artifacts"]) or not {
            "algorithm_scores.json", "fusion_policy.json", "resilience_policy.json",
            "resilience_evidence.json", "algorithm_results.json"}.issubset(required):
        fail("artifact allowlist mismatch")
    for name, expected_hash in manifest["artifacts"].items():
        if name != Path(name).name or name in (".", "..") or not (bundle / name).is_file():
            fail("unsafe or missing artifact")
        actual_hash = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            fail("artifact hash mismatch: " + name)
    scores = read_json(bundle / "algorithm_scores.json")
    fusion = read_json(bundle / "fusion_policy.json")
    fallback = read_json(bundle / "resilience_policy.json")
    evidence = read_json(bundle / "resilience_evidence.json")
    results = read_json(bundle / "algorithm_results.json")
    names = fallback["model_names"]
    if (names != list(MODEL_NAMES) or fusion["model_names"] != names
            or len(fusion["weights"]) != 4):
        fail("invalid four-model policy")
    rows = scores["test"]
    row_ids = [row["row_id"] for row in rows]
    labels = [row["label"] for row in rows]
    if not rows or len(set(row_ids)) != len(row_ids) or set(labels) != {0, 1}:
        fail("invalid held-out rows")
    columns = {name: [row["models"][name]["calibrated_score"] for row in rows]
               for name in names}
    budget = fusion["budget_fraction"]
    expected_fallback = validation_frozen_policy(scores["validation"], names, budget)
    expected_payload = {key: value for key, value in expected_fallback.items()
                        if key != "payload_sha256"}
    recorded_payload = {key: value for key, value in fallback.items()
                        if key != "payload_sha256"}
    if set(recorded_payload) != set(expected_payload):
        fail("validation-frozen policy keys differ")
    exact({key: value for key, value in recorded_payload.items()
           if key != "model_selection_metrics"},
          {key: value for key, value in expected_payload.items()
           if key != "model_selection_metrics"}, "validation-frozen policy")
    actual_metrics = recorded_payload["model_selection_metrics"]
    expected_metrics = expected_payload["model_selection_metrics"]
    if type(actual_metrics) is not dict or set(actual_metrics) != set(expected_metrics):
        fail("validation-frozen policy selection metric keys differ")
    for name in names:
        actual = actual_metrics[name]
        expected = expected_metrics[name]
        if type(actual) is not dict or set(actual) != set(expected):
            fail("validation-frozen policy selection metric shape differs")
        for metric in ("average_precision", "budget_recall"):
            value = actual[metric]
            if (type(value) is not float or not math.isfinite(value)
                    or not math.isclose(value, expected[metric], rel_tol=1e-14,
                                        abs_tol=1e-14)):
                fail("validation-frozen policy selection metric differs: " + name + "." + metric)
    # scikit-learn AP and this independent grouped-rank implementation can
    # differ by one floating-point ulp. Check the recorded payload hash against
    # its exact recorded bytes; all queue-affecting policy fields match exactly.
    if fallback.get("payload_sha256") != digest(recorded_payload):
        fail("validation-frozen policy payload digest differs")
    baseline_scores = [sum(fusion["weights"][j] * columns[name][i]
                           for j, name in enumerate(names)) for i in range(len(rows))]
    _, baseline_metrics = queue_and_metrics(labels, baseline_scores, row_ids,
                                            fusion["threshold"], budget)
    close(results["test_metrics"]["average_precision"],
          baseline_metrics["average_precision"], "baseline AP")
    close(evidence["row_ids"], row_ids, "resilience row IDs")
    strongest = names[max(range(4), key=lambda j: fusion["weights"][j])]
    if evidence["strongest_model"] != strongest or fallback["budget_fraction"] != budget:
        fail("resilience policy/baseline mismatch")
    if [item["name"] for item in evidence["scenarios"]] != list(SCENARIOS):
        fail("scenario set/order differs")
    reported = results["resilience"]
    if reported.get("classification") != "retrospective_posthoc_quality_not_runtime_input":
        fail("recovery result classification differs")
    close(reported.get("baseline_average_precision"),
          baseline_metrics["average_precision"], "recovery baseline AP")
    if [item["name"] for item in reported["scenarios"]] != list(SCENARIOS):
        fail("reported scenario set/order differs")
    quality_limit = reported["maximum_allowed_average_precision_drop"]
    if not math.isclose(quality_limit, .10):
        fail("quality limit differs")
    context = evidence["context"]
    measurement = evidence["runtime_measurement"]
    actual_calls = sum(sum(item["call_counts"].values()) for item in evidence["scenarios"])
    if (measurement.get("classification") != "eight_scenario_actual_model_callbacks_not_capacity_probe"
            or type(measurement.get("model_call_count")) is not int
            or measurement["model_call_count"] != actual_calls
            or type(measurement.get("feature_rows_per_scenario")) is not int
            or measurement["feature_rows_per_scenario"] != len(rows)
            or type(measurement.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(measurement["elapsed_seconds"])
            or measurement["elapsed_seconds"] <= 0):
        fail("runtime measurement does not match attempt counts")
    summary = []
    for item, report in zip(evidence["scenarios"], reported["scenarios"]):
        name = item["name"]
        expected_counts = {model: 2 if name == "all_models_unavailable" or (
            model == strongest and name in TRANSIENT | {"persistent_strongest_model_dropout"})
                           or name == "eight_bit_quantization" else 1 for model in names}
        # Eight-bit quantization alters every first model delivery, so all retry.
        if name == "eight_bit_quantization":
            expected_counts = dict.fromkeys(names, 2)
        close(item["call_counts"], expected_counts, name + ".calls")
        events = item["attempts"]
        if len(events) != sum(expected_counts.values()):
            fail(name + " attempt count differs")
        cursor = 0
        for model in names:
            for attempt in range(1, expected_counts[model] + 1):
                event = events[cursor]
                cursor += 1
                if event["model"] != model or event["attempt"] != attempt:
                    fail(name + " attempt order differs")
                reject = name == "all_models_unavailable" or (
                    attempt == 1 and model == strongest and name in TRANSIENT) or (
                    model == strongest and name == "persistent_strongest_model_dropout") or (
                    attempt == 1 and name == "eight_bit_quantization")
                if event["status"] != ("rejected" if reject else "accepted"):
                    fail(name + " attempt status differs")
                if name == "all_models_unavailable":
                    if event["receipt_sha256"] is not None or event["delivered_sha256"] is not None:
                        fail("unavailable model cannot have a receipt")
                    continue
                original = columns[model]
                receipt = {"context": context, "model": model,
                           "records": sorted(zip(row_ids, original))}
                delivery = score_delivery(original, row_ids, context, model,
                                          name, strongest, attempt)
                if event["receipt_sha256"] != digest(receipt) or event["delivered_sha256"] != digest(delivery):
                    fail(name + " receipt/delivery hash differs")
                if reject:
                    close(event["delivery"], delivery, name + ".failed_delivery")
        fail_closed = name == "all_models_unavailable"
        if fail_closed:
            if (item["status"] != "fail_closed" or item["route"] != "fail_closed"
                    or item["decision_count"] != 0 or item["coverage"] != 0
                    or item["scores"] != [] or item["queue"] is not None
                    or report["protected_metrics"] is not None
                    or report["quality_status"] != "not_applicable"):
                fail("all-model loss did not fail closed")
            summary.append({"scenario": name, "route": "fail_closed", "coverage": 0,
                            "protected_ap": None, "quality": "not_applicable"})
            continue
        if name == "persistent_strongest_model_dropout":
            healthy = [model for model in names if model != strongest]
            subset = next((part for part in fallback["subsets"]
                           if part["healthy_models"] == healthy), None)
            if subset is None or subset["selected_model"] not in healthy:
                fail("missing valid validation-frozen fallback")
            selected = subset["selected_model"]
            expected_scores = columns[selected]
            expected_route = "validation_frozen_fallback"
            expected_weights = [float(model == selected) for model in names]
            expected_threshold = subset["threshold"]
            expected_quarantined = [strongest]
        else:
            selected = None
            expected_scores = baseline_scores
            expected_route = "baseline_fusion"
            expected_weights = fusion["weights"]
            expected_threshold = fusion["threshold"]
            expected_quarantined = []
            healthy = list(names)
        if (item["status"] != "ok" or item["route"] != expected_route
                or item["selected_model"] != selected or item["healthy_models"] != healthy
                or item["quarantined_models"] != expected_quarantined
                or item["decision_count"] != len(rows) or item["coverage"] != 1):
            fail(name + " recovery route differs")
        close(item["weights"], expected_weights, name + ".weights")
        close(item["threshold"], expected_threshold, name + ".threshold")
        close(item["scores"], expected_scores, name + ".scores")
        queue, metrics = queue_and_metrics(labels, expected_scores, row_ids,
                                           expected_threshold, budget)
        close(item["queue"], queue, name + ".queue")
        close(report["protected_metrics"], metrics, name + ".protected_metrics")
        ap_loss = baseline_metrics["average_precision"] - metrics["average_precision"]
        close(report["observed_average_precision_drop"], ap_loss,
              name + ".AP_loss")
        expected_quality = "pass" if ap_loss <= quality_limit else "fail"
        if report["quality_status"] != expected_quality or report["runtime_status"] != "ok":
            fail(name + " quality status differs")
        control = results["stress"]["scenarios"].get(name)
        if control is not None:
            altered = {model: list(values) for model, values in columns.items()}
            if name == "strongest_model_dropout":
                altered[strongest] = [0.0] * len(rows)
            elif name == "one_model_row_roll":
                altered[strongest] = [columns[strongest][-1], *columns[strongest][:-1]]
            else:
                altered = {model: [round(value * 255) / 255 for value in values]
                           for model, values in columns.items()}
            old_scores = [sum(fusion["weights"][j] * altered[model][i]
                              for j, model in enumerate(names)) for i in range(len(rows))]
            _, old_metrics = queue_and_metrics(labels, old_scores, row_ids,
                                               fusion["threshold"], budget)
            old_ap = old_metrics["average_precision"]
            close(control["score_values"], old_scores, name + ".old_scores")
            close(control["metrics"], old_metrics, name + ".old_metrics")
            close(report["unprotected_control"]["average_precision"], old_ap,
                  name + ".reported_old_AP")
            old_loss = baseline_metrics["average_precision"] - old_ap
            close(control["observed_average_precision_drop"], old_loss,
                  name + ".control_AP_loss")
            close(report["unprotected_control"]["observed_average_precision_drop"],
                  old_loss, name + ".old_AP_loss")
            old_quality = "pass" if old_loss <= quality_limit else "fail"
            if (control["status"] != old_quality or
                    report["unprotected_control"]["quality_status"] != old_quality):
                fail(name + " old control quality status differs")
        elif report["unprotected_control"] is not None:
            fail(name + " fabricated unprotected control")
        summary.append({"scenario": name, "route": expected_route,
                        "coverage": item["coverage"],
                        "unprotected_ap": report["unprotected_control"]["average_precision"]
                        if report["unprotected_control"] else None,
                        "protected_ap": metrics["average_precision"],
                        "protected_tp": metrics["counts"]["tp"],
                        "protected_fp": metrics["counts"]["fp"],
                        "protected_fn": metrics["counts"]["fn"],
                        "quality": expected_quality})
    return {"bundle": str(bundle), "status": "pass", "baseline_ap": baseline_metrics["average_precision"],
            "recovery_model_calls": actual_calls,
            "recovery_elapsed_seconds_self_reported": measurement["elapsed_seconds"],
            "scenarios": summary}


def main(args):
    if not args:
        print("usage: audit_resilience_results.py BUNDLE [BUNDLE ...]", file=sys.stderr)
        return 2
    results = []
    failed = False
    for raw in args:
        try:
            results.append(audit_bundle(Path(raw)))
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            failed = True
            results.append({"bundle": raw, "status": "fail", "error": str(error)})
    print(json.dumps({"status": "fail" if failed else "pass", "bundles": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
