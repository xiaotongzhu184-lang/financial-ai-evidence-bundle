"""Pure, deterministic validation-period fusion and review-queue policy."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from itertools import product
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def _fraction(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("budget_fraction must be finite and strictly between zero and one")
    return float(value)


def _probabilities(values: Sequence[float], name: str = "scores") -> list[float]:
    if isinstance(values, (Mapping, Set)):
        raise ValueError(f"{name} must be an ordered nonempty sequence")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence") from error
    if not items:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item) or not 0 <= item <= 1 for item in items):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1] without booleans")
    return [float(item) for item in items]


def _row_ids(values: Sequence[int], expected: int) -> list[int]:
    if isinstance(values, (Mapping, Set)):
        raise ValueError("row_ids must be an ordered sequence of unique nonnegative integers")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("row_ids must be a sequence of unique nonnegative integers") from error
    if len(items) != expected or any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0 for item in items):
        raise ValueError("row_ids must be aligned nonnegative integers without booleans")
    ids = [int(item) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("row_ids must be unique")
    return ids


def _binary_labels(values: Sequence[int], expected: int) -> list[int]:
    if isinstance(values, (Mapping, Set)):
        raise ValueError("labels must be an ordered sequence of aligned binary integers")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("labels must be aligned binary integers") from error
    if len(items) != expected or any(isinstance(item, bool) or not isinstance(item, Integral) or item not in (0, 1) for item in items):
        raise ValueError("labels must be aligned binary integers without booleans")
    labels = [int(item) for item in items]
    if set(labels) != {0, 1}:
        raise ValueError("labels must include both classes")
    return labels


def _confusion(labels: list[int], selected: list[bool]) -> dict:
    tp = sum(label == 1 and chosen for label, chosen in zip(labels, selected))
    fp = sum(label == 0 and chosen for label, chosen in zip(labels, selected))
    fn = sum(label == 1 and not chosen for label, chosen in zip(labels, selected))
    tn = len(labels) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": len(labels)}


def apply_policy(scores: Sequence[float], row_ids: Sequence[int], threshold: float,
                 budget_fraction: float) -> dict:
    """Apply a strict threshold followed by a deterministic hard review cap."""
    probabilities = _probabilities(scores)
    ids = _row_ids(row_ids, len(probabilities))
    budget = _fraction(budget_fraction)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability without booleans")
    capacity = math.floor(budget * len(probabilities))
    threshold_alerted = [score > threshold for score in probabilities]
    ranked = sorted((index for index, alerted in enumerate(threshold_alerted) if alerted),
                    key=lambda index: (-probabilities[index], ids[index]))
    selected = set(ranked[:capacity])
    reviewed = [index in selected for index in range(len(probabilities))]
    deferred = [alerted and not selected_flag for alerted, selected_flag in zip(threshold_alerted, reviewed)]
    return {
        "threshold_alerted": threshold_alerted,
        "reviewed": reviewed,
        "deferred": deferred,
        "capacity": capacity,
        "threshold_alert_count": len(ranked),
        "reviewed_count": len(selected),
        "deferred_count": len(ranked) - len(selected),
        "budget_utilization": len(selected) / capacity if capacity else 0.0,
    }


def assess_scores(labels: Sequence[int], scores: Sequence[float], row_ids: Sequence[int],
                  threshold: float, budget_fraction: float) -> dict:
    """Measure ranking quality and distinct threshold-only/capped queues."""
    probabilities = _probabilities(scores)
    truth = _binary_labels(labels, len(probabilities))
    queue = apply_policy(probabilities, row_ids, threshold, budget_fraction)
    counts = _confusion(truth, queue["reviewed"])
    threshold_counts = _confusion(truth, queue["threshold_alerted"])
    return {
        "average_precision": float(average_precision_score(truth, probabilities)),
        "brier": float(brier_score_loss(truth, probabilities)),
        "counts": counts,
        "threshold_counts": threshold_counts,
        "precision": counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0,
        "recall": counts["tp"] / (counts["tp"] + counts["fn"]),
        "fpr": counts["fp"] / (counts["fp"] + counts["tn"]),
        "capacity": queue["capacity"],
        "threshold_alert_count": queue["threshold_alert_count"],
        "reviewed_count": queue["reviewed_count"],
        "deferred_count": queue["deferred_count"],
        "budget_utilization": queue["budget_utilization"],
    }


def _time_values(values: Sequence[float], expected: int) -> list[float]:
    if isinstance(values, (Mapping, Set)):
        raise ValueError("times must be an ordered sequence of aligned finite relative times")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("times must be aligned finite relative times") from error
    if len(items) != expected or any(isinstance(item, bool) or not isinstance(item, Real) or
                                     not math.isfinite(item) or item < 0 for item in items):
        raise ValueError("times must be aligned finite nonnegative relative times without booleans")
    return [float(item) for item in items]


def _split_after_ties(order: list[int], times: list[float]) -> tuple[list[int], list[int]]:
    boundary = len(order) // 2
    while 0 < boundary < len(order) and times[order[boundary - 1]] == times[order[boundary]]:
        boundary += 1
    if boundary == 0 or boundary == len(order):
        raise ValueError("equal-time groups leave an empty chronological period")
    return order[:boundary], order[boundary:]


def _top_budget(scores: np.ndarray, ids: list[int], budget: float) -> set[int]:
    capacity = math.floor(budget * len(scores))
    return set(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), ids[index]))[:capacity])


def _frozen_policy(matrix: np.ndarray, policy: list[int], ids: list[int],
                   weights: list[float], budget: float) -> dict:
    policy_scores = matrix[policy] @ np.asarray(weights, dtype=np.float64)
    capacity = math.floor(budget * len(policy))
    threshold = float(sorted(policy_scores, reverse=True)[capacity])
    queue = apply_policy(policy_scores.tolist(), [ids[index] for index in policy], threshold, budget)
    return {
        "weights": weights,
        "threshold": threshold,
        "capacity": queue["capacity"],
        "threshold_alert_count": queue["threshold_alert_count"],
        "reviewed_count": queue["reviewed_count"],
    }


def fit_fusion(scores: Sequence[Sequence[float]], labels: Sequence[int], times: Sequence[float],
               row_ids: Sequence[int], model_names: Sequence[str], budget_fraction: float = .005) -> dict:
    """Fit a fixed quarter-simplex mixture without consuming test-period labels."""
    budget = _fraction(budget_fraction)
    if isinstance(model_names, (str, bytes, Mapping, Set)):
        raise ValueError("model_names must identify four distinct models")
    try:
        names = list(model_names)
    except TypeError as error:
        raise ValueError("model_names must identify four distinct models") from error
    if len(names) != 4 or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != 4:
        raise ValueError("model_names must identify four distinct nonempty models")
    if isinstance(scores, (Mapping, Set)):
        raise ValueError("scores must be an ordered n-by-4 probability matrix")
    try:
        rows = list(scores)
    except TypeError as error:
        raise ValueError("scores must be an aligned n-by-4 probability matrix") from error
    if not rows:
        raise ValueError("scores must be a nonempty n-by-4 probability matrix")
    matrix_rows = [_probabilities(row, f"scores[{index}]") for index, row in enumerate(rows)]
    if any(len(row) != 4 for row in matrix_rows):
        raise ValueError("scores must have exactly four model columns")
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    truth = _binary_labels(labels, len(matrix))
    relative_times = _time_values(times, len(matrix))
    ids = _row_ids(row_ids, len(matrix))
    chronological = sorted(range(len(matrix)), key=lambda index: (relative_times[index], ids[index]))
    tuning, policy = _split_after_ties(chronological, relative_times)
    tune_first, tune_second = _split_after_ties(tuning, relative_times)
    for period in (tune_first, tune_second, policy):
        if {truth[index] for index in period} != {0, 1}:
            raise ValueError("each chronological tuning half and policy period must include both classes")
    candidates = []
    best_index: int | None = None
    best_single_candidate: int | None = None
    for quarters in product(range(5), repeat=4):
        if sum(quarters) != 4:
            continue
        weights = [part / 4 for part in quarters]
        fused = matrix @ np.asarray(weights, dtype=np.float64)
        half_recalls = []
        for period in (tune_first, tune_second):
            selected = _top_budget(fused[period], [ids[index] for index in period], budget)
            positives = sum(truth[index] for index in period)
            half_recalls.append(sum(truth[period[position]] for position in selected) / positives)
        candidate = {
            "weights": weights,
            "half_recalls": half_recalls,
            "minimum_recall": min(half_recalls),
            "average_precision": float(average_precision_score([truth[index] for index in tuning], fused[tuning])),
        }
        candidates.append(candidate)
        position = len(candidates) - 1
        if (best_index is None or
            (candidate["minimum_recall"], candidate["average_precision"]) >
            (candidates[best_index]["minimum_recall"], candidates[best_index]["average_precision"])):
            best_index = position
        if 1.0 in weights:
            if (best_single_candidate is None or
                (candidate["minimum_recall"], candidate["average_precision"]) >
                (candidates[best_single_candidate]["minimum_recall"],
                 candidates[best_single_candidate]["average_precision"])):
                best_single_candidate = position
    if best_index is None or best_single_candidate is None:
        raise RuntimeError("fusion policy search produced no valid candidate")
    chosen = candidates[best_index]
    best_single_index = candidates[best_single_candidate]["weights"].index(1.0)
    single_policies = [
        _frozen_policy(matrix, policy, ids, [float(index == model_index) for index in range(4)], budget)
        for model_index in range(4)
    ]
    ablation_policies = {
        "single_models": single_policies,
        "equal_weight": _frozen_policy(matrix, policy, ids, [.25] * 4, budget),
        "best_single": single_policies[best_single_index],
        "robust_weighted": _frozen_policy(matrix, policy, ids, chosen["weights"], budget),
    }
    robust_policy = ablation_policies["robust_weighted"]
    return {
        "model_names": names,
        "weights": chosen["weights"],
        "budget_fraction": budget,
        "tuning_row_ids": [ids[index] for index in tuning],
        "tuning_half_row_ids": [[ids[index] for index in period] for period in (tune_first, tune_second)],
        "policy_row_ids": [ids[index] for index in policy],
        "threshold": robust_policy["threshold"],
        "best_single_index": best_single_index,
        "selected_candidate_index": best_index,
        "candidates": candidates,
        "ablation_policies": ablation_policies,
        "policy_capacity": robust_policy["capacity"],
        "policy_threshold_alert_count": robust_policy["threshold_alert_count"],
        "policy_reviewed_count": robust_policy["reviewed_count"],
    }
