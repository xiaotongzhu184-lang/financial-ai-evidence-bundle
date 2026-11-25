"""Frozen-reference score drift and chronological review-queue diagnostics.

Only the PSI arithmetic is derived from Evidently; reference-only bins,
additive smoothing, windowing and queue assessment are YK Evidence changes.
ULB Time is relative seconds. No calendar date or production action is inferred.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from yk_evidence._vendor.evidently_psi import psi_components
from yk_evidence.fusion import apply_policy, assess_scores


_SOURCE = {
    "repository": "evidentlyai/evidently",
    "commit": "34771fe41c08ca074d96a52aaf4a45cd640cbaf5",
    "path": "src/evidently/legacy/calculations/stattests/psi.py",
    "license": "Apache-2.0",
    "modified": True,
}


def _ordered(values: Sequence, name: str) -> list:
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValueError(f"{name} must be an ordered sequence")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an ordered sequence") from error
    if not items:
        raise ValueError(f"{name} must be nonempty")
    return items


def _scores(values: Sequence[float], name: str) -> list[float]:
    items = _ordered(values, name)
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or
           not 0 <= value <= 1 for value in items):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1] without booleans")
    return [float(value) for value in items]


def _ids(values: Sequence[int], expected: int, name: str) -> list[int]:
    items = _ordered(values, name)
    if len(items) != expected or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                                     for value in items):
        raise ValueError(f"{name} must be aligned nonnegative integer row IDs without booleans")
    ids = [int(value) for value in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate row IDs")
    return ids


def _times(values: Sequence[float], expected: int) -> list[float]:
    items = _ordered(values, "test_relative_times")
    if len(items) != expected or any(isinstance(value, bool) or not isinstance(value, Real) or
                                     not math.isfinite(value) or value < 0 for value in items):
        raise ValueError("test_relative_times must be aligned finite nonnegative relative seconds")
    return [float(value) for value in items]


def _labels(values: Sequence[int], expected: int) -> list[int]:
    items = _ordered(values, "test_labels")
    if len(items) != expected or any(isinstance(value, bool) or not isinstance(value, Integral) or
                                     value not in (0, 1) for value in items):
        raise ValueError("test_labels must be aligned binary integers without booleans")
    return [int(value) for value in items]


def _three_time_windows(times: list[float], ids: list[int]) -> list[list[int]]:
    order = sorted(range(len(times)), key=lambda index: (times[index], ids[index]))
    n = len(order)
    if n < 3:
        raise ValueError("three nonempty chronological windows require at least three rows")
    cuts = []
    for target in (n // 3, 2 * n // 3):
        boundary = max(target, cuts[-1] + 1 if cuts else 1)
        while boundary < n and times[order[boundary - 1]] == times[order[boundary]]:
            boundary += 1
        if boundary >= n:
            raise ValueError("equal-time groups cannot form three nonempty chronological windows")
        cuts.append(boundary)
    return [order[:cuts[0]], order[cuts[0]:cuts[1]], order[cuts[1]:]]


def monitor_score_windows(
    reference_scores: Sequence[float], reference_row_ids: Sequence[int],
    test_scores: Sequence[float], test_relative_times: Sequence[float],
    test_row_ids: Sequence[int], test_labels: Sequence[int],
    threshold: float, budget_fraction: float,
) -> dict:
    """Compare three later relative-time blocks with fixed reference score bins.

    All scores are probabilities and all IDs are original row identities.
    PSI is descriptive. Small windows are flagged, not interpreted as failure.
    """
    reference = _scores(reference_scores, "reference_scores")
    reference_ids = _ids(reference_row_ids, len(reference), "reference_row_ids")
    test = _scores(test_scores, "test_scores")
    test_ids = _ids(test_row_ids, len(test), "test_row_ids")
    if not set(reference_ids).isdisjoint(test_ids):
        raise ValueError("reference and test row IDs must be disjoint")
    times = _times(test_relative_times, len(test))
    labels = _labels(test_labels, len(test))
    # Reuse the public fusion policy seam for identical threshold/budget validation.
    apply_policy(test, test_ids, threshold, budget_fraction)

    if len(set(reference)) == 1:
        # A constant reference value at an interior point must have a bin
        # separate from both domain endpoints; otherwise an extreme score
        # shift can occupy the same bin and produce a false PSI of zero.
        value = reference[0]
        edges = np.unique([0.0, value / 2, (value + 1.0) / 2, 1.0])
    else:
        quantiles = np.quantile(np.asarray(reference), np.linspace(0.0, 1.0, 11), method="linear")
        edges = np.unique(np.concatenate(([0.0], quantiles[1:-1], [1.0])))
    n_bins = len(edges) - 1
    alpha = 0.5
    min_rows = max(30, 5 * n_bins)
    reference_counts = np.histogram(reference, bins=edges)[0].astype(int)
    if int(reference_counts.sum()) != len(reference):
        raise ValueError("reference histogram did not conserve all rows")
    reference_smoothed = (reference_counts + alpha) / (len(reference) + alpha * n_bins)
    windows = []
    for window_index, positions in enumerate(_three_time_windows(times, test_ids)):
        scores = [test[position] for position in positions]
        ids = [test_ids[position] for position in positions]
        truth = [labels[position] for position in positions]
        counts = np.histogram(scores, bins=edges)[0].astype(int)
        if int(counts.sum()) != len(positions):
            raise ValueError("test histogram did not conserve all rows")
        smoothed = (counts + alpha) / (len(positions) + alpha * n_bins)
        psi = psi_components(reference_smoothed, smoothed)
        queue = apply_policy(scores, ids, threshold, budget_fraction)
        two_classes = set(truth) == {0, 1}
        windows.append({
            "window_index": window_index,
            "row_ids": ids,
            "relative_time_min": times[positions[0]],
            "relative_time_max": times[positions[-1]],
            "row_count": len(positions),
            "positive_count": sum(truth),
            "bin_counts": counts.tolist(),
            "raw_proportions": (counts / len(positions)).tolist(),
            "smoothed_proportions": smoothed.tolist(),
            "psi_contributions": psi["contributions"],
            "psi": psi["psi"],
            "small_sample": len(positions) < min_rows,
            "queue": queue,
            "queue_metrics": assess_scores(truth, scores, ids, threshold, budget_fraction) if two_classes else None,
            "queue_metrics_status": "measured" if two_classes else "single_class_unavailable",
        })
    return {
        "schema_version": "score-monitor-1",
        "method": "evidently_psi_core_modified",
        "source": dict(_SOURCE),
        "parameters": {
            "requested_bins": 10,
            "effective_bins": n_bins,
            "alpha": alpha,
            "window_count": 3,
            "small_sample_min_rows": min_rows,
            "threshold": float(threshold),
            "budget_fraction": float(budget_fraction),
            "interpretation": "descriptive_only",
        },
        "reference": {
            "row_ids": reference_ids,
            "row_count": len(reference),
            "bin_edges": edges.tolist(),
            "bin_counts": reference_counts.tolist(),
            "raw_proportions": (reference_counts / len(reference)).tolist(),
            "smoothed_proportions": reference_smoothed.tolist(),
            "small_sample": len(reference) < min_rows,
        },
        "windows": windows,
    }
