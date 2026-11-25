"""Lineage, input-drift and demonstration stress-gate evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "VERSION"
if not VERSION_FILE.is_file():
    VERSION_FILE = Path(__file__).resolve().parent / "assets/VERSION"
AMOUNT_BINS = (250.0, 500.0, 750.0, 1000.0)


def _sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _proportions(values: list[float]) -> list[float]:
    counts = [0] * (len(AMOUNT_BINS) + 1)
    for value in values:
        index = next((i for i, bound in enumerate(AMOUNT_BINS) if value < bound), len(AMOUNT_BINS))
        counts[index] += 1
    return [count / len(values) for count in counts]


def _psi(reference: list[float], current: list[float]) -> float:
    if not reference or not current:
        raise ValueError("PSI requires two nonempty samples")
    old = _proportions(reference)
    new = _proportions(current)
    epsilon = 1e-6
    return sum((b - a) * math.log((b + epsilon) / (a + epsilon)) for a, b in zip(old, new))


def evaluate_governance(
    benchmark: dict, rows: list[dict], psi_limit: float = 0.25, ring_share_limit: float = 0.15,
    focus_merchant: str = "ring-m",
) -> dict:
    if not rows or not 0 <= psi_limit or not 0 <= ring_share_limit <= 1:
        raise ValueError("nonempty rows and nonnegative gate limits required")
    cut = int(len(rows) * 0.75)
    train = rows[:cut]
    holdout = rows[cut:]
    train_amount = [float(x["amount"]) for x in train]
    test_amount = [float(x["amount"]) for x in holdout]
    baseline_psi = _psi(train_amount, test_amount)
    amount_scale_psi = _psi(train_amount, [value * 1.2 for value in test_amount])
    ring_rows = [row for row in holdout if row["merchant_id"] == focus_merchant]
    burst_share = (len(ring_rows) * 2) / (len(holdout) + len(ring_rows))
    scenarios = [
        {"scenario_id": "amount_scale_1_2_v1", "input_change": {"holdout_amount_multiplier": 1.2},
         "observed_amount_psi": amount_scale_psi, "limit": psi_limit,
         "status": "pass" if amount_scale_psi <= psi_limit else "fail"},
        {"scenario_id": "ring_burst_x2_v1" if focus_merchant == "ring-m" else "merchant_burst_x2_v1",
         "input_change": {"duplicate_focus_merchant_transactions": 1, "focus_merchant": focus_merchant},
         "observed_ring_share": burst_share, "limit": ring_share_limit,
         "status": "incomplete" if not ring_rows else "pass" if burst_share <= ring_share_limit else "fail"},
    ]
    models = benchmark.get("models", {})
    complete = bool(models) and all(model.get("metrics", {}).get("pr_auc") is not None for model in models.values())
    failures = ["Amount PSI exceeded limit" if scenarios[0]["status"] == "fail" else None,
                "Focus merchant share exceeded limit" if scenarios[1]["status"] == "fail" else None]
    reasons = [reason for reason in failures if reason]
    if not ring_rows:
        reasons.append("No shared focus merchant in holdout; link stress is unavailable")
    if not complete:
        reasons.append("Benchmark metrics missing or undefined")
    status = "incomplete" if not complete or not ring_rows else "fail" if failures[0] or failures[1] else "pass"
    return {
        "classification": benchmark.get("classification", "unknown"), "project_version": VERSION_FILE.read_text().strip(),
        "lineage": {"transactions_sha256": _sha(rows), "benchmark_sha256": _sha(benchmark),
                    "training_ids": benchmark.get("split", {}).get("train_ids", []),
                    "holdout_ids": benchmark.get("split", {}).get("test_ids", [])},
        "drift": {"amount_psi": baseline_psi, "bins_upper_exclusive": list(AMOUNT_BINS),
                  "reference": "chronological training transactions", "current": "held-out transactions"},
        "scenarios": scenarios,
        "gate": {"status": status, "reasons": reasons, "type": "synthetic_data_quality_demo",
                 "limitations": "Thresholds are demonstration settings; this data-quality gate does not use model rescores. See the separate stress_replay artifact for frozen-model predictions. This is not production model release approval."},
    }
