"""Translate frozen held-out risk scores into review-workload scenarios.

The scenarios are explicitly hypothetical. They do not calibrate scores or
estimate real-world fraud prevalence, cost, or release suitability.
"""

from __future__ import annotations

import math


DEFAULT_SCENARIOS = (
    {"scenario_id": "prevalence_1pct", "assumed_prevalence": 0.01,
     "transactions_per_period": 1000, "review_cost_units": 1.0,
     "miss_cost_units": 20.0, "review_capacity_per_period": 100},
    {"scenario_id": "prevalence_5pct", "assumed_prevalence": 0.05,
     "transactions_per_period": 1000, "review_cost_units": 1.0,
     "miss_cost_units": 20.0, "review_capacity_per_period": 100},
    {"scenario_id": "prevalence_10pct", "assumed_prevalence": 0.10,
     "transactions_per_period": 1000, "review_cost_units": 1.0,
     "miss_cost_units": 20.0, "review_capacity_per_period": 100},
)


def _number(value: object, name: str, minimum: float = 0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _wilson(successes: int, total: int) -> list[float] | None:
    """Two-sided 95% Wilson score interval for a binomial rate."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _validated_policies(policies: tuple[dict, ...] | list[dict]) -> list[dict]:
    if not policies:
        raise ValueError("at least one policy scenario is required")
    validated = []
    seen = set()
    for policy in policies:
        expected = {"scenario_id", "assumed_prevalence", "transactions_per_period",
                    "review_cost_units", "miss_cost_units", "review_capacity_per_period"}
        if not isinstance(policy, dict) or set(policy) != expected:
            raise ValueError("policy fields are incomplete or unknown")
        scenario_id = policy["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen:
            raise ValueError("scenario_id must be unique and nonempty")
        seen.add(scenario_id)
        prevalence = _number(policy["assumed_prevalence"], "assumed_prevalence", maximum=1)
        volume = policy["transactions_per_period"]
        capacity = policy["review_capacity_per_period"]
        if (isinstance(volume, bool) or not isinstance(volume, int) or volume <= 0
                or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError("transaction volume and review capacity must be nonnegative integers")
        validated.append({"scenario_id": scenario_id, "assumed_prevalence": prevalence,
                          "transactions_per_period": volume,
                          "review_cost_units": _number(policy["review_cost_units"], "review_cost_units"),
                          "miss_cost_units": _number(policy["miss_cost_units"], "miss_cost_units"),
                          "review_capacity_per_period": capacity})
    return validated


def build_decision_impact(benchmark: dict, run_id: str,
                          policies: tuple[dict, ...] | list[dict] = DEFAULT_SCENARIOS) -> dict:
    """Recompute operational scenarios from saved scores and training thresholds."""
    scenarios = _validated_policies(policies)
    models = benchmark.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("benchmark has no models")
    output = {}
    for model_id, model in models.items():
        threshold = _number(model["train_fpr_threshold"], "train_fpr_threshold", maximum=1)
        predictions = model["predictions"]
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("held-out predictions are missing")
        tp = fp = tn = fn = 0
        for prediction in predictions:
            label = prediction["label"]
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError("held-out label must be binary")
            score = _number(prediction["score"], "held-out score", maximum=1)
            flagged = score > threshold
            if label == 1 and flagged:
                tp += 1
            elif label == 0 and flagged:
                fp += 1
            elif label == 0:
                tn += 1
            else:
                fn += 1
        positives, negatives = tp + fn, fp + tn
        available = bool(positives and negatives)
        tpr = tp / positives if available else None
        fpr = fp / negatives if available else None
        cases = []
        if available:
            for policy in scenarios:
                volume = policy["transactions_per_period"]
                prevalence = policy["assumed_prevalence"]
                expected_true = volume * prevalence * tpr
                expected_false = volume * (1 - prevalence) * fpr
                alerts = expected_true + expected_false
                missed = volume * prevalence * (1 - tpr)
                cost = alerts * policy["review_cost_units"] + missed * policy["miss_cost_units"]
                cases.append({"scenario_id": policy["scenario_id"],
                              "expected_alerts_per_period": alerts,
                              "expected_false_alerts_per_period": expected_false,
                              "expected_missed_positives_per_period": missed,
                              "illustrative_cost_units": cost,
                              "capacity_status": ("capacity_exceeded" if alerts > policy["review_capacity_per_period"]
                                                  else "within_assumed_capacity")})
        output[model_id] = {
            "threshold_from_training": threshold, "threshold_rule": "score > threshold",
            "heldout_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "heldout_count": len(predictions), "heldout_positives": positives,
            "heldout_prevalence": positives / len(predictions),
            "heldout_tpr": tpr, "heldout_tpr_wilson95": _wilson(tp, positives) if available else None,
            "heldout_fpr": fpr, "heldout_fpr_wilson95": _wilson(fp, negatives) if available else None,
            "availability": "available" if available else "single_class_holdout",
            "scenarios": cases,
        }
    return {"run_id": run_id, "source_classification": benchmark["classification"],
            "assumption_status": "illustrative_not_observed_operating_costs",
            "policy_scenarios": scenarios, "models": output,
            "limitations": ("Held-out rates come from this dataset only. Scenario prevalence, volume, costs and capacity "
                            "are stated assumptions; scores are not calibrated real-world probabilities. "
                            "Wilson intervals describe sampling uncertainty only and do not cover dataset shift.")}
