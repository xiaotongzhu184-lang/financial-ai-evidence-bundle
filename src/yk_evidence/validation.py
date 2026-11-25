"""Synthetic-only stability and error analysis from saved benchmark outputs."""

from __future__ import annotations

from statistics import mean, pstdev

from .benchmark import run_benchmark
from .risk_data import generate_transactions


def evaluate_stability_and_errors(reference: dict, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("stability seeds must be nonempty and unique")
    count = reference["sample_count"]
    runs = []
    for seed in seeds:
        benchmark = reference if seed == reference["seed"] else run_benchmark(
            generate_transactions(seed=seed, count=count), seed=seed
        )
        runs.append({
            "seed": seed,
            "models": {
                name: {key: model["metrics"][key] for key in ("pr_auc", "f1_at_0_5", "recall_at_fpr_0_05", "observed_fpr")}
                for name, model in benchmark["models"].items()
            },
        })
    aggregate = {}
    for name in reference["models"]:
        aggregate[name] = {}
        for metric in ("pr_auc", "f1_at_0_5", "recall_at_fpr_0_05", "observed_fpr"):
            values = [item["models"][name][metric] for item in runs]
            aggregate[name][metric] = {
                "mean": mean(values) if all(value is not None for value in values) else None,
                "population_sd": pstdev(values) if all(value is not None for value in values) else None,
            }
    errors = {}
    for name, model in reference["models"].items():
        rows = model["predictions"]
        errors[name] = {
            "true_positives": sum(row["label"] == 1 and row["decision_at_0_5"] == 1 for row in rows),
            "false_positives": sum(row["label"] == 0 and row["decision_at_0_5"] == 1 for row in rows),
            "true_negatives": sum(row["label"] == 0 and row["decision_at_0_5"] == 0 for row in rows),
            "false_negatives": sum(row["label"] == 1 and row["decision_at_0_5"] == 0 for row in rows),
        }
    return {
        "classification": "synthetic_benchmark",
        "external_validity": "synthetic_only",
        "stability_runs": runs,
        "aggregate": aggregate,
        "error_analysis": errors,
        "limitations": "Seeds perturb generated amounts and benign identities, not the injected fraud rule; variation is not a confidence interval for real-world performance.",
    }


def evaluate_imported_errors(reference: dict) -> dict:
    """Report holdout errors without pretending a third-party CSV has generated seeds."""
    errors = {}
    for name, model in reference["models"].items():
        predictions = model["predictions"]
        errors[name] = {
            "true_positives": sum(item["label"] == 1 and item["decision_at_0_5"] == 1 for item in predictions),
            "false_positives": sum(item["label"] == 0 and item["decision_at_0_5"] == 1 for item in predictions),
            "true_negatives": sum(item["label"] == 0 and item["decision_at_0_5"] == 0 for item in predictions),
            "false_negatives": sum(item["label"] == 1 and item["decision_at_0_5"] == 0 for item in predictions),
        }
    return {
        "classification": reference["classification"], "external_validity": "not_established",
        "stability_runs": [], "aggregate": {}, "error_analysis": errors,
        "limitations": "No repeated-seed stability estimate for this imported file. The operator supplied its source and rights declaration; the software cannot independently verify those claims or infer real-world validity.",
    }
