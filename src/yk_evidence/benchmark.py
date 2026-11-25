"""Held-out synthetic risk benchmark using documented scikit-learn APIs."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .risk_data import graph_history_features


def _threshold_at_train_fpr(labels: list[int], scores: list[float], ceiling: float = 0.05) -> float:
    negative = sorted(score for label, score in zip(labels, scores) if label == 0)
    if not negative:
        return 1.0
    allowed = int(len(negative) * ceiling)
    return negative[-allowed - 1] if allowed < len(negative) else 0.0


def _metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return {"pr_auc": None, "f1_at_0_5": None, "recall_at_fpr_0_05": None,
                "observed_fpr": None, "undefined_reason": "holdout has only one class"}
    gate = [int(score > threshold) for score in scores]
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1_at_0_5": float(f1_score(labels, [int(score >= 0.5) for score in scores], zero_division=0)),
        "recall_at_fpr_0_05": sum(x and y for x, y in zip(gate, labels)) / positives,
        "observed_fpr": sum(x and not y for x, y in zip(gate, labels)) / negatives,
        "undefined_reason": None,
    }


def run_benchmark(rows: list[dict], seed: int, classification: str = "synthetic_benchmark") -> dict:
    if len(rows) < 80:
        raise ValueError("benchmark needs at least 80 rows")
    if len({row["transaction_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate transaction ID")
    if any(rows[i]["sequence"] >= rows[i + 1]["sequence"] for i in range(len(rows) - 1)):
        raise ValueError("transactions must be in sequence order")
    split = int(len(rows) * 0.75)
    labels = [int(row["label"]) for row in rows]
    if set(labels[:split]) != {0, 1}:
        raise ValueError("training split must contain both classes")
    graph = graph_history_features(rows)
    tabular = [[float(row["amount"]), float(row["hour"])] for row in rows]
    graph_input = [base + history for base, history in zip(tabular, graph)]
    models = {}

    def record(name: str, train_scores: list[float], test_scores: list[float], description: str) -> None:
        threshold = _threshold_at_train_fpr(labels[:split], train_scores)
        test_labels = labels[split:]
        models[name] = {
            "description": description, "train_fpr_threshold": threshold,
            "metrics": _metrics(test_labels, test_scores, threshold),
            "predictions": [
                {"transaction_id": row["transaction_id"], "label": label,
                 "score": float(score), "decision_at_0_5": int(score >= 0.5)}
                for row, label, score in zip(rows[split:], test_labels, test_scores)
            ],
        }

    rule_scores = [0.8 if row["amount"] >= 800 else 0.2 for row in rows]
    record("rule", rule_scores[:split], rule_scores[split:], "Fixed amount threshold; no fitting")
    for name, features, description in (
        ("tabular_logistic", tabular, "Amount and hour only"),
        ("graph_feature_logistic", graph_input, "Amount/hour plus prior customer and merchant network features"),
    ):
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
        model.fit(features[:split], labels[:split])
        train_scores = model.predict_proba(features[:split])[:, 1].tolist()
        test_scores = model.predict_proba(features[split:])[:, 1].tolist()
        record(name, train_scores, test_scores, description)
    forest = RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=seed, n_jobs=1,
    )
    forest.fit(graph_input[:split], labels[:split])
    record(
        "graph_feature_random_forest",
        forest.predict_proba(graph_input[:split])[:, 1].tolist(),
        forest.predict_proba(graph_input[split:])[:, 1].tolist(),
        "Random forest on amount/hour and prior network features; chronological split",
    )
    return {
        "classification": classification, "seed": seed, "sample_count": len(rows),
        "split": {"strategy": "chronological_75_25", "train_ids": [x["transaction_id"] for x in rows[:split]],
                  "test_ids": [x["transaction_id"] for x in rows[split:]]},
        "metric_definitions": {
            "pr_auc": "average_precision_score on held-out labels and raw scores",
            "f1_at_0_5": "binary F1 at fixed score threshold 0.5",
            "recall_at_fpr_0_05": "held-out recall using threshold selected from training negatives for FPR ceiling 0.05; actual held-out FPR is separately reported",
        },
        "models": models,
    }
