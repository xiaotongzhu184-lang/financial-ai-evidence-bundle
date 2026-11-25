"""Rescore declared synthetic scenarios with one frozen, fitted graph-feature model."""

from __future__ import annotations

import hashlib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark import _metrics
from .risk_data import graph_history_features


def _features(rows: list[dict]) -> list[list[float]]:
    history = graph_history_features(rows)
    return [
        [float(row["amount"]), float(row["hour"])] + prior
        for row, prior in zip(rows, history)
    ]


def replay_graph_stress(rows: list[dict], benchmark: dict, seed: int, focus_merchant: str = "ring-m") -> dict:
    split = benchmark["split"]
    cut = len(split["train_ids"])
    if benchmark["seed"] != seed or [row["transaction_id"] for row in rows[:cut]] != split["train_ids"] or [row["transaction_id"] for row in rows[cut:]] != split["test_ids"]:
        raise ValueError("reference benchmark seed or chronological split does not match rows")
    labels = [int(row["label"]) for row in rows]
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    original_features = _features(rows)
    model.fit(original_features[:cut], labels[:cut])
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    parameters = {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
    }
    model_hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    threshold = benchmark["models"]["graph_feature_logistic"]["train_fpr_threshold"]

    def score(changed: list[dict]) -> dict:
        predictions = [
            {"transaction_id": row["transaction_id"], "label": labels[index], "score": float(probability)}
            for index, (row, probability) in enumerate(
                zip(changed[cut:], model.predict_proba(_features(changed)[cut:])[:, 1]), start=cut
            )
        ]
        return {"predictions": predictions,
                "metrics": _metrics(labels[cut:], [item["score"] for item in predictions], threshold)}

    baseline = score(rows)
    baseline_ap = baseline["metrics"]["pr_auc"]
    scenarios = {}
    link_scenario = "ring_link_suppression" if focus_merchant == "ring-m" else "merchant_link_suppression"
    for scenario_id in ("amount_scale_1_2", link_scenario):
        changed = [dict(row) for row in rows]
        count = 0
        for index in range(cut, len(changed)):
            row = changed[index]
            if scenario_id == "amount_scale_1_2":
                row["amount"] = round(float(row["amount"]) * 1.2, 2)
            elif row["merchant_id"] == focus_merchant:
                row["merchant_id"] = f"missing-link-{row['transaction_id']}"
            else:
                continue
            count += 1
        result = score(changed)
        result["changed_rows"] = count
        result["model_sha256"] = model_hash
        result["maximum_allowed_pr_auc_drop"] = 0.10
        result["observed_pr_auc_drop"] = baseline_ap - result["metrics"]["pr_auc"]
        result["status"] = ("incomplete" if scenario_id == link_scenario and count == 0 else
                            "pass" if result["observed_pr_auc_drop"] <= 0.10 else "fail")
        scenarios[scenario_id] = result
    return {
        "classification": benchmark["classification"], "model_id": "graph_feature_logistic",
        "model_sha256": model_hash, "threshold_source": "original training negatives",
        "focus_merchant": focus_merchant,
        "baseline": baseline, "scenarios": scenarios,
        "limitations": "Only a frozen synthetic graph-history logistic model is rescored. The 0.10 AP-drop threshold is a demonstration choice, not a validated production limit.",
    }
