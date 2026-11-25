"""Recalculate claims in a v0.3 bundle from the saved, lower-level evidence.

This is an internal consistency audit. It cannot authenticate a source that an
attacker has replaced together with every related file and the manifest.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from .benchmark import _metrics
from .finance import answer_revenue_question, load_snapshot
from .governance import _sha


METRICS = ("pr_auc", "f1_at_0_5", "recall_at_fpr_0_05", "observed_fpr")


def _equal(actual: object, expected: object) -> bool:
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def _metrics_match(actual: dict, expected: dict) -> bool:
    return all(_equal(actual.get(name), expected.get(name)) for name in METRICS)


def audit_bundle(bundle: Path) -> list[str]:
    """Return semantic inconsistencies in a verified v0.3 run, without refitting."""
    errors: list[str] = []
    try:
        snapshot = load_snapshot(bundle / "financial_snapshot.json")
        memo = json.loads((bundle / "finance_memo.json").read_text(encoding="utf-8"))
        if memo != answer_revenue_question(snapshot):
            errors.append("finance claim mismatch: memo vs source facts")

        transactions = json.loads((bundle / "synthetic_transactions.json").read_text(encoding="utf-8"))
        rows = transactions["rows"]
        benchmark = json.loads((bundle / "benchmark.json").read_text(encoding="utf-8"))
        validation = json.loads((bundle / "validation.json").read_text(encoding="utf-8"))
        governance = json.loads((bundle / "governance.json").read_text(encoding="utf-8"))
        replay = json.loads((bundle / "stress_replay.json").read_text(encoding="utf-8"))
        record = json.loads((bundle / "run_record.json").read_text(encoding="utf-8"))
        metrics_file = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
        retrieval = json.loads((bundle / "retrieval_evidence.json").read_text(encoding="utf-8"))
        gate = json.loads((bundle / "release_gate.json").read_text(encoding="utf-8"))

        if (transactions.get("classification") not in ("synthetic_benchmark", "external_synthetic_benchmark")
                or benchmark.get("classification") != transactions.get("classification")
                or len(rows) != benchmark.get("sample_count")
                or transactions.get("seed") != benchmark.get("seed") != record.get("seed")):
            errors.append("input mismatch: transaction classification, count or seed")
        if transactions.get("classification") == "external_synthetic_benchmark":
            from .input_data import load_synthetic_csv, profile_import
            imported_rows, metadata = load_synthetic_csv(bundle / "transaction_source.csv",
                                                         bundle / "transaction_source_metadata.json")
            provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
            profile = json.loads((bundle / "import_profile.json").read_text(encoding="utf-8"))
            if (rows != imported_rows or profile != profile_import(imported_rows, metadata) or
                    provenance.get("transactions", {}).get("source_metadata") != metadata or
                    provenance.get("transactions", {}).get("source_csv_sha256") != hashlib.sha256((bundle / "transaction_source.csv").read_bytes()).hexdigest()):
                errors.append("imported transaction mismatch: raw CSV, normalized rows or provenance")
        ids = [row["transaction_id"] for row in rows]
        cut = int(len(rows) * 0.75)
        if (len(set(ids)) != len(ids) or
                [row["sequence"] for row in rows] != list(range(len(rows))) or
                benchmark["split"].get("train_ids") != ids[:cut] or
                benchmark["split"].get("test_ids") != ids[cut:]):
            errors.append("split mismatch: chronological source rows vs benchmark")
        if (governance["lineage"].get("transactions_sha256") != _sha(rows) or
                governance["lineage"].get("benchmark_sha256") != _sha(benchmark) or
                governance["lineage"].get("training_ids") != ids[:cut] or
                governance["lineage"].get("holdout_ids") != ids[cut:]):
            errors.append("lineage mismatch: governance vs source rows or benchmark")
        if (retrieval.get("answer") != memo.get("answer") or
                retrieval.get("citations") != memo.get("citations") or
                retrieval.get("source_snapshot_sha256") != snapshot["snapshot_sha256"] or
                retrieval.get("formal_rag") is not False):
            errors.append("retrieval evidence mismatch: memo or source snapshot")

        truth = {row["transaction_id"]: int(row["label"]) for row in rows[cut:]}
        if set(metrics_file.get("models", {})) != set(benchmark["models"]):
            errors.append("metrics mismatch: model set")
        for name, model in benchmark["models"].items():
            predictions = model["predictions"]
            prediction_ids = [item["transaction_id"] for item in predictions]
            if prediction_ids != ids[cut:] or any(item["label"] != truth[item["transaction_id"]] for item in predictions):
                errors.append(f"prediction mismatch: {name} IDs or labels")
                continue
            scores = [item["score"] for item in predictions]
            if any(not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
                errors.append(f"prediction mismatch: {name} score range")
                continue
            if any(item["decision_at_0_5"] != int(item["score"] >= 0.5) for item in predictions):
                errors.append(f"prediction mismatch: {name} decisions")
            calculated = _metrics([truth[item_id] for item_id in prediction_ids], scores, model["train_fpr_threshold"])
            if not _metrics_match(model["metrics"], calculated) or not _metrics_match(metrics_file.get("models", {}).get(name, {}), calculated):
                errors.append(f"metrics mismatch: {name} saved vs recomputed")
            confusion = {
                "true_positives": sum(item["label"] == 1 and item["decision_at_0_5"] == 1 for item in predictions),
                "false_positives": sum(item["label"] == 0 and item["decision_at_0_5"] == 1 for item in predictions),
                "true_negatives": sum(item["label"] == 0 and item["decision_at_0_5"] == 0 for item in predictions),
                "false_negatives": sum(item["label"] == 1 and item["decision_at_0_5"] == 0 for item in predictions),
            }
            if validation.get("error_analysis", {}).get(name) != confusion:
                errors.append(f"confusion mismatch: {name} saved vs predictions")

        graph = benchmark["models"]["graph_feature_logistic"]
        threshold = graph["train_fpr_threshold"]
        model_hash = replay["model_sha256"]
        baseline = replay["baseline"]
        if baseline["predictions"] != [{key: item[key] for key in ("transaction_id", "label", "score")} for item in graph["predictions"]]:
            errors.append("stress baseline mismatch: graph model predictions")
        for scenario_id, item in replay["scenarios"].items():
            predictions = item["predictions"]
            if ([p["transaction_id"] for p in predictions] != ids[cut:] or
                    any(p["label"] != truth[p["transaction_id"]] for p in predictions) or
                    item["model_sha256"] != model_hash):
                errors.append(f"stress evidence mismatch: {scenario_id}")
                continue
            calculated = _metrics([p["label"] for p in predictions], [p["score"] for p in predictions], threshold)
            drop = graph["metrics"]["pr_auc"] - calculated["pr_auc"]
            if (not _metrics_match(item["metrics"], calculated) or
                    not _equal(item["observed_pr_auc_drop"], drop) or
                    item["status"] != ("incomplete" if scenario_id in ("merchant_link_suppression", "ring_link_suppression") and item["changed_rows"] == 0 else
                                       "pass" if drop <= item["maximum_allowed_pr_auc_drop"] else "fail")):
                errors.append(f"stress metrics mismatch: {scenario_id}")
        if not _metrics_match(baseline["metrics"], graph["metrics"]):
            errors.append("stress baseline metrics mismatch")
        graph_path = bundle / "graphsage_evidence.json"
        if graph_path.exists():
            from sklearn.metrics import average_precision_score, f1_score

            graph_evidence = json.loads(graph_path.read_text(encoding="utf-8"))
            predictions = graph_evidence["predictions"]
            labels = [truth[item_id] for item_id in ids[cut:]]
            scores = [item["score"] for item in predictions]
            source_manifest_hash = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
            if (graph_evidence.get("input_bundle_run_id") != record["run_id"]
                    or graph_evidence.get("input_manifest_sha256") != source_manifest_hash
                    or graph_evidence.get("split", {}).get("train_ids") != ids[:cut]
                    or graph_evidence.get("split", {}).get("test_ids") != ids[cut:]
                    or [item["transaction_id"] for item in predictions] != ids[cut:]
                    or [item["label"] for item in predictions] != labels
                    or any(not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores)):
                errors.append("GraphSAGE evidence mismatch: source, split or scores")
            else:
                measured_ap = float(average_precision_score(labels, scores))
                measured_f1 = float(f1_score(labels, [score >= 0.5 for score in scores], zero_division=0))
                if (not _equal(graph_evidence["metrics"].get("pr_auc"), measured_ap)
                        or not _equal(graph_evidence["metrics"].get("f1_at_0_5"), measured_f1)
                        or not _equal(graph_evidence["metrics"].get("prevalence"), sum(labels) / len(labels))
                        or any(item["decision_at_0_5"] != int(item["score"] >= 0.5) for item in predictions)):
                    errors.append("GraphSAGE metrics mismatch: saved vs recomputed")
        expected_reasons = ["synthetic_benchmark", "formal_rag_not_implemented"]
        if governance["gate"]["status"] != "pass":
            expected_reasons.append("data_quality_gate_" + governance["gate"]["status"])
        if gate.get("status") != "blocked" or gate.get("reasons") != expected_reasons:
            errors.append("release gate mismatch: reasons or status")
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        errors.append(f"invalid semantic evidence: {exc}")
    return errors
