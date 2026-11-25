"""Export the integrated run to a versioned review contract.

The export preserves the raw 0.2 artifacts. Every added value is copied from,
or computed from, those artifacts; unavailable capabilities are marked blocked.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


REQUIRED_CONTRACT = (
    "run_metadata.json", "input_manifest.json", "provenance.json",
    "lineage.json", "retrieval_evidence.json", "risk_scores.csv",
    "model_metadata.json", "metrics.json", "drift_report.json",
    "stress_test_results.csv", "release_gate.json", "review_report.html",
)
REQUIRED_DECISION_IMPACT = REQUIRED_CONTRACT + ("decision_impact.json",)


def requires_decision_impact(version: str) -> bool:
    """Retain verification support for immutable rc1/rc2 evidence bundles."""
    return version not in ("0.3.0rc1", "0.3.0rc2")


def _read(bundle: Path, name: str) -> dict:
    return json.loads((bundle / name).read_text(encoding="utf-8"))


def _write(bundle: Path, name: str, value: dict) -> None:
    (bundle / name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def emit_contract(bundle: Path, digest) -> None:
    """Write the 0.3 contract from a completed integrated run."""
    record = _read(bundle, "run_record.json")
    snapshot = _read(bundle, "financial_snapshot.json")
    memo = _read(bundle, "finance_memo.json")
    transactions = _read(bundle, "synthetic_transactions.json")
    benchmark = _read(bundle, "benchmark.json")
    validation = _read(bundle, "validation.json")
    governance = _read(bundle, "governance.json")
    replay = _read(bundle, "stress_replay.json")
    imported_metadata = (_read(bundle, "transaction_source_metadata.json")
                         if (bundle / "transaction_source_metadata.json").exists() else None)

    _write(bundle, "run_metadata.json", {
        "schema_version": "0.3", "run_id": record["run_id"],
        "project_version": record["project_version"], "utc_time": record["utc_time"],
        "code_revision": record["code_revision"], "code_sha256": record["code_sha256"],
        "python_version": record["python_version"], "seed": record["seed"],
        "source_classification": record["source_classification"],
    })
    input_rows = [
            {"path": "financial_snapshot.json", "sha256": digest(bundle / "financial_snapshot.json"),
             "classification": snapshot["classification"], "accession": snapshot["accession"]},
            {"path": "synthetic_transactions.json", "sha256": digest(bundle / "synthetic_transactions.json"),
             "classification": transactions["classification"], "seed": transactions["seed"],
             "row_count": len(transactions["rows"])},
        ]
    if imported_metadata is not None:
        input_rows.extend([
            {"path": "transaction_source.csv", "sha256": digest(bundle / "transaction_source.csv"),
             "classification": imported_metadata["classification"], "row_count": len(transactions["rows"])},
            {"path": "transaction_source_metadata.json", "sha256": digest(bundle / "transaction_source_metadata.json"),
             "classification": "operator_declared_provenance"},
        ])
    _write(bundle, "input_manifest.json", {"run_id": record["run_id"], "inputs": input_rows})
    _write(bundle, "provenance.json", {
        "run_id": record["run_id"],
        "finance": {"source_url": snapshot["source_url"], "filing_url": snapshot["filing_url"],
                    "accession": snapshot["accession"], "retrieved_utc": snapshot["retrieved_utc"],
                    "raw_source_sha256": snapshot["raw_source_sha256"],
                    "snapshot_sha256": digest(bundle / "financial_snapshot.json")},
        "transactions": ({"classification": "external_synthetic_benchmark",
                          "generator": "yk_evidence.input_data.load_synthetic_csv", "seed": transactions["seed"],
                          "source_metadata": imported_metadata,
                          "source_csv_sha256": digest(bundle / "transaction_source.csv"),
                          "rights_status": "operator declaration; not independently authenticated"}
                         if imported_metadata is not None else
                         {"classification": "synthetic_benchmark", "generator": "yk_evidence.risk_data.generate_transactions",
                          "seed": transactions["seed"], "redistribution": "project-generated synthetic example"}),
        "limitations": "The included finance file is a two-fact extract, not the full source filing."
    })
    _write(bundle, "lineage.json", {
        "run_id": record["run_id"], "input_sha256": record["input_sha256"],
        "result_sha256": record["result_sha256"], "split": benchmark["split"],
        "governance_lineage": governance["lineage"],
    })
    _write(bundle, "retrieval_evidence.json", {
        "run_id": record["run_id"], "question": memo["question"], "answer": memo["answer"],
        "status": memo["status"], "source_snapshot_sha256": memo["snapshot_sha256"],
        "citations": memo["citations"], "method": memo["method"],
        "retrieval_mode": "structured_fact_lookup", "formal_rag": False,
        "limitations": "No full-document chunk index, retrieval ranking log or generative model was used.",
    })
    with (bundle / "risk_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run_id", "model_id", "transaction_id", "label", "score", "decision_at_0_5"))
        writer.writeheader()
        for model_id, result in benchmark["models"].items():
            for prediction in result["predictions"]:
                writer.writerow({"run_id": record["run_id"], "model_id": model_id, **prediction})
    _write(bundle, "model_metadata.json", {
        "run_id": record["run_id"], "data_classification": benchmark["classification"],
        "models": {name: {"description": value["description"], "train_fpr_threshold": value["train_fpr_threshold"],
                          "prediction_count": len(value["predictions"])}
                   for name, value in benchmark["models"].items()},
        "selected_stress_model": {"model_id": replay["model_id"], "model_sha256": replay["model_sha256"]},
    })
    _write(bundle, "metrics.json", {
        "run_id": record["run_id"], "data_classification": benchmark["classification"],
        "definitions": benchmark["metric_definitions"],
        "models": {name: value["metrics"] for name, value in benchmark["models"].items()},
        "stability_runs": validation["stability_runs"], "external_validity": validation["external_validity"],
    })
    _write(bundle, "drift_report.json", {
        "run_id": record["run_id"], "drift": governance["drift"],
        "scenarios": governance["scenarios"], "classification": governance["classification"],
    })
    with (bundle / "stress_test_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run_id", "kind", "scenario_id", "status", "observed", "limit", "measure", "model_id"))
        writer.writeheader()
        for item in governance["scenarios"]:
            measure = "amount_psi" if "observed_amount_psi" in item else "ring_share"
            writer.writerow({"run_id": record["run_id"], "kind": "data_quality", "scenario_id": item["scenario_id"],
                             "status": item["status"], "observed": item[f"observed_{measure}"],
                             "limit": item["limit"], "measure": measure, "model_id": ""})
        for scenario_id, item in replay["scenarios"].items():
            writer.writerow({"run_id": record["run_id"], "kind": "frozen_model", "scenario_id": scenario_id,
                             "status": item["status"], "observed": item["observed_pr_auc_drop"],
                             "limit": item["maximum_allowed_pr_auc_drop"], "measure": "pr_auc_drop",
                             "model_id": replay["model_id"]})
    reasons = ["synthetic_benchmark", "formal_rag_not_implemented"]
    if governance["gate"]["status"] != "pass":
        reasons.append("data_quality_gate_" + governance["gate"]["status"])
    _write(bundle, "release_gate.json", {
        "run_id": record["run_id"], "status": "blocked", "reasons": reasons,
        "data_quality_gate": governance["gate"],
        "scope": "Public research demo candidate only; never a production deployment approval.",
    })
    if requires_decision_impact(record["project_version"]):
        from .decision_impact import build_decision_impact
        _write(bundle, "decision_impact.json", build_decision_impact(benchmark, record["run_id"]))
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{value['metrics']['pr_auc']:.4f}</td><td>{len(value['predictions'])}</td></tr>"
        for name, value in benchmark["models"].items()
    )
    report = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Evidence review {html.escape(record['run_id'])}</title></head><body>"
        f"<h1>Evidence review: {html.escape(record['run_id'])}</h1>"
        "<p>Release gate: BLOCKED. Synthetic benchmark; structured SEC fact lookup only.</p>"
        f"<p>Finance finding: {html.escape(memo['answer'])}</p>"
        "<table><thead><tr><th>Model</th><th>Average precision</th><th>Held-out predictions</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>Inspect the JSON and CSV artifacts for citations, raw predictions, drift, stress results, thresholds and limitations.</p>"
        "</body></html>\n"
    )
    (bundle / "review_report.html").write_text(report, encoding="utf-8")
    _write(bundle, "bundle_manifest.json", {
        "schema_version": "0.3", "project_version": record["project_version"],
        "run_id": record["run_id"], "source_classification": record["source_classification"],
        "required_artifacts": list(REQUIRED_DECISION_IMPACT if requires_decision_impact(record["project_version"])
                                   else REQUIRED_CONTRACT),
        "artifacts": {path.name: digest(path) for path in sorted(bundle.iterdir()) if path.is_file() and path.name != "bundle_manifest.json"},
        "limitations": "Local hashes and cross-file checks are not an independent signature or timestamp.",
    })


def verify_contract(bundle: Path, manifest: dict) -> list[str]:
    """Check semantic relationships after all listed file hashes pass."""
    errors: list[str] = []
    try:
        from jsonschema import Draft202012Validator
        from .pipeline import asset

        schema = json.loads(asset("schemas/contract_0_3.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        for name, ref in schema["x-artifacts"].items():
            if name == "decision_impact.json" and not requires_decision_impact(manifest["project_version"]):
                continue
            value = _read(bundle, name)
            validator = Draft202012Validator(
                {"$defs": schema["$defs"], "$ref": ref},
                format_checker=Draft202012Validator.FORMAT_CHECKER,
            )
            for issue in validator.iter_errors(value):
                field = ".".join(map(str, issue.absolute_path)) or "root"
                errors.append(f"schema violation: {name}:{field}: {issue.message}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid contract schema: {exc}")
    if errors:
        return errors
    try:
        metadata = _read(bundle, "run_metadata.json")
        record = _read(bundle, "run_record.json")
    except (OSError, ValueError) as exc:
        return [f"invalid contract metadata: {exc}"]
    if not isinstance(metadata, dict) or not isinstance(record, dict):
        return ["invalid contract metadata object"]
    for field in ("run_id", "project_version", "source_classification"):
        if metadata.get(field) != manifest.get(field) or record.get(field) != manifest.get(field):
            errors.append(f"{field} mismatch: bundle_manifest, run_metadata, run_record")
    if metadata.get("code_sha256") != record.get("code_sha256"):
        errors.append("code_sha256 mismatch: run_metadata vs run_record")
    if metadata.get("seed") != record.get("seed"):
        errors.append("seed mismatch: run_metadata vs run_record")
    expected_required = list(REQUIRED_DECISION_IMPACT if requires_decision_impact(manifest["project_version"])
                             else REQUIRED_CONTRACT)
    if manifest.get("required_artifacts") != expected_required:
        errors.append("required artifact list mismatch")
    if requires_decision_impact(manifest["project_version"]):
        try:
            from .decision_impact import build_decision_impact
            expected_impact = build_decision_impact(_read(bundle, "benchmark.json"), manifest["run_id"])
            if _read(bundle, "decision_impact.json") != expected_impact:
                errors.append("decision impact mismatch: saved scenarios vs held-out predictions and fixed policy")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid decision impact evidence: {exc}")
    try:
        inputs = _read(bundle, "input_manifest.json")
        snapshot = _read(bundle, "financial_snapshot.json")
        transactions = _read(bundle, "synthetic_transactions.json")
        expected_inputs = [
            {"path": "financial_snapshot.json", "sha256": manifest["artifacts"]["financial_snapshot.json"],
             "classification": snapshot["classification"], "accession": snapshot["accession"]},
            {"path": "synthetic_transactions.json", "sha256": manifest["artifacts"]["synthetic_transactions.json"],
             "classification": transactions["classification"], "seed": transactions["seed"],
             "row_count": len(transactions["rows"])},
        ]
        if (bundle / "transaction_source_metadata.json").exists():
            imported_metadata = _read(bundle, "transaction_source_metadata.json")
            expected_inputs.extend([
                {"path": "transaction_source.csv", "sha256": manifest["artifacts"]["transaction_source.csv"],
                 "classification": imported_metadata["classification"], "row_count": len(transactions["rows"])},
                {"path": "transaction_source_metadata.json", "sha256": manifest["artifacts"]["transaction_source_metadata.json"],
                 "classification": "operator_declared_provenance"},
            ])
        if inputs.get("run_id") != manifest.get("run_id") or inputs.get("inputs") != expected_inputs:
            errors.append("input manifest mismatch: raw inputs or bundle manifest")
        memo = _read(bundle, "finance_memo.json")
        retrieval = _read(bundle, "retrieval_evidence.json")
        expected_retrieval = {"question": memo["question"], "answer": memo["answer"],
                              "status": memo["status"], "source_snapshot_sha256": memo["snapshot_sha256"],
                              "citations": memo["citations"], "method": memo["method"]}
        if (retrieval.get("run_id") != manifest.get("run_id")
                or any(retrieval.get(key) != value for key, value in expected_retrieval.items())
                or retrieval.get("formal_rag") is not False
                or retrieval.get("retrieval_mode") != "structured_fact_lookup"
                or memo["snapshot_sha256"] != manifest["artifacts"]["financial_snapshot.json"]):
            errors.append("retrieval evidence mismatch: raw finance memo or source snapshot")
        validation = _read(bundle, "validation.json")
        metrics = _read(bundle, "metrics.json")
        model_metrics = {name: value["metrics"] for name, value in _read(bundle, "benchmark.json")["models"].items()}
        if (metrics.get("run_id") != manifest.get("run_id")
                or metrics.get("models") != model_metrics
                or metrics.get("stability_runs") != validation["stability_runs"]
                or metrics.get("external_validity") != validation["external_validity"]):
            errors.append("metrics mismatch: raw benchmark or validation")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(f"invalid contract cross-file evidence: {exc}")
    try:
        benchmark = _read(bundle, "benchmark.json")
        expected = {
            (model_id, str(item["transaction_id"])): item
            for model_id, result in benchmark["models"].items()
            for item in result["predictions"]
        }
        with (bundle / "risk_scores.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(expected):
            errors.append("risk score count mismatch: CSV vs benchmark predictions")
        seen = set()
        for row in rows:
            key = (row.get("model_id"), row.get("transaction_id"))
            item = expected.get(key)
            if key in seen or item is None or row.get("run_id") != manifest.get("run_id"):
                errors.append("risk score mismatch: unknown, duplicate or wrong-run row")
                break
            seen.add(key)
            score = float(row["score"])
            if (not math.isfinite(score) or score < 0 or score > 1
                    or score != float(item["score"])
                    or int(row["label"]) != item["label"]
                    or int(row["decision_at_0_5"]) != item["decision_at_0_5"]):
                errors.append(f"risk score mismatch: {key[0]}/{key[1]}")
                break
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid risk score evidence: {exc}")
    try:
        from sklearn.metrics import average_precision_score, f1_score

        benchmark = _read(bundle, "benchmark.json")
        model_metadata = _read(bundle, "model_metadata.json")
        transactions = _read(bundle, "synthetic_transactions.json")
        source_ids = [row["transaction_id"] for row in transactions["rows"]]
        cut = int(len(source_ids) * 0.75)
        if len(set(source_ids)) != len(source_ids):
            errors.append("source-order split mismatch: duplicate source transaction IDs")
        if (benchmark.get("sample_count") != len(source_ids)
                or benchmark["split"].get("strategy") != "chronological_75_25"
                or benchmark["split"].get("train_ids") != source_ids[:cut]
                or benchmark["split"].get("test_ids") != source_ids[cut:]):
            errors.append("source-order split mismatch: expected chronological 75/25 input rows")
        source_labels = {row["transaction_id"]: int(row["label"]) for row in transactions["rows"]}
        with (bundle / "risk_scores.csv").open(newline="", encoding="utf-8") as handle:
            score_rows = list(csv.DictReader(handle))
        if set(model_metadata["models"]) != set(benchmark["models"]):
            errors.append("prediction count mismatch: model sets differ")
        test_ids = benchmark["split"]["test_ids"]
        if len(set(test_ids)) != len(test_ids):
            errors.append("held-out transaction IDs mismatch: duplicate declared test IDs")
        if set(test_ids).intersection(benchmark["split"]["train_ids"]):
            errors.append("held-out transaction IDs mismatch: training/test overlap")
        for model_id, result in benchmark["models"].items():
            raw = [row for row in score_rows if row["model_id"] == model_id]
            if (len(raw) != len(result["predictions"])
                    or len(raw) != len(test_ids)
                    or model_metadata["models"][model_id]["prediction_count"] != len(raw)):
                errors.append(f"prediction count mismatch: {model_id}")
            if ([item["transaction_id"] for item in result["predictions"]] != test_ids
                    or [row["transaction_id"] for row in raw] != test_ids):
                errors.append(f"held-out transaction IDs mismatch: {model_id}")
            labels = [int(row["label"]) for row in raw]
            scores = [float(row["score"]) for row in raw]
            if any(source_labels.get(row["transaction_id"]) != int(row["label"]) for row in raw):
                errors.append(f"raw transaction label mismatch: {model_id}")
            threshold = result["train_fpr_threshold"]
            if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                    or not math.isfinite(threshold) or not 0 <= threshold <= 1):
                errors.append(f"invalid train-selected threshold: {model_id}")
                continue
            if model_metadata["models"][model_id]["train_fpr_threshold"] != threshold:
                errors.append(f"train-selected threshold mismatch: {model_id}")
            if 0 < sum(labels) < len(labels):
                if result["metrics"].get("undefined_reason") is not None:
                    errors.append(f"recomputed metric mismatch: {model_id}/undefined_reason")
                observed_ap = float(average_precision_score(labels, scores))
                reported_ap = result["metrics"]["pr_auc"]
                if not isinstance(reported_ap, (int, float)) or not math.isclose(
                    reported_ap, observed_ap, rel_tol=1e-12, abs_tol=1e-12
                ):
                    errors.append(f"recomputed metric mismatch: {model_id}/pr_auc")
                observed_f1 = float(f1_score(labels, [score >= 0.5 for score in scores], zero_division=0))
                reported_f1 = result["metrics"].get("f1_at_0_5")
                if not isinstance(reported_f1, (int, float)) or not math.isclose(
                    reported_f1, observed_f1, rel_tol=1e-12, abs_tol=1e-12
                ):
                    errors.append(f"recomputed metric mismatch: {model_id}/f1_at_0_5")
                chosen = [score > threshold for score in scores]
                positives = sum(labels)
                negatives = len(labels) - positives
                expected_rates = {
                    "recall_at_fpr_0_05": sum(pred and label for pred, label in zip(chosen, labels)) / positives,
                    "observed_fpr": sum(pred and not label for pred, label in zip(chosen, labels)) / negatives,
                }
                for metric, observed in expected_rates.items():
                    reported = result["metrics"].get(metric)
                    if not isinstance(reported, (int, float)) or not math.isclose(
                        reported, observed, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        errors.append(f"recomputed metric mismatch: {model_id}/{metric}")
            else:
                metric_names = ("pr_auc", "f1_at_0_5", "recall_at_fpr_0_05", "observed_fpr")
                if (any(result["metrics"].get(name) is not None for name in metric_names)
                        or result["metrics"].get("undefined_reason") != "holdout has only one class"):
                    errors.append(f"single-class metric mismatch: {model_id}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"invalid recomputed metric evidence: {exc}")
    try:
        gate = _read(bundle, "release_gate.json")
        retrieval = _read(bundle, "retrieval_evidence.json")
        transactions = _read(bundle, "synthetic_transactions.json")
        governance = _read(bundle, "governance.json")
        if (gate.get("run_id") != manifest.get("run_id")
                or gate.get("data_quality_gate") != governance.get("gate")):
            errors.append("release gate mismatch: run or governance decision")
        if (transactions.get("classification") == "synthetic_benchmark"
                or retrieval.get("formal_rag") is not True
                or governance.get("gate", {}).get("status") != "pass") and gate.get("status") != "blocked":
            errors.append("release gate mismatch: incomplete evidence cannot pass")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"invalid release gate evidence: {exc}")
    if not errors:
        from .semantic_audit import audit_bundle
        errors.extend(audit_bundle(bundle))
    return errors
