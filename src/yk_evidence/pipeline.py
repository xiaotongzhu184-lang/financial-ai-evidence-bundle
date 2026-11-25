"""Dependency-free, synthetic first vertical slice."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"


def asset(relative: str) -> Path:
    """Read checked-in fixtures from a checkout or an installed wheel."""
    source = ROOT / relative
    return source if source.is_file() else ASSETS / relative
REQUIRED = (
    "source_snapshot.txt", "transactions_snapshot.csv", "finance_memo.json",
    "risk_output.json", "run_record.json", "stress_test.json", "summary.md",
)
REQUIRED_INTEGRATED = (
    "financial_snapshot.json", "finance_memo.json", "synthetic_transactions.json",
    "benchmark.json", "validation.json", "stress_replay.json", "governance.json", "run_record.json",
    "data_card.md", "model_card.md", "summary.md",
)
STOPWORDS = {"what", "was", "were", "the", "how", "did", "and", "for", "with", "company", "synthetic"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def grounded_memo(source: str, question: str) -> dict:
    """Return exact source sentences with line citations, or abstain."""
    keywords = {w for w in re.findall(r"[a-zA-Z]+", question.lower()) if len(w) > 2 and w not in STOPWORDS}
    citations = []
    for line_no, line in enumerate(source.splitlines(), 1):
        if keywords.intersection(re.findall(r"[a-zA-Z]+", line.lower())):
            citations.append({"source_id": "synthetic_finance", "line": line_no, "quote": line})
    if not citations:
        return {"question": question, "status": "no_supported_answer", "answer": "No source sentence supports an answer.", "citations": []}
    return {
        "question": question,
        "status": "source_excerpt_only",
        "answer": " ".join(item["quote"] for item in citations),
        "citations": citations,
        "limitations": "Extractive keyword retrieval; no LLM, semantic retrieval, or financial inference.",
    }


def score_transactions(rows: list[dict], amount_multiplier: float = 1.0) -> list[dict]:
    findings = []
    for row in rows:
        amount = float(row["amount"]) * amount_multiplier
        related = sorted(other["transaction_id"] for other in rows if other["transaction_id"] != row["transaction_id"] and (other["merchant_id"] == row["merchant_id"] or other["customer_id"] == row["customer_id"]))
        score = 0.75 if amount >= 800 and related else 0.40 if amount >= 800 else 0.10
        findings.append({
            "transaction_id": row["transaction_id"], "score": score,
            "decision": "review" if score >= 0.70 else "routine",
            "related_entities": related,
            "explanation": f"Amount {amount:.2f} synthetic units; connected transactions: {', '.join(related) if related else 'none'}.",
            "model_id": "transparent_rule_baseline_v0",
        })
    return findings


def run(output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be 1-64 safe filename characters")
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / run_id
    bundle.mkdir(exist_ok=False)
    shutil.copyfile(asset("examples/synthetic_finance.txt"), bundle / "source_snapshot.txt")
    shutil.copyfile(asset("examples/synthetic_transactions.csv"), bundle / "transactions_snapshot.csv")
    source = (bundle / "source_snapshot.txt").read_text(encoding="utf-8")
    with (bundle / "transactions_snapshot.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    memo = grounded_memo(source, "What was revenue in each period?")
    memo["source_sha256"] = digest(bundle / "source_snapshot.txt")
    write_json(bundle / "finance_memo.json", memo)
    findings = score_transactions(rows)
    write_json(bundle / "risk_output.json", {
        "data_classification": "synthetic_demo", "model_id": "transparent_rule_baseline_v0",
        "findings": findings, "limitations": "Illustrative fixed rule scores; not trained, calibrated, or validated fraud probabilities.",
    })
    stressed = score_transactions(rows, amount_multiplier=1.2)
    baseline_count = sum(x["decision"] == "review" for x in findings)
    stressed_count = sum(x["decision"] == "review" for x in stressed)
    increase = stressed_count - baseline_count
    write_json(bundle / "stress_test.json", {
        "scenario_id": "synthetic_amount_x1_2_v0", "parameter": {"amount_multiplier": 1.2},
        "baseline_review_count": baseline_count, "stressed_review_count": stressed_count,
        "observed_increase": increase, "maximum_allowed_increase": 1,
        "status": "pass" if increase <= 1 else "fail",
        "limitations": "Demo-only rule-gate result, not a production release decision or statistical robustness test.",
    })
    write_json(bundle / "run_record.json", {
        "run_id": run_id, "project_version": asset("VERSION").read_text().strip(),
        "utc_time": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
        "source_classification": "synthetic_demo", "model_id": "transparent_rule_baseline_v0",
        "finance_source_sha256": digest(bundle / "source_snapshot.txt"),
        "transactions_sha256": digest(bundle / "transactions_snapshot.csv"),
        "code_revision": None, "code_revision_reason": "No release commit pinned for development slice.",
        "seed": None, "seed_reason": "Fixed input fixtures and deterministic rules use no random generator.",
    })
    (bundle / "summary.md").write_text(
        "# Synthetic evidence bundle summary\n\n"
        f"Run: `{run_id}`. Project version: `{asset('VERSION').read_text().strip()}`.\n\n"
        f"Finance: {len(memo['citations'])} exact source-line citations. "
        f"Risk: {baseline_count} of {len(findings)} transactions flagged for review by an illustrative rule. "
        f"Stress: {stressed_count} flagged after multiplying amounts by 1.2; demo gate "
        f"{'pass' if increase <= 1 else 'fail'}.\n\n"
        "All data and results are synthetic demonstrations. No model performance, regulatory compliance, "
        "production readiness or public-company conclusion is claimed.\n",
        encoding="utf-8",
    )
    write_json(bundle / "manifest.json", {
        "schema_version": "0.1", "project_version": asset("VERSION").read_text().strip(),
        "run_id": run_id, "source_classification": "synthetic_demo",
        "artifacts": {name: digest(bundle / name) for name in REQUIRED},
        "verification": "python -m yk_evidence verify BUNDLE_PATH",
    })
    return bundle


def integrated_run(output_root: Path, run_id: str, seed: int = 42,
                   transactions_csv: Path | None = None, transactions_metadata: Path | None = None) -> Path:
    """Produce one public-source plus synthetic-benchmark evidence bundle."""
    from .benchmark import run_benchmark
    from .finance import answer_revenue_question, load_snapshot
    from .governance import evaluate_governance
    from .risk_data import generate_transactions
    from .stress_replay import replay_graph_stress
    from .validation import evaluate_imported_errors, evaluate_stability_and_errors
    from .input_data import load_synthetic_csv, profile_import
    from collections import Counter

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be 1-64 safe filename characters")
    if (transactions_csv is None) != (transactions_metadata is None):
        raise ValueError("provide both transactions_csv and transactions_metadata")
    imported = transactions_csv is not None
    if imported:
        rows, input_metadata = load_synthetic_csv(transactions_csv, transactions_metadata)
        classification = "external_synthetic_benchmark"
        source_classification = "public_sec_filing_and_external_synthetic_benchmark"
        train_merchants = Counter(row["merchant_id"] for row in rows[:int(len(rows) * .75)])
        holdout_merchants = {row["merchant_id"] for row in rows[int(len(rows) * .75):]}
        available = [(count, merchant) for merchant, count in train_merchants.items() if merchant in holdout_merchants]
        focus_merchant = max(available)[1] if available else "__no_shared_merchant__"
    else:
        rows = generate_transactions(seed=seed, count=400)
        input_metadata = None
        classification = "synthetic_benchmark"
        source_classification = "public_sec_filing_and_synthetic_benchmark"
        focus_merchant = "ring-m"
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = output_root / run_id
    bundle.mkdir(exist_ok=False)
    shutil.copyfile(asset("data/financial/msft_2025_10k_revenue.json"), bundle / "financial_snapshot.json")
    for source, target in (("DATA_CARD.md", "data_card.md"), ("MODEL_CARD.md", "model_card.md")):
        shutil.copyfile(asset("docs/project/" + source), bundle / target)
    memo = answer_revenue_question(load_snapshot(bundle / "financial_snapshot.json"))
    write_json(bundle / "finance_memo.json", memo)
    if imported:
        shutil.copyfile(transactions_csv, bundle / "transaction_source.csv")
        shutil.copyfile(transactions_metadata, bundle / "transaction_source_metadata.json")
        write_json(bundle / "import_profile.json", profile_import(rows, input_metadata))
    write_json(bundle / "synthetic_transactions.json", {"classification": classification, "seed": seed, "rows": rows})
    benchmark = run_benchmark(rows, seed=seed, classification=classification)
    write_json(bundle / "benchmark.json", benchmark)
    validation = (evaluate_imported_errors(benchmark) if imported else
                  evaluate_stability_and_errors(benchmark, seeds=[seed - 2, seed - 1, seed, seed + 1, seed + 2]))
    write_json(bundle / "validation.json", validation)
    replay = replay_graph_stress(rows, benchmark, seed=seed, focus_merchant=focus_merchant)
    write_json(bundle / "stress_replay.json", replay)
    governance = evaluate_governance(benchmark, rows, focus_merchant=focus_merchant)
    write_json(bundle / "governance.json", governance)
    code_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for source in sorted(source_root.rglob("*.py")):
        code_hash.update(source.relative_to(source_root).as_posix().encode())
        code_hash.update(source.read_bytes())
    source_revision = "source-sha256:" + code_hash.hexdigest()
    git = shutil.which("git")
    if git:
        try:
            revision = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
            git_root = subprocess.run([git, "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True, check=False)
            tracked_changes = subprocess.run([git, "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=False)
            if (revision.returncode == 0 and git_root.returncode == 0 and tracked_changes.returncode == 0
                    and not tracked_changes.stdout.strip() and Path(git_root.stdout.strip()).resolve() == ROOT.resolve()):
                source_revision = revision.stdout.strip()
        except OSError:
            pass
    write_json(bundle / "run_record.json", {
        "run_id": run_id, "project_version": asset("VERSION").read_text().strip(),
        "utc_time": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
        "code_sha256": code_hash.hexdigest(), "code_revision": source_revision,
        "seed": seed, "source_classification": source_classification,
        "input_sha256": {"finance": digest(bundle / "financial_snapshot.json"),
                         "transactions": digest(bundle / "synthetic_transactions.json"),
                         **({"transaction_source": digest(bundle / "transaction_source.csv"),
                             "transaction_source_metadata": digest(bundle / "transaction_source_metadata.json")} if imported else {})},
        "result_sha256": {"benchmark": digest(bundle / "benchmark.json"),
                          "validation": digest(bundle / "validation.json"),
                          "stress_replay": digest(bundle / "stress_replay.json"),
                          "governance": digest(bundle / "governance.json")},
        "model_ids": list(benchmark["models"]),
    })
    metrics = benchmark["models"]
    summary = [
        "# Integrated evidence bundle summary", "",
        f"Run `{run_id}`; project version `{asset('VERSION').read_text().strip()}`.", "",
        "## Public financial source", "",
        memo["answer"],
        f"Both cited facts come from SEC Form 10-K accession `{memo['accession']}`. See `finance_memo.json` and `financial_snapshot.json` for exact trace and hashes.", "",
        "## Synthetic transaction benchmark", "",
        "| Model | PR-AUC | F1 at 0.5 | Recall at train-selected FPR threshold | Held-out FPR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in metrics.items():
        m = result["metrics"]
        summary.append(f"| {name} | {m['pr_auc']:.4f} | {m['f1_at_0_5']:.4f} | {m['recall_at_fpr_0_05']:.4f} | {m['observed_fpr']:.4f} |")
    summary.extend(["", ("Imported synthetic input: no repeated-seed estimate; held-out confusion counts are saved in `validation.json`." if imported else "Five-seed synthetic stability and held-out confusion counts are saved in `validation.json`.") + " Two scenarios are rescored with a frozen fitted model in `stress_replay.json`.", "",
                    f"Synthetic data-quality stress gate: **{governance['gate']['status']}**. See `governance.json` for scenarios and thresholds.", "",
                    "These benchmark numbers are measured only on synthetic transactions. They do not estimate real fraud performance. The finance memo is deterministic arithmetic over SEC facts; it is not an LLM or evaluated RAG system.", ""])
    (bundle / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    write_json(bundle / "manifest.json", {
        "schema_version": "0.2", "project_version": asset("VERSION").read_text().strip(),
        "run_id": run_id, "source_classification": source_classification,
        "artifacts": {name: digest(bundle / name) for name in REQUIRED_INTEGRATED + (("transaction_source.csv", "transaction_source_metadata.json", "import_profile.json") if imported else ())},
        "verification": "yk-evidence verify BUNDLE_PATH",
        "limitations": "Manifest checks internal integrity only; it is not signed or independently authenticated.",
    })
    return bundle


def contract_run(output_root: Path, run_id: str, seed: int = 42, graph_epochs: int = 0,
                 transactions_csv: Path | None = None, transactions_metadata: Path | None = None) -> Path:
    """Run the integrated demo, optionally adding GraphSAGE in the same bundle."""
    from .contract import emit_contract

    if graph_epochs < 0:
        raise ValueError("graph_epochs must be nonnegative")
    if graph_epochs:
        import importlib.util
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            raise ValueError("GraphSAGE requires optional torch and torch-geometric packages")
    bundle = integrated_run(output_root, run_id, seed=seed,
                            transactions_csv=transactions_csv, transactions_metadata=transactions_metadata)
    if graph_epochs:
        import torch
        import torch_geometric
        from .graph_benchmark import run_graph_benchmark

        source = json.loads((bundle / "synthetic_transactions.json").read_text(encoding="utf-8"))
        graph = run_graph_benchmark(source["rows"], seed=seed, epochs=graph_epochs,
                                    classification=source["classification"])
        graph["input_bundle_run_id"] = run_id
        graph["input_manifest_sha256"] = digest(bundle / "manifest.json")
        graph["torch_version"] = torch.__version__
        graph["torch_geometric_version"] = torch_geometric.__version__
        write_json(bundle / "graphsage_evidence.json", graph)
    emit_contract(bundle, digest)
    return bundle


def verify(bundle: Path) -> list[str]:
    errors = []
    manifest_name = "bundle_manifest.json" if (bundle / "bundle_manifest.json").exists() else "manifest.json"
    try:
        manifest = json.loads((bundle / manifest_name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"missing or invalid manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["invalid manifest: expected a JSON object"]
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return ["invalid artifact map"]
    schema = manifest.get("schema_version")
    if schema == "0.6-inference":
        from .inference import verify_inference
        return verify_inference(bundle)
    if schema == "0.7-project":
        from .project import verify_project
        return verify_project(bundle)
    if schema in ("0.5-algorithm", "0.6-algorithm"):
        from .algorithm import verify_algorithm
        return verify_algorithm(bundle)
    if schema == "0.4":
        from .research import verify_research
        return verify_research(bundle)
    if schema not in ("0.1", "0.2", "0.3"):
        return [f"unsupported schema: {schema}"]
    if schema == "0.3":
        from .contract import REQUIRED_CONTRACT, REQUIRED_DECISION_IMPACT, requires_decision_impact
        required = (REQUIRED_DECISION_IMPACT if requires_decision_impact(manifest.get("project_version", ""))
                    else REQUIRED_CONTRACT)
    else:
        required = REQUIRED_INTEGRATED if schema == "0.2" else REQUIRED
    for name in required:
        if name not in artifacts:
            errors.append(f"missing manifest entry: {name}")
    for name, expected in artifacts.items():
        if not isinstance(name, str) or not isinstance(expected, str) or Path(name).is_absolute() or len(Path(name).parts) != 1 or name in (".", ".."):
            errors.append(f"unsafe path or hash: {name}")
            continue
        target = bundle / name
        if not target.is_file() or target.is_symlink():
            errors.append(f"missing or unsafe artifact: {name}")
        elif digest(target) != expected:
            errors.append(f"hash mismatch: {name}")
    if schema == "0.3":
        actual = {path.name for path in bundle.iterdir() if path.is_file() and path.name != "bundle_manifest.json"}
        if actual != set(artifacts):
            errors.append("artifact inventory mismatch: unlisted or missing file")
    if schema == "0.3" and not errors:
        from .contract import verify_contract
        errors.extend(verify_contract(bundle, manifest))
        if not errors:
            from .review import read_reviews
            try:
                read_reviews(bundle)
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid local review journal: {exc}")
    if schema == "0.2" and not errors:
        try:
            record = json.loads((bundle / "run_record.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid run record: {exc}")
        else:
            if not isinstance(record, dict) or record.get("run_id") != manifest.get("run_id"):
                errors.append("run_id mismatch: manifest vs run_record.json")
            if isinstance(record, dict):
                if record.get("project_version") != manifest.get("project_version"):
                    errors.append("project_version mismatch: manifest vs run_record.json")
                if record.get("source_classification") != manifest.get("source_classification"):
                    errors.append("source_classification mismatch: manifest vs run_record.json")
                for field, names in (
                    ("input_sha256", {"finance": "financial_snapshot.json", "transactions": "synthetic_transactions.json"}),
                    ("result_sha256", {"benchmark": "benchmark.json", "validation": "validation.json",
                                       "stress_replay": "stress_replay.json", "governance": "governance.json"}),
                ):
                    recorded = record.get(field)
                    if not isinstance(recorded, dict):
                        errors.append(f"invalid {field} in run_record.json")
                        continue
                    for key, name in names.items():
                        if recorded.get(key) != artifacts.get(name):
                            errors.append(f"{'input' if field == 'input_sha256' else 'result'} hash mismatch: {name}")
    return errors
