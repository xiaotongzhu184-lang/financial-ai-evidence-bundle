"""Grounded finance memo from a frozen SEC Company Facts extract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_ACCESSION = "0000950170-25-100235"
EXPECTED_CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"
EXPECTED_PERIODS = ("2024-06-30", "2025-06-30")


def load_snapshot(path: Path) -> dict:
    """Load an immutable local snapshot and attach its own file digest."""
    raw = path.read_bytes()
    data = json.loads(raw)
    data["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def answer_revenue_question(snapshot: dict) -> dict:
    """Derive the two-year revenue difference only from cited filing facts."""
    if snapshot.get("accession") != EXPECTED_ACCESSION:
        raise ValueError("unexpected accession")
    if snapshot.get("concept") != EXPECTED_CONCEPT:
        raise ValueError("unexpected revenue concept")
    if snapshot.get("classification") != "public_sec_filing":
        raise ValueError("unexpected source classification")
    if not snapshot.get("raw_source_sha256") or len(snapshot["raw_source_sha256"]) != 64:
        raise ValueError("missing raw source hash")
    facts = snapshot.get("facts", [])
    if len(facts) != 2 or tuple(x.get("end") for x in facts) != EXPECTED_PERIODS:
        raise ValueError("expected 2024 and 2025 annual periods")
    if any(x.get("accn") != EXPECTED_ACCESSION or x.get("form") != "10-K" for x in facts):
        raise ValueError("fact accession or form mismatch")
    if any(x.get("unit") != "USD" for x in facts):
        raise ValueError("fact unit must be USD")
    if any(not isinstance(x.get("val"), int) or x["val"] < 0 for x in facts):
        raise ValueError("invalid revenue value")
    old, new = facts
    increase = new["val"] - old["val"]
    citations = [
        {
            "source_id": f"SEC:{EXPECTED_ACCESSION}:us-gaap:{EXPECTED_CONCEPT}:{fact['end']}",
            "period_end": fact["end"], "value_usd": fact["val"],
            "source_url": snapshot["source_url"], "filing_url": snapshot["filing_url"],
            "accession": EXPECTED_ACCESSION, "source_sha256": snapshot["raw_source_sha256"],
        }
        for fact in facts
    ]
    return {
        "status": "supported", "question": "How did Microsoft's annual revenue change from FY2024 to FY2025?",
        "answer": f"Microsoft reported revenue of ${old['val']:,} for FY2024 and ${new['val']:,} for FY2025, an increase of ${increase:,}.",
        "increase_usd": increase, "accession": EXPECTED_ACCESSION,
        "snapshot_sha256": snapshot.get("snapshot_sha256"), "citations": citations,
        "method": "Deterministic arithmetic over two XBRL facts from the same 2025 Form 10-K accession; no LLM generation.",
    }
