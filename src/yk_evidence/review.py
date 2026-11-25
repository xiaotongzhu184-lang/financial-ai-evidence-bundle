"""Append-only local review notes bound to a particular evidence manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path


DECISIONS = {"accept_demo_evidence", "request_changes", "reject_evidence"}
MAX_REVIEW_FORM_BYTES = 32_768
_WRITE_LOCK = threading.Lock()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _manifest_run_id(bundle: Path) -> str:
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest.get("run_id") if isinstance(manifest, dict) else None
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("bundle manifest has no safe run_id for the review journal")
    return run_id


def review_path(bundle: Path) -> Path:
    """Return the journal path from the immutable run identity, not a folder name."""
    run_id = _manifest_run_id(bundle)
    return bundle.parent / "reviews" / (run_id + ".jsonl")


def read_reviews(bundle: Path) -> list[dict]:
    """Validate each local journal link; never treat it as a digital signature."""
    path = review_path(bundle)
    if not path.exists():
        return []
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("review journal may not be a symbolic link")
    manifest = bundle / "bundle_manifest.json"
    manifest_hash = _sha(manifest.read_bytes())
    records: list[dict] = []
    previous = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"review journal invalid at line {number}")
        claimed = record.pop("record_sha256", None)
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        if (claimed != _sha(canonical)
                or record.get("previous_record_sha256") != previous
                or record.get("bundle_manifest_sha256") != manifest_hash
                or record.get("decision") not in DECISIONS):
            raise ValueError(f"review journal invalid at line {number}")
        record["record_sha256"] = claimed
        records.append(record)
        previous = claimed
    return records


def append_review(bundle: Path, reviewer: str, decision: str, rationale: str) -> dict:
    """Record a demo evidence decision; this cannot grant production release."""
    if not re.fullmatch(r"[^\x00-\x1f\x7f]{2,80}", reviewer.strip()):
        raise ValueError("reviewer must contain 2-80 printable characters")
    if decision not in DECISIONS:
        raise ValueError("invalid review decision")
    if not 10 <= len(rationale.strip()) <= 2000 or any(ord(char) < 32 and char not in "\n\t" for char in rationale):
        raise ValueError("rationale must contain 10-2000 printable characters")
    # The caller verifies the bundle immediately before append. Recheck the
    # journal so a corrupt or stale chain cannot silently gain another entry.
    with _WRITE_LOCK:
        previous = read_reviews(bundle)
        run_id = _manifest_run_id(bundle)
        record = {
            "schema_version": "1.0",
            "bundle_run_id": run_id,
            "bundle_manifest_sha256": _sha((bundle / "bundle_manifest.json").read_bytes()),
            "reviewed_utc": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer.strip(),
            "decision": decision,
            "rationale": rationale.strip(),
            "previous_record_sha256": previous[-1]["record_sha256"] if previous else None,
            "scope": "Local review of demonstration evidence only; no production or public release authority.",
        }
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        record["record_sha256"] = _sha(canonical)
        path = review_path(bundle)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ValueError("review journal directory may not be a symbolic link")
        # A single append plus fsync keeps ordinary interruptions from losing
        # an acknowledged decision; this journal is not a digital signature.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
            if os.write(fd, encoded) != len(encoded):
                raise OSError("short review journal write")
            os.fsync(fd)
        finally:
            os.close(fd)
        return record
