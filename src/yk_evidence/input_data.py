"""Strict adapter for a local, rights-declared synthetic transaction CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


REQUIRED_COLUMNS = {"transaction_id", "timestamp_utc", "customer_id", "merchant_id", "amount", "label"}
REQUIRED_METADATA = {"source_name", "source_url", "dataset_version", "license_id", "label_definition",
                     "csv_sha256", "classification", "redistribution_permission"}
MAX_ROWS = 200_000
MAX_BYTES = 150_000_000


def load_synthetic_csv(csv_path: Path, metadata_path: Path) -> tuple[list[dict], dict]:
    """Return normalized rows and provenance; reject ambiguous time/rights data."""
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError("transactions CSV and metadata JSON must be existing files")
    if csv_path.stat().st_size > MAX_BYTES:
        raise ValueError(f"transaction CSV exceeds the {MAX_BYTES}-byte local limit")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or REQUIRED_METADATA - metadata.keys():
        raise ValueError("transaction metadata is missing required source, license or hash fields")
    if metadata["classification"] not in ("synthetic_public", "synthetic_private"):
        raise ValueError("only rights-declared synthetic transaction inputs are supported")
    if metadata["redistribution_permission"] not in ("permitted", "private_local_only"):
        raise ValueError("redistribution_permission must be permitted or private_local_only")
    if any(not isinstance(metadata[key], str) or not metadata[key].strip() for key in REQUIRED_METADATA):
        raise ValueError("all transaction metadata fields must be nonempty strings")
    with csv_path.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if metadata["csv_sha256"].lower() != actual_hash:
        raise ValueError("transaction CSV hash does not match metadata")
    rows: list[dict] = []
    seen: set[str] = set()
    previous_time: datetime | None = None
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or REQUIRED_COLUMNS - set(reader.fieldnames):
            raise ValueError("transaction CSV needs transaction_id,timestamp_utc,customer_id,merchant_id,amount,label")
        for line_number, raw in enumerate(reader, start=2):
            if len(rows) >= MAX_ROWS:
                raise ValueError(f"transaction CSV exceeds the {MAX_ROWS}-row local limit")
            if None in raw or any(raw.get(key) is None for key in REQUIRED_COLUMNS):
                raise ValueError(f"malformed transaction row at line {line_number}")
            transaction_id = raw["transaction_id"].strip()
            customer = raw["customer_id"].strip()
            merchant = raw["merchant_id"].strip()
            if (not transaction_id or len(transaction_id) > 128 or transaction_id in seen
                    or not customer or not merchant or len(customer) > 128 or len(merchant) > 128):
                raise ValueError(f"empty, duplicate or oversized identifier at line {line_number}")
            seen.add(transaction_id)
            try:
                instant = datetime.fromisoformat(raw["timestamp_utc"].replace("Z", "+00:00"))
                if instant.tzinfo is None:
                    raise ValueError("timezone missing")
                instant = instant.astimezone(timezone.utc)
                amount = float(raw["amount"])
                label = int(raw["label"])
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"invalid time, amount or label at line {line_number}: {exc}") from exc
            if previous_time is not None and instant < previous_time:
                raise ValueError(f"transaction CSV is not in chronological order at line {line_number}")
            if not math.isfinite(amount) or amount <= 0 or label not in (0, 1) or raw["label"].strip() not in ("0", "1"):
                raise ValueError(f"invalid positive amount or binary label at line {line_number}")
            rows.append({"transaction_id": transaction_id, "sequence": len(rows), "customer_id": customer,
                         "merchant_id": merchant, "amount": amount, "hour": instant.hour, "label": label,
                         "timestamp_utc": instant.isoformat()})
            previous_time = instant
    if len(rows) < 80:
        raise ValueError("transaction CSV needs at least 80 rows")
    cut = int(len(rows) * .75)
    if rows[cut - 1]["timestamp_utc"] == rows[cut]["timestamp_utc"]:
        raise ValueError("train/test cutoff splits equal-timestamp transactions")
    if {row["label"] for row in rows[:cut]} != {0, 1} or {row["label"] for row in rows[cut:]} != {0, 1}:
        raise ValueError("train and test periods must each contain both label classes")
    return rows, metadata


def profile_import(rows: list[dict], metadata: dict) -> dict:
    """Compact, reproducible profile for the decision to use an imported file."""
    amounts = [row["amount"] for row in rows]
    cut = int(len(rows) * .75)
    return {
        "source_name": metadata["source_name"], "classification": metadata["classification"],
        "license_id_operator_declared": metadata["license_id"],
        "row_count": len(rows), "unique_transaction_ids": len({row["transaction_id"] for row in rows}),
        "unique_customers": len({row["customer_id"] for row in rows}),
        "unique_merchants": len({row["merchant_id"] for row in rows}),
        "first_utc": rows[0]["timestamp_utc"], "last_utc": rows[-1]["timestamp_utc"],
        "positive_count": sum(row["label"] for row in rows),
        "train_positive_count": sum(row["label"] for row in rows[:cut]),
        "test_positive_count": sum(row["label"] for row in rows[cut:]),
        "amount_min": min(amounts), "amount_median": float(median(amounts)), "amount_max": max(amounts),
        "grain": "one timestamped customer-to-merchant transaction per unique transaction_id",
        "rights_limit": "Source and license are declared by the operator; not independently authenticated by the software.",
    }
