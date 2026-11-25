"""Typed DuckDB checks for a ULB-shaped source, before fitting any model."""
from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import duckdb

ULB_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def profile_ulb(path: Path, memory_mb: int = 128, threads: int = 2) -> dict:
    """Scan finite numeric values with a bounded DuckDB buffer budget.

    The budget controls DuckDB's buffer manager, not total process RSS. The
    downstream trainer separately materializes its finite feature matrix.
    """
    path = Path(path)
    if memory_mb < 32 or not 1 <= threads <= 8:
        raise ValueError("profile budget requires >=32 MB and 1-8 threads")
    if not path.is_file() or path.is_symlink():
        raise ValueError("source must be a regular nonsymlink CSV file")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        if next(csv.reader(handle), None) != ULB_COLUMNS:
            raise ValueError("expected the original ULB column names and order")
    start = time.perf_counter()
    columns = ", ".join(f"'{name}': 'DOUBLE'" for name in ULB_COLUMNS)
    finite = " OR ".join(f'"{name}" IS NULL OR NOT isfinite("{name}")' for name in ULB_COLUMNS)
    connection = duckdb.connect(config={"threads": str(threads), "memory_limit": f"{memory_mb}MB"})
    try:
        # Only ULB_COLUMNS, a module constant checked against the CSV header,
        # contributes identifiers to this statement. The file path remains a
        # bound DuckDB parameter.
        query = f'''SELECT count(*), sum("Class"), min("Time"), max("Time"),
            min("Amount"), max("Amount"), count(*) FILTER (WHERE "Amount" = 0),
            count(*) FILTER (WHERE {finite} OR "Time" < 0 OR "Amount" < 0
                              OR "Class" NOT IN (0,1))
            FROM read_csv(?, header=true, columns={{{columns}}}, auto_detect=false,
                          strict_mode=true, null_padding=false)'''  # nosec B608
        row = connection.execute(query, [str(path)]).fetchone()
    except duckdb.Error as exc:
        raise ValueError(f"typed CSV profile failed: {exc}") from exc
    finally:
        connection.close()
    if not row or row[0] == 0:
        raise ValueError("source CSV is empty")
    if row[7]:
        raise ValueError(f"source contains {row[7]} invalid/null/nonfinite rows")
    with path.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"engine": "duckdb", "engine_version": duckdb.__version__,
            "source_sha256": fingerprint, "source_bytes": path.stat().st_size,
            "row_count": int(row[0]), "positive_count": int(row[1]),
            "time_range_seconds": [float(row[2]), float(row[3])],
            "amount_range": [float(row[4]), float(row[5])], "zero_amount_count": int(row[6]),
            "invalid_count": 0, "scan_seconds": time.perf_counter() - start,
            "buffer_memory_mb": memory_mb, "threads": threads,
            "time_semantics": "seconds since the first source transaction; no invented calendar date",
            "limits": "DuckDB buffer budget is not total process RSS; training materializes arrays."}
