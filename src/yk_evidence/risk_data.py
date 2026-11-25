"""Original, deterministic synthetic transaction-network generator."""

from __future__ import annotations

import random


def generate_transactions(seed: int, count: int) -> list[dict]:
    if count < 80:
        raise ValueError("count must be at least 80 for a meaningful holdout")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        injected_ring = index % 19 == 0
        rows.append({
            "transaction_id": f"tx-{index:06d}",
            "sequence": index,
            "customer_id": f"ring-c{(index // 19) % 5}" if injected_ring else f"c{rng.randrange(200):03d}",
            "merchant_id": "ring-m" if injected_ring else f"m{rng.randrange(40):03d}",
            "amount": round(rng.uniform(200, 1100) if injected_ring else rng.uniform(10, 1100), 2),
            "hour": index % 24,
            "label": int(injected_ring),
        })
    return rows


def graph_history_features(rows: list[dict]) -> list[list[float]]:
    """Use only earlier transactions sharing a customer or merchant; never labels."""
    customer_count: dict[str, int] = {}
    merchant_count: dict[str, int] = {}
    merchant_amount: dict[str, float] = {}
    result = []
    for row in rows:
        customer = row["customer_id"]
        merchant = row["merchant_id"]
        prior_customer = customer_count.get(customer, 0)
        prior_merchant = merchant_count.get(merchant, 0)
        prior_mean = merchant_amount.get(merchant, 0.0) / prior_merchant if prior_merchant else 0.0
        result.append([float(prior_customer), float(prior_merchant), prior_mean])
        customer_count[customer] = prior_customer + 1
        merchant_count[merchant] = prior_merchant + 1
        merchant_amount[merchant] = merchant_amount.get(merchant, 0.0) + float(row["amount"])
    return result
