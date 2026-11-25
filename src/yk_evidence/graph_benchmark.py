"""Optional two-layer GraphSAGE edge classifier on the generated fixture.

Training target edges are removed from each mini-batch's message-passing context.
Test inference uses only training-period edges as context. This deliberately
strict context differs from the graph-history logistic model's sequential
access to earlier test-period unlabeled transactions.
"""

from __future__ import annotations

import hashlib
import math
import random

import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch_geometric.nn import SAGEConv


class EdgeSAGE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = SAGEConv(3, 16)
        self.second = SAGEConv(16, 16)
        self.head = nn.Sequential(nn.Linear(34, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, node_features, edge_index, source, target, edge_features):
        hidden = torch.relu(self.first(node_features, edge_index))
        hidden = torch.relu(self.second(hidden, edge_index))
        joined = torch.cat([hidden[source], hidden[target], edge_features], dim=1)
        return self.head(joined).squeeze(1)


def run_graph_benchmark(rows: list[dict], seed: int, epochs: int = 20,
                        classification: str = "synthetic_benchmark") -> dict:
    if len(rows) < 80 or epochs < 1:
        raise ValueError("at least 80 rows and one training epoch required")
    if len({row["transaction_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate transaction ID")
    if any(rows[i]["sequence"] >= rows[i + 1]["sequence"] for i in range(len(rows) - 1)):
        raise ValueError("transactions must be in sequence order")
    cut = int(len(rows) * .75)
    if set(int(row["label"]) for row in rows[:cut]) != {0, 1} or set(int(row["label"]) for row in rows[cut:]) != {0, 1}:
        raise ValueError("both train and holdout must contain both classes")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    names = sorted({"u:" + str(row["customer_id"]) for row in rows} | {"m:" + str(row["merchant_id"]) for row in rows})
    lookup = {name: index for index, name in enumerate(names)}
    source = torch.tensor([lookup["u:" + str(row["customer_id"])] for row in rows], dtype=torch.long)
    target = torch.tensor([lookup["m:" + str(row["merchant_id"])] for row in rows], dtype=torch.long)
    amount_mean = sum(float(row["amount"]) for row in rows[:cut]) / cut
    amount_sd = math.sqrt(sum((float(row["amount"]) - amount_mean) ** 2 for row in rows[:cut]) / cut) or 1.0
    edge_features = torch.tensor([[(float(row["amount"]) - amount_mean) / amount_sd, float(row["hour"]) / 23] for row in rows], dtype=torch.float32)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.float32)

    def context(indices: list[int]):
        degree = [0] * len(names)
        amount_sum = [0.0] * len(names)
        connections = []
        for index in indices:
            left, right = int(source[index]), int(target[index])
            connections.extend(((left, right), (right, left)))
            for node in (left, right):
                degree[node] += 1
                amount_sum[node] += float(rows[index]["amount"])
        features = torch.tensor([
            [1.0 if name.startswith("u:") else 0.0, math.log1p(degree[index]),
             (amount_sum[index] / degree[index] / 1000.0) if degree[index] else 0.0]
            for index, name in enumerate(names)
        ], dtype=torch.float32)
        edges = torch.tensor(connections, dtype=torch.long).T.contiguous() if connections else torch.empty((2, 0), dtype=torch.long)
        return features, edges

    model = EdgeSAGE()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0001)
    positive = int(labels[:cut].sum().item())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((cut - positive) / positive))
    train_ids = list(range(cut))
    trace = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train_ids)
        batch_losses = []
        for start in range(0, cut, 50):
            batch = train_ids[start:start + 50]
            withheld = set(batch)
            node_features, edge_index = context([index for index in range(cut) if index not in withheld])
            logits = model(node_features, edge_index, source[batch], target[batch], edge_features[batch])
            loss = loss_fn(logits, labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        trace.append(sum(batch_losses) / len(batch_losses))

    model.eval()
    with torch.no_grad():
        node_features, edge_index = context(list(range(cut)))
        scores = torch.sigmoid(model(node_features, edge_index, source[cut:], target[cut:], edge_features[cut:])).tolist()
    test_labels = [int(row["label"]) for row in rows[cut:]]
    predictions = [
        {"transaction_id": row["transaction_id"], "label": label, "score": float(score), "decision_at_0_5": int(score >= .5)}
        for row, label, score in zip(rows[cut:], test_labels, scores)
    ]
    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state.update(name.encode())
        state.update(tensor.detach().numpy().tobytes())
    return {
        "classification": classification, "model_family": "GraphSAGE", "seed": seed, "epochs": epochs,
        "model_sha256": state.hexdigest(), "training_edge_count": cut,
        "context_policy": "Train mini-batch target edges excluded from message passing; test embeddings use training-period edges only. No test edges or labels enter context.",
        "training_loss_by_epoch": trace,
        "split": {"train_ids": [row["transaction_id"] for row in rows[:cut]],
                  "test_ids": [row["transaction_id"] for row in rows[cut:]]},
        "metrics": {
            "pr_auc": float(average_precision_score(test_labels, scores)),
            "f1_at_0_5": float(f1_score(test_labels, [item["decision_at_0_5"] for item in predictions], zero_division=0)),
            "prevalence": sum(test_labels) / len(test_labels),
        },
        "predictions": predictions,
        "limitations": "Small generated network; fixed epochs, no held-out tuning or calibration. Graph-history logistic sees earlier unlabeled test rows while this GraphSAGE uses training-only context; scores are not a controlled architecture ranking.",
    }
