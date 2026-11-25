"""Render publication-oriented figures from one verified synthetic bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from yk_evidence.pipeline import verify


INK = "#183247"
TEAL = "#087e82"
ORANGE = "#c46b22"
BLUE = "#4565a2"
PURPLE = "#74539a"
LABELS = {"rule": "Amount rule", "tabular_logistic": "Tabular logistic", "graph_feature_logistic": "Graph-history logistic", "graph_feature_random_forest": "Graph-history random forest"}
COLORS = {"rule": ORANGE, "tabular_logistic": BLUE, "graph_feature_logistic": TEAL, "graph_feature_random_forest": PURPLE}


def _finish(fig, ax, path: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    fig.subplots_adjust(bottom=.22)
    fig.text(0.08, 0.025, "Generated 400-row synthetic fixture; not real-world fraud performance.", fontsize=8, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_figures(bundle: Path, output: Path) -> list[str]:
    errors = verify(bundle)
    if errors:
        raise ValueError("bundle verification failed: " + "; ".join(errors))
    benchmark = json.loads((bundle / "benchmark.json").read_text())
    validation = json.loads((bundle / "validation.json").read_text())
    replay = json.loads((bundle / "stress_replay.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, model in benchmark["models"].items():
        labels = [row["label"] for row in model["predictions"]]
        scores = [row["score"] for row in model["predictions"]]
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.step(recall, precision, where="post", label=f"{LABELS[name]} (AP {model['metrics']['pr_auc']:.3f})", color=COLORS[name], linewidth=2)
    prevalence = sum(row["label"] for row in benchmark["models"]["rule"]["predictions"]) / len(benchmark["models"]["rule"]["predictions"])
    ax.axhline(prevalence, color=INK, linestyle="--", linewidth=1, label=f"Holdout prevalence ({prevalence:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Held-out precision–recall on the synthetic fixture")
    ax.legend(frameon=False, fontsize=9)
    _finish(fig, ax, output / "precision_recall.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    names = list(benchmark["models"])
    positions = range(len(names))
    for index, name in enumerate(names):
        values = [run["models"][name]["pr_auc"] for run in validation["stability_runs"]]
        ax.scatter([index] * len(values), values, color=COLORS[name], s=42, alpha=.75, zorder=3)
        ax.scatter([index], [sum(values) / len(values)], color=INK, marker="_", s=280, linewidths=3, zorder=4)
    ax.set_xticks(list(positions), [LABELS[name] for name in names])
    ax.set(ylabel="Average precision", ylim=(0, 1), title="Five-seed variation within one synthetic generator")
    ax.grid(axis="y", color="#dbe4e9", linewidth=.7)
    _finish(fig, ax, output / "stability.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scenario_names = ["Baseline", "Amount ×1.2", "Ring links hidden"]
    values = [replay["baseline"]["metrics"]["pr_auc"], replay["scenarios"]["amount_scale_1_2"]["metrics"]["pr_auc"], replay["scenarios"]["ring_link_suppression"]["metrics"]["pr_auc"]]
    bars = ax.bar(scenario_names, values, color=[INK, BLUE, ORANGE], width=.62)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(ylabel="Average precision", ylim=(0, 1), title="Frozen-model scenario replay")
    ax.grid(axis="y", color="#dbe4e9", linewidth=.7)
    ax.set_axisbelow(True)
    _finish(fig, ax, output / "stress_replay.png")
    return ["precision_recall.png", "stability.png", "stress_replay.png"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print("\n".join(render_figures(arguments.bundle, arguments.output)))
