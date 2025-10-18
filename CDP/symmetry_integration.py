"""Minimal utilities for plotting Pareto frontiers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

try:  # pragma: no cover - optional dependency for plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def plot_pareto_history(
    history: Sequence[Tuple[float, float, float]],
    output_path: Path,
) -> Path:
    """Create a Pareto frontier plot from scalarised α evaluations."""

    if plt is None:  # pragma: no cover - plotting is optional
        raise RuntimeError(
            "matplotlib is required for plotting the Pareto frontier. "
            "Install it via 'pip install matplotlib'."
        )
    if not history:
        raise ValueError("No history data provided to plot the Pareto frontier.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    penalties = [entry[2] for entry in history]
    dispersions = [entry[1] if math.isfinite(entry[1]) else 0.0 for entry in history]
    labels = [f"α={entry[0]:.2f}" for entry in history]

    plt.figure(figsize=(8, 5))
    plt.plot(dispersions, penalties, marker="o", linestyle="-", color="#1f77b4")
    plt.scatter(
        dispersions[0],
        penalties[0],
        color="#d62728",
        marker="s",
        s=120,
        label="Initial solution",
        zorder=3,
    )
    if len(history) > 1:
        plt.scatter(
            dispersions[1:],
            penalties[1:],
            color="#2ca02c",
            marker="o",
            s=80,
            label="Explored frontier",
            zorder=3,
        )
    for label, x_val, y_val in zip(labels, dispersions, penalties):
        plt.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize="small",
        )
    plt.xlabel("CDP objective (dispersion)")
    plt.ylabel("Symmetry penalty")
    plt.title("CDP-Symmetry Pareto frontier")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path.resolve()
