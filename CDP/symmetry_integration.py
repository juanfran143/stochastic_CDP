"""Minimal utilities for plotting Epsilon Pareto frontiers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

try:  # pragma: no cover - optional dependency for plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def plot_epsilon_frontier(
        # Historical Data Format: (epsilon, dispersion, colour_count)
        history: Sequence[Tuple[int, float, int]],
        output_path: Path,
) -> Path:
    """Create an Epsilon Pareto frontier plot from constraint evaluations."""

    if plt is None:  # pragma: no cover - plotting is optional
        raise RuntimeError(
            "matplotlib is required for plotting the Pareto frontier. "
            "Install it via 'pip install matplotlib'."
        )
    if not history:
        raise ValueError("No history data provided to plot the Epsilon frontier.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    # history: (epsilon, dispersion, colour_count)
    dispersions = [entry[1] if math.isfinite(entry[1]) else 0.0 for entry in history]
    colour_counts = [entry[2] for entry in history]
    labels = [f"ε={entry[0]}" for entry in history]

    # (dispersion, colour_count)
    unique_points = sorted(list(set(zip(dispersions, colour_counts))), key=lambda x: x[1])

    # Re-extract unique coordinates and labels for plotting
    plot_dispersions = [p[0] for p in unique_points]
    plot_counts = [p[1] for p in unique_points]

    plt.figure(figsize=(8, 5))

    # Drawing Connecting Lines (Number of Colors vs. Dispersion)
    plt.plot(plot_counts, plot_dispersions, marker="o", linestyle="-", color="#1f77b4", zorder=2)

    for epsilon, x_val, y_val in zip(labels, dispersions, colour_counts):
        plt.scatter(
            y_val,  # (Colour Count)
            x_val,  # (Dispersion)
            color="#2ca02c",
            marker="o",
            s=80,
            zorder=3
        )

        plt.annotate(
            epsilon,
            (y_val, x_val),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize="small",
            zorder=4
        )

    if dispersions:
        plt.scatter(
            colour_counts[0],
            dispersions[0],
            color="#d62728",
            marker="s",
            s=120,
            label=f"Start: ε={labels[0].split('=')[-1]}",  # 标记开始点
            zorder=5,
        )

    plt.xlabel("Colour Count ($f_2$)")
    plt.ylabel("CDP objective (Dispersion, $f_1$)")
    plt.title(r"CDP Epsilon-Constraint Frontier (Maximize $f_1$ s.t. $f_2 \le \epsilon$)")
    plt.grid(True, linestyle="--", alpha=0.4)

    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path.resolve()