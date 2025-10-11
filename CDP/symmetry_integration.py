"""Utilities to combine CDP heuristics with the symmetry-aware toolkit."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Instance import Instance
from Solution import Solution
from cdp_symmetry import (
    CandidateSolution,
    EpsilonConstraintSolver,
    Node,
    ProblemInstance,
    evaluate_subset,
)

try:  # pragma: no cover - optional dependency for plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


@dataclass(slots=True)
class SymmetryAnalysis:
    """Container gathering the different symmetry evaluations for a solution."""

    base_solution: CandidateSolution
    epsilon_front: List[CandidateSolution]


def _build_nodes(instance: Instance) -> List[Node]:
    colours = instance.colours if instance.colours else ["default" for _ in range(instance.node_count)]
    return [
        Node(
            node_id=str(index),
            capacity=instance.capacities[index],
            color=colours[index],
            coordinates=(float(index), 0.0),
        )
        for index in range(instance.node_count)
    ]


def _build_distances(instance: Instance) -> dict[Tuple[str, str], float]:
    distances: dict[Tuple[str, str], float] = {}
    for i in range(instance.node_count):
        for j in range(instance.node_count):
            if i == j:
                continue
            value = instance.distances[i][j]
            if value:
                distances[(str(i), str(j))] = value
    return distances


def build_problem_instance(instance: Instance) -> ProblemInstance:
    """Translate a :class:`Instance` into the symmetry-aware representation."""

    return ProblemInstance(
        nodes=_build_nodes(instance),
        demand=instance.min_capacity,
        lambda_penalty=instance.lambda_penalty,
        distances=_build_distances(instance),
        gamma_override=instance.gamma_override,
    )


def evaluate_solution_with_symmetry(instance: Instance, solution: Solution) -> CandidateSolution:
    """Return the symmetry-aware evaluation of a CDP solution."""

    problem_instance = build_problem_instance(instance)
    subset = tuple(str(vertex) for vertex in solution.selected_vertices)
    candidate = evaluate_subset(problem_instance, subset)
    penalty, breakdown = instance.symmetry_penalty(
        solution.selected_vertices, return_breakdown=True
    )
    solution.symmetry_penalty = penalty
    solution.symmetry_breakdown = breakdown
    return candidate


def candidate_to_solution(instance: Instance, candidate: CandidateSolution) -> Solution:
    """Convert a symmetry candidate into the native :class:`Solution` object."""

    new_solution = Solution(instance)
    new_solution.selected_vertices = [int(node_id) for node_id in candidate.selected_nodes]
    new_solution.capacity = sum(instance.capacities[vertex] for vertex in new_solution.selected_vertices)
    new_solution.reevaluate()
    return new_solution


def generate_epsilon_front(
    instance: Instance,
    solution: Solution,
    *,
    steps: int = 5,
) -> List[CandidateSolution]:
    """Enumerate epsilon-constrained solutions favouring higher symmetry.

    The routine starts from the symmetry penalty yielded by the provided
    ``solution`` (typically built without considering symmetry in its
    objective) and iteratively tightens the epsilon constraint so that each
    subsequent candidate enjoys the same or a better symmetry level (i.e. a
    lower penalty). Repeated solutions are discarded so the returned list
    traces a Pareto frontier ordered by the progressively stricter symmetry
    requirement.
    """

    base_candidate = evaluate_solution_with_symmetry(instance, solution)
    if steps <= 0:
        raise ValueError("steps must be strictly positive.")

    solver = EpsilonConstraintSolver(build_problem_instance(instance))
    front: List[CandidateSolution] = []
    seen: set[Tuple[str, ...]] = {base_candidate.selected_nodes}

    penalty = max(base_candidate.symmetry_penalty, 0.0)
    if math.isclose(penalty, 0.0, abs_tol=1e-12):
        targets = [0.0] * steps
    else:
        step = penalty / steps
        targets = [max(penalty - step * (index + 1), 0.0) for index in range(steps)]

    for epsilon in targets:
        try:
            candidate = solver.solve_max_dispersion(epsilon=epsilon)
        except ValueError:
            continue
        if candidate.selected_nodes in seen:
            continue
        seen.add(candidate.selected_nodes)
        front.append(candidate)
    return front


def analyse_solution(instance: Instance, solution: Solution, *, steps: int = 5) -> SymmetryAnalysis:
    """Produce a detailed symmetry report for the provided solution."""

    base = evaluate_solution_with_symmetry(instance, solution)
    front = generate_epsilon_front(instance, solution, steps=steps)
    return SymmetryAnalysis(base_solution=base, epsilon_front=front)


def pareto_points_to_rows(candidates: Iterable[CandidateSolution]) -> List[Tuple[float, float]]:
    """Return dispersion/penalty pairs for the provided candidates."""

    rows: List[Tuple[float, float]] = []
    for candidate in candidates:
        dispersion = (
            candidate.dispersion if math.isfinite(candidate.dispersion) else 0.0
        )
        rows.append((dispersion, candidate.symmetry_penalty))
    return rows


def plot_pareto_front(
    base_candidate: CandidateSolution,
    frontier: Iterable[CandidateSolution],
    output_path: Path,
) -> Path:
    """Create a scatter plot displaying the Pareto frontier.

    Parameters
    ----------
    base_candidate:
        Solution generated without symmetry pressure. It is highlighted as the
        starting point in the plot.
    frontier:
        Iterable describing the progressive improvements in symmetry obtained
        through tightened epsilon constraints.
    output_path:
        Destination for the generated image.
    """

    if plt is None:  # pragma: no cover - plotting is optional
        raise RuntimeError(
            "matplotlib is required for plotting the Pareto frontier. "
            "Install it via 'pip install matplotlib'."
        )

    rows = pareto_points_to_rows([base_candidate, *frontier])
    if not rows:
        raise ValueError("No candidate data available to plot the Pareto frontier.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    dispersions = [row[0] for row in rows]
    penalties = [row[1] for row in rows]
    plt.plot(dispersions, penalties, marker="o", linestyle="-", color="#1f77b4")
    plt.scatter(
        dispersions[0],
        penalties[0],
        color="#d62728",
        marker="s",
        s=120,
        label="Sin simetría",
        zorder=3,
    )
    if len(rows) > 1:
        plt.scatter(
            dispersions[1:],
            penalties[1:],
            color="#2ca02c",
            marker="o",
            s=80,
            label="Restricciones ε",
            zorder=3,
        )
    plt.xlabel("Dispersión")
    plt.ylabel("Penalización de simetría")
    plt.title("Frontera de Pareto CDP-Simetría")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path.resolve()

