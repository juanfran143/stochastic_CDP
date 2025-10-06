"""cdp_symmetry
=================

Bi-objective Capacitated Dispersion Problem with Symmetry Penalty.

This module implements a self-contained toolkit for solving a bi-objective
extension of the Capacitated Dispersion Problem (CDP). In addition to the
traditional dispersion objective, solutions are evaluated by a symmetry
penalty that discourages selecting nodes of different colours. The
implementation provides:

* Data structures to represent nodes, problem instances, and candidate
  solutions.
* A black-box linear symmetry model that translates colour diversity into a
  penalty. The more homogeneous the selected colours are, the lower the
  penalty becomes. The gamma scaling factor can be provided directly or derived
  from the instance.
* Exact solvers based on exhaustive enumeration for the weighted-sum and the
  ε-constraint methods. These approaches are adequate for small to
  medium-sized instances and allow users to explore the Pareto frontier.
* A greedy constructive heuristic that scales to larger instances by
  combining capacity coverage with the symmetry-aware penalty.
* A plotting utility to compare candidate solutions in the plane, highlighting
  how well the proposed solution balances coverage, dispersion, and colour
  coherence.
* Convenience helpers to randomise colour assignments reproducibly so that
  existing instances without colour data can be used directly.

Usage Example
-------------

>>> from cdp_symmetry import ProblemInstance, WeightedSumSolver
>>> instance = ProblemInstance.from_json_path("example_instance.json")
>>> solver = WeightedSumSolver(instance=instance, alpha=0.7)
>>> result = solver.solve()
>>> result.objective_values
{'capacity': 11.0, 'dispersion': 6.02, 'symmetry_penalty': 0.0, 'weighted_objective': 4.214}

The module also exposes a ``main`` function that demonstrates how to load the
example instance, compute different optimisation strategies, and export a plot
of the resulting solution. Run ``python -m cdp_symmetry`` to execute the
demonstration script.

Complexity Notes
----------------

The exact solvers rely on enumerating all subsets of candidate nodes, leading
to :math:`O(2^n)` time complexity with :math:`O(n)` memory, where :math:`n`
is the number of nodes. This is practical for small instances and serves as a
reference to validate heuristics. The greedy heuristic operates in
``O(n log n)`` time due to sorting, providing a scalable albeit approximate
alternative for larger inputs.

The codebase targets Python 3.10+, follows snake_case naming, and integrates
type hints, logging, and unit tests for clarity and maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
import logging
from math import inf
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


DEFAULT_COLOUR_PALETTE: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the module-level logger.

    Parameters
    ----------
    level:
        Logging level applied to the default stream handler.
    """

    if logger.handlers:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


@dataclass(frozen=True)
class Node:
    """Candidate facility node.

    Attributes
    ----------
    node_id:
        Unique identifier for the node.
    capacity:
        Available capacity contributed by the node.
    color:
        Categorical colour label.
    coordinates:
        Tuple representing planar coordinates used for plotting and distance
        calculations.
    """

    node_id: str
    capacity: float
    color: str
    coordinates: Tuple[float, float]


@dataclass(slots=True)
class ProblemInstance:
    """Encapsulates all input data for the CDP with symmetry penalty."""

    nodes: List[Node]
    demand: float
    lambda_penalty: float
    distances: Dict[Tuple[str, str], float] = field(default_factory=dict)
    gamma_override: Optional[float] = None
    _min_distance_cache: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Node identifiers must be unique.")
        if self.demand <= 0:
            raise ValueError("Demand must be strictly positive.")
        if self.lambda_penalty < 0:
            raise ValueError("Lambda penalty must be non-negative.")
        if self.gamma_override is not None and self.gamma_override < 0:
            raise ValueError("Gamma must be non-negative when provided.")
        for (i, j), value in self.distances.items():
            if i == j and value != 0:
                raise ValueError("Distance from a node to itself must be zero.")
            if (j, i) in self.distances and self.distances[(j, i)] != value:
                raise ValueError("Distances must be symmetric.")
        missing_pairs = [
            (i.node_id, j.node_id)
            for i in self.nodes
            for j in self.nodes
            if i.node_id != j.node_id and (i.node_id, j.node_id) not in self.distances
        ]
        if missing_pairs:
            raise ValueError(
                "Missing distance entries for pairs: %s" % ", ".join(
                    f"{i}-{j}" for i, j in missing_pairs
                )
            )

    @property
    def node_ids(self) -> List[str]:
        """Return a list of node identifiers preserving input order."""

        return [node.node_id for node in self.nodes]

    def min_nonzero_distance(self) -> float:
        """Return the minimum strictly positive distance in the instance."""

        if self._min_distance_cache is None:
            positive_distances = [
                value
                for (node_a, node_b), value in self.distances.items()
                if node_a != node_b
            ]
            if not positive_distances:
                raise ValueError("Instance must include at least one inter-node distance.")
            self._min_distance_cache = min(positive_distances)
        return self._min_distance_cache

    @property
    def gamma(self) -> float:
        """Return the gamma scaling factor for the symmetry penalty."""

        if self.gamma_override is not None:
            return self.gamma_override
        return self.lambda_penalty * self.min_nonzero_distance()

    def distance(self, node_a: str, node_b: str) -> float:
        """Retrieve the symmetric distance between two nodes."""

        if node_a == node_b:
            return 0.0
        try:
            return self.distances[(node_a, node_b)]
        except KeyError:
            return self.distances[(node_b, node_a)]

    def with_random_colours(
        self,
        seed: Optional[int] = None,
        *,
        num_colours: Optional[int] = None,
        palette: Sequence[str] = DEFAULT_COLOUR_PALETTE,
    ) -> "ProblemInstance":
        """Return a copy of the instance with randomly generated colours.

        Colours are sampled between three and four distinct values whenever
        possible, depending on the number of nodes available. When the number of
        nodes is less than three, the method falls back to the maximum number of
        distinct colours that can be assigned. The sampling process is
        reproducible when ``seed`` is provided.
        """

        if not self.nodes:
            raise ValueError("Cannot generate colours for an instance with no nodes.")
        if not palette:
            raise ValueError("Palette must contain at least one colour.")
        rng = random.Random(seed)
        max_allowed = min(len(palette), len(self.nodes))
        if num_colours is None:
            if max_allowed >= 4 and len(self.nodes) >= 4:
                num_colours = 4
            elif max_allowed >= 3:
                num_colours = 3
            else:
                num_colours = max_allowed
        if num_colours <= 0 or num_colours > max_allowed:
            raise ValueError(
                "num_colours must be between 1 and min(len(palette), len(nodes))."
            )
        chosen_palette = list(rng.sample(palette, k=num_colours))
        assigned_colours: List[str] = list(chosen_palette)
        while len(assigned_colours) < len(self.nodes):
            assigned_colours.append(rng.choice(chosen_palette))
        rng.shuffle(assigned_colours)
        new_nodes = [
            Node(
                node_id=node.node_id,
                capacity=node.capacity,
                color=assigned_colours[index],
                coordinates=node.coordinates,
            )
            for index, node in enumerate(self.nodes)
        ]
        return ProblemInstance(
            nodes=new_nodes,
            demand=self.demand,
            lambda_penalty=self.lambda_penalty,
            distances=dict(self.distances),
            gamma_override=self.gamma_override,
        )

    @classmethod
    def from_json_path(cls, path: str | Path) -> "ProblemInstance":
        """Load a :class:`ProblemInstance` from a JSON file."""

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        nodes = []
        for index, node_data in enumerate(data["nodes"]):
            colour_value = node_data.get("color")
            if colour_value is None:
                colour_value = DEFAULT_COLOUR_PALETTE[index % len(DEFAULT_COLOUR_PALETTE)]
            nodes.append(
                Node(
                    node_id=str(node_data["id"]),
                    capacity=float(node_data["capacity"]),
                    color=str(colour_value),
                    coordinates=(
                        float(node_data["x"]),
                        float(node_data["y"]),
                    ),
                )
            )
        distances = {
            (str(entry["from"]), str(entry["to"])): float(entry["distance"])
            for entry in data["distances"]
        }
        return cls(
            nodes=nodes,
            demand=float(data["demand"]),
            lambda_penalty=float(data.get("lambda_penalty", 0.0)),
            distances=distances,
            gamma_override=(
                float(data["gamma"]) if "gamma" in data and data["gamma"] is not None else None
            ),
        )


@dataclass(slots=True)
class CandidateSolution:
    """Represents a feasible selection of nodes and its evaluation."""

    selected_nodes: Tuple[str, ...]
    capacity: float
    dispersion: float
    symmetry_penalty: float
    weighted_objective: Optional[float] = None

    @property
    def objective_values(self) -> Dict[str, float]:
        """Expose the raw objective values for reporting."""

        values = {
            "capacity": self.capacity,
            "dispersion": self.dispersion,
            "symmetry_penalty": self.symmetry_penalty,
        }
        if self.weighted_objective is not None:
            values["weighted_objective"] = self.weighted_objective
        return values


def iter_subsets(nodes: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """Yield all non-empty subsets of the provided nodes."""

    for r in range(1, len(nodes) + 1):
        yield from combinations(nodes, r)


def compute_capacity(instance: ProblemInstance, subset: Sequence[str]) -> float:
    """Compute the aggregated capacity of a subset of nodes."""

    capacities = {
        node.node_id: node.capacity
        for node in instance.nodes
    }
    return sum(capacities[node_id] for node_id in subset)


def compute_dispersion(instance: ProblemInstance, subset: Sequence[str]) -> float:
    """Return the minimum pairwise distance among selected nodes."""

    if len(subset) <= 1:
        return inf
    min_distance = inf
    for node_a, node_b in combinations(subset, 2):
        min_distance = min(min_distance, instance.distance(node_a, node_b))
    return min_distance


def symmetry_black_box_linear(
    instance: ProblemInstance,
    subset: Sequence[str],
    gamma: Optional[float] = None,
) -> float:
    """Black-box linear symmetry penalty.

    The penalty equals ``gamma * (|C_S| - 1)``, where ``gamma`` defaults to the
    instance-derived scaling factor. The term is linear in the number of
    colours activated by ``subset`` and returns ``0`` when all nodes share the
    same colour. Users may provide an explicit ``gamma`` value to test alternate
    penalty intensities.
    """

    if not subset:
        return 0.0
    colour_by_node = {
        node.node_id: node.color
        for node in instance.nodes
    }
    active_colours = {colour_by_node[node_id] for node_id in subset}
    if len(active_colours) <= 1:
        return 0.0
    gamma_value = instance.gamma if gamma is None else gamma
    penalty = gamma_value * (len(active_colours) - 1)
    logger.debug(
        "Symmetry penalty computed: subset=%s colours=%s penalty=%.3f",
        subset,
        active_colours,
        penalty,
    )
    return penalty


def evaluate_subset(
    instance: ProblemInstance,
    subset: Sequence[str],
) -> CandidateSolution:
    """Evaluate a subset of nodes and return a :class:`CandidateSolution`."""

    capacity = compute_capacity(instance, subset)
    dispersion = compute_dispersion(instance, subset)
    symmetry_penalty = symmetry_black_box_linear(instance, subset)
    solution = CandidateSolution(
        selected_nodes=tuple(sorted(subset)),
        capacity=capacity,
        dispersion=dispersion,
        symmetry_penalty=symmetry_penalty,
    )
    logger.debug("Evaluated subset %s -> %s", subset, solution)
    return solution


class WeightedSumSolver:
    """Exact solver using the weighted-sum scalarisation."""

    def __init__(self, instance: ProblemInstance, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.instance = instance
        self.alpha = alpha

    def solve(self) -> CandidateSolution:
        """Enumerate feasible subsets and return the best solution."""

        best_solution: Optional[CandidateSolution] = None
        best_score = -inf
        for subset in iter_subsets(self.instance.node_ids):
            solution = evaluate_subset(self.instance, subset)
            if solution.capacity < self.instance.demand:
                continue
            dispersion_term = (
                solution.dispersion if solution.dispersion < inf else 0.0
            )
            score = self.alpha * dispersion_term - (1 - self.alpha) * solution.symmetry_penalty
            logger.debug(
                "Subset %s -> dispersion %.3f penalty %.3f weighted %.3f",
                subset,
                dispersion_term,
                solution.symmetry_penalty,
                score,
            )
            if score > best_score:
                best_score = score
                best_solution = CandidateSolution(
                    selected_nodes=solution.selected_nodes,
                    capacity=solution.capacity,
                    dispersion=solution.dispersion,
                    symmetry_penalty=solution.symmetry_penalty,
                    weighted_objective=score,
                )
        if best_solution is None:
            raise ValueError("No feasible solution meets the demand constraint.")
        logger.info(
            "Weighted sum best solution %s with score %.3f",
            best_solution.selected_nodes,
            best_score,
        )
        return best_solution


class EpsilonConstraintSolver:
    """Exact solver for ε-constraint formulations."""

    def __init__(self, instance: ProblemInstance) -> None:
        self.instance = instance

    def solve_max_dispersion(
        self,
        epsilon: float,
    ) -> CandidateSolution:
        """Maximise dispersion subject to a symmetry penalty limit."""

        feasible: List[CandidateSolution] = []
        for subset in iter_subsets(self.instance.node_ids):
            solution = evaluate_subset(self.instance, subset)
            if solution.capacity < self.instance.demand:
                continue
            if solution.symmetry_penalty <= epsilon:
                feasible.append(solution)
        if not feasible:
            raise ValueError("No feasible solution satisfies the epsilon constraint.")
        best_solution = max(
            feasible,
            key=lambda sol: sol.dispersion if sol.dispersion < inf else 0.0,
        )
        logger.info(
            "Epsilon constraint (max dispersion) selected %s",
            best_solution.selected_nodes,
        )
        return best_solution

    def solve_min_penalty(
        self,
        epsilon: float,
    ) -> CandidateSolution:
        """Minimise symmetry penalty while enforcing a dispersion threshold."""

        feasible: List[CandidateSolution] = []
        for subset in iter_subsets(self.instance.node_ids):
            solution = evaluate_subset(self.instance, subset)
            if solution.capacity < self.instance.demand:
                continue
            dispersion_value = solution.dispersion if solution.dispersion < inf else 0.0
            if dispersion_value >= epsilon:
                feasible.append(solution)
        if not feasible:
            raise ValueError("No feasible solution satisfies the dispersion constraint.")
        best_solution = min(feasible, key=lambda sol: sol.symmetry_penalty)
        logger.info(
            "Epsilon constraint (min penalty) selected %s",
            best_solution.selected_nodes,
        )
        return best_solution


class GreedySymmetryHeuristic:
    """Greedy heuristic balancing capacity, dispersion, and symmetry."""

    def __init__(self, instance: ProblemInstance) -> None:
        self.instance = instance

    def solve(self) -> CandidateSolution:
        """Construct a solution by iteratively adding promising nodes."""

        nodes_sorted = sorted(
            self.instance.nodes,
            key=lambda node: (node.capacity, node.color),
            reverse=True,
        )
        selected: List[str] = []
        for node in nodes_sorted:
            trial_subset = selected + [node.node_id]
            trial_solution = evaluate_subset(self.instance, trial_subset)
            if trial_solution.capacity < self.instance.demand:
                selected.append(node.node_id)
                continue
            current_solution = evaluate_subset(self.instance, selected)
            current_score = self._heuristic_score(current_solution)
            trial_score = self._heuristic_score(trial_solution)
            if trial_score >= current_score:
                selected.append(node.node_id)
        if compute_capacity(self.instance, selected) < self.instance.demand:
            logger.warning("Heuristic could not reach demand. Falling back to best effort.")
        final_solution = evaluate_subset(self.instance, selected)
        logger.info(
            "Greedy heuristic selected %s with score %.3f",
            final_solution.selected_nodes,
            self._heuristic_score(final_solution),
        )
        return final_solution

    @staticmethod
    def _heuristic_score(solution: CandidateSolution) -> float:
        dispersion_term = solution.dispersion if solution.dispersion < inf else 0.0
        return dispersion_term - solution.symmetry_penalty


def plot_solution(
    instance: ProblemInstance,
    base_solution: CandidateSolution,
    proposed_solution: CandidateSolution,
    output_path: Path,
) -> Path:
    """Generate a scatter plot comparing two solutions.

    Parameters
    ----------
    instance:
        Problem data providing coordinates and colours.
    base_solution:
        Reference solution (e.g., heuristic outcome) plotted with hollow markers.
    proposed_solution:
        Highlighted solution (e.g., exact solver result) plotted with filled markers.
    output_path:
        Destination path where the image will be stored.

    Returns
    -------
    Path
        Absolute path to the generated image file.
    """

    if plt is None:  # pragma: no cover - guard for optional dependency
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via 'pip install matplotlib'."
        )

    colours = {node.node_id: node.color for node in instance.nodes}
    coords = {node.node_id: node.coordinates for node in instance.nodes}

    plt.figure(figsize=(8, 6))
    for node_id, (x_coord, y_coord) in coords.items():
        colour = colours[node_id]
        marker = "o" if node_id in proposed_solution.selected_nodes else "x"
        fill_style = "full" if node_id in proposed_solution.selected_nodes else "none"
        alpha = 0.9 if node_id in proposed_solution.selected_nodes else 0.5
        plt.scatter(
            x_coord,
            y_coord,
            label=f"{node_id} ({colour})",
            marker=marker,
            facecolors=colour if fill_style == "full" else "none",
            edgecolors=colour,
            s=120,
            alpha=alpha,
        )
    for node_id in base_solution.selected_nodes:
        if node_id in proposed_solution.selected_nodes:
            continue
        x_coord, y_coord = coords[node_id]
        colour = colours[node_id]
        plt.scatter(
            x_coord,
            y_coord,
            marker="s",
            facecolors="none",
            edgecolors=colour,
            s=140,
            alpha=0.5,
        )
    plt.title("CDP Symmetry Solutions")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Plot saved to %s", output_path)
    return output_path.resolve()


def main() -> None:
    """Demonstration entry point executed via ``python -m cdp_symmetry``."""

    configure_logging()
    example_path = Path(__file__).with_name("example_instance.json")
    if not example_path.exists():
        raise FileNotFoundError("example_instance.json not found next to module.")
    instance = ProblemInstance.from_json_path(example_path)
    instance = instance.with_random_colours(seed=42)
    weighted_solver = WeightedSumSolver(instance, alpha=0.6)
    weighted_solution = weighted_solver.solve()
    epsilon_solver = EpsilonConstraintSolver(instance)
    epsilon_solution = epsilon_solver.solve_max_dispersion(
        epsilon=weighted_solution.symmetry_penalty + 1e-9
    )
    heuristic = GreedySymmetryHeuristic(instance)
    heuristic_solution = heuristic.solve()
    plot_path: Optional[Path] = None
    try:
        plot_path = plot_solution(
            instance=instance,
            base_solution=heuristic_solution,
            proposed_solution=weighted_solution,
            output_path=Path("output") / "cdp_symmetry_solution.png",
        )
    except RuntimeError as exc:  # pragma: no cover - optional plotting
        logger.warning("Plotting skipped: %s", exc)
    logger.info("Weighted solution: %s", weighted_solution.objective_values)
    logger.info("Epsilon solution: %s", epsilon_solution.objective_values)
    logger.info("Heuristic solution: %s", heuristic_solution.objective_values)
    if plot_path is not None:
        logger.info("Plot saved at %s", plot_path)


if __name__ == "__main__":
    main()

