"""Solution representation and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - to avoid circular imports at runtime
    from Instance import Instance


@dataclass
class Solution:
    """Represents a feasible (or partially feasible) solution for the CDP."""

    instance: "Instance"
    selected_vertices: List[int] = field(default_factory=list)
    min_distance_vertex1: int = -1
    min_distance_vertex2: int = -1
    objective_value: float = field(init=False)
    cdp_objective: float = field(init=False)
    capacity: float = 0.0
    time: float = 0.0
    symmetry_penalty: float = 0.0
    symmetry_breakdown: Dict[Tuple[str, str], float] = field(default_factory=dict)
    symmetry_objective: float = 0.0
    objective_alpha: float = 1.0
    alpha_history: List[float] = field(default_factory=list)
    pareto_history: List[Tuple[float, float, float]] = field(default_factory=list)
    reliability: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    stochastic_capacity: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    stochastic_objective: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    mean_stochastic_objective: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})

    def __post_init__(self) -> None:
        baseline = self.instance.sorted_edges[0].distance * 10
        self.objective_value = baseline
        self.cdp_objective = baseline

    def copy(self) -> "Solution":
        clone = Solution(self.instance)
        clone.selected_vertices = list(self.selected_vertices)
        clone.min_distance_vertex1 = self.min_distance_vertex1
        clone.min_distance_vertex2 = self.min_distance_vertex2
        clone.objective_value = self.objective_value
        clone.cdp_objective = self.cdp_objective
        clone.capacity = self.capacity
        clone.time = self.time
        clone.symmetry_penalty = self.symmetry_penalty
        clone.symmetry_breakdown = dict(self.symmetry_breakdown)
        clone.symmetry_objective = self.symmetry_objective
        clone.objective_alpha = self.objective_alpha
        clone.alpha_history = list(self.alpha_history)
        clone.pareto_history = list(self.pareto_history)
        clone.reliability = dict(self.reliability)
        clone.stochastic_capacity = dict(self.stochastic_capacity)
        clone.stochastic_objective = dict(self.stochastic_objective)
        clone.mean_stochastic_objective = dict(self.mean_stochastic_objective)
        return clone

    def add_vertex(self, vertex: int) -> None:
        self.selected_vertices.append(vertex)
        self.capacity += self.instance.capacities[vertex]

    def remove_vertex(self, vertex: int) -> None:
        self.selected_vertices.remove(vertex)
        self.capacity -= self.instance.capacities[vertex]

    def distance_to(self, vertex: int) -> Tuple[int, float]:
        min_distance = self.instance.sorted_edges[0].distance * 10
        min_vertex = -1
        for selected in self.selected_vertices:
            candidate_distance = self.instance.distances[selected][vertex]
            if candidate_distance < min_distance:
                min_distance = candidate_distance
                min_vertex = selected
        return min_vertex, min_distance

    def is_feasible(self) -> bool:
        return self.capacity >= self.instance.min_capacity

    def update_objective(self, vertex1: int, vertex2: int, distance: float) -> None:
        self.objective_value = distance
        self.cdp_objective = distance
        self.min_distance_vertex1 = vertex1
        self.min_distance_vertex2 = vertex2

    def evaluate_complete(self, alpha: float | None = None) -> float:
        if alpha is not None:
            self.objective_alpha = alpha
        self.objective_value = self.instance.sorted_edges[0].distance * 10
        self.cdp_objective = self.objective_value
        for vertex1 in self.selected_vertices:
            for vertex2 in self.selected_vertices:
                if vertex1 == vertex2:
                    continue
                distance = self.instance.distances[vertex1][vertex2]
                if distance < self.objective_value:
                    self.objective_value = distance
                    self.cdp_objective = distance
        self.evaluate_symmetry()
        self.objective_value = self.weighted_objective()
        return self.objective_value

    def reevaluate(self, alpha: float | None = None) -> None:
        if alpha is not None:
            self.objective_alpha = alpha
        self.objective_value = self.instance.sorted_edges[0].distance * 10
        self.cdp_objective = self.objective_value
        for vertex1 in self.selected_vertices:
            for vertex2 in self.selected_vertices:
                if vertex1 == vertex2:
                    continue
                distance = self.instance.distances[vertex1][vertex2]
                if distance < self.objective_value:
                    self.objective_value = distance
                    self.min_distance_vertex1 = vertex1
                    self.min_distance_vertex2 = vertex2
                    self.cdp_objective = distance
        self.evaluate_symmetry()
        self.objective_value = self.weighted_objective()

    def evaluate_symmetry(self) -> float:
        """Update and return the symmetry penalty for the current selection."""

        penalty, breakdown = self.instance.symmetry_penalty(
            self.selected_vertices, return_breakdown=True
        )
        self.symmetry_penalty = penalty
        self.symmetry_breakdown = breakdown
        self.symmetry_objective = penalty
        return self.symmetry_penalty

    def weighted_objective(self, alpha: float | None = None) -> float:
        """Return the scalarised objective balancing dispersion and symmetry."""

        if alpha is not None:
            self.objective_alpha = alpha
        effective_alpha = self.objective_alpha
        dispersion_value = self.cdp_objective if self.selected_vertices else 0.0
        return effective_alpha * dispersion_value - (1 - effective_alpha) * self.symmetry_penalty

