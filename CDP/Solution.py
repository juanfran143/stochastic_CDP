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
    selectedVertices: List[int] = field(default_factory=list)
    minDistanceVertex1: int = -1
    minDistanceVertex2: int = -1
    objectiveValue: float = field(init=False)
    capacity: float = 0.0
    time: float = 0.0
    colourCounts: Dict[str, int] = field(default_factory=dict)
    reliability: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    stochasticCapacity: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    stochasticObjective: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})
    meanStochasticObjective: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0})

    def __post_init__(self) -> None:
        self.objectiveValue = self.instance.sorted_edges[0].distance * 10

    def copy(self) -> "Solution":
        clone = Solution(self.instance)
        clone.selectedVertices = list(self.selectedVertices)
        clone.minDistanceVertex1 = self.minDistanceVertex1
        clone.minDistanceVertex2 = self.minDistanceVertex2
        clone.objectiveValue = self.objectiveValue
        clone.capacity = self.capacity
        clone.time = self.time
        clone.colourCounts = dict(self.colourCounts)
        clone.reliability = dict(self.reliability)
        clone.stochasticCapacity = dict(self.stochasticCapacity)
        clone.stochasticObjective = dict(self.stochasticObjective)
        clone.meanStochasticObjective = dict(self.meanStochasticObjective)
        return clone

    def add_vertex(self, vertex: int) -> None:
        self.selectedVertices.append(vertex)
        self.capacity += self.instance.capacities[vertex]

        # Maintenance state: Update colour count when adding vertices
        if getattr(self.instance, "colours", None):
            colour = self.instance.colours[vertex]
            self.colourCounts[colour] = self.colourCounts.get(colour, 0) + 1

    def remove_vertex(self, vertex: int) -> None:
        self.selectedVertices.remove(vertex)
        self.capacity -= self.instance.capacities[vertex]

        # Maintenance state: Update colour count when removing vertices
        if getattr(self.instance, "colours", None):
            colour = self.instance.colours[vertex]
            self.colourCounts[colour] -= 1
            # If the count for that colour is 0, remove it from the dictionary to ensure len(colourCounts) correctly reflects f2(S).
            if self.colourCounts[colour] == 0:
                del self.colourCounts[colour]

    def distance_to(self, vertex: int) -> Tuple[int, float]:
        minDistance = self.instance.sorted_edges[0].distance * 10
        minVertex = -1
        for selected in self.selectedVertices:
            candidateDistance = self.instance.distances[selected][vertex]
            if candidateDistance < minDistance:
                minDistance = candidateDistance
                minVertex = selected
        return minVertex, minDistance

    def is_feasible(self) -> bool:
        return self.capacity >= self.instance.min_capacity

    def update_objective(self, vertex1: int, vertex2: int, distance: float) -> None:
        self.objectiveValue = distance
        self.minDistanceVertex1 = vertex1
        self.minDistanceVertex2 = vertex2

    def evaluate_complete(self) -> float:
        self.objectiveValue = self.instance.sorted_edges[0].distance * 10
        for vertex1 in self.selectedVertices:
            for vertex2 in self.selectedVertices:
                if vertex1 == vertex2:
                    continue
                distance = self.instance.distances[vertex1][vertex2]
                if distance < self.objectiveValue:
                    self.objectiveValue = distance
        return self.objectiveValue

    def reevaluate(self) -> None:
        self.objectiveValue = self.instance.sorted_edges[0].distance * 10
        for vertex1 in self.selectedVertices:
            for vertex2 in self.selectedVertices:
                if vertex1 == vertex2:
                    continue
                distance = self.instance.distances[vertex1][vertex2]
                if distance < self.objectiveValue:
                    self.objectiveValue = distance
                    self.minDistanceVertex1 = vertex1
                    self.minDistanceVertex2 = vertex2


    def get_colour_type_count(self) -> int:
        # Return the number of distinct colours in the solution
        return len(self.colourCounts)