"""Constructive heuristics used by the multi-start framework."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from Instance import Instance
from Solution import Solution
from objects import Candidate, WeightedCandidate


class ConstructiveHeuristic:
    """Generates initial solutions and supports local search adjustments."""

    def __init__(
        self,
        alpha: float,
        betaConstruction: float,
        betaLocalSearch: float,
        instance: Instance,
        weight: float,
    ) -> None:
        self.alpha = alpha
        self.beta = betaConstruction
        self.betaLocalSearch = betaLocalSearch
        self.instance = instance
        self.configuredWeight: Optional[float] = weight if 0.0 <= weight <= 1.0 else None
        self.weight = self.configuredWeight if self.configuredWeight is not None else weight
        self.firstEdgeIndex = 0
        self.maxMinDistance = 1.0
        self.maxCapacity = max(instance.capacities)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def initial_solution(self) -> Solution:
        solution = Solution(self.instance)
        edge = self.instance.sortedEdges[self.firstEdgeIndex]
        solution.add_vertex(edge.vertex1)
        solution.add_vertex(edge.vertex2)
        solution.update_objective(edge.vertex1, edge.vertex2, edge.distance)
        self.maxCapacity = max(
            self.instance.capacities[edge.vertex1],
            self.instance.capacities[edge.vertex2],
        )
        self.maxMinDistance = edge.distance
        return solution

    def weighted_score(self, distance: float, capacity: float) -> float:
        distanceComponent = distance / self.maxMinDistance if self.maxMinDistance else 0.0
        capacityComponent = capacity / self.maxCapacity if self.maxCapacity else 0.0
        return distanceComponent * self.weight + capacityComponent * (1 - self.weight)

    def apply_configured_weight(self, lower: float = 0.6, upper: float = 0.9) -> None:
        if self.configuredWeight is not None:
            self.weight = self.configuredWeight
        else:
            self.weight = random.uniform(lower, upper)

    def build_candidate_list(self, solution: Solution) -> List[Candidate]:
        candidates: List[Candidate] = []
        for vertex in range(self.instance.nodeCount):
            if vertex in solution.selectedVertices:
                continue
            nearestVertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearestVertex, distance))
        candidates.sort(key=lambda item: item.distance, reverse=True)
        return candidates

    def build_weighted_candidate_list(self, solution: Solution) -> List[WeightedCandidate]:
        weightedCandidates: List[WeightedCandidate] = []
        selectedCapacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.maxCapacity = max(selectedCapacities) if selectedCapacities else max(self.instance.capacities)

        candidates: List[Candidate] = []
        for vertex in range(self.instance.nodeCount):
            if vertex in solution.selectedVertices:
                continue
            nearestVertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearestVertex, distance))

        self.maxMinDistance = max((candidate.distance for candidate in candidates), default=1.0)

        for candidate in candidates:
            score = self.weighted_score(candidate.distance, self.instance.capacities[candidate.vertex])
            weightedCandidates.append(
                WeightedCandidate(candidate.vertex, candidate.nearestVertex, candidate.distance, score)
            )

        weightedCandidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return weightedCandidates

    def random_index(self, size: int, beta: float) -> int:
        position = int(math.log(random.random()) / math.log(1 - beta))
        return position % size if size else 0

    # ------------------------------------------------------------------
    # Deterministic constructions
    # ------------------------------------------------------------------
    def construct_greedy_solution(self) -> Solution:
        solution = self.initial_solution()
        candidateList = self.build_candidate_list(solution)
        alpha = self.alpha if self.alpha >= 0 else random.random()
        while not solution.is_feasible():
            limit = candidateList[0].distance - (alpha * candidateList[-1].distance)
            bestIndex = max(
                range(len(candidateList)),
                key=lambda index: (
                    candidateList[index].distance >= limit,
                    self.instance.capacities[candidateList[index].vertex],
                ),
            )
            candidate = candidateList.pop(bestIndex)
            solution.add_vertex(candidate.vertex)
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_candidate_list(solution, candidateList, candidate.vertex)
        return solution

    # ------------------------------------------------------------------
    # Biased randomized constructions
    # ------------------------------------------------------------------
    def construct_biased_solution(self) -> Tuple[Solution, List[Candidate]]:
        solution = self.initial_solution()
        candidateList = self.build_candidate_list(solution)
        while not solution.is_feasible():
            position = self.random_index(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.add_vertex(candidate.vertex)
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_candidate_list(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def construct_biased_capacity_solution(self) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        self.apply_configured_weight(0.6, 0.9)
        candidateList = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def construct_biased_capacity_simulation_solution(
        self,
        simulation: "Simheuristic",
        reliabilityThreshold: float,
    ) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        self.apply_configured_weight(0.6, 0.9)
        candidateList = self.build_weighted_candidate_list(solution)

        lowerBound, _ = simulation.run_fast_simulation(solution)
        while lowerBound < reliabilityThreshold:
            position = self.random_index(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
            lowerBound, _ = simulation.run_fast_simulation(solution)
        return solution, candidateList

    def construct_biased_fixed_weight_solution(self, weight: float) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        self.weight = weight
        candidateList = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def construct_biased_distribution_solution(
        self, distribution: Dict[Tuple[float, float], float]
    ) -> Tuple[Solution, List[WeightedCandidate], Tuple[float, float]]:
        solution = self.initial_solution()

        randomValue = random.random()
        cumulative = 0.0
        selectedInterval = next(reversed(distribution))
        for interval, probability in distribution.items():
            cumulative += probability
            if randomValue <= cumulative:
                selectedInterval = interval
                break

        lower, upper = selectedInterval
        if self.configuredWeight is not None:
            self.weight = self.configuredWeight
        else:
            self.weight = random.uniform(lower, upper if upper <= 1 else 1)
        candidateList = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
        return solution, candidateList, selectedInterval

    # ------------------------------------------------------------------
    # Candidate list maintenance
    # ------------------------------------------------------------------
    def update_candidate_list(self, solution: Solution, candidateList: List[Candidate], lastVertex: int) -> None:
        for candidate in candidateList:
            distance = self.instance.distances[lastVertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearestVertex = lastVertex
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def update_weighted_candidate_list(
        self, solution: Solution, candidateList: List[WeightedCandidate], lastVertex: int
    ) -> None:
        for candidate in candidateList:
            distance = self.instance.distances[lastVertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearestVertex = lastVertex
        self.maxMinDistance = max((candidate.distance for candidate in candidateList), default=1.0)
        for candidate in candidateList:
            candidate.score = self.weighted_score(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidateList.sort(key=lambda item: item.score, reverse=True)

    def insert_candidate(self, candidateList: List[Candidate], solution: Solution, vertex: int) -> None:
        nearestVertex, distance = solution.distance_to(vertex)
        candidateList.append(Candidate(vertex, nearestVertex, distance))
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def insert_weighted_candidate(
        self, candidateList: List[WeightedCandidate], solution: Solution, vertex: int
    ) -> None:
        nearestVertex, distance = solution.distance_to(vertex)
        self.maxCapacity = max(self.maxCapacity, self.instance.capacities[vertex])
        self.maxMinDistance = max(self.maxMinDistance, distance)
        score = self.weighted_score(distance, self.instance.capacities[vertex])
        candidateList.append(WeightedCandidate(vertex, nearestVertex, distance, score))
        candidateList.sort(key=lambda item: item.score, reverse=True)

    def recalculate_candidate_list(
        self, solution: Solution, candidateList: List[Candidate], removedVertex: int
    ) -> None:
        for candidate in candidateList:
            if candidate.nearestVertex == removedVertex:
                candidate.nearestVertex, candidate.distance = solution.distance_to(candidate.vertex)
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def recalculate_weighted_candidate_list(
        self, solution: Solution, candidateList: List[WeightedCandidate], removedVertex: int
    ) -> None:
        for candidate in candidateList:
            if candidate.nearestVertex == removedVertex:
                candidate.nearestVertex, candidate.distance = solution.distance_to(candidate.vertex)

        selectedCapacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.maxCapacity = max(selectedCapacities) if selectedCapacities else max(self.instance.capacities)
        self.maxMinDistance = max((candidate.distance for candidate in candidateList), default=1.0)

        for candidate in candidateList:
            candidate.score = self.weighted_score(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidateList.sort(key=lambda item: item.score, reverse=True)

    # ------------------------------------------------------------------
    # Partial reconstructions for tabu search
    # ------------------------------------------------------------------
    def partial_reconstruction(self, solution: Solution, candidateList: List[Candidate]) -> Solution:
        while not solution.is_feasible():
            index = self.random_index(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.add_vertex(candidate.vertex)
            self.update_candidate_list(solution, candidateList, candidate.vertex)
        return solution

    def partial_reconstruction_capacity(
        self, solution: Solution, candidateList: List[WeightedCandidate]
    ) -> Solution:
        while not solution.is_feasible():
            index = self.random_index(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
        return solution

    def partial_reconstruction_simulation(
        self,
        solution: Solution,
        candidateList: List[WeightedCandidate],
        simulation: "Simheuristic",
        reliabilityThreshold: float,
    ) -> Solution:
        lowerBound, _ = simulation.run_fast_simulation(solution)
        while lowerBound < reliabilityThreshold:
            index = self.random_index(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.add_vertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            self.update_weighted_candidate_list(solution, candidateList, candidate.vertex)
            lowerBound, _ = simulation.run_fast_simulation(solution)
        return solution


# Avoid circular imports at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from simheuristic import Simheuristic

