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
    def initialSolution(self) -> Solution:
        solution = Solution(self.instance)
        edge = self.instance.sortedEdges[self.firstEdgeIndex]
        solution.addVertex(edge.vertex1)
        solution.addVertex(edge.vertex2)
        solution.updateObjective(edge.vertex1, edge.vertex2, edge.distance)
        self.maxCapacity = max(
            self.instance.capacities[edge.vertex1],
            self.instance.capacities[edge.vertex2],
        )
        self.maxMinDistance = edge.distance
        return solution

    def weightedScore(self, distance: float, capacity: float) -> float:
        distanceComponent = distance / self.maxMinDistance if self.maxMinDistance else 0.0
        capacityComponent = capacity / self.maxCapacity if self.maxCapacity else 0.0
        return distanceComponent * self.weight + capacityComponent * (1 - self.weight)

    def applyConfiguredWeight(self, lower: float = 0.6, upper: float = 0.9) -> None:
        if self.configuredWeight is not None:
            self.weight = self.configuredWeight
        else:
            self.weight = random.uniform(lower, upper)

    def buildCandidateList(self, solution: Solution) -> List[Candidate]:
        candidates: List[Candidate] = []
        for vertex in range(self.instance.nodeCount):
            if vertex in solution.selectedVertices:
                continue
            nearestVertex, distance = solution.distanceTo(vertex)
            candidates.append(Candidate(vertex, nearestVertex, distance))
        candidates.sort(key=lambda item: item.distance, reverse=True)
        return candidates

    def buildWeightedCandidateList(self, solution: Solution) -> List[WeightedCandidate]:
        weightedCandidates: List[WeightedCandidate] = []
        selectedCapacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.maxCapacity = max(selectedCapacities) if selectedCapacities else max(self.instance.capacities)

        candidates: List[Candidate] = []
        for vertex in range(self.instance.nodeCount):
            if vertex in solution.selectedVertices:
                continue
            nearestVertex, distance = solution.distanceTo(vertex)
            candidates.append(Candidate(vertex, nearestVertex, distance))

        self.maxMinDistance = max((candidate.distance for candidate in candidates), default=1.0)

        for candidate in candidates:
            score = self.weightedScore(candidate.distance, self.instance.capacities[candidate.vertex])
            weightedCandidates.append(
                WeightedCandidate(candidate.vertex, candidate.nearestVertex, candidate.distance, score)
            )

        weightedCandidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return weightedCandidates

    def randomIndex(self, size: int, beta: float) -> int:
        position = int(math.log(random.random()) / math.log(1 - beta))
        return position % size if size else 0

    # ------------------------------------------------------------------
    # Deterministic constructions
    # ------------------------------------------------------------------
    def constructGreedySolution(self) -> Solution:
        solution = self.initialSolution()
        candidateList = self.buildCandidateList(solution)
        alpha = self.alpha if self.alpha >= 0 else random.random()
        while not solution.isFeasible():
            limit = candidateList[0].distance - (alpha * candidateList[-1].distance)
            bestIndex = max(
                range(len(candidateList)),
                key=lambda index: (
                    candidateList[index].distance >= limit,
                    self.instance.capacities[candidateList[index].vertex],
                ),
            )
            candidate = candidateList.pop(bestIndex)
            solution.addVertex(candidate.vertex)
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateCandidateList(solution, candidateList, candidate.vertex)
        return solution

    # ------------------------------------------------------------------
    # Biased randomized constructions
    # ------------------------------------------------------------------
    def constructBiasedSolution(self) -> Tuple[Solution, List[Candidate]]:
        solution = self.initialSolution()
        candidateList = self.buildCandidateList(solution)
        while not solution.isFeasible():
            position = self.randomIndex(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.addVertex(candidate.vertex)
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateCandidateList(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def constructBiasedCapacitySolution(self) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initialSolution()
        self.applyConfiguredWeight(0.6, 0.9)
        candidateList = self.buildWeightedCandidateList(solution)

        while not solution.isFeasible():
            position = self.randomIndex(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def constructBiasedCapacitySimulationSolution(
        self,
        simulation: "Simheuristic",
        reliabilityThreshold: float,
    ) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initialSolution()
        self.applyConfiguredWeight(0.6, 0.9)
        candidateList = self.buildWeightedCandidateList(solution)

        lowerBound, _ = simulation.runFastSimulation(solution)
        while lowerBound < reliabilityThreshold:
            position = self.randomIndex(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
            lowerBound, _ = simulation.runFastSimulation(solution)
        return solution, candidateList

    def constructBiasedFixedWeightSolution(self, weight: float) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initialSolution()
        self.weight = weight
        candidateList = self.buildWeightedCandidateList(solution)

        while not solution.isFeasible():
            position = self.randomIndex(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
        return solution, candidateList

    def constructBiasedDistributionSolution(
        self, distribution: Dict[Tuple[float, float], float]
    ) -> Tuple[Solution, List[WeightedCandidate], Tuple[float, float]]:
        solution = self.initialSolution()

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
        candidateList = self.buildWeightedCandidateList(solution)

        while not solution.isFeasible():
            position = self.randomIndex(len(candidateList), self.beta)
            candidate = candidateList.pop(position)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
        return solution, candidateList, selectedInterval

    # ------------------------------------------------------------------
    # Candidate list maintenance
    # ------------------------------------------------------------------
    def updateCandidateList(self, solution: Solution, candidateList: List[Candidate], lastVertex: int) -> None:
        for candidate in candidateList:
            distance = self.instance.distances[lastVertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearestVertex = lastVertex
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def updateWeightedCandidateList(
        self, solution: Solution, candidateList: List[WeightedCandidate], lastVertex: int
    ) -> None:
        for candidate in candidateList:
            distance = self.instance.distances[lastVertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearestVertex = lastVertex
        self.maxMinDistance = max((candidate.distance for candidate in candidateList), default=1.0)
        for candidate in candidateList:
            candidate.score = self.weightedScore(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidateList.sort(key=lambda item: item.score, reverse=True)

    def insertCandidate(self, candidateList: List[Candidate], solution: Solution, vertex: int) -> None:
        nearestVertex, distance = solution.distanceTo(vertex)
        candidateList.append(Candidate(vertex, nearestVertex, distance))
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def insertWeightedCandidate(
        self, candidateList: List[WeightedCandidate], solution: Solution, vertex: int
    ) -> None:
        nearestVertex, distance = solution.distanceTo(vertex)
        self.maxCapacity = max(self.maxCapacity, self.instance.capacities[vertex])
        self.maxMinDistance = max(self.maxMinDistance, distance)
        score = self.weightedScore(distance, self.instance.capacities[vertex])
        candidateList.append(WeightedCandidate(vertex, nearestVertex, distance, score))
        candidateList.sort(key=lambda item: item.score, reverse=True)

    def recalculateCandidateList(
        self, solution: Solution, candidateList: List[Candidate], removedVertex: int
    ) -> None:
        for candidate in candidateList:
            if candidate.nearestVertex == removedVertex:
                candidate.nearestVertex, candidate.distance = solution.distanceTo(candidate.vertex)
        candidateList.sort(key=lambda item: item.distance, reverse=True)

    def recalculateWeightedCandidateList(
        self, solution: Solution, candidateList: List[WeightedCandidate], removedVertex: int
    ) -> None:
        for candidate in candidateList:
            if candidate.nearestVertex == removedVertex:
                candidate.nearestVertex, candidate.distance = solution.distanceTo(candidate.vertex)

        selectedCapacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.maxCapacity = max(selectedCapacities) if selectedCapacities else max(self.instance.capacities)
        self.maxMinDistance = max((candidate.distance for candidate in candidateList), default=1.0)

        for candidate in candidateList:
            candidate.score = self.weightedScore(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidateList.sort(key=lambda item: item.score, reverse=True)

    # ------------------------------------------------------------------
    # Partial reconstructions for tabu search
    # ------------------------------------------------------------------
    def partialReconstruction(self, solution: Solution, candidateList: List[Candidate]) -> Solution:
        while not solution.isFeasible():
            index = self.randomIndex(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.addVertex(candidate.vertex)
            self.updateCandidateList(solution, candidateList, candidate.vertex)
        return solution

    def partialReconstructionCapacity(
        self, solution: Solution, candidateList: List[WeightedCandidate]
    ) -> Solution:
        while not solution.isFeasible():
            index = self.randomIndex(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
        return solution

    def partialReconstructionSimulation(
        self,
        solution: Solution,
        candidateList: List[WeightedCandidate],
        simulation: "Simheuristic",
        reliabilityThreshold: float,
    ) -> Solution:
        lowerBound, _ = simulation.runFastSimulation(solution)
        while lowerBound < reliabilityThreshold:
            index = self.randomIndex(len(candidateList), self.betaLocalSearch)
            candidate = candidateList.pop(index)
            solution.addVertex(candidate.vertex)
            self.maxCapacity = max(self.maxCapacity, self.instance.capacities[candidate.vertex])
            self.updateWeightedCandidateList(solution, candidateList, candidate.vertex)
            lowerBound, _ = simulation.runFastSimulation(solution)
        return solution


# Avoid circular imports at module level
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from simheuristic import Simheuristic

