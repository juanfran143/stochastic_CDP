"""Simulation-based evaluation utilities used by the simheuristic framework."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from Solution import Solution
from objects import WeightedCandidate


@dataclass
class Simheuristic:
    """Provides stochastic simulation capabilities for candidate solutions."""

    simulationRuns: int
    variance: float

    def randomCapacity(self, mean: float) -> float:
        return random.lognormvariate(math.log(mean), self.variance)

    def runReliabilitySimulation(self, solution: Solution) -> None:
        failures = 0
        capacities: List[float] = []
        solution.stochasticObjective[1] = solution.objectiveValue

        for _ in range(self.simulationRuns):
            stochasticCapacity = sum(
                solution.instance.capacities[node] - (self.randomCapacity(solution.instance.capacities[node]) - solution.instance.capacities[node])
                for node in solution.selectedVertices
            )

            if stochasticCapacity < solution.instance.minCapacity:
                failures += 1
            capacities.append(stochasticCapacity)

        solution.reliability[1] = (self.simulationRuns - failures) / self.simulationRuns
        solution.stochasticCapacity[1] = float(sum(capacities) / len(capacities)) if capacities else 0.0

    def runStochasticEvaluation(self, solution: Solution, candidateList: List[WeightedCandidate]) -> None:
        failures = 0
        capacities: List[float] = []
        solution.stochasticObjective[2] = []

        for _ in range(self.simulationRuns):
            auxSolution = solution.copy()
            stochasticCapacity = sum(
                solution.instance.capacities[node] - (self.randomCapacity(solution.instance.capacities[node]) - solution.instance.capacities[node])
                for node in solution.selectedVertices
            )

            if stochasticCapacity < solution.instance.minCapacity:
                failures += 1
                index = 0
                while stochasticCapacity < solution.instance.minCapacity and index < len(candidateList):
                    candidate = candidateList[index]
                    stochasticCapacity += solution.instance.capacities[candidate.vertex]
                    if auxSolution.objectiveValue > candidate.distance:
                        auxSolution.updateObjective(candidate.vertex, candidate.nearestVertex, candidate.distance)
                    self.updateWeightedCandidateList(auxSolution, candidateList, candidate.vertex)
                    index += 1
                stochasticCapacity = sum(solution.instance.capacities)

            capacities.append(stochasticCapacity)
            solution.stochasticObjective[2].append(auxSolution.objectiveValue)

        solution.reliability[2] = (self.simulationRuns - failures) / self.simulationRuns
        solution.stochasticCapacity[2] = (
            float(sum(capacities) / len(capacities)) if capacities else 0.0
        )
        values = solution.stochasticObjective[2]
        solution.meanStochasticObjective[2] = (
            float(sum(values) / len(values)) if values else 0.0
        )

    def runFastSimulation(self, solution: Solution) -> Tuple[float, float]:
        failures = 0
        for _ in range(self.simulationRuns):
            stochasticCapacity = sum(
                solution.instance.capacities[node] - (self.randomCapacity(solution.instance.capacities[node]) - solution.instance.capacities[node])
                for node in solution.selectedVertices
            )
            if stochasticCapacity < solution.instance.minCapacity:
                failures += 1

        probability = (self.simulationRuns - failures) / self.simulationRuns
        variance = ((probability * (1 - probability)) / self.simulationRuns) ** 0.5
        lowerBound = probability - 1.96 * variance
        upperBound = probability + 1.96 * variance
        return lowerBound, upperBound

    def updateWeightedCandidateList(
        self, solution: Solution, candidateList: List[WeightedCandidate], lastVertex: int
    ) -> None:
        maxDistance = 1.0
        for candidate in candidateList:
            distance = solution.instance.distances[lastVertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearestVertex = lastVertex
            maxDistance = max(maxDistance, candidate.distance)

        maxCapacity = max((solution.instance.capacities[v] for v in solution.selectedVertices), default=1.0)

        for candidate in candidateList:
            distanceComponent = candidate.distance / maxDistance if maxDistance else 0.0
            capacityComponent = solution.instance.capacities[candidate.vertex] / maxCapacity if maxCapacity else 0.0
            candidate.score = distanceComponent * 0.8 + capacityComponent * 0.2

        candidateList.sort(key=lambda element: element.score, reverse=True)

