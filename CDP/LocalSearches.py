"""Local search procedures (tabu-inspired) for solution refinement."""

from __future__ import annotations

from typing import List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Solution import Solution
from objects import Candidate, WeightedCandidate


def tabuSearch(
    initialSolution: Solution,
    candidateList: List[Candidate],
    maxIterations: int,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[Candidate]]:
    bestSolution = initialSolution.copy()
    currentSolution = initialSolution.copy()
    iterationsWithoutImprovement = 0

    while iterationsWithoutImprovement < maxIterations:
        removedVertex = currentSolution.selectedVertices[0]
        currentSolution.removeVertex(removedVertex)
        heuristic.recalculateCandidateList(currentSolution, candidateList, removedVertex)
        currentSolution = heuristic.partialReconstruction(currentSolution, candidateList)
        currentSolution.reevaluate()
        heuristic.insertCandidate(candidateList, currentSolution, removedVertex)

        if currentSolution.objectiveValue > bestSolution.objectiveValue:
            bestSolution = currentSolution.copy()
            iterationsWithoutImprovement = 0
        else:
            iterationsWithoutImprovement += 1

    return bestSolution, candidateList


def tabuSearchCapacity(
    initialSolution: Solution,
    candidateList: List[WeightedCandidate],
    maxIterations: int,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    bestSolution = initialSolution.copy()
    currentSolution = initialSolution.copy()
    iterationsWithoutImprovement = 0

    while iterationsWithoutImprovement < maxIterations:
        removedVertex = currentSolution.selectedVertices[0]
        currentSolution.removeVertex(removedVertex)
        heuristic.recalculateWeightedCandidateList(currentSolution, candidateList, removedVertex)
        currentSolution = heuristic.partialReconstructionCapacity(currentSolution, candidateList)
        currentSolution.reevaluate()
        heuristic.insertWeightedCandidate(candidateList, currentSolution, removedVertex)

        if currentSolution.objectiveValue > bestSolution.objectiveValue:
            bestSolution = currentSolution.copy()
            iterationsWithoutImprovement = 0
        else:
            iterationsWithoutImprovement += 1

    return bestSolution, candidateList


def tabuSearchCapacitySimulation(
    initialSolution: Solution,
    candidateList: List[WeightedCandidate],
    maxIterations: int,
    heuristic: ConstructiveHeuristic,
    simulation: "Simheuristic",
    reliabilityThreshold: float,
) -> Tuple[Solution, List[WeightedCandidate]]:
    bestSolution = initialSolution.copy()
    currentSolution = initialSolution.copy()
    iterationsWithoutImprovement = 0

    while iterationsWithoutImprovement < maxIterations:
        removedVertex = currentSolution.selectedVertices[0]
        currentSolution.removeVertex(removedVertex)
        heuristic.recalculateWeightedCandidateList(currentSolution, candidateList, removedVertex)
        currentSolution = heuristic.partialReconstructionSimulation(
            currentSolution, candidateList, simulation, reliabilityThreshold
        )
        currentSolution.reevaluate()
        heuristic.insertWeightedCandidate(candidateList, currentSolution, removedVertex)

        if currentSolution.objectiveValue > bestSolution.objectiveValue:
            bestSolution = currentSolution.copy()
            iterationsWithoutImprovement = 0
        else:
            iterationsWithoutImprovement += 1

    return bestSolution, candidateList


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from simheuristic import Simheuristic

