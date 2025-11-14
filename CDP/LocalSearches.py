"""Local search procedures (tabu-inspired) for solution refinement."""

from __future__ import annotations

from typing import List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Solution import Solution
from objects import Candidate, WeightedCandidate

def tabu_search_capacity(
    initial_solution: Solution,
    candidateList: List[WeightedCandidate],
    maxIterations: int,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    bestSolution = initial_solution.copy()
    currentSolution = initial_solution.copy()
    iterationsWithoutImprovement = 0

    while iterationsWithoutImprovement < maxIterations:
        removedVertex = currentSolution.selectedVertices[0]
        currentSolution.remove_vertex(removedVertex)

        heuristic.recalculate_weighted_candidate_list(currentSolution, candidateList, removedVertex)
        currentSolution = heuristic.partial_reconstruction_capacity(currentSolution, candidateList)
        currentSolution.reevaluate()
        heuristic.insert_weighted_candidate(candidateList, currentSolution, removedVertex)

        if currentSolution.objectiveValue > bestSolution.objectiveValue:
            bestSolution = currentSolution.copy()
            iterationsWithoutImprovement = 0
        else:
            iterationsWithoutImprovement += 1

    return bestSolution, candidateList



