"""Local search procedures (tabu-inspired) for solution refinement."""

from __future__ import annotations

from typing import List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Solution import Solution
from objects import Candidate, WeightedCandidate


def tabu_search(
    initial_solution: Solution,
    candidate_list: List[Candidate],
    max_iterations: int,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[Candidate]]:
    best_solution = initial_solution.copy()
    current_solution = initial_solution.copy()
    iterations_without_improvement = 0

    while iterations_without_improvement < max_iterations:
        removed_vertex = current_solution.selected_vertices[0]
        current_solution.remove_vertex(removed_vertex)
        heuristic.recalculate_candidate_list(current_solution, candidate_list, removed_vertex)
        current_solution = heuristic.partial_reconstruction(current_solution, candidate_list)
        current_solution.reevaluate()
        heuristic.insert_candidate(candidate_list, current_solution, removed_vertex)

        if current_solution.objective_value > best_solution.objective_value:
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    return best_solution, candidate_list


def tabu_search_capacity(
    initial_solution: Solution,
    candidate_list: List[WeightedCandidate],
    max_iterations: int,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    best_solution = initial_solution.copy()
    current_solution = initial_solution.copy()
    iterations_without_improvement = 0

    while iterations_without_improvement < max_iterations:
        removed_vertex = current_solution.selected_vertices[0]
        current_solution.remove_vertex(removed_vertex)
        heuristic.recalculate_weighted_candidate_list(current_solution, candidate_list, removed_vertex)
        current_solution = heuristic.partial_reconstruction_capacity(current_solution, candidate_list)
        current_solution.reevaluate()
        heuristic.insert_weighted_candidate(candidate_list, current_solution, removed_vertex)

        if current_solution.objective_value > best_solution.objective_value:
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    return best_solution, candidate_list



