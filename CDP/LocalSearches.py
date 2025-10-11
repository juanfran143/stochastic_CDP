"""Local search procedures (tabu-inspired) for solution refinement."""

from __future__ import annotations

import math
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

        improved_dispersion = (
            current_solution.objective_value > best_solution.objective_value + 1e-9
        )
        same_dispersion = math.isclose(
            current_solution.objective_value,
            best_solution.objective_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        better_symmetry = (
            current_solution.symmetry_penalty < best_solution.symmetry_penalty - 1e-9
        )
        if improved_dispersion or (same_dispersion and better_symmetry):
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    best_solution.reevaluate()
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

        improved_dispersion = (
            current_solution.objective_value > best_solution.objective_value + 1e-9
        )
        same_dispersion = math.isclose(
            current_solution.objective_value,
            best_solution.objective_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        better_symmetry = (
            current_solution.symmetry_penalty < best_solution.symmetry_penalty - 1e-9
        )
        if improved_dispersion or (same_dispersion and better_symmetry):
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

    best_solution.reevaluate()
    return best_solution, candidate_list



