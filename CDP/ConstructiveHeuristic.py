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
        beta_construction: float,
        beta_local_search: float,
        instance: Instance,
        weight: float,
    ) -> None:
        self.alpha = alpha
        self.beta = beta_construction
        self.beta_local_search = beta_local_search
        self.instance = instance
        self.configured_weight: Optional[float] = weight if 0.0 <= weight <= 1.0 else None
        self.weight = self.configured_weight if self.configured_weight is not None else weight
        self.first_edge_index = 0
        self.max_min_distance = 1.0
        self.max_capacity = max(instance.capacities)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def initial_solution(self) -> Solution:
        solution = Solution(self.instance)
        edge = self.instance.sorted_edges[self.first_edge_index]
        solution.add_vertex(edge.vertex1)
        solution.add_vertex(edge.vertex2)
        solution.update_objective(edge.vertex1, edge.vertex2, edge.distance)
        self.max_capacity = max(
            self.instance.capacities[edge.vertex1],
            self.instance.capacities[edge.vertex2],
        )
        self.max_min_distance = edge.distance
        solution.evaluate_symmetry()
        return solution

    def weighted_score(self, distance: float, capacity: float) -> float:
        distance_component = distance / self.max_min_distance if self.max_min_distance else 0.0
        capacity_component = capacity / self.max_capacity if self.max_capacity else 0.0
        return distance_component * self.weight + capacity_component * (1 - self.weight)

    def apply_configured_weight(self, lower: float = 0.6, upper: float = 0.9) -> None:
        if self.configured_weight is not None:
            self.weight = self.configured_weight
        else:
            self.weight = random.uniform(lower, upper)

    def build_candidate_list(self, solution: Solution) -> List[Candidate]:
        candidates: List[Candidate] = []
        for vertex in range(self.instance.node_count):
            if vertex in solution.selected_vertices:
                continue
            nearest_vertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearest_vertex, distance))
        candidates.sort(key=lambda item: item.distance, reverse=True)
        return candidates

    def build_weighted_candidate_list(self, solution: Solution) -> List[WeightedCandidate]:
        weighted_candidates: List[WeightedCandidate] = []
        selected_capacities = [self.instance.capacities[v] for v in solution.selected_vertices]
        self.max_capacity = (
            max(selected_capacities) if selected_capacities else max(self.instance.capacities)
        )

        candidates: List[Candidate] = []
        for vertex in range(self.instance.node_count):
            if vertex in solution.selected_vertices:
                continue
            nearest_vertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearest_vertex, distance))

        self.max_min_distance = max((candidate.distance for candidate in candidates), default=1.0)

        for candidate in candidates:
            score = self.weighted_score(candidate.distance, self.instance.capacities[candidate.vertex])
            weighted_candidates.append(
                WeightedCandidate(candidate.vertex, candidate.nearest_vertex, candidate.distance, score)
            )

        weighted_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return weighted_candidates

    def random_index(self, size: int, beta: float) -> int:
        position = int(math.log(random.random()) / math.log(1 - beta))
        return position % size if size else 0

    # ------------------------------------------------------------------
    # Deterministic constructions
    # ------------------------------------------------------------------
    def construct_greedy_solution(self) -> Solution:
        solution = self.initial_solution()
        candidate_list = self.build_candidate_list(solution)
        alpha = self.alpha if self.alpha >= 0 else random.random()
        while not solution.is_feasible():
            limit = candidate_list[0].distance - (alpha * candidate_list[-1].distance)
            best_index = max(
                range(len(candidate_list)),
                key=lambda index: (
                    candidate_list[index].distance >= limit,
                    self.instance.capacities[candidate_list[index].vertex],
                ),
            )
            candidate = candidate_list.pop(best_index)
            solution.add_vertex(candidate.vertex)
            if candidate.distance < solution.objective_value:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_candidate_list(solution, candidate_list, candidate.vertex)
        solution.evaluate_symmetry()
        return solution

    # ------------------------------------------------------------------
    # Biased randomized constructions
    # ------------------------------------------------------------------
    def construct_biased_solution(self) -> Tuple[Solution, List[Candidate]]:
        solution = self.initial_solution()
        candidate_list = self.build_candidate_list(solution)
        while not solution.is_feasible():
            position = self.random_index(len(candidate_list), self.beta)
            candidate = candidate_list.pop(position)
            solution.add_vertex(candidate.vertex)
            if candidate.distance < solution.objective_value:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_candidate_list(solution, candidate_list, candidate.vertex)
        solution.evaluate_symmetry()
        return solution, candidate_list

    def construct_biased_capacity_solution(self) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        self.apply_configured_weight(0.6, 0.9)
        candidate_list = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidate_list), self.beta)
            candidate = candidate_list.pop(position)
            solution.add_vertex(candidate.vertex)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objective_value:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        solution.evaluate_symmetry()
        return solution, candidate_list

    def construct_biased_fixed_weight_solution(self, weight: float) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        self.weight = weight
        candidate_list = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidate_list), self.beta)
            candidate = candidate_list.pop(position)
            solution.add_vertex(candidate.vertex)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objective_value:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        solution.evaluate_symmetry()
        return solution, candidate_list

    def construct_biased_distribution_solution(
        self, distribution: Dict[Tuple[float, float], float]
    ) -> Tuple[Solution, List[WeightedCandidate], Tuple[float, float]]:
        solution = self.initial_solution()

        random_value = random.random()
        cumulative = 0.0
        selected_interval = next(reversed(distribution))
        for interval, probability in distribution.items():
            cumulative += probability
            if random_value <= cumulative:
                selected_interval = interval
                break

        lower, upper = selected_interval
        if self.configured_weight is not None:
            self.weight = self.configured_weight
        else:
            self.weight = random.uniform(lower, upper if upper <= 1 else 1)
        candidate_list = self.build_weighted_candidate_list(solution)

        while not solution.is_feasible():
            position = self.random_index(len(candidate_list), self.beta)
            candidate = candidate_list.pop(position)
            solution.add_vertex(candidate.vertex)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objective_value:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        solution.evaluate_symmetry()
        return solution, candidate_list, selected_interval

    # ------------------------------------------------------------------
    # Candidate list maintenance
    # ------------------------------------------------------------------
    def update_candidate_list(self, solution: Solution, candidate_list: List[Candidate], last_vertex: int) -> None:
        for candidate in candidate_list:
            distance = self.instance.distances[last_vertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearest_vertex = last_vertex
        candidate_list.sort(key=lambda item: item.distance, reverse=True)

    def update_weighted_candidate_list(
        self, solution: Solution, candidate_list: List[WeightedCandidate], last_vertex: int
    ) -> None:
        for candidate in candidate_list:
            distance = self.instance.distances[last_vertex][candidate.vertex]
            if distance < candidate.distance:
                candidate.distance = distance
                candidate.nearest_vertex = last_vertex
        self.max_min_distance = max((candidate.distance for candidate in candidate_list), default=1.0)
        for candidate in candidate_list:
            candidate.score = self.weighted_score(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    def insert_candidate(self, candidate_list: List[Candidate], solution: Solution, vertex: int) -> None:
        nearest_vertex, distance = solution.distance_to(vertex)
        candidate_list.append(Candidate(vertex, nearest_vertex, distance))
        candidate_list.sort(key=lambda item: item.distance, reverse=True)

    def insert_weighted_candidate(
        self, candidate_list: List[WeightedCandidate], solution: Solution, vertex: int
    ) -> None:
        nearest_vertex, distance = solution.distance_to(vertex)
        self.max_capacity = max(self.max_capacity, self.instance.capacities[vertex])
        self.max_min_distance = max(self.max_min_distance, distance)
        score = self.weighted_score(distance, self.instance.capacities[vertex])
        candidate_list.append(WeightedCandidate(vertex, nearest_vertex, distance, score))
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    def recalculate_candidate_list(
        self, solution: Solution, candidate_list: List[Candidate], removed_vertex: int
    ) -> None:
        for candidate in candidate_list:
            if candidate.nearest_vertex == removed_vertex:
                candidate.nearest_vertex, candidate.distance = solution.distance_to(candidate.vertex)
        candidate_list.sort(key=lambda item: item.distance, reverse=True)

    def recalculate_weighted_candidate_list(
        self, solution: Solution, candidate_list: List[WeightedCandidate], removed_vertex: int
    ) -> None:
        for candidate in candidate_list:
            if candidate.nearest_vertex == removed_vertex:
                candidate.nearest_vertex, candidate.distance = solution.distance_to(candidate.vertex)

        selected_capacities = [self.instance.capacities[v] for v in solution.selected_vertices]
        self.max_capacity = (
            max(selected_capacities) if selected_capacities else max(self.instance.capacities)
        )
        self.max_min_distance = max((candidate.distance for candidate in candidate_list), default=1.0)

        for candidate in candidate_list:
            candidate.score = self.weighted_score(
                candidate.distance, self.instance.capacities[candidate.vertex]
            )
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    # ------------------------------------------------------------------
    # Partial reconstructions for tabu search
    # ------------------------------------------------------------------
    def partial_reconstruction(self, solution: Solution, candidate_list: List[Candidate]) -> Solution:
        while not solution.is_feasible():
            index = self.random_index(len(candidate_list), self.beta_local_search)
            candidate = candidate_list.pop(index)
            solution.add_vertex(candidate.vertex)
            self.update_candidate_list(solution, candidate_list, candidate.vertex)
        return solution

    def partial_reconstruction_capacity(
        self, solution: Solution, candidate_list: List[WeightedCandidate]
    ) -> Solution:
        while not solution.is_feasible():
            index = self.random_index(len(candidate_list), self.beta_local_search)
            candidate = candidate_list.pop(index)
            solution.add_vertex(candidate.vertex)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        return solution

