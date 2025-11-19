"""Constructive heuristics used by the multi-start framework."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from Instance import Instance
from Solution import Solution
from objects import Candidate, Edge, WeightedCandidate


class ConstructiveHeuristic:
    """Generates initial solutions and supports local search adjustments."""

    def __init__(
        self,
        epsilon: int,
        beta_construction: float,
        beta_local_search: float,
        instance: Instance,
        weight: float,
    ) -> None:
        colour_pool = len(instance.unique_colours) or len(getattr(instance, "colours", []))
        self.max_colour_types = max(colour_pool, 1)
        self.epsilon = max(1, min(epsilon, self.max_colour_types))
        self.beta = beta_construction
        self.beta_local_search = beta_local_search
        self.instance = instance
        self.weight = min(max(weight, 0.0), 1.0)
        self.first_edge_index = 0
        self.max_min_distance = 1.0
        self.max_capacity = max(instance.capacities)
        self.alpha = self._compute_alpha()

    def _compute_alpha(self) -> float:
        if self.max_colour_types <= 1:
            return 0.0
        unused_capacity = self.max_colour_types - self.epsilon
        return unused_capacity / self.max_colour_types

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def initial_solution(self) -> Solution:
        solution = Solution(self.instance)
        edge = self.select_initial_edge()
        solution.add_vertex(edge.vertex1)
        solution.add_vertex(edge.vertex2)
        solution.update_objective(edge.vertex1, edge.vertex2, edge.distance)
        self.max_capacity = max(
            self.instance.capacities[edge.vertex1],
            self.instance.capacities[edge.vertex2],
        )
        self.max_min_distance = edge.distance
        return solution

    def select_initial_edge(self) -> Edge:
        """Choose the starting edge according to the current value and the constraint of Epsilon"""

        # When no colours are provided we simply return the farthest edge as before.
        if not getattr(self.instance, "colours", None):
            return (
                self.instance.sorted_edges[self.first_edge_index]
                if self.instance.sorted_edges
                else Edge(0, 0, 0.0)
            )

        max_distance = (
            self.instance.sorted_edges[0].distance if self.instance.sorted_edges else 1.0
        ) or 1.0
        best_score = -math.inf
        best_tiebreaker = -math.inf
        best_index = self.first_edge_index


        # --- 1. Filter Edges based on Epsilon Constraint ---

        # In the epsilon-constraint method, the f2 constraint must be satisfied.
        # Since the initial solution S has two vertices, f2(S) is either 1 (same colour) or 2 (different colours).
        indexed_candidate_edges: List[Tuple[int, Edge]] = []

        if self.epsilon == 1:
            for index, edge in enumerate(self.instance.sorted_edges):
                if self.instance.colours[edge.vertex1] == self.instance.colours[edge.vertex2]:
                    indexed_candidate_edges.append((index, edge))

        else:
            indexed_candidate_edges = list(enumerate(self.instance.sorted_edges))

        if not indexed_candidate_edges and self.instance.sorted_edges:
            indexed_candidate_edges = list(enumerate(self.instance.sorted_edges))

        # --- 2. Build Restricted Candidate List (RCL) based on f1 Score (Distance) ---
        best_original_index = indexed_candidate_edges[0][0] if indexed_candidate_edges else self.first_edge_index

        for original_index, edge in indexed_candidate_edges:
            distance_score = edge.distance / max_distance if max_distance else 0.0
            combined_score = distance_score  # combined_score is distance_score

            if (
                    combined_score > best_score
                    or (
                    math.isclose(combined_score, best_score)
                    and distance_score > best_tiebreaker)
            ):
                best_score = combined_score
                best_tiebreaker = distance_score
                best_original_index = original_index

        self.first_edge_index = best_original_index
        return (
            self.instance.sorted_edges[self.first_edge_index]
            if self.instance.sorted_edges
            else Edge(0, 0, 0.0)
        )


    # def colour_penalty(self, solution: Solution, vertex: int) -> int:
    #     if not self.instance.colours or not solution.selectedVertices:
    #         return 0
    #     vertex_colour = self.instance.colours[vertex]
    #     selected_colours = {self.instance.colours[selected] for selected in solution.selectedVertices}
    #     return 0 if vertex_colour in selected_colours else 1


    # remove alpha
    def weighted_score(self, distance: float, capacity: float, penalty: float = 0.0) -> float:
        distance_component = distance / self.max_min_distance if self.max_min_distance else 0.0
        capacity_component = capacity / self.max_capacity if self.max_capacity else 0.0
        base_score = distance_component * self.weight + capacity_component * (1 - self.weight)
        return base_score - penalty

    def colour_penalty(self, solution: Solution, vertex: int) -> float:
        if not getattr(self.instance, "colours", None):
            return 0.0
        colour = self.instance.colours[vertex]
        if colour in solution.colourCounts:
            return 0.0
        return self.alpha

    def build_candidate_list(self, solution: Solution) -> List[Candidate]:
        candidates: List[Candidate] = []
        for vertex in range(self.instance.node_count):
            if vertex in solution.selectedVertices:
                continue
            nearest_vertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearest_vertex, distance))
        candidates.sort(key=lambda item: item.distance, reverse=True)
        return candidates

    def build_weighted_candidate_list(self, solution: Solution) -> List[WeightedCandidate]:
        weighted_candidates: List[WeightedCandidate] = []
        selected_capacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.max_capacity = (
            max(selected_capacities) if selected_capacities else max(self.instance.capacities)
        )

        candidates: List[Candidate] = []
        for vertex in range(self.instance.node_count):
            if vertex in solution.selectedVertices:
                continue
            nearest_vertex, distance = solution.distance_to(vertex)
            candidates.append(Candidate(vertex, nearest_vertex, distance))

        self.max_min_distance = max((candidate.distance for candidate in candidates), default=1.0)

        for candidate in candidates:
            penalty = self.colour_penalty(solution, candidate.vertex)
            score = self.weighted_score(
                candidate.distance,
                self.instance.capacities[candidate.vertex],
                penalty,
            )
            weighted_candidates.append(
                WeightedCandidate(candidate.vertex, candidate.nearest_vertex, candidate.distance, score)
            )

        weighted_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return weighted_candidates

    def random_index(self, size: int, beta: float) -> int:
        position = int(math.log(random.random()) / math.log(1 - beta))
        return position % size if size else 0

    def construct_biased_capacity_solution(self) -> Tuple[Solution, List[WeightedCandidate]]:
        solution = self.initial_solution()
        candidate_list = self.build_weighted_candidate_list(solution)
        feasible = True
        while not solution.is_feasible():
            if len(candidate_list) == 0:
                feasible = False
                break
            position = self.random_index(len(candidate_list), self.beta)
            candidate = candidate_list.pop(position)
            vertex_to_add = candidate.vertex

            # New: Epsilon constraint check
            current_count = solution.get_colour_type_count()
            vertex_colour = self.instance.colours[vertex_to_add]
            colour_exists = vertex_colour in solution.colourCounts

            if not colour_exists:
                new_colour_count = current_count + 1
            else:
                new_colour_count = current_count
                
            if new_colour_count > self.epsilon:
                continue

            solution.add_vertex(vertex_to_add)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(candidate.vertex, candidate.nearest_vertex, candidate.distance)
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        return solution, candidate_list, feasible



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
                candidate.distance,
                self.instance.capacities[candidate.vertex]
            )
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    def insert_weighted_candidate(
        self, candidate_list: List[WeightedCandidate], solution: Solution, vertex: int
    ) -> None:
        nearest_vertex, distance = solution.distance_to(vertex)
        self.max_capacity = max(self.max_capacity, self.instance.capacities[vertex])
        self.max_min_distance = max(self.max_min_distance, distance)
        penalty = self.colour_penalty(solution, vertex)
        score = self.weighted_score(
            distance,
            self.instance.capacities[vertex],
            penalty,
        )
        candidate_list.append(WeightedCandidate(vertex, nearest_vertex, distance, score))
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    def recalculate_weighted_candidate_list(
        self, solution: Solution, candidate_list: List[WeightedCandidate], removed_vertex: int
    ) -> None:
        for candidate in candidate_list:
            if candidate.nearest_vertex == removed_vertex:
                candidate.nearest_vertex, candidate.distance = solution.distance_to(candidate.vertex)

        selected_capacities = [self.instance.capacities[v] for v in solution.selectedVertices]
        self.max_capacity = (
            max(selected_capacities) if selected_capacities else max(self.instance.capacities)
        )
        self.max_min_distance = max((candidate.distance for candidate in candidate_list), default=1.0)

        for candidate in candidate_list:
            penalty = self.colour_penalty(solution, candidate.vertex)
            candidate.score = self.weighted_score(
                candidate.distance,
                self.instance.capacities[candidate.vertex],
                penalty,
            )
        candidate_list.sort(key=lambda item: item.score, reverse=True)

    # ------------------------------------------------------------------
    # Partial reconstructions for tabu search
    # ------------------------------------------------------------------
    def partial_reconstruction_capacity(
        self, solution: Solution, candidate_list: List[WeightedCandidate]
    ) -> Solution:
        while not solution.is_feasible():
            if len(candidate_list) == 0:
                break

            index = self.random_index(len(candidate_list), self.beta_local_search)
            candidate = candidate_list.pop(index)
            vertex_to_add = candidate.vertex
            
            # Epsilon Constraint Check
            current_count = solution.get_colour_type_count()
            vertex_colour = self.instance.colours[vertex_to_add]
            colour_exists = vertex_colour in solution.colourCounts

            if not colour_exists:
                new_colour_count = current_count + 1
            else:
                new_colour_count = current_count

            if new_colour_count > self.epsilon:
                continue

            solution.add_vertex(vertex_to_add)
            self.max_capacity = max(self.max_capacity, self.instance.capacities[candidate.vertex])
            if candidate.distance < solution.objectiveValue:
                solution.update_objective(
                    candidate.vertex, candidate.nearest_vertex, candidate.distance
                )
            self.update_weighted_candidate_list(solution, candidate_list, candidate.vertex)
        return solution

