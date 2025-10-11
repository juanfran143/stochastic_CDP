"""Problem instance definition and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from objects import Edge


@dataclass(slots=True)
class Instance:
    """Encapsulates all the data required to evaluate a solution."""

    path: str
    colours: List[str] = field(default_factory=list)
    lambda_penalty: float = 0.0
    gamma_override: Optional[float] = None
    name: str = field(init=False)
    node_count: int = field(init=False, default=0)
    min_capacity: float = field(init=False, default=0)
    capacities: List[float] = field(init=False, default_factory=list)
    distances: List[List[float]] = field(init=False, default_factory=list)
    sorted_edges: List[Edge] = field(init=False, default_factory=list)
    _min_positive_distance: Optional[float] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = Path(self.path).name
        self.load_instance()

    def load_instance(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if len(lines) < 3:
            raise ValueError("Instance file is incomplete.")

        self.node_count = int(lines[0])
        self.min_capacity = float(lines[1])
        self.capacities = [float(value) for value in lines[2].split("\t")]
        if len(self.capacities) != self.node_count:
            raise ValueError("Capacity vector length does not match node count.")

        self.distances = [
            [0.0 for _ in range(self.node_count)] for _ in range(self.node_count)
        ]
        self.sorted_edges = []

        for row_index, raw_row in enumerate(lines[3:]):
            values = [float(value) for value in raw_row.split("\t")]
            if len(values) != self.node_count:
                raise ValueError("Distance matrix row length mismatch.")
            for column_index, distance in enumerate(values):
                if distance:
                    self.distances[row_index][column_index] = distance
                    self.sorted_edges.append(Edge(row_index, column_index, distance))

        self.sorted_edges.sort(key=lambda edge: edge.distance, reverse=True)

    # ------------------------------------------------------------------
    # Symmetry helpers
    # ------------------------------------------------------------------
    def assign_colours(self, colours: Sequence[str]) -> None:
        """Assign a colour label to every vertex in the instance."""

        if len(colours) != self.node_count:
            raise ValueError("Number of colours must match the number of vertices.")
        self.colours = list(colours)

    def set_symmetry_parameters(
        self, *, lambda_penalty: float, gamma_override: Optional[float] = None
    ) -> None:
        """Configure the symmetry penalty scaling parameters."""

        if lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative.")
        if gamma_override is not None and gamma_override < 0:
            raise ValueError("gamma_override must be non-negative when provided.")
        self.lambda_penalty = lambda_penalty
        self.gamma_override = gamma_override
        self._min_positive_distance = None

    def min_positive_distance(self) -> float:
        """Return the minimum strictly positive distance among vertices."""

        if self._min_positive_distance is None:
            positive = [edge.distance for edge in self.sorted_edges if edge.distance > 0]
            if not positive:
                raise ValueError("Instance must contain at least one positive distance.")
            self._min_positive_distance = min(positive)
        return self._min_positive_distance

    @property
    def gamma(self) -> float:
        """Scaling factor used by the linear symmetry penalty."""

        if self.gamma_override is not None:
            return self.gamma_override
        if self.lambda_penalty == 0:
            return 0.0
        return self.lambda_penalty * self.min_positive_distance()

    def _ordered_active_colours(self, selected_vertices: Sequence[int]) -> List[str]:
        """Return active colours ordered by the lowest-index vertex using them."""

        colour_first_index: Dict[str, int] = {}
        for vertex in selected_vertices:
            colour = self.colours[vertex]
            if colour not in colour_first_index or vertex < colour_first_index[colour]:
                colour_first_index[colour] = vertex
        ordered = [
            colour
            for colour, _ in sorted(colour_first_index.items(), key=lambda item: item[1])
        ]
        return ordered

    def colour_pair_coefficients(
        self, selected_vertices: Sequence[int]
    ) -> Dict[Tuple[str, str], float]:
        """Return the per-colour coefficients contributing to the penalty.

        Colours are made distinct by vertex index, so the earliest colour present
        in the selection is treated as the reference tone. Each additional active
        colour contributes ``gamma`` to the total penalty and is reported as a
        pair ``(reference_colour, new_colour)``.
        """

        if not selected_vertices or not self.colours:
            return {}
        ordered_colours = self._ordered_active_colours(selected_vertices)
        if len(ordered_colours) <= 1:
            return {}
        reference_colour = ordered_colours[0]
        coefficients: Dict[Tuple[str, str], float] = {}
        for colour in ordered_colours[1:]:
            coefficients[(reference_colour, colour)] = self.gamma
        return coefficients

    def symmetry_penalty(
        self,
        selected_vertices: Sequence[int],
        *,
        return_breakdown: bool = False,
    ) -> float | Tuple[float, Dict[Tuple[str, str], float]]:
        """Compute the linear colour penalty for a set of vertices.

        When ``return_breakdown`` is enabled the method also returns the
        per-colour coefficients explaining how the penalty was accumulated.
        """

        if not selected_vertices or not self.colours:
            return (0.0, {}) if return_breakdown else 0.0
        active_colours = {self.colours[vertex] for vertex in selected_vertices}
        if len(active_colours) <= 1:
            return (0.0, {}) if return_breakdown else 0.0
        penalty = self.gamma * (len(active_colours) - 1)
        if not return_breakdown:
            return penalty
        breakdown = self.colour_pair_coefficients(selected_vertices)
        return penalty, breakdown

