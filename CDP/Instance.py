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
    unique_colours: List[str] = field(init=False, default_factory=list)
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

        self.node_count = int(lines[0])
        self.min_capacity = float(lines[1])
        self.capacities = [float(value) for value in lines[2].split("\t")]

        self.distances = [
            [0.0 for _ in range(self.node_count)] for _ in range(self.node_count)
        ]
        self.sorted_edges = []

        for row_index, raw_row in enumerate(lines[3:]):
            values = [float(value) for value in raw_row.split("\t")]
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

        palette = list(colours)
        if len(palette) < self.node_count and palette:
            repeats = (self.node_count - len(palette) + len(palette) - 1) // len(palette)
            palette.extend(palette * repeats)
        self.colours = palette[: self.node_count]
        self.unique_colours = sorted(list(set(self.colours)))






