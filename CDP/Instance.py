"""Problem instance definition and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from objects import Edge


@dataclass(slots=True)
class Instance:
    """Encapsulates all the data required to evaluate a solution."""

    path: str
    name: str = field(init=False)
    node_count: int = field(init=False, default=0)
    min_capacity: float = field(init=False, default=0)
    capacities: List[float] = field(init=False, default_factory=list)
    distances: List[List[float]] = field(init=False, default_factory=list)
    sorted_edges: List[Edge] = field(init=False, default_factory=list)

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

