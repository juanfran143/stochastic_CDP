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
    nodeCount: int = field(init=False, default=0)
    minCapacity: float = field(init=False, default=0)
    capacities: List[float] = field(init=False, default_factory=list)
    distances: List[List[float]] = field(init=False, default_factory=list)
    sortedEdges: List[Edge] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.name = Path(self.path).name
        self.load_instance()

    def load_instance(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if len(lines) < 3:
            raise ValueError("Instance file is incomplete.")

        self.nodeCount = int(lines[0])
        self.minCapacity = float(lines[1])
        self.capacities = [float(value) for value in lines[2].split("\t")]
        if len(self.capacities) != self.nodeCount:
            raise ValueError("Capacity vector length does not match node count.")

        self.distances = [
            [0.0 for _ in range(self.nodeCount)] for _ in range(self.nodeCount)
        ]
        self.sortedEdges = []

        for rowIndex, rawRow in enumerate(lines[3:]):
            values = [float(value) for value in rawRow.split("\t")]
            if len(values) != self.nodeCount:
                raise ValueError("Distance matrix row length mismatch.")
            for columnIndex, distance in enumerate(values):
                if distance:
                    self.distances[rowIndex][columnIndex] = distance
                    self.sortedEdges.append(Edge(rowIndex, columnIndex, distance))

        self.sortedEdges.sort(key=lambda edge: edge.distance, reverse=True)

