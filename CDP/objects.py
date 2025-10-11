"""Core data structures for the stochastic critical distance problem."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Candidate:
    """Candidate node considered during construction or improvement steps."""

    vertex: int
    nearest_vertex: int
    distance: float


@dataclass(slots=True)
class WeightedCandidate(Candidate):
    """Candidate enriched with a weighted score used in biased selections."""

    score: float


@dataclass(slots=True)
class Edge:
    """Edge represented by the pair of vertices and the associated distance."""

    vertex1: int
    vertex2: int
    distance: float


@dataclass(slots=True)
class TestCase:
    """Execution parameters loaded from the benchmark configuration file."""

    instance_name: str
    seed: int
    max_time: int
    beta_construction: float
    beta_local_search: float
    max_iterations: int
    weight: float

