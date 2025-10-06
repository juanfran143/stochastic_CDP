"""Core data structures for the stochastic critical distance problem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Candidate:
    """Candidate node considered during construction or improvement steps."""

    vertex: int
    nearestVertex: int
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

    instanceName: str
    seed: int
    maxTime: int
    betaConstruction: float
    betaLocalSearch: float
    maxIterations: int
    reliabilityThreshold: float
    shortSimulationRuns: int
    longSimulationRuns: int
    variance: float
    deterministic: bool
    skipPenaltyCost: bool
    weight: float
    inverseRatio: float

