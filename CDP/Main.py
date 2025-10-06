"""Entry point for executing stochastic critical distance problem experiments."""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Instance import Instance
from LocalSearches import (
    tabuSearchCapacity,
    tabuSearchCapacitySimulation,
)
from Solution import Solution
from objects import TestCase, WeightedCandidate
from simheuristic import Simheuristic


@dataclass
class SummaryFile:
    path: Path
    header: str

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(self.header, encoding="utf-8")

    def append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def cloneWeightedCandidates(candidates: Iterable[WeightedCandidate]) -> List[WeightedCandidate]:
    return [WeightedCandidate(candidate.vertex, candidate.nearestVertex, candidate.distance, candidate.score) for candidate in candidates]


def loadTestCases(testName: str) -> List[TestCase]:
    filePath = Path("test") / f"{testName}.txt"
    testCases: List[TestCase] = []
    with filePath.open("r", encoding="utf-8") as handle:
        for rawLine in handle:
            line = rawLine.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split("\t")
            deterministicFlag = values[10].lower() == "true"
            skipPenalty = values[11].lower() == "true" if values[11] != "-" else False
            testCases.append(
                TestCase(
                    instanceName=values[0],
                    seed=int(values[1]),
                    maxTime=int(values[2]),
                    betaConstruction=float(values[3]),
                    betaLocalSearch=float(values[4]),
                    maxIterations=int(values[5]),
                    reliabilityThreshold=float(values[6]),
                    shortSimulationRuns=int(values[7]),
                    longSimulationRuns=int(values[8]),
                    variance=float(values[9]),
                    deterministic=deterministicFlag,
                    skipPenaltyCost=skipPenalty,
                    weight=float(values[12]),
                    inverseRatio=float(values[13]),
                )
            )
    return testCases


def writeDeterministicSummary(solution: Solution, testCase: TestCase, writer: SummaryFile) -> None:
    writer.append(
        "\t".join(
            [
                testCase.instanceName,
                f"{testCase.betaConstruction}",
                f"{testCase.seed}",
                f"{solution.objectiveValue}",
                f"{solution.time}",
                f"{solution.capacity}",
                f"{testCase.inverseRatio}",
                f"{testCase.weight}",
            ]
        )
    )


def writeStochasticSummary(
    solution: Solution,
    testCase: TestCase,
    writer: SummaryFile,
    simulationType: str,
    skipPenalty: bool,
) -> None:
    writer.append(
        "\t".join(
            [
                testCase.instanceName,
                f"{testCase.betaConstruction}",
                f"{solution.objectiveValue}",
                f"{solution.time}",
                f"{solution.capacity}",
                f"{solution.reliability[1 if skipPenalty else 2]}",
                f"{testCase.variance}",
                f"{solution.objectiveValue if skipPenalty else solution.meanStochasticObjective[2]}",
                f"{solution.stochasticCapacity[1 if skipPenalty else 2]}",
                f"{testCase.deterministic}",
                simulationType,
                f"{testCase.inverseRatio}",
                f"{testCase.weight}",
                f"{testCase.seed}",
            ]
        )
    )


def deterministicMultiStart(
    initial: Tuple[Solution, List[WeightedCandidate]],
    testCase: TestCase,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    initialSolution, initialCandidates = initial
    if 0.0 <= testCase.weight <= 1.0:
        weightSamples = {testCase.weight: 0.0}
    else:
        weightSamples = {step / 10: 0.0 for step in range(5, 11)}
    for _ in range(10):
        for weight in weightSamples:
            candidateSolution, _ = heuristic.constructBiasedFixedWeightSolution(weight)
            weightSamples[weight] += candidateSolution.objectiveValue

    rankedWeights = sorted(weightSamples.items(), key=lambda item: item[1], reverse=True)
    weightOptions = [rankedWeights[i][0] for i in range(min(3, len(rankedWeights)))]
    bestSolution = initialSolution.copy()
    bestCandidates = cloneWeightedCandidates(initialCandidates)

    start = time.process_time()
    while time.process_time() - start < testCase.maxTime:
        if 0.0 <= testCase.weight <= 1.0:
            sampledWeight = testCase.weight
        else:
            randomValue = random.random()
            if randomValue < 0.7:
                reference = weightOptions[0]
            elif randomValue < 0.9 and len(weightOptions) > 1:
                reference = weightOptions[1]
            else:
                reference = weightOptions[min(2, len(weightOptions) - 1)]

            lower = max(reference - 0.05, 0.0)
            upper = min(reference + 0.05, 1.0)
            sampledWeight = random.uniform(lower, upper)

        candidateSolution, candidateList = heuristic.constructBiasedFixedWeightSolution(sampledWeight)
        candidateSolution, candidateList = tabuSearchCapacity(
            candidateSolution, candidateList, testCase.maxIterations, heuristic
        )

        if candidateSolution.objectiveValue > bestSolution.objectiveValue:
            bestSolution = candidateSolution.copy()
            bestSolution.time = time.process_time() - start
            bestCandidates = cloneWeightedCandidates(candidateList)

    longSimulation = Simheuristic(testCase.longSimulationRuns, testCase.variance)
    longSimulation.runReliabilitySimulation(bestSolution)
    longSimulation.runStochasticEvaluation(bestSolution, bestCandidates)
    return bestSolution, bestCandidates


def stochasticMultiStart(
    initial: Tuple[Solution, List[WeightedCandidate]],
    testCase: TestCase,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    initialSolution, initialCandidates = initial
    smallSimulation = Simheuristic(testCase.shortSimulationRuns, testCase.variance)
    smallSimulation.runStochasticEvaluation(initialSolution, initialCandidates)

    elite: List[Tuple[Solution, List[WeightedCandidate]]] = [
        (initialSolution.copy(), cloneWeightedCandidates(initialCandidates))
    ]
    bestSolution = initialSolution.copy()
    bestCandidates = cloneWeightedCandidates(initialCandidates)

    start = time.process_time()
    while time.process_time() - start < testCase.maxTime:
        candidateSolution, candidateList = heuristic.constructBiasedCapacitySolution()
        candidateSolution, candidateList = tabuSearchCapacity(
            candidateSolution, candidateList, testCase.maxIterations, heuristic
        )

        if candidateSolution.objectiveValue > bestSolution.objectiveValue:
            smallSimulation.runStochasticEvaluation(candidateSolution, candidateList)
            if candidateSolution.meanStochasticObjective[2] >= bestSolution.meanStochasticObjective[2]:
                bestSolution = candidateSolution.copy()
                bestSolution.time = time.process_time() - start
                bestCandidates = cloneWeightedCandidates(candidateList)
                elite.append((bestSolution.copy(), cloneWeightedCandidates(candidateList)))

    longSimulation = Simheuristic(testCase.longSimulationRuns, testCase.variance)
    for solution, candidates in elite:
        longSimulation.runStochasticEvaluation(solution, candidates)

    bestSolution, bestCandidates = max(
        elite,
        key=lambda pair: pair[0].meanStochasticObjective[2],
    )
    return bestSolution, bestCandidates


def stochasticMultiStartSimulation(
    testCase: TestCase,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    exploratorySimulation = Simheuristic(20, testCase.variance)
    candidateSolution, candidateList = heuristic.constructBiasedCapacitySimulationSolution(
        exploratorySimulation, 0.9
    )
    candidateSolution, candidateList = tabuSearchCapacitySimulation(
        candidateSolution,
        candidateList,
        testCase.maxIterations,
        heuristic,
        exploratorySimulation,
        0.9,
    )

    smallSimulation = Simheuristic(testCase.shortSimulationRuns, testCase.variance)
    smallSimulation.runReliabilitySimulation(candidateSolution)

    elite: List[Tuple[Solution, List[WeightedCandidate]]] = [
        (candidateSolution.copy(), cloneWeightedCandidates(candidateList))
    ]
    backup: List[Tuple[Solution, List[WeightedCandidate]]] = []
    reliabilityTargetReached = candidateSolution.reliability[1] >= testCase.reliabilityThreshold

    start = time.process_time()
    while time.process_time() - start < testCase.maxTime:
        newSolution, newCandidates = heuristic.constructBiasedCapacitySimulationSolution(
            exploratorySimulation, 0.9
        )
        newSolution, newCandidates = tabuSearchCapacitySimulation(
            newSolution,
            newCandidates,
            testCase.maxIterations,
            heuristic,
            exploratorySimulation,
            0.9,
        )

        if newSolution.objectiveValue > candidateSolution.objectiveValue:
            smallSimulation.runReliabilitySimulation(newSolution)
            if newSolution.reliability[1] >= testCase.reliabilityThreshold:
                reliabilityTargetReached = True
                candidateSolution = newSolution.copy()
                candidateSolution.time = time.process_time() - start
                candidateList = cloneWeightedCandidates(newCandidates)
                elite.append((candidateSolution.copy(), cloneWeightedCandidates(newCandidates)))
            elif not reliabilityTargetReached:
                backup.append((newSolution.copy(), cloneWeightedCandidates(newCandidates)))

    longSimulation = Simheuristic(testCase.longSimulationRuns, testCase.variance)
    if not reliabilityTargetReached and backup:
        for solution, _ in backup:
            longSimulation.runReliabilitySimulation(solution)
        bestSolution, bestCandidates = max(backup, key=lambda pair: pair[0].reliability[1])
    else:
        for solution, _ in elite:
            longSimulation.runReliabilitySimulation(solution)
        bestSolution, bestCandidates = max(elite, key=lambda pair: pair[0].reliability[1])

    return bestSolution, bestCandidates


def executeTestCase(testCase: TestCase) -> Tuple[Solution, List[WeightedCandidate]]:
    instancePath = Path("CDP") / testCase.instanceName
    instance = Instance(str(instancePath))
    if testCase.inverseRatio:
        instance.minCapacity = sum(instance.capacities) * testCase.inverseRatio

    heuristic = ConstructiveHeuristic(0.0, testCase.betaConstruction, testCase.betaLocalSearch, instance, testCase.weight)

    if testCase.deterministic:
        solution, candidates = heuristic.constructBiasedCapacitySolution()
        solution, candidates = tabuSearchCapacity(solution, candidates, testCase.maxIterations, heuristic)
        return deterministicMultiStart((solution, candidates), testCase, heuristic)

    if testCase.skipPenaltyCost:
        return stochasticMultiStartSimulation(testCase, heuristic)

    solution, candidates = heuristic.constructBiasedCapacitySolution()
    solution, candidates = tabuSearchCapacity(solution, candidates, testCase.maxIterations, heuristic)
    return stochasticMultiStart((solution, candidates), testCase, heuristic)


def run(testCases: Iterable[TestCase]) -> List[Tuple[TestCase, Solution, List[WeightedCandidate]]]:
    results = []
    for testCase in testCases:
        random.seed(testCase.seed)
        solution, candidates = executeTestCase(testCase)
        results.append((testCase, solution, candidates))
    return results


def performSanityCheck(results: Iterable[Tuple[TestCase, Solution, List[WeightedCandidate]]]) -> None:
    for _, solution, _ in results:
        if not solution.isFeasible():
            raise RuntimeError("Generated solution violates the minimum capacity constraint.")


def main() -> None:
    tests = loadTestCases("run")
    results = run(tests)
    performSanityCheck(results)

    stochasticWriter = SummaryFile(
        Path("output") / "ResumeOutputs_paper_2.txt",
        "Instance\tbetaLS\tCostSol\ttime\tCapacity\treliability\tvariance\tstochastic_of\tstochastic_capacity\tdeterministic\ttype_simulation\tinversa\tweight\tseed\n",
    )
    deterministicWriter = SummaryFile(
        Path("output") / "ResumeOutputs_def_STOCHASTIC.txt",
        "Instance\tbetaLS\tseed\tCostSol\ttime\tCapacity\tinversa\tweight\n",
    )

    for testCase, solution, candidates in results:
        if testCase.deterministic:
            writeDeterministicSummary(solution, testCase, deterministicWriter)
            writeStochasticSummary(solution, testCase, stochasticWriter, "True", True)
            writeStochasticSummary(solution, testCase, stochasticWriter, "False", False)
        else:
            skipPenalty = testCase.skipPenaltyCost
            simulationLabel = "True" if skipPenalty else "False"
            writeStochasticSummary(solution, testCase, stochasticWriter, simulationLabel, skipPenalty)


if __name__ == "__main__":
    main()
    sys.exit(0)

