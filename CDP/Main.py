"""Entry point for executing deterministic critical distance problem experiments."""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Instance import Instance
from LocalSearches import tabu_search_capacity
from Solution import Solution
from objects import TestCase, WeightedCandidate


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


def clone_weighted_candidates(candidates: Iterable[WeightedCandidate]) -> List[WeightedCandidate]:
    return [
        WeightedCandidate(candidate.vertex, candidate.nearest_vertex, candidate.distance, candidate.score)
        for candidate in candidates
    ]


def load_test_cases(test_name: str) -> List[TestCase]:
    file_path = Path("test") / f"{test_name}.txt"
    test_cases: List[TestCase] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split("\t")
            if len(values) != 7:
                raise ValueError(
                    "Each test case line must contain exactly seven tab-separated values."
                )
            test_cases.append(
                TestCase(
                    instance_name=values[0],
                    seed=int(values[1]),
                    max_time=int(values[2]),
                    beta_construction=float(values[3]),
                    beta_local_search=float(values[4]),
                    max_iterations=int(values[5]),
                    weight=float(values[6]),
                )
            )
    return test_cases


def write_deterministic_summary(solution: Solution, test_case: TestCase, writer: SummaryFile) -> None:
    writer.append(
        "\t".join(
            [
                test_case.instance_name,
                f"{test_case.beta_local_search}",
                f"{test_case.seed}",
                f"{solution.objective_value}",
                f"{solution.time}",
                f"{solution.capacity}",
                f"{test_case.weight}",
            ]
        )
    )


def deterministic_multi_start(
    initial: Tuple[Solution, List[WeightedCandidate]],
    test_case: TestCase,
    heuristic: ConstructiveHeuristic,
) -> Tuple[Solution, List[WeightedCandidate]]:
    initial_solution, initial_candidates = initial
    if 0.0 <= test_case.weight <= 1.0:
        weight_samples = {test_case.weight: 0.0}
    else:
        weight_samples = {step / 10: 0.0 for step in range(5, 11)}

    for _ in range(10):
        for weight in weight_samples:
            candidate_solution, _ = heuristic.construct_biased_fixed_weight_solution(weight)
            weight_samples[weight] += candidate_solution.objective_value

    ranked_weights = sorted(weight_samples.items(), key=lambda item: item[1], reverse=True)
    weight_options = [ranked_weights[i][0] for i in range(min(3, len(ranked_weights)))]
    best_solution = initial_solution.copy()
    best_candidates = clone_weighted_candidates(initial_candidates)

    start = time.process_time()
    while time.process_time() - start < test_case.max_time:
        if 0.0 <= test_case.weight <= 1.0:
            sampled_weight = test_case.weight
        else:
            random_value = random.random()
            if random_value < 0.7:
                reference = weight_options[0]
            elif random_value < 0.9 and len(weight_options) > 1:
                reference = weight_options[1]
            else:
                reference = weight_options[min(2, len(weight_options) - 1)]

            lower = max(reference - 0.05, 0.0)
            upper = min(reference + 0.05, 1.0)
            sampled_weight = random.uniform(lower, upper)

        candidate_solution, candidate_list = heuristic.construct_biased_fixed_weight_solution(sampled_weight)
        candidate_solution, candidate_list = tabu_search_capacity(
            candidate_solution,
            candidate_list,
            test_case.max_iterations,
            heuristic,
        )

        if candidate_solution.objective_value > best_solution.objective_value:
            best_solution = candidate_solution.copy()
            best_solution.time = time.process_time() - start
            best_candidates = clone_weighted_candidates(candidate_list)

    if best_solution.time == 0.0:
        best_solution.time = time.process_time() - start

    return best_solution, best_candidates


def execute_test_case(test_case: TestCase) -> Tuple[Solution, List[WeightedCandidate]]:
    instance_path = Path("CDP") / test_case.instance_name
    instance = Instance(str(instance_path))
    heuristic = ConstructiveHeuristic(
        0.0,
        test_case.beta_construction,
        test_case.beta_local_search,
        instance,
        test_case.weight,
    )

    solution, candidates = heuristic.construct_biased_capacity_solution()
    solution, candidates = tabu_search_capacity(
        solution,
        candidates,
        test_case.max_iterations,
        heuristic,
    )
    return deterministic_multi_start((solution, candidates), test_case, heuristic)


def run(test_cases: Iterable[TestCase]) -> List[Tuple[TestCase, Solution, List[WeightedCandidate]]]:
    results = []
    for test_case in test_cases:
        random.seed(test_case.seed)
        solution, candidates = execute_test_case(test_case)
        results.append((test_case, solution, candidates))
    return results


def perform_sanity_check(results: Iterable[Tuple[TestCase, Solution, List[WeightedCandidate]]]) -> None:
    for _, solution, _ in results:
        if not solution.is_feasible():
            raise RuntimeError("Generated solution violates the minimum capacity constraint.")


def main() -> None:
    tests = load_test_cases("run")
    results = run(tests)
    perform_sanity_check(results)

    deterministic_writer = SummaryFile(
        Path("output") / "deterministic_summary.txt",
        "Instance\tbeta_ls\tseed\tcost\ttime\tcapacity\tweight\n",
    )

    for test_case, solution, _ in results:
        write_deterministic_summary(solution, test_case, deterministic_writer)


if __name__ == "__main__":
    main()
    sys.exit(0)
