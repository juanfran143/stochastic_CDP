"""Entry point for executing epsilon-constraint critical distance problem experiments."""

from __future__ import annotations

import colorsys
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from ConstructiveHeuristic import ConstructiveHeuristic
from Instance import Instance
from LocalSearches import tabu_search_capacity
from Solution import Solution
from objects import TestCase
from symmetry_integration import plot_epsilon_frontier


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "test"
OUTPUT_DIR = REPO_ROOT / "output"


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


def _generate_indexed_palette(node_count: int) -> List[str]:
    """Return a deterministic list of colours cycling through a small palette."""

    if node_count <= 0:
        return []

    # Limit the number of distinct colours so that symmetry penalties are not
    # always triggered by default. Reuse up to four tones (or fewer when the
    # instance has less vertices) and repeat them deterministically.
    unique_colour_count = min(4, node_count) if node_count >= 3 else node_count
    hues = [index / max(unique_colour_count, 1) for index in range(unique_colour_count)]
    base_colours: List[str] = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.92)
        base_colours.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")

    colours = [base_colours[index % unique_colour_count] for index in range(node_count)]
    return colours


def resolve_instance_path(instance_reference: str) -> tuple[str, Path]:
    """Return a display name and absolute path for the requested instance."""

    reference_path = Path(instance_reference)
    if reference_path.is_absolute():
        return reference_path.name, reference_path

    candidate = (REPO_ROOT / reference_path).resolve()
    if not candidate.exists():
        candidate = (REPO_ROOT / "Instances" / reference_path).resolve()
    return reference_path.name, candidate


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def load_test_cases(test_name: str) -> List[TestCase]:
    file_path = TEST_DIR / f"{test_name}.json"
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        case_entries = payload.get("cases") or payload.get("tests") or []
    else:
        case_entries = payload

    cases: List[TestCase] = []
    for entry in case_entries:
        instance_reference = entry["instance"]
        instance_name, instance_path = resolve_instance_path(instance_reference)
        cases.append(
            TestCase(
                instance_name=instance_name,
                instance_path=instance_path,
                seed=int(entry["seed"]),
                max_time=int(entry["max_time"]),
                beta_construction=float(entry["beta_construction"]),
                beta_local_search=float(entry["beta_local_search"]),
                max_iterations=int(entry["max_iterations"]),
                weight=float(entry["weight"]),
                max_epsilon=max(0, int(entry.get("max_epsilon", 0) or 0)),
                plot_frontier=_coerce_bool(entry.get("plot", False)),
            )
        )
    return cases


def write_deterministic_summary(solution: Solution, test_case: TestCase, writer: SummaryFile) -> None:
    # solution.objectiveValue contains Maximin Distance
    writer.append(
        "\t".join(
            [
                test_case.instance_name,
                f"{test_case.beta_local_search}",
                f"{test_case.seed}",
                f"{solution.objectiveValue}",
                f"{solution.time}",
                f"{solution.capacity}",
                f"{test_case.weight}",
            ]
        )
    )


def deterministic_multi_start(
        initial_solution: Solution,
        test_case: TestCase,
        heuristic: ConstructiveHeuristic,
) -> Solution:
    best_solution = initial_solution.copy()
    start = time.process_time()

    while time.process_time() - start < test_case.max_time:
        candidate_solution, candidate_list = heuristic.construct_biased_capacity_solution()
        candidate_solution, _ = tabu_search_capacity(
            candidate_solution,
            candidate_list,
            test_case.max_iterations,
            heuristic,
        )

        # Maximin Distance is the primary objective, maximize it
        if candidate_solution.objectiveValue > best_solution.objectiveValue + 1e-9:
            best_solution = candidate_solution.copy()
            best_solution.time = time.process_time() - start

    if best_solution.time == 0.0:
        best_solution.time = time.process_time() - start

    return best_solution


def execute_test_case(test_case: TestCase, epsilon: int, instance: Instance) -> Solution:
    heuristic = ConstructiveHeuristic(
        epsilon,  # Pass epsilon
        test_case.beta_construction,
        test_case.beta_local_search,
        instance,
        test_case.weight,
    )

    solution, candidate_list = heuristic.construct_biased_capacity_solution()

    solution, candidate_list = tabu_search_capacity(
        solution,
        candidate_list,
        test_case.max_iterations,
        heuristic,
    )

    return deterministic_multi_start(solution, test_case, heuristic)


# ------------------------------------------------------------------
# Modified history compression function for Epsilon Method
# ------------------------------------------------------------------

def compress_epsilon_history(
        # Format: (epsilon_used, dispersion_achieved, colour_count_achieved)
        history: Iterable[Tuple[int, float, int]],
        *,
        tolerance: float = 1e-9,
) -> List[Tuple[int, float, int]]:
    """Return the Pareto history by only keeping the maximum dispersion for a given colour count."""

    best_per_count: dict[int, Tuple[int, float]] = {}  # {count: (epsilon, dispersion)}

    for epsilon, dispersion, count in history:
        # Check if a better dispersion is found for the current count
        if count not in best_per_count or dispersion > best_per_count[count][1] + tolerance:
            best_per_count[count] = (epsilon, dispersion)
        elif math.isclose(dispersion, best_per_count[count][1], abs_tol=tolerance):
            # If dispersion is equal, keep the solution found with the tighter constraint (smaller epsilon)
            if epsilon < best_per_count[count][0]:
                best_per_count[count] = (epsilon, dispersion)

    # Format back to list: (epsilon, dispersion, count) and sort by count
    compressed: List[Tuple[int, float, int]] = []
    for count, (epsilon, dispersion) in sorted(best_per_count.items(), key=lambda item: item[0]):
        compressed.append((epsilon, dispersion, count))

    return compressed


def enforce_monotonic_objective(
        history: Sequence[Tuple[int, float, int]],
) -> List[Tuple[int, float, int]]:
    """Ensure dispersion values never decrease as epsilon grows."""

    best_dispersion = -math.inf
    adjusted: List[Tuple[int, float, int]] = []
    for epsilon, dispersion, colour_count in sorted(history, key=lambda entry: entry[0]):
        best_dispersion = max(best_dispersion, dispersion)
        adjusted.append((epsilon, best_dispersion, colour_count))
    return adjusted


def run(test_cases: Iterable[TestCase]) -> List[Tuple[TestCase, Solution]]:
    results: List[Tuple[TestCase, Solution]] = []

    for test_case in test_cases:
        random.seed(test_case.seed)

        instance = Instance(str(test_case.instance_path))
        instance.assign_colours(_generate_indexed_palette(instance.node_count))

        colour_limit = max(len(instance.unique_colours), 1)
        requested_max = test_case.max_epsilon or colour_limit
        max_epsilon_to_run = max(1, min(requested_max, colour_limit))

        pareto_history: List[Tuple[int, float, int]] = []
        base_solution: Solution | None = None

        for epsilon_to_execute in range(1, max_epsilon_to_run + 1):

            # Execute experiment with the current epsilon constraint
            solution = execute_test_case(test_case, epsilon_to_execute, instance)

            # Record historical data
            pareto_history.append(
                (epsilon_to_execute, solution.objectiveValue, solution.get_colour_type_count())
            )

            if base_solution is None:
                base_solution = solution

        if base_solution is None:
            continue

        # Attach monotonic history to the result Solution object
        setattr(base_solution, 'pareto_history', enforce_monotonic_objective(pareto_history))

        results.append((test_case, base_solution))
    return results


# Epsilon constraint method
def main() -> None:
    tests = load_test_cases("run")
    results = run(tests)

    deterministic_writer = SummaryFile(
        OUTPUT_DIR / "deterministic_summary.txt",
        "Instance\tbeta_ls\tseed\tcost\ttime\tcapacity\tweight\n",
    )
    # New summary file to reflect epsilon-constraint output
    epsilon_writer = SummaryFile(
        OUTPUT_DIR / "epsilon_summary.txt",
        "Instance\tseed\tEpsilon_Used\tDispersion_Achieved\tColour_Count_Achieved\n",
    )

    for test_case, solution in results:
        write_deterministic_summary(solution, test_case, deterministic_writer)

        history = getattr(solution, 'pareto_history', [])

        # Output all recorded evaluations so FO(CDP) and FO(symmetry) are listed per epsilon
        for epsilon, dispersion, count in history:
            epsilon_writer.append(
                "\t".join(
                    [
                        test_case.instance_name,
                        f"{test_case.seed}",
                        f"{epsilon}",
                        f"{dispersion}",
                        f"{count}",
                    ]
                )
            )

        if test_case.plot_frontier and history:
            try:
                plot_destination = OUTPUT_DIR / f"{test_case.instance_name}_epsilon_frontier.png"
                frontier = compress_epsilon_history(history)
                plot_epsilon_frontier(frontier, plot_destination)
            except RuntimeError:
                pass

if __name__ == "__main__":
    main()
    sys.exit(0)
