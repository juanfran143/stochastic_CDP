"""Entry point for executing epsilon-constraint critical distance problem experiments."""

from __future__ import annotations

import colorsys
import math
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
from objects import TestCase
from symmetry_integration import plot_epsilon_frontier


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
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Requested instance '{reference_path}' was not found."
            )
        return reference_path.name, reference_path

    search_roots = [Path("../Instances"), Path("."), Path("..")]
    for root in search_roots:
        candidate = (root / reference_path).resolve()
        if candidate.exists():
            return reference_path.name, candidate

    raise FileNotFoundError(
        "Requested instance was not found. "
        f"Searched in: {', '.join(str((root / reference_path).resolve()) for root in search_roots)}."
    )


def load_test_cases(test_name: str) -> List[TestCase]:
    file_path = Path("../test") / f"{test_name}.txt"
    cases: List[TestCase] = []
    with file_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split("\t")
            if len(values) not in {7, 8}:
                raise ValueError(
                    "Each test case line must contain seven or eight tab-separated values."
                )
            (
                instance_reference,
                seed,
                max_time,
                beta_c,
                beta_ls,
                max_iterations,
                weight,
                *epsilon_step,
            ) = values
            instance_name, instance_path = resolve_instance_path(instance_reference)
            cases.append(
                TestCase(
                    instance_name=instance_name,
                    instance_path=instance_path,
                    seed=int(seed),
                    max_time=int(max_time),
                    beta_construction=float(beta_c),
                    beta_local_search=float(beta_ls),
                    max_iterations=int(max_iterations),
                    weight=float(weight),
                    epsilon_step=int(float(epsilon_step[0])) if epsilon_step else DEFAULT_EPSILON_STEP,
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
        epsilon: int,
) -> Solution:
    initial_solution.reevaluate()
    best_solution = initial_solution.copy()
    best_solution.reevaluate()
    start = time.process_time()
    iteration_count = 0  # Add an iteration counter

    while time.process_time() - start < test_case.max_time:
        iteration_count += 1
        candidate_solution, candidate_list = heuristic.construct_biased_capacity_solution()
        candidate_solution.reevaluate()  # Removed alpha parameter
        candidate_solution, _ = tabu_search_capacity(
            candidate_solution,
            candidate_list,
            test_case.max_iterations,
            heuristic,
        )
        candidate_solution.reevaluate()  # Removed alpha parameter

        # Maximin Distance is the primary objective, maximize it
        if candidate_solution.objectiveValue > best_solution.objectiveValue + 1e-9:
            elapsed_time = time.process_time() - start
            print(
                f"  [IMPROVEMENT] New Best Found @ {elapsed_time:.2f}s (Iter {iteration_count}) "
                f"for Epsilon {epsilon}: Dispersion improved from {best_solution.objectiveValue:.3f} to {candidate_solution.objectiveValue:.3f}"
            )
            best_solution = candidate_solution.copy()
            best_solution.time = time.process_time() - start
            best_solution.reevaluate()  # Removed alpha parameter

    if best_solution.time == 0.0:
        best_solution.time = time.process_time() - start

    best_solution.reevaluate()  # Removed alpha parameter
    return best_solution


def execute_test_case(test_case: TestCase, epsilon: int) -> Solution:
    instance_path = test_case.instance_path
    instance = Instance(str(instance_path))
    instance.assign_colours(_generate_indexed_palette(instance.node_count))

    # Removed Instance.set_symmetry_parameters call (related to lambda/gamma)

    heuristic = ConstructiveHeuristic(
        epsilon,  # Pass epsilon
        test_case.beta_construction,
        test_case.beta_local_search,
        instance,
        test_case.weight,
    )

    solution, candidate_list = heuristic.construct_biased_capacity_solution()
    solution.reevaluate()

    initial_dispersion = solution.objectiveValue

    # Initial solution
    print(
        f"  [LS START] Epsilon {epsilon}: Initial Dispersion={initial_dispersion:.3f}, "
        f"Count={solution.get_colour_type_count()}"
    )

    solution, candidate_list = tabu_search_capacity(
        solution,
        candidate_list,
        test_case.max_iterations,
        heuristic,
    )
    solution.reevaluate()

    final_dispersion = solution.objectiveValue

    # Improvement solution
    improvement = final_dispersion - initial_dispersion
    print(
        f"  [LS END] Epsilon {epsilon}: Final Dispersion={final_dispersion:.3f} (Improvement: {improvement:.3f})"
    )

    return deterministic_multi_start(solution, test_case, heuristic, epsilon)


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


def run(test_cases: Iterable[TestCase]) -> List[Tuple[TestCase, Solution]]:
    results: List[Tuple[TestCase, Solution]] = []

    for test_case in test_cases:
        random.seed(test_case.seed)

        instance = Instance(str(test_case.instance_path))
        instance.assign_colours(_generate_indexed_palette(instance.node_count))

        max_epsilon = len(instance.unique_colours)
        epsilon_step = test_case.epsilon_step

        # === DEBUG CHECK ===
        print(f"[DEBUG] Instance Node Count: {instance.node_count}")
        print(f"[DEBUG] Calculated Max Epsilon (len(unique_colours)): {max_epsilon}")

        # Generate the list of epsilon values to execute.
        # It must include max_epsilon, decrease by step, and include 1.
        # The sorted(list(set(...))) handles redundancy/order.
        epsilon_values_to_run = sorted(
            list(set(range(1, max_epsilon + 1, epsilon_step)) | {1, max_epsilon}),
            reverse=True
        )

        # --- DEBUG CHECK 3: Verify Epsilon Sequence ---
        print(f"[DEBUG] Epsilon sequence to run: {epsilon_values_to_run}")

        pareto_history: List[Tuple[int, float, int]] = []
        base_solution: Solution | None = None

        # Iterate directly over the calculated sequence of integers
        for epsilon_to_execute in epsilon_values_to_run:

            print(
                f"\n[PROGRESS] Running Test Case: {test_case.instance_name}, Seed: {test_case.seed}, Epsilon: {epsilon_to_execute}",
                flush=True)

            # Execute experiment with the current epsilon constraint
            solution = execute_test_case(test_case, epsilon_to_execute)
            solution.reevaluate()

            # Record historical data
            pareto_history.append(
                (epsilon_to_execute, solution.objectiveValue, solution.get_colour_type_count())
            )

            if base_solution is None:
                base_solution = solution

            print(
                f"[RESULT] Epsilon {epsilon_to_execute} completed: "
                f"Dispersion={solution.objectiveValue:.3f}, "
                f"Colour Count={solution.get_colour_type_count()}"
            )

        if base_solution is None:
            raise RuntimeError("No feasible solution generated for the provided test case.")

        # Attach history to the result Solution object
        setattr(base_solution, 'pareto_history', pareto_history)

        results.append((test_case, base_solution))
    return results

def perform_sanity_check(results: Iterable[Tuple[TestCase, Solution]]) -> None:
    for test_case, solution in results:
        if solution.is_feasible():
            print(
                f"[sanity] {test_case.instance_name} (seed={test_case.seed}) -> feasible",
                flush=True,
            )
            continue

        print(
            f"[sanity] {test_case.instance_name} (seed={test_case.seed}) -> not feasible",
            flush=True,
        )
        raise RuntimeError("Generated solution violates the minimum capacity constraint.")


# Epsilon constraint method
def main() -> None:
    tests = load_test_cases("run")
    results = run(tests)
    perform_sanity_check(results)

    deterministic_writer = SummaryFile(
        Path("../output") / "deterministic_summary.txt",
        "Instance\tbeta_ls\tseed\tcost\ttime\tcapacity\tweight\n",
    )
    # New summary file to reflect epsilon-constraint output
    epsilon_writer = SummaryFile(
        Path("../output") / "epsilon_summary.txt",
        "Instance\tseed\tEpsilon_Used\tDispersion_Achieved\tColour_Count_Achieved\n",
    )

    for test_case, solution in results:
        write_deterministic_summary(solution, test_case, deterministic_writer)

        # Use compressed history data
        history_data = compress_epsilon_history(getattr(solution, 'pareto_history', []))

        if history_data:
            # Initial solution (usually the least constrained point, the first in history_data)
            initial_epsilon, initial_dispersion, initial_count = history_data[0]

            # Find the solution with the maximum Maximin Distance
            best_solution_data = max(
                history_data,
                key=lambda entry: entry[1],  # Maximize dispersion (objectiveValue)
            )
            best_epsilon, best_dispersion, best_count = best_solution_data

            frontier_size = len(history_data)
        else:
            initial_epsilon, initial_dispersion, initial_count = (
            0, solution.objectiveValue, solution.get_colour_type_count())
            best_epsilon, best_dispersion, best_count = initial_epsilon, initial_dispersion, initial_count
            frontier_size = 0

        # Output all points on the Pareto frontier to the file
        for epsilon, dispersion, count in history_data:
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

        print(f"Epsilon frontier for {test_case.instance_name} (seed={test_case.seed}):")

        print(
            "  Constraint-agnostic reference: "
            f"Epsilon_used={initial_epsilon}, "
            f"Dispersion={initial_dispersion:.3f}, "
            f"Colour_Count={initial_count}"
        )

        # Print all points on the Pareto frontier
        for epsilon, dispersion, count in history_data:
            print(
                "  "
                f"Epsilon={epsilon}: Dispersion={dispersion:.3f}, "
                f"Colour_Count={count}"
            )

        # Print the best solution found
        print(
            "  Best Dispersion (Across all Epsilon): "
            f"Epsilon_used={best_epsilon}, Dispersion={best_dispersion:.3f}, "
            f"Colour_Count={best_count}"
        )

        # Plotting logic
        plot_error: str | None = None
        pareto_plot_path: Path | None = None
        try:
            if history_data:
                plot_destination = Path("../output") / f"{test_case.instance_name}_epsilon_frontier.png"
                # Call the new plotting function
                pareto_plot_path = plot_epsilon_frontier(history_data, plot_destination)
        except RuntimeError as error:
            plot_error = str(error)

        if pareto_plot_path is not None:
            print(f"  Plot saved to: {pareto_plot_path}")
        elif plot_error:
            print(f"  Plot was not generated: {plot_error}")



DEFAULT_EPSILON_STEP = 1
if __name__ == "__main__":
    main()
    sys.exit(0)