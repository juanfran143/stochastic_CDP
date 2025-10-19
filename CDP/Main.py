"""Entry point for executing deterministic critical distance problem experiments."""

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
from symmetry_integration import plot_pareto_history


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
                *alpha_step,
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
                    alpha_step=float(alpha_step[0]) if alpha_step else DEFAULT_ALPHA_STEP,
                )
            )
    return cases


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
    initial_solution: Solution,
    test_case: TestCase,
    heuristic: ConstructiveHeuristic,
    alpha: float,
) -> Solution:
    initial_solution.reevaluate(alpha)
    best_solution = initial_solution.copy()
    best_solution.reevaluate(alpha)

    start = time.process_time()
    while time.process_time() - start < test_case.max_time:
        candidate_solution, candidate_list = heuristic.construct_biased_capacity_solution()
        candidate_solution.reevaluate(alpha)
        candidate_solution, _ = tabu_search_capacity(
            candidate_solution,
            candidate_list,
            test_case.max_iterations,
            heuristic,
        )
        candidate_solution.reevaluate(alpha)

        if candidate_solution.objective_value > best_solution.objective_value + 1e-9:
            best_solution = candidate_solution.copy()
            best_solution.time = time.process_time() - start
            best_solution.reevaluate(alpha)

    if best_solution.time == 0.0:
        best_solution.time = time.process_time() - start

    best_solution.reevaluate(alpha)
    return best_solution


def execute_test_case(test_case: TestCase, alpha: float) -> Solution:
    instance_path = test_case.instance_path
    instance = Instance(str(instance_path))
    instance.assign_colours(_generate_indexed_palette(instance.node_count))
    instance.set_symmetry_parameters(
        lambda_penalty=DEFAULT_LAMBDA_PENALTY,
        gamma_override=DEFAULT_GAMMA_OVERRIDE,
    )
    heuristic = ConstructiveHeuristic(
        alpha,
        test_case.beta_construction,
        test_case.beta_local_search,
        instance,
        test_case.weight,
    )

    solution, candidate_list = heuristic.construct_biased_capacity_solution()
    solution.reevaluate(alpha)
    solution, candidate_list = tabu_search_capacity(
        solution,
        candidate_list,
        test_case.max_iterations,
        heuristic,
    )
    solution.reevaluate(alpha)
    return deterministic_multi_start(solution, test_case, heuristic, alpha)


def compress_pareto_history(
    history: Iterable[Tuple[float, float, float]],
    *,
    tolerance: float = 1e-9,
) -> List[Tuple[float, float, float]]:
    """Return the Pareto history without consecutive duplicate entries."""

    compressed: List[Tuple[float, float, float]] = []
    previous: Tuple[float, float, float] | None = None
    for alpha, dispersion, penalty in history:
        if previous is not None:
            _, prev_dispersion, prev_penalty = previous
            if math.isclose(dispersion, prev_dispersion, abs_tol=tolerance) and math.isclose(
                penalty, prev_penalty, abs_tol=tolerance
            ):
                previous = (alpha, dispersion, penalty)
                continue
        entry = (alpha, dispersion, penalty)
        compressed.append(entry)
        previous = entry
    return compressed


def run(test_cases: Iterable[TestCase]) -> List[Tuple[TestCase, Solution]]:
    results: List[Tuple[TestCase, Solution]] = []
    for test_case in test_cases:
        random.seed(test_case.seed)
        alpha_values: List[float] = []
        pareto_history: List[Tuple[float, float, float]] = []
        base_solution: Solution | None = None

        current_alpha = 1.0
        while True:
            solution = execute_test_case(test_case, current_alpha)
            solution.reevaluate(current_alpha)
            alpha_values.append(current_alpha)
            pareto_history.append(
                (current_alpha, solution.cdp_objective, solution.symmetry_penalty)
            )
            if base_solution is None:
                base_solution = solution
            current_alpha = max(current_alpha - test_case.alpha_step, 0.0)
            if math.isclose(alpha_values[-1], 0.0, abs_tol=1e-9):
                break

        if base_solution is None:
            raise RuntimeError("No feasible solution generated for the provided test case.")

        base_solution.alpha_history = alpha_values
        base_solution.pareto_history = pareto_history
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
    symmetry_writer = SummaryFile(
        Path("../output") / "symmetry_summary.txt",
        "Instance\tseed\tbase_dispersion\tbase_penalty\tbest_dispersion\tbest_penalty\tfront_size\n",
    )

    for test_case, solution in results:
        write_deterministic_summary(solution, test_case, deterministic_writer)
        history_data = compress_pareto_history(solution.pareto_history)
        if history_data:
            base_dispersion = history_data[0][1]
            base_penalty = history_data[0][2]
            best_alpha, best_dispersion, best_penalty = min(
                history_data,
                key=lambda entry: (
                    entry[2],
                    -entry[1] if math.isfinite(entry[1]) else 0.0,
                ),
            )
        else:
            base_dispersion = solution.cdp_objective
            base_penalty = solution.symmetry_penalty
            best_alpha = solution.objective_alpha
            best_dispersion = base_dispersion
            best_penalty = base_penalty

        frontier_size = max(len(history_data) - 1, 0)
        pareto_plot_path: Path | None = None
        plot_error: str | None = None
        try:
            if history_data:
                plot_destination = Path("../output") / f"{test_case.instance_name}_pareto.png"
                pareto_plot_path = plot_pareto_history(history_data, plot_destination)
        except RuntimeError as error:
            plot_error = str(error)
        symmetry_writer.append(
            "\t".join(
                [
                    test_case.instance_name,
                    f"{test_case.seed}",
                    f"{base_dispersion}",
                    f"{base_penalty}",
                    f"{best_dispersion}",
                    f"{best_penalty}",
                    f"{frontier_size}",
                ]
            )
        )

        print(f"Pareto frontier for {test_case.instance_name} (seed={test_case.seed}):")
        print(
            "  Symmetry-agnostic heuristic reference: "
            f"cdp_objective={base_dispersion:.3f}, "
            f"symmetry_penalty={base_penalty:.3f}"
        )
        for alpha, dispersion, penalty in history_data:
            print(
                "  "
                f"α={alpha:.2f}: cdp_objective={dispersion:.3f}, "
                f"symmetry_penalty={penalty:.3f}"
            )
        print(
            "  Best α combination: "
            f"α={best_alpha:.2f}, cdp_objective={best_dispersion:.3f}, "
            f"symmetry_penalty={best_penalty:.3f}"
        )
        if frontier_size == 0:
            print("  No additional improvements were found on the frontier.")
        if pareto_plot_path is not None:
            print(f"  Plot saved to: {pareto_plot_path}")
        elif plot_error:
            print(f"  Plot was not generated: {plot_error}")


DEFAULT_LAMBDA_PENALTY = 0.1
DEFAULT_GAMMA_OVERRIDE: float | None = None
DEFAULT_ALPHA_STEP = 0.05
if __name__ == "__main__":
    main()
    sys.exit(0)
