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
from objects import TestCase, WeightedCandidate
from symmetry_integration import analyse_solution, pareto_points_to_rows, plot_pareto_front


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


def load_colour_configuration(instance_path: Path, node_count: int) -> List[str]:
    """Load colour labels from disk or fall back to index-based colours."""
    a = Path(instance_path)
    colour_path = a.with_suffix(a.suffix + ".colors")
    if colour_path.exists():
        colours = [
            line.strip()
            for line in colour_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(colours) != node_count:
            raise ValueError(
                "Colour configuration must specify exactly one label per vertex."
            )
        return colours

    return _generate_indexed_palette(node_count)


def load_lambda_parameter(instance_path: Path) -> float:
    """Return the symmetry lambda penalty, using defaults when unspecified."""
    a = Path(instance_path)
    lambda_path = a.with_suffix(a.suffix + ".lambda")
    if lambda_path.exists():
        value = float(lambda_path.read_text(encoding="utf-8").strip())
        if value < 0:
            raise ValueError("Lambda penalty must be non-negative.")
        return value
    return 0.1


def load_gamma_override(instance_path: Path) -> float | None:
    """Load an optional gamma override value for the symmetry penalty."""
    a = Path(instance_path)
    gamma_path = a.with_suffix(a.suffix + ".gamma")
    if gamma_path.exists():
        value = float(gamma_path.read_text(encoding="utf-8").strip())
        if value < 0:
            raise ValueError("Gamma override must be non-negative.")
        return value
    return None


DEFAULT_ALPHA_STEP = 0.05


def resolve_instance_path(instance_reference: str) -> tuple[str, Path]:
    """Return a display name and absolute path for the requested instance."""

    reference_path = Path(instance_reference)
    if reference_path.is_absolute():
        if not reference_path.exists():
            raise FileNotFoundError(
                f"No se encontró la instancia solicitada en '{reference_path}'."
            )
        return reference_path.name, reference_path

    search_roots = [Path("../Instances"), Path("."), Path("..")]
    for root in search_roots:
        candidate = (root / reference_path).resolve()
        if candidate.exists():
            return reference_path.name, candidate

    raise FileNotFoundError(
        "No se encontró la instancia solicitada. "
        f"Se buscó en: {', '.join(str((root / reference_path).resolve()) for root in search_roots)}."
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

    best_solution.reevaluate()
    return best_solution, best_candidates


def execute_test_case(test_case: TestCase) -> Tuple[Solution, List[WeightedCandidate]]:
    instance_path = test_case.instance_path
    instance = Instance(str(instance_path))
    instance.assign_colours(load_colour_configuration(instance_path, instance.node_count))
    instance.set_symmetry_parameters(
        lambda_penalty=load_lambda_parameter(instance_path),
        gamma_override=load_gamma_override(instance_path),
    )
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
    solution.reevaluate()
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
        Path("../output") / "deterministic_summary.txt",
        "Instance\tbeta_ls\tseed\tcost\ttime\tcapacity\tweight\n",
    )
    symmetry_writer = SummaryFile(
        Path("../output") / "symmetry_summary.txt",
        "Instance\tseed\tbase_dispersion\tbase_penalty\tbest_dispersion\tbest_penalty\tfront_size\n",
    )

    for test_case, solution, _ in results:
        write_deterministic_summary(solution, test_case, deterministic_writer)
        analysis = analyse_solution(
            solution.instance,
            solution,
            steps=6,
            alpha_step=test_case.alpha_step,
        )
        base_dispersion = (
            analysis.base_solution.dispersion
            if math.isfinite(analysis.base_solution.dispersion)
            else 0.0
        )
        base_penalty = analysis.base_solution.symmetry_penalty
        pareto_plot_path: Path | None = None
        plot_error: str | None = None
        alpha_labels: list[str] | None = None
        if analysis.weighted_front:
            candidate_pool = [entry.candidate for entry in analysis.weighted_front]
            alpha_labels = [
                f"α=({entry.alpha_pair[0]:.2f}, {entry.alpha_pair[1]:.2f})"
                for entry in analysis.weighted_front
            ]
        else:
            candidate_pool = [analysis.base_solution, *analysis.epsilon_front]
        if candidate_pool:
            best_candidate = min(
                candidate_pool,
                key=lambda candidate: (
                    candidate.symmetry_penalty,
                    -(
                        candidate.dispersion
                        if math.isfinite(candidate.dispersion)
                        else 0.0
                    ),
                ),
            )
            best_dispersion = (
                best_candidate.dispersion if math.isfinite(best_candidate.dispersion) else 0.0
            )
            best_penalty = best_candidate.symmetry_penalty
        else:
            candidate_pool = [analysis.base_solution]
            best_dispersion = base_dispersion
            best_penalty = base_penalty
        try:
            pareto_plot_path = plot_pareto_front(
                candidate_pool[0],
                candidate_pool[1:],
                Path("../output") / f"{test_case.instance_name}_pareto.png",
                alpha_labels=alpha_labels,
            )
        except RuntimeError as error:
            pareto_plot_path = None
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
                    f"{max(len(candidate_pool) - 1, 0)}",
                ]
            )
        )

        print(f"Frontera de Pareto para {test_case.instance_name} (seed={test_case.seed}):")
        print(
            "  Referencia heurística sin simetría: "
            f"objetivo_CDP={base_dispersion:.3f}, "
            f"objetivo_simetría={base_penalty:.3f}"
        )
        rows = pareto_points_to_rows(candidate_pool)
        if alpha_labels is not None:
            labels = alpha_labels
        else:
            labels = [
                "Sin simetría",
                *[f"Iteración {index}" for index in range(1, len(rows))],
            ]
        for label, (penalty, dispersion) in zip(labels, rows):
            print(
                "  "
                f"{label}: objetivo_CDP={dispersion:.3f}, "
                f"objetivo_simetría={penalty:.3f}"
            )
        if len(rows) == 1:
            print(
                "  No se encontraron mejoras adicionales en la frontera con las "
                "combinaciones de α evaluadas."
            )
        if pareto_plot_path is not None:
            print(f"  Plot guardado en: {pareto_plot_path}")
        elif plot_error:
            print(f"  No se generó plot: {plot_error}")


if __name__ == "__main__":
    main()
    sys.exit(0)
