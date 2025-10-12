from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = (ROOT / "Instances").resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CDP"))

from Instance import Instance
from Main import DEFAULT_ALPHA_STEP, load_test_cases
from Solution import Solution
from symmetry_integration import (
    MAX_ENUMERABLE_NODES,
    analyse_solution,
    default_alpha_schedule,
    generate_weighted_front,
    pareto_points_to_rows,
    plot_pareto_front,
    WeightedFrontEntry,
    _enforce_monotonic_weighted_front,
)
from cdp_symmetry import CandidateSolution

try:  # pragma: no cover - optional dependency for plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


class TestSymmetryIntegration(unittest.TestCase):
    """Validate the integration between stochastic CDP heuristics and symmetry tools."""

    @classmethod
    def setUpClass(cls) -> None:
        instance_path = ROOT / "CDP" / "sample_instance.txt"
        cls.instance = Instance(str(instance_path))
        cls.instance.assign_colours(["red", "red", "green"])
        cls.instance.set_symmetry_parameters(lambda_penalty=0.2)

    def test_solution_symmetry_penalty(self) -> None:
        solution = Solution(self.instance)
        for vertex in range(self.instance.node_count):
            solution.add_vertex(vertex)
        solution.reevaluate()
        expected_penalty = self.instance.gamma * (
            len({self.instance.colours[v] for v in solution.selected_vertices}) - 1
        )
        self.assertAlmostEqual(solution.symmetry_penalty, expected_penalty, places=6)
        self.assertEqual(
            solution.symmetry_breakdown,
            {("red", "green"): expected_penalty} if expected_penalty else {},
        )

    def test_colour_pair_coefficients(self) -> None:
        solution = Solution(self.instance)
        solution.add_vertex(0)
        solution.add_vertex(2)
        solution.evaluate_symmetry()
        self.assertEqual(solution.symmetry_breakdown, {("red", "green"): self.instance.gamma})

    def test_analyse_solution_returns_epsilon_front(self) -> None:
        solution = Solution(self.instance)
        solution.add_vertex(0)
        solution.add_vertex(2)
        solution.reevaluate()
        analysis = analyse_solution(self.instance, solution, steps=4)
        penalties = [candidate.symmetry_penalty for candidate in analysis.epsilon_front]
        self.assertTrue(penalties)
        self.assertIn(0.0, penalties)
        self.assertTrue(analysis.weighted_front)
        self.assertAlmostEqual(sum(analysis.weighted_front[0].alpha_pair), 1.0)

    def test_iterative_front_monotonic_penalty(self) -> None:
        solution = Solution(self.instance)
        solution.selected_vertices = [0, 1]
        solution.reevaluate()
        analysis = analyse_solution(self.instance, solution, steps=5)
        penalties = [
            analysis.base_solution.symmetry_penalty,
            *[candidate.symmetry_penalty for candidate in analysis.epsilon_front],
        ]
        for previous, current in zip(penalties, penalties[1:]):
            self.assertLessEqual(current + 1e-9, previous)

    def test_analyse_solution_honours_custom_alpha_step(self) -> None:
        solution = Solution(self.instance)
        solution.selected_vertices = [0, 2]
        solution.reevaluate()
        custom_step = 0.37
        analysis = analyse_solution(
            self.instance,
            solution,
            steps=3,
            alpha_step=custom_step,
        )
        expected_pairs = default_alpha_schedule(step=custom_step)
        produced_pairs = {entry.alpha_pair for entry in analysis.weighted_front}
        default_pairs = set(default_alpha_schedule())
        distinctive_pairs = {pair for pair in expected_pairs if pair not in default_pairs}
        if distinctive_pairs:
            self.assertTrue(
                produced_pairs & distinctive_pairs,
                "Weighted front should include at least one α pair derived from the custom step.",
            )
        else:
            self.assertTrue(produced_pairs)

    def test_large_instances_skip_exact_enumeration(self) -> None:
        node_count = MAX_ENUMERABLE_NODES + 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "large_instance.txt"
            capacities = "\t".join(["1.0"] * node_count)
            rows = []
            for row in range(node_count):
                values = []
                for col in range(node_count):
                    values.append("0.0" if row == col else "1.0")
                rows.append("\t".join(values))
            contents = "\n".join([
                str(node_count),
                "5.0",
                capacities,
                *rows,
            ])
            tmp_path.write_text(contents, encoding="utf-8")

            instance = Instance(str(tmp_path))
            colours = ["red" if index % 2 == 0 else "blue" for index in range(node_count)]
            instance.assign_colours(colours)
            instance.set_symmetry_parameters(lambda_penalty=0.2)

            solution = Solution(instance)
            for vertex in range(6):
                solution.add_vertex(vertex)
            solution.reevaluate()

            analysis = analyse_solution(instance, solution, steps=4)

        self.assertFalse(
            analysis.epsilon_front,
            "ε-constraint enumeration should be skipped for large instances.",
        )
        self.assertFalse(
            analysis.weighted_front,
            "Weighted-sum enumeration should be skipped for large instances.",
        )

    @unittest.skipIf(plt is None, "matplotlib not available in test environment")
    def test_plot_pareto_front_creates_image(self) -> None:
        solution = Solution(self.instance)
        solution.selected_vertices = [0, 1, 2]
        solution.reevaluate()
        analysis = analyse_solution(self.instance, solution, steps=3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "pareto.png"
            weighted_candidates = [entry.candidate for entry in analysis.weighted_front]
            alpha_labels = [
                f"α=({entry.alpha_pair[0]:.2f}, {entry.alpha_pair[1]:.2f})"
                for entry in analysis.weighted_front
            ]
            base_candidate = (
                weighted_candidates[0] if weighted_candidates else analysis.base_solution
            )
            frontier = (
                weighted_candidates[1:] if weighted_candidates else analysis.epsilon_front
            )
            generated = plot_pareto_front(
                base_candidate,
                frontier,
                output_path,
                alpha_labels=alpha_labels if weighted_candidates else None,
            )
            self.assertTrue(generated.exists())
            points = pareto_points_to_rows([base_candidate, *frontier])
            self.assertTrue(points)
            first_dispersion, first_penalty = points[0]
            self.assertGreaterEqual(first_dispersion, 0.0)
            self.assertGreaterEqual(first_penalty, 0.0)

    def test_default_alpha_schedule_generates_expected_pairs(self) -> None:
        schedule = default_alpha_schedule(step=0.25)
        self.assertEqual(schedule[0], (1.0, 0.0))
        self.assertIn((0.5, 0.5), schedule)
        self.assertEqual(schedule[-1], (0.0, 1.0))

    def test_generate_weighted_front_respects_alpha_pairs(self) -> None:
        schedule = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
        front = generate_weighted_front(self.instance, schedule)
        self.assertTrue(front)
        self.assertAlmostEqual(sum(front[0].alpha_pair), 1.0)
        self.assertEqual(front[0].alpha_pair, schedule[0])
        produced_pairs = {entry.alpha_pair for entry in front}
        self.assertIn(schedule[-1], produced_pairs)

    def test_monotonic_weighted_front_filters_penalty_increases(self) -> None:
        entries = [
            WeightedFrontEntry(
                alpha_pair=(1.0, 0.0),
                candidate=CandidateSolution(
                    selected_nodes=("0",),
                    capacity=0.0,
                    dispersion=144.8,
                    symmetry_penalty=13.84,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.85, 0.15),
                candidate=CandidateSolution(
                    selected_nodes=("1",),
                    capacity=0.0,
                    dispersion=142.9,
                    symmetry_penalty=13.84,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.75, 0.25),
                candidate=CandidateSolution(
                    selected_nodes=("0", "1"),
                    capacity=0.0,
                    dispersion=140.3,
                    symmetry_penalty=6.92,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.60, 0.40),
                candidate=CandidateSolution(
                    selected_nodes=("0", "2"),
                    capacity=0.0,
                    dispersion=142.9,
                    symmetry_penalty=6.92,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.00, 1.0),
                candidate=CandidateSolution(
                    selected_nodes=("2",),
                    capacity=0.0,
                    dispersion=70.4,
                    symmetry_penalty=6.92,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.35, 0.65),
                candidate=CandidateSolution(
                    selected_nodes=("0", "3"),
                    capacity=0.0,
                    dispersion=55.0,
                    symmetry_penalty=0.0,
                ),
            ),
            WeightedFrontEntry(
                alpha_pair=(0.00, 1.0),
                candidate=CandidateSolution(
                    selected_nodes=("3",),
                    capacity=0.0,
                    dispersion=80.0,
                    symmetry_penalty=8.0,
                ),
            ),
        ]

        filtered = _enforce_monotonic_weighted_front(entries, tolerance=1e-6)

        self.assertEqual(len(filtered), 3)
        penalties = [entry.candidate.symmetry_penalty for entry in filtered]
        self.assertTrue(
            all(curr <= prev + 1e-6 for prev, curr in zip(penalties, penalties[1:]))
        )
        dispersions = [entry.candidate.dispersion for entry in filtered]
        self.assertListEqual(dispersions, [142.9, 70.4, 55.0])


class TestConfigurationLoading(unittest.TestCase):
    def test_load_test_cases_reads_alpha_step(self) -> None:
        config_name = "temp_alpha"
        config_path = ROOT / "test" / f"{config_name}.txt"
        config_path.write_text(
            "\n".join(
                [
                    "# instance\tseed\tmax_time\tbeta_c\tbeta_ls\tmax_iter\tweight",
                    "sample_instance.txt\t0\t1\t0.5\t0.5\t10\t0.7",
                    "sample_instance.txt\t0\t1\t0.5\t0.5\t10\t0.7\t0.2",
                ]
            ),
            encoding="utf-8",
        )
        original_cwd = Path.cwd()
        try:
            os.chdir(ROOT / "CDP")
            cases = load_test_cases(config_name)
        finally:
            os.chdir(original_cwd)
            if config_path.exists():
                config_path.unlink()
        self.assertEqual(len(cases), 2)
        self.assertAlmostEqual(cases[0].alpha_step, DEFAULT_ALPHA_STEP)
        self.assertAlmostEqual(cases[1].alpha_step, 0.2)
        for case in cases:
            self.assertEqual(case.instance_path.name, "sample_instance.txt")

    def test_load_test_cases_prefers_instances_folder(self) -> None:
        config_name = "temp_instances"
        config_path = ROOT / "test" / f"{config_name}.txt"
        instance_name = "SOM-a_11_n50_b02_m5.txt"
        config_path.write_text(
            "\n".join(
                [
                    "# instance\tseed\tmax_time\tbeta_c\tbeta_ls\tmax_iter\tweight",
                    f"{instance_name}\t0\t1\t0.5\t0.5\t10\t0.7",
                ]
            ),
            encoding="utf-8",
        )
        original_cwd = Path.cwd()
        try:
            os.chdir(ROOT / "CDP")
            cases = load_test_cases(config_name)
        finally:
            os.chdir(original_cwd)
            if config_path.exists():
                config_path.unlink()
        self.assertEqual(len(cases), 1)
        resolved_path = cases[0].instance_path
        self.assertTrue(resolved_path.exists())
        self.assertTrue(resolved_path.is_file())
        self.assertEqual(resolved_path.name, instance_name)
        self.assertEqual(resolved_path.parent, INSTANCES_DIR)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
