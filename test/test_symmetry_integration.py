from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "CDP"))

from Instance import Instance
from Solution import Solution
from symmetry_integration import (
    analyse_solution,
    default_alpha_schedule,
    generate_weighted_front,
    pareto_points_to_rows,
    plot_pareto_front,
)

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
            first_penalty, first_dispersion = points[0]
            self.assertGreaterEqual(first_penalty, 0.0)
            self.assertGreaterEqual(first_dispersion, 0.0)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
