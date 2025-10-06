"""Basic regression tests for the :mod:`cdp_symmetry` module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cdp_symmetry import (
    EpsilonConstraintSolver,
    GreedySymmetryHeuristic,
    ProblemInstance,
    WeightedSumSolver,
    plot_solution,
    symmetry_black_box_linear,
    plt,
)


INSTANCE_PATH = Path(__file__).resolve().with_name("example_instance.json")


class TestCDPSymmetry(unittest.TestCase):
    """Collection of lightweight regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = ProblemInstance.from_json_path(INSTANCE_PATH)

    def test_instance_loading(self) -> None:
        self.assertEqual(len(self.instance.nodes), 4)
        self.assertAlmostEqual(self.instance.demand, 8.0)
        self.assertAlmostEqual(self.instance.lambda_penalty, 0.4)

    def test_symmetry_black_box_linear(self) -> None:
        penalty = symmetry_black_box_linear(self.instance, ("A", "B", "D"))
        self.assertAlmostEqual(penalty, 2.08, places=2)

    def test_symmetry_black_box_linear_custom_gamma(self) -> None:
        penalty = symmetry_black_box_linear(self.instance, ("A", "B"), gamma=2.5)
        self.assertAlmostEqual(penalty, 2.5, places=3)

    def test_weighted_sum_solver(self) -> None:
        solver = WeightedSumSolver(instance=self.instance, alpha=0.6)
        solution = solver.solve()
        self.assertEqual(solution.selected_nodes, ("A", "C"))
        self.assertAlmostEqual(solution.dispersion, 6.02, places=2)
        self.assertAlmostEqual(solution.symmetry_penalty, 0.0)

    def test_epsilon_constraint_solvers(self) -> None:
        solver = EpsilonConstraintSolver(instance=self.instance)
        best_dispersion = solver.solve_max_dispersion(epsilon=0.0)
        self.assertEqual(best_dispersion.selected_nodes, ("A", "C"))
        min_penalty = solver.solve_min_penalty(epsilon=3.0)
        self.assertEqual(min_penalty.selected_nodes, ("A", "C"))

    def test_greedy_heuristic(self) -> None:
        heuristic = GreedySymmetryHeuristic(instance=self.instance)
        solution = heuristic.solve()
        self.assertEqual(solution.selected_nodes, ("A", "C"))

    def test_random_colour_generation_is_reproducible(self) -> None:
        first = self.instance.with_random_colours(seed=123)
        second = self.instance.with_random_colours(seed=123)
        colours_first = [node.color for node in first.nodes]
        colours_second = [node.color for node in second.nodes]
        self.assertEqual(colours_first, colours_second)
        distinct_colours = len(set(colours_first))
        if len(self.instance.nodes) >= 4:
            self.assertIn(distinct_colours, {3, 4})
        else:
            self.assertLessEqual(distinct_colours, len(self.instance.nodes))

    @unittest.skipIf(plt is None, "matplotlib not available in test environment")
    def test_plot_solution_creates_file(self) -> None:
        solver = WeightedSumSolver(instance=self.instance, alpha=0.5)
        exact_solution = solver.solve()
        heuristic = GreedySymmetryHeuristic(instance=self.instance)
        heuristic_solution = heuristic.solve()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "plot.png"
            result_path = plot_solution(
                instance=self.instance,
                base_solution=heuristic_solution,
                proposed_solution=exact_solution,
                output_path=output,
            )
            self.assertTrue(result_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
