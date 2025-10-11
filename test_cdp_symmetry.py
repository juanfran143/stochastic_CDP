"""Basic regression tests for the :mod:`cdp_symmetry` module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cdp_symmetry import (
    EpsilonConstraintSolver,
    GreedySymmetryHeuristic,
    Node,
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

    def test_rule_based_colour_generation(self) -> None:
        coloured = self.instance.with_random_colours()
        colours = [node.color for node in coloured.nodes]
        self.assertEqual(colours, ["#1f77b4", "#1f77b4", "#ff7f0e", "#1f77b4"])

    def test_rule_based_colour_divisibility(self) -> None:
        nodes = [
            Node(node_id=str(idx), capacity=1.0, color="#000000", coordinates=(0.0, 0.0))
            for idx in range(1, 8)
        ]
        distances = {
            (str(i), str(j)): 1.0
            for i in range(1, 8)
            for j in range(1, 8)
            if i != j
        }
        instance = ProblemInstance(
            nodes=nodes,
            demand=5.0,
            lambda_penalty=0.0,
            distances=distances,
        )
        coloured = instance.with_random_colours()
        colours = [node.color for node in coloured.nodes]
        self.assertEqual(
            colours,
            [
                "#1f77b4",  # fallback
                "#1f77b4",  # divisible by 2
                "#ff7f0e",  # divisible by 3
                "#1f77b4",  # divisible by 2
                "#2ca02c",  # divisible by 5
                "#1f77b4",  # divisible by 2 takes precedence over 3
                "#d62728",  # divisible by 7
            ],
        )

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
