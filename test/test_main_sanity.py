from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CDP"))

from Main import perform_sanity_check
from Instance import Instance
from Solution import Solution
from objects import TestCase as BenchmarkTestCase


class TestMainSanity(unittest.TestCase):
    """Validate the final sanity checks performed after solving instances."""

    @classmethod
    def setUpClass(cls) -> None:
        instance_path = ROOT / "CDP" / "sample_instance.txt"
        cls.instance = Instance(str(instance_path))
        cls.instance.assign_colours(["red", "blue", "green"])
        cls.instance.set_symmetry_parameters(lambda_penalty=0.2)
        cls.test_case = BenchmarkTestCase(
            instance_name="sample",
            instance_path=instance_path,
            seed=0,
            max_time=0,
            beta_construction=0.5,
            beta_local_search=0.5,
            max_iterations=0,
            weight=0.5,
        )

    def _build_solution(self) -> Solution:
        solution = Solution(self.instance)
        for vertex in range(self.instance.node_count):
            solution.add_vertex(vertex)
        solution.reevaluate(alpha=0.75)
        return solution

    def test_perform_sanity_check_detects_inconsistent_objective(self) -> None:
        solution = self._build_solution()
        solution.objective_value += 1.0
        with self.assertRaises(RuntimeError):
            perform_sanity_check([(self.test_case, solution, [])])

    def test_perform_sanity_check_enforces_alpha_one_dispersion(self) -> None:
        solution = self._build_solution()
        solution.reevaluate(alpha=1.0)
        perform_sanity_check([(self.test_case, solution, [])])
        self.assertAlmostEqual(solution.objective_value, solution.cdp_objective)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
