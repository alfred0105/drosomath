import unittest

from drosomath.interference_benchmark import VARIANT_NAMES
from drosomath.interference_v2 import run_benchmark


class InterferenceBenchmarkTests(unittest.TestCase):
    def test_seeded_benchmark_runs_all_variants(self) -> None:
        result = run_benchmark(runs=2, seed_start=10)
        self.assertEqual(result["experiment"], "catastrophic_interference_v2")
        self.assertEqual(result["runs_per_variant"], 2)
        self.assertEqual(result["variants"], len(VARIANT_NAMES))
        self.assertEqual(result["total_model_runs"], 2 * len(VARIANT_NAMES))
        self.assertEqual(len(result["results"]), len(VARIANT_NAMES))
        self.assertEqual(
            [entry["name"] for entry in result["results"]],
            list(VARIANT_NAMES),
        )
        for entry in result["results"]:
            self.assertEqual(entry["runs"], 2)
            self.assertEqual(len(entry["raw"]), 2)
            self.assertIn("mean", entry["acquisition_accuracy"])
            self.assertIn("mean", entry["final_accuracy"])
            self.assertIn("mean", entry["learned_task_retention"])
            self.assertIn("mean", entry["mean_forgetting"])
            self.assertGreaterEqual(entry["catastrophic_forgetting_rate"], 0.0)
            self.assertLessEqual(entry["catastrophic_forgetting_rate"], 1.0)

    def test_same_seed_range_is_deterministic(self) -> None:
        first = run_benchmark(runs=1, seed_start=3)
        second = run_benchmark(runs=1, seed_start=3)
        self.assertEqual(first, second)

    def test_unlearned_tasks_are_not_counted_as_perfect_retention(self) -> None:
        result = run_benchmark(runs=1, seed_start=0)
        for entry in result["results"]:
            raw = entry["raw"][0]
            for task, acquisition in raw["acquisition_scores"].items():
                if acquisition == 0.0:
                    self.assertNotIn(task, raw["learned_retention"])


if __name__ == "__main__":
    unittest.main()
