import unittest

from drosomath.interference_benchmark import VARIANT_NAMES, run_benchmark


class InterferenceBenchmarkTests(unittest.TestCase):
    def test_seeded_benchmark_runs_all_variants(self) -> None:
        result = run_benchmark(runs=2, seed_start=10)
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
            self.assertIn("mean", entry["final_accuracy"])
            self.assertIn("mean", entry["mean_retention"])
            self.assertIn("mean", entry["mean_forgetting"])

    def test_same_seed_range_is_deterministic(self) -> None:
        first = run_benchmark(runs=1, seed_start=3)
        second = run_benchmark(runs=1, seed_start=3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
