import unittest

from drosomath.malecns.phase1b_benchmark import _aggregate, _gate


class Phase1BBenchmarkTests(unittest.TestCase):
    def test_aggregate_and_gate(self) -> None:
        def run(seed, base, new):
            return {
                "seed": seed,
                "common_eval": {
                    "v2": {"tasks": {k: {"accuracy": v} for k, v in base.items()}},
                    "v21": {"tasks": {k: {"accuracy": v} for k, v in new.items()}},
                },
            }

        runs = [
            run(7, {"laterality": .95, "numerosity_1_4": .32, "compare_1_3": .42, "addition_1_3": .30},
                   {"laterality": .96, "numerosity_1_4": .36, "compare_1_3": .45, "addition_1_3": .31}),
            run(17, {"laterality": .96, "numerosity_1_4": .31, "compare_1_3": .41, "addition_1_3": .29},
                    {"laterality": .97, "numerosity_1_4": .34, "compare_1_3": .44, "addition_1_3": .30}),
            run(27, {"laterality": .97, "numerosity_1_4": .33, "compare_1_3": .43, "addition_1_3": .32},
                    {"laterality": .98, "numerosity_1_4": .35, "compare_1_3": .46, "addition_1_3": .33}),
        ]
        summary = _aggregate(runs)
        self.assertGreater(summary["numerosity_1_4"]["paired_delta"]["mean"], 0.0)
        self.assertEqual(summary["compare_1_3"]["v21_wins"], 3)
        self.assertTrue(_gate(summary)["passed"])


if __name__ == "__main__":
    unittest.main()
