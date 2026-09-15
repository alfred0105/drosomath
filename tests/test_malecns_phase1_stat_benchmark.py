import unittest


class Phase1StatBenchmarkTests(unittest.TestCase):
    def _runs(self):
        def evals(lat, num, cmp_, add):
            return {
                "tasks": {
                    "laterality": {"accuracy": lat},
                    "numerosity_1_4": {"accuracy": num},
                    "compare_1_3": {"accuracy": cmp_},
                    "addition_1_3": {"accuracy": add},
                },
                "macro_accuracy": (lat + num + cmp_ + add) / 4,
            }

        return [
            {"seed": 7, "common_eval": {"v1": evals(.90, .30, .35, .25), "v2": evals(.92, .36, .45, .27)}},
            {"seed": 17, "common_eval": {"v1": evals(.88, .28, .34, .24), "v2": evals(.90, .34, .44, .26)}},
            {"seed": 27, "common_eval": {"v1": evals(.91, .31, .36, .26), "v2": evals(.93, .35, .46, .28)}},
        ]

    def test_defaults_are_larger_than_original_pilot(self):
        from drosomath.malecns.phase1_stat_benchmark import Phase1StatConfig

        cfg = Phase1StatConfig()
        self.assertEqual(cfg.seeds, (7, 17, 27))
        self.assertEqual(cfg.stage_trials, 512)
        self.assertEqual(cfg.validation_trials_per_label, 32)

    def test_paired_summary_and_gate(self):
        from drosomath.malecns.phase1_stat_benchmark import _gate, _task_summary

        summary = _task_summary(self._runs())
        self.assertGreater(summary["numerosity_1_4"]["paired_delta"]["mean"], 0.0)
        self.assertEqual(summary["compare_1_3"]["seed_wins_v2"], 3)
        self.assertTrue(_gate(summary)["passed"])

    def test_html_renders_summary(self):
        from drosomath.malecns.phase1_stat_benchmark import build_html, _gate, _task_summary

        runs = self._runs()
        tasks = _task_summary(runs)
        report = {
            "config": {"seeds": [7, 17, 27], "stage_trials": 512, "validation_trials_per_label": 32},
            "runs": runs,
            "summary": {"tasks": tasks},
            "phase1_stat_gate": _gate(tasks),
        }
        page = build_html(report)
        self.assertIn("Phase-1 statistical benchmark", page)
        self.assertIn("numerosity_1_4", page)
        self.assertIn("Seed 17", page)


if __name__ == "__main__":
    unittest.main()
