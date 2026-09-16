import unittest


class StrictConceptFoundationGateTests(unittest.TestCase):
    def _report(self, *, before=0.50, after=0.85, final=0.82):
        names = ["object_presence", "single_vs_multiple", "latent_quantity_1_3"]
        thresholds = [0.80, 0.70, 0.50]
        chances = [0.50, 0.50, 1 / 3]
        stages = []
        final_tasks = {}
        for name, threshold, chance in zip(names, thresholds, chances):
            held = max(after if name == "object_presence" else threshold + 0.05, chance + 0.12)
            pre = min(before, held - 0.05)
            stages.append({
                "stage": name,
                "before_heldout": {"accuracy": pre},
                "after_train": {"accuracy": min(1.0, held + 0.05)},
                "after_heldout": {"accuracy": held, "chance": chance},
            })
            final_tasks[name] = {"heldout_accuracy": max(final, threshold)}
        return {
            "stage_reports": stages,
            "retention_history": [{"tasks": final_tasks}],
        }

    def test_gate_passes_when_training_improves_and_generalizes(self):
        from drosomath.malecns.concept_foundation_strict import strict_foundation_gate

        gate = strict_foundation_gate(self._report(), min_learning_gain=0.03)
        self.assertTrue(gate["passed"])

    def test_gate_rejects_high_accuracy_without_learning_gain(self):
        from drosomath.malecns.concept_foundation_strict import strict_foundation_gate

        report = self._report()
        row = report["stage_reports"][0]
        row["before_heldout"]["accuracy"] = 0.84
        row["after_heldout"]["accuracy"] = 0.85
        gate = strict_foundation_gate(report, min_learning_gain=0.03)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["tasks"]["object_presence"]["learned_from_training"])


if __name__ == "__main__":
    unittest.main()
