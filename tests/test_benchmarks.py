import unittest

from drosomath.benchmarks import BenchmarkVariant, ContinualAblationBenchmark
from drosomath.core import PlasticityTracker, RewardWeightRule, SpikingNetwork, SynapseState
from drosomath.experiments import AssociativeTrial, SequentialMemoryExperiment, SequentialTask


class ContinualAblationBenchmarkTests(unittest.TestCase):
    def _task(self, name: str) -> SequentialTask:
        trial = AssociativeTrial({0: 1.0}, 1)
        return SequentialTask(name=name, train_trials=(trial,), eval_trials=(trial,))

    def _experiment(self) -> SequentialMemoryExperiment:
        tracker = PlasticityTracker(
            [SynapseState(0, 1, 0.6)],
            weight_rule=RewardWeightRule(learning_rate=0.01),
        )
        network = SpikingNetwork.from_ids(
            [0, 1],
            tracker,
            threshold=0.5,
            decay=1.0,
        )
        return SequentialMemoryExperiment(
            network,
            tracker,
            output_neurons=[1],
            response_steps=2,
        )

    def test_variants_use_independent_experiments(self) -> None:
        built = []

        def factory() -> SequentialMemoryExperiment:
            experiment = self._experiment()
            built.append(experiment)
            return experiment

        benchmark = ContinualAblationBenchmark([self._task("A")])
        result = benchmark.run(
            [
                BenchmarkVariant("one", factory),
                BenchmarkVariant("two", factory),
            ]
        )

        self.assertEqual(len(built), 2)
        self.assertIsNot(built[0], built[1])
        self.assertEqual([item.name for item in result.variants], ["one", "two"])
        self.assertEqual(result.variants[0].summary["mean_retention"], 1.0)
        self.assertEqual(result.variants[1].summary["mean_retention"], 1.0)

    def test_after_stage_hook_runs_once_per_task(self) -> None:
        stages = []
        benchmark = ContinualAblationBenchmark(
            [self._task("A"), self._task("B")]
        )
        benchmark.run(
            [
                BenchmarkVariant(
                    "hooked",
                    self._experiment,
                    after_stage=lambda _experiment, stage: stages.append(stage),
                )
            ]
        )
        self.assertEqual(stages, [0, 1])

    def test_duplicate_variant_names_are_rejected(self) -> None:
        benchmark = ContinualAblationBenchmark([self._task("A")])
        variant = BenchmarkVariant("same", self._experiment)
        with self.assertRaises(ValueError):
            benchmark.run([variant, variant])


if __name__ == "__main__":
    unittest.main()
