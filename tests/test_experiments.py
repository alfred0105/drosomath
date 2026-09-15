import unittest

from drosomath.core import PlasticityTracker, RewardWeightRule, SpikingNetwork, SynapseState
from drosomath.experiments import AssociativeTrial, SequentialMemoryExperiment, SequentialTask


class SequentialMemoryExperimentTests(unittest.TestCase):
    def _experiment(self):
        first = SynapseState(0, 1, 0.6)
        second = SynapseState(2, 3, 0.6)
        tracker = PlasticityTracker(
            [first, second],
            reward_window=8,
            weight_rule=RewardWeightRule(learning_rate=0.1),
        )
        network = SpikingNetwork.from_ids(
            [0, 1, 2, 3],
            tracker,
            threshold=0.5,
            decay=1.0,
        )
        experiment = SequentialMemoryExperiment(
            network,
            tracker,
            output_neurons=[1, 3],
            response_steps=2,
        )
        return experiment, tracker, first, second

    def test_training_trial_applies_reward_to_used_path(self) -> None:
        experiment, _, first, second = self._experiment()
        trial = AssociativeTrial(stimulus={0: 1.0}, expected_output=1)

        result = experiment.run_trial(trial, training=True)

        self.assertTrue(result.correct)
        self.assertGreater(first.weight, 0.6)
        self.assertAlmostEqual(second.weight, 0.6)

    def test_evaluation_freezes_usage_and_weight_learning(self) -> None:
        experiment, _, first, _ = self._experiment()
        task = SequentialTask(
            name="A",
            train_trials=(),
            eval_trials=(AssociativeTrial({0: 1.0}, 1),),
        )
        before_weight = first.weight
        before_usage = first.usage_count

        score = experiment.evaluate_task(task)

        self.assertEqual(score, 1.0)
        self.assertAlmostEqual(first.weight, before_weight)
        self.assertEqual(first.usage_count, before_usage)

    def test_sequence_records_retention_for_seen_tasks(self) -> None:
        experiment, _, _, _ = self._experiment()
        task_a = SequentialTask(
            name="A",
            train_trials=(AssociativeTrial({0: 1.0}, 1),),
            eval_trials=(AssociativeTrial({0: 1.0}, 1),),
        )
        task_b = SequentialTask(
            name="B",
            train_trials=(AssociativeTrial({2: 1.0}, 3),),
            eval_trials=(AssociativeTrial({2: 1.0}, 3),),
        )

        evaluator = experiment.run_sequence([task_a, task_b], epochs_per_task=1)

        summary = evaluator.summary()
        self.assertEqual(summary["tasks"], 2)
        self.assertAlmostEqual(evaluator.retention("A"), 1.0)
        self.assertAlmostEqual(evaluator.retention("B"), 1.0)
        self.assertAlmostEqual(evaluator.mean_forgetting(), 0.0)


if __name__ == "__main__":
    unittest.main()
