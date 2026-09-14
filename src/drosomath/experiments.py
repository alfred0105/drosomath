from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .core import MemoryConsolidator, PlasticityTracker, SpikingNetwork
from .evaluation import ContinualMemoryEvaluator


@dataclass(frozen=True, slots=True)
class AssociativeTrial:
    stimulus: Mapping[int, float]
    expected_output: int


@dataclass(frozen=True, slots=True)
class SequentialTask:
    name: str
    train_trials: tuple[AssociativeTrial, ...]
    eval_trials: tuple[AssociativeTrial, ...]


@dataclass(frozen=True, slots=True)
class TrialResult:
    expected_output: int
    predicted_output: int | None
    correct: bool
    reward: float


class SequentialMemoryExperiment:
    """Generic train/evaluate loop for continual-memory experiments.

    The harness only presents stimuli, reads designated output neurons, and
    returns reward. It does not calculate an answer inside the simulated brain.
    Evaluation freezes both reward tracking and STDP so retention measurements do
    not themselves alter the learned synaptic state.
    """

    def __init__(
        self,
        network: SpikingNetwork,
        tracker: PlasticityTracker,
        *,
        output_neurons: Iterable[int],
        response_steps: int = 2,
        correct_reward: float = 1.0,
        wrong_reward: float = -1.0,
        consolidator: MemoryConsolidator | None = None,
    ) -> None:
        outputs = tuple(sorted(set(output_neurons)))
        if not outputs:
            raise ValueError("at least one output neuron is required")
        if response_steps < 1:
            raise ValueError("response_steps must be >= 1")
        for neuron_id in outputs:
            if neuron_id not in network.neurons:
                raise KeyError(f"unknown output neuron {neuron_id}")
        if network.tracker is not tracker:
            raise ValueError("network and tracker must reference the same tracker")

        self.network = network
        self.tracker = tracker
        self.output_neurons = outputs
        self.response_steps = response_steps
        self.correct_reward = float(correct_reward)
        self.wrong_reward = float(wrong_reward)
        self.consolidator = consolidator

    def run_trial(self, trial: AssociativeTrial, *, training: bool) -> TrialResult:
        if trial.expected_output not in self.output_neurons:
            raise ValueError("expected_output must be one of output_neurons")

        self.network.reset_state(clear_spike_history=True)
        self.tracker.clear_recent()
        previous_learning = self.network.learning_enabled
        self.network.set_learning_enabled(training)
        try:
            spike_counts = {neuron_id: 0 for neuron_id in self.output_neurons}
            for index in range(self.response_steps):
                result = self.network.step(trial.stimulus if index == 0 else None)
                for neuron_id in result.fired:
                    if neuron_id in spike_counts:
                        spike_counts[neuron_id] += 1

            active = [
                (count, neuron_id)
                for neuron_id, count in spike_counts.items()
                if count > 0
            ]
            predicted = None
            if active:
                # Highest spike count wins; neuron ID is a deterministic tie-break.
                predicted = max(active, key=lambda item: (item[0], -item[1]))[1]

            correct = predicted == trial.expected_output
            reward = self.correct_reward if correct else self.wrong_reward
            if training:
                self.tracker.apply_reward(
                    reward=reward,
                    step=self.network.step_index,
                )
                if self.consolidator is not None:
                    self.consolidator.consolidate(self.tracker.synapses)

            return TrialResult(
                expected_output=trial.expected_output,
                predicted_output=predicted,
                correct=correct,
                reward=reward,
            )
        finally:
            self.network.set_learning_enabled(previous_learning)

    def train_task(self, task: SequentialTask, *, epochs: int = 1) -> tuple[TrialResult, ...]:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        results: list[TrialResult] = []
        for _ in range(epochs):
            for trial in task.train_trials:
                results.append(self.run_trial(trial, training=True))
        return tuple(results)

    def evaluate_task(self, task: SequentialTask) -> float:
        if not task.eval_trials:
            return 0.0
        correct = sum(
            self.run_trial(trial, training=False).correct
            for trial in task.eval_trials
        )
        return correct / len(task.eval_trials)

    def run_sequence(
        self,
        tasks: Iterable[SequentialTask],
        *,
        epochs_per_task: int = 1,
    ) -> ContinualMemoryEvaluator:
        evaluator = ContinualMemoryEvaluator()
        seen: list[SequentialTask] = []
        for stage, task in enumerate(tasks):
            self.train_task(task, epochs=epochs_per_task)
            seen.append(task)
            scores = {seen_task.name: self.evaluate_task(seen_task) for seen_task in seen}
            evaluator.record(stage=stage, scores=scores)
        return evaluator
