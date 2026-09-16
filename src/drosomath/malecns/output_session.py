from __future__ import annotations

import math
from dataclasses import dataclass

from drosomath.whole_brain import OutgoingBudgetNormalizer, UsageRewardRule

from .output_readout import PopulationReadout, ReadoutTrainResult


NO_OUTPUT = "NO_OUTPUT"


@dataclass(frozen=True, slots=True)
class OutputSessionConfig:
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    no_output_reward: float = -0.35
    reset_fast_state_before_trial: bool = True


@dataclass(frozen=True, slots=True)
class OutputObservation:
    prediction: str
    confidence: float
    spike_counts: object
    features: object
    total_output_spikes: int


@dataclass(frozen=True, slots=True)
class BrainOutputTrialResult:
    target: str
    prediction: str
    confidence: float
    correct: bool
    reward: float
    total_output_spikes: int
    learning: dict[str, object]


class MaleCNSOutputSession:
    """Two-stage output curriculum for a plastic MaleCNS brain.

    Phase A trains only the external population decoder while synaptic usage
    tracking is disabled. Phase B requires the decoder to be frozen and then
    trains only the CNS through reward-modulated local plasticity.

    A silent output population is represented explicitly as ``NO_OUTPUT``.
    This prevents the decoder bias from turning zero-spike trials into an
    arbitrary task-label response.
    """

    def __init__(
        self,
        brain,
        readout: PopulationReadout,
        *,
        reward_rule: UsageRewardRule | None = None,
        normalizer: OutgoingBudgetNormalizer | None = None,
        config: OutputSessionConfig | None = None,
    ) -> None:
        np = readout.np
        self.np = np
        self.brain = brain
        self.readout = readout
        self.reward_rule = reward_rule or UsageRewardRule()
        self.normalizer = normalizer or OutgoingBudgetNormalizer()
        self.config = config or OutputSessionConfig()

        output_indices = np.asarray(readout.population.indices, dtype=np.int32)
        if len(output_indices) == 0:
            raise ValueError("output population must not be empty")
        if np.any(output_indices < 0) or np.any(output_indices >= brain.connectome.neuron_count):
            raise ValueError("output population contains indices outside the brain")
        if len(np.unique(output_indices)) != len(output_indices):
            raise ValueError("output population contains duplicate neurons")

        self._output_lookup = np.full(
            brain.connectome.neuron_count,
            -1,
            dtype=np.int32,
        )
        self._output_lookup[output_indices] = np.arange(len(output_indices), dtype=np.int32)
        # Concept curricula repeatedly sample from a finite bank of scenes. Avoid
        # rebuilding body-id -> simulation-index arrays on every presentation.
        self._stimulus_index_cache: dict[tuple[int, ...], object] = {}
        self._step_count_cache: dict[float, int] = {}

    def _indices_for_stimulus(self, stimulus_body_ids):
        key = tuple(int(x) for x in stimulus_body_ids)
        cached = self._stimulus_index_cache.get(key)
        if cached is None:
            cached = self.brain.indices_for_ids(key)
            self._stimulus_index_cache[key] = cached
        return cached

    def _run_window(
        self,
        *,
        stimulus_body_ids,
        duration_ms: float,
        stimulus_rate_hz: float,
    ) -> OutputObservation:
        np = self.np
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if stimulus_rate_hz < 0.0:
            raise ValueError("stimulus_rate_hz must be >= 0")

        if self.config.reset_fast_state_before_trial:
            self.brain.reset()

        stimulus_indices = self._indices_for_stimulus(stimulus_body_ids)
        duration_key = float(duration_ms)
        steps = self._step_count_cache.get(duration_key)
        if steps is None:
            steps = max(1, int(math.ceil(duration_ms / self.brain.params.dt_ms)))
            self._step_count_cache[duration_key] = steps
        counts = np.zeros(len(self.readout.population.body_ids), dtype=np.int32)

        for _ in range(steps):
            fired, _ = self.brain.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            if len(fired):
                local = self._output_lookup[fired]
                local = local[local >= 0]
                if len(local):
                    np.add.at(counts, local, 1)

        features = self.readout.features_from_counts(counts, duration_ms=duration_ms)
        total = int(counts.sum())
        if total == 0:
            prediction, confidence = NO_OUTPUT, 0.0
        else:
            prediction, confidence = self.readout.predict(features)
        return OutputObservation(
            prediction=prediction,
            confidence=confidence,
            spike_counts=counts,
            features=features,
            total_output_spikes=total,
        )

    def train_decoder_on_features(self, features, *, target: str) -> ReadoutTrainResult:
        return self.readout.train(features, target=target)

    def train_decoder_trial(
        self,
        *,
        stimulus_body_ids,
        target: str,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 300.0,
    ) -> tuple[OutputObservation, ReadoutTrainResult]:
        previous = self.brain.set_plasticity_tracking(False)
        try:
            observation = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            training = self.readout.train(observation.features, target=target)
            return observation, training
        finally:
            self.brain.set_plasticity_tracking(previous)

    def freeze_decoder(self) -> None:
        self.readout.freeze()

    def train_brain_trial(
        self,
        *,
        stimulus_body_ids,
        target: str,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 300.0,
    ) -> BrainOutputTrialResult:
        if not self.readout.frozen:
            raise RuntimeError("freeze the output decoder before training the CNS")
        if target not in self.readout.labels:
            raise KeyError(f"unknown output label: {target}")

        previous = self.brain.set_plasticity_tracking(True)
        try:
            observation = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            correct = observation.prediction == target
            if observation.total_output_spikes == 0:
                reward = self.config.no_output_reward
            else:
                reward = self.config.correct_reward if correct else self.config.incorrect_reward
            learning = self.brain.learn_from_reward(
                reward=reward,
                rule=self.reward_rule,
                normalizer=self.normalizer,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)

        return BrainOutputTrialResult(
            target=target,
            prediction=observation.prediction,
            confidence=observation.confidence,
            correct=correct,
            reward=float(reward),
            total_output_spikes=observation.total_output_spikes,
            learning=learning,
        )

    def evaluate_trial(
        self,
        *,
        stimulus_body_ids,
        target: str,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 300.0,
    ) -> dict[str, object]:
        previous = self.brain.set_plasticity_tracking(False)
        try:
            observation = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)

        return {
            "target": target,
            "prediction": observation.prediction,
            "confidence": observation.confidence,
            "correct": observation.prediction == target,
            "total_output_spikes": observation.total_output_spikes,
            "silent": observation.total_output_spikes == 0,
            "readout": self.readout.summary(),
        }

    def summary(self) -> dict[str, object]:
        return {
            "readout": self.readout.summary(),
            "correct_reward": self.config.correct_reward,
            "incorrect_reward": self.config.incorrect_reward,
            "no_output_reward": self.config.no_output_reward,
            "plasticity": self.brain.plasticity.summary(),
            "stimulus_cache_entries": len(self._stimulus_index_cache),
        }
