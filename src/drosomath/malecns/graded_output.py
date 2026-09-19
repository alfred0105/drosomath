from __future__ import annotations

import math
from dataclasses import dataclass

from drosomath.whole_brain import OutgoingBudgetNormalizer, UsageRewardRule


NO_OUTPUT = "NO_OUTPUT"
BETWEEN_LEVELS = "BETWEEN_LEVELS"


@dataclass(frozen=True, slots=True)
class TwoLevelRateCalibration:
    """Empirical rate-code calibration for the first two output levels."""

    background_mean_hz: float
    background_std_hz: float
    present_mean_hz: float
    present_std_hz: float
    separation_hz: float
    usable: bool
    reason: str

    def code_config(
        self,
        *,
        max_level: int = 7,
        max_teaching_signal: float = 1.0,
        step_override_hz: float | None = None,
    ) -> "GradedRateCodeConfig":
        step = (
            self.present_mean_hz - self.background_mean_hz
            if step_override_hz is None
            else float(step_override_hz)
        )
        if step == 0.0:
            step = 1e-6
        tolerance = max(1e-6, abs(step) * 0.24)
        return GradedRateCodeConfig(
            base_rate_hz=self.background_mean_hz,
            level_step_hz=step,
            tolerance_hz=tolerance,
            max_level=max_level,
            max_teaching_signal=max_teaching_signal,
        )

    def summary(self) -> dict[str, object]:
        return {
            "background_mean_hz": self.background_mean_hz,
            "background_std_hz": self.background_std_hz,
            "present_mean_hz": self.present_mean_hz,
            "present_std_hz": self.present_std_hz,
            "separation_hz": self.separation_hz,
            "usable": self.usable,
            "reason": self.reason,
        }


def calibrate_two_level_rate_code(
    background_rates_hz,
    present_rates_hz,
    *,
    min_separation_hz: float = 0.10,
) -> TwoLevelRateCalibration:
    """Measure output distributions before learning and derive a fixed code."""
    background = tuple(float(x) for x in background_rates_hz)
    present = tuple(float(x) for x in present_rates_hz)
    if not background or not present:
        raise ValueError("both calibration classes must contain at least one rate")
    if min_separation_hz < 0.0:
        raise ValueError("min_separation_hz must be >= 0")

    def mean(values):
        return sum(values) / len(values)

    def std(values, center):
        return (sum((x - center) ** 2 for x in values) / len(values)) ** 0.5

    background_mean = mean(background)
    present_mean = mean(present)
    separation = abs(present_mean - background_mean)
    usable = separation >= float(min_separation_hz)
    return TwoLevelRateCalibration(
        background_mean_hz=background_mean,
        background_std_hz=std(background, background_mean),
        present_mean_hz=present_mean,
        present_std_hz=std(present, present_mean),
        separation_hz=separation,
        usable=usable,
        reason=(
            "measured separation is large enough"
            if usable
            else "background and present output-rate distributions are not separated"
        ),
    )


@dataclass(frozen=True, slots=True)
class GradedRateCodeConfig:
    """Fixed, non-trainable population-rate code for MaleCNS outputs.

    One real output population carries an ordinal value by its mean firing rate.
    Level ``k`` has a fixed target rate ``base_rate_hz + k * level_step_hz``.
    No learned classifier or label-specific output population is used.
    """

    base_rate_hz: float = 1.0
    level_step_hz: float = 1.5
    tolerance_hz: float = 0.50
    max_level: int = 7
    max_teaching_signal: float = 1.0

    def __post_init__(self) -> None:
        if self.base_rate_hz < 0.0:
            raise ValueError("base_rate_hz must be >= 0")
        if self.level_step_hz == 0.0:
            raise ValueError("level_step_hz must be non-zero")
        if self.tolerance_hz <= 0.0:
            raise ValueError("tolerance_hz must be > 0")
        if self.tolerance_hz >= abs(self.level_step_hz) / 2.0:
            raise ValueError("tolerance_hz must be < half the level spacing")
        if self.max_level < 1:
            raise ValueError("max_level must be >= 1")
        if self.max_teaching_signal <= 0.0:
            raise ValueError("max_teaching_signal must be > 0")

    def target_rate_hz(self, level: int) -> float:
        level = int(level)
        if level < 0 or level > self.max_level:
            raise ValueError(f"level must be in [0, {self.max_level}]")
        return self.base_rate_hz + self.level_step_hz * level


@dataclass(frozen=True, slots=True)
class GradedObservation:
    predicted_level: int | None
    status: str
    population_rate_hz: float
    total_output_spikes: int
    target_level: int | None = None
    target_rate_hz: float | None = None
    signed_error_hz: float | None = None
    teaching_signal: float = 0.0
    correct: bool = False


@dataclass(frozen=True, slots=True)
class GradedTrainResult:
    observation: GradedObservation
    learning: dict[str, object]


class GradedPopulationCode:
    """Decode one population firing rate into fixed ordinal levels."""

    def __init__(self, population_size: int, config: GradedRateCodeConfig | None = None) -> None:
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        self.population_size = int(population_size)
        self.config = config or GradedRateCodeConfig()

    def population_rate_hz(self, total_spikes: int, *, duration_ms: float) -> float:
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        return float(total_spikes) * 1000.0 / (self.population_size * float(duration_ms))

    def decode_rate(self, rate_hz: float) -> tuple[int | None, str]:
        if rate_hz < 0.0:
            raise ValueError("rate_hz must be >= 0")
        cfg = self.config
        nearest = int(round((float(rate_hz) - cfg.base_rate_hz) / cfg.level_step_hz))
        nearest = min(cfg.max_level, max(0, nearest))
        distance = abs(float(rate_hz) - cfg.target_rate_hz(nearest))
        if distance <= cfg.tolerance_hz:
            return nearest, f"LEVEL_{nearest}"
        if rate_hz == 0.0:
            return None, NO_OUTPUT
        return None, BETWEEN_LEVELS

    def observe(
        self,
        total_spikes: int,
        *,
        duration_ms: float,
        target_level: int | None = None,
    ) -> GradedObservation:
        rate = self.population_rate_hz(total_spikes, duration_ms=duration_ms)
        predicted, status = self.decode_rate(rate)
        if target_level is None:
            return GradedObservation(
                predicted_level=predicted,
                status=status,
                population_rate_hz=rate,
                total_output_spikes=int(total_spikes),
            )

        target = self.config.target_rate_hz(int(target_level))
        error = target - rate
        correct = predicted == int(target_level)
        if correct:
            teaching = 0.0
        else:
            teaching = error / self.config.level_step_hz
            teaching = max(-self.config.max_teaching_signal, min(self.config.max_teaching_signal, teaching))
        return GradedObservation(
            predicted_level=predicted,
            status=status,
            population_rate_hz=rate,
            total_output_spikes=int(total_spikes),
            target_level=int(target_level),
            target_rate_hz=target,
            signed_error_hz=error,
            teaching_signal=float(teaching),
            correct=bool(correct),
        )

    def summary(self) -> dict[str, object]:
        return {
            "type": "fixed_population_rate_code",
            "population_size": self.population_size,
            "base_rate_hz": self.config.base_rate_hz,
            "level_step_hz": self.config.level_step_hz,
            "tolerance_hz": self.config.tolerance_hz,
            "max_level": self.config.max_level,
            "target_rates_hz": [
                self.config.target_rate_hz(level)
                for level in range(self.config.max_level + 1)
            ],
            "trainable_decoder": False,
        }


class GradedPopulationSession:
    """Run and train MaleCNS with one fixed graded output population.

    The only learned state is inside the CNS.  The output code itself is fixed.
    A positive teaching signal means the observed population rate is below the
    target zone; a negative signal means it is above the target zone.
    """

    def __init__(
        self,
        brain,
        output_population,
        *,
        code_config: GradedRateCodeConfig | None = None,
        reward_rule: UsageRewardRule | None = None,
        normalizer: OutgoingBudgetNormalizer | None = None,
        probe_indices=None,
    ) -> None:
        np = brain.np
        self.np = np
        self.brain = brain
        self.output_population = output_population
        self.code = GradedPopulationCode(len(output_population.body_ids), code_config)
        self.reward_rule = reward_rule or UsageRewardRule()
        self.normalizer = normalizer or OutgoingBudgetNormalizer()

        output_indices = np.asarray(output_population.indices, dtype=np.int32)
        if len(output_indices) == 0:
            raise ValueError("output population must not be empty")
        self._output_lookup = np.full(brain.connectome.neuron_count, -1, dtype=np.int32)
        self._output_lookup[output_indices] = np.arange(len(output_indices), dtype=np.int32)
        if probe_indices is not None:
            probe_indices = np.asarray(probe_indices, dtype=np.int32)
            if len(probe_indices) == 0:
                raise ValueError("activity probe must not be empty")
            if np.any(probe_indices < 0) or np.any(probe_indices >= brain.connectome.neuron_count):
                raise ValueError("activity probe contains indices outside the brain")
            if len(np.unique(probe_indices)) != len(probe_indices):
                raise ValueError("activity probe contains duplicate neurons")
            self._probe_indices = probe_indices
            self._probe_lookup = np.full(brain.connectome.neuron_count, -1, dtype=np.int32)
            self._probe_lookup[probe_indices] = np.arange(len(probe_indices), dtype=np.int32)
        else:
            self._probe_indices = None
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
        collect_probe: bool = False,
    ):
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if stimulus_rate_hz < 0.0:
            raise ValueError("stimulus_rate_hz must be >= 0")
        self.brain.reset()
        stimulus_indices = self._indices_for_stimulus(stimulus_body_ids)
        steps = self._step_count_cache.get(float(duration_ms))
        if steps is None:
            steps = max(1, int(math.ceil(float(duration_ms) / self.brain.params.dt_ms)))
            self._step_count_cache[float(duration_ms)] = steps

        total = 0
        probe_counts = None
        if collect_probe:
            if self._probe_indices is None:
                raise RuntimeError("activity probe is not configured")
            probe_counts = self.np.zeros(len(self._probe_indices), dtype=self.np.int32)
        for _ in range(steps):
            fired, _ = self.brain.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=float(stimulus_rate_hz),
            )
            if len(fired):
                local = self._output_lookup[fired]
                total += int((local >= 0).sum())
                if probe_counts is not None:
                    probe_local = self._probe_lookup[fired]
                    probe_local = probe_local[probe_local >= 0]
                    if len(probe_local):
                        self.np.add.at(probe_counts, probe_local, 1)
        return (total, probe_counts) if probe_counts is not None else total

    def measure_trial_rate(
        self,
        *,
        stimulus_body_ids,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 205.0,
    ) -> GradedObservation:
        """Measure an untrained output rate without applying learning."""
        previous = self.brain.set_plasticity_tracking(False)
        try:
            total = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)
        return self.code.observe(total, duration_ms=duration_ms)

    def measure_trial_activity(
        self,
        *,
        stimulus_body_ids,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 205.0,
    ):
        """Return the output observation and a fixed internal probe vector."""
        previous = self.brain.set_plasticity_tracking(False)
        try:
            total, probe_counts = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
                collect_probe=True,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)
        return self.code.observe(total, duration_ms=duration_ms), probe_counts

    def evaluate_trial(
        self,
        *,
        stimulus_body_ids,
        target_level: int,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 205.0,
    ) -> GradedObservation:
        previous = self.brain.set_plasticity_tracking(False)
        try:
            total = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)
        return self.code.observe(total, duration_ms=duration_ms, target_level=target_level)

    def train_trial(
        self,
        *,
        stimulus_body_ids,
        target_level: int,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 205.0,
    ) -> GradedTrainResult:
        previous = self.brain.set_plasticity_tracking(True)
        try:
            total = self._run_window(
                stimulus_body_ids=stimulus_body_ids,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            observation = self.code.observe(total, duration_ms=duration_ms, target_level=target_level)
            learning = self.brain.learn_from_reward(
                reward=observation.teaching_signal,
                rule=self.reward_rule,
                normalizer=self.normalizer,
            )
        finally:
            self.brain.set_plasticity_tracking(previous)
        return GradedTrainResult(observation=observation, learning=learning)

    def summary(self) -> dict[str, object]:
        return {
            "code": self.code.summary(),
            "activity_probe_size": 0 if self._probe_indices is None else int(len(self._probe_indices)),
            "stimulus_cache_entries": len(self._stimulus_index_cache),
            "plasticity": self.brain.plasticity.summary(),
        }
