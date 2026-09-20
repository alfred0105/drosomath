"""Foundation task for delayed-cue recurrent working memory (F.2A).

This module is intentionally learning-disabled.  It presents a cue, a silent
delay, and one common GO cue in a single recurrent episode, then scores only
the GO-phase output populations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from itertools import combinations
import numpy as np

from .symbol_interface import SYMBOLS, SymbolInterface
from .symbol_learning import build_symbol_learning_signal


GO_SYMBOL = "GO"
GO_ALLOCATION_SEED = 23


class WorkingMemoryInterface:
    """Separate input vocabulary ``A/B/C/D/GO`` from output vocabulary."""

    def __init__(
        self,
        connectome,
        symbol_interface: SymbolInterface,
        *,
        go_population_size: int = 32,
        go_seed: int = GO_ALLOCATION_SEED,
    ) -> None:
        if go_population_size != 32:
            raise ValueError("F.2A GO population size is exactly 32")
        if symbol_interface.encoder.connectome is not connectome:
            raise ValueError("symbol interface and connectome must match")
        self.connectome = connectome
        self.symbol_interface = symbol_interface
        self.go_seed = int(go_seed)
        self.go_population_size = int(go_population_size)
        sensory = np.concatenate(tuple(symbol_interface.sensory_populations.values()))
        output = np.concatenate(tuple(symbol_interface.output_populations.values()))
        candidates = np.flatnonzero(np.diff(connectome.indptr) > 0).astype(np.int32, copy=False)
        excluded = np.unique(np.concatenate((sensory, output))).astype(np.int32, copy=False)
        candidates = candidates[~np.isin(candidates, excluded)]
        if len(candidates) < go_population_size:
            raise ValueError("not enough disjoint real sensory candidates for GO")
        rng = np.random.default_rng(self.go_seed)
        selected = np.sort(rng.permutation(candidates)[:go_population_size]).astype(np.int32, copy=False)
        selected.setflags(write=False)
        self._go_population = selected

    @property
    def input_vocabulary(self) -> tuple[str, ...]:
        return (*SYMBOLS, GO_SYMBOL)

    @property
    def output_vocabulary(self) -> tuple[str, ...]:
        return SYMBOLS

    @property
    def go_population(self) -> np.ndarray:
        return self._go_population.copy()

    @property
    def sensory_populations(self) -> dict[str, np.ndarray]:
        values = self.symbol_interface.sensory_populations
        values[GO_SYMBOL] = self.go_population
        return values

    @property
    def output_populations(self) -> dict[str, np.ndarray]:
        return self.symbol_interface.output_populations

    def population_for_input(self, label: str) -> np.ndarray:
        if label == GO_SYMBOL:
            return self.go_population
        if label in SYMBOLS:
            return self.symbol_interface.encoder.indices_for(label)
        raise KeyError(f"unknown working-memory input {label!r}")

    def allocation_summary(self) -> dict[str, object]:
        sensory = np.concatenate(tuple(self.symbol_interface.sensory_populations.values()))
        output = np.concatenate(tuple(self.output_populations.values()))
        go = self.go_population
        return {
            "go_population_size": int(len(go)),
            "go_seed": int(self.go_seed),
            "sensory_overlap": int(len(np.intersect1d(go, sensory))),
            "output_overlap": int(len(np.intersect1d(go, output))),
            "go_indices": [int(value) for value in go],
        }


@dataclass(frozen=True, slots=True)
class DelayedCueTrialResult:
    cue: str
    reset_before_go: bool
    decision: str
    cue_output_rates_hz: dict[str, float]
    delay_output_rates_hz: dict[str, float]
    go_output_rates_hz: dict[str, float]
    cue_output_spikes: int
    delay_output_spikes: int
    go_output_spikes: int
    pre_go_active_indices: tuple[int, ...]
    pre_go_fingerprint: str
    pre_go_active_count: int
    pre_go_membrane_norm: float
    pre_go_conductance_norm: float
    post_reset_active_count: int
    post_reset_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cue": self.cue,
            "reset_before_go": bool(self.reset_before_go),
            "decision": self.decision,
            "cue_output_rates_hz": dict(self.cue_output_rates_hz),
            "delay_output_rates_hz": dict(self.delay_output_rates_hz),
            "go_output_rates_hz": dict(self.go_output_rates_hz),
            "cue_output_spikes": int(self.cue_output_spikes),
            "delay_output_spikes": int(self.delay_output_spikes),
            "go_output_spikes": int(self.go_output_spikes),
            "pre_go_fingerprint": self.pre_go_fingerprint,
            "pre_go_active_count": int(self.pre_go_active_count),
            "pre_go_membrane_norm": float(self.pre_go_membrane_norm),
            "pre_go_conductance_norm": float(self.pre_go_conductance_norm),
            "post_reset_active_count": int(self.post_reset_active_count),
            "post_reset_fingerprint": self.post_reset_fingerprint,
        }


class DelayedCueSession:
    """Run one-reset delayed-cue episodes without invoking learning."""

    cue_ms = 20.0
    delay_ms = 20.0
    go_ms = 20.0
    stimulus_rate_hz = 205.0

    def __init__(
        self,
        brain,
        interface: WorkingMemoryInterface,
        *,
        learning_enabled: bool = False,
        delay_ms: float | None = None,
    ):
        if brain.connectome is not interface.connectome:
            raise ValueError("brain and working-memory interface must match")
        if learning_enabled:
            raise NotImplementedError("F.2A is learning-disabled; defer learning to F.2B")
        self.brain = brain
        self.interface = interface
        self.learning_enabled = False
        self.delay_ms = self.__class__.delay_ms if delay_ms is None else float(delay_ms)
        if self.delay_ms < 0.0:
            raise ValueError("delay_ms must be >= 0")
        self._output_lookup = np.full(brain.connectome.neuron_count, -1, dtype=np.int8)
        for index, symbol in enumerate(SYMBOLS):
            self._output_lookup[interface.output_populations[symbol]] = index

    def _run_phase(self, stimulus_indices, duration_ms: float, stimulus_rate_hz: float):
        counts = np.zeros(len(SYMBOLS), dtype=np.int32)
        if duration_ms <= 0.0:
            return {symbol: 0.0 for symbol in SYMBOLS}, 0, 0
        steps = int(math.ceil(duration_ms / self.brain.params.dt_ms))
        total_spikes = 0
        for _ in range(steps):
            fired, _ = self.brain.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            total_spikes += int(len(fired))
            if len(fired):
                local = self._output_lookup[fired]
                local = local[local >= 0]
                if len(local):
                    np.add.at(counts, local, 1)
        seconds = duration_ms / 1000.0
        rates = {
            symbol: float(counts[index]) / seconds
            for index, symbol in enumerate(SYMBOLS)
        }
        return rates, int(counts.sum()), int(total_spikes)

    def _capture_pre_go_state(self) -> tuple[np.ndarray, str, float, float]:
        np_mod = self.brain.np
        candidates = np.asarray(
            getattr(self.brain, "_fast_active", np_mod.empty(0, dtype=np_mod.int32)),
            dtype=np_mod.int32,
        )
        if len(candidates):
            resting = float(self.brain.params.resting_mv)
            live = (
                (np_mod.abs(self.brain.v[candidates] - resting) > 1e-7)
                | (np_mod.abs(self.brain.g[candidates]) > 1e-7)
                | (self.brain.refractory_until[candidates] > self.brain.step_index)
            )
            active = candidates[live].copy()
        else:
            active = np_mod.empty(0, dtype=np_mod.int32)
        active = np_mod.sort(active)
        fingerprint = hashlib.sha256(active.tobytes()).hexdigest()
        membrane_norm = float(np_mod.linalg.norm(self.brain.v[active] - self.brain.params.resting_mv)) if len(active) else 0.0
        conductance_norm = float(np_mod.linalg.norm(self.brain.g[active])) if len(active) else 0.0
        return active, fingerprint, membrane_norm, conductance_norm

    def run_trial(
        self,
        cue: str,
        *,
        reset_before_go: bool = False,
        track_eligibility: bool = False,
        retain_tracking: bool = False,
        phase_observer=None,
    ) -> DelayedCueTrialResult:
        if cue not in SYMBOLS:
            raise KeyError(f"unknown cue {cue!r}")
        previous_tracking = self.brain.set_plasticity_tracking(track_eligibility)
        try:
            self.brain.reset()
            cue_rates, cue_output_spikes, _ = self._run_phase(
                self.interface.population_for_input(cue), self.cue_ms, self.stimulus_rate_hz
            )
            if phase_observer is not None:
                phase_observer("cue", self.brain)
            delay_rates, delay_output_spikes, _ = self._run_phase(None, self.delay_ms, 0.0)
            if phase_observer is not None:
                phase_observer("delay", self.brain)
            active, fingerprint, membrane_norm, conductance_norm = self._capture_pre_go_state()
            post_reset_count = len(active)
            post_reset_fingerprint = fingerprint
            if reset_before_go:
                self.brain.reset()
                post_reset_active, post_reset_fingerprint, _, _ = self._capture_pre_go_state()
                post_reset_count = int(len(post_reset_active))
            go_rates, go_output_spikes, _ = self._run_phase(
                self.interface.population_for_input(GO_SYMBOL), self.go_ms, self.stimulus_rate_hz
            )
            if phase_observer is not None:
                phase_observer("go", self.brain)
            decision = self.interface.symbol_interface.decision_surface.decide(
                go_rates, total_output_spikes=go_output_spikes
            )
            return DelayedCueTrialResult(
                cue=cue,
                reset_before_go=bool(reset_before_go),
                decision=decision,
                cue_output_rates_hz=cue_rates,
                delay_output_rates_hz=delay_rates,
                go_output_rates_hz=go_rates,
                cue_output_spikes=cue_output_spikes,
                delay_output_spikes=delay_output_spikes,
                go_output_spikes=go_output_spikes,
                pre_go_active_indices=tuple(int(value) for value in active),
                pre_go_fingerprint=fingerprint,
                pre_go_active_count=int(len(active)),
                pre_go_membrane_norm=membrane_norm,
                pre_go_conductance_norm=conductance_norm,
                post_reset_active_count=post_reset_count,
                post_reset_fingerprint=post_reset_fingerprint,
            )
        finally:
            if not retain_tracking:
                self.brain.set_plasticity_tracking(previous_tracking)

    @staticmethod
    def build_learning_signal(result: DelayedCueTrialResult):
        """Prepare the future F.2B signal without applying it in F.2A."""
        return build_symbol_learning_signal(
            target=result.cue,
            decision=result.decision,
            output_rates_hz=result.go_output_rates_hz,
        )


def pairwise_set_jaccard(rows: list[DelayedCueTrialResult]) -> dict[str, float]:
    by_cue: dict[str, set[int]] = {cue: set() for cue in SYMBOLS}
    for row in rows:
        by_cue[row.cue].update(row.pre_go_active_indices)
    return {
        f"{left}|{right}": float(len(by_cue[left] & by_cue[right]) / len(by_cue[left] | by_cue[right]))
        if by_cue[left] | by_cue[right] else 0.0
        for left, right in combinations(SYMBOLS, 2)
    }


__all__ = [
    "GO_ALLOCATION_SEED",
    "GO_SYMBOL",
    "DelayedCueSession",
    "DelayedCueTrialResult",
    "WorkingMemoryInterface",
    "pairwise_set_jaccard",
]
