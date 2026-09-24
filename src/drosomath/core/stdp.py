from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class STDPRule:
    """Pair-based spike-timing-dependent plasticity with bounded weights."""

    potentiation_rate: float = 0.01
    depression_rate: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    window: int = 40
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.potentiation_rate < 0.0:
            raise ValueError("potentiation_rate must be >= 0")
        if self.depression_rate < 0.0:
            raise ValueError("depression_rate must be >= 0")
        if self.tau_plus <= 0.0 or self.tau_minus <= 0.0:
            raise ValueError("STDP time constants must be > 0")
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")

    def delta(self, delta_t: int) -> float:
        """Return weight change for t_post - t_pre."""
        if delta_t == 0 or abs(delta_t) > self.window:
            return 0.0
        if delta_t > 0:
            return self.potentiation_rate * exp(-delta_t / self.tau_plus)
        return -self.depression_rate * exp(delta_t / self.tau_minus)

    def apply(self, synapse: SynapseState, *, delta_t: int) -> float:
        if not synapse.alive:
            return 0.0
        change = self.delta(delta_t)
        if change == 0.0:
            return 0.0
        old_weight = synapse.weight
        synapse.weight = min(
            self.max_weight,
            max(self.min_weight, old_weight + change),
        )
        return synapse.weight - old_weight


class STDPPlasticity:
    """Event-driven STDP state that remembers only each neuron's last spike."""

    def __init__(self, *, rule: STDPRule | None = None) -> None:
        self.rule = rule or STDPRule()
        self._last_spike: dict[int, int] = {}

    @property
    def last_spikes(self) -> dict[int, int]:
        return dict(self._last_spike)

    def reset(self) -> None:
        self._last_spike.clear()

    def observe_spikes(
        self,
        fired: Iterable[int],
        synapses: Iterable[SynapseState],
        *,
        step: int,
    ) -> int:
        if step < 0:
            raise ValueError("step must be >= 0")
        fired_set = set(fired)
        if not fired_set:
            return 0

        updates = 0
        for synapse in synapses:
            if not synapse.alive:
                continue

            # Causal pair: pre fired previously, post fires now -> potentiate.
            if synapse.post_id in fired_set:
                pre_step = self._last_spike.get(synapse.pre_id)
                if pre_step is not None and pre_step < step:
                    if self.rule.apply(synapse, delta_t=step - pre_step) != 0.0:
                        updates += 1

            # Anti-causal pair: post fired previously, pre fires now -> depress.
            if synapse.pre_id in fired_set:
                post_step = self._last_spike.get(synapse.post_id)
                if post_step is not None and post_step < step:
                    if self.rule.apply(synapse, delta_t=post_step - step) != 0.0:
                        updates += 1

        for neuron_id in fired_set:
            self._last_spike[neuron_id] = step
        return updates
