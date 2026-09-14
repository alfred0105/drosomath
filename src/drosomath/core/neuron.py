from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .plasticity import PlasticityTracker
from .synapse import SynapseState


@dataclass(slots=True)
class NeuronState:
    neuron_id: int
    threshold: float = 1.0
    decay: float = 0.95
    reset_potential: float = 0.0
    potential: float = 0.0
    fired: bool = False
    last_spike_step: int = -1

    def __post_init__(self) -> None:
        if self.threshold <= 0.0:
            raise ValueError("threshold must be > 0")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class StepResult:
    step: int
    fired: tuple[int, ...]
    transferred_synapses: int


class SpikingNetwork:
    """Executable reference leaky integrate-and-fire network.

    This engine prioritizes correctness and inspectability for small networks.
    A future sparse/tensor backend can reuse the same plasticity interfaces.
    """

    def __init__(
        self,
        neurons: Iterable[NeuronState],
        tracker: PlasticityTracker,
    ) -> None:
        neuron_list = tuple(neurons)
        if not neuron_list:
            raise ValueError("at least one neuron is required")
        ids = [neuron.neuron_id for neuron in neuron_list]
        if len(ids) != len(set(ids)):
            raise ValueError("neuron IDs must be unique")

        self.neurons = {neuron.neuron_id: neuron for neuron in neuron_list}
        self.tracker = tracker
        self.step_index = 0
        self._pending_current: dict[int, float] = {}
        self._outgoing: dict[int, list[SynapseState]] = {}
        self._seen_topology_version = -1
        self.refresh_topology()

    @classmethod
    def from_ids(
        cls,
        neuron_ids: Iterable[int],
        tracker: PlasticityTracker,
        *,
        threshold: float = 1.0,
        decay: float = 0.95,
    ) -> "SpikingNetwork":
        ids = tuple(neuron_ids)
        return cls(
            (NeuronState(neuron_id=i, threshold=threshold, decay=decay) for i in ids),
            tracker,
        )

    def refresh_topology(self) -> None:
        outgoing: dict[int, list[SynapseState]] = {}
        for synapse in self.tracker.synapses:
            if not synapse.alive:
                continue
            if synapse.pre_id not in self.neurons or synapse.post_id not in self.neurons:
                raise ValueError(
                    f"synapse ({synapse.pre_id}, {synapse.post_id}) references unknown neuron"
                )
            outgoing.setdefault(synapse.pre_id, []).append(synapse)
        self._outgoing = outgoing
        self._seen_topology_version = self.tracker.topology_version

    def inject(self, currents: Mapping[int, float]) -> None:
        for neuron_id, current in currents.items():
            if neuron_id not in self.neurons:
                raise KeyError(f"unknown neuron {neuron_id}")
            self._pending_current[neuron_id] = (
                self._pending_current.get(neuron_id, 0.0) + float(current)
            )

    def step(self, currents: Mapping[int, float] | None = None) -> StepResult:
        if self._seen_topology_version != self.tracker.topology_version:
            self.refresh_topology()
        if currents:
            self.inject(currents)

        for neuron_id, neuron in self.neurons.items():
            neuron.potential *= neuron.decay
            neuron.potential += self._pending_current.get(neuron_id, 0.0)

        fired: list[int] = []
        for neuron in self.neurons.values():
            neuron.fired = neuron.potential >= neuron.threshold
            if neuron.fired:
                neuron.last_spike_step = self.step_index
                fired.append(neuron.neuron_id)
                neuron.potential = neuron.reset_potential

        next_current: dict[int, float] = {}
        transferred = 0
        for pre_id in fired:
            for synapse in self._outgoing.get(pre_id, ()):
                if not synapse.alive:
                    continue
                if self.tracker.record_transfer(
                    pre_id=synapse.pre_id,
                    post_id=synapse.post_id,
                    step=self.step_index,
                ):
                    next_current[synapse.post_id] = (
                        next_current.get(synapse.post_id, 0.0) + synapse.weight
                    )
                    transferred += 1

        result = StepResult(
            step=self.step_index,
            fired=tuple(fired),
            transferred_synapses=transferred,
        )
        self._pending_current = next_current
        self.step_index += 1
        return result

    def run(
        self,
        steps: int,
        *,
        stimulus: Mapping[int, float] | None = None,
    ) -> tuple[StepResult, ...]:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        return tuple(
            self.step(stimulus if index == 0 else None)
            for index in range(steps)
        )
