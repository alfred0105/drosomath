from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from .neuron import SpikingNetwork, StepResult
from .plasticity import PlasticityTracker


@dataclass(slots=True)
class BridgeSynapseState:
    source_brain: str
    pre_id: int
    target_brain: str
    post_id: int
    weight: float
    stability: float = 0.0
    usage_count: int = 0
    reward_ema: float = 0.0
    last_used_step: int = -1
    alive: bool = True


@dataclass(frozen=True, slots=True)
class BridgeUse:
    step: int
    key: tuple[str, int, str, int]


class AdaptiveBridge:
    """Sparse reward-modulated communication between otherwise independent brains."""

    def __init__(
        self,
        synapses: Iterable[BridgeSynapseState] = (),
        *,
        reward_window: int = 32,
        reward_alpha: float = 0.05,
        learning_rate: float = 0.01,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> None:
        if reward_window < 0:
            raise ValueError("reward_window must be >= 0")
        if not 0.0 < reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be in (0, 1]")
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        if min_weight > max_weight:
            raise ValueError("min_weight must be <= max_weight")

        self.reward_window = reward_window
        self.reward_alpha = reward_alpha
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._synapses: dict[tuple[str, int, str, int], BridgeSynapseState] = {}
        self._recent: deque[BridgeUse] = deque()
        for synapse in synapses:
            self.register(synapse)

    @property
    def synapses(self) -> tuple[BridgeSynapseState, ...]:
        return tuple(self._synapses.values())

    def register(self, synapse: BridgeSynapseState) -> None:
        if synapse.source_brain == synapse.target_brain:
            raise ValueError("bridge synapse must connect different brains")
        key = self._key(synapse)
        if key in self._synapses:
            raise ValueError(f"duplicate bridge synapse {key}")
        self._synapses[key] = synapse

    def transfer(
        self,
        *,
        source_brain: str,
        fired: Iterable[int],
        target_networks: Mapping[str, SpikingNetwork],
        step: int,
    ) -> int:
        self._evict_old(step)
        fired_set = set(fired)
        if not fired_set:
            return 0

        transferred = 0
        for key, synapse in self._synapses.items():
            if not synapse.alive:
                continue
            if synapse.source_brain != source_brain or synapse.pre_id not in fired_set:
                continue
            target = target_networks.get(synapse.target_brain)
            if target is None:
                raise KeyError(f"unknown target brain {synapse.target_brain}")
            if synapse.post_id not in target.neurons:
                raise KeyError(
                    f"bridge target neuron {synapse.target_brain}:{synapse.post_id} does not exist"
                )

            target.inject({synapse.post_id: synapse.weight})
            synapse.usage_count += 1
            synapse.last_used_step = step
            self._recent.append(BridgeUse(step=step, key=key))
            transferred += 1
        return transferred

    def apply_reward(self, *, reward: float, step: int) -> int:
        self._evict_old(step)
        credited: set[tuple[str, int, str, int]] = set()
        for event in self._recent:
            if event.key in credited:
                continue
            synapse = self._synapses.get(event.key)
            if synapse is None or not synapse.alive:
                continue
            synapse.reward_ema += self.reward_alpha * (reward - synapse.reward_ema)
            synapse.weight = min(
                self.max_weight,
                max(self.min_weight, synapse.weight + self.learning_rate * reward),
            )
            credited.add(event.key)
        return len(credited)

    @staticmethod
    def _key(synapse: BridgeSynapseState) -> tuple[str, int, str, int]:
        return (
            synapse.source_brain,
            synapse.pre_id,
            synapse.target_brain,
            synapse.post_id,
        )

    def _evict_old(self, step: int) -> None:
        cutoff = step - self.reward_window
        while self._recent and self._recent[0].step < cutoff:
            self._recent.popleft()


@dataclass(slots=True)
class BrainCore:
    name: str
    network: SpikingNetwork
    tracker: PlasticityTracker

    def __post_init__(self) -> None:
        if self.network.tracker is not self.tracker:
            raise ValueError("BrainCore network and tracker must reference the same tracker")


@dataclass(frozen=True, slots=True)
class DualBrainStepResult:
    step: int
    brain_a: StepResult
    brain_b: StepResult
    bridge_transfers: int


class DualBrainSystem:
    """Two independent spiking brains connected only through a sparse bridge."""

    def __init__(
        self,
        brain_a: BrainCore,
        brain_b: BrainCore,
        bridge: AdaptiveBridge,
    ) -> None:
        if brain_a.name == brain_b.name:
            raise ValueError("brain names must be different")
        self.brain_a = brain_a
        self.brain_b = brain_b
        self.bridge = bridge
        self.step_index = 0
        self._networks = {
            brain_a.name: brain_a.network,
            brain_b.name: brain_b.network,
        }

    def step(
        self,
        *,
        currents_a: Mapping[int, float] | None = None,
        currents_b: Mapping[int, float] | None = None,
    ) -> DualBrainStepResult:
        result_a = self.brain_a.network.step(currents_a)
        result_b = self.brain_b.network.step(currents_b)

        transfers = self.bridge.transfer(
            source_brain=self.brain_a.name,
            fired=result_a.fired,
            target_networks=self._networks,
            step=self.step_index,
        )
        transfers += self.bridge.transfer(
            source_brain=self.brain_b.name,
            fired=result_b.fired,
            target_networks=self._networks,
            step=self.step_index,
        )

        result = DualBrainStepResult(
            step=self.step_index,
            brain_a=result_a,
            brain_b=result_b,
            bridge_transfers=transfers,
        )
        self.step_index += 1
        return result

    def apply_reward(self, reward: float) -> tuple[int, int, int]:
        """Apply one shared outcome without assigning fixed roles to either brain."""
        step = self.step_index
        credited_a = self.brain_a.tracker.apply_reward(reward=reward, step=step)
        credited_b = self.brain_b.tracker.apply_reward(reward=reward, step=step)
        credited_bridge = self.bridge.apply_reward(reward=reward, step=step)
        return credited_a, credited_b, credited_bridge
