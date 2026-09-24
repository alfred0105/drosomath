from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .dual_brain import AdaptiveBridge, BridgeSynapseState


BridgeKey = tuple[str, int, str, int]


@dataclass(frozen=True, slots=True)
class BridgeStructuralConfig:
    """Slow-timescale fixed-budget pruning/regrowth for inter-brain bridges."""

    min_age_cycles: int = 2
    stale_steps: int = 32
    reward_threshold: float = 0.0
    protected_stability: float = 0.8
    max_rewire_per_cycle: int = 64
    regrow_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.min_age_cycles < 0:
            raise ValueError("min_age_cycles must be >= 0")
        if self.stale_steps < 0:
            raise ValueError("stale_steps must be >= 0")
        if self.max_rewire_per_cycle < 0:
            raise ValueError("max_rewire_per_cycle must be >= 0")
        if self.protected_stability < 0.0:
            raise ValueError("protected_stability must be >= 0")


@dataclass(frozen=True, slots=True)
class BridgeRewireResult:
    pruned: tuple[BridgeKey, ...]
    regrown: tuple[BridgeKey, ...]

    @property
    def changed(self) -> int:
        return len(self.pruned)


class BridgeStructuralPlasticityManager:
    """Reallocate a fixed sparse bridge budget between two brains.

    Candidate bridge endpoints are supplied externally so later routing logic can
    bias regrowth toward useful modules or gateway neurons. This manager is only
    responsible for safe pruning, endpoint validation, and exact budget
    preservation.
    """

    def __init__(
        self,
        bridge: AdaptiveBridge,
        brain_neurons: Mapping[str, Iterable[int]],
        *,
        config: BridgeStructuralConfig | None = None,
    ) -> None:
        self.bridge = bridge
        self.config = config or BridgeStructuralConfig()
        self._brain_neurons = {
            name: frozenset(neuron_ids) for name, neuron_ids in brain_neurons.items()
        }
        if len(self._brain_neurons) < 2:
            raise ValueError("at least two brains are required")

    def rewire(
        self,
        *,
        step: int,
        candidate_keys: Iterable[BridgeKey],
    ) -> BridgeRewireResult:
        if step < 0:
            raise ValueError("step must be >= 0")

        current = self.bridge.synapses
        for synapse in current:
            if synapse.alive:
                synapse.age += 1

        prune_candidates = sorted(
            (synapse for synapse in current if self._eligible_for_prune(synapse, step)),
            key=self._prune_sort_key,
        )
        prune_limit = min(len(prune_candidates), self.config.max_rewire_per_cycle)
        if prune_limit == 0:
            return BridgeRewireResult(pruned=(), regrown=())

        existing = {self._key(synapse) for synapse in current if synapse.alive}
        regrow_keys: list[BridgeKey] = []
        seen: set[BridgeKey] = set()

        for key in candidate_keys:
            if len(regrow_keys) >= prune_limit:
                break
            if not self._valid_candidate(key):
                continue
            if key in existing or key in seen:
                continue
            seen.add(key)
            regrow_keys.append(key)

        # Never shrink the bridge merely because replacement candidates are
        # temporarily unavailable.
        prune_selected = prune_candidates[: len(regrow_keys)]
        if not prune_selected:
            return BridgeRewireResult(pruned=(), regrown=())

        pruned_keys: list[BridgeKey] = []
        for synapse in prune_selected:
            key = self._key(synapse)
            removed = self.bridge.unregister(key)
            if removed is None:
                continue
            removed.alive = False
            pruned_keys.append(key)

        regrow_keys = regrow_keys[: len(pruned_keys)]
        for source_brain, pre_id, target_brain, post_id in regrow_keys:
            self.bridge.register(
                BridgeSynapseState(
                    source_brain=source_brain,
                    pre_id=pre_id,
                    target_brain=target_brain,
                    post_id=post_id,
                    weight=self.config.regrow_weight,
                )
            )

        return BridgeRewireResult(
            pruned=tuple(pruned_keys),
            regrown=tuple(regrow_keys),
        )

    def _eligible_for_prune(self, synapse: BridgeSynapseState, step: int) -> bool:
        if not synapse.alive:
            return False
        if synapse.age < self.config.min_age_cycles:
            return False
        if synapse.stability >= self.config.protected_stability:
            return False
        if synapse.reward_ema > self.config.reward_threshold:
            return False
        if synapse.last_used_step < 0:
            return True
        return step - synapse.last_used_step >= self.config.stale_steps

    def _valid_candidate(self, key: BridgeKey) -> bool:
        source_brain, pre_id, target_brain, post_id = key
        if source_brain == target_brain:
            return False
        source_ids = self._brain_neurons.get(source_brain)
        target_ids = self._brain_neurons.get(target_brain)
        if source_ids is None or target_ids is None:
            return False
        return pre_id in source_ids and post_id in target_ids

    @staticmethod
    def _key(synapse: BridgeSynapseState) -> BridgeKey:
        return (
            synapse.source_brain,
            synapse.pre_id,
            synapse.target_brain,
            synapse.post_id,
        )

    @staticmethod
    def _prune_sort_key(
        synapse: BridgeSynapseState,
    ) -> tuple[float, int, float, int]:
        return (
            synapse.reward_ema,
            synapse.usage_count,
            synapse.weight,
            synapse.last_used_step,
        )
