from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Mapping

from .dual_brain import AdaptiveBridge, BrainCore
from .synapse import SynapseState


BridgeKey = tuple[str, int, str, int]


@dataclass(frozen=True, slots=True)
class BridgeActivityCandidateConfig:
    """Low-cost activity/reward-biased bridge candidate generation."""

    max_candidates: int = 128
    pool_size: int = 32
    usage_weight: float = 1.0
    reward_weight: float = 4.0
    recency_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be >= 0")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if self.usage_weight < 0.0:
            raise ValueError("usage_weight must be >= 0")
        if self.reward_weight < 0.0:
            raise ValueError("reward_weight must be >= 0")
        if self.recency_weight < 0.0:
            raise ValueError("recency_weight must be >= 0")


class BridgeActivityCandidateGenerator:
    """Generate sparse cross-brain edges near useful internal activity.

    Brain roles are not hard-coded. Neurons are ranked only by recent internal
    activity and reward history, then high-value endpoints from different brains
    are paired. Existing bridge edges are excluded.
    """

    def __init__(
        self,
        *,
        config: BridgeActivityCandidateConfig | None = None,
    ) -> None:
        self.config = config or BridgeActivityCandidateConfig()

    def generate(
        self,
        brains: Mapping[str, BrainCore],
        bridge: AdaptiveBridge,
        *,
        step: int,
    ) -> tuple[BridgeKey, ...]:
        if step < 0:
            raise ValueError("step must be >= 0")
        if self.config.max_candidates == 0 or len(brains) < 2:
            return ()

        existing = {
            (
                synapse.source_brain,
                synapse.pre_id,
                synapse.target_brain,
                synapse.post_id,
            )
            for synapse in bridge.synapses
            if synapse.alive
        }

        ranked_neurons: dict[str, list[tuple[float, int]]] = {}
        for name, brain in brains.items():
            scores = self._neuron_scores(brain, step=step)
            ranked = sorted(
                ((scores.get(neuron_id, 0.0), neuron_id) for neuron_id in brain.network.neurons),
                key=lambda item: (-item[0], item[1]),
            )
            ranked_neurons[name] = ranked[: self.config.pool_size]

        candidates: list[tuple[float, str, int, str, int]] = []
        names = sorted(brains)
        for source_name in names:
            for target_name in names:
                if source_name == target_name:
                    continue
                for source_score, pre_id in ranked_neurons[source_name]:
                    for target_score, post_id in ranked_neurons[target_name]:
                        key = (source_name, pre_id, target_name, post_id)
                        if key in existing:
                            continue
                        pair_score = source_score + target_score
                        candidates.append(
                            (pair_score, source_name, pre_id, target_name, post_id)
                        )

        candidates.sort(
            key=lambda item: (-item[0], item[1], item[2], item[3], item[4])
        )
        return tuple(
            (source_name, pre_id, target_name, post_id)
            for _, source_name, pre_id, target_name, post_id in candidates[
                : self.config.max_candidates
            ]
        )

    def _neuron_scores(self, brain: BrainCore, *, step: int) -> dict[int, float]:
        scores: dict[int, float] = {}
        for synapse in brain.tracker.synapses:
            if not synapse.alive:
                continue
            score = self._synapse_score(synapse, step=step)
            scores[synapse.pre_id] = scores.get(synapse.pre_id, 0.0) + score
            scores[synapse.post_id] = scores.get(synapse.post_id, 0.0) + score
        return scores

    def _synapse_score(self, synapse: SynapseState, *, step: int) -> float:
        usage = self.config.usage_weight * log1p(synapse.usage_count)
        reward = self.config.reward_weight * max(0.0, synapse.reward_ema)
        recency = 0.0
        if synapse.last_used_step >= 0:
            age = max(0, step - synapse.last_used_step)
            recency = self.config.recency_weight / (1.0 + float(age))
        return usage + reward + recency
