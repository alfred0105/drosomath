from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
from math import log1p
from typing import Iterable

from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class ActivityBiasedCandidateConfig:
    """Configuration for low-cost activity-biased synapse regrowth candidates."""

    max_candidates: int = 256
    pool_size: int = 64
    usage_weight: float = 1.0
    reward_weight: float = 4.0
    recency_weight: float = 2.0
    allow_self_connections: bool = False

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


class ActivityBiasedCandidateGenerator:
    """Prefer new edges near neurons involved in useful recent activity.

    This generator is intentionally lightweight. It derives source and target
    scores from the synapses already tracked by DrosoMath, selects only small
    top-k neuron pools, then ranks candidate cross-products. That keeps the slow
    structural-plasticity loop practical without constructing an N x N matrix.
    """

    def __init__(
        self,
        *,
        config: ActivityBiasedCandidateConfig | None = None,
    ) -> None:
        self.config = config or ActivityBiasedCandidateConfig()

    def generate(
        self,
        synapses: Iterable[SynapseState],
        *,
        step: int,
        neuron_ids: Iterable[int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        if step < 0:
            raise ValueError("step must be >= 0")
        if self.config.max_candidates == 0:
            return ()

        alive = tuple(s for s in synapses if s.alive)
        existing = {(s.pre_id, s.post_id) for s in alive}
        nodes = {node for s in alive for node in (s.pre_id, s.post_id)}
        if neuron_ids is not None:
            nodes.update(neuron_ids)
        if len(nodes) < 2:
            return ()

        source_scores: dict[int, float] = {}
        target_scores: dict[int, float] = {}
        for synapse in alive:
            score = self._synapse_activity_score(synapse, step=step)
            source_scores[synapse.pre_id] = source_scores.get(synapse.pre_id, 0.0) + score
            target_scores[synapse.post_id] = target_scores.get(synapse.post_id, 0.0) + score

        pool_size = min(self.config.pool_size, len(nodes))
        source_pool = nlargest(
            pool_size,
            nodes,
            key=lambda node: (source_scores.get(node, 0.0), -node),
        )
        target_pool = nlargest(
            pool_size,
            nodes,
            key=lambda node: (target_scores.get(node, 0.0), -node),
        )

        ranked: list[tuple[float, int, int]] = []
        for pre_id in source_pool:
            for post_id in target_pool:
                if not self.config.allow_self_connections and pre_id == post_id:
                    continue
                if (pre_id, post_id) in existing:
                    continue

                pair_score = (
                    source_scores.get(pre_id, 0.0)
                    + target_scores.get(post_id, 0.0)
                )
                ranked.append((pair_score, pre_id, post_id))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return tuple(
            (pre_id, post_id)
            for _, pre_id, post_id in ranked[: self.config.max_candidates]
        )

    def _synapse_activity_score(self, synapse: SynapseState, *, step: int) -> float:
        usage = self.config.usage_weight * log1p(synapse.usage_count)
        positive_reward = self.config.reward_weight * max(0.0, synapse.reward_ema)

        recency = 0.0
        if synapse.last_used_step >= 0:
            age = max(0, step - synapse.last_used_step)
            recency = self.config.recency_weight / (1.0 + float(age))

        return usage + positive_reward + recency
