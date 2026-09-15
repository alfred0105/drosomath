from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    min_usage: int = 3
    min_reward: float = 0.05
    growth_rate: float = 0.1
    decay_rate: float = 0.005
    max_stability: float = 1.0

    def __post_init__(self) -> None:
        if self.min_usage < 0:
            raise ValueError("min_usage must be >= 0")
        if self.growth_rate < 0.0:
            raise ValueError("growth_rate must be >= 0")
        if self.decay_rate < 0.0:
            raise ValueError("decay_rate must be >= 0")
        if self.max_stability <= 0.0:
            raise ValueError("max_stability must be > 0")


class MemoryConsolidator:
    """Turn repeatedly useful pathways into pruning-resistant long-term memory.

    Frequent use alone is not enough: a connection must also carry positive
    reward credit before its stability grows. Unreinforced connections slowly
    lose stability so the fixed resource budget can eventually be reused.
    """

    def __init__(self, *, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def consolidate(self, synapses: Iterable[SynapseState]) -> tuple[int, int]:
        strengthened = 0
        weakened = 0
        cfg = self.config

        for synapse in synapses:
            if not synapse.alive:
                continue
            if synapse.usage_count >= cfg.min_usage and synapse.reward_ema >= cfg.min_reward:
                reward_factor = min(1.0, max(0.0, synapse.reward_ema))
                delta = cfg.growth_rate * reward_factor * (
                    cfg.max_stability - synapse.stability
                )
                if delta > 0.0:
                    synapse.stability = min(cfg.max_stability, synapse.stability + delta)
                    strengthened += 1
            elif synapse.stability > 0.0 and cfg.decay_rate > 0.0:
                new_value = max(0.0, synapse.stability - cfg.decay_rate)
                if new_value < synapse.stability:
                    synapse.stability = new_value
                    weakened += 1

        return strengthened, weakened
