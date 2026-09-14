from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .plasticity import PlasticityTracker
from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class StructuralPlasticityConfig:
    """Slow-timescale pruning/regrowth policy with a fixed synapse budget."""

    min_age_cycles: int = 2
    stale_steps: int = 32
    reward_threshold: float = 0.0
    protected_stability: float = 0.8
    max_rewire_per_cycle: int = 128
    regrow_weight: float = 0.05
    allow_self_connections: bool = False

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
class RewireResult:
    pruned: tuple[tuple[int, int], ...]
    regrown: tuple[tuple[int, int], ...]

    @property
    def changed(self) -> int:
        return len(self.pruned)


class StructuralPlasticityManager:
    """Prune low-value stale synapses and regrow the same number elsewhere.

    The manager never grows the registered synapse count during a rewire cycle.
    Candidate pairs are supplied from outside so later phases can bias regrowth
    toward active modules, bridge gateways, or exploratory connections without
    changing this budget-preserving core.
    """

    def __init__(
        self,
        tracker: PlasticityTracker,
        *,
        config: StructuralPlasticityConfig | None = None,
    ) -> None:
        self.tracker = tracker
        self.config = config or StructuralPlasticityConfig()

    def rewire(
        self,
        *,
        step: int,
        candidate_pairs: Iterable[tuple[int, int]],
    ) -> RewireResult:
        """Run one slow structural-plasticity cycle."""
        if step < 0:
            raise ValueError("step must be >= 0")

        current = self.tracker.synapses
        for synapse in current:
            synapse.tick()

        prune_candidates = sorted(
            (synapse for synapse in current if self._eligible_for_prune(synapse, step)),
            key=self._prune_sort_key,
        )
        prune_limit = min(
            len(prune_candidates),
            self.config.max_rewire_per_cycle,
        )
        if prune_limit == 0:
            return RewireResult(pruned=(), regrown=())

        existing = {
            (synapse.pre_id, synapse.post_id)
            for synapse in current
            if synapse.alive
        }
        regrow_pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        for pre_id, post_id in candidate_pairs:
            key = (pre_id, post_id)
            if len(regrow_pairs) >= prune_limit:
                break
            if not self.config.allow_self_connections and pre_id == post_id:
                continue
            if key in existing or key in seen:
                continue
            seen.add(key)
            regrow_pairs.append(key)

        # Preserve the budget exactly: never prune more than can be regrown now.
        prune_selected = prune_candidates[: len(regrow_pairs)]
        if not prune_selected:
            return RewireResult(pruned=(), regrown=())

        pruned_keys: list[tuple[int, int]] = []
        for synapse in prune_selected:
            key = (synapse.pre_id, synapse.post_id)
            removed = self.tracker.unregister_synapse(
                pre_id=synapse.pre_id,
                post_id=synapse.post_id,
            )
            if removed is None:
                continue
            removed.prune()
            pruned_keys.append(key)

        # In normal operation these counts match. Slice defensively if a caller
        # modified the tracker during the cycle.
        regrow_pairs = regrow_pairs[: len(pruned_keys)]
        for pre_id, post_id in regrow_pairs:
            self.tracker.register_synapse(
                SynapseState(
                    pre_id=pre_id,
                    post_id=post_id,
                    weight=self.config.regrow_weight,
                )
            )

        return RewireResult(
            pruned=tuple(pruned_keys),
            regrown=tuple(regrow_pairs),
        )

    def _eligible_for_prune(self, synapse: SynapseState, step: int) -> bool:
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

    @staticmethod
    def _prune_sort_key(synapse: SynapseState) -> tuple[float, int, float, int]:
        """Prefer low-reward, little-used, weak, old-unused connections first."""
        return (
            synapse.reward_ema,
            synapse.usage_count,
            synapse.weight,
            synapse.last_used_step,
        )
