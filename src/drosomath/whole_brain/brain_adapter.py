from __future__ import annotations

from dataclasses import asdict

from drosomath.flywire_real import FlyBrainParams, FlyWireConnectome, SparseFlyBrain

from .homeostasis import OutgoingBudgetNormalizer
from .plastic_state import PlasticStateConfig, SparsePlasticityState
from .usage_learning import UsageRewardRule


class PlasticSparseFlyBrain(SparseFlyBrain):
    """Sparse whole-brain simulator with sign-preserving synaptic learning.

    The anatomical graph stays immutable in ``connectome``. Learned memory is
    stored in ``plasticity.multiplier`` and related compact float32 arrays.
    """

    def __init__(
        self,
        connectome: FlyWireConnectome,
        *,
        params: FlyBrainParams | None = None,
        seed: int = 0,
        plasticity: SparsePlasticityState | None = None,
        plasticity_config: PlasticStateConfig | None = None,
        usage_alpha: float = 0.05,
        eligibility_gain: float = 1.0,
    ) -> None:
        super().__init__(connectome, params=params, seed=seed)
        if not 0.0 < usage_alpha <= 1.0:
            raise ValueError("usage_alpha must be in (0, 1]")
        if eligibility_gain < 0.0:
            raise ValueError("eligibility_gain must be >= 0")

        if plasticity is None:
            plasticity = SparsePlasticityState(
                connectome.edge_count,
                config=plasticity_config,
            )
        elif plasticity.edge_count != connectome.edge_count:
            raise ValueError("plasticity edge count must match connectome edge count")

        self.plasticity = plasticity
        self.usage_alpha = usage_alpha
        self.eligibility_gain = eligibility_gain
        self._recent_presynaptic: set[int] = set()

    def reset(self) -> None:
        """Reset fast neural state while deliberately preserving learned memory."""
        super().reset()
        self._recent_presynaptic.clear()

    def reset_all(self) -> None:
        """Reset both neural dynamics and learned synaptic state."""
        self.reset()
        self.plasticity.reset_learning_state()

    def _schedule_spike_outputs(self, fired_indices) -> int:
        np = self.np
        if len(fired_indices) == 0:
            return 0

        target_slot = self._delay_ring[
            (self.step_index + self.delay_steps) % len(self._delay_ring)
        ]
        transferred = 0
        scale = self.params.mv_per_synapse
        indptr = self.connectome.indptr
        posts = self.connectome.post_indices
        base_signed = self.connectome.signed_synapse_counts

        for pre_raw in fired_indices:
            pre = int(pre_raw)
            start = int(indptr[pre])
            stop = int(indptr[pre + 1])
            if start == stop:
                continue

            effective = self.plasticity.effective_signed_slice(
                base_signed,
                start,
                stop,
            )
            np.add.at(target_slot, posts[start:stop], effective * scale)
            self.plasticity.record_use_slice(
                start,
                stop,
                usage_alpha=self.usage_alpha,
                eligibility_gain=self.eligibility_gain,
            )
            self._recent_presynaptic.add(pre)
            transferred += stop - start

        return transferred

    def learn_from_reward(
        self,
        *,
        reward: float,
        rule: UsageRewardRule,
        normalizer: OutgoingBudgetNormalizer | None = None,
        usage_decay: float = 0.995,
        eligibility_decay: float = 0.90,
        clear_eligibility: bool = True,
    ) -> dict[str, object]:
        """Turn recent synaptic use into long-term weight changes.

        Positive reward strengthens frequently used eligible routes. Negative
        reward weakens them, with consolidated (stable) routes changing less.
        Optional outgoing-budget normalization prevents runaway rich-get-richer
        dynamics.
        """
        update = rule.apply(self.plasticity, reward=reward)

        budget = None
        if normalizer is not None and self._recent_presynaptic:
            budget = normalizer.normalize_presynaptic(
                self.plasticity,
                indptr=self.connectome.indptr,
                base_abs=self.np.abs(self.connectome.signed_synapse_counts),
                presynaptic_indices=sorted(self._recent_presynaptic),
            )

        self.plasticity.decay_episode(
            usage_decay=usage_decay,
            eligibility_decay=eligibility_decay,
        )
        if clear_eligibility:
            self.plasticity.clear_eligibility()
        self._recent_presynaptic.clear()

        return {
            "reward": float(reward),
            "learning": asdict(update),
            "budget": asdict(budget) if budget is not None else None,
            "plasticity": self.plasticity.summary(),
        }
