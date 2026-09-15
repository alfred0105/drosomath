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
    Plasticity tracking can be disabled while an external decoder is being
    trained, so that readout pretraining does not contaminate the brain state.

    Live telemetry is opt-in. The simulator still computes the full sparse
    graph, but only a capped sample of the strongest recently transmitted edges
    is copied for visualization. This avoids turning a 6M-edge simulation into
    a browser-rendering benchmark.
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
        self.plasticity_tracking_enabled = True
        self._recent_presynaptic: set[int] = set()

        self._telemetry_enabled = False
        self._telemetry_max_edges = 128
        self._telemetry_edges_per_neuron = 4
        self._last_telemetry: dict[str, object] = {
            "step": 0,
            "fired_neuron_ids": [],
            "fired_neuron_count": 0,
            "transferred_synapses": 0,
            "active_edges": [],
        }

    def set_plasticity_tracking(self, enabled: bool) -> bool:
        """Enable/disable usage and eligibility recording.

        Returns the previous state so callers can restore it in a ``finally``
        block. Disabling tracking never changes the anatomical graph or learned
        multipliers.
        """
        previous = self.plasticity_tracking_enabled
        self.plasticity_tracking_enabled = bool(enabled)
        if not self.plasticity_tracking_enabled:
            self._recent_presynaptic.clear()
            self.plasticity.clear_eligibility()
        return previous

    def configure_live_telemetry(
        self,
        enabled: bool = True,
        *,
        max_active_edges: int = 128,
        edges_per_firing_neuron: int = 4,
    ) -> None:
        """Enable a bounded visualization sample of recent spike transfers."""
        if max_active_edges < 1:
            raise ValueError("max_active_edges must be >= 1")
        if edges_per_firing_neuron < 1:
            raise ValueError("edges_per_firing_neuron must be >= 1")
        self._telemetry_enabled = bool(enabled)
        self._telemetry_max_edges = int(max_active_edges)
        self._telemetry_edges_per_neuron = int(edges_per_firing_neuron)
        if not self._telemetry_enabled:
            self._last_telemetry = {
                "step": int(self.step_index),
                "fired_neuron_ids": [],
                "fired_neuron_count": 0,
                "transferred_synapses": 0,
                "active_edges": [],
            }

    def live_telemetry_snapshot(self) -> dict[str, object]:
        """Return the latest immutable-ish telemetry payload for UI polling."""
        snap = self._last_telemetry
        return {
            "step": int(snap["step"]),
            "fired_neuron_ids": list(snap["fired_neuron_ids"]),
            "fired_neuron_count": int(snap["fired_neuron_count"]),
            "transferred_synapses": int(snap["transferred_synapses"]),
            "active_edges": [dict(row) for row in snap["active_edges"]],
        }

    def reset(self) -> None:
        """Reset fast neural state while deliberately preserving learned memory."""
        super().reset()
        self._recent_presynaptic.clear()

    def reset_all(self) -> None:
        """Reset both neural dynamics and learned synaptic state."""
        self.reset()
        self.plasticity.reset_learning_state()

    def _neuron_identifier(self, index: int) -> int:
        ids = getattr(self.connectome, "body_ids", None)
        if ids is None:
            ids = getattr(self.connectome, "flywire_ids")
        return int(ids[int(index)])

    def _sample_live_edges(self, *, pre: int, start: int, stop: int, effective) -> list[dict[str, object]]:
        np = self.np
        count = stop - start
        if count <= 0:
            return []
        k = min(self._telemetry_edges_per_neuron, count)
        magnitude = np.abs(effective)
        if count <= k:
            local = np.arange(count, dtype=np.int32)
        else:
            local = np.argpartition(magnitude, -k)[-k:]
            local = local[np.argsort(magnitude[local])[::-1]]

        rows: list[dict[str, object]] = []
        posts = self.connectome.post_indices
        base = self.connectome.signed_synapse_counts
        scale = self.params.mv_per_synapse
        for rel_raw in local:
            rel = int(rel_raw)
            edge = start + rel
            post = int(posts[edge])
            rows.append(
                {
                    "edge_index": int(edge),
                    "pre_id": self._neuron_identifier(pre),
                    "post_id": self._neuron_identifier(post),
                    "base_signed_synapses": float(base[edge]),
                    "multiplier": float(self.plasticity.multiplier[edge]),
                    "signal_mv": float(effective[rel] * scale),
                    "stability": float(self.plasticity.stability[edge]),
                    "usage_ema": float(self.plasticity.usage_ema[edge]),
                }
            )
        return rows

    def _schedule_spike_outputs(self, fired_indices) -> int:
        np = self.np
        if len(fired_indices) == 0:
            if self._telemetry_enabled:
                self._last_telemetry = {
                    "step": int(self.step_index),
                    "fired_neuron_ids": [],
                    "fired_neuron_count": 0,
                    "transferred_synapses": 0,
                    "active_edges": [],
                }
            return 0

        target_slot = self._delay_ring[
            (self.step_index + self.delay_steps) % len(self._delay_ring)
        ]
        transferred = 0
        scale = self.params.mv_per_synapse
        indptr = self.connectome.indptr
        posts = self.connectome.post_indices
        base_signed = self.connectome.signed_synapse_counts
        telemetry_edges: list[dict[str, object]] = []

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

            if self._telemetry_enabled and len(telemetry_edges) < self._telemetry_max_edges:
                telemetry_edges.extend(
                    self._sample_live_edges(pre=pre, start=start, stop=stop, effective=effective)
                )

            if self.plasticity_tracking_enabled:
                self.plasticity.record_use_slice(
                    start,
                    stop,
                    usage_alpha=self.usage_alpha,
                    eligibility_gain=self.eligibility_gain,
                )
                self._recent_presynaptic.add(pre)
            transferred += stop - start

        if self._telemetry_enabled:
            if len(telemetry_edges) > self._telemetry_max_edges:
                telemetry_edges = sorted(
                    telemetry_edges,
                    key=lambda row: abs(float(row["signal_mv"])),
                    reverse=True,
                )[: self._telemetry_max_edges]
            fired_ids = [self._neuron_identifier(int(x)) for x in fired_indices[:256]]
            self._last_telemetry = {
                "step": int(self.step_index),
                "fired_neuron_ids": fired_ids,
                "fired_neuron_count": int(len(fired_indices)),
                "transferred_synapses": int(transferred),
                "active_edges": telemetry_edges,
            }

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
        if not self.plasticity_tracking_enabled:
            raise RuntimeError("cannot learn from reward while plasticity tracking is disabled")

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