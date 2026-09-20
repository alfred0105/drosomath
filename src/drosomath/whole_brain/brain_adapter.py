from __future__ import annotations

import os
import time
from dataclasses import asdict

from drosomath.flywire_real import FlyBrainParams, FlyWireConnectome, SparseFlyBrain

from .homeostasis import OutgoingBudgetNormalizer
from .plastic_state import PlasticStateConfig, SparsePlasticityState
from .structural_overlay import LearnedStructuralOverlay, StructuralOverlayConfig
from .usage_learning import UsageRewardRule


class PlasticSparseFlyBrain(SparseFlyBrain):
    """Sparse whole-brain simulator with weight and structural plasticity.

    The anatomical graph stays immutable in ``connectome``. Learned anatomical
    weight memory is stored in ``plasticity``. Optional structural plasticity is
    stored in a separate learned-edge overlay; donor anatomical edges are only
    functionally silenced, never removed from the source connectome.
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
        # Anatomy is immutable for the lifetime of a brain instance.  Cache
        # this 6M-edge absolute-strength vector instead of rebuilding it on
        # every reward/homeostasis update.
        self._base_abs_synapse_counts = self.np.abs(
            connectome.signed_synapse_counts
        ).astype(self.np.float32, copy=False)
        self.usage_alpha = usage_alpha
        self.eligibility_gain = eligibility_gain
        self.plasticity_tracking_enabled = True
        self._recent_presynaptic: set[int] = set()
        self.structural_overlay: LearnedStructuralOverlay | None = None
        self._structural_reward_events = 0
        self.structural_rewire_interval = max(
            1,
            int(os.environ.get("DROSOMATH_STRUCTURAL_INTERVAL", "64")),
        )

        if os.environ.get("DROSOMATH_STRUCTURAL", "").strip().lower() in {"1", "true", "yes", "on"}:
            self.configure_structural_plasticity()

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

    def configure_structural_plasticity(
        self,
        config: StructuralOverlayConfig | None = None,
    ) -> LearnedStructuralOverlay:
        """Enable auditable learned-edge structural plasticity."""
        overlay = LearnedStructuralOverlay(self.connectome, self.plasticity, config=config)
        self.structural_overlay = overlay
        self._structural_reward_events = 0
        return overlay

    def run_structural_cycle(self, *, cycle_label: str = "") -> dict[str, object]:
        if self.structural_overlay is None:
            return {"enabled": False, "added": 0, "replaced": 0, "active_edges": 0}
        return self.structural_overlay.rewire(cycle_label=cycle_label)

    def structural_summary(self) -> dict[str, object]:
        if self.structural_overlay is None:
            return {"enabled": False, "active_edges": 0}
        return self.structural_overlay.summary()

    def set_plasticity_tracking(self, enabled: bool) -> bool:
        """Enable/disable usage and eligibility recording."""
        previous = self.plasticity_tracking_enabled
        self.plasticity_tracking_enabled = bool(enabled)
        if not self.plasticity_tracking_enabled:
            self._recent_presynaptic.clear()
            self.plasticity.clear_eligibility()
            if self.structural_overlay is not None:
                self.structural_overlay.eligibility.fill(0.0)
                self.structural_overlay._clear_trial_activity()
        return previous

    def configure_live_telemetry(
        self,
        enabled: bool = True,
        *,
        max_active_edges: int = 128,
        edges_per_firing_neuron: int = 4,
    ) -> None:
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
        snap = self._last_telemetry
        return {
            "step": int(snap["step"]),
            "fired_neuron_ids": list(snap["fired_neuron_ids"]),
            "fired_neuron_count": int(snap["fired_neuron_count"]),
            "transferred_synapses": int(snap["transferred_synapses"]),
            "active_edges": [dict(row) for row in snap["active_edges"]],
        }

    def reset(self) -> None:
        """Reset fast neural state while preserving learned memory/structure."""
        super().reset()
        self._recent_presynaptic.clear()

    def reset_all(self) -> None:
        """Reset neural dynamics, learned weights, and learned structure."""
        self.reset()
        if self.structural_overlay is not None:
            self.structural_overlay.reset()
        self._structural_reward_events = 0
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
                    "learned_structural": False,
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

    def _sample_structural_live_edges(self, *, pre: int, slots) -> list[dict[str, object]]:
        overlay = self.structural_overlay
        if overlay is None or len(slots) == 0:
            return []
        np = self.np
        k = min(self._telemetry_edges_per_neuron, len(slots))
        magnitude = np.abs(overlay.signed_strength[slots])
        if len(slots) <= k:
            chosen = slots
        else:
            local = np.argpartition(magnitude, -k)[-k:]
            chosen = slots[local[np.argsort(magnitude[local])[::-1]]]
        scale = self.params.mv_per_synapse
        rows = []
        for slot_raw in chosen:
            slot = int(slot_raw)
            post = int(overlay.post_index[slot])
            rows.append(
                {
                    "edge_index": -1,
                    "structural_slot": slot,
                    "learned_structural": True,
                    "pre_id": self._neuron_identifier(pre),
                    "post_id": self._neuron_identifier(post),
                    "base_signed_synapses": 0.0,
                    "multiplier": 1.0,
                    "signal_mv": float(overlay.signed_strength[slot] * scale),
                    "stability": float(overlay.stability[slot]),
                    "usage_ema": float(overlay.usage_ema[slot]),
                    "donor_edge": int(overlay.donor_edge[slot]),
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

        if self.plasticity_tracking_enabled and self.structural_overlay is not None:
            self.structural_overlay.record_firing(fired_indices)

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
            if stop > start:
                effective = self.plasticity.effective_signed_slice(base_signed, start, stop)
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

            overlay = self.structural_overlay
            if overlay is not None:
                structural_slots = overlay.slots_for_pre(pre)
                if len(structural_slots):
                    structural_posts = overlay.post_index[structural_slots]
                    structural_strength = overlay.signed_strength[structural_slots]
                    np.add.at(target_slot, structural_posts, structural_strength * scale)
                    if self.plasticity_tracking_enabled:
                        overlay.record_use(structural_slots)
                    if self._telemetry_enabled and len(telemetry_edges) < self._telemetry_max_edges:
                        telemetry_edges.extend(
                            self._sample_structural_live_edges(pre=pre, slots=structural_slots)
                        )
                    transferred += len(structural_slots)

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
        include_plasticity_summary: bool = True,
        profile_timing: bool = False,
        post_reward_hook=None,
        normalizer_observer=None,
        reward_credit=None,
    ) -> dict[str, object]:
        """Turn recent synaptic use into long-term weight/structural changes."""
        if not self.plasticity_tracking_enabled:
            raise RuntimeError("cannot learn from reward while plasticity tracking is disabled")

        timings: dict[str, float] | None = {} if profile_timing else None
        started = time.perf_counter() if profile_timing else 0.0
        recent_presynaptic = sorted(self._recent_presynaptic)
        reward_started = time.perf_counter() if timings is not None else 0.0
        if clear_eligibility:
            update = rule.apply_recent_presynaptic(
                self.plasticity,
                reward=reward,
                indptr=self.connectome.indptr,
                presynaptic_indices=recent_presynaptic,
                reward_credit=reward_credit,
            )
        else:
            # A caller retaining eligibility may intentionally credit traces
            # from earlier trials, so retain the complete reference scan.
            update = rule.apply(self.plasticity, reward=reward, reward_credit=reward_credit)
        if timings is not None:
            timings["reward_update_seconds"] = time.perf_counter() - reward_started
        structural_learning = None
        structural_rewire = None
        if self.structural_overlay is not None:
            structural_learning = self.structural_overlay.learn_from_reward(float(reward))
            self._structural_reward_events += 1
            if self._structural_reward_events % self.structural_rewire_interval == 0:
                structural_rewire = self.run_structural_cycle(
                    cycle_label=f"reward_event_{self._structural_reward_events}"
                )

        # A task-specific local teacher may use current-trial eligibility, but
        # it must run after the global reward update and before normalization,
        # decay, and the single lifecycle clear below.
        post_started = time.perf_counter() if timings is not None else 0.0
        post_reward = (
            post_reward_hook(self.plasticity)
            if post_reward_hook is not None
            else None
        )
        if timings is not None:
            timings["post_reward_directional_seconds"] = time.perf_counter() - post_started

        budget = None
        if normalizer is not None and self._recent_presynaptic:
            if normalizer_observer is not None:
                normalizer_observer("before", self.plasticity)
            normalizer_started = time.perf_counter() if timings is not None else 0.0
            budget = normalizer.normalize_presynaptic(
                self.plasticity,
                indptr=self.connectome.indptr,
                base_abs=self._base_abs_synapse_counts,
                presynaptic_indices=recent_presynaptic,
            )
            if normalizer_observer is not None:
                normalizer_observer("after", self.plasticity)
            if timings is not None:
                timings["normalizer_seconds"] = time.perf_counter() - normalizer_started

        decay_started = time.perf_counter() if timings is not None else 0.0
        self.plasticity.decay_episode(
            usage_decay=usage_decay,
            # The public operation below clears eligibility immediately. A
            # preceding decay therefore cannot affect any observable state.
            eligibility_decay=1.0 if clear_eligibility else eligibility_decay,
        )
        if clear_eligibility:
            self.plasticity.clear_eligibility()
        if timings is not None:
            timings["plastic_lifecycle_seconds"] = time.perf_counter() - decay_started
        self._recent_presynaptic.clear()

        result = {
            "reward": float(reward),
            "learning": asdict(update),
            "budget": asdict(budget) if budget is not None else None,
            # Full summary scans every edge.  Interactive training only needs
            # per-update stats; checkpoint/final reports still request it.
            "plasticity": self.plasticity.summary() if include_plasticity_summary else None,
            "structural_learning": structural_learning,
            "structural_rewire": structural_rewire,
            "structural_reward_events": int(self._structural_reward_events),
            "structural": self.structural_summary(),
            "post_reward": post_reward,
        }
        if timings is not None:
            timings["total_seconds"] = time.perf_counter() - started
            result["timing"] = timings
        return result
