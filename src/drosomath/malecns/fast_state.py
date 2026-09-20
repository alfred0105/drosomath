from __future__ import annotations

import math
import os
import time

from drosomath.whole_brain.structural_overlay import LearnedStructuralOverlay

from .numba_kernels import (
    NUMBA_AVAILABLE,
    advance_sparse_lif,
    advance_sparse_lif_with_adaptation,
    scatter_csr_rows,
    scatter_csr_rows_with_plasticity,
    scatter_csr_rows_with_std,
    scatter_csr_rows_with_plasticity_and_std,
)


class FastLearnedStructuralOverlay(LearnedStructuralOverlay):
    """MaleCNS structural overlay with bounded deterministic donor scans.

    The generic overlay scans every anatomical edge to choose a low-value donor.
    On the 6.2M-edge MaleCNS graph that becomes expensive when rewiring repeats.
    This variant evaluates a deterministic pseudo-strided sample spread across
    the whole edge array while preserving the same donor eligibility/protection
    criteria. Small graphs are scanned exhaustively.
    """

    donor_scan_size: int = 65_536

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        n = max(1, int(self.plasticity.edge_count))
        stride = 104_729
        while math.gcd(stride, n) != 1:
            stride += 2
        self._fast_donor_stride = int(stride)

    def _select_donor_edges(self, count: int):
        np = self.np
        if count <= 0:
            return np.empty(0, dtype=np.int32)
        n = int(self.plasticity.edge_count)
        if n == 0:
            return np.empty(0, dtype=np.int32)

        scan_n = min(n, max(self.donor_scan_size, int(count) * 256))
        if scan_n == n:
            idx = np.arange(n, dtype=np.int64)
        else:
            base = np.arange(scan_n, dtype=np.int64)
            offset = (int(self._cycles) * 97_531) % n
            idx = (offset + base * self._fast_donor_stride) % n

        eligible = self.plasticity.plastic_mask[idx]
        eligible &= self.plasticity.stability[idx] < self.config.donor_protected_stability
        eligible &= (
            self.plasticity.multiplier[idx]
            > self.plasticity.config.min_multiplier + 1e-6
        )

        used = self.donor_edge[self.active]
        used = used[used >= 0]
        if len(used):
            eligible &= ~np.isin(idx, used, assume_unique=False)
        idx = idx[eligible]
        if len(idx) == 0:
            return np.empty(0, dtype=np.int32)

        score = (
            self.plasticity.usage_ema[idx]
            + 2.0 * self.plasticity.stability[idx]
            + 0.10 * np.abs(self.plasticity.multiplier[idx] - 1.0)
        )
        k = min(int(count), len(idx))
        if len(idx) <= k:
            order = np.argsort(score)
        else:
            order = np.argpartition(score, k - 1)[:k]
            order = order[np.argsort(score[order])]
        return idx[order].astype(np.int32, copy=False)


class FastSparseStateMixin:
    """Exact sparse-state update fast path for reset-between-trial MaleCNS runs.

    Only neurons that have ever become relevant in the current trial are updated:
    prior active neurons, delayed synaptic targets, and stimulus neurons. Neurons
    outside that set remain exactly at rest with zero conductance, so skipping
    their dense vector update does not change the LIF equations.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        np = self.np
        # ``_fast_active`` is kept sorted because the old implementation used
        # ``np.unique`` (which also sorts) before every sparse state update.
        # The mask makes membership checks O(1) and lets us only sort newly
        # activated neurons instead of repeatedly uniquing the full history.
        self._fast_active_mask = np.zeros(
            int(self.connectome.neuron_count), dtype=np.bool_
        )
        self._fast_due_mask = np.zeros(
            int(self.connectome.neuron_count), dtype=np.bool_
        )
        self._fast_active = np.empty(0, dtype=np.int32)
        self._fast_fired = np.empty(int(self.connectome.neuron_count), dtype=np.int32)
        self._fast_delay_touched: list[list[object]] = [
            [] for _ in range(len(self._delay_ring))
        ]
        self._fast_peak_active = 0
        self.adaptation_mv = np.zeros(
            int(self.connectome.neuron_count), dtype=np.float32
        )
        self._slow_adaptation_decay = (
            self.slow_adaptation_config.decay_factor(self.params.dt_ms)
            if self.slow_adaptation_config.enabled
            else 1.0
        )
        requested_numba = os.environ.get("DROSOMATH_NUMBA", "auto").strip().lower()
        self._numba_enabled = (
            NUMBA_AVAILABLE
            and requested_numba not in {"0", "false", "off", "no"}
        )
        self._std_config = getattr(self, "presynaptic_depression_config", None)
        self._std_enabled = bool(self._std_config is not None and self._std_config.enabled)
        if self._std_enabled:
            self.release_factor = np.ones(
                int(self.connectome.neuron_count), dtype=np.float32
            )
            self.last_release_step = np.full(
                int(self.connectome.neuron_count), -1, dtype=np.int64
            )
        else:
            # Disabled STD must not add another dense neuron-sized state array
            # to the normal P.8 path.
            self.release_factor = np.empty(0, dtype=np.float32)
            self.last_release_step = np.empty(0, dtype=np.int64)
        # Optional P.3 audit hook.  It is None in all normal runs, so timing
        # cannot alter model state or consume RNG values.
        self._neural_timing_profiler = None

    def configure_neural_timing(self, profiler=None):
        """Attach an explicit neural-step profiler and return the previous one."""
        previous = self._neural_timing_profiler
        self._neural_timing_profiler = profiler
        return previous

    def reset(self) -> None:
        super().reset()
        self._fast_active_mask.fill(False)
        self._fast_due_mask.fill(False)
        self._fast_active = self.np.empty(0, dtype=self.np.int32)
        for chunks in self._fast_delay_touched:
            chunks.clear()
        self._fast_peak_active = 0
        self.adaptation_mv.fill(0.0)
        if self._std_enabled:
            self.release_factor.fill(1.0)
            self.last_release_step.fill(-1)

    @property
    def slow_adaptation_enabled(self) -> bool:
        return bool(self.slow_adaptation_config.enabled)

    def slow_adaptation_summary(self) -> dict[str, float | int | bool]:
        values = self.adaptation_mv
        positive = values[values > 0.0]
        return {
            "enabled": self.slow_adaptation_enabled,
            "adapted_neurons": int(len(positive)),
            "mean_adaptation_mv": float(positive.mean()) if len(positive) else 0.0,
            "max_adaptation_mv": float(values.max()) if len(values) else 0.0,
            "extra_state_bytes": int(values.nbytes),
        }

    def _apply_slow_adaptation_decay(self, active) -> None:
        if not self.slow_adaptation_enabled or len(active) == 0:
            return
        self.adaptation_mv[active] *= self._slow_adaptation_decay

    def _apply_slow_adaptation_spikes(self, fired) -> None:
        if not self.slow_adaptation_enabled or len(fired) == 0:
            return
        np = self.np
        config = self.slow_adaptation_config
        values = self.adaptation_mv[fired] + np.float32(config.spike_increment_mv)
        np.minimum(values, np.float32(config.max_adaptation_mv), out=values)
        self.adaptation_mv[fired] = values

    def _std_release_for_pre(self, pre: int) -> float:
        """Recover lazily, then depress after the current firing event."""
        if not self._std_enabled:
            return 1.0
        np = self.np
        config = self._std_config
        old = float(self.release_factor[pre])
        last = int(self.last_release_step[pre])
        if last >= 0:
            elapsed_ms = (int(self.step_index) - last) * float(self.params.dt_ms)
            recovered = 1.0 - (1.0 - old) * math.exp(
                -elapsed_ms / float(config.recovery_tau_ms)
            )
        else:
            recovered = old
        recovered = max(float(config.min_release_factor), recovered)
        self.release_factor[pre] = np.float32(
            max(float(config.min_release_factor), recovered * (1.0 - float(config.depression_fraction)))
        )
        self.last_release_step[pre] = int(self.step_index)
        return recovered

    def presynaptic_depression_summary(self) -> dict[str, float | int | bool]:
        """Compact transient STD state summary; never serializes dense arrays."""
        if not self._std_enabled:
            return {
                "enabled": False,
                "depressed_neurons": 0,
                "mean_release_factor_depressed": 1.0,
                "minimum_release_factor": 1.0,
                "fraction_at_floor": 0.0,
                "mean_recovery_now": 1.0,
                "extra_state_bytes": 0,
            }
        np = self.np
        depressed = self.release_factor < np.float32(1.0 - 1e-7)
        values = self.release_factor[depressed]
        recovered_touched = self._std_recovered_factors()
        return {
            "enabled": True,
            "touched_neurons": int(np.count_nonzero(self.last_release_step >= 0)),
            "depressed_neurons": int(depressed.sum()),
            "mean_release_factor_depressed": float(values.mean()) if len(values) else 1.0,
            "minimum_release_factor": float(self.release_factor.min()) if len(self.release_factor) else 1.0,
            "fraction_at_floor": float(
                np.mean(self.release_factor <= np.float32(self._std_config.min_release_factor + 1e-6))
            ) if len(self.release_factor) else 0.0,
            "mean_recovery_now": float(recovered_touched.mean()) if len(recovered_touched) else 1.0,
            "extra_state_bytes": int(self.release_factor.nbytes + self.last_release_step.nbytes),
        }

    def _std_recovered_factors(self):
        """Return recovered factors only for touched neurons.

        Telemetry must not turn the sparse lazy state into a dense per-neuron
        work array.  The returned temporary is bounded by the neurons that
        actually fired since the last reset.
        """
        if not self._std_enabled:
            return self.np.empty(0, dtype=self.np.float32)
        np = self.np
        touched = np.flatnonzero(self.last_release_step >= 0)
        if not len(touched):
            return np.empty(0, dtype=np.float32)
        result = self.release_factor[touched].astype(np.float32, copy=True)
        elapsed = (int(self.step_index) - self.last_release_step[touched]).astype(np.float32) * np.float32(self.params.dt_ms)
        result = np.maximum(
            np.float32(self._std_config.min_release_factor),
            1.0 - (1.0 - result) * np.exp(
                -elapsed / np.float32(self._std_config.recovery_tau_ms)
            ),
        )
        return result

    def fast_state_summary(self) -> dict[str, int | float]:
        n = int(self.connectome.neuron_count)
        active = int(len(self._fast_active))
        return {
            "active_state_neurons": active,
            "peak_active_state_neurons": int(self._fast_peak_active),
            "neuron_count": n,
            "active_fraction": (active / n) if n else 0.0,
        }

    def _record_fast_delay_targets(self, fired_indices) -> None:
        if len(fired_indices) == 0:
            return
        target_index = (self.step_index + self.delay_steps) % len(self._delay_ring)
        chunks = self._fast_delay_touched[target_index]
        indptr = self.connectome.indptr
        posts = self.connectome.post_indices
        overlay = getattr(self, "structural_overlay", None)
        for pre_raw in fired_indices:
            pre = int(pre_raw)
            start = int(indptr[pre])
            stop = int(indptr[pre + 1])
            if stop > start:
                chunks.append(posts[start:stop])
            if overlay is not None:
                slots = overlay.slots_for_pre(pre)
                if len(slots):
                    chunks.append(overlay.post_index[slots])

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
        scale = self.params.mv_per_synapse
        indptr = self.connectome.indptr
        posts = self.connectome.post_indices
        base_signed = self.connectome.signed_synapse_counts
        fired = np.asarray(fired_indices, dtype=np.int32)
        starts = indptr[fired]
        stops = indptr[fired + 1]
        canonical_total = int(np.sum(stops - starts, dtype=np.int64))
        overlay = self.structural_overlay
        if overlay is None and not self._telemetry_enabled:
            # This is the normal keyboard-training path.  Gather all outgoing
            # anatomical edge slices into one vectorized batch instead of
            # crossing the Python loop once per fired neuron.
            if canonical_total:
                if self._numba_enabled:
                    if self.plasticity_tracking_enabled:
                        if self._std_enabled:
                            config = self._std_config
                            scatter_csr_rows_with_plasticity_and_std(
                                fired, indptr, posts, base_signed,
                                self.plasticity.multiplier, target_slot,
                                self.plasticity.plastic_mask,
                                self.plasticity.usage_ema,
                                self.plasticity.eligibility,
                                scale, self.usage_alpha, self.eligibility_gain,
                                self.release_factor, self.last_release_step,
                                self.step_index, self.params.dt_ms,
                                config.recovery_tau_ms, config.depression_fraction,
                                config.min_release_factor,
                            )
                        else:
                            scatter_csr_rows_with_plasticity(
                                fired, indptr, posts, base_signed,
                                self.plasticity.multiplier, target_slot,
                                self.plasticity.plastic_mask,
                                self.plasticity.usage_ema,
                                self.plasticity.eligibility,
                                scale, self.usage_alpha, self.eligibility_gain,
                            )
                    else:
                        if self._std_enabled:
                            config = self._std_config
                            scatter_csr_rows_with_std(
                                fired, indptr, posts, base_signed,
                                self.plasticity.multiplier, target_slot, scale,
                                self.release_factor, self.last_release_step,
                                self.step_index, self.params.dt_ms,
                                config.recovery_tau_ms, config.depression_fraction,
                                config.min_release_factor,
                            )
                        else:
                            scatter_csr_rows(
                                fired, indptr, posts, base_signed,
                                self.plasticity.multiplier, target_slot, scale,
                            )
                else:
                    lengths = (stops - starts).astype(np.int64, copy=False)
                    offsets = np.arange(canonical_total, dtype=np.int64)
                    block_offsets = np.repeat(
                        np.cumsum(lengths, dtype=np.int64) - lengths,
                        lengths,
                    )
                    edge_indices = (
                        np.repeat(starts, lengths)
                        + offsets
                        - block_offsets
                    )
                    if self._std_enabled:
                        for pre_raw, start_raw, stop_raw in zip(fired, starts, stops):
                            pre = int(pre_raw)
                            release = self._std_release_for_pre(pre)
                            if int(stop_raw) > int(start_raw):
                                edges = np.arange(int(start_raw), int(stop_raw), dtype=np.int32)
                                np.add.at(
                                    target_slot,
                                    posts[edges],
                                    base_signed[edges]
                                    * self.plasticity.multiplier[edges]
                                    * np.float32(release)
                                    * scale,
                                )
                    else:
                        np.add.at(
                            target_slot,
                            posts[edge_indices],
                            base_signed[edge_indices]
                            * self.plasticity.multiplier[edge_indices]
                            * scale,
                        )
                    if self.plasticity_tracking_enabled:
                        self.plasticity.record_use_indices(
                            edge_indices,
                            usage_alpha=self.usage_alpha,
                            eligibility_gain=self.eligibility_gain,
                        )
                if self.plasticity_tracking_enabled:
                    self._recent_presynaptic.update(int(pre) for pre in fired)
            if self._std_enabled and canonical_total == 0:
                for pre_raw in fired:
                    self._std_release_for_pre(int(pre_raw))
            self._record_fast_delay_targets(fired)
            return canonical_total

        structural_groups = []
        structural_total = 0
        if overlay is not None:
            for pre_raw in fired:
                slots = overlay.slots_for_pre(int(pre_raw))
                structural_groups.append(slots)
                structural_total += len(slots)

        total_edges = canonical_total + structural_total
        output_posts = np.empty(total_edges, dtype=np.int32)
        output_values = np.empty(total_edges, dtype=np.float32)
        cursor = 0
        transferred = 0
        telemetry_edges: list[dict[str, object]] = []

        for offset, (pre_raw, start_raw, stop_raw) in enumerate(zip(fired, starts, stops)):
            pre = int(pre_raw)
            start = int(start_raw)
            stop = int(stop_raw)
            std_release = self._std_release_for_pre(pre) if self._std_enabled else 1.0
            if stop > start:
                count = stop - start
                effective = self.plasticity.effective_signed_slice(base_signed, start, stop)
                if self._std_enabled:
                    effective = effective * np.float32(std_release)
                output_posts[cursor : cursor + count] = posts[start:stop]
                output_values[cursor : cursor + count] = effective * scale
                cursor += count
                transferred += count

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

            if overlay is not None:
                structural_slots = structural_groups[offset]
                if len(structural_slots):
                    structural_count = len(structural_slots)
                    structural_posts = overlay.post_index[structural_slots]
                    structural_strength = overlay.signed_strength[structural_slots]
                    output_posts[cursor : cursor + structural_count] = structural_posts
                    output_values[cursor : cursor + structural_count] = structural_strength * scale
                    cursor += structural_count
                    transferred += structural_count
                    if self.plasticity_tracking_enabled:
                        overlay.record_use(structural_slots)
                    if self._telemetry_enabled and len(telemetry_edges) < self._telemetry_max_edges:
                        telemetry_edges.extend(
                            self._sample_structural_live_edges(pre=pre, slots=structural_slots)
                        )

        if cursor:
            # Preserve fired-neuron/edge order while doing one indexed
            # accumulation per timestep instead of one np.add.at per pre.
            np.add.at(target_slot, output_posts[:cursor], output_values[:cursor])

        if self._telemetry_enabled:
            if len(telemetry_edges) > self._telemetry_max_edges:
                telemetry_edges = sorted(
                    telemetry_edges,
                    key=lambda row: abs(float(row["signal_mv"])),
                    reverse=True,
                )[: self._telemetry_max_edges]
            fired_ids = [self._neuron_identifier(int(x)) for x in fired[:256]]
            self._last_telemetry = {
                "step": int(self.step_index),
                "fired_neuron_ids": fired_ids,
                "fired_neuron_count": int(len(fired)),
                "transferred_synapses": int(transferred),
                "active_edges": telemetry_edges,
            }

        self._record_fast_delay_targets(fired)
        return transferred

    def _due_indices(self, ring_index: int):
        np = self.np
        chunks = self._fast_delay_touched[ring_index]
        if not chunks:
            return np.empty(0, dtype=np.int32)
        raw = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        # A due slot is a fresh batch: duplicates only need to be removed
        # within this batch.  Sorting the batch once and comparing adjacent
        # values is equivalent to the old np.unique path, but avoids a full
        # neuron-sized boolean mask write/read for every due slot.
        fresh = np.sort(raw) if len(raw) > 1 else raw
        if len(fresh) > 1:
            keep = np.empty(len(fresh), dtype=np.bool_)
            keep[0] = True
            keep[1:] = fresh[1:] != fresh[:-1]
            fresh = fresh[keep]
        chunks.clear()
        return fresh.astype(np.int32, copy=False)

    def _activate_indices(self, *index_groups):
        """Add new active neurons while preserving the sorted active cache.

        The old path built ``active = unique(old_active + due + stimulus)`` on
        every timestep.  Here the boolean mask removes duplicates first, then
        only the newly activated subset is sorted and merged into the cached
        sorted array.  The resulting order is identical to ``np.unique``.
        """
        np = self.np
        additions = []
        for indices in index_groups:
            if indices is None or len(indices) == 0:
                continue
            indices = np.asarray(indices, dtype=np.int32)
            unseen = indices[~self._fast_active_mask[indices]]
            self._fast_active_mask[indices] = True
            if len(unseen):
                additions.append(unseen)

        if not additions:
            return self._fast_active

        new = additions[0] if len(additions) == 1 else np.concatenate(additions)
        if len(new) > 1:
            new = np.sort(new)
            keep = np.empty(len(new), dtype=np.bool_)
            keep[0] = True
            keep[1:] = new[1:] != new[:-1]
            new = new[keep]

        if len(self._fast_active) == 0:
            self._fast_active = new.astype(np.int32, copy=False)
        else:
            # ``new`` is sorted and disjoint from the cache by construction.
            # Insert it into the sorted cache by rank instead of sorting the
            # entire active history again.
            old = self._fast_active
            positions = np.searchsorted(old, new, side="left")
            positions += np.arange(len(new), dtype=np.int32)
            merged = np.empty(len(old) + len(new), dtype=np.int32)
            occupied = np.zeros(len(merged), dtype=np.bool_)
            merged[positions] = new
            occupied[positions] = True
            merged[~occupied] = old
            self._fast_active = merged
        return self._fast_active

    def step(self, *, stimulus_indices=None, stimulus_rate_hz: float = 0.0):
        np = self.np
        p = self.params
        profiler = self._neural_timing_profiler
        if profiler is None:
            ring_index = self.step_index % len(self._delay_ring)
            due_slot = self._delay_ring[ring_index]
            due = self._due_indices(ring_index)
            active = self._activate_indices(due, stimulus_indices)
        else:
            delay_started = time.perf_counter()
            with profiler.section("due_index_collection_seconds"):
                ring_index = self.step_index % len(self._delay_ring)
                due_slot = self._delay_ring[ring_index]
                due = self._due_indices(ring_index)
            with profiler.section("active_set_merge_seconds"):
                active = self._activate_indices(due, stimulus_indices)
            # Compatibility aggregate.  It is intentionally not included in
            # P.4's disjoint subcomponent sum because it is the sum of the two
            # sections above, not an additional interval.
            profiler.add("delay_ring_handling_seconds", time.perf_counter() - delay_started)

        stimulated = np.empty(0, dtype=np.int32)
        if profiler is None:
            if (
                stimulus_indices is not None
                and len(stimulus_indices)
                and stimulus_rate_hz > 0.0
            ):
                probability = min(1.0, stimulus_rate_hz * p.dt_ms / 1000.0)
                stimulated = stimulus_indices[
                    self.rng.random(len(stimulus_indices)) < probability
                ]
        else:
            with profiler.section("stimulus_injection_seconds"):
                if (
                    stimulus_indices is not None
                    and len(stimulus_indices)
                    and stimulus_rate_hz > 0.0
                ):
                    probability = min(1.0, stimulus_rate_hz * p.dt_ms / 1000.0)
                    stimulated = stimulus_indices[
                        self.rng.random(len(stimulus_indices)) < probability
                    ]
        if profiler is None:
            if self._numba_enabled:
                if self.slow_adaptation_enabled:
                    fired_count = advance_sparse_lif_with_adaptation(
                        active, due, due_slot, self.v, self.g, self.refractory_until,
                        self.adaptation_mv, stimulated, self._fast_fired, self.step_index,
                        np.float32(p.resting_mv), np.float32(p.threshold_mv),
                        np.float32(p.reset_mv), np.float32(self._membrane_decay),
                        np.float32(self._g_to_v), np.float32(self._synapse_decay),
                        np.float32(p.mv_per_synapse * p.poisson_drive_scale),
                        self.refractory_steps, np.float32(self._slow_adaptation_decay),
                        np.float32(self.slow_adaptation_config.spike_increment_mv),
                        np.float32(self.slow_adaptation_config.max_adaptation_mv),
                    )
                else:
                    fired_count = advance_sparse_lif(
                        active, due, due_slot, self.v, self.g, self.refractory_until,
                        stimulated, self._fast_fired, self.step_index,
                        np.float32(p.resting_mv), np.float32(p.threshold_mv),
                        np.float32(p.reset_mv), np.float32(self._membrane_decay),
                        np.float32(self._g_to_v), np.float32(self._synapse_decay),
                        np.float32(p.mv_per_synapse * p.poisson_drive_scale),
                        self.refractory_steps,
                    )
                fired = self._fast_fired[:fired_count]
            else:
                fired = (
                    self._advance_python_with_slow_adaptation(active, due, due_slot, stimulated, p)
                    if self.slow_adaptation_enabled
                    else self._advance_python(active, due, due_slot, stimulated, p)
                )
        else:
            with profiler.section("active_neuron_state_update_seconds"):
                if self._numba_enabled:
                    if self.slow_adaptation_enabled:
                        fired_count = advance_sparse_lif_with_adaptation(
                            active, due, due_slot, self.v, self.g, self.refractory_until,
                            self.adaptation_mv, stimulated, self._fast_fired, self.step_index,
                            np.float32(p.resting_mv), np.float32(p.threshold_mv),
                            np.float32(p.reset_mv), np.float32(self._membrane_decay),
                            np.float32(self._g_to_v), np.float32(self._synapse_decay),
                            np.float32(p.mv_per_synapse * p.poisson_drive_scale),
                            self.refractory_steps, np.float32(self._slow_adaptation_decay),
                            np.float32(self.slow_adaptation_config.spike_increment_mv),
                            np.float32(self.slow_adaptation_config.max_adaptation_mv),
                        )
                    else:
                        fired_count = advance_sparse_lif(
                            active, due, due_slot, self.v, self.g, self.refractory_until,
                            stimulated, self._fast_fired, self.step_index,
                            np.float32(p.resting_mv), np.float32(p.threshold_mv),
                            np.float32(p.reset_mv), np.float32(self._membrane_decay),
                            np.float32(self._g_to_v), np.float32(self._synapse_decay),
                            np.float32(p.mv_per_synapse * p.poisson_drive_scale),
                            self.refractory_steps,
                        )
                    fired = self._fast_fired[:fired_count]
                else:
                    fired = (
                        self._advance_python_with_slow_adaptation(active, due, due_slot, stimulated, p)
                        if self.slow_adaptation_enabled
                        else self._advance_python(active, due, due_slot, stimulated, p)
                    )

        if profiler is None:
            transferred = self._schedule_spike_outputs(fired)
        else:
            with profiler.section("synaptic_scheduling_seconds"):
                transferred = self._schedule_spike_outputs(fired)

        if len(fired) and not self._numba_enabled:
            self.v[fired] = p.reset_mv
            self.g[fired] = 0.0
            self.refractory_until[fired] = self.step_index + self.refractory_steps

        # Conservatively retain every neuron that has mattered this trial. This
        # avoids threshold approximations while still skipping untouched CNS cells.
        self._fast_active = active
        if len(active) > self._fast_peak_active:
            self._fast_peak_active = int(len(active))
        self.step_index += 1
        return fired, transferred

    def _advance_python(self, active, due, due_slot, stimulated, p):
        """Reference sparse LIF branch used by the optional timing wrapper."""
        np = self.np
        if len(due):
            self.g[due] += due_slot[due]
            due_slot[due] = 0.0
        if len(active):
            old_g = self.g[active]
            y = self.v[active] - p.resting_mv
            self.v[active] = p.resting_mv + y * self._membrane_decay + old_g * self._g_to_v
            self.g[active] = old_g * self._synapse_decay
            refractory = self.step_index < self.refractory_until[active]
            if refractory.any():
                blocked = active[refractory]
                self.v[blocked] = p.resting_mv
                self.g[blocked] = 0.0
        if len(stimulated):
            self.v[stimulated] += p.mv_per_synapse * p.poisson_drive_scale
        if len(active):
            local_fire = (
                (self.v[active] > p.threshold_mv)
                & (self.step_index >= self.refractory_until[active])
            )
            fired = active[local_fire].astype(np.int32, copy=False)
        else:
            fired = np.empty(0, dtype=np.int32)
        return fired

    def _advance_python_with_slow_adaptation(self, active, due, due_slot, stimulated, p):
        """Reference sparse LIF update with transient threshold adaptation."""
        np = self.np
        if len(due):
            self.g[due] += due_slot[due]
            due_slot[due] = 0.0
        if len(active):
            old_g = self.g[active]
            y = self.v[active] - p.resting_mv
            self.v[active] = p.resting_mv + y * self._membrane_decay + old_g * self._g_to_v
            self.g[active] = old_g * self._synapse_decay
            refractory = self.step_index < self.refractory_until[active]
            if refractory.any():
                blocked = active[refractory]
                self.v[blocked] = p.resting_mv
                self.g[blocked] = 0.0
            self._apply_slow_adaptation_decay(active)
        if len(stimulated):
            self.v[stimulated] += p.mv_per_synapse * p.poisson_drive_scale
        if len(active):
            local_fire = (
                (self.v[active] > (p.threshold_mv + self.adaptation_mv[active]))
                & (self.step_index >= self.refractory_until[active])
            )
            fired = active[local_fire].astype(np.int32, copy=False)
        else:
            fired = np.empty(0, dtype=np.int32)
        self._apply_slow_adaptation_spikes(fired)
        return fired
