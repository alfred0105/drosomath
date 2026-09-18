from __future__ import annotations

import math
import os

from drosomath.whole_brain.structural_overlay import LearnedStructuralOverlay

from .numba_kernels import (
    NUMBA_AVAILABLE,
    advance_sparse_lif,
    scatter_csr_rows,
    scatter_csr_rows_with_plasticity,
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
        requested_numba = os.environ.get("DROSOMATH_NUMBA", "auto").strip().lower()
        self._numba_enabled = (
            NUMBA_AVAILABLE
            and requested_numba not in {"0", "false", "off", "no"}
        )

    def reset(self) -> None:
        super().reset()
        self._fast_active_mask.fill(False)
        self._fast_due_mask.fill(False)
        self._fast_active = self.np.empty(0, dtype=self.np.int32)
        for chunks in self._fast_delay_touched:
            chunks.clear()
        self._fast_peak_active = 0

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
                        scatter_csr_rows_with_plasticity(
                            fired,
                            indptr,
                            posts,
                            base_signed,
                            self.plasticity.multiplier,
                            target_slot,
                            self.plasticity.plastic_mask,
                            self.plasticity.usage_ema,
                            self.plasticity.eligibility,
                            scale,
                            self.usage_alpha,
                            self.eligibility_gain,
                        )
                    else:
                        scatter_csr_rows(
                            fired,
                            indptr,
                            posts,
                            base_signed,
                            self.plasticity.multiplier,
                            target_slot,
                            scale,
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
            if stop > start:
                count = stop - start
                effective = self.plasticity.effective_signed_slice(base_signed, start, stop)
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
        # This mask is local to the current due slot. It removes duplicate
        # post-neurons without hashing the whole array through np.unique.
        fresh = raw[~self._fast_due_mask[raw]]
        self._fast_due_mask[raw] = True
        if len(fresh) > 1:
            fresh = np.sort(fresh)
            keep = np.empty(len(fresh), dtype=np.bool_)
            keep[0] = True
            keep[1:] = fresh[1:] != fresh[:-1]
            fresh = fresh[keep]
        self._fast_due_mask[raw] = False
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
        ring_index = self.step_index % len(self._delay_ring)
        due_slot = self._delay_ring[ring_index]
        due = self._due_indices(ring_index)
        active = self._activate_indices(due, stimulus_indices)

        stimulated = np.empty(0, dtype=np.int32)
        if (
            stimulus_indices is not None
            and len(stimulus_indices)
            and stimulus_rate_hz > 0.0
        ):
            probability = min(1.0, stimulus_rate_hz * p.dt_ms / 1000.0)
            stimulated = stimulus_indices[
                self.rng.random(len(stimulus_indices)) < probability
            ]
        if self._numba_enabled:
            fired_count = advance_sparse_lif(
                active,
                due,
                due_slot,
                self.v,
                self.g,
                self.refractory_until,
                stimulated,
                self._fast_fired,
                self.step_index,
                np.float32(p.resting_mv),
                np.float32(p.threshold_mv),
                np.float32(p.reset_mv),
                np.float32(self._membrane_decay),
                np.float32(self._g_to_v),
                np.float32(self._synapse_decay),
                np.float32(p.mv_per_synapse * p.poisson_drive_scale),
                self.refractory_steps,
            )
            fired = self._fast_fired[:fired_count]
        else:
            if len(due):
                self.g[due] += due_slot[due]
                due_slot[due] = 0.0

            if len(active):
                # Integer-array indexing already returns a detached gather; an
                # additional ``.copy()`` only adds one temporary allocation.
                old_g = self.g[active]
                y = self.v[active] - p.resting_mv
                self.v[active] = (
                    p.resting_mv
                    + y * self._membrane_decay
                    + old_g * self._g_to_v
                )
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
