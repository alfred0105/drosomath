from __future__ import annotations

import math

from drosomath.whole_brain.structural_overlay import LearnedStructuralOverlay


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
        self._fast_active = np.empty(0, dtype=np.int32)
        self._fast_delay_touched: list[list[object]] = [
            [] for _ in range(len(self._delay_ring))
        ]
        self._fast_peak_active = 0

    def reset(self) -> None:
        super().reset()
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
        transferred = super()._schedule_spike_outputs(fired_indices)
        self._record_fast_delay_targets(fired_indices)
        return transferred

    def _due_indices(self, ring_index: int):
        np = self.np
        chunks = self._fast_delay_touched[ring_index]
        if not chunks:
            return np.empty(0, dtype=np.int32)
        if len(chunks) == 1:
            due = np.unique(chunks[0]).astype(np.int32, copy=False)
        else:
            due = np.unique(np.concatenate(chunks)).astype(np.int32, copy=False)
        chunks.clear()
        return due

    def step(self, *, stimulus_indices=None, stimulus_rate_hz: float = 0.0):
        np = self.np
        p = self.params
        ring_index = self.step_index % len(self._delay_ring)
        due_slot = self._delay_ring[ring_index]
        due = self._due_indices(ring_index)

        pieces = []
        if len(self._fast_active):
            pieces.append(self._fast_active)
        if len(due):
            pieces.append(due)
        if stimulus_indices is not None and len(stimulus_indices):
            pieces.append(stimulus_indices)

        if pieces:
            if len(pieces) == 1:
                active = np.unique(pieces[0]).astype(np.int32, copy=False)
            else:
                active = np.unique(np.concatenate(pieces)).astype(np.int32, copy=False)
        else:
            active = np.empty(0, dtype=np.int32)

        if len(due):
            self.g[due] += due_slot[due]
            due_slot[due] = 0.0

        if len(active):
            old_g = self.g[active].copy()
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

        if (
            stimulus_indices is not None
            and len(stimulus_indices)
            and stimulus_rate_hz > 0.0
        ):
            probability = min(1.0, stimulus_rate_hz * p.dt_ms / 1000.0)
            stimulated = stimulus_indices[
                self.rng.random(len(stimulus_indices)) < probability
            ]
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

        if len(fired):
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
