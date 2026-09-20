"""Optional compiled kernels for the sparse MaleCNS reference simulator.

The kernels preserve the CSR edge order used by the NumPy reference path.  A
missing Numba install is deliberately harmless: callers select the existing
NumPy implementation instead.
"""

from __future__ import annotations

import numpy as np

try:  # Optional performance dependency, not a required simulation dependency.
    from numba import njit
except ImportError:  # pragma: no cover - exercised on minimal installations
    NUMBA_AVAILABLE = False
    advance_sparse_lif = None
    advance_sparse_lif_with_adaptation = None
    scatter_csr_rows = None
    scatter_csr_rows_with_plasticity = None
    scatter_csr_rows_with_std = None
    scatter_csr_rows_with_plasticity_and_std = None
    scatter_csr_rows_with_hebbian = None
    scatter_csr_rows_with_plasticity_and_hebbian = None
    update_hebbian_bindings = None
else:
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:

    @njit(cache=True, nogil=True)
    def scatter_csr_rows_with_plasticity(
        fired,
        indptr,
        posts,
        signed,
        multiplier,
        target,
        plastic_mask,
        usage_ema,
        eligibility,
        scale,
        usage_alpha,
        eligibility_gain,
    ):
        """Scatter CSR rows and update plastic traces in original edge order."""
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                target[posts[edge]] += signed[edge] * multiplier[edge] * scale
                if plastic_mask[edge]:
                    usage_ema[edge] += usage_alpha * (1.0 - usage_ema[edge])
                    eligibility[edge] += eligibility_gain


    @njit(cache=True, nogil=True)
    def scatter_csr_rows(
        fired,
        indptr,
        posts,
        signed,
        multiplier,
        target,
        scale,
    ):
        """Scatter CSR rows when plastic trace collection is disabled."""
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                target[posts[edge]] += signed[edge] * multiplier[edge] * scale


    @njit(cache=True, nogil=True)
    def scatter_csr_rows_with_std(
        fired,
        indptr,
        posts,
        signed,
        multiplier,
        target,
        scale,
        release_factor,
        last_release_step,
        step_index,
        dt_ms,
        recovery_tau_ms,
        depression_fraction,
        min_release_factor,
    ):
        """CSR scatter with lazy generic presynaptic release depression."""
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            old = release_factor[pre]
            last = last_release_step[pre]
            recovered = old
            if last >= 0:
                elapsed_ms = (step_index - last) * dt_ms
                recovered = 1.0 - (1.0 - old) * np.exp(-elapsed_ms / recovery_tau_ms)
            if recovered < min_release_factor:
                recovered = min_release_factor
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                target[posts[edge]] += signed[edge] * multiplier[edge] * recovered * scale
            new_factor = recovered * (1.0 - depression_fraction)
            if new_factor < min_release_factor:
                new_factor = min_release_factor
            release_factor[pre] = new_factor
            last_release_step[pre] = step_index


    @njit(cache=True, nogil=True)
    def scatter_csr_rows_with_plasticity_and_std(
        fired,
        indptr,
        posts,
        signed,
        multiplier,
        target,
        plastic_mask,
        usage_ema,
        eligibility,
        scale,
        usage_alpha,
        eligibility_gain,
        release_factor,
        last_release_step,
        step_index,
        dt_ms,
        recovery_tau_ms,
        depression_fraction,
        min_release_factor,
    ):
        """STD scatter plus the existing exact plastic trace updates."""
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            old = release_factor[pre]
            last = last_release_step[pre]
            recovered = old
            if last >= 0:
                elapsed_ms = (step_index - last) * dt_ms
                recovered = 1.0 - (1.0 - old) * np.exp(-elapsed_ms / recovery_tau_ms)
            if recovered < min_release_factor:
                recovered = min_release_factor
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                target[posts[edge]] += signed[edge] * multiplier[edge] * recovered * scale
                if plastic_mask[edge]:
                    usage_ema[edge] += usage_alpha * (1.0 - usage_ema[edge])
                    eligibility[edge] += eligibility_gain
            new_factor = recovered * (1.0 - depression_fraction)
            if new_factor < min_release_factor:
                new_factor = min_release_factor
            release_factor[pre] = new_factor
            last_release_step[pre] = step_index


    @njit(cache=True, nogil=True)
    def scatter_csr_rows_with_hebbian(
        fired, indptr, posts, signed, multiplier, target, scale,
        binding_gain, binding_last_step, step_index, dt_ms, binding_tau_ms,
    ):
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                gain = binding_gain[edge]
                last = binding_last_step[edge]
                if gain > 0.0 and last >= 0:
                    elapsed_ms = (step_index - last) * dt_ms
                    gain = gain * np.exp(-elapsed_ms / binding_tau_ms)
                    binding_gain[edge] = gain
                    binding_last_step[edge] = step_index
                target[posts[edge]] += signed[edge] * multiplier[edge] * (1.0 + gain) * scale


    @njit(cache=True, nogil=True)
    def scatter_csr_rows_with_plasticity_and_hebbian(
        fired, indptr, posts, signed, multiplier, target, plastic_mask,
        usage_ema, eligibility, scale, usage_alpha, eligibility_gain,
        binding_gain, binding_last_step, step_index, dt_ms, binding_tau_ms,
    ):
        for fired_offset in range(len(fired)):
            pre = fired[fired_offset]
            start = indptr[pre]
            stop = indptr[pre + 1]
            for edge in range(start, stop):
                gain = binding_gain[edge]
                last = binding_last_step[edge]
                if gain > 0.0 and last >= 0:
                    elapsed_ms = (step_index - last) * dt_ms
                    gain = gain * np.exp(-elapsed_ms / binding_tau_ms)
                    binding_gain[edge] = gain
                    binding_last_step[edge] = step_index
                target[posts[edge]] += signed[edge] * multiplier[edge] * (1.0 + gain) * scale
                if plastic_mask[edge]:
                    usage_ema[edge] += usage_alpha * (1.0 - usage_ema[edge])
                    eligibility[edge] += eligibility_gain


    @njit(cache=True, nogil=True)
    def update_hebbian_bindings(
        fired_posts, incoming_indptr, incoming_edges, edge_pre,
        pre_trace, pre_last_step, binding_gain, binding_last_step,
        binding_active_mask, binding_active_edges, binding_active_count,
        step_index, dt_ms, pre_trace_tau_ms, binding_tau_ms,
        binding_increment, max_binding_gain,
    ):
        count = binding_active_count[0]
        for post_offset in range(len(fired_posts)):
            post = fired_posts[post_offset]
            start = incoming_indptr[post]
            stop = incoming_indptr[post + 1]
            for cursor in range(start, stop):
                edge = incoming_edges[cursor]
                pre = edge_pre[cursor]
                trace = pre_trace[pre]
                last_pre = pre_last_step[pre]
                if last_pre >= 0:
                    elapsed_pre = (step_index - last_pre) * dt_ms
                    trace = trace * np.exp(-elapsed_pre / pre_trace_tau_ms)
                if trace <= 1.0e-6:
                    continue
                gain = binding_gain[edge]
                last_binding = binding_last_step[edge]
                if gain > 0.0 and last_binding >= 0:
                    elapsed_binding = (step_index - last_binding) * dt_ms
                    gain = gain * np.exp(-elapsed_binding / binding_tau_ms)
                if not binding_active_mask[edge]:
                    binding_active_mask[edge] = True
                    binding_active_edges[count] = edge
                    count += 1
                gain += binding_increment * trace
                if gain > max_binding_gain:
                    gain = max_binding_gain
                binding_gain[edge] = gain
                binding_last_step[edge] = step_index
        binding_active_count[0] = count


    @njit(cache=True, nogil=True)
    def advance_sparse_lif(
        active,
        due,
        due_slot,
        v,
        g,
        refractory_until,
        stimulated,
        fired_out,
        step_index,
        resting_mv,
        threshold_mv,
        reset_mv,
        membrane_decay,
        g_to_v,
        synapse_decay,
        poisson_drive_mv,
        refractory_steps,
    ):
        """Advance the sparse LIF state without temporary indexed gathers.

        The order intentionally mirrors the NumPy reference: deliver delayed
        current, decay/integrate active cells, clamp refractory cells, inject
        stimulus, then detect and reset spikes.
        """
        for offset in range(len(due)):
            neuron = due[offset]
            g[neuron] += due_slot[neuron]
            due_slot[neuron] = 0.0

        for offset in range(len(active)):
            neuron = active[offset]
            old_g = g[neuron]
            v[neuron] = (
                resting_mv
                + (v[neuron] - resting_mv) * membrane_decay
                + old_g * g_to_v
            )
            g[neuron] = old_g * synapse_decay
            if step_index < refractory_until[neuron]:
                v[neuron] = resting_mv
                g[neuron] = 0.0

        for offset in range(len(stimulated)):
            v[stimulated[offset]] += poisson_drive_mv

        fired_count = 0
        for offset in range(len(active)):
            neuron = active[offset]
            if v[neuron] > threshold_mv and step_index >= refractory_until[neuron]:
                fired_out[fired_count] = neuron
                fired_count += 1
                v[neuron] = reset_mv
                g[neuron] = 0.0
                refractory_until[neuron] = step_index + refractory_steps
        return fired_count

    @njit(cache=True, nogil=True)
    def advance_sparse_lif_with_adaptation(
        active,
        due,
        due_slot,
        v,
        g,
        refractory_until,
        adaptation,
        stimulated,
        fired_out,
        step_index,
        resting_mv,
        threshold_mv,
        reset_mv,
        membrane_decay,
        g_to_v,
        synapse_decay,
        poisson_drive_mv,
        refractory_steps,
        adaptation_decay,
        spike_increment_mv,
        max_adaptation_mv,
    ):
        """Sparse LIF update with generic transient threshold adaptation."""
        for offset in range(len(due)):
            neuron = due[offset]
            g[neuron] += due_slot[neuron]
            due_slot[neuron] = 0.0

        for offset in range(len(active)):
            neuron = active[offset]
            old_g = g[neuron]
            v[neuron] = (
                resting_mv
                + (v[neuron] - resting_mv) * membrane_decay
                + old_g * g_to_v
            )
            g[neuron] = old_g * synapse_decay
            if step_index < refractory_until[neuron]:
                v[neuron] = resting_mv
                g[neuron] = 0.0
            adaptation[neuron] *= adaptation_decay

        for offset in range(len(stimulated)):
            v[stimulated[offset]] += poisson_drive_mv

        fired_count = 0
        for offset in range(len(active)):
            neuron = active[offset]
            if v[neuron] > threshold_mv + adaptation[neuron] and step_index >= refractory_until[neuron]:
                fired_out[fired_count] = neuron
                fired_count += 1
                adaptation[neuron] += spike_increment_mv
                if adaptation[neuron] > max_adaptation_mv:
                    adaptation[neuron] = max_adaptation_mv
                v[neuron] = reset_mv
                g[neuron] = 0.0
                refractory_until[neuron] = step_index + refractory_steps
        return fired_count
