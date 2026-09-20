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
