"""Optional CuPy sparse backend for GPU validation.

This module is deliberately separate from the NumPy reference simulator.  It
uses the same CSR topology and LIF constants, but keeps neural state and sparse
matrix operations on the CUDA device.  Plasticity is not silently claimed here:
this first backend is a GPU execution/benchmark path and reports that fact in
its result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams

from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1


def _require_cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - optional GPU path
        raise RuntimeError(
            "GPU mode needs CuPy. Install with: "
            "python -m pip install -e \".[gpu]\""
        ) from exc
    return cp


class GpuSparseFlyBrain:
    """CuPy CSR LIF runner using the real connectome topology.

    The source connectome remains host-side and immutable.  Its CSR arrays,
    synaptic values, membrane state, delay ring, and spike vectors are copied
    to the selected CUDA device.  ``run`` returns host scalars only at the end
    of each call, so the hot path does not copy the full state back to Python.
    """

    def __init__(
        self,
        connectome,
        *,
        params: FlyBrainParams | None = None,
        seed: int = 0,
        device: int = 0,
    ) -> None:
        cp = _require_cupy()
        self.cp = cp
        self.connectome = connectome
        self.params = params or FlyBrainParams()
        self.device = cp.cuda.Device(int(device))
        self.device.use()
        self.rng = cp.random.default_rng(seed)

        n = connectome.neuron_count
        self.n = int(n)
        self.indptr = cp.asarray(connectome.indptr, dtype=cp.int64)
        self.post_indices = cp.asarray(connectome.post_indices, dtype=cp.int32)
        signed = cp.asarray(connectome.signed_synapse_counts, dtype=cp.float32)
        self.signed = signed
        self._scatter_kernel = cp.RawKernel(
            r'''
            extern "C" __global__
            void scatter_csr_rows(
                const int* fired,
                const long long* indptr,
                const int* posts,
                const float* signed_values,
                float* target,
                const float scale
            ) {
                const int row = blockIdx.x;
                const int pre = fired[row];
                const long long start = indptr[pre];
                const long long stop = indptr[pre + 1];
                for (long long edge = start + threadIdx.x;
                     edge < stop;
                     edge += blockDim.x) {
                    atomicAdd(&target[posts[edge]], signed_values[edge] * scale);
                }
            }
            ''',
            "scatter_csr_rows",
        )

        self.v = cp.full(n, self.params.resting_mv, dtype=cp.float32)
        self.g = cp.zeros(n, dtype=cp.float32)
        self.refractory_until = cp.zeros(n, dtype=cp.int64)
        self.step_index = 0
        self.delay_steps = max(1, int(round(self.params.delay_ms / self.params.dt_ms)))
        self.refractory_steps = max(
            1, int(math.ceil(self.params.refractory_ms / self.params.dt_ms))
        )
        self.delay_ring = [cp.zeros(n, dtype=cp.float32) for _ in range(self.delay_steps + 1)]

        dt = self.params.dt_ms
        tm = self.params.membrane_tau_ms
        ts = self.params.synapse_tau_ms
        self._membrane_decay = math.exp(-dt / tm)
        self._synapse_decay = math.exp(-dt / ts)
        if math.isclose(tm, ts):
            self._g_to_v = (dt / tm) * self._membrane_decay
        else:
            self._g_to_v = ts / (ts - tm) * (self._synapse_decay - self._membrane_decay)

    @property
    def device_name(self) -> str:
        raw = self.cp.cuda.runtime.getDeviceProperties(self.device.id)["name"]
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    def reset(self) -> None:
        self.v.fill(self.params.resting_mv)
        self.g.fill(0.0)
        self.refractory_until.fill(0)
        for slot in self.delay_ring:
            slot.fill(0.0)
        self.step_index = 0

    def indices_for_ids(self, body_ids):
        # ID lookup is a setup operation, not part of the GPU hot path.
        return self.cp.asarray(
            [self.connectome.index_of(int(body_id)) for body_id in body_ids],
            dtype=self.cp.int32,
        )

    def _schedule_spike_outputs(self, fired) -> int:
        """Scatter only the CSR edge ranges touched by this step.

        A full sparse-matrix multiply scans all 6.2M graph edges even when one
        neuron fires.  The CPU reference already exploits CSR row locality;
        this GPU path does the same, but performs the final accumulation on the
        device.  The small host transfer is only the list of fired neuron IDs.
        """
        cp = self.cp
        if len(fired) == 0:
            return 0
        lengths = self.indptr[fired + 1] - self.indptr[fired]
        active_fired = fired[lengths > 0]
        if len(active_fired) == 0:
            return 0
        target = self.delay_ring[
            (self.step_index + self.delay_steps) % len(self.delay_ring)
        ]
        self._scatter_kernel(
            (int(active_fired.shape[0]),),
            (128,),
            (
                active_fired,
                self.indptr,
                self.post_indices,
                self.signed,
                target,
                cp.float32(self.params.mv_per_synapse),
            ),
        )
        return int(cp.sum(lengths).get())

    def step(self, *, stimulus_indices=None, stimulus_rate_hz: float = 0.0):
        cp = self.cp
        p = self.params
        ring_index = self.step_index % len(self.delay_ring)
        due = self.delay_ring[ring_index]
        self.g += due
        due.fill(0.0)

        old_g = self.g.copy()
        y = self.v - p.resting_mv
        self.v = p.resting_mv + y * self._membrane_decay + old_g * self._g_to_v
        self.g = old_g * self._synapse_decay

        refractory = self.step_index < self.refractory_until
        self.v[refractory] = p.resting_mv
        self.g[refractory] = 0.0

        if stimulus_indices is not None and len(stimulus_indices) and stimulus_rate_hz > 0.0:
            probability = min(1.0, stimulus_rate_hz * p.dt_ms / 1000.0)
            stimulated = stimulus_indices[self.rng.random(len(stimulus_indices)) < probability]
            if len(stimulated):
                self.v[stimulated] += p.mv_per_synapse * p.poisson_drive_scale

        fired = cp.flatnonzero(
            (self.v > p.threshold_mv) & (self.step_index >= self.refractory_until)
        ).astype(cp.int32, copy=False)
        if len(fired):
            self._schedule_spike_outputs(fired)
            self.v[fired] = p.reset_mv
            self.g[fired] = 0.0
            self.refractory_until[fired] = self.step_index + self.refractory_steps

        self.step_index += 1
        return fired

    def run(self, *, duration_ms: float, stimulus_body_ids, stimulus_rate_hz: float = 100.0) -> dict[str, object]:
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        stimulus_indices = self.indices_for_ids(stimulus_body_ids)
        steps = max(1, int(math.ceil(duration_ms / self.params.dt_ms)))
        spike_counts = self.cp.zeros(self.n, dtype=self.cp.int32)
        started = time.perf_counter()
        for _ in range(steps):
            fired = self.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            if len(fired):
                spike_counts[fired] += 1
        self.cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        active = self.cp.flatnonzero(spike_counts)
        total_spikes = int(self.cp.sum(spike_counts).get())
        return {
            "backend": "cupy_cuda_csr",
            "gpu_used": True,
            "plasticity_used": False,
            "device": self.device_name,
            "device_id": int(self.device.id),
            "neuron_count": self.n,
            "edge_count": int(self.connectome.edge_count),
            "duration_ms": float(duration_ms),
            "dt_ms": float(self.params.dt_ms),
            "steps": steps,
            "stimulus_rate_hz": float(stimulus_rate_hz),
            "active_neurons": int(len(active)),
            "total_spikes": total_spikes,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / max(elapsed, 1e-9),
            "simulation_ms_per_second": (steps * self.params.dt_ms) / max(elapsed, 1e-9),
            "peak_memory_bytes": int(self.cp.get_default_memory_pool().used_bytes()),
        }


def run_gpu_smoke(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    duration_ms: float = 20.0,
    stimulus_rate_hz: float = 300.0,
    min_connection_synapses: int = 5,
    seed: int = 7,
) -> dict[str, object]:
    connectome = load_malecns_v1(
        data_dir,
        min_connection_synapses=min_connection_synapses,
    )
    stimulus = connectome.strongest_outgoing_ids(2)
    brain = GpuSparseFlyBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
    )
    result = brain.run(
        duration_ms=duration_ms,
        stimulus_body_ids=stimulus,
        stimulus_rate_hz=stimulus_rate_hz,
    )
    result["dataset"] = connectome.source
    result["stimulus_body_ids"] = list(stimulus)
    result["connectome"] = connectome.summary()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPU execution on the real MaleCNS CSR graph.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--stimulus-rate-hz", type=float, default=300.0)
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.download:
        download_malecns(args.data_dir)
    print(json.dumps(run_gpu_smoke(
        args.data_dir,
        duration_ms=args.duration_ms,
        stimulus_rate_hz=args.stimulus_rate_hz,
        min_connection_synapses=args.min_syn,
        seed=args.seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
