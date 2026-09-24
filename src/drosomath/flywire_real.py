from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SHIU_COMPLETENESS = "Completeness_783.csv"
SHIU_CONNECTIVITY = "Connectivity_783.parquet"
EXPECTED_NEURONS_V783 = 139_255


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - setup path
        raise RuntimeError(
            "FlyWire mode needs NumPy. Run: python -m pip install -e \".[flywire]\""
        ) from exc
    return np


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - setup path
        raise RuntimeError(
            "FlyWire data loading needs PyArrow. Run: python -m pip install -e \".[flywire]\""
        ) from exc
    return pq


@dataclass(frozen=True, slots=True)
class FlyBrainParams:
    """Whole-brain LIF parameters from Shiu et al., Nature 2024."""

    resting_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    mv_per_synapse: float = 0.275
    poisson_drive_scale: float = 250.0
    dt_ms: float = 0.2

    def __post_init__(self) -> None:
        if self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        if self.membrane_tau_ms <= 0.0 or self.synapse_tau_ms <= 0.0:
            raise ValueError("time constants must be > 0")
        if self.threshold_mv <= self.resting_mv:
            raise ValueError("threshold_mv must exceed resting_mv")


@dataclass(slots=True)
class FlyWireConnectome:
    """CSR-like sparse topology retaining the real FlyWire neuron IDs."""

    flywire_ids: object
    indptr: object
    post_indices: object
    signed_synapse_counts: object
    outgoing_strength: object
    source: str = "FlyWire FAFB v783 / Shiu et al. 2024"

    @property
    def neuron_count(self) -> int:
        return int(len(self.flywire_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.post_indices))

    @property
    def synapse_count_abs(self) -> int:
        np = _require_numpy()
        return int(np.abs(self.signed_synapse_counts).sum())

    def index_of(self, flywire_id: int) -> int:
        np = _require_numpy()
        matches = np.flatnonzero(self.flywire_ids == int(flywire_id))
        if matches.size == 0:
            raise KeyError(f"FlyWire neuron {flywire_id} is not present")
        return int(matches[0])

    def strongest_outgoing_ids(self, count: int) -> tuple[int, ...]:
        np = _require_numpy()
        if count < 1:
            return ()
        count = min(count, self.neuron_count)
        if count == self.neuron_count:
            idx = np.argsort(self.outgoing_strength)[::-1]
        else:
            idx = np.argpartition(self.outgoing_strength, -count)[-count:]
            idx = idx[np.argsort(self.outgoing_strength[idx])[::-1]]
        return tuple(int(x) for x in self.flywire_ids[idx])


def _read_flywire_ids(path: Path):
    """Read the first CSV column used as FlyWire ID by the published model."""
    np = _require_numpy()
    ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                ids.append(int(row[0]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse FlyWire ID from first column of {path} "
                    f"at row {row_number}"
                ) from exc
    if not ids:
        raise ValueError(f"{path} contains no neuron IDs")
    return np.asarray(ids, dtype=np.int64)


def load_shiu_v783(
    data_dir: str | Path,
    *,
    min_connection_synapses: int = 1,
    require_published_neuron_count: bool = True,
) -> FlyWireConnectome:
    """Load the real FlyWire v783 topology distributed with Shiu et al. 2024.

    ``Completeness_783.csv`` defines the mapping from simulation index to real
    FlyWire root ID. ``Connectivity_783.parquet`` contains the signed connection
    strength used by the published whole-brain LIF model.
    """
    if min_connection_synapses < 1:
        raise ValueError("min_connection_synapses must be >= 1")

    np = _require_numpy()
    pq = _require_pyarrow_parquet()
    root = Path(data_dir)
    completeness = root / SHIU_COMPLETENESS
    connectivity = root / SHIU_CONNECTIVITY
    missing = [str(p) for p in (completeness, connectivity) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing FlyWire v783 files: " + ", ".join(missing))

    flywire_ids = _read_flywire_ids(completeness)
    if require_published_neuron_count and len(flywire_ids) != EXPECTED_NEURONS_V783:
        raise ValueError(
            f"Expected {EXPECTED_NEURONS_V783:,} FlyWire v783 neurons, "
            f"got {len(flywire_ids):,}"
        )

    table = pq.read_table(
        connectivity,
        columns=[
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Excitatory x Connectivity",
        ],
    )
    pre = table["Presynaptic_Index"].to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    post = table["Postsynaptic_Index"].to_numpy(zero_copy_only=False).astype(
        np.int32, copy=False
    )
    signed = table["Excitatory x Connectivity"].to_numpy(
        zero_copy_only=False
    ).astype(np.float32, copy=False)

    n = len(flywire_ids)
    valid = (pre >= 0) & (post >= 0) & (pre < n) & (post < n)
    valid &= np.abs(signed) >= float(min_connection_synapses)
    pre = pre[valid]
    post = post[valid]
    signed = signed[valid]

    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = post[order]
    signed = signed[order]

    counts = np.bincount(pre, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    outgoing_strength = np.bincount(
        pre,
        weights=np.abs(signed),
        minlength=n,
    ).astype(np.float32, copy=False)

    return FlyWireConnectome(
        flywire_ids=flywire_ids,
        indptr=indptr,
        post_indices=post,
        signed_synapse_counts=signed,
        outgoing_strength=outgoing_strength,
    )


class SparseFlyBrain:
    """Sparse whole-brain LIF simulator on the real FlyWire connectome.

    The parameterization follows Shiu et al. 2024. State decay is calculated
    analytically between discrete event times; synaptic delivery is sparse and
    occurs only for neurons that actually spike.
    """

    def __init__(
        self,
        connectome: FlyWireConnectome,
        *,
        params: FlyBrainParams | None = None,
        seed: int = 0,
    ) -> None:
        np = _require_numpy()
        self.np = np
        self.connectome = connectome
        self.params = params or FlyBrainParams()
        self.rng = np.random.default_rng(seed)
        n = connectome.neuron_count
        self.v = np.full(n, self.params.resting_mv, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.refractory_until = np.zeros(n, dtype=np.int64)
        self.step_index = 0

        self.delay_steps = max(1, int(round(self.params.delay_ms / self.params.dt_ms)))
        self.refractory_steps = max(
            1, int(math.ceil(self.params.refractory_ms / self.params.dt_ms))
        )
        self._delay_ring = [
            np.zeros(n, dtype=np.float32) for _ in range(self.delay_steps + 1)
        ]
        self._id_to_index = {
            int(flywire_id): idx
            for idx, flywire_id in enumerate(connectome.flywire_ids)
        }

        dt = self.params.dt_ms
        tm = self.params.membrane_tau_ms
        ts = self.params.synapse_tau_ms
        self._membrane_decay = math.exp(-dt / tm)
        self._synapse_decay = math.exp(-dt / ts)
        if math.isclose(tm, ts):
            self._g_to_v = (dt / tm) * self._membrane_decay
        else:
            self._g_to_v = ts / (ts - tm) * (
                self._synapse_decay - self._membrane_decay
            )

    def reset(self) -> None:
        self.v.fill(self.params.resting_mv)
        self.g.fill(0.0)
        self.refractory_until.fill(0)
        for slot in self._delay_ring:
            slot.fill(0.0)
        self.step_index = 0

    def indices_for_ids(self, flywire_ids: Iterable[int]):
        np = self.np
        idx: list[int] = []
        missing: list[int] = []
        for flywire_id in flywire_ids:
            found = self._id_to_index.get(int(flywire_id))
            if found is None:
                missing.append(int(flywire_id))
            else:
                idx.append(found)
        if missing:
            raise KeyError(f"Unknown FlyWire IDs: {missing[:8]}")
        return np.asarray(idx, dtype=np.int32)

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
        signed = self.connectome.signed_synapse_counts
        for pre in fired_indices:
            start = int(indptr[int(pre)])
            stop = int(indptr[int(pre) + 1])
            if start == stop:
                continue
            np.add.at(target_slot, posts[start:stop], signed[start:stop] * scale)
            transferred += stop - start
        return transferred

    def step(
        self,
        *,
        stimulus_indices=None,
        stimulus_rate_hz: float = 0.0,
    ):
        np = self.np
        p = self.params

        due_slot = self._delay_ring[self.step_index % len(self._delay_ring)]
        self.g += due_slot
        due_slot.fill(0.0)

        old_g = self.g.copy()
        y = self.v - p.resting_mv
        self.v[:] = p.resting_mv + y * self._membrane_decay + old_g * self._g_to_v
        self.g[:] = old_g * self._synapse_decay

        refractory = self.step_index < self.refractory_until
        if refractory.any():
            self.v[refractory] = p.resting_mv
            self.g[refractory] = 0.0

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

        fired = np.flatnonzero(
            (self.v > p.threshold_mv)
            & (self.step_index >= self.refractory_until)
        ).astype(np.int32, copy=False)
        transferred = self._schedule_spike_outputs(fired)

        if len(fired):
            self.v[fired] = p.reset_mv
            self.g[fired] = 0.0
            self.refractory_until[fired] = self.step_index + self.refractory_steps

        self.step_index += 1
        return fired, transferred

    def run(
        self,
        *,
        duration_ms: float,
        stimulus_ids: Iterable[int],
        stimulus_rate_hz: float = 100.0,
        top_fired: int = 20,
    ) -> dict[str, object]:
        np = self.np
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        stimulus_ids = tuple(int(x) for x in stimulus_ids)
        stimulus_indices = self.indices_for_ids(stimulus_ids)
        steps = max(1, int(math.ceil(duration_ms / self.params.dt_ms)))
        spike_counts = np.zeros(self.connectome.neuron_count, dtype=np.int32)
        total_transfers = 0

        for _ in range(steps):
            fired, transferred = self.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            if len(fired):
                spike_counts[fired] += 1
            total_transfers += transferred

        active = np.flatnonzero(spike_counts)
        if len(active):
            order = active[np.argsort(spike_counts[active])[::-1]][:top_fired]
            top = [
                {
                    "flywire_id": int(self.connectome.flywire_ids[i]),
                    "spikes": int(spike_counts[i]),
                }
                for i in order
            ]
        else:
            top = []

        return {
            "dataset": self.connectome.source,
            "neuron_count": self.connectome.neuron_count,
            "edge_count": self.connectome.edge_count,
            "absolute_synapse_count_in_loaded_edges": self.connectome.synapse_count_abs,
            "duration_ms": float(duration_ms),
            "dt_ms": self.params.dt_ms,
            "stimulus_rate_hz": float(stimulus_rate_hz),
            "stimulus_flywire_ids": list(stimulus_ids),
            "active_neurons": int(len(active)),
            "total_spikes": int(spike_counts.sum()),
            "total_synaptic_edge_deliveries": int(total_transfers),
            "top_firing_neurons": top,
            "parameters": {
                "resting_mv": self.params.resting_mv,
                "threshold_mv": self.params.threshold_mv,
                "membrane_tau_ms": self.params.membrane_tau_ms,
                "synapse_tau_ms": self.params.synapse_tau_ms,
                "refractory_ms": self.params.refractory_ms,
                "delay_ms": self.params.delay_ms,
                "mv_per_synapse": self.params.mv_per_synapse,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real FlyWire v783 whole-brain DrosoMath simulation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/flywire_v783"))
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--stimulus-rate-hz", type=float, default=100.0)
    parser.add_argument("--auto-stimuli", type=int, default=3)
    parser.add_argument("--stimulus-id", type=int, action="append", default=[])
    parser.add_argument("--min-connection-synapses", type=int, default=1)
    parser.add_argument("--dt-ms", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    connectome = load_shiu_v783(
        args.data_dir,
        min_connection_synapses=args.min_connection_synapses,
    )
    stimulus_ids = tuple(args.stimulus_id)
    if not stimulus_ids:
        stimulus_ids = connectome.strongest_outgoing_ids(args.auto_stimuli)

    brain = SparseFlyBrain(
        connectome,
        params=FlyBrainParams(dt_ms=args.dt_ms),
        seed=args.seed,
    )
    result = brain.run(
        duration_ms=args.duration_ms,
        stimulus_ids=stimulus_ids,
        stimulus_rate_hz=args.stimulus_rate_hz,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
