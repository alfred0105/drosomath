"""Generic, diagnostic-only symbol input and decision surfaces.

Phase F.1A deliberately keeps this module outside the learning stack.  It
maps a small fixed vocabulary to real connectome neurons and reports direct
population activity; it does not learn embeddings or train an external
decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np


SYMBOLS = ("A", "B", "C", "D")
NO_DECISION = "NO_DECISION"


def _as_neuron_ids(connectome):
    ids = getattr(connectome, "body_ids", None)
    if ids is None:
        ids = getattr(connectome, "flywire_ids", None)
    if ids is None:
        raise TypeError("connectome must expose real body_ids")
    return np.asarray(ids)


def _pairwise_jaccard(populations: Mapping[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for left, right in combinations(SYMBOLS, 2):
        a = set(int(x) for x in populations[left])
        b = set(int(x) for x in populations[right])
        union = a | b
        result[f"{left}|{right}"] = float(len(a & b) / len(union)) if union else 0.0
    return result


@dataclass(frozen=True, slots=True)
class SymbolInterfaceConfig:
    """Deterministic F.1A allocation and presentation settings."""

    symbols: tuple[str, ...] = SYMBOLS
    sensory_population_size: int = 32
    output_population_size: int = 32
    seed: int = 7
    dt_ms: float = 0.2
    default_duration_ms: float = 20.0
    default_stimulus_rate_hz: float = 205.0
    plastic_fraction: float = 0.05

    def __post_init__(self) -> None:
        if tuple(self.symbols) != SYMBOLS:
            raise ValueError("F.1A vocabulary is exactly A, B, C, D")
        if self.sensory_population_size < 1:
            raise ValueError("sensory_population_size must be >= 1")
        if self.output_population_size < 1:
            raise ValueError("output_population_size must be >= 1")
        if self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        if self.default_duration_ms <= 0.0:
            raise ValueError("default_duration_ms must be > 0")
        if self.default_stimulus_rate_hz < 0.0:
            raise ValueError("default_stimulus_rate_hz must be >= 0")
        if not 0.0 <= self.plastic_fraction <= 1.0:
            raise ValueError("plastic_fraction must be in [0, 1]")


class DistributedSymbolEncoder:
    """Map each initial symbol to a fixed disjoint real sensory population."""

    def __init__(self, connectome, config: SymbolInterfaceConfig | None = None):
        self.config = config or SymbolInterfaceConfig()
        self.connectome = connectome
        self._body_ids = _as_neuron_ids(connectome)
        n = int(getattr(connectome, "neuron_count", len(self._body_ids)))
        if len(self._body_ids) != n:
            raise ValueError("connectome body_ids and neuron_count disagree")

        indptr = np.asarray(connectome.indptr)
        outgoing = np.diff(indptr) > 0
        candidates = np.flatnonzero(outgoing).astype(np.int32, copy=False)
        required = len(SYMBOLS) * self.config.sensory_population_size
        if len(candidates) < required:
            raise ValueError(
                f"connectome has only {len(candidates)} usable sensory neurons; "
                f"{required} are required"
            )

        # This RNG is local to allocation.  In particular, never use
        # ``brain.rng``: constructing the interface must not alter simulation
        # stochasticity.
        rng = np.random.default_rng(self.config.seed)
        selected = rng.permutation(candidates)[:required]
        self._indices: dict[str, np.ndarray] = {}
        cursor = 0
        for symbol in SYMBOLS:
            values = np.sort(selected[cursor : cursor + self.config.sensory_population_size])
            values = values.astype(np.int32, copy=False)
            values.setflags(write=False)
            self._indices[symbol] = values
            cursor += self.config.sensory_population_size

    @property
    def symbols(self) -> tuple[str, ...]:
        return SYMBOLS

    @property
    def populations(self) -> dict[str, np.ndarray]:
        return {symbol: values.copy() for symbol, values in self._indices.items()}

    @property
    def body_id_populations(self) -> dict[str, np.ndarray]:
        return {symbol: self._body_ids[values].copy() for symbol, values in self._indices.items()}

    def indices_for(self, symbol: str) -> np.ndarray:
        try:
            return self._indices[symbol].copy()
        except KeyError as exc:
            raise KeyError(f"unknown symbol {symbol!r}; expected one of {SYMBOLS}") from exc

    def body_ids_for(self, symbol: str) -> np.ndarray:
        return self._body_ids[self.indices_for(symbol)].copy()


class SymbolDecisionSurface:
    """Read direct output population rates without a trainable decoder."""

    def __init__(self, connectome, config: SymbolInterfaceConfig | None = None, *, excluded_indices=()):
        self.config = config or SymbolInterfaceConfig()
        self.connectome = connectome
        self._body_ids = _as_neuron_ids(connectome)
        n = int(getattr(connectome, "neuron_count", len(self._body_ids)))
        incoming = np.bincount(
            np.asarray(connectome.post_indices, dtype=np.int64),
            minlength=n,
        )
        candidates = np.flatnonzero(incoming > 0).astype(np.int32, copy=False)
        excluded = np.asarray(tuple(excluded_indices), dtype=np.int64)
        if len(excluded):
            if int(excluded.min()) < 0 or int(excluded.max()) >= n:
                raise IndexError("excluded output indices are outside the connectome")
            candidates = candidates[~np.isin(candidates, excluded)]

        required = len(SYMBOLS) * self.config.output_population_size
        if len(candidates) < required:
            raise ValueError(
                f"connectome has only {len(candidates)} usable output neurons "
                f"after sensory exclusion; {required} are required"
            )
        rng = np.random.default_rng(self.config.seed + 1)
        selected = rng.permutation(candidates)[:required]
        self._indices: dict[str, np.ndarray] = {}
        cursor = 0
        for symbol in SYMBOLS:
            values = np.sort(selected[cursor : cursor + self.config.output_population_size])
            values = values.astype(np.int32, copy=False)
            values.setflags(write=False)
            self._indices[symbol] = values
            cursor += self.config.output_population_size

    @property
    def symbols(self) -> tuple[str, ...]:
        return SYMBOLS

    @property
    def populations(self) -> dict[str, np.ndarray]:
        return {symbol: values.copy() for symbol, values in self._indices.items()}

    @property
    def body_id_populations(self) -> dict[str, np.ndarray]:
        return {symbol: self._body_ids[values].copy() for symbol, values in self._indices.items()}

    def indices_for(self, symbol: str) -> np.ndarray:
        try:
            return self._indices[symbol].copy()
        except KeyError as exc:
            raise KeyError(f"unknown symbol {symbol!r}; expected one of {SYMBOLS}") from exc

    def body_ids_for(self, symbol: str) -> np.ndarray:
        return self._body_ids[self.indices_for(symbol)].copy()

    def decide(
        self,
        output_rates_hz: Mapping[str, float],
        *,
        total_output_spikes: int | None = None,
    ) -> str:
        values = {symbol: float(output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS}
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("output rates must be finite and non-negative")
        highest = max(values.values())
        if highest <= 0.0:
            return NO_DECISION
        winners = [symbol for symbol, value in values.items() if value == highest]
        if len(winners) != 1:
            return NO_DECISION
        return winners[0]

    def pairwise_jaccard(self) -> dict[str, float]:
        return _pairwise_jaccard(self._indices)


def _reachable_frontier(connectome, frontier: np.ndarray, visited: np.ndarray) -> np.ndarray:
    """Expand one CSR graph hop without changing the connectome."""
    if len(frontier) == 0:
        return np.empty(0, dtype=np.int32)
    indptr = np.asarray(connectome.indptr)
    posts = np.asarray(connectome.post_indices, dtype=np.int32)
    starts = indptr[frontier]
    stops = indptr[frontier + 1]
    lengths = (stops - starts).astype(np.int64, copy=False)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32)
    offsets = np.arange(total, dtype=np.int64)
    block_offsets = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    edge_indices = np.repeat(starts, lengths) + offsets - block_offsets
    candidates = np.unique(posts[edge_indices])
    fresh = candidates[~visited[candidates]]
    if len(fresh):
        visited[fresh] = True
    return fresh.astype(np.int32, copy=False)


def audit_symbol_reachability(connectome, interface: SymbolInterface, *, max_hops: int = 3) -> dict[str, dict[str, dict[str, object]]]:
    """Measure bounded sensory-to-output reachability on immutable anatomy.

    Counts are cumulative reachable output neurons (within one, two, or three
    hops), so duplicate paths never duplicate a neuron.  The function only
    reads CSR topology and interface populations; it does not access a brain,
    plasticity state, labels, or a random generator.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    n = int(getattr(connectome, "neuron_count", len(_as_neuron_ids(connectome))))
    outputs = interface.output_populations
    result: dict[str, dict[str, dict[str, object]]] = {}
    for input_symbol in SYMBOLS:
        visited = np.zeros(n, dtype=np.bool_)
        frontier = interface.encoder.indices_for(input_symbol)
        visited[frontier] = True
        first_hops = {
            symbol: np.zeros(len(indices), dtype=np.int8)
            for symbol, indices in outputs.items()
        }
        cumulative_counts = {symbol: [0] * max_hops for symbol in SYMBOLS}
        for hop in range(1, max_hops + 1):
            frontier = _reachable_frontier(connectome, frontier, visited)
            for output_symbol, indices in outputs.items():
                newly_reached = (first_hops[output_symbol] == 0) & np.isin(indices, frontier)
                first_hops[output_symbol][newly_reached] = hop
                cumulative_counts[output_symbol][hop - 1] = int(
                    np.count_nonzero(first_hops[output_symbol] > 0)
                )
        result[input_symbol] = {}
        for output_symbol, indices in outputs.items():
            counts = cumulative_counts[output_symbol]
            hops = first_hops[output_symbol]
            reachable_hops = hops[hops > 0]
            result[input_symbol][output_symbol] = {
                "hop1_output_neurons": counts[0],
                "hop2_output_neurons": counts[1] if max_hops >= 2 else counts[0],
                "hop3_output_neurons": counts[2] if max_hops >= 3 else counts[-1],
                "shortest_reachable_hop": int(reachable_hops.min()) if len(reachable_hops) else None,
                "reachable_output_fraction_at_1hop": float(counts[0] / len(indices)),
                "reachable_output_fraction_at_2hop": float(
                    (counts[1] if max_hops >= 2 else counts[0]) / len(indices)
                ),
                "reachable_output_fraction_at_3hop": float(
                    (counts[2] if max_hops >= 3 else counts[-1]) / len(indices)
                ),
            }
    return result


@dataclass(frozen=True, slots=True)
class SymbolPresentationResult:
    input_symbol: str
    output_population_rates_hz: dict[str, float]
    decision: str
    total_output_spikes: int
    network_activity: dict[str, int]
    first_output_spike_ms: float | None = None
    active_neuron_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "input_symbol": self.input_symbol,
            "output_population_rates_hz": dict(self.output_population_rates_hz),
            "decision": self.decision,
            "total_output_spikes": int(self.total_output_spikes),
            "network_activity": dict(self.network_activity),
            "first_output_spike_ms": self.first_output_spike_ms,
        }


def symbol_decision_surface_ready(
    observations: Mapping[str, list[SymbolPresentationResult]],
) -> bool:
    """Require measurable output activity for every input symbol."""
    return all(
        any(int(observation.total_output_spikes) > 0 for observation in observations.get(symbol, ()))
        for symbol in SYMBOLS
    )


def symbol_f1b_ready(*, sensory_interface_ready: bool, decision_surface_ready: bool) -> bool:
    """F.1B may start only after both observational surfaces are usable."""
    return bool(sensory_interface_ready and decision_surface_ready)


class SymbolInterface:
    """Own both deterministic populations and expose structural diagnostics."""

    def __init__(self, connectome, config: SymbolInterfaceConfig | None = None):
        self.config = config or SymbolInterfaceConfig()
        self.encoder = DistributedSymbolEncoder(connectome, self.config)
        self.decision_surface = SymbolDecisionSurface(
            connectome,
            self.config,
            excluded_indices=np.concatenate(tuple(self.encoder.populations.values())),
        )

    @property
    def sensory_populations(self) -> dict[str, np.ndarray]:
        return self.encoder.populations

    @property
    def output_populations(self) -> dict[str, np.ndarray]:
        return self.decision_surface.populations

    def allocation_summary(self) -> dict[str, object]:
        sensory = self.sensory_populations
        output = self.output_populations
        sensory_union = np.concatenate(tuple(sensory.values()))
        output_union = np.concatenate(tuple(output.values()))
        return {
            "sensory_population_size": self.config.sensory_population_size,
            "output_population_size": self.config.output_population_size,
            "sensory_output_overlap": int(len(np.intersect1d(sensory_union, output_union))),
            "sensory_pairwise_overlap": _pairwise_jaccard(sensory),
            "output_pairwise_overlap": self.decision_surface.pairwise_jaccard(),
            "sensory_body_ids": {
                symbol: [int(x) for x in self.encoder.body_ids_for(symbol)] for symbol in SYMBOLS
            },
            "output_body_ids": {
                symbol: [int(x) for x in self.decision_surface.body_ids_for(symbol)] for symbol in SYMBOLS
            },
        }


def _snapshot_persistent_state(brain) -> dict[str, object]:
    """Capture the state F.1A promises not to mutate during observation."""
    plasticity = brain.plasticity
    snapshot: dict[str, object] = {
        "multiplier": plasticity.multiplier.copy(),
        "usage_ema": plasticity.usage_ema.copy(),
        "eligibility": plasticity.eligibility.copy(),
        "stability": plasticity.stability.copy(),
        "plastic_mask": plasticity.plastic_mask.copy(),
        "allocation_overrides": {
            key: value.copy() for key, value in plasticity.allocation_overrides().items()
        },
    }
    overlay = getattr(brain, "structural_overlay", None)
    if overlay is not None:
        snapshot["overlay"] = {
            name: getattr(overlay, name).copy()
            for name in ("post_index", "signed_strength", "eligibility", "usage_ema")
            if hasattr(overlay, name)
        }
    return snapshot


def _restore_persistent_state(brain, snapshot: dict[str, object]) -> None:
    plasticity = brain.plasticity
    for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask"):
        getattr(plasticity, name)[:] = snapshot[name]
    overrides = snapshot["allocation_overrides"]
    plasticity.restore_allocation_overrides(
        overrides["promoted_edges"], overrides["retired_edges"]
    )
    overlay_snapshot = snapshot.get("overlay")
    overlay = getattr(brain, "structural_overlay", None)
    if overlay is not None and overlay_snapshot is not None:
        for name, values in overlay_snapshot.items():
            getattr(overlay, name)[:] = values


class SymbolSession:
    """Read-only symbol presentation on an existing PlasticMaleCNSBrain."""

    def __init__(self, brain, interface: SymbolInterface):
        self.brain = brain
        self.interface = interface
        if brain.connectome is not interface.encoder.connectome:
            raise ValueError("brain and symbol interface must use the same connectome")
        n = brain.connectome.neuron_count
        self._output_lookup = np.full(n, -1, dtype=np.int8)
        self._output_symbols = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}
        for symbol, indices in interface.output_populations.items():
            self._output_lookup[indices] = self._output_symbols[symbol]

    def present(
        self,
        *,
        symbol: str,
        duration_ms: float | None = None,
        stimulus_rate_hz: float | None = None,
        learn: bool = False,
    ) -> SymbolPresentationResult:
        if learn:
            raise NotImplementedError("F.1A is observational; symbol learning belongs to F.1B")
        if symbol not in SYMBOLS:
            raise KeyError(f"unknown symbol {symbol!r}; expected one of {SYMBOLS}")
        duration = self.interface.config.default_duration_ms if duration_ms is None else float(duration_ms)
        rate = self.interface.config.default_stimulus_rate_hz if stimulus_rate_hz is None else float(stimulus_rate_hz)
        if duration <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if rate < 0.0:
            raise ValueError("stimulus_rate_hz must be >= 0")

        steps = max(1, int(math.ceil(duration / self.brain.params.dt_ms)))
        counts = np.zeros(len(SYMBOLS), dtype=np.int32)
        active_neurons: set[int] = set()
        network_spikes = 0
        first_output_spike_ms: float | None = None
        snapshot = _snapshot_persistent_state(self.brain)
        previous_tracking = self.brain.set_plasticity_tracking(False)
        try:
            self.brain.reset()
            sensory_indices = self.interface.encoder.indices_for(symbol)
            for step_index in range(steps):
                fired, _ = self.brain.step(
                    stimulus_indices=sensory_indices,
                    stimulus_rate_hz=rate,
                )
                network_spikes += int(len(fired))
                if len(fired):
                    active_neurons.update(int(index) for index in fired)
                    local = self._output_lookup[fired]
                    local = local[local >= 0]
                    if len(local):
                        np.add.at(counts, local, 1)
                        if first_output_spike_ms is None:
                            first_output_spike_ms = float(step_index * self.brain.params.dt_ms)
        finally:
            self.brain.set_plasticity_tracking(previous_tracking)
            _restore_persistent_state(self.brain, snapshot)

        seconds = duration / 1000.0
        rates = {
            symbol_name: float(counts[index]) / seconds
            for index, symbol_name in enumerate(SYMBOLS)
        }
        decision = self.interface.decision_surface.decide(
            rates,
            total_output_spikes=int(counts.sum()),
        )
        return SymbolPresentationResult(
            input_symbol=symbol,
            output_population_rates_hz=rates,
            decision=decision,
            total_output_spikes=int(counts.sum()),
            network_activity={
                "total_spikes": int(network_spikes),
                "unique_neurons": int(len(active_neurons)),
                "steps": int(steps),
            },
            first_output_spike_ms=first_output_spike_ms,
            active_neuron_indices=tuple(sorted(active_neurons)),
        )


__all__ = [
    "DistributedSymbolEncoder",
    "NO_DECISION",
    "SYMBOLS",
    "SymbolDecisionSurface",
    "SymbolInterface",
    "SymbolInterfaceConfig",
    "SymbolPresentationResult",
    "SymbolSession",
    "audit_symbol_reachability",
    "symbol_decision_surface_ready",
    "symbol_f1b_ready",
]
