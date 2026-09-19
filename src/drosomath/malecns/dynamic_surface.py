"""Label-independent dynamic generic decision-surface construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .symbol_interface import SYMBOLS, SymbolInterface, SymbolInterfaceConfig


@dataclass(frozen=True, slots=True)
class DynamicCandidateFeatures:
    neuron_index: int
    body_id: int
    distinct_symbol_count_active: int
    distinct_seed_count_active: int
    total_active_run_count: int


def collect_dynamic_candidate_features(
    connectome,
    discovery_runs: Iterable[Mapping[str, object]],
    *,
    sensory_indices: Iterable[int],
) -> tuple[list[DynamicCandidateFeatures], int]:
    """Aggregate generic activity breadth and exclude all sensory neurons.

    Each discovery run contains only ``input_symbol``, ``brain_seed`` and an
    iterable of real ``active_neuron_indices``.  The symbol is used only to
    count breadth, never to assign an output label.
    """
    body_ids = np.asarray(getattr(connectome, "body_ids", connectome.flywire_ids))
    n = int(getattr(connectome, "neuron_count", len(body_ids)))
    excluded = {int(index) for index in sensory_indices}
    per_neuron_symbols: dict[int, set[str]] = {}
    per_neuron_seeds: dict[int, set[int]] = {}
    per_neuron_runs: Counter[int] = Counter()
    for run in discovery_runs:
        symbol = str(run["input_symbol"])
        seed = int(run["brain_seed"])
        active_indices = run.get("active_neuron_indices", run.get("_active_neuron_indices", ()))
        for raw_index in active_indices:
            index = int(raw_index)
            if index < 0 or index >= n or index in excluded:
                continue
            per_neuron_symbols.setdefault(index, set()).add(symbol)
            per_neuron_seeds.setdefault(index, set()).add(seed)
            per_neuron_runs[index] += 1
    features = [
        DynamicCandidateFeatures(
            neuron_index=index,
            body_id=int(body_ids[index]),
            distinct_symbol_count_active=len(per_neuron_symbols[index]),
            distinct_seed_count_active=len(per_neuron_seeds[index]),
            total_active_run_count=int(per_neuron_runs[index]),
        )
        for index in per_neuron_symbols
    ]
    features.sort(
        key=lambda item: (
            -item.distinct_symbol_count_active,
            -item.distinct_seed_count_active,
            -item.total_active_run_count,
            item.neuron_index,
            item.body_id,
        )
    )
    return features, len(features)


def select_dynamic_generic_candidates(
    features: Iterable[DynamicCandidateFeatures],
    *,
    selected_count: int = 128,
) -> list[DynamicCandidateFeatures]:
    """Select the top generic candidates with no target-label criterion."""
    if selected_count < 1:
        raise ValueError("selected_count must be >= 1")
    ranked = sorted(
        list(features),
        key=lambda item: (
            -item.distinct_symbol_count_active,
            -item.distinct_seed_count_active,
            -item.total_active_run_count,
            item.neuron_index,
            item.body_id,
        ),
    )
    if len(ranked) < selected_count:
        raise ValueError(
            f"dynamic generic candidate pool has {len(ranked)} neurons; "
            f"{selected_count} are required"
        )
    return ranked[:selected_count]


def partition_dynamic_generic_candidates(
    selected: Iterable[DynamicCandidateFeatures],
    *,
    seed: int = 17,
    population_size: int = 32,
) -> dict[str, np.ndarray]:
    """Shuffle once with a local RNG and split into four equal groups."""
    selected_list = list(selected)
    required = len(SYMBOLS) * population_size
    if len(selected_list) != required:
        raise ValueError(f"exactly {required} selected candidates are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(selected_list))
    populations: dict[str, np.ndarray] = {}
    for offset, symbol in enumerate(SYMBOLS):
        chosen = [selected_list[int(index)].neuron_index for index in order[offset * population_size : (offset + 1) * population_size]]
        populations[symbol] = np.asarray(chosen, dtype=np.int32)
    return populations


def build_dynamic_generic_interface(
    connectome,
    config: SymbolInterfaceConfig,
    selected: Iterable[DynamicCandidateFeatures],
    *,
    decision_surface_seed: int = 17,
) -> SymbolInterface:
    """Freeze a label-independent dynamic surface after discovery."""
    populations = partition_dynamic_generic_candidates(
        selected,
        seed=decision_surface_seed,
        population_size=config.output_population_size,
    )
    return SymbolInterface.with_output_populations(
        connectome,
        config,
        populations,
        source="dynamic_generic",
    )


def viability_histogram(
    selected: Iterable[DynamicCandidateFeatures],
    field: str,
) -> dict[str, int]:
    values = [int(getattr(feature, field)) for feature in selected]
    counts = Counter(values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


__all__ = [
    "DynamicCandidateFeatures",
    "build_dynamic_generic_interface",
    "collect_dynamic_candidate_features",
    "partition_dynamic_generic_candidates",
    "select_dynamic_generic_candidates",
    "viability_histogram",
]
