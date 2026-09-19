"""Task-independent aggregation helpers for the F.1A.2 audit."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


def classify_output_observability(active_seed_count: int, total_seed_count: int) -> str:
    """Classify measured output activity without treating correctness as signal."""
    if total_seed_count < 1:
        raise ValueError("total_seed_count must be >= 1")
    if active_seed_count < 0 or active_seed_count > total_seed_count:
        raise ValueError("active_seed_count must be within total_seed_count")
    if active_seed_count >= 2:
        return "reliably_observable"
    if active_seed_count == 1:
        return "intermittently_observable"
    return "silent"


def burst_outlier_symbols(
    network_spikes_by_symbol: Mapping[str, int | float],
    *,
    multiplier: float = 5.0,
) -> list[dict[str, float | str]]:
    """Return symbols at least ``multiplier`` times the other-symbol median."""
    if multiplier <= 0.0:
        raise ValueError("multiplier must be > 0")
    result: list[dict[str, float | str]] = []
    for symbol, raw_value in network_spikes_by_symbol.items():
        others = [float(value) for key, value in network_spikes_by_symbol.items() if key != symbol]
        if not others:
            continue
        others.sort()
        middle = len(others) // 2
        other_median = (
            others[middle]
            if len(others) % 2
            else (others[middle - 1] + others[middle]) / 2.0
        )
        value = float(raw_value)
        if other_median > 0.0 and value >= multiplier * other_median:
            result.append({
                "symbol": symbol,
                "network_spikes": value,
                "other_symbol_median": other_median,
                "ratio": value / other_median,
            })
    return result


def count_candidate_dynamic_pool(
    active_neurons_by_symbol: Mapping[str, Iterable[int]],
    *,
    excluded_indices: Iterable[int] = (),
) -> dict[str, int]:
    """Count real neurons active for at least N symbol contexts.

    The input is already aggregated per symbol (for example, the union across
    matched seeds).  It deliberately does not assign neurons to labels.
    """
    excluded = {int(index) for index in excluded_indices}
    symbol_sets = {
        symbol: {int(index) for index in indices if int(index) not in excluded}
        for symbol, indices in active_neurons_by_symbol.items()
    }
    counts = Counter(index for indices in symbol_sets.values() for index in indices)
    return {
        "active_for_at_least_1_symbol": int(sum(value >= 1 for value in counts.values())),
        "active_for_at_least_2_symbols": int(sum(value >= 2 for value in counts.values())),
        "active_for_at_least_3_symbols": int(sum(value >= 3 for value in counts.values())),
        "active_for_all_4_symbols": int(sum(value >= 4 for value in counts.values())),
    }


def allocation_fingerprint(interface) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    """Return a deterministic comparison key for an unchanged interface."""
    return tuple(
        (
            symbol,
            tuple(int(index) for index in interface.sensory_populations[symbol]),
            tuple(int(index) for index in interface.output_populations[symbol]),
        )
        for symbol in ("A", "B", "C", "D")
    )


__all__ = [
    "allocation_fingerprint",
    "burst_outlier_symbols",
    "classify_output_observability",
    "count_candidate_dynamic_pool",
]
