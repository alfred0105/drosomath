"""Read-only diagnostics for the Phase F.1B.1 extended curve."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from .symbol_interface import NO_DECISION, SYMBOLS
from .symbol_learning import SymbolTrialResult, balanced_symbol_schedule, summarize_results


EXTENDED_CHECKPOINTS = (0, 400, 800, 1200, 1600)


def extended_symbol_schedule(*, seed: int) -> tuple[str, ...]:
    """Return the fixed 1600-trial balanced schedule used by F.1B.1."""
    return balanced_symbol_schedule(cycles=400, seed=seed + 10_000)


def target_rank(output_rates_hz: Mapping[str, float], target: str) -> float | None:
    """Return average competition rank, or ``None`` for a silent output."""
    if target not in SYMBOLS:
        raise KeyError(f"unknown target {target!r}")
    values = {symbol: float(output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS}
    if sum(values.values()) <= 0.0:
        return None
    target_value = values[target]
    greater = sum(value > target_value for symbol, value in values.items() if symbol != target)
    tied = 1 + sum(value == target_value for symbol, value in values.items() if symbol != target)
    return float(1 + greater + (tied - 1) / 2.0)


def _rank_summary(results: Sequence[SymbolTrialResult], symbol: str) -> dict[str, object]:
    selected = [result for result in results if result.target == symbol]
    ranks = [target_rank(result.output_rates_hz, symbol) for result in selected]
    non_silent = [rank for rank in ranks if rank is not None]
    margins = [
        float(result.output_rates_hz.get(symbol, 0.0))
        - max((value for other, value in result.output_rates_hz.items() if other != symbol), default=0.0)
        for result in selected
    ]
    return {
        "mean_target_rank": float(np.mean(non_silent)) if non_silent else None,
        "silent_count": int(sum(rank is None for rank in ranks)),
        "silent_fraction": float(sum(rank is None for rank in ranks) / len(ranks)) if ranks else 0.0,
        "fraction_target_rank_1": float(sum(rank == 1.0 for rank in non_silent) / len(non_silent)) if non_silent else 0.0,
        "fraction_target_rank_2_or_better": float(sum(rank <= 2.0 for rank in non_silent) / len(non_silent)) if non_silent else 0.0,
        "fraction_positive_margin": float(sum(margin > 0.0 for margin in margins) / len(margins)) if margins else 0.0,
    }


def extended_checkpoint_metrics(results: Sequence[SymbolTrialResult]) -> dict[str, object]:
    """Combine existing F.1B metrics with rank/boundary diagnostics."""
    summary = summarize_results(results)
    per_symbol = dict(summary["per_symbol"])
    for symbol in SYMBOLS:
        per_symbol[symbol] = {
            **per_symbol[symbol],
            **_rank_summary(results, symbol),
        }
    all_ranks = [
        target_rank(result.output_rates_hz, result.target)
        for result in results
    ]
    non_silent = [rank for rank in all_ranks if rank is not None]
    all_margins = [
        result.output_rates_hz.get(result.target, 0.0)
        - max((value for other, value in result.output_rates_hz.items() if other != result.target), default=0.0)
        for result in results
    ]
    summary["per_symbol"] = per_symbol
    summary["target_rank"] = {
        "mean": float(np.mean(non_silent)) if non_silent else None,
        "silent_count": int(sum(rank is None for rank in all_ranks)),
        "silent_fraction": float(sum(rank is None for rank in all_ranks) / len(all_ranks)) if all_ranks else 0.0,
        "fraction_target_rank_1": float(sum(rank == 1.0 for rank in non_silent) / len(non_silent)) if non_silent else 0.0,
        "fraction_target_rank_2_or_better": float(sum(rank <= 2.0 for rank in non_silent) / len(non_silent)) if non_silent else 0.0,
        "fraction_positive_margin": float(sum(margin > 0.0 for margin in all_margins) / len(all_margins)) if all_margins else 0.0,
    }
    return summary


def classify_margin_interval(delta_hz: float) -> str:
    """Descriptive, predeclared margin movement classification."""
    if delta_hz > 10.0:
        return "still_improving"
    if delta_hz < -10.0:
        return "regressing"
    return "plateau"


def margin_intervals(checkpoints: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    ordered = [str(value) for value in EXTENDED_CHECKPOINTS]
    result: dict[str, dict[str, object]] = {}
    for left, right in zip(ordered, ordered[1:]):
        left_metrics = checkpoints[left]
        right_metrics = checkpoints[right]
        per_symbol: dict[str, object] = {}
        for symbol in SYMBOLS:
            left_margin = float(left_metrics["per_symbol"][symbol]["target_minus_best_competitor_margin"])
            right_margin = float(right_metrics["per_symbol"][symbol]["target_minus_best_competitor_margin"])
            delta = right_margin - left_margin
            per_symbol[symbol] = {
                "delta_hz": float(delta),
                "classification": classify_margin_interval(delta),
            }
        left_margin = float(left_metrics["target_minus_best_competitor_margin"])
        right_margin = float(right_metrics["target_minus_best_competitor_margin"])
        delta = right_margin - left_margin
        result[f"{left}->{right}"] = {
            "delta_hz": float(delta),
            "classification": classify_margin_interval(delta),
            "per_symbol": per_symbol,
        }
    return result


def boundary_crossings(checkpoints: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, str | None]]:
    result = {}
    for symbol in SYMBOLS:
        margin_checkpoint = None
        accuracy_checkpoint = None
        for checkpoint in EXTENDED_CHECKPOINTS:
            row = checkpoints[str(checkpoint)]["per_symbol"][symbol]
            if margin_checkpoint is None and float(row["target_minus_best_competitor_margin"]) > 0.0:
                margin_checkpoint = str(checkpoint)
            if accuracy_checkpoint is None and float(row["accuracy"]) >= 0.5:
                accuracy_checkpoint = str(checkpoint)
        result[symbol] = {
            "first_mean_positive_margin_checkpoint": margin_checkpoint,
            "first_accuracy_at_least_0_5_checkpoint": accuracy_checkpoint,
        }
    return result


def channel_credit_telemetry(
    results: Sequence[SymbolTrialResult],
    *,
    unique_edge_counts: Mapping[str, int] | None = None,
) -> dict[str, dict[str, object]]:
    """Aggregate generic-channel directional telemetry without changing learning."""
    output = {
        channel: {
            "positive_direction_requests": 0,
            "negative_direction_requests": 0,
            "edge_updates": 0,
            "sum_abs_multiplier_delta": 0.0,
            "1hop_updates": 0,
            "2hop_updates": 0,
            "unique_edges_modified": set(),
        }
        for channel in (f"symbol/{symbol}" for symbol in SYMBOLS)
    }
    for result in results:
        update = result.directional_update
        for channel, direction in result.directional_error.items():
            if channel not in output:
                continue
            if direction > 0.0:
                output[channel]["positive_direction_requests"] += 1
            elif direction < 0.0:
                output[channel]["negative_direction_requests"] += 1
        for channel, count in update.get("channel_updates", {}).items():
            if channel in output:
                output[channel]["edge_updates"] += int(count)
        for channel, value in update.get("channel_sum_abs_delta", {}).items():
            if channel in output:
                output[channel]["sum_abs_multiplier_delta"] += float(value)
        for channel, counts in update.get("channel_hop_counts", {}).items():
            if channel in output:
                output[channel]["1hop_updates"] += int(counts.get(1, 0))
                output[channel]["2hop_updates"] += int(counts.get(2, 0))
        for channel, count in update.get("channel_unique_edge_updates", {}).items():
            if channel in output:
                output[channel]["unique_edges_modified"] .add(int(count))
    for channel, value in output.items():
        if unique_edge_counts is not None:
            # The session maintains exact cumulative edge-ID sets without
            # retaining those potentially large sets in every trial record.
            value["unique_edges_modified"] = int(unique_edge_counts.get(channel, 0))
        else:
            value["unique_edges_modified"] = int(sum(value["unique_edges_modified"]))
    return output


def cumulative_plasticity_telemetry(session, *, trial_count: int) -> dict[str, object]:
    """Read cumulative update telemetry and current persistent plastic state."""
    state = session.brain.plasticity
    directional_edges: set[int] = set()
    hop_counts: Counter[str] = Counter()
    excitatory = inhibitory = reinforcement_trials = consolidated = 0
    edge_updates = 0
    for result in session.trial_results[:trial_count]:
        update = result.directional_update
        edge_updates += int(update.get("edge_updates", 0))
        directional_edges.update(int(edge) for edge in update.get("updated_edge_indices", ()))
        hop_counts.update({str(key): int(value) for key, value in update.get("hop_counts", {}).items()})
        excitatory += int(update.get("excitatory_updates", 0))
        inhibitory += int(update.get("inhibitory_updates", 0))
        reinforcement_trials += int(bool(result.reinforcement))
        consolidated += int(update.get("consolidated_edges", 0))
    saturation = (state.multiplier <= state.config.min_multiplier) | (
        state.multiplier >= state.config.max_multiplier
    )
    return {
        "trial_count": int(trial_count),
        "mean_multiplier": float(np.mean(state.multiplier)),
        "multiplier_saturation_fraction": float(np.mean(saturation)),
        "mean_stability": float(np.mean(state.stability)),
        "plastic_budget": int(state.plastic_edge_count),
        "budget_drift": int(state.plastic_edge_count - session.plastic_budget_start),
        "unique_directional_edges_modified_cumulative": int(len(directional_edges)),
        "directional_edge_updates_cumulative": int(edge_updates),
        "1hop_updates_cumulative": int(hop_counts.get("1", 0)),
        "2hop_updates_cumulative": int(hop_counts.get("2", 0)),
        "excitatory_updates_cumulative": int(excitatory),
        "inhibitory_updates_cumulative": int(inhibitory),
        "positive_reinforcement_trials_cumulative": int(reinforcement_trials),
        "consolidated_edges_cumulative": int(consolidated),
        "adaptive_plastic_budget": False,
    }


__all__ = [
    "EXTENDED_CHECKPOINTS",
    "boundary_crossings",
    "channel_credit_telemetry",
    "classify_margin_interval",
    "cumulative_plasticity_telemetry",
    "extended_checkpoint_metrics",
    "extended_symbol_schedule",
    "margin_intervals",
    "target_rank",
]
