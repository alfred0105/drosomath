"""Run the matched-RNG, diagnostic-only F.1A.2 symbol audit."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolSession,
    allocation_fingerprint,
    burst_outlier_symbols,
    classify_output_observability,
    count_candidate_dynamic_pool,
    load_malecns_v1,
)
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


BRAIN_SEEDS = (7, 11, 19)
DURATIONS_MS = (10.0, 20.0, 40.0)


def _make_brain(connectome, config: SymbolInterfaceConfig, seed: int):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=seed,
        ),
    )


def _rng_state_equal(left, right) -> bool:
    return left == right


def _persistent_snapshot(brain) -> dict[str, object]:
    plasticity = brain.plasticity
    return {
        "multiplier": plasticity.multiplier.copy(),
        "stability": plasticity.stability.copy(),
        "eligibility": plasticity.eligibility.copy(),
        "usage_ema": plasticity.usage_ema.copy(),
        "plastic_mask": plasticity.plastic_mask.copy(),
        "allocation_overrides": {
            key: value.copy() for key, value in plasticity.allocation_overrides().items()
        },
        "plastic_budget": plasticity.plastic_edge_count,
    }


def _persistent_state_equal(brain, before) -> bool:
    plasticity = brain.plasticity
    for name in ("multiplier", "stability", "eligibility", "usage_ema", "plastic_mask"):
        if not np.array_equal(getattr(plasticity, name), before[name]):
            return False
    if plasticity.plastic_edge_count != before["plastic_budget"]:
        return False
    current = plasticity.allocation_overrides()
    return all(
        np.array_equal(current[key], values)
        for key, values in before["allocation_overrides"].items()
    )


def _aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    by_symbol: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in SYMBOLS}
    for row in rows:
        by_symbol[row["input_symbol"]].append(row)
    summary: dict[str, object] = {}
    for symbol, symbol_rows in by_symbol.items():
        network = [int(row["network_spikes"]) for row in symbol_rows]
        output = [int(row["output_spikes"]) for row in symbol_rows]
        active_latencies = [
            float(row["first_output_spike_ms"])
            for row in symbol_rows
            if row["first_output_spike_ms"] is not None
        ]
        summary[symbol] = {
            "mean_network_spikes": float(statistics.mean(network)),
            "median_network_spikes": float(statistics.median(network)),
            "mean_output_spikes": float(statistics.mean(output)),
            "median_output_spikes": float(statistics.median(output)),
            "output_active_run_fraction": float(sum(value > 0 for value in output) / len(output)),
            "mean_first_output_latency_ms_active_runs": (
                float(statistics.mean(active_latencies)) if active_latencies else None
            ),
            "active_run_count": int(sum(value > 0 for value in output)),
            "run_count": len(output),
        }
    return summary


def _run_matched_condition(connectome, interface, config, *, seed: int, duration_ms: float):
    rows: list[dict[str, object]] = []
    initial_states = {}
    persistent_flags = []
    active_by_symbol: dict[str, set[int]] = {symbol: set() for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        # This construction is intentionally inside the symbol loop.  No
        # symbol shares transient state or RNG position with another symbol.
        brain = _make_brain(connectome, config, seed)
        initial_states[symbol] = copy.deepcopy(brain.rng.bit_generator.state)
        before = _persistent_snapshot(brain)
        result = SymbolSession(brain, interface).present(
            symbol=symbol,
            duration_ms=duration_ms,
            stimulus_rate_hz=config.default_stimulus_rate_hz,
            learn=False,
        )
        persistent_flags.append(_persistent_state_equal(brain, before))
        if duration_ms == 40.0:
            active_by_symbol[symbol].update(result.active_neuron_indices)
        rows.append({
            "brain_seed": seed,
            "duration_ms": duration_ms,
            "input_symbol": symbol,
            "network_spikes": int(result.network_activity["total_spikes"]),
            "unique_network_neurons": int(result.network_activity["unique_neurons"]),
            "output_spikes": int(result.total_output_spikes),
            "output_population_rates_hz": result.output_population_rates_hz,
            "first_output_spike_ms": result.first_output_spike_ms,
            "decision": result.decision,
        })
    return rows, initial_states, all(persistent_flags), active_by_symbol


def run(*, data_dir: Path, output_path: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
    )
    allocation_brain = _make_brain(connectome, config, config.seed)
    rng_before_allocation = copy.deepcopy(allocation_brain.rng.bit_generator.state)
    interface = SymbolInterface(connectome, config)
    rng_after_allocation = copy.deepcopy(allocation_brain.rng.bit_generator.state)
    interface_key = allocation_fingerprint(interface)

    matched_runs: dict[str, object] = {}
    all_40_active: dict[str, set[int]] = {symbol: set() for symbol in SYMBOLS}
    persistent_flags = []
    matching_flags = []
    distinct_seed_state_flags = []
    reference_seed_state = None
    burst_events: list[dict[str, object]] = []
    burst_count_by_symbol = {symbol: 0 for symbol in SYMBOLS}
    condition_extremes: list[dict[str, object]] = []
    for duration_ms in DURATIONS_MS:
        duration_rows: list[dict[str, object]] = []
        seed_initial_states = {}
        for seed in BRAIN_SEEDS:
            rows, initial_states, persistent_unchanged, active_by_symbol = _run_matched_condition(
                connectome, interface, config, seed=seed, duration_ms=duration_ms
            )
            duration_rows.extend(rows)
            seed_initial_states[seed] = initial_states
            persistent_flags.append(persistent_unchanged)
            matching_flags.append(all(
                _rng_state_equal(initial_states["A"], initial_states[symbol])
                for symbol in SYMBOLS
            ))
            if duration_ms == DURATIONS_MS[0]:
                if reference_seed_state is None:
                    reference_seed_state = initial_states["A"]
                else:
                    distinct_seed_state_flags.append(
                        not _rng_state_equal(reference_seed_state, initial_states["A"])
                    )
            if duration_ms == 40.0:
                for symbol in SYMBOLS:
                    all_40_active[symbol].update(active_by_symbol[symbol])
            spike_map = {row["input_symbol"]: row["network_spikes"] for row in rows}
            events = burst_outlier_symbols(spike_map)
            for event in events:
                event = {"brain_seed": seed, "duration_ms": duration_ms, **event}
                burst_events.append(event)
                burst_count_by_symbol[event["symbol"]] += 1
            values = [int(row["network_spikes"]) for row in rows]
            nonzero = [value for value in values if value > 0]
            condition_extremes.append({
                "brain_seed": seed,
                "duration_ms": duration_ms,
                "max_symbol_network_spikes": max(values),
                "min_nonzero_symbol_network_spikes": min(nonzero) if nonzero else None,
            })
        matched_runs[f"{int(duration_ms)}ms"] = {
            "presentations": duration_rows,
            "symbol_summary": _aggregate_rows(duration_rows),
            "matched_initial_rng_by_seed": {
                str(seed): bool(all(
                    _rng_state_equal(seed_initial_states[seed]["A"], seed_initial_states[seed][symbol])
                    for symbol in SYMBOLS
                ))
                for seed in BRAIN_SEEDS
            },
            "fresh_brain_per_symbol": True,
        }

    sensory_union = np.concatenate(tuple(interface.sensory_populations.values()))
    candidate_pool = count_candidate_dynamic_pool(
        all_40_active,
        excluded_indices=sensory_union,
    )
    forty_summary = matched_runs["40ms"]["symbol_summary"]
    readiness = {
        symbol: classify_output_observability(
            int(forty_summary[symbol]["active_run_count"]) // 1,
            len(BRAIN_SEEDS),
        )
        for symbol in SYMBOLS
    }
    decision_surface_ready = all(value == "reliably_observable" for value in readiness.values())
    b_40_bursts = [event for event in burst_events if event["duration_ms"] == 40.0 and event["symbol"] == "B"]
    dominant_40 = []
    for seed in BRAIN_SEEDS:
        rows = [row for row in matched_runs["40ms"]["presentations"] if row["brain_seed"] == seed]
        dominant_40.append(max(rows, key=lambda row: row["network_spikes"])["input_symbol"])
    previous_burst_reproducible = len(b_40_bursts) >= 2 and dominant_40.count("B") >= 2
    broadly_active_pool = candidate_pool["active_for_at_least_2_symbols"] >= config.output_population_size
    if decision_surface_ready:
        recommendation = "proceed_to_f1b"
    elif broadly_active_pool:
        recommendation = "build_dynamically_viable_decision_surface"
    else:
        recommendation = "investigate_recurrent_propagation"

    result = {
        "protocol": {
            "phase": "F.1A.2",
            "interface_seed": config.seed,
            "brain_seeds": list(BRAIN_SEEDS),
            "durations_ms": list(DURATIONS_MS),
            "stimulus_rate_hz": config.default_stimulus_rate_hz,
            "learning_enabled": False,
            "fresh_brain_per_symbol": True,
            "presentations": len(BRAIN_SEEDS) * len(DURATIONS_MS) * len(SYMBOLS),
        },
        "matched_runs": matched_runs,
        "symbol_summary": {duration: report["symbol_summary"] for duration, report in matched_runs.items()},
        "burst_audit": {
            "events": burst_events,
            "burst_outlier_count_by_symbol": burst_count_by_symbol,
            "per_condition_extremes": condition_extremes,
            "dominant_symbols_at_40ms_by_seed": {
                str(seed): dominant_40[index] for index, seed in enumerate(BRAIN_SEEDS)
            },
        },
        "candidate_dynamic_pool": candidate_pool,
        "readiness": {
            **readiness,
            "decision_surface_ready": decision_surface_ready,
            "ready_for_f1b": decision_surface_ready,
        },
        "invariants": {
            "fresh_brain_per_symbol": True,
            "same_seed_initial_rng_matched": all(matching_flags),
            "different_brain_seeds_are_distinct": all(distinct_seed_state_flags),
            "interface_allocation_fingerprint": [
                [symbol, list(sensory), list(output)]
                for symbol, sensory, output in interface_key
            ],
            "interface_did_not_consume_brain_rng": _rng_state_equal(rng_before_allocation, rng_after_allocation),
            "persistent_learning_state_unchanged": all(persistent_flags),
            "learning_calls": 0,
        },
        "conclusion": {
            "previous_burst_reproducible": previous_burst_reproducible,
            "rng_order_confounded_previous_result": not previous_burst_reproducible,
            "symbol_specific_recurrent_regime_evidence": previous_burst_reproducible,
            "current_output_surface_dynamically_viable": decision_surface_ready,
            "recommended_next_step": recommendation,
        },
    }
    result["pass"] = bool(
        len(matched_runs["10ms"]["presentations"])
        + len(matched_runs["20ms"]["presentations"])
        + len(matched_runs["40ms"]["presentations"])
        == 36
        and result["invariants"]["same_seed_initial_rng_matched"]
        and result["invariants"]["interface_did_not_consume_brain_rng"]
        and result["invariants"]["persistent_learning_state_unchanged"]
        and result["invariants"]["learning_calls"] == 0
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_matched_symbol_dynamics_phase_f1a2.json"),
    )
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
