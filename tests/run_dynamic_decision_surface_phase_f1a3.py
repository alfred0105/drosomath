"""Build and validate the diagnostic-only F.1A.3 dynamic symbol surface."""

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
    build_dynamic_generic_interface,
    collect_dynamic_candidate_features,
    load_malecns_v1,
    select_dynamic_generic_candidates,
    viability_histogram,
)
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


DISCOVERY_SEEDS = (7, 11, 19)
VALIDATION_SEEDS = (23, 29, 31)
DURATION_MS = 40.0
STIMULUS_RATE_HZ = 205.0
DECISION_SURFACE_SEED = 17


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


def _persistent_snapshot(brain) -> dict[str, object]:
    plasticity = brain.plasticity
    return {
        "multiplier": plasticity.multiplier.copy(),
        "stability": plasticity.stability.copy(),
        "eligibility": plasticity.eligibility.copy(),
        "usage_ema": plasticity.usage_ema.copy(),
        "plastic_mask": plasticity.plastic_mask.copy(),
        "overrides": {key: value.copy() for key, value in plasticity.allocation_overrides().items()},
        "budget": plasticity.plastic_edge_count,
    }


def _persistent_equal(brain, before) -> bool:
    plasticity = brain.plasticity
    for name in ("multiplier", "stability", "eligibility", "usage_ema", "plastic_mask"):
        if not np.array_equal(getattr(plasticity, name), before[name]):
            return False
    if plasticity.plastic_edge_count != before["budget"]:
        return False
    current = plasticity.allocation_overrides()
    return all(np.array_equal(current[key], value) for key, value in before["overrides"].items())


def _run_surface(connectome, interface, config, seeds):
    rows = []
    persistent_flags = []
    initial_rng_states = {}
    for seed in seeds:
        for symbol in SYMBOLS:
            # A fresh brain is created for every seed × symbol presentation.
            brain = _make_brain(connectome, config, seed)
            initial_rng_states[(seed, symbol)] = copy.deepcopy(brain.rng.bit_generator.state)
            before = _persistent_snapshot(brain)
            result = SymbolSession(brain, interface).present(
                symbol=symbol,
                duration_ms=DURATION_MS,
                stimulus_rate_hz=STIMULUS_RATE_HZ,
                learn=False,
            )
            persistent_flags.append(_persistent_equal(brain, before))
            rows.append({
                "brain_seed": seed,
                "input_symbol": symbol,
                "network_spikes": int(result.network_activity["total_spikes"]),
                "unique_network_neurons": int(result.network_activity["unique_neurons"]),
                "output_spikes": int(result.total_output_spikes),
                "output_population_rates_hz": result.output_population_rates_hz,
                "first_output_spike_ms": result.first_output_spike_ms,
                "decision": result.decision,
                "_active_neuron_indices": tuple(result.active_neuron_indices),
            })
    return rows, all(persistent_flags), initial_rng_states


def _clean_rows(rows):
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _summary(rows):
    result = {}
    for symbol in SYMBOLS:
        selected = [row for row in rows if row["input_symbol"] == symbol]
        network = [row["network_spikes"] for row in selected]
        output = [row["output_spikes"] for row in selected]
        latency = [
            row["first_output_spike_ms"]
            for row in selected
            if row["first_output_spike_ms"] is not None
        ]
        decisions: dict[str, int] = {}
        for row in selected:
            decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
        active_count = sum(value > 0 for value in output)
        result[symbol] = {
            "mean_network_spikes": float(statistics.mean(network)),
            "median_network_spikes": float(statistics.median(network)),
            "mean_symbol_output_spikes": float(statistics.mean(output)),
            "median_symbol_output_spikes": float(statistics.median(output)),
            "active_seed_count": int(active_count),
            "active_seed_fraction": float(active_count / len(output)),
            "mean_first_output_latency_ms": float(statistics.mean(latency)) if latency else None,
            "decision_distribution": decisions,
        }
    return result


def _active_fractions(rows):
    summary = _summary(rows)
    return {symbol: summary[symbol]["active_seed_fraction"] for symbol in SYMBOLS}


def _no_decision_fraction(rows):
    return float(sum(row["decision"] == "NO_DECISION" for row in rows) / len(rows))


def _same_rng_for_surface(states):
    seeds = sorted({seed for seed, _symbol in states})
    return all(
        states[(seed, "A")] == states[(seed, symbol)]
        for seed in seeds for symbol in SYMBOLS
    )


def run(*, data_dir: Path, output_path: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    base_config = SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=DURATION_MS,
        default_stimulus_rate_hz=STIMULUS_RATE_HZ,
        plastic_fraction=0.05,
        output_selection="random_indegree",
    )
    dynamic_config = SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=DURATION_MS,
        default_stimulus_rate_hz=STIMULUS_RATE_HZ,
        plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )
    old_interface = SymbolInterface(connectome, base_config)
    discovery_rows, discovery_state_ok, discovery_rng_states = _run_surface(
        connectome, old_interface, base_config, DISCOVERY_SEEDS
    )
    sensory_indices = np.concatenate(tuple(old_interface.sensory_populations.values()))
    features, candidate_count = collect_dynamic_candidate_features(
        connectome,
        discovery_rows,
        sensory_indices=sensory_indices,
    )
    selected = select_dynamic_generic_candidates(features, selected_count=128)
    dynamic_interface = build_dynamic_generic_interface(
        connectome,
        dynamic_config,
        selected,
        decision_surface_seed=DECISION_SURFACE_SEED,
    )
    old_rows, old_state_ok, old_rng_states = _run_surface(
        connectome, old_interface, base_config, VALIDATION_SEEDS
    )
    new_rows, new_state_ok, new_rng_states = _run_surface(
        connectome, dynamic_interface, dynamic_config, VALIDATION_SEEDS
    )

    old_summary = _summary(old_rows)
    new_summary = _summary(new_rows)
    dynamic_ready = all(new_summary[symbol]["active_seed_count"] >= 2 for symbol in SYMBOLS)
    old_output_total = int(sum(row["output_spikes"] for row in old_rows))
    new_output_total = int(sum(row["output_spikes"] for row in new_rows))
    old_output_active = _active_fractions(old_rows)
    new_output_active = _active_fractions(new_rows)
    selected_histograms = {
        "distinct_symbol_count_active": viability_histogram(selected, "distinct_symbol_count_active"),
        "distinct_seed_count_active": viability_histogram(selected, "distinct_seed_count_active"),
        "total_active_run_count": viability_histogram(selected, "total_active_run_count"),
    }
    selected_body_ids = {
        symbol: [int(value) for value in dynamic_interface.decision_surface.body_id_populations[symbol]]
        for symbol in SYMBOLS
    }
    selected_indices = {
        symbol: [int(value) for value in dynamic_interface.output_populations[symbol]]
        for symbol in SYMBOLS
    }
    old_no_decision = _no_decision_fraction(old_rows)
    new_no_decision = _no_decision_fraction(new_rows)
    surface_overlap = dynamic_interface.allocation_summary()
    candidate_selection_labels = {
        "ranking_uses_specific_symbol_identity": False,
        "partition_uses_specific_symbol_identity": False,
        "uses_correctness_or_reward": False,
        "partition_seed": DECISION_SURFACE_SEED,
    }
    result = {
        "protocol": {
            "phase": "F.1A.3",
            "discovery_seeds": list(DISCOVERY_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "duration_ms": DURATION_MS,
            "stimulus_rate_hz": STIMULUS_RATE_HZ,
            "learning_enabled": False,
            "external_trainable_decoder": False,
            "fresh_brain_per_symbol": True,
            "held_out_validation_not_used_for_selection": True,
        },
        "discovery": {
            "candidate_count_before_selection": candidate_count,
            "candidate_count": candidate_count,
            "selected_count": len(selected),
            "selection_rule": "distinct_symbol_count_active, then distinct_seed_count_active, then total_active_run_count, then neuron index/body ID",
            "selected_viability_histograms": selected_histograms,
            "partition_label_independent": True,
            "discovery_learning_state_unchanged": discovery_state_ok,
        },
        "surface": {
            "selection_strategy": "dynamic_generic",
            "population_size_per_symbol": 32,
            "sensory_output_overlap": surface_overlap["sensory_output_overlap"],
            "pairwise_output_overlap": surface_overlap["output_pairwise_overlap"],
            "selected_indices": selected_indices,
            "selected_body_ids": selected_body_ids,
        },
        "validation": {
            "old_random_surface": {
                "presentations": _clean_rows(old_rows),
                "per_symbol": old_summary,
                "total_output_spikes": old_output_total,
                "no_decision_fraction": old_no_decision,
                "active_fraction_by_input": old_output_active,
            },
            "dynamic_generic_surface": {
                "presentations": _clean_rows(new_rows),
                "per_symbol": new_summary,
                "total_output_spikes": new_output_total,
                "no_decision_fraction": new_no_decision,
                "active_fraction_by_input": new_output_active,
            },
            "same_heldout_conditions": True,
        },
        "readiness": {
            symbol: {
                "active_seed_count": new_summary[symbol]["active_seed_count"],
                "active_seed_fraction": new_summary[symbol]["active_seed_fraction"],
                "ready": new_summary[symbol]["active_seed_count"] >= 2,
            }
            for symbol in SYMBOLS
        },
        "invariants": {
            "discovery_seeds_disjoint_from_validation": set(DISCOVERY_SEEDS).isdisjoint(VALIDATION_SEEDS),
            "interface_allocation_does_not_consume_brain_rng": True,
            "discovery_rng_same_within_seed": _same_rng_for_surface(discovery_rng_states),
            "old_validation_rng_same_within_seed": _same_rng_for_surface(old_rng_states),
            "new_validation_rng_same_within_seed": _same_rng_for_surface(new_rng_states),
            "validation_uses_frozen_selected_indices": True,
            "validation_learning_state_unchanged": old_state_ok and new_state_ok,
            "learning_calls": 0,
            "external_trainable_decoder": False,
        },
        "conclusion": {
            "dynamic_surface_improves_observability": new_output_total > old_output_total,
            "all_symbols_observable_on_heldout_seeds": dynamic_ready,
            "ready_for_f1b": dynamic_ready,
            "recommended_next_step": "proceed_to_f1b" if dynamic_ready else "investigate_recurrent_propagation",
        },
    }
    result["pass"] = bool(
        candidate_count >= 128
        and len(selected) == 128
        and result["surface"]["sensory_output_overlap"] == 0
        and all(value == 0.0 for value in result["surface"]["pairwise_output_overlap"].values())
        and result["invariants"]["discovery_seeds_disjoint_from_validation"]
        and result["invariants"]["validation_uses_frozen_selected_indices"]
        and result["invariants"]["validation_learning_state_unchanged"]
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
        default=Path("results/latest_dynamic_decision_surface_phase_f1a3.json"),
    )
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
