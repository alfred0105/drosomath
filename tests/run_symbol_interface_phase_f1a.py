"""Run the real MaleCNS F.1A symbol-interface diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolSession,
    load_malecns_v1,
)
from drosomath.whole_brain import PlasticStateConfig


def _same_rng_state(left, right) -> bool:
    return left == right


def _persistent_snapshot(brain) -> dict[str, object]:
    plasticity = brain.plasticity
    return {
        "multiplier": plasticity.multiplier.copy(),
        "usage_ema": plasticity.usage_ema.copy(),
        "eligibility": plasticity.eligibility.copy(),
        "stability": plasticity.stability.copy(),
        "plastic_mask": plasticity.plastic_mask.copy(),
        "budget": plasticity.plastic_edge_count,
        "overrides": {
            key: value.copy() for key, value in plasticity.allocation_overrides().items()
        },
    }


def _persistent_equal(brain, before) -> bool:
    plasticity = brain.plasticity
    for name, expected in (
        ("multiplier", before["multiplier"]),
        ("usage_ema", before["usage_ema"]),
        ("eligibility", before["eligibility"]),
        ("stability", before["stability"]),
        ("plastic_mask", before["plastic_mask"]),
    ):
        if not np.array_equal(getattr(plasticity, name), expected):
            return False
    if plasticity.plastic_edge_count != before["budget"]:
        return False
    current = plasticity.allocation_overrides()
    return all(np.array_equal(current[key], values) for key, values in before["overrides"].items())


def _jaccard_zero(summary: dict[str, float]) -> bool:
    return all(value == 0.0 for value in summary.values())


def run(*, data_dir: Path, output_path: Path, repeats: int = 3) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = SymbolInterfaceConfig(
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=10.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
    )
    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=config.seed,
        ),
    )
    brain_rng_before = copy.deepcopy(brain.rng.bit_generator.state)
    interface = SymbolInterface(connectome, config)
    brain_rng_after = copy.deepcopy(brain.rng.bit_generator.state)
    session = SymbolSession(brain, interface)
    persistent_before = _persistent_snapshot(brain)

    observations: dict[str, list[object]] = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for _ in range(repeats):
            observations[symbol].append(session.present(symbol=symbol, learn=False))

    per_symbol: dict[str, object] = {}
    for symbol, rows in observations.items():
        network_spikes = [row.network_activity["total_spikes"] for row in rows]
        output_spikes = [row.total_output_spikes for row in rows]
        decision_counts: dict[str, int] = {}
        for row in rows:
            decision_counts[row.decision] = decision_counts.get(row.decision, 0) + 1
        per_symbol[symbol] = {
            "sensory_population_size": config.sensory_population_size,
            "output_population_size": config.output_population_size,
            "mean_network_spikes": float(np.mean(network_spikes)),
            "mean_output_spikes": float(np.mean(output_spikes)),
            "no_decision_count": int(sum(row.decision == "NO_DECISION" for row in rows)),
            "decision_counts": decision_counts,
        }

    allocation = interface.allocation_summary()
    sensory = interface.sensory_populations
    output = interface.output_populations
    all_indices = np.concatenate(tuple(sensory.values()) + tuple(output.values()))
    sensory_union = np.concatenate(tuple(sensory.values()))
    output_union = np.concatenate(tuple(output.values()))
    real_neurons_only = bool(all_indices.min() >= 0 and all_indices.max() < connectome.neuron_count)
    persistent_unchanged = _persistent_equal(brain, persistent_before)
    network_activity_present = all(
        per_symbol[symbol]["mean_network_spikes"] > 0.0 for symbol in SYMBOLS
    )
    symbol_input_ready = bool(
        real_neurons_only
        and len(np.unique(sensory_union)) == len(sensory_union)
        and network_activity_present
    )
    decision_surface_ready = all(
        per_symbol[symbol]["mean_output_spikes"] > 0.0 for symbol in SYMBOLS
    )
    result = {
        "protocol": {
            "phase": "F.1A",
            "seed": config.seed,
            "symbols": list(SYMBOLS),
            "learning_enabled": False,
            "external_trainable_decoder": False,
            "repeats_per_symbol": repeats,
            "duration_ms": config.default_duration_ms,
            "stimulus_rate_hz": config.default_stimulus_rate_hz,
            "connectome": "MaleCNS v1.0, min_connection_synapses=5",
        },
        "allocation": {
            "sensory_population_size": config.sensory_population_size,
            "output_population_size": config.output_population_size,
            "sensory_output_overlap": int(len(np.intersect1d(sensory_union, output_union))),
            "sensory_pairwise_overlap": allocation["sensory_pairwise_overlap"],
            "output_pairwise_overlap": allocation["output_pairwise_overlap"],
            "selection_basis": "real graph viability: nonzero outgoing sensory and nonzero indegree output",
        },
        "per_symbol": per_symbol,
        "readiness": {
            "symbol_input_interface_ready": symbol_input_ready,
            "symbol_decision_surface_ready": decision_surface_ready,
            "ready_for_supervised_symbol_learning_phase_f1b": bool(
                symbol_input_ready and decision_surface_ready
            ),
        },
        "invariants": {
            "real_neurons_only": real_neurons_only,
            "disjoint_output_populations": len(np.unique(output_union)) == len(output_union),
            "disjoint_sensory_populations": len(np.unique(sensory_union)) == len(sensory_union),
            "disjoint_sensory_output": not np.intersect1d(sensory_union, output_union).size,
            "brain_rng_unchanged_by_allocation": _same_rng_state(brain_rng_before, brain_rng_after),
            "persistent_learning_state_unchanged": persistent_unchanged,
        },
        "conclusion": {
            "symbol_input_interface_ready": symbol_input_ready,
            "symbol_decision_surface_ready": decision_surface_ready,
            "ready_for_supervised_symbol_learning_phase_f1b": bool(
                symbol_input_ready and decision_surface_ready
            ),
        },
    }
    result["pass"] = bool(
        result["invariants"]["real_neurons_only"]
        and result["invariants"]["disjoint_output_populations"]
        and result["invariants"]["disjoint_sensory_populations"]
        and result["invariants"]["disjoint_sensory_output"]
        and result["invariants"]["brain_rng_unchanged_by_allocation"]
        and result["invariants"]["persistent_learning_state_unchanged"]
        and _jaccard_zero(result["allocation"]["sensory_pairwise_overlap"])
        and _jaccard_zero(result["allocation"]["output_pairwise_overlap"])
        and network_activity_present
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
        default=Path("results/latest_symbol_interface_phase_f1a.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output, repeats=args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
