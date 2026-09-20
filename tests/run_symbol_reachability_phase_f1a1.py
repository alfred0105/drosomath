"""Real MaleCNS structural reachability and fixed-window F.1A.1 diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
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
    audit_symbol_reachability,
    load_malecns_v1,
)
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


DURATIONS_MS = (10.0, 20.0, 40.0)


def _persistent_snapshot(brain) -> dict[str, object]:
    plasticity = brain.plasticity
    return {
        "multiplier": plasticity.multiplier.copy(),
        "usage_ema": plasticity.usage_ema.copy(),
        "eligibility": plasticity.eligibility.copy(),
        "stability": plasticity.stability.copy(),
        "plastic_mask": plasticity.plastic_mask.copy(),
        "budget": plasticity.plastic_edge_count,
        "overrides": {key: value.copy() for key, value in plasticity.allocation_overrides().items()},
    }


def _persistent_equal(brain, before) -> bool:
    plasticity = brain.plasticity
    for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask"):
        if not np.array_equal(getattr(plasticity, name), before[name]):
            return False
    if plasticity.plastic_edge_count != before["budget"]:
        return False
    current = plasticity.allocation_overrides()
    return all(np.array_equal(current[key], values) for key, values in before["overrides"].items())


def _make_brain(connectome, *, seed: int, config: SymbolInterfaceConfig):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=seed,
        ),
    )


def _dynamic_condition(connectome, interface, config, duration_ms: float) -> tuple[dict[str, object], bool]:
    # A new brain with the same seed is created for each duration condition.
    brain = _make_brain(connectome, seed=config.seed, config=config)
    before = _persistent_snapshot(brain)
    session = SymbolSession(brain, interface)
    per_symbol: dict[str, object] = {}
    for symbol in SYMBOLS:
        result = session.present(
            symbol=symbol,
            duration_ms=duration_ms,
            stimulus_rate_hz=config.default_stimulus_rate_hz,
            learn=False,
        )
        per_symbol[symbol] = {
            "network_spikes": int(result.network_activity["total_spikes"]),
            "output_spikes": int(result.total_output_spikes),
            "first_output_spike_ms": result.first_output_spike_ms,
            "decision": result.decision,
            "output_population_rates_hz": result.output_population_rates_hz,
        }
    return {
        "duration_ms": duration_ms,
        "per_symbol": per_symbol,
        "persistent_learning_state_unchanged": _persistent_equal(brain, before),
    }, _persistent_equal(brain, before)


def _classify(reachability, dynamic_latency):
    rows = [
        row
        for input_rows in reachability.values()
        for row in input_rows.values()
    ]
    has_routes = any(row["hop3_output_neurons"] > 0 for row in rows)
    output_by_duration = {
        float(duration.removesuffix("ms")): any(
            values["output_spikes"] > 0
            for values in report["per_symbol"].values()
        )
        for duration, report in dynamic_latency.items()
    }
    any_output = any(output_by_duration.values())
    longer_latency = (
        not output_by_duration[10.0]
        and (output_by_duration[20.0] or output_by_duration[40.0])
    )
    structural_problem = not has_routes
    dynamically_silent = has_routes and not any_output
    if structural_problem:
        recommendation = "reachable_output_selection"
    elif longer_latency:
        recommendation = "cognition_integration_window"
    elif dynamically_silent:
        recommendation = "propagation_dynamics_investigation"
    else:
        recommendation = "proceed_to_f1b"
    return {
        "structural_reachability_problem": structural_problem,
        "dynamically_silent_despite_routes": dynamically_silent,
        "output_appears_with_longer_latency": longer_latency,
        "observable_output_by_duration": {f"{int(duration)}ms": value for duration, value in output_by_duration.items()},
        "recommended_next_step": recommendation,
    }


def run(*, data_dir: Path, output_path: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = SymbolInterfaceConfig(
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
    )

    allocation_brain = _make_brain(connectome, seed=config.seed, config=config)
    brain_rng_before = copy.deepcopy(allocation_brain.rng.bit_generator.state)
    interface = SymbolInterface(connectome, config)
    brain_rng_after = copy.deepcopy(allocation_brain.rng.bit_generator.state)

    reachability = audit_symbol_reachability(connectome, interface, max_hops=3)
    dynamic_latency: dict[str, object] = {}
    persistent_flags = []
    for duration_ms in DURATIONS_MS:
        report, unchanged = _dynamic_condition(connectome, interface, config, duration_ms)
        dynamic_latency[f"{int(duration_ms)}ms"] = report
        persistent_flags.append(unchanged)

    sensory = interface.sensory_populations
    output = interface.output_populations
    sensory_union = np.concatenate(tuple(sensory.values()))
    output_union = np.concatenate(tuple(output.values()))
    all_dynamic_rows = [
        row
        for report in dynamic_latency.values()
        for row in report["per_symbol"].values()
    ]
    sensory_ready = bool(
        len(np.unique(sensory_union)) == len(sensory_union)
        and all(row["network_spikes"] > 0 for row in all_dynamic_rows)
    )
    observed_output_by_symbol = {
        symbol: any(
            dynamic_latency[f"{int(duration)}ms"]["per_symbol"][symbol]["output_spikes"] > 0
            for duration in DURATIONS_MS
        )
        for symbol in SYMBOLS
    }
    decision_surface_ready = all(observed_output_by_symbol.values())
    diagnosis = _classify(reachability, dynamic_latency)
    result = {
        "protocol": {
            "phase": "F.1A.1",
            "seed": config.seed,
            "symbols": list(SYMBOLS),
            "learning_enabled": False,
            "stimulus_rate_hz": config.default_stimulus_rate_hz,
            "durations_ms": list(DURATIONS_MS),
            "allocation_preserved": True,
            "output_selection_changed": False,
            "connectome": "MaleCNS v1.0, min_connection_synapses=5",
        },
        "structural_reachability": reachability,
        "dynamic_latency": dynamic_latency,
        "readiness": {
            "sensory_interface_ready": sensory_ready,
            "decision_surface_ready": decision_surface_ready,
            "ready_for_f1b": bool(sensory_ready and decision_surface_ready),
            "observable_output_by_symbol": observed_output_by_symbol,
        },
        "invariants": {
            "brain_rng_unchanged_by_allocation": brain_rng_before == brain_rng_after,
            "persistent_learning_state_unchanged": all(persistent_flags),
            "sensory_output_overlap": int(len(np.intersect1d(sensory_union, output_union))),
            "sensory_population_size": config.sensory_population_size,
            "output_population_size": config.output_population_size,
        },
        "conclusion": diagnosis,
    }
    # pass means the audit completed and preserved invariants.  It is not a
    # claim that F.1B is ready; readiness is reported separately above.
    result["pass"] = bool(
        result["invariants"]["brain_rng_unchanged_by_allocation"]
        and result["invariants"]["persistent_learning_state_unchanged"]
        and result["invariants"]["sensory_output_overlap"] == 0
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
        default=Path("results/latest_symbol_reachability_phase_f1a1.json"),
    )
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
