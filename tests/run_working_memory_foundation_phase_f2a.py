"""Run the learning-disabled delayed-cue working-memory foundation (F.2A)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.malecns import (  # noqa: E402
    DelayedCueSession,
    GO_ALLOCATION_SEED,
    GO_SYMBOL,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    balanced_symbol_schedule,
    load_malecns_v1,
    pairwise_set_jaccard,
)
from drosomath.malecns.symbol_learning_extended import target_rank  # noqa: E402
from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import (  # noqa: E402
    run_benchmark,
    run_route_cache_equivalence,
)


SEEDS = (71, 73, 79)
TRIALS_PER_CUE = 10
GO_RATE_HZ = 205.0
DATA_DIR = Path("data/malecns_v1")
ARTIFACT = Path("results/latest_working_memory_foundation_phase_f2a.json")


def _config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=GO_RATE_HZ,
        plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


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


def _array_digest(value) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _persistent_digest(brain) -> dict[str, str]:
    state = brain.plasticity
    return {
        "multiplier": _array_digest(state.multiplier),
        "usage_ema": _array_digest(state.usage_ema),
        "eligibility": _array_digest(state.eligibility),
        "stability": _array_digest(state.stability),
        "plastic_mask": _array_digest(state.plastic_mask),
        "promoted_edges": _array_digest(state.allocation_overrides()["promoted_edges"]),
        "retired_edges": _array_digest(state.allocation_overrides()["retired_edges"]),
    }


def _trial_summary(rows):
    per_cue = {}
    for cue in SYMBOLS:
        selected = [row for row in rows if row.cue == cue]
        correct = sum(row.decision == cue for row in selected)
        no_decision = sum(row.decision == "NO_DECISION" for row in selected)
        ranks = [target_rank(row.go_output_rates_hz, cue) for row in selected]
        ranks = [value for value in ranks if value is not None]
        margins = [
            float(row.go_output_rates_hz.get(cue, 0.0))
            - max((value for other, value in row.go_output_rates_hz.items() if other != cue), default=0.0)
            for row in selected
        ]
        per_cue[cue] = {
            "trials": len(selected),
            "accuracy": float(correct / len(selected)) if selected else 0.0,
            "no_decision_fraction": float(no_decision / len(selected)) if selected else 0.0,
            "mean_target_rank": float(np.mean(ranks)) if ranks else None,
            "mean_target_margin_hz": float(np.mean(margins)) if margins else 0.0,
            "mean_go_output_rates_hz": {
                symbol: float(np.mean([row.go_output_rates_hz[symbol] for row in selected]))
                if selected else 0.0
                for symbol in SYMBOLS
            },
            "mean_pre_go_active_count": float(np.mean([row.pre_go_active_count for row in selected]))
            if selected else 0.0,
        }
    all_ranks = [target_rank(row.go_output_rates_hz, row.cue) for row in rows]
    all_ranks = [value for value in all_ranks if value is not None]
    return {
        "trials": len(rows),
        "accuracy": float(sum(row.decision == row.cue for row in rows) / len(rows)) if rows else 0.0,
        "no_decision_fraction": float(sum(row.decision == "NO_DECISION" for row in rows) / len(rows)) if rows else 0.0,
        "mean_target_rank": float(np.mean(all_ranks)) if all_ranks else None,
        "per_cue": per_cue,
    }


def _output_separation(rows) -> float:
    means = []
    for cue in SYMBOLS:
        selected = [row for row in rows if row.cue == cue]
        if selected:
            means.append(np.asarray([
                np.mean([row.go_output_rates_hz[symbol] for row in selected])
                for symbol in SYMBOLS
            ], dtype=np.float64))
    distances = [float(np.linalg.norm(left - right, ord=1)) for index, left in enumerate(means) for right in means[index + 1:]]
    return float(np.mean(distances)) if distances else 0.0


def _run_arm(connectome, interface, wm_interface, config, *, reset_before_go: bool):
    arm = {}
    digest_before = {}
    digest_after = {}
    for seed in SEEDS:
        brain = _make_brain(connectome, config, seed)
        digest_before[str(seed)] = _persistent_digest(brain)
        session = DelayedCueSession(brain, wm_interface, learning_enabled=False)
        schedule = balanced_symbol_schedule(cycles=TRIALS_PER_CUE, seed=seed + 30_000)
        rows = [
            session.run_trial(symbol, reset_before_go=reset_before_go)
            for symbol in schedule
        ]
        digest_after[str(seed)] = _persistent_digest(brain)
        arm[str(seed)] = {
            "schedule": list(schedule),
            "rows": rows,
            "summary": _trial_summary(rows),
            "pairwise_pre_go_jaccard": pairwise_set_jaccard(rows),
            "output_separation_l1_hz": _output_separation(rows),
            "pre_go_active_count_mean": float(np.mean([row.pre_go_active_count for row in rows])),
            "pre_go_active_count_nonzero_fraction": float(
                np.mean([row.pre_go_active_count > 0 for row in rows])
            ),
            "post_reset_active_count_mean": float(np.mean([row.post_reset_active_count for row in rows])),
            "post_reset_empty_fraction": float(
                np.mean([row.post_reset_active_count == 0 for row in rows])
            ),
            "unique_pre_go_fingerprints": int(len({row.pre_go_fingerprint for row in rows})),
            "go_output_spikes_total": int(sum(row.go_output_spikes for row in rows)),
        }
    return arm, digest_before, digest_after


def _strip_rows(arm):
    return {
        seed: {
            key: value
            for key, value in payload.items()
            if key != "rows"
        }
        | {
            "rows": [row.to_dict() for row in payload["rows"]]
        }
        for seed, payload in arm.items()
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = _config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, config)
    wm_interface = WorkingMemoryInterface(connectome, interface, go_seed=GO_ALLOCATION_SEED)

    intact, intact_before, intact_after = _run_arm(
        connectome, interface, wm_interface, config, reset_before_go=False
    )
    reset, reset_before, reset_after = _run_arm(
        connectome, interface, wm_interface, config, reset_before_go=True
    )

    all_intact = [row for payload in intact.values() for row in payload["rows"]]
    all_reset = [row for payload in reset.values() for row in payload["rows"]]
    disjoint = wm_interface.allocation_summary()
    persistent_unchanged = all(
        intact_before[seed] == intact_after[seed]
        and reset_before[seed] == reset_after[seed]
        for seed in intact_before
    )
    intact_retains_state = all(row.pre_go_active_count > 0 for row in all_intact)
    reset_clears_transient = all(row.post_reset_active_count == 0 for row in all_reset)
    go_measurable = all(
        any(row.go_output_spikes > 0 for row in all_intact if row.cue == cue)
        for cue in SYMBOLS
    )
    interface_ok = (
        disjoint["go_population_size"] == 32
        and disjoint["go_seed"] == GO_ALLOCATION_SEED
        and disjoint["sensory_overlap"] == 0
        and disjoint["output_overlap"] == 0
    )

    # The corrected profiler uses disjoint phase sections.  Run the existing
    # P.1 protocol after the timing correction so its result is comparable.
    benchmark_baseline = run_benchmark(connectome, interface, config, route_cache_enabled=False)
    benchmark_optimized = run_benchmark(connectome, interface, config, route_cache_enabled=True)
    route_cache_equivalence = run_route_cache_equivalence(connectome, interface, config)
    profile_seconds = benchmark_optimized["warm"]["median_profile"]["seconds"]
    disjoint_timing = {
        "reward_update_seconds": profile_seconds.get("reward_update_seconds", 0.0),
        "post_reward_directional_seconds": profile_seconds.get("post_reward_directional_seconds", 0.0),
        "normalizer_seconds": profile_seconds.get("normalizer_seconds", 0.0),
        "plastic_lifecycle_seconds": profile_seconds.get("plastic_lifecycle_seconds", 0.0),
    }
    measured_total = sum(disjoint_timing.values())
    total = profile_seconds.get("total_training_seconds", 0.0)
    throughput = benchmark_optimized["warm"]["median_trials_per_second"]
    local_baseline_throughput = benchmark_baseline["warm"]["median_trials_per_second"]
    throughput_baseline = 19.03
    artifact = {
        "protocol": {
            "phase": "F.2A",
            "learning_enabled": False,
            "input_vocabulary": [*SYMBOLS, GO_SYMBOL],
            "output_vocabulary": list(SYMBOLS),
            "seeds": list(SEEDS),
            "trials_per_cue": TRIALS_PER_CUE,
            "trials_per_seed_per_arm": 4 * TRIALS_PER_CUE,
            "cue_ms": DelayedCueSession.cue_ms,
            "delay_ms": DelayedCueSession.delay_ms,
            "go_ms": DelayedCueSession.go_ms,
            "stimulus_rate_hz": GO_RATE_HZ,
            "reset_once_at_trial_start": True,
            "decision_phase": "GO_only",
            "go_population_size": 32,
            "go_allocation_seed": GO_ALLOCATION_SEED,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "allocation": disjoint,
        "intact": _strip_rows(intact),
        "reset_ablation": _strip_rows(reset),
        "state_memory": {
            "intact_preserves_recurrent_state": bool(intact_retains_state),
            "reset_ablation_clears_transient_state": bool(reset_clears_transient),
            "intact_pre_go_active_count_mean": float(np.mean([row.pre_go_active_count for row in all_intact])),
            "reset_post_reset_active_count_mean": float(np.mean([row.post_reset_active_count for row in all_reset])),
            "intact_output_separation_l1_hz": _output_separation(all_intact),
            "reset_output_separation_l1_hz": _output_separation(all_reset),
            "intact_unique_pre_go_fingerprints": int(len({row.pre_go_fingerprint for row in all_intact})),
            "reset_unique_pre_go_fingerprints": int(len({row.pre_go_fingerprint for row in all_reset})),
            "go_output_spikes_total_intact": int(sum(row.go_output_spikes for row in all_intact)),
            "go_output_spikes_total_reset": int(sum(row.go_output_spikes for row in all_reset)),
        },
        "learning_disabled_invariants": {
            "persistent_arrays_unchanged": bool(persistent_unchanged),
            "no_reward_or_directional_update": True,
            "no_multiplier_update": True,
            "no_stability_update": True,
            "no_plastic_allocation_update": True,
            "target_identity_not_used_by_network": True,
            "no_decoder": True,
        },
        "performance": {
            "benchmark": {
                "baseline_reference_route_cache_disabled": benchmark_baseline,
                "optimized_route_cache_enabled": benchmark_optimized,
            },
            "corrected_disjoint_timing_seconds": disjoint_timing,
            "disjoint_timing_sum_seconds": float(measured_total),
            "total_training_seconds": float(total),
            "disjoint_sections_within_total": bool(measured_total <= total + 1e-9),
            "throughput_trials_per_second": float(throughput),
            "local_baseline_throughput_trials_per_second": float(local_baseline_throughput),
            "local_speedup": float(throughput / max(local_baseline_throughput, 1e-12)),
            "reference_throughput_trials_per_second": throughput_baseline,
            "throughput_regression_fraction": float(throughput / throughput_baseline - 1.0),
            "throughput_within_5_percent": bool(throughput >= throughput_baseline * 0.95),
            "route_cache_equivalence": route_cache_equivalence,
        },
        "readiness": {
            "interface_invariants": bool(interface_ok),
            "intact_recurrent_state": bool(intact_retains_state),
            "reset_ablation": bool(reset_clears_transient),
            "GO_output_measurable": bool(go_measurable),
            "persistent_state_unchanged": bool(persistent_unchanged),
            "ready_for_f2b_learning": bool(interface_ok and intact_retains_state and reset_clears_transient and go_measurable and persistent_unchanged),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({
        "ready_for_f2b_learning": artifact["readiness"]["ready_for_f2b_learning"],
        "throughput_trials_per_second": artifact["performance"]["throughput_trials_per_second"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
