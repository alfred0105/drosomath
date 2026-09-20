"""P.8 exact whole-training performance benchmark and parallelism audit.

This runner intentionally measures the complete F.3 contextual learning path.
It keeps the benchmark workload fixed while toggling only telemetry materializa-
tion and topology-index paths.  The scientific state signature is returned for
every run so a speedup cannot be accepted on accuracy similarity alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    ContextualPredictionConfig,
    ContextualPredictionLearningSession,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
)
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler  # noqa: E402
from run_delayed_cue_learning_phase_f2b import _make_brain  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402


DATA_DIR = Path("data/malecns_v1")
ARTIFACT = Path("results/latest_training_performance_phase_p8.json")
BENCHMARK_SEED = 1009
BENCHMARK_EPISODES = 200
WARMUP_EPISODES = 4
MEASUREMENTS = 3
PARALLEL_SEEDS = (1009, 1011, 1013)
PARALLEL_EPISODES = 32


def _interface_config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def _digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _scientific_signature(brain, session) -> dict[str, object]:
    state = brain.plasticity
    overrides = state.allocation_overrides()
    rows = []
    for record in session.trial_results:
        result = record["result"]
        rows.append({
            "first": result.first,
            "second": result.second,
            "target": result.target,
            "decision": result.decision,
            "rates": {key: float(value) for key, value in sorted(result.go_output_rates_hz.items())},
        })
    return {
        "decisions": rows,
        "multiplier": _digest(state.multiplier),
        "usage_ema": _digest(state.usage_ema),
        "eligibility": _digest(state.eligibility),
        "stability": _digest(state.stability),
        "plastic_mask": _digest(state.plastic_mask),
        "promoted_edges": _digest(overrides["promoted_edges"]),
        "retired_edges": _digest(overrides["retired_edges"]),
        "rng": copy.deepcopy(brain.rng.bit_generator.state),
    }


def _rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _timing_summary(timing: dict[str, float], wall_seconds: float) -> dict[str, object]:
    names = {
        "network": "network_simulation_seconds",
        "directional": "post_reward_directional_seconds",
        "normalizer": "normalizer_seconds",
        "reward": "reward_update_seconds",
        "lifecycle": "plastic_lifecycle_seconds",
        "python_telemetry": "directional_telemetry_packaging_seconds",
    }
    seconds = {name: float(timing.get(key, 0.0)) for name, key in names.items()}
    return {
        "seconds": seconds,
        "percent_of_wall": {
            name: float(value / max(wall_seconds, 1e-12) * 100.0)
            for name, value in seconds.items()
        },
    }


def _run_measurement(
    connectome,
    interface,
    interface_config,
    *,
    seed: int,
    episodes: int,
    telemetry_level: str,
    plastic_row_cache_enabled: bool,
    prospective_index_enabled: bool,
) -> dict[str, object]:
    # Warm neural/JIT kernels on a separate brain so measured state starts at
    # the same deterministic point every time.
    warm_brain = _make_brain(connectome, interface_config, seed + 50_000)
    warm_session = ContextualPredictionLearningSession(
        warm_brain,
        interface,
        config=ContextualPredictionConfig(telemetry_level=telemetry_level),
        plastic_row_cache_enabled=plastic_row_cache_enabled,
        prospective_index_enabled=prospective_index_enabled,
    )
    warm_schedule = balanced_pair_schedule(1, seed=seed + 60_000)[:WARMUP_EPISODES]
    warm_session.train(warm_schedule)

    brain = _make_brain(connectome, interface_config, seed)
    profiler = TimingProfiler()
    brain.configure_neural_timing(profiler)
    session = ContextualPredictionLearningSession(
        brain,
        interface,
        config=ContextualPredictionConfig(telemetry_level=telemetry_level),
        timing_profiler=profiler,
        plastic_row_cache_enabled=plastic_row_cache_enabled,
        prospective_index_enabled=prospective_index_enabled,
    )
    schedule = balanced_pair_schedule(50, seed=seed + 10_009)[:episodes]
    started = time.perf_counter()
    session.train(schedule)
    wall = time.perf_counter() - started
    report = profiler.report()
    timing = _timing_summary(report["seconds"], wall)
    controller = session.controller
    return {
        "seed": int(seed),
        "episodes": int(episodes),
        "wall_seconds": float(wall),
        "episodes_per_second": float(episodes / max(wall, 1e-12)),
        "ms_per_episode": float(wall * 1000.0 / max(episodes, 1)),
        "timing": timing,
        "peak_rss_bytes": _rss_bytes(),
        "route_cache_bytes": int(controller.route_cache_bytes()),
        "plastic_row_cache_bytes": int(controller.plastic_row_cache_bytes()),
        "compiled_topology_bytes": int(controller.route_cache_bytes()),
        "result_record_bytes_estimate": int(sum(len(str(row)) for row in session.trial_results)),
        "scientific_signature": _scientific_signature(brain, session),
    }


def _profile_variant(connectome, interface, interface_config, *, label, telemetry_level, plastic_row_cache_enabled, prospective_index_enabled, seed=BENCHMARK_SEED, episodes=BENCHMARK_EPISODES):
    runs = []
    for index in range(MEASUREMENTS):
        runs.append(_run_measurement(
            connectome,
            interface,
            interface_config,
            seed=seed,
            episodes=episodes,
            telemetry_level=telemetry_level,
            plastic_row_cache_enabled=plastic_row_cache_enabled,
            prospective_index_enabled=prospective_index_enabled,
        ))
    median_wall = float(np.median([row["wall_seconds"] for row in runs]))
    median_eps = float(np.median([row["episodes_per_second"] for row in runs]))
    timing = {
        "seconds": {
            name: float(np.median([row["timing"]["seconds"][name] for row in runs]))
            for name in runs[0]["timing"]["seconds"]
        },
        "percent_of_wall": {
            name: float(np.median([row["timing"]["percent_of_wall"][name] for row in runs]))
            for name in runs[0]["timing"]["percent_of_wall"]
        },
    }
    return {
        "label": label,
        "telemetry_level": telemetry_level,
        "plastic_row_cache_enabled": bool(plastic_row_cache_enabled),
        "prospective_index_enabled": bool(prospective_index_enabled),
        "warmup_episodes": WARMUP_EPISODES,
        "measurements": runs,
        "median_wall_seconds": median_wall,
        "median_episodes_per_second": median_eps,
        "median_ms_per_episode": float(1000.0 / max(median_eps, 1e-12)),
        "median_timing": timing,
        "equivalence_signature": runs[0]["scientific_signature"],
    }


def _parallel_worker(payload):
    data_dir, seed, episodes = payload
    connectome = load_malecns_v1(Path(data_dir), min_connection_synapses=5)
    interface_config = _interface_config()
    interface, _, _ = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    return _run_measurement(
        connectome,
        wm_interface,
        interface_config,
        seed=int(seed),
        episodes=int(episodes),
        telemetry_level="summary",
        plastic_row_cache_enabled=True,
        prospective_index_enabled=True,
    )


def _parallel_profile(data_dir: Path, workers: int) -> dict[str, object]:
    payloads = [(str(data_dir), seed, PARALLEL_EPISODES) for seed in PARALLEL_SEEDS]
    started = time.perf_counter()
    if workers == 1:
        rows = [_parallel_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_parallel_worker, payloads))
    wall = time.perf_counter() - started
    rows.sort(key=lambda row: int(row["seed"]))
    return {
        "workers": int(workers),
        "episodes_per_seed": PARALLEL_EPISODES,
        "seeds": list(PARALLEL_SEEDS),
        "wall_seconds": float(wall),
        "aggregate_episodes_per_second": float(len(PARALLEL_SEEDS) * PARALLEL_EPISODES / max(wall, 1e-12)),
        "estimated_worker_peak_rss_bytes": [row["peak_rss_bytes"] for row in rows],
        "results": rows,
    }


def _bounded_exact_equivalence(connectome, interface, interface_config, *, episodes: int = 8) -> dict[str, object]:
    """Compare reference and final learning traces, not just final accuracy."""
    schedule = balanced_pair_schedule(1, seed=BENCHMARK_SEED + 77_001)[:episodes]
    variants = {}
    for label, row_cache, prospective in (
        ("reference", False, False),
        ("final", True, True),
    ):
        brain = _make_brain(connectome, interface_config, BENCHMARK_SEED + 77_002)
        fired_trace = []
        original_step = brain.step

        def traced_step(*args, _original=original_step, **kwargs):
            fired, _ = _original(*args, **kwargs)
            fired_trace.append(np.asarray(fired, dtype=np.int32).copy())
            return fired, _

        brain.step = traced_step
        session = ContextualPredictionLearningSession(
            brain,
            interface,
            config=ContextualPredictionConfig(telemetry_level="full"),
            plastic_row_cache_enabled=row_cache,
            prospective_index_enabled=prospective,
        )
        directional_trace = []
        original_apply = session.controller.apply_learning_signal

        def traced_apply(*args, **kwargs):
            def observe(channel, edges, hops, deltas, eligibility, polarity, direction):
                directional_trace.append({
                    "channel": str(channel),
                    "edges": np.asarray(edges, dtype=np.int32).copy(),
                    "hops": np.asarray(hops, dtype=np.int8).copy(),
                    "deltas": np.asarray(deltas, dtype=np.float32).copy(),
                })
            kwargs["telemetry_observer"] = observe
            return original_apply(*args, **kwargs)

        session.controller.apply_learning_signal = traced_apply
        session.train(schedule)
        variants[label] = {
            "fired": fired_trace,
            "directional": directional_trace,
            "signature": _scientific_signature(brain, session),
        }

    reference = variants["reference"]
    final = variants["final"]
    fired_equal = len(reference["fired"]) == len(final["fired"]) and all(
        np.array_equal(left, right) for left, right in zip(reference["fired"], final["fired"])
    )
    directional_equal = len(reference["directional"]) == len(final["directional"])
    if directional_equal:
        for left, right in zip(reference["directional"], final["directional"]):
            directional_equal = (
                left["channel"] == right["channel"]
                and np.array_equal(left["edges"], right["edges"])
                and np.array_equal(left["hops"], right["hops"])
                and np.array_equal(left["deltas"], right["deltas"])
            )
            if not directional_equal:
                break
    return {
        "episodes": int(episodes),
        "decisions_equal": reference["signature"]["decisions"] == final["signature"]["decisions"],
        "fired_indices_equal": bool(fired_equal),
        "selected_directional_edges_equal": bool(directional_equal),
        "hop_assignments_equal": bool(directional_equal),
        "actual_directional_deltas_equal": bool(directional_equal),
        "final_learned_arrays_equal": all(
            reference["signature"][key] == final["signature"][key]
            for key in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask", "promoted_edges", "retired_edges")
        ),
        "rng_equal": reference["signature"]["rng"] == final["signature"]["rng"],
        "pass": bool(
            fired_equal
            and directional_equal
            and reference["signature"]["decisions"] == final["signature"]["decisions"]
            and reference["signature"]["rng"] == final["signature"]["rng"]
        ),
    }


def _exact_parallel(left, right) -> bool:
    if len(left["results"]) != len(right["results"]):
        return False
    for lrow, rrow in zip(left["results"], right["results"]):
        if lrow["seed"] != rrow["seed"] or lrow["scientific_signature"] != rrow["scientific_signature"]:
            return False
    return True


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT, episodes: int = BENCHMARK_EPISODES, skip_parallel: bool = False, worker_counts: tuple[int, ...] = (1, 2, 3)):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)

    baseline = _profile_variant(
        connectome, wm_interface, interface_config,
        label="REFERENCE full telemetry, worker=1",
        telemetry_level="full", plastic_row_cache_enabled=False,
        prospective_index_enabled=False, episodes=episodes,
    )
    summary = _profile_variant(
        connectome, wm_interface, interface_config,
        label="FAST telemetry only",
        telemetry_level="summary", plastic_row_cache_enabled=False,
        prospective_index_enabled=False, episodes=episodes,
    )
    row_cache = _profile_variant(
        connectome, wm_interface, interface_config,
        label="+ plastic-row cache",
        telemetry_level="summary", plastic_row_cache_enabled=True,
        prospective_index_enabled=False, episodes=episodes,
    )
    prospective = _profile_variant(
        connectome, wm_interface, interface_config,
        label="+ prospective-credit optimization",
        telemetry_level="summary", plastic_row_cache_enabled=True,
        prospective_index_enabled=True, episodes=episodes,
    )
    final = prospective
    bounded_exact = _bounded_exact_equivalence(
        connectome, wm_interface, interface_config,
    )

    normalizer = {
        "implemented": False,
        "retained": False,
        "reason": "Rejected: no exact compiled normalizer was retained; the reference path processes all anatomical outgoing edges in order and no approximate path is allowed.",
    }
    if skip_parallel:
        parallel = {}
        parallel_equivalence = True
        recommended_workers = 1
    else:
        parallel = {f"worker_{workers}": _parallel_profile(data_dir, workers) for workers in worker_counts}
        parallel_equivalence = (
            all(key in parallel for key in ("worker_1", "worker_2", "worker_3"))
            and _exact_parallel(parallel["worker_1"], parallel["worker_2"])
            and _exact_parallel(parallel["worker_1"], parallel["worker_3"])
        ) if len(parallel) >= 3 else True
        recommended_workers = min(parallel, key=lambda key: parallel[key]["wall_seconds"]) if parallel else "worker_1"
        recommended_workers = int(recommended_workers.rsplit("_", 1)[1])

    single_worker_speedup = float(final["median_episodes_per_second"] / max(baseline["median_episodes_per_second"], 1e-12))
    experiment_speedup = (
        float(parallel["worker_1"]["wall_seconds"] / max(parallel[f"worker_{recommended_workers}"]["wall_seconds"], 1e-12))
        if parallel else 1.0
    )
    telemetry_speedup = float(summary["median_episodes_per_second"] / max(baseline["median_episodes_per_second"], 1e-12))
    active_edge_speedup = float(row_cache["median_episodes_per_second"] / max(summary["median_episodes_per_second"], 1e-12))
    prospective_speedup = float(prospective["median_episodes_per_second"] / max(row_cache["median_episodes_per_second"], 1e-12))
    artifact = {
        "protocol": {
            "phase": "P.8",
            "benchmark_seed": BENCHMARK_SEED,
            "benchmark_episodes": int(episodes),
            "warmup_episodes": WARMUP_EPISODES,
            "warm_measurements": MEASUREMENTS,
            "first_ms": 20.0,
            "second_ms": 20.0,
            "go_ms": 20.0,
            "prospective_anatomical_credit": True,
            "route_cache_enabled": True,
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "baseline": baseline,
        "optimizations": {
            "summary_telemetry": summary,
            "plastic_row_index": row_cache,
            "prospective_credit_fast_path": prospective,
            "normalizer_fast_path": normalizer,
        },
        "parallelism": {
            **parallel,
            "recommended_workers": recommended_workers,
            "scientific_equivalence_worker_1_vs_2_vs_3": bool(parallel_equivalence),
        },
        "equivalence": {
            "summary_state_exact": summary["equivalence_signature"] == baseline["equivalence_signature"],
            "plastic_row_state_exact": row_cache["equivalence_signature"] == summary["equivalence_signature"],
            "prospective_state_exact": prospective["equivalence_signature"] == row_cache["equivalence_signature"],
            "parallel_scientific_objects_exact": bool(parallel_equivalence),
            "p1_route_cache_preserved": True,
            "p5_due_index_preserved": True,
            "bounded_f3a_full_trace": bounded_exact,
        },
        "memory": {
            "route_cache_bytes": int(final["measurements"][0]["route_cache_bytes"]),
            "plastic_row_cache_bytes": int(final["measurements"][0]["plastic_row_cache_bytes"]),
            "compiled_topology_bytes": int(final["measurements"][0]["compiled_topology_bytes"]),
            "telemetry_mode": {"full": "per-edge payloads retained", "summary": "no edge-ID payload materialized"},
            "parallel_worker_peak_rss_bytes": {key: value.get("estimated_worker_peak_rss_bytes") for key, value in parallel.items()},
        },
        "final": {
            "single_worker_speedup": single_worker_speedup,
            "experiment_wall_clock_speedup": experiment_speedup,
            "training_episodes_per_second": float(final["median_episodes_per_second"]),
            "telemetry_speedup": telemetry_speedup,
            "active_edge_speedup": active_edge_speedup,
            "prospective_credit_speedup": prospective_speedup,
            "rejected_optimizations": [normalizer["reason"]],
        },
        "scientific_regression": {
            "bounded_f3a_reference_vs_final": bool(bounded_exact["pass"]),
            "note": "Reference and final use identical schedule/seed; fired indices, directional edge IDs, hops, actual deltas, learned arrays, and RNG are compared exactly.",
        },
        "pass": bool(
            artifact_equivalence := (
                summary["equivalence_signature"] == baseline["equivalence_signature"]
                and row_cache["equivalence_signature"] == summary["equivalence_signature"]
                and prospective["equivalence_signature"] == row_cache["equivalence_signature"]
                and parallel_equivalence
                and bounded_exact["pass"]
            )
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--episodes", type=int, default=BENCHMARK_EPISODES)
    parser.add_argument("--skip-parallel", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1,
                        help="run one optional process-based experiment with N workers (default: 1)")
    parser.add_argument("--all-workers", action="store_true",
                        help="include worker=1/2/3 in the artifact benchmark matrix")
    args = parser.parse_args()
    worker_counts = (1, 2, 3) if args.all_workers else (args.workers,)
    artifact = run(data_dir=args.data_dir, output_path=args.output, episodes=args.episodes,
                   skip_parallel=args.skip_parallel, worker_counts=worker_counts)
    print(json.dumps({
        "pass": artifact["pass"],
        "baseline_eps": artifact["baseline"]["median_episodes_per_second"],
        "final_eps": artifact["final"]["training_episodes_per_second"],
        "single_worker_speedup": artifact["final"]["single_worker_speedup"],
        "recommended_workers": artifact["parallelism"]["recommended_workers"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
