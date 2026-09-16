from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .curriculum_v2_memory import MemoryV2Config, run_memory_curriculum
from .curriculum_v21_memory import MemoryV21Config, run_memory_v21_curriculum
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1
from .phase1_stat_benchmark import _common_eval


DEFAULT_RESULT_ROOT = Path("results/phase1b_stat")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/phase1b_stat")
DEFAULT_BASELINE_RESULT_ROOT = Path("results/phase1_stat")
DEFAULT_BASELINE_CHECKPOINT_ROOT = Path("checkpoints/phase1_stat")
DEFAULT_RESULT = Path("results/latest_malecns_phase1b_stat.json")
DEFAULT_HTML = Path("results/latest_malecns_phase1b_stat.html")


@dataclass(frozen=True, slots=True)
class Phase1BConfig:
    seeds: tuple[int, ...] = (7, 17, 27)
    stage_trials: int = 512
    validation_trials_per_label: int = 32
    decoder_epochs: int = 8
    checkpoint_every: int = 64
    min_connection_synapses: int = 5
    common_eval_fraction: float = 0.70
    common_eval_rate_hz: float = 205.0


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(root: Path, checkpoint_root: Path, seed: int, version: str) -> dict[str, Path]:
    return {
        "result": root / f"seed_{seed}/{version}.json",
        "progress": root / f"seed_{seed}/{version}_progress.json",
        "checkpoint": checkpoint_root / f"seed_{seed}/{version}_brain.npz",
        "readouts": checkpoint_root / f"seed_{seed}/{version}_readouts",
    }


def _completed(report: dict[str, object] | None, *, experiment: str, seed: int, config: Phase1BConfig) -> bool:
    if not report or report.get("experiment") != experiment:
        return False
    cfg = report.get("config") or {}
    return (
        int(cfg.get("seed", -1)) == seed
        and int(cfg.get("stage_trials", -1)) == config.stage_trials
        and int(cfg.get("validation_trials_per_label", -1)) == config.validation_trials_per_label
        and int(cfg.get("decoder_epochs", -1)) == config.decoder_epochs
    )


def _clean(paths: dict[str, Path]) -> None:
    for key in ("result", "progress", "checkpoint"):
        if paths[key].is_file():
            paths[key].unlink()
    if paths["readouts"].is_dir():
        shutil.rmtree(paths["readouts"])


def _ensure_v2(connectome, *, seed: int, config: Phase1BConfig, baseline_result_root: Path, baseline_checkpoint_root: Path):
    paths = _paths(baseline_result_root, baseline_checkpoint_root, seed, "v2")
    report = _load(paths["result"])
    reusable = (
        _completed(report, experiment="malecns_memory_v2", seed=seed, config=config)
        and paths["checkpoint"].is_file()
        and paths["readouts"].is_dir()
    )
    if reusable:
        print(f"[v2 seed={seed}] reusing Phase-1A baseline")
        return report, paths, True

    print(f"[v2 seed={seed}] Phase-1A baseline unavailable; retraining")
    _clean(paths)
    run_config = MemoryV2Config(
        min_connection_synapses=config.min_connection_synapses,
        stage_trials=config.stage_trials,
        validation_trials_per_label=config.validation_trials_per_label,
        decoder_epochs=config.decoder_epochs,
        checkpoint_every=config.checkpoint_every,
        seed=seed,
    )
    report = run_memory_curriculum(
        connectome,
        config=run_config,
        checkpoint_path=paths["checkpoint"],
        readout_dir=paths["readouts"],
        progress_path=paths["progress"],
        resume=False,
        v1_baseline=None,
    )
    _write(paths["result"], report)
    return report, paths, False


def _train_v21(connectome, *, seed: int, config: Phase1BConfig, result_root: Path, checkpoint_root: Path, fresh: bool):
    paths = _paths(result_root, checkpoint_root, seed, "v21")
    existing = None if fresh else _load(paths["result"])
    reusable = (
        _completed(existing, experiment="malecns_memory_v2_1", seed=seed, config=config)
        and paths["checkpoint"].is_file()
        and paths["readouts"].is_dir()
    )
    if reusable:
        print(f"[v2.1 seed={seed}] completed result found; reusing")
        return existing, paths

    _clean(paths)
    run_config = MemoryV21Config(
        min_connection_synapses=config.min_connection_synapses,
        stage_trials=config.stage_trials,
        validation_trials_per_label=config.validation_trials_per_label,
        decoder_epochs=config.decoder_epochs,
        checkpoint_every=config.checkpoint_every,
        seed=seed,
    )
    print(f"[v2.1 seed={seed}] training adaptive replay + stage-local consolidation")
    report = run_memory_v21_curriculum(
        connectome,
        config=run_config,
        checkpoint_path=paths["checkpoint"],
        readout_dir=paths["readouts"],
        progress_path=paths["progress"],
        resume=False,
    )
    _write(paths["result"], report)
    return report, paths


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) >= 2 else 0.0}


def _replay_allocation(report: dict[str, object]) -> dict[str, object]:
    rows = {}
    for stage in report.get("replay_history", []):
        per_task = stage.get("per_task") or {}
        if per_task:
            rows[str(stage["stage"])] = per_task
    return rows


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    tasks = list(runs[0]["common_eval"]["v2"]["tasks"].keys())
    out = {}
    for task in tasks:
        base = [float(r["common_eval"]["v2"]["tasks"][task]["accuracy"]) for r in runs]
        new = [float(r["common_eval"]["v21"]["tasks"][task]["accuracy"]) for r in runs]
        delta = [b - a for a, b in zip(base, new)]
        out[task] = {
            "v2": _mean_sd(base),
            "v21": _mean_sd(new),
            "paired_delta": _mean_sd(delta),
            "v21_wins": sum(int(x > 0) for x in delta),
            "ties": sum(int(abs(x) < 1e-12) for x in delta),
            "v21_losses": sum(int(x < 0) for x in delta),
            "per_seed": [{"seed": int(runs[i]["seed"]), "v2": base[i], "v21": new[i], "delta": delta[i]} for i in range(len(runs))],
        }
    return out


def _gate(summary: dict[str, object]) -> dict[str, object]:
    thresholds = {"laterality": 0.80, "numerosity_1_4": 0.30, "compare_1_3": 0.40}
    checks = {}
    passed = True
    for task, threshold in thresholds.items():
        row = summary[task]
        mean_new = float(row["v21"]["mean"])
        delta = float(row["paired_delta"]["mean"])
        ok = mean_new >= threshold and delta >= 0.0
        checks[task] = {"v21_mean_accuracy": mean_new, "threshold": threshold, "mean_paired_delta_vs_v2": delta, "passed": ok}
        passed &= ok
    return {"passed": bool(passed), "criterion": "same-seed common held-out evaluation: v2.1 clears retention threshold and does not regress versus v2", "tasks": checks}


def build_html(report: dict[str, object]) -> str:
    summary = report["summary"]["tasks"]
    gate = report["phase1b_gate"]
    rows = []
    for task, row in summary.items():
        rows.append(f"<tr><td>{task}</td><td>{100*row['v2']['mean']:.1f}% ± {100*row['v2']['sd']:.1f}</td><td>{100*row['v21']['mean']:.1f}% ± {100*row['v21']['sd']:.1f}</td><td>{100*row['paired_delta']['mean']:+.1f} pp</td><td>{row['v21_wins']}/{len(report['runs'])}</td></tr>")
    seed_tables = []
    for run in report["runs"]:
        rs = []
        for task, base in run["common_eval"]["v2"]["tasks"].items():
            new = run["common_eval"]["v21"]["tasks"][task]
            rs.append(f"<tr><td>{task}</td><td>{100*base['accuracy']:.1f}%</td><td>{100*new['accuracy']:.1f}%</td><td>{100*(new['accuracy']-base['accuracy']):+.1f} pp</td></tr>")
        seed_tables.append(f"<h2>Seed {run['seed']}</h2><table><tr><th>Task</th><th>v2</th><th>v2.1</th><th>Δ</th></tr>{''.join(rs)}</table>")
    cls = "pass" if gate["passed"] else "fail"
    status = "PASS" if gate["passed"] else "FAIL"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>DrosoMath Phase-1B</title><style>body{{font-family:system-ui;background:#101318;color:#e8edf5;max-width:1150px;margin:auto;padding:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #2b3442;text-align:right}}th:first-child,td:first-child{{text-align:left}}.card{{padding:16px;border:1px solid #2b3442;border-radius:12px;background:#171c24}}.pass{{color:#6ee7a8}}.fail{{color:#ff8a8a}}</style></head><body><h1>DrosoMath Phase-1B: adaptive memory</h1><div class='card'>Gate: <b class='{cls}'>{status}</b><br>v2 baseline vs v2.1 adaptive replay + stage-local consolidation</div><h2>Paired summary</h2><table><tr><th>Task</th><th>v2 mean ± SD</th><th>v2.1 mean ± SD</th><th>Mean Δ</th><th>v2.1 wins</th></tr>{''.join(rows)}</table>{''.join(seed_tables)}</body></html>"""


def run_benchmark(*, data_dir: Path, config: Phase1BConfig, result_root: Path = DEFAULT_RESULT_ROOT, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, baseline_result_root: Path = DEFAULT_BASELINE_RESULT_ROOT, baseline_checkpoint_root: Path = DEFAULT_BASELINE_CHECKPOINT_ROOT, fresh: bool = False, download: bool = False) -> dict[str, object]:
    if download:
        download_malecns(data_dir)
    connectome = load_malecns_v1(data_dir, min_connection_synapses=config.min_connection_synapses)
    runs = []
    for seed in config.seeds:
        v2_report, v2_paths, baseline_reused = _ensure_v2(connectome, seed=seed, config=config, baseline_result_root=baseline_result_root, baseline_checkpoint_root=baseline_checkpoint_root)
        v21_report, v21_paths = _train_v21(connectome, seed=seed, config=config, result_root=result_root, checkpoint_root=checkpoint_root, fresh=fresh)
        common_v2 = _common_eval(connectome, seed, config, v2_paths["checkpoint"], v2_paths["readouts"])
        common_v21 = _common_eval(connectome, seed, config, v21_paths["checkpoint"], v21_paths["readouts"])
        runs.append({"seed": seed, "baseline_reused": baseline_reused, "common_eval": {"v2": common_v2, "v21": common_v21}, "v2": {"phase1_gate": v2_report.get("phase1_gate")}, "v21": {"phase1_gate": v21_report.get("phase1_gate"), "replay_allocation": _replay_allocation(v21_report), "consolidation_history": v21_report.get("consolidation_history")}})
        gc.collect()
    task_summary = _aggregate(runs)
    macro_v2 = [float(r["common_eval"]["v2"]["macro_accuracy"]) for r in runs]
    macro_v21 = [float(r["common_eval"]["v21"]["macro_accuracy"]) for r in runs]
    macro_delta = [b - a for a, b in zip(macro_v2, macro_v21)]
    return {"experiment": "malecns_phase1b_adaptive_memory_benchmark", "config": asdict(config), "connectome": connectome.summary(), "runs": runs, "summary": {"tasks": task_summary, "macro_accuracy": {"v2": _mean_sd(macro_v2), "v21": _mean_sd(macro_v21), "paired_delta": _mean_sd(macro_delta)}}, "phase1b_gate": _gate(task_summary)}


def _parse_seeds(value: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not out:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare MaleCNS memory v2 against adaptive memory v2.1")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seeds", type=_parse_seeds, default=(7, 17, 27))
    p.add_argument("--stage-trials", type=int, default=512)
    p.add_argument("--validation-trials", type=int, default=32)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = p.parse_args()
    config = Phase1BConfig(seeds=tuple(args.seeds), stage_trials=args.stage_trials, validation_trials_per_label=args.validation_trials, decoder_epochs=args.decoder_epochs, checkpoint_every=args.checkpoint_every, min_connection_synapses=args.min_syn)
    report = run_benchmark(data_dir=args.data_dir, config=config, fresh=args.fresh, download=args.download)
    _write(args.result, report)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({"phase1b_gate": report["phase1b_gate"], "summary": report["summary"]}, indent=2, sort_keys=True))
    print(f"saved result: {args.result}")
    print(f"saved dashboard: {args.html}")


if __name__ == "__main__":
    main()
