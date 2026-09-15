from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import PlasticStateConfig

from .brain import PlasticMaleCNSBrain
from .checkpoint import restore_learning_checkpoint, restore_readout_checkpoint
from .curriculum_v1 import (
    CurriculumV1Config,
    _degrade,
    _readout_path,
    build_curriculum,
    run_curriculum,
)
from .curriculum_v2_memory import MemoryV2Config, run_memory_curriculum
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1
from .output_readout import OutputReadoutConfig, PopulationReadout
from .output_session import MaleCNSOutputSession


DEFAULT_ROOT = Path("results/phase1_stat")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/phase1_stat")
DEFAULT_RESULT = Path("results/latest_malecns_phase1_stat.json")
DEFAULT_HTML = Path("results/latest_malecns_phase1_stat.html")


@dataclass(frozen=True, slots=True)
class Phase1StatConfig:
    seeds: tuple[int, ...] = (7, 17, 27)
    stage_trials: int = 512
    validation_trials_per_label: int = 32
    decoder_epochs: int = 8
    checkpoint_every: int = 64
    min_connection_synapses: int = 5
    common_eval_fraction: float = 0.70
    common_eval_rate_hz: float = 205.0

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.stage_trials < 1 or self.validation_trials_per_label < 1:
            raise ValueError("trial counts must be >= 1")
        if self.decoder_epochs < 1 or self.checkpoint_every < 0:
            raise ValueError("invalid decoder/checkpoint settings")
        if not 0.0 < self.common_eval_fraction <= 1.0:
            raise ValueError("common_eval_fraction must be in (0, 1]")
        if self.common_eval_rate_hz <= 0.0:
            raise ValueError("common_eval_rate_hz must be > 0")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(result_root: Path, checkpoint_root: Path, seed: int, version: str) -> dict[str, Path]:
    stem = f"seed_{seed}/{version}"
    return {
        "result": result_root / f"{stem}.json",
        "checkpoint": checkpoint_root / f"{stem}_brain.npz",
        "readouts": checkpoint_root / f"{stem}_readouts",
        "progress": result_root / f"{stem}_progress.json",
    }


def _clean(paths: dict[str, Path]) -> None:
    for key in ("result", "checkpoint", "progress"):
        if paths[key].is_file():
            paths[key].unlink()
    if paths["readouts"].is_dir():
        shutil.rmtree(paths["readouts"])


def _matches(report: dict[str, object] | None, version: str, seed: int, config: Phase1StatConfig) -> bool:
    if not report:
        return False
    expected = "malecns_curriculum_v1" if version == "v1" else "malecns_memory_v2"
    cfg = report.get("config") or {}
    return (
        report.get("experiment") == expected
        and int(cfg.get("seed", -1)) == seed
        and int(cfg.get("stage_trials", -1)) == config.stage_trials
        and int(cfg.get("validation_trials_per_label", -1)) == config.validation_trials_per_label
        and int(cfg.get("decoder_epochs", -1)) == config.decoder_epochs
    )


def _train_one(connectome, version: str, seed: int, config: Phase1StatConfig, paths: dict[str, Path], fresh: bool) -> dict[str, object]:
    cached = None if fresh else _load_json(paths["result"])
    if _matches(cached, version, seed, config):
        print(f"[{version} seed={seed}] reusing completed run")
        return cached

    # Completed model/seed runs are resumable at benchmark level. An interrupted
    # individual run is restarted so its stage/retention report cannot be partial.
    _clean(paths)
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    if version == "v1":
        cfg = CurriculumV1Config(
            min_connection_synapses=config.min_connection_synapses,
            stage_trials=config.stage_trials,
            validation_trials_per_label=config.validation_trials_per_label,
            decoder_epochs=config.decoder_epochs,
            checkpoint_every=config.checkpoint_every,
            seed=seed,
        )
        print(f"[v1 seed={seed}] {config.stage_trials} trials/stage")
        report = run_curriculum(
            connectome,
            config=cfg,
            checkpoint_path=paths["checkpoint"],
            readout_dir=paths["readouts"],
            resume=False,
        )
    elif version == "v2":
        cfg = MemoryV2Config(
            min_connection_synapses=config.min_connection_synapses,
            stage_trials=config.stage_trials,
            validation_trials_per_label=config.validation_trials_per_label,
            decoder_epochs=config.decoder_epochs,
            checkpoint_every=config.checkpoint_every,
            seed=seed,
        )
        print(f"[v2 seed={seed}] {config.stage_trials} trials/stage + replay")
        report = run_memory_curriculum(
            connectome,
            config=cfg,
            checkpoint_path=paths["checkpoint"],
            readout_dir=paths["readouts"],
            progress_path=paths["progress"],
            resume=False,
            observer=None,
            live_telemetry=False,
            v1_baseline=None,
        )
    else:
        raise ValueError(version)

    _write_json(paths["result"], report)
    return report


def _common_eval(connectome, seed: int, config: Phase1StatConfig, checkpoint: Path, readout_dir: Path) -> dict[str, object]:
    """Evaluate v1/v2 on the same fresh held-out stimulus and simulator streams."""
    import numpy as np

    cfg = CurriculumV1Config(
        min_connection_synapses=config.min_connection_synapses,
        stage_trials=config.stage_trials,
        validation_trials_per_label=config.validation_trials_per_label,
        decoder_epochs=config.decoder_epochs,
        checkpoint_every=config.checkpoint_every,
        seed=seed,
    )
    tasks, output = build_curriculum(connectome, config=cfg)
    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed + 777_777,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.20, seed=seed),
    )
    restore_learning_checkpoint(checkpoint, brain=brain)

    task_rows: dict[str, object] = {}
    total_correct = 0
    total_trials = 0
    for i, task in enumerate(tasks):
        readout = PopulationReadout(
            output,
            task.labels,
            config=OutputReadoutConfig(learning_rate=0.08, seed=seed + 100 + i),
        )
        restore_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        session = MaleCNSOutputSession(brain, readout)
        rng = np.random.default_rng(900_000 + seed * 100 + i)
        correct = silent = spikes = 0
        n = len(task.labels) * config.validation_trials_per_label
        for label in task.labels:
            for _ in range(config.validation_trials_per_label):
                stimulus = _degrade(rng, task.sample(label, rng), config.common_eval_fraction)
                row = session.evaluate_trial(
                    stimulus_body_ids=stimulus,
                    target=label,
                    duration_ms=cfg.duration_ms,
                    stimulus_rate_hz=config.common_eval_rate_hz,
                )
                correct += int(bool(row["correct"]))
                silent += int(bool(row.get("silent")))
                spikes += int(row["total_output_spikes"])
        total_correct += correct
        total_trials += n
        task_rows[task.name] = {
            "accuracy": correct / n,
            "correct": correct,
            "trials": n,
            "chance": 1.0 / len(task.labels),
            "silent_fraction": silent / n,
            "mean_output_spikes": spikes / n,
        }

    return {
        "tasks": task_rows,
        "macro_accuracy": statistics.mean(float(x["accuracy"]) for x in task_rows.values()),
        "micro_accuracy": total_correct / total_trials,
        "total_correct": total_correct,
        "total_trials": total_trials,
    }


def _stage_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        str(row["stage"]): {
            "before_accuracy": float(row["before_accuracy"]),
            "after_accuracy": float(row["after_accuracy"]),
            "delta_accuracy": float(row["delta_accuracy"]),
            "training_accuracy": float(row["training_accuracy"]),
        }
        for row in report.get("stages", [])
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def _task_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    names = list(runs[0]["common_eval"]["v1"]["tasks"].keys())
    out: dict[str, object] = {}
    for name in names:
        v1 = [float(r["common_eval"]["v1"]["tasks"][name]["accuracy"]) for r in runs]
        v2 = [float(r["common_eval"]["v2"]["tasks"][name]["accuracy"]) for r in runs]
        delta = [b - a for a, b in zip(v1, v2)]
        out[name] = {
            "v1": _mean_sd(v1),
            "v2": _mean_sd(v2),
            "paired_delta": _mean_sd(delta),
            "seed_wins_v2": sum(x > 0 for x in delta),
            "seed_ties": sum(abs(x) < 1e-12 for x in delta),
            "seed_losses_v2": sum(x < 0 for x in delta),
            "per_seed": [
                {"seed": int(runs[i]["seed"]), "v1": v1[i], "v2": v2[i], "delta": delta[i]}
                for i in range(len(runs))
            ],
        }
    return out


def _gate(summary: dict[str, object]) -> dict[str, object]:
    thresholds = {"laterality": 0.80, "numerosity_1_4": 0.30, "compare_1_3": 0.40}
    checks = {}
    passed = True
    for name, threshold in thresholds.items():
        row = summary[name]
        mean_v2 = float(row["v2"]["mean"])
        mean_delta = float(row["paired_delta"]["mean"])
        ok = mean_v2 >= threshold and mean_delta >= 0.0
        checks[name] = {
            "v2_mean_accuracy": mean_v2,
            "threshold": threshold,
            "mean_paired_delta_vs_v1": mean_delta,
            "passed": ok,
        }
        passed &= ok
    return {
        "passed": bool(passed),
        "criterion": "paired common held-out evaluation: v2 mean retention clears threshold and does not regress versus same-seed v1",
        "tasks": checks,
    }


def build_html(report: dict[str, object]) -> str:
    cfg = report["config"]
    summary = report["summary"]["tasks"]
    gate = report["phase1_stat_gate"]
    rows = []
    for name, x in summary.items():
        rows.append(
            f"<tr><td>{name}</td><td>{100*x['v1']['mean']:.1f}% ± {100*x['v1']['sd']:.1f}</td>"
            f"<td>{100*x['v2']['mean']:.1f}% ± {100*x['v2']['sd']:.1f}</td>"
            f"<td>{100*x['paired_delta']['mean']:+.1f} pp</td><td>{x['seed_wins_v2']}/{len(report['runs'])}</td></tr>"
        )
    seeds = []
    for run in report["runs"]:
        rs = []
        for name, a in run["common_eval"]["v1"]["tasks"].items():
            b = run["common_eval"]["v2"]["tasks"][name]
            rs.append(
                f"<tr><td>{name}</td><td>{100*a['accuracy']:.1f}%</td><td>{100*b['accuracy']:.1f}%</td>"
                f"<td>{100*(b['accuracy']-a['accuracy']):+.1f} pp</td></tr>"
            )
        seeds.append(f"<h2>Seed {run['seed']}</h2><table><tr><th>Task</th><th>v1</th><th>v2</th><th>Δ</th></tr>{''.join(rs)}</table>")
    status = "PASS" if gate["passed"] else "FAIL"
    cls = "pass" if gate["passed"] else "fail"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>DrosoMath Phase-1 statistical benchmark</title>
<style>body{{font-family:system-ui;background:#101318;color:#e8edf5;max-width:1150px;margin:auto;padding:28px}}table{{width:100%;border-collapse:collapse;margin:14px 0 28px}}th,td{{padding:9px;border-bottom:1px solid #2b3442;text-align:right}}th:first-child,td:first-child{{text-align:left}}.card{{padding:16px;border:1px solid #2b3442;border-radius:12px;background:#171c24}}.pass{{color:#6ee7a8}}.fail{{color:#ff8a8a}}</style></head><body>
<h1>DrosoMath Phase-1 statistical benchmark</h1><div class='card'>Gate: <b class='{cls}'>{status}</b><br>Seeds: {', '.join(str(x) for x in cfg['seeds'])} · {cfg['stage_trials']} trials/stage · {cfg['validation_trials_per_label']} validation/class</div>
<h2>Paired v1 vs v2</h2><table><tr><th>Task</th><th>v1 mean ± SD</th><th>v2 mean ± SD</th><th>Mean Δ</th><th>v2 wins</th></tr>{''.join(rows)}</table>{''.join(seeds)}</body></html>"""


def run_benchmark(data_dir: Path, config: Phase1StatConfig, result_root: Path = DEFAULT_ROOT, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, fresh: bool = False, download: bool = False) -> dict[str, object]:
    if download:
        download_malecns(data_dir)
    connectome = load_malecns_v1(data_dir, min_connection_synapses=config.min_connection_synapses)
    runs = []
    for seed in config.seeds:
        pair: dict[str, object] = {"seed": seed, "reports": {}, "common_eval": {}}
        for version in ("v1", "v2"):
            paths = _paths(result_root, checkpoint_root, seed, version)
            report = _train_one(connectome, version, seed, config, paths, fresh)
            pair["reports"][version] = {
                "stage_summary": _stage_summary(report),
                "final_plasticity": report["final_plasticity"],
                "phase1_gate": report.get("phase1_gate"),
                "result_path": str(paths["result"]),
            }
            pair["common_eval"][version] = _common_eval(connectome, seed, config, paths["checkpoint"], paths["readouts"])
            gc.collect()
        runs.append(pair)

    tasks = _task_summary(runs)
    v1_macro = [float(r["common_eval"]["v1"]["macro_accuracy"]) for r in runs]
    v2_macro = [float(r["common_eval"]["v2"]["macro_accuracy"]) for r in runs]
    report = {
        "experiment": "malecns_phase1_stat_benchmark",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "runs": runs,
        "summary": {
            "tasks": tasks,
            "macro_accuracy": {
                "v1": _mean_sd(v1_macro),
                "v2": _mean_sd(v2_macro),
                "paired_delta": _mean_sd([b - a for a, b in zip(v1_macro, v2_macro)]),
            },
        },
        "phase1_stat_gate": _gate(tasks),
    }
    return report


def _seed_list(text: str) -> tuple[int, ...]:
    out = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not out:
        raise argparse.ArgumentTypeError("empty seed list")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Paired multi-seed MaleCNS v1 vs memory-v2 benchmark")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seeds", type=_seed_list, default=(7, 17, 27))
    p.add_argument("--stage-trials", type=int, default=512)
    p.add_argument("--validation-trials", type=int, default=32)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()

    config = Phase1StatConfig(
        seeds=tuple(a.seeds),
        stage_trials=a.stage_trials,
        validation_trials_per_label=a.validation_trials,
        decoder_epochs=a.decoder_epochs,
        checkpoint_every=a.checkpoint_every,
        min_connection_synapses=a.min_syn,
    )
    report = run_benchmark(a.data_dir, config, a.result_root, a.checkpoint_root, a.fresh, a.download)
    _write_json(a.result, report)
    a.html.parent.mkdir(parents=True, exist_ok=True)
    a.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({"phase1_stat_gate": report["phase1_stat_gate"], "summary": report["summary"]}, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")
    print(f"saved dashboard: {a.html}")


if __name__ == "__main__":
    main()
