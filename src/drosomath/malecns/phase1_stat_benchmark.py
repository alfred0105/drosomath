from __future__ import annotations

import argparse
import gc
import json
import math
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
        if len(self.seeds) < 1:
            raise ValueError("at least one seed is required")
        if self.stage_trials < 1:
            raise ValueError("stage_trials must be >= 1")
        if self.validation_trials_per_label < 1:
            raise ValueError("validation_trials_per_label must be >= 1")
        if self.decoder_epochs < 1:
            raise ValueError("decoder_epochs must be >= 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0")
        if not 0.0 < self.common_eval_fraction <= 1.0:
            raise ValueError("common_eval_fraction must be in (0, 1]")
        if self.common_eval_rate_hz <= 0.0:
            raise ValueError("common_eval_rate_hz must be > 0")


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_paths(result_root: Path, checkpoint_root: Path, seed: int, version: str) -> dict[str, Path]:
    stem = f"seed_{seed}/{version}"
    return {
        "result": result_root / f"{stem}.json",
        "checkpoint": checkpoint_root / f"{stem}_brain.npz",
        "readouts": checkpoint_root / f"{stem}_readouts",
        "progress": result_root / f"{stem}_progress.json",
    }


def _clean_paths(paths: dict[str, Path]) -> None:
    for key in ("result", "checkpoint", "progress"):
        path = paths[key]
        if path.is_file():
            path.unlink()
    readouts = paths["readouts"]
    if readouts.is_dir():
        shutil.rmtree(readouts)


def _completed_result_matches(report: dict[str, object] | None, *, version: str, seed: int, config: Phase1StatConfig) -> bool:
    if not report:
        return False
    expected_experiment = "malecns_curriculum_v1" if version == "v1" else "malecns_memory_v2"
    if report.get("experiment") != expected_experiment:
        return False
    cfg = report.get("config") or {}
    return (
        int(cfg.get("seed", -1)) == int(seed)
        and int(cfg.get("stage_trials", -1)) == config.stage_trials
        and int(cfg.get("validation_trials_per_label", -1)) == config.validation_trials_per_label
        and int(cfg.get("decoder_epochs", -1)) == config.decoder_epochs
    )


def _train_one(connectome, *, version: str, seed: int, config: Phase1StatConfig, paths: dict[str, Path], fresh: bool) -> dict[str, object]:
    existing = None if fresh else _json_load(paths["result"])
    if _completed_result_matches(existing, version=version, seed=seed, config=config):
        print(f"[{version} seed={seed}] completed result found; reusing it")
        return existing  # type: ignore[return-value]

    # A partial v1 checkpoint does not contain prior stage reports, so an interrupted
    # benchmark run is restarted from the beginning. Completed seed/version runs are
    # still reused, making benchmark-level resume safe.
    _clean_paths(paths)
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    if version == "v1":
        run_config = CurriculumV1Config(
            min_connection_synapses=config.min_connection_synapses,
            stage_trials=config.stage_trials,
            validation_trials_per_label=config.validation_trials_per_label,
            decoder_epochs=config.decoder_epochs,
            checkpoint_every=config.checkpoint_every,
            seed=seed,
        )
        print(f"[v1 seed={seed}] training {config.stage_trials} trials/stage")
        report = run_curriculum(
            connectome,
            config=run_config,
            checkpoint_path=paths["checkpoint"],
            readout_dir=paths["readouts"],
            resume=False,
        )
    elif version == "v2":
        run_config = MemoryV2Config(
            min_connection_synapses=config.min_connection_synapses,
            stage_trials=config.stage_trials,
            validation_trials_per_label=config.validation_trials_per_label,
            decoder_epochs=config.decoder_epochs,
            checkpoint_every=config.checkpoint_every,
            seed=seed,
        )
        print(f"[v2 seed={seed}] training {config.stage_trials} trials/stage + replay")
        report = run_memory_curriculum(
            connectome,
            config=run_config,
            checkpoint_path=paths["checkpoint"],
            readout_dir=paths["readouts"],
            progress_path=paths["progress"],
            resume=False,
            observer=None,
            live_telemetry=False,
            v1_baseline=None,
        )
    else:
        raise ValueError(f"unknown version: {version}")

    _json_write(paths["result"], report)
    return report


def _common_final_evaluation(
    connectome,
    *,
    seed: int,
    config: Phase1StatConfig,
    checkpoint_path: Path,
    readout_dir: Path,
) -> dict[str, object]:
    """Re-evaluate a trained checkpoint on exactly the same held-out streams.

    This intentionally ignores version-specific evaluation seeds in the original
    v1/v2 reports. Both checkpoints get a fresh brain with the same RNG seed and
    the same task/example RNG streams, so paired differences are much cleaner.
    """
    import numpy as np

    eval_config = CurriculumV1Config(
        min_connection_synapses=config.min_connection_synapses,
        stage_trials=config.stage_trials,
        validation_trials_per_label=config.validation_trials_per_label,
        decoder_epochs=config.decoder_epochs,
        checkpoint_every=config.checkpoint_every,
        seed=seed,
    )
    tasks, output = build_curriculum(connectome, config=eval_config)
    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed + 777_777,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.20, seed=seed),
    )
    restore_learning_checkpoint(checkpoint_path, brain=brain)

    task_rows: dict[str, object] = {}
    total_correct = 0
    total_trials = 0
    for task_index, task in enumerate(tasks):
        readout = PopulationReadout(
            output,
            task.labels,
            config=OutputReadoutConfig(learning_rate=0.08, seed=seed + 100 + task_index),
        )
        restore_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        session = MaleCNSOutputSession(brain, readout)
        rng = np.random.default_rng(900_000 + seed * 100 + task_index)
        correct = 0
        silent = 0
        spikes = 0
        rows = []
        for label in task.labels:
            for _ in range(config.validation_trials_per_label):
                stimulus = _degrade(rng, task.sample(label, rng), config.common_eval_fraction)
                row = session.evaluate_trial(
                    stimulus_body_ids=stimulus,
                    target=label,
                    duration_ms=eval_config.duration_ms,
                    stimulus_rate_hz=config.common_eval_rate_hz,
                )
                correct += int(bool(row["correct"]))
                silent += int(bool(row.get("silent")))
                spikes += int(row["total_output_spikes"])
                rows.append({
                    "target": row["target"],
                    "prediction": row["prediction"],
                    "correct": bool(row["correct"]),
                    "silent": bool(row.get("silent")),
                })
        n = len(rows)
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
        "macro_accuracy": statistics.mean(float(v["accuracy"]) for v in task_rows.values()),
        "micro_accuracy": total_correct / total_trials,
        "total_correct": total_correct,
        "total_trials": total_trials,
    }


def _stage_summary(report: dict[str, object]) -> dict[str, object]:
    out = {}
    for row in report.get("stages", []):
        out[str(row["stage"])] = {
            "before_accuracy": float(row["before_accuracy"]),
            "after_accuracy": float(row["after_accuracy"]),
            "delta_accuracy": float(row["delta_accuracy"]),
            "training_accuracy": float(row["training_accuracy"]),
            "training_silent_fraction": float(row["training_silent_fraction"]),
        }
    return out


def _mean_sd(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def _aggregate_task_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {}
    task_names = list(runs[0]["common_eval"]["v1"]["tasks"].keys())
    summary: dict[str, object] = {}
    for task in task_names:
        v1 = [float(row["common_eval"]["v1"]["tasks"][task]["accuracy"]) for row in runs]
        v2 = [float(row["common_eval"]["v2"]["tasks"][task]["accuracy"]) for row in runs]
        delta = [b - a for a, b in zip(v1, v2)]
        summary[task] = {
            "v1": _mean_sd(v1),
            "v2": _mean_sd(v2),
            "paired_delta": _mean_sd(delta),
            "seed_wins_v2": sum(int(x > 0.0) for x in delta),
            "seed_ties": sum(int(abs(x) < 1e-12) for x in delta),
            "seed_losses_v2": sum(int(x < 0.0) for x in delta),
            "per_seed": [
                {"seed": int(runs[i]["seed"]), "v1": v1[i], "v2": v2[i], "delta": delta[i]}
                for i in range(len(runs))
            ],
        }
    return summary


def _phase1_stat_gate(summary: dict[str, object]) -> dict[str, object]:
    thresholds = {"laterality": 0.80, "numerosity_1_4": 0.30, "compare_1_3": 0.40}
    checks = {}
    passed = True
    for task, threshold in thresholds.items():
        row = summary.get(task)
        if not row:
            checks[task] = {"passed": False, "reason": "missing"}
            passed = False
            continue
        mean_v2 = float(row["v2"]["mean"])
        mean_delta = float(row["paired_delta"]["mean"])
        # Phase 1 is about preserving prior knowledge. Require an absolute retained
        # level and no average regression versus the unprotected baseline.
        ok = mean_v2 >= threshold and mean_delta >= 0.0
        checks[task] = {
            "v2_mean_accuracy": mean_v2,
            "threshold": threshold,
            "mean_paired_delta_vs_v1": mean_delta,
            "passed": ok,
        }
        passed &= ok
    return {
        "passed": bool(passed),
        "criterion": "3-seed common held-out evaluation: v2 mean retention must clear task threshold and not regress vs paired v1",
        "tasks": checks,
    }


def build_html(report: dict[str, object]) -> str:
    esc = __import__("html").escape
    cfg = report["config"]
    summary = report["summary"]["tasks"]
    gate = report["phase1_stat_gate"]
    rows = []
    for task, values in summary.items():
        v1 = values["v1"]
        v2 = values["v2"]
        delta = values["paired_delta"]
        rows.append(
            "<tr>"
            f"<td>{esc(task)}</td>"
            f"<td>{100*float(v1['mean']):.1f}% ± {100*float(v1['sd']):.1f}</td>"
            f"<td>{100*float(v2['mean']):.1f}% ± {100*float(v2['sd']):.1f}</td>"
            f"<td>{100*float(delta['mean']):+.1f} pp</td>"
            f"<td>{values['seed_wins_v2']}/{len(report['runs'])}</td>"
            "</tr>"
        )
    seed_blocks = []
    for run in report["runs"]:
        task_rows = []
        for task, v1row in run["common_eval"]["v1"]["tasks"].items():
            v2row = run["common_eval"]["v2"]["tasks"][task]
            task_rows.append(
                f"<tr><td>{esc(task)}</td><td>{100*float(v1row['accuracy']):.1f}%</td>"
                f"<td>{100*float(v2row['accuracy']):.1f}%</td>"
                f"<td>{100*(float(v2row['accuracy'])-float(v1row['accuracy'])):+.1f} pp</td></tr>"
            )
        seed_blocks.append(
            f"<section><h2>Seed {run['seed']}</h2><table><thead><tr><th>Task</th><th>v1</th><th>v2</th><th>Δ</th></tr></thead>"
            f"<tbody>{''.join(task_rows)}</tbody></table></section>"
        )
    status = "PASS" if gate["passed"] else "FAIL"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>DrosoMath Phase-1 statistical benchmark</title>
<style>
body{{font-family:system-ui,sans-serif;background:#101318;color:#e8edf5;margin:0;padding:28px;max-width:1200px;margin:auto}}
h1,h2{{margin:.3em 0}} .card{{background:#171c24;border:1px solid #2a3340;border-radius:14px;padding:18px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;margin:12px 0 24px}} th,td{{padding:10px;border-bottom:1px solid #2a3340;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.pass{{color:#6ee7a8}} .fail{{color:#ff8a8a}} code{{background:#202633;padding:2px 6px;border-radius:6px}}
</style></head><body>
<h1>DrosoMath Phase-1 statistical benchmark</h1>
<div class='card'><b>Gate: <span class='{'pass' if gate['passed'] else 'fail'}'>{status}</span></b><br>
Seeds: {', '.join(str(x) for x in cfg['seeds'])} · {cfg['stage_trials']} trials/stage · {cfg['validation_trials_per_label']} validation trials/class · common paired held-out evaluation</div>
<section><h2>Paired v1 vs v2 summary</h2><table><thead><tr><th>Task</th><th>v1 mean ± SD</th><th>v2 mean ± SD</th><th>Mean Δ</th><th>v2 wins</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(seed_blocks)}
</body></html>"""


def run_benchmark(
    *,
    data_dir: Path,
    config: Phase1StatConfig,
    result_root: Path = DEFAULT_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    fresh: bool = False,
    download: bool = False,
) -> dict[str, object]:
    if download:
        download_malecns(data_dir)
    connectome = load_malecns_v1(data_dir, min_connection_synapses=config.min_connection_synapses)
    runs = []

    for seed in config.seeds:
        pair: dict[str, object] = {"seed": int(seed), "reports": {}, "common_eval": {}}
        for version in ("v1", "v2"):
            paths = _run_paths(result_root, checkpoint_root, seed, version)
            report = _train_one(
                connectome,
                version=version,
                seed=seed,
                config=config,
                paths=paths,
                fresh=fresh,
            )
            pair["reports"][version] = {
                "experiment": report["experiment"],
                "stage_summary": _stage_summary(report),
                "final_plasticity": report["final_plasticity"],
                "phase1_gate": report.get("phase1_gate"),
                "result_path": str(paths["result"]),
            }
            pair["common_eval"][version] = _common_final_evaluation(
                connectome,
                seed=seed,
                config=config,
                checkpoint_path=paths["checkpoint"],
                readout_dir=paths["readouts"],
            )
            gc.collect()
        runs.append(pair)

    task_summary = _aggregate_task_summary(runs)
    macro_v1 = [float(row["common_eval"]["v1"]["macro_accuracy"]) for row in runs]
    macro_v2 = [float(row["common_eval"]["v2"]["macro_accuracy"]) for row in runs]
    macro_delta = [b - a for a, b in zip(macro_v1, macro_v2)]
    report = {
        "experiment": "malecns_phase1_stat_benchmark",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "runs": runs,
        "summary": {
            "tasks": task_summary,
            "macro_accuracy": {
                "v1": _mean_sd(macro_v1),
                "v2": _mean_sd(macro_v2),
                "paired_delta": _mean_sd(macro_delta),
            },
        },
        "phase1_stat_gate": _phase1_stat_gate(task_summary),
    }
    return report


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    return seeds


def main() -> None:
    p = argparse.ArgumentParser(description="Paired multi-seed MaleCNS v1 vs memory-v2 benchmark")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--seeds", type=_parse_seeds, default=(7, 17, 27))
    p.add_argument("--stage-trials", type=int, default=512)
    p.add_argument("--validation-trials", type=int, default=32)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--fresh", action="store_true")n    a = p.parse_args()

    config = Phase1StatConfig(
        seeds=tuple(a.seeds),
        stage_trials=a.stage_trials,
        validation_trials_per_label=a.validation_trials,
        decoder_epochs=a.decoder_epochs,
        checkpoint_every=a.checkpoint_every,
        min_connection_synapses=a.min_syn,
    )
    report = run_benchmark(
        data_dir=a.data_dir,
        config=config,
        result_root=a.result_root,
        checkpoint_root=a.checkpoint_root,
        fresh=a.fresh,
        download=a.download,
    )
    _json_write(a.result, report)
    a.html.parent.mkdir(parents=True, exist_ok=True)
    a.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({"phase1_stat_gate": report["phase1_stat_gate"], "summary": report["summary"]}, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")
    print(f"saved dashboard: {a.html}")


if __name__ == "__main__":
    main()
