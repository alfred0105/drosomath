from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import (
    ConsolidationConfig,
    MemoryConsolidator,
    OutgoingBudgetNormalizer,
    PlasticStateConfig,
    ProtectedRewardRule,
    ReplayConfig,
    ReplayScheduler,
)

from .brain import PlasticMaleCNSBrain
from .checkpoint import (
    restore_learning_checkpoint,
    restore_readout_checkpoint,
    save_learning_checkpoint,
    save_readout_checkpoint,
)
from .curriculum_v1 import (
    CurriculumV1Config,
    _calibrate,
    _degrade,
    _evaluate,
    _readout_path,
    _train_decoder,
    build_curriculum,
)
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1
from .output_readout import OutputReadoutConfig, PopulationReadout
from .output_session import MaleCNSOutputSession, OutputSessionConfig


DEFAULT_RESULT = Path("results/latest_malecns_v2_memory.json")
DEFAULT_PROGRESS = Path("results/malecns_v2_memory_progress.json")
DEFAULT_CHECKPOINT = Path("checkpoints/malecns_v2_memory_brain.npz")
DEFAULT_READOUT_DIR = Path("checkpoints/malecns_v2_memory_readouts")
DEFAULT_V1_BASELINE = Path("results/latest_malecns_v1_curriculum.json")

Observer = Callable[[dict[str, object], PlasticMaleCNSBrain], None]


@dataclass(frozen=True, slots=True)
class MemoryV2Config(CurriculumV1Config):
    replay_interval: int = 4
    replay_fraction: float = 0.80
    replay_rate_hz: float = 230.0
    replay_max_prior_tasks: int = 3
    positive_protection: float = 0.35
    negative_protection: float = 0.90
    stability_protection: float = 0.85
    consolidation_top_fraction: float = 0.15
    consolidation_boost: float = 0.30


def _make_memory_session(brain, output, task, config: MemoryV2Config, seed: int):
    readout = PopulationReadout(
        output,
        task.labels,
        config=OutputReadoutConfig(learning_rate=0.08, seed=seed),
    )
    session = MaleCNSOutputSession(
        brain,
        readout,
        reward_rule=ProtectedRewardRule(
            learning_rate=config.learning_rate,
            positive_protection=config.positive_protection,
            negative_protection=config.negative_protection,
        ),
        normalizer=OutgoingBudgetNormalizer(
            strength=config.budget_strength,
            stability_protection=config.stability_protection,
        ),
        config=OutputSessionConfig(
            correct_reward=1.0,
            incorrect_reward=-1.0,
            no_output_reward=config.silent_reward,
        ),
    )
    return readout, session


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_progress(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _chance(task) -> float:
    return 1.0 / len(task.labels)


def _phase1_gate(tasks, stage_reports, retention_history) -> dict[str, object]:
    if not stage_reports or not retention_history:
        return {"passed": False, "reason": "no_completed_stages", "tasks": {}}

    by_name = {row["stage"]: row for row in stage_reports}
    final_retention = retention_history[-1]["tasks"]
    checks: dict[str, object] = {}
    passed = True

    for task in tasks[:-1]:  # addition is the final interference source, not a retained prior task.
        stage = by_name.get(task.name)
        final = final_retention.get(task.name)
        if stage is None or final is None:
            checks[task.name] = {"passed": False, "reason": "missing_measurement"}
            passed = False
            continue

        immediate = float(stage["after_accuracy"])
        final_acc = float(final["accuracy"])
        chance = _chance(task)
        ratio = final_acc / immediate if immediate > 1e-9 else 0.0
        measurable = immediate >= chance + 0.05

        if task.name == "laterality":
            threshold = max(0.80, immediate * 0.80)
        elif task.name == "numerosity_1_4":
            threshold = max(0.30, immediate * 0.75)
        else:  # comparison
            threshold = max(0.40, immediate * 0.80)

        ok = measurable and final_acc >= threshold
        checks[task.name] = {
            "chance": chance,
            "immediate_accuracy": immediate,
            "final_retention_accuracy": final_acc,
            "retention_ratio": ratio,
            "required_accuracy": threshold,
            "measurable_learning": measurable,
            "passed": ok,
        }
        passed &= ok

    return {
        "passed": bool(passed),
        "criterion": "prior tasks must first exceed chance+5pp, then survive later tasks at configured retention thresholds",
        "tasks": checks,
    }


def _compare_v1(v1_report: dict[str, object] | None, v2_retention_history) -> dict[str, object] | None:
    if not v1_report or not v2_retention_history:
        return None
    v1_hist = v1_report.get("retention_history") or []
    if not v1_hist:
        return None
    v1_final = v1_hist[-1].get("tasks", {})
    v2_final = v2_retention_history[-1].get("tasks", {})
    rows = {}
    for name in sorted(set(v1_final) | set(v2_final)):
        a = v1_final.get(name)
        b = v2_final.get(name)
        if not a or not b:
            continue
        old = float(a["accuracy"])
        new = float(b["accuracy"])
        rows[name] = {
            "v1_accuracy": old,
            "v2_accuracy": new,
            "delta_accuracy": new - old,
            "v1_silent_fraction": float(a.get("silent_fraction", 0.0)),
            "v2_silent_fraction": float(b.get("silent_fraction", 0.0)),
        }
    return {"final_retention": rows}


def run_memory_curriculum(
    connectome,
    *,
    config: MemoryV2Config,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    readout_dir: Path = DEFAULT_READOUT_DIR,
    progress_path: Path = DEFAULT_PROGRESS,
    resume: bool = True,
    observer: Observer | None = None,
    live_telemetry: bool = False,
    v1_baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    np = __import__("numpy")
    tasks, output = build_curriculum(connectome, config=config)

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=config.seed),
    )
    if live_telemetry:
        brain.configure_live_telemetry(True, max_active_edges=128, edges_per_firing_neuron=4)

    resume_info = None
    if resume and checkpoint_path.is_file():
        resume_info = restore_learning_checkpoint(checkpoint_path, brain=brain)

    sessions = {}
    readouts = {}
    for i, task in enumerate(tasks):
        readout, session = _make_memory_session(brain, output, task, config, config.seed + 200 + i)
        rp = _readout_path(readout_dir, task.name)
        if resume and rp.is_file():
            restore_readout_checkpoint(rp, readout)
        readouts[task.name] = readout
        sessions[task.name] = session

    progress = _load_progress(progress_path) if resume else None
    stage_reports = list(progress.get("stages", [])) if progress else []
    retention_history = list(progress.get("retention_history", [])) if progress else []
    task_conditions = dict(progress.get("task_conditions", {})) if progress else {}
    consolidation_history = list(progress.get("consolidation_history", [])) if progress else []
    replay_history = list(progress.get("replay_history", [])) if progress else []

    start_stage = len(stage_reports)
    completed_in_stage = 0
    if resume_info:
        saved_stage = str(resume_info.get("stage", ""))
        if saved_stage == "complete":
            start_stage = len(tasks)
        else:
            for i, task in enumerate(tasks):
                if task.name == saved_stage:
                    start_stage = i
                    completed_in_stage = int(resume_info.get("completed_trials", 0))
                    if completed_in_stage >= config.stage_trials:
                        start_stage = max(start_stage, len(stage_reports))
                        completed_in_stage = 0
                    break

    consolidator = MemoryConsolidator(
        ConsolidationConfig(
            top_fraction=config.consolidation_top_fraction,
            boost=config.consolidation_boost,
        )
    )
    replay = ReplayScheduler(
        ReplayConfig(
            interval=config.replay_interval,
            fraction=config.replay_fraction,
            rate_hz=config.replay_rate_hz,
            max_prior_tasks=config.replay_max_prior_tasks,
        )
    )

    def emit(**event: object) -> None:
        if observer is not None:
            observer(event, brain)

    for stage_index in range(start_stage, len(tasks)):
        task = tasks[stage_index]
        session = sessions[task.name]
        readout = readouts[task.name]
        unlock = brain.plasticity.set_plastic_fraction(task.plastic_fraction)

        emit(
            phase=task.name,
            status="decoder",
            stage_index=stage_index + 1,
            total_stages=len(tasks),
            completed_brain_trials=0,
            total_brain_trials=config.stage_trials,
            message=f"Memory v2 stage {stage_index+1}/{len(tasks)}: {task.name}",
        )

        stage_rng = np.random.default_rng(config.seed + 10_000 + stage_index)
        if not readout.frozen:
            decoder_report = _train_decoder(session, task, config, stage_rng)
            save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        else:
            decoder_report = {"restored": True, "steps": readout.train_steps}

        emit(phase=task.name, status="calibration", stage_index=stage_index + 1, total_stages=len(tasks))
        if task.name in task_conditions:
            fraction = float(task_conditions[task.name]["fraction"])
            rate_hz = float(task_conditions[task.name]["rate_hz"])
            calibration = task_conditions[task.name].get("calibration", [])
        else:
            fraction, rate_hz, calibration = _calibrate(
                session, task, config, config.seed + 20_000 + stage_index * 100
            )
            task_conditions[task.name] = {
                "fraction": fraction,
                "rate_hz": rate_hz,
                "calibration": calibration,
            }

        before_acc, before_silent, before_spikes, _ = _evaluate(
            session,
            task,
            config,
            fraction=fraction,
            rate_hz=rate_hz,
            seed=config.seed + 30_000 + stage_index * 100,
        )

        start_trial = completed_in_stage if stage_index == start_stage else 0
        rows = []
        correct = 0
        silent_count = 0
        replay_rows = []

        for trial in range(start_trial, config.stage_trials):
            trial_rng = np.random.default_rng(config.seed + stage_index * 1_000_000 + trial)
            label = task.labels[int(trial_rng.integers(0, len(task.labels)))]
            stimulus = _degrade(trial_rng, task.sample(label, trial_rng), fraction)
            result = session.train_brain_trial(
                stimulus_body_ids=stimulus,
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=rate_hz,
            )
            correct += int(result.correct)
            silent_count += int(result.total_output_spikes == 0)
            row = {
                "trial": trial + 1,
                "target": result.target,
                "prediction": result.prediction,
                "correct": result.correct,
                "confidence": result.confidence,
                "reward": result.reward,
                "output_spikes": result.total_output_spikes,
                "edge_updates": result.learning["learning"]["edge_updates"],
            }
            rows.append(row)

            if replay.should_replay(trial + 1, stage_index):
                prior_index = replay.choose_prior_index(trial_rng, stage_index)
                prior = tasks[prior_index]
                prior_session = sessions[prior.name]
                prior_label = prior.labels[int(trial_rng.integers(0, len(prior.labels)))]
                prior_stimulus = _degrade(
                    trial_rng,
                    prior.sample(prior_label, trial_rng),
                    replay.config.fraction,
                )
                replay_result = prior_session.train_brain_trial(
                    stimulus_body_ids=prior_stimulus,
                    target=prior_label,
                    duration_ms=config.duration_ms,
                    stimulus_rate_hz=replay.config.rate_hz,
                )
                replay_rows.append(
                    {
                        "at_current_trial": trial + 1,
                        "task": prior.name,
                        "target": replay_result.target,
                        "prediction": replay_result.prediction,
                        "correct": replay_result.correct,
                        "silent": replay_result.total_output_spikes == 0,
                        "edge_updates": replay_result.learning["learning"]["edge_updates"],
                    }
                )

            emit(
                phase=task.name,
                status="training",
                stage_index=stage_index + 1,
                total_stages=len(tasks),
                completed_brain_trials=trial + 1,
                total_brain_trials=config.stage_trials,
                running_accuracy=correct / max(1, len(rows)),
                running_silent_fraction=silent_count / max(1, len(rows)),
                replay_count=len(replay_rows),
                selected_input_fraction=fraction,
                selected_stimulus_rate_hz=rate_hz,
                before_accuracy=before_acc,
                last_trial=row,
            )

            if config.checkpoint_every and (trial + 1) % config.checkpoint_every == 0:
                save_learning_checkpoint(
                    checkpoint_path,
                    brain=brain,
                    readout=readout,
                    config=config,
                    completed_trials=trial + 1,
                    stage=task.name,
                )
                save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
                _write_progress(
                    progress_path,
                    {
                        "experiment": "malecns_memory_v2_progress",
                        "stages": stage_reports,
                        "retention_history": retention_history,
                        "task_conditions": task_conditions,
                        "consolidation_history": consolidation_history,
                        "replay_history": replay_history,
                        "current_stage": task.name,
                        "completed_trials": trial + 1,
                    },
                )
                print(
                    f"[{task.name}] {trial+1}/{config.stage_trials} "
                    f"acc={correct/max(1,len(rows)):.3f} "
                    f"silent={silent_count/max(1,len(rows)):.3f} replay={len(replay_rows)}"
                )

        emit(phase=task.name, status="consolidating", stage_index=stage_index + 1, total_stages=len(tasks))
        consolidation = consolidator.consolidate(brain.plasticity)
        consolidation_history.append({"after_stage": task.name, **consolidation})

        after_acc, after_silent, after_spikes, _ = _evaluate(
            session,
            task,
            config,
            fraction=fraction,
            rate_hz=rate_hz,
            seed=config.seed + 40_000 + stage_index * 100,
        )

        save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        save_learning_checkpoint(
            checkpoint_path,
            brain=brain,
            readout=readout,
            config=config,
            completed_trials=config.stage_trials,
            stage=task.name,
        )

        replay_correct = sum(int(x["correct"]) for x in replay_rows)
        replay_silent = sum(int(x["silent"]) for x in replay_rows)
        replay_report = {
            "stage": task.name,
            "count": len(replay_rows),
            "accuracy": replay_correct / len(replay_rows) if replay_rows else None,
            "silent_fraction": replay_silent / len(replay_rows) if replay_rows else None,
            "last_rows": replay_rows[-12:],
        }
        replay_history.append(replay_report)

        stage_reports.append(
            {
                "stage": task.name,
                "labels": list(task.labels),
                "plasticity_unlock": unlock,
                "decoder": decoder_report,
                "challenge": {"fraction": fraction, "rate_hz": rate_hz, "calibration": calibration},
                "before_accuracy": before_acc,
                "after_accuracy": after_acc,
                "delta_accuracy": after_acc - before_acc,
                "before_silent_fraction": before_silent,
                "after_silent_fraction": after_silent,
                "before_mean_spikes": before_spikes,
                "after_mean_spikes": after_spikes,
                "training_accuracy": correct / max(1, len(rows)),
                "training_silent_fraction": silent_count / max(1, len(rows)),
                "replay": replay_report,
                "consolidation": consolidation,
                "last_rows": rows[-16:],
            }
        )

        emit(phase=task.name, status="retention", stage_index=stage_index + 1, total_stages=len(tasks))
        retention = {"after_stage": task.name, "tasks": {}}
        for prior_index, prior in enumerate(tasks[: stage_index + 1]):
            acc, silent, spikes, _ = _evaluate(
                sessions[prior.name],
                prior,
                config,
                fraction=0.70,
                rate_hz=205.0,
                seed=config.seed + 50_000 + stage_index * 100 + prior_index,
            )
            retention["tasks"][prior.name] = {
                "accuracy": acc,
                "silent_fraction": silent,
                "mean_output_spikes": spikes,
            }
        retention_history.append(retention)
        completed_in_stage = 0

        _write_progress(
            progress_path,
            {
                "experiment": "malecns_memory_v2_progress",
                "stages": stage_reports,
                "retention_history": retention_history,
                "task_conditions": task_conditions,
                "consolidation_history": consolidation_history,
                "replay_history": replay_history,
                "current_stage": task.name,
                "completed_trials": config.stage_trials,
            },
        )

    save_learning_checkpoint(
        checkpoint_path,
        brain=brain,
        config=config,
        completed_trials=0,
        stage="complete",
    )

    plast = brain.plasticity.summary()
    changed = int(np.count_nonzero(np.abs(brain.plasticity.multiplier - 1.0) > 1e-7))
    gate = _phase1_gate(tasks, stage_reports, retention_history)
    result = {
        "experiment": "malecns_memory_v2",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "resume": resume_info,
        "stages": stage_reports,
        "retention_history": retention_history,
        "consolidation_history": consolidation_history,
        "replay_history": replay_history,
        "task_conditions": task_conditions,
        "phase1_gate": gate,
        "v1_comparison": _compare_v1(v1_baseline, retention_history),
        "final_plasticity": {**plast, "changed_edges": changed},
        "checkpoint": str(checkpoint_path),
        "readout_dir": str(readout_dir),
    }
    _write_progress(progress_path, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Run DrosoMath Phase-1 memory consolidation curriculum")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--stage-trials", type=int, default=256)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--validation-trials", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--replay-interval", type=int, default=4)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    p.add_argument("--v1-baseline", type=Path, default=DEFAULT_V1_BASELINE)
    a = p.parse_args()

    if a.download:
        download_malecns(a.data_dir)
    connectome = load_malecns_v1(a.data_dir, min_connection_synapses=a.min_syn)
    config = MemoryV2Config(
        min_connection_synapses=a.min_syn,
        stage_trials=a.stage_trials,
        decoder_epochs=a.decoder_epochs,
        validation_trials_per_label=a.validation_trials,
        checkpoint_every=a.checkpoint_every,
        seed=a.seed,
        replay_interval=a.replay_interval,
    )
    baseline = None
    if a.v1_baseline.is_file():
        baseline = json.loads(a.v1_baseline.read_text(encoding="utf-8"))

    if a.fresh:
        for path in (a.checkpoint, a.progress):
            if path.is_file():
                path.unlink()

    report = run_memory_curriculum(
        connectome,
        config=config,
        checkpoint_path=a.checkpoint,
        readout_dir=a.readout_dir,
        progress_path=a.progress,
        resume=not a.fresh,
        v1_baseline=baseline,
    )
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "phase1_gate": report["phase1_gate"],
        "v1_comparison": report["v1_comparison"],
        "final_plasticity": report["final_plasticity"],
    }, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")


if __name__ == "__main__":
    main()
