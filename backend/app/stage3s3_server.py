from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .run_logging import RunLogger
from .stage3s2_server import LAYOUT, NEURON_COUNT, Metrics, make_display_activity
from .stage3s3_curriculum import (
    PROFILE_ORDER, PROFILE_TRIALS, TASK_NAMES, TRAIN_TRIALS, TRAINING_MILESTONES,
    profile_example, training_spec,
)
from .symbolic_stage3s3 import SymbolicStage3S3Experiment

app = FastAPI(title="DrosoMath telemetry API", version="0.16.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PROBE_EVERY = 40
TRIALS_PER_UI_FRAME = 20
UI_INTERVAL_SECONDS = 0.16
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "stage3s3_seed67_t600000_v1.npz"


def _milestone_checkpoint(trial: int) -> Path:
    return CHECKPOINT_DIR / f"stage3s3_seed67_t{trial}_v1.npz"


def prepare_experiment() -> tuple[SymbolicStage3S3Experiment, dict[str, Any], dict[str, Any], str]:
    exp = SymbolicStage3S3Experiment()
    if FINAL_CHECKPOINT.exists():
        meta = exp.load_checkpoint(FINAL_CHECKPOINT)
        return exp, dict(meta.get("training_snapshot") or {}), dict(meta.get("training_snapshots") or {}), "checkpoint"

    overall = Metrics()
    meters = {task: Metrics() for task in TASK_NAMES}
    snapshots: dict[str, Any] = {}
    for trial in range(1, TRAIN_TRIALS + 1):
        spec = training_spec(exp, trial)
        task = str(spec["task"])
        probe = trial % PROBE_EVERY == 0
        result = exp.step(
            trial, task=task, learn=not probe, a=spec.get("a"), pair_pool=spec.get("pair_pool"),
            token_plasticity_scale=float(spec["token"]),
            recurrent_plasticity_scale=float(spec["recurrent"]),
            operator_plasticity_scale=float(spec["operator"]),
            output_plasticity_scale=float(spec["output"]),
            trial_kind="probe" if probe else "train",
        )
        overall.record(int(result["target"]), int(result["choice"]))
        meters[task].record(int(result["target"]), int(result["choice"]))
        if trial in TRAINING_MILESTONES:
            snap = overall.snapshot()
            snap["curriculum_phase"] = spec["phase"]
            snap["tasks"] = {name: meter.snapshot() for name, meter in meters.items()}
            snap["plasticity_scales"] = result["plasticity_scales"]
            snapshots[str(trial)] = snap
            exp.save_checkpoint(_milestone_checkpoint(trial), {
                "stage": "3S.3_symbolic_pretraining", "training_trials": trial,
                "target_training_trials": TRAIN_TRIALS, "training_snapshot": snap,
                "training_snapshots": dict(snapshots), "learner": exp.config_dict(),
            })

    final = snapshots[str(TRAIN_TRIALS)]
    exp.save_checkpoint(FINAL_CHECKPOINT, {
        "stage": "3S.3_symbolic_pretraining", "training_trials": TRAIN_TRIALS,
        "target_training_trials": TRAIN_TRIALS, "training_snapshot": final,
        "training_snapshots": snapshots, "learner": exp.config_dict(),
    })
    return exp, final, snapshots, "deterministic_600k_stage3s3_curriculum"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok", "experiment": "symbolic_stage3s3",
        "mode": "600k_operator_transition_scaffold_then_frozen_generalization",
        "training_trials": TRAIN_TRIALS, "training_milestones": list(TRAINING_MILESTONES),
        "profiles": list(PROFILE_ORDER), "profile_trials": PROFILE_TRIALS,
        "addition_chance_accuracy": 1 / 19, "digit_task_chance_accuracy": .1,
    }


@app.get("/api/layout")
def layout() -> dict[str, Any]:
    return LAYOUT


def _profile_snapshots(meters: dict[str, Metrics]) -> dict[str, Any]:
    return {name: meter.snapshot() if meter.attempts else None for name, meter in meters.items()}


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    exp, training_snapshot, training_snapshots, source = await asyncio.to_thread(prepare_experiment)
    meters = {profile: Metrics() for profile in PROFILE_ORDER}
    learner = exp.config_dict() | {
        "training_trials": TRAIN_TRIALS, "training_milestones": list(TRAINING_MILESTONES),
        "training_snapshot": training_snapshot, "training_snapshots": training_snapshots,
        "state_source": source,
    }
    logger = RunLogger({
        "experiment": "symbolic_stage3s3", "telemetry_source": "symbolic_math_prototype",
        "activity_source": "display_proxy_not_connectome_spikes",
        "learning_model": "shared_operator_transition_reward_gated_recurrent_associator",
        "chance_accuracy": 1 / 19, "layout_source": LAYOUT["source"], "layout_count": NEURON_COUNT,
        "evaluation_profile": "operator_composition_and_symbolic_arithmetic_frozen",
        "evaluation_profiles": list(PROFILE_ORDER), "profile_trials": PROFILE_TRIALS,
        "total_evaluation_trials": PROFILE_TRIALS * len(PROFILE_ORDER),
        "evaluation_learning_enabled": False, "learner": learner,
        "scientific_scope": (
            "Arbitrary symbol codes; NEXT/PREV have random-initialized shared plastic transition matrices. "
            "Guided intermediate digit symbols exist only in scaffold training. Frozen composition tests are autonomous; "
            "FlyWire coordinates are visualization-only."
        ),
    })
    eval_trial = 1
    last_frame = last_metrics = None
    finalized = False
    status = "completed"
    try:
        for profile_index, profile in enumerate(PROFILE_ORDER, 1):
            meter = meters[profile]
            for profile_trial in range(1, PROFILE_TRIALS + 1):
                task, a, b = profile_example(profile, profile_trial - 1)
                result = exp.step(eval_trial, task=task, learn=False, a=a, b=b, trial_kind="stage3s3_frozen_eval")
                meter.record(int(result["target"]), int(result["choice"]))
                metrics = meter.snapshot() | {
                    "profiles": _profile_snapshots(meters), "current_profile": profile,
                    "profile_trial": profile_trial, "profile_trials_target": PROFILE_TRIALS,
                    "experiment_complete": False,
                }
                frame = {
                    "type": "telemetry", "telemetry_source": "symbolic_math_prototype",
                    "activity_source": "display_proxy_not_connectome_spikes",
                    "phase": "stage3s3_operator_composition_frozen", "trial_kind": result["trial_kind"],
                    "learning_enabled": False, "timestamp": time.time(), "trial": eval_trial,
                    "evaluation_profile": profile, "profile_trial": profile_trial,
                    "profile_trials_target": PROFILE_TRIALS, "profile_index": profile_index,
                    "profile_count": len(PROFILE_ORDER), "task": task, "tokens": result["tokens"],
                    "expression": result["expression"], "target": int(result["target"]),
                    "answer": int(result["choice"]), "correct": bool(result["correct"]),
                    "reward": float(result["reward"]), "active_output_count": int(result["active_output_count"]),
                    "accuracy": metrics["overall"], "metrics": metrics, "policy": result["policy"],
                    "activity": make_display_activity(result), "plasticity": result["plasticity"],
                    "stimulus": {"kind": "symbol_sequence", "controls": {}},
                }
                logger.record(frame, metrics)
                last_frame, last_metrics = frame, metrics
                eval_trial += 1
                if profile_trial % TRIALS_PER_UI_FRAME == 0:
                    await websocket.send_json(frame)
                    await asyncio.sleep(UI_INTERVAL_SECONDS)

        if last_frame is not None and last_metrics is not None:
            final_metrics = dict(last_metrics)
            final_metrics["profiles"] = _profile_snapshots(meters)
            final_metrics["experiment_complete"] = True
            final_frame = dict(last_frame)
            final_frame["metrics"] = final_metrics
            final_frame["experiment_complete"] = True
            await websocket.send_json(final_frame)
            logger.finalize(status="completed", frame=final_frame, metrics=final_metrics)
            finalized = True
    except WebSocketDisconnect:
        status = "interrupted"
    except Exception:
        status = "error"
        raise
    finally:
        if not finalized:
            logger.finalize(status=status, frame=last_frame, metrics=last_metrics)
