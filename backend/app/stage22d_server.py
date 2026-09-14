from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .flywire import FAFB_V783_TOTAL_NEURONS, load_fafb_soma_layout
from .numerosity_stage22d import (
    STAGE22D_PROFILES,
    Stage22DExperiment,
)
from .run_logging import RunLogger


app = FastAPI(title="DrosoMath telemetry API", version="0.12.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIALS_PER_UI_FRAME = 12
UI_INTERVAL_SECONDS = 0.24
PROBE_EVERY = 10
STAGE22D_TRAIN_TRIALS = 60_000
TRAINING_MILESTONES = (20_000, 40_000, 60_000)
PROFILE_TRIALS = 2_000
PROFILE_ORDER = list(STAGE22D_PROFILES)
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
STAGE22D_CHECKPOINT = CHECKPOINT_DIR / "stage2_2d_seed7_t60000_v1.npz"


def _training_profile(trial: int) -> str:
    if trial <= 20_000:
        return "train_brightness_weak"
    if trial <= 40_000:
        return "train_brightness_medium"
    return "train_brightness_strong"


def _milestone_checkpoint(trial: int) -> Path:
    return CHECKPOINT_DIR / f"stage2_2d_seed7_t{trial}_v1.npz"


def make_mock_layout() -> dict[str, Any]:
    rng = random.Random(42)
    neurons = [
        {
            "id": i,
            "root_id": None,
            "x": rng.gauss(0.0, 1.9),
            "y": rng.gauss(0.0, 1.2),
            "z": rng.gauss(0.0, 0.9),
            "region": ("visual", "mushroom_body", "central", "dopamine")[i % 4],
            "super_class": "mock",
            "side": "",
        }
        for i in range(800)
    ]
    return {
        "neurons": neurons,
        "source": "mock geometry — run scripts/download_fafb783.py for real FlyWire coordinates",
        "count": len(neurons),
        "total_connectome_neurons": FAFB_V783_TOTAL_NEURONS,
        "coordinate_kind": "mock",
    }


LAYOUT = load_fafb_soma_layout() or make_mock_layout()
NEURONS = LAYOUT["neurons"]
NEURON_COUNT = len(NEURONS)

VISUAL_POOL = [
    i for i, neuron in enumerate(NEURONS)
    if neuron.get("region") in {"optic", "sensory", "visual_projection", "visual_centrifugal", "visual"}
]
CENTRAL_POOL = [
    i for i, neuron in enumerate(NEURONS)
    if neuron.get("region") in {"central", "mushroom_body"}
]
OUTPUT_POOL = [
    i for i, neuron in enumerate(NEURONS)
    if neuron.get("region") in {"descending", "motor", "ascending"}
]
if not VISUAL_POOL:
    VISUAL_POOL = list(range(0, NEURON_COUNT, 3))
if not CENTRAL_POOL:
    CENTRAL_POOL = list(range(1, NEURON_COUNT, 3))
if not OUTPUT_POOL:
    OUTPUT_POOL = list(range(2, NEURON_COUNT, 3))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "layout_source": LAYOUT["source"],
        "layout_count": NEURON_COUNT,
        "telemetry_source": "numerosity_prototype",
        "experiment": "numerosity_0_2_stage2_2d",
        "mode": "60k_progressive_brightness_curriculum_then_frozen_ood",
        "stage22d_training_trials": STAGE22D_TRAIN_TRIALS,
        "training_milestones": list(TRAINING_MILESTONES),
        "profile_trials": PROFILE_TRIALS,
        "profiles": PROFILE_ORDER,
    }


@app.get("/api/layout")
def layout() -> dict[str, Any]:
    return LAYOUT


class SuccessMetrics:
    def __init__(self) -> None:
        self.total_attempts = 0
        self.total_successes = 0
        self.outcomes: deque[int] = deque(maxlen=500)
        self.target_attempts = [0, 0, 0]
        self.target_successes = [0, 0, 0]
        self.confusion = [[0, 0, 0] for _ in range(3)]
        self.probe_attempts = 0
        self.probe_successes = 0
        self.probe_outcomes: deque[int] = deque(maxlen=500)
        self.probe_target_attempts = [0, 0, 0]
        self.probe_target_successes = [0, 0, 0]
        self.probe_confusion = [[0, 0, 0] for _ in range(3)]

    def record(self, correct: bool, target: int, choice: int, *, is_probe: bool) -> None:
        outcome = int(correct)
        self.total_attempts += 1
        self.total_successes += outcome
        self.outcomes.append(outcome)
        self.target_attempts[target] += 1
        self.target_successes[target] += outcome
        self.confusion[target][choice] += 1
        if is_probe:
            self.probe_attempts += 1
            self.probe_successes += outcome
            self.probe_outcomes.append(outcome)
            self.probe_target_attempts[target] += 1
            self.probe_target_successes[target] += outcome
            self.probe_confusion[target][choice] += 1

    @staticmethod
    def _rate(values: deque[int], window: int) -> float | None:
        if not values:
            return None
        recent = list(values)[-window:]
        return sum(recent) / len(recent)

    @staticmethod
    def _class_rates(successes: list[int], attempts: list[int]) -> dict[str, float | None]:
        return {
            str(i): successes[i] / attempts[i] if attempts[i] else None
            for i in range(3)
        }

    @staticmethod
    def _balanced(successes: list[int], attempts: list[int]) -> float | None:
        rates = [successes[i] / attempts[i] for i in range(3) if attempts[i]]
        return sum(rates) / len(rates) if rates else None

    @staticmethod
    def _one_two(confusion: list[list[int]]) -> float | None:
        attempts = sum(confusion[1]) + sum(confusion[2])
        return (confusion[1][1] + confusion[2][2]) / attempts if attempts else None

    def snapshot(self) -> dict[str, Any]:
        overall = self.total_successes / self.total_attempts if self.total_attempts else None
        probe_overall = self.probe_successes / self.probe_attempts if self.probe_attempts else None
        return {
            "overall": overall,
            "balanced_accuracy": self._balanced(self.target_successes, self.target_attempts),
            "one_vs_two_accuracy": self._one_two(self.confusion),
            "recent_20": self._rate(self.outcomes, 20),
            "recent_100": self._rate(self.outcomes, 100),
            "recent_500": self._rate(self.outcomes, 500),
            "successes": self.total_successes,
            "attempts": self.total_attempts,
            "by_target_accuracy": self._class_rates(self.target_successes, self.target_attempts),
            "confusion_matrix": [row[:] for row in self.confusion],
            "probe": {
                "overall": probe_overall,
                "balanced_accuracy": self._balanced(self.probe_target_successes, self.probe_target_attempts),
                "one_vs_two_accuracy": self._one_two(self.probe_confusion),
                "recent_20": self._rate(self.probe_outcomes, 20),
                "recent_100": self._rate(self.probe_outcomes, 100),
                "recent_500": self._rate(self.probe_outcomes, 500),
                "successes": self.probe_successes,
                "attempts": self.probe_attempts,
                "by_target_accuracy": self._class_rates(self.probe_target_successes, self.probe_target_attempts),
                "confusion_matrix": [row[:] for row in self.probe_confusion],
            },
        }


def prepare_stage22d_experiment() -> tuple[Stage22DExperiment, dict[str, Any], dict[str, Any], str]:
    experiment = Stage22DExperiment()
    if STAGE22D_CHECKPOINT.exists():
        metadata = experiment.load_checkpoint(STAGE22D_CHECKPOINT)
        return (
            experiment,
            dict(metadata.get("pretraining_snapshot") or {}),
            dict(metadata.get("training_snapshots") or {}),
            "checkpoint",
        )

    metrics = SuccessMetrics()
    snapshots: dict[str, Any] = {}
    for trial in range(1, STAGE22D_TRAIN_TRIALS + 1):
        profile = _training_profile(trial)
        is_probe = trial % PROBE_EVERY == 0
        result = experiment.step(
            trial,
            learn=not is_probe,
            trial_kind="probe" if is_probe else "train",
            stimulus_profile=profile,
        )
        metrics.record(
            bool(result["correct"]),
            int(result["target"]),
            int(result["choice"]),
            is_probe=is_probe,
        )

        if trial in TRAINING_MILESTONES:
            snapshot = metrics.snapshot()
            snapshot["curriculum_phase"] = Stage22DExperiment.curriculum_phase(profile)
            snapshots[str(trial)] = snapshot
            experiment.save_checkpoint(
                _milestone_checkpoint(trial),
                {
                    "stage": "2.2D_pretraining",
                    "pretraining_trials": trial,
                    "target_training_trials": STAGE22D_TRAIN_TRIALS,
                    "probe_every": PROBE_EVERY,
                    "pretraining_snapshot": snapshot,
                    "training_snapshots": dict(snapshots),
                    "learner": experiment.config_dict(),
                    "purpose": "progressive brightness curriculum before frozen Stage-2.2D OOD evaluation",
                },
            )

    return (
        experiment,
        snapshots[str(STAGE22D_TRAIN_TRIALS)],
        snapshots,
        "deterministic_60k_curriculum_training",
    )


def _profile_results(profile_metrics: dict[str, SuccessMetrics]) -> dict[str, Any]:
    return {
        profile: metrics.snapshot() if metrics.total_attempts else None
        for profile, metrics in profile_metrics.items()
    }


def _add_pool_activity(
    activity: dict[int, float], pool: list[int], start: int, count: int, value: float, step: int
) -> None:
    if not pool:
        return
    for n in range(min(count, len(pool))):
        index = pool[(start + n * step) % len(pool)]
        activity[index] = max(activity.get(index, 0.0), min(1.0, value))


def make_display_activity(result: dict[str, Any]) -> list[list[float | int]]:
    """Display proxy only; not a prediction of FlyWire spikes."""
    activity: dict[int, float] = {}
    trial = int(result["trial"])
    for dot_index, dot in enumerate(result["stimulus"]["dots"]):
        key = int(dot["x"] * 10007 + dot["y"] * 20011 + trial * 17 + dot_index * 97)
        _add_pool_activity(activity, VISUAL_POOL, key, 55, 0.88, 173)
    for kc_index, kc_value in result["kc_activity"]:
        if not CENTRAL_POOL:
            break
        mapped = CENTRAL_POOL[(int(kc_index) * 7919 + trial * 13) % len(CENTRAL_POOL)]
        activity[mapped] = max(activity.get(mapped, 0.0), 0.35 + 0.65 * float(kc_value))
    probabilities = [
        float(result["policy"]["p0"]),
        float(result["policy"]["p1"]),
        float(result["policy"]["p2"]),
    ]
    for action, probability in enumerate(probabilities):
        intensity = 0.30 + 0.62 * probability + (0.08 if action == result["choice"] else 0.0)
        _add_pool_activity(activity, OUTPUT_POOL, action * 997 + trial * 31, 35, intensity, 149)
    return [[index, round(value, 3)] for index, value in activity.items()]


def make_frame(result: dict[str, Any], *, profile: str, profile_trial: int, profile_index: int) -> dict[str, Any]:
    return {
        "type": "telemetry",
        "telemetry_source": "numerosity_prototype",
        "activity_source": "display_proxy_not_connectome_spikes",
        "learning_model": "reward_modulated_sparse_associator",
        "phase": "dots_0_2_stage2_2d_60k_brightness_curriculum_frozen_ood",
        "trial_kind": result["trial_kind"],
        "learning_enabled": result["learning_enabled"],
        "timestamp": time.time(),
        "trial": result["trial"],
        "problem": "How many dots?",
        "target": int(result["target"]),
        "answer": int(result["choice"]),
        "correct": bool(result["correct"]),
        "reward": float(result["reward"]),
        "evaluation_profile": profile,
        "profile_trial": profile_trial,
        "profile_trials_target": PROFILE_TRIALS,
        "profile_index": profile_index,
        "profile_count": len(PROFILE_ORDER),
        "stimulus": result["stimulus"],
        "policy": result["policy"],
        "activity": make_display_activity(result),
        "plasticity": result["plasticity"],
    }


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()

    experiment, pretraining_snapshot, training_snapshots, state_source = await asyncio.to_thread(
        prepare_stage22d_experiment
    )
    profile_metrics = {profile: SuccessMetrics() for profile in PROFILE_ORDER}
    eval_trial = 1

    learner_config = experiment.config_dict()
    learner_config["training_trials"] = STAGE22D_TRAIN_TRIALS
    learner_config["training_milestones"] = list(TRAINING_MILESTONES)
    learner_config["training_snapshots"] = training_snapshots

    logger = RunLogger(
        {
            "experiment": "numerosity_0_2_stage2_2d",
            "telemetry_source": "numerosity_prototype",
            "activity_source": "display_proxy_not_connectome_spikes",
            "learning_model": "reward_modulated_sparse_associator",
            "chance_accuracy": 1.0 / 3.0,
            "classes": [0, 1, 2],
            "layout_source": LAYOUT["source"],
            "layout_count": NEURON_COUNT,
            "total_connectome_neurons": FAFB_V783_TOTAL_NEURONS,
            "rolling_windows": [20, 100, 500],
            "trials_per_ui_frame": TRIALS_PER_UI_FRAME,
            "ui_interval_seconds": UI_INTERVAL_SECONDS,
            "stage22d_training_trials": STAGE22D_TRAIN_TRIALS,
            "stage22d_state_source": state_source,
            "stage22d_pretraining_snapshot": pretraining_snapshot,
            "stage22d_training_snapshots": training_snapshots,
            "evaluation_profile": "progressive_brightness_frozen_retest",
            "evaluation_profiles": PROFILE_ORDER,
            "profile_trials": PROFILE_TRIALS,
            "total_evaluation_trials": PROFILE_TRIALS * len(PROFILE_ORDER),
            "evaluation_learning_enabled": False,
            "learner": learner_config,
            "scientific_scope": (
                "Stage-2.2D preserves the Stage-2.2C visual encoder and nuisance controls while progressively raising "
                "two-dot brightness asymmetry over three 20k learning phases. Signal-energy and area distributions remain "
                "matched across 1 vs 2. Frozen OOD evaluation uses Position, stronger-Brightness, and Combined shifts with "
                "zero plastic updates. No explicit object, peak, component, or numerosity count is injected. FlyWire "
                "coordinates remain visualization-only; activity is a display proxy rather than connectome spiking."
            ),
        }
    )

    status = "completed"
    finalized = False
    last_frame: dict[str, Any] | None = None
    try:
        for profile_index, profile in enumerate(PROFILE_ORDER, start=1):
            metrics = profile_metrics[profile]
            for profile_trial in range(1, PROFILE_TRIALS + 1):
                result = experiment.step(
                    eval_trial,
                    learn=False,
                    trial_kind="stage22d_ood_eval",
                    stimulus_profile=profile,
                )
                frame = make_frame(
                    result,
                    profile=profile,
                    profile_trial=profile_trial,
                    profile_index=profile_index,
                )
                metrics.record(
                    bool(frame["correct"]),
                    int(frame["target"]),
                    int(frame["answer"]),
                    is_probe=True,
                )
                snapshot = metrics.snapshot()
                snapshot["current_profile"] = profile
                snapshot["profile_trial"] = profile_trial
                snapshot["profile_trials_target"] = PROFILE_TRIALS
                snapshot["profiles"] = _profile_results(profile_metrics)
                snapshot["experiment_complete"] = False
                frame["metrics"] = snapshot
                frame["accuracy"] = snapshot["overall"]
                logger.record(frame, snapshot)
                last_frame = frame
                eval_trial += 1

                if profile_trial % TRIALS_PER_UI_FRAME == 0 or profile_trial == PROFILE_TRIALS:
                    await websocket.send_json(frame)
                    await asyncio.sleep(UI_INTERVAL_SECONDS)

        if last_frame is not None:
            final_snapshot = dict(last_frame["metrics"])
            final_snapshot["profiles"] = _profile_results(profile_metrics)
            final_snapshot["experiment_complete"] = True
            final_snapshot["current_profile"] = PROFILE_ORDER[-1]
            final_snapshot["profile_trial"] = PROFILE_TRIALS
            last_frame["metrics"] = final_snapshot
            last_frame["experiment_complete"] = True
            logger.finalize(status="completed", frame=last_frame, metrics=final_snapshot)
            finalized = True
            await websocket.send_json(last_frame)
        else:
            logger.finalize(status="completed")
            finalized = True

        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        status = "completed" if finalized else "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        if not finalized:
            logger.finalize(status=status)
