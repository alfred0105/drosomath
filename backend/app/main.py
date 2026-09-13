import asyncio
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .flywire import FAFB_V783_TOTAL_NEURONS, load_fafb_soma_layout
from .numerosity import NumerosityExperiment, STAGE22B_PROFILES
from .run_logging import RunLogger

app = FastAPI(title="DrosoMath telemetry API", version="0.10.0")

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
STAGE22B_REPLAY_TRIALS = 10_000
PROFILE_TRIALS = 1_500
PROFILE_ORDER = list(STAGE22B_PROFILES)
REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE22B_CHECKPOINT = REPO_ROOT / "checkpoints" / "stage2_2b_seed7_t10000_v1.npz"


def make_mock_layout() -> dict[str, Any]:
    neuron_count = 800
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
        for i in range(neuron_count)
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
    i
    for i, neuron in enumerate(NEURONS)
    if neuron.get("region") in {"optic", "sensory", "visual_projection", "visual_centrifugal", "visual"}
]
CENTRAL_POOL = [
    i
    for i, neuron in enumerate(NEURONS)
    if neuron.get("region") in {"central", "mushroom_body"}
]
OUTPUT_POOL = [
    i
    for i, neuron in enumerate(NEURONS)
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
        "experiment": "numerosity_0_2_stage2_2b",
        "mode": "robust_encoder_replay_then_frozen_ood",
        "stage22b_replay_trials": STAGE22B_REPLAY_TRIALS,
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
        outcome = 1 if correct else 0
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
            str(target): successes[target] / attempts[target] if attempts[target] else None
            for target in range(3)
        }

    @staticmethod
    def _balanced_accuracy(successes: list[int], attempts: list[int]) -> float | None:
        rates = [
            successes[target] / attempts[target]
            for target in range(3)
            if attempts[target]
        ]
        return sum(rates) / len(rates) if rates else None

    @staticmethod
    def _one_vs_two_accuracy(confusion: list[list[int]]) -> float | None:
        attempts = sum(confusion[1]) + sum(confusion[2])
        if attempts <= 0:
            return None
        return (confusion[1][1] + confusion[2][2]) / attempts

    def snapshot(self) -> dict[str, Any]:
        overall = self.total_successes / self.total_attempts if self.total_attempts else None
        probe_overall = self.probe_successes / self.probe_attempts if self.probe_attempts else None
        return {
            "overall": overall,
            "balanced_accuracy": self._balanced_accuracy(self.target_successes, self.target_attempts),
            "one_vs_two_accuracy": self._one_vs_two_accuracy(self.confusion),
            "recent_20": self._rate(self.outcomes, 20),
            "recent_100": self._rate(self.outcomes, 100),
            "recent_500": self._rate(self.outcomes, 500),
            "successes": self.total_successes,
            "attempts": self.total_attempts,
            "by_target_accuracy": self._class_rates(self.target_successes, self.target_attempts),
            "confusion_matrix": [row[:] for row in self.confusion],
            "probe": {
                "overall": probe_overall,
                "balanced_accuracy": self._balanced_accuracy(
                    self.probe_target_successes,
                    self.probe_target_attempts,
                ),
                "one_vs_two_accuracy": self._one_vs_two_accuracy(self.probe_confusion),
                "recent_20": self._rate(self.probe_outcomes, 20),
                "recent_100": self._rate(self.probe_outcomes, 100),
                "recent_500": self._rate(self.probe_outcomes, 500),
                "successes": self.probe_successes,
                "attempts": self.probe_attempts,
                "by_target_accuracy": self._class_rates(
                    self.probe_target_successes,
                    self.probe_target_attempts,
                ),
                "confusion_matrix": [row[:] for row in self.probe_confusion],
            },
        }


def prepare_stage22b_experiment() -> tuple[NumerosityExperiment, dict[str, Any], str]:
    """Train/replay the robust encoder on Stage-2.1-like data, then freeze it."""
    experiment = NumerosityExperiment()

    if STAGE22B_CHECKPOINT.exists():
        metadata = experiment.load_checkpoint(STAGE22B_CHECKPOINT)
        snapshot = dict(metadata.get("pretraining_snapshot") or {})
        return experiment, snapshot, "checkpoint"

    metrics = SuccessMetrics()
    for trial in range(1, STAGE22B_REPLAY_TRIALS + 1):
        is_probe = trial % PROBE_EVERY == 0
        result = experiment.step(
            trial,
            learn=not is_probe,
            trial_kind="probe" if is_probe else "train",
            stimulus_profile="train",
        )
        metrics.record(
            bool(result["correct"]),
            int(result["target"]),
            int(result["choice"]),
            is_probe=is_probe,
        )

    snapshot = metrics.snapshot()
    experiment.save_checkpoint(
        STAGE22B_CHECKPOINT,
        {
            "stage": "2.2B_pretraining",
            "pretraining_trials": STAGE22B_REPLAY_TRIALS,
            "probe_every": PROBE_EVERY,
            "pretraining_snapshot": snapshot,
            "learner": experiment.config_dict(),
            "purpose": "robust visual encoder state before frozen Stage-2.2B OOD evaluation",
        },
    )
    return experiment, snapshot, "deterministic_replay"


def _profile_results(profile_metrics: dict[str, SuccessMetrics]) -> dict[str, Any]:
    return {
        profile: metrics.snapshot() if metrics.total_attempts else None
        for profile, metrics in profile_metrics.items()
    }


def _add_pool_activity(
    activity: dict[int, float],
    pool: list[int],
    start: int,
    count: int,
    value: float,
    step: int,
) -> None:
    if not pool:
        return
    for n in range(min(count, len(pool))):
        index = pool[(start + n * step) % len(pool)]
        activity[index] = max(activity.get(index, 0.0), min(1.0, value))


def make_display_activity(result: dict[str, Any]) -> list[list[float | int]]:
    """Visualization proxy; these are not claimed to be FlyWire spike predictions."""
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

    policy = result["policy"]
    probabilities = [float(policy["p0"]), float(policy["p1"]), float(policy["p2"])]
    for action, probability in enumerate(probabilities):
        start = action * 997 + trial * 31
        intensity = 0.30 + 0.62 * probability
        if action == result["choice"]:
            intensity += 0.08
        _add_pool_activity(activity, OUTPUT_POOL, start, 35, intensity, 149)

    return [[index, round(value, 3)] for index, value in activity.items()]


def make_frame(
    result: dict[str, Any],
    *,
    profile: str,
    profile_trial: int,
    profile_index: int,
) -> dict[str, Any]:
    return {
        "type": "telemetry",
        "telemetry_source": "numerosity_prototype",
        "activity_source": "display_proxy_not_connectome_spikes",
        "learning_model": "reward_modulated_sparse_associator",
        "phase": "dots_0_2_stage2_2b_robust_ood",
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

    experiment, pretraining_snapshot, state_source = prepare_stage22b_experiment()
    profile_metrics = {profile: SuccessMetrics() for profile in PROFILE_ORDER}
    eval_trial = 1

    logger = RunLogger(
        {
            "experiment": "numerosity_0_2_stage2_2b",
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
            "stage22b_replay_trials": STAGE22B_REPLAY_TRIALS,
            "stage22b_state_source": state_source,
            "stage22b_pretraining_snapshot": pretraining_snapshot,
            "evaluation_profile": "robust_encoder_retest",
            "evaluation_profiles": PROFILE_ORDER,
            "profile_trials": PROFILE_TRIALS,
            "total_evaluation_trials": PROFILE_TRIALS * len(PROFILE_ORDER),
            "evaluation_learning_enabled": False,
            "learner": experiment.config_dict(),
            "scientific_scope": (
                "Stage-2.2B robust-vision retest. The learner is trained for 10,000 Stage-2.1-like trials using the new "
                "continuous Gaussian rasterizer and anti-alias/divisive-contrast/sub-pixel-phase visual front end. The "
                "resulting state is then frozen. Position-only, brightness-only, and combined OOD blocks receive no "
                "plastic updates. Total signal-energy remains identically distributed for 1 and 2, and preprocessing "
                "restores global L1 energy after local contrast/phase pooling. No object count, peak count, or target "
                "feature is injected. FlyWire coordinates remain visualization-only."
            ),
        }
    )

    status = "completed"
    finalized = False
    last_frame: dict[str, Any] | None = None
    try:
        for profile_index, profile in enumerate(PROFILE_ORDER, start=1):
            metrics = profile_metrics[profile]
            profile_trial = 1

            while profile_trial <= PROFILE_TRIALS:
                latest_frame: dict[str, Any] | None = None

                for _ in range(TRIALS_PER_UI_FRAME):
                    if profile_trial > PROFILE_TRIALS:
                        break
                    result = experiment.step(
                        eval_trial,
                        learn=False,
                        trial_kind="robust_ood_eval",
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
                    latest_frame = frame
                    last_frame = frame
                    profile_trial += 1
                    eval_trial += 1

                if latest_frame is not None:
                    await websocket.send_json(latest_frame)
                await asyncio.sleep(UI_INTERVAL_SECONDS)

        if last_frame is not None:
            final_snapshot = dict(last_frame.get("metrics") or {})
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

        # Keep the socket open so the frontend does not reconnect and create a
        # duplicate completed experiment. Closing the browser/server exits here.
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
