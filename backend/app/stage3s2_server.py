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
from .run_logging import RunLogger
from .symbolic_stage3s2 import (
    ADDITION_HELDOUT_PAIRS,
    ADDITION_REVERSE_TEST_PAIRS,
    ADDITION_TRAIN_PAIRS,
    MICRO_ADDITION_PAIRS,
    N_OUTPUTS,
    SMALL_ADDITION_PAIRS,
    SUCCESSOR2_HELDOUT_STARTS,
    SUCCESSOR2_TRAIN_STARTS,
    SUCCESSOR3_HELDOUT_STARTS,
    SUCCESSOR3_TRAIN_STARTS,
    SymbolicStage3S2Experiment,
)


app = FastAPI(title="DrosoMath telemetry API", version="0.15.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROBE_EVERY = 25
TRAIN_TRIALS = 360_000
TRAINING_MILESTONES = (30_000, 90_000, 150_000, 210_000, 280_000, 360_000)
PROFILE_TRIALS = 1_200
PROFILE_ORDER = (
    "identity",
    "successor",
    "successor2_seen",
    "successor2_heldout",
    "successor3_seen",
    "successor3_heldout",
    "addition_seen",
    "addition_commutativity",
    "addition_heldout_pairs",
)
TRIALS_PER_UI_FRAME = 20
UI_INTERVAL_SECONDS = 0.20

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "stage3s2_seed53_t360000_v1.npz"


def _milestone_checkpoint(trial: int) -> Path:
    return CHECKPOINT_DIR / f"stage3s2_seed53_t{trial}_v1.npz"


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


class Metrics:
    def __init__(self) -> None:
        self.attempts = 0
        self.successes = 0
        self.absolute_error_sum = 0.0
        self.within_one = 0
        self.recent: deque[int] = deque(maxlen=500)
        self.target_attempts = [0] * N_OUTPUTS
        self.target_successes = [0] * N_OUTPUTS
        self.confusion = [[0] * N_OUTPUTS for _ in range(N_OUTPUTS)]

    def record(self, target: int, choice: int) -> None:
        correct = target == choice
        self.attempts += 1
        self.successes += int(correct)
        self.absolute_error_sum += abs(choice - target)
        self.within_one += int(abs(choice - target) <= 1)
        self.recent.append(int(correct))
        self.target_attempts[target] += 1
        self.target_successes[target] += int(correct)
        self.confusion[target][choice] += 1

    @staticmethod
    def _rate(values: deque[int], n: int) -> float | None:
        if not values:
            return None
        subset = list(values)[-n:]
        return sum(subset) / len(subset)

    def snapshot(self) -> dict[str, Any]:
        class_rates = {
            str(i): self.target_successes[i] / self.target_attempts[i]
            if self.target_attempts[i]
            else None
            for i in range(N_OUTPUTS)
        }
        valid_rates = [v for v in class_rates.values() if v is not None]
        return {
            "overall": self.successes / self.attempts if self.attempts else None,
            "balanced_accuracy": sum(valid_rates) / len(valid_rates) if valid_rates else None,
            "mean_absolute_error": self.absolute_error_sum / self.attempts if self.attempts else None,
            "within_one_accuracy": self.within_one / self.attempts if self.attempts else None,
            "recent_20": self._rate(self.recent, 20),
            "recent_100": self._rate(self.recent, 100),
            "recent_500": self._rate(self.recent, 500),
            "successes": self.successes,
            "attempts": self.attempts,
            "by_target_accuracy": class_rates,
            "confusion_matrix": [row[:] for row in self.confusion],
            "probe": {},
        }


def _choose_start(exp: SymbolicStage3S2Experiment, pool: tuple[int, ...]) -> int:
    return int(pool[int(exp.rng.integers(0, len(pool)))])


def _training_spec(exp: SymbolicStage3S2Experiment, trial: int) -> dict[str, Any]:
    """Return one deterministic curriculum sample and its consolidation schedule."""
    if trial <= 30_000:
        return {
            "task": "identity",
            "phase": "01_identity_grounding_0_30k",
            "token_scale": 1.0,
            "rec_scale": 0.0,
            "out_scale": 1.0,
        }

    if trial <= 90_000:
        r = float(exp.rng.random())
        task = "successor" if r < 0.82 else "identity"
        return {
            "task": task,
            "phase": "02_successor_30_90k",
            "token_scale": 0.65,
            "rec_scale": 1.0,
            "out_scale": 1.0,
        }

    if trial <= 150_000:
        r = float(exp.rng.random())
        if r < 0.35:
            task = "successor2"
            a = _choose_start(exp, SUCCESSOR2_TRAIN_STARTS)
        elif r < 0.60:
            task = "successor3"
            a = _choose_start(exp, SUCCESSOR3_TRAIN_STARTS)
        elif r < 0.85:
            task = "successor"
            a = None
        else:
            task = "identity"
            a = None
        return {
            "task": task,
            "a": a,
            "phase": "03_successor_composition_90_150k",
            "token_scale": 0.25,
            "rec_scale": 0.70,
            "out_scale": 1.0,
        }

    if trial <= 210_000:
        r = float(exp.rng.random())
        if r < 0.55:
            task, pool, a = "addition", MICRO_ADDITION_PAIRS, None
        elif r < 0.70:
            task, pool, a = "successor2", None, _choose_start(exp, SUCCESSOR2_TRAIN_STARTS)
        elif r < 0.80:
            task, pool, a = "successor3", None, _choose_start(exp, SUCCESSOR3_TRAIN_STARTS)
        elif r < 0.90:
            task, pool, a = "successor", None, None
        else:
            task, pool, a = "identity", None, None
        return {
            "task": task,
            "a": a,
            "pair_pool": pool,
            "phase": "04_micro_addition_0_2_150_210k",
            "token_scale": 0.08,
            "rec_scale": 0.40,
            "out_scale": 1.0,
        }

    if trial <= 280_000:
        r = float(exp.rng.random())
        if r < 0.65:
            task, pool, a = "addition", SMALL_ADDITION_PAIRS, None
        elif r < 0.77:
            task, pool, a = "successor2", None, _choose_start(exp, SUCCESSOR2_TRAIN_STARTS)
        elif r < 0.85:
            task, pool, a = "successor3", None, _choose_start(exp, SUCCESSOR3_TRAIN_STARTS)
        elif r < 0.93:
            task, pool, a = "successor", None, None
        else:
            task, pool, a = "identity", None, None
        return {
            "task": task,
            "a": a,
            "pair_pool": pool,
            "phase": "05_small_addition_0_4_210_280k",
            "token_scale": 0.04,
            "rec_scale": 0.25,
            "out_scale": 1.0,
        }

    r = float(exp.rng.random())
    if r < 0.76:
        task, pool, a = "addition", ADDITION_TRAIN_PAIRS, None
    elif r < 0.84:
        task, pool, a = "successor2", None, _choose_start(exp, SUCCESSOR2_TRAIN_STARTS)
    elif r < 0.90:
        task, pool, a = "successor3", None, _choose_start(exp, SUCCESSOR3_TRAIN_STARTS)
    elif r < 0.96:
        task, pool, a = "successor", None, None
    else:
        task, pool, a = "identity", None, None
    return {
        "task": task,
        "a": a,
        "pair_pool": pool,
        "phase": "06_full_addition_280_360k",
        "token_scale": 0.015,
        "rec_scale": 0.15,
        "out_scale": 1.0,
    }


def prepare_experiment() -> tuple[SymbolicStage3S2Experiment, dict[str, Any], dict[str, Any], str]:
    experiment = SymbolicStage3S2Experiment()
    if FINAL_CHECKPOINT.exists():
        metadata = experiment.load_checkpoint(FINAL_CHECKPOINT)
        return (
            experiment,
            dict(metadata.get("training_snapshot") or {}),
            dict(metadata.get("training_snapshots") or {}),
            "checkpoint",
        )

    task_names = ("identity", "successor", "successor2", "successor3", "addition")
    overall = Metrics()
    task_metrics = {task: Metrics() for task in task_names}
    snapshots: dict[str, Any] = {}

    for trial in range(1, TRAIN_TRIALS + 1):
        spec = _training_spec(experiment, trial)
        task = str(spec["task"])
        is_probe = trial % PROBE_EVERY == 0
        result = experiment.step(
            trial,
            task=task,
            learn=not is_probe,
            a=spec.get("a"),
            pair_pool=spec.get("pair_pool"),
            token_plasticity_scale=float(spec["token_scale"]),
            recurrent_plasticity_scale=float(spec["rec_scale"]),
            output_plasticity_scale=float(spec["out_scale"]),
            trial_kind="probe" if is_probe else "train",
        )
        overall.record(int(result["target"]), int(result["choice"]))
        task_metrics[task].record(int(result["target"]), int(result["choice"]))

        if trial in TRAINING_MILESTONES:
            snapshot = overall.snapshot()
            snapshot["curriculum_phase"] = spec["phase"]
            snapshot["tasks"] = {name: meter.snapshot() for name, meter in task_metrics.items()}
            snapshot["plasticity_scales"] = result["plasticity_scales"]
            snapshots[str(trial)] = snapshot
            metadata = {
                "stage": "3S.2_symbolic_pretraining",
                "training_trials": trial,
                "target_training_trials": TRAIN_TRIALS,
                "probe_every": PROBE_EVERY,
                "training_snapshot": snapshot,
                "training_snapshots": dict(snapshots),
                "learner": experiment.config_dict(),
            }
            experiment.save_checkpoint(_milestone_checkpoint(trial), metadata)

    return experiment, snapshots[str(TRAIN_TRIALS)], snapshots, "deterministic_360k_stage3s2_curriculum"


def _profile_example(profile: str, index: int) -> tuple[str, int | None, int | None]:
    if profile == "identity":
        return "identity", index % 10, None
    if profile == "successor":
        return "successor", index % 9, None
    if profile == "successor2_seen":
        return "successor2", SUCCESSOR2_TRAIN_STARTS[index % len(SUCCESSOR2_TRAIN_STARTS)], None
    if profile == "successor2_heldout":
        return "successor2", SUCCESSOR2_HELDOUT_STARTS[index % len(SUCCESSOR2_HELDOUT_STARTS)], None
    if profile == "successor3_seen":
        return "successor3", SUCCESSOR3_TRAIN_STARTS[index % len(SUCCESSOR3_TRAIN_STARTS)], None
    if profile == "successor3_heldout":
        return "successor3", SUCCESSOR3_HELDOUT_STARTS[index % len(SUCCESSOR3_HELDOUT_STARTS)], None
    if profile == "addition_seen":
        pair = ADDITION_TRAIN_PAIRS[index % len(ADDITION_TRAIN_PAIRS)]
    elif profile == "addition_commutativity":
        pair = ADDITION_REVERSE_TEST_PAIRS[index % len(ADDITION_REVERSE_TEST_PAIRS)]
    elif profile == "addition_heldout_pairs":
        pair = ADDITION_HELDOUT_PAIRS[index % len(ADDITION_HELDOUT_PAIRS)]
    else:
        raise ValueError(f"Unknown Stage 3S.2 profile: {profile}")
    return "addition", pair[0], pair[1]


def _add_pool_activity(activity: dict[int, float], pool: list[int], start: int, count: int, value: float, step: int) -> None:
    if not pool:
        return
    for n in range(min(count, len(pool))):
        index = pool[(start + n * step) % len(pool)]
        activity[index] = max(activity.get(index, 0.0), min(1.0, value))


def make_display_activity(result: dict[str, Any]) -> list[list[float | int]]:
    """Visualization proxy only; not a FlyWire spike prediction."""
    activity: dict[int, float] = {}
    trial = int(result["trial"])

    for slot, token in enumerate(result["tokens"]):
        token_key = sum((i + 1) * ord(ch) for i, ch in enumerate(token))
        _add_pool_activity(activity, VISUAL_POOL, token_key * 97 + slot * 991 + trial * 7, 40, 0.82, 173)

    for state_index, state_value in result["state_activity"]:
        if CENTRAL_POOL:
            mapped = CENTRAL_POOL[(int(state_index) * 7919 + trial * 13) % len(CENTRAL_POOL)]
            activity[mapped] = max(activity.get(mapped, 0.0), 0.35 + 0.65 * float(state_value))

    for rank, item in enumerate(result["policy"]["top"][:3]):
        intensity = 0.32 + 0.62 * float(item["p"])
        _add_pool_activity(
            activity,
            OUTPUT_POOL,
            int(item["choice"]) * 997 + trial * 31 + rank * 53,
            30,
            intensity,
            149,
        )
    return [[index, round(value, 3)] for index, value in activity.items()]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "experiment": "symbolic_stage3s2",
        "mode": "360k_multistage_consolidated_symbolic_curriculum_then_frozen_generalization",
        "training_trials": TRAIN_TRIALS,
        "training_milestones": list(TRAINING_MILESTONES),
        "profiles": list(PROFILE_ORDER),
        "profile_trials": PROFILE_TRIALS,
        "addition_chance_accuracy": 1.0 / N_OUTPUTS,
        "identity_successor_chance_accuracy": 0.1,
    }


@app.get("/api/layout")
def layout() -> dict[str, Any]:
    return LAYOUT


def _profile_snapshots(profile_metrics: dict[str, Metrics]) -> dict[str, Any]:
    return {
        profile: meter.snapshot() if meter.attempts else None
        for profile, meter in profile_metrics.items()
    }


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()

    experiment, training_snapshot, training_snapshots, state_source = await asyncio.to_thread(prepare_experiment)
    profile_metrics = {profile: Metrics() for profile in PROFILE_ORDER}
    eval_trial = 1

    learner = experiment.config_dict()
    learner["training_trials"] = TRAIN_TRIALS
    learner["training_milestones"] = list(TRAINING_MILESTONES)
    learner["training_snapshot"] = training_snapshot
    learner["training_snapshots"] = training_snapshots
    learner["state_source"] = state_source

    logger = RunLogger(
        {
            "experiment": "symbolic_stage3s2",
            "telemetry_source": "symbolic_math_prototype",
            "activity_source": "display_proxy_not_connectome_spikes",
            "learning_model": "consolidated_reward_gated_recurrent_symbolic_associator",
            "chance_accuracy": 1.0 / N_OUTPUTS,
            "layout_source": LAYOUT["source"],
            "layout_count": NEURON_COUNT,
            "evaluation_profile": "symbolic_multistep_frozen_generalization",
            "evaluation_profiles": list(PROFILE_ORDER),
            "profile_trials": PROFILE_TRIALS,
            "total_evaluation_trials": PROFILE_TRIALS * len(PROFILE_ORDER),
            "evaluation_learning_enabled": False,
            "learner": learner,
            "scientific_scope": (
                "Stage 3S.2 triples training to 360k and inserts explicit learning stages without injecting numeric "
                "answers: symbol grounding, one-step NEXT, repeated-NEXT composition, 0..2 addition, 0..4 addition, "
                "then full addition. Repeated NEXT is represented only by repeated visible NEXT tokens. Most addition "
                "pairs expose both operand orders, a small one-way subset tests learned commutativity transfer, and a "
                "separate unordered set remains completely held out. Internal plasticity decays across stages to protect "
                "earlier representations. FlyWire coordinates remain visualization-only, not simulated connectome spikes."
            ),
        }
    )

    finalized = False
    status = "completed"
    last_frame: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None

    try:
        for profile_index, profile in enumerate(PROFILE_ORDER, start=1):
            meter = profile_metrics[profile]
            profile_trial = 1

            while profile_trial <= PROFILE_TRIALS:
                latest_frame: dict[str, Any] | None = None
                for _ in range(TRIALS_PER_UI_FRAME):
                    if profile_trial > PROFILE_TRIALS:
                        break

                    task, a, b = _profile_example(profile, profile_trial - 1)
                    result = experiment.step(
                        eval_trial,
                        task=task,
                        learn=False,
                        a=a,
                        b=b,
                        token_plasticity_scale=0.0,
                        recurrent_plasticity_scale=0.0,
                        output_plasticity_scale=0.0,
                        trial_kind="stage3s2_frozen_eval",
                    )
                    meter.record(int(result["target"]), int(result["choice"]))
                    snapshot = meter.snapshot()
                    snapshot["profiles"] = _profile_snapshots(profile_metrics)
                    snapshot["current_profile"] = profile
                    snapshot["profile_trial"] = profile_trial
                    snapshot["profile_trials_target"] = PROFILE_TRIALS
                    snapshot["experiment_complete"] = False

                    frame = {
                        "type": "telemetry",
                        "telemetry_source": "symbolic_math_prototype",
                        "activity_source": "display_proxy_not_connectome_spikes",
                        "phase": "stage3s2_symbolic_multistep_frozen_generalization",
                        "trial_kind": result["trial_kind"],
                        "learning_enabled": False,
                        "timestamp": time.time(),
                        "trial": eval_trial,
                        "evaluation_profile": profile,
                        "profile_trial": profile_trial,
                        "profile_trials_target": PROFILE_TRIALS,
                        "profile_index": profile_index,
                        "profile_count": len(PROFILE_ORDER),
                        "task": task,
                        "tokens": result["tokens"],
                        "expression": result["expression"],
                        "target": int(result["target"]),
                        "answer": int(result["choice"]),
                        "correct": bool(result["correct"]),
                        "reward": float(result["reward"]),
                        "active_output_count": int(result["active_output_count"]),
                        "accuracy": snapshot["overall"],
                        "metrics": snapshot,
                        "policy": result["policy"],
                        "activity": make_display_activity(result),
                        "plasticity": result["plasticity"],
                        "stimulus": {"kind": "symbol_sequence", "controls": {}},
                    }
                    logger.record(frame, snapshot)
                    latest_frame = frame
                    last_frame = frame
                    last_metrics = snapshot
                    eval_trial += 1
                    profile_trial += 1

                if latest_frame is not None:
                    await websocket.send_json(latest_frame)
                await asyncio.sleep(UI_INTERVAL_SECONDS)

        if last_frame is not None and last_metrics is not None:
            final_snapshot = dict(last_metrics)
            final_snapshot["profiles"] = _profile_snapshots(profile_metrics)
            final_snapshot["experiment_complete"] = True
            final_snapshot["current_profile"] = PROFILE_ORDER[-1]
            final_snapshot["profile_trial"] = PROFILE_TRIALS
            final_frame = dict(last_frame)
            final_frame["metrics"] = final_snapshot
            final_frame["experiment_complete"] = True
            last_frame = final_frame
            last_metrics = final_snapshot
            await websocket.send_json(final_frame)
            logger.finalize(status="completed", frame=final_frame, metrics=final_snapshot)
            finalized = True
    except WebSocketDisconnect:
        status = "interrupted"
    except Exception:
        status = "error"
        raise
    finally:
        if not finalized:
            logger.finalize(status=status, frame=last_frame, metrics=last_metrics)
