import asyncio
import random
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .flywire import FAFB_V783_TOTAL_NEURONS, load_fafb_soma_layout
from .numerosity import NumerosityExperiment
from .run_logging import RunLogger

app = FastAPI(title="DrosoMath telemetry API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIALS_PER_UI_FRAME = 5
UI_INTERVAL_SECONDS = 0.1


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

# Robust fallback for datasets whose region labels differ from FAFB classification.
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
        "experiment": "numerosity_0_2_stage1",
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

    def record(self, correct: bool, target: int, choice: int) -> None:
        outcome = 1 if correct else 0
        self.total_attempts += 1
        self.total_successes += outcome
        self.outcomes.append(outcome)
        self.target_attempts[target] += 1
        self.target_successes[target] += outcome
        self.confusion[target][choice] += 1

    def _recent_rate(self, window: int) -> float | None:
        if not self.outcomes:
            return None
        values = list(self.outcomes)[-window:]
        return sum(values) / len(values)

    def snapshot(self) -> dict[str, Any]:
        overall = self.total_successes / self.total_attempts if self.total_attempts else None
        by_target = {
            str(target): (
                self.target_successes[target] / self.target_attempts[target]
                if self.target_attempts[target]
                else None
            )
            for target in range(3)
        }
        return {
            "overall": overall,
            "recent_20": self._recent_rate(20),
            "recent_100": self._recent_rate(100),
            "recent_500": self._recent_rate(500),
            "successes": self.total_successes,
            "attempts": self.total_attempts,
            "by_target_accuracy": by_target,
            "confusion_matrix": [row[:] for row in self.confusion],
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
    """Map prototype state onto real anatomy for visualization only.

    This is deliberately labeled as a proxy. These points are NOT claimed to be
    measured/predicted FlyWire spikes. Actual connectome activity comes later.
    """
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


def make_frame(result: dict[str, Any]) -> dict[str, Any]:
    target = int(result["target"])
    return {
        "type": "telemetry",
        "telemetry_source": "numerosity_prototype",
        "activity_source": "display_proxy_not_connectome_spikes",
        "learning_model": "reward_modulated_sparse_associator",
        "phase": "dots_0_2",
        "timestamp": time.time(),
        "trial": result["trial"],
        "problem": "How many dots?",
        "target": target,
        "answer": result["choice"],
        "correct": result["correct"],
        "reward": result["reward"],
        "stimulus": result["stimulus"],
        "policy": result["policy"],
        "activity": make_display_activity(result),
        "plasticity": result["plasticity"],
    }


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    experiment = NumerosityExperiment()
    metrics = SuccessMetrics()
    trial = 1

    logger = RunLogger(
        {
            "experiment": "numerosity_0_2_stage1",
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
            "learner": experiment.config_dict(),
            "scientific_scope": (
                "Stage-1 reinforcement-learning protocol validation. Anatomical coordinates are FlyWire, "
                "but neural dynamics are not yet the whole-connectome LIF simulation."
            ),
        }
    )

    status = "completed"
    try:
        while True:
            latest_frame: dict[str, Any] | None = None
            latest_snapshot: dict[str, Any] | None = None

            for _ in range(TRIALS_PER_UI_FRAME):
                result = experiment.step(trial)
                frame = make_frame(result)
                metrics.record(bool(frame["correct"]), int(frame["target"]), int(frame["answer"]))
                snapshot = metrics.snapshot()
                frame["metrics"] = snapshot
                frame["accuracy"] = snapshot["overall"]
                logger.record(frame, snapshot)
                latest_frame = frame
                latest_snapshot = snapshot
                trial += 1

            if latest_frame is not None and latest_snapshot is not None:
                await websocket.send_json(latest_frame)
            await asyncio.sleep(UI_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        status = "completed"
    except Exception:
        status = "failed"
        raise
    finally:
        logger.finalize(status=status)
