import asyncio
import math
import random
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .flywire import FAFB_V783_TOTAL_NEURONS, load_fafb_soma_layout
from .run_logging import RunLogger

app = FastAPI(title="DrosoMath telemetry API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "source": "mock geometry — run scripts/download_fafb783.py for real soma coordinates",
        "count": len(neurons),
        "total_connectome_neurons": FAFB_V783_TOTAL_NEURONS,
        "coordinate_kind": "mock",
    }


LAYOUT = load_fafb_soma_layout() or make_mock_layout()
NEURONS = LAYOUT["neurons"]
NEURON_COUNT = len(NEURONS)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "layout_source": LAYOUT["source"],
        "layout_count": NEURON_COUNT,
        "telemetry_source": "mock",
    }


@app.get("/api/layout")
def layout() -> dict[str, Any]:
    return LAYOUT


class SuccessMetrics:
    """Session-level success metrics for the current training stream.

    The real simulator will eventually own these counters. Keeping the metric
    contract here now lets the UI use the same overall/rolling definitions when
    mock telemetry is replaced with real experiment trials.
    """

    def __init__(self) -> None:
        self.total_attempts = 0
        self.total_successes = 0
        self.outcomes: deque[int] = deque(maxlen=500)

    def record(self, correct: bool) -> None:
        outcome = 1 if correct else 0
        self.total_attempts += 1
        self.total_successes += outcome
        self.outcomes.append(outcome)

    def _recent_rate(self, window: int) -> float | None:
        if not self.outcomes:
            return None
        values = list(self.outcomes)[-window:]
        return sum(values) / len(values)

    def snapshot(self) -> dict[str, Any]:
        overall = (
            self.total_successes / self.total_attempts
            if self.total_attempts
            else None
        )
        return {
            "overall": overall,
            "recent_20": self._recent_rate(20),
            "recent_100": self._recent_rate(100),
            "recent_500": self._recent_rate(500),
            "successes": self.total_successes,
            "attempts": self.total_attempts,
        }


def make_frame(t: float, trial: int) -> dict[str, Any]:
    # Geometry can now be real FlyWire data, but neural activity remains mock until
    # the connectome simulator adapter is integrated. Stream only a sparse active
    # subset so the browser path scales to tens of thousands of displayed somas.
    activity: list[list[float | int]] = []
    active_count = min(700, NEURON_COUNT)
    if NEURON_COUNT:
        start = (trial * 997) % NEURON_COUNT
        step = 37
        for k in range(active_count):
            index = (start + k * step) % NEURON_COUNT
            neuron = NEURONS[index]
            phase = index * 0.031
            base = 0.30 + 0.32 * math.sin(t * 2.4 + phase)
            region = neuron.get("region", "")
            burst = 0.0
            if region in ("central", "mushroom_body") and trial % 8 in (5, 6):
                burst = 0.48
            if region in ("sensory", "dopamine") and trial % 8 == 7:
                burst = 0.66
            value = max(0.08, min(1.0, base + burst + ((index * 17 + trial) % 13) / 100.0))
            activity.append([index, round(value, 3)])

    # Deterministic mock outcome: exactly 4 of every 5 trials are marked correct.
    # This exists only to test the UI/metric pipeline and is NOT learned behavior.
    answer = 2 if trial % 5 == 0 else 3
    correct = answer == 3
    return {
        "type": "telemetry",
        "telemetry_source": "mock",
        "timestamp": time.time(),
        "trial": trial,
        "problem": "2 + 1",
        "answer": answer,
        "correct": correct,
        "reward": 1.0 if correct else -1.0,
        "activity": activity,
        "plasticity": {
            "mean_delta_w": round(0.03 * math.sin(t * 0.8), 4),
            "active_synapses": 120 + (trial % 31),
        },
    }


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    started = time.monotonic()
    trial = 1
    metrics = SuccessMetrics()
    logger = RunLogger(
        {
            "experiment": "mock_2_plus_1",
            "telemetry_source": "mock",
            "layout_source": LAYOUT["source"],
            "layout_count": NEURON_COUNT,
            "total_connectome_neurons": FAFB_V783_TOTAL_NEURONS,
            "rolling_windows": [20, 100, 500],
            "mock_rule": "trial divisible by 5 -> wrong answer; all other trials -> correct",
        }
    )

    status = "completed"
    try:
        while True:
            t = time.monotonic() - started
            frame = make_frame(t, trial)
            metrics.record(bool(frame["correct"]))
            snapshot = metrics.snapshot()
            frame["metrics"] = snapshot
            # Compatibility alias for clients that still expect `accuracy`.
            frame["accuracy"] = snapshot["overall"]

            logger.record(frame, snapshot)
            await websocket.send_json(frame)
            trial += 1
            await asyncio.sleep(0.1)  # 10 Hz UI telemetry
    except WebSocketDisconnect:
        status = "completed"
    except Exception:
        status = "failed"
        raise
    finally:
        logger.finalize(status=status)
