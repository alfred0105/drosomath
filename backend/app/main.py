import asyncio
import math
import random
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .flywire import FAFB_V783_TOTAL_NEURONS, load_fafb_soma_layout

app = FastAPI(title="DrosoMath telemetry API", version="0.2.0")

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
            base = 0.35 + 0.35 * math.sin(t * 2.4 + phase)
            region = neuron.get("region", "")
            burst = 0.0
            if region in ("central", "mushroom_body") and trial % 8 in (5, 6):
                burst = 0.28
            if region in ("sensory", "dopamine") and trial % 8 == 7:
                burst = 0.42
            value = max(0.12, min(1.0, base + burst + ((index * 17 + trial) % 13) / 100.0))
            activity.append([index, round(value, 3)])

    answer = 3 if trial % 5 else 2
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
        "accuracy": round(0.5 + 0.45 * (1.0 - math.exp(-trial / 80.0)), 3),
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
    trial = 0
    try:
        while True:
            t = time.monotonic() - started
            await websocket.send_json(make_frame(t, trial))
            trial += 1
            await asyncio.sleep(0.1)  # 10 Hz UI telemetry
    except WebSocketDisconnect:
        return
