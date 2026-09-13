import asyncio
import math
import random
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DrosoMath telemetry API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock layout only. Replace this adapter with FlyWire/Codex coordinates later.
NEURON_COUNT = 800
random.seed(42)
NEURONS = [
    {
        "id": i,
        "x": random.gauss(0.0, 1.9),
        "y": random.gauss(0.0, 1.2),
        "z": random.gauss(0.0, 0.9),
        "region": ("visual", "mushroom_body", "central", "dopamine")[i % 4],
    }
    for i in range(NEURON_COUNT)
]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/layout")
def layout() -> dict[str, Any]:
    return {"neurons": NEURONS, "source": "mock", "count": len(NEURONS)}


def make_frame(t: float, trial: int) -> dict[str, Any]:
    activity = []
    for neuron in NEURONS:
        phase = neuron["id"] * 0.031
        base = 0.5 + 0.5 * math.sin(t * 2.4 + phase)
        burst = 0.0
        if neuron["region"] == "mushroom_body" and trial % 8 in (5, 6):
            burst = 0.35
        if neuron["region"] == "dopamine" and trial % 8 == 7:
            burst = 0.55
        value = max(0.0, min(1.0, base * 0.55 + burst + random.random() * 0.08))
        if value > 0.18:
            activity.append([neuron["id"], round(value, 3)])

    answer = 3 if trial % 5 else 2
    correct = answer == 3
    return {
        "type": "telemetry",
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
