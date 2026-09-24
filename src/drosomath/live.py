from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import parse_qs, urlparse

from .core import (
    ActivityBiasedCandidateConfig,
    ActivityBiasedCandidateGenerator,
    ConsolidationConfig,
    MemoryConsolidator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    STDPPlasticity,
    STDPRule,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)


@dataclass(frozen=True, slots=True)
class LiveTrial:
    name: str
    stimulus: dict[int, float]
    expected_output: int


class LiveSimulation:
    """Small continuously-learning network used by the local live dashboard."""

    RESPONSE_STEPS = 3

    def __init__(self, *, hz: float = 8.0) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be > 0")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = True
        self.hz = float(hz)
        self._build_model()

    def _build_model(self) -> None:
        synapses = [
            SynapseState(0, 2, 0.58),
            SynapseState(1, 3, 0.58),
            SynapseState(2, 4, 0.58),
            SynapseState(3, 5, 0.58),
            SynapseState(0, 3, 0.18),
            SynapseState(1, 2, 0.18),
            SynapseState(2, 5, 0.16),
            SynapseState(3, 4, 0.16),
            SynapseState(6, 2, 0.08),
            SynapseState(7, 3, 0.08),
        ]
        self.tracker = PlasticityTracker(
            synapses,
            reward_window=12,
            weight_rule=RewardWeightRule(
                learning_rate=0.018,
                min_weight=0.0,
                max_weight=1.0,
            ),
        )
        self.network = SpikingNetwork.from_ids(
            range(8),
            self.tracker,
            threshold=0.5,
            decay=0.90,
            stdp=STDPPlasticity(
                rule=STDPRule(
                    potentiation_rate=0.006,
                    depression_rate=0.007,
                    tau_plus=8.0,
                    tau_minus=8.0,
                    window=8,
                )
            ),
        )
        self.consolidator = MemoryConsolidator(
            config=ConsolidationConfig(
                min_usage=3,
                min_reward=0.03,
                growth_rate=0.08,
                decay_rate=0.001,
            )
        )
        self.generator = ActivityBiasedCandidateGenerator(
            config=ActivityBiasedCandidateConfig(
                max_candidates=32,
                pool_size=8,
            )
        )
        self.structural = StructuralPlasticityManager(
            self.tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=2,
                stale_steps=12,
                reward_threshold=-0.01,
                protected_stability=0.65,
                max_rewire_per_cycle=1,
                regrow_weight=0.08,
            ),
        )
        self.trials = (
            LiveTrial("pattern_A", {0: 1.0}, 4),
            LiveTrial("pattern_B", {1: 1.0}, 5),
        )
        self.trial_index = 0
        self.phase = 0
        self.output_counts = {4: 0, 5: 0}
        self.last_reward = 0.0
        self.last_prediction: int | None = None
        self.last_expected = self.trials[0].expected_output
        self.last_correct = False
        self.last_rewire: dict[str, Any] | None = None
        self.completed_trials = 0
        self.correct_trials = 0
        self.last_result = None
        self.history: deque[dict[str, float | int]] = deque(maxlen=160)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            with self._lock:
                if self.running:
                    self._advance_locked()
                hz = self.hz
            elapsed = time.perf_counter() - started
            time.sleep(max(0.002, 1.0 / hz - elapsed))

    def advance_once(self) -> dict[str, Any]:
        with self._lock:
            self._advance_locked()
            return self._snapshot_locked()

    def _advance_locked(self) -> None:
        trial = self.trials[self.trial_index]
        if self.phase == 0:
            self.network.reset_state(clear_spike_history=True)
            self.tracker.clear_recent()
            self.output_counts = {4: 0, 5: 0}
            stimulus = trial.stimulus
        else:
            stimulus = None

        result = self.network.step(stimulus)
        self.last_result = result
        for neuron_id in result.fired:
            if neuron_id in self.output_counts:
                self.output_counts[neuron_id] += 1

        self.phase += 1
        if self.phase < self.RESPONSE_STEPS:
            return

        active = [
            (count, neuron_id)
            for neuron_id, count in self.output_counts.items()
            if count > 0
        ]
        prediction = None
        if active:
            prediction = max(active, key=lambda item: (item[0], -item[1]))[1]

        correct = prediction == trial.expected_output
        reward = 1.0 if correct else -0.55
        self.tracker.apply_reward(reward=reward, step=self.network.step_index)
        self.consolidator.consolidate(self.tracker.synapses)

        self.last_prediction = prediction
        self.last_expected = trial.expected_output
        self.last_correct = correct
        self.last_reward = reward
        self.completed_trials += 1
        if correct:
            self.correct_trials += 1

        self.last_rewire = None
        if self.completed_trials % 12 == 0:
            candidates = self.generator.generate(
                self.tracker.synapses,
                step=self.network.step_index,
                neuron_ids=self.network.neurons,
            )
            rewired = self.structural.rewire(
                step=self.network.step_index,
                candidate_pairs=candidates,
            )
            if rewired.changed:
                self.last_rewire = {
                    "pruned": [list(edge) for edge in rewired.pruned],
                    "regrown": [list(edge) for edge in rewired.regrown],
                }

        synapses = self.tracker.synapses
        self.history.append(
            {
                "step": self.network.step_index,
                "reward": reward,
                "accuracy": self.accuracy,
                "mean_weight": fmean(s.weight for s in synapses) if synapses else 0.0,
                "mean_stability": fmean(s.stability for s in synapses) if synapses else 0.0,
            }
        )
        self.phase = 0
        self.trial_index = (self.trial_index + 1) % len(self.trials)

    @property
    def accuracy(self) -> float:
        if self.completed_trials == 0:
            return 0.0
        return self.correct_trials / self.completed_trials

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        synapses = self.tracker.synapses
        result = self.last_result
        current_trial = self.trials[self.trial_index]
        fired = set(result.fired) if result is not None else set()
        return {
            "step": self.network.step_index,
            "running": self.running,
            "hz": self.hz,
            "phase": self.phase,
            "task": current_trial.name,
            "expected_output": current_trial.expected_output,
            "last_expected": self.last_expected,
            "last_prediction": self.last_prediction,
            "last_reward": self.last_reward,
            "last_correct": self.last_correct,
            "completed_trials": self.completed_trials,
            "accuracy": self.accuracy,
            "last_rewire": self.last_rewire,
            "metrics": {
                "fired_count": len(fired),
                "transferred_synapses": result.transferred_synapses if result else 0,
                "stdp_updates": result.stdp_updates if result else 0,
                "synapse_count": len(synapses),
                "mean_weight": fmean(s.weight for s in synapses) if synapses else 0.0,
                "mean_stability": fmean(s.stability for s in synapses) if synapses else 0.0,
            },
            "neurons": [
                {
                    "id": neuron.neuron_id,
                    "potential": neuron.potential,
                    "fired": neuron.neuron_id in fired,
                    "last_spike_step": neuron.last_spike_step,
                    "role": (
                        "input"
                        if neuron.neuron_id in (0, 1)
                        else "output"
                        if neuron.neuron_id in (4, 5)
                        else "internal"
                    ),
                }
                for neuron in self.network.neurons.values()
            ],
            "synapses": [
                {
                    "pre": synapse.pre_id,
                    "post": synapse.post_id,
                    "weight": synapse.weight,
                    "stability": synapse.stability,
                    "usage": synapse.usage_count,
                    "reward_ema": synapse.reward_ema,
                    "alive": synapse.alive,
                }
                for synapse in synapses
            ],
            "history": list(self.history),
        }

    def control(self, action: str) -> dict[str, Any]:
        with self._lock:
            if action == "pause":
                self.running = False
            elif action == "resume":
                self.running = True
            elif action == "reset":
                hz = self.hz
                running = self.running
                self._build_model()
                self.hz = hz
                self.running = running
            elif action == "faster":
                self.hz = min(60.0, self.hz * 1.5)
            elif action == "slower":
                self.hz = max(0.5, self.hz / 1.5)
            elif action == "step":
                self._advance_locked()
            else:
                raise ValueError(f"unknown control action: {action}")
            return self._snapshot_locked()


class LiveRequestHandler(BaseHTTPRequestHandler):
    simulation: LiveSimulation
    dashboard_html: bytes

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(self.dashboard_html, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(self.simulation.snapshot())
            return
        if parsed.path == "/api/control":
            action = parse_qs(parsed.query).get("action", [""])[0]
            try:
                payload = self.simulation.control(action)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if parsed.path == "/events":
            self._serve_events()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                payload = json.dumps(self.simulation.snapshot(), separators=(",", ":"))
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.12)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _dashboard_html() -> bytes:
    path = Path(__file__).with_name("live_dashboard.html")
    return path.read_bytes()


def run_server(*, host: str = "127.0.0.1", port: int = 8765, hz: float = 8.0, open_browser: bool = True) -> None:
    simulation = LiveSimulation(hz=hz)
    handler = type(
        "DrosoMathLiveHandler",
        (LiveRequestHandler,),
        {"simulation": simulation, "dashboard_html": _dashboard_html()},
    )
    server = ThreadingHTTPServer((host, port), handler)
    simulation.start()
    url = f"http://{host}:{port}/"
    print(f"DrosoMath Live: {url}")
    print("Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        simulation.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="DrosoMath realtime brain/learning dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        simulation = LiveSimulation(hz=args.hz)
        for _ in range(9):
            snapshot = simulation.advance_once()
        print(json.dumps(snapshot, sort_keys=True))
        return

    run_server(
        host=args.host,
        port=args.port,
        hz=args.hz,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
