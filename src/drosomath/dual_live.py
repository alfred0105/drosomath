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
    AdaptiveBridge,
    AdaptiveModuleManager,
    BrainCore,
    BridgeActivityCandidateConfig,
    BridgeActivityCandidateGenerator,
    BridgeStructuralConfig,
    BridgeStructuralPlasticityManager,
    BridgeSynapseState,
    ConsolidationConfig,
    DualBrainSystem,
    MemoryConsolidator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    STDPPlasticity,
    STDPRule,
    SynapseState,
)


@dataclass(frozen=True, slots=True)
class DualLiveTrial:
    name: str
    stimulus_a: dict[int, float]
    expected_output_b: int


def _build_brain(name: str) -> BrainCore:
    synapses = [
        SynapseState(0, 2, 0.60),
        SynapseState(1, 3, 0.60),
        SynapseState(2, 4, 0.62),
        SynapseState(3, 5, 0.62),
        SynapseState(0, 3, 0.10),
        SynapseState(1, 2, 0.10),
        SynapseState(2, 5, 0.08),
        SynapseState(3, 4, 0.08),
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=16,
        weight_rule=RewardWeightRule(
            learning_rate=0.012,
            min_weight=0.0,
            max_weight=1.0,
        ),
    )
    network = SpikingNetwork.from_ids(
        range(8),
        tracker,
        threshold=0.5,
        decay=0.90,
        stdp=STDPPlasticity(
            rule=STDPRule(
                potentiation_rate=0.004,
                depression_rate=0.005,
                tau_plus=8.0,
                tau_minus=8.0,
                window=8,
            )
        ),
    )
    return BrainCore(name=name, network=network, tracker=tracker)


class DualLiveSimulation:
    """Realtime two-brain learning demo with an adaptive sparse bridge."""

    RESPONSE_STEPS = 6

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
        self.brain_a = _build_brain("A")
        self.brain_b = _build_brain("B")
        self.bridge = AdaptiveBridge(
            [
                BridgeSynapseState("A", 4, "B", 0, 0.70),
                BridgeSynapseState("A", 5, "B", 1, 0.70),
                BridgeSynapseState("A", 6, "B", 6, 0.05),
                BridgeSynapseState("B", 7, "A", 7, 0.05),
            ],
            reward_window=16,
            reward_alpha=0.08,
            learning_rate=0.015,
            min_weight=0.0,
            max_weight=1.0,
        )
        self.system = DualBrainSystem(self.brain_a, self.brain_b, self.bridge)
        self.brains = {"A": self.brain_a, "B": self.brain_b}

        self.modules = {
            "A": AdaptiveModuleManager(range(8), module_count=2),
            "B": AdaptiveModuleManager(range(8), module_count=2),
        }
        self.consolidators = {
            "A": MemoryConsolidator(
                config=ConsolidationConfig(
                    min_usage=3,
                    min_reward=0.03,
                    growth_rate=0.07,
                    decay_rate=0.001,
                )
            ),
            "B": MemoryConsolidator(
                config=ConsolidationConfig(
                    min_usage=3,
                    min_reward=0.03,
                    growth_rate=0.07,
                    decay_rate=0.001,
                )
            ),
        }
        self.bridge_candidates = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(
                max_candidates=48,
                pool_size=8,
            )
        )
        self.bridge_structural = BridgeStructuralPlasticityManager(
            self.bridge,
            {"A": range(8), "B": range(8)},
            config=BridgeStructuralConfig(
                min_age_cycles=1,
                stale_steps=12,
                reward_threshold=0.01,
                protected_stability=0.75,
                max_rewire_per_cycle=1,
                regrow_weight=0.08,
            ),
        )

        self.trials = (
            DualLiveTrial("pattern_A", {0: 1.0}, 4),
            DualLiveTrial("pattern_B", {1: 1.0}, 5),
        )
        self.trial_index = 0
        self.phase = 0
        self.output_counts = {4: 0, 5: 0}
        self._trial_fired = {"A": set(), "B": set()}
        self.last_reward = 0.0
        self.last_prediction: int | None = None
        self.last_expected = self.trials[0].expected_output_b
        self.last_correct = False
        self.last_bridge_rewire: dict[str, Any] | None = None
        self.last_reward_credit = {"A": 0, "B": 0, "bridge": 0}
        self.completed_trials = 0
        self.correct_trials = 0
        self.last_result = None
        self.history: deque[dict[str, float | int]] = deque(maxlen=180)

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
            for brain in self.brains.values():
                brain.network.reset_state(clear_spike_history=True)
                brain.tracker.clear_recent()
            self.bridge.clear_recent()
            self.output_counts = {4: 0, 5: 0}
            self._trial_fired = {"A": set(), "B": set()}
            stimulus_a = trial.stimulus_a
        else:
            stimulus_a = None

        result = self.system.step(currents_a=stimulus_a)
        self.last_result = result
        self._trial_fired["A"].update(result.brain_a.fired)
        self._trial_fired["B"].update(result.brain_b.fired)

        self.modules["A"].observe(result.brain_a.fired, context=trial.name)
        self.modules["B"].observe(result.brain_b.fired, context=trial.name)

        for neuron_id in result.brain_b.fired:
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

        correct = prediction == trial.expected_output_b
        reward = 1.0 if correct else -0.55
        credited_a, credited_b, credited_bridge = self.system.apply_reward(reward)
        self.last_reward_credit = {
            "A": credited_a,
            "B": credited_b,
            "bridge": credited_bridge,
        }

        for name, brain in self.brains.items():
            self.consolidators[name].consolidate(brain.tracker.synapses)
            self.modules[name].observe(
                self._trial_fired[name],
                reward=reward,
                context=trial.name,
            )

        self.last_prediction = prediction
        self.last_expected = trial.expected_output_b
        self.last_correct = correct
        self.last_reward = reward
        self.completed_trials += 1
        if correct:
            self.correct_trials += 1

        self.last_bridge_rewire = None
        if self.completed_trials % 10 == 0:
            candidates = self.bridge_candidates.generate(
                self.brains,
                self.bridge,
                step=self.system.step_index,
            )
            rewired = self.bridge_structural.rewire(
                step=self.system.step_index,
                candidate_keys=candidates,
            )
            if rewired.changed:
                self.last_bridge_rewire = {
                    "pruned": [list(edge) for edge in rewired.pruned],
                    "regrown": [list(edge) for edge in rewired.regrown],
                }

        internal = self.brain_a.tracker.synapses + self.brain_b.tracker.synapses
        bridge_synapses = self.bridge.synapses
        all_weights = [s.weight for s in internal] + [s.weight for s in bridge_synapses]
        all_stability = [s.stability for s in internal]
        self.history.append(
            {
                "step": self.system.step_index,
                "reward": reward,
                "accuracy": self.accuracy,
                "mean_weight": fmean(all_weights) if all_weights else 0.0,
                "mean_stability": fmean(all_stability) if all_stability else 0.0,
                "bridge_weight": (
                    fmean(s.weight for s in bridge_synapses) if bridge_synapses else 0.0
                ),
            }
        )

        self.phase = 0
        self.trial_index = (self.trial_index + 1) % len(self.trials)

    @property
    def accuracy(self) -> float:
        if self.completed_trials == 0:
            return 0.0
        return self.correct_trials / self.completed_trials

    def _brain_snapshot(self, name: str) -> dict[str, Any]:
        brain = self.brains[name]
        module_manager = self.modules[name]
        if self.last_result is None:
            result = None
        elif name == "A":
            result = self.last_result.brain_a
        else:
            result = self.last_result.brain_b
        fired = set(result.fired) if result is not None else set()

        if name == "A":
            role_map = {0: "input", 1: "input", 4: "gateway", 5: "gateway", 6: "spare", 7: "spare"}
        else:
            role_map = {0: "bridge-in", 1: "bridge-in", 4: "output", 5: "output", 6: "spare", 7: "spare"}

        synapses = brain.tracker.synapses
        return {
            "name": name,
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
                    "role": role_map.get(neuron.neuron_id, "internal"),
                    "module": module_manager.module_of(neuron.neuron_id),
                }
                for neuron in brain.network.neurons.values()
            ],
            "synapses": [
                {
                    "pre": synapse.pre_id,
                    "post": synapse.post_id,
                    "weight": synapse.weight,
                    "stability": synapse.stability,
                    "usage": synapse.usage_count,
                    "reward_ema": synapse.reward_ema,
                    "active": synapse.last_used_step == brain.network.step_index - 1,
                }
                for synapse in synapses
            ],
            "modules": [
                {
                    "id": module_id,
                    "neurons": sorted(state.neuron_ids),
                    "activity": state.activity_ema,
                    "reward": state.reward_ema,
                }
                for module_id, state in sorted(module_manager.modules.items())
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        current_trial = self.trials[self.trial_index]
        bridge_synapses = self.bridge.synapses
        bridge_active_step = self.system.step_index - 1
        result = self.last_result
        bridge_transfers = result.bridge_transfers if result is not None else 0

        brain_a = self._brain_snapshot("A")
        brain_b = self._brain_snapshot("B")
        total_stdp = brain_a["metrics"]["stdp_updates"] + brain_b["metrics"]["stdp_updates"]
        total_fired = brain_a["metrics"]["fired_count"] + brain_b["metrics"]["fired_count"]
        total_internal_transfers = (
            brain_a["metrics"]["transferred_synapses"]
            + brain_b["metrics"]["transferred_synapses"]
        )
        internal_synapses = self.brain_a.tracker.synapses + self.brain_b.tracker.synapses
        all_weights = [s.weight for s in internal_synapses] + [s.weight for s in bridge_synapses]

        current_specialization = fmean(
            self.modules[name].context_specialization(current_trial.name)
            for name in ("A", "B")
        )

        return {
            "mode": "dual",
            "step": self.system.step_index,
            "running": self.running,
            "hz": self.hz,
            "phase": self.phase,
            "task": current_trial.name,
            "expected_output": current_trial.expected_output_b,
            "last_expected": self.last_expected,
            "last_prediction": self.last_prediction,
            "last_reward": self.last_reward,
            "last_correct": self.last_correct,
            "completed_trials": self.completed_trials,
            "accuracy": self.accuracy,
            "last_rewire": self.last_bridge_rewire,
            "reward_credit": dict(self.last_reward_credit),
            "metrics": {
                "fired_count": total_fired,
                "transferred_synapses": total_internal_transfers + bridge_transfers,
                "stdp_updates": total_stdp,
                "synapse_count": len(internal_synapses) + len(bridge_synapses),
                "mean_weight": fmean(all_weights) if all_weights else 0.0,
                "mean_stability": (
                    fmean(s.stability for s in internal_synapses) if internal_synapses else 0.0
                ),
                "bridge_transfers": bridge_transfers,
                "bridge_mean_weight": (
                    fmean(s.weight for s in bridge_synapses) if bridge_synapses else 0.0
                ),
                "module_specialization": current_specialization,
            },
            "brains": [brain_a, brain_b],
            "bridge": {
                "synapse_count": len(bridge_synapses),
                "transfers": bridge_transfers,
                "mean_weight": (
                    fmean(s.weight for s in bridge_synapses) if bridge_synapses else 0.0
                ),
                "synapses": [
                    {
                        "source_brain": synapse.source_brain,
                        "pre": synapse.pre_id,
                        "target_brain": synapse.target_brain,
                        "post": synapse.post_id,
                        "weight": synapse.weight,
                        "stability": synapse.stability,
                        "usage": synapse.usage_count,
                        "reward_ema": synapse.reward_ema,
                        "age": synapse.age,
                        "active": synapse.last_used_step == bridge_active_step,
                    }
                    for synapse in bridge_synapses
                ],
            },
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


class DualLiveRequestHandler(BaseHTTPRequestHandler):
    simulation: DualLiveSimulation
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
    return Path(__file__).with_name("dual_live_dashboard.html").read_bytes()


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    hz: float = 8.0,
    open_browser: bool = True,
) -> None:
    simulation = DualLiveSimulation(hz=hz)
    handler = type(
        "DrosoMathDualLiveHandler",
        (DualLiveRequestHandler,),
        {"simulation": simulation, "dashboard_html": _dashboard_html()},
    )
    server = ThreadingHTTPServer((host, port), handler)
    simulation.start()
    url = f"http://{host}:{port}/"
    print(f"DrosoMath Dual Live: {url}")
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
    parser = argparse.ArgumentParser(description="DrosoMath realtime dual-brain dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        simulation = DualLiveSimulation(hz=args.hz)
        snapshot = simulation.snapshot()
        for _ in range(18):
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
