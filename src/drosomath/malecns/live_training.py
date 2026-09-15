from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .adaptive_training import AdaptiveTrainingConfig, run_adaptive_training
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1
from .visualize_training import render_training_report


DEFAULT_RESULT = Path("results/latest_malecns_adaptive_training.json")
DEFAULT_HTML = Path("results/latest_malecns_adaptive_training.html")
DEFAULT_CHECKPOINT = Path("checkpoints/latest_malecns_adaptive_training.npz")


class LiveTrainingState:
    """Thread-safe bridge between adaptive training and the browser dashboard."""

    def __init__(self, *, history_limit: int = 512) -> None:
        self._lock = threading.RLock()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._pause.clear()
        self._stop.clear()
        self.started_at = time.time()
        self.finished = False
        self.error: str | None = None
        self.report_path: str | None = None
        self.dashboard_path: str | None = None
        self.latest: dict[str, Any] = {
            "phase": "loading",
            "status": "starting",
            "message": "Loading MaleCNS v1.0",
        }
        self.history: deque[dict[str, Any]] = deque(maxlen=history_limit)

    def wait_if_paused(self) -> None:
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.05)
        if self._stop.is_set():
            raise RuntimeError("live training stopped by user")

    def update(self, event: dict[str, object], brain) -> None:
        self.wait_if_paused()
        telemetry = brain.live_telemetry_snapshot()
        payload: dict[str, Any] = dict(event)
        payload["telemetry"] = telemetry
        payload["wall_time_s"] = time.time() - self.started_at

        # Keep the browser payload bounded. Full synaptic learning state stays in
        # the simulator/checkpoint; the live view receives only recent summaries.
        trial = payload.get("last_trial")
        if isinstance(trial, dict):
            self.history.append(
                {
                    "trial": trial.get("trial"),
                    "correct": trial.get("correct"),
                    "reward": trial.get("reward"),
                    "confidence": trial.get("confidence"),
                    "output_spikes": trial.get("output_spikes"),
                    "edge_updates": trial.get("edge_updates"),
                    "running_accuracy": payload.get("running_accuracy"),
                }
            )

        with self._lock:
            self.latest = payload

    def set_finished(self, *, report_path: Path, dashboard_path: Path) -> None:
        with self._lock:
            self.finished = True
            self.report_path = str(report_path)
            self.dashboard_path = str(dashboard_path)
            self.latest = {
                **self.latest,
                "phase": "complete",
                "status": "complete",
                "finished": True,
            }

    def set_error(self, exc: BaseException) -> None:
        with self._lock:
            self.error = f"{type(exc).__name__}: {exc}"
            self.finished = True
            self.latest = {
                **self.latest,
                "phase": "error",
                "status": "error",
                "error": self.error,
                "finished": True,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.latest,
                "finished": self.finished,
                "error": self.error,
                "paused": self._pause.is_set(),
                "report_path": self.report_path,
                "dashboard_path": self.dashboard_path,
                "history": list(self.history),
            }

    def control(self, action: str) -> dict[str, Any]:
        if action == "pause":
            self._pause.set()
        elif action == "resume":
            self._pause.clear()
        elif action == "stop":
            self._stop.set()
            self._pause.clear()
        else:
            raise ValueError(f"unknown action: {action}")
        return self.snapshot()


class LiveTrainingRunner:
    def __init__(
        self,
        *,
        data_dir: Path,
        config: AdaptiveTrainingConfig,
        result_path: Path,
        html_path: Path,
        checkpoint_path: Path,
        download: bool,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.result_path = Path(result_path)
        self.html_path = Path(html_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.download = bool(download)
        self.state = LiveTrainingState()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if self.download:
                download_malecns(self.data_dir)
            connectome = load_malecns_v1(
                self.data_dir,
                min_connection_synapses=self.config.min_connection_synapses,
            )
            report = run_adaptive_training(
                connectome,
                config=self.config,
                checkpoint_path=self.checkpoint_path,
                observer=self.state.update,
                live_telemetry=True,
            )
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            render_training_report(report, self.html_path)
            self.state.set_finished(
                report_path=self.result_path,
                dashboard_path=self.html_path,
            )
        except BaseException as exc:  # keep error visible to the dashboard
            self.state.set_error(exc)


class LiveTrainingHandler(BaseHTTPRequestHandler):
    runner: LiveTrainingRunner
    dashboard_html: bytes

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(self.dashboard_html, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(self.runner.state.snapshot())
            return
        if parsed.path == "/api/control":
            action = parse_qs(parsed.query).get("action", [""])[0]
            try:
                payload = self.runner.state.control(action)
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
                payload = json.dumps(self.runner.state.snapshot(), separators=(",", ":"))
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
    return Path(__file__).with_name("live_training_dashboard.html").read_bytes()


def run_server(
    *,
    runner: LiveTrainingRunner,
    host: str = "127.0.0.1",
    port: int = 8770,
    open_browser: bool = True,
) -> None:
    handler = type(
        "MaleCNSLiveTrainingHandler",
        (LiveTrainingHandler,),
        {"runner": runner, "dashboard_html": _dashboard_html()},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"MaleCNS live training: {url}")
    print("The full connectome is simulated; the browser renders a bounded active-edge sample.")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    runner.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime MaleCNS adaptive-training telemetry server")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--input-per-side", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--decoder-epochs", type=int, default=8)
    parser.add_argument("--brain-trials", type=int, default=192)
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--plastic-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    config = AdaptiveTrainingConfig(
        min_connection_synapses=args.min_syn,
        input_per_side=args.input_per_side,
        output_population_size=args.output_size,
        decoder_epochs=args.decoder_epochs,
        brain_trials=args.brain_trials,
        validation_trials_per_class=args.validation_trials,
        duration_ms=args.duration_ms,
        plastic_fraction=args.plastic_fraction,
        seed=args.seed,
    )
    runner = LiveTrainingRunner(
        data_dir=args.data_dir,
        config=config,
        result_path=args.result,
        html_path=args.html,
        checkpoint_path=args.checkpoint,
        download=args.download,
    )
    run_server(
        runner=runner,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
