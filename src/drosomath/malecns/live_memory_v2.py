from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

from .curriculum_v2_memory import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PROGRESS,
    DEFAULT_READOUT_DIR,
    DEFAULT_RESULT,
    DEFAULT_V1_BASELINE,
    MemoryV2Config,
    run_memory_curriculum,
)
from .download import DEFAULT_DATA_DIR, download_malecns
from .live_training import LiveTrainingHandler, LiveTrainingState, _dashboard_html
from .loader import load_malecns_v1
from .morphology import MorphologySpace, SkeletonCache
from .visualize_memory_v2 import DEFAULT_HTML, build_memory_v2_html


class LiveMemoryV2Runner:
    """Phase-1 memory curriculum with the existing real-coordinate 3-D telemetry UI."""

    def __init__(
        self,
        *,
        data_dir: Path,
        config: MemoryV2Config,
        result_path: Path = DEFAULT_RESULT,
        html_path: Path = DEFAULT_HTML,
        progress_path: Path = DEFAULT_PROGRESS,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        readout_dir: Path = DEFAULT_READOUT_DIR,
        baseline_path: Path = DEFAULT_V1_BASELINE,
        download: bool = False,
        resume: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.result_path = Path(result_path)
        self.html_path = Path(html_path)
        self.progress_path = Path(progress_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.readout_dir = Path(readout_dir)
        self.baseline_path = Path(baseline_path)
        self.download = bool(download)
        self.resume = bool(resume)
        self.state = LiveTrainingState(history_limit=2048)
        self.morphology: MorphologySpace | None = None
        self.skeletons: SkeletonCache | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def geometry_payload(self, *, max_points: int = 30_000) -> dict[str, object]:
        if self.morphology is None:
            raise RuntimeError("MaleCNS morphology is still loading or unavailable")
        return self.morphology.point_cloud(max_points=max_points)

    def skeleton_payload(self, body_id: int, *, max_segments: int = 3500) -> dict[str, object]:
        if self.skeletons is None:
            raise RuntimeError("MaleCNS morphology is still loading or unavailable")
        return self.skeletons.payload(body_id, max_segments=max_segments)

    def _run(self) -> None:
        try:
            if self.download:
                download_malecns(self.data_dir)
            connectome = load_malecns_v1(
                self.data_dir,
                min_connection_synapses=self.config.min_connection_synapses,
            )

            try:
                self.morphology = MorphologySpace.from_annotations(connectome.body_ids, self.data_dir)
                self.skeletons = SkeletonCache(self.morphology, self.data_dir)
                self.state.set_morphology(self.morphology)
            except BaseException as exc:
                self.state.set_morphology_error(exc)

            baseline = None
            if self.baseline_path.is_file():
                baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))

            report = run_memory_curriculum(
                connectome,
                config=self.config,
                checkpoint_path=self.checkpoint_path,
                readout_dir=self.readout_dir,
                progress_path=self.progress_path,
                resume=self.resume,
                observer=self.state.update,
                live_telemetry=True,
                v1_baseline=baseline,
            )
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            self.html_path.parent.mkdir(parents=True, exist_ok=True)
            self.html_path.write_text(build_memory_v2_html(report), encoding="utf-8")
            self.state.set_finished(report_path=self.result_path, dashboard_path=self.html_path)
        except BaseException as exc:
            self.state.set_error(exc)


def run_memory_server(
    *,
    runner: LiveMemoryV2Runner,
    host: str = "127.0.0.1",
    port: int = 8772,
    open_browser: bool = True,
) -> None:
    handler = type(
        "MaleCNSMemoryV2Handler",
        (LiveTrainingHandler,),
        {"runner": runner, "dashboard_html": _dashboard_html()},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"MaleCNS Memory v2 live training: {url}")
    print("Protected plasticity + replay + consolidation on the full connectome.")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    runner.start()

    def stop_when_done() -> None:
        while not runner.state.finished:
            time.sleep(0.25)
        # Give the browser one final SSE/state refresh before the live server exits.
        time.sleep(0.75)
        server.shutdown()

    threading.Thread(target=stop_when_done, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        runner.state.control("stop")
    finally:
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase-1 MaleCNS memory v2 with realtime 3-D telemetry")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--stage-trials", type=int, default=256)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--validation-trials", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--replay-interval", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8772)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    p.add_argument("--v1-baseline", type=Path, default=DEFAULT_V1_BASELINE)
    a = p.parse_args()

    if a.fresh:
        for path in (a.checkpoint, a.progress, a.result, a.html):
            if path.is_file():
                path.unlink()
        if a.readout_dir.is_dir():
            shutil.rmtree(a.readout_dir)

    config = MemoryV2Config(
        min_connection_synapses=a.min_syn,
        stage_trials=a.stage_trials,
        decoder_epochs=a.decoder_epochs,
        validation_trials_per_label=a.validation_trials,
        checkpoint_every=a.checkpoint_every,
        replay_interval=a.replay_interval,
        seed=a.seed,
    )
    runner = LiveMemoryV2Runner(
        data_dir=a.data_dir,
        config=config,
        result_path=a.result,
        html_path=a.html,
        progress_path=a.progress,
        checkpoint_path=a.checkpoint,
        readout_dir=a.readout_dir,
        baseline_path=a.v1_baseline,
        download=a.download,
        resume=not a.fresh,
    )
    run_memory_server(
        runner=runner,
        host=a.host,
        port=a.port,
        open_browser=not a.no_browser,
    )
    if runner.state.error:
        raise RuntimeError(runner.state.error)


if __name__ == "__main__":
    main()
