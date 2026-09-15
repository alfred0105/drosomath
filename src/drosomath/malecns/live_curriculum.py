from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import PlasticStateConfig

from .brain import PlasticMaleCNSBrain
from .checkpoint import (
    restore_learning_checkpoint,
    restore_readout_checkpoint,
    save_learning_checkpoint,
    save_readout_checkpoint,
)
from .curriculum_v1 import (
    DEFAULT_CHECKPOINT,
    DEFAULT_READOUT_DIR,
    DEFAULT_RESULT,
    CurriculumV1Config,
    _calibrate,
    _degrade,
    _evaluate,
    _make_session,
    _readout_path,
    _train_decoder,
    build_curriculum,
)
from .download import DEFAULT_DATA_DIR, download_malecns
from .live_training import LiveTrainingState, run_server
from .loader import load_malecns_v1
from .morphology import MorphologySpace, SkeletonCache
from .visualize_curriculum import DEFAULT_HTML, build_curriculum_html


class LiveCurriculumRunner:
    """Run the v1 curriculum while exposing bounded whole-CNS telemetry."""

    def __init__(
        self,
        *,
        data_dir: Path,
        config: CurriculumV1Config,
        result_path: Path = DEFAULT_RESULT,
        html_path: Path = DEFAULT_HTML,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        readout_dir: Path = DEFAULT_READOUT_DIR,
        download: bool = False,
        resume: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config
        self.result_path = Path(result_path)
        self.html_path = Path(html_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.readout_dir = Path(readout_dir)
        self.download = bool(download)
        self.resume = bool(resume)
        self.state = LiveTrainingState(history_limit=1024)
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

    def _emit(self, brain, **event: object) -> None:
        self.state.update(dict(event), brain)

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

            report = self._run_curriculum(connectome)
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            self.html_path.parent.mkdir(parents=True, exist_ok=True)
            self.html_path.write_text(build_curriculum_html(report), encoding="utf-8")
            self.state.set_finished(report_path=self.result_path, dashboard_path=self.html_path)
        except BaseException as exc:
            self.state.set_error(exc)

    def _run_curriculum(self, connectome) -> dict[str, object]:
        np = __import__("numpy")
        rng = np.random.default_rng(self.config.seed)
        tasks, output = build_curriculum(connectome, config=self.config)

        brain = PlasticMaleCNSBrain(
            connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=self.config.seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=self.config.seed),
        )
        brain.configure_live_telemetry(True, max_active_edges=128, edges_per_firing_neuron=4)

        resume_info = None
        if self.resume and self.checkpoint_path.is_file():
            resume_info = restore_learning_checkpoint(self.checkpoint_path, brain=brain)

        sessions = {}
        readouts = {}
        for i, task in enumerate(tasks):
            readout, session = _make_session(brain, output, task, self.config, self.config.seed + 100 + i)
            rp = _readout_path(self.readout_dir, task.name)
            if self.resume and rp.is_file():
                restore_readout_checkpoint(rp, readout)
            readouts[task.name] = readout
            sessions[task.name] = session

        start_stage = 0
        completed_in_stage = 0
        if resume_info:
            saved_stage = str(resume_info.get("stage", ""))
            if saved_stage == "complete":
                start_stage = len(tasks)
            else:
                for i, task in enumerate(tasks):
                    if task.name == saved_stage:
                        start_stage = i
                        completed_in_stage = int(resume_info.get("completed_trials", 0))
                        if completed_in_stage >= self.config.stage_trials:
                            start_stage = i + 1
                            completed_in_stage = 0
                        break

        stage_reports = []
        retention_history = []
        total_stages = len(tasks)

        for stage_index in range(start_stage, total_stages):
            task = tasks[stage_index]
            session = sessions[task.name]
            readout = readouts[task.name]
            unlock = brain.plasticity.set_plastic_fraction(task.plastic_fraction)
            self._emit(
                brain,
                phase=task.name,
                status="decoder",
                stage_index=stage_index + 1,
                total_stages=total_stages,
                completed_brain_trials=0,
                total_brain_trials=self.config.stage_trials,
                message=f"Stage {stage_index+1}/{total_stages}: {task.name}",
            )

            if not readout.frozen:
                decoder_report = _train_decoder(session, task, self.config, rng)
                save_readout_checkpoint(_readout_path(self.readout_dir, task.name), readout)
            else:
                decoder_report = {"restored": True, "steps": readout.train_steps}

            self._emit(brain, phase=task.name, status="calibration", stage_index=stage_index + 1, total_stages=total_stages)
            fraction, rate_hz, calibration = _calibrate(
                session, task, self.config, self.config.seed + 10000 + stage_index * 100
            )
            before_acc, before_silent, before_spikes, _ = _evaluate(
                session,
                task,
                self.config,
                fraction=fraction,
                rate_hz=rate_hz,
                seed=self.config.seed + 20000 + stage_index * 100,
            )

            start_trial = completed_in_stage if stage_index == start_stage else 0
            rows = []
            correct = 0
            silent_count = 0
            for trial in range(start_trial, self.config.stage_trials):
                label = task.labels[int(rng.integers(0, len(task.labels)))]
                stimulus = _degrade(rng, task.sample(label, rng), fraction)
                result = session.train_brain_trial(
                    stimulus_body_ids=stimulus,
                    target=label,
                    duration_ms=self.config.duration_ms,
                    stimulus_rate_hz=rate_hz,
                )
                correct += int(result.correct)
                silent_count += int(result.total_output_spikes == 0)
                row = {
                    "trial": trial + 1,
                    "target": result.target,
                    "prediction": result.prediction,
                    "correct": result.correct,
                    "confidence": result.confidence,
                    "reward": result.reward,
                    "output_spikes": result.total_output_spikes,
                    "edge_updates": result.learning["learning"]["edge_updates"],
                }
                rows.append(row)
                self._emit(
                    brain,
                    phase=task.name,
                    status="training",
                    stage_index=stage_index + 1,
                    total_stages=total_stages,
                    completed_brain_trials=trial + 1,
                    total_brain_trials=self.config.stage_trials,
                    running_accuracy=correct / max(1, len(rows)),
                    running_silent_fraction=silent_count / max(1, len(rows)),
                    selected_input_fraction=fraction,
                    selected_stimulus_rate_hz=rate_hz,
                    before_accuracy=before_acc,
                    last_trial=row,
                )

                if self.config.checkpoint_every and (trial + 1) % self.config.checkpoint_every == 0:
                    save_learning_checkpoint(
                        self.checkpoint_path,
                        brain=brain,
                        readout=readout,
                        config=self.config,
                        completed_trials=trial + 1,
                        stage=task.name,
                    )
                    save_readout_checkpoint(_readout_path(self.readout_dir, task.name), readout)

            after_acc, after_silent, after_spikes, _ = _evaluate(
                session,
                task,
                self.config,
                fraction=fraction,
                rate_hz=rate_hz,
                seed=self.config.seed + 30000 + stage_index * 100,
            )
            save_readout_checkpoint(_readout_path(self.readout_dir, task.name), readout)
            save_learning_checkpoint(
                self.checkpoint_path,
                brain=brain,
                readout=readout,
                config=self.config,
                completed_trials=self.config.stage_trials,
                stage=task.name,
            )

            stage_reports.append(
                {
                    "stage": task.name,
                    "labels": list(task.labels),
                    "plasticity_unlock": unlock,
                    "decoder": decoder_report,
                    "challenge": {"fraction": fraction, "rate_hz": rate_hz, "calibration": calibration},
                    "before_accuracy": before_acc,
                    "after_accuracy": after_acc,
                    "delta_accuracy": after_acc - before_acc,
                    "before_silent_fraction": before_silent,
                    "after_silent_fraction": after_silent,
                    "before_mean_spikes": before_spikes,
                    "after_mean_spikes": after_spikes,
                    "training_accuracy": correct / max(1, len(rows)),
                    "training_silent_fraction": silent_count / max(1, len(rows)),
                    "last_rows": rows[-16:],
                }
            )

            retention = {"after_stage": task.name, "tasks": {}}
            self._emit(brain, phase=task.name, status="retention", stage_index=stage_index + 1, total_stages=total_stages)
            for prior in tasks[: stage_index + 1]:
                acc, silent, spikes, _ = _evaluate(
                    sessions[prior.name],
                    prior,
                    self.config,
                    fraction=0.70,
                    rate_hz=205.0,
                    seed=self.config.seed + 40000 + stage_index * 100 + len(retention["tasks"]),
                )
                retention["tasks"][prior.name] = {
                    "accuracy": acc,
                    "silent_fraction": silent,
                    "mean_output_spikes": spikes,
                }
            retention_history.append(retention)
            completed_in_stage = 0

        save_learning_checkpoint(
            self.checkpoint_path,
            brain=brain,
            config=self.config,
            completed_trials=0,
            stage="complete",
        )
        plast = brain.plasticity.summary()
        changed = int(np.count_nonzero(np.abs(brain.plasticity.multiplier - 1.0) > 1e-7))
        return {
            "experiment": "malecns_curriculum_v1_live",
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {field: getattr(self.config, field) for field in self.config.__dataclass_fields__},
            "connectome": connectome.summary(),
            "resume": resume_info,
            "stages": stage_reports,
            "retention_history": retention_history,
            "final_plasticity": {**plast, "changed_edges": changed, "plastic_fraction": float(brain.plasticity.plastic_fraction)},
            "checkpoint": str(self.checkpoint_path),
            "readout_dir": str(self.readout_dir),
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full MaleCNS v1 curriculum with realtime 3D telemetry")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--stage-trials", type=int, default=256)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--validation-trials", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    a = p.parse_args()

    config = CurriculumV1Config(
        min_connection_synapses=a.min_syn,
        stage_trials=a.stage_trials,
        decoder_epochs=a.decoder_epochs,
        validation_trials_per_label=a.validation_trials,
        checkpoint_every=a.checkpoint_every,
        seed=a.seed,
    )
    runner = LiveCurriculumRunner(
        data_dir=a.data_dir,
        config=config,
        result_path=a.result,
        html_path=a.html,
        checkpoint_path=a.checkpoint,
        readout_dir=a.readout_dir,
        download=a.download,
        resume=not a.fresh,
    )
    run_server(runner=runner, port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
