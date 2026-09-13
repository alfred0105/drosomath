from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _balanced_accuracy(metrics: dict[str, Any]) -> float | None:
    by_target = metrics.get("by_target_accuracy", {})
    values = [by_target.get(str(i)) for i in range(3)]
    valid = [float(v) for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _one_vs_two_accuracy(metrics: dict[str, Any]) -> float | None:
    confusion = metrics.get("confusion_matrix") or []
    if len(confusion) < 3 or len(confusion[1]) < 3 or len(confusion[2]) < 3:
        return None
    attempts = sum(confusion[1]) + sum(confusion[2])
    if attempts <= 0:
        return None
    correct = confusion[1][1] + confusion[2][2]
    return correct / attempts


class RunLogger:
    """Persist compact experiment outcomes for later Git/GitHub review."""

    def __init__(self, config: dict[str, Any], runs_dir: Path | None = None) -> None:
        self.started_at = _now()
        self.run_id = self.started_at.strftime("%Y%m%d_%H%M%S_%f%z")
        self.run_dir = (runs_dir or DEFAULT_RUNS_DIR) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.config = dict(config)
        self.config["run_id"] = self.run_id
        self.config["started_at"] = _iso(self.started_at)

        self._metrics_path = self.run_dir / "metrics.csv"
        self._summary_path = self.run_dir / "summary.json"
        self._config_path = self.run_dir / "config.json"
        self._last_frame: dict[str, Any] | None = None
        self._last_metrics: dict[str, Any] | None = None
        self._rows = 0

        self._write_json(self._config_path, self.config)
        with self._metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "trial",
                    "timestamp",
                    "phase",
                    "trial_kind",
                    "learning_enabled",
                    "target",
                    "answer",
                    "correct",
                    "reward",
                    "p0",
                    "p1",
                    "p2",
                    "policy_entropy",
                    "stimulus_profile",
                    "signal_energy",
                    "area_factor",
                    "requested_area_factor",
                    "post_noise_energy",
                    "min_pair_distance",
                    "gain_ratio",
                    "overall",
                    "balanced_accuracy",
                    "one_vs_two_accuracy",
                    "recent_20",
                    "recent_100",
                    "recent_500",
                    "successes",
                    "attempts",
                    "target_0_accuracy",
                    "target_1_accuracy",
                    "target_2_accuracy",
                    "probe_overall",
                    "probe_balanced_accuracy",
                    "probe_one_vs_two_accuracy",
                    "probe_recent_20",
                    "probe_recent_100",
                    "probe_recent_500",
                    "probe_successes",
                    "probe_attempts",
                    "probe_target_0_accuracy",
                    "probe_target_1_accuracy",
                    "probe_target_2_accuracy",
                    "active_synapses",
                    "mean_delta_w",
                ]
            )

        self._write_summary(status="running")

    def record(self, frame: dict[str, Any], metrics: dict[str, Any]) -> None:
        plasticity = frame.get("plasticity", {})
        policy = frame.get("policy", {})
        by_target = metrics.get("by_target_accuracy", {})
        probe = metrics.get("probe", {})
        probe_by_target = probe.get("by_target_accuracy", {})
        controls = (frame.get("stimulus") or {}).get("controls", {})
        balanced = _balanced_accuracy(metrics)
        one_vs_two = _one_vs_two_accuracy(metrics)
        probe_balanced = _balanced_accuracy(probe)
        probe_one_vs_two = _one_vs_two_accuracy(probe)

        with self._metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    frame.get("trial"),
                    frame.get("timestamp"),
                    frame.get("phase"),
                    frame.get("trial_kind"),
                    int(bool(frame.get("learning_enabled"))),
                    frame.get("target"),
                    frame.get("answer"),
                    int(bool(frame.get("correct"))),
                    frame.get("reward"),
                    policy.get("p0"),
                    policy.get("p1"),
                    policy.get("p2"),
                    policy.get("entropy"),
                    controls.get("stimulus_profile"),
                    controls.get("signal_energy"),
                    controls.get("area_factor"),
                    controls.get("requested_area_factor"),
                    controls.get("post_noise_energy"),
                    controls.get("min_pair_distance"),
                    controls.get("gain_ratio"),
                    metrics.get("overall"),
                    balanced,
                    one_vs_two,
                    metrics.get("recent_20"),
                    metrics.get("recent_100"),
                    metrics.get("recent_500"),
                    metrics.get("successes"),
                    metrics.get("attempts"),
                    by_target.get("0"),
                    by_target.get("1"),
                    by_target.get("2"),
                    probe.get("overall"),
                    probe_balanced,
                    probe_one_vs_two,
                    probe.get("recent_20"),
                    probe.get("recent_100"),
                    probe.get("recent_500"),
                    probe.get("successes"),
                    probe.get("attempts"),
                    probe_by_target.get("0"),
                    probe_by_target.get("1"),
                    probe_by_target.get("2"),
                    plasticity.get("active_synapses"),
                    plasticity.get("mean_delta_w"),
                ]
            )

        self._last_frame = frame
        self._last_metrics = metrics
        self._rows += 1

        if self._rows == 1 or self._rows % 25 == 0:
            self._write_summary(status="running")

    def finalize(self, status: str = "completed") -> None:
        self._write_summary(status=status, ended_at=_now())

    def _write_summary(self, status: str, ended_at: datetime | None = None) -> None:
        last = self._last_frame or {}
        metrics = self._last_metrics or {
            "overall": None,
            "recent_20": None,
            "recent_100": None,
            "recent_500": None,
            "successes": 0,
            "attempts": 0,
            "by_target_accuracy": {"0": None, "1": None, "2": None},
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "probe": {
                "overall": None,
                "recent_20": None,
                "recent_100": None,
                "recent_500": None,
                "successes": 0,
                "attempts": 0,
                "by_target_accuracy": {"0": None, "1": None, "2": None},
                "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            },
        }
        probe = metrics.get("probe", {})
        controls = (last.get("stimulus") or {}).get("controls", {})
        now = ended_at or _now()
        summary = {
            "run_id": self.run_id,
            "status": status,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(now),
            "ended_at": _iso(ended_at) if ended_at else None,
            "duration_seconds": round((now - self.started_at).total_seconds(), 3),
            "experiment": self.config.get("experiment"),
            "telemetry_source": self.config.get("telemetry_source"),
            "activity_source": self.config.get("activity_source"),
            "learning_model": self.config.get("learning_model"),
            "chance_accuracy": self.config.get("chance_accuracy"),
            "layout_source": self.config.get("layout_source"),
            "layout_count": self.config.get("layout_count"),
            "stage21_replay_trials": self.config.get("stage21_replay_trials"),
            "stage21_state_source": self.config.get("stage21_state_source"),
            "stage21_pretraining_snapshot": self.config.get("stage21_pretraining_snapshot"),
            "evaluation_profile": self.config.get("evaluation_profile"),
            "evaluation_learning_enabled": self.config.get("evaluation_learning_enabled"),
            "successes": metrics.get("successes", 0),
            "attempts": metrics.get("attempts", 0),
            "overall_accuracy": metrics.get("overall"),
            "balanced_accuracy": _balanced_accuracy(metrics),
            "one_vs_two_accuracy": _one_vs_two_accuracy(metrics),
            "recent_20": metrics.get("recent_20"),
            "recent_100": metrics.get("recent_100"),
            "recent_500": metrics.get("recent_500"),
            "by_target_accuracy": metrics.get("by_target_accuracy"),
            "confusion_matrix_rows_target_cols_choice": metrics.get("confusion_matrix"),
            "probe": {
                **probe,
                "balanced_accuracy": _balanced_accuracy(probe),
                "one_vs_two_accuracy": _one_vs_two_accuracy(probe),
            },
            "last_trial_kind": last.get("trial_kind"),
            "last_target": last.get("target"),
            "last_answer": last.get("answer"),
            "last_correct": last.get("correct"),
            "last_reward": last.get("reward"),
            "last_policy": last.get("policy"),
            "last_stimulus_controls": controls,
            "learner": self.config.get("learner"),
            "scientific_scope": self.config.get("scientific_scope"),
            "note": (
                "This run validates the reward-learning protocol. It is not yet evidence that the full FlyWire connectome learned numerosity."
                if self.config.get("telemetry_source") == "numerosity_prototype"
                else None
            ),
        }
        self._write_json(self._summary_path, summary)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
