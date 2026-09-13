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


class RunLogger:
    """Persist lightweight experiment results for later Git/GitHub review.

    We intentionally do not log per-neuron activity here. The committed run
    artifacts stay small and focus on experiment outcomes and learning metrics.
    """

    def __init__(self, config: dict[str, Any], runs_dir: Path | None = None) -> None:
        self.started_at = _now()
        self.run_id = self.started_at.strftime("%Y%m%d_%H%M%S%z")
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
                    "problem",
                    "answer",
                    "correct",
                    "reward",
                    "overall",
                    "recent_20",
                    "recent_100",
                    "recent_500",
                    "successes",
                    "attempts",
                    "active_synapses",
                    "mean_delta_w",
                ]
            )

        self._write_summary(status="running")

    def record(self, frame: dict[str, Any], metrics: dict[str, Any]) -> None:
        plasticity = frame.get("plasticity", {})
        with self._metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    frame.get("trial"),
                    frame.get("timestamp"),
                    frame.get("problem"),
                    frame.get("answer"),
                    int(bool(frame.get("correct"))),
                    frame.get("reward"),
                    metrics.get("overall"),
                    metrics.get("recent_20"),
                    metrics.get("recent_100"),
                    metrics.get("recent_500"),
                    metrics.get("successes"),
                    metrics.get("attempts"),
                    plasticity.get("active_synapses"),
                    plasticity.get("mean_delta_w"),
                ]
            )

        self._last_frame = frame
        self._last_metrics = metrics
        self._rows += 1

        # Keep summary reasonably fresh without rewriting it at 10 Hz forever.
        if self._rows == 1 or self._rows % 10 == 0:
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
        }
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
            "layout_source": self.config.get("layout_source"),
            "layout_count": self.config.get("layout_count"),
            "successes": metrics.get("successes", 0),
            "attempts": metrics.get("attempts", 0),
            "overall_accuracy": metrics.get("overall"),
            "recent_20": metrics.get("recent_20"),
            "recent_100": metrics.get("recent_100"),
            "recent_500": metrics.get("recent_500"),
            "last_problem": last.get("problem"),
            "last_answer": last.get("answer"),
            "last_correct": last.get("correct"),
            "last_reward": last.get("reward"),
            "note": "Mock telemetry is UI/pipeline validation only; do not interpret it as learned behavior."
            if self.config.get("telemetry_source") == "mock"
            else None,
        }
        self._write_json(self._summary_path, summary)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
