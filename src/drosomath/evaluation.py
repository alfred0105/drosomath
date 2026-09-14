from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TaskScore:
    task: str
    score: float


@dataclass(slots=True)
class ContinualMemoryEvaluator:
    """Track retention and catastrophic forgetting across sequential tasks."""

    initial_scores: dict[str, float] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)
    latest_scores: dict[str, float] = field(default_factory=dict)
    history: list[tuple[int, tuple[TaskScore, ...]]] = field(default_factory=list)

    def record(self, *, stage: int, scores: dict[str, float]) -> None:
        if stage < 0:
            raise ValueError("stage must be >= 0")
        normalized: list[TaskScore] = []
        for task, score in scores.items():
            value = float(score)
            self.initial_scores.setdefault(task, value)
            self.best_scores[task] = max(self.best_scores.get(task, value), value)
            self.latest_scores[task] = value
            normalized.append(TaskScore(task=task, score=value))
        self.history.append((stage, tuple(sorted(normalized, key=lambda item: item.task))))

    def retention(self, task: str) -> float:
        """Latest score / first measured score; 1.0 means fully retained."""
        initial = self.initial_scores[task]
        latest = self.latest_scores[task]
        if initial == 0.0:
            return 1.0 if latest == 0.0 else float("inf")
        return latest / initial

    def forgetting(self, task: str) -> float:
        """Drop from the best historical score to the latest score."""
        return max(0.0, self.best_scores[task] - self.latest_scores[task])

    def mean_retention(self) -> float:
        if not self.latest_scores:
            return 0.0
        values = [self.retention(task) for task in self.latest_scores]
        finite = [value for value in values if value != float("inf")]
        return sum(finite) / len(finite) if finite else float("inf")

    def mean_forgetting(self) -> float:
        if not self.latest_scores:
            return 0.0
        return sum(self.forgetting(task) for task in self.latest_scores) / len(
            self.latest_scores
        )

    def summary(self) -> dict[str, object]:
        return {
            "tasks": len(self.latest_scores),
            "mean_retention": self.mean_retention(),
            "mean_forgetting": self.mean_forgetting(),
            "retention": {
                task: self.retention(task) for task in sorted(self.latest_scores)
            },
            "forgetting": {
                task: self.forgetting(task) for task in sorted(self.latest_scores)
            },
        }
