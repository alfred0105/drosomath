from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .evaluation import ContinualMemoryEvaluator
from .experiments import SequentialMemoryExperiment, SequentialTask


StageHook = Callable[[SequentialMemoryExperiment, int], None]
ExperimentFactory = Callable[[], SequentialMemoryExperiment]


@dataclass(frozen=True, slots=True)
class BenchmarkVariant:
    """One independently constructed continual-learning configuration."""

    name: str
    build_experiment: ExperimentFactory
    after_stage: StageHook | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variant name must not be empty")


@dataclass(frozen=True, slots=True)
class VariantBenchmarkResult:
    name: str
    summary: dict[str, object]
    synapse_count: int


@dataclass(frozen=True, slots=True)
class ContinualBenchmarkResult:
    variants: tuple[VariantBenchmarkResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "variants": [
                {
                    "name": result.name,
                    "synapse_count": result.synapse_count,
                    **result.summary,
                }
                for result in self.variants
            ]
        }


class ContinualAblationBenchmark:
    """Run the same sequential curriculum across isolated model variants.

    Each variant receives a freshly constructed experiment, preventing learned
    weights, spike history, reward traces, or structural changes from leaking
    between configurations. Optional stage hooks allow structural-plasticity or
    maintenance policies to run after a task has been trained but before the
    retention evaluation for that stage.
    """

    def __init__(
        self,
        tasks: Iterable[SequentialTask],
        *,
        epochs_per_task: int = 1,
    ) -> None:
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("at least one task is required")
        if epochs_per_task < 1:
            raise ValueError("epochs_per_task must be >= 1")
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("task names must be unique")
        self.epochs_per_task = epochs_per_task

    def run_variant(self, variant: BenchmarkVariant) -> VariantBenchmarkResult:
        experiment = variant.build_experiment()
        evaluator = ContinualMemoryEvaluator()
        seen: list[SequentialTask] = []

        for stage, task in enumerate(self.tasks):
            experiment.train_task(task, epochs=self.epochs_per_task)
            if variant.after_stage is not None:
                variant.after_stage(experiment, stage)
            seen.append(task)
            evaluator.record(
                stage=stage,
                scores={
                    seen_task.name: experiment.evaluate_task(seen_task)
                    for seen_task in seen
                },
            )

        return VariantBenchmarkResult(
            name=variant.name,
            summary=evaluator.summary(),
            synapse_count=experiment.tracker.synapse_count,
        )

    def run(self, variants: Iterable[BenchmarkVariant]) -> ContinualBenchmarkResult:
        variants = tuple(variants)
        if not variants:
            raise ValueError("at least one benchmark variant is required")
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise ValueError("variant names must be unique")
        return ContinualBenchmarkResult(
            variants=tuple(self.run_variant(variant) for variant in variants)
        )
