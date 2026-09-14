from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from typing import Callable

import numpy as np

from .dual_brain_v2 import DualBrainV2, DualBrainV2Config


TaskSampler = Callable[[np.random.Generator], tuple[list[str], int, int]]


def identity_task(rng: np.random.Generator) -> tuple[list[str], int, int]:
    value = int(rng.integers(0, 10))
    return [str(value)], value, 10


def next_task(rng: np.random.Generator) -> tuple[list[str], int, int]:
    value = int(rng.integers(0, 9))
    return [str(value), "NEXT"], value + 1, 10


def previous_task(rng: np.random.Generator) -> tuple[list[str], int, int]:
    value = int(rng.integers(1, 10))
    return [str(value), "PREV"], value - 1, 10


TASKS = [
    ("identity", identity_task),
    ("next", next_task),
    ("previous", previous_task),
]


def train_block(
    model: DualBrainV2,
    sampler: TaskSampler,
    task_name: str,
    steps: int,
) -> None:
    for _ in range(steps):
        tokens, target, outputs = sampler(model.rng)
        model.step(
            tokens,
            target,
            active_outputs=outputs,
            task_name=task_name,
            learn=True,
            sample_action=True,
        )


def evaluate(
    model: DualBrainV2,
    sampler: TaskSampler,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(trials):
        tokens, target, outputs = sampler(rng)
        result = model.step(
            tokens,
            target,
            active_outputs=outputs,
            learn=False,
            sample_action=False,
        )
        correct += int(result["correct"])
    return correct / float(trials)


def run_sequential_retention(
    config: DualBrainV2Config,
    *,
    steps_per_task: int,
    eval_trials: int,
) -> dict[str, object]:
    """Train sequential tasks and measure catastrophic forgetting.

    Task labels are used only to group telemetry. They are not passed into the
    routing or plasticity rules.
    """

    model = DualBrainV2(config)
    history: list[dict[str, object]] = []
    peak: dict[str, float] = {}

    for phase, (task_name, sampler) in enumerate(TASKS, start=1):
        train_block(model, sampler, task_name, steps_per_task)
        scores: dict[str, float] = {}
        for offset, (eval_name, eval_sampler) in enumerate(TASKS):
            score = evaluate(
                model,
                eval_sampler,
                eval_trials,
                config.seed + phase * 100 + offset,
            )
            scores[eval_name] = score
            if eval_name == task_name:
                peak[eval_name] = max(
                    peak.get(eval_name, 0.0),
                    score,
                )

        retention: dict[str, float | None] = {}
        for eval_name, score in scores.items():
            best = peak.get(eval_name)
            retention[eval_name] = (
                None if not best else score / best
            )

        history.append(
            {
                "after_training": task_name,
                "accuracy": scores,
                "retention_vs_peak": retention,
                "bridge": model.bridge_bank.snapshot(),
                "module_usage": model.module_usage_ema.tolist(),
                "task_activity": {
                    name: values.tolist()
                    for name, values in sorted(
                        model.task_activity_ema.items()
                    )
                },
            }
        )

    return {
        "config": asdict(config),
        "resources": model.resource_report(),
        "history": history,
        "topology": model.topology_snapshot(
            max_edges_per_bank=8
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DrosoMath DualBrainV2 continual-memory benchmark"
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="training trials per sequential task",
    )
    parser.add_argument(
        "--eval-trials",
        type=int,
        default=300,
        help="deterministic evaluation trials per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260914,
    )
    parser.add_argument(
        "--bridge",
        type=int,
        default=256,
        help="fixed cross-brain synapse budget",
    )
    parser.add_argument("--no-rewire", action="store_true")
    args = parser.parse_args()

    config = DualBrainV2Config(
        seed=args.seed,
        bridge_synapses=args.bridge,
    )
    if args.no_rewire:
        config = replace(config, rewire_fraction=0.0)

    result = run_sequential_retention(
        config,
        steps_per_task=args.steps,
        eval_trials=args.eval_trials,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
