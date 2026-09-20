"""Compare single-worker and process-parallel F.3R artifacts exactly."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def _scientific_projection(value):
    value = copy.deepcopy(value)
    value.pop("protocol", None)
    value.pop("performance", None)
    value.pop("parallelism", None)
    value.pop("pass", None)
    value.get("representation_comparison", {}).pop("workers", None)
    return value


def compare(worker_1: Path, worker_3: Path, final: Path) -> dict[str, object]:
    one = json.loads(worker_1.read_text(encoding="utf-8"))
    three = json.loads(worker_3.read_text(encoding="utf-8"))
    equivalent = _scientific_projection(one) == _scientific_projection(three)
    three["parallelism"] = {
        "worker_1_runtime_seconds": float(one["performance"]["total_runtime_seconds"]),
        "worker_3_runtime_seconds": float(three["performance"]["total_runtime_seconds"]),
        "wall_clock_speedup_worker_3_vs_worker_1": float(one["performance"]["total_runtime_seconds"] / max(three["performance"]["total_runtime_seconds"], 1e-12)),
        "scientific_equivalence": bool(equivalent),
        "deterministic_result_ordering": bool([row["seed"] for row in one["representation_comparison"]["per_seed"]] == [233, 239, 241] and [row["seed"] for row in three["representation_comparison"]["per_seed"]] == [233, 239, 241]),
        "workers_tested": [1, 3],
        "peak_memory": "not sampled by runner; three independent connectome workers were used",
    }
    three["pass"] = bool(three["pass"] and equivalent)
    final.write_text(json.dumps(three, indent=2, sort_keys=True), encoding="utf-8")
    return three["parallelism"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-1", type=Path, required=True)
    parser.add_argument("--worker-3", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.worker_1, args.worker_3, args.final), indent=2))
