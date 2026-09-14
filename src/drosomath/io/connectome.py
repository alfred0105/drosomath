from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from drosomath.core import SynapseState


@dataclass(frozen=True, slots=True)
class ConnectomeLoadConfig:
    pre_column: str = "pre_id"
    post_column: str = "post_id"
    weight_column: str | None = "weight"
    count_column: str | None = None
    default_weight: float = 0.1
    count_scale: float = 0.01
    min_count: float = 0.0
    max_edges: int | None = None

    def __post_init__(self) -> None:
        if self.default_weight < 0.0:
            raise ValueError("default_weight must be >= 0")
        if self.count_scale < 0.0:
            raise ValueError("count_scale must be >= 0")
        if self.max_edges is not None and self.max_edges < 1:
            raise ValueError("max_edges must be >= 1 when set")


def load_synapses_csv(
    path: str | Path,
    *,
    config: ConnectomeLoadConfig | None = None,
) -> tuple[SynapseState, ...]:
    """Stream an edge-list CSV into SynapseState objects.

    Supported input modes:
    - explicit weight column (default: ``weight``)
    - contact/synapse-count column converted with ``count_scale``
    - constant ``default_weight`` when neither is provided

    This loader is intended for small/medium experiments and connectome subsets.
    Full FlyWire-scale runs will use a compact sparse backend rather than tens of
    millions of Python objects.
    """

    cfg = config or ConnectomeLoadConfig()
    result: list[SynapseState] = []
    seen: set[tuple[int, int]] = set()

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header")
        required = {cfg.pre_column, cfg.post_column}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        for row in reader:
            pre_id = int(row[cfg.pre_column])
            post_id = int(row[cfg.post_column])
            key = (pre_id, post_id)
            if key in seen:
                continue

            weight = cfg.default_weight
            if cfg.weight_column and row.get(cfg.weight_column) not in (None, ""):
                weight = float(row[cfg.weight_column])
            elif cfg.count_column and row.get(cfg.count_column) not in (None, ""):
                count = float(row[cfg.count_column])
                if count < cfg.min_count:
                    continue
                weight = count * cfg.count_scale

            seen.add(key)
            result.append(SynapseState(pre_id=pre_id, post_id=post_id, weight=weight))
            if cfg.max_edges is not None and len(result) >= cfg.max_edges:
                break

    return tuple(result)


def infer_neuron_ids(synapses: Iterable[SynapseState]) -> tuple[int, ...]:
    ids = {node for synapse in synapses for node in (synapse.pre_id, synapse.post_id)}
    return tuple(sorted(ids))
