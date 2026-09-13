from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path
from typing import Any

FAFB_V783_TOTAL_NEURONS = 139_255

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "flywire" / "fafb783"


def data_dir() -> Path:
    configured = os.environ.get("DROSOMATH_FLYWIRE_DIR")
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_DATA_DIR


def _iter_gzip_csv(path: Path):
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def _parse_position(raw: str) -> tuple[float, float, float] | None:
    parts = raw.strip().strip("[]").replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def load_fafb_soma_layout() -> dict[str, Any] | None:
    """Load real FAFB v783 soma positions when the Codex dumps are present.

    coordinates.csv.gz contains soma coordinates only, so this layout is a real
    anatomical subset of the 139,255-neuron connectome rather than fabricated
    fallback positions for cells without soma coordinates.
    """
    root = data_dir()
    coordinates_path = root / "coordinates.csv.gz"
    classification_path = root / "classification.csv.gz"

    if not coordinates_path.exists() or not classification_path.exists():
        return None

    classification: dict[str, tuple[str, str]] = {}
    for row in _iter_gzip_csv(classification_path):
        root_id = (row.get("root_id") or "").strip()
        if not root_id:
            continue
        classification[root_id] = (
            (row.get("super_class") or "central").strip() or "central",
            (row.get("side") or "").strip(),
        )

    positions: dict[str, tuple[float, float, float]] = {}
    for row in _iter_gzip_csv(coordinates_path):
        root_id = (row.get("root_id") or "").strip()
        if not root_id or root_id in positions:
            continue
        parsed = _parse_position(row.get("position") or "")
        if parsed is not None:
            positions[root_id] = parsed

    usable_ids = sorted(set(positions).intersection(classification), key=int)
    if not usable_ids:
        raise RuntimeError(
            f"FlyWire files were found in {root}, but no usable coordinates matched classification rows."
        )

    xs = [positions[root_id][0] for root_id in usable_ids]
    ys = [positions[root_id][1] for root_id in usable_ids]
    zs = [positions[root_id][2] for root_id in usable_ids]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    longest_span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    scale = 20.0 / longest_span if longest_span else 1.0

    neurons: list[dict[str, Any]] = []
    for local_id, root_id in enumerate(usable_ids):
        x_nm, y_nm, z_nm = positions[root_id]
        super_class, side = classification[root_id]
        neurons.append(
            {
                # Browser-safe dense index. FlyWire root IDs exceed JS safe integer range,
                # so the true ID is kept as a string in root_id.
                "id": local_id,
                "root_id": root_id,
                "x": round((x_nm - cx) * scale, 4),
                "y": round(-(y_nm - cy) * scale, 4),
                "z": round(-(z_nm - cz) * scale, 4),
                "region": super_class,
                "super_class": super_class,
                "side": side,
            }
        )

    return {
        "neurons": neurons,
        "source": "FlyWire Codex FAFB v783 soma coordinates",
        "count": len(neurons),
        "total_connectome_neurons": FAFB_V783_TOTAL_NEURONS,
        "coordinate_kind": "soma",
        "data_dir": str(root),
    }
