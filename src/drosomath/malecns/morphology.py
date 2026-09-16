from __future__ import annotations

import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .download import DEFAULT_DATA_DIR, path_for

SKELETON_BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-swc"
)
SKELETON_CACHE_DIRNAME = "skeletons_swc"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'MaleCNS morphology needs NumPy. Install with: python -m pip install -e ".[malecns]"'
        ) from exc
    return np


def _require_feather():
    try:
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'MaleCNS morphology needs PyArrow. Install with: python -m pip install -e ".[malecns]"'
        ) from exc
    return feather


def _coerce_location(value):
    """Return a numeric (x, y, z) tuple or None from Arrow/Python location data."""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            value = (value["x"], value["y"], value["z"])
        else:
            return None
    try:
        if len(value) != 3:
            return None
        xyz = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in xyz):
        return None
    return xyz


@dataclass(slots=True)
class MorphologySpace:
    """MaleCNS soma/soma-tract positions aligned to the connectome body-ID order.

    MaleCNS SWC morphology and soma positions use the EM coordinate frame.  We
    retain raw 8-nm voxel coordinates and expose one common normalization for
    the browser.  This means a neuron point, a live synapse endpoint, and a
    downloaded SWC skeleton all occupy the same 3-D coordinate system.
    """

    body_ids: object
    xyz_voxels: object
    center_voxels: object
    scale_voxels: float
    valid_mask: object
    source: str = "MaleCNS v1.0 somaLocation/tosomaLocation"

    @classmethod
    def from_annotations(
        cls,
        body_ids,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> "MorphologySpace":
        np = _require_numpy()
        feather = _require_feather()
        body_ids = np.asarray(body_ids, dtype=np.int64)
        table = feather.read_table(path_for("annotations", data_dir), memory_map=True)
        names = table.schema.names
        if "bodyId" not in names:
            raise ValueError("MaleCNS annotation table is missing bodyId")

        loc_columns = [name for name in ("somaLocation", "tosomaLocation") if name in names]
        xyz = np.full((len(body_ids), 3), np.nan, dtype=np.float32)
        if loc_columns:
            ann_body = table["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            raw_columns = {name: table[name].to_pylist() for name in loc_columns}
            pos = np.searchsorted(body_ids, ann_body)
            in_range = pos < len(body_ids)
            rows = np.flatnonzero(in_range)
            if len(rows):
                exact = body_ids[pos[rows]] == ann_body[rows]
                rows = rows[exact]
            for row in rows:
                loc = None
                for name in ("somaLocation", "tosomaLocation"):
                    values = raw_columns.get(name)
                    if values is not None:
                        loc = _coerce_location(values[int(row)])
                        if loc is not None:
                            break
                if loc is not None:
                    xyz[int(pos[row])] = loc

        valid = np.isfinite(xyz).all(axis=1)
        if not valid.any():
            raise ValueError(
                "MaleCNS annotations did not provide usable somaLocation/tosomaLocation coordinates"
            )

        valid_xyz = xyz[valid].astype(np.float64, copy=False)
        # Robust bounds keep a handful of tract-coordinate outliers from making
        # the whole CNS tiny in the browser.
        low = np.percentile(valid_xyz, 0.5, axis=0)
        high = np.percentile(valid_xyz, 99.5, axis=0)
        center = ((low + high) * 0.5).astype(np.float32)
        span = np.maximum(high - low, 1.0)
        scale = float(np.max(span) * 0.5)

        return cls(
            body_ids=body_ids,
            xyz_voxels=xyz,
            center_voxels=center,
            scale_voxels=scale,
            valid_mask=valid,
        )

    @property
    def positioned_count(self) -> int:
        np = _require_numpy()
        return int(np.count_nonzero(self.valid_mask))

    def index_of(self, body_id: int) -> int:
        np = _require_numpy()
        pos = int(np.searchsorted(self.body_ids, int(body_id)))
        if pos >= len(self.body_ids) or int(self.body_ids[pos]) != int(body_id):
            raise KeyError(f"MaleCNS body {body_id} is not in morphology space")
        return pos

    def normalized_at_index(self, index: int) -> list[float] | None:
        np = _require_numpy()
        i = int(index)
        row = self.xyz_voxels[i]
        if not np.isfinite(row).all():
            return None
        out = (row - self.center_voxels) / self.scale_voxels
        return [float(out[0]), float(out[1]), float(out[2])]

    def normalized(self, body_id: int) -> list[float] | None:
        return self.normalized_at_index(self.index_of(body_id))

    def normalize_xyz(self, xyz) -> list[float]:
        np = _require_numpy()
        row = np.asarray(xyz, dtype=np.float32)
        out = (row - self.center_voxels) / self.scale_voxels
        return [float(out[0]), float(out[1]), float(out[2])]

    def point_cloud(
        self,
        *,
        max_points: int = 30_000,
        include_body_ids: Iterable[int] = (),
    ) -> dict[str, object]:
        np = _require_numpy()
        if max_points < 100:
            raise ValueError("max_points must be >= 100")
        valid_idx = np.flatnonzero(self.valid_mask)
        required: list[int] = []
        for body_id in include_body_ids:
            try:
                i = self.index_of(int(body_id))
            except KeyError:
                continue
            if bool(self.valid_mask[i]):
                required.append(i)

        if len(valid_idx) > max_points:
            # Deterministic even sampling preserves the gross brain/VNC shape.
            take = np.linspace(0, len(valid_idx) - 1, max_points, dtype=np.int64)
            chosen = valid_idx[take]
        else:
            chosen = valid_idx
        if required:
            chosen = np.unique(
                np.concatenate([chosen, np.asarray(required, dtype=np.int64)])
            )

        coords = (self.xyz_voxels[chosen] - self.center_voxels) / self.scale_voxels
        return {
            "source": self.source,
            "coordinate_units": "8nm_voxels",
            "positioned_neurons": self.positioned_count,
            "rendered_neurons": int(len(chosen)),
            "body_ids": [int(x) for x in self.body_ids[chosen]],
            "xyz": coords.astype(np.float32, copy=False).tolist(),
        }

    def annotate_telemetry(self, telemetry: dict[str, object]) -> dict[str, object]:
        out = dict(telemetry)
        fired_rows: list[dict[str, object]] = []
        for body_id in telemetry.get("fired_neuron_ids", []):
            xyz = self.normalized(int(body_id))
            if xyz is not None:
                fired_rows.append({"body_id": int(body_id), "xyz": xyz})
        out["fired_positions"] = fired_rows

        edge_rows: list[dict[str, object]] = []
        for edge in telemetry.get("active_edges", []):
            row = dict(edge)
            pre_xyz = self.normalized(int(row["pre_id"]))
            post_xyz = self.normalized(int(row["post_id"]))
            if pre_xyz is not None:
                row["pre_xyz"] = pre_xyz
            if post_xyz is not None:
                row["post_xyz"] = post_xyz
            edge_rows.append(row)
        out["active_edges"] = edge_rows
        return out


class SkeletonCache:
    """On-demand public MaleCNS SWC downloader and compact browser serializer."""

    def __init__(
        self,
        morphology: MorphologySpace,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        self.morphology = morphology
        self.cache_dir = Path(data_dir) / SKELETON_CACHE_DIRNAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, body_id: int) -> Path:
        return self.cache_dir / f"{int(body_id)}.swc"

    def ensure(self, body_id: int) -> Path:
        body_id = int(body_id)
        # Validate the ID before touching the network.
        self.morphology.index_of(body_id)
        destination = self.path_for(body_id)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination

        partial = destination.with_suffix(".swc.part")
        url = f"{SKELETON_BASE_URL}/{body_id}.swc"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "DrosoMath-MaleCNS/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(f"No public MaleCNS skeleton for body {body_id}") from exc
            raise
        partial.replace(destination)
        return destination

    def payload(self, body_id: int, *, max_segments: int = 3500) -> dict[str, object]:
        np = _require_numpy()
        if max_segments < 10:
            raise ValueError("max_segments must be >= 10")
        path = self.ensure(body_id)
        nodes: dict[int, tuple[float, float, float]] = {}
        links: list[tuple[int, int]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    node_id = int(float(parts[0]))
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    parent = int(float(parts[6]))
                except ValueError:
                    continue
                if not all(math.isfinite(v) for v in (x, y, z)):
                    continue
                nodes[node_id] = (x, y, z)
                if parent >= 0:
                    links.append((node_id, parent))

        segments: list[list[float]] = []
        for child, parent in links:
            a = nodes.get(child)
            b = nodes.get(parent)
            if a is None or b is None:
                continue
            na = self.morphology.normalize_xyz(a)
            nb = self.morphology.normalize_xyz(b)
            segments.append([*na, *nb])

        total = len(segments)
        if total > max_segments:
            take = np.linspace(0, total - 1, max_segments, dtype=np.int64)
            segments = [segments[int(i)] for i in take]

        return {
            "body_id": int(body_id),
            "source": "MaleCNS v1.0 public SWC centerline",
            "coordinate_units": "8nm_voxels",
            "cached_path": str(path),
            "total_segments": total,
            "rendered_segments": len(segments),
            "segments": segments,
        }
