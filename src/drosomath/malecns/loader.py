from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .download import DEFAULT_DATA_DIR, path_for

EXPECTED_NEURONS_V1 = 166_700
EXPECTED_EDGES_MIN1 = 25_582_938
EXPECTED_EDGES_MIN5 = 6_242_118
INHIBITORY_NT = frozenset({"gaba", "glutamate", "histamine"})
SIGN_MODEL = (
    "presynaptic consensus_nt in {gaba, glutamate, histamine} -> inhibitory; "
    "all other/unclear -> positive drive"
)


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'MaleCNS mode needs NumPy. Install with: python -m pip install -e ".[malecns]"'
        ) from exc
    return np


def _require_arrow():
    try:
        import pyarrow.dataset as ds
        import pyarrow.feather as feather
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'MaleCNS mode needs PyArrow. Install with: python -m pip install -e ".[malecns]"'
        ) from exc
    return ds, feather


def _as_object_array(column):
    np = _require_numpy()
    return np.asarray(column.to_pylist(), dtype=object)


def _detect_column(names: list[str], exact: tuple[str, ...], contains: str | None = None) -> str:
    lowered = {name.lower(): name for name in names}
    for candidate in exact:
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    if contains is not None:
        for name in names:
            if contains.lower() in name.lower():
                return name
    raise ValueError(f"could not identify required column from {names}")


def _detect_weight_columns(names: list[str]) -> tuple[str, str, str]:
    pre = _detect_column(names, ("body_pre", "pre", "pre_body", "bodypre"), "pre")
    post = _detect_column(names, ("body_post", "post", "post_body", "bodypost"), "post")
    weight_candidates = ("weight", "syn_count", "count", "n", "synapses")
    lowered = {name.lower(): name for name in names}
    weight = next((lowered[x] for x in weight_candidates if x in lowered), None)
    if weight is None:
        leftovers = [name for name in names if name not in (pre, post)]
        if not leftovers:
            raise ValueError(f"could not identify weight column from {names}")
        weight = leftovers[-1]
    return pre, post, weight


@dataclass(slots=True)
class MaleCNSConnectome:
    """CSR representation of the selected MaleCNS v1.0 neuronal graph.

    ``signed_synapse_counts`` contains the anatomical connection count with a
    model sign derived from the presynaptic consensus neurotransmitter. Raw
    counts and neurotransmitter metadata remain available separately so the
    anatomical data and our dynamical assumptions are not conflated.
    """

    body_ids: object
    indptr: object
    post_indices: object
    synapse_counts: object
    signed_synapse_counts: object
    outgoing_strength: object
    presynaptic_sign: object
    consensus_nt: object
    metadata: dict[str, object] = field(default_factory=dict)
    min_connection_synapses: int = 1
    source: str = "MaleCNS v1.0 / HHMI Janelia"

    @property
    def neuron_count(self) -> int:
        return int(len(self.body_ids))

    @property
    def edge_count(self) -> int:
        return int(len(self.post_indices))

    @property
    def synapse_count_abs(self) -> int:
        np = _require_numpy()
        return int(np.asarray(self.synapse_counts, dtype=np.int64).sum())

    @property
    def flywire_ids(self):
        """Compatibility alias for the existing sparse simulator ID interface."""
        return self.body_ids

    def index_of(self, body_id: int) -> int:
        np = _require_numpy()
        pos = int(np.searchsorted(self.body_ids, int(body_id)))
        if pos >= self.neuron_count or int(self.body_ids[pos]) != int(body_id):
            raise KeyError(f"MaleCNS body {body_id} is not in the selected neuron set")
        return pos

    def strongest_outgoing_ids(self, count: int) -> tuple[int, ...]:
        np = _require_numpy()
        if count < 1:
            return ()
        count = min(int(count), self.neuron_count)
        if count == self.neuron_count:
            idx = np.argsort(self.outgoing_strength)[::-1]
        else:
            idx = np.argpartition(self.outgoing_strength, -count)[-count:]
            idx = idx[np.argsort(self.outgoing_strength[idx])[::-1]]
        return tuple(int(x) for x in self.body_ids[idx])

    def select_indices(self, **criteria: str) -> object:
        """Return neuron indices matching stored annotation fields exactly."""
        np = _require_numpy()
        mask = np.ones(self.neuron_count, dtype=np.bool_)
        for key, value in criteria.items():
            if key == "consensus_nt":
                values = self.consensus_nt
            else:
                try:
                    values = self.metadata[key]
                except KeyError as exc:
                    raise KeyError(f"metadata field not loaded: {key}") from exc
            mask &= np.asarray(values, dtype=object) == value
        return np.flatnonzero(mask).astype(np.int32, copy=False)

    def summary(self) -> dict[str, object]:
        np = _require_numpy()
        return {
            "dataset": self.source,
            "neuron_count": self.neuron_count,
            "edge_count": self.edge_count,
            "synapse_count_in_loaded_edges": self.synapse_count_abs,
            "min_connection_synapses": int(self.min_connection_synapses),
            "inhibitory_neuron_fraction": float(np.mean(self.presynaptic_sign < 0)),
            "sign_model": SIGN_MODEL,
        }


def _load_neuron_table(data_dir: Path):
    np = _require_numpy()
    _, feather = _require_arrow()

    ann = feather.read_table(path_for("annotations", data_dir), memory_map=True)
    names = ann.schema.names
    body_col = _detect_column(names, ("bodyId", "body", "body_id"))
    superclass_col = _detect_column(names, ("superclass",))

    body_all = ann[body_col].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    superclass_all = _as_object_array(ann[superclass_col])
    neuron_mask = np.asarray([x is not None and str(x) != "" for x in superclass_all])
    selected_rows = np.flatnonzero(neuron_mask)
    selected_bodies = body_all[selected_rows]
    order = np.argsort(selected_bodies, kind="stable")
    selected_rows = selected_rows[order]
    body_ids = selected_bodies[order]

    wanted = (
        "type",
        "flywireType",
        "hemibrainType",
        "superclass",
        "class",
        "subclass",
        "somaSide",
        "rootSide",
        "status",
        "statusLabel",
        "instance",
        "somaNeuromere",
        "group",
    )
    metadata: dict[str, object] = {}
    for name in wanted:
        if name in names:
            metadata[name] = _as_object_array(ann[name])[selected_rows]

    return body_ids, metadata


def _load_neurotransmitters(data_dir: Path, body_ids):
    np = _require_numpy()
    _, feather = _require_arrow()
    table = feather.read_table(path_for("neurotransmitters", data_dir), memory_map=True)
    names = table.schema.names
    body_col = _detect_column(names, ("body", "bodyId", "body_id"))
    nt_col = _detect_column(names, ("consensus_nt", "consensusNt"), "consensus")

    nt_body = table[body_col].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    nt_value = _as_object_array(table[nt_col])
    consensus = np.full(len(body_ids), "unclear", dtype=object)

    pos = np.searchsorted(body_ids, nt_body)
    in_range = pos < len(body_ids)
    valid_rows = np.flatnonzero(in_range)
    if len(valid_rows):
        valid_pos = pos[valid_rows]
        exact = body_ids[valid_pos] == nt_body[valid_rows]
        valid_rows = valid_rows[exact]
        valid_pos = valid_pos[exact]
        for row, dst in zip(valid_rows, valid_pos, strict=True):
            value = nt_value[row]
            if value is not None and str(value):
                consensus[int(dst)] = str(value).lower()

    sign = np.asarray(
        [-1 if str(nt).lower() in INHIBITORY_NT else 1 for nt in consensus],
        dtype=np.int8,
    )
    return consensus, sign


def load_malecns_v1(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    min_connection_synapses: int = 1,
    require_published_neuron_count: bool = True,
    batch_size: int = 4_000_000,
) -> MaleCNSConnectome:
    """Load MaleCNS v1.0 into the sparse layout used by DrosoMath.

    The 1.1 GB weights table is scanned in Arrow record batches. Only edges for
    annotated neurons survive, and only after the requested synapse-count
    threshold. This keeps peak host memory below a naive pandas materialization.
    """
    if min_connection_synapses < 1:
        raise ValueError("min_connection_synapses must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    np = _require_numpy()
    ds, _ = _require_arrow()
    root = Path(data_dir)
    required = [path_for(key, root) for key in ("annotations", "neurotransmitters", "weights")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MaleCNS v1.0 files: " + ", ".join(missing))

    body_ids, metadata = _load_neuron_table(root)
    if require_published_neuron_count and len(body_ids) != EXPECTED_NEURONS_V1:
        raise ValueError(
            f"Expected {EXPECTED_NEURONS_V1:,} MaleCNS neurons under superclass rule, "
            f"got {len(body_ids):,}"
        )

    consensus_nt, sign = _load_neurotransmitters(root, body_ids)

    dataset = ds.dataset(str(path_for("weights", root)), format="ipc")
    pre_col, post_col, weight_col = _detect_weight_columns(dataset.schema.names)
    scanner = dataset.scanner(
        columns=[pre_col, post_col, weight_col],
        batch_size=batch_size,
        use_threads=True,
    )

    pre_parts: list[object] = []
    post_parts: list[object] = []
    count_parts: list[object] = []
    n = len(body_ids)

    for batch in scanner.to_batches():
        pre_body = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post_body = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weights = batch.column(2).to_numpy(zero_copy_only=False)

        pre_idx = np.searchsorted(body_ids, pre_body)
        post_idx = np.searchsorted(body_ids, post_body)
        in_range = (pre_idx < n) & (post_idx < n)
        if not in_range.any():
            continue

        safe_pre = np.minimum(pre_idx, n - 1)
        safe_post = np.minimum(post_idx, n - 1)
        keep = in_range
        keep &= body_ids[safe_pre] == pre_body
        keep &= body_ids[safe_post] == post_body
        keep &= weights >= min_connection_synapses
        if not keep.any():
            continue

        pre_parts.append(pre_idx[keep].astype(np.int32, copy=False))
        post_parts.append(post_idx[keep].astype(np.int32, copy=False))
        count_parts.append(weights[keep].astype(np.int32, copy=False))

    if pre_parts:
        pre = np.concatenate(pre_parts)
        post = np.concatenate(post_parts)
        counts = np.concatenate(count_parts)
        order = np.lexsort((post, pre))
        pre = pre[order]
        post = post[order]
        counts = counts[order]
    else:
        pre = np.empty(0, dtype=np.int32)
        post = np.empty(0, dtype=np.int32)
        counts = np.empty(0, dtype=np.int32)

    signed = counts.astype(np.float32) * sign[pre].astype(np.float32)
    degree = np.bincount(pre, minlength=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    outgoing_strength = np.bincount(
        pre,
        weights=counts.astype(np.float64),
        minlength=n,
    ).astype(np.float32)

    return MaleCNSConnectome(
        body_ids=body_ids,
        indptr=indptr,
        post_indices=post,
        synapse_counts=counts,
        signed_synapse_counts=signed,
        outgoing_strength=outgoing_strength,
        presynaptic_sign=sign,
        consensus_nt=consensus_nt,
        metadata=metadata,
        min_connection_synapses=min_connection_synapses,
    )
