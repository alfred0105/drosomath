from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path


def save_readout_checkpoint(path: str | Path, readout) -> dict[str, object]:
    np = readout.np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=readout.weights,
        bias=readout.bias,
        labels=np.asarray(readout.labels, dtype="U64"),
        output_body_ids=np.asarray(readout.population.body_ids, dtype=np.int64),
        train_steps=np.asarray([readout.train_steps], dtype=np.int64),
        frozen=np.asarray([readout.frozen], dtype=np.bool_),
    )
    return {"path": str(path), "labels": list(readout.labels), "train_steps": readout.train_steps}


def restore_readout_checkpoint(path: str | Path, readout) -> dict[str, object]:
    np = readout.np
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        labels = tuple(str(x) for x in data["labels"].tolist())
        body_ids = tuple(int(x) for x in data["output_body_ids"].tolist())
        if labels != tuple(readout.labels):
            raise ValueError("readout labels do not match checkpoint")
        if body_ids != tuple(readout.population.body_ids):
            raise ValueError("output population does not match checkpoint")
        if data["weights"].shape != readout.weights.shape:
            raise ValueError("readout weight shape mismatch")
        readout.weights[:] = data["weights"]
        readout.bias[:] = data["bias"]
        readout.train_steps = int(data["train_steps"][0])
        readout.frozen = bool(data["frozen"][0])
    return {"path": str(path), "labels": list(labels), "train_steps": readout.train_steps}


def save_learning_checkpoint(
    path: str | Path,
    *,
    brain,
    readout=None,
    config=None,
    completed_trials: int = 0,
    stage: str = "",
    session_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Persist long-term learned state without copying immutable anatomy."""
    np = brain.np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only plastic or historically plastic edges can differ from the default
    # state through the public learning API. This avoids a full 6.2M-edge scan
    # at every periodic keyboard checkpoint while retaining frozen curriculum
    # state required for a later unlock.
    candidates = brain.plasticity.lifecycle_indices
    candidate_changed = (
        (np.abs(brain.plasticity.multiplier[candidates] - 1.0) > 1e-7)
        | (brain.plasticity.stability[candidates] > 0.0)
        | (brain.plasticity.usage_ema[candidates] > 1e-7)
    )
    changed = candidates[candidate_changed].astype(np.int32, copy=False)

    payload: dict[str, object] = {
        "format_version": np.asarray([3], dtype=np.int32),
        "neuron_count": np.asarray([brain.connectome.neuron_count], dtype=np.int64),
        "edge_count": np.asarray([brain.connectome.edge_count], dtype=np.int64),
        "min_connection_synapses": np.asarray(
            [int(getattr(brain.connectome, "min_connection_synapses", 1))], dtype=np.int32
        ),
        "changed_edge_indices": changed,
        "multipliers": brain.plasticity.multiplier[changed],
        "stability": brain.plasticity.stability[changed],
        "usage_ema": brain.plasticity.usage_ema[changed],
        "plastic_fraction": np.asarray(
            [float(brain.plasticity.config.plastic_fraction)], dtype=np.float32
        ),
        "plastic_seed": np.asarray([int(brain.plasticity.config.seed)], dtype=np.int64),
        "completed_trials": np.asarray([int(completed_trials)], dtype=np.int64),
        "stage": np.asarray([str(stage)], dtype="U64"),
    }

    structural = getattr(brain, "structural_overlay", None)
    if structural is not None:
        payload.update(structural.checkpoint_payload())

    if config is not None:
        cfg = asdict(config) if is_dataclass(config) else dict(config)
        for key, value in cfg.items():
            if isinstance(value, bool):
                payload[f"config__{key}"] = np.asarray([value], dtype=np.bool_)
            elif isinstance(value, int):
                payload[f"config__{key}"] = np.asarray([value], dtype=np.int64)
            elif isinstance(value, float):
                payload[f"config__{key}"] = np.asarray([value], dtype=np.float64)
            elif isinstance(value, str):
                payload[f"config__{key}"] = np.asarray([value], dtype="U128")

    if readout is not None:
        payload.update(
            {
                "readout_weights": readout.weights,
                "readout_bias": readout.bias,
                "readout_labels": np.asarray(readout.labels, dtype="U64"),
                "output_body_ids": np.asarray(readout.population.body_ids, dtype=np.int64),
                "readout_train_steps": np.asarray([readout.train_steps], dtype=np.int64),
                "readout_frozen": np.asarray([readout.frozen], dtype=np.bool_),
            }
        )
    if session_state is not None:
        payload["session_state_json"] = np.asarray(
            [json.dumps(session_state, sort_keys=True)], dtype="U65535"
        )

    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "format_version": 3,
        "changed_edge_count": int(len(changed)),
        "structural_edge_count": int(structural.edge_count) if structural is not None else 0,
        "completed_trials": int(completed_trials),
        "stage": str(stage),
        "session_state_saved": session_state is not None,
    }


def _restore_structural_if_present(data, brain) -> int:
    if "structural__present" not in data or not bool(data["structural__present"][0]):
        return 0

    from drosomath.whole_brain import StructuralOverlayConfig

    structural = getattr(brain, "structural_overlay", None)
    if structural is None:
        kwargs = {}
        for field in fields(StructuralOverlayConfig):
            key = "structural__config__" + field.name
            if key not in data:
                continue
            value = data[key][0]
            if field.type is bool or isinstance(field.default, bool):
                value = bool(value)
            elif field.type is int or isinstance(field.default, int):
                value = int(value)
            elif field.type is float or isinstance(field.default, float):
                value = float(value)
            kwargs[field.name] = value
        structural = brain.configure_structural_plasticity(StructuralOverlayConfig(**kwargs))
    structural.restore_from_checkpoint(data)
    return structural.edge_count


def restore_learning_checkpoint(
    path: str | Path,
    *,
    brain,
    readout=None,
    strict_readout: bool = True,
) -> dict[str, object]:
    """Restore long-term plastic and optional structural state."""
    np = brain.np
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        neuron_count = int(data["neuron_count"][0])
        edge_count = int(data["edge_count"][0])
        if neuron_count != brain.connectome.neuron_count:
            raise ValueError(
                f"checkpoint neuron count {neuron_count} != graph {brain.connectome.neuron_count}"
            )
        if edge_count != brain.connectome.edge_count:
            raise ValueError(
                f"checkpoint edge count {edge_count} != graph {brain.connectome.edge_count}"
            )

        changed = data["changed_edge_indices"].astype(np.int64, copy=False)
        if len(changed) and (int(changed.min()) < 0 or int(changed.max()) >= edge_count):
            raise ValueError("checkpoint contains out-of-range edge indices")

        brain.plasticity.multiplier[changed] = data["multipliers"]
        brain.plasticity.stability[changed] = data["stability"]
        if "usage_ema" in data:
            brain.plasticity.usage_ema[changed] = data["usage_ema"]
        brain.plasticity.clear_eligibility()
        structural_edge_count = _restore_structural_if_present(data, brain)

        readout_restored = False
        if readout is not None and "readout_weights" in data:
            labels = tuple(str(x) for x in data["readout_labels"].tolist())
            body_ids = tuple(int(x) for x in data["output_body_ids"].tolist())
            compatible = labels == tuple(readout.labels) and body_ids == tuple(readout.population.body_ids)
            if not compatible and strict_readout:
                raise ValueError("checkpoint readout labels/output population do not match")
            if compatible:
                if data["readout_weights"].shape != readout.weights.shape:
                    raise ValueError("checkpoint readout weight shape mismatch")
                readout.weights[:] = data["readout_weights"]
                readout.bias[:] = data["readout_bias"]
                readout.train_steps = int(data["readout_train_steps"][0])
                readout.frozen = bool(data["readout_frozen"][0])
                readout_restored = True

        return {
            "path": str(path),
            "changed_edge_count": int(len(changed)),
            "structural_edge_count": int(structural_edge_count),
            "completed_trials": int(data["completed_trials"][0]) if "completed_trials" in data else 0,
            "stage": str(data["stage"][0]) if "stage" in data else "",
            "readout_restored": readout_restored,
            "session_state": (
                json.loads(str(data["session_state_json"][0]))
                if "session_state_json" in data
                else None
            ),
        }
