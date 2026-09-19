"""Reproducible compact Phase D.1 fixed-budget validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drosomath.malecns.checkpoint import restore_learning_checkpoint, save_learning_checkpoint
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.plasticity_budget import PlasticityBudgetManager


def make_state() -> SparsePlasticityState:
    state = SparsePlasticityState(10, config=PlasticStateConfig(plastic_fraction=1.0, seed=17))
    state.retire_edges([4, 5, 6, 7, 8, 9])
    state.stability[0] = 0.9
    state.usage_ema[0] = 0.8
    state.multiplier[0] = 1.6
    state.usage_ema[1] = 0.7
    state.multiplier[1] = 1.4
    state.usage_ema[2] = 0.01
    state.usage_ema[3] = 0.02
    return state


def fake_brain(state):
    return SimpleNamespace(
        np=np,
        connectome=SimpleNamespace(neuron_count=4, edge_count=state.edge_count, min_connection_synapses=1),
        plasticity=state,
    )


def run() -> dict[str, object]:
    state = make_state()
    before_mask = state.plastic_mask.copy()
    before_count = state.plastic_edge_count
    before_edge_count = state.edge_count
    manager = PlasticityBudgetManager()
    exchange = manager.reallocate(state, [4, 5], need_scores=[0.4, 0.9], count=2)
    after_mask = state.plastic_mask.copy()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "phase_d1.npz"
        save_learning_checkpoint(path, brain=fake_brain(state))
        restored = make_state()
        restore_info = restore_learning_checkpoint(path, brain=fake_brain(restored))

    mask_equal = bool(np.array_equal(state.plastic_mask, restored.plastic_mask))
    learned_equal = bool(
        np.array_equal(state.multiplier, restored.multiplier)
        and np.array_equal(state.stability, restored.stability)
        and np.array_equal(state.usage_ema, restored.usage_ema)
    )
    result = {
        "before": {"plastic_edges": before_count, "plastic_indices": np.flatnonzero(before_mask).tolist()},
        "exchange": {
            "promoted": exchange["promoted_edges"].tolist(),
            "retired": exchange["retired_edges"].tolist(),
            "plastic_edges_before": exchange["plastic_edges_before"],
            "plastic_edges_after": exchange["plastic_edges_after"],
            "budget_delta": exchange["budget_delta"],
            "protected_edges_retained": bool(state.plastic_mask[0]),
        },
        "after": {
            "plastic_edges": state.plastic_edge_count,
            "plastic_indices": np.flatnonzero(after_mask).tolist(),
            "promoted_edges_plastic": bool(state.plastic_mask[4] and state.plastic_mask[5]),
            "retired_edges_frozen": bool((~state.plastic_mask[[2, 3]]).all()),
        },
        "anatomy": {
            "edge_count_before": before_edge_count,
            "edge_count_after": state.edge_count,
            "unchanged": before_edge_count == state.edge_count,
        },
        "checkpoint_round_trip": {
            "plastic_mask_equal": mask_equal,
            "learned_state_equal": learned_equal,
            "dynamic_allocation_restored": bool(restore_info["dynamic_allocation_restored"]),
        },
    }
    result["pass"] = bool(
        exchange["budget_delta"] == 0
        and state.plastic_edge_count == before_count
        and result["exchange"]["protected_edges_retained"]
        and result["anatomy"]["unchanged"]
        and mask_equal and learned_equal
    )
    return result


if __name__ == "__main__":
    output = run()
    path = Path("results/latest_plasticity_budget_phase_d1_validation.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
