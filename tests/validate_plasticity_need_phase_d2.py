"""Deterministic executable Phase D.2 validation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.checkpoint import restore_learning_checkpoint, save_learning_checkpoint
from drosomath.whole_brain.directional_modulation import DirectionalModulationConfig, DirectionalUpdate, PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.plasticity_budget_adaptation import PlasticityBudgetAdaptation, PlasticityBudgetAdaptationConfig
from drosomath.whole_brain.plasticity_need import PlasticityNeedConfig


def make_brain():
    graph = SimpleNamespace(
        neuron_count=8,
        edge_count=10,
        indptr=np.asarray([0, 3, 4, 4, 5, 6, 7, 8, 10], dtype=np.int32),
        post_indices=np.asarray([1, 2, 5, 2, 2, 3, 4, 4, 2, 3], dtype=np.int32),
        signed_synapse_counts=np.asarray([1, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        min_connection_synapses=1,
    )
    state = SparsePlasticityState(10, config=PlasticStateConfig(plastic_fraction=1.0, seed=4))
    state.retire_edges([0, 1, 2, 3, 8, 9])
    state.stability[4] = 0.90  # protected high-value donor
    return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0})


def make_adaptation():
    route_controller = PlasticityController(DirectionalModulationConfig(learning_rate=0.10))
    return PlasticityBudgetAdaptation(
        config=PlasticityBudgetAdaptationConfig(
            enabled=True,
            max_promotions_per_event=1,
            need=PlasticityNeedConfig(
                need_decay=0.90,
                minimum_observations=3,
                promotion_threshold=1.0,
            ),
        ),
        route_controller=route_controller,
    )


def zero_update():
    return DirectionalUpdate(0, {}, 0.0, np.empty(0, dtype=np.int32), {}, 0, 0, 0, 0, (), 0)


def fake_brain(state):
    return SimpleNamespace(
        np=np,
        connectome=SimpleNamespace(neuron_count=8, edge_count=state.edge_count, min_connection_synapses=1),
        plasticity=state,
    )


def run() -> dict[str, object]:
    brain = make_brain()
    adaptation = make_adaptation()
    signal = LearningSignal(reward=-1.0, directional_error={"out": 1.0}, success=False)
    output_context = {"out": [2]}
    budget_before = brain.plasticity.plastic_edge_count
    anatomy_before = brain.connectome.edge_count
    observations = []
    result = {}
    for trial in range(1, 4):
        result = adaptation.observe_directional_failure(
            brain=brain,
            signal=signal,
            output_context=output_context,
            directional_update=zero_update(),
        )
        observations.append({
            "trial": trial,
            "candidates_observed": result["structural_need_candidates_observed"],
            "tracked": result["tracked_need_candidates"],
            "ready": result["ready_need_candidates"],
            "reallocation_triggered": result["reallocation_triggered"],
        })
    promoted = [int(edge) for edge in result["promoted_edges"]]
    retired = [int(edge) for edge in result["retired_edges"]]

    promoted_edge = promoted[0] if promoted else -1
    eligibility_before_next_trial = float(brain.plasticity.eligibility[promoted_edge]) if promoted_edge >= 0 else -1.0
    recorded = brain.plasticity.record_use_indices([promoted_edge]) if promoted_edge >= 0 else 0
    multiplier_before_next_update = float(brain.plasticity.multiplier[promoted_edge]) if promoted_edge >= 0 else 0.0
    next_update = adaptation.route_controller.apply_learning_signal(brain, signal, output_context)
    learned_next_trial = bool(
        promoted_edge >= 0
        and recorded == 1
        and next_update.edge_updates > 0
        and float(brain.plasticity.multiplier[promoted_edge]) != multiplier_before_next_update
    )

    tracker_round_trip = False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "phase_d2.npz"
        save_learning_checkpoint(path, brain=brain, need_tracker=adaptation.need_tracker)
        restored_brain = make_brain()
        restored_adaptation = make_adaptation()
        restore_info = restore_learning_checkpoint(
            path,
            brain=restored_brain,
            need_tracker=restored_adaptation.need_tracker,
        )
        tracker_round_trip = bool(
            restore_info["need_state_restored"]
            and adaptation.need_tracker.event_count == restored_adaptation.need_tracker.event_count
            and len(adaptation.need_tracker.records()) == len(restored_adaptation.need_tracker.records())
            and all(
                abs(left.need_score - right.need_score) < 1e-5
                for left, right in zip(adaptation.need_tracker.records(), restored_adaptation.need_tracker.records())
            )
        )

    protected_survived = bool(brain.plasticity.plastic_mask[4])
    irrelevant_frozen = bool(not brain.plasticity.plastic_mask[2])
    budget_after = brain.plasticity.plastic_edge_count
    output = {
        "persistent_error": {
            "trials_before_trigger": next(item["trial"] for item in observations if item["reallocation_triggered"]),
            "candidate_observations": observations,
            "max_need": float(result["max_candidate_need"]),
        },
        "reallocation": {
            "triggered": bool(result["reallocation_triggered"]),
            "promoted_edges": promoted,
            "retired_edges": retired,
            "budget_before": budget_before,
            "budget_after": budget_after,
            "budget_delta": budget_after - budget_before,
        },
        "next_trial": {
            "promoted_edge": promoted_edge,
            "promoted_edge_recorded_eligibility": bool(eligibility_before_next_trial == 0.0 and recorded == 1),
            "generic_directional_update": bool(next_update.edge_updates > 0),
            "promoted_route_learned": learned_next_trial,
        },
        "safety": {
            "protected_donor_retained": protected_survived,
            "irrelevant_edge_not_promoted": irrelevant_frozen,
            "anatomy_unchanged": anatomy_before == brain.connectome.edge_count,
            "negative_inhibitory_candidate_supported": bool(1 in promoted or 1 in adaptation.need_tracker.records()),
        },
        "checkpoint": {"need_state_restored": tracker_round_trip},
    }
    output["pass"] = bool(
        output["reallocation"]["triggered"]
        and output["reallocation"]["budget_delta"] == 0
        and output["next_trial"]["promoted_edge_recorded_eligibility"]
        and output["next_trial"]["promoted_route_learned"]
        and output["safety"]["protected_donor_retained"]
        and output["safety"]["irrelevant_edge_not_promoted"]
        and output["safety"]["anatomy_unchanged"]
        and output["checkpoint"]["need_state_restored"]
    )
    return output


if __name__ == "__main__":
    output = run()
    path = Path("results/latest_plasticity_need_phase_d2_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
