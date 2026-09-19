"""Reproducible compact Phase C.1 reward-locality validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain.directional_modulation import PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.usage_learning import UsageRewardRule


EDGES = (
    (0, 10, 1),   # direct output A route
    (0, 11, 1),   # shared input to output B
    (1, 11, 1),   # B-only route
    (2, 3, 1),    # upstream of A, two-hop
    (3, 10, 1),   # output A route
    (4, 5, 1),    # upstream of an opposing A route
    (5, 10, -1),  # opposing A route
    (6, 12, 1),   # unrelated active edge
)


def make_brain():
    ordered = sorted(EDGES, key=lambda item: item[0])
    indptr, posts, signs, cursor = [0], [], [], 0
    for pre in range(16):
        while cursor < len(ordered) and ordered[cursor][0] == pre:
            _, post, sign = ordered[cursor]
            posts.append(post); signs.append(float(sign)); cursor += 1
        indptr.append(len(posts))
    graph = SimpleNamespace(
        neuron_count=16,
        indptr=np.asarray(indptr, dtype=np.int32),
        post_indices=np.asarray(posts, dtype=np.int32),
        signed_synapse_counts=np.asarray(signs, dtype=np.float32),
    )
    state = SparsePlasticityState(
        len(EDGES), config=PlasticStateConfig(plastic_fraction=1.0, seed=123),
    )
    state.usage_ema[:] = 1.0
    state.eligibility[:] = 1.0
    return SimpleNamespace(
        np=np, connectome=graph, plasticity=state,
        _recent_presynaptic={0, 1, 2, 3, 4, 5, 6},
    )


def run() -> dict[str, object]:
    signal = LearningSignal(1.0, reinforcement={"motor/a": 1.0}, success=True)
    context = {"motor/a": np.asarray([10], dtype=np.int32), "motor/b": np.asarray([11], dtype=np.int32)}
    rule = UsageRewardRule(learning_rate=0.1)

    legacy_brain = make_brain()
    legacy_before = legacy_brain.plasticity.multiplier.copy()
    legacy_stats = rule.apply_recent_presynaptic(
        legacy_brain.plasticity, reward=1.0,
        indptr=legacy_brain.connectome.indptr,
        presynaptic_indices=sorted(legacy_brain._recent_presynaptic),
    )
    legacy_changed = np.flatnonzero(legacy_brain.plasticity.multiplier != legacy_before)

    localized_brain = make_brain()
    localized_before = localized_brain.plasticity.multiplier.copy()
    controller = PlasticityController()
    reward_credit = controller.build_reward_credit(localized_brain, signal, context)
    route_edges = controller._credit_edges(
        localized_brain,
        controller._active_edges(localized_brain),
        context["motor/a"],
    )
    localized_stats = rule.apply_recent_presynaptic(
        localized_brain.plasticity, reward=1.0,
        indptr=localized_brain.connectome.indptr,
        presynaptic_indices=sorted(localized_brain._recent_presynaptic),
        reward_credit=reward_credit,
    )
    localized_changed = np.flatnonzero(localized_brain.plasticity.multiplier != localized_before)
    selected = set(int(x) for x in reward_credit.edge_indices)
    opposing = set(int(x) for x in route_edges.edges[route_edges.path_polarities < 0.0])
    legacy_set = set(int(x) for x in legacy_changed)
    localized_set = set(int(x) for x in localized_changed)
    b_only = {index for index, (_, post, _) in enumerate(EDGES) if post == 11}

    interference_legacy = make_brain()
    interference_localized = make_brain()
    before_legacy_b = interference_legacy.plasticity.multiplier.copy()
    before_localized_b = interference_localized.plasticity.multiplier.copy()
    rule.apply_recent_presynaptic(
        interference_legacy.plasticity, reward=1.0,
        indptr=interference_legacy.connectome.indptr,
        presynaptic_indices=sorted(interference_legacy._recent_presynaptic),
    )
    b_credit = controller.build_reward_credit(interference_localized, signal, context)
    rule.apply_recent_presynaptic(
        interference_localized.plasticity, reward=1.0,
        indptr=interference_localized.connectome.indptr,
        presynaptic_indices=sorted(interference_localized._recent_presynaptic),
        reward_credit=b_credit,
    )
    legacy_b_changed = int(np.count_nonzero(interference_legacy.plasticity.multiplier[list(b_only)] != before_legacy_b[list(b_only)]))
    localized_b_changed = int(np.count_nonzero(interference_localized.plasticity.multiplier[list(b_only)] != before_localized_b[list(b_only)]))
    result = {
        "legacy": {
            "legacy_eligible_edges": legacy_stats.eligible_reward_edges_before_localization,
            "legacy_updated_edges": legacy_stats.actual_reward_updated_edges,
            "eligible_edges": legacy_stats.eligible_reward_edges_before_localization,
            "updated_edges": legacy_stats.actual_reward_updated_edges,
            "mean_abs_delta": legacy_stats.mean_abs_delta,
        },
        "localized": {
            "localized_eligible_edges": localized_stats.eligible_reward_edges_before_localization,
            "localized_selected_credit_edges": localized_stats.selected_credit_edges,
            "localized_actual_updated_edges": localized_stats.actual_reward_updated_edges,
            "selected_credit_edges": localized_stats.selected_credit_edges,
            "actual_updated_edges": localized_stats.actual_reward_updated_edges,
            "actual_credited_updated_edges": localized_stats.actual_credited_reward_updated_edges,
            "opposing_path_edges_skipped": localized_stats.opposing_path_edges_skipped,
            "mean_abs_delta": localized_stats.mean_abs_delta,
        },
        "reduction": {
            "update_reduction_fraction": 1.0 - localized_stats.actual_reward_updated_edges / max(1, legacy_stats.actual_reward_updated_edges),
            "reinforced_route_edges_updated": len(localized_set & selected),
            "opposing_route_edges_updated": len(localized_set & opposing),
            "unrelated_edges_updated": len(localized_set - selected - opposing),
        },
        "interference_probe": {
            "output_b_only_edges": len(b_only),
            "legacy_b_only_edges_updated": legacy_b_changed,
            "localized_b_only_edges_updated": localized_b_changed,
        },
        "pass": bool(
            legacy_stats.actual_reward_updated_edges > localized_stats.actual_reward_updated_edges
            and localized_stats.opposing_path_edges_skipped > 0
            and len(localized_set & opposing) == 0
            and localized_b_changed == 0
        ),
    }
    return result


if __name__ == "__main__":
    output = run()
    path = Path("results/latest_reward_locality_phase_c1_validation.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
