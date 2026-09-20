"""Correct F.3D conclusions without rerunning the 400-episode experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drosomath.malecns import (  # noqa: E402
    ContextualPredictionConfig,
    ContextualPredictionLearningSession,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
)
from run_presynaptic_std_binding_phase_f3d import (  # noqa: E402
    DATA_DIR,
    _interface_config,
    _make_brain,
)
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402


ARTIFACT = ROOT / "results/latest_presynaptic_std_binding_phase_f3d.json"


def _evaluation_rows(artifact):
    return artifact["behavior"]["per_seed"]


def derive_prediction_criteria(artifact):
    rows = _evaluation_rows(artifact)
    control = [row["arms"]["CONTROL"]["evaluation"]["400"]["intact"] for row in rows]
    std = [row["arms"]["STD_BINDING"]["evaluation"]["400"]["intact"] for row in rows]
    reset = [row["arms"]["STD_BINDING"]["evaluation"]["400"]["between_item_reset"] for row in rows]
    improved = [s["accuracy"] > c["accuracy"] for s, c in zip(std, control)]
    dependent = [s["accuracy"] > r["accuracy"] for s, r in zip(std, reset)]
    no_collapse = not any(item["output_collapse"] for item in control + std + reset)
    std_prediction_improved = bool(
        np.mean([item["accuracy"] for item in std]) > np.mean([item["accuracy"] for item in control])
        and sum(improved) >= 2
        and np.mean([item["target_minus_best_competitor_margin_hz"] for item in std])
        > np.mean([item["target_minus_best_competitor_margin_hz"] for item in control])
        and no_collapse
    )
    std_context_dependence_supported = bool(
        np.mean([item["accuracy"] for item in std]) > np.mean([item["accuracy"] for item in reset])
        and sum(dependent) >= 2
        and np.mean([item["target_minus_best_competitor_margin_hz"] for item in std])
        > np.mean([item["target_minus_best_competitor_margin_hz"] for item in reset])
    )
    strong_prediction = bool(
        np.mean([item["accuracy"] for item in std]) >= 0.50
        and sum(item["accuracy"] >= 0.50 for item in std) >= 2
        and std_context_dependence_supported
        and no_collapse
    )
    return {
        "std_prediction_improved": std_prediction_improved,
        "std_context_dependence_supported": std_context_dependence_supported,
        "strong_prediction": strong_prediction,
        "discrete_contextual_prediction_demonstrated": strong_prediction,
        "improved_seed_count": int(sum(improved)),
        "intact_gt_reset_seed_count": int(sum(dependent)),
        "no_output_collapse": bool(no_collapse),
    }


def derive_representation_criteria(artifact):
    rows = artifact["representation"]["per_seed"]
    control = [row["arms"]["CONTROL"] for row in rows]
    std = [row["arms"]["STD_BINDING"] for row in rows]
    control_same_first = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in control]
    std_same_first = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in std]
    broader_counts = []
    for row in std:
        values = row["conjunctive_specific_pools"]["pre_go"]["per_context"].values()
        broader_counts.append(int(sum(item["broader_conjunction_count"] > 0 for item in values)))
    control_mean = float(np.mean(control_same_first))
    std_mean = float(np.mean(std_same_first))
    return {
        "control_same_first_jaccard": control_mean,
        "std_same_first_jaccard": std_mean,
        "std_to_control_ratio": float(std_mean / max(control_mean, 1e-12)),
        "broader_context_counts_by_seed": broader_counts,
        "context_binding_improved": bool(std_mean <= 0.80 * control_mean and sum(value >= 12 for value in broader_counts) >= 2),
    }


def derive_guards(artifact):
    rows = artifact["representation"]["per_seed"]
    std_rep = [row["arms"]["STD_BINDING"] for row in rows]
    control_rep = [row["arms"]["CONTROL"] for row in rows]
    std_same_second = float(np.mean([
        row["pairwise_context_similarity"]["pre_go"]["same_second_different_first"]["mean"]
        for row in std_rep
    ]))
    std_same_first = float(np.mean([
        row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"]
        for row in std_rep
    ]))
    second_domination = bool(std_same_second >= 0.80 and std_same_second > std_same_first)

    def active_mean(rep):
        values = rep["state_metrics"]["pre_go"]["per_context"].values()
        return float(np.mean([item["active_count_mean"] for item in values]))

    control_active = float(np.mean([active_mean(rep) for rep in control_rep]))
    std_active = float(np.mean([active_mean(rep) for rep in std_rep]))
    ratio = float(std_active / max(control_active, 1e-12))
    near_silence = bool(all(
        item["stable_active_count"] <= 1
        for rep in std_rep
        for item in rep["state_metrics"]["pre_go"]["per_context"].values()
    ))
    return {
        "second_domination_guard": {
            "std_same_first_jaccard": std_same_first,
            "std_same_second_jaccard": std_same_second,
            "threshold": 0.80,
            "second_cue_domination": second_domination,
        },
        "activity_guard": {
            "control_mean_pre_go_active": control_active,
            "std_mean_pre_go_active": std_active,
            "std_to_control_activity_ratio": ratio,
            "near_total_silence": near_silence,
            "activity_pathology": bool(ratio > 2.0 or near_silence),
        },
    }


def run_bounded_phase_credit(*, data_dir: Path = DATA_DIR, seed: int = 179, max_incorrect: int = 8):
    """Replay only until eight incorrect diagnostic episodes per arm."""
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, _, _ = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    output = {}
    for arm, std_enabled in (("CONTROL", False), ("STD_BINDING", True)):
        brain = _make_brain(connectome, interface_config, seed, std_enabled=std_enabled)
        session = ContextualPredictionLearningSession(
            brain,
            wm_interface,
            config=ContextualPredictionConfig(telemetry_level="summary"),
            route_cache_enabled=True,
            plastic_row_cache_enabled=True,
            prospective_index_enabled=True,
        )
        schedule = balanced_pair_schedule(25, seed=seed + session.config.schedule_seed_offset)
        samples = []
        attempts = 0
        for first, second in schedule:
            attempts += 1
            record = session.train_trial(first, second, capture_phase_credit=True)
            if not record["success"] and record["phase_credit"] is not None:
                samples.append(record["phase_credit"])
            if len(samples) >= max_incorrect:
                break
        output[arm] = {
            "seed": int(seed),
            "episodes_attempted": attempts,
            "incorrect_phase_credit_samples": len(samples),
            "samples": samples,
            "nonzero_updated_edge_samples": int(sum(sample["updated_edges"] > 0 for sample in samples)),
            "all_samples_have_edge_attribution": bool(all(sample["updated_edges"] > 0 for sample in samples)),
        }
    output["full_training_rerun"] = False
    return output


def correct(*, artifact_path: Path = ARTIFACT, data_dir: Path = DATA_DIR):
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    prediction = derive_prediction_criteria(artifact)
    representation = derive_representation_criteria(artifact)
    guards = derive_guards(artifact)
    bounded = run_bounded_phase_credit(data_dir=data_dir)
    first_memory_destroyed = bool(artifact["first_memory_guard"]["flagged"])
    artifact["validation_mode"] = {
        "reused_existing_training_artifact": True,
        "full_training_rerun": False,
        "bounded_phase_credit_only": True,
    }
    artifact["representation"]["corrected_criteria"] = representation
    artifact["behavior"]["corrected_criteria"] = prediction
    artifact.update(guards)
    artifact["bounded_phase_credit"] = bounded
    artifact["conclusion"] = {
        "context_binding_improved": representation["context_binding_improved"],
        "std_prediction_improved": prediction["std_prediction_improved"],
        "std_context_dependence_supported": prediction["std_context_dependence_supported"],
        "discrete_contextual_prediction_demonstrated": prediction["discrete_contextual_prediction_demonstrated"],
        "first_memory_destroyed": first_memory_destroyed,
        "second_cue_domination": artifact["second_domination_guard"]["second_cue_domination"],
        "activity_pathology": artifact["activity_guard"]["activity_pathology"],
        "recommended_next_step": (
            "test_transient_hebbian_synaptic_binding"
            if not representation["context_binding_improved"]
            else "continue_context_binding_validation"
        ),
        "interpretation": {
            "behavioral_modulation_observed": prediction["std_prediction_improved"],
            "conjunctive_representation_demonstrated": representation["context_binding_improved"],
        },
    }
    # pass describes validation/safety completion, not success of the STD
    # context-binding hypothesis.
    artifact["pass"] = bool(
        artifact["safety"]["plastic_budget_fixed"]
        and artifact["safety"]["anatomy_unchanged"]
        and artifact["safety"]["disabled_exact_smoke"]["passed"]
        and not first_memory_destroyed
        and all(value["incorrect_phase_credit_samples"] >= 8 for value in bounded.values() if isinstance(value, dict))
    )
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    result = correct(artifact_path=args.artifact, data_dir=args.data_dir)
    print(json.dumps({"pass": result["pass"], "conclusion": result["conclusion"], "second_domination_guard": result["second_domination_guard"], "activity_guard": result["activity_guard"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
