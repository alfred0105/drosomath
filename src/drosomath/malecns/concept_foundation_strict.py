from __future__ import annotations

import argparse
import json
from pathlib import Path

from .concept_foundation import (
    ConceptFoundationConfig,
    DEFAULT_CHECKPOINT,
    DEFAULT_PROGRESS,
    DEFAULT_READOUT_DIR,
    DEFAULT_RESULT,
    build_html,
    run_concept_foundation,
)
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1


DEFAULT_HTML = Path("results/latest_malecns_concept_foundation.html")


def strict_foundation_gate(report: dict[str, object], *, min_learning_gain: float = 0.03) -> dict[str, object]:
    thresholds = {
        "object_presence": 0.80,
        "single_vs_multiple": 0.70,
        "latent_quantity_1_3": 0.50,
    }
    stages = report.get("stage_reports") or []
    retention_history = report.get("retention_history") or []
    if not stages or not retention_history:
        return {"passed": False, "reason": "incomplete", "tasks": {}}

    final = retention_history[-1]["tasks"]
    checks = {}
    passed = True
    for row in stages:
        name = str(row["stage"])
        threshold = thresholds[name]
        before = float(row["before_heldout"]["accuracy"])
        heldout = float(row["after_heldout"]["accuracy"])
        train = float(row["after_train"]["accuracy"])
        final_acc = float(final[name]["heldout_accuracy"])
        chance = float(row["after_heldout"]["chance"])
        gain = heldout - before
        gap = train - heldout

        above_chance = heldout >= chance + 0.10
        learned_from_training = gain >= min_learning_gain
        generalizes = gap <= 0.20
        retained = final_acc >= max(chance + 0.10, threshold * 0.90)
        ok = (
            heldout >= threshold
            and above_chance
            and learned_from_training
            and generalizes
            and retained
        )
        checks[name] = {
            "heldout_accuracy_before_stage": before,
            "heldout_accuracy_after_stage": heldout,
            "learning_gain": gain,
            "minimum_learning_gain": min_learning_gain,
            "train_accuracy_after_stage": train,
            "position_generalization_gap": gap,
            "final_heldout_retention": final_acc,
            "threshold": threshold,
            "above_chance": above_chance,
            "learned_from_training": learned_from_training,
            "generalizes_to_unseen_positions": generalizes,
            "retained_after_later_concepts": retained,
            "passed": ok,
        }
        passed &= ok

    return {
        "passed": bool(passed),
        "criterion": (
            "accuracy must improve after CNS learning, exceed chance, transfer to unseen virtual positions, "
            "and survive later concept stages"
        ),
        "minimum_learning_gain": min_learning_gain,
        "tasks": checks,
    }


def checkpoint_structural_summary(path: Path) -> dict[str, object]:
    """Read compact structural metrics without reloading the full MaleCNS graph."""
    import numpy as np

    if not path.is_file():
        return {"enabled": False, "active_edges": 0, "reason": "checkpoint_missing"}
    with np.load(path, allow_pickle=False) as data:
        if "structural__present" not in data or not bool(data["structural__present"][0]):
            return {"enabled": False, "active_edges": 0}
        slots = data["structural__slots"]
        strength = data["structural__signed_strength"]
        stability = data["structural__stability"]
        return {
            "enabled": True,
            "active_edges": int(len(slots)),
            "rewire_cycles": int(data["structural__cycles"][0]) if "structural__cycles" in data else 0,
            "total_regrown": int(data["structural__total_regrown"][0]) if "structural__total_regrown" in data else int(len(slots)),
            "total_replaced": int(data["structural__total_replaced"][0]) if "structural__total_replaced" in data else 0,
            "donor_edges_silenced": int(len(slots)),
            "mean_abs_strength": float(np.mean(np.abs(strength))) if len(strength) else 0.0,
            "mean_stability": float(np.mean(stability)) if len(stability) else 0.0,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Run strict concept-first MaleCNS foundation curriculum")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--stage-trials", type=int, default=2048)
    p.add_argument("--validation-trials", type=int, default=32)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-learning-gain", type=float, default=0.03)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    a = p.parse_args()

    if a.download:
        download_malecns(a.data_dir)
    connectome = load_malecns_v1(a.data_dir, min_connection_synapses=a.min_syn)
    config = ConceptFoundationConfig(
        min_connection_synapses=a.min_syn,
        stage_trials=a.stage_trials,
        validation_trials_per_label=a.validation_trials,
        decoder_epochs=a.decoder_epochs,
        checkpoint_every=a.checkpoint_every,
        train_examples_per_label=384,
        heldout_examples_per_label=128,
        seed=a.seed,
    )
    report = run_concept_foundation(
        connectome,
        config=config,
        checkpoint_path=a.checkpoint,
        readout_dir=a.readout_dir,
        progress_path=a.progress,
    )
    report["foundation_gate"] = strict_foundation_gate(
        report,
        min_learning_gain=a.min_learning_gain,
    )
    report["final_structural"] = checkpoint_structural_summary(a.checkpoint)
    report["execution_profile"] = {
        "stage_trials": int(a.stage_trials),
        "train_examples_per_label": 384,
        "heldout_examples_per_label": 128,
        "checkpoint_every": int(a.checkpoint_every),
        "male_cns_sparse_active_state": True,
        "stimulus_index_cache": True,
    }
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    a.html.parent.mkdir(parents=True, exist_ok=True)
    a.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({
        "foundation_gate": report["foundation_gate"],
        "curriculum_order": report["curriculum_order"],
        "final_plasticity": report["final_plasticity"],
        "final_structural": report["final_structural"],
        "execution_profile": report["execution_profile"],
    }, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")
    print(f"saved dashboard: {a.html}")


if __name__ == "__main__":
    main()
