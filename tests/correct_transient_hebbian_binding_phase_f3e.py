"""Add derived F.3E performance pathology to an existing Stage-A artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/latest_transient_hebbian_binding_phase_f3e.json"


def correct(path: Path = ARTIFACT):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    performance = artifact["performance"]
    control = float(performance["control_episodes_per_second"])
    hebb = float(performance["hebb_episodes_per_second"])
    performance["performance_pathology_threshold"] = 0.50
    performance["performance_pathology"] = bool(hebb < 0.50 * control)
    artifact["validation_mode"] = {
        "stage_a_artifact_reused": True,
        "stage_a_rerun_for_derived_field": False,
        "stage_b_executed": False,
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = correct(args.artifact)
    print(json.dumps({"performance_pathology": artifact["performance"]["performance_pathology"], "training_stage_executed": artifact["training_stage_executed"]}, indent=2))


if __name__ == "__main__":
    main()
