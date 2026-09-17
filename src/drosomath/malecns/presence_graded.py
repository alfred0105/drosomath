from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import OutgoingBudgetNormalizer, PlasticStateConfig, UsageRewardRule

from .brain import PlasticMaleCNSBrain
from .checkpoint import save_learning_checkpoint
from .concept_foundation import ConceptFoundationConfig, build_concept_foundation
from .download import DEFAULT_DATA_DIR, download_malecns
from .graded_output import (
    GradedPopulationSession,
    GradedRateCodeConfig,
    calibrate_two_level_rate_code,
)
from .loader import load_malecns_v1


DEFAULT_RESULT = Path("results/latest_malecns_presence_graded.json")
DEFAULT_HTML = Path("results/latest_malecns_presence_graded.html")
DEFAULT_PROGRESS = Path("results/malecns_presence_graded_progress.json")
DEFAULT_CHECKPOINT = Path("checkpoints/malecns_presence_graded_brain.npz")


@dataclass(frozen=True, slots=True)
class PresenceMasteryConfig:
    min_connection_synapses: int = 5
    duration_ms: float = 100.0
    stimulus_rate_hz: float = 205.0
    calibration_trials_per_class: int = 32
    calibration_min_separation_hz: float = 0.10
    representation_probe_size: int = 512
    representation_trials_per_class: int = 16
    fixed_full_trials: int = 2048
    varied_full_trials: int = 3072
    varied_dropout_trials: int = 3072
    validation_trials_per_class: int = 64
    mastery_accuracy: float = 0.85
    max_attempts_per_phase: int = 3
    checkpoint_every: int = 512
    learning_rate: float = 0.02
    budget_strength: float = 0.25
    base_rate_hz: float = 1.0
    level_step_hz: float = 1.5
    tolerance_hz: float = 0.50
    max_output_level: int = 7
    seed: int = 7

    def __post_init__(self) -> None:
        if self.fixed_full_trials < 1 or self.varied_full_trials < 1 or self.varied_dropout_trials < 1:
            raise ValueError("phase trial counts must be positive")
        if self.validation_trials_per_class < 1:
            raise ValueError("validation_trials_per_class must be positive")
        if self.calibration_trials_per_class < 2:
            raise ValueError("calibration_trials_per_class must be >= 2")
        if self.calibration_min_separation_hz < 0.0:
            raise ValueError("calibration_min_separation_hz must be >= 0")
        if self.representation_probe_size < 1:
            raise ValueError("representation_probe_size must be positive")
        if self.representation_trials_per_class < 2:
            raise ValueError("representation_trials_per_class must be >= 2")
        if not 0.5 < self.mastery_accuracy <= 1.0:
            raise ValueError("mastery_accuracy must be in (0.5, 1]")
        if self.max_attempts_per_phase < 1:
            raise ValueError("max_attempts_per_phase must be >= 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")


@dataclass(frozen=True, slots=True)
class PresencePhase:
    name: str
    trials: int
    varied_position: bool
    keep_min: float
    keep_max: float
    evaluation_split: str


def _choose_representation_probe(connectome, field, output, size: int):
    np = __import__("numpy")
    if size < 1:
        raise ValueError("representation probe size must be positive")
    input_ids = set(int(x) for x in field.background_body_ids)
    input_ids.update(int(x) for group in field.position_body_ids for x in group)
    input_indices = set(int(connectome.index_of(x)) for x in input_ids)
    output_indices = set(int(x) for x in output.indices)
    excluded = input_indices | output_indices
    candidates = np.asarray(
        [i for i in range(connectome.neuron_count) if i not in excluded],
        dtype=np.int32,
    )
    if len(candidates) < size:
        raise ValueError("not enough non-input, non-output neurons for representation probe")
    return candidates[:size]


def _cosine(a, b) -> float:
    np = __import__("numpy")
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _mean_pairwise(vectors) -> float:
    if len(vectors) < 2:
        return 0.0
    values = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            values.append(_cosine(vectors[i], vectors[j]))
    return sum(values) / len(values)


def _representation_audit(session, field, phase, config: PresenceMasteryConfig, *, seed: int) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    vectors = {0: [], 1: []}
    for target_level in (0, 1):
        for _ in range(config.representation_trials_per_class):
            stimulus = _scene(
                field,
                present=bool(target_level),
                phase=phase,
                split="heldout",
                rng=rng,
            )
            _, vector = session.measure_trial_activity(
                stimulus_body_ids=stimulus,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.stimulus_rate_hz,
            )
            vectors[target_level].append(np.asarray(vector, dtype=np.float32))

    between = []
    for left in vectors[0]:
        for right in vectors[1]:
            between.append(_cosine(left, right))
    all_vectors = vectors[0] + vectors[1]
    return {
        "probe_size": int(len(all_vectors[0])) if all_vectors else 0,
        "within_background_cosine": _mean_pairwise(vectors[0]),
        "within_present_cosine": _mean_pairwise(vectors[1]),
        "between_class_cosine": sum(between) / len(between) if between else 0.0,
        "invariance_margin": _mean_pairwise(vectors[1]) - (sum(between) / len(between) if between else 0.0),
        "nonzero_vector_fraction": sum(int(float(np.linalg.norm(x)) > 0.0) for x in all_vectors) / max(1, len(all_vectors)),
        "split": "heldout",
    }


def _phases(config: PresenceMasteryConfig) -> tuple[PresencePhase, ...]:
    return (
        PresencePhase(
            name="presence_fixed_full",
            trials=config.fixed_full_trials,
            varied_position=False,
            keep_min=1.0,
            keep_max=1.0,
            evaluation_split="train_fixed",
        ),
        PresencePhase(
            name="presence_varied_full",
            trials=config.varied_full_trials,
            varied_position=True,
            keep_min=1.0,
            keep_max=1.0,
            evaluation_split="heldout",
        ),
        PresencePhase(
            name="presence_varied_dropout",
            trials=config.varied_dropout_trials,
            varied_position=True,
            keep_min=0.50,
            keep_max=1.0,
            evaluation_split="heldout",
        ),
    )


def mastery_passed(accuracy: float, threshold: float) -> bool:
    return float(accuracy) >= float(threshold)


def _scene(field, *, present: bool, phase: PresencePhase, split: str, rng) -> tuple[int, ...]:
    if not present:
        return field.encode((), rng=rng, keep_min=1.0, keep_max=1.0)

    if split == "train_fixed" or not phase.varied_position:
        pool = field.train_positions
        position = int(pool[len(pool) // 2])
    elif split == "heldout":
        pool = field.heldout_positions
        position = int(pool[int(rng.integers(0, len(pool)))])
    else:
        pool = field.train_positions
        position = int(pool[int(rng.integers(0, len(pool)))])

    return field.encode(
        (position,),
        rng=rng,
        keep_min=phase.keep_min,
        keep_max=phase.keep_max,
    )


def _evaluate(session, field, phase: PresencePhase, config: PresenceMasteryConfig, *, seed: int) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    rows = []
    split = phase.evaluation_split
    for present, target_level in ((False, 0), (True, 1)):
        for _ in range(config.validation_trials_per_class):
            stimulus = _scene(field, present=present, phase=phase, split=split, rng=rng)
            obs = session.evaluate_trial(
                stimulus_body_ids=stimulus,
                target_level=target_level,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.stimulus_rate_hz,
            )
            rows.append(obs)

    n = len(rows)
    correct = sum(int(x.correct) for x in rows)
    rates = [float(x.population_rate_hz) for x in rows]
    no_output = sum(int(x.status == "NO_OUTPUT") for x in rows)
    between = sum(int(x.status == "BETWEEN_LEVELS") for x in rows)
    return {
        "accuracy": correct / max(1, n),
        "correct": correct,
        "trials": n,
        "mean_population_rate_hz": sum(rates) / max(1, n),
        "no_output_fraction": no_output / max(1, n),
        "between_levels_fraction": between / max(1, n),
        "split": split,
    }


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_presence_mastery(
    connectome,
    *,
    config: PresenceMasteryConfig,
    result_path: Path = DEFAULT_RESULT,
    progress_path: Path = DEFAULT_PROGRESS,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, object]:
    np = __import__("numpy")

    # Reuse the exact v1 virtual retina/output selection so this run is directly
    # comparable to the decoder baseline.  Tiny example banks are sufficient here
    # because this curriculum samples scenes online.
    foundation_config = ConceptFoundationConfig(
        min_connection_synapses=config.min_connection_synapses,
        train_examples_per_label=4,
        heldout_examples_per_label=4,
        anchor_examples_per_label=1,
        stage_trials=1,
        validation_trials_per_label=4,
        seed=config.seed,
    )
    bundle = build_concept_foundation(connectome, config=foundation_config)
    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=config.seed),
    )
    representation_probe = _choose_representation_probe(
        connectome,
        bundle.field,
        bundle.output,
        config.representation_probe_size,
    )
    session = GradedPopulationSession(
        brain,
        bundle.output,
        code_config=GradedRateCodeConfig(
            base_rate_hz=config.base_rate_hz,
            level_step_hz=config.level_step_hz,
            tolerance_hz=config.tolerance_hz,
            max_level=config.max_output_level,
        ),
        reward_rule=UsageRewardRule(learning_rate=config.learning_rate),
        normalizer=OutgoingBudgetNormalizer(strength=config.budget_strength),
        probe_indices=representation_probe,
    )

    # Calibrate the fixed rate code from the untrained MaleCNS output
    # distribution.  This is measurement only: tracking is disabled and no
    # synaptic state is changed.  Do not train if the two classes are not
    # separable at the chosen output population and time window.
    calibration_phase = _phases(config)[0]
    calibration_rng = np.random.default_rng(config.seed + 90_000)
    background_rates = []
    present_rates = []
    for _ in range(config.calibration_trials_per_class):
        background = _scene(
            bundle.field,
            present=False,
            phase=calibration_phase,
            split="train",
            rng=calibration_rng,
        )
        present = _scene(
            bundle.field,
            present=True,
            phase=calibration_phase,
            split="train",
            rng=calibration_rng,
        )
        background_rates.append(
            session.measure_trial_rate(
                stimulus_body_ids=background,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.stimulus_rate_hz,
            ).population_rate_hz
        )
        present_rates.append(
            session.measure_trial_rate(
                stimulus_body_ids=present,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.stimulus_rate_hz,
            ).population_rate_hz
        )

    calibration = calibrate_two_level_rate_code(
        background_rates,
        present_rates,
        min_separation_hz=config.calibration_min_separation_hz,
    )
    if calibration.usable:
        session.code.config = calibration.code_config(
            max_level=config.max_output_level,
        )
    else:
        report = {
            "experiment": "malecns_presence_graded_v1",
            "purpose": "master object absence/presence before any quantity curriculum",
            "config": asdict(config),
            "connectome": connectome.summary(),
            "provenance": {
                **bundle.provenance,
                "representation_probe_size": len(representation_probe),
                "representation_probe_scope": "deterministic non-input, non-output neurons",
            },
            "calibration": calibration.summary(),
            "output_code": session.code.summary(),
            "phase_reports": [],
            "completed_training_trials": 0,
            "curriculum_passed": False,
            "failed_phase": "output_rate_calibration",
            "next_curriculum_unlocked": False,
            "final_plasticity": {**brain.plasticity.summary(), "changed_edges": 0},
            "structural": brain.structural_summary(),
            "execution_profile": {
                "backend": "numpy_cpu",
                "gpu_used": False,
                "sparse_active_state": True,
                "output_rate_calibration": True,
            },
            "checkpoint": str(checkpoint_path),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _write_progress(progress_path, report)
        return report

    phase_reports = []
    completed_training_trials = 0
    curriculum_passed = True
    failed_phase = None

    for phase_index, phase in enumerate(_phases(config)):
        attempts = []
        before = _evaluate(
            session,
            bundle.field,
            phase,
            config,
            seed=config.seed + 100_000 + phase_index * 10_000,
        )
        representation_before = _representation_audit(
            session,
            bundle.field,
            phase,
            config,
            seed=config.seed + 110_000 + phase_index * 10_000,
        )
        phase_passed = False

        for attempt_index in range(config.max_attempts_per_phase):
            correct = 0
            between = 0
            signed_error_sum = 0.0
            last_rows = []

            for trial in range(phase.trials):
                rng = np.random.default_rng(
                    config.seed
                    + phase_index * 10_000_000
                    + attempt_index * 1_000_000
                    + trial
                )
                present = bool(int(rng.integers(0, 2)))
                target_level = 1 if present else 0
                stimulus = _scene(
                    bundle.field,
                    present=present,
                    phase=phase,
                    split="train",
                    rng=rng,
                )
                result = session.train_trial(
                    stimulus_body_ids=stimulus,
                    target_level=target_level,
                    duration_ms=config.duration_ms,
                    stimulus_rate_hz=config.stimulus_rate_hz,
                )
                obs = result.observation
                correct += int(obs.correct)
                between += int(obs.status == "BETWEEN_LEVELS")
                signed_error_sum += float(obs.signed_error_hz or 0.0)
                completed_training_trials += 1

                if len(last_rows) >= 16:
                    last_rows.pop(0)
                last_rows.append(
                    {
                        "trial": trial + 1,
                        "target_level": target_level,
                        "predicted_level": obs.predicted_level,
                        "status": obs.status,
                        "rate_hz": obs.population_rate_hz,
                        "signed_error_hz": obs.signed_error_hz,
                        "teaching_signal": obs.teaching_signal,
                        "correct": obs.correct,
                    }
                )

                if completed_training_trials % config.checkpoint_every == 0:
                    save_learning_checkpoint(
                        checkpoint_path,
                        brain=brain,
                        config=config,
                        completed_trials=completed_training_trials,
                        stage=f"{phase.name}_attempt_{attempt_index + 1}",
                    )

            evaluation = _evaluate(
                session,
                bundle.field,
                phase,
                config,
                seed=config.seed + 200_000 + phase_index * 10_000 + attempt_index,
            )
            representation_after = _representation_audit(
                session,
                bundle.field,
                phase,
                config,
                seed=config.seed + 210_000 + phase_index * 10_000 + attempt_index,
            )
            phase_passed = mastery_passed(evaluation["accuracy"], config.mastery_accuracy)
            attempt_report = {
                "attempt": attempt_index + 1,
                "training_trials": phase.trials,
                "training_accuracy": correct / max(1, phase.trials),
                "training_between_levels_fraction": between / max(1, phase.trials),
                "mean_signed_error_hz": signed_error_sum / max(1, phase.trials),
                "evaluation": evaluation,
                "representation": representation_after,
                "mastery_threshold": config.mastery_accuracy,
                "passed": phase_passed,
                "last_rows": last_rows,
            }
            attempts.append(attempt_report)

            partial = {
                "experiment": "malecns_presence_graded_v1",
                "config": asdict(config),
                "current_phase": phase.name,
                "phase_reports": phase_reports + [{
                    "phase": phase.name,
                    "before": before,
                    "representation_before": representation_before,
                    "attempts": attempts,
                    "passed": phase_passed,
                }],
                "completed_training_trials": completed_training_trials,
                "output_code": session.code.summary(),
            }
            _write_progress(progress_path, partial)

            if phase_passed:
                break

        phase_reports.append(
            {
                "phase": phase.name,
                "training_trials_per_attempt": phase.trials,
                "evaluation_split": phase.evaluation_split,
                "before": before,
                "attempts": attempts,
                "attempts_used": len(attempts),
                "retraining_count": max(0, len(attempts) - 1),
                "passed": phase_passed,
            }
        )

        save_learning_checkpoint(
            checkpoint_path,
            brain=brain,
            config=config,
            completed_trials=completed_training_trials,
            stage=phase.name,
        )

        if not phase_passed:
            curriculum_passed = False
            failed_phase = phase.name
            break

    plast = brain.plasticity.summary()
    changed = int(np.count_nonzero(np.abs(brain.plasticity.multiplier - 1.0) > 1e-7))
    report = {
        "experiment": "malecns_presence_graded_v1",
        "purpose": "master object absence/presence before any quantity curriculum",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "provenance": {
            **bundle.provenance,
            "representation_probe_size": len(representation_probe),
            "representation_probe_scope": "deterministic non-input, non-output neurons",
        },
        "calibration": calibration.summary(),
        "output_code": session.code.summary(),
        "phase_reports": phase_reports,
        "completed_training_trials": completed_training_trials,
        "curriculum_passed": curriculum_passed,
        "failed_phase": failed_phase,
        "next_curriculum_unlocked": bool(curriculum_passed),
        "final_plasticity": {**plast, "changed_edges": changed},
        "structural": brain.structural_summary(),
        "execution_profile": {
            "backend": "numpy_cpu",
            "gpu_used": False,
            "sparse_active_state": True,
            "output_rate_calibration": True,
        },
        "checkpoint": str(checkpoint_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_progress(progress_path, report)
    return report


def build_html(report: dict[str, object]) -> str:
    rows = []
    for phase in report["phase_reports"]:
        last = phase["attempts"][-1]
        before = 100.0 * float(phase["before"]["accuracy"])
        after = 100.0 * float(last["evaluation"]["accuracy"])
        rows.append(
            f"<tr><td>{phase['phase']}</td><td>{before:.1f}%</td>"
            f"<td>{phase['attempts_used']}</td><td>{after:.1f}%</td>"
            f"<td>{'PASS' if phase['passed'] else 'FAIL'}</td></tr>"
        )
    status = "PASS" if report["curriculum_passed"] else "STOPPED AFTER RETRAINING"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>DrosoMath Presence Mastery</title>
<style>body{{font-family:system-ui;background:#101318;color:#e8edf5;max-width:1050px;margin:auto;padding:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #2b3442;text-align:right}}th:first-child,td:first-child{{text-align:left}}.card{{padding:16px;border:1px solid #2b3442;border-radius:12px;background:#171c24}}</style></head><body>
<h1>DrosoMath Presence Mastery</h1><div class='card'>Status: <b>{status}</b><br>Fixed graded population output; no trainable external decoder.<br>Each phase receives up to {report['config']['max_attempts_per_phase']} education attempts before stopping.</div>
<h2>Mastery phases</h2><table><tr><th>Phase</th><th>Before</th><th>Attempts</th><th>Final</th><th>Status</th></tr>{''.join(rows)}</table>
<p>Quantity 2/3 and number symbols are locked until all presence phases pass.</p></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Train MaleCNS on graded absence/presence mastery before quantity")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--fixed-trials", type=int, default=2048)
    p.add_argument("--varied-trials", type=int, default=3072)
    p.add_argument("--dropout-trials", type=int, default=3072)
    p.add_argument("--duration-ms", type=float, default=100.0)
    p.add_argument("--stimulus-rate-hz", type=float, default=205.0)
    p.add_argument("--calibration-trials", type=int, default=32)
    p.add_argument("--calibration-min-separation-hz", type=float, default=0.10)
    p.add_argument("--representation-probe-size", type=int, default=512)
    p.add_argument("--representation-trials", type=int, default=16)
    p.add_argument("--validation-trials", type=int, default=64)
    p.add_argument("--mastery", type=float, default=0.85)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--checkpoint-every", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=DEFAULT_HTML)
    p.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    a = p.parse_args()

    if a.download:
        download_malecns(a.data_dir)
    connectome = load_malecns_v1(a.data_dir, min_connection_synapses=a.min_syn)
    config = PresenceMasteryConfig(
        min_connection_synapses=a.min_syn,
        fixed_full_trials=a.fixed_trials,
        varied_full_trials=a.varied_trials,
        varied_dropout_trials=a.dropout_trials,
        duration_ms=a.duration_ms,
        stimulus_rate_hz=a.stimulus_rate_hz,
        calibration_trials_per_class=a.calibration_trials,
        calibration_min_separation_hz=a.calibration_min_separation_hz,
        representation_probe_size=a.representation_probe_size,
        representation_trials_per_class=a.representation_trials,
        validation_trials_per_class=a.validation_trials,
        mastery_accuracy=a.mastery,
        max_attempts_per_phase=a.max_attempts,
        checkpoint_every=a.checkpoint_every,
        seed=a.seed,
    )
    report = run_presence_mastery(
        connectome,
        config=config,
        result_path=a.result,
        progress_path=a.progress,
        checkpoint_path=a.checkpoint,
    )
    a.html.parent.mkdir(parents=True, exist_ok=True)
    a.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({
        "curriculum_passed": report["curriculum_passed"],
        "failed_phase": report["failed_phase"],
        "completed_training_trials": report["completed_training_trials"],
        "output_code": report["output_code"],
        "phase_reports": [
            {
                "phase": x["phase"],
                "attempts_used": x["attempts_used"],
                "passed": x["passed"],
                "final_accuracy": x["attempts"][-1]["evaluation"]["accuracy"],
            }
            for x in report["phase_reports"]
        ],
    }, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")
    print(f"saved dashboard: {a.html}")


if __name__ == "__main__":
    main()
