from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import OutgoingBudgetNormalizer, PlasticStateConfig, UsageRewardRule

from .brain import PlasticMaleCNSBrain
from .download import DEFAULT_DATA_DIR, download_malecns
from .first_training import choose_default_populations, save_checkpoint
from .loader import MaleCNSConnectome, load_malecns_v1
from .output_readout import OutputReadoutConfig, PopulationReadout
from .output_session import MaleCNSOutputSession, OutputSessionConfig


DEFAULT_RESULT = Path("results/latest_malecns_adaptive_training.json")
DEFAULT_CHECKPOINT = Path("checkpoints/latest_malecns_adaptive_training.npz")


@dataclass(frozen=True, slots=True)
class AdaptiveTrainingConfig:
    min_connection_synapses: int = 5
    input_per_side: int = 32
    output_population_size: int = 512
    decoder_epochs: int = 8
    calibration_trials_per_class: int = 6
    validation_trials_per_class: int = 16
    brain_trials: int = 192
    duration_ms: float = 20.0
    decoder_stimulus_rate_hz: float = 350.0
    plastic_fraction: float = 0.05
    synaptic_learning_rate: float = 0.02
    decoder_learning_rate: float = 0.08
    budget_strength: float = 0.25
    target_baseline_low: float = 0.55
    target_baseline_high: float = 0.80
    seed: int = 1

    def __post_init__(self) -> None:
        if self.min_connection_synapses < 1:
            raise ValueError("min_connection_synapses must be >= 1")
        if self.input_per_side < 1 or self.output_population_size < 2:
            raise ValueError("population sizes must be positive")
        if self.decoder_epochs < 1 or self.brain_trials < 1:
            raise ValueError("training counts must be positive")
        if self.calibration_trials_per_class < 1 or self.validation_trials_per_class < 1:
            raise ValueError("evaluation counts must be positive")
        if self.duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if not 0.0 <= self.plastic_fraction <= 1.0:
            raise ValueError("plastic_fraction must be in [0, 1]")
        if not 0.0 <= self.target_baseline_low <= self.target_baseline_high <= 1.0:
            raise ValueError("invalid target baseline interval")


# Increasingly difficult visual degradation.  The calibration phase selects a
# non-silent condition near the requested 55-80% pre-learning accuracy window.
CHALLENGE_GRID: tuple[tuple[float, float], ...] = (
    (0.60, 230.0),
    (0.50, 205.0),
    (0.40, 180.0),
    (0.35, 155.0),
    (0.30, 135.0),
    (0.25, 115.0),
    (0.20, 95.0),
    (0.15, 80.0),
    (0.10, 65.0),
)


def _degraded(rng, ids: tuple[int, ...], fraction: float) -> tuple[int, ...]:
    count = max(1, int(round(len(ids) * fraction)))
    if count >= len(ids):
        return ids
    pick = rng.choice(len(ids), size=count, replace=False)
    return tuple(ids[int(i)] for i in pick)


def _evaluate_degraded(
    session: MaleCNSOutputSession,
    stimuli: dict[str, tuple[int, ...]],
    *,
    fraction: float,
    rate_hz: float,
    trials_per_class: int,
    duration_ms: float,
    seed: int,
) -> tuple[float, float, list[dict[str, object]]]:
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for label in session.readout.labels:
        for _ in range(trials_per_class):
            subset = _degraded(rng, stimuli[label], fraction)
            row = session.evaluate_trial(
                stimulus_body_ids=subset,
                target=label,
                duration_ms=duration_ms,
                stimulus_rate_hz=rate_hz,
            )
            rows.append(row)
    accuracy = sum(bool(row["correct"]) for row in rows) / len(rows)
    mean_spikes = sum(int(row["total_output_spikes"]) for row in rows) / len(rows)
    return float(accuracy), float(mean_spikes), rows


def _select_challenge(
    session: MaleCNSOutputSession,
    stimuli: dict[str, tuple[int, ...]],
    config: AdaptiveTrainingConfig,
) -> tuple[float, float, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for i, (fraction, rate_hz) in enumerate(CHALLENGE_GRID):
        acc, mean_spikes, _ = _evaluate_degraded(
            session,
            stimuli,
            fraction=fraction,
            rate_hz=rate_hz,
            trials_per_class=config.calibration_trials_per_class,
            duration_ms=config.duration_ms,
            seed=config.seed + 1000 + i,
        )
        candidates.append(
            {
                "fraction": fraction,
                "rate_hz": rate_hz,
                "accuracy": acc,
                "mean_output_spikes": mean_spikes,
            }
        )

    target = (config.target_baseline_low + config.target_baseline_high) / 2.0
    viable = [
        row
        for row in candidates
        if row["mean_output_spikes"] > 0.0
        and config.target_baseline_low <= row["accuracy"] <= config.target_baseline_high
    ]
    pool = viable or [row for row in candidates if row["mean_output_spikes"] > 0.0]
    if not pool:
        raise RuntimeError("all automatic challenge conditions silenced the output population")
    chosen = min(pool, key=lambda row: abs(float(row["accuracy"]) - target))
    return float(chosen["fraction"]), float(chosen["rate_hz"]), candidates


def _plasticity_stats(brain: PlasticMaleCNSBrain) -> dict[str, object]:
    np = __import__("numpy")
    mask = brain.plasticity.plastic_mask
    multipliers = brain.plasticity.multiplier[mask]
    changed = np.abs(multipliers - 1.0) > 1e-7
    changed_values = multipliers[changed]
    if len(changed_values):
        quantiles = np.quantile(changed_values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        q = [float(x) for x in quantiles]
    else:
        q = [1.0] * 7
    return {
        "plastic_edges": int(mask.sum()),
        "changed_edges": int(changed.sum()),
        "mean_multiplier": float(multipliers.mean()) if len(multipliers) else 1.0,
        "mean_stability": float(brain.plasticity.stability[mask].mean()) if mask.any() else 0.0,
        "multiplier_quantiles": {
            "min": q[0], "p10": q[1], "p25": q[2], "median": q[3],
            "p75": q[4], "p90": q[5], "max": q[6],
        },
    }


def run_adaptive_training(
    connectome: MaleCNSConnectome,
    *,
    config: AdaptiveTrainingConfig,
    checkpoint_path: Path | None = DEFAULT_CHECKPOINT,
) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(config.seed)
    stimuli, output, population_info = choose_default_populations(
        connectome,
        input_per_side=config.input_per_side,
        output_population_size=config.output_population_size,
    )

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=config.seed,
        ),
    )
    readout = PopulationReadout(
        output,
        ("LEFT", "RIGHT"),
        config=OutputReadoutConfig(
            learning_rate=config.decoder_learning_rate,
            seed=config.seed,
        ),
    )
    session = MaleCNSOutputSession(
        brain,
        readout,
        reward_rule=UsageRewardRule(learning_rate=config.synaptic_learning_rate),
        normalizer=OutgoingBudgetNormalizer(strength=config.budget_strength),
        config=OutputSessionConfig(correct_reward=1.0, incorrect_reward=-1.0),
    )

    # A. Teach only the fixed external decoder on clean left/right examples.
    decoder_rows: list[dict[str, object]] = []
    for _ in range(config.decoder_epochs):
        labels = list(readout.labels)
        rng.shuffle(labels)
        for label in labels:
            obs, trained = session.train_decoder_trial(
                stimulus_body_ids=stimuli[label],
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.decoder_stimulus_rate_hz,
            )
            decoder_rows.append(
                {
                    "target": label,
                    "loss": trained.loss,
                    "prediction_after": trained.prediction_after,
                    "confidence_after": trained.confidence_after,
                    "output_spikes": obs.total_output_spikes,
                }
            )
    session.freeze_decoder()

    # B. Find a challenge the decoder cannot already solve perfectly.
    fraction, rate_hz, calibration = _select_challenge(session, stimuli, config)

    # C. Measure the frozen brain on fresh degraded stimuli.
    before_acc, before_spikes, before_rows = _evaluate_degraded(
        session,
        stimuli,
        fraction=fraction,
        rate_hz=rate_hz,
        trials_per_class=config.validation_trials_per_class,
        duration_ms=config.duration_ms,
        seed=config.seed + 20_000,
    )

    # D. Decoder remains frozen; only MaleCNS local synaptic state may change.
    brain_rows: list[dict[str, object]] = []
    labels = list(readout.labels)
    for trial in range(config.brain_trials):
        label = labels[int(rng.integers(0, len(labels)))]
        subset = _degraded(rng, stimuli[label], fraction)
        result = session.train_brain_trial(
            stimulus_body_ids=subset,
            target=label,
            duration_ms=config.duration_ms,
            stimulus_rate_hz=rate_hz,
        )
        brain_rows.append(
            {
                "trial": trial + 1,
                "target": result.target,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "correct": result.correct,
                "reward": result.reward,
                "output_spikes": result.total_output_spikes,
                "edge_updates": result.learning["learning"]["edge_updates"],
            }
        )

    # E. Re-evaluate under the same challenge distribution with a fresh mask stream.
    after_acc, after_spikes, after_rows = _evaluate_degraded(
        session,
        stimuli,
        fraction=fraction,
        rate_hz=rate_hz,
        trials_per_class=config.validation_trials_per_class,
        duration_ms=config.duration_ms,
        seed=config.seed + 30_000,
    )

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = save_checkpoint(Path(checkpoint_path), brain=brain, readout=readout, config=config)

    return {
        "experiment": "malecns_adaptive_output_curriculum_v2",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "populations": population_info,
        "stimulus_body_ids": {key: list(value) for key, value in stimuli.items()},
        "decoder_training": {
            "steps": len(decoder_rows),
            "last_rows": decoder_rows[-8:],
            "readout": readout.summary(),
        },
        "challenge": {
            "selected_input_fraction": fraction,
            "selected_stimulus_rate_hz": rate_hz,
            "calibration_candidates": calibration,
        },
        "evaluation": {
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "delta_accuracy": after_acc - before_acc,
            "before_mean_output_spikes": before_spikes,
            "after_mean_output_spikes": after_spikes,
            "before": before_rows,
            "after": after_rows,
        },
        "brain_training": {
            "trials": len(brain_rows),
            "training_accuracy": sum(bool(row["correct"]) for row in brain_rows) / len(brain_rows),
            "rows": brain_rows,
        },
        "plasticity": _plasticity_stats(brain),
        "checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an automatically calibrated MaleCNS learning challenge.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--input-per-side", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--decoder-epochs", type=int, default=8)
    parser.add_argument("--brain-trials", type=int, default=192)
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--plastic-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if args.download:
        download_malecns(args.data_dir)
    connectome = load_malecns_v1(args.data_dir, min_connection_synapses=args.min_syn)
    config = AdaptiveTrainingConfig(
        min_connection_synapses=args.min_syn,
        input_per_side=args.input_per_side,
        output_population_size=args.output_size,
        decoder_epochs=args.decoder_epochs,
        brain_trials=args.brain_trials,
        validation_trials_per_class=args.validation_trials,
        duration_ms=args.duration_ms,
        plastic_fraction=args.plastic_fraction,
        seed=args.seed,
    )
    report = run_adaptive_training(connectome, config=config, checkpoint_path=args.checkpoint)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "experiment": report["experiment"],
        "challenge": report["challenge"],
        "evaluation": {k: v for k, v in report["evaluation"].items() if k not in ("before", "after")},
        "plasticity": report["plasticity"],
    }, indent=2, sort_keys=True))
    print(f"saved result: {args.result}")
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
