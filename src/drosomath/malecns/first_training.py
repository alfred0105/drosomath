from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import (
    OutgoingBudgetNormalizer,
    PlasticStateConfig,
    UsageRewardRule,
)

from .brain import PlasticMaleCNSBrain
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import MaleCNSConnectome, load_malecns_v1
from .output_readout import OutputPopulation, OutputReadoutConfig, PopulationReadout
from .output_session import MaleCNSOutputSession, OutputSessionConfig


DEFAULT_RESULT = Path("results/latest_malecns_training.json")
DEFAULT_CHECKPOINT = Path("checkpoints/latest_malecns_training.npz")


@dataclass(frozen=True, slots=True)
class FirstTrainingConfig:
    min_connection_synapses: int = 5
    input_per_side: int = 32
    output_population_size: int = 512
    decoder_epochs: int = 8
    brain_trials: int = 64
    validation_trials_per_class: int = 4
    duration_ms: float = 20.0
    decoder_stimulus_rate_hz: float = 350.0
    brain_stimulus_rate_hz: float = 250.0
    brain_input_fraction: float = 0.65
    plastic_fraction: float = 0.05
    synaptic_learning_rate: float = 0.02
    decoder_learning_rate: float = 0.08
    budget_strength: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.min_connection_synapses < 1:
            raise ValueError("min_connection_synapses must be >= 1")
        if self.input_per_side < 1 or self.output_population_size < 2:
            raise ValueError("population sizes must be positive")
        if self.decoder_epochs < 1 or self.brain_trials < 1:
            raise ValueError("training counts must be positive")
        if self.validation_trials_per_class < 1:
            raise ValueError("validation_trials_per_class must be >= 1")
        if self.duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if not 0.0 < self.brain_input_fraction <= 1.0:
            raise ValueError("brain_input_fraction must be in (0, 1]")
        if not 0.0 <= self.plastic_fraction <= 1.0:
            raise ValueError("plastic_fraction must be in [0, 1]")


def _normalise_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("R"):
        return "R"
    return ""


def _top_by_outgoing(connectome: MaleCNSConnectome, indices, count: int):
    np = __import__("numpy")
    idx = np.asarray(indices, dtype=np.int32)
    if len(idx) <= count:
        order = np.argsort(connectome.outgoing_strength[idx])[::-1]
        return idx[order]
    strengths = connectome.outgoing_strength[idx]
    local = np.argpartition(strengths, -count)[-count:]
    local = local[np.argsort(strengths[local])[::-1]]
    return idx[local]


def choose_default_populations(
    connectome: MaleCNSConnectome,
    *,
    input_per_side: int,
    output_population_size: int,
) -> tuple[dict[str, tuple[int, ...]], OutputPopulation, dict[str, object]]:
    """Choose a reproducible first curriculum from real MaleCNS annotations.

    Inputs are left/right ``visual_projection`` neurons.  Outputs are real
    ``descending_neuron`` cells.  This makes the first task a biologically
    grounded left-vs-right visual discrimination rather than stimulation of
    arbitrary neuron IDs.
    """
    np = __import__("numpy")
    superclass = np.asarray(connectome.metadata.get("superclass"), dtype=object)
    if superclass.shape != (connectome.neuron_count,):
        raise ValueError("MaleCNS superclass metadata is required for auto curriculum")

    visual = np.flatnonzero(superclass == "visual_projection").astype(np.int32)
    if len(visual) < 2:
        raise ValueError("MaleCNS contains too few visual_projection neurons")

    side_field = None
    side_values = None
    for candidate in ("rootSide", "somaSide"):
        values = connectome.metadata.get(candidate)
        if values is not None and len(values) == connectome.neuron_count:
            side_field = candidate
            side_values = np.asarray(values, dtype=object)
            break

    if side_values is not None:
        sides = np.asarray([_normalise_side(x) for x in side_values], dtype=object)
        left_candidates = visual[sides[visual] == "L"]
        right_candidates = visual[sides[visual] == "R"]
    else:
        # Deterministic fallback only if side annotations are absent.
        left_candidates = visual[::2]
        right_candidates = visual[1::2]
        side_field = "deterministic_split"

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        # Some releases can leave side metadata sparse; retain a deterministic
        # fallback so a fresh public release remains runnable without hand edits.
        left_candidates = visual[::2]
        right_candidates = visual[1::2]
        side_field = "deterministic_split"

    left = _top_by_outgoing(connectome, left_candidates, input_per_side)
    right = _top_by_outgoing(connectome, right_candidates, input_per_side)
    if len(left) == 0 or len(right) == 0:
        raise ValueError("could not construct two non-empty visual input populations")

    descending = np.flatnonzero(superclass == "descending_neuron").astype(np.int32)
    if len(descending) < 2:
        raise ValueError("MaleCNS contains too few descending_neuron outputs")

    # Rank descending neurons by total anatomical connectivity.  The decoder
    # remains external; this ranking only avoids wasting its features on very
    # weakly connected cells in the first pilot.
    descending = _top_by_outgoing(connectome, descending, output_population_size)
    output_ids = tuple(int(connectome.body_ids[i]) for i in descending)
    output = OutputPopulation.from_body_ids(connectome, output_ids)

    stimuli = {
        "LEFT": tuple(int(connectome.body_ids[i]) for i in left),
        "RIGHT": tuple(int(connectome.body_ids[i]) for i in right),
    }
    provenance = {
        "input_superclass": "visual_projection",
        "input_side_field": side_field,
        "left_input_count": len(stimuli["LEFT"]),
        "right_input_count": len(stimuli["RIGHT"]),
        "output_superclass": "descending_neuron",
        "output_count": len(output.body_ids),
    }
    return stimuli, output, provenance


def _accuracy(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def _evaluate(
    session: MaleCNSOutputSession,
    stimuli: dict[str, tuple[int, ...]],
    *,
    trials_per_class: int,
    duration_ms: float,
    stimulus_rate_hz: float,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for label in session.readout.labels:
        for _ in range(trials_per_class):
            rows.append(
                session.evaluate_trial(
                    stimulus_body_ids=stimuli[label],
                    target=label,
                    duration_ms=duration_ms,
                    stimulus_rate_hz=stimulus_rate_hz,
                )
            )
    return _accuracy(rows), rows


def _degraded_stimulus(rng, ids: tuple[int, ...], fraction: float) -> tuple[int, ...]:
    if fraction >= 1.0 or len(ids) <= 1:
        return ids
    count = max(1, int(round(len(ids) * fraction)))
    chosen = rng.choice(len(ids), size=count, replace=False)
    return tuple(ids[int(i)] for i in chosen)


def save_checkpoint(path: Path, *, brain: PlasticMaleCNSBrain, readout: PopulationReadout, config: FirstTrainingConfig) -> dict[str, object]:
    np = __import__("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = np.flatnonzero(
        (np.abs(brain.plasticity.multiplier - 1.0) > 1e-7)
        | (brain.plasticity.stability > 0.0)
    ).astype(np.int32, copy=False)
    np.savez_compressed(
        path,
        changed_edge_indices=changed,
        multipliers=brain.plasticity.multiplier[changed],
        stability=brain.plasticity.stability[changed],
        readout_weights=readout.weights,
        readout_bias=readout.bias,
        output_body_ids=np.asarray(readout.population.body_ids, dtype=np.int64),
        plastic_mask_seed=np.asarray([config.seed], dtype=np.int64),
        plastic_fraction=np.asarray([config.plastic_fraction], dtype=np.float32),
    )
    return {
        "path": str(path),
        "changed_edge_count": int(len(changed)),
        "output_weight_shape": list(readout.weights.shape),
    }


def run_first_training(
    connectome: MaleCNSConnectome,
    *,
    config: FirstTrainingConfig,
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

    # Phase A: teach the fixed external decoder how to read the CNS output.
    decoder_rows: list[dict[str, object]] = []
    for _ in range(config.decoder_epochs):
        order = list(readout.labels)
        rng.shuffle(order)
        for label in order:
            obs, trained = session.train_decoder_trial(
                stimulus_body_ids=stimuli[label],
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.decoder_stimulus_rate_hz,
            )
            decoder_rows.append(
                {
                    "target": label,
                    "prediction_before": trained.prediction_before,
                    "prediction_after": trained.prediction_after,
                    "loss": trained.loss,
                    "confidence_after": trained.confidence_after,
                    "output_spikes": obs.total_output_spikes,
                }
            )

    session.freeze_decoder()
    pre_brain_accuracy, pre_eval = _evaluate(
        session,
        stimuli,
        trials_per_class=config.validation_trials_per_class,
        duration_ms=config.duration_ms,
        stimulus_rate_hz=config.brain_stimulus_rate_hz,
    )

    # Phase B: lower the drive and randomly drop some visual input neurons.  The
    # decoder is frozen; only the CNS may adapt to preserve the learned output.
    brain_rows: list[dict[str, object]] = []
    labels = list(readout.labels)
    for trial in range(config.brain_trials):
        label = labels[int(rng.integers(0, len(labels)))]
        degraded = _degraded_stimulus(rng, stimuli[label], config.brain_input_fraction)
        result = session.train_brain_trial(
            stimulus_body_ids=degraded,
            target=label,
            duration_ms=config.duration_ms,
            stimulus_rate_hz=config.brain_stimulus_rate_hz,
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

    post_accuracy, post_eval = _evaluate(
        session,
        stimuli,
        trials_per_class=config.validation_trials_per_class,
        duration_ms=config.duration_ms,
        stimulus_rate_hz=config.brain_stimulus_rate_hz,
    )

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = save_checkpoint(
            Path(checkpoint_path),
            brain=brain,
            readout=readout,
            config=config,
        )

    return {
        "experiment": "malecns_first_output_curriculum_v1",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "populations": population_info,
        "stimulus_body_ids": {key: list(value) for key, value in stimuli.items()},
        "readout": readout.summary(),
        "decoder_training": {
            "steps": len(decoder_rows),
            "last_rows": decoder_rows[-8:],
        },
        "brain_training": {
            "trials": len(brain_rows),
            "training_accuracy": _accuracy(brain_rows),
            "last_rows": brain_rows[-12:],
            "plasticity": brain.plasticity.summary(),
        },
        "evaluation": {
            "before_brain_training_accuracy": pre_brain_accuracy,
            "after_brain_training_accuracy": post_accuracy,
            "delta_accuracy": post_accuracy - pre_brain_accuracy,
            "before": pre_eval,
            "after": post_eval,
        },
        "checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DrosoMath's first automatic MaleCNS output-learning curriculum."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--input-per-side", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--decoder-epochs", type=int, default=8)
    parser.add_argument("--brain-trials", type=int, default=64)
    parser.add_argument("--validation-trials", type=int, default=4)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--plastic-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if args.download:
        download_malecns(args.data_dir)

    connectome = load_malecns_v1(
        args.data_dir,
        min_connection_synapses=args.min_syn,
    )
    config = FirstTrainingConfig(
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
    report = run_first_training(
        connectome,
        config=config,
        checkpoint_path=args.checkpoint,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved result: {args.result}")
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
