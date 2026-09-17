from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import OutgoingBudgetNormalizer, PlasticStateConfig, UsageRewardRule

from .brain import PlasticMaleCNSBrain
from .checkpoint import (
    restore_learning_checkpoint,
    restore_readout_checkpoint,
    save_learning_checkpoint,
    save_readout_checkpoint,
)
from .download import DEFAULT_DATA_DIR, download_malecns
from .first_training import choose_default_populations
from .loader import MaleCNSConnectome, load_malecns_v1
from .output_readout import OutputPopulation, OutputReadoutConfig, PopulationReadout
from .output_session import MaleCNSOutputSession, OutputSessionConfig


DEFAULT_RESULT = Path("results/latest_malecns_v1_curriculum.json")
DEFAULT_CHECKPOINT = Path("checkpoints/malecns_v1_brain.npz")
DEFAULT_READOUT_DIR = Path("checkpoints/malecns_v1_readouts")


@dataclass(frozen=True, slots=True)
class CurriculumV1Config:
    min_connection_synapses: int = 5
    output_population_size: int = 512
    visual_pool_size: int = 192
    token_size: int = 6
    decoder_epochs: int = 8
    stage_trials: int = 256
    validation_trials_per_label: int = 8
    checkpoint_every: int = 64
    duration_ms: float = 20.0
    decoder_rate_hz: float = 350.0
    learning_rate: float = 0.02
    budget_strength: float = 0.25
    silent_reward: float = -0.35
    seed: int = 7
    numeric_first: bool = False


@dataclass(frozen=True, slots=True)
class CurriculumTask:
    name: str
    labels: tuple[str, ...]
    exemplars: dict[str, tuple[tuple[int, ...], ...]]
    plastic_fraction: float

    def sample(self, label: str, rng) -> tuple[int, ...]:
        items = self.exemplars[label]
        return items[int(rng.integers(0, len(items)))]


def _top_visual_ids(connectome: MaleCNSConnectome, count: int) -> tuple[int, ...]:
    np = __import__("numpy")
    superclass = np.asarray(connectome.metadata.get("superclass"), dtype=object)
    visual = np.flatnonzero(superclass == "visual_projection").astype(np.int32)
    if len(visual) < count:
        raise ValueError(f"need {count} visual_projection neurons, found {len(visual)}")
    strength = connectome.outgoing_strength[visual]
    local = np.argpartition(strength, -count)[-count:]
    picked = visual[local[np.argsort(strength[local])[::-1]]]
    return tuple(int(connectome.body_ids[i]) for i in picked)


def _tokens(ids: tuple[int, ...], token_size: int) -> tuple[tuple[int, ...], ...]:
    if token_size < 1:
        raise ValueError("token_size must be >= 1")
    return tuple(
        tuple(ids[i : i + token_size])
        for i in range(0, len(ids) - token_size + 1, token_size)
    )


def _merge(groups) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for value in group:
            if int(value) not in seen:
                seen.add(int(value))
                out.append(int(value))
    return tuple(out)


def _random_quantity_examples(rng, bank, n: int, count: int = 12):
    examples: list[tuple[int, ...]] = []
    for _ in range(count):
        chosen = rng.choice(len(bank), size=n, replace=False)
        examples.append(_merge(bank[int(i)] for i in chosen))
    return tuple(examples)


def build_curriculum(
    connectome: MaleCNSConnectome,
    *,
    config: CurriculumV1Config,
) -> tuple[tuple[CurriculumTask, ...], OutputPopulation]:
    """Build deterministic visual-symbol tasks on real MaleCNS neurons.

    Numerosity varies token identity, so the same label is represented by many
    different visual-neuron subsets rather than one memorized fixed pattern.
    Comparison and addition use disjoint operand banks.
    """
    np = __import__("numpy")
    rng = np.random.default_rng(config.seed + 991)

    lateral, output, _ = choose_default_populations(
        connectome,
        input_per_side=32,
        output_population_size=config.output_population_size,
    )
    laterality = CurriculumTask(
        name="laterality",
        labels=("LEFT", "RIGHT"),
        exemplars={
            "LEFT": (tuple(lateral["LEFT"]),),
            "RIGHT": (tuple(lateral["RIGHT"]),),
        },
        plastic_fraction=0.05,
    )

    ids = _top_visual_ids(connectome, config.visual_pool_size)
    token_bank = _tokens(ids, config.token_size)
    if len(token_bank) < 24:
        raise ValueError("visual pool must yield at least 24 tokens")

    # Quantity is represented by the number of active object tokens, while the
    # actual token identities vary between exemplars.
    qbank = token_bank[:8]
    numerosity_examples: dict[str, tuple[tuple[int, ...], ...]] = {}
    for n in range(1, 5):
        numerosity_examples[f"N{n}"] = _random_quantity_examples(rng, qbank, n, 16)
    numerosity = CurriculumTask(
        name="numerosity_1_4",
        labels=("N1", "N2", "N3", "N4"),
        exemplars=numerosity_examples,
        plastic_fraction=0.10,
    )

    # Relational task: compare quantities represented in two disjoint banks.
    left_bank = token_bank[8:14]
    right_bank = token_bank[14:20]
    comparison: dict[str, list[tuple[int, ...]]] = {"LT": [], "EQ": [], "GT": []}
    for a in range(1, 4):
        for b in range(1, 4):
            label = "LT" if a < b else "GT" if a > b else "EQ"
            for _ in range(4):
                aa = _random_quantity_examples(rng, left_bank, a, 1)[0]
                bb = _random_quantity_examples(rng, right_bank, b, 1)[0]
                comparison[label].append(_merge((aa, bb)))
    compare_task = CurriculumTask(
        name="compare_1_3",
        labels=("LT", "EQ", "GT"),
        exemplars={k: tuple(v) for k, v in comparison.items()},
        plastic_fraction=0.15,
    )

    # Small addition. Different operand identities can map to the same sum.
    add_a = token_bank[20:24]
    add_b = token_bank[24:28] if len(token_bank) >= 28 else token_bank[4:8]
    addition: dict[str, list[tuple[int, ...]]] = {f"SUM{s}": [] for s in range(2, 7)}
    for a in range(1, 4):
        for b in range(1, 4):
            for _ in range(5):
                aa = _random_quantity_examples(rng, add_a, a, 1)[0]
                bb = _random_quantity_examples(rng, add_b, b, 1)[0]
                addition[f"SUM{a+b}"].append(_merge((aa, bb)))
    addition_task = CurriculumTask(
        name="addition_1_3",
        labels=("SUM2", "SUM3", "SUM4", "SUM5", "SUM6"),
        exemplars={k: tuple(v) for k, v in addition.items()},
        plastic_fraction=0.20,
    )

    if config.numeric_first:
        return (numerosity, compare_task, addition_task), output
    return (laterality, numerosity, compare_task, addition_task), output


def _degrade(rng, ids: tuple[int, ...], fraction: float) -> tuple[int, ...]:
    if fraction >= 1.0 or len(ids) <= 1:
        return ids
    n = max(1, int(round(len(ids) * fraction)))
    idx = rng.choice(len(ids), size=n, replace=False)
    return tuple(ids[int(i)] for i in idx)


def _make_session(brain, output, task: CurriculumTask, config: CurriculumV1Config, seed: int):
    readout = PopulationReadout(
        output,
        task.labels,
        config=OutputReadoutConfig(learning_rate=0.08, seed=seed),
    )
    session = MaleCNSOutputSession(
        brain,
        readout,
        reward_rule=UsageRewardRule(learning_rate=config.learning_rate),
        normalizer=OutgoingBudgetNormalizer(strength=config.budget_strength),
        config=OutputSessionConfig(
            correct_reward=1.0,
            incorrect_reward=-1.0,
            no_output_reward=config.silent_reward,
        ),
    )
    return readout, session


def _train_decoder(session, task, config, rng) -> dict[str, object]:
    rows = []
    for _ in range(config.decoder_epochs):
        labels = list(task.labels)
        rng.shuffle(labels)
        for label in labels:
            stimulus = task.sample(label, rng)
            obs, result = session.train_decoder_trial(
                stimulus_body_ids=stimulus,
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.decoder_rate_hz,
            )
            rows.append(
                {
                    "target": label,
                    "prediction_after": result.prediction_after,
                    "confidence_after": result.confidence_after,
                    "loss": result.loss,
                    "output_spikes": obs.total_output_spikes,
                }
            )
    session.freeze_decoder()
    return {"steps": len(rows), "last_rows": rows[-min(12, len(rows)): ]}


def _evaluate(session, task, config, *, fraction: float, rate_hz: float, seed: int):
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    rows = []
    for label in task.labels:
        for _ in range(config.validation_trials_per_label):
            stimulus = _degrade(rng, task.sample(label, rng), fraction)
            rows.append(
                session.evaluate_trial(
                    stimulus_body_ids=stimulus,
                    target=label,
                    duration_ms=config.duration_ms,
                    stimulus_rate_hz=rate_hz,
                )
            )
    accuracy = sum(bool(x["correct"]) for x in rows) / len(rows)
    silent = sum(bool(x.get("silent")) for x in rows) / len(rows)
    spikes = sum(int(x["total_output_spikes"]) for x in rows) / len(rows)
    return float(accuracy), float(silent), float(spikes), rows


def _calibrate(session, task, config, seed: int):
    grid = ((0.80, 230.0), (0.70, 205.0), (0.60, 180.0), (0.50, 155.0),
            (0.40, 135.0), (0.35, 115.0), (0.30, 100.0), (0.25, 85.0))
    candidates = []
    for i, (fraction, rate) in enumerate(grid):
        acc, silent, spikes, _ = _evaluate(
            session, task, config, fraction=fraction, rate_hz=rate, seed=seed + i
        )
        candidates.append(
            {"fraction": fraction, "rate_hz": rate, "accuracy": acc,
             "silent_fraction": silent, "mean_output_spikes": spikes}
        )
    viable = [x for x in candidates if x["silent_fraction"] <= 0.25 and 0.45 <= x["accuracy"] <= 0.80]
    pool = viable or [x for x in candidates if x["silent_fraction"] <= 0.40] or candidates
    chosen = min(pool, key=lambda x: abs(float(x["accuracy"]) - 0.65) + float(x["silent_fraction"]) * 0.5)
    return float(chosen["fraction"]), float(chosen["rate_hz"]), candidates


def _readout_path(root: Path, task_name: str) -> Path:
    return root / f"{task_name}.npz"


def run_curriculum(
    connectome: MaleCNSConnectome,
    *,
    config: CurriculumV1Config,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    readout_dir: Path = DEFAULT_READOUT_DIR,
    resume: bool = True,
) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(config.seed)
    tasks, output = build_curriculum(connectome, config=config)

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=config.seed),
    )

    resume_info = None
    if resume and checkpoint_path.is_file():
        resume_info = restore_learning_checkpoint(checkpoint_path, brain=brain)

    sessions: dict[str, MaleCNSOutputSession] = {}
    readouts: dict[str, PopulationReadout] = {}
    for i, task in enumerate(tasks):
        readout, session = _make_session(brain, output, task, config, config.seed + 100 + i)
        rp = _readout_path(readout_dir, task.name)
        if resume and rp.is_file():
            restore_readout_checkpoint(rp, readout)
        readouts[task.name] = readout
        sessions[task.name] = session

    start_stage = 0
    completed_in_stage = 0
    if resume_info:
        saved_stage = str(resume_info.get("stage", ""))
        if saved_stage == "complete":
            start_stage = len(tasks)
        else:
            for i, task in enumerate(tasks):
                if task.name == saved_stage:
                    start_stage = i
                    completed_in_stage = int(resume_info.get("completed_trials", 0))
                    if completed_in_stage >= config.stage_trials:
                        start_stage = i + 1
                        completed_in_stage = 0
                    break

    stage_reports = []
    retention_history = []

    for stage_index in range(start_stage, len(tasks)):
        task = tasks[stage_index]
        session = sessions[task.name]
        readout = readouts[task.name]
        unlock = brain.plasticity.set_plastic_fraction(task.plastic_fraction)

        if not readout.frozen:
            decoder_report = _train_decoder(session, task, config, rng)
            save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        else:
            decoder_report = {"restored": True, "steps": readout.train_steps}

        fraction, rate_hz, calibration = _calibrate(
            session, task, config, config.seed + 10000 + stage_index * 100
        )
        before_acc, before_silent, before_spikes, _ = _evaluate(
            session, task, config,
            fraction=fraction, rate_hz=rate_hz,
            seed=config.seed + 20000 + stage_index * 100,
        )

        start_trial = completed_in_stage if stage_index == start_stage else 0
        rows = []
        correct = 0
        silent_count = 0
        for trial in range(start_trial, config.stage_trials):
            label = task.labels[int(rng.integers(0, len(task.labels)))]
            stimulus = _degrade(rng, task.sample(label, rng), fraction)
            result = session.train_brain_trial(
                stimulus_body_ids=stimulus,
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=rate_hz,
            )
            correct += int(result.correct)
            silent_count += int(result.total_output_spikes == 0)
            row = {
                "trial": trial + 1,
                "target": result.target,
                "prediction": result.prediction,
                "correct": result.correct,
                "confidence": result.confidence,
                "reward": result.reward,
                "output_spikes": result.total_output_spikes,
                "edge_updates": result.learning["learning"]["edge_updates"],
            }
            rows.append(row)

            if config.checkpoint_every and (trial + 1) % config.checkpoint_every == 0:
                save_learning_checkpoint(
                    checkpoint_path,
                    brain=brain,
                    readout=readout,
                    config=config,
                    completed_trials=trial + 1,
                    stage=task.name,
                )
                save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
                print(
                    f"[{task.name}] {trial+1}/{config.stage_trials} "
                    f"acc={correct/max(1,len(rows)):.3f} silent={silent_count/max(1,len(rows)):.3f}"
                )

        after_acc, after_silent, after_spikes, _ = _evaluate(
            session, task, config,
            fraction=fraction, rate_hz=rate_hz,
            seed=config.seed + 30000 + stage_index * 100,
        )
        save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)
        save_learning_checkpoint(
            checkpoint_path,
            brain=brain,
            readout=readout,
            config=config,
            completed_trials=config.stage_trials,
            stage=task.name,
        )

        stage_reports.append(
            {
                "stage": task.name,
                "labels": list(task.labels),
                "plasticity_unlock": unlock,
                "decoder": decoder_report,
                "challenge": {
                    "fraction": fraction,
                    "rate_hz": rate_hz,
                    "calibration": calibration,
                },
                "before_accuracy": before_acc,
                "after_accuracy": after_acc,
                "delta_accuracy": after_acc - before_acc,
                "before_silent_fraction": before_silent,
                "after_silent_fraction": after_silent,
                "before_mean_spikes": before_spikes,
                "after_mean_spikes": after_spikes,
                "training_accuracy": correct / max(1, len(rows)),
                "training_silent_fraction": silent_count / max(1, len(rows)),
                "last_rows": rows[-16:],
            }
        )

        # Test all learned tasks after every stage to expose catastrophic forgetting.
        retention = {"after_stage": task.name, "tasks": {}}
        for prior in tasks[: stage_index + 1]:
            prior_session = sessions[prior.name]
            acc, silent, spikes, _ = _evaluate(
                prior_session,
                prior,
                config,
                fraction=0.70,
                rate_hz=205.0,
                seed=config.seed + 40000 + stage_index * 100 + len(retention["tasks"]),
            )
            retention["tasks"][prior.name] = {
                "accuracy": acc,
                "silent_fraction": silent,
                "mean_output_spikes": spikes,
            }
        retention_history.append(retention)
        completed_in_stage = 0

    save_learning_checkpoint(
        checkpoint_path,
        brain=brain,
        config=config,
        completed_trials=0,
        stage="complete",
    )

    plast = brain.plasticity.summary()
    changed = int(np.count_nonzero(np.abs(brain.plasticity.multiplier - 1.0) > 1e-7))
    return {
        "experiment": "malecns_curriculum_v1",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "resume": resume_info,
        "stages": stage_reports,
        "retention_history": retention_history,
        "final_plasticity": {**plast, "changed_edges": changed},
        "checkpoint": str(checkpoint_path),
        "readout_dir": str(readout_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete DrosoMath MaleCNS v1 curriculum")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--stage-trials", type=int, default=256)
    parser.add_argument("--decoder-epochs", type=int, default=8)
    parser.add_argument("--validation-trials", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--numeric-first",
        action="store_true",
        help="start with quantity concepts instead of the legacy laterality stage",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    args = parser.parse_args()

    if args.download:
        download_malecns(args.data_dir)
    connectome = load_malecns_v1(args.data_dir, min_connection_synapses=args.min_syn)
    config = CurriculumV1Config(
        min_connection_synapses=args.min_syn,
        stage_trials=args.stage_trials,
        decoder_epochs=args.decoder_epochs,
        validation_trials_per_label=args.validation_trials,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        numeric_first=args.numeric_first,
    )
    report = run_curriculum(
        connectome,
        config=config,
        checkpoint_path=args.checkpoint,
        readout_dir=args.readout_dir,
        resume=not args.fresh,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "experiment": report["experiment"],
        "stages": [
            {k: stage[k] for k in ("stage", "before_accuracy", "after_accuracy", "delta_accuracy", "training_accuracy")}
            for stage in report["stages"]
        ],
        "final_plasticity": report["final_plasticity"],
        "checkpoint": report["checkpoint"],
    }, indent=2, sort_keys=True))
    print(f"saved result: {args.result}")


if __name__ == "__main__":
    main()
