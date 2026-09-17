from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import (
    AdaptiveReplayConfig,
    AdaptiveReplayScheduler,
    ConsolidationConfig,
    MemoryConsolidator,
    PlasticStateConfig,
)

from .brain import PlasticMaleCNSBrain
from .checkpoint import save_learning_checkpoint, save_readout_checkpoint
from .curriculum_v1 import _readout_path, _top_visual_ids
from .curriculum_v2_memory import MemoryV2Config, _make_memory_session
from .download import DEFAULT_DATA_DIR, download_malecns
from .first_training import choose_route_aware_output_population
from .loader import load_malecns_v1


DEFAULT_RESULT = Path("results/latest_malecns_concept_foundation.json")
DEFAULT_CHECKPOINT = Path("checkpoints/malecns_concept_foundation_brain.npz")
DEFAULT_READOUT_DIR = Path("checkpoints/malecns_concept_foundation_readouts")
DEFAULT_PROGRESS = Path("results/malecns_concept_foundation_progress.json")


@dataclass(frozen=True, slots=True)
class ConceptFoundationConfig(MemoryV2Config):
    """Concept-first curriculum before symbols, comparison, or arithmetic.

    The virtual point field is only an experimental encoder over real MaleCNS
    visual_projection neurons. Grid coordinates are NOT claimed to be biological
    retinotopic coordinates.
    """

    field_width: int = 5
    field_height: int = 5
    neurons_per_position: int = 6
    background_neurons: int = 6
    train_examples_per_label: int = 96
    heldout_examples_per_label: int = 64
    anchor_examples_per_label: int = 2
    object_neuron_keep_min: float = 0.50
    object_neuron_keep_max: float = 1.00
    stage_trials: int = 512
    validation_trials_per_label: int = 32
    replay_interval: int = 4
    seed: int = 7

    def __post_init__(self) -> None:
        if self.field_width < 3 or self.field_height < 3:
            raise ValueError("point field must be at least 3x3")
        if self.neurons_per_position < 2 or self.background_neurons < 1:
            raise ValueError("neuron groups must be non-empty")
        if self.train_examples_per_label < 4 or self.heldout_examples_per_label < 4:
            raise ValueError("need multiple train/held-out examples per label")
        if self.anchor_examples_per_label < 1:
            raise ValueError("anchor_examples_per_label must be >= 1")
        if not 0.0 < self.object_neuron_keep_min <= self.object_neuron_keep_max <= 1.0:
            raise ValueError("object neuron keep range must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class VirtualPointField:
    width: int
    height: int
    neurons_per_position: int
    background_body_ids: tuple[int, ...]
    position_body_ids: tuple[tuple[int, ...], ...]
    train_positions: tuple[int, ...]
    heldout_positions: tuple[int, ...]

    def xy(self, index: int) -> tuple[int, int]:
        return int(index % self.width), int(index // self.width)

    def encode(
        self,
        positions: tuple[int, ...],
        *,
        rng,
        keep_min: float,
        keep_max: float,
    ) -> tuple[int, ...]:
        """Encode point positions as real visual_projection body IDs.

        Every scene carries a small background/fixation population so an empty
        scene is a real neural input rather than NO_OUTPUT/silence. Per-object
        dropout prevents exact active-neuron count from being the only cue.
        """
        out = list(self.background_body_ids)
        seen = set(out)
        for position in positions:
            group = self.position_body_ids[int(position)]
            keep_fraction = float(rng.uniform(keep_min, keep_max))
            keep_n = max(1, int(round(len(group) * keep_fraction)))
            if keep_n >= len(group):
                chosen = group
            else:
                idx = rng.choice(len(group), size=keep_n, replace=False)
                chosen = tuple(group[int(i)] for i in idx)
            for body_id in chosen:
                if body_id not in seen:
                    seen.add(body_id)
                    out.append(int(body_id))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class ConceptTask:
    name: str
    labels: tuple[str, ...]
    anchors: dict[str, tuple[tuple[int, ...], ...]]
    train_examples: dict[str, tuple[tuple[int, ...], ...]]
    heldout_examples: dict[str, tuple[tuple[int, ...], ...]]
    plastic_fraction: float
    meaning: dict[str, object]

    def sample(self, label: str, rng, *, split: str = "train") -> tuple[int, ...]:
        if split == "anchor":
            bank = self.anchors[label]
        elif split == "heldout":
            bank = self.heldout_examples[label]
        else:
            bank = self.train_examples[label]
        return bank[int(rng.integers(0, len(bank)))]


@dataclass(frozen=True, slots=True)
class ConceptFoundationBundle:
    tasks: tuple[ConceptTask, ...]
    output: object
    field: VirtualPointField
    provenance: dict[str, object]


def _choose_positions(rng, pool: tuple[int, ...], quantity: int) -> tuple[int, ...]:
    if quantity == 0:
        return ()
    idx = rng.choice(len(pool), size=quantity, replace=False)
    return tuple(int(pool[int(i)]) for i in idx)


def _examples_for_quantity(
    field: VirtualPointField,
    *,
    pool: tuple[int, ...],
    quantity: int,
    count: int,
    rng,
    config: ConceptFoundationConfig,
) -> tuple[tuple[int, ...], ...]:
    rows = []
    for _ in range(count):
        positions = _choose_positions(rng, pool, quantity)
        rows.append(
            field.encode(
                positions,
                rng=rng,
                keep_min=config.object_neuron_keep_min,
                keep_max=config.object_neuron_keep_max,
            )
        )
    return tuple(rows)


def _anchors_for_quantity(
    field: VirtualPointField,
    *,
    quantity: int,
    count: int,
    rng,
    config: ConceptFoundationConfig,
) -> tuple[tuple[int, ...], ...]:
    rows = []
    pool = field.train_positions
    for offset in range(count):
        if quantity == 0:
            positions = ()
        else:
            start = (offset * max(1, quantity)) % len(pool)
            positions = tuple(pool[(start + j) % len(pool)] for j in range(quantity))
        rows.append(
            field.encode(
                positions,
                rng=rng,
                keep_min=1.0,
                keep_max=1.0,
            )
        )
    return tuple(rows)


def build_concept_foundation(
    connectome,
    *,
    config: ConceptFoundationConfig,
) -> ConceptFoundationBundle:
    """Build a bottom-up point/object curriculum on real MaleCNS inputs.

    No number symbols, comparison operators, or arithmetic symbols are present.
    The final latent labels A/B/C are arbitrary output channels; they are never
    injected into the brain as input and are not yet bound to 1/2/3 symbols.
    """
    np = __import__("numpy")
    total_positions = config.field_width * config.field_height
    visual_needed = (
        total_positions * config.neurons_per_position + config.background_neurons
    )
    visual_ids = list(_top_visual_ids(connectome, visual_needed))
    if len(visual_ids) < visual_needed:
        raise ValueError(f"need {visual_needed} visual_projection neurons")

    # Break any accidental strength-order-to-grid correlation. Coordinates are a
    # virtual experimental retina, not anatomical receptive-field coordinates.
    rng = np.random.default_rng(config.seed + 71_001)
    rng.shuffle(visual_ids)
    background = tuple(int(x) for x in visual_ids[: config.background_neurons])
    body = visual_ids[config.background_neurons :]
    groups = tuple(
        tuple(int(x) for x in body[i : i + config.neurons_per_position])
        for i in range(0, total_positions * config.neurons_per_position, config.neurons_per_position)
    )

    heldout = []
    train = []
    for position in range(total_positions):
        x = position % config.field_width
        y = position // config.field_width
        # Spatially interleaved holdout cells prevent a trivial train-left/test-right split.
        if (x + 2 * y) % 3 == 0:
            heldout.append(position)
        else:
            train.append(position)
    if len(train) < 6 or len(heldout) < 4:
        raise ValueError("virtual point field split is too small")

    field = VirtualPointField(
        width=config.field_width,
        height=config.field_height,
        neurons_per_position=config.neurons_per_position,
        background_body_ids=background,
        position_body_ids=groups,
        train_positions=tuple(train),
        heldout_positions=tuple(heldout),
    )

    route_input_ids = tuple(
        list(field.background_body_ids)
        + [body_id for group in field.position_body_ids for body_id in group]
    )
    output, output_provenance = choose_route_aware_output_population(
        connectome,
        route_input_ids,
        output_population_size=config.output_population_size,
        max_hops=2,
    )
    output_provenance = {
        **output_provenance,
        "input_superclass": "visual_projection",
        "route_input_scope": "background plus every virtual field position",
        "biological_retinotopy_claimed": False,
    }

    def q_examples(quantity: int, *, split: str, count: int, salt: int):
        local_rng = np.random.default_rng(config.seed + salt)
        pool = field.train_positions if split == "train" else field.heldout_positions
        return _examples_for_quantity(
            field,
            pool=pool,
            quantity=quantity,
            count=count,
            rng=local_rng,
            config=config,
        )

    anchor_rng = np.random.default_rng(config.seed + 72_000)

    presence = ConceptTask(
        name="object_presence",
        labels=("BACKGROUND", "OBJECT_PRESENT"),
        anchors={
            "BACKGROUND": _anchors_for_quantity(field, quantity=0, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
            "OBJECT_PRESENT": _anchors_for_quantity(field, quantity=1, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
        },
        train_examples={
            "BACKGROUND": q_examples(0, split="train", count=config.train_examples_per_label, salt=73_001),
            "OBJECT_PRESENT": q_examples(1, split="train", count=config.train_examples_per_label, salt=73_002),
        },
        heldout_examples={
            "BACKGROUND": q_examples(0, split="heldout", count=config.heldout_examples_per_label, salt=74_001),
            "OBJECT_PRESENT": q_examples(1, split="heldout", count=config.heldout_examples_per_label, salt=74_002),
        },
        plastic_fraction=0.05,
        meaning={
            "goal": "detect an object independent of its virtual position",
            "symbol_binding": False,
            "numeric_concept": False,
        },
    )

    single_multi_train_rng = np.random.default_rng(config.seed + 75_000)
    single_multi_hold_rng = np.random.default_rng(config.seed + 76_000)

    def mixed_multi(pool, count, rng):
        rows = []
        for _ in range(count):
            quantity = 2 if int(rng.integers(0, 2)) == 0 else 3
            rows.extend(
                _examples_for_quantity(
                    field,
                    pool=pool,
                    quantity=quantity,
                    count=1,
                    rng=rng,
                    config=config,
                )
            )
        return tuple(rows)

    one_vs_many = ConceptTask(
        name="single_vs_multiple",
        labels=("SINGLE_OBJECT", "MULTIPLE_OBJECTS"),
        anchors={
            "SINGLE_OBJECT": _anchors_for_quantity(field, quantity=1, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
            "MULTIPLE_OBJECTS": _anchors_for_quantity(field, quantity=2, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
        },
        train_examples={
            "SINGLE_OBJECT": q_examples(1, split="train", count=config.train_examples_per_label, salt=75_101),
            "MULTIPLE_OBJECTS": mixed_multi(field.train_positions, config.train_examples_per_label, single_multi_train_rng),
        },
        heldout_examples={
            "SINGLE_OBJECT": q_examples(1, split="heldout", count=config.heldout_examples_per_label, salt=76_101),
            "MULTIPLE_OBJECTS": mixed_multi(field.heldout_positions, config.heldout_examples_per_label, single_multi_hold_rng),
        },
        plastic_fraction=0.10,
        meaning={
            "goal": "separate one object from more-than-one without number symbols",
            "symbol_binding": False,
            "numeric_concept": "coarse_quantity",
        },
    )

    latent = ConceptTask(
        name="latent_quantity_1_3",
        labels=("LATENT_A", "LATENT_B", "LATENT_C"),
        anchors={
            "LATENT_A": _anchors_for_quantity(field, quantity=1, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
            "LATENT_B": _anchors_for_quantity(field, quantity=2, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
            "LATENT_C": _anchors_for_quantity(field, quantity=3, count=config.anchor_examples_per_label, rng=anchor_rng, config=config),
        },
        train_examples={
            "LATENT_A": q_examples(1, split="train", count=config.train_examples_per_label, salt=77_001),
            "LATENT_B": q_examples(2, split="train", count=config.train_examples_per_label, salt=77_002),
            "LATENT_C": q_examples(3, split="train", count=config.train_examples_per_label, salt=77_003),
        },
        heldout_examples={
            "LATENT_A": q_examples(1, split="heldout", count=config.heldout_examples_per_label, salt=78_001),
            "LATENT_B": q_examples(2, split="heldout", count=config.heldout_examples_per_label, salt=78_002),
            "LATENT_C": q_examples(3, split="heldout", count=config.heldout_examples_per_label, salt=78_003),
        },
        plastic_fraction=0.15,
        meaning={
            "goal": "form three location-invariant latent quantity states before symbol binding",
            "symbol_binding": False,
            "latent_cardinality_for_audit_only": {"LATENT_A": 1, "LATENT_B": 2, "LATENT_C": 3},
        },
    )

    provenance = {
        "input_superclass": "visual_projection",
        "position_mapping": "virtual_grid_over_visual_projection_neurons",
        "biological_retinotopy_claimed": False,
        "field_width": field.width,
        "field_height": field.height,
        "neurons_per_position": field.neurons_per_position,
        "background_neuron_count": len(field.background_body_ids),
        "train_position_count": len(field.train_positions),
        "heldout_position_count": len(field.heldout_positions),
        "train_positions_xy": [field.xy(i) for i in field.train_positions],
        "heldout_positions_xy": [field.xy(i) for i in field.heldout_positions],
        "output": output_provenance,
    }
    return ConceptFoundationBundle(
        tasks=(presence, one_vs_many, latent),
        output=output,
        field=field,
        provenance=provenance,
    )


def _train_anchor_decoder(session, task: ConceptTask, config: ConceptFoundationConfig, seed: int) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(config.decoder_epochs):
        labels = list(task.labels)
        rng.shuffle(labels)
        for label in labels:
            stimulus = task.sample(label, rng, split="anchor")
            observation, result = session.train_decoder_trial(
                stimulus_body_ids=stimulus,
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=config.decoder_rate_hz,
            )
            rows.append(
                {
                    "target": label,
                    "prediction_after": result.prediction_after,
                    "loss": result.loss,
                    "output_spikes": observation.total_output_spikes,
                }
            )
    session.freeze_decoder()
    return {
        "strategy": "anchor_only_then_freeze",
        "anchor_examples_per_label": config.anchor_examples_per_label,
        "steps": len(rows),
        "last_rows": rows[-12:],
    }


def _evaluate_split(
    session,
    task: ConceptTask,
    config: ConceptFoundationConfig,
    *,
    split: str,
    rate_hz: float,
    seed: int,
) -> dict[str, object]:
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    rows = []
    for label in task.labels:
        for _ in range(config.validation_trials_per_label):
            stimulus = task.sample(label, rng, split=split)
            rows.append(
                session.evaluate_trial(
                    stimulus_body_ids=stimulus,
                    target=label,
                    duration_ms=config.duration_ms,
                    stimulus_rate_hz=rate_hz,
                )
            )
    n = len(rows)
    correct = sum(int(bool(x["correct"])) for x in rows)
    silent = sum(int(bool(x.get("silent"))) for x in rows)
    spikes = sum(int(x["total_output_spikes"]) for x in rows)
    return {
        "split": split,
        "accuracy": correct / n,
        "correct": correct,
        "trials": n,
        "silent_fraction": silent / n,
        "mean_output_spikes": spikes / n,
        "chance": 1.0 / len(task.labels),
    }


def _calibrate_rate(session, task, config, seed: int) -> tuple[float, list[dict[str, object]]]:
    rates = (350.0, 300.0, 250.0, 205.0, 180.0, 155.0, 135.0, 115.0)
    rows = []
    for i, rate in enumerate(rates):
        result = _evaluate_split(
            session,
            task,
            config,
            split="train",
            rate_hz=rate,
            seed=seed + i,
        )
        rows.append({"rate_hz": rate, **result})
    viable = [x for x in rows if float(x["silent_fraction"]) <= 0.25]
    pool = viable or rows
    chosen = min(
        pool,
        key=lambda x: abs(float(x["accuracy"]) - 0.65) + 0.5 * float(x["silent_fraction"]),
    )
    return float(chosen["rate_hz"]), rows


def _foundation_gate(stage_reports, retention_history) -> dict[str, object]:
    thresholds = {
        "object_presence": 0.80,
        "single_vs_multiple": 0.70,
        "latent_quantity_1_3": 0.50,
    }
    if not stage_reports or not retention_history:
        return {"passed": False, "reason": "incomplete", "tasks": {}}
    final = retention_history[-1]["tasks"]
    checks = {}
    passed = True
    for row in stage_reports:
        name = row["stage"]
        threshold = thresholds[name]
        heldout = float(row["after_heldout"]["accuracy"])
        train = float(row["after_train"]["accuracy"])
        final_acc = float(final[name]["heldout_accuracy"])
        chance = float(row["after_heldout"]["chance"])
        gap = train - heldout
        learned = heldout >= chance + 0.10
        generalizes = gap <= 0.20
        retained = final_acc >= max(chance + 0.10, threshold * 0.90)
        ok = heldout >= threshold and learned and generalizes and retained
        checks[name] = {
            "heldout_accuracy_after_stage": heldout,
            "train_accuracy_after_stage": train,
            "position_generalization_gap": gap,
            "final_heldout_retention": final_acc,
            "threshold": threshold,
            "learned_above_chance": learned,
            "generalizes_to_unseen_positions": generalizes,
            "retained_after_later_concepts": retained,
            "passed": ok,
        }
        passed &= ok
    return {
        "passed": bool(passed),
        "criterion": "learn concept above chance, transfer to unseen virtual positions, and retain it after later concept stages",
        "tasks": checks,
    }


def run_concept_foundation(
    connectome,
    *,
    config: ConceptFoundationConfig,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    readout_dir: Path = DEFAULT_READOUT_DIR,
    progress_path: Path = DEFAULT_PROGRESS,
) -> dict[str, object]:
    np = __import__("numpy")
    bundle = build_concept_foundation(connectome, config=config)
    tasks = bundle.tasks

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=config.seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=config.seed),
    )

    sessions = {}
    readouts = {}
    for i, task in enumerate(tasks):
        readout, session = _make_memory_session(
            brain,
            bundle.output,
            task,
            config,
            config.seed + 800 + i,
        )
        sessions[task.name] = session
        readouts[task.name] = readout

    replay = AdaptiveReplayScheduler(
        AdaptiveReplayConfig(
            interval=config.replay_interval,
            fraction=1.0,
            rate_hz=205.0,
            max_prior_tasks=3,
            error_power=2.0,
            min_weight=0.03,
            ema_decay=0.85,
        )
    )
    consolidator = MemoryConsolidator(
        ConsolidationConfig(
            top_fraction=config.consolidation_top_fraction,
            boost=config.consolidation_boost,
            min_stage_gain=1e-6,
        )
    )

    stage_reports = []
    retention_history = []
    replay_history = []
    consolidation_history = []

    for stage_index, task in enumerate(tasks):
        session = sessions[task.name]
        readout = readouts[task.name]
        unlock = brain.plasticity.set_plastic_fraction(task.plastic_fraction)

        decoder = _train_anchor_decoder(
            session,
            task,
            config,
            seed=config.seed + 10_000 + stage_index * 100,
        )
        save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)

        rate_hz, calibration = _calibrate_rate(
            session,
            task,
            config,
            seed=config.seed + 20_000 + stage_index * 100,
        )
        before_train = _evaluate_split(
            session,
            task,
            config,
            split="train",
            rate_hz=rate_hz,
            seed=config.seed + 30_000 + stage_index * 100,
        )
        before_heldout = _evaluate_split(
            session,
            task,
            config,
            split="heldout",
            rate_hz=rate_hz,
            seed=config.seed + 31_000 + stage_index * 100,
        )

        if retention_history:
            latest = retention_history[-1]["tasks"]
            for prior_index, prior in enumerate(tasks[:stage_index]):
                replay.set_accuracy(prior_index, float(latest[prior.name]["heldout_accuracy"]))

        stability_before = brain.plasticity.stability.copy()
        correct = 0
        silent = 0
        rows = []
        replay_rows = []
        replay_counts = {prior.name: 0 for prior in tasks[:stage_index]}

        for trial in range(config.stage_trials):
            trial_rng = np.random.default_rng(config.seed + stage_index * 1_000_000 + trial)
            label = task.labels[int(trial_rng.integers(0, len(task.labels)))]
            stimulus = task.sample(label, trial_rng, split="train")
            result = session.train_brain_trial(
                stimulus_body_ids=stimulus,
                target=label,
                duration_ms=config.duration_ms,
                stimulus_rate_hz=rate_hz,
            )
            correct += int(result.correct)
            silent += int(result.total_output_spikes == 0)
            rows.append(
                {
                    "trial": trial + 1,
                    "target": result.target,
                    "prediction": result.prediction,
                    "correct": bool(result.correct),
                    "reward": result.reward,
                    "output_spikes": result.total_output_spikes,
                    "edge_updates": result.learning["learning"]["edge_updates"],
                }
            )

            if replay.should_replay(trial + 1, stage_index):
                prior_index = replay.choose_prior_index(trial_rng, stage_index)
                prior = tasks[prior_index]
                prior_session = sessions[prior.name]
                prior_label = prior.labels[int(trial_rng.integers(0, len(prior.labels)))]
                prior_stimulus = prior.sample(prior_label, trial_rng, split="train")
                rr = prior_session.train_brain_trial(
                    stimulus_body_ids=prior_stimulus,
                    target=prior_label,
                    duration_ms=config.duration_ms,
                    stimulus_rate_hz=205.0,
                )
                replay.update(prior_index, correct=bool(rr.correct))
                replay_counts[prior.name] += 1
                replay_rows.append(
                    {
                        "at_current_trial": trial + 1,
                        "task": prior.name,
                        "correct": bool(rr.correct),
                        "post_accuracy_ema": replay.accuracy(prior_index),
                    }
                )

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
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(
                    json.dumps(
                        {
                            "experiment": "malecns_concept_foundation_progress",
                            "stage": task.name,
                            "completed_trials": trial + 1,
                            "training_accuracy": correct / max(1, len(rows)),
                            "replay_count": len(replay_rows),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"[concept {task.name}] {trial+1}/{config.stage_trials} "
                    f"acc={correct/max(1,len(rows)):.3f} replay={len(replay_rows)}"
                )

        consolidation = consolidator.consolidate(
            brain.plasticity,
            baseline_stability=stability_before,
        )
        consolidation_history.append({"after_stage": task.name, **consolidation})

        after_train = _evaluate_split(
            session,
            task,
            config,
            split="train",
            rate_hz=rate_hz,
            seed=config.seed + 40_000 + stage_index * 100,
        )
        after_heldout = _evaluate_split(
            session,
            task,
            config,
            split="heldout",
            rate_hz=rate_hz,
            seed=config.seed + 41_000 + stage_index * 100,
        )

        save_learning_checkpoint(
            checkpoint_path,
            brain=brain,
            readout=readout,
            config=config,
            completed_trials=config.stage_trials,
            stage=task.name,
        )
        save_readout_checkpoint(_readout_path(readout_dir, task.name), readout)

        replay_report = {
            "stage": task.name,
            "count": len(replay_rows),
            "per_task_count": replay_counts,
            "policy": replay.snapshot(stage_index),
            "last_rows": replay_rows[-12:],
        }
        replay_history.append(replay_report)

        stage_reports.append(
            {
                "stage": task.name,
                "labels": list(task.labels),
                "meaning": task.meaning,
                "plasticity_unlock": unlock,
                "decoder": decoder,
                "selected_rate_hz": rate_hz,
                "calibration": calibration,
                "before_train": before_train,
                "before_heldout": before_heldout,
                "after_train": after_train,
                "after_heldout": after_heldout,
                "training_accuracy": correct / max(1, len(rows)),
                "training_silent_fraction": silent / max(1, len(rows)),
                "replay": replay_report,
                "consolidation": consolidation,
                "last_rows": rows[-16:],
            }
        )

        retention = {"after_stage": task.name, "tasks": {}}
        for prior_index, prior in enumerate(tasks[: stage_index + 1]):
            prior_session = sessions[prior.name]
            prior_result = _evaluate_split(
                prior_session,
                prior,
                config,
                split="heldout",
                rate_hz=205.0,
                seed=config.seed + 50_000 + stage_index * 100 + prior_index,
            )
            retention["tasks"][prior.name] = {
                "heldout_accuracy": prior_result["accuracy"],
                "silent_fraction": prior_result["silent_fraction"],
                "mean_output_spikes": prior_result["mean_output_spikes"],
            }
            replay.set_accuracy(prior_index, float(prior_result["accuracy"]))
        retention_history.append(retention)

    save_learning_checkpoint(
        checkpoint_path,
        brain=brain,
        config=config,
        completed_trials=0,
        stage="complete",
    )

    plast = brain.plasticity.summary()
    changed = int(np.count_nonzero(np.abs(brain.plasticity.multiplier - 1.0) > 1e-7))
    report = {
        "experiment": "malecns_concept_foundation_v1",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "provenance": bundle.provenance,
        "curriculum_order": [task.name for task in tasks],
        "stage_reports": stage_reports,
        "retention_history": retention_history,
        "replay_history": replay_history,
        "consolidation_history": consolidation_history,
        "foundation_gate": _foundation_gate(stage_reports, retention_history),
        "final_plasticity": {**plast, "changed_edges": changed},
        "checkpoint": str(checkpoint_path),
        "readout_dir": str(readout_dir),
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_html(report: dict[str, object]) -> str:
    gate = report["foundation_gate"]
    rows = []
    for stage in report["stage_reports"]:
        train = 100.0 * float(stage["after_train"]["accuracy"])
        held = 100.0 * float(stage["after_heldout"]["accuracy"])
        before = 100.0 * float(stage["before_heldout"]["accuracy"])
        rows.append(
            f"<tr><td>{stage['stage']}</td><td>{before:.1f}%</td>"
            f"<td>{train:.1f}%</td><td>{held:.1f}%</td><td>{train-held:+.1f} pp</td></tr>"
        )
    status = "PASS" if gate["passed"] else "FAIL"
    cls = "pass" if gate["passed"] else "fail"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>DrosoMath Concept Foundation</title>
<style>body{{font-family:system-ui;background:#101318;color:#e8edf5;max-width:1100px;margin:auto;padding:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #2b3442;text-align:right}}th:first-child,td:first-child{{text-align:left}}.card{{padding:16px;border:1px solid #2b3442;border-radius:12px;background:#171c24}}.pass{{color:#6ee7a8}}.fail{{color:#ff8a8a}}</style></head><body>
<h1>DrosoMath Concept Foundation</h1>
<div class='card'>Gate: <b class='{cls}'>{status}</b><br>점/물체 → 위치 불변성 → 1개 vs 여러 개 → 숫자기호 없는 잠재 수량 상태 A/B/C</div>
<h2>Concept transfer</h2><table><tr><th>Stage</th><th>Held-out before</th><th>Train after</th><th>Unseen-position after</th><th>Position gap</th></tr>{''.join(rows)}</table>
<p>Important: grid positions are a virtual experimental mapping over real visual_projection neurons, not claimed biological retinotopic coordinates.</p>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Run concept-first MaleCNS foundation curriculum")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--download", action="store_true")
    p.add_argument("--min-syn", type=int, default=5)
    p.add_argument("--stage-trials", type=int, default=512)
    p.add_argument("--validation-trials", type=int, default=32)
    p.add_argument("--decoder-epochs", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    p.add_argument("--html", type=Path, default=Path("results/latest_malecns_concept_foundation.html"))
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
        seed=a.seed,
    )
    report = run_concept_foundation(
        connectome,
        config=config,
        checkpoint_path=a.checkpoint,
        readout_dir=a.readout_dir,
        progress_path=a.progress,
    )
    a.result.parent.mkdir(parents=True, exist_ok=True)
    a.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    a.html.parent.mkdir(parents=True, exist_ok=True)
    a.html.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({
        "foundation_gate": report["foundation_gate"],
        "curriculum_order": report["curriculum_order"],
        "final_plasticity": report["final_plasticity"],
    }, indent=2, sort_keys=True))
    print(f"saved result: {a.result}")
    print(f"saved dashboard: {a.html}")


if __name__ == "__main__":
    main()
