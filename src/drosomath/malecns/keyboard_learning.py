"""Closed-loop MaleCNS learning task for matching tokens to virtual keys."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import OutgoingBudgetNormalizer, PlasticStateConfig, UsageRewardRule

from .brain import PlasticMaleCNSBrain
from .checkpoint import save_learning_checkpoint
from .download import DEFAULT_DATA_DIR, download_malecns
from .first_training import _top_by_outgoing, choose_route_aware_output_population
from .loader import load_malecns_v1
from .virtual_keyboard import KEY_LABELS, KeyboardMatchingTask


DEFAULT_RESULT = Path("results/latest_malecns_keyboard_matching.json")
DEFAULT_PROGRESS = Path("results/malecns_keyboard_matching_progress.json")
DEFAULT_CHECKPOINT = Path("checkpoints/malecns_keyboard_matching_brain.npz")


def _live_status(message: str) -> None:
    """Show progress in-place in a terminal, or line-buffered otherwise."""
    text = str(message)
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[2K" + text)
        sys.stdout.flush()
    else:
        print(text, flush=True)


def _finish_live_status() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[2K\n")
        sys.stdout.flush()


@dataclass(frozen=True, slots=True)
class KeyboardTrainingConfig:
    min_connection_synapses: int = 5
    token_neurons: int = 6
    background_neurons: int = 6
    motor_population_size: int = 32
    # Sixty physical keys need repeated visits; the default gives every key
    # 32 trials (a balanced first keyboard curriculum).
    trials: int = 1920
    duration_ms: float = 100.0
    control_window_ms: float = 20.0
    max_control_windows: int = 30
    stimulus_rate_hz: float = 205.0
    plastic_fraction: float = 0.05
    learning_rate: float = 0.02
    budget_strength: float = 0.25
    checkpoint_every: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        if self.min_connection_synapses < 1:
            raise ValueError("min_connection_synapses must be >= 1")
        if self.token_neurons < 2 or self.background_neurons < 1:
            raise ValueError("input groups must be non-empty")
        if self.motor_population_size < 2:
            raise ValueError("motor_population_size must be >= 2")
        if self.trials < 1 or self.checkpoint_every < 1:
            raise ValueError("trials and checkpoint_every must be positive")
        if self.duration_ms <= 0.0 or self.control_window_ms <= 0.0:
            raise ValueError("durations must be > 0")
        if self.max_control_windows < 1:
            raise ValueError("max_control_windows must be >= 1")


@dataclass(frozen=True, slots=True)
class TokenVisualEncoder:
    background_body_ids: tuple[int, ...]
    token_body_ids: dict[str, tuple[int, ...]]

    def encode(self, label: str) -> tuple[int, ...]:
        try:
            token = self.token_body_ids[str(label)]
        except KeyError as exc:
            raise KeyError(f"unknown keyboard token: {label!r}") from exc
        return self.background_body_ids + token


def build_token_encoder(connectome, *, config: KeyboardTrainingConfig) -> TokenVisualEncoder:
    """Assign deterministic visual_projection groups to the physical key set."""
    np = __import__("numpy")
    superclass = np.asarray(connectome.metadata.get("superclass"), dtype=object)
    visual = np.flatnonzero(superclass == "visual_projection").astype(np.int32)
    needed = config.background_neurons + len(KEY_LABELS) * config.token_neurons
    if len(visual) < needed:
        raise ValueError(f"need {needed} visual_projection neurons, found {len(visual)}")
    selected = _top_by_outgoing(connectome, visual, needed)
    rng = np.random.default_rng(config.seed + 81_001)
    selected = np.asarray(selected, dtype=np.int32).copy()
    rng.shuffle(selected)
    background = tuple(int(connectome.body_ids[i]) for i in selected[: config.background_neurons])
    cursor = config.background_neurons
    groups: dict[str, tuple[int, ...]] = {}
    for label in KEY_LABELS:
        group = selected[cursor : cursor + config.token_neurons]
        groups[label] = tuple(int(connectome.body_ids[i]) for i in group)
        cursor += config.token_neurons
    return TokenVisualEncoder(background_body_ids=background, token_body_ids=groups)


class KeyboardNeuralSession:
    """Run neural prompt trials and close the loop through four virtual arms."""

    motor_channel_count = 20

    def __init__(self, connectome, *, config: KeyboardTrainingConfig) -> None:
        np = __import__("numpy")
        self.np = np
        self.config = config
        self.encoder = build_token_encoder(connectome, config=config)
        route_inputs = self.encoder.background_body_ids + tuple(
            body_id
            for group in self.encoder.token_body_ids.values()
            for body_id in group
        )
        self.output, self.route_provenance = choose_route_aware_output_population(
            connectome,
            route_inputs,
            output_population_size=self.motor_channel_count * config.motor_population_size,
            max_hops=2,
        )
        self.channel_groups = tuple(
            np.asarray(self.output.indices[i : i + config.motor_population_size], dtype=np.int32)
            for i in range(0, len(self.output.indices), config.motor_population_size)
        )
        if len(self.channel_groups) != self.motor_channel_count:
            raise ValueError("route-aware output did not produce 20 motor populations")
        self.channel_lookup = np.full(connectome.neuron_count, -1, dtype=np.int16)
        for channel, group in enumerate(self.channel_groups):
            self.channel_lookup[group] = channel
        self.brain = PlasticMaleCNSBrain(
            connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=config.seed,
            plasticity_config=PlasticStateConfig(
                plastic_fraction=config.plastic_fraction,
                seed=config.seed,
            ),
        )
        self.task = KeyboardMatchingTask(seed=config.seed)
        self.reward_rule = UsageRewardRule(learning_rate=config.learning_rate)
        self.normalizer = OutgoingBudgetNormalizer(strength=config.budget_strength)
        self.steps_per_window = max(
            1,
            int(round(config.control_window_ms / self.brain.params.dt_ms)),
        )
        self.max_control_windows = min(
            config.max_control_windows,
            max(1, int(math.ceil(config.duration_ms / config.control_window_ms))),
        )
        self.rng = np.random.default_rng(config.seed + 82_001)

    def run_trial(self, label: str) -> dict[str, object]:
        np = self.np
        self.brain.reset()
        self.task.reset(label)
        stimulus_indices = self.brain.indices_for_ids(self.encoder.encode(label))
        last = None
        windows = 0
        started = time.perf_counter()
        for windows in range(1, self.max_control_windows + 1):
            counts = np.zeros(self.motor_channel_count, dtype=np.int32)
            for _ in range(self.steps_per_window):
                fired, _ = self.brain.step(
                    stimulus_indices=stimulus_indices,
                    stimulus_rate_hz=self.config.stimulus_rate_hz,
                )
                if len(fired):
                    local = self.channel_lookup[fired]
                    local = local[local >= 0]
                    if len(local):
                        np.add.at(counts, local, 1)
            rates = counts.astype(np.float32) * (
                1000.0 / (self.config.control_window_ms * self.config.motor_population_size)
            )
            last = self.task.step_from_rates(rates)
            if last.done:
                break

        correct = bool(last is not None and last.clicked_label == label)
        reward = 1.0 if correct else -0.5
        learning = self.brain.learn_from_reward(
            reward=reward,
            rule=self.reward_rule,
            normalizer=self.normalizer,
        )
        return {
            "label": label,
            "correct": correct,
            "clicked_label": last.clicked_label if last is not None else None,
            "clicked_arm": last.clicked_arm if last is not None else None,
            "reward": reward,
            "windows": windows,
            "elapsed_seconds": time.perf_counter() - started,
            "learning": learning,
        }


def run_keyboard_training(
    connectome,
    *,
    config: KeyboardTrainingConfig,
    result_path: Path = DEFAULT_RESULT,
    progress_path: Path = DEFAULT_PROGRESS,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, object]:
    _live_status("keyboard matching: building token encoder and motor populations...")
    session = KeyboardNeuralSession(connectome, config=config)
    _finish_live_status()
    print(
        "keyboard matching: ready "
        f"tokens={len(KEY_LABELS)} motor_channels={session.motor_channel_count} "
        f"route_outputs={len(session.output.body_ids)}",
        flush=True,
    )
    rows = []
    correct = 0
    started = time.perf_counter()
    for trial in range(1, config.trials + 1):
        label = KEY_LABELS[(trial - 1) % len(KEY_LABELS)]
        row = session.run_trial(label)
        row["trial"] = trial
        rows.append(row)
        correct += int(row["correct"])
        update_stats = row["learning"].get("learning", {})
        _live_status(
            f"keyboard_matching trial={trial:>4}/{config.trials:<4} "
            f"key={label:<2} clicked={str(row['clicked_label'] or '-'): <2} "
            f"correct={int(row['correct'])} reward={row['reward']:+.2f} "
            f"windows={row['windows']:>2} updates={int(update_stats.get('edge_updates', 0)):>6} "
            f"accuracy={correct / trial:.3f} "
            f"elapsed={time.perf_counter() - started:6.1f}s"
        )
        payload = {
            "experiment": "malecns_virtual_keyboard_matching_v1",
            "config": asdict(config),
            "completed_trials": trial,
            "accuracy": correct / trial,
            "route_provenance": session.route_provenance,
            "motor_channel_count": session.motor_channel_count,
            "recent_trials": rows[-16:],
        }
        if trial % config.checkpoint_every == 0 or trial == config.trials:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            save_learning_checkpoint(
                checkpoint_path,
                brain=session.brain,
                config=config,
                completed_trials=trial,
                stage="keyboard_matching",
            )
            if sys.stdout.isatty():
                _finish_live_status()
                print(
                    f"checkpoint trial={trial} accuracy={correct / trial:.3f} "
                    f"path={checkpoint_path}",
                    flush=True,
                )

    _finish_live_status()

    np = session.np
    report = {
        "experiment": "malecns_virtual_keyboard_matching_v1",
        "purpose": "match physical Korean, English, O/X, and digit keys to virtual clicks with four-arm motor output",
        "config": asdict(config),
        "connectome": connectome.summary(),
        "token_labels": list(KEY_LABELS),
        "token_encoder": {
            "type": "virtual_visual_projection_groups",
            "background_count": len(session.encoder.background_body_ids),
            "token_group_size": config.token_neurons,
        },
        "route_provenance": session.route_provenance,
        "motor_channel_count": session.motor_channel_count,
        "completed_trials": config.trials,
        "accuracy": correct / max(1, config.trials),
        "final_plasticity": {
            **session.brain.plasticity.summary(),
            "changed_edges": int(np.count_nonzero(np.abs(session.brain.plasticity.multiplier - 1.0) > 1e-7)),
        },
        "execution_profile": {
            "backend": "numpy_cpu",
            "gpu_used": False,
            "external_decoder": False,
            "closed_loop_virtual_body": True,
        },
        "checkpoint": str(checkpoint_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaleCNS to match tokens to a virtual keyboard.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1920)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--control-window-ms", type=float, default=20.0)
    parser.add_argument("--max-control-windows", type=int, default=30)
    parser.add_argument("--stimulus-rate-hz", type=float, default=205.0)
    parser.add_argument("--motor-population-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.download:
        download_malecns(args.data_dir)
    config = KeyboardTrainingConfig(
        min_connection_synapses=args.min_syn,
        trials=args.trials,
        duration_ms=args.duration_ms,
        control_window_ms=args.control_window_ms,
        max_control_windows=args.max_control_windows,
        stimulus_rate_hz=args.stimulus_rate_hz,
        motor_population_size=args.motor_population_size,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(
        f"keyboard matching: loading MaleCNS data from {args.data_dir} "
        f"(min_syn={args.min_syn})...",
        flush=True,
    )
    connectome = load_malecns_v1(args.data_dir, min_connection_synapses=args.min_syn)
    print(
        f"keyboard matching: loaded neurons={connectome.neuron_count} "
        f"edges={connectome.edge_count}; backend=numpy_cpu; gpu_used=False",
        flush=True,
    )
    report = run_keyboard_training(
        connectome,
        config=config,
    )
    print(json.dumps({
        "completed_trials": report["completed_trials"],
        "accuracy": report["accuracy"],
        "route_provenance": report["route_provenance"],
        "changed_edges": report["final_plasticity"]["changed_edges"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
