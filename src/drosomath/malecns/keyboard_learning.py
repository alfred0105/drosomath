"""Closed-loop MaleCNS learning task for matching tokens to virtual keys."""

from __future__ import annotations

import argparse
import html
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
DEFAULT_HTML = Path("results/latest_malecns_keyboard_matching.html")


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


def build_label_schedule(trials: int, *, seed: int) -> tuple[str, ...]:
    """Shuffle every cycle while keeping key exposure balanced."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    np = __import__("numpy")
    rng = np.random.default_rng(seed)
    full_cycles, remainder = divmod(int(trials), len(KEY_LABELS))
    schedule = []
    for _ in range(full_cycles):
        schedule.extend(str(x) for x in rng.permutation(KEY_LABELS))
    if remainder:
        schedule.extend(str(x) for x in rng.permutation(KEY_LABELS)[:remainder])
    return tuple(schedule)


def build_keyboard_html(payload: dict[str, object]) -> str:
    """Render a self-refreshing progress dashboard for the keyboard task."""
    config = payload.get("config", {})
    completed = int(payload.get("completed_trials", 0))
    trials = int(config.get("trials", 0)) if isinstance(config, dict) else 0
    accuracy = float(payload.get("accuracy", 0.0))
    labels = payload.get("token_labels", list(KEY_LABELS))
    recent = payload.get("recent_trials", [])
    per_key = payload.get("per_key", {})
    latest = recent[-1] if recent else {}
    latest_label = str(latest.get("label", "")) if isinstance(latest, dict) else ""
    key_cards = []
    for label in labels:
        stats = per_key.get(str(label), {}) if isinstance(per_key, dict) else {}
        key_accuracy = float(stats.get("accuracy", 0.0)) if isinstance(stats, dict) else 0.0
        mastered = bool(stats.get("mastered", False)) if isinstance(stats, dict) else False
        active = " active" if str(label) == latest_label else ""
        state = " mastered" if mastered else " learning"
        key_cards.append(
            f"<div class='key{active}{state}'>"
            f"<b>{html.escape(str(label))}</b><small>{key_accuracy:.0%}</small></div>"
        )
    rows = []
    for row in reversed(recent):
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{row.get('trial', '')}</td>"
            f"<td>{html.escape(str(row.get('label', '')))}</td>"
            f"<td>{html.escape(str(row.get('clicked_label') or '-'))}</td>"
            f"<td>{'PASS' if row.get('correct') else 'MISS'}</td>"
            f"<td>{float(row.get('reward', 0.0)):+.2f}</td>"
            f"<td>{int(row.get('windows', 0))}</td>"
            "</tr>"
        )
    progress = 100.0 * completed / max(1, trials)
    mastered_count = sum(
        int(isinstance(stats, dict) and stats.get("mastered", False))
        for stats in per_key.values()
    ) if isinstance(per_key, dict) else 0
    target_accuracy = float(payload.get("target_accuracy", 0.80))
    return f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta http-equiv='refresh' content='5'>
<title>DrosoMath Virtual Keyboard Matching</title>
<style>
body{{font-family:system-ui,sans-serif;background:#101318;color:#e8edf5;max-width:1100px;margin:auto;padding:24px}}
.grid{{display:grid;grid-template-columns:repeat(12,minmax(42px,1fr));gap:6px;margin:18px 0}}
.key{{padding:10px 4px;text-align:center;border:1px solid #344052;border-radius:7px;background:#1b2230;min-height:20px}}
.key small{{display:block;color:#aebbd0;margin-top:4px}}
.key.active{{background:#285b8f;border-color:#77b7ff}}
.key.mastered{{border-color:#4ecb8a;background:#163629}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.card{{background:#171d27;border:1px solid #2b3442;border-radius:9px;padding:12px}}
.value{{font-size:1.35rem;font-weight:700}}
progress{{width:100%;height:18px}}
table{{width:100%;border-collapse:collapse;margin-top:14px}}th,td{{padding:7px;border-bottom:1px solid #2b3442;text-align:right}}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
</style></head><body>
<h1>DrosoMath 가상 키보드 매칭</h1>
<div class='cards'>
<div class='card'>Trial<div class='value'>{completed}/{trials}</div></div>
<div class='card'>진행률<div class='value'>{progress:.1f}%</div></div>
<div class='card'>정확도<div class='value'>{accuracy:.3f}</div></div>
<div class='card'>숙련 키<div class='value'>{mastered_count}/{len(labels)}</div></div>
</div>
<p><progress value='{progress:.3f}' max='100'></progress></p>
<h2>가상 키보드 · 목표 {target_accuracy:.0%}</h2><div class='grid'>{''.join(key_cards)}</div>
<h2>최근 trial</h2><table><tr><th>Trial</th><th>입력 키</th><th>클릭</th><th>결과</th><th>보상</th><th>Window</th></tr>{''.join(rows)}</table>
<p>페이지는 5초마다 자동 갱신된다. Backend: numpy_cpu / closed-loop virtual body.</p>
</body></html>"""


@dataclass(frozen=True, slots=True)
class KeyboardTrainingConfig:
    min_connection_synapses: int = 5
    token_neurons: int = 6
    background_neurons: int = 6
    motor_population_size: int = 32
    # Safety cap. Training stops early only after every key reaches target
    # accuracy with the minimum number of observations.
    trials: int = 20_000
    min_trials_per_key: int = 32
    target_accuracy: float = 0.80
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
        if self.min_trials_per_key < 1:
            raise ValueError("min_trials_per_key must be >= 1")
        if not 0.5 < self.target_accuracy <= 1.0:
            raise ValueError("target_accuracy must be in (0.5, 1]")
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
        peak_motor_rate_hz = 0.0
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
            peak_motor_rate_hz = max(peak_motor_rate_hz, float(rates.max()))
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
            "peak_motor_rate_hz": peak_motor_rate_hz,
            "click_threshold_hz": self.task.motor.config.click_threshold_hz,
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
    html_path: Path = DEFAULT_HTML,
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
    np = session.np
    rng = np.random.default_rng(config.seed + 83_001)
    per_key_trials = {label: 0 for label in KEY_LABELS}
    per_key_correct = {label: 0 for label in KEY_LABELS}
    mastery_reached = False
    completed = 0

    while completed < config.trials:
        # First guarantee enough observations for every key. Afterwards focus
        # random sampling on keys still below the target accuracy.
        under_minimum = [
            label for label in KEY_LABELS
            if per_key_trials[label] < config.min_trials_per_key
        ]
        if under_minimum:
            candidates = under_minimum
        else:
            candidates = [
                label for label in KEY_LABELS
                if per_key_correct[label] / max(1, per_key_trials[label]) < config.target_accuracy
            ]
            if not candidates:
                mastery_reached = True
                break
        label = str(candidates[int(rng.integers(0, len(candidates)))])
        completed += 1
        trial = completed
        row = session.run_trial(label)
        row["trial"] = trial
        rows.append(row)
        correct += int(row["correct"])
        per_key_trials[label] += 1
        per_key_correct[label] += int(row["correct"])
        per_key = {
            key: {
                "trials": per_key_trials[key],
                "correct": per_key_correct[key],
                "accuracy": per_key_correct[key] / max(1, per_key_trials[key]),
                "mastered": (
                    per_key_trials[key] >= config.min_trials_per_key
                    and per_key_correct[key] / max(1, per_key_trials[key]) >= config.target_accuracy
                ),
            }
            for key in KEY_LABELS
        }
        mastered_count = sum(int(stats["mastered"]) for stats in per_key.values())
        mastery_reached = mastered_count == len(KEY_LABELS)
        update_stats = row["learning"].get("learning", {})
        _live_status(
            f"keyboard_matching trial={trial:>4}/{config.trials:<4} "
            f"key={label:<2} clicked={str(row['clicked_label'] or '-'): <2} "
            f"correct={int(row['correct'])} reward={row['reward']:+.2f} "
            f"windows={row['windows']:>2} updates={int(update_stats.get('edge_updates', 0)):>6} "
            f"peak={row['peak_motor_rate_hz']:>5.1f}Hz "
            f"accuracy={correct / trial:.3f} mastered={mastered_count:>2}/{len(KEY_LABELS)} "
            f"elapsed={time.perf_counter() - started:6.1f}s"
        )
        payload = {
            "experiment": "malecns_virtual_keyboard_matching_v1",
            "config": asdict(config),
            "completed_trials": trial,
            "accuracy": correct / trial,
            "target_accuracy": config.target_accuracy,
            "min_trials_per_key": config.min_trials_per_key,
            "mastery_reached": mastery_reached,
            "per_key": per_key,
            "route_provenance": session.route_provenance,
            "motor_channel_count": session.motor_channel_count,
            "recent_trials": rows[-16:],
        }
        if trial % config.checkpoint_every == 0 or trial == config.trials or mastery_reached:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            save_learning_checkpoint(
                checkpoint_path,
                brain=session.brain,
                config=config,
                completed_trials=trial,
                stage="keyboard_matching",
            )
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(build_keyboard_html(payload), encoding="utf-8")
            if sys.stdout.isatty():
                _finish_live_status()
                print(
                    f"checkpoint trial={trial} accuracy={correct / trial:.3f} "
                    f"path={checkpoint_path}",
                    flush=True,
                )
        if mastery_reached:
            _finish_live_status()
            print(
                f"keyboard mastery reached: all {len(KEY_LABELS)} keys >= "
                f"{config.target_accuracy:.0%} after {trial} trials",
                flush=True,
            )
            break

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
        "completed_trials": completed,
        "accuracy": correct / max(1, completed),
        "target_accuracy": config.target_accuracy,
        "min_trials_per_key": config.min_trials_per_key,
        "mastery_reached": mastery_reached,
        "stopped_reason": "all_keys_mastered" if mastery_reached else "max_trials_reached",
        "per_key": per_key if completed else {},
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
        "html": str(html_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_keyboard_html({
        **report,
        "recent_trials": rows[-16:],
    }), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaleCNS to match tokens to a virtual keyboard.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--min-trials-per-key", type=int, default=32)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
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
        min_trials_per_key=args.min_trials_per_key,
        target_accuracy=args.target_accuracy,
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
