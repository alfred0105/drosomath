"""Closed-loop MaleCNS learning task for matching tokens to virtual keys."""

from __future__ import annotations

import argparse
from collections import deque
import html
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain import (
    ChannelHomeostasis,
    DirectionalModulationConfig,
    OutgoingBudgetNormalizer,
    PlasticStateConfig,
    PlasticityBudgetAdaptation,
    PlasticityBudgetAdaptationConfig,
    PlasticityController,
    UsageRewardRule,
)

from .brain import PlasticMaleCNSBrain
from .checkpoint import restore_learning_checkpoint, save_learning_checkpoint
from .download import DEFAULT_DATA_DIR, download_malecns
from .first_training import _top_by_outgoing, choose_route_aware_output_population
from .loader import load_malecns_v1
from .virtual_body import ArmAction, OneArmWorld
from .virtual_keyboard import CircularKeyboard, FanKeyboard, KEY_LABELS, KeyboardMatchingTask
from .virtual_motor import FourArmMotorConfig, OneArmMotorAdapter


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


def build_temporal_window_record(
    *,
    window_index: int,
    total_spikes: int,
    fired_neurons,
    recent_presynaptic_count: int,
    click_output_spikes: int,
    click_output_neurons,
    click_rate_hz: float,
    motor_channel_rates,
    causal_summary: dict[str, object],
    active_neuron_jaccard: float | None,
    click_causal_jaccard: float | None,
) -> dict[str, object]:
    """Build compact, read-only telemetry for one existing control window."""
    return {
        "window_index": int(window_index),
        "total_spikes": int(total_spikes),
        "unique_fired_neurons": int(len(fired_neurons)),
        "recent_presynaptic_count": int(recent_presynaptic_count),
        "click_output_spikes": int(click_output_spikes),
        "active_click_output_neurons": int(len(click_output_neurons)),
        "click_rate_hz": float(click_rate_hz),
        "motor_channel_rates": [float(value) for value in motor_channel_rates],
        **causal_summary,
        "active_neuron_jaccard": active_neuron_jaccard,
        "click_causal_jaccard": click_causal_jaccard,
    }


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


def recent_key_mastered(history: list[bool], window_size: int) -> bool:
    """Return whether one key was correct on each of its recent attempts."""
    if len(history) < window_size:
        return False
    return all(history[-window_size:])


def calculate_success_margin_deficit(
    *,
    peak_click_rate_hz: float,
    threshold_hz: float,
    target_hz: float,
) -> float:
    """Return bounded output-margin deficit for a successful click.

    The behavioral threshold remains the success gate.  This value only
    describes how far a successful output is from the optional robust target;
    outputs at or above the target receive no additional directional signal.
    """
    denominator = max(float(target_hz) - float(threshold_hz), 1e-9)
    return float(
        min(
            1.0,
            max(0.0, (float(target_hz) - float(peak_click_rate_hz)) / denominator),
        )
    )


def _movement_summary(points: list[tuple[float, float]]) -> dict[str, object]:
    """Summarize one endpoint trajectory without treating it as a large region."""
    if not points:
        points = [(0.0, 0.0)]
    last_x, last_y = points[-1]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return {
        "endpoint": [float(last_x), float(last_y)],
        "center": [float(center_x), float(center_y)],
        "min": [float(min_x), float(min_y)],
        "max": [float(max_x), float(max_y)],
        "trajectory_span": float(math.hypot(max_x - min_x, max_y - min_y)),
        "samples": len(points),
    }


def build_evaluation_schedule(
    labels: tuple[str, ...],
    window_size: int,
    rng,
) -> tuple[str, ...]:
    """Create one randomized, non-training exam: every key appears N times."""
    np = __import__("numpy")
    label_indices = np.repeat(np.arange(len(labels), dtype=np.int32), window_size)
    return tuple(str(labels[int(index)]) for index in rng.permutation(label_indices))


def build_balanced_coverage_cycle(labels, rng) -> tuple[str, ...]:
    """Return one shuffled presentation of every key for starvation-free training."""
    cycle = list(labels)
    rng.shuffle(cycle)
    return tuple(cycle)


def summarize_per_key_stats(
    labels: tuple[str, ...],
    trials: dict[str, int],
    correct: dict[str, int],
    recent: dict[str, list[bool]],
    *,
    window_size: int,
    click_only: bool,
    min_trials: int,
    target_accuracy: float,
) -> dict[str, dict[str, object]]:
    """Build the training-only per-key telemetry shown by the dashboard."""
    result = {}
    for key in labels:
        history = recent[key]
        recent_correct = sum(history)
        overall_accuracy = correct[key] / max(1, trials[key])
        result[key] = {
            "trials": trials[key],
            "correct": correct[key],
            "accuracy": overall_accuracy,
            "recent_attempts": len(history),
            "recent_correct": recent_correct,
            "recent_accuracy": recent_correct / max(1, len(history)),
            "recent_clicks": list(history),
            "mastered": (
                recent_key_mastered(history, window_size)
                if click_only
                else (
                    trials[key] >= min_trials
                    and overall_accuracy >= target_accuracy
                )
            ),
        }
    return result


def summarize_recent_outcome_transitions(
    per_key_recent: dict[str, list[bool]],
) -> dict[str, object]:
    """Summarize recent success/failure flips without using task labels for learning."""
    per_key: dict[str, dict[str, float | int]] = {}
    for label, history in per_key_recent.items():
        transitions = sum(
            int(previous != current)
            for previous, current in zip(history, history[1:])
        )
        rate = transitions / max(1, len(history) - 1) if len(history) > 1 else 0.0
        per_key[str(label)] = {
            "attempts": int(len(history)),
            "transition_count": int(transitions),
            "transition_rate": float(rate),
            "recent_accuracy": float(sum(history) / max(1, len(history))),
        }
    ranked = sorted(
        per_key.items(),
        key=lambda item: (float(item[1]["recent_accuracy"]), item[0]),
    )
    weak = [item for _, item in ranked[:10]]
    return {
        "per_key": per_key,
        "mean_recent_outcome_transition_rate": float(
            sum(float(item["transition_rate"]) for item in per_key.values())
            / max(1, len(per_key))
        ),
        "weak_key_labels": [label for label, _ in ranked[:10]],
        "weak_key_outcome_transition_rate": float(
            sum(float(item["transition_rate"]) for item in weak) / max(1, len(weak))
        ),
    }


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
    layout = payload.get("keyboard_layout", [])
    layout = layout if isinstance(layout, list) else []
    latest_body = latest.get("body", {}) if isinstance(latest, dict) else {}
    if not isinstance(latest_body, dict) or not latest_body:
        latest_body = payload.get("body", {})
    latest_body = latest_body if isinstance(latest_body, dict) else {}
    latest_movement = latest.get("movement", {}) if isinstance(latest, dict) else {}
    latest_movement = latest_movement if isinstance(latest_movement, dict) else {}
    latest_target_distance = float(latest.get("min_target_distance", 0.0)) if isinstance(latest, dict) else 0.0
    latest_distance_penalty = float(latest.get("distance_penalty", 0.0)) if isinstance(latest, dict) else 0.0
    latest_correct_distance_penalty = float(latest.get("correct_distance_penalty", 0.0)) if isinstance(latest, dict) else 0.0
    latest_streak = int(latest.get("correct_streak", 0)) if isinstance(latest, dict) else 0
    latest_bonus = float(latest.get("consecutive_bonus", 0.0)) if isinstance(latest, dict) else 0.0
    latest_click_threshold = float(
        latest.get("click_threshold_hz", config.get("click_gate_threshold_hz", 9.0))
    ) if isinstance(latest, dict) and isinstance(config, dict) else 9.0
    latest_click_evidence = float(latest.get("peak_click_evidence_hz", 0.0)) if isinstance(latest, dict) else 0.0
    latest_click_margin_bonus = float(latest.get("click_margin_bonus", 0.0)) if isinstance(latest, dict) else 0.0
    latest_low_peak_penalty = float(latest.get("low_peak_click_penalty", 0.0)) if isinstance(latest, dict) else 0.0
    latest_regression_penalty = float(latest.get("peak_regression_penalty", 0.0)) if isinstance(latest, dict) else 0.0
    latest_updates = 0
    latest_teacher_updates = 0
    if isinstance(latest, dict):
        latest_learning = latest.get("learning", {})
        if isinstance(latest_learning, dict):
            nested_learning = latest_learning.get("learning", {})
            if isinstance(nested_learning, dict):
                latest_updates = int(nested_learning.get("edge_updates", 0))
        teacher = latest_learning.get("motor_teacher", {})
        if isinstance(teacher, dict):
            latest_teacher_updates = int(teacher.get("edge_updates", 0))

    def svg_x(value: object) -> float:
        return float(value) * 1000.0

    def svg_y(value: object) -> float:
        return (1.0 - float(value)) * 600.0

    key_cards = []
    click_window_size = int(payload.get("click_window_size", 20))
    click_only = bool(config.get("click_only", True)) if isinstance(config, dict) else True
    for label in labels:
        stats = per_key.get(str(label), {}) if isinstance(per_key, dict) else {}
        key_accuracy = float(stats.get("accuracy", 0.0)) if isinstance(stats, dict) else 0.0
        recent_correct = int(stats.get("recent_correct", 0)) if isinstance(stats, dict) else 0
        total_trials = int(stats.get("trials", 0)) if isinstance(stats, dict) else 0
        mastered = bool(stats.get("mastered", False)) if isinstance(stats, dict) else False
        active = " active" if str(label) == latest_label else ""
        state = " mastered" if mastered else " learning"
        key_cards.append(
            f"<div class='key{active}{state}'>"
            f"<b>{html.escape(str(label))}</b>"
            f"<small>최근 {click_window_size}회 {recent_correct}/{click_window_size} · "
            f"전체 정답률 {key_accuracy:.0%} · 총 시도 {total_trials}회</small></div>"
        )
    svg_keys = []
    for key in layout:
        if not isinstance(key, dict):
            continue
        label = str(key.get("label", ""))
        x = svg_x(key.get("x", 0.0))
        y = svg_y(key.get("y", 0.0))
        width = float(key.get("width", 0.1)) * 1000.0
        height = float(key.get("height", 0.1)) * 600.0
        left = x - width / 2.0
        top = y - height / 2.0
        state = " target" if label == latest_label else ""
        svg_keys.append(
            f"<g class='svg-key{state}' aria-label='키 {html.escape(label, quote=True)}'>"
            f"<rect x='{left:.2f}' y='{top:.2f}' width='{width:.2f}' height='{height:.2f}' rx='5'></rect>"
            f"<text x='{x:.2f}' y='{y + 5:.2f}'>{html.escape(label)}</text></g>"
        )
    svg_targets = []
    targets = latest_body.get("targets", [])
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_x = svg_x(target.get("x", 0.0))
            target_y = svg_y(target.get("y", 0.0))
            if target.get("hitbox_shape") == "box" or target.get("hitbox_width") is not None:
                width = float(target.get("hitbox_width", 0.05)) * 1000.0
                height = float(target.get("hitbox_height", 0.05)) * 600.0
                left = target_x - width / 2.0
                top = target_y - height / 2.0
                svg_targets.append(
                    f"<rect class='target target-box' x='{left:.2f}' y='{top:.2f}' "
                    f"width='{width:.2f}' height='{height:.2f}' rx='5'>"
                    f"<title>목표 박스 {html.escape(str(target.get('id', '')))}</title></rect>"
                )
            else:
                radius = max(10.0, float(target.get("radius", 0.05)) * 1000.0)
                svg_targets.append(
                    f"<circle class='target' cx='{target_x:.2f}' cy='{target_y:.2f}' r='{radius:.2f}'>"
                    f"<title>목표 {html.escape(str(target.get('id', '')))}</title></circle>"
                )
    svg_arms = []
    arms = latest_body.get("arms", [])
    if isinstance(arms, list):
        for index, arm in enumerate(arms):
            if not isinstance(arm, dict):
                continue
            shoulder = arm.get("shoulder", [0.0, 0.0])
            elbow = arm.get("elbow", [0.0, 0.0])
            endpoint = arm.get("endpoint", [0.0, 0.0])
            sx, sy = svg_x(shoulder[0]), svg_y(shoulder[1])
            ex, ey = svg_x(elbow[0]), svg_y(elbow[1])
            tx, ty = svg_x(endpoint[0]), svg_y(endpoint[1])
            svg_arms.append(
                f"<g class='arm arm-{index + 1}' aria-label='팔 {index + 1}'>"
                f"<line class='segment' x1='{sx:.2f}' y1='{sy:.2f}' x2='{ex:.2f}' y2='{ey:.2f}'></line>"
                f"<line class='segment' x1='{ex:.2f}' y1='{ey:.2f}' x2='{tx:.2f}' y2='{ty:.2f}'></line>"
                f"<circle class='joint shoulder' cx='{sx:.2f}' cy='{sy:.2f}' r='10'></circle>"
                f"<circle class='joint' cx='{ex:.2f}' cy='{ey:.2f}' r='8'></circle>"
                f"<circle class='endpoint' cx='{tx:.2f}' cy='{ty:.2f}' r='9'></circle>"
                f"<text class='arm-label' x='{tx + 12:.2f}' y='{ty - 12:.2f}'>A{index + 1}</text></g>"
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
            f"<td>{float(row.get('peak_motor_rate_hz', 0.0)):.1f}Hz</td>"
            f"<td>{float(row.get('peak_click_rate_hz', 0.0)):.1f}Hz</td>"
            f"<td>{float(row.get('peak_click_evidence_hz', 0.0)):.1f}Hz</td>"
            f"<td>{html.escape(str(row.get('click_attempted_arms') or '-'))}</td>"
            f"<td>{float((row.get('movement') or {}).get('trajectory_span', 0.0)):.3f}</td>"
            f"<td>{float(row.get('min_target_distance', 0.0)):.3f}</td>"
            f"<td>{float(row.get('distance_penalty', 0.0)):.2f}</td>"
            f"<td>{float(row.get('correct_distance_penalty', 0.0)):.2f}</td>"
            f"<td>{int(row.get('correct_streak', 0))}</td>"
            f"<td>{float(row.get('consecutive_bonus', 0.0)):+.2f}</td>"
            f"<td>{float(row.get('click_margin_bonus', 0.0)):+.2f}</td>"
            f"<td>-{float(row.get('low_peak_click_penalty', 0.0)):.2f}</td>"
            "</tr>"
        )
    unlimited = trials <= 0
    progress = 0.0 if unlimited else 100.0 * completed / max(1, trials)
    trial_label = f"{completed}/∞" if unlimited else f"{completed}/{trials}"
    progress_label = "무제한 훈련" if unlimited else f"{progress:.1f}%"
    mastered_count = sum(
        int(isinstance(stats, dict) and stats.get("mastered", False))
        for stats in per_key.values()
    ) if isinstance(per_key, dict) else 0
    target_accuracy = float(payload.get("target_accuracy", 0.80))
    click_window = payload.get("click_window", [])
    click_window = click_window if isinstance(click_window, list) else []
    click_window_hits = sum(bool(value) for value in click_window[-click_window_size:])
    click_window_reached = bool(payload.get("click_window_reached", False))
    evaluation_ready = bool(payload.get("evaluation_ready", False))
    evaluation_correct = int(payload.get("evaluation_correct", 0))
    evaluation_total = int(payload.get("evaluation_total", len(labels) * click_window_size))
    evaluation_accuracy = float(payload.get("evaluation_accuracy", 0.0))
    evaluation_passed = bool(payload.get("evaluation_passed", False))
    click_status = "시험 통과" if evaluation_passed else "훈련/시험 대기"
    evaluation_status = (
        f"{evaluation_accuracy:.1%} · {'시험 통과' if evaluation_passed else '시험 불합격 · 재훈련'}"
        if evaluation_ready
        else "훈련 중 · 시험 전"
    )
    keyboard_goal = (
        f"키별 최근 {click_window_size}회 연속 100%"
        if click_only
        else f"목표 {target_accuracy:.0%}"
    )
    route_live = payload.get("route_diagnostics_live", {})
    route_live = route_live if isinstance(route_live, dict) else {}
    route_status = str(route_live.get("status", "idle"))
    route_stage = str(route_live.get("stage", "대기"))
    route_completed = int(route_live.get("completed", 0))
    route_total = int(route_live.get("total", 0))
    route_label = str(route_live.get("label") or "-")
    route_card = (
        "<div class='card'>경로 진단 (실시간)"
        f"<div class='value'>{html.escape(route_stage)}</div>"
        f"<small>{html.escape(route_status)} · {route_completed}/{route_total} · 키 {html.escape(route_label)}</small></div>"
        if route_live else ""
    )
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
.scene{{margin:18px 0;border:1px solid #2b3442;border-radius:9px;background:#0d1117;padding:8px}}
.scene svg{{display:block;width:100%;height:auto;min-height:360px;background:radial-gradient(circle at center,#172131,#0d1117)}}
.svg-key rect{{fill:#1b2230;stroke:#344052;stroke-width:1.2}}
.svg-key text{{fill:#e8edf5;text-anchor:middle;font-size:12px;font-weight:600;dominant-baseline:middle}}
.svg-key.target rect{{fill:#285b8f;stroke:#77b7ff;stroke-width:2.5}}
.target{{fill:#f2b84b;fill-opacity:.18;stroke:#f2b84b;stroke-width:2;stroke-dasharray:6 5}}
.arm .segment{{stroke:#77b7ff;stroke-width:7;stroke-linecap:round;opacity:.88}}
.arm-2 .segment{{stroke:#d891ff}} .arm-3 .segment{{stroke:#4ecb8a}} .arm-4 .segment{{stroke:#ff9f6e}}
.joint{{fill:#e8edf5;stroke:#101318;stroke-width:3}} .shoulder{{fill:#77b7ff}}
.endpoint{{fill:#f2b84b;stroke:#101318;stroke-width:3}}
.arm-label{{fill:#e8edf5;font-size:13px;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.card{{background:#171d27;border:1px solid #2b3442;border-radius:9px;padding:12px}}
.value{{font-size:1.35rem;font-weight:700}}
progress{{width:100%;height:18px}}
table{{width:100%;border-collapse:collapse;margin-top:14px}}th,td{{padding:7px;border-bottom:1px solid #2b3442;text-align:right}}th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
@media(max-width:720px){{body{{padding:14px}}.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:repeat(6,minmax(42px,1fr))}}.scene svg{{min-height:260px}}}}
</style></head><body>
<h1>DrosoMath 가상 키보드 매칭</h1>
<div class='cards'>
<div class='card'>Trial<div class='value'>{trial_label}</div></div>
<div class='card'>진행률<div class='value'>{progress_label}</div></div>
<div class='card'>정확도<div class='value'>{accuracy:.3f}</div></div>
<div class='card'>숙련 키<div class='value'>{mastered_count}/{len(labels)}</div></div>
<div class='card'>키별 최근 클릭<div class='value'>{mastered_count}/{len(labels)}</div><small>각 키 최근 {click_window_size}회 연속 100% · {click_status}</small></div>
<div class='card'>평가 정답률<div class='value'>{evaluation_correct}/{evaluation_total}</div><small>{evaluation_status}</small></div>
<div class='card'>현재 단계<div class='value'>{html.escape(str(payload.get('phase', 'training')).upper())}</div><small>최근 키: {html.escape(latest_label or '-')} · {'팔 고정' if bool(payload.get('config', {}).get('lock_arm_during_click_training', False)) else '팔 이동'}</small></div>
<div class='card'>최근 이동 범위<div class='value'>{float(latest_movement.get('trajectory_span', 0.0)):.3f}</div><small>팔 끝점 궤적 폭</small></div>
<div class='card'>목표까지 거리<div class='value'>{latest_target_distance:.3f}</div><small>팔 끝점과 목표 중심</small></div>
<div class='card'>실패 거리 처벌<div class='value'>-{latest_distance_penalty:.2f}</div><small>멀수록 증가</small></div>
<div class='card'>정답 위치 보정<div class='value'>-{latest_correct_distance_penalty:.2f}</div><small>박스 안 중심에서 멀수록</small></div>
<div class='card'>키별 연속 정답<div class='value'>{latest_streak}회</div><small>현재 키 기준</small></div>
<div class='card'>연속 정답 보너스<div class='value'>+{latest_bonus:.2f}</div><small>키별 누적 · 최대 +{float(payload.get('config', {}).get('max_consecutive_bonus', 1.0)):.2f}</small></div>
<div class='card'>클릭 임계값<div class='value'>{latest_click_threshold:.2f}Hz</div><small>20ms Peak click이 이 값을 넘으면 클릭</small></div>
<div class='card'>100ms 클릭 평균<div class='value'>{latest_click_evidence:.2f}Hz</div><small>최근 {int(config.get('click_evidence_windows', 5)) if isinstance(config, dict) else 5}개 window 관측값 · 행동 판정에는 미사용</small></div>
<div class='card'>저 peak 추가 처벌<div class='value'>-{latest_low_peak_penalty:.2f}</div><small>9Hz보다 낮을수록 증가</small></div>
<div class='card'>키별 peak 하락 처벌<div class='value'>-{latest_regression_penalty:.2f}</div><small>같은 키의 직전 peak보다 낮을 때 국소 교습 강화</small></div>
<div class='card'>시냅스 업데이트<div class='value'>{latest_updates:,}</div><small>최근 trial reward 적용량</small></div>
<div class='card'>클릭 교습 시냅스<div class='value'>{latest_teacher_updates:,}</div><small>저 peak 실패 때 click 출력으로만 강화</small></div>
{route_card}
</div>
<p><progress value='{progress:.3f}' max='100'></progress></p>
<h2>2D 가상 초파리 몸체</h2>
<div class='scene'>
<svg viewBox='0 0 1000 600' role='img' aria-label='현재 목표 키와 네 개의 가상 팔 상태'>
<title>가상 초파리 몸체와 키보드</title><desc>네 개의 팔 관절, 팔 끝점, 현재 목표 키를 표시한다.</desc>
{''.join(svg_keys)}{''.join(svg_targets)}{''.join(svg_arms)}
</svg></div>
<h2>가상 키보드 · {keyboard_goal}</h2><div class='grid'>{''.join(key_cards)}</div>
<h2>최근 trial</h2><table><tr><th>Trial</th><th>입력 키</th><th>클릭</th><th>결과</th><th>보상</th><th>Window</th><th>Peak motor</th><th>Peak click</th><th>누적 click</th><th>click arm</th><th>이동 폭</th><th>목표 거리</th><th>실패 거리 처벌</th><th>정답 위치 보정</th><th>키별 streak</th><th>streak 보너스</th><th>안정 클릭 보너스</th><th>저 peak 처벌</th></tr>{''.join(rows)}</table>
<p>페이지는 5초마다 자동 갱신된다. 클릭은 <b>Peak click</b>의 단일 20ms bin 값으로 판정한다. <b>누적 click</b>은 최근 100ms 평균을 보여 주는 안정성 관측값이며 클릭을 막지 않는다. Backend: numpy_cpu / closed-loop virtual body.</p>
</body></html>"""


@dataclass(frozen=True, slots=True)
class KeyboardTrainingConfig:
    min_connection_synapses: int = 5
    token_neurons: int = 6
    background_neurons: int = 6
    # 64 neurons per motor channel gives a 0.78Hz click-rate resolution at a
    # 20ms control window, reducing threshold jitter versus the old 32-neuron
    # (1.56Hz) readout without expanding the connectome itself.
    motor_population_size: int = 64
    body_mode: str = "one_arm_fan"
    click_only: bool = True
    lock_arm_during_click_training: bool = True
    click_gate_threshold_hz: float = 9.0
    # Action is an event: a strong 20ms click burst should not be discarded.
    # Longer averaging remains telemetry only, so it can diagnose stability
    # without turning an otherwise valid action into a failure.
    click_integration_windows: int = 1
    click_evidence_windows: int = 5
    click_margin_target_hz: float = 15.0
    click_margin_reward_scale: float = 0.50
    # Optional E.2 signal: only marginal successes receive a small generic
    # directional correction toward the existing click-margin target.
    success_margin_directional_learning: bool = False
    success_margin_directional_scale: float = 0.25
    click_penalty: float = 1.20
    # Click-only failures are corrected by motor-targeted teaching plasticity;
    # do not amplify their global negative reward.
    low_peak_click_penalty_scale: float = 0.0
    click_teacher_learning_rate: float = 0.08
    click_teacher_credit_floor: float = 0.05
    # A 20ms motor-rate sample is quantized. Ignore a one-bin fluctuation,
    # then apply a fixed local correction for a meaningful per-key drop.
    peak_regression_trigger_hz: float = 1.5
    peak_regression_fixed_penalty: float = 0.80
    peak_regression_escalation: float = 0.50
    peak_regression_max_penalty: float = 2.00
    persistent_peak_ema_alpha: float = 0.25
    persistent_deficit_ema_alpha: float = 0.25
    persistent_deficit_streak_gain: float = 0.15
    persistent_deficit_max_strength: float = 2.0
    teacher_confirmation_stability_gain: float = 0.05
    teacher_memory_edges_per_key: int = 128
    teacher_stability_protection: float = 0.85
    distance_penalty_scale: float = 1.20
    correct_distance_penalty_scale: float = 0.20
    consecutive_correct_bonus: float = 0.25
    max_consecutive_bonus: float = 1.00
    click_window_size: int = 20
    curriculum_stage: str = "click_accuracy"
    # ``0`` means unlimited training until the separate 1200-trial exam passes.
    trials: int = 20_000
    min_trials_per_key: int = 5
    target_accuracy: float = 0.80
    # Every Nth post-warmup trial is drawn from a shuffled all-key cycle.
    # Other trials sample hard examples probabilistically, never via argmin.
    coverage_interval: int = 4
    hard_mining_floor: float = 0.10
    hard_mining_power: float = 2.0
    robust_mastery_enabled: bool = False
    robust_mastery_macro_accuracy: float = 0.95
    robust_mastery_min_key_accuracy: float = 0.90
    retention_probe_interval: int = 600
    # Retention is a diagnostic measurement.  A single Poisson-driven response
    # is not evidence that a label was forgotten.
    retention_trials_per_key: int = 5
    retention_degraded_accuracy: float = 0.80
    retention_replay_bonus: float = 0.50
    duration_ms: float = 100.0
    control_window_ms: float = 20.0
    max_control_windows: int = 30
    stimulus_rate_hz: float = 205.0
    plastic_fraction: float = 0.05
    learning_rate: float = 0.02
    # D.2 is opt-in so existing keyboard runs remain byte-for-byte equivalent
    # in their learning policy unless explicitly enabled.
    adaptive_plastic_budget: bool = False
    adaptive_budget_max_promotions_per_event: int = 1
    # D.2.3 is a read-only matched probe.  It is opt-in so ordinary learning
    # runs do not spend diagnostics or alter their behavior.
    route_health_diagnostic: bool = False
    # E.1 records already-generated control-window activity only.  It never
    # replays the brain or participates in learning.
    temporal_engagement_diagnostic: bool = False
    budget_strength: float = 0.25
    checkpoint_every: int = 32
    dashboard_update_interval_seconds: float = 0.5
    profile_timing: bool = False
    resume: bool = False
    seed: int = 7

    def __post_init__(self) -> None:
        if self.min_connection_synapses < 1:
            raise ValueError("min_connection_synapses must be >= 1")
        if self.token_neurons < 2 or self.background_neurons < 1:
            raise ValueError("input groups must be non-empty")
        if self.motor_population_size < 2:
            raise ValueError("motor_population_size must be >= 2")
        if self.body_mode not in {
            "one_arm_fan",
            "one_arm_circle",
            "one_arm_circular",
            "four_arm_grid",
        }:
            raise ValueError("body_mode must be one_arm_fan, one_arm_circle, or four_arm_grid")
        if self.click_penalty < 0.0:
            raise ValueError("click_penalty must be >= 0")
        if self.low_peak_click_penalty_scale < 0.0:
            raise ValueError("low_peak_click_penalty_scale must be >= 0")
        if self.click_teacher_learning_rate < 0.0:
            raise ValueError("click_teacher_learning_rate must be >= 0")
        if self.click_teacher_credit_floor < 0.0:
            raise ValueError("click_teacher_credit_floor must be >= 0")
        if self.peak_regression_trigger_hz < 0.0:
            raise ValueError("peak_regression_trigger_hz must be >= 0")
        if self.peak_regression_fixed_penalty < 0.0:
            raise ValueError("peak_regression_fixed_penalty must be >= 0")
        if self.peak_regression_escalation < 0.0:
            raise ValueError("peak_regression_escalation must be >= 0")
        if self.peak_regression_max_penalty < self.peak_regression_fixed_penalty:
            raise ValueError("peak_regression_max_penalty must be >= fixed penalty")
        if not 0.0 < self.persistent_peak_ema_alpha <= 1.0:
            raise ValueError("persistent_peak_ema_alpha must be in (0, 1]")
        if not 0.0 < self.persistent_deficit_ema_alpha <= 1.0:
            raise ValueError("persistent_deficit_ema_alpha must be in (0, 1]")
        if self.persistent_deficit_streak_gain < 0.0 or self.persistent_deficit_max_strength < 0.0:
            raise ValueError("persistent deficit strengths must be >= 0")
        if self.teacher_confirmation_stability_gain < 0.0:
            raise ValueError("teacher_confirmation_stability_gain must be >= 0")
        if self.teacher_memory_edges_per_key < 1:
            raise ValueError("teacher_memory_edges_per_key must be positive")
        if not 0.0 <= self.teacher_stability_protection <= 1.0:
            raise ValueError("teacher_stability_protection must be in [0, 1]")
        if self.click_gate_threshold_hz <= 5.0:
            raise ValueError("click_gate_threshold_hz must be > baseline rate (5.0Hz)")
        if self.click_integration_windows < 1:
            raise ValueError("click_integration_windows must be >= 1")
        if self.click_evidence_windows < 1:
            raise ValueError("click_evidence_windows must be >= 1")
        if self.click_margin_target_hz <= self.click_gate_threshold_hz:
            raise ValueError("click_margin_target_hz must exceed click_gate_threshold_hz")
        if self.click_margin_reward_scale < 0.0:
            raise ValueError("click_margin_reward_scale must be >= 0")
        if self.success_margin_directional_scale < 0.0:
            raise ValueError("success_margin_directional_scale must be >= 0")
        if self.distance_penalty_scale < 0.0:
            raise ValueError("distance_penalty_scale must be >= 0")
        if self.correct_distance_penalty_scale < 0.0:
            raise ValueError("correct_distance_penalty_scale must be >= 0")
        if self.consecutive_correct_bonus < 0.0:
            raise ValueError("consecutive_correct_bonus must be >= 0")
        if self.max_consecutive_bonus < 0.0:
            raise ValueError("max_consecutive_bonus must be >= 0")
        if self.click_window_size < 1:
            raise ValueError("click_window_size must be >= 1")
        if self.curriculum_stage not in {"click_gate", "click_accuracy"}:
            raise ValueError("curriculum_stage must be click_gate or click_accuracy")
        if self.trials < 0 or self.checkpoint_every < 1:
            raise ValueError("trials must be >= 0 (0 means unlimited) and checkpoint_every must be positive")
        if self.min_trials_per_key < 1:
            raise ValueError("min_trials_per_key must be >= 1")
        if self.coverage_interval < 1:
            raise ValueError("coverage_interval must be >= 1")
        if self.hard_mining_floor <= 0.0:
            raise ValueError("hard_mining_floor must be > 0")
        if self.hard_mining_power <= 0.0:
            raise ValueError("hard_mining_power must be > 0")
        if not 0.5 < self.robust_mastery_macro_accuracy <= 1.0:
            raise ValueError("robust_mastery_macro_accuracy must be in (0.5, 1]")
        if not 0.5 < self.robust_mastery_min_key_accuracy <= 1.0:
            raise ValueError("robust_mastery_min_key_accuracy must be in (0.5, 1]")
        if self.retention_probe_interval < 1:
            raise ValueError("retention_probe_interval must be positive")
        if self.retention_trials_per_key < 1:
            raise ValueError("retention_trials_per_key must be positive")
        if not 0.0 <= self.retention_degraded_accuracy <= 1.0:
            raise ValueError("retention_degraded_accuracy must be in [0, 1]")
        if self.retention_replay_bonus < 0.0:
            raise ValueError("retention_replay_bonus must be >= 0")
        if not 0.5 < self.target_accuracy <= 1.0:
            raise ValueError("target_accuracy must be in (0.5, 1]")
        if self.duration_ms <= 0.0 or self.control_window_ms <= 0.0:
            raise ValueError("durations must be > 0")
        if self.max_control_windows < 1:
            raise ValueError("max_control_windows must be >= 1")
        if self.dashboard_update_interval_seconds <= 0.0:
            raise ValueError("dashboard_update_interval_seconds must be > 0")
        if self.adaptive_budget_max_promotions_per_event < 1:
            raise ValueError("adaptive_budget_max_promotions_per_event must be >= 1")


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
    """Run neural prompt trials through a selectable virtual motor body."""

    def __init__(self, connectome, *, config: KeyboardTrainingConfig) -> None:
        np = __import__("numpy")
        self.np = np
        self.config = config
        self.encoder = build_token_encoder(connectome, config=config)
        if config.body_mode in {"one_arm_fan", "one_arm_circle", "one_arm_circular"}:
            world = OneArmWorld()
            keyboard_type = CircularKeyboard if config.body_mode == "one_arm_circle" else FanKeyboard
            self.task = KeyboardMatchingTask(
                keyboard=keyboard_type.for_arm(world.arm_configs[0]),
                world=world,
                motor=OneArmMotorAdapter(
                    config=FourArmMotorConfig(
                        click_threshold_hz=config.click_gate_threshold_hz,
                        click_integration_windows=config.click_integration_windows,
                    ),
                ),
                seed=config.seed,
            )
        else:
            self.task = KeyboardMatchingTask(seed=config.seed)
        self.motor_channel_count = self.task.motor.channel_count
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
        # ``output.indices`` is descending route-score order.  Contiguous
        # slicing previously handed the click channel (index 4) only the
        # lowest-ranked 64 neurons.  Interleave ranks so every motor channel,
        # including click, receives the same score range.
        self.channel_groups = tuple(
            np.asarray(self.output.indices[channel::self.motor_channel_count], dtype=np.int32)
            for channel in range(self.motor_channel_count)
        )
        if len(self.channel_groups) != self.motor_channel_count:
            raise ValueError("route-aware output did not produce 20 motor populations")
        selected_scores = np.asarray(self.route_provenance.get("selected_route_scores", []), dtype=np.float32)
        self.route_provenance["channel_allocation"] = "rank_round_robin"
        self.route_provenance["per_channel_route_scores"] = [
            {
                "channel": channel,
                "mean": float(selected_scores[channel::self.motor_channel_count].mean()),
                "minimum": float(selected_scores[channel::self.motor_channel_count].min()),
                "maximum": float(selected_scores[channel::self.motor_channel_count].max()),
            }
            for channel in range(self.motor_channel_count)
        ]
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
        self.output_context = {
            name: self.channel_groups[index]
            for index, name in enumerate((
                "motor/shoulder_positive", "motor/shoulder_negative",
                "motor/elbow_positive", "motor/elbow_negative", "motor/click",
            ))
        }
        self.plasticity_controller = PlasticityController(
            DirectionalModulationConfig(learning_rate=config.learning_rate)
        )
        self.plasticity_budget_adaptation = PlasticityBudgetAdaptation(
            config=PlasticityBudgetAdaptationConfig(
                enabled=config.adaptive_plastic_budget,
                max_promotions_per_event=config.adaptive_budget_max_promotions_per_event,
            ),
            route_controller=self.plasticity_controller,
        )
        self.channel_homeostasis = ChannelHomeostasis(target_rate_hz=config.click_gate_threshold_hz)
        click_output = self.channel_groups[4]
        self.click_teacher_edge_indices = np.flatnonzero(
            np.isin(connectome.post_indices, click_output, assume_unique=False)
            & (connectome.signed_synapse_counts > 0.0)
        ).astype(np.int32, copy=False)
        self.click_teacher_pre_indices = np.searchsorted(
            connectome.indptr,
            self.click_teacher_edge_indices,
            side="right",
        ).astype(np.int32, copy=False) - 1
        self.stimulus_indices_by_label = {
            label: self.brain.indices_for_ids(self.encoder.encode(label))
            for label in KEY_LABELS
        }
        self.click_teacher_edges_by_label = {
            label: self._find_click_teacher_edges(indices)
            for label, indices in self.stimulus_indices_by_label.items()
        }
        self.reward_rule = UsageRewardRule(learning_rate=config.learning_rate)
        self.normalizer = OutgoingBudgetNormalizer(
            strength=config.budget_strength,
            stability_protection=config.teacher_stability_protection,
        )
        self.steps_per_window = max(
            1,
            int(round(config.control_window_ms / self.brain.params.dt_ms)),
        )
        self.max_control_windows = min(
            config.max_control_windows,
            max(1, int(math.ceil(config.duration_ms / config.control_window_ms))),
        )
        self.rate_scale = np.float32(
            1000.0 / (config.control_window_ms * config.motor_population_size)
        )
        self.rng = np.random.default_rng(config.seed + 82_001)
        self.correct_streak_by_label = {label: 0 for label in KEY_LABELS}
        self.previous_peak_by_label = {label: None for label in KEY_LABELS}
        self.peak_regression_streak_by_label = {label: 0 for label in KEY_LABELS}
        self.peak_ema_by_label = {label: None for label in KEY_LABELS}
        self.deficit_ema_by_label = {label: 0.0 for label in KEY_LABELS}
        self.subthreshold_streak_by_label = {label: 0 for label in KEY_LABELS}
        self.teacher_memory_edges_by_label = {
            label: np.empty(0, dtype=np.int32) for label in KEY_LABELS
        }
        self._pending_teacher_normalizer: dict[str, object] | None = None

    def _find_click_teacher_edges(self, stimulus_indices) -> object:
        """Find real two-hop excitatory paths from one token to click output."""
        np = self.np
        connectome = self.brain.connectome
        reachable = np.zeros(connectome.neuron_count, dtype=np.bool_)
        stimulus = np.asarray(stimulus_indices, dtype=np.int32)
        reachable[stimulus] = True
        indptr = connectome.indptr
        posts = connectome.post_indices
        for pre in stimulus:
            start, stop = int(indptr[pre]), int(indptr[pre + 1])
            reachable[posts[start:stop]] = True
        return self.click_teacher_edge_indices[
            reachable[self.click_teacher_pre_indices]
        ]

    def _apply_low_peak_click_teacher(
        self,
        label: str,
        peak_click_rate_hz: float,
        regression_strength: float,
    ) -> dict[str, float | int]:
        """Strengthen only active excitatory inputs into the click readout.

        A no-click trial has no correct action to reinforce.  Applying a larger
        global negative reward would depress its already weak active pathway.
        This three-factor teaching update instead uses the click-rate deficit to
        potentiate eligible anatomical inputs ending at the designated click
        motor population.  It never changes inhibitory edge signs or anatomy.
        """
        np = self.np
        threshold = self.config.click_gate_threshold_hz
        deficit = min(1.0, max(0.0, (threshold - peak_click_rate_hz) / threshold))
        state = self.brain.plasticity
        candidates = self.click_teacher_edges_by_label[label]
        candidate_count = int(len(candidates))
        candidate_pre = self.click_teacher_pre_indices[
            np.isin(self.click_teacher_edge_indices, candidates, assume_unique=False)
        ]
        plastic = state.plastic_mask[candidates]
        eligible = state.eligibility[candidates] > 0.0
        active = plastic & eligible
        plastic_edges = candidates[plastic]
        edges = candidates[active]
        teacher_mode = "DIRECT_CLICK_EDGE"
        # A dead final click edge must not turn a real active upstream route
        # into an unlearnable label.  Walk exactly one real, excitatory hop
        # from recently firing neurons into the click-input presynaptic set.
        # This never invents anatomy or changes inhibitory signs.
        if not len(edges) and len(candidate_pre) and hasattr(self.brain.connectome, "neuron_count"):
            click_input = np.zeros(self.brain.connectome.neuron_count, dtype=np.bool_)
            click_input[np.unique(candidate_pre)] = True
            upstream_chunks = []
            indptr = self.brain.connectome.indptr
            posts = self.brain.connectome.post_indices
            signed = self.brain.connectome.signed_synapse_counts
            for pre in sorted(self.brain._recent_presynaptic):
                start, stop = int(indptr[pre]), int(indptr[pre + 1])
                local = np.arange(start, stop, dtype=np.int32)
                valid = (
                    click_input[posts[start:stop]]
                    & (signed[start:stop] > 0.0)
                    & state.plastic_mask[local]
                    & (state.eligibility[local] > 0.0)
                )
                if np.any(valid):
                    upstream_chunks.append(local[valid])
            if upstream_chunks:
                edges = np.unique(np.concatenate(upstream_chunks)).astype(np.int32, copy=False)
                teacher_mode = "UPSTREAM_ROUTE_FALLBACK"
        eligibility_nonzero = int(np.count_nonzero(eligible))
        mean_before = float(state.multiplier[edges].mean()) if len(edges) else 0.0
        effective_before = float(
            np.abs(
                self.brain.connectome.signed_synapse_counts[edges]
                * state.multiplier[edges]
                * self.brain.params.mv_per_synapse
            ).sum()
        ) if len(edges) else 0.0
        common = {
            "label": label,
            "candidate_edge_count": candidate_count,
            "plastic_edge_count": int(len(plastic_edges)),
            "nonzero_eligibility_edge_count": eligibility_nonzero,
            "usage_nonzero_edge_count": int(np.count_nonzero(state.usage_ema[candidates] > 0.0)),
            "total_eligibility": float(state.eligibility[candidates].sum()),
            "total_credit": float((state.usage_ema[candidates] * state.eligibility[candidates]).sum()),
            "active_eligible_edge_count": int(len(edges)),
            "relevant_presynaptic_count": int(len(np.unique(candidate_pre))),
            "active_presynaptic_count": int(len(np.unique(candidate_pre[active]))),
            "teacher_update_mode": teacher_mode,
            "upstream_fallback_edge_count": int(len(edges)) if teacher_mode == "UPSTREAM_ROUTE_FALLBACK" else 0,
            "mean_multiplier_before": mean_before,
            "effective_strength_before": effective_before,
        }
        if deficit <= 0.0 or self.config.click_teacher_learning_rate <= 0.0 or not len(edges):
            return {
                **common,
                "edge_updates": 0,
                "mean_delta": 0.0,
                "max_delta": 0.0,
                "mean_multiplier_after": mean_before,
                "effective_strength_after": effective_before,
                "deficit": deficit,
                "regression_strength": regression_strength,
            }
        old = state.multiplier[edges].copy()
        local_credit = np.maximum(
            state.usage_ema[edges] * state.eligibility[edges],
            self.config.click_teacher_credit_floor,
        )
        teaching_strength = 1.0 + regression_strength
        delta = (
            self.config.click_teacher_learning_rate
            * teaching_strength
            * deficit
            * local_credit
        )
        state.multiplier[edges] = np.clip(
            old + delta,
            state.config.min_multiplier,
            state.config.max_multiplier,
        )
        actual = state.multiplier[edges] - old
        mean_after = float(state.multiplier[edges].mean())
        effective_after = float(
            np.abs(
                self.brain.connectome.signed_synapse_counts[edges]
                * state.multiplier[edges]
                * self.brain.params.mv_per_synapse
            ).sum()
        )
        self._pending_teacher_normalizer = {
            "label": label,
            "edges": edges.copy(),
            "teacher_mean_before": mean_before,
            "teacher_mean_after": mean_after,
            "teacher_gain": mean_after - mean_before,
        }
        # An ordered recency buffer, not ``np.unique`` (which sorts by edge ID).
        remembered = [int(edge) for edge in self.teacher_memory_edges_by_label[label] if int(edge) not in set(map(int, edges))]
        remembered.extend(int(edge) for edge in edges)
        self.teacher_memory_edges_by_label[label] = np.asarray(
            remembered[-self.config.teacher_memory_edges_per_key:], dtype=np.int32
        )
        return {
            **common,
            "edge_updates": int(len(edges)),
            "mean_delta": float(actual.mean()) if len(actual) else 0.0,
            "max_delta": float(actual.max()) if len(actual) else 0.0,
            "mean_multiplier_after": mean_after,
            "effective_strength_after": effective_after,
            "deficit": deficit,
            "regression_strength": regression_strength,
        }

    def _consolidate_confirmed_teacher_memory(self, label: str, *, confirmed: bool) -> dict[str, object]:
        edges = self.teacher_memory_edges_by_label[label]
        if not confirmed or len(edges) == 0 or self.config.teacher_confirmation_stability_gain <= 0.0:
            return {"confirmed": bool(confirmed), "edge_updates": 0, "mean_stability": 0.0}
        state = self.brain.plasticity
        edges = edges[state.plastic_mask[edges]]
        if not len(edges):
            return {"confirmed": True, "edge_updates": 0, "mean_stability": 0.0}
        stability = state.stability[edges]
        stability += self.config.teacher_confirmation_stability_gain * (1.0 - stability)
        self.np.clip(stability, 0.0, 1.0, out=stability)
        return {"confirmed": True, "edge_updates": int(len(edges)), "mean_stability": float(stability.mean())}

    def _learning_signal_for_trial(
        self,
        *,
        reward: float,
        correct: bool,
        current_deficit: float,
        click_gain: float,
        success_margin_directional_error: float = 0.0,
    ) -> LearningSignal:
        """Translate keyboard outcome into task-independent brain feedback."""
        if self.config.click_only and correct:
            return LearningSignal(
                reward=reward,
                directional_error={"motor/click": float(success_margin_directional_error)},
                reinforcement={"motor/click": 1.0},
                surprise=current_deficit,
                success=True,
            )
        return LearningSignal(
            reward=reward,
            directional_error={
                "motor/click": current_deficit * click_gain if self.config.click_only else 0.0,
            },
            reinforcement={},
            surprise=current_deficit,
            success=correct,
        )

    def run_trial(self, label: str, *, learn: bool = True) -> dict[str, object]:
        np = self.np
        timings: dict[str, float] | None = {} if self.config.profile_timing else None
        reset_started = time.perf_counter() if timings is not None else 0.0
        self.brain.reset()
        if timings is not None:
            timings["brain_reset_seconds"] = time.perf_counter() - reset_started
        previous_tracking = None
        if not learn:
            previous_tracking = self.brain.set_plasticity_tracking(False)
        self.task.reset(label)
        if self.config.click_only:
            target = self.task.keyboard.key(label)
            if not isinstance(self.task.world, OneArmWorld):
                raise ValueError("click_only curriculum requires a one-arm world")
            self.task.world.position_arm_to(0, target.x, target.y)
        stimulus_indices = self.stimulus_indices_by_label[label]
        last = None
        windows = 0
        peak_motor_rate_hz = 0.0
        peak_click_rate_hz = 0.0
        peak_click_evidence_hz = 0.0
        click_evidence_history: list[float] = [0.0] * self.config.click_evidence_windows
        click_attempted_arms: set[int] = set()
        click_output_spike_count = 0
        active_click_outputs: set[int] = set()
        counts = np.zeros(self.motor_channel_count, dtype=np.int32)
        temporal_windows: list[dict[str, object]] = []
        previous_window_neurons: set[int] | None = None
        previous_window_causal_edges: set[int] | None = None
        first_any_network_activity_window = None
        first_click_causal_activity_window = None
        first_click_output_spike_window = None
        first_window_click_rate_above_25pct_threshold = None
        first_window_click_rate_above_50pct_threshold = None
        first_window_click_rate_above_75pct_threshold = None
        first_window_click_rate_above_threshold = None
        early_window_limit = max(1, int(math.ceil(self.max_control_windows * 0.25)))
        initial_body = self.task.observation().get("body", {})
        initial_endpoints = (
            initial_body.get("endpoints", [])
            if isinstance(initial_body, dict)
            else []
        )
        movement_points = [
            (float(endpoint[0]), float(endpoint[1]))
            for endpoint in initial_endpoints
            if isinstance(endpoint, (list, tuple)) and len(endpoint) >= 2
        ]
        started = time.perf_counter()
        neural_step_seconds = 0.0
        motor_count_seconds = 0.0
        for windows in range(1, self.max_control_windows + 1):
            counts.fill(0)
            window_total_spikes = 0
            window_fired_neurons: set[int] = set()
            window_click_spikes = 0
            window_click_neurons: set[int] = set()
            for _ in range(self.steps_per_window):
                step_started = time.perf_counter() if timings is not None else 0.0
                fired, _ = self.brain.step(
                    stimulus_indices=stimulus_indices,
                    stimulus_rate_hz=self.config.stimulus_rate_hz,
                )
                window_total_spikes += int(len(fired))
                window_fired_neurons.update(int(index) for index in fired)
                if timings is not None:
                    neural_step_seconds += time.perf_counter() - step_started
                    count_started = time.perf_counter()
                if len(fired):
                    local_click = fired[self.channel_lookup[fired] == 4]
                    window_click_spikes += int(len(local_click))
                    window_click_neurons.update(int(index) for index in local_click)
                    local = self.channel_lookup[fired]
                    local = local[local >= 0]
                    if len(local):
                        counts += np.bincount(
                            local,
                            minlength=self.motor_channel_count,
                        ).astype(np.int32, copy=False)
                if timings is not None:
                    motor_count_seconds += time.perf_counter() - count_started
            rates = counts.astype(np.float32) * self.rate_scale
            peak_motor_rate_hz = max(peak_motor_rate_hz, float(rates.max()))
            click_rates = rates[4::5]
            window_click_rate_hz = float(click_rates.max())
            click_output_spike_count += window_click_spikes
            active_click_outputs.update(window_click_neurons)
            peak_click_rate_hz = max(peak_click_rate_hz, window_click_rate_hz)
            click_evidence_history.append(float(click_rates.max()))
            del click_evidence_history[:-self.config.click_evidence_windows]
            peak_click_evidence_hz = max(
                peak_click_evidence_hz,
                sum(click_evidence_history) / len(click_evidence_history),
            )
            if temporal_windows is not None and self.config.temporal_engagement_diagnostic:
                causal_probe = self.plasticity_controller.diagnose_activity_window(
                    self.brain,
                    np.asarray(sorted(window_fired_neurons), dtype=np.int32),
                    {"motor/click": self.output_context["motor/click"]},
                )["motor/click"]
                causal_indices = set(
                    int(edge) for edge in causal_probe.pop("_causal_edge_indices", ())
                )
                if first_any_network_activity_window is None and window_total_spikes > 0:
                    first_any_network_activity_window = windows
                if first_click_causal_activity_window is None and causal_probe["active_click_causal_edges"] > 0:
                    first_click_causal_activity_window = windows
                if first_click_output_spike_window is None and window_click_spikes > 0:
                    first_click_output_spike_window = windows
                threshold = float(self.config.click_gate_threshold_hz)
                threshold_events = (
                    (0.25, "first_window_click_rate_above_25pct_threshold"),
                    (0.50, "first_window_click_rate_above_50pct_threshold"),
                    (0.75, "first_window_click_rate_above_75pct_threshold"),
                    (1.00, "first_window_click_rate_above_threshold"),
                )
                for fraction, attribute in threshold_events:
                    current_value = {
                        "first_window_click_rate_above_25pct_threshold": first_window_click_rate_above_25pct_threshold,
                        "first_window_click_rate_above_50pct_threshold": first_window_click_rate_above_50pct_threshold,
                        "first_window_click_rate_above_75pct_threshold": first_window_click_rate_above_75pct_threshold,
                        "first_window_click_rate_above_threshold": first_window_click_rate_above_threshold,
                    }[attribute]
                    if current_value is None and window_click_rate_hz >= fraction * threshold:
                        if attribute == "first_window_click_rate_above_25pct_threshold":
                            first_window_click_rate_above_25pct_threshold = windows
                        elif attribute == "first_window_click_rate_above_50pct_threshold":
                            first_window_click_rate_above_50pct_threshold = windows
                        elif attribute == "first_window_click_rate_above_75pct_threshold":
                            first_window_click_rate_above_75pct_threshold = windows
                        else:
                            first_window_click_rate_above_threshold = windows
                active_neurons = set(window_fired_neurons)
                active_jaccard = (
                    len(active_neurons & previous_window_neurons)
                    / max(1, len(active_neurons | previous_window_neurons))
                    if previous_window_neurons is not None else None
                )
                causal_jaccard = (
                    len(causal_indices & previous_window_causal_edges)
                    / max(1, len(causal_indices | previous_window_causal_edges))
                    if previous_window_causal_edges is not None else None
                )
                temporal_windows.append(build_temporal_window_record(
                    window_index=windows,
                    total_spikes=window_total_spikes,
                    fired_neurons=window_fired_neurons,
                    recent_presynaptic_count=len(self.brain._recent_presynaptic),
                    click_output_spikes=window_click_spikes,
                    click_output_neurons=window_click_neurons,
                    click_rate_hz=window_click_rate_hz,
                    motor_channel_rates=rates,
                    causal_summary=causal_probe,
                    active_neuron_jaccard=active_jaccard,
                    click_causal_jaccard=causal_jaccard,
                ))
                previous_window_neurons = active_neurons
                previous_window_causal_edges = causal_indices
            actions = self.task.motor.decode(rates)
            if self.config.click_only and self.config.lock_arm_during_click_training:
                # Click-gate curriculum: the target starts under the endpoint.
                # Learn the token-to-click timing before teaching navigation;
                # otherwise a no-click window can move the endpoint away first.
                actions = tuple(ArmAction(click=action.click) for action in actions)
            elif self.config.click_only:
                # A click is an event at the current endpoint.  Do not let a
                # simultaneous joint command move a correct click away from
                # its target; joint exploration remains active on no-click
                # windows and is still learned from failures.
                actions = tuple(
                    ArmAction(click=action.click) if action.click else action
                    for action in actions
                )
            click_attempted_arms.update(
                arm_index for arm_index, action in enumerate(actions) if action.click
            )
            last = self.task.step(actions)
            step_body = self.task.observation().get("body", {})
            step_endpoints = (
                step_body.get("endpoints", [])
                if isinstance(step_body, dict)
                else []
            )
            movement_points.extend(
                (float(endpoint[0]), float(endpoint[1]))
                for endpoint in step_endpoints
                if isinstance(endpoint, (list, tuple)) and len(endpoint) >= 2
            )
            if last.done:
                break

        if previous_tracking is not None:
            self.brain.set_plasticity_tracking(previous_tracking)

        temporal_engagement = None
        if self.config.temporal_engagement_diagnostic:
            observed_windows = len(temporal_windows)
            early_windows = temporal_windows[:early_window_limit]

            def mean_window(name, rows=None):
                rows = temporal_windows if rows is None else rows
                values = [float(row[name]) for row in rows]
                return float(sum(values) / len(values)) if values else 0.0

            active_jaccards = [
                float(row["active_neuron_jaccard"])
                for row in temporal_windows
                if row["active_neuron_jaccard"] is not None
            ]
            causal_jaccards = [
                float(row["click_causal_jaccard"])
                for row in temporal_windows
                if row["click_causal_jaccard"] is not None
            ]
            total_eligibility = [
                float(row["total_click_route_eligibility"])
                for row in temporal_windows
            ]
            peak_row = max(
                temporal_windows,
                key=lambda row: float(row["click_rate_hz"]),
                default=None,
            )
            temporal_engagement = {
                "window_count": observed_windows,
                "allowed_window_count": self.max_control_windows,
                "early_window_definition": {
                    "type": "first_fraction_of_allowed_windows",
                    "fraction": 0.25,
                    "window_count": early_window_limit,
                    "observed_count": len(early_windows),
                },
                "windows": temporal_windows,
                "first_any_network_activity_window": first_any_network_activity_window,
                "first_click_causal_activity_window": first_click_causal_activity_window,
                "first_click_output_spike_window": first_click_output_spike_window,
                "first_window_click_rate_above_25pct_threshold": first_window_click_rate_above_25pct_threshold,
                "first_window_click_rate_above_50pct_threshold": first_window_click_rate_above_50pct_threshold,
                "first_window_click_rate_above_75pct_threshold": first_window_click_rate_above_75pct_threshold,
                "first_window_click_rate_above_threshold": first_window_click_rate_above_threshold,
                "mean_adjacent_window_active_neuron_jaccard": float(sum(active_jaccards) / len(active_jaccards)) if active_jaccards else 0.0,
                "mean_adjacent_window_click_causal_jaccard": float(sum(causal_jaccards) / len(causal_jaccards)) if causal_jaccards else 0.0,
                "activity_persistence_windows": int(sum(value >= 0.5 for value in active_jaccards)),
                "peak_unique_neurons": max((int(row["unique_fired_neurons"]) for row in temporal_windows), default=0),
                "mean_unique_neurons_per_window": mean_window("unique_fired_neurons"),
                "mean_windows_per_trial": float(observed_windows),
                "early_active_click_causal_edges": mean_window("active_click_causal_edges", early_windows),
                "early_active_plastic_click_causal_edges": mean_window("active_plastic_click_causal_edges", early_windows),
                "early_active_frozen_click_causal_edges": mean_window("active_frozen_click_causal_edges", early_windows),
                "early_net_click_route_influence": mean_window("net_click_route_influence", early_windows),
                "mean_net_click_route_influence_per_window": mean_window("net_click_route_influence"),
                "mean_positive_effect_magnitude_per_window": mean_window("positive_effect_magnitude"),
                "mean_negative_effect_magnitude_per_window": mean_window("negative_effect_magnitude"),
                "raw_total_click_route_eligibility": float(sum(total_eligibility)),
                "eligibility_per_window": float(sum(total_eligibility) / max(1, observed_windows)),
                "early_eligibility_per_window": mean_window("total_click_route_eligibility", early_windows),
                "eligibility_accumulation_rate": float(
                    (total_eligibility[-1] - total_eligibility[0]) / max(1, len(total_eligibility) - 1)
                ) if total_eligibility else 0.0,
                "mean_click_route_eligibility_per_window": mean_window("mean_click_route_eligibility"),
                "mean_eligible_click_route_edges_per_window": mean_window("eligible_click_route_edges"),
                "unique_click_output_neurons_recruited": len(active_click_outputs),
                "total_click_output_spikes": click_output_spike_count,
                "peak_click_output_rate_hz": peak_click_rate_hz,
                "time_to_peak_click_rate_window": int(peak_row["window_index"]) if peak_row is not None else None,
                "mean_click_output_synchrony_proxy": float(
                    sum(
                        float(row["active_click_output_neurons"]) / max(1, len(self.channel_groups[4]))
                        for row in temporal_windows
                    ) / max(1, observed_windows)
                ),
                "peak_simultaneous_click_output_recruitment": max(
                    (int(row["active_click_output_neurons"]) for row in temporal_windows),
                    default=0,
                ),
            }

        correct = bool(last is not None and last.clicked_label == label)
        movement = _movement_summary(movement_points)
        body = self.task.observation()["body"]
        target_key = self.task.keyboard.key(label)
        endpoints = body.get("endpoints", []) if isinstance(body, dict) else []
        min_target_distance = min(
            (
                math.hypot(float(endpoint[0]) - target_key.x, float(endpoint[1]) - target_key.y)
                for endpoint in endpoints
            ),
            default=float("inf"),
        )
        hitbox_half_diagonal = math.hypot(
            target_key.width / 2.0,
            target_key.height / 2.0,
        )
        outside_distance = max(0.0, min_target_distance - hitbox_half_diagonal)
        arm_index = target_key.owner_arm if target_key.owner_arm is not None else 0
        arm_config = self.task.world.arm_configs[arm_index]
        max_target_distance = max(
            1e-9,
            2.0 * (arm_config.upper_arm_length + arm_config.forearm_length),
        )
        distance_ratio = min(1.0, outside_distance / max_target_distance)
        distance_penalty = (
            self.config.distance_penalty_scale * distance_ratio
            if learn and not correct
            else 0.0
        )
        correct_distance_ratio = min(
            1.0,
            min_target_distance / max(hitbox_half_diagonal, 1e-9),
        )
        correct_distance_penalty = (
            self.config.correct_distance_penalty_scale * correct_distance_ratio
            if learn and correct
            else 0.0
        )
        previous_correct_streak = self.correct_streak_by_label.get(label, 0)
        if learn:
            current_correct_streak = (
                previous_correct_streak + 1
                if correct
                else 0
            )
            self.correct_streak_by_label[label] = current_correct_streak
        else:
            current_correct_streak = previous_correct_streak
        consecutive_bonus = (
            min(
                self.config.max_consecutive_bonus,
                max(0, current_correct_streak - 1)
                * self.config.consecutive_correct_bonus,
            )
            if learn and correct
            else 0.0
        )
        click_margin_bonus = (
            self.config.click_margin_reward_scale * min(
                1.0,
                max(
                    0.0,
                    (peak_click_rate_hz - self.config.click_gate_threshold_hz)
                    / (self.config.click_margin_target_hz - self.config.click_gate_threshold_hz),
                ),
            )
            if learn and correct
            else 0.0
        )
        success_margin_deficit = (
            calculate_success_margin_deficit(
                peak_click_rate_hz=peak_click_rate_hz,
                threshold_hz=self.config.click_gate_threshold_hz,
                target_hz=self.config.click_margin_target_hz,
            )
            if correct
            else 0.0
        )
        success_margin_directional_error = (
            self.config.success_margin_directional_scale * success_margin_deficit
            if learn and correct and self.config.success_margin_directional_learning
            else 0.0
        )
        low_peak_click_penalty = (
            self.config.low_peak_click_penalty_scale * min(
                1.0,
                max(
                    0.0,
                    (self.config.click_gate_threshold_hz - peak_click_rate_hz)
                    / self.config.click_gate_threshold_hz,
                ),
            )
            if learn and self.config.click_only and not correct
            else 0.0
        )
        previous_peak = self.previous_peak_by_label[label]
        peak_regression_ratio = (
            min(
                1.0,
                max(
                    0.0,
                    (float(previous_peak) - peak_click_rate_hz)
                    / max(float(previous_peak), self.config.click_gate_threshold_hz),
                ),
            )
            if previous_peak is not None
            else 0.0
        )
        meaningful_regression = bool(
            learn
            and self.config.click_only
            and not correct
            and previous_peak is not None
            and peak_click_rate_hz
            < float(previous_peak) - self.config.peak_regression_trigger_hz
        )
        if meaningful_regression:
            regression_streak = self.peak_regression_streak_by_label[label] + 1
            self.peak_regression_streak_by_label[label] = regression_streak
            peak_regression_penalty = min(
                self.config.peak_regression_max_penalty,
                self.config.peak_regression_fixed_penalty
                + self.config.peak_regression_escalation * (regression_streak - 1),
            )
        else:
            # Recovery, a correct click, or a small quantization wobble clears
            # the local correction immediately for this key only.
            regression_streak = 0
            self.peak_regression_streak_by_label[label] = 0
            peak_regression_penalty = 0.0
        regression_strength = (
            peak_regression_penalty
            / max(self.config.peak_regression_fixed_penalty, 1e-9)
            if meaningful_regression
            else 0.0
        )
        current_deficit = min(
            1.0,
            max(0.0, (self.config.click_gate_threshold_hz - peak_click_rate_hz)
                 / self.config.click_gate_threshold_hz),
        )
        channel_names = tuple(self.output_context)
        final_rates = rates if "rates" in locals() else np.zeros(self.motor_channel_count, dtype=np.float32)
        homeostatic_gains = self.channel_homeostasis.observe({
            name: float(final_rates[index]) for index, name in enumerate(channel_names)
        })
        old_peak_ema = self.peak_ema_by_label[label]
        peak_ema = (
            peak_click_rate_hz if old_peak_ema is None else
            (1.0 - self.config.persistent_peak_ema_alpha) * float(old_peak_ema)
            + self.config.persistent_peak_ema_alpha * peak_click_rate_hz
        )
        old_deficit_ema = self.deficit_ema_by_label[label]
        deficit_ema = (
            (1.0 - self.config.persistent_deficit_ema_alpha) * old_deficit_ema
            + self.config.persistent_deficit_ema_alpha * current_deficit
        )
        subthreshold_streak = (
            self.subthreshold_streak_by_label[label] + 1
            if learn and not correct and peak_click_rate_hz < self.config.click_gate_threshold_hz
            else 0
        )
        self.peak_ema_by_label[label] = peak_ema
        self.deficit_ema_by_label[label] = deficit_ema
        self.subthreshold_streak_by_label[label] = subthreshold_streak
        persistence_strength = min(
            self.config.persistent_deficit_max_strength,
            deficit_ema + self.config.persistent_deficit_streak_gain * max(0, subthreshold_streak - 1),
        )
        teacher_strength = regression_strength + persistence_strength
        self.previous_peak_by_label[label] = peak_click_rate_hz
        # Failed trials are graded by endpoint distance from the target.  The
        # key hitbox diagonal is treated as zero-error space, then the
        # remaining distance is normalized by the arm workspace diameter.
        reward = (
            1.0 + consecutive_bonus + click_margin_bonus - correct_distance_penalty
            if correct
            else -(
                self.config.click_penalty
                + distance_penalty
                + low_peak_click_penalty
                + peak_regression_penalty
            )
        )
        global_learning_reward = reward
        if learn and self.config.click_only and not correct:
            # No-click is not an action to punish globally: use the targeted
            # click teacher below so low output is pushed upward instead.
            global_learning_reward = 0.0
        if learn:
            learning_started = time.perf_counter() if timings is not None else 0.0
            learning_signal = self._learning_signal_for_trial(
                reward=float(global_learning_reward),
                correct=bool(correct),
                current_deficit=current_deficit,
                click_gain=homeostatic_gains.get("motor/click", 1.0),
                success_margin_directional_error=success_margin_directional_error,
            )
            reward_credit = (
                self.plasticity_controller.build_reward_credit(
                    self.brain, learning_signal, self.output_context,
                )
                if global_learning_reward > 0.0 and learning_signal.positive_reinforcements()
                else None
            )
            teacher_normalizer_followup: dict[str, object] = {}
            directional_update: dict[str, object] = {"edge_updates": 0, "channel_updates": {}}
            generic_update = None
            structural_need: dict[str, object] = {}
            route_health: dict[str, object] = {}
            counterfactual_route_health: dict[str, object] = {}
            teacher_stats: dict[str, object] = {
                "edge_updates": 0,
                "mean_delta": 0.0,
                "deficit": 0.0,
                "regression_strength": 0.0,
            }
            teacher_seconds = 0.0

            def apply_teacher_before_normalization(state) -> dict[str, object]:
                nonlocal teacher_stats, teacher_seconds, directional_update, generic_update, structural_need, route_health, counterfactual_route_health
                if self.config.route_health_diagnostic:
                    counterfactual_probe = LearningSignal(
                        reward=0.0,
                        directional_error={"motor/click": 1.0},
                        success=None,
                    )
                    counterfactual_route_health = {
                        name: asdict(health)
                        for name, health in self.plasticity_controller.diagnose_route_health(
                            self.brain,
                            counterfactual_probe,
                            self.output_context,
                            standardized=True,
                        ).items()
                    }
                if not learning_signal.success and learning_signal.nonzero_directions():
                    route_health = {
                        name: asdict(health)
                        for name, health in self.plasticity_controller.diagnose_route_health(
                            self.brain, learning_signal, self.output_context
                        ).items()
                    }
                generic = self.plasticity_controller.apply_learning_signal(
                    self.brain, learning_signal, self.output_context
                )
                generic_update = generic
                directional_update = {
                    "edge_updates": generic.edge_updates,
                    "channel_updates": generic.channel_updates,
                    "mean_abs_delta": generic.mean_abs_delta,
                    "sum_abs_delta": generic.sum_abs_delta,
                    "hop_counts": generic.hop_counts,
                    "excitatory_updates": generic.excitatory_updates,
                    "inhibitory_updates": generic.inhibitory_updates,
                    "consolidated_edges": generic.consolidated_edges,
                    "unique_edge_updates": generic.unique_edge_updates,
                    "reinforced_channels": list(generic.reinforced_channels),
                    "ambiguous_path_edges_skipped": generic.ambiguous_path_edges_skipped,
                    "legacy_rescue_used": False,
                }
                structural_need = self.plasticity_budget_adaptation.observe_directional_failure(
                    brain=self.brain,
                    signal=learning_signal,
                    output_context=self.output_context,
                    directional_update=generic,
                    legacy_rescue_used=False,
                )
                if not self.config.click_only:
                    return teacher_stats
                # Legacy label-route teacher is rescue only: use it when the
                # generic output-direction path had no eligible edge.
                if generic.edge_updates > 0 or learning_signal.success or not learning_signal.nonzero_directions():
                    return teacher_stats
                teacher_started = time.perf_counter() if timings is not None else 0.0
                directional_update["legacy_rescue_used"] = True
                teacher_stats = self._apply_low_peak_click_teacher(
                    label,
                    peak_click_rate_hz,
                    teacher_strength,
                )
                directional_update["legacy_rescue_used"] = bool(teacher_stats.get("edge_updates", 0))
                structural_need["legacy_rescue_used"] = directional_update["legacy_rescue_used"]
                if timings is not None:
                    teacher_seconds = time.perf_counter() - teacher_started
                return teacher_stats

            def observe_normalizer(phase: str, state) -> None:
                pending = self._pending_teacher_normalizer
                if pending is None:
                    return
                edges = pending["edges"]
                mean = float(state.multiplier[edges].mean()) if len(edges) else 0.0
                if phase == "before":
                    pending["before_normalizer_mean"] = mean
                elif phase == "after":
                    before = float(pending.get("before_normalizer_mean", mean))
                    after = mean
                    teacher_normalizer_followup.update({
                        "label": pending["label"],
                        "teacher_multiplier_after_previous_update": float(pending["teacher_mean_after"]),
                        "teacher_multiplier_before_next_normalization": before,
                        "teacher_multiplier_after_next_normalization": after,
                        "teacher_gain": float(pending["teacher_gain"]),
                        "normalizer_loss": after - before,
                        # ``after - teacher_after`` is normalization loss, not
                        # total net learning. Keep both quantities explicit.
                        "true_net_gain": after - float(pending["teacher_mean_before"]),
                        "net_gain": after - float(pending["teacher_mean_before"]),
                    })
                    self._pending_teacher_normalizer = None

            learning = self.brain.learn_from_reward(
                reward=global_learning_reward,
                rule=self.reward_rule,
                normalizer=self.normalizer,
                include_plasticity_summary=False,
                profile_timing=self.config.profile_timing,
                post_reward_hook=apply_teacher_before_normalization,
                normalizer_observer=observe_normalizer,
                reward_credit=reward_credit,
            )
            if timings is not None:
                timings["brain_learning_seconds"] = time.perf_counter() - learning_started
            learning["motor_teacher"] = teacher_stats
            learning["learning_signal"] = {
                "reward": learning_signal.reward,
                "directional_error": dict(learning_signal.directional_error),
                "reinforcement": dict(learning_signal.reinforcement),
                "novelty": learning_signal.novelty,
                "surprise": learning_signal.surprise,
            }
            learning["directional_modulation"] = directional_update
            learning["success_margin"] = {
                "directional_edge_updates": int(
                    generic_update.edge_updates
                    if success_margin_directional_error > 0.0
                    else 0
                ),
                "sum_abs_delta": float(
                    generic_update.sum_abs_delta
                    if success_margin_directional_error > 0.0
                    else 0.0
                ),
            }
            success_margin_directional_edge_updates = int(
                generic_update.edge_updates
                if success_margin_directional_error > 0.0
                else 0
            )
            success_margin_sum_abs_delta = float(
                generic_update.sum_abs_delta
                if success_margin_directional_error > 0.0
                else 0.0
            )
            learning["structural_need"] = structural_need
            if counterfactual_route_health:
                learning["counterfactual_route_health"] = counterfactual_route_health
            if route_health:
                for health in route_health.values():
                    health["actual_directional_edge_updates"] = int(generic_update.edge_updates)
                    health["mean_abs_delta"] = float(generic_update.mean_abs_delta)
                    health["sum_abs_delta"] = float(generic_update.sum_abs_delta)
                learning["route_health"] = route_health
            reward_learning = learning.get("learning", {})
            learning["reward_locality"] = {
                "positive_reward": float(max(0.0, global_learning_reward)),
                "eligible_reward_edges_before_localization": int(reward_learning.get("eligible_reward_edges_before_localization", 0)),
                "selected_credit_edges": int(reward_learning.get("selected_credit_edges", reward_learning.get("credited_reward_edges", 0))),
                "credited_reward_edges": int(reward_learning.get("selected_credit_edges", reward_learning.get("credited_reward_edges", 0))),
                "actual_credited_reward_updated_edges": int(reward_learning.get("actual_credited_reward_updated_edges", 0)),
                "actual_reward_updated_edges": int(reward_learning.get("actual_reward_updated_edges", reward_learning.get("edge_updates", 0))),
                "aligned_one_hop_edges": int(reward_learning.get("aligned_one_hop_edges", 0)),
                "aligned_two_hop_edges": int(reward_learning.get("aligned_two_hop_edges", 0)),
                "opposing_path_edges_skipped": int(reward_learning.get("opposing_path_edges_skipped", 0)),
                "ambiguous_path_edges_skipped": int(reward_learning.get("ambiguous_path_edges_skipped", 0)),
                "uncredited_reward_updated_edges": int(reward_learning.get("uncredited_reward_updated_edges", reward_learning.get("uncredited_edges_updated", 0))),
                "unaligned_edges_skipped": int(reward_learning.get("opposing_path_edges_skipped", 0)) + int(reward_learning.get("ambiguous_path_edges_skipped", 0)),
                "uncredited_edges_updated": int(reward_learning.get("uncredited_reward_updated_edges", reward_learning.get("uncredited_edges_updated", 0))),
                "reward_credit_selection_fraction": float(reward_learning.get("reward_credit_selection_fraction", reward_learning.get("reward_credit_fraction", 0.0))),
                "reward_credit_fraction": float(reward_learning.get("reward_credit_selection_fraction", reward_learning.get("reward_credit_fraction", 0.0))),
                "reward_update_fraction": float(reward_learning.get("reward_update_fraction", 0.0)),
                "mean_reward_credit_weight": float(reward_learning.get("mean_reward_credit_weight", 0.0)),
            }
            learning["homeostasis"] = {"channel_rates": {name: float(final_rates[index]) for index, name in enumerate(channel_names)}, "gains": homeostatic_gains, "ema": dict(self.channel_homeostasis.ema)}
            learning["teacher_normalizer_followup"] = teacher_normalizer_followup
            learning["active_route"] = {
                "candidate_teacher_edges": int(teacher_stats.get("candidate_edge_count", 0)),
                "plastic_teacher_edges": int(teacher_stats.get("plastic_edge_count", 0)),
                "active_eligible_teacher_edges": int(teacher_stats.get("active_eligible_edge_count", 0)),
                "active_teacher_presynaptic_neurons": int(teacher_stats.get("active_presynaptic_count", 0)),
                "teacher_edges_usage_nonzero": int(teacher_stats.get("usage_nonzero_edge_count", 0)),
                "total_teacher_eligibility": float(teacher_stats.get("total_eligibility", 0.0)),
                "total_teacher_credit": float(teacher_stats.get("total_credit", 0.0)),
                "effective_teacher_strength": float(teacher_stats.get("effective_strength_after", teacher_stats.get("effective_strength_before", 0.0))),
                "threshold_margin_hz": float(peak_click_rate_hz - self.config.click_gate_threshold_hz),
                "active_teacher_fraction": float(teacher_stats.get("active_eligible_edge_count", 0)) / max(1, int(teacher_stats.get("plastic_edge_count", 0))),
                "status": (
                    "NO_ANATOMICAL_ROUTE" if int(teacher_stats.get("candidate_edge_count", 0)) == 0 else
                    "NO_ACTIVE_PLASTIC_ROUTE" if int(teacher_stats.get("plastic_edge_count", 0)) > 0 and int(teacher_stats.get("active_eligible_edge_count", 0)) == 0 and int(teacher_stats.get("upstream_fallback_edge_count", 0)) == 0 else
                    "SATURATED_BUT_WEAK" if float(teacher_stats.get("mean_multiplier_before", 0.0)) >= self.brain.plasticity.config.max_multiplier - 1e-3 and peak_click_rate_hz < self.config.click_gate_threshold_hz else
                    "WEAK_ACTIVE" if peak_click_rate_hz < self.config.click_gate_threshold_hz else "HEALTHY"
                ),
            }
            # Successful consolidation is generic above.  Legacy label memory
            # remains a failed-route rescue and is not reinforced on success.
            learning["teacher_confirmation"] = {
                "confirmed": bool(correct), "edge_updates": 0, "mean_stability": 0.0,
            }
            if timings is not None:
                timings["motor_teacher_seconds"] = teacher_seconds
        else:
            reward = 0.0
            success_margin_directional_edge_updates = 0
            success_margin_sum_abs_delta = 0.0
            learning = {
                "learning": {
                    "edge_updates": 0,
                    "evaluation_only": True,
                }
            }
        if isinstance(body, dict) and not body.get("targets"):
            body = {
                **body,
                "targets": [{
                    "id": target_key.label,
                    "x": target_key.x,
                    "y": target_key.y,
                    "radius": target_key.click_radius,
                    "owner_arm": target_key.owner_arm,
                    "hitbox_width": target_key.width,
                    "hitbox_height": target_key.height,
                    "hitbox_shape": "box",
                }],
            }
        result = {
            "label": label,
            "correct": correct,
            "clicked_label": last.clicked_label if last is not None else None,
            "clicked_arm": last.clicked_arm if last is not None else None,
            "body": body,
            "reward": reward,
            "global_learning_reward": global_learning_reward,
            "windows": windows,
            "peak_motor_rate_hz": peak_motor_rate_hz,
            "peak_click_rate_hz": peak_click_rate_hz,
            "peak_click_evidence_hz": peak_click_evidence_hz,
            "click_attempted_arms": sorted(click_attempted_arms),
            "click_population_spike_count": click_output_spike_count,
            "active_click_output_neuron_count": int(len(active_click_outputs)),
            "click_threshold_hz": self.task.motor.config.click_threshold_hz,
            "min_target_distance": min_target_distance,
            "distance_ratio": distance_ratio,
            "distance_penalty": distance_penalty,
            "correct_distance_ratio": correct_distance_ratio,
            "correct_distance_penalty": correct_distance_penalty,
            "max_target_distance": max_target_distance,
            "movement": movement,
            "correct_streak": current_correct_streak,
            "consecutive_bonus": consecutive_bonus,
            "click_margin_bonus": click_margin_bonus,
            "success_margin_target_hz": float(self.config.click_margin_target_hz),
            "success_margin_deficit": float(success_margin_deficit),
            "success_margin_directional_error": float(success_margin_directional_error),
            "success_margin_directional_edge_updates": int(success_margin_directional_edge_updates),
            "success_margin_sum_abs_delta": float(success_margin_sum_abs_delta),
            "low_peak_click_penalty": low_peak_click_penalty,
            "previous_peak_click_rate_hz": previous_peak,
            "peak_regression_ratio": peak_regression_ratio,
            "peak_regression_penalty": peak_regression_penalty,
            "peak_regression_triggered": meaningful_regression,
            "peak_regression_streak": regression_streak,
            "peak_regression_strength": regression_strength,
            "peak_ema_hz": peak_ema,
            "deficit_ema": deficit_ema,
            "subthreshold_streak": subthreshold_streak,
            "persistence_strength": persistence_strength,
            "elapsed_seconds": time.perf_counter() - started,
            "learned": bool(learn),
            "learning": learning,
        }
        if temporal_engagement is not None:
            result["temporal_engagement"] = temporal_engagement
        if timings is not None:
            timings["neural_step_seconds"] = neural_step_seconds
            timings["motor_count_seconds"] = motor_count_seconds
            timings["total_trial_seconds"] = time.perf_counter() - started
            result["timing"] = timings
        return result


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
    resume_info = None
    if config.resume and checkpoint_path.is_file():
        resume_info = restore_learning_checkpoint(
            checkpoint_path,
            brain=session.brain,
            need_tracker=session.plasticity_budget_adaptation.need_tracker,
        )
    _finish_live_status()
    print(
        "keyboard matching: ready "
        f"mode={config.body_mode} tokens={len(KEY_LABELS)} "
        f"motor_channels={session.motor_channel_count} "
        f"route_outputs={len(session.output.body_ids)}",
        flush=True,
    )
    if resume_info is not None:
        print(
            "keyboard matching: restored learned synapses "
            f"changed_edges={resume_info['changed_edge_count']:,} "
            f"from_trial={resume_info['completed_trials']:,}",
            flush=True,
        )
    keyboard_layout = session.task.keyboard.layout()
    evaluation_total = len(KEY_LABELS) * config.click_window_size
    initial_payload = {
        "experiment": "malecns_virtual_keyboard_matching_v1",
        "config": asdict(config),
        "resume": resume_info,
        "completed_trials": 0,
        "training_trials": 0,
        "evaluation_trials": 0,
        "phase": "training",
        "accuracy": 0.0,
        "target_accuracy": config.target_accuracy,
        "min_trials_per_key": config.min_trials_per_key,
        "mastery_reached": False,
        "click_window_size": config.click_window_size,
        "click_window": [],
        "click_window_reached": False,
        "key_accuracy_reached": False,
        "evaluation_ready": False,
        "evaluation_correct": 0,
        "evaluation_total": evaluation_total,
        "evaluation_accuracy": 0.0,
        "evaluation_passed": False,
        "evaluation_number": 0,
        "evaluation_failures": 0,
        "per_key": {
            label: {
                "trials": 0,
                "correct": 0,
                "accuracy": 0.0,
                "recent_attempts": 0,
                "recent_correct": 0,
                "recent_accuracy": 0.0,
                "recent_clicks": [],
                "mastered": False,
            }
            for label in KEY_LABELS
        },
        "token_labels": list(KEY_LABELS),
        "keyboard_layout": keyboard_layout,
        "body": session.task.observation()["body"],
        "recent_trials": [],
    }
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_keyboard_html(initial_payload), encoding="utf-8")
    rows = deque(maxlen=16)
    timing_totals: dict[str, float] = {}
    timing_samples = 0
    last_dashboard_write = time.monotonic()
    correct = 0
    training_correct = 0
    evaluation_correct = 0
    training_trials = 0
    evaluation_trials = 0
    started = time.perf_counter()
    np = session.np
    plastic_budget_start = session.brain.plasticity.plastic_edge_count
    legacy_rescue_events = 0
    directional_generic_update_count = 0
    localized_positive_reward_updated_edge_count = 0
    successful_trials_below_margin_target = 0
    successful_trials_at_or_above_margin_target = 0
    margin_update_trial_count = 0
    margin_directional_edge_update_count = 0
    margin_directional_sum_abs_delta = 0.0
    success_peak_click_rate_sum = 0.0
    success_peak_click_rate_count = 0
    failure_peak_click_rate_sum = 0.0
    failure_peak_click_rate_count = 0
    success_synchrony_sum = 0.0
    success_unique_click_neurons_sum = 0.0
    success_temporal_count = 0
    failure_synchrony_sum = 0.0
    failure_unique_click_neurons_sum = 0.0
    failure_temporal_count = 0
    route_health_by_key = {
        label: {
            "attempts": 0,
            "failures": 0,
            "zero_update_failures": 0,
            "status_counts": {},
            "sums": {
                "active_plastic_edges": 0.0,
                "total_eligibility": 0.0,
                "total_route_credit": 0.0,
                "estimated_available_adjustment": 0.0,
                "useful_frozen_edges": 0.0,
                "estimated_frozen_capacity": 0.0,
                "frozen_to_plastic_capacity_ratio": 0.0,
                "saturated_fraction": 0.0,
                "sum_abs_delta": 0.0,
                "active_plastic_candidate_edges": 0.0,
                "active_frozen_candidate_edges": 0.0,
                "useful_plastic_edges": 0.0,
                "plastic_structural_opportunity": 0.0,
                "frozen_structural_opportunity": 0.0,
                "frozen_to_plastic_structural_opportunity_ratio": 0.0,
                "realized_plastic_capacity": 0.0,
                "plastic_engagement_efficiency": 0.0,
                "plastic_multiplier_headroom": 0.0,
                "frozen_multiplier_headroom": 0.0,
                "useful_plastic_route_fraction": 0.0,
                "useful_frozen_route_fraction": 0.0,
                "mean_effective_route_influence": 0.0,
                "frozen_raw_candidate_edges": 0.0,
                "frozen_unique_candidate_edges": 0.0,
                "frozen_direct_candidate_edges": 0.0,
                "frozen_two_hop_candidate_edges": 0.0,
            },
        }
        for label in KEY_LABELS
    }
    counterfactual_metric_names = (
        "active_plastic_candidate_edges",
        "active_frozen_candidate_edges",
        "useful_plastic_edges",
        "useful_frozen_edges",
        "plastic_structural_opportunity",
        "frozen_structural_opportunity",
        "frozen_to_plastic_structural_opportunity_ratio",
        "realized_plastic_capacity",
        "plastic_engagement_efficiency",
        "saturated_fraction",
        "plastic_multiplier_headroom",
        "frozen_multiplier_headroom",
        "useful_plastic_route_fraction",
        "useful_frozen_route_fraction",
        "mean_effective_route_influence",
        "frozen_raw_candidate_edges",
        "frozen_unique_candidate_edges",
        "frozen_direct_candidate_edges",
        "frozen_two_hop_candidate_edges",
    )
    counterfactual_route_health_by_key = {
        label: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "sums": {name: 0.0 for name in counterfactual_metric_names},
            "success_sums": {name: 0.0 for name in counterfactual_metric_names},
            "failure_sums": {name: 0.0 for name in counterfactual_metric_names},
            "status_counts": {},
        }
        for label in KEY_LABELS
    }
    temporal_engagement_trials: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed + 83_001)
    per_key_trials = {label: 0 for label in KEY_LABELS}
    per_key_correct = {label: 0 for label in KEY_LABELS}
    per_key_recent = {label: [] for label in KEY_LABELS}
    retraining_trials = {label: 0 for label in KEY_LABELS}
    last_evaluation_per_key = {
        label: {"attempts": 0, "correct": 0, "accuracy": 0.0}
        for label in KEY_LABELS
    }
    evaluation_per_key = {
        label: {"attempts": 0, "correct": 0, "accuracy": 0.0}
        for label in KEY_LABELS
    }
    evaluation_schedule: tuple[str, ...] = ()
    evaluation_index = 0
    evaluation_number = 0
    evaluation_failures = 0
    phase = "training"
    evaluation_ready = False
    evaluation_passed = False
    robust_mastery_passed = False
    retention_schedule: tuple[str, ...] = ()
    retention_index = 0
    retention_per_key = {label: {"attempts": 0, "correct": 0, "accuracy": 0.0} for label in KEY_LABELS}
    retention_degraded: set[str] = set()
    next_retention_probe = config.retention_probe_interval
    evaluation_score = 0
    evaluation_accuracy = 0.0
    click_window: list[bool] = []
    mastery_reached = False
    click_window_reached = False
    key_accuracy_reached = False
    completed = 0
    coverage_cycle = build_balanced_coverage_cycle(KEY_LABELS, rng)
    coverage_cursor = 0
    if resume_info is not None and resume_info.get("session_state"):
        saved = resume_info["session_state"]
        completed = int(saved.get("completed", completed))
        correct = int(saved.get("correct", correct))
        training_correct = int(saved.get("training_correct", training_correct))
        training_trials = int(saved.get("training_trials", training_trials))
        evaluation_trials = int(saved.get("evaluation_trials", evaluation_trials))
        phase = str(saved.get("phase", phase))
        evaluation_index = int(saved.get("evaluation_index", evaluation_index))
        evaluation_number = int(saved.get("evaluation_number", evaluation_number))
        evaluation_score = int(saved.get("evaluation_score", evaluation_score))
        evaluation_correct = int(saved.get("evaluation_correct", evaluation_correct))
        evaluation_failures = int(saved.get("evaluation_failures", evaluation_failures))
        evaluation_schedule = tuple(saved.get("evaluation_schedule", evaluation_schedule))
        coverage_cycle = tuple(saved.get("coverage_cycle", coverage_cycle))
        coverage_cursor = int(saved.get("coverage_cursor", coverage_cursor))
        for name, target in (("per_key_trials", per_key_trials), ("per_key_correct", per_key_correct), ("retraining_trials", retraining_trials)):
            target.update({key: int(value) for key, value in saved.get(name, {}).items()})
        for key, values in saved.get("per_key_recent", {}).items():
            per_key_recent[key] = [bool(value) for value in values]
        click_window = [bool(value) for value in saved.get("click_window", click_window)]
        if "rng_state" in saved:
            rng.bit_generator.state = saved["rng_state"]
        session.correct_streak_by_label.update(saved.get("correct_streak_by_label", {}))
        session.previous_peak_by_label.update(saved.get("previous_peak_by_label", {}))
        session.peak_regression_streak_by_label.update(saved.get("peak_regression_streak_by_label", {}))
        session.peak_ema_by_label.update(saved.get("peak_ema_by_label", {}))
        session.deficit_ema_by_label.update(saved.get("deficit_ema_by_label", {}))
        session.subthreshold_streak_by_label.update(saved.get("subthreshold_streak_by_label", {}))
        session.channel_homeostasis.ema.update(saved.get("channel_homeostasis_ema", {}))
        for label, edges in saved.get("teacher_memory_edges_by_label", {}).items():
            if label in session.teacher_memory_edges_by_label:
                session.teacher_memory_edges_by_label[label] = np.asarray(edges, dtype=np.int32)
        print(f"keyboard matching: restored curriculum state at trial={completed:,}", flush=True)
    per_key = summarize_per_key_stats(
        KEY_LABELS,
        per_key_trials,
        per_key_correct,
        per_key_recent,
        window_size=config.click_window_size,
        click_only=config.click_only,
        min_trials=config.min_trials_per_key,
        target_accuracy=config.target_accuracy,
    )

    while config.trials <= 0 or completed < config.trials:
        if (
            config.click_only and phase == "training" and training_trials > 0
            and training_trials >= next_retention_probe
        ):
            phase = "retention"
            retention_index = 0
            retention_per_key = {label: {"attempts": 0, "correct": 0, "accuracy": 0.0} for label in KEY_LABELS}
            next_retention_probe += config.retention_probe_interval
            retention_schedule = build_evaluation_schedule(
                KEY_LABELS, config.retention_trials_per_key, rng
            )
            print(
                "retention probe started: balanced frozen "
                f"{len(retention_schedule)}-trial check ({config.retention_trials_per_key}/key)",
                flush=True,
            )
            continue
        if config.click_only and phase == "training":
            recent_accuracies = {
                label: sum(per_key_recent[label][-config.click_window_size:])
                / max(1, len(per_key_recent[label][-config.click_window_size:]))
                for label in KEY_LABELS
            }
            window_filled = min(per_key_trials.values()) >= config.click_window_size
            retraining_ready = (
                evaluation_number == 0
                or min(retraining_trials.values()) >= config.min_trials_per_key
            )
            accuracy_ready = all(
                recent_key_mastered(
                    per_key_recent[label],
                    config.click_window_size,
                )
                for label in KEY_LABELS
            )
            if window_filled and retraining_ready and accuracy_ready:
                phase = "evaluation"
                evaluation_number += 1
                evaluation_schedule = build_evaluation_schedule(
                    KEY_LABELS,
                    config.click_window_size,
                    rng,
                )
                evaluation_index = 0
                evaluation_score = 0
                evaluation_per_key = {
                    label: {"attempts": 0, "correct": 0, "accuracy": 0.0}
                    for label in KEY_LABELS
                }
                print(
                    f"evaluation {evaluation_number} started: "
                    f"{evaluation_total} randomized trials, learning frozen",
                    flush=True,
                )
                continue

            if not window_filled:
                minimum_trials = min(per_key_trials.values())
                candidates = [
                    label for label in KEY_LABELS
                    if per_key_trials[label] == minimum_trials
                ]
                selection_source = "warmup_coverage"
            elif evaluation_number > 0 and not retraining_ready:
                minimum_round = min(retraining_trials.values())
                candidates = [
                    label for label in KEY_LABELS
                    if retraining_trials[label] == minimum_round
                ]
                selection_source = "retraining_coverage"
            else:
                priority_accuracy = {
                    label: min(
                        recent_accuracies[label],
                        float(last_evaluation_per_key[label]["accuracy"]),
                    )
                    if evaluation_number > 0 and last_evaluation_per_key[label]["attempts"]
                    else recent_accuracies[label]
                    for label in KEY_LABELS
                }
                # A deterministic all-key cycle prevents a chronic weak key
                # from monopolizing training. Between coverage slots, sample
                # by error weight rather than a hard minimum so weak keys are
                # emphasized without starving every other key.
                if training_trials % config.coverage_interval == 0:
                    if coverage_cursor == len(coverage_cycle):
                        coverage_cycle = build_balanced_coverage_cycle(KEY_LABELS, rng)
                        coverage_cursor = 0
                    label = coverage_cycle[coverage_cursor]
                    coverage_cursor += 1
                    selection_source = "balanced_coverage"
                else:
                    scores = np.asarray(
                        [
                            config.hard_mining_floor
                            + (1.0 - priority_accuracy[key]) ** config.hard_mining_power
                            + (config.retention_replay_bonus if key in retention_degraded else 0.0)
                            for key in KEY_LABELS
                        ],
                        dtype=np.float64,
                    )
                    probabilities = scores / scores.sum()
                    label = str(KEY_LABELS[int(rng.choice(len(KEY_LABELS), p=probabilities))])
                    selection_source = "hard_mining"
                candidates = None
            if candidates is not None:
                label = str(candidates[int(rng.integers(0, len(candidates)))])
            learn = True
        elif config.click_only and phase == "evaluation":
            label = evaluation_schedule[evaluation_index]
            selection_source = "evaluation"
            learn = False
        elif config.click_only and phase == "retention":
            label = retention_schedule[retention_index]
            selection_source = "retention_probe"
            learn = False
        else:
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
            selection_source = "legacy"
            learn = True

        completed += 1
        trial = completed
        row_phase = phase
        row = session.run_trial(label, learn=learn)
        row["trial"] = trial
        row["phase"] = row_phase
        row["selection_source"] = selection_source
        rows.append(row)
        row_correct = int(row["correct"])
        row_learning = row.get("learning", {})
        peak_click_rate = float(row.get("peak_click_rate_hz", 0.0))
        if row_correct:
            success_peak_click_rate_sum += peak_click_rate
            success_peak_click_rate_count += 1
            if peak_click_rate < config.click_margin_target_hz:
                successful_trials_below_margin_target += 1
            else:
                successful_trials_at_or_above_margin_target += 1
        else:
            failure_peak_click_rate_sum += peak_click_rate
            failure_peak_click_rate_count += 1
        row_margin = row_learning.get("success_margin", {})
        if isinstance(row_margin, dict):
            margin_updates = int(row_margin.get("directional_edge_updates", 0))
            if margin_updates > 0:
                margin_update_trial_count += 1
            margin_directional_edge_update_count += margin_updates
            margin_directional_sum_abs_delta += float(row_margin.get("sum_abs_delta", 0.0))
        temporal = row.get("temporal_engagement")
        if isinstance(temporal, dict):
            if row_correct:
                success_synchrony_sum += float(temporal.get("mean_click_output_synchrony_proxy", 0.0))
                success_unique_click_neurons_sum += float(temporal.get("unique_click_output_neurons_recruited", 0.0))
                success_temporal_count += 1
            else:
                failure_synchrony_sum += float(temporal.get("mean_click_output_synchrony_proxy", 0.0))
                failure_unique_click_neurons_sum += float(temporal.get("unique_click_output_neurons_recruited", 0.0))
                failure_temporal_count += 1
        if config.temporal_engagement_diagnostic and row.get("temporal_engagement") is not None:
            temporal_engagement_trials.append({
                "trial": trial,
                "label": label,
                "correct": row_correct,
                "temporal_engagement": row["temporal_engagement"],
            })
        row_directional = row_learning.get("directional_modulation", {})
        row_legacy_rescue = bool(row_directional.get("legacy_rescue_used", False))
        legacy_rescue_events += int(row_legacy_rescue)
        directional_generic_update_count += int(row_directional.get("edge_updates", 0))
        localized_positive_reward_updated_edge_count += int(
            row_learning.get("reward_locality", {}).get("actual_credited_reward_updated_edges", 0)
        )
        for health in row_learning.get("route_health", {}).values():
            bucket = route_health_by_key[label]
            bucket["attempts"] += 1
            bucket["failures"] += int(row_correct == 0)
            bucket["zero_update_failures"] += int(health.get("actual_directional_edge_updates", 0) == 0)
            status = str(health.get("route_health_status", "UNKNOWN"))
            bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
            for name in bucket["sums"]:
                bucket["sums"][name] += float(health.get(name, 0.0))
        for health in row_learning.get("counterfactual_route_health", {}).values():
            bucket = counterfactual_route_health_by_key[label]
            bucket["attempts"] += 1
            bucket["successes"] += row_correct
            bucket["failures"] += int(not row_correct)
            target_sums = bucket["success_sums"] if row_correct else bucket["failure_sums"]
            for name in counterfactual_metric_names:
                value = float(health.get(name, 0.0))
                bucket["sums"][name] += value
                target_sums[name] += value
            status = str(health.get("route_health_status", "UNKNOWN"))
            bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
        if config.profile_timing:
            timing_samples += 1
            for name, value in row.get("timing", {}).items():
                timing_totals[name] = timing_totals.get(name, 0.0) + float(value)
        correct += row_correct

        if row_phase == "evaluation":
            evaluation_trials += 1
            evaluation_score += row_correct
            stats = evaluation_per_key[label]
            stats["attempts"] += 1
            stats["correct"] += row_correct
            stats["accuracy"] = stats["correct"] / stats["attempts"]
            evaluation_index += 1
            if evaluation_index == evaluation_total:
                evaluation_ready = True
                evaluation_correct = evaluation_score
                evaluation_accuracy = evaluation_score / evaluation_total
                last_evaluation_per_key = {
                    key: dict(stats) for key, stats in evaluation_per_key.items()
                }
                evaluation_passed = evaluation_score == evaluation_total
                evaluation_macro_accuracy = sum(
                    float(stats["accuracy"]) for stats in evaluation_per_key.values()
                ) / len(KEY_LABELS)
                evaluation_min_key_accuracy = min(
                    float(stats["accuracy"]) for stats in evaluation_per_key.values()
                )
                robust_mastery_passed = (
                    evaluation_macro_accuracy >= config.robust_mastery_macro_accuracy
                    and evaluation_min_key_accuracy >= config.robust_mastery_min_key_accuracy
                )
                if evaluation_passed or (config.robust_mastery_enabled and robust_mastery_passed):
                    mastery_reached = True
                    phase = "complete"
                else:
                    evaluation_failures += 1
                    phase = "training"
                    retraining_trials = {key: 0 for key in KEY_LABELS}
                    print(
                        f"evaluation {evaluation_number} failed: "
                        f"{evaluation_score}/{evaluation_total}; returning to training",
                        flush=True,
                    )
        elif row_phase == "retention":
            stats = retention_per_key[label]
            stats["attempts"] += 1
            stats["correct"] += row_correct
            stats["accuracy"] = stats["correct"] / stats["attempts"]
            retention_index += 1
            if retention_index == len(retention_schedule):
                retention_degraded = {
                    key for key, item in retention_per_key.items()
                    if float(item["accuracy"]) < config.retention_degraded_accuracy
                }
                phase = "training"
                print(
                    f"retention probe complete: degraded={len(retention_degraded)}/{len(KEY_LABELS)}",
                    flush=True,
                )
        else:
            training_trials += 1
            training_correct += row_correct
            per_key_trials[label] += 1
            per_key_correct[label] += row_correct
            retraining_trials[label] += 1
            per_key_recent[label].append(bool(row_correct))
            if len(per_key_recent[label]) > config.click_window_size:
                del per_key_recent[label][:-config.click_window_size]
            click_window.append(bool(row_correct))
            if len(click_window) > config.click_window_size:
                del click_window[:-config.click_window_size]
            click_window_reached = (
                len(click_window) >= config.click_window_size
                and all(click_window)
            )

        per_key = summarize_per_key_stats(
            KEY_LABELS,
            per_key_trials,
            per_key_correct,
            per_key_recent,
            window_size=config.click_window_size,
            click_only=config.click_only,
            min_trials=config.min_trials_per_key,
            target_accuracy=config.target_accuracy,
        )
        mastered_count = sum(int(stats["mastered"]) for stats in per_key.values())
        key_accuracy_reached = (
            evaluation_passed if config.click_only else mastered_count == len(KEY_LABELS)
        )
        update_stats = row["learning"].get("learning", {})
        teacher_stats = row["learning"].get("motor_teacher", {})
        trial_limit = str(config.trials) if config.trials > 0 else "∞"
        phase_label = (
            f"EXAM {evaluation_index}/{evaluation_total}"
            if row_phase == "evaluation"
            else phase.upper()
        )
        _live_status(
            f"keyboard_matching trial={trial:>6}/{trial_limit:<6} phase={phase_label:<14} "
            f"key={label:<2} clicked={str(row['clicked_label'] or '-'): <2} "
            f"correct={row_correct} learn={str(learn):<5} "
            f"peak={float(row.get('peak_click_rate_hz', 0.0)):>5.2f}Hz "
            f"target_dist={float(row.get('min_target_distance', 0.0)):.3f} "
            f"distance_penalty={float(row.get('distance_penalty', 0.0)):.2f} "
            f"low_peak_penalty={float(row.get('low_peak_click_penalty', 0.0)):.2f} "
            f"peak_regression={float(row.get('peak_regression_penalty', 0.0)):.2f} "
            f"correct_edge={float(row.get('correct_distance_penalty', 0.0)):.2f} "
            f"key_streak={int(row.get('correct_streak', 0)):>2} "
            f"bonus={float(row.get('consecutive_bonus', 0.0)):+.2f} "
            f"updates={int(update_stats.get('edge_updates', 0)):>6} "
            f"click_teacher={int(teacher_stats.get('edge_updates', 0)):>5} "
            f"train={training_trials:>5} exam={evaluation_correct:>4}/{evaluation_total:<4} "
            f"lowkey={label:<2} accuracy={correct / max(1, completed):.3f} "
            f"elapsed={time.perf_counter() - started:6.1f}s"
        )
        payload = {
            "experiment": "malecns_virtual_keyboard_matching_v1",
            "config": asdict(config),
            "resume": resume_info,
            "completed_trials": trial,
            "training_trials": training_trials,
            "evaluation_trials": evaluation_trials,
            "phase": phase,
            "accuracy": correct / max(1, trial),
            "training_accuracy": training_correct / max(1, training_trials),
            "target_accuracy": config.target_accuracy,
            "min_trials_per_key": config.min_trials_per_key,
            "mastery_reached": mastery_reached,
            "click_window_size": config.click_window_size,
            "click_window": list(click_window),
            "click_window_reached": click_window_reached,
            "key_accuracy_reached": key_accuracy_reached,
            "evaluation_ready": evaluation_ready,
            "evaluation_correct": evaluation_correct,
            "evaluation_total": evaluation_total,
            "evaluation_accuracy": evaluation_accuracy,
            "evaluation_passed": evaluation_passed,
            "perfect_1200_passed": evaluation_passed,
            "robust_mastery_passed": robust_mastery_passed,
            "mastery_passed": mastery_reached,
            "evaluation_number": evaluation_number,
            "evaluation_failures": evaluation_failures,
            "evaluation_per_key": evaluation_per_key,
            "retention_per_key": retention_per_key,
            "retention_degraded_keys": sorted(retention_degraded),
            "retention_accuracy": sum(float(item["accuracy"]) for item in retention_per_key.values()) / len(KEY_LABELS),
            "macro_recent_accuracy": sum(float(item["recent_accuracy"]) for item in per_key.values()) / len(KEY_LABELS),
            "median_recent_accuracy": float(np.median([float(item["recent_accuracy"]) for item in per_key.values()])),
            "minimum_recent_accuracy": min(float(item["recent_accuracy"]) for item in per_key.values()),
            "per_key": per_key,
            "route_provenance": session.route_provenance,
            "motor_channel_count": session.motor_channel_count,
            "token_labels": list(KEY_LABELS),
            "keyboard_layout": keyboard_layout,
            "body": row.get("body", {}),
            "recent_trials": list(rows),
        }
        checkpoint_due = trial % config.checkpoint_every == 0 or trial == config.trials or mastery_reached
        dashboard_due = time.monotonic() - last_dashboard_write >= config.dashboard_update_interval_seconds
        if dashboard_due or checkpoint_due:
            dashboard_started = time.perf_counter() if config.profile_timing else 0.0
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(build_keyboard_html(payload), encoding="utf-8")
            last_dashboard_write = time.monotonic()
            if config.profile_timing:
                timing_totals["dashboard_write_seconds"] = timing_totals.get("dashboard_write_seconds", 0.0) + (time.perf_counter() - dashboard_started)
        if checkpoint_due:
            checkpoint_started = time.perf_counter() if config.profile_timing else 0.0
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            save_learning_checkpoint(
                checkpoint_path,
                brain=session.brain,
                config=config,
                completed_trials=trial,
                stage="keyboard_matching",
                session_state={
                    "completed": completed, "correct": correct,
                    "training_correct": training_correct, "training_trials": training_trials,
                    "evaluation_trials": evaluation_trials, "phase": phase,
                    "evaluation_index": evaluation_index, "evaluation_number": evaluation_number,
                    "evaluation_score": evaluation_score, "evaluation_correct": evaluation_correct,
                    "evaluation_failures": evaluation_failures,
                    "evaluation_schedule": list(evaluation_schedule),
                    "coverage_cycle": list(coverage_cycle), "coverage_cursor": coverage_cursor,
                    "per_key_trials": per_key_trials, "per_key_correct": per_key_correct,
                    "per_key_recent": per_key_recent, "retraining_trials": retraining_trials,
                    "click_window": click_window, "rng_state": rng.bit_generator.state,
                    "correct_streak_by_label": session.correct_streak_by_label,
                    "previous_peak_by_label": session.previous_peak_by_label,
                    "peak_regression_streak_by_label": session.peak_regression_streak_by_label,
                    "peak_ema_by_label": session.peak_ema_by_label,
                    "deficit_ema_by_label": session.deficit_ema_by_label,
                    "subthreshold_streak_by_label": session.subthreshold_streak_by_label,
                    "teacher_memory_edges_by_label": {label: edges.tolist() for label, edges in session.teacher_memory_edges_by_label.items()},
                    "channel_homeostasis_ema": session.channel_homeostasis.ema,
                },
                need_tracker=session.plasticity_budget_adaptation.need_tracker,
            )
            if config.profile_timing:
                timing_totals["checkpoint_seconds"] = timing_totals.get("checkpoint_seconds", 0.0) + (time.perf_counter() - checkpoint_started)
            if sys.stdout.isatty():
                _finish_live_status()
                print(
                    f"checkpoint trial={trial} accuracy={correct / trial:.3f} "
                    f"path={checkpoint_path}",
                    flush=True,
                )
        if mastery_reached:
            _finish_live_status()
            if config.click_only:
                print(
                    f"evaluation passed: {evaluation_correct}/{evaluation_total} "
                    f"(100%) after {trial} total trials; next_stage=arm_navigation",
                    flush=True,
                )
            else:
                print(
                    f"keyboard mastery reached: all {len(KEY_LABELS)} keys >= "
                    f"{config.target_accuracy:.0%} after {trial} trials",
                    flush=True,
                )
            break

    _finish_live_status()

    np = session.np
    route_health_report = {}
    route_health_status_counts: dict[str, int] = {}
    zero_update_failure_count = 0
    route_health_failure_count = 0
    for label, bucket in route_health_by_key.items():
        attempts = int(bucket["attempts"])
        route_health_failure_count += attempts
        zero_update_failure_count += int(bucket["zero_update_failures"])
        for status, count in bucket["status_counts"].items():
            route_health_status_counts[status] = route_health_status_counts.get(status, 0) + int(count)
        route_health_report[label] = {
            "attempts": attempts,
            "failures": int(bucket["failures"]),
            "zero_update_failures": int(bucket["zero_update_failures"]),
            "status_counts": dict(bucket["status_counts"]),
            **{
                f"mean_{name}": total / max(1, attempts)
                for name, total in bucket["sums"].items()
            },
        }
    counterfactual_route_health_report = {}
    counterfactual_trial_count = 0
    counterfactual_success_count = 0
    counterfactual_failure_count = 0
    for label, bucket in counterfactual_route_health_by_key.items():
        attempts = int(bucket["attempts"])
        successes = int(bucket["successes"])
        failures = int(bucket["failures"])
        counterfactual_trial_count += attempts
        counterfactual_success_count += successes
        counterfactual_failure_count += failures

        def means(values, count):
            return {
                f"mean_{name}": float(total) / max(1, count)
                for name, total in values.items()
            }

        counterfactual_route_health_report[label] = {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "status_counts": dict(bucket["status_counts"]),
            **means(bucket["sums"], attempts),
            "success": {
                "attempts": successes,
                **means(bucket["success_sums"], successes),
            },
            "failure": {
                "attempts": failures,
                **means(bucket["failure_sums"], failures),
            },
        }
    report = {
        "experiment": "malecns_virtual_keyboard_matching_v1",
        "purpose": "learn click-gated matching for Korean, English, O/X, and digit keys before arm navigation",
        "config": asdict(config),
        "resume": resume_info,
        "connectome": connectome.summary(),
        "token_labels": list(KEY_LABELS),
        "keyboard_layout": keyboard_layout,
        "body": rows[-1].get("body", {}) if rows else session.task.observation()["body"],
        "token_encoder": {
            "type": "virtual_visual_projection_groups",
            "background_count": len(session.encoder.background_body_ids),
            "token_group_size": config.token_neurons,
        },
        "route_provenance": session.route_provenance,
        "motor_channel_count": session.motor_channel_count,
        "body_mode": config.body_mode,
        "completed_trials": completed,
        "training_trials": training_trials,
        "evaluation_trials": evaluation_trials,
        "phase": phase,
        "accuracy": correct / max(1, completed),
        "training_accuracy": training_correct / max(1, training_trials),
        "target_accuracy": config.target_accuracy,
        "min_trials_per_key": config.min_trials_per_key,
        "mastery_reached": mastery_reached,
        "click_window_size": config.click_window_size,
        "click_window": list(click_window),
        "click_window_reached": click_window_reached,
        "key_accuracy_reached": key_accuracy_reached,
        "evaluation_ready": evaluation_ready,
        "evaluation_correct": evaluation_correct,
        "evaluation_total": evaluation_total,
        "evaluation_accuracy": evaluation_accuracy,
        "evaluation_passed": evaluation_passed,
        "perfect_1200_passed": evaluation_passed,
        "robust_mastery_passed": robust_mastery_passed,
        "mastery_passed": mastery_reached,
        "evaluation_number": evaluation_number,
        "evaluation_failures": evaluation_failures,
        "evaluation_per_key": evaluation_per_key,
        "retention_per_key": retention_per_key,
        "retention_accuracy": sum(float(item["accuracy"]) for item in retention_per_key.values()) / len(KEY_LABELS),
        "macro_recent_accuracy": sum(float(item["recent_accuracy"]) for item in per_key.values()) / len(KEY_LABELS),
        "median_recent_accuracy": float(np.median([float(item["recent_accuracy"]) for item in per_key.values()])),
        "minimum_recent_accuracy": min(float(item["recent_accuracy"]) for item in per_key.values()),
        "curriculum_stage": config.curriculum_stage,
        "next_stage": "arm_navigation" if mastery_reached else config.curriculum_stage,
        "stopped_reason": (
            "evaluation_1200_perfect"
            if config.click_only and mastery_reached
            else "all_keys_mastered"
            if (not config.click_only and mastery_reached)
            else "max_trials_reached"
        ),
        "per_key": per_key,
        "final_plasticity": {
            **session.brain.plasticity.summary(),
            "changed_edges": int(np.count_nonzero(np.abs(session.brain.plasticity.multiplier - 1.0) > 1e-7)),
            "multiplier_saturation_fraction": float(
                np.count_nonzero(
                    session.brain.plasticity.plastic_mask
                    & (
                        session.brain.plasticity.multiplier
                        >= session.brain.plasticity.config.max_multiplier - 1e-6
                    )
                ) / max(1, session.brain.plasticity.plastic_edge_count)
            ),
        },
        "success_margin_summary": {
            "enabled": bool(config.success_margin_directional_learning),
            "target_hz": float(config.click_margin_target_hz),
            "directional_scale": float(config.success_margin_directional_scale),
            "successful_trials_below_margin_target": int(successful_trials_below_margin_target),
            "successful_trials_at_or_above_margin_target": int(successful_trials_at_or_above_margin_target),
            "margin_update_trial_count": int(margin_update_trial_count),
            "margin_directional_edge_update_count": int(margin_directional_edge_update_count),
            "margin_directional_sum_abs_delta": float(margin_directional_sum_abs_delta),
            "mean_success_peak_click_rate_hz": success_peak_click_rate_sum / max(1, success_peak_click_rate_count),
            "mean_failure_peak_click_rate_hz": failure_peak_click_rate_sum / max(1, failure_peak_click_rate_count),
            "mean_success_click_output_synchrony": success_synchrony_sum / max(1, success_temporal_count),
            "mean_failure_click_output_synchrony": failure_synchrony_sum / max(1, failure_temporal_count),
            "mean_success_unique_click_output_neurons": success_unique_click_neurons_sum / max(1, success_temporal_count),
            "mean_failure_unique_click_output_neurons": failure_unique_click_neurons_sum / max(1, failure_temporal_count),
        },
        "outcome_transition_summary": summarize_recent_outcome_transitions(per_key_recent),
        "adaptive_budget": {
            **session.plasticity_budget_adaptation.telemetry(),
            "adaptive_plastic_budget": bool(config.adaptive_plastic_budget),
            "legacy_rescue_events": int(legacy_rescue_events),
            "directional_generic_update_count": int(directional_generic_update_count),
            "localized_positive_reward_updated_edge_count": int(localized_positive_reward_updated_edge_count),
            "plastic_budget_start": int(plastic_budget_start),
            "plastic_budget_end": int(session.brain.plasticity.plastic_edge_count),
            "budget_delta": int(session.brain.plasticity.plastic_edge_count - plastic_budget_start),
        },
        "route_health_by_key": route_health_report,
        "counterfactual_route_health_by_key": counterfactual_route_health_report,
        "route_health_summary": {
            "failed_directional_trial_count": route_health_failure_count,
            "zero_update_failure_count": zero_update_failure_count,
            "route_health_status_counts": route_health_status_counts,
        },
        "counterfactual_route_health_summary": {
            "trial_count": counterfactual_trial_count,
            "success_count": counterfactual_success_count,
            "failure_count": counterfactual_failure_count,
            "keys_with_samples": sum(
                int(item["attempts"] > 0)
                for item in counterfactual_route_health_report.values()
            ),
        },
        "temporal_engagement_trials": temporal_engagement_trials,
        "execution_profile": {
            "backend": "numpy_cpu",
            "gpu_used": False,
            "external_decoder": False,
            "closed_loop_virtual_body": True,
            "body_mode": config.body_mode,
            "timing_enabled": config.profile_timing,
            "timing_samples": timing_samples,
            "mean_seconds": {
                name: value / max(1, timing_samples)
                for name, value in timing_totals.items()
            } if config.profile_timing else None,
        },
        "checkpoint": str(checkpoint_path),
        "html": str(html_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_keyboard_html({
        **report,
        "recent_trials": list(rows),
    }), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaleCNS to match tokens to a virtual keyboard.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--min-trials-per-key", type=int, default=5)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument("--coverage-interval", type=int, default=4)
    parser.add_argument("--hard-mining-floor", type=float, default=0.10)
    parser.add_argument("--hard-mining-power", type=float, default=2.0)
    parser.add_argument("--click-penalty", type=float, default=1.20)
    parser.add_argument("--low-peak-click-penalty-scale", type=float, default=0.0)
    parser.add_argument("--click-teacher-learning-rate", type=float, default=0.08)
    parser.add_argument("--click-teacher-credit-floor", type=float, default=0.05)
    parser.add_argument("--peak-regression-trigger-hz", type=float, default=1.5)
    parser.add_argument("--peak-regression-fixed-penalty", type=float, default=0.80)
    parser.add_argument("--peak-regression-escalation", type=float, default=0.50)
    parser.add_argument("--peak-regression-max-penalty", type=float, default=2.00)
    parser.add_argument("--click-gate-threshold-hz", type=float, default=9.0)
    parser.add_argument("--click-integration-windows", type=int, default=1)
    parser.add_argument("--click-evidence-windows", type=int, default=5)
    parser.add_argument("--click-margin-target-hz", type=float, default=15.0)
    parser.add_argument("--click-margin-reward-scale", type=float, default=0.50)
    parser.add_argument(
        "--success-margin-directional-learning",
        action="store_true",
        help="Enable conservative generic correction for marginal successful clicks.",
    )
    parser.add_argument("--success-margin-directional-scale", type=float, default=0.25)
    parser.add_argument("--distance-penalty-scale", type=float, default=1.20)
    parser.add_argument("--correct-distance-penalty-scale", type=float, default=0.20)
    parser.add_argument(
        "--allow-click-training-movement",
        action="store_true",
        help="Enable joint movement during the click-only curriculum.",
    )
    parser.add_argument("--consecutive-correct-bonus", type=float, default=0.25)
    parser.add_argument("--max-consecutive-bonus", type=float, default=1.00)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--control-window-ms", type=float, default=20.0)
    parser.add_argument("--max-control-windows", type=int, default=30)
    parser.add_argument("--stimulus-rate-hz", type=float, default=205.0)
    parser.add_argument("--motor-population-size", type=int, default=64)
    parser.add_argument(
        "--adaptive-plastic-budget",
        action="store_true",
        help="Enable conservative Phase D.2 sparse plastic-capacity adaptation.",
    )
    parser.add_argument("--adaptive-budget-max-promotions-per-event", type=int, default=1)
    parser.add_argument(
        "--body-mode",
        choices=("one_arm_fan", "one_arm_circle", "one_arm_circular", "four_arm_grid"),
        default="one_arm_fan",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=("click_gate", "click_accuracy"),
        default="click_accuracy",
    )
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--retention-trials-per-key", type=int, default=5)
    parser.add_argument("--diagnose-routes", action="store_true", help="Write read-only anatomical/active route diagnostics.")
    parser.add_argument("--diagnose-background", action="store_true", help="Include frozen background-only and token-only probes.")
    parser.add_argument("--diagnose-interference", action="store_true", help="Measure one-trial cross-label interference in a restored sandbox.")
    parser.add_argument("--diagnostic-key-count", type=int, default=5)
    parser.add_argument("--diagnostic-full-matrix", action="store_true")
    parser.add_argument("--diagnostic-output", type=Path, default=Path("results/latest_keyboard_route_diagnostics.json"))
    parser.add_argument("--diagnostic-html", type=Path, default=Path("results/latest_keyboard_route_diagnostics.html"))
    parser.add_argument("--dashboard-update-interval-seconds", type=float, default=0.5)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Restore learned synapse state from the checkpoint.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.download:
        download_malecns(args.data_dir)
    config = KeyboardTrainingConfig(
        min_connection_synapses=args.min_syn,
        trials=args.trials,
        min_trials_per_key=args.min_trials_per_key,
        target_accuracy=args.target_accuracy,
        coverage_interval=args.coverage_interval,
        hard_mining_floor=args.hard_mining_floor,
        hard_mining_power=args.hard_mining_power,
        lock_arm_during_click_training=not args.allow_click_training_movement,
        click_penalty=args.click_penalty,
        low_peak_click_penalty_scale=args.low_peak_click_penalty_scale,
        click_teacher_learning_rate=args.click_teacher_learning_rate,
        click_teacher_credit_floor=args.click_teacher_credit_floor,
        peak_regression_trigger_hz=args.peak_regression_trigger_hz,
        peak_regression_fixed_penalty=args.peak_regression_fixed_penalty,
        peak_regression_escalation=args.peak_regression_escalation,
        peak_regression_max_penalty=args.peak_regression_max_penalty,
        click_gate_threshold_hz=args.click_gate_threshold_hz,
        click_integration_windows=args.click_integration_windows,
        click_evidence_windows=args.click_evidence_windows,
        click_margin_target_hz=args.click_margin_target_hz,
        click_margin_reward_scale=args.click_margin_reward_scale,
        success_margin_directional_learning=args.success_margin_directional_learning,
        success_margin_directional_scale=args.success_margin_directional_scale,
        distance_penalty_scale=args.distance_penalty_scale,
        correct_distance_penalty_scale=args.correct_distance_penalty_scale,
        consecutive_correct_bonus=args.consecutive_correct_bonus,
        max_consecutive_bonus=args.max_consecutive_bonus,
        duration_ms=args.duration_ms,
        control_window_ms=args.control_window_ms,
        max_control_windows=args.max_control_windows,
        stimulus_rate_hz=args.stimulus_rate_hz,
        motor_population_size=args.motor_population_size,
        adaptive_plastic_budget=args.adaptive_plastic_budget,
        adaptive_budget_max_promotions_per_event=args.adaptive_budget_max_promotions_per_event,
        body_mode=args.body_mode,
        curriculum_stage=args.curriculum_stage,
        checkpoint_every=args.checkpoint_every,
        retention_trials_per_key=args.retention_trials_per_key,
        dashboard_update_interval_seconds=args.dashboard_update_interval_seconds,
        profile_timing=args.profile_timing,
        resume=args.resume,
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
    if args.diagnose_routes or args.diagnose_background or args.diagnose_interference:
        from .keyboard_diagnostics import build_keyboard_route_diagnostics_html, run_keyboard_route_diagnostics

        session = KeyboardNeuralSession(connectome, config=config)
        if not args.resume:
            raise ValueError("diagnostics require --resume so they inspect the saved learned state")
        restore_learning_checkpoint(args.checkpoint, brain=session.brain)
        progress = json.loads(args.progress.read_text(encoding="utf-8"))
        def write_diagnostic_progress(live_progress: dict[str, object]) -> None:
            partial = live_progress.get("report")
            if not isinstance(partial, dict):
                partial = {
                    "summary": {"state_integrity": "checking"},
                    "weak_labels": [], "strong_labels": [],
                    "weak_vs_strong": {"weak": [], "strong": []},
                    "background_token_probes": {},
                    "retention_analysis": {}, "path_overlap_matrix": {},
                }
            partial = {**partial, "live_progress": live_progress}
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_output.write_text(json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            args.diagnostic_html.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_html.write_text(build_keyboard_route_diagnostics_html(partial), encoding="utf-8")
            # Keep the existing virtual-body/keyboard dashboard intact and add
            # one live diagnostic card to it instead of replacing it.
            dashboard_payload = {
                **progress,
                "route_diagnostics_live": live_progress,
            }
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(build_keyboard_html(dashboard_payload), encoding="utf-8")
        report = run_keyboard_route_diagnostics(
            session,
            per_key=progress["per_key"],
            diagnostic_key_count=args.diagnostic_key_count,
            include_background=args.diagnose_background,
            include_interference=args.diagnose_interference,
            full_matrix=args.diagnostic_full_matrix,
            retention_trials_per_key=args.retention_trials_per_key,
            on_progress=write_diagnostic_progress,
        )
        args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        args.diagnostic_html.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_html.write_text(build_keyboard_route_diagnostics_html(report), encoding="utf-8")
        print(json.dumps({"diagnostic_output": str(args.diagnostic_output), "diagnostic_html": str(args.diagnostic_html), "weak_labels": report["weak_labels"], "strong_labels": report["strong_labels"]}, ensure_ascii=False, indent=2))
        return
    report = run_keyboard_training(
        connectome,
        config=config,
        result_path=args.result,
        progress_path=args.progress,
        checkpoint_path=args.checkpoint,
        html_path=args.html,
    )
    print(json.dumps({
        "completed_trials": report["completed_trials"],
        "accuracy": report["accuracy"],
        "route_provenance": report["route_provenance"],
        "changed_edges": report["final_plasticity"]["changed_edges"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
