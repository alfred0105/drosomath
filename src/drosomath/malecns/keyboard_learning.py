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
    click_penalty: float = 1.20
    # Click-only failures are corrected by motor-targeted teaching plasticity;
    # do not amplify their global negative reward.
    low_peak_click_penalty_scale: float = 0.0
    click_teacher_learning_rate: float = 0.08
    click_teacher_credit_floor: float = 0.05
    peak_regression_penalty_scale: float = 0.80
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
    duration_ms: float = 100.0
    control_window_ms: float = 20.0
    max_control_windows: int = 30
    stimulus_rate_hz: float = 205.0
    plastic_fraction: float = 0.05
    learning_rate: float = 0.02
    budget_strength: float = 0.25
    checkpoint_every: int = 32
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
        if self.peak_regression_penalty_scale < 0.0:
            raise ValueError("peak_regression_penalty_scale must be >= 0")
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
        self.normalizer = OutgoingBudgetNormalizer(strength=config.budget_strength)
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
        regression_ratio: float,
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
        if deficit <= 0.0 or self.config.click_teacher_learning_rate <= 0.0:
            return {"edge_updates": 0, "mean_delta": 0.0, "deficit": deficit}
        state = self.brain.plasticity
        candidates = self.click_teacher_edges_by_label[label]
        credit = state.usage_ema[candidates] * state.eligibility[candidates]
        active = state.plastic_mask[candidates]
        if not active.any():
            return {"edge_updates": 0, "mean_delta": 0.0, "deficit": deficit}
        edges = candidates[active]
        old = state.multiplier[edges].copy()
        local_credit = np.maximum(
            credit[active],
            self.config.click_teacher_credit_floor,
        )
        teaching_strength = 1.0 + regression_ratio
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
        return {
            "edge_updates": int(len(edges)),
            "mean_delta": float(actual.mean()) if len(actual) else 0.0,
            "deficit": deficit,
            "regression_ratio": regression_ratio,
        }

    def run_trial(self, label: str, *, learn: bool = True) -> dict[str, object]:
        np = self.np
        self.brain.reset()
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
        counts = np.zeros(self.motor_channel_count, dtype=np.int32)
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
        for windows in range(1, self.max_control_windows + 1):
            counts.fill(0)
            for _ in range(self.steps_per_window):
                fired, _ = self.brain.step(
                    stimulus_indices=stimulus_indices,
                    stimulus_rate_hz=self.config.stimulus_rate_hz,
                )
                if len(fired):
                    local = self.channel_lookup[fired]
                    local = local[local >= 0]
                    if len(local):
                        counts += np.bincount(
                            local,
                            minlength=self.motor_channel_count,
                        ).astype(np.int32, copy=False)
            rates = counts.astype(np.float32) * self.rate_scale
            peak_motor_rate_hz = max(peak_motor_rate_hz, float(rates.max()))
            click_rates = rates[4::5]
            peak_click_rate_hz = max(peak_click_rate_hz, float(click_rates.max()))
            click_evidence_history.append(float(click_rates.max()))
            del click_evidence_history[:-self.config.click_evidence_windows]
            peak_click_evidence_hz = max(
                peak_click_evidence_hz,
                sum(click_evidence_history) / len(click_evidence_history),
            )
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
        peak_regression_penalty = (
            self.config.peak_regression_penalty_scale * peak_regression_ratio
            if learn and self.config.click_only and not correct
            else 0.0
        )
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
            learning = self.brain.learn_from_reward(
                reward=global_learning_reward,
                rule=self.reward_rule,
                normalizer=self.normalizer,
                include_plasticity_summary=False,
            )
            learning["motor_teacher"] = (
                self._apply_low_peak_click_teacher(
                    label,
                    peak_click_rate_hz,
                    peak_regression_ratio,
                )
                if self.config.click_only and not correct
                else {
                    "edge_updates": 0,
                    "mean_delta": 0.0,
                    "deficit": 0.0,
                    "regression_ratio": 0.0,
                }
            )
        else:
            reward = 0.0
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
        return {
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
            "low_peak_click_penalty": low_peak_click_penalty,
            "previous_peak_click_rate_hz": previous_peak,
            "peak_regression_ratio": peak_regression_ratio,
            "peak_regression_penalty": peak_regression_penalty,
            "elapsed_seconds": time.perf_counter() - started,
            "learned": bool(learn),
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
    resume_info = None
    if config.resume and checkpoint_path.is_file():
        resume_info = restore_learning_checkpoint(checkpoint_path, brain=session.brain)
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
    rows = []
    correct = 0
    training_correct = 0
    evaluation_correct = 0
    training_trials = 0
    evaluation_trials = 0
    started = time.perf_counter()
    np = session.np
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
    evaluation_score = 0
    evaluation_accuracy = 0.0
    click_window: list[bool] = []
    mastery_reached = False
    click_window_reached = False
    key_accuracy_reached = False
    completed = 0
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
            elif evaluation_number > 0 and not retraining_ready:
                minimum_round = min(retraining_trials.values())
                candidates = [
                    label for label in KEY_LABELS
                    if retraining_trials[label] == minimum_round
                ]
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
                lowest_accuracy = min(priority_accuracy.values())
                candidates = [
                    label for label in KEY_LABELS
                    if priority_accuracy[label] == lowest_accuracy
                ]
            label = str(candidates[int(rng.integers(0, len(candidates)))])
            learn = True
        elif config.click_only and phase == "evaluation":
            label = evaluation_schedule[evaluation_index]
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
            learn = True

        completed += 1
        trial = completed
        row_phase = phase
        row = session.run_trial(label, learn=learn)
        row["trial"] = trial
        row["phase"] = row_phase
        rows.append(row)
        row_correct = int(row["correct"])
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
                if evaluation_passed:
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
            "evaluation_number": evaluation_number,
            "evaluation_failures": evaluation_failures,
            "evaluation_per_key": evaluation_per_key,
            "per_key": per_key,
            "route_provenance": session.route_provenance,
            "motor_channel_count": session.motor_channel_count,
            "token_labels": list(KEY_LABELS),
            "keyboard_layout": keyboard_layout,
            "body": row.get("body", {}),
            "recent_trials": rows[-16:],
        }
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(build_keyboard_html(payload), encoding="utf-8")
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
        "evaluation_number": evaluation_number,
        "evaluation_failures": evaluation_failures,
        "evaluation_per_key": evaluation_per_key,
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
        },
        "execution_profile": {
            "backend": "numpy_cpu",
            "gpu_used": False,
            "external_decoder": False,
            "closed_loop_virtual_body": True,
            "body_mode": config.body_mode,
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
    parser.add_argument("--min-trials-per-key", type=int, default=5)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument("--click-penalty", type=float, default=1.20)
    parser.add_argument("--low-peak-click-penalty-scale", type=float, default=0.0)
    parser.add_argument("--click-teacher-learning-rate", type=float, default=0.08)
    parser.add_argument("--click-teacher-credit-floor", type=float, default=0.05)
    parser.add_argument("--peak-regression-penalty-scale", type=float, default=0.80)
    parser.add_argument("--click-gate-threshold-hz", type=float, default=9.0)
    parser.add_argument("--click-integration-windows", type=int, default=1)
    parser.add_argument("--click-evidence-windows", type=int, default=5)
    parser.add_argument("--click-margin-target-hz", type=float, default=15.0)
    parser.add_argument("--click-margin-reward-scale", type=float, default=0.50)
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
        lock_arm_during_click_training=not args.allow_click_training_movement,
        click_penalty=args.click_penalty,
        low_peak_click_penalty_scale=args.low_peak_click_penalty_scale,
        click_teacher_learning_rate=args.click_teacher_learning_rate,
        click_teacher_credit_floor=args.click_teacher_credit_floor,
        peak_regression_penalty_scale=args.peak_regression_penalty_scale,
        click_gate_threshold_hz=args.click_gate_threshold_hz,
        click_integration_windows=args.click_integration_windows,
        click_evidence_windows=args.click_evidence_windows,
        click_margin_target_hz=args.click_margin_target_hz,
        click_margin_reward_scale=args.click_margin_reward_scale,
        distance_penalty_scale=args.distance_penalty_scale,
        correct_distance_penalty_scale=args.correct_distance_penalty_scale,
        consecutive_correct_bonus=args.consecutive_correct_bonus,
        max_consecutive_bonus=args.max_consecutive_bonus,
        duration_ms=args.duration_ms,
        control_window_ms=args.control_window_ms,
        max_control_windows=args.max_control_windows,
        stimulus_rate_hz=args.stimulus_rate_hz,
        motor_population_size=args.motor_population_size,
        body_mode=args.body_mode,
        curriculum_stage=args.curriculum_stage,
        checkpoint_every=args.checkpoint_every,
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
