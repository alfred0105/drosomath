"""Read-only route diagnostics for the virtual-keyboard curriculum.

This module deliberately does not alter the curriculum or learning rule.  It
uses a short-lived snapshot for frozen probes, so diagnostic execution cannot
silently change the checkpointed learned state.
"""

from __future__ import annotations

import copy
import html
import json
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from .virtual_keyboard import KEY_LABELS

if TYPE_CHECKING:  # pragma: no cover
    from .keyboard_learning import KeyboardNeuralSession


def jaccard(left, right) -> float:
    """Return set Jaccard, defining two empty sets as fully overlapping."""
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _edge_pre(connectome, edges):
    return np.searchsorted(connectome.indptr, edges, side="right").astype(np.int32) - 1


def _hops(connectome, stimulus):
    """Return first and second hop reachable masks from ``stimulus``."""
    n = connectome.neuron_count
    first = np.zeros(n, dtype=bool)
    second = np.zeros(n, dtype=bool)
    for pre in np.asarray(stimulus, dtype=np.int32):
        first[connectome.post_indices[connectome.indptr[pre]:connectome.indptr[pre + 1]]] = True
    for pre in np.flatnonzero(first):
        second[connectome.post_indices[connectome.indptr[pre]:connectome.indptr[pre + 1]]] = True
    return first, second


def route_for_label(session: "KeyboardNeuralSession", label: str) -> dict[str, object]:
    """Summarize actual anatomical two-hop routes into the click population."""
    graph, state = session.brain.connectome, session.brain.plasticity
    stimulus = session.stimulus_indices_by_label[label]
    first, second = _hops(graph, stimulus)
    click = session.channel_groups[4]
    click_mask = np.zeros(graph.neuron_count, dtype=bool)
    click_mask[click] = True
    all_click_edges = np.flatnonzero(click_mask[graph.post_indices]).astype(np.int32)
    all_click_pre = _edge_pre(graph, all_click_edges)
    routed = all_click_edges[first[all_click_pre]]
    routed_signed = graph.signed_synapse_counts[routed]
    teacher = session.click_teacher_edges_by_label[label]
    plastic_teacher = teacher[state.plastic_mask[teacher]]
    multipliers = state.multiplier[plastic_teacher]
    signed_teacher = graph.signed_synapse_counts[plastic_teacher]
    saturation = multipliers >= state.config.max_multiplier - 1e-3
    reachable_click = click[first[click] | second[click]]
    direct = all_click_edges[np.isin(all_click_pre, stimulus, assume_unique=False)]
    direct_signed = graph.signed_synapse_counts[direct]
    return {
        "label": label,
        "token_specific_input_neuron_count": len(session.encoder.token_body_ids[label]),
        "shared_background_neuron_count": len(session.encoder.background_body_ids),
        "total_stimulated_neuron_count": int(len(stimulus)),
        "first_hop_reachable_neuron_count": int(first.sum()),
        "second_hop_reachable_neuron_count": int(second.sum()),
        "teacher_candidate_edge_count": int(len(teacher)),
        "plastic_teacher_edge_count": int(len(plastic_teacher)),
        "distinct_teacher_presynaptic_neuron_count": int(len(np.unique(_edge_pre(graph, teacher)))) if len(teacher) else 0,
        "direct_excitatory_edge_count_into_click": int((direct_signed > 0).sum()),
        "direct_inhibitory_edge_count_into_click": int((direct_signed < 0).sum()),
        "total_positive_anatomical_strength_into_click": float(routed_signed[routed_signed > 0].sum()),
        "total_negative_anatomical_strength_into_click": float(routed_signed[routed_signed < 0].sum()),
        "net_signed_strength": float(routed_signed.sum()),
        "mean_anatomical_strength_teacher_edges": float(signed_teacher.mean()) if len(signed_teacher) else 0.0,
        "max_anatomical_strength_teacher_edges": float(signed_teacher.max()) if len(signed_teacher) else 0.0,
        "reachable_click_output_neuron_count": int(len(reachable_click)),
        "click_output_neurons_receiving_path_count": int(len(reachable_click)),
        "teacher_multiplier_mean": float(multipliers.mean()) if len(multipliers) else 0.0,
        "teacher_multiplier_median": float(np.median(multipliers)) if len(multipliers) else 0.0,
        "teacher_multiplier_max": float(multipliers.max()) if len(multipliers) else 0.0,
        "saturated_teacher_edge_count": int(saturation.sum()),
        "saturated_fraction": float(saturation.mean()) if len(saturation) else 0.0,
        "teacher_edge_mean_stability": float(state.stability[plastic_teacher].mean()) if len(plastic_teacher) else 0.0,
        "teacher_edge_mean_usage_ema": float(state.usage_ema[plastic_teacher].mean()) if len(plastic_teacher) else 0.0,
        "effective_teacher_strength": float(np.abs(signed_teacher * multipliers * session.brain.params.mv_per_synapse).sum()) if len(plastic_teacher) else 0.0,
    }


def _snapshot(session: "KeyboardNeuralSession") -> dict[str, object]:
    """Capture every mutable learning/curriculum item touched by a trial."""
    state = session.brain.plasticity
    return {
        "multiplier": state.multiplier.copy(), "stability": state.stability.copy(),
        "usage": state.usage_ema.copy(), "eligibility": state.eligibility.copy(),
        "brain_rng": copy.deepcopy(session.brain.rng.bit_generator.state),
        "task_rng": session.task.rng.getstate(),
        "session_rng": copy.deepcopy(session.rng.bit_generator.state),
        "correct_streak": dict(session.correct_streak_by_label),
        "previous_peak": dict(session.previous_peak_by_label),
        "regression_streak": dict(session.peak_regression_streak_by_label),
        "peak_ema": dict(session.peak_ema_by_label), "deficit_ema": dict(session.deficit_ema_by_label),
        "subthreshold": dict(session.subthreshold_streak_by_label),
        "teacher_memory": {key: value.copy() for key, value in session.teacher_memory_edges_by_label.items()},
        "pending": copy.deepcopy(session._pending_teacher_normalizer),
    }


def _restore(session: "KeyboardNeuralSession", snap: dict[str, object]) -> None:
    state = session.brain.plasticity
    state.multiplier[:] = snap["multiplier"]
    state.stability[:] = snap["stability"]
    state.usage_ema[:] = snap["usage"]
    state.eligibility[:] = snap["eligibility"]
    session.brain.rng.bit_generator.state = snap["brain_rng"]
    session.task.rng.setstate(snap["task_rng"])
    session.rng.bit_generator.state = snap["session_rng"]
    session.correct_streak_by_label = dict(snap["correct_streak"])
    session.previous_peak_by_label = dict(snap["previous_peak"])
    session.peak_regression_streak_by_label = dict(snap["regression_streak"])
    session.peak_ema_by_label = dict(snap["peak_ema"])
    session.deficit_ema_by_label = dict(snap["deficit_ema"])
    session.subthreshold_streak_by_label = dict(snap["subthreshold"])
    session.teacher_memory_edges_by_label = {key: value.copy() for key, value in snap["teacher_memory"].items()}
    session._pending_teacher_normalizer = copy.deepcopy(snap["pending"])


def _assert_unchanged(session: "KeyboardNeuralSession", snap: dict[str, object]) -> None:
    state = session.brain.plasticity
    for name, current in (("multiplier", state.multiplier), ("stability", state.stability), ("usage", state.usage_ema), ("eligibility", state.eligibility)):
        if not np.array_equal(current, snap[name]):
            raise AssertionError(f"frozen probe changed {name}")


def _probe_row(session: "KeyboardNeuralSession", label: str, *, mode: str = "combined", rng_seed: int | None = None) -> dict[str, object]:
    """Run one response probe and restore all learning state afterwards."""
    original = session.stimulus_indices_by_label[label]
    if mode == "background":
        stimulus = session.brain.indices_for_ids(session.encoder.background_body_ids)
    elif mode == "token":
        stimulus = session.brain.indices_for_ids(session.encoder.token_body_ids[label])
    elif mode == "combined":
        stimulus = original
    else:
        raise ValueError(f"unknown probe mode: {mode}")
    snap = _snapshot(session)
    try:
        # learn=True obtains true active eligibility; the full state is restored
        # immediately after response collection, before another action can see it.
        session.stimulus_indices_by_label[label] = stimulus
        if rng_seed is not None:
            session.brain.rng.bit_generator.state = np.random.default_rng(rng_seed).bit_generator.state
        row = session.run_trial(label, learn=True)
        teacher = row["learning"].get("motor_teacher", {})
        return {
            "click_peak_hz": float(row["peak_click_rate_hz"]),
            "mean_click_rate_hz": float(row["peak_click_evidence_hz"]),
            "click_occurred": bool(row["clicked_label"] is not None),
            "click_population_spike_count": int(row["click_population_spike_count"]),
            "active_click_output_neuron_count": int(row["active_click_output_neuron_count"]),
            "active_route": {
                "candidate_teacher_edges": int(teacher.get("candidate_edge_count", 0)),
                "plastic_teacher_edges": int(teacher.get("plastic_edge_count", 0)),
                "active_eligible_teacher_edges": int(teacher.get("active_eligible_edge_count", 0)),
                "active_teacher_presynaptic_neuron_count": int(teacher.get("active_presynaptic_count", 0)),
                "teacher_edges_usage_nonzero": int(teacher.get("usage_nonzero_edge_count", 0)),
                "total_teacher_eligibility": float(teacher.get("total_eligibility", 0.0)),
                "total_teacher_credit": float(teacher.get("total_credit", 0.0)),
                "effective_teacher_strength": float(teacher.get("effective_strength_after", 0.0)),
                "threshold_margin_hz": float(row["peak_click_rate_hz"] - session.config.click_gate_threshold_hz),
                "active_teacher_fraction": float(teacher.get("active_eligible_edge_count", 0)) / max(1, int(teacher.get("plastic_edge_count", 0))),
            },
        }
    finally:
        session.stimulus_indices_by_label[label] = original
        _restore(session, snap)
        _assert_unchanged(session, snap)


def frozen_probe(session: "KeyboardNeuralSession", label: str, mode: str = "combined", *, rng_seed: int | None = None) -> dict[str, object]:
    return _probe_row(session, label, mode=mode, rng_seed=rng_seed)


def _comparison_rows(per_key: dict[str, object], routes: dict[str, object], probes: dict[str, object], labels) -> list[dict[str, object]]:
    rows = []
    for label in labels:
        key, route, probe = per_key[label], routes[label], probes[label]
        rows.append({
            "label": label, "recent_accuracy": float(key.get("recent_accuracy", 0.0)),
            "cumulative_accuracy": float(key.get("accuracy", 0.0)), "attempts": int(key.get("trials", 0)),
            "average_peak_click_hz": float(probe["click_peak_hz"]), "median_peak_click_hz": float(probe["click_peak_hz"]),
            "threshold_crossing_rate": float(probe["click_peak_hz"] >= 9.0),
            "candidate_edges": route["teacher_candidate_edge_count"], "plastic_teacher_edges": route["plastic_teacher_edge_count"],
            "active_teacher_fraction": probe["active_route"]["active_teacher_fraction"],
            "saturated_teacher_fraction": route["saturated_fraction"], "teacher_effective_strength": route["effective_teacher_strength"],
            "mean_stability": route["teacher_edge_mean_stability"], "mean_usage_ema": route["teacher_edge_mean_usage_ema"],
            "reachable_click_outputs": route["reachable_click_output_neuron_count"],
        })
    return rows


def _overlap_matrix(session: "KeyboardNeuralSession", labels) -> dict[str, dict[str, object]]:
    graph = session.brain.connectome
    first_hops = {label: np.flatnonzero(_hops(graph, session.stimulus_indices_by_label[label])[0]) for label in labels}
    result: dict[str, dict[str, object]] = {}
    for source in labels:
        result[source] = {}
        for target in labels:
            result[source][target] = {
                "teacher_edge_jaccard": jaccard(session.click_teacher_edges_by_label[source], session.click_teacher_edges_by_label[target]),
                "teacher_presynaptic_jaccard": jaccard(_edge_pre(graph, session.click_teacher_edges_by_label[source]), _edge_pre(graph, session.click_teacher_edges_by_label[target])),
                "first_hop_jaccard": jaccard(first_hops[source], first_hops[target]),
                "stimulated_neuron_jaccard": jaccard(session.stimulus_indices_by_label[source], session.stimulus_indices_by_label[target]),
            }
    return result


def run_keyboard_route_diagnostics(session: "KeyboardNeuralSession", *, per_key: dict[str, object], diagnostic_key_count: int = 5, include_background: bool = False, include_interference: bool = False, full_matrix: bool = False, retention_trials_per_key: int = 5, on_progress=None) -> dict[str, object]:
    """Produce diagnosis-only evidence without persisting a learning update."""
    labels = tuple(KEY_LABELS)
    def emit(stage: str, completed: int, total: int, label: str | None = None) -> None:
        if on_progress is not None:
            on_progress({"status": "running", "stage": stage, "completed": completed, "total": total, "label": label})

    routes = {}
    for index, label in enumerate(labels, 1):
        routes[label] = route_for_label(session, label)
        emit("anatomical_routes", index, len(labels), label)
    ranked = sorted(labels, key=lambda label: (float(per_key[label].get("recent_accuracy", 0.0)), label))
    weak, strong = ranked[:diagnostic_key_count], list(reversed(ranked[-diagnostic_key_count:]))
    selected = list(dict.fromkeys(weak + strong))
    probe_labels = labels if full_matrix else selected
    combined = {}
    for index, label in enumerate(probe_labels, 1):
        combined[label] = frozen_probe(session, label)
        emit("combined_frozen_probes", index, len(probe_labels), label)
    background = {}
    if include_background:
        for index, label in enumerate(selected, 1):
            background[label] = {mode: frozen_probe(session, label, mode) for mode in ("background", "token", "combined")}
            emit("background_token_probes", index, len(selected), label)
    overlap_labels = labels if full_matrix else selected
    overlap = _overlap_matrix(session, overlap_labels)
    interference: dict[str, dict[str, object]] = {}
    reward_spread: dict[str, object] = {}
    if include_interference:
        # Restore the complete original state after each source-learning experiment.
        for source_index, source in enumerate(selected, 1):
            snap = _snapshot(session)
            pre = {target: frozen_probe(session, target)["click_peak_hz"] for target in selected}
            before_multiplier = session.brain.plasticity.multiplier.copy()
            source_row = session.run_trial(source, learn=True)
            post = {target: frozen_probe(session, target)["click_peak_hz"] for target in selected}
            interference[source] = {target: {
                "pre_peak_hz": pre[target], "post_peak_hz": post[target], "interference_hz": post[target] - pre[target],
                "pre_threshold": pre[target] >= session.config.click_gate_threshold_hz,
                "post_threshold": post[target] >= session.config.click_gate_threshold_hz,
            } for target in selected}
            if source_row["correct"] and float(source_row["global_learning_reward"]) > 0.0:
                changed = np.flatnonzero(np.abs(session.brain.plasticity.multiplier - before_multiplier) > 1e-8)
                current = set(map(int, session.click_teacher_edges_by_label[source]))
                changed_set = set(map(int, changed))
                overlaps = sorted(({
                    "label": other,
                    "updated_teacher_edges": len(changed_set & set(map(int, session.click_teacher_edges_by_label[other]))),
                } for other in KEY_LABELS if other != source), key=lambda row: row["updated_teacher_edges"], reverse=True)[:5]
                reward_spread[source] = {
                    "total_reward_updated_edges": int(len(changed)),
                    "current_label_teacher_edges_updated": int(len(changed_set & current)),
                    "outside_current_teacher_path": int(len(changed_set - current)),
                    "current_label_teacher_fraction": float(len(changed_set & current) / max(1, len(changed))),
                    "distinct_updated_presynaptic_neurons": int(len(np.unique(_edge_pre(session.brain.connectome, changed)))) if len(changed) else 0,
                    "top_other_label_teacher_overlaps": overlaps,
                }
            else:
                reward_spread[source] = {"available": False, "reason": "source_trial_not_positive_reward"}
            _restore(session, snap)
            _assert_unchanged(session, snap)
            emit("cross_label_interference", source_index, len(selected), source)
    overlap_interference = {"sample_size": 0, "teacher_edge_jaccard_correlation": None}
    if interference:
        overlap_values, interference_values = [], []
        for source, targets in interference.items():
            for target, row in targets.items():
                if source == target:
                    continue
                overlap_values.append(float(overlap[source][target]["teacher_edge_jaccard"]))
                interference_values.append(float(row["interference_hz"]))
        if len(overlap_values) >= 3 and np.std(overlap_values) > 0.0 and np.std(interference_values) > 0.0:
            overlap_interference["teacher_edge_jaccard_correlation"] = float(np.corrcoef(overlap_values, interference_values)[0, 1])
        overlap_interference["sample_size"] = len(overlap_values)
    saturation = [{"label": label, "classification": "SATURATED_BUT_WEAK" if routes[label]["saturated_fraction"] >= .5 and float(per_key[label].get("recent_accuracy", 0.0)) < .5 else "OK", **routes[label]} for label in labels]
    retention = {}
    for label_index, label in enumerate(labels, 1):
        label_index = labels.index(label)
        attempts = [
            frozen_probe(session, label, rng_seed=910_000 + label_index * 100 + attempt)["click_peak_hz"]
            >= session.config.click_gate_threshold_hz
            for attempt in range(retention_trials_per_key)
        ]
        retention[label] = {"attempts": retention_trials_per_key, "correct": int(sum(attempts)), "accuracy": float(sum(attempts) / retention_trials_per_key), "standard_error": float((sum(attempts) / retention_trials_per_key * (1 - sum(attempts) / retention_trials_per_key) / retention_trials_per_key) ** .5)}
        emit("multi_trial_retention", label_index, len(labels), label)
    ret_acc = [row["accuracy"] for row in retention.values()]
    report = {
        "summary": {"diagnostic_key_count": diagnostic_key_count, "retention_trials_per_key": retention_trials_per_key, "state_integrity": "passed", "labels": len(labels)},
        "weak_labels": weak, "strong_labels": strong, "per_label_routes": routes,
        "active_route_summary": combined,
        "weak_vs_strong": {"weak": _comparison_rows(per_key, routes, combined, [x for x in weak if x in combined]), "strong": _comparison_rows(per_key, routes, combined, [x for x in strong if x in combined])},
        "background_token_probes": background,
        "interference_matrix": interference,
        "path_overlap_matrix": {"matrix": overlap, "interference_correlation": overlap_interference},
        "positive_reward_spread": reward_spread,
        "saturation_analysis": saturation,
        "retention_analysis": {"per_label": retention, "macro_accuracy": float(np.mean(ret_acc)), "minimum_accuracy": float(np.min(ret_acc)), "median_accuracy": float(median(ret_acc))},
    }
    if on_progress is not None:
        on_progress({"status": "complete", "stage": "complete", "completed": 1, "total": 1, "label": None, "report": report})
    return report


def build_keyboard_route_diagnostics_html(report: dict[str, object]) -> str:
    """Render a portable, offline inspection page for the route evidence."""
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><html lang='ko'><meta charset='utf-8'><meta http-equiv='refresh' content='1'><title>DrosoMath · route diagnostics</title>
<style>body{{margin:0;background:#0d131d;color:#e8eef9;font:14px system-ui;padding:24px}}h1{{margin-top:0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}}section{{background:#151e2b;border:1px solid #2b3b52;border-radius:10px;padding:14px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:7px;border-bottom:1px solid #29384c;text-align:right}}th:first-child,td:first-child{{text-align:left}}.bad{{color:#ff8e8e}}.good{{color:#72e0a5}}.muted{{color:#aab8cd}}</style>
<h1>DrosoMath · Keyboard Route Diagnostics</h1><div id='app'></div><script>const R=__DATA__;const pct=x=>(100*Number(x||0)).toFixed(1)+'%';const hz=x=>Number(x||0).toFixed(2)+' Hz';
const rows=a=>(a||[]).map(x=>`<tr><td class="${x.recent_accuracy<.5?'bad':'good'}">${x.label}</td><td>${pct(x.recent_accuracy)}</td><td>${pct(x.cumulative_accuracy)}</td><td>${x.attempts}</td><td>${hz(x.average_peak_click_hz)}</td><td>${x.active_teacher_fraction.toFixed(2)}</td><td>${x.plastic_teacher_edges}</td><td>${pct(x.saturated_teacher_fraction)}</td></tr>`).join('');
const table=(title,a)=>`<section><h2>${title}</h2><table><tr><th>key</th><th>recent</th><th>total</th><th>attempts</th><th>peak</th><th>active</th><th>plastic</th><th>saturated</th></tr>${rows(a)}</table></section>`;
const bg=Object.entries(R.background_token_probes||{}).map(([k,v])=>`<tr><td>${k}</td>${['background','token','combined'].map(m=>`<td>${hz(v[m].click_peak_hz)}</td>`).join('')}</tr>`).join('');
const P=R.live_progress||{};const A=R.retention_analysis||{};const C=R.path_overlap_matrix?.interference_correlation||{};document.querySelector('#app').innerHTML=`<div class='grid'><section><h2>${P.status==='running'?'Live diagnostic running':'Summary'}</h2><p>stage: <b>${P.stage||'complete'}</b> · ${P.completed??1}/${P.total??1} · label: ${P.label||'—'}</p><p>state integrity: <b class='good'>${R.summary?.state_integrity||'checking'}</b></p><p>retention: macro ${A.macro_accuracy==null?'—':pct(A.macro_accuracy)} · median ${A.median_accuracy==null?'—':pct(A.median_accuracy)} · minimum ${A.minimum_accuracy==null?'—':pct(A.minimum_accuracy)}</p><p>overlap/interference correlation: ${C.teacher_edge_jaccard_correlation?.toFixed(3)??'—'} (${C.sample_size??0} pairs)</p></section>${table('Weak labels',R.weak_vs_strong?.weak)}${table('Strong labels',R.weak_vs_strong?.strong)}<section><h2>Frozen input probes</h2><table><tr><th>key</th><th>background</th><th>token</th><th>combined</th></tr>${bg}</table></section></div>`;</script></html>"""
    return page.replace("__DATA__", data).replace("{{", "{").replace("}}", "}")
