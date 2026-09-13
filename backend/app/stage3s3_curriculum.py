from __future__ import annotations

from typing import Any

from .symbolic_stage3s3 import (
    ADDITION_HELDOUT_PAIRS,
    ADDITION_REVERSE_TEST_PAIRS,
    ADDITION_TRAIN_PAIRS,
    BRIDGE_ADDITION_PAIRS,
    NEXT2_HELDOUT_STARTS,
    NEXT2_TRAIN_STARTS,
    NEXT3_HELDOUT_STARTS,
    NEXT3_TRAIN_STARTS,
    PREV2_HELDOUT_STARTS,
    PREV2_TRAIN_STARTS,
    PREV3_HELDOUT_STARTS,
    PREV3_TRAIN_STARTS,
    SMALL_ADDITION_PAIRS,
    SymbolicStage3S3Experiment,
)

TRAIN_TRIALS = 600_000
TRAINING_MILESTONES = (40_000, 120_000, 220_000, 320_000, 400_000, 500_000, 600_000)
PROFILE_TRIALS = 1_200
PROFILE_ORDER = (
    "identity", "next", "prev",
    "next2_seen", "next2_heldout", "next3_seen", "next3_heldout",
    "prev2_seen", "prev2_heldout", "prev3_seen", "prev3_heldout",
    "addition_seen", "addition_commutativity", "addition_heldout_pairs",
)
TASK_NAMES = (
    "identity", "next", "prev",
    "guided_next2", "guided_next3", "guided_prev2", "guided_prev3",
    "next2", "next3", "prev2", "prev3", "addition",
)


def _pick(exp: SymbolicStage3S3Experiment, pool: tuple[int, ...]) -> int:
    return int(pool[int(exp.rng.integers(0, len(pool)))])


def _operator_sample(exp: SymbolicStage3S3Experiment, guided: bool) -> tuple[str, int]:
    r = float(exp.rng.random())
    prefix = "guided_" if guided else ""
    if r < 0.30:
        return prefix + "next2", _pick(exp, NEXT2_TRAIN_STARTS)
    if r < 0.50:
        return prefix + "next3", _pick(exp, NEXT3_TRAIN_STARTS)
    if r < 0.78:
        return prefix + "prev2", _pick(exp, PREV2_TRAIN_STARTS)
    return prefix + "prev3", _pick(exp, PREV3_TRAIN_STARTS)


def training_spec(exp: SymbolicStage3S3Experiment, trial: int) -> dict[str, Any]:
    if trial <= 40_000:
        return dict(task="identity", phase="01_identity_0_40k", token=1.0, recurrent=0.0, operator=0.0, output=1.0)
    if trial <= 120_000:
        r = float(exp.rng.random())
        task = "next" if r < 0.42 else "prev" if r < 0.84 else "identity"
        return dict(task=task, phase="02_next_prev_single_40_120k", token=.55, recurrent=.30, operator=1.0, output=1.0)
    if trial <= 220_000:
        r = float(exp.rng.random())
        if r < .78:
            task, a = _operator_sample(exp, True)
        elif r < .88:
            task, a = "next", None
        elif r < .98:
            task, a = "prev", None
        else:
            task, a = "identity", None
        return dict(task=task, a=a, phase="03_guided_multistep_120_220k", token=.15, recurrent=.15, operator=.85, output=1.0)
    if trial <= 320_000:
        r = float(exp.rng.random())
        if r < .82:
            task, a = _operator_sample(exp, False)
        elif r < .90:
            task, a = "next", None
        elif r < .98:
            task, a = "prev", None
        else:
            task, a = "identity", None
        return dict(task=task, a=a, phase="04_autonomous_multistep_220_320k", token=.04, recurrent=.08, operator=.65, output=1.0)
    if trial <= 400_000:
        r = float(exp.rng.random())
        if r < .55:
            task, a, pool = "addition", None, BRIDGE_ADDITION_PAIRS
        elif r < .82:
            task, a = _operator_sample(exp, False); pool = None
        elif r < .90:
            task, a, pool = "next", None, None
        elif r < .98:
            task, a, pool = "prev", None, None
        else:
            task, a, pool = "identity", None, None
        return dict(task=task, a=a, pair_pool=pool, phase="05_addition_bridge_320_400k", token=.02, recurrent=.05, operator=.35, output=1.0)
    if trial <= 500_000:
        r = float(exp.rng.random())
        if r < .65:
            task, a, pool = "addition", None, SMALL_ADDITION_PAIRS
        elif r < .75:
            task, a, pool = "addition", None, BRIDGE_ADDITION_PAIRS
        elif r < .91:
            task, a = _operator_sample(exp, False); pool = None
        elif r < .95:
            task, a, pool = "next", None, None
        elif r < .99:
            task, a, pool = "prev", None, None
        else:
            task, a, pool = "identity", None, None
        return dict(task=task, a=a, pair_pool=pool, phase="06_small_addition_400_500k", token=.01, recurrent=.03, operator=.20, output=1.0)
    r = float(exp.rng.random())
    if r < .72:
        task, a, pool = "addition", None, ADDITION_TRAIN_PAIRS
    elif r < .80:
        task, a, pool = "addition", None, BRIDGE_ADDITION_PAIRS
    elif r < .92:
        task, a = _operator_sample(exp, False); pool = None
    elif r < .96:
        task, a, pool = "next", None, None
    elif r < .995:
        task, a, pool = "prev", None, None
    else:
        task, a, pool = "identity", None, None
    return dict(task=task, a=a, pair_pool=pool, phase="07_full_addition_500_600k", token=.005, recurrent=.015, operator=.12, output=1.0)


def profile_example(profile: str, index: int) -> tuple[str, int | None, int | None]:
    if profile == "identity": return "identity", index % 10, None
    if profile == "next": return "next", index % 9, None
    if profile == "prev": return "prev", 1 + index % 9, None
    starts = {
        "next2_seen": ("next2", NEXT2_TRAIN_STARTS), "next2_heldout": ("next2", NEXT2_HELDOUT_STARTS),
        "next3_seen": ("next3", NEXT3_TRAIN_STARTS), "next3_heldout": ("next3", NEXT3_HELDOUT_STARTS),
        "prev2_seen": ("prev2", PREV2_TRAIN_STARTS), "prev2_heldout": ("prev2", PREV2_HELDOUT_STARTS),
        "prev3_seen": ("prev3", PREV3_TRAIN_STARTS), "prev3_heldout": ("prev3", PREV3_HELDOUT_STARTS),
    }
    if profile in starts:
        task, pool = starts[profile]
        return task, pool[index % len(pool)], None
    if profile == "addition_seen": pair = ADDITION_TRAIN_PAIRS[index % len(ADDITION_TRAIN_PAIRS)]
    elif profile == "addition_commutativity": pair = ADDITION_REVERSE_TEST_PAIRS[index % len(ADDITION_REVERSE_TEST_PAIRS)]
    elif profile == "addition_heldout_pairs": pair = ADDITION_HELDOUT_PAIRS[index % len(ADDITION_HELDOUT_PAIRS)]
    else: raise ValueError(profile)
    return "addition", pair[0], pair[1]
