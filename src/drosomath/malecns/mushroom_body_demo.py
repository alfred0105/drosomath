from __future__ import annotations

import argparse
import json
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import PlasticStateConfig, UsageRewardRule

from .brain import PlasticMaleCNSBrain
from .download import DEFAULT_DATA_DIR, download_malecns
from .loader import load_malecns_v1
from .mushroom_body import MushroomBodyCircuit


DEFAULT_RESULT = Path("results/latest_mushroom_body_trial.json")


def _default_stimuli(connectome, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("stimulus count must be >= 1")
    try:
        indices = connectome.select_indices(superclass="visual_projection")
    except KeyError as exc:
        raise ValueError(
            "no visual_projection annotation is available; pass --stimulus-body-id explicitly"
        ) from exc
    if len(indices) < count:
        raise ValueError(
            f"requested {count} visual stimulus neurons, found only {len(indices)}"
        )
    return tuple(int(connectome.body_ids[int(i)]) for i in indices[:count])


def run_trials(
    *,
    data_dir: Path,
    min_syn: int,
    stimulus_body_ids: tuple[int, ...],
    stimulus_count: int,
    trials: int,
    duration_ms: float,
    stimulus_rate_hz: float,
    reward: float,
    seed: int,
) -> dict[str, object]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    connectome = load_malecns_v1(data_dir, min_connection_synapses=min_syn)
    if not stimulus_body_ids:
        stimulus_body_ids = _default_stimuli(connectome, stimulus_count)

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=1.0,
            seed=seed,
        ),
    )
    circuit = MushroomBodyCircuit.from_connectome(connectome)
    circuit.attach(brain)

    rule = UsageRewardRule(learning_rate=0.02)
    reports = []
    for trial in range(trials):
        reports.append(
            circuit.run_trial(
                stimulus_body_ids=stimulus_body_ids,
                reward=reward,
                duration_ms=duration_ms,
                stimulus_rate_hz=stimulus_rate_hz,
                rule=rule,
            )
        )

    return {
        "experiment": "connectome_mushroom_body_trial",
        "connectome": connectome.summary(),
        "mushroom_body": circuit.summary(),
        "stimulus_body_ids": list(stimulus_body_ids),
        "trials": reports,
        "config": {
            "min_connection_synapses": min_syn,
            "trial_count": trials,
            "duration_ms": duration_ms,
            "stimulus_rate_hz": stimulus_rate_hz,
            "reward": reward,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reward learning on real annotated KC -> MBON connectome edges."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--stimulus-body-id", type=int, action="append", default=[])
    parser.add_argument("--stimulus-count", type=int, default=8)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--stimulus-rate-hz", type=float, default=205.0)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()

    if args.download:
        download_malecns(args.data_dir)

    report = run_trials(
        data_dir=args.data_dir,
        min_syn=args.min_syn,
        stimulus_body_ids=tuple(args.stimulus_body_id),
        stimulus_count=args.stimulus_count,
        trials=args.trials,
        duration_ms=args.duration_ms,
        stimulus_rate_hz=args.stimulus_rate_hz,
        reward=args.reward,
        seed=args.seed,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "experiment": report["experiment"],
        "dataset": report["connectome"]["dataset"],
        "neuron_count": report["connectome"]["neuron_count"],
        "edge_count": report["connectome"]["edge_count"],
        "mushroom_body": report["mushroom_body"],
        "stimulus_body_ids": report["stimulus_body_ids"],
        "trial_count": len(report["trials"]),
        "saved_result": str(args.result),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
