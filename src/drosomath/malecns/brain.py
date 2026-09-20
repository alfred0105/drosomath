from __future__ import annotations

import argparse
import json
from pathlib import Path

from drosomath.flywire_real import FlyBrainParams
from drosomath.whole_brain import (
    OutgoingBudgetNormalizer,
    PlasticSparseFlyBrain,
    PlasticStateConfig,
    PresynapticDepressionConfig,
    SlowAdaptationConfig,
    StructuralOverlayConfig,
    UsageRewardRule,
)

from .download import DEFAULT_DATA_DIR, download_malecns
from .fast_state import FastLearnedStructuralOverlay, FastSparseStateMixin
from .loader import MaleCNSConnectome, load_malecns_v1


class PlasticMaleCNSBrain(FastSparseStateMixin, PlasticSparseFlyBrain):
    """Plastic MaleCNS simulator with sparse active-state acceleration."""

    def __init__(
        self,
        connectome: MaleCNSConnectome,
        *,
        params: FlyBrainParams | None = None,
        seed: int = 0,
        plasticity_config: PlasticStateConfig | None = None,
        usage_alpha: float = 0.05,
        eligibility_gain: float = 1.0,
        slow_adaptation_config: SlowAdaptationConfig | None = None,
        presynaptic_depression_config: PresynapticDepressionConfig | None = None,
    ) -> None:
        self.slow_adaptation_config = slow_adaptation_config or SlowAdaptationConfig()
        self.presynaptic_depression_config = (
            presynaptic_depression_config or PresynapticDepressionConfig()
        )
        super().__init__(
            connectome,
            params=params,
            seed=seed,
            plasticity_config=plasticity_config,
            usage_alpha=usage_alpha,
            eligibility_gain=eligibility_gain,
        )

    def configure_structural_plasticity(
        self,
        config: StructuralOverlayConfig | None = None,
    ) -> FastLearnedStructuralOverlay:
        """Enable MaleCNS structural plasticity with bounded donor scans."""
        overlay = FastLearnedStructuralOverlay(
            self.connectome,
            self.plasticity,
            config=config,
        )
        self.structural_overlay = overlay
        self._structural_reward_events = 0
        return overlay

    def run_malecns(
        self,
        *,
        duration_ms: float,
        stimulus_body_ids,
        stimulus_rate_hz: float = 100.0,
        top_fired: int = 20,
    ) -> dict[str, object]:
        result = super().run(
            duration_ms=duration_ms,
            stimulus_ids=stimulus_body_ids,
            stimulus_rate_hz=stimulus_rate_hz,
            top_fired=top_fired,
        )

        # The inherited engine predates MaleCNS and labels its generic IDs as
        # FlyWire IDs. Rewrite only the report labels; the simulation itself
        # already uses MaleCNS body IDs through the compatibility property.
        if "stimulus_flywire_ids" in result:
            result["stimulus_body_ids"] = result.pop("stimulus_flywire_ids")
        for row in result.get("top_firing_neurons", []):
            if "flywire_id" in row:
                row["body_id"] = row.pop("flywire_id")
        result["dataset"] = self.connectome.source
        result["fast_state"] = self.fast_state_summary()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the plastic DrosoMath engine on MaleCNS v1.0."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-connection-synapses", type=int, default=5)
    parser.add_argument("--duration-ms", type=float, default=10.0)
    parser.add_argument("--stimulus-rate-hz", type=float, default=300.0)
    parser.add_argument("--auto-stimuli", type=int, default=2)
    parser.add_argument("--stimulus-body-id", type=int, action="append", default=[])
    parser.add_argument("--plastic-fraction", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--budget-strength", type=float, default=0.25)
    parser.add_argument("--dt-ms", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.download:
        download_malecns(args.data_dir)

    connectome = load_malecns_v1(
        args.data_dir,
        min_connection_synapses=args.min_connection_synapses,
    )
    stimulus_ids = tuple(int(x) for x in args.stimulus_body_id)
    if not stimulus_ids:
        stimulus_ids = connectome.strongest_outgoing_ids(args.auto_stimuli)

    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=args.dt_ms),
        seed=args.seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=args.plastic_fraction,
            seed=args.seed,
        ),
    )
    simulation = brain.run_malecns(
        duration_ms=args.duration_ms,
        stimulus_body_ids=stimulus_ids,
        stimulus_rate_hz=args.stimulus_rate_hz,
    )
    learning = brain.learn_from_reward(
        reward=args.reward,
        rule=UsageRewardRule(learning_rate=args.learning_rate),
        normalizer=OutgoingBudgetNormalizer(strength=args.budget_strength),
    )

    report = {
        "connectome": connectome.summary(),
        "simulation": simulation,
        "learning": learning,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
