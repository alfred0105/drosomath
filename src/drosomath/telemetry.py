from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean

from .core import AdaptiveBridge, AdaptiveModuleManager, PlasticityTracker, StepResult


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    step: int
    fired_count: int
    transferred_synapses: int
    synapse_count: int
    mean_weight: float
    mean_stability: float
    bridge_synapse_count: int = 0
    bridge_mean_weight: float = 0.0
    module_activity: tuple[tuple[int, float], ...] = ()
    module_reward: tuple[tuple[int, float], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def capture_snapshot(
    result: StepResult,
    tracker: PlasticityTracker,
    *,
    bridge: AdaptiveBridge | None = None,
    modules: AdaptiveModuleManager | None = None,
) -> TelemetrySnapshot:
    synapses = tracker.synapses
    bridge_synapses = bridge.synapses if bridge is not None else ()
    module_activity = ()
    module_reward = ()
    if modules is not None:
        module_activity = tuple(
            sorted(
                (module_id, state.activity_ema)
                for module_id, state in modules.modules.items()
            )
        )
        module_reward = tuple(
            sorted(
                (module_id, state.reward_ema)
                for module_id, state in modules.modules.items()
            )
        )

    return TelemetrySnapshot(
        step=result.step,
        fired_count=len(result.fired),
        transferred_synapses=result.transferred_synapses,
        synapse_count=len(synapses),
        mean_weight=fmean(s.weight for s in synapses) if synapses else 0.0,
        mean_stability=fmean(s.stability for s in synapses) if synapses else 0.0,
        bridge_synapse_count=len(bridge_synapses),
        bridge_mean_weight=(
            fmean(s.weight for s in bridge_synapses) if bridge_synapses else 0.0
        ),
        module_activity=module_activity,
        module_reward=module_reward,
    )
