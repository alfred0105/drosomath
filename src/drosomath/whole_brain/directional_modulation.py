"""Sparse, task-independent output-direction neuromodulation."""

from __future__ import annotations

from dataclasses import dataclass

from drosomath.learning_signal import LearningSignal


@dataclass(frozen=True, slots=True)
class DirectionalModulationConfig:
    learning_rate: float = 0.02
    stability_protection: float = 0.5


@dataclass(frozen=True, slots=True)
class DirectionalUpdate:
    edge_updates: int
    channel_updates: dict[str, int]
    mean_abs_delta: float


class PlasticityController:
    """Apply local output-direction feedback using only active eligibility.

    ``output_context`` maps generic channel names to output-neuron indices.  It
    contains no environment-specific label or teacher edge set.
    """

    def __init__(self, config: DirectionalModulationConfig | None = None) -> None:
        self.config = config or DirectionalModulationConfig()

    def apply_learning_signal(self, brain, signal: LearningSignal, output_context) -> DirectionalUpdate:
        np = brain.np
        state, graph = brain.plasticity, brain.connectome
        total, sum_abs = 0, 0.0
        per_channel: dict[str, int] = {}
        for name, direction in signal.nonzero_directions().items():
            outputs = output_context.get(name)
            if outputs is None:
                continue
            output_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
            output_mask[np.asarray(outputs, dtype=np.int32)] = True
            edges = np.flatnonzero(output_mask[graph.post_indices]).astype(np.int32, copy=False)
            active = edges[state.plastic_mask[edges] & (state.eligibility[edges] > 0.0)]
            if not len(active):
                per_channel[name] = 0
                continue
            # Stability is a bounded metaplastic factor: consolidated edges
            # still learn, but unrelated directional perturbation is slower.
            metaplastic = 1.0 - self.config.stability_protection * state.stability[active]
            delta = self.config.learning_rate * float(direction) * state.eligibility[active] * metaplastic
            old = state.multiplier[active].copy()
            state.multiplier[active] = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
            actual = state.multiplier[active] - old
            count = int(len(active))
            total += count
            sum_abs += float(np.abs(actual).sum())
            per_channel[name] = count
        return DirectionalUpdate(total, per_channel, sum_abs / total if total else 0.0)
