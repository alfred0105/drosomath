from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from drosomath.whole_brain import OutgoingBudgetNormalizer, UsageRewardRule


@dataclass(frozen=True, slots=True)
class MushroomBodyConfig:
    """Annotation patterns for a connectome-backed mushroom-body learning path."""

    kenyon_patterns: tuple[str, ...] = (
        r"\bkenyon\b",
        r"\bkc\b",
        r"\bkc[geh]\b",
        r"mushroom[ _-]*body[ _-]*intrinsic",
    )
    mbon_patterns: tuple[str, ...] = (
        r"\bmbon\b",
        r"mushroom[ _-]*body[ _-]*output",
    )
    dopamine_patterns: tuple[str, ...] = (
        r"\bdan\b",
        r"dopamin",
        r"\bpam\b",
        r"\bppl1\b",
    )
    min_population_size: int = 1
    require_dopamine_population: bool = False

    def __post_init__(self) -> None:
        if self.min_population_size < 1:
            raise ValueError("min_population_size must be >= 1")


@dataclass(frozen=True, slots=True)
class MushroomBodyRoles:
    """Indices of annotated neurons participating in the learning circuit."""

    kenyon_cells: object
    mbons: object
    dopamine_neurons: object


class MushroomBodyCircuit:
    """Connectome-backed KC -> MBON reward-learning circuit.

    The anatomical graph is never replaced. Instead, this adapter identifies
    real annotated neurons and restricts plasticity to anatomical edges from
    Kenyon cells to MBONs. DAN activity is reported as a circuit observable;
    the experiment supplies the scalar reward/punishment signal at trial end.
    """

    def __init__(self, connectome, roles: MushroomBodyRoles, *, config: MushroomBodyConfig):
        self.connectome = connectome
        self.roles = roles
        self.config = config
        self._brain = None
        self._eligible_edge_mask = None

    @classmethod
    def from_connectome(
        cls,
        connectome,
        *,
        config: MushroomBodyConfig | None = None,
    ) -> "MushroomBodyCircuit":
        config = config or MushroomBodyConfig()
        metadata = getattr(connectome, "metadata", None)
        if not metadata:
            raise ValueError(
                "Mushroom-body mapping requires connectome neuron annotations; "
                "load the MaleCNS annotation table or provide a role map."
            )

        np = _require_numpy()
        neuron_count = int(connectome.neuron_count)
        texts = [_annotation_text(metadata, index) for index in range(neuron_count)]

        def match(patterns: tuple[str, ...]):
            compiled = tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
            return np.asarray(
                [
                    any(pattern.search(text) is not None for pattern in compiled)
                    for text in texts
                ],
                dtype=np.bool_,
            )

        kc = np.flatnonzero(match(config.kenyon_patterns)).astype(np.int32, copy=False)
        mbon = np.flatnonzero(match(config.mbon_patterns)).astype(np.int32, copy=False)
        dan = np.flatnonzero(match(config.dopamine_patterns)).astype(np.int32, copy=False)

        if len(kc) < config.min_population_size:
            raise ValueError(f"could not identify enough Kenyon cells: found {len(kc)}")
        if len(mbon) < config.min_population_size:
            raise ValueError(f"could not identify enough MBONs: found {len(mbon)}")
        if config.require_dopamine_population and len(dan) < config.min_population_size:
            raise ValueError(f"could not identify enough dopamine neurons: found {len(dan)}")

        return cls(
            connectome,
            MushroomBodyRoles(
                kenyon_cells=kc,
                mbons=mbon,
                dopamine_neurons=dan,
            ),
            config=config,
        )

    @property
    def eligible_edge_mask(self):
        """Boolean mask for real anatomical KC -> MBON edges."""
        if self._eligible_edge_mask is not None:
            return self._eligible_edge_mask

        np = _require_numpy()
        n_edges = int(self.connectome.edge_count)
        mask = np.zeros(n_edges, dtype=np.bool_)
        mbon_mask = np.zeros(int(self.connectome.neuron_count), dtype=np.bool_)
        mbon_mask[self.roles.mbons] = True
        indptr = self.connectome.indptr
        posts = self.connectome.post_indices

        for pre_raw in self.roles.kenyon_cells:
            pre = int(pre_raw)
            start = int(indptr[pre])
            stop = int(indptr[pre + 1])
            if stop > start:
                mask[start:stop] = mbon_mask[posts[start:stop]]

        self._eligible_edge_mask = mask
        return mask

    def attach(self, brain, *, freeze_non_mushroom_edges: bool = True) -> dict[str, object]:
        """Attach this circuit and freeze non-KC->MBON plasticity by default."""
        if brain.connectome is not self.connectome:
            raise ValueError("brain and mushroom-body circuit use different connectomes")
        mask = self.eligible_edge_mask
        if freeze_non_mushroom_edges:
            brain.plasticity.plastic_mask[:] = mask
        else:
            brain.plasticity.plastic_mask[:] &= mask
        self._brain = brain
        return self.summary()

    def learn_from_reward(
        self,
        *,
        reward: float,
        rule: UsageRewardRule | None = None,
        normalizer: OutgoingBudgetNormalizer | None = None,
    ) -> dict[str, object]:
        """Apply delayed reward only to recently used KC->MBON edges."""
        if self._brain is None:
            raise RuntimeError("attach the circuit to a PlasticSparseFlyBrain first")
        report = self._brain.learn_from_reward(
            reward=float(reward),
            rule=rule or UsageRewardRule(),
            normalizer=normalizer,
        )
        report["mushroom_body"] = self.summary()
        return report

    def run_trial(
        self,
        *,
        stimulus_body_ids: Iterable[int],
        reward: float,
        duration_ms: float = 20.0,
        stimulus_rate_hz: float = 205.0,
        rule: UsageRewardRule | None = None,
        normalizer: OutgoingBudgetNormalizer | None = None,
    ) -> dict[str, object]:
        """Run one connectome trial and then deliver its scalar reward.

        The only task supervision is the final reward value. No target label,
        class index, or answer is injected into the neural state.
        """
        if self._brain is None:
            raise RuntimeError("attach the circuit to a PlasticSparseFlyBrain first")
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if stimulus_rate_hz < 0.0:
            raise ValueError("stimulus_rate_hz must be >= 0")

        np = _require_numpy()
        brain = self._brain
        stimulus_body_ids = tuple(int(x) for x in stimulus_body_ids)
        previous_tracking = brain.set_plasticity_tracking(True)
        try:
            brain.reset()
            stimulus_indices = brain.indices_for_ids(stimulus_body_ids)
            steps = max(1, int(math.ceil(float(duration_ms) / brain.params.dt_ms)))
            kc_mask = np.zeros(brain.connectome.neuron_count, dtype=np.bool_)
            kc_mask[self.roles.kenyon_cells] = True
            mbon_mask = np.zeros(brain.connectome.neuron_count, dtype=np.bool_)
            mbon_mask[self.roles.mbons] = True
            dan_mask = np.zeros(brain.connectome.neuron_count, dtype=np.bool_)
            dan_mask[self.roles.dopamine_neurons] = True

            kc_spikes = 0
            mbon_spikes = 0
            dan_spikes = 0
            total_spikes = 0
            for _ in range(steps):
                fired, _ = brain.step(
                    stimulus_indices=stimulus_indices,
                    stimulus_rate_hz=float(stimulus_rate_hz),
                )
                if len(fired):
                    total_spikes += int(len(fired))
                    kc_spikes += int(kc_mask[fired].sum())
                    mbon_spikes += int(mbon_mask[fired].sum())
                    dan_spikes += int(dan_mask[fired].sum())

            learning = self.learn_from_reward(
                reward=float(reward),
                rule=rule,
                normalizer=normalizer,
            )
            return {
                "duration_ms": float(duration_ms),
                "stimulus_body_ids": list(int(x) for x in stimulus_body_ids),
                "reward": float(reward),
                "spikes": {
                    "total": total_spikes,
                    "kenyon_cells": kc_spikes,
                    "mbons": mbon_spikes,
                    "dopamine_neurons": dan_spikes,
                },
                "learning": learning,
            }
        finally:
            brain.set_plasticity_tracking(previous_tracking)

    def summary(self) -> dict[str, object]:
        return {
            "type": "connectome_mushroom_body",
            "kenyon_cell_count": int(len(self.roles.kenyon_cells)),
            "mbon_count": int(len(self.roles.mbons)),
            "dopamine_neuron_count": int(len(self.roles.dopamine_neurons)),
            "kc_to_mbon_edge_count": int(self.eligible_edge_mask.sum()),
            "plasticity_scope": "anatomical KC -> MBON edges only",
            "dopamine_role": "reported circuit activity; reward remains an external trial signal",
        }


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mushroom-body learning needs NumPy") from exc
    return np


def _annotation_text(metadata: dict[str, object], index: int) -> str:
    values = []
    for value in metadata.values():
        item = value[index]
        if item is not None:
            values.append(str(item))
    return " | ".join(values).lower()
