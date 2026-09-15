from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'MaleCNS output learning needs NumPy. Install with: '
            'python -m pip install -e ".[malecns]"'
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class OutputReadoutConfig:
    learning_rate: float = 0.08
    l2: float = 1e-4
    feature_scale_hz: float = 100.0
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        if self.l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        if self.feature_scale_hz <= 0.0:
            raise ValueError("feature_scale_hz must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True, slots=True)
class OutputPopulation:
    """A selected set of real MaleCNS neurons used only as the external readout surface."""

    body_ids: tuple[int, ...]
    indices: object
    source: str

    @classmethod
    def from_body_ids(cls, connectome, body_ids: Iterable[int]) -> "OutputPopulation":
        np = _require_numpy()
        ids = tuple(int(x) for x in body_ids)
        if not ids:
            raise ValueError("output population must contain at least one neuron")
        indices = np.asarray([connectome.index_of(x) for x in ids], dtype=np.int32)
        return cls(body_ids=ids, indices=indices, source="explicit_body_ids")

    @classmethod
    def from_metadata(
        cls,
        connectome,
        *,
        field: str,
        values: Sequence[str],
        limit: int | None = None,
    ) -> "OutputPopulation":
        np = _require_numpy()
        accepted = tuple(str(x) for x in values)
        if not accepted:
            raise ValueError("values must not be empty")
        if field == "consensus_nt":
            metadata = np.asarray(connectome.consensus_nt, dtype=object)
        else:
            try:
                metadata = np.asarray(connectome.metadata[field], dtype=object)
            except KeyError as exc:
                raise KeyError(f"metadata field not loaded: {field}") from exc

        mask = np.zeros(connectome.neuron_count, dtype=np.bool_)
        for value in accepted:
            mask |= metadata == value
        indices = np.flatnonzero(mask).astype(np.int32, copy=False)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            indices = indices[: int(limit)]
        if len(indices) == 0:
            raise ValueError(
                f"no MaleCNS neurons matched {field} in {accepted}; "
                "inspect metadata values before selecting a population"
            )
        ids = tuple(int(connectome.body_ids[i]) for i in indices)
        return cls(body_ids=ids, indices=indices, source=f"metadata:{field}={accepted}")


@dataclass(frozen=True, slots=True)
class ReadoutTrainResult:
    target: str
    prediction_before: str
    prediction_after: str
    loss: float
    confidence_after: float


class PopulationReadout:
    """Small online linear decoder for a real-neuron output population.

    This decoder is deliberately external to the CNS. It is trained first on
    easy labelled examples, then frozen. Once frozen, only the CNS plasticity
    is allowed to change, so later improvements cannot be attributed to a
    moving classifier.
    """

    def __init__(
        self,
        population: OutputPopulation,
        labels: Sequence[str],
        *,
        config: OutputReadoutConfig | None = None,
    ) -> None:
        np = _require_numpy()
        self.np = np
        self.population = population
        self.labels = tuple(str(x) for x in labels)
        if len(self.labels) < 2:
            raise ValueError("at least two output labels are required")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("output labels must be unique")
        self._label_to_index = {label: i for i, label in enumerate(self.labels)}
        self.config = config or OutputReadoutConfig()
        rng = np.random.default_rng(self.config.seed)
        self.weights = rng.normal(
            0.0,
            1e-3,
            size=(len(self.labels), len(self.population.body_ids)),
        ).astype(np.float32)
        self.bias = np.zeros(len(self.labels), dtype=np.float32)
        self.frozen = False
        self.train_steps = 0

    def features_from_counts(self, spike_counts, *, duration_ms: float):
        np = self.np
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        counts = np.asarray(spike_counts, dtype=np.float32)
        if counts.shape != (len(self.population.body_ids),):
            raise ValueError("spike_counts shape does not match output population")
        rates = counts * (1000.0 / float(duration_ms))
        # log compression avoids a handful of high-rate neurons dominating the decoder.
        return np.log1p(rates) / np.log1p(self.config.feature_scale_hz)

    def _logits(self, features):
        np = self.np
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (len(self.population.body_ids),):
            raise ValueError("feature shape does not match output population")
        return (self.weights @ x + self.bias) / self.config.temperature

    def predict_proba(self, features):
        np = self.np
        logits = self._logits(features)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def predict(self, features) -> tuple[str, float]:
        probs = self.predict_proba(features)
        idx = int(self.np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def train(self, features, *, target: str) -> ReadoutTrainResult:
        np = self.np
        if self.frozen:
            raise RuntimeError("output decoder is frozen")
        try:
            target_idx = self._label_to_index[target]
        except KeyError as exc:
            raise KeyError(f"unknown output label: {target}") from exc

        x = np.asarray(features, dtype=np.float32)
        before, _ = self.predict(x)
        probs = self.predict_proba(x)
        loss = -float(np.log(max(float(probs[target_idx]), 1e-12)))

        grad_logits = probs.astype(np.float32, copy=True)
        grad_logits[target_idx] -= 1.0
        lr = self.config.learning_rate
        grad_w = grad_logits[:, None] * x[None, :]
        if self.config.l2:
            grad_w += self.config.l2 * self.weights
        self.weights -= lr * grad_w
        self.bias -= lr * grad_logits
        self.train_steps += 1

        after, confidence = self.predict(x)
        return ReadoutTrainResult(
            target=target,
            prediction_before=before,
            prediction_after=after,
            loss=loss,
            confidence_after=confidence,
        )

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def summary(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "population_size": len(self.population.body_ids),
            "population_source": self.population.source,
            "frozen": self.frozen,
            "train_steps": self.train_steps,
        }
