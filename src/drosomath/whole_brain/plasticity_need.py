"""Sparse, task-independent evidence for missing plastic routes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlasticityNeedConfig:
    """Conservative accumulation settings for structural need."""

    need_decay: float = 0.90
    success_decay: float = 0.25
    minimum_observations: int = 3
    promotion_threshold: float = 1.0
    max_tracked_candidates: int = 4096
    prune_threshold: float = 1e-6


@dataclass(frozen=True, slots=True)
class PlasticityNeedRecord:
    edge_index: int
    need_score: float
    observation_count: int
    last_seen_event: int


class PlasticityNeedTracker:
    """Bounded sparse state keyed by anatomical edge index.

    The tracker deliberately stores only edges observed as useful candidates;
    it never allocates an array indexed by the full connectome edge count.
    """

    def __init__(self, config: PlasticityNeedConfig | None = None) -> None:
        self.config = config or PlasticityNeedConfig()
        if not 0.0 <= self.config.need_decay <= 1.0:
            raise ValueError("need_decay must be in [0, 1]")
        if not 0.0 <= self.config.success_decay <= 1.0:
            raise ValueError("success_decay must be in [0, 1]")
        if self.config.minimum_observations < 1:
            raise ValueError("minimum_observations must be >= 1")
        if self.config.max_tracked_candidates < 1:
            raise ValueError("max_tracked_candidates must be >= 1")
        self._records: dict[int, PlasticityNeedRecord] = {}
        self.event_count = 0

    @property
    def tracked_count(self) -> int:
        return len(self._records)

    def records(self) -> tuple[PlasticityNeedRecord, ...]:
        return tuple(self._records[edge] for edge in sorted(self._records))

    def advance(self, *, success: bool = False) -> None:
        """Advance time and decay old evidence before observing a new event."""
        self.event_count += 1
        factor = self.config.success_decay if success else self.config.need_decay
        updated: dict[int, PlasticityNeedRecord] = {}
        for edge, record in self._records.items():
            score = record.need_score * factor
            if score > self.config.prune_threshold:
                updated[edge] = PlasticityNeedRecord(
                    edge, score, record.observation_count, record.last_seen_event
                )
        self._records = updated

    def observe(self, edge_indices, need_weights) -> None:
        """Add one failure observation for each sparse candidate edge."""
        edges = list(int(edge) for edge in edge_indices)
        weights = list(float(weight) for weight in need_weights)
        if len(edges) != len(weights):
            raise ValueError("edge_indices and need_weights must have equal length")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("need weights must be non-negative")
        # One observation per edge per event, while duplicate route evidence
        # still contributes additively to the score.
        grouped: dict[int, float] = {}
        for edge, weight in zip(edges, weights):
            if weight > 0.0:
                grouped[edge] = grouped.get(edge, 0.0) + weight
        for edge in sorted(grouped):
            old = self._records.get(edge)
            self._records[edge] = PlasticityNeedRecord(
                edge,
                (old.need_score if old else 0.0) + grouped[edge],
                (old.observation_count if old else 0) + 1,
                self.event_count,
            )
        self._trim()

    def ready_candidates(self, *, threshold: float | None = None,
                         minimum_observations: int | None = None,
                         count: int | None = None):
        threshold = self.config.promotion_threshold if threshold is None else float(threshold)
        minimum = self.config.minimum_observations if minimum_observations is None else int(minimum_observations)
        ready = [record for record in self._records.values()
                 if record.need_score >= threshold and record.observation_count >= minimum]
        ready.sort(key=lambda record: (-record.need_score, record.edge_index))
        if count is not None:
            if count < 0:
                raise ValueError("count must be >= 0")
            ready = ready[:int(count)]
        np_edges = __import__("numpy").asarray([r.edge_index for r in ready], dtype="int32")
        np_scores = __import__("numpy").asarray([r.need_score for r in ready], dtype="float32")
        return np_edges, np_scores

    def discard(self, edge_indices) -> None:
        """Forget candidates that are no longer frozen after promotion."""
        for edge in edge_indices:
            self._records.pop(int(edge), None)

    def checkpoint_payload(self, np=None) -> dict[str, object]:
        if np is None:
            np = __import__("numpy")
        records = self.records()
        return {
            "edge_indices": np.asarray([r.edge_index for r in records], dtype=np.int32),
            "scores": np.asarray([r.need_score for r in records], dtype=np.float32),
            "observations": np.asarray([r.observation_count for r in records], dtype=np.int32),
            "last_seen": np.asarray([r.last_seen_event for r in records], dtype=np.int64),
            "event_count": np.asarray([self.event_count], dtype=np.int64),
        }

    def restore_from_checkpoint(self, payload) -> None:
        edges = payload["edge_indices"]
        scores = payload["scores"]
        observations = payload["observations"]
        last_seen = payload["last_seen"]
        if not (len(edges) == len(scores) == len(observations) == len(last_seen)):
            raise ValueError("inconsistent plasticity-need checkpoint lengths")
        self._records = {
            int(edge): PlasticityNeedRecord(
                int(edge), float(score), int(observation), int(seen)
            )
            for edge, score, observation, seen in zip(edges, scores, observations, last_seen)
            if float(score) > self.config.prune_threshold
        }
        self.event_count = int(payload.get("event_count", [0])[0])
        self._trim()

    def _trim(self) -> None:
        if len(self._records) <= self.config.max_tracked_candidates:
            return
        keep = sorted(
            self._records.values(),
            key=lambda record: (-record.need_score, record.edge_index),
        )[: self.config.max_tracked_candidates]
        self._records = {record.edge_index: record for record in keep}
