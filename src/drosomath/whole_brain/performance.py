"""Optional, state-free timing telemetry for whole-brain experiments."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time


class TimingProfiler:
    """Accumulate wall-clock sections without touching model or RNG state.

    The disabled path is intentionally inert.  Callers only pass an instance
    when timing is explicitly requested, so normal training has no timing
    calls or per-trial timing dictionaries.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._totals[str(name)] += float(seconds)
        self._counts[str(name)] += 1

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def absorb(self, timings: dict[str, float] | None) -> None:
        if not self.enabled or not timings:
            return
        for name, seconds in timings.items():
            self.add(name, float(seconds))

    def report(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "seconds": {name: float(value) for name, value in sorted(self._totals.items())},
            "counts": {name: int(value) for name, value in sorted(self._counts.items())},
        }


__all__ = ["TimingProfiler"]
