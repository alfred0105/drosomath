import math
import unittest

import numpy as np

from drosomath.malecns.sequence_memory import TwoCueSequenceSession


class _FakeConnectome:
    neuron_count = 16


class _FakeParams:
    dt_ms = 0.2


class _FakeInterface:
    def __init__(self, connectome):
        self.connectome = connectome
        self.symbol_interface = type(
            "_FakeSymbolInterface",
            (),
            {"decision_surface": type("_FakeDecisionSurface", (), {"decide": lambda *_args, **_kwargs: "A"})()},
        )()
        self.output_populations = {symbol: np.asarray([index], dtype=np.int32) for index, symbol in enumerate("ABCD")}
        self._inputs = {symbol: np.asarray([index + 4], dtype=np.int32) for index, symbol in enumerate("ABCD")}
        self._inputs["GO"] = np.asarray([8], dtype=np.int32)

    def population_for_input(self, symbol):
        return self._inputs[symbol]


class _FakeBrain:
    def __init__(self):
        self.connectome = _FakeConnectome()
        self.params = _FakeParams()
        self.calls = []
        self.tracking = False

    def set_plasticity_tracking(self, enabled):
        previous = self.tracking
        self.tracking = bool(enabled)
        return previous

    def reset(self):
        pass

    def step(self, *, stimulus_indices, stimulus_rate_hz):
        self.calls.append((tuple(int(value) for value in stimulus_indices), float(stimulus_rate_hz)))
        return np.empty(0, dtype=np.int32), None


class SequenceStimulusRateTests(unittest.TestCase):
    def _session(self):
        brain = _FakeBrain()
        return brain, TwoCueSequenceSession(brain, _FakeInterface(brain.connectome))

    def test_default_rates_preserve_205_hz_behavior(self):
        brain, session = self._session()
        session.run_trial("A", "B")
        self.assertEqual([rate for _, rate in brain.calls], [205.0] * 300)

    def test_custom_item_rate_changes_first_and_second_only(self):
        brain, session = self._session()
        session = TwoCueSequenceSession(session.brain, session.interface, item_stimulus_rate_hz=25.0)
        session.run_trial("A", "B")
        rates = [rate for _, rate in brain.calls]
        self.assertEqual(rates[:100], [25.0] * 100)
        self.assertEqual(rates[100:200], [25.0] * 100)
        self.assertEqual(rates[200:], [205.0] * 100)

    def test_custom_go_rate_is_independent(self):
        brain, session = self._session()
        session = TwoCueSequenceSession(session.brain, session.interface, go_stimulus_rate_hz=17.0)
        session.run_trial("A", "B")
        rates = [rate for _, rate in brain.calls]
        self.assertEqual(rates[:200], [205.0] * 200)
        self.assertEqual(rates[200:], [17.0] * 100)

    def test_rates_must_be_finite_and_nonnegative(self):
        brain, interface = self._session()
        for kwargs in (
            {"item_stimulus_rate_hz": -1.0},
            {"go_stimulus_rate_hz": -1.0},
            {"item_stimulus_rate_hz": math.nan},
            {"go_stimulus_rate_hz": math.inf},
        ):
            with self.assertRaises(ValueError):
                TwoCueSequenceSession(brain, interface.interface if hasattr(interface, "interface") else interface, **kwargs)


if __name__ == "__main__":
    unittest.main()
