import json
import unittest

from drosomath.live import LiveSimulation, _dashboard_html


class LiveDashboardTests(unittest.TestCase):
    def test_snapshot_is_json_serializable_and_advances(self) -> None:
        simulation = LiveSimulation(hz=8.0)
        first = simulation.snapshot()
        for _ in range(9):
            latest = simulation.advance_once()

        self.assertGreater(latest["step"], first["step"])
        self.assertEqual(len(latest["neurons"]), 8)
        self.assertGreater(len(latest["synapses"]), 0)
        self.assertIn("mean_weight", latest["metrics"])
        self.assertIn("mean_stability", latest["metrics"])
        self.assertGreaterEqual(latest["completed_trials"], 1)
        json.dumps(latest)

    def test_controls_do_not_require_background_server(self) -> None:
        simulation = LiveSimulation(hz=8.0)
        simulation.control("pause")
        self.assertFalse(simulation.snapshot()["running"])
        old_hz = simulation.snapshot()["hz"]
        simulation.control("faster")
        self.assertGreater(simulation.snapshot()["hz"], old_hz)
        simulation.control("resume")
        self.assertTrue(simulation.snapshot()["running"])
        simulation.control("reset")
        self.assertEqual(simulation.snapshot()["step"], 0)

    def test_dashboard_asset_is_available(self) -> None:
        html = _dashboard_html()
        self.assertIn(b"DrosoMath Live", html)
        self.assertIn(b"/events", html)


if __name__ == "__main__":
    unittest.main()
