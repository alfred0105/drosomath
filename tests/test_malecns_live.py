import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "MaleCNS live tests need NumPy")
class MaleCNSLiveTests(unittest.TestCase):
    def test_live_state_merges_training_event_and_synapse_telemetry(self) -> None:
        from drosomath.malecns.live_training import LiveTrainingState

        class FakeBrain:
            def live_telemetry_snapshot(self):
                return {
                    "step": 42,
                    "fired_neuron_ids": [101],
                    "fired_neuron_count": 1,
                    "transferred_synapses": 7,
                    "active_edges": [
                        {
                            "pre_id": 101,
                            "post_id": 202,
                            "signal_mv": 1.5,
                            "multiplier": 1.1,
                            "stability": 0.2,
                        }
                    ],
                }

        state = LiveTrainingState(history_limit=4)
        state.update(
            {
                "phase": "brain_training",
                "status": "running",
                "completed_brain_trials": 3,
                "total_brain_trials": 10,
                "running_accuracy": 2 / 3,
                "last_trial": {
                    "trial": 3,
                    "correct": True,
                    "reward": 1.0,
                    "confidence": 0.8,
                    "output_spikes": 9,
                    "edge_updates": 123,
                },
            },
            FakeBrain(),
        )
        snap = state.snapshot()
        self.assertEqual(snap["telemetry"]["step"], 42)
        self.assertEqual(snap["telemetry"]["transferred_synapses"], 7)
        self.assertEqual(snap["telemetry"]["active_edges"][0]["pre_id"], 101)
        self.assertEqual(snap["history"][-1]["edge_updates"], 123)

    def test_dashboard_resource_contains_realtime_activity_view(self) -> None:
        from drosomath.malecns.live_training import _dashboard_html

        html = _dashboard_html().decode("utf-8")
        self.assertIn("Realtime activity", html)
        self.assertIn("Active synapses", html)
        self.assertIn("EventSource('/events')", html)


if __name__ == "__main__":
    unittest.main()
