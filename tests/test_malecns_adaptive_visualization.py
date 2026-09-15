import json
import tempfile
import unittest
from pathlib import Path


class MaleCNSAdaptiveVisualizationTests(unittest.TestCase):
    def test_adaptive_config_and_html_report(self) -> None:
        from drosomath.malecns.adaptive_training import AdaptiveTrainingConfig
        from drosomath.malecns.visualize_training import build_html, render

        config = AdaptiveTrainingConfig(brain_trials=4, validation_trials_per_class=2)
        self.assertEqual(config.brain_trials, 4)

        report = {
            "experiment": "test_malecns_adaptive",
            "connectome": {"neuron_count": 10, "edge_count": 20},
            "evaluation": {
                "before_accuracy": 0.5,
                "after_accuracy": 0.75,
                "delta_accuracy": 0.25,
            },
            "brain_training": {
                "training_accuracy": 0.75,
                "rows": [
                    {"trial": 1, "target": "LEFT", "prediction": "RIGHT", "correct": False, "reward": -1.0, "confidence": 0.55, "output_spikes": 2, "edge_updates": 4},
                    {"trial": 2, "target": "LEFT", "prediction": "LEFT", "correct": True, "reward": 1.0, "confidence": 0.75, "output_spikes": 3, "edge_updates": 5},
                ],
            },
            "challenge": {
                "calibration_candidates": [
                    {"fraction": 0.4, "rate_hz": 180.0, "accuracy": 0.75, "mean_output_spikes": 3.0}
                ]
            },
            "plasticity": {
                "changed_edges": 5,
                "plastic_edges": 10,
                "mean_multiplier": 1.01,
                "mean_stability": 0.02,
                "multiplier_quantiles": {"min": 0.9, "median": 1.0, "max": 1.1},
            },
        }
        page = build_html(report)
        self.assertIn("MaleCNS learning report", page)
        self.assertIn("Rolling training accuracy", page)
        self.assertIn("test_malecns_adaptive", page)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "result.json"
            output = root / "result.html"
            source.write_text(json.dumps(report), encoding="utf-8")
            rendered = render(source, output)
            self.assertEqual(rendered, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
