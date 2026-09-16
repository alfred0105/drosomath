import tempfile
import unittest
from pathlib import Path

from drosomath.io import ConnectomeLoadConfig, infer_neuron_ids, load_synapses_csv


class ConnectomeLoaderTests(unittest.TestCase):
    def test_loads_weighted_edge_list_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text(
                "pre_id,post_id,weight\n1,2,0.4\n1,2,0.8\n2,3,0.6\n",
                encoding="utf-8",
            )
            synapses = load_synapses_csv(path)

        self.assertEqual(len(synapses), 2)
        self.assertAlmostEqual(synapses[0].weight, 0.4)
        self.assertEqual(infer_neuron_ids(synapses), (1, 2, 3))

    def test_can_convert_contact_count_to_weight_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text(
                "pre,post,contacts\n1,2,2\n2,3,10\n",
                encoding="utf-8",
            )
            config = ConnectomeLoadConfig(
                pre_column="pre",
                post_column="post",
                weight_column=None,
                count_column="contacts",
                count_scale=0.1,
                min_count=5,
            )
            synapses = load_synapses_csv(path, config=config)

        self.assertEqual(len(synapses), 1)
        self.assertEqual((synapses[0].pre_id, synapses[0].post_id), (2, 3))
        self.assertAlmostEqual(synapses[0].weight, 1.0)


if __name__ == "__main__":
    unittest.main()
