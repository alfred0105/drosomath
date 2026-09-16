import importlib.util
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
ARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(
    NUMPY_AVAILABLE and ARROW_AVAILABLE,
    "MaleCNS morphology tests need NumPy and PyArrow",
)
class MaleCNSMorphologyTests(unittest.TestCase):
    def _write_annotations(self, root: Path) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        from drosomath.malecns.download import FILES

        feather.write_feather(
            pa.table(
                {
                    "bodyId": pa.array([10, 20, 30, 999], type=pa.int64()),
                    "superclass": pa.array(["sensory", "central", "descending_neuron", None]),
                    "somaLocation": pa.array(
                        [[100.0, 200.0, 300.0], None, [700.0, 800.0, 900.0], None],
                        type=pa.list_(pa.float32()),
                    ),
                    "tosomaLocation": pa.array(
                        [None, [400.0, 500.0, 600.0], None, None],
                        type=pa.list_(pa.float32()),
                    ),
                }
            ),
            root / FILES["annotations"],
        )

    def test_annotation_positions_align_to_body_ids_and_fallback_to_tosoma(self) -> None:
        import numpy as np

        from drosomath.malecns.morphology import MorphologySpace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_annotations(root)
            space = MorphologySpace.from_annotations(
                np.asarray([10, 20, 30], dtype=np.int64),
                root,
            )

        self.assertEqual(space.positioned_count, 3)
        np.testing.assert_allclose(space.xyz_voxels[0], [100.0, 200.0, 300.0])
        np.testing.assert_allclose(space.xyz_voxels[1], [400.0, 500.0, 600.0])
        np.testing.assert_allclose(space.xyz_voxels[2], [700.0, 800.0, 900.0])
        self.assertEqual(len(space.normalized(20)), 3)
        cloud = space.point_cloud(max_points=100)
        self.assertEqual(cloud["positioned_neurons"], 3)
        self.assertEqual(cloud["rendered_neurons"], 3)

    def test_telemetry_receives_real_coordinate_endpoints(self) -> None:
        import numpy as np

        from drosomath.malecns.morphology import MorphologySpace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_annotations(root)
            space = MorphologySpace.from_annotations(
                np.asarray([10, 20, 30], dtype=np.int64),
                root,
            )
            annotated = space.annotate_telemetry(
                {
                    "step": 3,
                    "fired_neuron_ids": [10, 30],
                    "fired_neuron_count": 2,
                    "transferred_synapses": 1,
                    "active_edges": [
                        {
                            "pre_id": 10,
                            "post_id": 20,
                            "signal_mv": 1.25,
                        }
                    ],
                }
            )

        self.assertEqual(len(annotated["fired_positions"]), 2)
        edge = annotated["active_edges"][0]
        self.assertEqual(len(edge["pre_xyz"]), 3)
        self.assertEqual(len(edge["post_xyz"]), 3)

    def test_cached_swc_is_parsed_without_network_and_shares_coordinate_frame(self) -> None:
        import numpy as np

        from drosomath.malecns.morphology import MorphologySpace, SkeletonCache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_annotations(root)
            space = MorphologySpace.from_annotations(
                np.asarray([10, 20, 30], dtype=np.int64),
                root,
            )
            cache = SkeletonCache(space, root)
            path = cache.path_for(10)
            path.write_text(
                "# synthetic SWC\n"
                "1 1 100 200 300 1 -1\n"
                "2 3 120 220 320 1 1\n"
                "3 3 150 240 330 1 2\n",
                encoding="utf-8",
            )
            payload = cache.payload(10, max_segments=10)

        self.assertEqual(payload["body_id"], 10)
        self.assertEqual(payload["total_segments"], 2)
        self.assertEqual(payload["rendered_segments"], 2)
        self.assertEqual(len(payload["segments"][0]), 6)


if __name__ == "__main__":
    unittest.main()
