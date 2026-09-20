import unittest


class MaleCNSRouteSelectionTests(unittest.TestCase):
    def _graph(self):
        import numpy as np
        from drosomath.malecns.loader import MaleCNSConnectome

        # 10(input) -> 20(intermediate) -> 30/40(descending), plus an
        # unrelated high-outgoing descending cell 50.
        return MaleCNSConnectome(
            body_ids=np.asarray([10, 20, 30, 40, 50], dtype=np.int64),
            indptr=np.asarray([0, 1, 3, 3, 3, 3], dtype=np.int64),
            post_indices=np.asarray([1, 2, 3], dtype=np.int32),
            synapse_counts=np.asarray([8, 4, 4], dtype=np.int32),
            signed_synapse_counts=np.asarray([8.0, 4.0, 4.0], dtype=np.float32),
            outgoing_strength=np.asarray([8.0, 8.0, 0.0, 0.0, 100.0], dtype=np.float32),
            presynaptic_sign=np.ones(5, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * 5, dtype=object),
            metadata={
                "superclass": np.asarray(
                    [
                        "visual_projection",
                        "intermediate",
                        "descending_neuron",
                        "descending_neuron",
                        "descending_neuron",
                    ],
                    dtype=object,
                )
            },
            min_connection_synapses=1,
        )

    def test_two_hop_route_prefers_downstream_outputs(self):
        from drosomath.malecns.first_training import choose_route_aware_output_population

        output, info = choose_route_aware_output_population(
            self._graph(),
            [10],
            output_population_size=2,
            max_hops=2,
        )

        self.assertEqual(set(output.body_ids), {30, 40})
        self.assertEqual(info["selection"], "route_aware_2hop")
        self.assertEqual(info["route_direct_edge_count"], 0)
        self.assertEqual(info["route_second_hop_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
