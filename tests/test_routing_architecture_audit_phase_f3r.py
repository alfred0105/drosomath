import inspect
import unittest

import numpy as np

from drosomath.malecns.loader import MaleCNSConnectome
from run_routing_architecture_audit_phase_f3r import (
    _role_evidence,
    bounded_reachability,
    run_seed_jobs,
    select_mb_routed_candidate,
)


def tiny_connectome():
    # Four ALPN projection neurons, four KCs, two MBONs, two DANs, and
    # additional unrelated cells.  All selections remain annotation-derived.
    n = 20
    rows = {
        0: [4, 5, 8], 1: [6, 7, 9], 2: [4, 6, 10], 3: [5, 7, 11],
        4: [12], 5: [12], 6: [13], 7: [13],
        8: [14], 9: [14], 10: [15], 11: [15],
    }
    indptr = [0]
    posts = []
    signed = []
    for index in range(n):
        values = rows.get(index, [])
        posts.extend(values)
        signed.extend([1.0] * len(values))
        indptr.append(len(posts))
    metadata = {
        "class": np.asarray(["ALPN"] * 4 + ["Kenyon_Cell"] * 4 + ["MBON"] * 2 + ["DAN"] * 2 + ["other"] * 8, dtype=object),
        "superclass": np.asarray(["cb_intrinsic"] * 12 + ["vnc_motor"] * 8, dtype=object),
        **{name: np.asarray(["x"] * n, dtype=object) for name in ("type", "flywireType", "hemibrainType", "subclass", "somaNeuromere", "rootSide")},
    }
    return MaleCNSConnectome(
        body_ids=np.arange(100, 100 + n, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        post_indices=np.asarray(posts, dtype=np.int32),
        synapse_counts=np.ones(len(posts), dtype=np.int32),
        signed_synapse_counts=np.asarray(signed, dtype=np.float32),
        outgoing_strength=np.bincount(np.repeat(np.arange(n), np.diff(indptr)), minlength=n).astype(np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["acetylcholine"] * n, dtype=object),
        metadata=metadata,
    )


class RoutingArchitectureAuditTests(unittest.TestCase):
    def test_annotation_classification_is_evidence_only(self):
        self.assertIn("projection_neuron_annotation", _role_evidence({"class": "ALPN", "superclass": "cb_intrinsic"}))
        self.assertIn("sensory_annotation", _role_evidence({"class": "olfactory", "superclass": "ol_sensory"}))
        self.assertIn("higher_order_or_mushroom_body_annotation", _role_evidence({"class": "Kenyon_Cell", "superclass": "cb_intrinsic"}))

    def test_candidate_selection_is_deterministic_and_disjoint(self):
        connectome = tiny_connectome()
        first, summary = select_mb_routed_candidate(connectome, seed=233, population_size=1)
        second, _ = select_mb_routed_candidate(connectome, seed=233, population_size=1)
        self.assertEqual({key: tuple(value) for key, value in first.items()}, {key: tuple(value) for key, value in second.items()})
        merged = np.concatenate(tuple(first.values()))
        self.assertEqual(len(np.unique(merged)), len(merged))
        self.assertTrue(summary["disjoint"])

    def test_topology_reachability_has_exact_bounded_hops(self):
        connectome = tiny_connectome()
        metrics, hop_sets = bounded_reachability(connectome, np.asarray([0], dtype=np.int32), {"kc": np.asarray([4, 5, 6, 7], dtype=np.int32)}, max_hops=2)
        self.assertEqual(metrics["kc"]["per_hop"][0]["newly_reached_neurons"], 2)
        self.assertEqual(metrics["kc"]["per_hop"][1]["newly_reached_neurons"], 0)
        self.assertEqual(hop_sets[0], {4, 5, 8})

    def test_audit_source_does_not_train_or_use_grammar(self):
        import run_routing_architecture_audit_phase_f3r as module

        source = inspect.getsource(module)
        self.assertNotIn("CONTEXTUAL_GRAMMAR", source)
        self.assertNotIn("learn=True", source)
        self.assertIn("ProcessPoolExecutor", source)

    def test_worker_contract_and_fixed_worker_choices(self):
        import run_routing_architecture_audit_phase_f3r as module

        source = inspect.getsource(module.run_seed_jobs)
        self.assertIn("workers not in (1, 2, 3)", source)
        self.assertIn("sorted(rows", source)


if __name__ == "__main__":
    unittest.main()
