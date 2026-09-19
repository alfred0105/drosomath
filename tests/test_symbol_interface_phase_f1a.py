from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    NO_DECISION,
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolSession,
)
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain import PlasticStateConfig


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    indptr = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    signed = np.where(np.arange(2 * n) % 3 == 0, -2.0, 3.0).astype(np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
        indptr=indptr,
        post_indices=posts.astype(np.int32),
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.full(n, 5.0, dtype=np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * n, dtype=object),
        min_connection_synapses=5,
    )


@pytest.fixture(scope="module")
def connectome():
    return make_connectome()


@pytest.fixture(scope="module")
def interface(connectome):
    return SymbolInterface(
        connectome,
        SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
        ),
    )


def test_exact_vocabulary(interface):
    assert tuple(interface.encoder.symbols) == SYMBOLS == ("A", "B", "C", "D")


def test_same_seed_is_exactly_reproducible(connectome):
    config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=19)
    left = SymbolInterface(connectome, config)
    right = SymbolInterface(connectome, config)
    for symbol in SYMBOLS:
        np.testing.assert_array_equal(left.sensory_populations[symbol], right.sensory_populations[symbol])
        np.testing.assert_array_equal(left.output_populations[symbol], right.output_populations[symbol])


def test_sensory_size_and_no_duplicates(interface):
    values = np.concatenate(tuple(interface.sensory_populations.values()))
    assert all(len(interface.sensory_populations[symbol]) == 8 for symbol in SYMBOLS)
    assert len(np.unique(values)) == len(values)


def test_sensory_mutual_disjoint(interface):
    assert all(
        not np.intersect1d(interface.sensory_populations[left], interface.sensory_populations[right]).size
        for left in SYMBOLS
        for right in SYMBOLS
        if left < right
    )


def test_output_equal_size_and_mutual_disjoint(interface):
    values = np.concatenate(tuple(interface.output_populations.values()))
    assert all(len(interface.output_populations[symbol]) == 8 for symbol in SYMBOLS)
    assert len(np.unique(values)) == len(values)


def test_sensory_and_output_disjoint(interface):
    sensory = np.concatenate(tuple(interface.sensory_populations.values()))
    output = np.concatenate(tuple(interface.output_populations.values()))
    assert not np.intersect1d(sensory, output).size


def test_indices_are_real_valid_connectome_indices(connectome, interface):
    all_indices = np.concatenate(
        tuple(interface.sensory_populations.values()) + tuple(interface.output_populations.values())
    )
    assert int(all_indices.min()) >= 0
    assert int(all_indices.max()) < connectome.neuron_count
    body_ids = np.concatenate(
        tuple(interface.encoder.body_id_populations.values())
        + tuple(interface.decision_surface.body_id_populations.values())
    )
    assert set(int(x) for x in body_ids).issubset(set(int(x) for x in connectome.body_ids))


def test_unknown_symbol_fails_clearly(interface):
    with pytest.raises(KeyError, match="unknown symbol"):
        interface.encoder.indices_for("?")


def test_all_zero_is_no_decision(interface):
    assert interface.decision_surface.decide({symbol: 0.0 for symbol in SYMBOLS}) == NO_DECISION


def test_unique_highest_is_selected(interface):
    rates = {symbol: 0.0 for symbol in SYMBOLS}
    rates["C"] = 4.0
    assert interface.decision_surface.decide(rates) == "C"


def test_exact_tie_has_no_arbitrary_selection(interface):
    rates = {symbol: 0.0 for symbol in SYMBOLS}
    rates["A"] = rates["B"] = 4.0
    assert interface.decision_surface.decide(rates) == NO_DECISION


def test_decision_consumes_no_rng(interface):
    rng = np.random.default_rng(77)
    before = copy.deepcopy(rng.bit_generator.state)
    interface.decision_surface.decide({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})
    after = copy.deepcopy(rng.bit_generator.state)
    assert before == after


def test_allocation_does_not_consume_brain_rng(connectome):
    rng = np.random.default_rng(123)
    before = copy.deepcopy(rng.bit_generator.state)
    SymbolInterface(connectome, SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7))
    assert before == rng.bit_generator.state


def _make_brain(connectome):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=11,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=11),
    )


def test_observational_session_preserves_persistent_learning_state(connectome, interface):
    brain = _make_brain(connectome)
    before = {
        "multiplier": brain.plasticity.multiplier.copy(),
        "usage": brain.plasticity.usage_ema.copy(),
        "eligibility": brain.plasticity.eligibility.copy(),
        "stability": brain.plasticity.stability.copy(),
        "mask": brain.plasticity.plastic_mask.copy(),
        "overrides": {
            key: value.copy() for key, value in brain.plasticity.allocation_overrides().items()
        },
        "budget": brain.plasticity.plastic_edge_count,
    }
    result = SymbolSession(brain, interface).present(symbol="A", duration_ms=2.0, stimulus_rate_hz=205.0)
    assert result.input_symbol == "A"
    np.testing.assert_array_equal(brain.plasticity.multiplier, before["multiplier"])
    np.testing.assert_array_equal(brain.plasticity.usage_ema, before["usage"])
    np.testing.assert_array_equal(brain.plasticity.eligibility, before["eligibility"])
    np.testing.assert_array_equal(brain.plasticity.stability, before["stability"])
    np.testing.assert_array_equal(brain.plasticity.plastic_mask, before["mask"])
    assert brain.plasticity.plastic_edge_count == before["budget"]
    for key, values in before["overrides"].items():
        np.testing.assert_array_equal(brain.plasticity.allocation_overrides()[key], values)


def test_learn_true_is_not_silently_implemented(connectome, interface):
    brain = _make_brain(connectome)
    with pytest.raises(NotImplementedError, match="F.1B"):
        SymbolSession(brain, interface).present(symbol="A", learn=True)


def test_no_trainable_external_decoder_in_symbol_module():
    source = inspect.getsource(__import__("drosomath.malecns.symbol_interface", fromlist=["x"]))
    assert "PopulationReadout" not in source
    assert "softmax" not in source.lower()
    assert "linear" not in source.lower()
