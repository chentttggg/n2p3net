"""Public integration contracts for complete shared-variable partitions."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from data.candidate_task import MembershipDecision
from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.latent_response_reference import ResponseState, selection_design
from research.response_frontier import (
    FrontierBudgetExceeded,
    LogFactor,
    PointFactors,
    Roster,
    StateKey,
    sum_product,
)
from research.structured_response_frontier import (
    NativeSelection,
    StoredNoiseCarrier,
    StructuredPointwiseFrontier,
    combine_record_factors,
)
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest


def test_complete_frontier_retains_weak_candidates_and_named_budget_outcome():
    q = Roster("q", (StateKey("a", 0), StateKey("b", 0)), np.array([0., -900.]))
    h = Roster("h", (StateKey("h", 0), StateKey("h", 1)), np.log([.2, .8]))
    point = PointFactors((q, h), (LogFactor((0, 1), np.array([[0., 1.], [901., 900.]])),),
        0, ("b", "a"), 0., (1,), {})
    with pytest.raises(FrontierBudgetExceeded):
        sum_product(point, maximum_intermediate_entries=3)
    result = sum_product(point, maximum_intermediate_entries=4)
    expected = np.array([logsumexp(np.log([.2, .8])+[1., 0.]),
                         logsumexp(np.log([.2, .8])+[0., 1.])])
    np.testing.assert_allclose(result.log_joint_mass_by_q, expected)
    np.testing.assert_array_equal(result.current_log_prior_mass_by_q, [-900., 0.])
    assert result.current_q_values == ("b", "a")


def test_integrated_native_records_share_q_and_source_but_have_local_shapes_and_clocks():
    vocabulary = ("red", "green", "blue")
    rosters = (
        Roster("shape-A", tuple(StateKey("h", i) for i in range(2)), np.log([.3, .7])),
        Roster("q", tuple(StateKey(q, "candidate") for q in vocabulary), np.log([.1, .6, .3])),
        Roster("shape-B", tuple(StateKey("h", i) for i in range(2)), np.log([.8, .2])),
    )
    basis = np.array([[0.], [1.], [.2]])
    dictionary = NativeResponseDictionary.from_source_matrix(basis, sample_rate_hz=8.,
        source_basis_sha256=array_digest(basis), terminal_values=np.zeros(1))
    shapes = (ResponseState(1., .8, 0., 1., .5), ResponseState(.7, 1.1, .125, 1., .5))
    mean = np.array([[.1, -.2], [.4, .2]])
    factor = np.array([[[.1], [.03]], [[-.04], [.08]]])
    u = np.array([.23])
    geometry = NoiseGeometry(("A", "B"), 8., "V", "declared_reference", "native", 2)
    engines, drifts = [], []
    for ordinal, (origin, count, shape_id) in enumerate(((50, 9, "shape-A"), (200, 11, "shape-B"))):
        decision = MembershipDecision.direct(vocabulary, vocabulary,
            event_ids=tuple(f"record{ordinal}:{q}" for q in vocabulary),
            repetition_indices=np.zeros(3, dtype=int), evidence_indices=np.arange(3),
            repetitions=1, timeline_complete=True)
        selection = NativeSelection(f"block-{ordinal}", decision, origin+np.array([0, 2, 4]),
            "q", shape_id, shapes)
        noise = FittedChannelNoise.from_parameters([[.1], [.2]], [[.9, .12], [.12, 1.3]],
            geometry=geometry, conditioning_samples=3, reference_id=f"noise-{ordinal}", family="private")
        observed = np.column_stack((np.sin(np.arange(count)*.3)*.2, np.cos(np.arange(count)*.2)*.1))
        engines.append(StructuredPointwiseFrontier(observed, dictionary, mean, factor,
            (selection,), StoredNoiseCarrier(noise, eta=.9), rosters, origin_sample=origin,
            current_variable_id="q", current_q_values=vocabulary,
            maximum_intermediate_entries=12, record_id=f"record-{ordinal}"))
        drifts.append(np.array([[.01, -.02], [.002*(ordinal+1), -.001]]))
    points = tuple(engine.factors(u, drift, len(engine.y))
        for engine, drift in zip(engines, drifts, strict=True))
    result = sum_product(combine_record_factors(points), maximum_intermediate_entries=12)
    # Independent full raw Gaussian enumeration, with one global prior product.
    bins = {q: [] for q in vocabulary}
    beta = mean+np.einsum("kcr,r->kc", factor, u)
    for coordinates in itertools.product(range(2), range(3), range(2)):
        value = sum(roster.log_mass[index] for roster, index in zip(rosters, coordinates, strict=True))
        q = vocabulary[coordinates[1]]
        for engine, drift in zip(engines, drifts, strict=True):
            selection = engine.selections[0]
            shape_index = coordinates[engine.variable_ids.index(selection.shape_variable)]
            design, _ = selection_design(dictionary, selection.onsets,
                selection.membership.membership[:, vocabulary.index(q)],
                origin_sample=engine.origin, observed_samples=len(engine.y), state=shapes[shape_index])
            times = (engine.origin+np.arange(len(engine.y)))/8.
            residual = engine.y-design@beta-drift[0]-times[:, None]*drift[1]
            white = engine.carrier.noise.whiten(residual).ravel()/np.sqrt(.9)
            value += engine.carrier.noise.whitening_logabsdet(len(engine.y))
            value -= .5*len(white)*np.log(.9*2*np.pi)+.5*float(white@white)
        bins[q].append(value)
    np.testing.assert_allclose(result.log_joint_mass_by_q,
        [logsumexp(bins[q]) for q in vocabulary], rtol=0., atol=5e-13)
    np.testing.assert_allclose(result.current_log_prior_mass_by_q, np.log([.1, .6, .3]))
    with pytest.raises(ValueError, match="twice"):
        combine_record_factors((points[0], points[0]))
