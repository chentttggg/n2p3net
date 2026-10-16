"""Independent exact covariance checks and explicit factor-codec replay."""

import copy
import json
from fractions import Fraction as Q

import numpy as np
import pytest

from research import channel_noise_mechanism as cm
from research import shared_stationary_factor as factor
from research.execution import UnidentifiableFitError
from research.stationary_noise_artifact import stationary_root
from transfer.noise_model import metadata_digest
from transfer.temporal_noise import TemporalNoise


def _matmul(a, b):
    return [[sum((x*y for x, y in zip(row, col, strict=True)), Q())
             for col in zip(*b, strict=True)] for row in a]


def _transpose(a):
    return list(map(list, zip(*a, strict=True)))


def _inverse(a):
    n = len(a)
    rows = [list(row)+[Q(i == j) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        rows[i] = [v/rows[i][i] for v in rows[i]]
        for j in range(n):
            if j != i:
                coefficient = rows[j][i]
                rows[j] = [v-coefficient*w for v, w in zip(rows[j], rows[i], strict=True)]
    return [row[n:] for row in rows]


def _rational_covariance(reflections, sigma, prefix):
    """AR0/1/2 scalar YW in closed form, independent of the production owner."""
    k = [Q.from_float(float(v)) for v in reflections]
    if not k:
        lags = [Q(1)]+[Q(0)]*(prefix-1)
    else:
        phi1, phi2 = (-k[0], Q(0)) if len(k) == 1 else (-k[0]*(1+k[1]), -k[1])
        zero = 1/(1-phi2**2-phi1**2*(1+phi2)/(1-phi2))
        lags = [zero, phi1*zero/(1-phi2)]
        for _ in range(2, prefix):
            lags.append(phi1*lags[-1]+phi2*lags[-2])
    channels = len(sigma)
    return [[lags[abs(i//channels-j//channels)]*Q.from_float(float(sigma[i % channels, j % channels]))
             for j in range(prefix*channels)] for i in range(prefix*channels)]


@pytest.mark.parametrize("reflections", [[], [-.9375], [-.71, .23]])
@pytest.mark.parametrize("scale", [1., 1e-300, 1e-320, 1e200])
def test_certificate_covers_independent_exact_joint_pullback(reflections, scale):
    sigma = scale*np.array([[1., .2], [.2, .75]])
    root, state = factor.compile_shared_stationary_root(reflections, sigma, 4)
    # Direct exact inverse/K/Gram calculations deliberately use no factor
    # decomposition or certificate helper, so product rounding is visible.
    inverse = _inverse([[Q.from_float(float(v)) for v in row] for row in root])
    pulled = _matmul(_matmul(inverse, _rational_covariance(reflections, sigma, 4)), _transpose(inverse))
    squared = sum(((value-Q(i == j))**2 for i, row in enumerate(pulled)
                   for j, value in enumerate(row)), Q())
    bound = Q.from_float(state["certificate"]["joint_law_pullback_F_bound"])
    assert squared <= bound**2
    assert bound < Q.from_float(1e-8*np.sqrt(8))
    dense = stationary_root([reflections]*2, sigma, 4)
    # Decimal96 dense elimination can leave 1e-96 relative residue at exact
    # structural zeros. Compare in a common nonunderflowing coordinate chart.
    scale_root = np.max(np.abs(root))
    np.testing.assert_allclose(root/scale_root, dense/scale_root, rtol=8e-15, atol=1e-95)
    assert factor.admit_shared_stationary_root(reflections, sigma, 4, root, state) is not None


def _noise(prefix=4):
    temporal = TemporalNoise.from_reflections([-.625, .125], innovation_variance=7., reference_id="saved-shape")
    geometry = cm.NoiseGeometry(("a", "b"), 1000., "V", "nose", "native", 2)
    provenance = {"fit_scope": "analytic_reference", "reference_id": "factor-oracle", "records": []}
    return cm.FittedChannelNoise.from_shared_stationary_factors((temporal, temporal),
        np.array([[1., .3], [.3, .7]])*1e-12, geometry=geometry,
        conditioning_samples=prefix, provenance=provenance)


def test_saved_factor_codec_restores_original_state_without_any_root_compiler(tmp_path, monkeypatch):
    import research.stationary_noise_artifact as stationary
    import transfer.temporal_noise as temporal

    noise = _noise()
    original = json.loads(json.dumps(noise.record()))
    assert original["schema"] == "n2p3_channel_noise_mechanism/4"
    # Distinct original raw source hashes survive a new consumer host/version.
    original["prefix_compiler"]["source_sha256"] = "a"*64
    original["prefix_compiler"]["factor_law_source_sha256"] = "b"*64
    original["prefix_compiler"]["scalar_rounding_source_sha256"] = "c"*64
    path = tmp_path/"factor.json"
    path.write_text(json.dumps({"state": original, "state_sha256": metadata_digest(original)}), encoding="utf8")
    samples = np.arange(24).reshape(12, 2)*1e-7
    white = noise.whiten(samples)

    def forbidden(*args, **kwargs):
        raise AssertionError("saved factor codec called a new root compiler")

    for owner, name in ((factor, "_cholesky"), (stationary, "_cholesky"),
            (cm, "_prefix_root"), (temporal, "_innovation_factor"), (temporal, "_startup")):
        monkeypatch.setattr(owner, name, forbidden)
    restored = cm.FittedChannelNoise.load(path)
    assert json.loads(json.dumps(restored.record())) == original
    np.testing.assert_array_equal(restored.compiled_prefix_root, noise.compiled_prefix_root)
    np.testing.assert_array_equal(restored.whiten(samples), white)
    assert restored.whitening_logabsdet(12) == noise.whitening_logabsdet(12)
    with pytest.raises(TypeError, match="frozen"):
        restored.shared_factor_state["program"] = "changed"


def test_default_dense_codec_and_private_parameters_keep_original_path(monkeypatch):
    n = _noise()

    def forbidden(*args, **kwargs):
        raise AssertionError("default constructor silently selected shared factor producer")

    monkeypatch.setattr(cm, "compile_shared_stationary_root", forbidden)
    dense = cm.FittedChannelNoise(n.temporal, n.innovation_covariance, n.geometry, 4, "shared", n.provenance)
    assert dense.record()["schema"] == "n2p3_channel_noise_mechanism/3"
    assert "temporal_root" not in dense.record()["prefix_compiler"]
    with pytest.raises(ValueError, match="shared factor state"):
        cm.FittedChannelNoise(n.temporal, n.innovation_covariance, n.geometry, 4, "private", n.provenance,
                             n.compiled_prefix_root, shared_factor_state=n.shared_factor_state)
    other = TemporalNoise.from_reflections([-.6, .125], innovation_variance=7., reference_id="private")
    with pytest.raises(ValueError, match="identical stored PARCOR"):
        cm.FittedChannelNoise.from_shared_stationary_factors((n.temporal[0], other), n.innovation_covariance,
            geometry=n.geometry, conditioning_samples=4, provenance=n.provenance)


@pytest.mark.parametrize("mutation", ["root", "factor", "law", "extra", "missing", "bool", "certificate"])
def test_resigned_malformed_or_altered_factor_states_are_rejected(mutation):
    root, state = factor.compile_shared_stationary_root([-.5], [[1., .2], [.2, .7]], 3)
    state = copy.deepcopy(state)
    if mutation == "root":
        root[3, 3] *= 1.001
        from transfer.noise_model import array_digest
        state["compiled_root_sha256"] = array_digest(root)
    elif mutation == "factor":
        state["spatial_root"][0][0] *= 1.001
    elif mutation == "law":
        state["law"] = "rounded_AR_polynomial"
    elif mutation == "extra":
        state["ignored"] = True
    elif mutation == "missing":
        state.pop("certificate")
    elif mutation == "bool":
        state["channels"] = True
    else:
        state["certificate"]["joint_law_pullback_F_bound"] = 0.
    with pytest.raises(ValueError):
        factor.admit_shared_stationary_root([-.5], [[1., .2], [.2, .7]], 3, root, state)


def test_exact_inverse_defect_and_square_root_upper_bounds():
    # A noncommuting BA defect checks inverse orientation and both pullbacks.
    root = np.array([[.7, 0], [.2, 1.3]])
    actual = [[Q.from_float(float(v)) for v in row] for row in root]
    covariance = [[Q(1), Q(1, 3)], [Q(1, 3), Q(2)]]
    bound, inverse_bound, condition, eta = factor._factor_certificate(root, covariance)
    inverse = _inverse(actual)
    pulled = _matmul(_matmul(inverse, covariance), _transpose(inverse))
    squared = sum(((v-Q(i == j))**2 for i, row in enumerate(pulled) for j, v in enumerate(row)), Q())
    assert squared <= bound**2 and eta < 1
    assert sum((v*v for row in inverse for v in row), Q()) <= inverse_bound**2
    positive = _matmul([[abs(v) for v in row] for row in inverse], [[abs(v) for v in row] for row in actual])
    assert sum((v*v for row in positive for v in row), Q()) <= condition**2
    for value in (Q(0), Q(2), Q(1, 2**2150), Q(2**2040), Q(1, 3)):
        assert Q.from_float(factor._sqrt_upper(value))**2 >= value


def test_product_underflow_has_a_nonzero_exact_defect_covered_by_the_bound():
    tiny = 2.**-600
    root, state = factor.compile_shared_stationary_root([-tiny], [[1., tiny], [tiny, 1.]], 2)
    a = [[Q.from_float(v) for v in row] for row in state["temporal_root"]]
    b = [[Q.from_float(v) for v in row] for row in state["spatial_root"]]
    product = [[a[i//2][j//2]*b[i % 2][j % 2] for j in range(4)] for i in range(4)]
    delta = [[Q.from_float(float(root[i, j]))-product[i][j] for j in range(4)] for i in range(4)]
    assert root[3, 0] == 0 and product[3][0] == Q(1, 2**1200)
    pulled = _matmul(_inverse(product), delta)
    bound = Q.from_float(state["certificate"]["product_rounding_F_bound"])
    assert 0 < sum((v*v for row in pulled for v in row), Q()) <= bound**2


def test_certificate_does_not_accept_a_boolean_in_place_of_an_exact_zero_bound():
    root, state = factor.compile_shared_stationary_root([], [[1.]], 2)
    assert state["certificate"]["temporal_law_pullback_F_bound"] == 0.
    state["certificate"]["temporal_law_pullback_F_bound"] = False
    with pytest.raises(ValueError, match="certificate differs"):
        factor.admit_shared_stationary_root([], [[1.]], 2, root, state)


def test_nonpositive_law_and_unrepresentable_coordinate_range_are_typed():
    with pytest.raises(UnidentifiableFitError):
        factor.compile_shared_stationary_root([-.5], [[1., 2.], [2., 1.]], 4)
    with pytest.raises(UnidentifiableFitError, match="coordinate range"):
        factor._normalize(np.diag([1e308, 1e-320]))
    with pytest.raises(ValueError):
        factor.compile_shared_stationary_root([-.5], [[1., .2], [.1, 1.]], 4)
    with pytest.raises(ValueError):
        factor.compile_shared_stationary_root([1.], [[1.]], 4)


@pytest.mark.parametrize("reflections,sigma", [
    ([np.nextafter(-1., 0.)], [[1., .2], [.2, .7]]),
    ([-.6], [[1., 1-2.**-48], [1-2.**-48, 1.]]),
])
def test_unresolved_weak_directions_keep_original_policy_instead_of_parameter_repair(reflections, sigma):
    before = np.array(sigma)
    with pytest.raises(UnidentifiableFitError, match="declared stationary noise model"):
        factor.compile_shared_stationary_root(reflections, sigma, 4)
    np.testing.assert_array_equal(sigma, before)


def test_factor_noise_joint_flat_save_load_replays_original_numeric_prediction(tmp_path, monkeypatch):
    from research.joint_flat_artifact import load_joint_flat_state, save_joint_flat_state
    from research.joint_flat_authority import _parameters
    from research.joint_flat_response import (
        compile_owned_flat_decision,
        predict_flat_tail,
        prepare_flat_model,
        prepare_flat_prediction,
    )
    from research.joint_response_sampling import run_gibbs
    from research.joint_response_scale import FixedHyperparameters, GlobalGaussianPrior

    noise = _noise()
    prior = GlobalGaussianPrior([0., 0.], np.eye(2)*1e-6, "external")
    hyper = FixedHyperparameters(4., 1., "externally_fixed")
    rng = np.random.default_rng(811)
    y, d = rng.normal(size=(14, 2))*1e-6, rng.normal(size=(14, 2, 2))
    z = np.broadcast_to(np.eye(2), (14, 2, 2)).copy()
    full = compile_owned_flat_decision(prior, hyper, noise, decision_id="same-record",
        role="target_calibration", native_samples=14, raw_batches=((0, y, d, z),), conditioning_samples=6)
    model = prepare_flat_model(prior, (full.prefix,), hyper)
    samples = run_gibbs(model, burnin=2, draws=16, seeds=(23, 41), initial_charts=np.zeros((2, 2)))
    prediction = prepare_flat_prediction(model, full)
    policy = dict(batch_size=4, max_rhat=1000., min_bulk_ess=0., min_tail_ess=0.)
    before = predict_flat_tail(samples, prediction, **policy)
    parameters = _parameters(prior, hyper, np.ones((2, 1)), noise, {"noise": noise.record(), "scope": "software_only"})
    identity = save_joint_flat_state(tmp_path/"package", samples, predictions=(prediction,), parameters=parameters)

    def forbidden(*args, **kwargs):
        raise AssertionError("model-only replay refit or recompiled original data")

    monkeypatch.setattr(factor, "_cholesky", forbidden)
    monkeypatch.setattr(cm.FittedChannelNoise, "fit", forbidden)
    monkeypatch.setattr(cm.FittedChannelNoise, "whiten", forbidden)
    restored, predictions, _, actual = load_joint_flat_state(tmp_path/"package", artifact_sha256=identity)
    after = predict_flat_tail(restored, predictions[0], **policy)
    assert actual.noise.record() == noise.record()
    np.testing.assert_array_equal(before["component_tail_logs"], after["component_tail_logs"])
    np.testing.assert_array_equal(before["prefix_weights"], after["prefix_weights"])
    assert before["log_density"] == after["log_density"]
