"""Dense physical posterior, separable limits and highest B05 score oracles."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import block_diag, null_space
from scipy.stats import multivariate_normal

from research.channel_noise_mechanism import (
    FittedChannelNoise,
    NoiseGeometry,
    PhysicalCoefficientPrior,
    ResponseFitRecord,
)
from research.execution import UnidentifiableFitError
from transfer.noise_model import array_digest


def geometry(unit="V"):
    return NoiseGeometry(("Cz", "Pz"), 128., unit, "declared_native_reference", "native-software", 2)


def fixture(*, family="private", sigma=None, scale=1., noise=None):
    g = geometry("V" if scale == 1 else "uV")
    sigma = np.array([[1., .3], [.3, 1.4]]) if sigma is None else sigma
    noise = noise or FittedChannelNoise.from_parameters([[.55], [-.2]] if family == "private" else [[.4], [.4]],
        sigma*scale**2, geometry=g, conditioning_samples=4, reference_id="fixed-known-software", family=family)
    basis = np.array([[.1, .2], [.8, .6], [.4, .9]])
    mean = np.array([[.2, -.1], [.1, .3], [.6, .2], [-.2, .4]])*scale
    covariance = np.diag([.7, .5, .6, .8, .4, .9, .5, .6])*scale**2
    covariance[0, 6] = covariance[6, 0] = .1*scale**2
    precision = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(8))
    prior = PhysicalCoefficientPrior(mean, precision, basis, g, "fixed-physical-software",
                                     {"role": "analytic_reference", "source_sha256": "a"*64})
    records = []
    rng = np.random.default_rng(1445)
    for index in range(2):
        n = 18+index
        labels = np.array([0, 1, index])
        design = event_design(basis, [5, 9, 13], labels, n)
        drift = np.column_stack((np.ones(n), (2*np.arange(n)-3)/3))
        nuisance = np.kron(drift, np.eye(2)).reshape(n, 2, 4)
        actual_mean = np.array([[.5, -.2], [.7, .6], [1.2, .9], [-.3, .2]])*scale
        samples = design@actual_mean+(nuisance.reshape(n*2, 4)@np.array([.2, -.1, .1, .03])).reshape(n, 2)*scale
        samples += .3*rng.normal(size=(n, 2))*scale
        records.append(ResponseFitRecord(samples, design, nuisance, labels, array_digest(basis), g,
            {"record_id": f"C{index}", "participant_key": "subject", "recording_id": f"raw{index}",
             "read_support": [10, 10+n], "label_support": [15, 24], "fit_scope": "target_calibration",
             "source_sha256": "b"*64, "source_input_sha256": "c"*64, "design_role": "actual_known_C_event_design"}))
    return noise, prior, records


def event_design(basis, onsets, labels, n):
    impulse = np.zeros((n, 2))
    for onset, label in zip(onsets, labels, strict=True):
        impulse[onset] = [1, label]
    return np.column_stack([np.convolve(values, column)[:n] for values in impulse.T for column in basis.T])


def ar1_covariance(coefficients, sigma, n):
    """Closed diagonal VAR1 covariance; independent of actual whitening code."""
    phi = np.asarray(coefficients)
    stationary = sigma/(1-phi[:, None]*phi[None, :])
    covariance = np.empty((n*len(phi), n*len(phi)))
    for t in range(n):
        for s in range(n):
            block = np.diag(phi**abs(t-s))@stationary
            covariance[t*len(phi):(t+1)*len(phi), s*len(phi):(s+1)*len(phi)] = block if t >= s else block.T
    return covariance


def dense_joint(noise, prior, records):
    d = prior.mean.size
    count = d+sum(row.nuisance.shape[2] for row in records)
    precision = np.zeros((count, count))
    precision[:d, :d] = prior.precision_root.T@prior.precision_root
    score = np.r_[precision[:d, :d]@prior.mean.ravel(), np.zeros(count-d)]
    offset = d
    for row in records:
        n, c = row.samples.shape
        design = np.zeros((n*c, count))
        design[:, :d] = np.kron(row.design, np.eye(c))
        z = row.nuisance.reshape(n*c, -1)
        design[:, offset:offset+z.shape[1]] = z
        offset += z.shape[1]
        covariance = ar1_covariance([item.coefficients[0] for item in noise.temporal], noise.innovation_covariance, n)
        solved = np.linalg.solve(covariance, design)
        precision += design.T@solved
        score += design.T@np.linalg.solve(covariance, row.samples.ravel())
    covariance = np.linalg.inv(precision)
    return np.linalg.solve(precision, score), covariance


@pytest.mark.parametrize("family", ["shared", "private"])
def test_full_general_posterior_and_all_drift_crosscov_match_independent_dense_joint(family):
    noise, prior, records = fixture(family=family)
    before = noise.record()
    actual = noise.fit_response_given_frozen_noise(prior, records, response_id="given-noise-software")
    mean, covariance = dense_joint(noise, prior, records)
    vector = np.r_[actual.response.mean.ravel(), *(item.mean for item in actual.drifts)]
    factor = actual.joint_coefficient_drift_factor()
    np.testing.assert_allclose(vector, mean, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(factor@factor.T, covariance, rtol=3e-11, atol=3e-11)
    assert np.max(np.abs(covariance[:8, 8:])) > .001
    assert np.max(np.abs(covariance[8:12, 12:])) > .001
    assert noise.record() == before
    assert actual.binding["matrix_normal_claimed"] is False


def test_noise_tied_mn_limit_matches_shared_row_posterior_and_keeps_crossblocks():
    noise, prior, records = fixture(family="shared")
    lam = 2.
    physical = np.kron(np.eye(4)/lam, noise.innovation_covariance)
    prior = replace(prior, precision_root=np.linalg.solve(np.linalg.cholesky(physical), np.eye(8)))
    fit = noise.fit_response_given_frozen_noise(prior, records, response_id="mn-limit")
    gram, score = lam*np.eye(4), lam*prior.mean.copy()
    for row in records:
        n = len(row.samples)
        phi = noise.temporal[0].coefficients[0]
        temporal_cov = phi**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])/(1-phi**2)
        whitening = np.linalg.inv(np.linalg.cholesky(temporal_cov))
        drift = row.nuisance[:, 0, ::2]
        z = whitening@drift
        complement = null_space(z.T)
        design, response = complement.T@whitening@row.design, complement.T@whitening@row.samples
        gram += design.T@design
        score += design.T@response
    h = np.linalg.inv(gram)
    np.testing.assert_allclose(fit.response.mean, np.linalg.solve(gram, score), rtol=3e-11, atol=3e-11)
    factor = fit.response.coefficient_factor
    np.testing.assert_allclose(factor@factor.T, np.kron(h, noise.innovation_covariance), rtol=3e-11, atol=3e-11)
    contrast = np.kron(np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]]), np.eye(2))
    full_target = contrast@(factor@factor.T)@contrast.T
    wrong = np.kron(h[:2, :2]+h[2:, 2:], noise.innovation_covariance)
    assert np.max(np.abs(full_target-wrong)) > .001


def test_sigma_only_cancels_in_noise_tied_mean_but_changes_fixed_physical_prior_mean():
    first, prior, records = fixture(family="shared")
    second = FittedChannelNoise.from_parameters([[.4], [.4]], 4*first.innovation_covariance,
        geometry=first.geometry, conditioning_samples=4, reference_id="same-shape-other-Sigma", family="shared")
    fixed_a = first.fit_response_given_frozen_noise(prior, records, response_id="fixed-a")
    fixed_b = second.fit_response_given_frozen_noise(prior, records, response_id="fixed-b")
    assert np.max(np.abs(fixed_a.response.mean-fixed_b.response.mean)) > .05
    tied = []
    for noise in (first, second):
        covariance = np.kron(np.eye(4)/2, noise.innovation_covariance)
        arm_prior = replace(prior, precision_root=np.linalg.solve(np.linalg.cholesky(covariance), np.eye(8)))
        tied.append(noise.fit_response_given_frozen_noise(arm_prior, records, response_id="tied").response.mean)
    np.testing.assert_allclose(tied[0], tied[1], rtol=3e-11, atol=3e-11)
    assert fixed_a.binding["prior"] == fixed_b.binding["prior"]


def test_physical_unit_change_and_nonorthogonal_basis_pullback_preserve_mean_covariance():
    noise, prior, records = fixture()
    reference = noise.fit_response_given_frozen_noise(prior, records, response_id="physical-reference")
    scaled_noise, scaled_prior, scaled_records = fixture(scale=1e6)
    scaled = scaled_noise.fit_response_given_frozen_noise(scaled_prior, scaled_records, response_id="unit-change")
    np.testing.assert_allclose(scaled.response.mean/1e6, reference.response.mean, rtol=3e-11, atol=3e-11)
    sf = scaled.response.coefficient_factor/1e6
    rf = reference.response.coefficient_factor
    np.testing.assert_allclose(sf@sf.T, rf@rf.T, rtol=3e-11, atol=3e-11)
    change = np.array([[2., .3], [-.1, .6]])
    block = block_diag(change, change)
    vector = np.kron(block, np.eye(2))
    transformed_prior = replace(prior, basis=prior.basis@change, mean=np.linalg.solve(block, prior.mean),
                                precision_root=prior.precision_root@vector)
    transformed_records = [replace(row, design=row.design@block, basis_sha256=array_digest(transformed_prior.basis)) for row in records]
    fit = noise.fit_response_given_frozen_noise(transformed_prior, transformed_records, response_id="basis-change")
    np.testing.assert_allclose(block@fit.response.mean, reference.response.mean, rtol=3e-11, atol=3e-11)
    tf = vector@fit.response.coefficient_factor
    np.testing.assert_allclose(tf@tf.T, rf@rf.T, rtol=3e-11, atol=3e-11)
    rng = np.random.default_rng(155)
    held = rng.normal(size=(12, 2))
    designs = rng.normal(size=(2, 12, 4))
    args = dict(candidate_ids=("one", "two"), record_id="held-unit-basis", recording_id="held-physical", sample_start=0)
    original_score = noise.score(held, designs, reference.response, geometry=noise.geometry, **args)
    basis_score = noise.score(held, designs@block, fit.response, geometry=noise.geometry, **args)
    unit_score = scaled_noise.score(held*1e6, designs, scaled.response, geometry=scaled_noise.geometry, **args)
    np.testing.assert_allclose(basis_score["log_density"], original_score["log_density"], rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(unit_score["log_probability"], original_score["log_probability"], rtol=3e-11, atol=3e-11)
    shift = -(len(held)-noise.conditioning_samples)*2*np.log(1e6)
    np.testing.assert_allclose(unit_score["log_density"], np.asarray(original_score["log_density"])+shift, rtol=3e-11, atol=3e-11)


def test_common_nuisance_coordinate_change_and_record_stream_permutation_keep_physical_posterior():
    noise, prior, records = fixture()
    reference = noise.fit_response_given_frozen_noise(prior, records, response_id="reference")
    change = np.array([[2., .2, 0., 0.], [0., .5, .1, 0.], [0., 0., .8, .3], [0., 0., 0., 1.2]])
    transformed = [replace(row, nuisance=row.nuisance@change) for row in records]
    result = noise.fit_response_given_frozen_noise(prior, transformed[::-1], response_id="transformed")
    np.testing.assert_allclose(result.response.mean, reference.response.mean, rtol=3e-11, atol=3e-11)
    a, b = result.response.coefficient_factor, reference.response.coefficient_factor
    np.testing.assert_allclose(a@a.T, b@b.T, rtol=3e-11, atol=3e-11)
    by_id = {item.record_id: item for item in reference.drifts}
    for item in result.drifts:
        np.testing.assert_allclose(change@item.mean, by_id[item.record_id].mean, rtol=3e-11, atol=3e-11)


def test_fitted_response_scores_same_prefix_common_measure_against_independent_dense_ratio():
    noise, prior, records = fixture()
    fit = noise.fit_response_given_frozen_noise(prior, records, response_id="highest-score-seam")
    rng = np.random.default_rng(377)
    n = 16
    designs = rng.normal(size=(2, n, 4))
    samples = rng.normal(size=(n, 2))
    score = noise.score(samples, designs, fit.response, candidate_ids=("one", "two"), geometry=noise.geometry,
                        record_id="held", recording_id="held-recording", sample_start=0)
    sigma = ar1_covariance([item.coefficients[0] for item in noise.temporal], noise.innovation_covariance, n)
    z = np.kron(np.column_stack((np.ones(n), (2*np.arange(n)-3)/3)), np.eye(2))
    factor = fit.response.coefficient_factor
    expected = []
    for design in designs:
        mapping = np.kron(design, np.eye(2))
        covariance = sigma+mapping@factor@factor.T@mapping.T
        residual = samples.ravel()-mapping@fit.response.mean.ravel()
        potentials = []
        for stop in (n*2, 4*2):
            complement = null_space(z[:stop].T)
            potentials.append(multivariate_normal.logpdf(complement.T@residual[:stop],
                cov=complement.T@covariance[:stop, :stop]@complement)
                -.5*np.linalg.slogdet(z[:stop].T@z[:stop])[1])
        expected.append(potentials[0]-potentials[1])
    np.testing.assert_allclose(score["log_density"], expected, rtol=3e-11, atol=3e-11)
    assert score["parameters_updated"] is False


def test_wrong_c_role_basis_or_overlapping_support_cannot_be_signed_as_fit():
    noise, prior, records = fixture()
    with pytest.raises(ValueError, match="never held"):
        replace(records[0], authorization={**records[0].authorization, "fit_scope": "held_diagnostic"})
    with pytest.raises(ValueError, match="basis"):
        noise.fit_response_given_frozen_noise(prior, [replace(records[0], basis_sha256="d"*64)], response_id="wrong-basis")
    overlap = replace(records[1], authorization={**records[1].authorization, "recording_id": records[0].authorization["recording_id"]})
    with pytest.raises(ValueError, match="duplicate actual"):
        noise.fit_response_given_frozen_noise(prior, [records[0], overlap], response_id="duplicate")
    z = records[0].nuisance.copy()
    z[:4] = 0
    with pytest.raises(UnidentifiableFitError, match="diffuse"):
        noise.fit_response_given_frozen_noise(prior, [replace(records[0], nuisance=z)], response_id="no-prefix-info")


def test_private_noise_with_same_mn_prior_produces_nonseparable_full_response_covariance():
    noise, prior, records = fixture(family="private")
    covariance = np.kron(np.eye(4)/2, noise.innovation_covariance)
    prior = replace(prior, precision_root=np.linalg.solve(np.linalg.cholesky(covariance), np.eye(8)))
    fit = noise.fit_response_given_frozen_noise(prior, records, response_id="private-not-MN")
    factor = fit.response.coefficient_factor
    posterior = factor@factor.T
    # Every separable H x S reshapes into an outer product (rank one).
    reshaped = posterior.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3).reshape(16, 4)
    singular = np.linalg.svd(reshaped, compute_uv=False)
    assert singular[1]/singular[0] > .01
    _, dense = dense_joint(noise, prior, records)
    np.testing.assert_allclose(posterior, dense[:8, :8], rtol=3e-11, atol=3e-11)


def test_saved_general_response_factor_replays_existing_score_without_refitting(tmp_path, monkeypatch):
    import research.channel_noise_mechanism as owner
    from research.channel_noise_mechanism import FrozenERPResponse
    from transfer.temporal_noise import TemporalNoise

    noise, prior, records = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("given-frozen-noise fit/replay refitted or recompiled actual noise")
    monkeypatch.setattr(FittedChannelNoise, "fit", forbidden)
    monkeypatch.setattr(TemporalNoise, "fit", forbidden)
    monkeypatch.setattr(owner, "_prefix_root", forbidden)
    result = noise.fit_response_given_frozen_noise(prior, records, response_id="actual-response-state")
    artifact = tmp_path/"response-arrays.npz"
    np.savez(artifact, mean=result.response.mean, coefficient_factor=result.response.coefficient_factor,
             precision_root=result.precision_root, precision_response=result.precision_response,
             joint_factor=result.joint_coefficient_drift_factor(), augmented_root=result.augmented_root,
             **{f"information_{index}": value for index, value in enumerate(result.information_roots)},
             **{f"information_response_{index}": value for index, value in enumerate(result.information_responses)})
    with np.load(artifact, allow_pickle=False) as saved:
        restored = FrozenERPResponse(saved["mean"], saved["coefficient_factor"], prior.geometry,
            result.response.source_model_sha256, result.response.response_id, result.response.provenance)
        np.testing.assert_array_equal(saved["precision_root"], result.precision_root)
        np.testing.assert_array_equal(saved["joint_factor"], result.joint_coefficient_drift_factor())
        np.testing.assert_array_equal(saved["augmented_root"], result.augmented_root)
    designs = np.random.default_rng(477).normal(size=(2, 12, 4))
    y = np.random.default_rng(478).normal(size=(12, 2))
    args = dict(candidate_ids=("one", "two"), geometry=prior.geometry, record_id="held-replay",
                recording_id="held-new", sample_start=0)
    original = noise.score(y, designs, result.response, **args)
    replay = noise.score(y, designs, restored, **args)
    assert replay == original
    assert restored.record() == result.response.record()


def test_actual_per_record_roots_and_streamed_posterior_preserve_prior_information_relation():
    noise, prior, records = fixture()
    result = noise.fit_response_given_frozen_noise(prior, records, response_id="actual-root-relation")
    stacked = np.vstack((prior.precision_root, *result.information_roots))
    response = np.r_[prior.precision_root@prior.mean.ravel(), *result.information_responses]
    np.testing.assert_allclose(result.precision_root.T@result.precision_root, stacked.T@stacked, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(result.precision_root.T@result.precision_response, stacked.T@response, rtol=3e-12, atol=3e-12)
    mean, covariance = dense_joint(noise, prior, records)
    np.testing.assert_allclose(result.response.mean.ravel(), mean[:8], rtol=3e-11, atol=3e-11)
    factor = result.joint_coefficient_drift_factor()
    np.testing.assert_allclose(factor@factor.T, covariance, rtol=3e-11, atol=3e-11)
    objective = (stacked@result.response.mean.ravel()-response)
    compressed = result.augmented_root
    scales = np.asarray(result.response.provenance["prior_column_scales"])
    vector = np.r_[scales*result.response.mean.ravel(), -1.]
    np.testing.assert_allclose(objective@objective, np.sum((compressed@vector)**2), rtol=3e-12, atol=3e-12)
