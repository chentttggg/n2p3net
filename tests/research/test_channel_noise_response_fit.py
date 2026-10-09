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


@pytest.mark.parametrize("voltage_scale", [1., 1e6])
@pytest.mark.parametrize("noise_scale,offset,unresolved", [
    (1e-12, .35e-6, False), (1e-48, 0., True), (1e-12, .35e14, True)])
def test_pure_diffuse_public_fit_preserves_full_prior_or_reports_numeric_unresolved(
        voltage_scale, noise_scale, offset, unresolved):
    from research.channel_noise_mechanism import DiffuseProjectionResolutionError

    native, channels, prefix = 7, 2, 2
    unit = "V" if voltage_scale == 1 else "uV"
    geometry = NoiseGeometry(("toy-left", "toy-right"), 256., unit, "toy-earlobe", "toy-native", channels)
    sigma = np.array([[.9, .22], [.22, .6]])*noise_scale*voltage_scale**2
    noise = FittedChannelNoise.from_parameters([[.62], [-.27]], sigma, geometry=geometry,
        conditioning_samples=prefix, reference_id="pure-Z-software", family="private")
    basis = np.ones((native, 1))
    # A complete nontriangular precision root, rather than marginal deviations.
    root = np.array([[1., .35, 0., .2], [.1, 1.2, .3, 0.],
                     [0., -.2, .8, .1], [.15, 0., .2, 1.1]])/(.6e-6*voltage_scale)
    prior = PhysicalCoefficientPrior(np.array([[.1, -.2], [.3, .1]])*1e-6*voltage_scale,
        root, basis, geometry, "independent-physical-prior",
        {"role": "analytic_reference", "source_sha256": "a"*64})
    design = np.column_stack((np.ones(native), np.zeros(native)))
    nuisance = np.kron(np.column_stack((np.ones(native), 2*np.arange(native)-1)),
        np.diag(np.sqrt(np.diag(sigma)))).reshape(native, channels, 4)
    row = ResponseFitRecord(np.tile(np.array([offset, -.6*offset])*voltage_scale, (native, 1)),
        design, nuisance, np.array([0]), array_digest(basis), geometry,
        {"record_id": "pure-Z-C", "participant_key": "toy", "recording_id": "toy-recording",
         "read_support": [0, native], "label_support": [0, 1], "fit_scope": "target_calibration",
         "source_sha256": "b"*64, "source_input_sha256": "c"*64,
         "design_role": "actual_known_C_event_design"})
    if unresolved:
        with pytest.raises(DiffuseProjectionResolutionError, match="proper-prior/noise chart"):
            noise.fit_response_given_frozen_noise(prior, [row], response_id="unresolved-toy")
    else:
        result = noise.fit_response_given_frozen_noise(prior, [row], response_id="ordinary-toy")
        np.testing.assert_allclose(result.response.mean, prior.mean, rtol=3e-13, atol=1e-18*voltage_scale)
        chart_factor = root@result.response.coefficient_factor
        np.testing.assert_allclose(chart_factor@chart_factor.T, np.eye(4), rtol=3e-13, atol=3e-13)
        resolution = result.response.provenance["compression"][0]["diffuse_projection_resolution"]
        assert resolution["structural_zero"] is False
        assert resolution["normalized_error_bound"] > 0


def test_structurally_saturated_nuisance_quotient_is_zero_at_large_operand_scales():
    from research.channel_noise_mechanism import project_diffuse_nuisance

    z = np.array([[1., 2.], [1., -1.]])
    result = project_diffuse_nuisance(np.array([[2., -3.], [5., 1.]])*1e100,
        np.array([.4, -.7])*1e100, z, prior_factor=np.eye(2))
    np.testing.assert_array_equal(result.design, np.zeros((2, 2)))
    np.testing.assert_array_equal(result.response, np.zeros(2))
    assert result.resolution["structural_zero"] is True


def test_diffuse_resolution_uses_complete_correlated_prior_not_precision_column_maxima():
    from research.channel_noise_mechanism import (
        DiffuseProjectionResolutionError,
        project_diffuse_nuisance,
    )

    # Precision entries have order 1e6, but the common weak direction has a
    # huge proper variance. Marginal max-column scaling would miss that axis.
    precision = np.array([[1., 1.], [1., 1.+1e-10]])*1e6
    d = np.tile(np.eye(2), (7, 1))*1e6
    z = np.kron(np.column_stack((np.ones(7), 2*np.arange(7)-1)), np.eye(2))
    with pytest.raises(DiffuseProjectionResolutionError, match="proper-prior/noise chart"):
        project_diffuse_nuisance(d, np.zeros(14), z,
            prior_factor=np.linalg.solve(precision, np.eye(2)))
    # This intentionally different, tightly independent prior is resolved.
    resolved = project_diffuse_nuisance(d, np.zeros(14), z, prior_factor=np.eye(2)*1e-6)
    assert resolved.resolution["normalized_error_bound"] < resolved.resolution["resolution_target"]


def test_genuinely_large_quotient_information_is_resolved_without_blanket_amplitude_rejection():
    from research.channel_noise_mechanism import project_diffuse_nuisance

    n = 9
    z = np.column_stack((np.ones(n), np.arange(n)))
    d = (np.arange(n) % 2)[:, None]*1e12
    result = project_diffuse_nuisance(d, np.zeros(n), z, prior_factor=np.eye(1))
    complement = null_space(z.T)
    expected = np.sum((complement.T@d)**2)
    assert np.sum(result.design**2) == pytest.approx(expected, rel=1e-13)
    assert expected > 1e24
    assert result.resolution["normalized_error_bound"] < result.resolution["resolution_target"]


def test_small_gamma_rate_reference_cannot_admit_pure_flat_response_roundoff_as_scale_evidence():
    from research.channel_noise_mechanism import (
        DiffuseProjectionResolutionError,
        project_diffuse_nuisance,
    )

    z = np.column_stack((np.ones(7), np.arange(7)))
    d, y = np.zeros((7, 1)), np.ones(7)
    ordinary = project_diffuse_nuisance(d, y, z, prior_factor=np.eye(1))
    assert ordinary.resolution["normalized_error_bound"] < ordinary.resolution["resolution_target"]
    # The same rounding in q is material against the declared tiny nu/2 rate.
    with pytest.raises(DiffuseProjectionResolutionError, match="proper-prior/noise chart"):
        project_diffuse_nuisance(d, y, z, prior_factor=np.eye(1), response_reference_scale=1e-15)


def test_response_rate_reference_keeps_gaussian_noise_unit_and_explicit_known_mean_chart():
    from research.channel_noise_mechanism import project_diffuse_nuisance

    y = np.array([1., 2., -.5, .7])
    z = np.ones((4, 1))
    quotient = project_diffuse_nuisance(np.empty((4, 0)), y, z, prior_factor=np.empty((0, 0)))
    np.testing.assert_allclose(quotient.response, y-y.mean(), atol=1e-14)
    assert quotient.design.shape == (4, 0)
    with pytest.raises(ValueError, match="retain one noise unit"):
        project_diffuse_nuisance(np.empty((4, 0)), y, z,
            prior_factor=np.empty((0, 0)), response_reference_scale=1e15)


@pytest.mark.parametrize("known_mean", [False, True])
def test_public_score_flat_translation_is_equivariant_or_typed_unresolved(known_mean):
    from research.channel_noise_mechanism import FrozenERPResponse

    geometry = NoiseGeometry(("toy-left", "toy-right"), 256., "V", "toy-earlobe", "toy-native", 2)
    noise = FittedChannelNoise.from_parameters([[.62], [-.27]], np.array([[.9, .22], [.22, .6]])*1e-12,
        geometry=geometry, conditioning_samples=2, reference_id="score-pure-Z", family="private")
    factor = np.empty((4, 0)) if known_mean else np.diag([.6e-6, .7e-6, .6e-6, .7e-6])
    response = FrozenERPResponse(np.zeros((2, 2)), factor, geometry, "a"*64, "frozen-toy",
        {"fit_scope": "analytic_reference"})
    designs = np.tile(np.column_stack((np.ones(7), np.zeros(7))), (2, 1, 1))
    args = dict(candidate_ids=("one", "two"), geometry=geometry,
        record_id="held", recording_id="held-physical", sample_start=0)
    reference = noise.score(np.zeros((7, 2)), designs, response, **args)
    ordinary = noise.score(np.tile([.35e-6, -.22e-6], (7, 1)), designs, response, **args)
    np.testing.assert_allclose(ordinary["log_density"], reference["log_density"], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(ordinary["probabilities"], [.5, .5], atol=1e-14)
    assert ordinary["reason"] == "tie"
    # The declared intercept chart removes an exactly represented common
    # offset before the actual causal map, rather than rejecting legal pure Z.
    huge = np.tile([.35e14, -.22e14], (7, 1))
    translated = noise.score(huge, designs, response, **args)
    np.testing.assert_array_equal(translated["log_density"], reference["log_density"])
    assert translated["samples_sha256"] == array_digest(huge)


@pytest.mark.parametrize("family", ["shared", "private"])
def test_actual_square_whitening_logdet_matches_independent_dense_native_covariance(family):
    noise, _, _ = fixture(family=family)
    before = noise.record()
    for native in (1, 2, noise.conditioning_samples, noise.conditioning_samples+1, 13):
        covariance = ar1_covariance([item.coefficients[0] for item in noise.temporal],
            noise.innovation_covariance, native)
        expected = -.5*np.linalg.slogdet(covariance)[1]
        assert noise.whitening_logabsdet(native) == pytest.approx(expected, rel=3e-13, abs=3e-13)
    assert noise.record() == before
    for invalid in (0, -1, True, 2., np.int64(2)):
        with pytest.raises(ValueError, match="native sample count"):
            noise.whitening_logabsdet(invalid)


@pytest.mark.parametrize("offset", [0., 1e14])
def test_raw_conditional_density_cannot_borrow_full_quotient_relative_precision(offset):
    from research.channel_noise_mechanism import DiffuseProjectionResolutionError, FrozenERPResponse

    g = NoiseGeometry(("toy",), 128., "V", "explicit_toy_reference", "AR0_affine_translation", 1)
    noise = FittedChannelNoise.from_parameters(np.empty((1, 0)), np.ones((1, 1)), geometry=g,
        conditioning_samples=2, reference_id="mixed-prefix", family="shared")
    response = FrozenERPResponse(np.zeros((2, 1)), np.empty((2, 0)), g, "a"*64, "known-zero",
        {"fit_scope": "analytic_reference"})
    # The independent raw contrast covariance is I+H H.T, H_t=[1-t,t].
    # Exact q=33750000000000000000/7; even the output binary64 ULP cannot
    # meet sqrt(eps) nats per tail row. The old full quotient gate admitted both
    # cases, with a spurious 31,031,003 change in q after the exact flat shift.
    y = np.array([0., 0., 1e9, -1e9, 1e9, -1e9, 1e9])[:, None]+offset
    with pytest.raises(DiffuseProjectionResolutionError, match="raw conditional log-density"):
        noise.score(y, np.zeros((2, 7, 2)), response, candidate_ids=("a", "b"), geometry=g,
                    record_id="held-offset", recording_id="toy-native", sample_start=0)


@pytest.mark.parametrize("family", ["shared", "private"])
@pytest.mark.parametrize("known_mean", [False, True])
def test_raw_intercept_chart_keeps_exact_mixed_translation_and_original_source(family, known_mean):
    from research.channel_noise_mechanism import FrozenERPResponse

    g = geometry()
    sigma = np.array([[1., .25], [.25, 1.5]])
    coefficients = [[.5], [.5]] if family == "shared" else [[.5], [-.25]]
    noise = FittedChannelNoise.from_parameters(coefficients, sigma, geometry=g,
        conditioning_samples=2, reference_id="exact-chart", family=family)
    mean = np.array([[.25, -.125], [.125, .375]])
    factor = np.empty((4, 0)) if known_mean else np.array([
        [.5, .125, 0., 0.], [0., .375, .125, 0.], [.125, 0., .5, .125], [0., 0., 0., .25]])
    response = FrozenERPResponse(mean, factor, g, "a"*64, "fixed-correlated-proper",
                                 {"fit_scope": "analytic_reference"})
    y = np.array([[0., 0.], [.25, -.125], [.5, .25], [-.375, .125],
                  [.25, -.25], [-.5, .375], [.125, -.125]])
    designs = np.stack((np.column_stack((np.ones(7), np.arange(7)%2)),
                       np.column_stack((np.ones(7), (np.arange(7)+1)%2))))
    args = dict(candidate_ids=("a", "b"), geometry=g, record_id="held",
                recording_id="toy-native", sample_start=10, candidate_prior=np.array([.3, .7]))
    shifted_y = y+np.array([1e14, -1e14])
    np.testing.assert_array_equal(shifted_y-shifted_y[0], y-y[0])
    original, shifted = (noise.score(value, designs, response, **args) for value in (y, shifted_y))
    np.testing.assert_array_equal(shifted["log_density"], original["log_density"])
    np.testing.assert_array_equal(shifted["probabilities"], original["probabilities"])
    assert shifted["samples_sha256"] == array_digest(shifted_y)
    assert shifted["samples_sha256"] != original["samples_sha256"]
    assert shifted["read_support"] == [10, 17] and shifted["conditioning_support"] == [10, 12]
    for diagnostic in shifted["candidate_diagnostics"]:
        resolution = diagnostic["conditional_numerical_resolution"]
        assert resolution["actual_tail_scalar_rows"] == 10
        assert resolution["absolute_log_density_error_estimate"] <= resolution["absolute_log_density_error_budget"]
        assert resolution["all_input_certificate"] is False


def test_large_affine_slope_is_not_certified_by_intercept_centering():
    from research.channel_noise_mechanism import DiffuseProjectionResolutionError, FrozenERPResponse

    g = NoiseGeometry(("toy",), 128., "V", "toy-reference", "AR0-affine-slope", 1)
    noise = FittedChannelNoise.from_parameters(np.empty((1, 0)), np.ones((1, 1)), geometry=g,
        conditioning_samples=2, reference_id="slope", family="shared")
    response = FrozenERPResponse(np.zeros((1, 1)), np.empty((1, 0)), g, "a"*64, "known-zero",
        {"fit_scope": "analytic_reference"})
    # The exact representable affine nuisance is legal; this implementation's
    # intercept chart leaves large slope operands and must disclose unresolved
    # conditional arithmetic instead of asserting a nonzero residual density.
    with pytest.raises(DiffuseProjectionResolutionError, match="(proper-prior/noise chart|raw conditional)"):
        noise.score((np.arange(7)*1e14)[:, None], np.zeros((2, 7, 1)), response,
                    candidate_ids=("a", "b"), geometry=g, record_id="held",
                    recording_id="toy-native", sample_start=0)
