"""Fixed-prefix likelihoods against independent dense Gaussian references."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov, toeplitz
from scipy.special import logsumexp

from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel
from transfer.temporal_noise import TemporalNoise


def basis_and_design(onsets, targets, n):
    t = np.arange(13)
    basis = np.column_stack((np.exp(-((t - 4) / 3) ** 2), np.exp(-((t - 9) / 3) ** 2)))
    design = np.zeros((n, 4))
    for onset, target in zip(onsets, targets, strict=True):
        stop = min(n, onset + len(basis))
        design[onset:stop, :2] += basis[:stop - onset]
        design[onset:stop, 2:] += target * basis[:stop - onset]
    return basis, design


def dense_noise(coefficients, n):
    order = len(coefficients)
    if not order:
        return np.eye(n)
    transition = np.zeros((order, order))
    transition[0] = coefficients
    transition[1:, :-1] = np.eye(order - 1)
    innovation = np.zeros((order, order))
    innovation[0, 0] = 1
    covariance = solve_discrete_lyapunov(transition, innovation)
    lag = np.empty(n)
    state = covariance.copy()
    for index in range(n):
        lag[index] = state[0, 0]
        state = transition @ state
    return toeplitz(lag)


def diffuse_density(y, design, drift, time_covariance, mean, kernel_covariance, spatial):
    covariance = time_covariance + design @ kernel_covariance @ design.T
    root = cho_factor(covariance, lower=True)
    precision_drift = cho_solve(root, drift)
    drift_information = drift.T @ precision_drift
    residual = y - design @ mean
    precision_residual = cho_solve(root, residual)
    quadratic = residual.T @ precision_residual
    projected = drift.T @ precision_residual
    quadratic -= projected.T @ np.linalg.solve(drift_information, projected)
    n, channels = y.shape
    return -.5 * (
        (n - drift.shape[1]) * channels * np.log(2 * np.pi)
        + (n - drift.shape[1]) * np.linalg.slogdet(spatial)[1]
        + channels * (np.linalg.slogdet(covariance)[1] + np.linalg.slogdet(drift_information)[1])
        + np.trace(np.linalg.solve(spatial, quadratic))
    )


def analytic_model(order):
    reflections = .015 * np.cos(np.arange(order))
    if order:
        reflections[0] = -.55
    noise = TemporalNoise.from_reflections(reflections, innovation_variance=1., reference_id=f"AR{order}")
    basis, _ = basis_and_design(np.array([2]), np.array([0]), 89)
    model = ContinuousDecisionModel(
        basis, np.array([[.3, -.1], [.15, .2], [.45, .25], [-.2, .3]]),
        np.diag([.4, .3, .2, .5]), np.array([[1.2, .2], [.2, .8]]), None,
        50., ("Fz", "Pz"), ("a", "b", "c"), "native",
        {"fit_scope": "analytic_reference", "reference_id": "fixed-prefix", "records": []},
        noise=noise, conditioning_samples=32,
    )
    rng = np.random.default_rng(219 + order)
    samples = rng.normal(size=(89, 2))
    onsets = np.array([2, 18, 40, 60])
    membership = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    return model, samples, onsets, membership


def score(model, samples, onsets, membership):
    return model.score(samples, onsets, membership, operator_identity="native", sfreq=50.,
                       channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), record_id="held")


@pytest.mark.parametrize("order", [0, 1, 4, 8, 16, 32])
def test_fixed_prefix_density_matches_full_minus_prefix_diffuse_gaussian(order):
    model, samples, onsets, membership = analytic_model(order)
    n, h = len(samples), 32
    covariance = dense_noise(model.noise.coefficients, n)
    drift = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    expected = []
    for target in membership.T:
        _, design = basis_and_design(onsets, target, n)
        full = diffuse_density(samples, design, drift, covariance, model.kernel_mean,
                               model.kernel_row_covariance, model.innovation_covariance)
        prefix = diffuse_density(samples[:h], design[:h], drift[:h], covariance[:h, :h],
                                 model.kernel_mean, model.kernel_row_covariance, model.innovation_covariance)
        expected.append(full - prefix)
    actual = score(model, samples, onsets, membership)
    np.testing.assert_allclose(actual.log_evidence, expected, atol=3e-9, rtol=1e-10)
    np.testing.assert_allclose(actual.log_probability, expected - logsumexp(expected), atol=3e-9)
    assert actual.n_unique_samples == n
    assert actual.record()["conditioning_sample_support"] == [0, 32]
    assert actual.record()["score_sample_support"] == [32, n]
    assert actual.record()["n_scored_samples"] == n - 32


def test_ar0_prefix_updates_common_kernel_and_diffuse_drift_without_mutating_model():
    model, samples, onsets, membership = analytic_model(0)
    original = model._metadata()
    changed = samples.copy()
    changed[:32, 0] += 3 * basis_and_design(onsets, membership[:, 0], len(samples))[1][:32, 0]
    first = score(model, samples, onsets, membership)
    second = score(model, changed, onsets, membership)
    assert np.max(np.abs(first.probabilities - second.probabilities)) > 1e-5
    assert model._metadata() == original


@pytest.mark.parametrize("name", ["basis", "kernel_mean", "kernel_row_covariance", "innovation_covariance"])
def test_real_gaussian_parameters_do_not_silently_discard_complex_components(name):
    model, _, _, _ = analytic_model(1)
    value = getattr(model, name).astype(complex) + 1j
    with pytest.raises(ValueError, match="real numeric"):
        replace(model, **{name: value})


def calibration():
    rng = np.random.default_rng(61005)
    onsets = np.array([3, 25, 49, 75, 102, 132, 161, 190])
    coefficients = np.array([[.3, -.1], [.15, .2], [.65, .25], [-.2, .3]])
    records = []
    for index in range(4):
        labels = ((np.arange(len(onsets)) + index) % 3 == 0).astype(int)
        basis, design = basis_and_design(onsets, labels, 212)
        noise = rng.normal(scale=.4, size=(212, 2))
        for row in range(1, len(noise)):
            noise[row] += .6 * noise[row - 1]
        drift = np.column_stack((np.ones(212), np.linspace(-1, 1, 212)))
        samples = design @ coefficients + noise + drift @ rng.normal(scale=.2, size=(2, 2))
        records.append(CalibrationRecord(f"calibration{index}", samples, onsets, labels))
    return basis, records


def fit(records, basis, order, strategy):
    return ContinuousDecisionModel.fit(
        records, sfreq=50., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"),
        operator_identity="native", basis=basis, noise_order=order,
        conditioning_samples=32, noise_estimator=strategy,
    )


def dense_training_reference(model, records):
    full_a, full_y, prefix_a, prefix_y, tail_a, tail_y = [], [], [], [], [], []
    width = model.kernel_mean.shape[0] + 2 * len(records)
    for index, record in enumerate(records):
        n = len(record.samples)
        _, design = basis_and_design(record.onsets, record.target_mask, n)
        drift = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
        row_root = np.linalg.cholesky(dense_noise(model.noise.coefficients, n))
        a = np.zeros((n, width))
        a[:, :4] = design
        a[:, 4 + 2 * index:6 + 2 * index] = drift
        a, y = np.linalg.solve(row_root, a), np.linalg.solve(row_root, record.samples)
        full_a.append(a)
        full_y.append(y)
        prefix_a.append(a[:32])
        prefix_y.append(y[:32])
        tail_a.append(a[32:])
        tail_y.append(y[32:])
    full_a, full_y = np.vstack(full_a), np.vstack(full_y)
    prefix_a, prefix_y, tail_a, tail_y = map(np.vstack, (prefix_a, prefix_y, tail_a, tail_y))
    precision = np.diag([1.] * 4 + [0.] * (width - 4))
    information = precision + full_a.T @ full_a
    mean = np.linalg.solve(information, full_a.T @ full_y)
    covariance = np.linalg.inv(information)
    prefix_covariance = np.linalg.inv(precision + prefix_a.T @ prefix_a)
    prefix_mean = prefix_covariance @ prefix_a.T @ prefix_y
    prediction_covariance = np.eye(len(tail_y)) + tail_a @ prefix_covariance @ tail_a.T
    residual = tail_y - tail_a @ prefix_mean
    quadratic = residual.T @ np.linalg.solve(prediction_covariance, residual)
    spatial = quadratic / len(tail_y)
    objective = .5 * (len(tail_y) * np.linalg.slogdet(spatial)[1]
                      + spatial.shape[0] * np.linalg.slogdet(prediction_covariance)[1])
    return mean[:4], covariance[:4, :4], spatial, objective


def test_conditional_fit_jointly_updates_one_shared_kernel_and_profiles_tail_covariance():
    basis, records = calibration()
    model = fit(records, basis, 1, "burg")
    mean, covariance, spatial, objective = dense_training_reference(model, records)
    np.testing.assert_allclose(model.kernel_mean, mean, atol=2e-10)
    np.testing.assert_allclose(model.kernel_row_covariance, covariance, atol=2e-10)
    np.testing.assert_allclose(model.innovation_covariance, spatial, atol=2e-10)
    np.testing.assert_allclose(model.noise.provenance["optimization"]["final_objective"], objective, atol=2e-8)
    assert model.noise.provenance["objective_rows"] == 4 * (212 - 32)
    assert model.provenance["kernel_observation_support"] == "all_authorized_prefix_and_tail"
    assert all(tuple(item["score_sample_support"]) == (32, 212) for item in model.provenance["records"])


def test_conditional_records_have_independent_drift_and_starts_with_order_invariant_fit():
    basis, records = calibration()
    records[1] = replace(records[1], samples=np.vstack((records[1].samples, records[1].samples[-9:])))
    records[2] = replace(records[2], samples=records[2].samples[:-5])
    model = fit(records, basis, 4, "burg")
    reordered = fit(list(reversed(records)), basis, 4, "burg")
    np.testing.assert_allclose(model.kernel_mean, reordered.kernel_mean, atol=2e-10)
    np.testing.assert_allclose(model.kernel_row_covariance, reordered.kernel_row_covariance, atol=2e-10)
    np.testing.assert_allclose(model.innovation_covariance, reordered.innovation_covariance, atol=2e-10)
    assert model.noise.provenance["objective_rows"] == sum(len(record.samples) - 32 for record in records)
    initialization = model.noise.provenance["initialization"]
    assert initialization["provenance"]["available_lag_rows"] == sum(len(record.samples) - 4 for record in records)
    assert [row["n_samples"] for row in initialization["provenance"]["records"]] == [len(record.samples) for record in records]


def test_conditional_parcor_optimizer_uses_the_same_tail_objective_and_records_convergence(tmp_path):
    basis, records = calibration()
    model = fit(records, basis, 1, "conditional_profile")
    optimization = model.noise.provenance["optimization"]
    assert optimization["strategy"] == "conditional_profile" and optimization["converged"]
    assert optimization["final_objective"] <= optimization["initial_objective"] + 1e-8
    assert optimization["evaluations"] > 1
    _, _, _, expected = dense_training_reference(model, records)
    np.testing.assert_allclose(optimization["final_objective"], expected, atol=2e-8)
    path = tmp_path / "conditional.npz"
    model.save(path)
    restored = ContinuousDecisionModel.load(path)
    assert restored.conditioning_samples == 32
    held = replace(records[0], record_id="held", samples=records[0].samples + .01)
    membership = np.column_stack((held.target_mask, 1 - held.target_mask, np.arange(len(held.onsets)) % 2))
    before = score(model, held.samples, held.onsets, membership)
    after = score(restored, held.samples, held.onsets, membership)
    np.testing.assert_array_equal(after.log_probability, before.log_probability)


@pytest.mark.parametrize("change", ["estimator", "solver"])
def test_conditional_model_does_not_relabel_burg_as_actual_optimization(change):
    basis, records = calibration()
    model = fit(records, basis, 1, "burg")
    provenance = {**model.provenance, "noise_solver": dict(model.provenance["noise_solver"])}
    if change == "estimator":
        provenance["noise_estimator"] = "conditional_profile"
    else:
        provenance["noise_solver"]["method"] = "L-BFGS-B"
    with pytest.raises(ValueError, match="actual noise|solver"):
        replace(model, provenance=provenance)


def test_conditional_ar0_records_zero_shape_solver_instead_of_claiming_optimizer_call():
    basis, records = calibration()
    model = fit(records, basis, 0, "conditional_profile")
    assert model.provenance["noise_solver"]["method"] == "no_AR_shape_parameters"
    assert model.noise.provenance["optimization"]["iterations"] == 0


@pytest.mark.parametrize("prefix", [True, 1, 31, 212])
def test_conditional_support_refuses_unidentified_or_order_short_prefix(prefix):
    basis, records = calibration()
    with pytest.raises(ValueError, match="prefix|conditioning"):
        ContinuousDecisionModel.fit(records, sfreq=50., channels=("Fz", "Pz"),
            candidate_ids=("a", "b", "c"), operator_identity="native", basis=basis,
            noise_order=32, conditioning_samples=prefix)
