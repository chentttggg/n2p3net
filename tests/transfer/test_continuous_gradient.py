"""Fit gradient versus independent dense covariance/diffuse integration."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_discrete_lyapunov, toeplitz

from research.execution import UnidentifiableFitError
from transfer.continuous_decision import _conditional_prediction
from transfer.continuous_gradient import ConditionalGaussianObjective
from transfer.temporal_noise import TemporalNoise

PREFIX = 32
RIDGE = .8

# Actual A01 five-selection calibration Burg initialization. MAT SHA256:
# cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd.
# Initializer artifacts 2fd460a2...1393e (p8) and 7760cb67...738d (p32).
# Only the noise shapes are fixtures; no held observations or labels are used.
REAL_BURG_32 = np.array([
    -.9179361414037359, .941654433910308, -.8834138309108374, .7850526107482318,
    .1640890212603942, -.34264543125862656, -.09020187587414617, .2743362891496982,
    .25571179766118823, -.0014986013633522168, -.13973896945420738, -.043031075699770056,
    .10008948279252985, .09399265537572925, .026118500717667123, -.16492389534301538,
    .005912806358045626, .04177717376529358, .014716577263296813, -.01106186637823118,
    -.030370565292304908, -.030766063461246506, -.003639866836395913, .024272167638461142,
    .01835559738831268, -.014468391824641933, -.03255457783201604, -.006505970881834437,
    .007859939996292501, .029807231926041372, -.009113122875125307, -.03566044751394994,
])


def calibration_arrays():
    rng = np.random.default_rng(105710)
    samples, designs = [], []
    coefficients = np.array([[.7, -.25], [-.1, .3], [.2, .45]])
    for index, n in enumerate((69, 83, 101)):
        time = np.arange(n)
        design = np.column_stack((np.sin(time / 4 + index), np.cos(time / 9),
                                  np.exp(-((time - 18 - index) / 9) ** 2)))
        drift = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
        y = design @ coefficients + rng.normal(scale=.6, size=(n, 2))
        y += drift @ rng.normal(size=(2, 2))
        samples.append(y)
        designs.append(design)
    return samples, designs


def moderate_eta(order):
    reflections = .018 * np.cos(np.arange(order))
    if order:
        reflections[0] = -.54
    return np.arctanh(reflections)


def stationary_covariance(eta, n):
    """Independent expanded polynomial + stationary Lyapunov covariance."""
    polynomial = np.empty(0)
    for reflection in np.tanh(eta):
        polynomial = np.r_[polynomial + reflection * polynomial[::-1], reflection]
    if not len(polynomial):
        return np.eye(n)
    transition = np.zeros((len(polynomial), len(polynomial)))
    transition[0] = -polynomial
    transition[1:, :-1] = np.eye(len(polynomial) - 1)
    driving = np.zeros_like(transition)
    driving[0, 0] = 1.
    covariance = solve_discrete_lyapunov(transition, driving)
    state = covariance[:, 0].copy()
    autocovariance = []
    for _ in range(n):
        autocovariance.append(state[0])
        state = transition @ state
    return toeplitz(autocovariance)


def dense_diffuse_terms(y, design, drift, noise, prior):
    covariance = noise + design @ prior @ design.T
    root = cho_factor(covariance, lower=True)
    precision_drift = cho_solve(root, drift)
    information = drift.T @ precision_drift
    precision_y = cho_solve(root, y)
    projection = drift.T @ precision_y
    scatter = y.T @ precision_y - projection.T @ np.linalg.solve(information, projection)
    logdet = (2 * np.log(np.diag(root[0])).sum()
              + np.linalg.slogdet(information)[1])
    return scatter, logdet


def high_precision_stationary_covariance(eta, n):
    """60-digit Yule-Walker ACF, independent of lattice/expanded Lyapunov."""
    if not len(eta):
        return np.eye(n)
    with localcontext() as context:
        context.prec = 60
        reflections = [Decimal(float(value)) for value in np.tanh(eta)]
        error = Decimal(1)
        for reflection in reflections:
            error /= 1 - reflection * reflection
        autocovariance, polynomial = [error], []
        for reflection in reflections:
            lag = len(polynomial) + 1
            autocovariance.append(-reflection * error - sum(
                coefficient * autocovariance[lag - index - 1]
                for index, coefficient in enumerate(polynomial)
            ))
            polynomial = [coefficient + reflection * polynomial[-index - 1]
                          for index, coefficient in enumerate(polynomial)] + [reflection]
            error *= 1 - reflection * reflection
        for lag in range(len(autocovariance), n):
            autocovariance.append(-sum(coefficient * autocovariance[lag - index - 1]
                                      for index, coefficient in enumerate(polynomial)))
        return toeplitz([float(value) for value in autocovariance[:n]])


def dense_conditional_terms(samples, designs, eta, *, prior=None, covariance=stationary_covariance):
    channels, kernel_width = samples[0].shape[1], designs[0].shape[1]
    if prior is None:
        prior = np.eye(kernel_width) / RIDGE
    lengths = [len(y) for y in samples]
    y, design = np.vstack(samples), np.vstack(designs)
    drift = block_diag(*[np.column_stack((np.ones(n), np.linspace(-1, 1, n))) for n in lengths])
    noise = block_diag(*[covariance(eta, n) for n in lengths])
    starts = np.cumsum([0] + lengths[:-1])
    indices = np.concatenate([np.arange(start, start + PREFIX) for start in starts])
    full_q, full_logdet = dense_diffuse_terms(y, design, drift, noise, prior)
    prefix_q, prefix_logdet = dense_diffuse_terms(
        y[indices], design[indices], drift[indices], noise[np.ix_(indices, indices)], prior
    )
    scatter, logdet = full_q - prefix_q, full_logdet - prefix_logdet
    rows = sum(n - PREFIX for n in lengths)
    value = .5 * (np.linalg.slogdet(scatter / rows)[1] + channels * logdet / rows)
    return value, scatter, logdet


def central_difference(function, eta, *, step=1e-5):
    result = np.empty(len(eta), dtype=np.float64)
    for index in range(len(eta)):
        shift = np.zeros_like(eta)
        shift[index] = step
        result[index] = (function(eta + shift) - function(eta - shift)) / (2 * step)
    return result


def torch_dense_value(eta, samples, designs):
    """Autograd oracle via stationary ACF and dense Gaussian integration.

    This builds time covariance rather than whitening observations, then
    subtracts full/prefix diffuse densities. No producer lattice or QR is used.
    """
    reflections = torch.tanh(eta)
    maximum = max(len(y) for y in samples)
    if len(eta):
        error = 1 / torch.prod(1 - reflections ** 2)
        acf, polynomial = [error], []
        for reflection in reflections:
            lag = len(polynomial) + 1
            acf.append(-reflection * error - sum(
                coefficient * acf[lag - index - 1] for index, coefficient in enumerate(polynomial)
            ))
            polynomial = [coefficient + reflection * polynomial[-index - 1]
                          for index, coefficient in enumerate(polynomial)] + [reflection]
            error = error * (1 - reflection ** 2)
        for lag in range(len(acf), maximum):
            acf.append(-sum(coefficient * acf[lag - index - 1]
                            for index, coefficient in enumerate(polynomial)))
        acf = torch.stack(acf)
        row = torch.arange(maximum)
        largest_covariance = acf[torch.abs(row[:, None] - row[None, :])]
    else:
        largest_covariance = torch.eye(maximum, dtype=torch.float64)
    y = torch.tensor(np.vstack(samples), dtype=torch.float64)
    design = torch.tensor(np.vstack(designs), dtype=torch.float64)
    drift = torch.tensor(block_diag(*[
        np.column_stack((np.ones(len(value)), np.linspace(-1, 1, len(value))))
        for value in samples
    ]), dtype=torch.float64)
    noise = torch.block_diag(*[largest_covariance[:len(value), :len(value)] for value in samples])
    covariance = noise + design @ design.T / RIDGE
    prefix_indices = torch.tensor(np.concatenate([
        np.arange(start, start + PREFIX) for start in np.cumsum([0] + [len(value) for value in samples[:-1]])
    ]))

    def terms(data, z, sigma):
        root = torch.linalg.cholesky(sigma)
        yw = torch.linalg.solve_triangular(root, data, upper=False)
        zw = torch.linalg.solve_triangular(root, z, upper=False)
        information = zw.T @ zw
        drift_mean = torch.linalg.solve(information, zw.T @ yw)
        residual = yw - zw @ drift_mean
        return (residual.T @ residual,
                2 * torch.log(torch.diag(root)).sum() + torch.linalg.slogdet(information)[1])

    full_q, full_logdet = terms(y, drift, covariance)
    prefix_q, prefix_logdet = terms(
        y[prefix_indices], drift[prefix_indices], covariance[prefix_indices][:, prefix_indices]
    )
    rows = sum(len(value) - PREFIX for value in samples)
    return .5 * (torch.linalg.slogdet((full_q - prefix_q) / rows)[1]
                 + y.shape[1] * (full_logdet - prefix_logdet) / rows)


@pytest.mark.parametrize("order", [0, 1, 4, 8, 16, 32])
def test_value_scatter_and_exact_gradient_match_independent_dense_gaussian(order):
    samples, designs = calibration_arrays()
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    eta = moderate_eta(order)
    actual, gradient = producer.value_gradient(eta)
    expected, expected_q, expected_logdet = dense_conditional_terms(samples, designs, eta)
    scatter, logdet = producer._conditional_terms(torch.tensor(eta, dtype=torch.float64))
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(scatter.numpy(), expected_q, atol=3e-8, rtol=2e-10)
    np.testing.assert_allclose(float(logdet), expected_logdet, atol=2e-10, rtol=0)
    expected_gradient = central_difference(lambda point: dense_conditional_terms(samples, designs, point)[0], eta)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-7, rtol=2e-6)
    assert gradient.dtype == np.float64
    assert producer.objective_rows == sum(len(y) - PREFIX for y in samples)


@pytest.mark.parametrize("order", [1, 4, 8, 16, 32])
def test_gradient_matches_separate_covariance_autograd_oracle(order):
    samples, designs = calibration_arrays()
    eta = moderate_eta(order)
    point = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    expected = torch_dense_value(point, samples, designs)
    expected_gradient, = torch.autograd.grad(expected, point)
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    value, gradient = producer.value_gradient(eta)
    np.testing.assert_allclose(value, float(expected.detach()), atol=2e-10, rtol=0)
    np.testing.assert_allclose(gradient, expected_gradient.numpy(), atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("order", [8, 32])
def test_actual_high_correlation_burg_shapes_match_dense_and_central_difference(order):
    samples, designs = calibration_arrays()
    eta = np.arctanh(REAL_BURG_32[:order])
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    value, gradient = producer.value_gradient(eta)
    def reference(point):
        return dense_conditional_terms(samples, designs, point, covariance=high_precision_stationary_covariance)
    expected, expected_q, expected_logdet = reference(eta)
    scatter, logdet = producer._conditional_terms(torch.tensor(eta, dtype=torch.float64))
    np.testing.assert_allclose(value, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(scatter.numpy(), expected_q, atol=2e-6, rtol=2e-8)
    np.testing.assert_allclose(float(logdet), expected_logdet, atol=2e-7, rtol=0)
    expected_gradient = central_difference(lambda point: reference(point)[0], eta, step=1e-4)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-7, rtol=2e-7)
    point = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    dense_value = torch_dense_value(point, samples, designs)
    dense_gradient, = torch.autograd.grad(dense_value, point)
    np.testing.assert_allclose(value, float(dense_value.detach()), atol=2e-9, rtol=0)
    np.testing.assert_allclose(gradient, dense_gradient.numpy(), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("order", [0, 1, 4, 8, 16, 32])
@pytest.mark.parametrize("shape", ["moderate", "actual_A01_Burg"])
def test_matches_current_numpy_conditional_prediction_for_the_same_prior(order, shape):
    samples, designs = calibration_arrays()
    eta = moderate_eta(order) if shape == "moderate" else np.arctanh(REAL_BURG_32[:order])
    noise = TemporalNoise.from_reflections(np.tanh(eta), innovation_variance=1., reference_id="numeric_reference")
    expected_q, expected_logdet, rows, _, _ = _conditional_prediction(
        samples, designs, noise, PREFIX, np.zeros((3, 2)), np.eye(3) / RIDGE
    )
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    scatter, logdet = producer._conditional_terms(torch.tensor(eta, dtype=torch.float64))
    np.testing.assert_allclose(scatter.numpy(), expected_q, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(float(logdet), expected_logdet, atol=2e-10)
    assert rows == producer.objective_rows


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_voltage_units_change_profile_offset_and_preserve_gradient(scale):
    samples, designs = calibration_arrays()
    eta = np.arctanh(REAL_BURG_32)
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    scaled = ConditionalGaussianObjective([y * scale for y in samples], designs,
                                         conditioning_samples=PREFIX, ridge=RIDGE)
    value, gradient = producer.value_gradient(eta)
    scaled_value, scaled_gradient = scaled.value_gradient(eta)
    np.testing.assert_allclose(scaled_value - value, samples[0].shape[1] * np.log(scale), atol=3e-9)
    np.testing.assert_allclose(scaled_gradient, gradient, atol=3e-9, rtol=3e-9)


def test_record_permutation_and_independent_drift_do_not_leak_ar_state_or_padding():
    samples, designs = calibration_arrays()
    eta = np.arctanh(REAL_BURG_32)
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    value, gradient = producer.value_gradient(eta)
    order = [2, 0, 1]
    moved = ConditionalGaussianObjective([samples[index] for index in order], [designs[index] for index in order],
                                        conditioning_samples=PREFIX, ridge=RIDGE)
    moved_value, moved_gradient = moved.value_gradient(eta)
    np.testing.assert_allclose(moved_value, value, atol=2e-10)
    np.testing.assert_allclose(moved_gradient, gradient, atol=2e-9)
    shifted = [y.copy() for y in samples]
    shifted[1] += np.column_stack((np.ones(len(shifted[1])), np.arange(len(shifted[1])))) @ np.array([[19., -31.], [.6, -.35]])
    shifted_producer = ConditionalGaussianObjective(shifted, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    shifted_value, shifted_gradient = shifted_producer.value_gradient(eta)
    np.testing.assert_allclose(shifted_value, value, atol=3e-9)
    np.testing.assert_allclose(shifted_gradient, gradient, atol=3e-8)
    assert shifted_producer.objective_rows == producer.objective_rows == 157


def test_real_prefix_information_updates_one_shared_kernel_even_at_ar0():
    samples, designs = calibration_arrays()
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    changed = [y.copy() for y in samples]
    changed[0][:PREFIX] += designs[0][:PREFIX, :1] * np.array([2., -1.])
    changed_producer = ConditionalGaussianObjective(changed, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    original_value, _ = producer.value_gradient(np.array([]))
    changed_value, _ = changed_producer.value_gradient(np.array([]))
    assert abs(changed_value - original_value) > .01
    expected, _, _ = dense_conditional_terms(changed, designs, np.array([]))
    np.testing.assert_allclose(changed_value, expected, atol=2e-10)
    independent_value = sum(
        ConditionalGaussianObjective([y], [d], conditioning_samples=PREFIX, ridge=RIDGE).value_gradient(np.array([]))[0] * (len(y) - PREFIX)
        for y, d in zip(changed, designs, strict=True)
    ) / producer.objective_rows
    assert abs(changed_value - independent_value) > .001


def test_basis_coordinates_require_a_coupled_prior_beyond_orthogonal_changes():
    samples, designs = calibration_arrays()
    eta = moderate_eta(4)
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    value, gradient = producer.value_gradient(eta)
    rotation, _ = np.linalg.qr(np.array([[1., .3, .2], [-.2, 1., .4], [.5, -.1, 1.]]))
    rotated = ConditionalGaussianObjective(samples, [d @ rotation for d in designs],
                                          conditioning_samples=PREFIX, ridge=RIDGE)
    rotated_value, rotated_gradient = rotated.value_gradient(eta)
    np.testing.assert_allclose(rotated_value, value, atol=2e-10)
    np.testing.assert_allclose(rotated_gradient, gradient, atol=2e-9)
    change = np.diag([.2, 3., 5.])
    transformed_designs = [d @ change for d in designs]
    transformed = ConditionalGaussianObjective(samples, transformed_designs, conditioning_samples=PREFIX, ridge=RIDGE)
    assert abs(transformed.value_gradient(eta)[0] - value) > .001
    inverse = np.linalg.inv(change)
    coupled_prior = inverse @ (np.eye(3) / RIDGE) @ inverse.T
    coupled_value, _, _ = dense_conditional_terms(samples, transformed_designs, eta, prior=coupled_prior)
    np.testing.assert_allclose(coupled_value, value, atol=2e-10)


def test_owned_ingress_and_readonly_rows_survive_caller_mutation_and_grad_context():
    samples, designs = calibration_arrays()
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    eta = moderate_eta(8)
    value, gradient = producer.value_gradient(eta)
    for y, design in zip(samples, designs, strict=True):
        y[:] = 999.
        design[:] = -77.
    with torch.no_grad():
        new_value, new_gradient = producer.value_gradient(eta)
    assert new_value == value
    np.testing.assert_array_equal(new_gradient, gradient)
    with pytest.raises(AttributeError):
        producer.objective_rows = 1
    assert all(value.dtype == torch.float64 and value.device.type == "cpu" for value in producer._raw)


def test_cpu_float64_execution_is_explicit_under_a_different_default_device():
    samples, designs = calibration_arrays()
    eta = moderate_eta(4)
    expected = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE).value_gradient(eta)
    with torch.device("meta"):
        producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
        value, gradient = producer.value_gradient(eta)
    assert value == expected[0]
    np.testing.assert_array_equal(gradient, expected[1])
    assert all(tensor.dtype == torch.float64 and tensor.device.type == "cpu" for tensor in producer._raw)


@pytest.mark.parametrize("eta", [np.zeros((2, 1)), np.full(33, .1), np.array([np.nan]),
                                 np.array([np.inf]), np.array([1j]), np.array([30.])])
def test_parameter_errors_are_explicit(eta):
    samples, designs = calibration_arrays()
    producer = ConditionalGaussianObjective(samples, designs, conditioning_samples=PREFIX, ridge=RIDGE)
    with pytest.raises(ValueError, match="eta must"):
        producer.value_gradient(eta)


def test_unidentifiable_spatial_scatter_is_expected_fit_failure():
    samples, designs = calibration_arrays()
    producer = ConditionalGaussianObjective([np.zeros_like(y) for y in samples], designs,
                                            conditioning_samples=PREFIX, ridge=RIDGE)
    with pytest.raises(UnidentifiableFitError, match="not identifiable"):
        producer.value_gradient(moderate_eta(8))
