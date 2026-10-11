"""Numerical admission follows actual operations, charts and temporal support."""
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from research import channel_noise_mechanism as cm


def _dyadic_fixture():
    time = np.arange(24, dtype=float)
    design = np.column_stack(((time % 3 - 1) / 8, ((time // 3) % 2 - .5) / 4))
    design[:8] = 0.
    nuisance = np.column_stack((np.ones(24), (time - 3.5) * 2.**-36))
    response = (time % 5 - 2) / 8 + (time - 3.5) / 64
    return design, response, nuisance, 8


def _conditional(design, response, nuisance, cut):
    # The numerical owner must stabilize a flat coordinate scale without
    # changing the prior or the conditional model.
    nuisance, _, _ = cm._prefix_nuisance_chart(nuisance, cut)
    matrix = np.column_stack((design, nuisance))
    proper = np.zeros((design.shape[1], matrix.shape[1]))
    proper[:, :design.shape[1]] = np.eye(design.shape[1])
    rhs, prefix, prefix_q, smallest = cm._posterior_qr(matrix[:cut], response[:cut], proper)
    full = np.vstack((matrix[cut:], prefix))
    full_y = np.r_[response[cut:], rhs]
    q, root = np.linalg.qr(full, mode="reduced")
    full_rhs = cm._row_dot(q, full_y)
    mean = solve_triangular(root, full_rhs)
    remaining = response[cut:] - matrix[cut:] @ mean
    shift = prefix @ mean - rhs
    quadratic = float(remaining @ remaining + shift @ shift)
    resolution = cm._conditional_score_resolution(matrix, response, proper, cut, prefix_q,
        prefix, smallest, full, full_y, q, root, full_rhs, mean, remaining, shift, quadratic,
        cm._SegmentError(0., 0.), cm._SegmentError(0., 0.))
    logdet = 2 * (np.log(np.abs(np.diag(root))).sum()
                  - np.log(np.abs(np.diag(prefix))).sum())
    return -.5 * (quadratic + logdet + (len(response) - cut) * np.log(2 * np.pi)), resolution


def test_flat_coordinate_scale_does_not_create_conditional_unavailability():
    design, response, nuisance, cut = _dyadic_fixture()
    density, resolution = _conditional(design, response, nuisance, cut)
    # Independently computed 80-digit integral of original dyadic inputs;
    # preserves one raw-coordinate flat da and an I_2 proper prior.
    assert abs(density - (-17.335675285048127)) < 2e-11
    assert resolution["absolute_log_density_error_estimate"] < 16*np.sqrt(np.finfo(float).eps)
    unscaled = nuisance.copy()
    unscaled[:, 1] *= 2.**36
    reference, _ = _conditional(design, response, unscaled, cut)
    assert abs(reference-density) < 2e-11


def test_long_exact_flat_design_does_not_acquire_spurious_prior_information():
    # N=2^16 makes the constant nuisance Q exactly dyadic in ideal arithmetic.
    # The old full-length gamma_N rejects although the complete proper chart
    # contains no retained signal; a short actual reduction tree resolves it.
    nuisance = np.ones((65536, 1))
    design = nuisance * 256.
    result = cm.project_diffuse_nuisance(design, np.zeros(len(design)), nuisance,
        prior_factor=np.eye(1))
    assert np.linalg.norm(result.design) < 1e-8
    assert result.resolution["normalized_error_bound"] < np.sqrt(np.finfo(float).eps)


def test_causal_whitening_zero_prefix_error_keeps_nonzero_history_in_tail():
    geometry = cm.NoiseGeometry(("Cz",), 128., "V", "toy-ear", "toy-native", 1)
    noise = cm.FittedChannelNoise.from_parameters([[.5, -.125]], np.eye(1), geometry=geometry,
        conditioning_samples=4, reference_id="error-history", family="shared")
    raw = np.zeros((12, 1))
    raw[3] = .25
    raw[7] = -.5
    source_error = np.zeros_like(raw)
    source_error[3] = 2.**-30
    white = noise.whiten(raw)
    error = cm._score_whitening_error(noise, raw, white, source_error,
        cm._score_whitening_geometry(noise))
    assert error.prefix > 0 and error.tail > 0
    # Perturb only the last prefix raw sample: AR tail must use that history.
    changed = raw.copy()
    changed[3] += source_error[3]
    delta = noise.whiten(changed)-white
    assert np.linalg.norm(delta[:4]) <= error.prefix
    assert np.linalg.norm(delta[4:]) <= error.tail
    zero_prefix = raw.copy()
    zero_prefix[:4] = 0.
    structural = cm._score_whitening_error(noise, zero_prefix, noise.whiten(zero_prefix),
        np.zeros_like(raw), cm._score_whitening_geometry(noise))
    assert structural.prefix == 0.


def test_whitening_error_covers_exact_stored_prefix_triangular_solve():
    geometry = cm.NoiseGeometry(("Cz",), 128., "V", "toy-ear", "toy-native", 1)
    noise = cm.FittedChannelNoise.from_parameters([[.9375]], np.eye(1), geometry=geometry,
        conditioning_samples=8, reference_id="prefix-error", family="shared")
    raw = ((np.arange(16) % 5 - 2)/16)[:, None]
    white = noise.whiten(raw)
    error = cm._score_whitening_error(noise, raw, white, np.zeros_like(raw),
        cm._score_whitening_geometry(noise))
    with localcontext() as context:
        context.prec = 80
        root = noise.compiled_prefix_root
        exact = []
        for row in range(8):
            rhs = Decimal.from_float(float(raw[row, 0]))
            rhs -= sum(Decimal.from_float(float(root[row, col]))*exact[col] for col in range(row))
            exact.append(rhs/Decimal.from_float(float(root[row, row])))
        difference = sum((Decimal.from_float(float(white[row, 0]))-exact[row])**2
                         for row in range(8)).sqrt()
    assert float(difference) <= error.prefix


def test_common_voltage_density_offset_does_not_damage_probability_normalization():
    geometry = cm.NoiseGeometry(("Cz",), 128., "V", "toy-ear", "toy-native", 1)
    noise = cm.FittedChannelNoise.from_parameters([[]], np.array([[1e-48]]), geometry=geometry,
        conditioning_samples=4, reference_id="probability-offset", family="shared")
    response = cm.FrozenERPResponse(np.zeros((1, 1)), np.empty((1, 0)), geometry,
        "a"*64, "known-zero", {"fit_scope": "analytic_reference"})
    score = noise.score(np.zeros((128, 1)), np.zeros((3, 128, 1)), response,
        candidate_ids=("a", "b", "c"), geometry=geometry, record_id="held",
        recording_id="held-record", sample_start=0)
    assert abs(sum(score["probabilities"])-1.) < 8*np.finfo(float).eps
    np.testing.assert_allclose(score["probabilities"], [1/3]*3, rtol=0, atol=8*np.finfo(float).eps)


def test_dekker_internal_overflow_is_typed_even_when_stored_product_is_finite(monkeypatch):
    root = np.array([[np.sqrt(np.finfo(float).max)]])
    image = root.copy()
    raw = root@image
    assert np.isfinite(raw).all()
    monkeypatch.setattr(cm, "math", SimpleNamespace(fsum=math.fsum))
    with pytest.raises(cm.DiffuseProjectionResolutionError, match="whitening residual"):
        cm._triangular_residual(root, raw, image)
    geometry = cm.NoiseGeometry(("Cz",), 128., "V", "toy-ear", "toy-native", 1)
    # The existing covariance constructor's diagnostic Frobenius norm can
    # overflow at this legal endpoint; this fixture tests the solve boundary.
    with np.errstate(over="ignore"):
        noise = cm.FittedChannelNoise.from_parameters([[]], np.array([[np.finfo(float).max/4]]),
            geometry=geometry, conditioning_samples=2, reference_id="finite-Dekker-overflow", family="shared")
    response = cm.FrozenERPResponse(np.zeros((1, 1)), np.empty((1, 0)), geometry,
        "a"*64, "known-zero", {"fit_scope": "analytic_reference"})
    samples = np.zeros((7, 1))
    samples[1] = raw[0, 0]
    with pytest.raises(cm.DiffuseProjectionResolutionError):
        noise.score(samples, np.zeros((2, 7, 1)), response, candidate_ids=("a", "b"),
            geometry=geometry, record_id="held", recording_id="held-record", sample_start=0)
