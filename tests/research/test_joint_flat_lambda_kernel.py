"""Distinguish the fixed-q continuous law, measure, weak directions and endpoints."""
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from research.channel_noise_mechanism import DiffuseProjectionResolutionError
from research.joint_flat_lambda_kernel import (
    GammaEnvelope,
    SingleRecordLambdaKernel,
    integrate_single_record_precision,
    marginal_log_measure_constant,
    prepare_single_record_lambda_kernel,
)


def _prepare(h, b, y):
    h, b, y = (np.asarray(a, dtype=float) for a in (h, b, y))
    r, k = h.shape[1], b.shape[1]
    return prepare_single_record_lambda_kernel(np.column_stack((h, b, y)), np.zeros(r+k+1),
        np.empty((0, r+1)), np.empty((0, r+1)), response_width=k, channels=1, drift_width=r)


def _scalar(s, z, *, log_residual=-math.inf, d=1, coupling=0., source_h=0.):
    return SingleRecordLambdaKernel(np.asarray(s), np.asarray(z), log_residual, 0., d, len(s),
        np.full(len(s), coupling), 1., 0., source_h, 0., 0., {})


def _one_beta_decimal(h, b, y, lam, variance=1):
    """Direct 2x2 proper/physical-flat precision, not a spectral/QR formula."""
    with localcontext() as context:
        context.prec = 100
        h, b, y = ([Decimal.from_float(float(v)) for v in row] for row in (h, b, y))
        lam, variance = Decimal.from_float(lam), Decimal(variance)
        hh, bb, hb = sum(v*v for v in h), sum(v*v for v in b), sum(v*w for v, w in zip(h, b, strict=True))
        hy = sum(v*w for v, w in zip(h, y, strict=True))
        by = sum(v*w for v, w in zip(b, y, strict=True))
        yy = sum(v*v for v in y)
        a, diagonal = lam*hh, 1+lam*variance*bb
        det = a*diagonal-lam*lam*variance*hb*hb
        quadratic = diagonal*(lam*hy)**2-2*(lam*variance*hb)*(lam*hy)*(lam*by)+a*variance*(lam*by)**2
        energy = lam*yy-quadratic/det
        return -(energy+det.ln())/2


@pytest.mark.parametrize('duplicate', [False, True])
def test_proper_beta_occurs_once_and_keeps_its_unscaled_prior(duplicate):
    h, b, y = [1., 1., 1.], [-.7, .2, .8], [1., -.5, 2.]
    design = np.asarray(b)[:, None]
    if duplicate:
        design = np.column_stack((design, design))
    kernel = _prepare(np.asarray(h)[:, None], design, y)
    assert len(kernel.singular) == (2 if duplicate else 1)
    for lam in (1e-20, .003, 1., 1e8):
        value, error, _ = kernel.evaluate(math.log(lam))
        exact = _one_beta_decimal(h, b, y, lam, 2 if duplicate else 1)
        assert abs(Decimal.from_float(value)-exact) <= Decimal.from_float(error)


def test_zero_and_repeated_singular_values_keep_every_response_coordinate():
    zero = _prepare([[1.], [1.], [1.]], [[0.], [0.], [0.]], [1., -.5, 2.])
    assert zero.singular[0] == 0
    got, error, _ = zero.evaluate(math.log(3.))
    exact = _one_beta_decimal([1., 1., 1.], [0., 0., 0.], [1., -.5, 2.], 3.)
    assert abs(Decimal.from_float(got)-exact) <= Decimal.from_float(error)
    repeated = _scalar([1., 1.], [2., -3.], log_residual=math.log(16.))
    lam = 3.
    value, bound, _ = repeated.evaluate(math.log(lam))
    expected = -.5*(lam*16+lam*13/(1+lam))-.5*math.log(lam)-math.log1p(lam)
    assert abs(value-expected) <= bound


def test_direct_residual_retains_information_lost_by_energy_subtraction():
    kernel = _prepare([[1.], [0.], [0.]], [[0.], [1.], [0.]], [0., 1e8, 3.])
    assert kernel.residual_squared == pytest.approx(9., abs=1e-13)
    assert (1e16+9)-1e16 != kernel.residual_squared


@pytest.mark.parametrize('s,z,x,energy,beta', [
    (1e200, 1e200, 0., 1., 1.),
    (1e-150, 1e-150, math.log(1e300), .5, .5),
])
def test_log_products_recover_energy_and_beta_after_intermediate_underflow(s, z, x, energy, beta):
    value, error, diagnostics = _scalar([s], [z]).evaluate(x)
    assert math.isfinite(value) and math.isfinite(error)
    assert diagnostics['energy'] == pytest.approx(energy, rel=2e-12)
    assert diagnostics['beta_norm'] == pytest.approx(beta, rel=2e-12)


@pytest.mark.parametrize('norm,x', [(1e200, math.log(1e-300)), (1e-200, math.log(1e300))])
def test_direct_residual_log_scale_survives_squared_overflow_or_underflow(norm, x):
    value, error, _ = _scalar([0.], [0.], log_residual=2*math.log(norm)).evaluate(x)
    with localcontext() as context:
        context.prec = 100
        exact = -(Decimal.from_float(x).exp()*Decimal.from_float(norm)**2+Decimal.from_float(x))/2
        assert abs(Decimal.from_float(value)-exact) <= Decimal.from_float(error)


def test_source_metric_cannot_lose_recoverable_coupling_after_tau_underflows():
    kernel = _scalar([1e200], [0.], coupling=1e300, source_h=2e-100)
    with pytest.raises(DiffuseProjectionResolutionError, match='source metric'):
        kernel.evaluate(700.)


@pytest.mark.parametrize('kind', ['energy_sum', 'source_quadratic'])
def test_finite_inputs_with_unrepresentable_energy_or_error_raise_a_typed_failure(kind):
    if kind == 'energy_sum':
        kernel = _scalar([0., 0.], [1e154, 1e154])
    else:
        kernel = SingleRecordLambdaKernel(np.array([0.]), np.array([1.]), -math.inf,
            0., 1, 1, np.array([0.]), 1., 0., 0., 0., 1e200, {})
    with pytest.raises(DiffuseProjectionResolutionError, match='unrepresentable'):
        kernel.evaluate(0.)


def test_weak_response_is_retained_and_the_original_numerical_budget_can_be_applied():
    kernel = _prepare([[1.], [0.], [0.]], [[0.], [2.**-600], [0.]], [0., 1., 3.])
    assert kernel.singular[0] > 0
    value, error, _ = kernel.evaluate(0.)
    assert math.isfinite(value) and error < np.sqrt(np.finfo(float).eps)*3


def test_absolute_same_record_measure_matches_independent_student_t_conditional():
    yp, yt, nu, variance = np.array([1., -.5, 2.]), np.array([0., 3.]), 4., 9.
    values = {}
    for phase, y in (('P', yp), ('F', np.r_[yp, yt])):
        kernel = _prepare(np.ones((len(y), 1))/3, np.zeros((len(y), 1)), y/3)
        envelope = GammaEnvelope((len(y)+nu-1)/2, nu/2, -kernel.log_flat_root+1e-10)
        result = integrate_single_record_precision(kernel, raw_scalar_rows=len(y), nu=nu, envelope=envelope,
            centre_hint=0., scale_hint=.5, relative_tolerances=(1e-8, 1e-10), scan_step=.2)
        assert result['convergence']['status'] == 'numerical_checks_passed'
        values[phase] = result['log_integral']+marginal_log_measure_constant(raw_scalar_rows=len(y),
            flat_dimension=1, nu=nu, physical_flat_logabsdet=0., noise_logdet=len(y)*math.log(variance))
    mean, df = yp.mean(), nu+len(yp)-1
    scale = (nu+np.sum((yp-mean)**2)/variance)/df*variance*(np.eye(len(yt))+np.ones((len(yt), len(yt)))/len(yp))
    residual = yt-mean
    expected = (math.lgamma((df+len(yt))/2)-math.lgamma(df/2)-len(yt)/2*math.log(df*math.pi)
        -.5*np.linalg.slogdet(scale)[1]-(df+len(yt))/2*math.log1p(float(residual@np.linalg.solve(scale, residual))/df))
    assert values['F']-values['P'] == pytest.approx(expected, abs=1e-9)


def test_full_axis_integral_keeps_both_proper_beta_modes_and_original_endpoints():
    kernel = _prepare([[1.], [0.]], [[0.], [10.]], [0., math.sqrt(3000.)])
    result = integrate_single_record_precision(kernel, raw_scalar_rows=2, nu=4.,
        envelope=GammaEnvelope(2.5, 2., 1e-10), centre_hint=0., scale_hint=.5,
        relative_tolerances=(1e-8, 1e-10), scan_step=.1)
    assert result['log_integral'] == pytest.approx(-17.006063367176497, abs=1e-9)
    modes = [location for location, _ in result['mode_candidates']]
    assert any(location < -5 for location in modes) and any(-.2 < location < .1 for location in modes)
    assert result['complete_interval_endpoints_unchanged']
    assert result['convergence']['status'] == 'numerical_checks_passed'
    with pytest.raises(ValueError, match='raw scalar rows'):
        integrate_single_record_precision(kernel, raw_scalar_rows=3, nu=4.,
            envelope=GammaEnvelope(2.5, 2., 0.), centre_hint=0., scale_hint=.5)


def test_scope_arrays_are_frozen_and_unidentified_flat_support_is_rejected():
    kernel = _prepare([[1.], [1.], [1.]], [[-.7], [.2], [.8]], [1., -.5, 2.])
    with pytest.raises(ValueError):
        kernel.singular[0] = 0.
    with pytest.raises(DiffuseProjectionResolutionError, match='flat drift'):
        _prepare([[0.], [0.]], [[0.], [1.]], [0., 2.])
    with pytest.raises(ValueError, match='prior relation'):
        kernel.evaluate(0., prior_covariance_error=.5)
