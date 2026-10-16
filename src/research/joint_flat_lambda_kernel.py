"""One record's proper-beta/physical-flat Gaussian marginal for all lambda.

The caller supplies complete, whitened reduced operands for ONE fixed target
configuration. This module grants no record/target authority and changes no
captured sampler or replay program. The raw Gaussian row count and physical
measure constants remain explicit when constructing the Gamma integral.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_triangular, svd
from scipy.optimize import minimize_scalar

from research.channel_noise_mechanism import (
    DiffuseProjectionResolutionError,
    _column_norms,
    _inverse_root_geometry,
    _roundoff_gamma,
)
from research.joint_response_scale import _augmented_qr
from transfer.noise_model import finite_real_array, freeze_metadata, frozen_array


def _frob(a):
    # Avoid squaring extremely small/large entries before their norm is known.
    return math.hypot(*map(float, np.ravel(a)))


def _norm_upper(a):
    value = _frob(a)
    return math.nextafter(math.nextafter(value, math.inf), math.inf) if value else 0.


def _subtraction_norm_upper(stored_difference):
    eps, tiny = np.finfo(float).eps, np.nextafter(0., 1.)
    value = (_norm_upper(stored_difference)+math.sqrt(stored_difference.size)*tiny)/(1-eps)
    return math.nextafter(value, math.inf)


def _matmul_error_norm(left, right):
    """A positive-dot enclosure, including gradual-underflow dot errors."""
    n = left.shape[-1]
    gamma = math.nextafter(_roundoff_gamma(n), math.inf)
    positive = np.abs(left)@np.abs(right)
    absolute = n*np.nextafter(0., 1.)*math.sqrt(positive.size)
    value = (gamma*_norm_upper(positive)+absolute)/(1-gamma)
    return math.nextafter(value, math.inf)


def _exp_extended(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _log_positive(value):
    return math.log(value) if value else -math.inf


def _log_sum(values):
    values = tuple(map(float, values))
    maximum = max(values, default=-math.inf)
    if maximum in (-math.inf, math.inf):
        return maximum
    return maximum+math.log(math.fsum(math.exp(value-maximum) for value in values))


def _positive_log_upper(value, scale):
    if value == -math.inf:
        return 0.
    result = _exp_extended(value)
    if not math.isfinite(result):
        return math.inf
    return math.nextafter(result*(1+_roundoff_gamma(64)*(1+abs(value)+scale)), math.inf)


def _orthogonal_polar_defect(q):
    """A posteriori distance from stored Q to a genuine orthogonal matrix."""
    eta = _subtraction_norm_upper(q.T@q-np.eye(q.shape[1]))+_matmul_error_norm(q.T, q)
    eta = math.nextafter(eta, math.inf)
    if not eta < 1:
        raise DiffuseProjectionResolutionError('spectral orthogonality unresolved')
    return math.nextafter(eta/(1+math.sqrt(1-eta)), math.inf), eta


@dataclass(frozen=True)
class SingleRecordLambdaKernel:
    singular: np.ndarray
    coordinates: np.ndarray
    residual_log_squared: float
    log_flat_root: float
    flat_dimension: int
    proper_dimension: int
    coupling_norms: np.ndarray
    inverse_flat_norm: float
    drift_zero_norm: float
    source_flat_norm: float
    source_beta_norm: float
    source_rhs_norm: float
    preprocessing: dict

    @property
    def residual_squared(self):
        """Display view only; the log residual is the unique numerical state."""
        return _exp_extended(self.residual_log_squared)

    def __post_init__(self):
        singular = finite_real_array(self.singular, name='lambda spectral singular values')
        coordinates = finite_real_array(self.coordinates, name='lambda spectral response coordinates')
        coupling = finite_real_array(self.coupling_norms, name='lambda flat-response coupling bounds')
        if (singular.ndim != 1 or coordinates.shape != singular.shape or np.any(singular < 0)
                or type(self.proper_dimension) is not int or self.proper_dimension < len(singular)
                or type(self.flat_dimension) is not int or self.flat_dimension < 1
                or coupling.shape != (self.proper_dimension,) or np.any(coupling < 0)):
            raise ValueError('single-record spectral and proper/flat coordinates differ')
        scalars = (self.log_flat_root, self.inverse_flat_norm, self.drift_zero_norm,
                   self.source_flat_norm, self.source_beta_norm, self.source_rhs_norm)
        if (not all(math.isfinite(value) for value in scalars)
                or min(scalars[1:]) < 0 or self.inverse_flat_norm <= 0
                or not (math.isfinite(self.residual_log_squared) or self.residual_log_squared == -math.inf)):
            raise ValueError('single-record spectral/source bounds must be finite and nonnegative')
        for name, value in (('singular', singular), ('coordinates', coordinates), ('coupling_norms', coupling)):
            object.__setattr__(self, name, frozen_array(value))
        object.__setattr__(self, 'preprocessing', freeze_metadata(self.preprocessing))

    def evaluate(self, x, *, prior_covariance_error=0.):
        """Return an admitted value/error pair or a typed arithmetic failure."""
        try:
            return self._evaluate(x, prior_covariance_error=prior_covariance_error)
        except (OverflowError, FloatingPointError) as error:
            raise DiffuseProjectionResolutionError('continuous precision arithmetic is unrepresentable') from error

    def _evaluate(self, x, *, prior_covariance_error=0.):
        """m(exp(x)), source/arithmetic nats estimate, and scalar diagnostics.

        All products that can recover from intermediate underflow are combined
        before exponentiation. Unresolved source geometry raises, so a caller
        cannot consume a finite value as an admitted density at that point.
        """
        if (isinstance(x, bool) or not math.isfinite(x) or isinstance(prior_covariance_error, bool)
                or not math.isfinite(prior_covariance_error) or not 0 <= prior_covariance_error < .5):
            raise ValueError('finite log precision and original prior relation in [0,.5) are required')
        singular, z = self.singular, self.coordinates
        with np.errstate(divide='ignore'):
            log_singular = np.log(singular)
        log_den = np.logaddexp(0., x+2*log_singular)
        with np.errstate(divide='ignore'):
            log_z = np.log(np.abs(z))
        with np.errstate(over='ignore', under='ignore'):
            energy_terms = np.exp(x-log_den+2*log_z)
        residual_energy = _exp_extended(x+self.residual_log_squared) if self.residual_log_squared != -math.inf else 0.
        energy = math.fsum([residual_energy, *map(float, energy_terms)])
        logdet = self.log_flat_root+self.flat_dimension*x/2+.5*math.fsum(map(float, log_den))
        value = -.5*energy-logdet
        beta_log_coordinates = x-log_den+log_singular+log_z
        beta_log_norm = .5*_log_sum(2*beta_log_coordinates)
        beta_norm = _exp_extended(beta_log_norm)
        if not all(math.isfinite(v) for v in (energy, logdet, value, beta_norm)):
            raise DiffuseProjectionResolutionError('continuous precision energy or proper mean is unrepresentable')
        full_log_den = np.zeros(self.proper_dimension)
        full_log_den[:len(singular)] = log_den
        log_inverse_beta = -.5*float(full_log_den.min(initial=math.inf))
        with np.errstate(divide='ignore'):
            log_coupling = np.log(self.coupling_norms)
        log_coupled_inverse = .5*_log_sum(2*log_coupling-full_log_den)
        source_h, source_b, source_y = map(_log_positive, (
            self.source_flat_norm, self.source_beta_norm, self.source_rhs_norm))
        arithmetic_scale = 1+abs(x)+float(np.max(np.abs(log_singular[np.isfinite(log_singular)]), initial=0.))
        metric_log = _log_sum((source_h+_log_positive(self.inverse_flat_norm),
            x/2+source_b+log_inverse_beta, x/2+source_h+log_coupled_inverse))
        metric = _positive_log_upper(metric_log, arithmetic_scale)
        drift_log_norm = _log_sum((_log_positive(self.drift_zero_norm),
            *map(float, log_coupling[:len(singular)]+beta_log_coordinates)))
        drift_norm = _exp_extended(drift_log_norm)
        input_log = x/2+_log_sum((source_y, source_b+beta_log_norm, source_h+drift_log_norm))
        displacement_input = _positive_log_upper(input_log, arithmetic_scale)
        if not metric < 1:
            raise DiffuseProjectionResolutionError('continuous precision source metric is unresolved')
        displacement = (metric*math.sqrt(energy)+(1+metric)*displacement_input)/(1-metric)**2
        residual_error = displacement_input+metric*displacement
        quadratic_error = 2*math.sqrt(energy)*residual_error+residual_error**2
        determinant_error = -(self.proper_dimension+self.flat_dimension)*math.log1p(-metric)
        log_scale = 1+abs(x)+float(np.max(np.abs(log_singular[np.isfinite(log_singular)]), initial=0.))
        log_scale += float(np.max(np.abs(log_z[np.isfinite(log_z)]), initial=0.))
        scalar_error = _roundoff_gamma(64*(self.proper_dimension+self.flat_dimension+1))*(
            (1+energy)*log_scale+abs(self.log_flat_root)+self.flat_dimension*abs(x)
            +math.fsum(map(float, np.abs(log_den))))
        error = .5*quadratic_error+determinant_error+scalar_error
        prior_error = 0.
        if prior_covariance_error:
            delta = prior_covariance_error
            t = delta/(1-delta)
            if not t < 1:
                raise DiffuseProjectionResolutionError('original proper-prior relation unresolved')
            prior_error = .5*self.proper_dimension*(-math.log1p(-delta)-math.log1p(-t))
            beta_displacement = _positive_log_upper(log_inverse_beta+_log_positive(displacement), arithmetic_scale)
            prior_error += .5*t*(beta_norm+beta_displacement)**2/(1-t)
            error += prior_error
        if not math.isfinite(error):
            raise DiffuseProjectionResolutionError('continuous precision source error is unrepresentable')
        return value, error, {'source_metric': metric, 'energy': energy, 'logdet': logdet,
            'beta_norm': beta_norm, 'drift_norm_bound': drift_norm,
            'source_residual_error': residual_error, 'quadratic_error_bound': quadratic_error,
            'determinant_error_bound': determinant_error, 'scalar_error_estimate': scalar_error,
            'proper_prior_relation_nats_bound': prior_error, 'point_source_resolved': True}

    def log_integrand(self, x, *, raw_scalar_rows, nu, prior_covariance_error=0.):
        """Gamma-prior log integrand per d log(lambda), before raw constants."""
        _gamma_shape(raw_scalar_rows, self.flat_dimension, nu)
        value, error, diagnostics = self.evaluate(x, prior_covariance_error=prior_covariance_error)
        gamma_penalty = _exp_extended(x+math.log(nu/2))
        leading = (raw_scalar_rows+nu)*x/2
        result = leading-gamma_penalty+value
        if not math.isfinite(result):
            raise DiffuseProjectionResolutionError('continuous precision log integrand is unrepresentable')
        error += _roundoff_gamma(8)*(1+abs(leading)+gamma_penalty+abs(value))
        if not math.isfinite(error):
            raise DiffuseProjectionResolutionError('continuous precision log-integrand error is unrepresentable')
        return result, error, diagnostics


def prepare_single_record_lambda_kernel(root, errors, startup_rows, startup_errors, *, response_width, channels, drift_width):
    """Compile one fixed-q record; retain all singular values and all drift.

    Tail columns are [r drift, k unit-proper beta, c responses]; startup has
    r*c physical flat columns plus one response and structural beta design 0.
    Source errors are original column-norm bounds, not replacement data.
    """
    if any(type(value) is not int or value < 1 for value in (response_width, channels, drift_width)):
        raise ValueError('positive integer response, channel and physical drift widths are required')
    root = finite_real_array(root, name='complete single-record compressed root')
    errors = finite_real_array(errors, name='original single-record column errors')
    startup_rows = finite_real_array(startup_rows, name='complete original startup rows')
    startup_errors = finite_real_array(startup_errors, name='original startup source errors')
    k, c, r = response_width, channels, drift_width
    p, d = k*c, r*c
    if (root.ndim != 2 or root.shape[1] != r+k+c or errors.shape != (r+k+c,)
            or np.any(errors < 0) or startup_rows.ndim != 2 or startup_rows.shape[1] != d+1
            or startup_errors.shape != startup_rows.shape or np.any(startup_errors < 0)):
        raise ValueError('single-record tail/startup operands and source errors have different support')
    h = np.kron(root[:, :r], np.eye(c))
    b = np.kron(root[:, r:r+k], np.eye(c))
    y = root[:, r+k:].ravel()
    tail = np.column_stack((h, b, y))
    startup = np.zeros((len(startup_rows), d+p+1))
    startup[:, :d], startup[:, -1] = startup_rows[:, :-1], startup_rows[:, -1]
    augmented = np.vstack((tail, startup))
    inherited = np.r_[np.repeat(errors[:r], c), np.repeat(errors[r:r+k], c),
                      np.linalg.norm(errors[r+k:])]
    start_errors = _column_norms(startup_errors)
    inherited[:d] += start_errors[:-1]
    inherited[-1] += start_errors[-1]
    qr, qr_errors = _augmented_qr(augmented, len(augmented), with_error=True)
    inherited += qr_errors
    rh, cross, rhs_head = qr[:d, :d], qr[:d, d:d+p], qr[:d, -1]
    reduced_b, reduced_y = qr[d:, d:d+p], qr[d:, -1]
    if np.any(qr[d:, :d] != 0):
        raise DiffuseProjectionResolutionError('flat elimination failed to retain triangular rows')
    if len(rh) != d or np.any(np.diag(rh) == 0):
        raise DiffuseProjectionResolutionError('complete original record does not identify its flat drift')
    inverse_abs, inverse_norm = _inverse_root_geometry(rh)
    u, singular, vt = svd(reduced_b, full_matrices=True, lapack_driver='gesvd')
    rank_width = len(singular)
    polar_u, eta_u = _orthogonal_polar_defect(u)
    polar_v, eta_v = _orthogonal_polar_defect(vt.T)
    us = u[:, :rank_width]*singular
    reconstructed_b = us@vt[:rank_width]
    reconstruction_b = _subtraction_norm_upper(reduced_b-reconstructed_b)+_matmul_error_norm(us, vt[:rank_width])
    reconstruction_b += _roundoff_gamma(2)*_norm_upper(us)*math.sqrt(1+eta_v)
    spectral_b_error = reconstruction_b+_norm_upper(singular)*(polar_u*math.sqrt(1+eta_v)+polar_v)
    z = u.T@reduced_y
    reconstructed_y = u@z
    reconstruction_y = _subtraction_norm_upper(reduced_y-reconstructed_y)+_matmul_error_norm(u, z)
    # z is the stored coordinate witness: its actual dot error is already
    # included in the measured reconstruction, rather than charged twice.
    spectral_y_error = reconstruction_y+polar_u*_norm_upper(z)
    transformed_cross = cross@vt.T
    coupling = solve_triangular(rh, transformed_cross, lower=False)
    coupling_errors = _column_norms(rh@coupling-transformed_cross)
    coupling_errors += _matmul_error_norm(rh, coupling)+_matmul_error_norm(cross, vt.T)
    coupling_norms = _column_norms(coupling)+inverse_norm*coupling_errors
    coupling_norms += polar_v*inverse_norm*_norm_upper(cross)
    drift_zero = solve_triangular(rh, rhs_head, lower=False)
    drift_zero_error = _subtraction_norm_upper(rh@drift_zero-rhs_head)+_matmul_error_norm(rh, drift_zero)
    drift_zero_norm = _norm_upper(drift_zero)+inverse_norm*drift_zero_error
    source_h = _norm_upper(inherited[:d])
    source_b = _norm_upper(inherited[d:d+p])+spectral_b_error
    source_y = float(inherited[-1])+spectral_y_error
    # All trailing U coordinates are retained directly; no difference of
    # large energies determines the orthogonal residual or its lower bound.
    residual_coordinates = z[rank_width:]
    residual_norm = _frob(residual_coordinates)
    q_residual = residual_norm*residual_norm
    if np.any(residual_coordinates != 0):
        maximum = float(np.max(np.abs(residual_coordinates)))
        scaled = residual_coordinates/maximum
        log_q_residual = 2*math.log(maximum)+math.log(math.fsum(float(v)*float(v) for v in scaled))
    else:
        log_q_residual = -math.inf
    log_flat = math.fsum(map(float, np.log(np.abs(np.diag(rh)))))
    audit = {'input_rows': len(augmented), 'flat_dimension': d, 'proper_dimension': p,
        'reduced_shape': list(reduced_b.shape), 'singular_min': float(singular[-1]) if len(singular) else 0.,
        'singular_max': float(singular[0]) if len(singular) else 0., 'singular_all_retained': True,
        'residual_squared': q_residual, 'observed_projected_energy': _frob(reduced_y)**2,
        'log_flat_root': log_flat, 'source_H_F': source_h, 'source_B_F': source_b, 'source_y': source_y,
        'source_root_errors_H_F': _frob(inherited[:d]),
        'source_root_errors_B_F': _frob(inherited[d:d+p]), 'source_root_errors_y': float(inherited[-1]),
        'svd_B_reconstruction_bound': reconstruction_b, 'svd_B_total_backward_bound': spectral_b_error,
        'svd_y_total_backward_bound': spectral_y_error, 'polar_U_bound': polar_u, 'polar_V_bound': polar_v,
        'inverse_flat_norm': inverse_norm, 'coupling_norm_max': float(coupling_norms.max(initial=0.)),
        'drift_zero_norm': drift_zero_norm}
    return SingleRecordLambdaKernel(singular, z[:rank_width], log_q_residual, log_flat, d, p, coupling_norms,
        inverse_norm, drift_zero_norm, source_h, source_b, source_y, audit)


def _gamma_shape(raw_scalar_rows, flat_dimension, nu):
    if (type(raw_scalar_rows) is not int or not flat_dimension <= raw_scalar_rows < 2**53
            or isinstance(nu, bool) or not math.isfinite(nu) or nu <= 0):
        raise ValueError('original scalar rows and finite positive Gamma degrees of freedom are required')
    if nu/2 == 0:
        raise DiffuseProjectionResolutionError('Gamma prior shape is not representable')
    return (raw_scalar_rows-flat_dimension)/2+nu/2


def marginal_log_measure_constant(*, raw_scalar_rows, flat_dimension, nu,
                                  physical_flat_logabsdet, noise_logdet):
    """Original da/raw-Gaussian/Gamma constants; unit proper beta occurs once."""
    if type(flat_dimension) is not int or flat_dimension < 1:
        raise ValueError('the physical flat dimension must be a positive integer')
    _gamma_shape(raw_scalar_rows, flat_dimension, nu)
    if not all(math.isfinite(value) for value in (physical_flat_logabsdet, noise_logdet)):
        raise ValueError('original physical flat and raw noise measure constants must be finite')
    alpha = nu/2
    try:
        value = (physical_flat_logabsdet-(raw_scalar_rows-flat_dimension)/2*math.log(2*math.pi)
                 -.5*noise_logdet+alpha*math.log(alpha)-math.lgamma(alpha))
    except OverflowError as error:
        raise DiffuseProjectionResolutionError('original Gamma measure constant is unrepresentable') from error
    if not math.isfinite(value):
        raise DiffuseProjectionResolutionError('original measure constant is unrepresentable')
    return value


def _merge_nearby_breakpoints(values):
    """Merge ULP-close interior cuts; retain the complete original interval."""
    cuts = sorted(values)
    result = [cuts[0]]
    for value in cuts[1:-1]:
        scale = max(1., abs(result[-1]), abs(value))
        if value-result[-1] > 128*math.ulp(scale):
            result.append(value)
    if len(result) > 1:
        scale = max(1., abs(result[-1]), abs(cuts[-1]))
        if cuts[-1]-result[-1] <= 128*math.ulp(scale):
            result.pop()
    result.append(cuts[-1])
    return result



@dataclass(frozen=True)
class GammaEnvelope:
    shape: float
    rate: float
    log_prefactor: float

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in (self.shape, self.rate, self.log_prefactor))
                or self.shape <= 0 or self.rate <= 0):
            raise ValueError('analytical Gamma envelope must be finite with positive shape/rate')

    def log_lower_mass(self, limit):
        # exp(-rate*lambda)<=1, so this remains valid for every shape>0.
        return self.log_prefactor+self.shape*math.log(limit)-math.log(self.shape)

    def log_upper_mass(self, limit):
        slope = self.rate-(self.shape-1)/limit if self.shape >= 1 else self.rate
        if slope <= 0:
            return math.inf
        return (self.log_prefactor+(self.shape-1)*math.log(limit)-self.rate*limit-math.log(slope))



def _integrate_log_kernel(log_kernel, *, centre_hint, scale_hint, envelope,
                         relative_tolerances=(1e-4, 1e-6, 1e-8), scan_step=.25):
    """Mode-centred finite integration, with separately reported tail bounds.

    log_kernel(x) returns (log integrand per dx, source nats error estimate).
    The Gamma envelope bounds the reduced integral per d lambda. Its original
    flat determinant and residual lower bound must be supplied by the caller.
    """
    if (not math.isfinite(centre_hint) or not math.isfinite(scale_hint) or scale_hint <= 0
            or envelope.shape <= 0 or envelope.rate <= 0 or scan_step <= 0
            or not relative_tolerances or min(relative_tolerances) <= 0):
        raise ValueError('finite mode/scale, positive Gamma envelope and explicit precision sequence are required')
    points = {}
    def point(x):
        x = float(x)
        if x not in points:
            value, error = map(float, log_kernel(x))
            if not math.isfinite(value) or not math.isfinite(error) or error < 0:
                raise ArithmeticError('log-lambda kernel or its source error is unresolved')
            points[x] = value, error
        return points[x]
    initial = point(centre_hint)[0]+math.log(scale_hint)
    tail_target = initial+math.log(min(relative_tolerances)*1e-4)
    lower = math.exp(centre_hint-8*scale_hint)
    upper = math.exp(centre_hint+8*scale_hint)
    while envelope.log_lower_mass(lower) > tail_target:
        lower /= 2
        if lower <= 0:
            raise ArithmeticError('Gamma lower boundary is not representable')
    while envelope.log_upper_mass(upper) > tail_target:
        upper *= 2
        if not math.isfinite(upper):
            raise ArithmeticError('Gamma upper boundary is not representable')
    left, right = math.log(lower), math.log(upper)
    count = max(1, math.ceil((right-left)/scan_step))
    grid = [left+(right-left)*i/count for i in range(count+1)]
    values = [point(x)[0] for x in grid]
    modes = []
    for i in range(1, len(grid)-1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            optimum = minimize_scalar(lambda x: -point(x)[0], bounds=(grid[i-1], grid[i+1]),
                method='bounded', options={'xatol': max(1e-13, scale_hint*1e-8)})
            modes.append((float(optimum.x), point(optimum.x)[0]))
    modes.extend(((centre_hint, point(centre_hint)[0]), (grid[values.index(max(values))], max(values))))
    mode, height = max(modes, key=lambda item: item[1])
    knots = set(grid+[x for x, _ in modes])
    for location, _ in modes:
        knots.update(location+scale_hint*z for z in (-16, -8, -4, -2, 0, 2, 4, 8, 16)
                     if left < location+scale_hint*z < right)
    original_cuts = sorted((x-mode)/scale_hint for x in knots)
    cuts = _merge_nearby_breakpoints(original_cuts)
    evaluations = []
    for tolerance in relative_tolerances:
        total = estimated_error = lower_estimate = upper_estimate = 0.
        lower_quad_error = upper_quad_error = 0.
        messages = []
        for a, b in zip(cuts, cuts[1:], strict=False):
            def integrand(z, source_sign=0):
                value, error = point(mode+scale_hint*z)
                shifted = value-height+source_sign*error
                if shifted > 700:
                    raise ArithmeticError('mode scan missed a numerically significant peak')
                return math.exp(shifted)*scale_hint
            absolute = tolerance*scale_hint/max(1, len(cuts)-1)
            answer = quad(integrand, a, b, epsabs=absolute, epsrel=tolerance, limit=200, full_output=True)
            total += answer[0]
            estimated_error += answer[1]
            if len(answer) > 3:
                messages.append({'scope': 'central', 'segment': [a, b], 'message': answer[3]})
            lower_answer = quad(lambda z: integrand(z, -1), a, b,
                epsabs=absolute, epsrel=tolerance, limit=200, full_output=True)
            upper_answer = quad(lambda z: integrand(z, 1), a, b,
                epsabs=absolute, epsrel=tolerance, limit=200, full_output=True)
            lower_estimate += lower_answer[0]
            upper_estimate += upper_answer[0]
            lower_quad_error += lower_answer[1]
            upper_quad_error += upper_answer[1]
            for scope, source_answer in (('source_lower', lower_answer), ('source_upper', upper_answer)):
                if len(source_answer) > 3:
                    messages.append({'scope': scope, 'segment': [a, b], 'message': source_answer[3]})
        if not total > 0 or not math.isfinite(total):
            raise ArithmeticError('mode-centred integral is not finite and positive')
        log_integral = height+math.log(total)
        evaluations.append({'requested_relative_tolerance': tolerance, 'log_integral': log_integral,
            'quad_reported_relative_error_estimate': estimated_error/total,
            'source_lower_quad_reported_relative_error_estimate': lower_quad_error/lower_estimate,
            'source_upper_quad_reported_relative_error_estimate': upper_quad_error/upper_estimate,
            'integrated_source_nats_lower_estimate': height+math.log(lower_estimate),
            'integrated_source_nats_upper_estimate': height+math.log(upper_estimate),
            'lower_analytical_tail_log_bound_relative_to_numerical_integral': envelope.log_lower_mass(lower)-log_integral,
            'upper_analytical_tail_log_bound_relative_to_numerical_integral': envelope.log_upper_mass(upper)-log_integral,
            'quad_messages': messages})
    final = evaluations[-1]
    changes = [abs(left['log_integral']-right['log_integral']) for left, right in zip(evaluations, evaluations[1:], strict=False)]
    compatible = all(change <= left['requested_relative_tolerance']+right['requested_relative_tolerance']
                     for change, left, right in zip(changes, evaluations, evaluations[1:], strict=False))
    tolerances_met = all(all(row[key] <= row['requested_relative_tolerance'] for key in (
        'quad_reported_relative_error_estimate', 'source_lower_quad_reported_relative_error_estimate',
        'source_upper_quad_reported_relative_error_estimate')) for row in evaluations)
    tails_met = all(final[key] <= math.log(relative_tolerances[-1]*1e-4) for key in (
        'lower_analytical_tail_log_bound_relative_to_numerical_integral',
        'upper_analytical_tail_log_bound_relative_to_numerical_integral'))
    messages = [message for row in evaluations for message in row['quad_messages']]
    convergence = {'status': ('numerical_checks_passed' if compatible and tolerances_met and tails_met and not messages
        else 'requires_review'), 'all_quad_messages': messages, 'reported_tolerances_met': tolerances_met,
        'tolerance_sequence_log_integral_changes_nats': changes, 'tolerance_sequence_compatible': compatible,
        'analytical_tail_budget_met': tails_met,
        'integrated_source_error_span_nats': final['integrated_source_nats_upper_estimate']-final['integrated_source_nats_lower_estimate'],
        'interpretation': 'numerical convergence checks, not automatic scientific acceptance or an interval proof'}
    return {'log_integral': final['log_integral'], 'mode_x': mode, 'mode_lambda': math.exp(mode),
        'mode_log_kernel': height, 'scale_x_hint': scale_hint, 'lambda_bounds': [lower, upper],
        'analytical_gamma_envelope': {'shape': envelope.shape, 'rate': envelope.rate,
            'log_prefactor': envelope.log_prefactor}, 'evaluations': evaluations,
        'mode_scan_points': len(grid), 'mode_candidates': modes, 'distinct_kernel_evaluations': len(points),
        'merged_nearby_breakpoints': len(original_cuts)-len(cuts),
        'complete_interval_endpoints_unchanged': [cuts[0], cuts[-1]] == [original_cuts[0], original_cuts[-1]],
        'convergence': convergence,
        'max_evaluated_source_nats_error_estimate': max(error for _, error in points.values()),
        'interpretation': 'adaptive numerical estimates with analytical conditional tail envelope; not an interval proof'}


def integrate_single_record_precision(kernel, *, raw_scalar_rows, nu, envelope,
                                     centre_hint, scale_hint, prior_covariance_error=0.,
                                     relative_tolerances=(1e-4, 1e-6, 1e-8), scan_step=.25):
    """Integrate the fixed-q single-record kernel over the whole positive axis.

    The caller's Gamma envelope must bound the ORIGINAL source law, using its
    own residual lower bound and physical-flat determinant lower bound. Its
    shape is tied here to raw N and all d flat coordinates. Returned log I
    excludes marginal_log_measure_constant; the two are added once per phase.
    """
    if not isinstance(kernel, SingleRecordLambdaKernel):
        raise TypeError('a prepared single-record fixed-q kernel is required')
    shape = _gamma_shape(raw_scalar_rows, kernel.flat_dimension, nu)
    if not isinstance(envelope, GammaEnvelope) or envelope.shape != shape:
        raise ValueError('the analytical envelope shape must use original raw scalar rows and flat dimension')

    def point(x):
        value, error, _ = kernel.log_integrand(x, raw_scalar_rows=raw_scalar_rows, nu=nu,
                                             prior_covariance_error=prior_covariance_error)
        return value, error

    result = _integrate_log_kernel(point, centre_hint=centre_hint, scale_hint=scale_hint,
        envelope=envelope, relative_tolerances=relative_tolerances, scan_step=scan_step)
    return {**result, 'scope': 'single_record_fixed_target_continuous_precision_integral',
        'raw_scalar_rows': raw_scalar_rows, 'proper_dimension': kernel.proper_dimension,
        'physical_flat_dimension': kernel.flat_dimension, 'degrees_of_freedom': nu,
        'measure_constant_included': False}
