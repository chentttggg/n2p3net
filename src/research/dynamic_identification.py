"""Small-matrix reference for noise-only Student-t uncertainty.

For one declared decision, lambda ~ Gamma(nu/2, nu/2), independently of
u ~ N(0, I), and y | lambda,u = mean + gaussian_factor @ u + noise.
The noise covariance is noise_factor @ noise_factor.T / lambda. The proper
Gaussian factor can represent a frozen ERP posterior or a declared state prior;
it is NOT divided by lambda. No parameters, state, labels or read permissions
are estimated here. The single scale is shared by the complete supplied vector,
not independently redrawn per channel, flash or prefix.

This numerical research reference integrates over log(lambda) with explicit
Gaussian/Gamma tail upper bounds and QUADPACK error estimates. It supports
full-rank noise and proper Gaussian factors only; no diffuse-prior substitution,
jitter, pseudoinverse, recursive t approximation or production default.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from transfer.noise_model import finite_real_array


class ScaleMixturePrecisionError(ValueError):
    """The requested density accuracy was not established; no alternate score."""


@dataclass(frozen=True)
class ScaleMixtureDensity:
    log_density: float
    estimated_quadrature_relative_error: float
    analytic_omitted_tail_relative_bound: float
    gamma_normalizer_log_error_bound: float
    integration_log_gamma_bounds: tuple[float, float]
    dimensions: int
    degrees_of_freedom: float

    @property
    def relative_error_estimate(self):
        return (self.estimated_quadrature_relative_error + self.analytic_omitted_tail_relative_bound
                + np.expm1(self.gamma_normalizer_log_error_bound))


@dataclass(frozen=True)
class ConditionalScaleMixtureDensity:
    """Ratio of integrated joint and prefix densities, not prior-weight averaging."""

    joint: ScaleMixtureDensity
    prefix: ScaleMixtureDensity

    @property
    def log_density(self):
        return self.joint.log_density - self.prefix.log_density

    @property
    def relative_error_estimate(self):
        return (self.joint.relative_error_estimate + self.prefix.relative_error_estimate) / (
            1 - self.prefix.relative_error_estimate)


def _inputs(observed, mean, noise_factor, gaussian_factor):
    y = finite_real_array(observed, name="actual observed vector")
    m = finite_real_array(mean, name="frozen candidate mean")
    noise = finite_real_array(noise_factor, name="full-rank lower noise factor")
    gaussian = finite_real_array(gaussian_factor, name="independent proper Gaussian factor")
    if (y.ndim != 1 or not len(y) or m.shape != y.shape
            or noise.shape != (len(y), len(y)) or gaussian.ndim != 2
            or gaussian.shape[0] != len(y)):
        raise ValueError("observations, mean and both factors require the same nonempty row grid")
    if np.any(np.triu(noise, 1) != 0) or np.any(np.diag(noise) <= 0):
        raise ValueError("noise requires a positive-diagonal full-rank lower factor")
    return y, m, noise, gaussian


def noise_scale_mixture_logpdf(observed, mean, noise_factor, gaussian_factor, *,
                              degrees_of_freedom, relative_tolerance=1e-8):
    """Integrate N(mean, Knoise/lambda + KGaussian) against the Gamma prior.

The returned tail bound is analytical; QUADPACK's interior error is an estimate,
not a proof. Convergence failure is explicit. Units come from the supplied
physical factors; for nu <= 2 this density is proper but its noise covariance
    does not exist; ``mean`` is a location, not a finite marginal expectation
    when nu <= 1. There is no interpretation of nu as a participant count.
"""
    y, m, noise, gaussian = _inputs(observed, mean, noise_factor, gaussian_factor)
    nu = degrees_of_freedom
    if isinstance(nu, bool) or not np.isfinite(nu) or nu <= 0:
        raise ValueError("degrees_of_freedom must be explicitly finite and positive")
    if (isinstance(relative_tolerance, bool) or not np.isfinite(relative_tolerance)
            or not 1e-12 <= relative_tolerance <= 1e-3):
        raise ValueError("the numerical reference supports relative tolerance in [1e-12, 1e-3]")
    a = float(nu) / 2
    n = len(y)
    residual = solve_triangular(noise, y - m, lower=True)
    factor = solve_triangular(noise, gaussian, lower=True)
    if not np.isfinite(residual).all() or not np.isfinite(factor).all():
        raise ScaleMixturePrecisionError("whitened inputs exceed finite coordinates")
    basis, singular, _ = np.linalg.svd(factor, full_matrices=False)
    projected = basis.T @ residual
    orthogonal = residual - basis @ projected
    with np.errstate(over="ignore", invalid="ignore"):
        energies = np.r_[projected ** 2, orthogonal @ orthogonal]
        energy_sum = float(energies.sum())
    if not np.isfinite(energy_sum) or not np.isfinite(singular).all():
        raise ScaleMixturePrecisionError("reference energy exceeds finite coordinates")
    with np.errstate(divide="ignore"):
        log_energies = np.log(energies)
        log_variances = 2 * np.log(singular)
    log_a = log(a)
    noise_logdet = float(2 * np.log(np.diag(noise)).sum())
    # t = log(a * lambda), so the Gamma rate and shape are both a.
    constant = -0.5 * (n * log(2 * np.pi) + noise_logdet + n * log_a) - gammaln(a)
    k = a + n / 2
    # Evaluate the Gamma peak without subtracting O(a*log(a)) terms.
    # At a >= 100, the retained Stirling remainder is < 1/(1680*a**7)
    # <= 5.96e-18. Using cancelling gammaln terms up to a=1e4 violated a
    # supported 1e-12 density request even when quadrature had converged.
    if a >= 100:
        inverse_a = 1 / a
        correction = inverse_a / 12 - inverse_a ** 3 / 360 + inverse_a ** 5 / 1260
        gamma_at_unit = 0.5 * (log_a - log(2 * np.pi)) - correction
        normalizer_error = inverse_a ** 7 / 1680
    else:
        gamma_at_unit = a * log_a - a - gammaln(a)
        normalizer_error = 0.

    def log_integrand(t):
        if t > 700:
            return -np.inf
        log_lambda = t - log_a
        log_denominators = np.logaddexp(0., log_lambda + log_variances)
        terms = log_lambda + log_energies
        terms[:-1] -= log_denominators
        log_quadratic = float(logsumexp(terms))
        if log_quadratic > 700:
            return -np.inf
        quadratic = exp(log_quadratic) if np.isfinite(log_quadratic) else 0.
        gamma_change = (a * (log_lambda - np.expm1(log_lambda)) if abs(log_lambda) < 0.5
                        else a * log_lambda - exp(t) + a)
        gaussian_log = -0.5 * (n * log(2 * np.pi) + noise_logdet - n * log_lambda
                               + log_denominators.sum() + quadratic)
        return float(gamma_at_unit + gamma_change + gaussian_log)

    def tail_logs(lower, upper):
        # det(Knoise/lambda + GG') >= det(Knoise)/lambda**n;
        # dropping the Gaussian quadratic bounds the integrand by
        # exp(constant + k*t - exp(t)). These bounds cover all omitted modes.
        lower_log = constant + k * lower - log(k)
        slope = 1 - max(k - 1, 0) * exp(-upper)
        if slope <= 0:
            return lower_log, np.inf
        upper_log = constant + (k - 1) * upper - exp(upper) - log(slope)
        return lower_log, upper_log

    pure_mode = log(k) - np.logaddexp(0., log(energy_sum) - log(2 * a)) if energy_sum else log(k)
    lower, upper = min(-10., pure_mode - 10), max(4., log(k) + 4)
    # A mode is used solely for floating rescaling. It is not a unimodality
    # assumption: all interval mass is integrated, with analytic exterior bounds.
    for _ in range(8):
        optimum = minimize_scalar(lambda t: -log_integrand(t), bounds=(lower, upper),
                                  method="bounded", options={"xatol": 1e-10})
        centers = {float(optimum.x), float(pure_mode), float(log(k)), float(log_a), 0.}
        # QUADPACK can miss one half of a narrow mode on a long interval and
        # still report a tiny error. Explicit local knots resolve the Gamma
        # curvature around every candidate peak, including the Gaussian limit.
        width = 1 / np.sqrt(k)
        points = sorted({center + multiple * width for center in centers
                         for multiple in (-12., -4., -1., 0., 1., 4., 12.)})
        points = [point for point in points if lower < point < upper]
        height = max(log_integrand(point) for point in points)
        if not optimum.success or not np.isfinite(height):
            raise ScaleMixturePrecisionError("could not establish finite integration rescaling")

        def integrand(t, log_height=height):
            value = log_integrand(t) - log_height
            if value > 700:
                raise ScaleMixturePrecisionError("integration missed representable density rescaling")
            return exp(value)

        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            try:
                integral, error = quad(integrand, lower, upper, points=points,
                                       epsabs=1e-12, epsrel=relative_tolerance / 8, limit=400)
            except IntegrationWarning as exception:
                raise ScaleMixturePrecisionError("interior integration did not converge") from exception
        if not np.isfinite([integral, error]).all() or integral <= 0:
            raise ScaleMixturePrecisionError("integration produced no finite positive density")
        log_density = height + log(integral)
        tail_values = tail_logs(lower, upper)
        log_tail_relative = float(logsumexp(tail_values)) - log_density
        tail_relative = exp(log_tail_relative) if log_tail_relative < 700 else np.inf
        interior_relative = error / integral
        if interior_relative + tail_relative + np.expm1(normalizer_error) <= relative_tolerance:
            return ScaleMixtureDensity(float(log_density), interior_relative, tail_relative, normalizer_error,
                                       (float(lower), float(upper)), n, float(nu))
        if interior_relative > relative_tolerance / 2:
            raise ScaleMixturePrecisionError("estimated interior error exceeds requested density accuracy")
        target_log = log_density + log(relative_tolerance / 8)
        lower = min(lower, (target_log - constant + log(k)) / k - 2)
        while tail_logs(lower, upper)[1] > target_log:
            upper += 1.
    raise ScaleMixturePrecisionError("density/tail error bounds did not settle after refinement")


def noise_scale_mixture_conditional_logpdf(observed, mean, noise_factor, gaussian_factor, *,
                                          conditioning_rows, degrees_of_freedom,
                                          relative_tolerance=1e-8):
    """Score the actual tail given its contiguous prefix, integrating both.

Lower-triangular noise-factor leading rows give the true prefix covariance.
The caller owns physical row order and read authorization. The tail must be
available at scoring; this is not permission to smooth using future samples.
    The prior Gamma law is UPDATED by prefix evidence separately for each candidate.
    Both public density entries support requested tolerance in [1e-12, 1e-3].
    Near that floor, the final ratio must meet the ACTUAL combined error budget;
    otherwise it fails explicitly rather than silently exceeding the request.
"""
    y, m, noise, gaussian = _inputs(observed, mean, noise_factor, gaussian_factor)
    cut = conditioning_rows
    if type(cut) is not int or not 0 < cut < len(y):
        raise ValueError("a genuine nonempty prefix and nonempty scored tail are required")
    if (isinstance(relative_tolerance, bool) or not np.isfinite(relative_tolerance)
            or not 1e-12 <= relative_tolerance <= 1e-3):
        raise ValueError("the numerical reference supports relative tolerance in [1e-12, 1e-3]")
    kwargs = {"degrees_of_freedom": degrees_of_freedom,
              "relative_tolerance": max(relative_tolerance / 4, 1e-12)}
    joint = noise_scale_mixture_logpdf(y, m, noise, gaussian, **kwargs)
    prefix = noise_scale_mixture_logpdf(y[:cut], m[:cut], noise[:cut, :cut], gaussian[:cut], **kwargs)
    result = ConditionalScaleMixtureDensity(joint, prefix)
    if result.relative_error_estimate > relative_tolerance:
        raise ScaleMixturePrecisionError("joint/prefix ratio exceeds requested combined density accuracy")
    return result
