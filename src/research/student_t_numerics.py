"""Shared binary64 evaluation of a declared finite-df multivariate t density.

No finite nu is changed to Gaussian infinity. Integer observation dimensions
permit exact Gamma recurrences, avoiding subtraction of large log-Gamma values.
The only asymptotic evaluation is a half-step Gamma ratio; its analytic remainder
is bounded below. That bound excludes ordinary floating-point rounding and is
not a certificate for a likelihood integral or a sampled posterior.
"""

from functools import lru_cache
from math import fsum, isfinite, log, log1p

import numpy as np
from scipy.special import gammaln


class StudentTNumericalError(ValueError):
    """The declared t density cannot be represented by this numerical program."""


def _half_step(a):
    """log Gamma(a+1/2)-log Gamma(a)-log(a)/2, without large cancellation.

For a >= 16, the six alternating terms have an absolute remainder bounded by
5461/(425984*a**13), at most 2.847e-18. This follows by integrating the alternating
partial-fraction expansion of tanh(t/4) in
    -1/2 integral_0^inf exp(-a*t) tanh(t/4)/t dt.
For smaller a, Gamma(a+1)=a*Gamma(a) keeps both log-Gamma arguments bounded
away from zero. Floating-point evaluation adds rounding error in either branch.
"""
    if a < 16:
        return float(gammaln(a + .5) - gammaln(a + 1) + .5 * log(a))
    inverse = 1 / a
    squared = inverse * inverse
    return inverse * (-1 / 8 + squared * (1 / 192 + squared * (-1 / 640
        + squared * (17 / 14336 + squared * (-31 / 18432 + squared * 691 / 180224)))))


def _log_one_plus_ratio(numerator, denominator):
    ratio = numerator / denominator
    if isfinite(ratio):
        return log1p(ratio)
    # Only the ratio overflowed: its logarithm and the density may be finite.
    return log(numerator) - log(denominator) + log1p(denominator / numerator)


@lru_cache(maxsize=256)
def _scaled_gamma_log_ratio(nu, dimension):
    """log Gamma(a+n/2)-log Gamma(a)-(n/2)*log(a), a=nu/2.

Only nu/n determine the cached scalar; no observation or draw arrays are held.
The recurrence costs O(n) once per admitted pair, with O(1) working storage.
"""
    a = nu / 2
    if a == 0:
        raise StudentTNumericalError("finite-df Gamma shape nu/2 underflows binary64")
    if 2 * a != nu:
        raise StudentTNumericalError("finite-df Gamma shape nu/2 is not representable exactly")
    count, odd = divmod(dimension, 2)
    offset = .5 if odd else 0.
    value = fsum(_log_one_plus_ratio(index + offset, a) for index in range(count))
    if odd:
        value += _half_step(a)
    if not isfinite(value):
        raise StudentTNumericalError("finite-df Gamma normalization is not representable")
    return value


def _radial_penalty(quadratic, nu, dimension):
    """(nu+n)/2 * log1p(q/nu), preserving the nu part if q/nu underflows."""
    small = quadratic <= nu
    result = np.empty_like(quadratic)
    ratio = quadratic[small] / nu
    logarithm = np.log1p(ratio)
    factor = np.ones_like(ratio)
    np.divide(logarithm, ratio, out=factor, where=ratio != 0)
    # nu*log1p(q/nu) = q*[log1p(r)/r]; at an underflowed r its rounded factor is 1.
    result[small] = (.5 * quadratic[small]) * factor + (.5 * dimension) * logarithm
    large = quadratic[~small]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = large / nu
        logarithm = np.log1p(ratio)
        overflowed = np.isinf(ratio)
        logarithm[overflowed] = (np.log(large[overflowed]) - log(nu)
                                 + np.log1p(nu / large[overflowed]))
        result[~small] = (.5 * nu) * logarithm + (.5 * dimension) * logarithm
    if not np.isfinite(result).all():
        raise StudentTNumericalError("finite-df radial log density is not representable")
    return result


def student_t_log_density(quadratic, *, degrees_of_freedom, dimension, noise_logdet):
    """Evaluate proper t log density from actual n, q >= 0 and log det(shape).

The explicit Gaussian-infinity branch remains separate. This is a numerical
evaluation of the supplied model, not a guarantee of sampling/integration error.
Unrepresentable finite-df operands/results raise a typed numerical failure.
"""
    nu = float(degrees_of_freedom)
    q = np.asarray(quadratic, dtype=float)
    if (type(dimension) is not int or dimension < 0 or not isfinite(noise_logdet)
            or not np.isfinite(q).all() or np.any(q < 0)
            or not (np.isposinf(nu) or isfinite(nu) and nu > 0)):
        raise ValueError("t density requires actual dimension, finite nonnegative energy and valid df/shape")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        constant = -.5 * (dimension * np.log(2 * np.pi) + noise_logdet)
        if np.isposinf(nu):
            result = constant - .5 * q
        else:
            result = (constant + _scaled_gamma_log_ratio(nu, dimension)
                      - _radial_penalty(q, nu, dimension))
    if not np.isfinite(result).all():
        raise StudentTNumericalError("t log density is not representable")
    return result
