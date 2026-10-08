"""Full-C shared-beta/Gamma research reference, with proper priors only.

One physical Gaussian beta is shared by all supplied decisions. Each complete
decision has one independent Gamma precision multiplier. Source/external beta
prior and all actual noise roots are supplied, rather than refitted here.
The model integrals are exact definitions. Finite evaluation uses one positive,
normalized discrete Gamma-prior rule shared by full/prefix and every candidate.
It defines a proper finite Gaussian mixture; convergence to the continuous
Gamma model is a separate approximation claim.
Actual source/prior/known-C roles, native row maps and independent noise scopes
remain caller-owned. This module has no nuisance, hyperparameter estimator,
default degrees of freedom, scalable inference engine or production admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, roots_legendre
from scipy.stats import gamma

from transfer.noise_model import finite_real_array, frozen_array


class NumericalResolutionError(ValueError):
    """The declared proper small-matrix integral is not numerically resolved."""


def _array(value, ndim, name):
    result = finite_real_array(value, name=name)
    if result.ndim != ndim:
        raise ValueError(f"{name} requires finite actual arrays")
    return frozen_array(result)


def _degrees(value):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError("nu requires a positive real scalar or explicit Gaussian infinity")
    nu = float(raw)
    if not (np.isposinf(nu) or np.isfinite(nu) and nu > 0):
        raise ValueError("nu requires a positive real scalar or explicit Gaussian infinity")
    return nu


def _positive_scalar(value, name):
    raw = finite_real_array(value, name=name)
    if raw.ndim != 0 or raw <= 0:
        raise ValueError(f"{name} requires a positive finite real scalar")
    return float(raw)


def _numeric(*values, name):
    if any(not np.isfinite(value).all() for value in values):
        raise NumericalResolutionError(f"{name} is not numerically representable")


def _numeric_array(value, ndim, name):
    result = np.asarray(value, dtype=float)
    _numeric(result, name=name)
    if result.ndim != ndim:
        raise ValueError(f"{name} has the wrong output dimension")
    return frozen_array(result)


def _lower(value, name):
    result = _array(value, 2, name)
    if (not len(result) or result.shape != (len(result), len(result))
            or np.any(np.triu(result, 1) != 0) or np.any(np.diag(result) <= 0)):
        raise ValueError(f"{name} requires a proper positive lower root")
    return result


@dataclass(frozen=True)
class GlobalGaussianPrior:
    mean: np.ndarray
    root: np.ndarray
    program: str

    def __post_init__(self):
        mean, root = _array(self.mean, 1, "physical beta mean"), _lower(self.root, "physical beta prior")
        if len(mean) != len(root):
            raise ValueError("physical beta coordinates must align")
        if not isinstance(self.program, str) or self.program not in {"external", "source_only", "C_design_only"}:
            raise ValueError("a C-response posterior/plugin cannot be relabelled an independent prior")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "root", root)


@dataclass(frozen=True)
class FixedHyperparameters:
    degrees_of_freedom: float
    relative_noise_shape: float
    program: str

    def __post_init__(self):
        nu = _degrees(self.degrees_of_freedom)
        shape = _positive_scalar(self.relative_noise_shape, "relative noise shape")
        if not isinstance(self.program, str) or self.program not in {"externally_fixed", "frozen_C_plugin"}:
            raise ValueError("hyperparameters require an explicitly frozen conditional program")
        object.__setattr__(self, "degrees_of_freedom", nu)
        object.__setattr__(self, "relative_noise_shape", shape)


@dataclass(frozen=True)
class DecisionBlock:
    decision_id: str
    samples: np.ndarray
    design: np.ndarray
    noise_root: np.ndarray
    role: str

    def __post_init__(self):
        y, design = _array(self.samples, 1, "actual vector observation"), _array(self.design, 2, "actual design")
        root = _lower(self.noise_root, "actual fixed K0/spatial root")
        if not len(y) or design.shape[0] != len(y) or len(root) != len(y):
            raise ValueError("actual observed vector/design/noise grid must align")
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError("one actual complete-decision identity is required")
        if not isinstance(self.role, str) or self.role not in {"target_calibration", "read_only_inference"}:
            raise ValueError("this synthetic reference supports declared C and read-only test roles")
        for key, value in (("samples", y), ("design", design), ("noise_root", root)):
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class DiscreteGammaRule:
    """One positive probability measure, fixed before any Y/D/row-count read.

Actual nodes/weights define the finite model and can be replayed directly.
This is not an observation-dimension tilted integral evaluation rule.
"""

    nodes: np.ndarray
    weights: np.ndarray
    degrees_of_freedom: float
    quadrature_order: int
    log_precision_bounds: tuple[float, float] | None = None
    excluded_gamma_prior_probability: float = 0.
    method: str = "declared_positive_discrete_probability"

    def __post_init__(self):
        nodes, weights = _array(self.nodes, 1, "positive precision nodes"), _array(self.weights, 1, "positive prior weights")
        if (not len(nodes) or nodes.shape != weights.shape or np.any(nodes <= 0) or np.any(weights <= 0)
                or type(self.quadrature_order) is not int or self.quadrature_order < 0):
            raise ValueError("finite Gamma prior rule requires aligned positive nodes/weights")
        nu = _degrees(self.degrees_of_freedom)
        if np.isposinf(nu) and (len(nodes) != 1 or nodes[0] != 1 or self.quadrature_order != 0):
            raise ValueError("Gaussian infinity requires one deterministic unit precision")
        excluded = finite_real_array(self.excluded_gamma_prior_probability, name="excluded Gamma prior probability")
        if excluded.ndim != 0 or not 0 <= excluded < 1:
            raise ValueError("represented prior exclusion requires a probability in [0,1)")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("actual finite rule method must be nonempty text")
        object.__setattr__(self, "excluded_gamma_prior_probability", float(excluded))
        if self.log_precision_bounds is not None:
            lower, upper = _array(self.log_precision_bounds, 1, "actual log-precision range")
            if not np.isfinite([lower, upper]).all() or not lower < upper:
                raise ValueError("actual log-precision range must be finite and ordered")
            object.__setattr__(self, "log_precision_bounds", (float(lower), float(upper)))
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise NumericalResolutionError("finite Gamma prior total weight is unresolved")
        normalized = weights / total
        if np.any(normalized <= 0) or not np.isfinite(normalized).all():
            raise NumericalResolutionError("positive finite Gamma weights cannot be represented")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", _array(normalized, 1, "normalized prior weights"))
        object.__setattr__(self, "degrees_of_freedom", nu)


def gamma_prior_rule(degrees_of_freedom, quadrature_order, *, excluded_prior_probability=1e-14):
    """One prior-only positive log-lambda rule, with an explicit finite range.

Gamma quantiles choose the range without reading Y, D or dimensions. Legendre
weights integrate the Gamma density and its log-coordinate Jacobian; normalize
the represented finite probability measure once. Prior exclusion is NOT a
posterior conditional-density error bound. Node order and range are independent
approximation controls, never an automatic exact-Gamma precision certificate.
"""
    nu = _degrees(degrees_of_freedom)
    if np.isposinf(nu):
        return DiscreteGammaRule([1.], [1.], nu, 0)
    if not np.isfinite(nu) or nu <= 0 or type(quadrature_order) is not int or quadrature_order < 1:
        raise ValueError("positive finite nu and integer prior quadrature order required")
    if not np.isfinite(excluded_prior_probability) or not 0 < excluded_prior_probability < 1:
        raise ValueError("prior range requires an explicitly positive excluded probability below one")
    shape = nu / 2
    lower_precision = gamma.ppf(excluded_prior_probability / 2, shape, scale=1 / shape)
    upper_precision = gamma.isf(excluded_prior_probability / 2, shape, scale=1 / shape)
    if (not np.isfinite([lower_precision, upper_precision]).all()
            or not 0 < lower_precision < upper_precision):
        raise NumericalResolutionError("Gamma prior log-precision endpoints are unresolved")
    lower, upper = np.log(lower_precision), np.log(upper_precision)
    abscissae, base_weights = roots_legendre(quadrature_order)
    logarithms = (lower + upper) / 2 + (upper - lower) / 2 * abscissae
    nodes = np.exp(logarithms)
    log_weights = (np.log(base_weights) + np.log((upper - lower) / 2)
                   + shape * np.log(shape) - gammaln(shape) + shape * logarithms - shape * nodes)
    weights = np.exp(log_weights - logsumexp(log_weights))
    if not np.isfinite(nodes).all() or np.any(nodes <= 0) or np.any(weights <= 0):
        raise NumericalResolutionError("positive prior log-lambda quadrature operands are unresolved")
    excluded = float(gamma.cdf(lower_precision, shape, scale=1 / shape)
                     + gamma.sf(upper_precision, shape, scale=1 / shape))
    return DiscreteGammaRule(nodes, weights, nu, quadrature_order, (float(lower), float(upper)), excluded,
                             "normalized_log_lambda_Legendre_truncated_Gamma_prior")


@dataclass(frozen=True)
class JointEvidence:
    log_evidence: float
    beta_mean: np.ndarray
    beta_covariance: np.ndarray
    scale_mean: np.ndarray
    scale_covariance: np.ndarray
    decision_ids: tuple[str, ...]
    evaluated_nodes: int
    quadrature_order: int
    gaussian_limit: bool
    scale_rule: DiscreteGammaRule

    def __post_init__(self):
        mean = _numeric_array(self.beta_mean, 1, "posterior beta mean")
        covariance = _numeric_array(self.beta_covariance, 2, "posterior beta covariance")
        scales = _numeric_array(self.scale_mean, 1, "posterior scale mean")
        scale_covariance = _numeric_array(self.scale_covariance, 2, "posterior scale covariance")
        if (covariance.shape != (len(mean), len(mean))
                or scale_covariance.shape != (len(scales), len(scales))
                or len(scales) != len(self.decision_ids)):
            raise ValueError("joint posterior moments and actual decision axes must align")
        if np.any(np.diag(covariance) <= 0):
            raise NumericalResolutionError("proper beta posterior variances are unresolved")
        if (len(scales) and np.any(self.scale_rule.nodes != self.scale_rule.nodes[0])
                and np.any(np.diag(scale_covariance) <= 0)):
            raise NumericalResolutionError("nondegenerate finite-scale posterior variances are unresolved")
        _numeric(self.log_evidence, name="joint log evidence")
        for key, value in (("beta_mean", mean), ("beta_covariance", covariance),
                           ("scale_mean", scales), ("scale_covariance", scale_covariance)):
            object.__setattr__(self, key, value)

    @property
    def numerical_program(self):
        return "Gaussian_limit" if self.gaussian_limit else "proper_discrete_scale_joint_approximation"


@dataclass(frozen=True)
class ConditionalTailEvidence:
    conditional_log_density: float
    full: JointEvidence
    prefix: JointEvidence
    prior_program: str
    hyperparameter_program: str


def _prepare(prior, blocks):
    blocks = tuple(blocks)
    if not isinstance(prior, GlobalGaussianPrior) or any(not isinstance(block, DecisionBlock) for block in blocks):
        raise ValueError("owned physical prior and decision blocks are required")
    if len({block.decision_id for block in blocks}) != len(blocks):
        raise ValueError("C/test cannot repeat a complete decision or duplicate its Gamma draw")
    width = len(prior.mean)
    prepared = []
    for block in blocks:
        if block.design.shape[1] != width:
            raise ValueError("every decision must use the same actual physical beta chart")
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            response = block.design @ prior.root
            error = block.samples - block.design @ prior.mean
            _numeric(response, error, name="actual beta image and residual")
            image = solve_triangular(block.noise_root, response, lower=True)
            residual = solve_triangular(block.noise_root, error, lower=True)
        if not np.isfinite(image).all() or not np.isfinite(residual).all():
            raise NumericalResolutionError("whitened actual prior image is unresolved")
        prepared.append((image, residual, 2 * np.log(np.diag(block.noise_root)).sum()))
    return blocks, prepared


def _conditional_gaussian(prior, blocks, prepared, lambdas, relative_shape):
    """Integrate one shared beta; return its conditional Gaussian moments."""
    count, width = len(lambdas), len(prior.mean)
    precision = np.broadcast_to(np.eye(width), (count, width, width)).copy()
    linear = np.zeros((count, width))
    normalization = np.zeros(count)
    for j, (block, (image, residual, logdet)) in enumerate(zip(blocks, prepared, strict=True)):
        strength = lambdas[:, j] / relative_shape
        if not np.isfinite(strength).all() or np.any(strength <= 0):
            raise NumericalResolutionError("positive noise precision multiplier is unresolved")
        precision += strength[:, None, None] * (image.T @ image)
        linear += strength[:, None] * (image.T @ residual)
        normalization += logdet + len(block.samples) * (np.log(2 * np.pi * relative_shape) - np.log(lambdas[:, j]))
    try:
        factor = np.linalg.cholesky(precision)
        intermediate = np.linalg.solve(factor, linear[..., None])
        chart_mean = np.linalg.solve(factor.swapaxes(1, 2), intermediate)[..., 0]
        inverse_factor = np.linalg.solve(factor, np.broadcast_to(np.eye(width), factor.shape))
    except np.linalg.LinAlgError as error:
        raise NumericalResolutionError("proper beta conditional precision is numerically unresolved") from error
    chart_covariance = inverse_factor.swapaxes(1, 2) @ inverse_factor
    quadratic = np.sum(chart_mean**2, axis=1)
    for j, (image, residual, _) in enumerate(prepared):
        remaining = residual[None, :] - chart_mean @ image.T
        quadratic += lambdas[:, j] / relative_shape * np.sum(remaining**2, axis=1)
    log_density = -.5 * (normalization + 2 * np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(axis=1) + quadratic)
    beta_mean = prior.mean[None, :] + chart_mean @ prior.root.T
    beta_covariance = prior.root[None, :, :] @ chart_covariance @ prior.root.T[None, :, :]
    if not all(np.isfinite(v).all() for v in (log_density, beta_mean, beta_covariance)):
        raise NumericalResolutionError("conditional beta moments/density are not representable")
    if np.any(np.diagonal(beta_covariance, axis1=1, axis2=2) <= 0):
        raise NumericalResolutionError("proper conditional beta variances are unresolved")
    return log_density, beta_mean, beta_covariance


def conditional_gaussian_components(prior, blocks, hyperparameters, scale_values):
    """Reusable same-beta likelihood and moments for a batch of declared scales.

Returns immutable (log likelihood, beta means, beta covariances). ``scale_values``
has batch x actual-decision axes; each row integrates ONE global proper beta.
Future inference engines reuse this owner and these prior/block classes, rather
than inventing another model. No Gamma weights or hyperparameters are fit here.
"""
    if not isinstance(hyperparameters, FixedHyperparameters):
        raise ValueError("owned fixed hyperparameters are required")
    blocks, prepared = _prepare(prior, blocks)
    scales = _array(scale_values, 2, "actual batch of decision scales")
    if not len(scales) or scales.shape[1] != len(blocks) or np.any(scales <= 0):
        raise ValueError("positive scales must match the complete actual decision axis")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        values = _conditional_gaussian(prior, blocks, prepared, scales, hyperparameters.relative_noise_shape)
    return tuple(_numeric_array(value, value.ndim, "conditional Gaussian component") for value in values)


def _weighted_moments(log_terms, means, covariances=None):
    log_mass = float(logsumexp(log_terms))
    weights = np.exp(log_terms - log_mass)
    mean = weights @ means
    centered = means - mean
    covariance = (centered.T * weights) @ centered
    if covariances is not None:
        covariance += np.einsum("b,bij->ij", weights, covariances)
    _numeric(log_mass, mean, covariance, name="weighted posterior moments")
    return log_mass, mean, covariance


def _merge(old, new):
    if old is None:
        return new
    mass = float(np.logaddexp(old[0], new[0]))
    weight = float(np.exp(old[0] - mass))
    delta = old[1] - new[1]
    mean = weight * old[1] + (1 - weight) * new[1]
    covariance = (weight * old[2] + (1 - weight) * new[2]
                  + weight * (1 - weight) * np.outer(delta, delta))
    _numeric(mass, mean, covariance, name="merged posterior moments")
    return mass, mean, covariance


def joint_evidence(prior, blocks, hyperparameters, *, scale_rule, batch_size=2048):
    """Proper m(all supplied decisions), retaining C-noise posterior mixing.

Tensor cost is order**decisions. batch_size controls memory, not state truncation.
No variational factorization, C Gaussian re-signing, MAP lambda or jitter occurs.
All observations use the same positive discrete prior measure. This finite
model is proper; no continuous-Gamma precision certificate follows from a
node count or properness. Supply scale_rule to replay the actual prior rule.
"""
    if not isinstance(hyperparameters, FixedHyperparameters):
        raise ValueError("owned fixed hyperparameters are required")
    blocks, prepared = _prepare(prior, blocks)
    rule = scale_rule
    if not isinstance(rule, DiscreteGammaRule) or rule.degrees_of_freedom != hyperparameters.degrees_of_freedom:
        raise ValueError("actual finite rule must agree with the declared prior nu")
    if not blocks:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            covariance = prior.root @ prior.root.T
        return JointEvidence(0., prior.mean, covariance, np.empty(0), np.empty((0, 0)), (), 0, 0,
                             bool(np.isposinf(hyperparameters.degrees_of_freedom)), rule)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer memory batch size required")
    nu = float(hyperparameters.degrees_of_freedom)
    relative_shape = float(hyperparameters.relative_noise_shape)
    if np.isposinf(nu):
        lambdas = np.ones((1, len(blocks)))
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            density, means, covariances = _conditional_gaussian(prior, blocks, prepared, lambdas, relative_shape)
        return JointEvidence(float(density[0]), _array(means[0], 1, "beta posterior mean"),
            _array(covariances[0], 2, "beta posterior covariance"), _array(lambdas[0], 1, "fixed Gaussian scales"),
            _array(np.zeros((len(blocks), len(blocks))), 2, "fixed Gaussian scale covariance"),
            tuple(v.decision_id for v in blocks), 1, 0, True, rule)
    log_weights = np.log(rule.weights)
    log_weights -= logsumexp(log_weights)
    indices = product(range(len(rule.nodes)), repeat=len(blocks))
    beta_summary = scale_summary = None
    evaluated = 0
    while batch := tuple(islice(indices, batch_size)):
        positions = np.asarray(batch, dtype=int)
        lambdas = rule.nodes[positions]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            density, means, covariances = _conditional_gaussian(prior, blocks, prepared, lambdas, relative_shape)
            terms = density + log_weights[positions].sum(axis=1)
            beta_summary = _merge(beta_summary, _weighted_moments(terms, means, covariances))
            scale_summary = _merge(scale_summary, _weighted_moments(terms, lambdas))
        evaluated += len(batch)
    if beta_summary is None or scale_summary is None:
        raise NumericalResolutionError("no represented positive Gamma quadrature mass")
    return JointEvidence(beta_summary[0], _array(beta_summary[1], 1, "beta posterior mean"),
        _array(beta_summary[2], 2, "beta posterior covariance"), _array(scale_summary[1], 1, "lambda posterior mean"),
        _array(scale_summary[2], 2, "joint lambda posterior covariance"),
        tuple(v.decision_id for v in blocks), evaluated, rule.quadrature_order, False, rule)


def conditional_tail(prior, calibration, test, conditioning_rows, hyperparameters, *, scale_rule):
    """m(C + full,q)/m(C + actual prefix,q); C is present exactly once in each.

Rows must be actual chronological scalar coordinates. For multichannel input
the caller must supply sample-major arrays/root so the cut is a native prefix.
Roles and IDs enforce only the synthetic contract, never real read permission.
"""
    calibration = tuple(calibration)
    if (any(v.role != "target_calibration" for v in calibration) or test.role != "read_only_inference"
            or type(conditioning_rows) is not int or not 0 < conditioning_rows < len(test.samples)):
        raise ValueError("complete declared C and a genuine nonempty test prefix/tail required")
    prefix = DecisionBlock(test.decision_id, test.samples[:conditioning_rows], test.design[:conditioning_rows],
                           test.noise_root[:conditioning_rows, :conditioning_rows], test.role)
    rule = scale_rule
    full = joint_evidence(prior, calibration + (test,), hyperparameters, scale_rule=rule)
    past = joint_evidence(prior, calibration + (prefix,), hyperparameters, scale_rule=rule)
    return ConditionalTailEvidence(full.log_evidence - past.log_evidence, full, past,
                                   prior.program, hyperparameters.program)
