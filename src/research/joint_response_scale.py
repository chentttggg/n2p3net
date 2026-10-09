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

import hashlib
from dataclasses import dataclass, field
from itertools import islice, product

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, roots_legendre
from scipy.stats import gamma

from transfer.noise_model import finite_real_array, frozen_array, metadata_digest


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


def _input_digest(*arrays):
    """Hash actual operands without allocating a full byte-string copy."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _raw_streams(samples, channels, width):
    return tuple(hashlib.sha256(str((shape, np.dtype(float).str)).encode())
                 for shape in ((samples, channels), (samples, channels, width)))


def _raw_stream_identity(streams):
    return metadata_digest({"Y": streams[0].hexdigest(), "D": streams[1].hexdigest()})


def _chart_identity(prior, hyperparameters):
    return metadata_digest({"prior_program": prior.program,
                            "physical_prior": _input_digest(prior.mean, prior.root),
                            "nu": str(hyperparameters.degrees_of_freedom),
                            "relative_noise_shape": hyperparameters.relative_noise_shape,
                            "hyperparameter_program": hyperparameters.program})


@dataclass(frozen=True, init=False, eq=False)
class CompiledDecision:
    """Factory-owned proper likelihood rows, never a generic root admission API.

    ``observed_rows`` counts original scalar observations, including residual
    directions removed by QR. A conditional tail shares the full decision's
    scale and cannot enter an independent-decision observation set.
    Source/read authority remains outside this research numerical carrier.
    """

    decision_id: str
    role: str
    augmented_root: np.ndarray
    observed_rows: int
    noise_logdet: float
    row_start: int
    row_stop: int
    kind: str
    chart_identity: str
    source_identity: str
    map_identity: str
    raw_input_identity: str
    prefix: CompiledDecision | None = field(repr=False)
    tail: CompiledDecision | None = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("compiled proper likelihoods require a canonical compiler factory")


def _compiled(root, *, decision_id, role, rows, logdet, start, kind, chart, source, mapping, raw):
    _numeric(root, logdet, name="compiled augmented likelihood")
    result = object.__new__(CompiledDecision)
    fields = {"decision_id": decision_id, "role": role, "augmented_root": frozen_array(root),
              "observed_rows": rows, "noise_logdet": float(logdet), "row_start": start,
              "row_stop": start + rows, "kind": kind, "chart_identity": chart,
              "source_identity": source, "map_identity": mapping, "raw_input_identity": raw,
              "prefix": None, "tail": None}
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    return result


def _augmented_qr(rows, chunk_rows):
    """Preserve all p+1 columns and the final residual-only row."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("QR work-block size must be a positive integer")
    root = np.empty((0, rows.shape[1]))
    for start in range(0, len(rows), chunk_rows):
        root = np.linalg.qr(np.vstack((root, rows[start:start + chunk_rows])), mode="r")
    _numeric(root, name="augmented QR root")
    return root


def _compile_white(white, *, decision_id, role, full_logdet, prefix_logdet, cut,
                   chart, source, mapping, full_raw, prefix_raw, qr_chunk_rows):
    common = {"decision_id": decision_id, "role": role, "chart": chart,
              "source": source, "mapping": mapping}
    full = _compiled(_augmented_qr(white, qr_chunk_rows), rows=len(white), logdet=full_logdet,
                     start=0, kind="full", raw=full_raw, **common)
    if cut is not None:
        prefix = _compiled(_augmented_qr(white[:cut], qr_chunk_rows), rows=cut, logdet=prefix_logdet,
                           start=0, kind="prefix", raw=prefix_raw, **common)
        tail = _compiled(_augmented_qr(white[cut:], qr_chunk_rows), rows=len(white)-cut,
                         logdet=full_logdet-prefix_logdet, start=cut, kind="conditional_tail",
                         raw=full_raw, **common)
        object.__setattr__(full, "prefix", prefix)
        object.__setattr__(full, "tail", tail)
    return full


def _compilation_chart(prior, hyperparameters):
    if not isinstance(prior, GlobalGaussianPrior) or not isinstance(hyperparameters, FixedHyperparameters):
        raise TypeError("canonical proper physical prior and fixed hyperparameters are required")
    return _chart_identity(prior, hyperparameters)


def _prefix_cut(cut, count):
    if cut is not None and (type(cut) is not int or not 0 < cut < count):
        raise ValueError("an actual nonempty prefix and tail are required")


def compile_decision(prior, hyperparameters, block, *, conditioning_rows=None, qr_chunk_rows=2048):
    """Compile one dense proper decision; split only original whitened rows.

    Independently supplied dense prefixes bind by their exact y/D/leading
    noise-root operands, rather than shape-dependent QR rounding.
    """
    chart = _compilation_chart(prior, hyperparameters)
    if not isinstance(block, DecisionBlock):
        raise TypeError("a canonical proper raw DecisionBlock is required")
    p, n, cut = len(prior.mean), len(block.samples), conditioning_rows
    if block.design.shape[1] != p:
        raise ValueError("every decision must use the same actual physical beta chart")
    _prefix_cut(cut, n)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        augmented = np.column_stack((block.design @ prior.root, block.samples-block.design @ prior.mean))
        _numeric(augmented, name="actual beta image and residual")
        white = solve_triangular(block.noise_root, augmented, lower=True)
        white /= np.sqrt(hyperparameters.relative_noise_shape)
        full_logdet = 2*np.log(np.diag(block.noise_root)).sum()+n*np.log(hyperparameters.relative_noise_shape)
        prefix_logdet = (None if cut is None else 2*np.log(np.diag(block.noise_root)[:cut]).sum()
                         +cut*np.log(hyperparameters.relative_noise_shape))
    raw = _input_digest(block.samples, block.design, block.noise_root)
    prefix_raw = (None if cut is None else _input_digest(block.samples[:cut], block.design[:cut],
                                                        block.noise_root[:cut, :cut]))
    _numeric(white, name="actual full-record whitening")
    return _compile_white(white, decision_id=block.decision_id, role=block.role, cut=cut,
                          full_logdet=full_logdet, prefix_logdet=prefix_logdet, chart=chart,
                          source=raw, mapping=_input_digest(block.noise_root), full_raw=raw,
                          prefix_raw=prefix_raw, qr_chunk_rows=qr_chunk_rows)


def compile_owned_decision(prior, hyperparameters, noise, *, decision_id, role, native_samples,
                           raw_batches, conditioning_samples=None, qr_chunk_rows=2048):
    """Use the existing B05 full-record map and public whitening Jacobian once.

    ``raw_batches`` supplies contiguous (start_sample, Y[N,C], D[N,C,p]). B05
    still requires the whole record: chunks do not reset the temporal filter.
    All prefix/tail roots come independently from its original white rows.
    No dense square covariance or observation-by-draw matrix is constructed.
    """
    from research.channel_noise_mechanism import FittedChannelNoise

    chart = _compilation_chart(prior, hyperparameters)
    if not isinstance(noise, FittedChannelNoise):
        raise TypeError("the canonical proper B05 noise owner is required")
    if (not isinstance(decision_id, str) or not decision_id.strip()
            or role not in {"target_calibration", "read_only_inference"}
            or type(native_samples) is not int or native_samples < 1):
        raise ValueError("one actual complete decision, role and native sample count are required")
    _prefix_cut(conditioning_samples, native_samples)
    p, channels = len(prior.mean), len(noise.geometry.channels)
    augmented = np.empty((native_samples, channels, p+1))
    raw_streams = _raw_streams(native_samples, channels, p)
    prefix_streams = (None if conditioning_samples is None else
                      _raw_streams(conditioning_samples, channels, p))
    offset = 0
    for start, samples, design in raw_batches:
        y = finite_real_array(samples, name="actual source observations")
        d = finite_real_array(design, name="actual source design")
        if (type(start) is not int or start != offset or y.ndim != 2 or not len(y)
                or y.shape[1] != channels or d.shape != (*y.shape, p)):
            raise ValueError("actual contiguous sample-major physical Y/D chunks are required")
        stop = offset+len(y)
        if stop > native_samples:
            raise ValueError("source reader exceeds the declared complete decision")
        # Bind original canonical Y/D before the chart multiplication can round
        # away their differences. Independent streams make reader cuts irrelevant.
        for digest, value in zip(raw_streams, (y, d), strict=True):
            digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        if prefix_streams is not None:
            leading = min(len(y), max(conditioning_samples-offset, 0))
            if leading:
                for digest, value in zip(prefix_streams, (y[:leading], d[:leading]), strict=True):
                    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        augmented[offset:stop, :, :p] = d @ prior.root
        augmented[offset:stop, :, p] = y-d @ prior.mean
        offset = stop
    if offset != native_samples:
        raise ValueError("all declared decision rows must actually be supplied")
    _numeric(augmented, name="actual source augmented data")
    mapping = metadata_digest(noise.record())
    source = _raw_stream_identity(raw_streams)
    full_raw = metadata_digest({"source": source, "map": mapping})
    prefix_raw = (None if prefix_streams is None else
                  metadata_digest({"source": _raw_stream_identity(prefix_streams), "map": mapping}))
    augmented /= np.sqrt(hyperparameters.relative_noise_shape)
    white = noise.whiten(augmented).reshape(native_samples*channels, p+1)
    del augmented
    n, cut = len(white), None if conditioning_samples is None else conditioning_samples*channels
    full_logdet = -2*noise.whitening_logabsdet(native_samples)+n*np.log(hyperparameters.relative_noise_shape)
    prefix_logdet = (None if cut is None else -2*noise.whitening_logabsdet(conditioning_samples)
                     +cut*np.log(hyperparameters.relative_noise_shape))
    return _compile_white(white, decision_id=decision_id, role=role, cut=cut,
                          full_logdet=full_logdet, prefix_logdet=prefix_logdet, chart=chart,
                          source=source, mapping=mapping, full_raw=full_raw,
                          prefix_raw=prefix_raw, qr_chunk_rows=qr_chunk_rows)


def _prepare(prior, blocks, hyperparameters):
    chart = _compilation_chart(prior, hyperparameters)
    prepared = []
    for block in blocks:
        if isinstance(block, DecisionBlock):
            block = compile_decision(prior, hyperparameters, block)
        if not isinstance(block, CompiledDecision):
            raise TypeError("canonical proper raw or factory-compiled decisions are required")
        if block.chart_identity != chart:
            raise ValueError("compiled decisions must use the same physical prior and noise program")
        if block.kind == "conditional_tail":
            raise ValueError("a conditional tail cannot introduce another independent Gamma decision")
        prepared.append(block)
    if len({block.decision_id for block in prepared}) != len(prepared):
        raise ValueError("C/test cannot repeat a complete decision or duplicate its Gamma draw")
    return tuple(prepared)


def _qr_beta(records, scales, width):
    """Admitted numerical core: one prior I, all complete-decision roots."""
    terms = [np.column_stack((np.eye(width), np.zeros(width)))]
    terms.extend(np.sqrt(lam)*record.augmented_root for record, lam in zip(records, scales, strict=True))
    augmented = np.linalg.qr(np.vstack(terms), mode="r")
    root, rhs = augmented[:width, :width], augmented[:width, width]
    _numeric(root, rhs, name="proper conditional beta QR")
    if np.any(np.diag(root) == 0):
        raise NumericalResolutionError("proper conditional beta QR is numerically unresolved")
    mean = solve_triangular(root, rhs, lower=False)
    _numeric(mean, name="proper conditional beta mean")
    return mean, root


def _root_energy(record, charts):
    vectors = np.column_stack((charts, -np.ones(len(charts))))
    residual = vectors @ record.augmented_root.T
    result = np.einsum("bi,bi->b", residual, residual)
    _numeric(result, name="augmented residual energy")
    return result


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


def _conditional_gaussian(prior, prepared, lambdas):
    """Integrate one shared proper beta using augmented square-root statistics."""
    count, width = len(lambdas), len(prior.mean)
    log_density = np.empty(count)
    means = np.empty((count, width))
    covariances = np.empty((count, width, width))
    for index, scales in enumerate(lambdas):
        chart_mean, root = _qr_beta(prepared, scales, width)
        inverse = solve_triangular(root, np.eye(width), lower=False)
        chart_covariance = inverse @ inverse.T
        quadratic = float(chart_mean @ chart_mean)
        normalization = 0.
        for block, lam in zip(prepared, scales, strict=True):
            quadratic += lam * _root_energy(block, chart_mean[None])[0]
            normalization += block.noise_logdet + block.observed_rows * (np.log(2*np.pi)-np.log(lam))
        log_density[index] = -.5 * (normalization+2*np.log(abs(np.diag(root))).sum()+quadratic)
        means[index] = prior.mean+prior.root @ chart_mean
        covariances[index] = prior.root @ chart_covariance @ prior.root.T
    _numeric(log_density, means, covariances, name="conditional beta moments/density")
    if np.any(np.diagonal(covariances, axis1=1, axis2=2) <= 0):
        raise NumericalResolutionError("proper conditional beta variances are unresolved")
    return log_density, means, covariances


def conditional_gaussian_components(prior, blocks, hyperparameters, scale_values):
    """Reusable same-beta likelihood and moments for a batch of declared scales.

Returns immutable (log likelihood, beta means, beta covariances). ``scale_values``
has batch x actual-decision axes; each row integrates ONE global proper beta.
Future inference engines reuse this owner and these prior/block classes, rather
than inventing another model. No Gamma weights or hyperparameters are fit here.
"""
    if not isinstance(hyperparameters, FixedHyperparameters):
        raise ValueError("owned fixed hyperparameters are required")
    blocks = prepared = _prepare(prior, blocks, hyperparameters)
    scales = _array(scale_values, 2, "actual batch of decision scales")
    if not len(scales) or scales.shape[1] != len(blocks) or np.any(scales <= 0):
        raise ValueError("positive scales must match the complete actual decision axis")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        values = _conditional_gaussian(prior, prepared, scales)
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
    blocks = prepared = _prepare(prior, blocks, hyperparameters)
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
    if np.isposinf(nu):
        lambdas = np.ones((1, len(blocks)))
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            density, means, covariances = _conditional_gaussian(prior, prepared, lambdas)
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
            density, means, covariances = _conditional_gaussian(prior, prepared, lambdas)
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
    if isinstance(test, DecisionBlock):
        test = compile_decision(prior, hyperparameters, test, conditioning_rows=conditioning_rows)
    if (not isinstance(test, CompiledDecision) or test.role != "read_only_inference"
            or any(v.role != "target_calibration" for v in calibration)
            or type(conditioning_rows) is not int or test.prefix is None
            or test.prefix.observed_rows != conditioning_rows):
        raise ValueError("complete declared C and a genuine nonempty test prefix/tail required")
    prefix = test.prefix
    rule = scale_rule
    full = joint_evidence(prior, calibration + (test,), hyperparameters, scale_rule=rule)
    past = joint_evidence(prior, calibration + (prefix,), hyperparameters, scale_rule=rule)
    return ConditionalTailEvidence(full.log_evidence - past.log_evidence, full, past,
                                   prior.program, hyperparameters.program)
