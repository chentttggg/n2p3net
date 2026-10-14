"""Research full-C Gibbs and proper finite empirical predictive mixtures.

One proper global beta and one independent Gamma draw per complete decision.
All physical inputs/hyperparameters follow the existing full-C reference model.
Budgets and initial states are explicit caller arguments; no production entry,
automatic fallback, hyperparameter fit, exactness certificate or real-read grant.
Canonical prior, hyperparameters and decision classes are shared with
``research.joint_response_scale``. Its finite Gamma rule is NOT the Gibbs target:
this sampler targets the declared continuous-Gamma model with random error.
Finite Rhat/ESS/MCSE cannot detect modes missed by all chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtri
from scipy.stats import rankdata

from research.joint_response_scale import (
    CompiledDecision,
    DecisionBlock,
    FixedHyperparameters,
    GlobalGaussianPrior,
    NumericalResolutionError,
    _chart_identity,
    _prepare,
    _qr_beta,
    _root_energy,
    compile_decision,
)
from research.student_t_numerics import StudentTNumericalError, student_t_log_density
from transfer.noise_model import finite_real_array, frozen_array


class SamplingResolutionError(ValueError):
    """The declared numerical sampling calculation is not representable."""


class PredictiveSamplingError(ValueError):
    """Caller-declared Monte Carlo diagnostic thresholds were not met."""


@dataclass(frozen=True)
class PreparedModel:
    prior: GlobalGaussianPrior
    blocks: tuple[CompiledDecision, ...]
    hyperparameters: FixedHyperparameters

    def __post_init__(self):
        blocks = _prepare(self.prior, self.blocks, self.hyperparameters)
        if (any(v.role != "target_calibration" for v in blocks[:-1])
                or blocks and blocks[-1].role not in {"target_calibration", "read_only_inference"}):
            raise ValueError("only complete C followed by at most one actual inference prefix is supported")
        object.__setattr__(self, "blocks", blocks)


@dataclass(frozen=True)
class PreparedPrediction:
    """One actual full map, its independent original-row prefix/tail roots."""

    prior: GlobalGaussianPrior
    hyperparameters: FixedHyperparameters
    full: CompiledDecision
    prefix: CompiledDecision
    tail: CompiledDecision
    draw_batch_size: int

    def __post_init__(self):
        if (not isinstance(self.full, CompiledDecision) or self.full.kind != "full"
                or self.full.role != "read_only_inference" or self.prefix is None or self.tail is None
                or self.prefix is not self.full.prefix or self.tail is not self.full.tail):
            raise ValueError("prediction requires the same factory full/prefix/tail map")
        if self.full.chart_identity != _chart_identity(self.prior, self.hyperparameters):
            raise ValueError("prediction requires the same physical prior and noise program")
        if type(self.draw_batch_size) is not int or self.draw_batch_size < 1:
            raise ValueError("prediction draw work-block size must be a positive integer")


@dataclass(frozen=True)
class JointResponseSamples:
    """One immutable finite stochastic run, not a certified posterior artifact."""

    model: PreparedModel
    chart: np.ndarray
    scales: np.ndarray
    burnin_per_chain: int
    draws_per_chain: int
    chain_seeds: tuple[int, ...]
    elapsed_seconds: float
    beta: np.ndarray = field(init=False, repr=False, compare=False)
    gaussian_limit: bool = field(init=False)

    def __post_init__(self):
        chart = finite_real_array(self.chart, name="retained global charts")
        scales = finite_real_array(self.scales, name="retained decision precisions")
        expected = (len(self.chain_seeds), self.draws_per_chain)
        if (chart.shape != (*expected, len(self.model.prior.mean))
                or scales.shape != (*expected, len(self.model.blocks)) or np.any(scales <= 0)):
            raise ValueError("retained draws must preserve one global beta and every complete-decision lambda")
        gaussian = bool(np.isposinf(self.model.hyperparameters.degrees_of_freedom))
        if gaussian and np.any(scales != 1):
            raise ValueError("the explicit Gaussian limit has deterministic unit precisions")
        with np.errstate(over="ignore", invalid="ignore"):
            beta = self.model.prior.mean + chart @ self.model.prior.root.T
        if not np.isfinite(beta).all():
            raise SamplingResolutionError("sampled physical beta is not representable")
        for key, value in (("chart", chart), ("scales", scales), ("beta", beta)):
            object.__setattr__(self, key, frozen_array(value))
        object.__setattr__(self, "gaussian_limit", gaussian)


def prepare_model(prior, blocks, hyperparameters):
    """Admit proper raw or factory-compiled C/C+actual-prefix rows once.

    Physical coordinates, source/read authority and cross-decision independence
    remain caller-owned. A tail cannot add another independent Gamma variable.
    """
    return PreparedModel(prior, tuple(blocks), hyperparameters)


def prepare_prediction(model, test, *, conditioning_rows, draw_batch_size=256):
    """Compile the actual full test once, with the SAME original prefix rows."""
    if isinstance(test, DecisionBlock):
        test = compile_decision(model.prior, model.hyperparameters, test,
                                conditioning_rows=conditioning_rows)
    if (not isinstance(test, CompiledDecision) or test.role != "read_only_inference"
            or test.kind != "full" or test.prefix is None or test.tail is None
            or type(conditioning_rows) is not int or test.prefix.observed_rows != conditioning_rows):
        raise ValueError("a canonical full read-only test with its actual prefix and tail is required")
    if any(v.decision_id == test.decision_id and v.role == "target_calibration" for v in model.blocks):
        raise ValueError("test cannot duplicate a complete C decision or its Gamma draw")
    prediction = PreparedPrediction(model.prior, model.hyperparameters, test, test.prefix, test.tail,
                                    draw_batch_size)
    _prediction_binding(model, prediction)
    return prediction


def _prediction_binding(model, prediction):
    if (not isinstance(prediction, PreparedPrediction)
            or prediction.prior.program != model.prior.program
            or not np.array_equal(prediction.prior.mean, model.prior.mean)
            or not np.array_equal(prediction.prior.root, model.prior.root)
            or prediction.hyperparameters != model.hyperparameters):
        raise ValueError("prediction and sampled C must share the same physical prior and noise program")
    # Compile-time chart binds all proper beta/noise operands, including a no-C model.
    if prediction.full.chart_identity != _chart_identity(model.prior, model.hyperparameters):
        raise ValueError("prediction and sampled C must share the same physical prior and noise program")


def _targeted_prediction_binding(model, prediction):
    _prediction_binding(model, prediction)
    if not model.blocks:
        raise ValueError("targeted sampling requires the actual test prefix")
    last, expected = model.blocks[-1], prediction.prefix
    if (any(v.role != "target_calibration" for v in model.blocks[:-1])
            or last.decision_id != expected.decision_id or last.role != "read_only_inference"
            or last.observed_rows != expected.observed_rows or last.noise_logdet != expected.noise_logdet
            or last.raw_input_identity != expected.raw_input_identity):
        raise ValueError("the actual test prefix must have entered the global sampler exactly once")


def _conditional_beta_chart(model, values):
    try:
        mean, upper = _qr_beta(model.blocks, values, len(model.prior.mean))
    except (NumericalResolutionError, np.linalg.LinAlgError) as error:
        raise SamplingResolutionError("proper conditional beta QR is unresolved") from error
    # Existing single RNG loop solves lower.T * shift=z; precision is R.T R.
    return mean, upper.T


def conditional_beta_chart(model, lambdas):
    """u|lambda,y, using one prior I and complete augmented square-root rows."""
    values = finite_real_array(lambdas, name="conditional decision precisions")
    if values.shape != (len(model.blocks),) or np.any(values <= 0):
        raise ValueError("one finite positive conditional scale per supplied decision is required")
    return _conditional_beta_chart(model, values)


def _conditional_scale_parameters(model, u):
    try:
        energies = np.asarray([_root_energy(v, u[None])[0] for v in model.blocks])
    except NumericalResolutionError as error:
        raise SamplingResolutionError("conditional Gamma energy is not representable") from error
    nu = model.hyperparameters.degrees_of_freedom
    # Proper: n_raw. Factory-owned flat quotient: n_raw-rank(Z).
    # Neither compressed root height nor proper beta width changes this count.
    shapes = np.asarray([(nu+v.density_dimension)/2 for v in model.blocks])
    rates = (nu+energies)/2
    if not np.isfinite(shapes).all() or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise SamplingResolutionError("conditional Gamma operands are not representable")
    return shapes, rates


def conditional_scale_parameters(model, chart):
    """Independent Gamma((nu+n_raw)/2,(nu+||R[u;-1]||²)/2)."""
    u = finite_real_array(chart, name="global prior chart")
    if u.shape != (len(model.prior.mean),):
        raise ValueError("the same finite global prior chart is required")
    return _conditional_scale_parameters(model, u)


def run_gibbs(model, *, burnin, draws, seeds, initial_charts):
    """Systematic u->lambda Gibbs; finite runs are approximations, never exact draws.

Caller supplies independent chain seeds, overdispersed starts and actual finite
sampling program. Gaussian infinity has fixed lambda=1 and independent exact
conditional-Gaussian draws; no finite nu is silently relabelled infinity.
"""
    starts = finite_real_array(initial_charts, name="initial global charts")
    seeds = tuple(seeds)
    p, j = len(model.prior.mean), len(model.blocks)
    if (type(burnin) is not int or burnin < 0 or type(draws) is not int or draws < 2
            or starts.shape != (len(seeds), p) or len(set(seeds)) != len(seeds)
            or not len(seeds) or any(type(v) is not int or v < 0 for v in seeds)):
        raise ValueError("an explicit finite sampling program and independent chain identities are required")
    chart_draws = np.empty((len(seeds), draws, p))
    scale_draws = np.empty((len(seeds), draws, j))
    gaussian = np.isposinf(model.hyperparameters.degrees_of_freedom)
    gaussian_conditional = _conditional_beta_chart(model, np.ones(j)) if gaussian else None
    started = perf_counter()
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        u, lambdas = starts[chain].copy(), np.ones(j)
        if not gaussian:
            shape, rate = _conditional_scale_parameters(model, u)
            lambdas = rng.gamma(shape, 1 / rate)
        for iteration in range(burnin + draws):
            mean, root = gaussian_conditional if gaussian else _conditional_beta_chart(model, lambdas)
            u = mean + solve_triangular(root.T, rng.normal(size=p), lower=False)
            for record, precision in zip(model.blocks, lambdas, strict=True):
                record.validate_precision(u[None], precision)
            if not gaussian:
                shape, rate = _conditional_scale_parameters(model, u)
                lambdas = rng.gamma(shape, 1 / rate)
            if not np.isfinite(u).all() or not np.isfinite(lambdas).all() or np.any(lambdas <= 0):
                raise SamplingResolutionError("Gamma/Gaussian draw is not finite and positive")
            if iteration >= burnin:
                chart_draws[chain, iteration - burnin] = u
                scale_draws[chain, iteration - burnin] = lambdas
    return JointResponseSamples(model, chart_draws, scale_draws, burnin, draws, seeds, perf_counter() - started)


def _split(values):
    x = finite_real_array(values, name="scalar MCMC chains")
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError("multi-chain diagnostics require finite scalar chains of at least eight draws")
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def _variance(x):
    n = x.shape[1]
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(x, axis=1), ddof=1))
    return within, (n - 1) / n * within + between / n


def scalar_chain_diagnostics(values):
    """Rank-normalized/folded split Rhat and bulk/tail Geyer-monotone ESS.

These finite diagnostics cannot detect a mode missed by every initial chain,
or certify stationarity/unbiased sampling. ESS is functional-specific.
"""
    x = _split(values)

    def rank_normalize(z):
        ranks = rankdata(z.ravel(), method="average")
        return ndtri((ranks - 3 / 8) / (z.size + 1 / 4)).reshape(z.shape)

    def rhat(z):
        within, variance = _variance(z)
        return float(np.sqrt(variance / within)) if within > 0 else np.inf

    def ess(z):
        within, variance = _variance(z)
        if within <= 0 or variance <= 0:
            return 0.
        centered = z - z.mean(axis=1, keepdims=True)
        n = z.shape[1]
        length = 1 << (2 * n - 1).bit_length()
        transformed = np.fft.rfft(centered, n=length, axis=1)
        covariance = np.fft.irfft(transformed * transformed.conj(), n=length, axis=1)[:, :n] / n
        rho = 1 - (within - covariance.mean(axis=0)) / variance
        rho[0] = 1.
        paired = []
        for lag in range(0, n - 1, 2):
            pair = float(rho[lag] + rho[lag + 1])
            if pair <= 0:
                break
            paired.append(min(pair, paired[-1]) if paired else pair)
        # Conservatively cap antithetic ESS at actual draws; this is a diagnostic
        # convention, not a claim that importance/multi-chain draws are iid.
        return float(z.size / max(-1 + 2 * sum(paired), 1.))

    normalized = rank_normalize(x)
    folded = rank_normalize(abs(x - np.median(x)))
    lower, upper = np.quantile(x, [.05, .95])
    return {"rank_folded_split_rhat": max(rhat(normalized), rhat(folded)),
            "bulk_ess": ess(normalized),
            "tail_ess": min(ess((x <= lower).astype(float)), ess((x >= upper).astype(float)))}


def posterior_diagnostics(samples):
    values = samples.chart
    if not samples.gaussian_limit and samples.scales.shape[2]:
        values = np.concatenate((values, np.log(samples.scales)), axis=2)
    diagnostics = [scalar_chain_diagnostics(values[:, :, index]) for index in range(values.shape[2])]
    return {"max_rank_folded_split_rhat": max(v["rank_folded_split_rhat"] for v in diagnostics),
            "min_bulk_ess": min(v["bulk_ess"] for v in diagnostics),
            "min_tail_ess": min(v["tail_ess"] for v in diagnostics),
            "diagnosed_dimensions": len(diagnostics), "per_coordinate": diagnostics,
            "scope": "finite_MCMC_coordinate_diagnostics",
            "continuous_Gamma_target_accuracy_certified": False,
            "unvisited_modes_detected": False}


def _predictive_policy(max_rhat, min_bulk_ess, min_tail_ess):
    values = finite_real_array([max_rhat, min_bulk_ess, min_tail_ess], name="caller MCMC thresholds")
    if values[0] <= 0 or np.any(values[1:] < 0):
        raise ValueError("caller Rhat threshold must be positive and ESS thresholds nonnegative")


def _functional_diagnostics(values):
    if np.ptp(values) == 0:
        return {"constant_at_retained_draws": True, "rank_folded_split_rhat": None,
                "bulk_ess": None, "tail_ess": None,
                "unvisited_modes_detected": False}
    return scalar_chain_diagnostics(values)


def _t_logpdf(record, charts, nu):
    n = record.observed_rows
    try:
        quadratic = _root_energy(record, charts)
    except NumericalResolutionError as error:
        raise SamplingResolutionError("normalized predictive residual energy is unresolved") from error
    try:
        return student_t_log_density(quadratic, degrees_of_freedom=nu, dimension=n,
                                     noise_logdet=record.noise_logdet)
    except StudentTNumericalError as error:
        raise SamplingResolutionError("normalized finite-df predictive density is unresolved") from error


def conditional_tail_components(prediction, chart):
    """All retained beta draws, batched; one actual test lambda integrated out.

    Each normalized p(tail|u,prefix) uses original-row full/prefix roots from the
    same causal map. Prefix weights never inspect the tail. No draw is removed.
    """
    charts = finite_real_array(chart, name="sampled global charts")
    p = len(prediction.prior.mean)
    if charts.ndim not in (2, 3) or charts.shape[-1] != p:
        raise ValueError("prediction components require the actual sampled global chart width")
    u = charts.reshape(-1, p)
    full, prefix = np.empty(len(u)), np.empty(len(u))
    nu = prediction.hyperparameters.degrees_of_freedom
    for start in range(0, len(u), prediction.draw_batch_size):
        end = min(start+prediction.draw_batch_size, len(u))
        full[start:end] = _t_logpdf(prediction.full, u[start:end], nu)
        prefix[start:end] = _t_logpdf(prediction.prefix, u[start:end], nu)
    return full, prefix, full-prefix


def _conditional_beta_tail_logpdf(model, prediction, scales):
    """Private admitted joint-lambda state core; no repeated source admission."""
    mean, lower = _conditional_beta_chart(model, scales)
    prefix_root = lower.T
    tail = prediction.tail
    proper_prior = np.column_stack((prefix_root, prefix_root @ mean))
    augmented = np.linalg.qr(np.vstack((proper_prior, np.sqrt(scales[-1])*tail.augmented_root)), mode="r")
    p = len(mean)
    full_root, rhs = augmented[:p, :p], augmented[:p, p]
    updated = solve_triangular(full_root, rhs, lower=False)
    remaining = tail.augmented_root[:, :p] @ updated-tail.augmented_root[:, p]
    shift = prefix_root @ (updated-mean)
    # Positive energies: never subtract two large C/full likelihood quadratics.
    quadratic = scales[-1]*(remaining @ remaining)+shift @ shift
    logdet = 2*(np.log(abs(np.diag(full_root)))-np.log(abs(np.diag(prefix_root)))).sum()
    density = -.5*(tail.observed_rows*(np.log(2*np.pi)-np.log(scales[-1]))
                   +tail.noise_logdet+logdet+quadratic)
    if not np.isfinite(density):
        raise SamplingResolutionError("conditional-beta Gaussian tail density is not finite")
    return float(density)


def _conditional_beta_components(model, prediction, scales):
    logs = np.empty(len(scales))
    # Exact Gaussian-limit states repeat; their conditional component is identical.
    gaussian_value = None
    for start in range(0, len(scales), prediction.draw_batch_size):
        for index in range(start, min(start+prediction.draw_batch_size, len(scales))):
            if np.isposinf(model.hyperparameters.degrees_of_freedom):
                if gaussian_value is None:
                    gaussian_value = _conditional_beta_tail_logpdf(model, prediction, scales[index])
                logs[index] = gaussian_value
            else:
                logs[index] = _conditional_beta_tail_logpdf(model, prediction, scales[index])
    return logs


def conditional_beta_tail_components(model, prediction, scale_values):
    """Integrate proper beta|all SAME C+prefix joint scales for every state.

    Gaussianity is conditional on paired joint scales. This never relabels a
    separate Gaussian-C posterior as the finite-nu marginal beta posterior.
    """
    _targeted_prediction_binding(model, prediction)
    scales = finite_real_array(scale_values, name="targeted joint precision states")
    if scales.ndim not in (2, 3) or scales.shape[-1] != len(model.blocks) or np.any(scales <= 0):
        raise ValueError("positive joint scales must preserve every C and actual-prefix decision")
    if np.isposinf(model.hyperparameters.degrees_of_freedom) and np.any(scales != 1):
        raise ValueError("the explicit Gaussian limit has deterministic unit precisions")
    return _conditional_beta_components(model, prediction, scales.reshape(-1, len(model.blocks)))


def _batch_mcse(influence, batch_size):
    x = np.asarray(influence, dtype=float)
    if type(batch_size) is not int or batch_size < 2 or x.shape[1] % batch_size:
        raise ValueError("caller batch size must divide each chain length")
    batches = x.reshape(x.shape[0], -1, batch_size).mean(axis=2).ravel()
    if len(batches) < 4:
        raise ValueError("at least four actual batches are required for this MCSE estimate")
    return float(np.std(batches, ddof=1) / np.sqrt(len(batches)))


def prefix_importance_predict(samples, prediction, *, batch_size, min_weight_ess_fraction,
                              max_weight_fraction, max_rhat, min_bulk_ess, min_tail_ess):
    """One C posterior, then explicit prefix importance update under the SAME model.

Finite draws define one common discrete C-beta measure: prefix weights are fixed
before reading the tail. Its mixture of conditional t components is proper even
at finite sample size. MCSE/weight concentration remain approximations, not
certificates. Raw Kish weight ESS is NOT MCMC ESS or a variance guarantee.
"""
    model = samples.model
    _predictive_policy(max_rhat, min_bulk_ess, min_tail_ess)
    concentration = finite_real_array([min_weight_ess_fraction, max_weight_fraction],
                                     name="caller importance thresholds")
    if not 0 <= concentration[0] <= 1 or not 0 < concentration[1] <= 1:
        raise ValueError("caller importance thresholds must be valid positive mass fractions")
    _prediction_binding(model, prediction)
    if any(v.role != "target_calibration" for v in model.blocks):
        raise ValueError("C-only sampling and declared read-only test roles are required")
    diagnostics = posterior_diagnostics(samples)
    if (diagnostics["max_rank_folded_split_rhat"] > max_rhat
            or diagnostics["min_bulk_ess"] < min_bulk_ess
            or diagnostics["min_tail_ess"] < min_tail_ess):
        raise PredictiveSamplingError("caller-declared C posterior MCMC diagnostics were not met")
    full, prefix, tail = conditional_tail_components(prediction, samples.chart)
    weights = np.exp(prefix - logsumexp(prefix))
    raw_weight_ess = float(1 / np.sum(weights ** 2))
    largest = float(weights.max())
    if raw_weight_ess / len(weights) < min_weight_ess_fraction or largest > max_weight_fraction:
        raise PredictiveSamplingError("prefix importance weights collapsed under caller-declared thresholds")
    log_density = float(logsumexp(tail + prefix - logsumexp(prefix)))
    influence = (np.exp(full - (logsumexp(full) - np.log(len(full))))
                 - np.exp(prefix - (logsumexp(prefix) - np.log(len(prefix)))))
    mcse = _batch_mcse(influence.reshape(samples.chart.shape[:2]), batch_size)
    functional_diagnostics = _functional_diagnostics(influence.reshape(samples.chart.shape[:2]))
    return {"log_density": log_density, "log_density_batch_mcse": mcse,
            "raw_weight_ess": raw_weight_ess, "raw_weight_ess_fraction": raw_weight_ess / len(weights),
            "maximum_prefix_weight": largest, "prefix_weights": weights,
            "diagnostics": diagnostics, "log_density_influence_diagnostics": functional_diagnostics,
            "component_tail_logs": tail,
            "conditional_chart_mean": weights @ samples.chart.reshape(len(weights), -1),
            "conditional_C_scale_mean": weights @ samples.scales.reshape(len(weights), -1),
            "numerical_program": "finite_empirical_C_beta_measure_prefix_importance",
            "finite_predictive_mixture_proper": True,
            "continuous_Gamma_target_accuracy_certified": False,
            "uncertainty_scope": "paired_ratio_delta_method_batch_MCSE_estimate"}


def targeted_prefix_predict(samples, prediction, *, batch_size, max_rhat, min_bulk_ess, min_tail_ess):
    """Caller separately ran C+the SAME test prefix Gibbs; no implicit fallback.

Only proper conditional components are averaged. Hyperparameters are still
fixed; prefix inference updates beta/calibration/test lambdas, not a new fit.
"""
    model = samples.model
    _predictive_policy(max_rhat, min_bulk_ess, min_tail_ess)
    _targeted_prediction_binding(model, prediction)
    diagnostics = posterior_diagnostics(samples)
    if (diagnostics["max_rank_folded_split_rhat"] > max_rhat
            or diagnostics["min_bulk_ess"] < min_bulk_ess
            or diagnostics["min_tail_ess"] < min_tail_ess):
        raise PredictiveSamplingError("caller-declared C+prefix MCMC diagnostics were not met")
    _, _, tail = conditional_tail_components(prediction, samples.chart)
    log_density = float(logsumexp(tail) - np.log(len(tail)))
    influence = np.exp(tail - log_density) - 1
    return {"log_density": log_density,
            "log_density_batch_mcse": _batch_mcse(influence.reshape(samples.chart.shape[:2]), batch_size),
            "diagnostics": diagnostics,
            "log_density_influence_diagnostics": _functional_diagnostics(influence.reshape(samples.chart.shape[:2])),
            "component_tail_logs": tail,
            "conditional_chart_mean": samples.chart.mean(axis=(0, 1)),
            "conditional_scale_mean": samples.scales.mean(axis=(0, 1)),
            "numerical_program": "finite_empirical_C_plus_prefix_beta_measure",
            "finite_predictive_mixture_proper": True,
            "continuous_Gamma_target_accuracy_certified": False,
            "uncertainty_scope": "conditional_mean_delta_method_batch_MCSE_estimate"}


def targeted_conditional_beta_predict(samples, prediction, *, batch_size, max_rhat, min_bulk_ess, min_tail_ess):
    """Explicit alternate estimator: beta integrated under SAME targeted lambdas.

    These scales must come from C+the actual prefix joint posterior, rather than
    a C-only fit. Equal finite weights are fixed before tail evaluation. This
    estimator and the lambda-collapsed t estimator integrate different variables;
    neither universally has lower variance. Finite mixing/MCSE limits remain.
    """
    model = samples.model
    _predictive_policy(max_rhat, min_bulk_ess, min_tail_ess)
    _targeted_prediction_binding(model, prediction)
    diagnostics = posterior_diagnostics(samples)
    if (diagnostics["max_rank_folded_split_rhat"] > max_rhat
            or diagnostics["min_bulk_ess"] < min_bulk_ess
            or diagnostics["min_tail_ess"] < min_tail_ess):
        raise PredictiveSamplingError("caller-declared C+prefix MCMC diagnostics were not met")
    scales = samples.scales.reshape(-1, len(model.blocks))
    logs = _conditional_beta_components(model, prediction, scales)
    log_density = float(logsumexp(logs)-np.log(len(logs)))
    influence = (np.exp(logs-log_density)-1).reshape(samples.chart.shape[:2])
    concentrations = np.exp(logs-logsumexp(logs))
    return {"log_density": log_density,
            "log_density_batch_mcse": _batch_mcse(influence, batch_size),
            "diagnostics": diagnostics,
            "log_density_influence_diagnostics": _functional_diagnostics(influence),
            "point_likelihood_Kish_concentration": float(1/np.sum(concentrations**2)),
            "point_likelihood_maximum_concentration": float(concentrations.max()),
            "component_tail_logs": logs,
            "actual_retained_joint_scale_draws": len(logs),
            "finite_predictive_mixture_proper": True,
            "continuous_Gamma_target_accuracy_certified": False,
            "numerical_program": "conditional_beta_Gaussian_integration_under_targeted_joint_Gamma_draws",
            "uncertainty_scope": "conditional_mean_delta_method_batch_MCSE_estimate",
            "universal_variance_improvement_over_lambda_collapsed_t_not_claimed": True}
