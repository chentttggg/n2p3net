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
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import rankdata

from research.joint_response_scale import DecisionBlock, FixedHyperparameters, GlobalGaussianPrior
from transfer.noise_model import finite_real_array, frozen_array


class SamplingResolutionError(ValueError):
    """The declared numerical sampling calculation is not representable."""


class PredictiveSamplingError(ValueError):
    """Caller-declared Monte Carlo diagnostic thresholds were not met."""


@dataclass(frozen=True)
class PreparedDecision:
    decision_id: str
    image: np.ndarray
    residual: np.ndarray
    noise_logdet: float


@dataclass(frozen=True)
class PreparedModel:
    prior: GlobalGaussianPrior
    blocks: tuple[DecisionBlock, ...]
    prepared: tuple[PreparedDecision, ...]
    hyperparameters: FixedHyperparameters
    grams: np.ndarray
    linear: np.ndarray
    stacked_images: np.ndarray | None
    stacked_residuals: np.ndarray | None


@dataclass(frozen=True)
class PreparedPrediction:
    """Actual test whitening compiled once, tied to one physical prior/noise chart."""

    prior: GlobalGaussianPrior
    hyperparameters: FixedHyperparameters
    test: DecisionBlock
    conditioning_rows: int
    full: PreparedDecision
    prefix_logdet: float


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


def _prepared_decision(prior, block, hyperparameters):
    scale = np.sqrt(hyperparameters.relative_noise_shape)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        image = solve_triangular(block.noise_root, block.design @ prior.root, lower=True) / scale
        residual = solve_triangular(block.noise_root, block.samples - block.design @ prior.mean,
                                    lower=True) / scale
        logdet = float(2 * np.log(np.diag(block.noise_root)).sum()
                       + len(block.samples) * np.log(hyperparameters.relative_noise_shape))
    if not all(np.isfinite(v).all() for v in (image, residual, logdet)):
        raise SamplingResolutionError("physical whitened model is not representable")
    return PreparedDecision(block.decision_id, frozen_array(image), frozen_array(residual), logdet)


def prepare_model(prior, blocks, hyperparameters):
    """Compile actual C or C+one declared inference prefix exactly once.

All physical coordinates, source/plugin authority, read time and cross-decision
noise independence remain caller-owned; canonical classes enforce declarations.
"""
    blocks = tuple(blocks)
    if (not isinstance(prior, GlobalGaussianPrior) or not isinstance(hyperparameters, FixedHyperparameters)
            or any(not isinstance(v, DecisionBlock) for v in blocks)):
        raise ValueError("canonical prior, hyperparameters and decision blocks are required")
    if len({v.decision_id for v in blocks}) != len(blocks):
        raise ValueError("each complete decision must appear once in the supplied observation set")
    if (any(v.role != "target_calibration" for v in blocks[:-1])
            or blocks and blocks[-1].role not in {"target_calibration", "read_only_inference"}):
        raise ValueError("only complete C followed by at most one actual inference prefix is supported")
    p = len(prior.mean)
    prepared = []
    for block in blocks:
        if block.design.shape[1] != p:
            raise ValueError("every block must use the same actual global beta chart")
        prepared.append(_prepared_decision(prior, block, hyperparameters))
    with np.errstate(over="ignore", invalid="ignore"):
        grams = np.asarray([v.image.T @ v.image for v in prepared]).reshape(len(blocks), p, p)
        linear = np.asarray([v.image.T @ v.residual for v in prepared]).reshape(len(blocks), p)
    if not np.isfinite(grams).all() or not np.isfinite(linear).all():
        raise SamplingResolutionError("conditional Gaussian statistics are not representable")
    same_grid = bool(prepared) and len({len(v.residual) for v in prepared}) == 1
    images = np.stack([v.image for v in prepared]) if same_grid else None
    residuals = np.stack([v.residual for v in prepared]) if same_grid else None
    return PreparedModel(prior, blocks, tuple(prepared), hyperparameters, frozen_array(grams),
                         frozen_array(linear), frozen_array(images) if images is not None else None,
                         frozen_array(residuals) if residuals is not None else None)


def prepare_prediction(model, test, *, conditioning_rows):
    """Prepare the actual full test once; its leading rows are the SAME prefix."""
    if not isinstance(test, DecisionBlock) or test.role != "read_only_inference":
        raise ValueError("a canonical declared read-only test decision is required")
    if test.design.shape[1] != len(model.prior.mean):
        raise ValueError("test must use the same global physical beta chart")
    cut = conditioning_rows
    if type(cut) is not int or not 0 < cut < len(test.samples):
        raise ValueError("an actual nonempty test prefix and tail are required")
    if any(v.decision_id == test.decision_id and v.role == "target_calibration" for v in model.blocks):
        raise ValueError("test cannot duplicate a complete C decision or its Gamma draw")
    full = _prepared_decision(model.prior, test, model.hyperparameters)
    prefix_logdet = float(2 * np.log(np.diag(test.noise_root)[:cut]).sum()
                         + cut * np.log(model.hyperparameters.relative_noise_shape))
    return PreparedPrediction(model.prior, model.hyperparameters, test, cut, full, prefix_logdet)


def _prediction_binding(model, prediction):
    if (not isinstance(prediction, PreparedPrediction)
            or prediction.prior.program != model.prior.program
            or not np.array_equal(prediction.prior.mean, model.prior.mean)
            or not np.array_equal(prediction.prior.root, model.prior.root)
            or prediction.hyperparameters != model.hyperparameters):
        raise ValueError("prediction and sampled C must share the same physical prior and noise program")


def conditional_beta_chart(model, lambdas):
    """u|lambda,y with beta=m0+Lbeta*u and u-prior exactly N(0,I)."""
    values = finite_real_array(lambdas, name="conditional decision precisions")
    if values.shape != (len(model.blocks),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("one finite positive conditional scale per supplied decision is required")
    p = len(model.prior.mean)
    with np.errstate(over="ignore", invalid="ignore"):
        precision = np.eye(p) + np.einsum("j,jab->ab", values, model.grams)
        rhs = values @ model.linear
    if not np.isfinite(precision).all() or not np.isfinite(rhs).all():
        raise SamplingResolutionError("conditional beta precision/RHS is not representable")
    try:
        root = np.linalg.cholesky(precision)
        mean = cho_solve((root, True), rhs)
    except np.linalg.LinAlgError as error:
        raise SamplingResolutionError("proper conditional beta precision is unresolved") from error
    if not np.isfinite(root).all() or not np.isfinite(mean).all():
        raise SamplingResolutionError("conditional beta chart is not representable")
    return mean, root


def conditional_scale_parameters(model, chart):
    """lambda_j|u,y are independent Gamma((nu+n_j)/2,(nu+q_j(u))/2)."""
    u = finite_real_array(chart, name="global prior chart")
    if u.shape != (len(model.prior.mean),) or not np.isfinite(u).all():
        raise ValueError("the same finite global prior chart is required")
    with np.errstate(over="ignore", invalid="ignore"):
        if model.stacked_images is not None:
            residuals = model.stacked_residuals - np.einsum("jnp,p->jn", model.stacked_images, u)
            energies = np.einsum("jn,jn->j", residuals, residuals)
        else:
            energies = np.asarray([np.sum((v.residual - v.image @ u) ** 2) for v in model.prepared])
    nu = model.hyperparameters.degrees_of_freedom
    shapes = np.asarray([(nu + len(v.residual)) / 2 for v in model.prepared])
    rates = (nu + energies) / 2
    if not np.isfinite(shapes).all() or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise SamplingResolutionError("conditional Gamma operands are not representable")
    return shapes, rates


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
    gaussian_conditional = conditional_beta_chart(model, np.ones(j)) if gaussian else None
    started = perf_counter()
    for chain, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        u, lambdas = starts[chain].copy(), np.ones(j)
        if not gaussian:
            shape, rate = conditional_scale_parameters(model, u)
            lambdas = rng.gamma(shape, 1 / rate)
        for iteration in range(burnin + draws):
            mean, root = gaussian_conditional if gaussian else conditional_beta_chart(model, lambdas)
            u = mean + solve_triangular(root.T, rng.normal(size=p), lower=False)
            if not gaussian:
                shape, rate = conditional_scale_parameters(model, u)
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


def _t_logpdf(residuals, nu, logdet):
    n = residuals.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = np.einsum("si,si->s", residuals, residuals)
    if np.isposinf(nu):
        result = -.5 * (n * np.log(2 * np.pi) + logdet + quadratic)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            normalizer = gammaln((nu + n) / 2) - gammaln(nu / 2)
            result = (normalizer - .5 * (n * (np.log(nu) + np.log(np.pi)) + logdet)
                      - (nu + n) / 2 * np.log1p(quadratic / nu))
    if not np.isfinite(result).all():
        raise SamplingResolutionError("normalized finite-df predictive density is unresolved")
    return result


def conditional_tail_components(prediction, chart):
    """Each component is normalized p(tail|u,prefix), using ONE actual test lambda.

Full and prefix roots are the same physical lower factor/leading block. The
test mean/design shares the global beta chart; lambda is analytically integrated
after updating it with prefix evidence. No tail is used to weight components.
"""
    cut = prediction.conditioning_rows
    prepared = prediction.full
    charts = finite_real_array(chart, name="sampled global charts")
    if charts.ndim not in (2, 3) or charts.shape[-1] != len(prediction.prior.mean):
        raise ValueError("prediction components require the actual sampled global chart width")
    u = charts.reshape(-1, len(prediction.prior.mean))
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = prepared.residual - u @ prepared.image.T
    nu = prediction.hyperparameters.degrees_of_freedom
    full = _t_logpdf(residuals, nu, prepared.noise_logdet)
    prefix = _t_logpdf(residuals[:, :cut], nu, prediction.prefix_logdet)
    return full, prefix, full - prefix


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
    _prediction_binding(model, prediction)
    test, cut = prediction.test, prediction.conditioning_rows
    if not model.blocks:
        raise ValueError("targeted sampling requires C plus the actual test prefix")
    last = model.blocks[-1]
    if (last.decision_id != test.decision_id or last.role != "read_only_inference"
            or len(last.samples) != cut
            or not np.array_equal(last.samples, test.samples[:cut])
            or not np.array_equal(last.design, test.design[:cut])
            or not np.array_equal(last.noise_root, test.noise_root[:cut, :cut])
            or any(v.role != "target_calibration" for v in model.blocks[:-1])):
        raise ValueError("the actual test prefix must have entered the global sampler exactly once")
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
