"""Finite whole-selection history reference with one shared ERP posterior.

The caller supplies authorized native samples, a complete public event history,
and a joint label prior frozen from a declared source/C/protocol. This module
does not fit parameters, inspect truth, authenticate permission, truncate label
states, invent outside events, or change a production dictionary/scorer.

Default output is the fixed-current-prior conditional-tail scoring program.
Flat nuisance uses a common sigma-finite coefficient measure, shared between
joint and prefix; its marginals are not normalized full generative densities.
Prefix must identify the common nuisance columns. This small-matrix reference
enumerates the complete supplied finite state space exactly; it is not a claim
that arbitrary continuous history or latent latency can be enumerated cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from transfer.noise_model import (
    finite_real_array,
    freeze_metadata,
    frozen_array,
    require_identifiers,
)


class HistoryNuisanceResolutionError(ValueError):
    """The actual prefix cannot identify the declared common diffuse measure."""


class HistoryMixtureResolutionError(HistoryNuisanceResolutionError):
    """Component estimates do not resolve the actual complete state mixture."""


def _matrix(value, name):
    result = finite_real_array(value, name=name)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a real finite matrix")
    return result


def _positive_lower(value, name):
    result = _matrix(value, name)
    if (not len(result) or result.shape != (len(result), len(result))
            or np.any(np.triu(result, 1)) or np.any(np.diag(result) <= 0)):
        raise ValueError(f"{name} must be a full-rank positive lower covariance factor")
    return frozen_array(result)


@dataclass(frozen=True)
class ResponsePosterior:
    """Frozen MN response coefficients in the actual common/increment chart.

    Phi is dimensionless on t=n/fs and [0,L/fs). M has volts, R is the row
    precision factor, and the column covariance factor has volts. Data/source
    authority and the Gaussian/fitted-noise assumption belong to the caller.
    """

    basis: np.ndarray
    mean: np.ndarray
    precision_root: np.ndarray
    spatial_root: np.ndarray
    sample_rate_hz: float

    def __post_init__(self):
        basis, mean, root = (_matrix(v, n) for v, n in (
            (self.basis, "actual response basis"), (self.mean, "frozen response mean"),
            (self.precision_root, "frozen response row precision")))
        spatial = _positive_lower(self.spatial_root, "frozen innovation column covariance")
        width = 2*basis.shape[1]
        if (min(basis.shape) == 0 or mean.shape != (width, len(spatial))
                or root.shape != (width, width) or np.any(np.tril(root, -1)) or np.any(np.diag(root) == 0)):
            raise ValueError("response posterior must use the actual two-block kernel chart and measured columns")
        rate = np.asarray(self.sample_rate_hz)
        if (rate.ndim != 0 or rate.dtype.kind not in "fiu" or not np.isfinite(rate) or rate <= 0):
            raise ValueError("native response sample rate must be positive and finite")
        for key, value in (("basis", basis), ("mean", mean), ("precision_root", root), ("spatial_root", spatial)):
            object.__setattr__(self, key, frozen_array(value))
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))


@dataclass(frozen=True)
class PublicSelection:
    """Public clock/membership only; no selected target or truth is accepted."""

    selection_id: str
    candidate_ids: tuple[str, ...]
    onsets: np.ndarray
    membership: np.ndarray

    def __post_init__(self):
        ids = tuple(self.candidate_ids)
        require_identifiers(ids, "complete public candidate vocabulary")
        if len(ids) < 2 or not isinstance(self.selection_id, str) or not self.selection_id.strip():
            raise ValueError("public selection requires a name and at least two complete candidates")
        onsets = np.asarray(self.onsets)
        member = np.asarray(self.membership)
        if (onsets.ndim != 1 or onsets.dtype.kind not in "iu" or not len(onsets)
                or np.any(onsets > np.iinfo(np.int64).max) or member.dtype.kind not in "bfiu"
                or np.any(onsets < 0) or np.any(np.diff(onsets.astype(np.int64)) <= 0)
                or member.shape != (len(onsets), len(ids)) or not np.isin(member, [0, 1]).all()
                or np.any(member.sum(axis=1) == 0) or np.any(member.sum(axis=1) == len(ids))
                or np.any(member.sum(axis=0) == 0)):
            raise ValueError("selection events need distinct native times and the full proper public membership masks")
        integer = np.ascontiguousarray(onsets, dtype=np.int64)
        object.__setattr__(self, "onsets", np.frombuffer(integer.tobytes(), dtype=np.int64))
        object.__setattr__(self, "membership", frozen_array(member))
        object.__setattr__(self, "candidate_ids", ids)


@dataclass(frozen=True)
class NeighborEventPrefix:
    """All already occurred events of one neighbouring selection.

    Its complete target vocabulary stays present even if an as-yet-unflashed
    candidate has zero design, or several targets induce identical designs.
    Completeness of the event prefix is proved by a source owner, not this type.
    The current selection continues to use the strict PublicSelection gate.
    """

    selection_id: str
    candidate_ids: tuple[str, ...]
    onsets: np.ndarray
    membership: np.ndarray

    def __post_init__(self):
        ids = tuple(self.candidate_ids)
        require_identifiers(ids, "complete neighbouring target vocabulary")
        onsets, member = np.asarray(self.onsets), np.asarray(self.membership)
        if (len(ids) < 2 or not isinstance(self.selection_id, str) or not self.selection_id.strip()
                or onsets.ndim != 1 or onsets.dtype.kind not in "iu" or not len(onsets)
                or np.any(onsets < 0) or np.any(onsets > np.iinfo(np.int64).max)
                or np.any(np.diff(onsets.astype(np.int64)) <= 0)
                or member.shape != (len(onsets), len(ids)) or member.dtype.kind not in "bfiu"
                or not np.isin(member, [0, 1]).all()
                or np.any(member.sum(axis=1) == 0) or np.any(member.sum(axis=1) == len(ids))):
            raise ValueError("neighbour prefix requires actual ordered events and full proper row masks")
        object.__setattr__(self, "candidate_ids", ids)
        values = np.ascontiguousarray(onsets, dtype=np.int64)
        object.__setattr__(self, "onsets", np.frombuffer(values.tobytes(), dtype=np.int64))
        object.__setattr__(self, "membership", frozen_array(member))


@dataclass(frozen=True)
class PublicResponseHistory:
    """Complete supplied finite selections and explicit joint target prior.

    Prior axes follow selections and each actual candidate order. Zero-mass
    combinations are explicit; each requested current candidate keeps positive
    mass. This declaration is not evidence that the raw world's history is
    complete or that its source/C prior was lawfully estimated.
    """

    selections: tuple[PublicSelection | NeighborEventPrefix, ...]
    current_selection_id: str
    joint_label_prior: np.ndarray

    def __post_init__(self):
        selections = tuple(self.selections)
        if not selections or any(not isinstance(s, (PublicSelection, NeighborEventPrefix)) for s in selections):
            raise ValueError("history needs owned public selections")
        names = tuple(s.selection_id for s in selections)
        require_identifiers(names, "history selection identities")
        if self.current_selection_id not in names:
            raise ValueError("requested current selection is missing from the supplied history")
        if not isinstance(selections[names.index(self.current_selection_id)], PublicSelection):
            raise ValueError("current selection requires complete candidate evidence; a neighbour prefix is insufficient")
        onsets = np.concatenate([s.onsets for s in selections])
        if len(np.unique(onsets)) != len(onsets):
            raise ValueError("a public flash may not be counted in two supplied selections")
        prior = finite_real_array(self.joint_label_prior, name="frozen full joint label prior")
        shape = tuple(len(s.candidate_ids) for s in selections)
        if prior.shape != shape or np.any(prior < 0) or not np.isfinite(prior.sum()) or not np.isclose(prior.sum(), 1., rtol=0, atol=64*np.finfo(float).eps):
            raise ValueError("joint label prior must cover the complete Cartesian state table and sum to one")
        current_axis = names.index(self.current_selection_id)
        other_axes = tuple(i for i in range(len(selections)) if i != current_axis)
        marginal = prior.sum(axis=other_axes) if other_axes else prior
        if np.any(marginal <= 0):
            raise ValueError("every requested current candidate must retain positive prior mass")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "joint_label_prior", frozen_array(prior))


@dataclass(frozen=True)
class ResponseObservation:
    """One authorized native block and its frozen unit-innovation noise factor."""

    samples: np.ndarray
    origin_sample: int
    temporal_root: np.ndarray
    nuisance: np.ndarray
    conditioning_samples: int
    current_right_censored: bool = False

    def __post_init__(self):
        samples, nuisance = _matrix(self.samples, "actual observed volts"), _matrix(self.nuisance, "common nuisance coordinates")
        temporal = _positive_lower(self.temporal_root, "actual native temporal covariance factor")
        if (not len(samples) or samples.shape[1] < 1 or len(temporal) != len(samples)
                or nuisance.shape[0] != len(samples) or type(self.origin_sample) is not int or self.origin_sample < 0
                or type(self.conditioning_samples) is not int or not 1 <= self.conditioning_samples < len(samples)):
            raise ValueError("actual block, native origin, common nuisance and nonempty prefix/tail must align")
        if type(self.current_right_censored) is not bool:
            raise ValueError("current response censoring must be explicitly declared")
        for key, value in (("samples", samples), ("temporal_root", temporal), ("nuisance", nuisance)):
            object.__setattr__(self, key, frozen_array(value))


@dataclass(frozen=True)
class StructuredResponseObservation:
    """Native production operands; the stored noise map is evaluated structurally.

    No dense temporal covariance/root is materialized. Its scientific caller
    must supply the same full/prefix marginal evaluator from the actual noise.
    """

    samples: np.ndarray
    origin_sample: int
    nuisance: np.ndarray
    conditioning_samples: int
    current_right_censored: bool

    def __post_init__(self):
        samples, nuisance = _matrix(self.samples, "actual native history samples"), _matrix(
            self.nuisance, "same full/prefix nuisance chart")
        if (not len(samples) or not samples.shape[1] or nuisance.shape[0] != len(samples)
                or type(self.origin_sample) is not int or self.origin_sample < 0
                or type(self.conditioning_samples) is not int
                or not 1 <= self.conditioning_samples < len(samples)
                or type(self.current_right_censored) is not bool):
            raise ValueError("structured history requires actual prefix/tail and explicit censoring")
        object.__setattr__(self, "samples", frozen_array(samples))
        object.__setattr__(self, "nuisance", frozen_array(nuisance))


@dataclass(frozen=True)
class ConditionalHistoryEvidence:
    candidate_ids: tuple[str, ...]
    state_indices: tuple[tuple[int, ...], ...]
    conditional_log_likelihood: np.ndarray
    log_candidate_probability: np.ndarray
    state_joint_log_marginal: np.ndarray
    state_prefix_log_marginal: np.ndarray
    state_conditional_log_density: np.ndarray
    state_prefix_probability_given_current: np.ndarray
    current_prior: np.ndarray
    measure_kind: str
    evaluated_states: int
    identical_design_state_groups: tuple[tuple[int, ...], ...]
    state_resolution: tuple[dict | None, ...] = ()
    mixture_resolution: dict | None = None
    program: str = field(default="finite_history_conditional_tail_fixed_current_prior", init=False)

    @property
    def probabilities(self):
        return frozen_array(np.exp(self.log_candidate_probability))


def _history_mixture_resolution(history, observation, states, prior, prefix, conditional,
                                resolutions, current_prior, evidence, *, log_prior=None, log_prior_errors=None,
                                tail_scalar_rows=None, whole_observed=False):
    """Propagate central estimates through the actual stable logsumexp program.

    logsumexp is 1-Lipschitz in the infinity norm. Prefix posterior weights
    therefore contribute at most twice the maximum prefix error; the state
    conditional potentials contribute their maximum conditional error. These
    source estimates and the stated binary64 arithmetic estimate are kept
    separate. This does not turn the central estimates into interval proofs.
    """
    if not any(resolution is not None for resolution in resolutions):
        return None
    axis = tuple(s.selection_id for s in history.selections).index(history.current_selection_id)
    current = history.selections[axis]
    tail_rows = ((len(observation.samples)-observation.conditioning_samples)*observation.samples.shape[1]
                 if tail_scalar_rows is None else tail_scalar_rows)
    budget = float(tail_rows*np.sqrt(np.finfo(float).eps))
    records = []
    for q in range(len(current.candidate_ids)):
        indices = [i for i, state in enumerate(states) if state[axis] == q
                   and (prior[i] > 0 if log_prior is None else np.isfinite(log_prior[i]))]
        selected = [resolutions[i] for i in indices]
        if any(resolution is None or not {"full", "prefix", "conditional", "source_sha256"} <= resolution.keys()
               for resolution in selected):
            raise HistoryMixtureResolutionError("complete state mixture lacks its central full/prefix certificates")
        if any(resolution["conditional"]["actual_tail_scalar_rows"] != tail_rows for resolution in selected):
            raise HistoryMixtureResolutionError("mixture certificates disagree with actual native tail rows")
        full_error = max(float(r["full"]["absolute_log_density_error_estimate"]) for r in selected)
        prefix_error = max(float(r["prefix"]["absolute_log_density_error_estimate"]) for r in selected)
        conditional_error = max(float(r["conditional"]["absolute_log_density_error_estimate"]) for r in selected)
        if not np.isfinite([full_error, prefix_error, conditional_error]).all() or min(
                full_error, prefix_error, conditional_error) < 0:
            raise HistoryMixtureResolutionError("mixture source error estimates are invalid")
        p, c = prefix[indices], conditional[indices]
        logs = np.log(prior[indices]) if log_prior is None else log_prior[indices]
        # Actual leading-potential centering, two logsumexp reductions, log
        # prior and scalar additions. gamma uses the full surviving state
        # count, rather than assuming pairwise summation or a small mixture.
        operations = len(indices)+16
        product = operations*np.finfo(float).eps
        if product >= 1:
            raise HistoryMixtureResolutionError("complete mixture has no finite arithmetic error estimate")
        gamma = product/(1-product)
        magnitude = (4*float(np.max(np.abs(p)))+4*float(np.max(np.abs(c)))
                     +4*float(np.max(np.abs(logs)))+2*np.log(len(indices))+1.)
        arithmetic_error = float(gamma*magnitude)
        c_error = 0. if log_prior_errors is None else float(np.max(log_prior_errors[indices]))
        if not np.isfinite(c_error) or c_error < 0:
            raise HistoryMixtureResolutionError("finite C log-mass error estimates are invalid")
        source_error = ((c_error+prefix_error+conditional_error) if whole_observed
                        else 2*c_error+2*prefix_error+conditional_error)
        error = source_error+arithmetic_error
        if not np.isfinite(error) or error > budget:
            raise HistoryMixtureResolutionError(
                f"complete history mixture error estimate {error:.6g} exceeds {tail_rows}-row nats budget {budget:.6g}")
        records.append({"candidate_id": current.candidate_ids[q], "positive_mass_states": len(indices),
            "maximum_full_potential_error_estimate": full_error,
            "maximum_prefix_potential_error_estimate": prefix_error,
            "maximum_conditional_potential_error_estimate": conditional_error,
            **({"maximum_C_logmass_error_estimate": c_error} if log_prior_errors is not None else {}),
            "mixture_source_error_estimate": source_error,
            "mixture_arithmetic_error_estimate": arithmetic_error,
            "absolute_log_density_error_estimate": error,
            "absolute_log_density_error_budget": budget})
    maximum = max(record["absolute_log_density_error_estimate"] for record in records)
    probabilities_operations = len(records)+8
    product = probabilities_operations*np.finfo(float).eps
    gamma = product/(1-product)
    normalization_error = float(gamma*(4*float(np.max(np.abs(evidence)))
        +4*float(np.max(np.abs(np.log(current_prior))))+np.log(len(records))+1.))
    log_probability_error = 2*maximum+normalization_error
    if not np.isfinite(log_probability_error) or log_probability_error > budget:
        raise HistoryMixtureResolutionError(
            f"history normalized log-mass error estimate {log_probability_error:.6g} exceeds nats budget {budget:.6g}")
    return freeze_metadata({"method": ("whole_observed_q_common_prefix_anchor_logsumexp_Lipschitz_nats_estimate"
        if whole_observed else "prefix_posterior_weights_conditional_logsumexp_Lipschitz_nats_estimate"),
        "component_source_sha256": sorted({r["source_sha256"] for r in resolutions if r is not None}),
        "candidates": records, "actual_tail_scalar_rows": tail_rows,
        "absolute_log_density_error_budget": budget,
        "normalized_log_probability_error_estimate": log_probability_error,
        "normalization_arithmetic_error_estimate": normalization_error,
        "normalization_program": "max_centered_complete_candidate_log_mass",
        "arithmetic_model": "binary64_scalar_and_full_count_gamma_estimate_not_interval_certificate",
        "all_input_certificate": False})


def response_history_design(posterior, history, observation, state):
    """One shared beta chart; include actual visible tails of public past events."""
    n, k = len(observation.samples), posterior.basis.shape[1]
    if len(state) != len(history.selections) or any(type(i) is not int or not 0 <= i < len(s.candidate_ids)
                                                  for s, i in zip(history.selections, state, strict=True)):
        raise ValueError("one whole-selection hypothesis from each actual public vocabulary is required")
    design = np.zeros((n, 2*k))
    length = len(posterior.basis)
    for selection, candidate in zip(history.selections, state, strict=True):
        for event, onset in enumerate(selection.onsets):
            offset = int(onset)-observation.origin_sample
            first, stop = max(0, offset), min(n, offset+length)
            if first >= stop:
                continue
            response = posterior.basis[first-offset:stop-offset]
            design[first:stop, :k] += response
            design[first:stop, k:] += selection.membership[event, candidate]*response
    return frozen_array(design)


def _log_marginal(samples, design, noise_root, nuisance, posterior):
    """Proper Gaussian or common-flat-measure marginal; never a fake density."""
    n, channels = samples.shape
    width = nuisance.shape[1]
    image = solve_triangular(posterior.precision_root.T, design.T, lower=True).T
    if not np.isfinite(image).all():
        raise ValueError("shared response covariance image is not numerically representable")
    # A covariance factor sum, not an inverse Gram or duplicated neighbour Gamma.
    covariance_root = np.linalg.qr(np.column_stack((noise_root, image)).T, mode="r").T
    if not np.isfinite(covariance_root).all() or np.any(np.diag(covariance_root) == 0):
        raise ValueError("history component covariance is not numerically representable")
    residual = solve_triangular(covariance_root, samples-design@posterior.mean, lower=True)
    logdet_nuisance = 0.
    if width:
        white = solve_triangular(covariance_root, nuisance, lower=True)
        scales = np.max(np.abs(white), axis=0)
        if n < width or np.any(scales == 0):
            raise HistoryNuisanceResolutionError("prefix cannot identify the common nuisance measure")
        normalized = white/scales
        singular = np.linalg.svd(normalized, compute_uv=False)
        if len(singular) < width or singular[-1] <= np.finfo(float).eps*max(normalized.shape)*singular[0]:
            raise HistoryNuisanceResolutionError("prefix nuisance coordinates are numerically unresolved")
        orthogonal, root = np.linalg.qr(normalized, mode="reduced")
        residual -= orthogonal@(orthogonal.T@residual)
        logdet_nuisance = 2*float(np.log(np.abs(np.diag(root))).sum()+np.log(scales).sum())
    spatial_residual = solve_triangular(posterior.spatial_root, residual.T, lower=True)
    logdet_time = 2*float(np.log(np.abs(np.diag(covariance_root))).sum())
    logdet_spatial = 2*float(np.log(np.diag(posterior.spatial_root)).sum())
    result = -.5*(float(np.sum(spatial_residual**2)) + channels*(logdet_time+logdet_nuisance)
                 + (n-width)*(channels*np.log(2*np.pi)+logdet_spatial))
    if not np.isfinite(result):
        raise ValueError("history component marginal is not finitely representable")
    return float(result)


def conditional_history_evidence(posterior, history, observation, *, full_state_prior=None,
                                 design_builder=None, marginal_evaluator=None):
    """Exact finite-state conditional-tail mixture with the frozen current prior.

    Both integrals use the same nuisance columns/coordinate volume. The prefix
    updates unknown-neighbour probabilities and the shared response posterior.
    The output does not claim a full-data Bayes current-candidate posterior.
    """
    if (not isinstance(posterior, ResponsePosterior) or not isinstance(history, PublicResponseHistory)
            or not isinstance(observation, (ResponseObservation, StructuredResponseObservation))
            or observation.samples.shape[1] != posterior.mean.shape[1]):
        raise ValueError("history evidence requires aligned owned posterior/history/observations")
    if isinstance(observation, StructuredResponseObservation) and marginal_evaluator is None:
        raise ValueError("structured observations require their actual full/prefix marginal owner")
    axis = tuple(s.selection_id for s in history.selections).index(history.current_selection_id)
    current = history.selections[axis]
    end = observation.origin_sample+len(observation.samples)
    if (current.onsets[0] < observation.origin_sample or current.onsets[-1] >= end
            or not observation.current_right_censored and int(current.onsets[-1])+len(posterior.basis) > end):
        raise ValueError("requested current selection needs its complete observed native response tails")
    table = history.joint_label_prior if full_state_prior is None else finite_real_array(
        full_state_prior, name="complete frozen joint target/response-state prior")
    label_shape = tuple(len(s.candidate_ids) for s in history.selections)
    if (table.shape[:len(label_shape)] != label_shape or np.any(table < 0)
            or not np.isclose(table.sum(), 1., rtol=0, atol=64*np.finfo(float).eps)):
        raise ValueError("full finite state mass must preserve every target axis and sum to one")
    if full_state_prior is not None:
        extra = tuple(range(len(label_shape), table.ndim))
        label_mass = table.sum(axis=extra) if extra else table
        if not np.array_equal(label_mass, history.joint_label_prior):
            raise ValueError("joint response-state mass differs from the frozen label marginal")
        if design_builder is None:
            raise ValueError("complete response-state mass requires its actual owned design builder")
    states = tuple(product(*(range(size) for size in table.shape)))
    prior = table.reshape(-1)
    active = prior > 0
    logprior = np.full(len(states), -np.inf)
    logprior[active] = np.log(prior[active])
    joint, prefix, conditional = (np.full(len(states), -np.inf) for _ in range(3))
    resolutions = [None]*len(states)
    u = observation.conditioning_samples
    design_groups = {}
    for i, state in enumerate(states):
        if not active[i]:
            continue
        design = (response_history_design(posterior, history, observation, state) if design_builder is None
                  else design_builder(state))
        design_groups.setdefault((state[axis], design.tobytes()), []).append(i)
        if marginal_evaluator is None:
            joint[i] = _log_marginal(observation.samples, design, observation.temporal_root, observation.nuisance, posterior)
            prefix[i] = _log_marginal(observation.samples[:u], design[:u], observation.temporal_root[:u, :u],
                                      observation.nuisance[:u], posterior)
        else:
            pair = marginal_evaluator(design)
            if hasattr(pair, "conditional_log_density"):
                joint[i], prefix[i], conditional[i] = (
                    pair.joint_log_marginal, pair.prefix_log_marginal, pair.conditional_log_density)
                resolutions[i] = freeze_metadata(pair.resolution)
            else:
                joint[i], prefix[i] = pair
        if marginal_evaluator is None or not hasattr(pair, "conditional_log_density"):
            conditional[i] = joint[i]-prefix[i]
    q_index = np.asarray([state[axis] for state in states])
    evidence, current_prior = [], []
    weights = np.zeros((len(current.candidate_ids), len(states)))
    for q in range(len(current.candidate_ids)):
        selected = q_index == q
        positive = selected & active
        anchor = float(np.max(prefix[positive]))
        centered = logprior[positive]+(prefix[positive]-anchor)
        past = float(logsumexp(centered))
        # This is the same sum(full)/sum(prefix) ratio in a stable chart:
        # prefix updates the state mass, then its conditional density is
        # integrated. These are posterior prefix weights, never prior weights.
        prefix_weights = centered-past
        evidence.append(float(logsumexp(prefix_weights+conditional[positive])))
        current_prior.append(float(prior[selected].sum()))
        weights[q, positive] = np.exp(prefix_weights)
    scores = np.asarray(evidence)+np.log(current_prior)
    if not np.isfinite(scores).all():
        raise ValueError("requested candidate conditional mixture is not representable")
    centered_scores = scores-np.max(scores)
    normalized = centered_scores-logsumexp(centered_scores)
    arrays = [frozen_array(a) for a in (evidence, normalized, joint, prefix, conditional, weights, current_prior)]
    kind = "proper_Gaussian_density_ratio" if observation.nuisance.shape[1] == 0 else "common_sigma_finite_nuisance_marginal_ratio"
    groups = tuple(tuple(indices) for indices in design_groups.values() if len(indices) > 1)
    resolution = _history_mixture_resolution(history, observation, states, prior, prefix, conditional,
        resolutions, np.asarray(current_prior), np.asarray(evidence))
    return ConditionalHistoryEvidence(current.candidate_ids, states, *arrays, kind, int(active.sum()), groups,
        tuple(resolutions), resolution)


@dataclass(frozen=True)
class SparseConditionalHistoryEvidence:
    """Complete reachable states in log mass, conditional on each requested q."""

    candidate_ids: tuple[str, ...]
    state_indices: tuple[tuple[int, ...], ...]
    current_candidate_indices: np.ndarray
    state_log_prior: np.ndarray
    state_c_log_weights: np.ndarray
    state_prefix_log_weights: np.ndarray
    state_full_log_weights: np.ndarray
    state_joint_log_marginal: np.ndarray
    state_prefix_log_marginal: np.ndarray
    state_conditional_log_density: np.ndarray
    conditional_log_likelihood: np.ndarray
    log_prefix_evidence_by_current: np.ndarray
    log_prefix_current_probability: np.ndarray
    log_whole_observed_current_probability: np.ndarray
    log_candidate_probability: np.ndarray
    mixture_resolution: dict | None
    program: str

    @property
    def probabilities(self):
        return frozen_array(np.exp(self.log_candidate_probability))


def finite_conditional_mixture(candidate_ids, states, current_indices, log_prior, pairs, *, tail_scalar_rows,
                               c_logmass_errors=None, scoring_law="fixed_current_prior_conditional_tail"):
    """Reuse the history mixture owner with sparse complete positive log mass.

    C posterior component mass is supplied once. Each pair is an ordinary
    increment density under that actual component, not another C likelihood.
    Prefix and full state weights are normalized separately for each current
    candidate; the scoring program retains its declared uniform current prior.
    Sparse means impossible states are absent, never that tiny states are cut.
    The caller supplies actual state coverage and source/time permissions.
    """
    from types import SimpleNamespace

    candidate_ids, states, pairs = tuple(candidate_ids), tuple(tuple(s) for s in states), tuple(pairs)
    require_identifiers(candidate_ids, "complete current finite mixture vocabulary")
    current = np.asarray(current_indices)
    logs = finite_real_array(log_prior, name="complete reachable state log mass")
    c_errors = (None if c_logmass_errors is None else
                finite_real_array(c_logmass_errors, name="actual C component log-mass error estimates"))
    count = len(states)
    if scoring_law not in {"fixed_current_prior_conditional_tail", "whole_observed_current_q"}:
        raise ValueError("finite continuation scoring law must be explicitly declared")
    if (not count or len(set(states)) != count or len(pairs) != count or logs.shape != (count,)
            or current.shape != (count,) or current.dtype.kind not in "iu"
            or c_errors is not None and (c_errors.shape != (count,) or np.any(c_errors < 0))
            or set(current.tolist()) != set(range(len(candidate_ids)))
            or type(tail_scalar_rows) is not int or tail_scalar_rows < 1):
        raise ValueError("reachable finite states must retain every requested current candidate")
    logs = logs-logsumexp(logs)
    joint, prefix, conditional = (np.asarray([getattr(p, key) for p in pairs], dtype=float) for key in
        ("joint_log_marginal", "prefix_log_marginal", "conditional_log_density"))
    if not np.isfinite(np.r_[joint, prefix, conditional]).all():
        raise HistoryMixtureResolutionError("every positive finite state requires resolved component potentials")
    c_weights, prefix_weights, full_weights = (np.empty(count) for _ in range(3))
    evidence = np.empty(len(candidate_ids))
    masses = np.empty(len(candidate_ids))
    log_prefix = np.empty(len(candidate_ids))
    prefix_score, whole_score = np.empty(len(candidate_ids)), np.empty(len(candidate_ids))
    global_anchor = float(np.max(prefix))
    for q in range(len(candidate_ids)):
        selected = current == q
        masses[q] = logsumexp(logs[selected])
        c_weights[selected] = logs[selected]-masses[q]
        p = prefix[selected]-np.max(prefix[selected])
        terms = logs[selected]+p
        prefix_weights[selected] = terms-logsumexp(terms)
        full_terms = prefix_weights[selected]+conditional[selected]
        full_weights[selected] = full_terms-logsumexp(full_terms)
        evidence[q] = logsumexp(full_terms)
        centered_past = logs[selected]+(prefix[selected]-global_anchor)
        prefix_score[q] = logsumexp(centered_past)
        whole_score[q] = logsumexp(centered_past+conditional[selected])
        log_prefix[q] = prefix_score[q]+global_anchor-masses[q]
    expected = -np.log(len(candidate_ids))
    if not np.allclose(masses, expected, rtol=0, atol=128*np.finfo(float).eps):
        raise ValueError("finite continuation requires its predeclared uniform current candidate prior")
    score = evidence+expected if scoring_law == "fixed_current_prior_conditional_tail" else whole_score.copy()
    score -= np.max(score)
    score -= logsumexp(score)
    centered_prefix = prefix_score-np.max(prefix_score)
    centered_whole = whole_score-np.max(whole_score)
    prefix_probability = centered_prefix-logsumexp(centered_prefix)
    whole_probability = centered_whole-logsumexp(centered_whole)
    # Adapt only the numerical resolution interface, keeping q separate from
    # C component indices and never manufacturing a physical history authority.
    history = SimpleNamespace(current_selection_id="current", selections=(
        SimpleNamespace(selection_id="current", candidate_ids=candidate_ids),))
    resolution = _history_mixture_resolution(history, None,
        tuple((int(q),) for q in current), np.ones(count), prefix, conditional,
        tuple(p.resolution for p in pairs), np.full(len(candidate_ids), 1/len(candidate_ids)),
        evidence if scoring_law == "fixed_current_prior_conditional_tail" else whole_score,
        log_prior=logs, log_prior_errors=c_errors, tail_scalar_rows=tail_scalar_rows,
        whole_observed=scoring_law == "whole_observed_current_q")
    integer = np.ascontiguousarray(current, dtype=np.int64)
    immutable_current = np.frombuffer(integer.tobytes(), dtype=np.int64)
    arrays = [frozen_array(x) for x in (logs, c_weights, prefix_weights, full_weights,
        joint, prefix, conditional, evidence, log_prefix, prefix_probability, whole_probability, score)]
    return SparseConditionalHistoryEvidence(candidate_ids, states, immutable_current, *arrays, resolution, scoring_law)
