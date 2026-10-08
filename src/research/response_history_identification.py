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

from transfer.noise_model import finite_real_array, frozen_array, require_identifiers


class HistoryNuisanceResolutionError(ValueError):
    """The actual prefix cannot identify the declared common diffuse measure."""


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
class PublicResponseHistory:
    """Complete supplied finite selections and explicit joint target prior.

    Prior axes follow selections and each actual candidate order. Zero-mass
    combinations are explicit; each requested current candidate keeps positive
    mass. This declaration is not evidence that the raw world's history is
    complete or that its source/C prior was lawfully estimated.
    """

    selections: tuple[PublicSelection, ...]
    current_selection_id: str
    joint_label_prior: np.ndarray

    def __post_init__(self):
        selections = tuple(self.selections)
        if not selections or any(not isinstance(s, PublicSelection) for s in selections):
            raise ValueError("history needs owned public selections")
        names = tuple(s.selection_id for s in selections)
        require_identifiers(names, "history selection identities")
        if self.current_selection_id not in names:
            raise ValueError("requested current selection is missing from the supplied history")
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

    def __post_init__(self):
        samples, nuisance = _matrix(self.samples, "actual observed volts"), _matrix(self.nuisance, "common nuisance coordinates")
        temporal = _positive_lower(self.temporal_root, "actual native temporal covariance factor")
        if (not len(samples) or samples.shape[1] < 1 or len(temporal) != len(samples)
                or nuisance.shape[0] != len(samples) or type(self.origin_sample) is not int or self.origin_sample < 0
                or type(self.conditioning_samples) is not int or not 1 <= self.conditioning_samples < len(samples)):
            raise ValueError("actual block, native origin, common nuisance and nonempty prefix/tail must align")
        for key, value in (("samples", samples), ("temporal_root", temporal), ("nuisance", nuisance)):
            object.__setattr__(self, key, frozen_array(value))


@dataclass(frozen=True)
class ConditionalHistoryEvidence:
    candidate_ids: tuple[str, ...]
    state_indices: tuple[tuple[int, ...], ...]
    conditional_log_likelihood: np.ndarray
    log_candidate_probability: np.ndarray
    state_joint_log_marginal: np.ndarray
    state_prefix_log_marginal: np.ndarray
    state_prefix_probability_given_current: np.ndarray
    current_prior: np.ndarray
    measure_kind: str
    evaluated_states: int
    identical_design_state_groups: tuple[tuple[int, ...], ...]
    program: str = field(default="finite_history_conditional_tail_fixed_current_prior", init=False)

    @property
    def probabilities(self):
        return frozen_array(np.exp(self.log_candidate_probability))


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


def conditional_history_evidence(posterior, history, observation):
    """Exact finite-state conditional-tail mixture with the frozen current prior.

    Both integrals use the same nuisance columns/coordinate volume. The prefix
    updates unknown-neighbour probabilities and the shared response posterior.
    The output does not claim a full-data Bayes current-candidate posterior.
    """
    if (not isinstance(posterior, ResponsePosterior) or not isinstance(history, PublicResponseHistory)
            or not isinstance(observation, ResponseObservation)
            or observation.samples.shape[1] != posterior.mean.shape[1]):
        raise ValueError("history evidence requires aligned owned posterior/history/observations")
    axis = tuple(s.selection_id for s in history.selections).index(history.current_selection_id)
    current = history.selections[axis]
    end = observation.origin_sample+len(observation.samples)
    if current.onsets[0] < observation.origin_sample or int(current.onsets[-1])+len(posterior.basis) > end:
        raise ValueError("requested current selection needs its complete observed native response tails")
    states = tuple(product(*(range(len(s.candidate_ids)) for s in history.selections)))
    prior = history.joint_label_prior.reshape(-1)
    active = prior > 0
    logprior = np.full(len(states), -np.inf)
    logprior[active] = np.log(prior[active])
    joint, prefix = np.full(len(states), -np.inf), np.full(len(states), -np.inf)
    u = observation.conditioning_samples
    design_groups = {}
    for i, state in enumerate(states):
        if not active[i]:
            continue
        design = response_history_design(posterior, history, observation, state)
        design_groups.setdefault((state[axis], design.tobytes()), []).append(i)
        joint[i] = _log_marginal(observation.samples, design, observation.temporal_root, observation.nuisance, posterior)
        prefix[i] = _log_marginal(observation.samples[:u], design[:u], observation.temporal_root[:u, :u],
                                  observation.nuisance[:u], posterior)
    q_index = np.asarray([state[axis] for state in states])
    evidence, current_prior = [], []
    weights = np.zeros((len(current.candidate_ids), len(states)))
    for q in range(len(current.candidate_ids)):
        selected = q_index == q
        full = float(logsumexp(logprior[selected]+joint[selected]))
        past = float(logsumexp(logprior[selected]+prefix[selected]))
        evidence.append(full-past)
        current_prior.append(float(prior[selected].sum()))
        weights[q, selected] = np.exp(logprior[selected]+prefix[selected]-past)
    scores = np.asarray(evidence)+np.log(current_prior)
    if not np.isfinite(scores).all():
        raise ValueError("requested candidate conditional mixture is not representable")
    normalized = scores-logsumexp(scores)
    arrays = [frozen_array(a) for a in (evidence, normalized, joint, prefix, weights, current_prior)]
    kind = "proper_Gaussian_density_ratio" if observation.nuisance.shape[1] == 0 else "common_sigma_finite_nuisance_marginal_ratio"
    groups = tuple(tuple(indices) for indices in design_groups.values() if len(indices) > 1)
    return ConditionalHistoryEvidence(current.candidate_ids, states, *arrays, kind, int(active.sum()), groups)
