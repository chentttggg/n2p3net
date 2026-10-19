"""Finite selection response law; proper scalar-channel research reference.

The existing joint_response_scale owner integrates ONE proper Gaussian vector.
That vector contains global common/target coefficients and independent,
zero-mean proper selection deviations. Amplitude, latency and width remain a
finite non-Gaussian selection law. This is not a production artifact/compiler,
continuous-latent quadrature claim, hyperparameter fit or history authority.
Free/hard dictionaries and soft endpoint coefficient priors remain externally
declared research choices. No product caller or default is registered here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from scipy.special import logsumexp

from research.joint_response_scale import (
    DecisionBlock,
    FixedHyperparameters,
    GlobalGaussianPrior,
    NumericalResolutionError,
    compile_decision,
    conditional_gaussian_components,
)
from transfer.noise_model import array_digest
from transfer.response_dictionary import ResponseDictionary

__all__ = ["ResponseState", "analytic_basis", "hierarchical_prior", "selection_design",
           "finite_selection_evidence"]
_GAUSSIAN_NOISE = FixedHyperparameters(np.inf, 1., "externally_fixed")
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResponseState:
    common_amplitude: float
    target_amplitude: float
    latency_seconds: float
    width_multiplier: float
    probability: float


def _admit_law(states):
    states = tuple(states)
    values = np.asarray([[s.common_amplitude, s.target_amplitude, s.latency_seconds,
                          s.width_multiplier, s.probability] for s in states])
    if (not len(states) or not np.isfinite(values).all() or np.any(values[:, :2] <= 0)
            or np.any(values[:, 2] < 0) or np.any(values[:, 3:] <= 0)
            or not np.isclose(values[:, 4].sum(), 1, atol=2e-15, rtol=0)):
        raise ValueError("finite law needs positive amplitudes/width/mass and causal latencies")
    if not np.allclose(values[:, 4] @ values[:, :2], [1., 1.], atol=2e-15, rtol=0):
        raise ValueError("amplitude mean one is the declared global/selection scale anchor")
    return states


def analytic_basis(dictionary: ResponseDictionary, seconds, derivative=0):
    """Off-grid continuation of the saved analytic family, in seconds.

    Values (derivative=0) use the exact continuous dictionary owner for all five
    legal families/charts. Float seconds denote their binary64 real values.
    Derivatives 1/2 retain the historical NumPy research-only helper and units
    s^-derivative; they are not a new portable derivative artifact protocol.
    """
    if derivative not in (0, 1, 2):
        raise ValueError("finite physical time and derivatives 0, 1 or 2 are required")
    if derivative == 0:
        return dictionary.evaluate(seconds, coordinate="seconds", support="analytic").matrix
    u = np.asarray(seconds, dtype=np.float64)
    if not np.isfinite(u).all():
        raise ValueError("finite physical time and derivatives 0, 1 or 2 are required")
    if dictionary.spec.family == "cosine":
        omega = np.asarray(dictionary.metadata()["generation"]["angular_frequencies_per_second"])
        phase = u[..., None]*omega
        if derivative == 1:
            ambient = -omega*np.sin(phase)
        else:
            ambient = -omega**2*np.cos(phase)
    elif dictionary.spec.family == "gaussian":
        generation = dictionary.metadata()["generation"]
        center = np.asarray(generation["centers_seconds"])
        sigma = generation["width_seconds"]
        distance = u[..., None]-center
        ambient = np.exp(-.5*(distance/sigma)**2)
        if derivative == 1:
            ambient = -distance/sigma**2*ambient
        elif derivative == 2:
            ambient = (distance**2/sigma**4-1/sigma**2)*ambient
    else:
        raise NumericalResolutionError("off_grid_response_dictionary_family_is_unsupported")
    return ambient @ dictionary.constraint_map


def selection_design(dictionary, onsets, labels, *, origin_sample, observed_samples,
                     state: ResponseState, support="translated_physical"):
    """One selection common/target design, native actual clock, actual read cut.

    translated_physical means g((u-delta)/w), g=1_[0,T) Phi.
    fixed_mask means 1_[0,T)(u) Phi((u-delta)/w), a DIFFERENT model.
    Voltage amplitudes are retained; no hidden area-preserving 1/w factor.
    """
    if support not in {"translated_physical", "fixed_mask"}:
        raise ValueError("explicit physical support program is required")
    parameters = np.asarray([state.common_amplitude, state.target_amplitude,
                             state.latency_seconds, state.width_multiplier])
    if (not np.isfinite(parameters).all() or np.any(parameters[:2] <= 0)
            or parameters[2] < 0 or parameters[3] <= 0):
        raise ValueError("positive amplitudes/width and causal latency are required")
    onset = np.asarray(onsets)
    label = np.asarray(labels)
    if (onset.ndim != 1 or onset.dtype.kind not in "iu" or not len(onset)
            or np.any(onset[1:] <= onset[:-1]) or onset[0] < 0
            or label.shape != onset.shape or not np.isin(label, [0, 1]).all()
            or type(origin_sample) is not int or origin_sample < 0
            or type(observed_samples) is not int or observed_samples < 1):
        raise ValueError("ordered actual public sample clock and binary event design are required")
    n, k = observed_samples, dictionary.matrix.shape[1]
    dictionary.digest()
    design = np.zeros((n, 2*k))
    ledger = []
    rate = Fraction.from_float(float(dictionary.sample_rate_hz))
    duration = Fraction(len(dictionary.matrix), 1)/rate
    latency = Fraction.from_float(float(state.latency_seconds))
    width = Fraction.from_float(float(state.width_multiplier))
    for event, target in zip(onset, label, strict=True):
        start = Fraction(int(event), 1)/rate+latency
        stop = start+width*duration
        physical_start = Fraction(int(event), 1)/rate if support == "fixed_mask" else start
        physical_stop = Fraction(int(event), 1)/rate+duration if support == "fixed_mask" else stop
        native_start, native_stop = rate*physical_start, rate*physical_stop
        # Exact half-open physical support drives every sample/censor decision.
        # Stored binary64 scalars are rational; no epsilon or rounded ceil.
        sample_start = -(-native_start.numerator//native_start.denominator)
        sample_stop = -(-native_stop.numerator//native_stop.denominator)
        first, last = max(origin_sample, sample_start), min(origin_sample+n, sample_stop)
        retained = max(0, last-first)
        basis = np.zeros((n, k))
        evaluation = None
        if retained:
            if state.latency_seconds == 0 and state.width_multiplier == 1:
                basis[first-origin_sample:last-origin_sample] = dictionary.matrix[first-int(event):last-int(event)]
                evaluation = {"program": "original_saved_native_matrix_slice/1",
                    "dictionary_identity_digest": dictionary.identity_digest,
                    "native_rows": [first-int(event), last-int(event)],
                    "matrix_sha256": array_digest(dictionary.matrix[first-int(event):last-int(event)])}
            else:
                # Keep r=(sample-event-fs*delta)/w as its actual rational.
                # Binary64 conversion can collapse an interior cell onto L;
                # support and scalar evaluation must consume the same real r.
                coordinate = [(Fraction(sample-int(event))-rate*latency)/width for sample in range(first, last)]
                artifact = dictionary.evaluate(coordinate, coordinate="sample",
                    support="half_open" if support == "translated_physical" else "analytic")
                basis[first-origin_sample:last-origin_sample] = artifact.matrix
                evaluation = artifact.record()
        design[:, :k] += state.common_amplitude*basis
        design[:, k:] += float(target)*state.target_amplitude*basis
        try:
            displayed_seconds = [float(physical_start), float(physical_stop)]
        except OverflowError as error:
            raise NumericalResolutionError("physical_support_seconds_display_is_not_representable") from error
        if not np.isfinite(displayed_seconds).all():
            raise NumericalResolutionError("physical_support_seconds_display_is_not_representable")
        ledger.append({"onset_sample": int(event), "target_indicator": int(target), "response_seconds": displayed_seconds,
                       "response_seconds_display_only": True,
                       "response_seconds_exact_ratio": [[physical_start.numerator, physical_start.denominator],
                                                        [physical_stop.numerator, physical_stop.denominator]],
                       "response_sample_centers_half_open": [sample_start, sample_stop],
                       "retained_native_rows": retained,
                       "response_evaluation": evaluation,
                       "left_censored": sample_start < origin_sample,
                       "right_censored": sample_stop > origin_sample+n})
    return design, {"support_program": support, "observed_sample_support": [origin_sample, origin_sample+n],
                    "schema": "n2p3_exact_selection_response_design/1",
                    "program": "integer_actual_event_clock_exact_finite_delta_width_dictionary_design/1",
                    "dictionary_identity_digest": dictionary.identity_digest,
                    "dictionary_producer": dictionary.metadata()["implementation_sha256"],
                    "consumer_implementation_sha256": IMPLEMENTATION_SHA256,
                    "state_exact_ratio": {"latency_seconds": [latency.numerator, latency.denominator],
                        "width_multiplier": [width.numerator, width.denominator]},
                    "common_amplitude": float(state.common_amplitude), "target_amplitude": float(state.target_amplitude),
                    "physical_duration_definition": "exact_L_over_stored_sample_rate",
                    "native_support_arithmetic": "exact_stored_binary64_rational_half_open",
                    "events": ledger, "design_sha256": array_digest(design)}


def hierarchical_prior(global_mean, global_covariance, selection_covariance, selection_count):
    """External fixed physical prior; additive selection mean is exactly zero.

    Coefficient order: global(common,target), then each selection(common,target).
    Independent selection deviations do not delete global cross-selection
    covariance. Hyperparameters are supplied, never estimated from test/C here.
    """
    mean = np.asarray(global_mean, dtype=float)
    covariance = np.asarray(global_covariance, dtype=float)
    local = np.asarray(selection_covariance, dtype=float)
    if (mean.ndim != 1 or len(mean) == 0 or len(mean) % 2
            or covariance.shape != (len(mean), len(mean)) or local.shape != covariance.shape
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.isfinite(local).all() or type(selection_count) is not int or selection_count < 1
            or not np.array_equal(covariance, covariance.T)
            or not np.array_equal(local, local.T)):
        raise ValueError("proper fixed common/target and independent selection coordinates are required")
    total = block_diag(covariance, *([local]*selection_count))
    return GlobalGaussianPrior(np.concatenate((mean, np.zeros(len(mean)*selection_count))),
                               np.linalg.cholesky(total), "external")


def _coefficient_design(dictionary, unit_design, block_index, selections, state):
    """Global amplitude product plus additive proper selection deviation.

    Local deviations use the SAME transformed common/target temporal shape,
    but are not multiplied by random amplitude: this is a declared additive
    coefficient model, not a second copy of global beta uncertainty.
    """
    k = dictionary.matrix.shape[1]
    width = 2*k
    result = np.zeros((len(unit_design), width*(1+selections)))
    result[:, :k] = state.common_amplitude*unit_design[:, :k]
    result[:, k:width] = state.target_amplitude*unit_design[:, k:]
    result[:, width*(1+block_index):width*(2+block_index)] = unit_design
    return result


def _compiled_selection(prior, dictionary, samples, onsets, labels, noise_root, *, block_index,
                       selections, origin_sample, state, record_id, role, prefix=None):
    unit_state = ResponseState(1., 1., state.latency_seconds, state.width_multiplier, state.probability)
    design, _ = selection_design(dictionary, onsets, labels, origin_sample=origin_sample,
                                      observed_samples=len(samples), state=unit_state)
    full_design = _coefficient_design(dictionary, design, block_index, selections, state)
    block = DecisionBlock(record_id, samples, full_design, noise_root, role)
    return compile_decision(prior, _GAUSSIAN_NOISE, block, conditioning_rows=prefix)


def finite_selection_evidence(prior, dictionary, calibration, observation, membership, law, *, prefix):
    """Integrate finite response states and ONE proper hierarchical coefficient.

    Each mapping contains scalar-channel samples[n], an actual positive-lower
    noise_root[n,n] in volts, absolute integer onsets, origin_sample, record_id
    and available_sample. Only calibration has binary labels; inference uses
    the public event-by-candidate membership, without target truth.
    The caller proves full union-support availability or a declared observed-
    censored estimand, complete history, C independence and source authority.
    Identity/available_sample checks do not grant that authority. No candidate
    normalization, hyperparameter fitting or product admission is performed.
    """
    states = _admit_law(law)
    dictionary.digest()
    count = len(calibration)
    selections = count+1
    width = 2*dictionary.matrix.shape[1]
    if len(prior.mean) != width*(1+selections):
        raise ValueError("one global vector and all proper independent selection effects are required")
    groups = np.arange(len(prior.mean))//width
    if (prior.program not in {"external", "source_only"} or np.any(prior.mean[width:] != 0)
            or np.any(prior.root[groups[:, None] != groups[None, :]] != 0)):
        raise ValueError("independent zero-mean selection prior and external/source global prior are required")
    identities = [item["record_id"] for item in (*calibration, observation)]
    if len(set(identities)) != len(identities):
        raise ValueError("C and read-only inference cannot repeat a selection identity")
    for item in (*calibration, observation):
        available = item["available_sample"]
        if (type(available) is not int or available < item["origin_sample"]+len(item["samples"])
                or np.any(np.asarray(item["onsets"]) >= available)):
            raise ValueError("actual clocks and read support must already be available")
    member = np.asarray(membership)
    if (member.ndim != 2 or member.shape[0] != len(observation["onsets"])
            or member.shape[1] < 2 or not np.isin(member, [0, 1]).all()
            or np.any(member.sum(axis=1) == 0) or np.any(member.sum(axis=0) == 0)):
        raise ValueError("public complete candidate membership is required")
    c_compiled = []
    for index, item in enumerate(calibration):
        row = []
        for state in states:
            block = _compiled_selection(prior, dictionary, item["samples"], item["onsets"],
                item["labels"], item["noise_root"], block_index=index, selections=selections,
                origin_sample=item["origin_sample"], state=state, record_id=item["record_id"],
                role="target_calibration")
            row.append(block)
        c_compiled.append(row)
    c_states = list(product(range(len(states)), repeat=count))
    c_logmass = np.asarray([sum(np.log(states[j].probability) for j in state) for state in c_states])
    c_logs, c_means, c_covariances = [], [], []
    for state in c_states:
        blocks = tuple(c_compiled[index][node] for index, node in enumerate(state))
        logs, means, covariances = conditional_gaussian_components(prior, blocks, _GAUSSIAN_NOISE,
                                                                  np.ones((1, len(blocks))))
        c_logs.append(float(logs[0]))
        c_means.append(means[0])
        c_covariances.append(covariances[0])
    full, before, joint_keys = [], [], []
    # Reuse factory-owned C and inference QR states; never re-sign an artifact.
    for candidate in range(member.shape[1]):
        full_terms, prefix_terms = [], []
        for node, state in enumerate(states):
            test = _compiled_selection(prior, dictionary, observation["samples"], observation["onsets"],
                member[:, candidate], observation["noise_root"], block_index=count, selections=selections,
                origin_sample=observation["origin_sample"], state=state, record_id=observation["record_id"],
                role="read_only_inference", prefix=prefix)
            for index, c_state in enumerate(c_states):
                c_blocks = tuple(c_compiled[i][j] for i, j in enumerate(c_state))
                weight = c_logmass[index]+np.log(state.probability)
                for target, test_block in ((full_terms, test), (prefix_terms, test.prefix)):
                    logs, _, _ = conditional_gaussian_components(prior, (*c_blocks, test_block), _GAUSSIAN_NOISE,
                                                                  np.ones((1, count+1)))
                    target.append(float(logs[0])+weight)
                if candidate == 0:
                    joint_keys.append([*c_state, node])
        full.append(full_terms)
        before.append(prefix_terms)
    full, before = np.asarray(full), np.asarray(before)
    score = logsumexp(full, axis=1)-logsumexp(before, axis=1)
    return {"conditional_log_density": score, "full_joint_terms": full, "prefix_joint_terms": before,
            "state_posterior_given_C_prefix_q": np.exp(before-logsumexp(before, axis=1)[:, None]),
            "c_log_evidence": np.asarray(c_logs), "c_prior_logmass": c_logmass,
            "c_posterior_mass": np.exp(np.asarray(c_logs)+c_logmass-logsumexp(np.asarray(c_logs)+c_logmass)),
            "c_states": np.asarray(c_states), "joint_states": np.asarray(joint_keys),
            "c_coefficient_means": np.asarray(c_means), "c_coefficient_covariances": np.asarray(c_covariances),
            "finite_component_evaluations": len(c_states)*(1+2*member.shape[1]*len(states)),
            "evidence_scope": "exact_declared_finite_selection_law_given_external_hyperparameters",
            "normalized_candidate_posterior_claimed": False}
