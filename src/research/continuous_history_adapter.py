"""Read-only research adapter from the actual frozen continuous producer.

Only explicitly supplied, already available actual public clocks are accepted.
The external caller proves source, permission, reference and complete history;
this adapter checks numerical and native-clock consistency, not that authority.
It never refits/recompiles noise or response, reconstructs AR/Toeplitz noise,
changes a production scorer, or turns a scheduled future clock into actual data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from research.response_history_identification import (
    PublicResponseHistory,
    ResponseObservation,
    ResponsePosterior,
    StructuredResponseObservation,
    conditional_history_evidence,
)
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.noise_model import array_digest, finite_real_array, freeze_metadata


class ContinuousHistoryUnsupportedError(ValueError):
    """The producer/program does not support this exact research adapter."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

    def record(self):
        return {"kind": "unsupported_continuous_history_adapter", "reason": self.reason}


@dataclass(frozen=True)
class ContinuousHistoryInputs:
    posterior: ResponsePosterior
    history: PublicResponseHistory
    observation: ResponseObservation
    binding: dict

    def evidence(self):
        return conditional_history_evidence(self.posterior, self.history, self.observation)


def _stored_temporal_root(noise, samples):
    """Invert the actual stored runtime's square impulse whitening image."""
    if noise.startup_mode not in {"stationary", "stationary_innovations"}:
        raise ContinuousHistoryUnsupportedError("rectangular_conditional_prefix_has_no_joint_stationary_covariance")
    identity = np.eye(samples)
    whitening = noise.whiten(identity)
    if (whitening.shape != identity.shape or not np.isfinite(whitening).all()
            or np.any(np.triu(whitening, 1)) or np.any(np.diag(whitening) <= 0)):
        raise ContinuousHistoryUnsupportedError("actual_whitening_is_not_representable_positive_lower_square")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        root = solve_triangular(whitening, identity, lower=True)
        relation = whitening@root
        scale = np.abs(whitening)@np.abs(root)+identity
        diagonal = np.diag(whitening)*np.diag(root)
    if not (np.isfinite(root).all() and np.isfinite(relation).all() and np.isfinite(scale).all()):
        raise ContinuousHistoryUnsupportedError("stored_whitening_inverse_is_not_finitely_representable")
    eps = np.finfo(float).eps
    gamma = (samples+2)*eps/(1-(samples+2)*eps)
    if (np.any(np.abs(relation-identity) > 8*gamma*scale)
            or np.any(np.abs(diagonal-1) > 16*eps*np.abs(diagonal))):
        raise ContinuousHistoryUnsupportedError("stored_whitening_inverse_relation_is_numerically_unresolved")
    return root


def adapt_continuous_history(model, history, samples, *, origin_sample, available_sample,
                             sfreq, channels, operator_identity, record_id):
    """Bind actual model state to supplied authorized native history observations.

    Matching identity/clock values are consistency checks. The caller must prove
    their truth, permissions, held independence and event completeness separately.
    The model's existing frozen prefix and diffuse trend program are retained.
    This dense small-matrix interface supports already available actual clocks;
    it does not certify access to scheduled or unobserved future event times.
    """
    if not isinstance(model, ContinuousDecisionModel) or not isinstance(history, PublicResponseHistory):
        raise ValueError("actual continuous model and owned public history are required")
    model_digest = model.digest()
    if (model.provenance["fit_scope"] not in {"source_training", "target_calibration"}
            or model._posterior_precision_root is None):
        raise ContinuousHistoryUnsupportedError("actual_fitted_producer_posterior_is_required")
    if model.noise is None:
        raise ContinuousHistoryUnsupportedError("actual_stored_noise_is_required_no_legacy_phi_reconstruction")
    if (sfreq != model.sfreq or tuple(channels) != model.channels
            or operator_identity != model.operator_identity):
        raise ValueError("native fs/channel/reference-program differs from the frozen producer")
    current = next(s for s in history.selections if s.selection_id == history.current_selection_id)
    if current.candidate_ids != model.candidate_ids:
        raise ValueError("current candidate vocabulary/order differs from the frozen producer")
    values = finite_real_array(samples, name="actual observed native volts")
    if values.ndim != 2 or values.shape[1] != len(model.channels):
        raise ValueError("observed native columns differ from the frozen producer")
    limit = np.iinfo(np.int64).max
    if (type(origin_sample) is not int or type(available_sample) is not int
            or not 0 <= origin_sample <= available_sample <= limit
            or origin_sample+len(values) > available_sample):
        raise ValueError("actual observed block extends beyond its declared native availability")
    if any(np.any(s.onsets >= available_sample) for s in history.selections):
        raise ContinuousHistoryUnsupportedError("future_actual_public_event_clock_is_not_available")
    if (not isinstance(record_id, str) or not record_id.strip()
            or record_id in {r["record_id"] for r in model.provenance["records"]}):
        raise ValueError("history observations require a distinct held record identity")
    if array_digest(values, current.onsets-origin_sample) in {
        r["signal_sha256"] for r in model.provenance["records"]
    }:
        raise ValueError("history observations repeat an exact calibration signal/current clock")
    prefix = model.conditioning_samples
    if not 2 <= prefix < len(values):
        raise ContinuousHistoryUnsupportedError("frozen_common_diffuse_prefix_must_identify_trend_and_leave_tail")
    posterior = ResponsePosterior(model.basis, model._predictive_kernel_mean, model._posterior_precision_root,
                                  np.linalg.cholesky(model.innovation_covariance), model.sfreq)
    temporal = _stored_temporal_root(model.noise, len(values))
    nuisance = np.column_stack((np.ones(len(values)), (2*np.arange(len(values))-(prefix-1))/(prefix-1)))
    observation = ResponseObservation(values, origin_sample, temporal, nuisance, prefix)
    binding = freeze_metadata({
        "program": "actual_continuous_producer_to_finite_history_reference/1",
        "model_digest": model_digest, "noise_digest": model.noise.identity_digest,
        "model_producer_source": model.provenance["implementation_sha256"],
        "noise_producer_source": model.noise.record()["compiler"]["source_sha256"],
        "temporal_factor_program": "inverse_of_actual_stored_whitened_impulses",
        "clock_kind": "already_available_actual_public_native_onsets",
        "record_id": record_id, "operator_identity": operator_identity, "sfreq": sfreq,
        "channels": model.channels, "observed_sample_support": [origin_sample, origin_sample+len(values)],
        "available_sample": available_sample, "conditioning_samples": prefix,
        "authority": "external_caller_proves_source_permission_reference_time_and_complete_history",
    })
    return ContinuousHistoryInputs(posterior, history, observation, binding)


def score_authorized_history(model, capability, *, maximum_evaluated_states=100000):
    """Opt-in native full-history prediction with original-source authority.

    The legacy external-authority/dense research bridge above stays distinct.
    This route reuses actual C/noise/dictionary and the same finite mixture
    owner; no fit, covariance restart, producer replacement or future read.
    """
    from math import prod

    from research.channel_noise_mechanism import native_history_marginal_program
    from research.history_authority import AuthorizedResponseHistory, HistoryUnavailableError
    from research.latent_response_reference import selection_design
    from transfer.noise_model import metadata_digest

    if not isinstance(capability, AuthorizedResponseHistory):
        raise TypeError("native history requires its original-source-owned capability")
    capability.assert_model(model)
    if type(maximum_evaluated_states) is not int or maximum_evaluated_states < 1:
        raise ValueError("history resource limit must be a positive declared integer")
    view, law = capability.view, capability.law
    omitted = set(capability.binding["exact_zero_design_axis_marginalization"]["selection_ids"])
    selections = tuple(s for s in capability.selections if s.selection_id not in omitted)
    labels_shape = tuple(len(s.candidate_ids) for s in selections)
    states_shape = (len(law.states),)*len(selections)
    full_shape = labels_shape+states_shape
    count = prod(full_shape)
    if count > maximum_evaluated_states:
        raise HistoryUnavailableError(f"complete_finite_history_resource_limit:{count}>{maximum_evaluated_states}")
    if law.joint_mass is None:
        mass = np.ones(full_shape)

        def multiply_mass(factor):
            nonlocal mass
            result = mass*factor
            if np.any((mass > 0) & (factor > 0) & (result == 0)):
                raise HistoryUnavailableError("complete_finite_prior_positive_mass_underflow")
            mass = result

        for axis, selection in enumerate(selections):
            factor = np.asarray(capability.binding["target_prior_factors"][selection.selection_id])
            shape = [1]*len(full_shape)
            shape[axis] = len(factor)
            multiply_mass(factor.reshape(shape))
        probabilities = np.asarray([s.probability for s in law.states])
        for axis in range(len(selections), len(full_shape)):
            shape = [1]*len(full_shape)
            shape[axis] = len(probabilities)
            multiply_mass(probabilities.reshape(shape))
    else:
        # Condition only on source-owner-authorized C labels. A zero-image
        # axis may be correlated with current: sum its real joint mass, never
        # replace it with an assumed independent uniform factor.
        mass = capability.conditioned_joint_mass
        axes = tuple(axis for axis, s in enumerate(capability.selections) if s.selection_id in omitted)
        axes += tuple(axis+len(capability.selections) for axis in axes)
        if axes:
            mass = mass.sum(axis=axes)
    latent_axes = tuple(range(len(selections), len(full_shape)))
    label_mass = mass.sum(axis=latent_axes)
    history = PublicResponseHistory(selections, capability.current_selection_id, label_mass)
    prefix, n = model.conditioning_samples, len(view.samples)
    if not 2 <= prefix < n:
        raise HistoryUnavailableError("original_frozen_prefix_cannot_identify_common_flat_trend")
    observation = StructuredResponseObservation(view.samples, view.raw_scope[0],
        np.column_stack((np.ones(n), (2*np.arange(n)-(prefix-1))/(prefix-1))), prefix,
        capability.binding["current_response_right_censored"])
    posterior = ResponsePosterior(model.basis, model._predictive_kernel_mean, model._posterior_precision_root,
        np.linalg.cholesky(model.innovation_covariance), model.sfreq)
    primitive = native_history_marginal_program(
        mean=model._predictive_kernel_mean,
        precision_root=model._posterior_precision_root,
        innovation_covariance=model.innovation_covariance,
        temporal_noise=model.noise,
        samples=view.samples,
        nuisance=observation.nuisance,
        conditioning_samples=prefix,
    )
    design_records = {}

    def design_builder(state):
        design = np.zeros((n, model.kernel_mean.shape[0]))
        ledger = []
        for axis, selection in enumerate(selections):
            contribution, receipt = selection_design(model.dictionary, selection.onsets,
                selection.membership[:, state[axis]], origin_sample=view.raw_scope[0], observed_samples=n,
                state=law.states[state[len(selections)+axis]], support="translated_physical")
            design += contribution
            ledger.append({"selection_id": selection.selection_id, "execution": receipt})
        design_records[str(state)] = {"design_sha256": array_digest(design),
                                     "ledger_sha256": metadata_digest(ledger)}
        return design

    evidence = conditional_history_evidence(posterior, history, observation,
        full_state_prior=mass, design_builder=design_builder, marginal_evaluator=primitive)
    binding = freeze_metadata({"schema": "n2p3_authorized_history_prediction/1",
        "capability_digest": capability.digest, "model_digest": model.digest(),
        "old_model_producer_source": model.provenance["implementation_sha256"],
        "consumer_source_sha256": __import__("hashlib").sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest(),
        "noise_digest": model.noise.identity_digest, "dictionary_digest": model.dictionary.digest(),
        "law_digest": law.digest, "samples_sha256": array_digest(view.samples),
        "nuisance_sha256": array_digest(observation.nuisance),
        "posterior_precision_sha256": array_digest(model._posterior_precision_root),
        "complete_finite_state_shape": list(full_shape), "joint_mass_sha256": array_digest(mass),
        "state_designs": design_records, "maximum_evaluated_states": maximum_evaluated_states,
        "state_numerical_resolution": [{"state_indices": list(state), "resolution": resolution}
            for state, resolution in zip(evidence.state_indices, evidence.state_resolution, strict=True)],
        "mixture_numerical_resolution": evidence.mixture_resolution,
        "central_consumer_source_sha256": evidence.mixture_resolution["component_source_sha256"],
        "measure_kind": evidence.measure_kind, "score_program": evidence.program,
        "structured_primitive": "stored_noise_full_record_whitening_existing_augmented_QR_full_prefix/1",
        "n_raw_full": n, "n_raw_prefix": prefix,
        "conditioning_support": list(capability.binding["conditioning_support"]),
        "score_support": list(capability.binding["score_support"]),
        "log_evidence": evidence.conditional_log_likelihood.tolist(),
        "log_probability": evidence.log_candidate_probability.tolist(),
        "state_joint_log_marginal": [float(x) if np.isfinite(x) else None for x in evidence.state_joint_log_marginal],
        "state_prefix_log_marginal": [float(x) if np.isfinite(x) else None for x in evidence.state_prefix_log_marginal],
        "state_conditional_log_density": [float(x) if np.isfinite(x) else None for x in evidence.state_conditional_log_density],
        "current_prior": evidence.current_prior.tolist(),
        "fits": 0, "response_fits": 0, "new_noise_producers": 0})
    return evidence, binding
