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
