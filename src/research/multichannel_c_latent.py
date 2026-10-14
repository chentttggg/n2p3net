"""Actual native continuous C-latent producer and same-record continuation.

The first programme uses a predeclared finite selection amplitude law and
fixed external physical prior/noise parameters. Original MAT ownership grants
C fitting and later unlabelled history conditioning separately. One shared
beta and one record-local flat drift survive every selection boundary.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from itertools import product
from math import prod
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.channel_noise_mechanism import (
    GaussianQRPosteriorState,
    NoiseGeometry,
    PhysicalCoefficientPrior,
    StructuredGaussianMarginalProgram,
    _StoredTemporalNoiseAdapter,
)
from research.contracts import semantic_sha256
from research.history_authority import HistoryUnavailableError
from research.latent_response_reference import ResponseState, _admit_law, selection_design
from research.response_history_identification import NeighborEventPrefix, finite_conditional_mixture
from transfer.noise_model import array_digest, freeze_metadata, frozen_array
from transfer.response_dictionary import ResponseDictionary
from transfer.temporal_noise import TemporalNoise

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_multichannel_c_latent_model/1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _nuisance(samples, channels, leading):
    z = np.column_stack((np.ones(samples), (2*np.arange(samples)-(leading-1))/(leading-1)))
    return np.einsum("nj,cd->ncjd", z, np.eye(channels)).reshape(samples, channels, 2*channels)


def _assert_original(compiler, prefix):
    raw = compiler._record
    if (_sha(compiler._history_source_payload) != raw.source_sha256
            or compiler._history_source_identity != array_digest(raw.eeg_uv, raw.flash_sample,
                raw.candidate_id, raw.selection_index, raw.repetition_index)):
        raise ValueError("C-latent original decoder operands changed after source admission")
    return compiler.assert_original_prefix(prefix)


def _selections(compiler, prefix, stop):
    """Only actual occurred event/membership and authorized C binary labels."""
    from data.bnci2014_008_candidate import (
        BNCI2014_008_GRID_SHAPE,
        BNCI2014_008_REPETITIONS_PER_SELECTION,
    )
    from data.candidate_task import MembershipDecision

    raw, target = compiler._record, compiler._arguments["target"]
    occurred = np.flatnonzero(raw.flash_sample < stop)
    selections, allowed = [], {}
    for name in dict.fromkeys(raw.selection_id[occurred]):
        rows = occurred[raw.selection_id[occurred] == name]
        member = MembershipDecision.row_column(raw.candidate_id[rows], n_rows=BNCI2014_008_GRID_SHAPE[0],
            n_columns=BNCI2014_008_GRID_SHAPE[1],
            event_ids=tuple(f"{target['source_dataset_id']}:{raw.subject_id}:{int(r)}" for r in rows),
            repetition_indices=raw.repetition_index[rows], repetitions=BNCI2014_008_REPETITIONS_PER_SELECTION)
        selections.append(NeighborEventPrefix(str(name), member.candidate_vocabulary,
                                              raw.flash_sample[rows], member.membership))
        known = rows < prefix.budget
        possible = np.all(member.membership[known] == prefix.raw.target_mask[rows[known], None], axis=0)
        if not possible.any():
            raise ValueError("actual C binary labels contradict original public membership")
        allowed[str(name)] = tuple(int(x) for x in np.flatnonzero(possible))
    return tuple(selections), allowed


@dataclass(frozen=True)
class FiniteCProtocol:
    states: tuple[ResponseState, ...]
    plan_bytes: bytes
    record: dict

    @classmethod
    def from_plan(cls, payload, *, prefix, dictionary, prior, noise, innovation_covariance):
        from transfer.ensemble_views import calibration_prefix_identity

        if not isinstance(payload, bytes):
            raise TypeError("finite C protocol requires actual frozen plan bytes")
        plan = json.loads(payload)
        required = {"schema", "freeze_role", "parameter_role", "calibration_prefix_digest",
            "dictionary_digest", "physical_prior_digest", "temporal_noise_digest",
            "innovation_covariance_sha256", "conditioning_samples", "response_states",
            "maximum_calibration_components", "maximum_prediction_states", "scoring_laws", "default_scoring_law"}
        if (set(plan) != required or plan["schema"] != "n2p3_finite_c_latent_protocol/1"
                or plan["freeze_role"] != "predeclared_protocol"
                or plan["parameter_role"] != "fixed_external_numeric_parameters"
                or plan["calibration_prefix_digest"] != calibration_prefix_identity(prefix)
                or plan["dictionary_digest"] != dictionary.digest()
                or plan["physical_prior_digest"] != semantic_sha256(prior.record())
                or plan["temporal_noise_digest"] != noise.identity_digest
                or plan["innovation_covariance_sha256"] != array_digest(innovation_covariance)):
            raise ValueError("finite C protocol differs from actual prefix or fixed parameter producers")
        for key in ("conditioning_samples", "maximum_calibration_components", "maximum_prediction_states"):
            if type(plan[key]) is not int or plan[key] < (2 if key == "conditioning_samples" else 1):
                raise ValueError("finite C resource/leading-prefix policy must be explicitly positive")
        declared = plan["scoring_laws"]
        if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
                or not set(declared) <= {"fixed_current_prior_conditional_tail", "whole_observed_current_q"}
                or plan["default_scoring_law"] not in declared):
            raise ValueError("finite C protocol must predeclare its exact scoring laws and default")
        states = _admit_law(ResponseState(**s) for s in plan["response_states"])
        if any((s.latency_seconds, s.width_multiplier) != (0., 1.) for s in states):
            raise ValueError("first continuous C-latent producer declares an amplitude-only finite axis")
        if (prior.provenance["role"] != "analytic_reference"
                or noise.provenance["fit_scope"] != "analytic_reference"):
            raise ValueError("this programme owns fixed external numeric parameters, not a learned source/C plugin")
        return cls(states, payload, freeze_metadata({**plan, "plan_sha256": _sha(payload)}))

    @property
    def digest(self):
        return semantic_sha256(self.record)


@dataclass(frozen=True, init=False)
class AuthorizedFiniteC:
    prefix: object
    selections: tuple
    allowed_candidates: dict
    dictionary: ResponseDictionary
    prior: PhysicalCoefficientPrior
    noise: TemporalNoise
    innovation_covariance: np.ndarray
    protocol: FiniteCProtocol
    binding: dict


def _issue_c_latent_fit_scope(compiler, prefix, *, dictionary, prior, noise, plan,
                              innovation_covariance):
    from transfer.ensemble_views import calibration_prefix_identity

    prefix = _assert_original(compiler, prefix)
    if (prefix.raw is None or not isinstance(dictionary, ResponseDictionary)
            or not isinstance(prior, PhysicalCoefficientPrior) or not isinstance(noise, TemporalNoise)):
        raise TypeError("C-latent fitting requires original nonempty C and actual fixed parameter owners")
    target = compiler._arguments["target"]
    geometry = NoiseGeometry(tuple(target["channel_names"]), target["sample_rate_hz"], "V",
        target["reference"], "native_unfiltered_volts/1", len(target["channel_names"]))
    if (prior.geometry != geometry or dictionary.sample_rate_hz != geometry.sfreq
            or not np.array_equal(prior.basis, dictionary.matrix)
            or noise.startup_mode not in {"stationary", "stationary_innovations"}):
        raise ValueError("fixed C prior/noise/dictionary differ from original native layout/reference/clock")
    covariance = np.asarray(innovation_covariance, dtype=float)
    protocol = FiniteCProtocol.from_plan(plan, prefix=prefix, dictionary=dictionary, prior=prior,
        noise=noise, innovation_covariance=covariance)
    leading = protocol.record["conditioning_samples"]
    if len(prefix.raw.samples) <= leading:
        raise HistoryUnavailableError("actual_C_prefix_cannot_identify_shared_record_flat_drift")
    _StoredTemporalNoiseAdapter(noise, covariance, leading)
    stop = len(prefix.raw.samples)
    selections, allowed = _selections(compiler, prefix, stop)
    count = prod(len(allowed[s.selection_id])*len(protocol.states) for s in selections)
    if count > protocol.record["maximum_calibration_components"]:
        raise HistoryUnavailableError(f"reachable_C_component_resource_policy:{count}")
    binding = {"schema": "n2p3_authorized_finite_c_fit/1", "source_owner": "CalibrationPrefixCompiler",
        "source_owner_sha256": compiler.history_owner_source, "original_source_sha256": target["source_sha256"],
        "prefix_identity": calibration_prefix_identity(prefix), "target_contract": prefix.target_contract,
        "label_use": prefix.label_use, "C_input_sha256": array_digest(prefix.raw.samples,
            prefix.raw.onsets, prefix.raw.target_mask), "fit_read_support": [0, stop],
        "fit_label_event_indices": [0, prefix.budget], "conditioning_read_support": None,
        "score_support": None, "native_geometry": geometry.record(), "protocol_digest": protocol.digest,
        "dictionary_digest": dictionary.digest(), "physical_prior_digest": semantic_sha256(prior.record()),
        "temporal_noise_digest": noise.identity_digest, "innovation_covariance_sha256": array_digest(covariance),
        "C_selections": [{"selection_id": s.selection_id, "candidate_ids": list(s.candidate_ids),
            "onsets": s.onsets.tolist(), "membership": s.membership.tolist(),
            "allowed_target_indices": list(allowed[s.selection_id])} for s in selections],
        "reachable_positive_components": count, "noise_startups": 1,
        "clock_kind": "original_decoder_actual_native_integer_sample", "truth_consumed": False}
    result = object.__new__(AuthorizedFiniteC)
    for key, value in {"prefix": prefix, "selections": selections, "allowed_candidates": freeze_metadata(allowed),
        "dictionary": dictionary, "prior": prior, "noise": noise,
        "innovation_covariance": frozen_array(covariance), "protocol": protocol,
        "binding": freeze_metadata(binding)}.items():
        object.__setattr__(result, key, value)
    return result


def _program(prior, dictionary, noise, covariance, samples, leading, *, transform=None,
             factor=None, factor_source_error=None):
    # PhysicalCoefficientPrior owns a general precision factor, not necessarily
    # an upper-triangular QR root after a physical basis change.
    if factor is None:
        factor = np.linalg.solve(prior.precision_root, np.eye(prior.mean.size))
        defect = np.linalg.norm(np.eye(prior.mean.size)-prior.precision_root@factor, ord=2)
        defect += (prior.mean.size+1)*np.finfo(float).eps*np.linalg.norm(
            np.abs(prior.precision_root)@np.abs(factor), ord=2)
        if not np.isfinite(defect) or defect >= 1:
            raise HistoryUnavailableError("fixed_physical_prior_inverse_unresolved")
        factor_source_error = np.full(factor.shape, np.linalg.norm(factor, ord=2)*defect/(1-defect))
    adapter = _StoredTemporalNoiseAdapter(noise, covariance, leading)
    return StructuredGaussianMarginalProgram(adapter, samples, prior.mean, factor,
        _nuisance(len(samples), len(prior.geometry.channels), leading), intercept_chart=True,
        flat_coordinate_transform=transform, factor_source_error=factor_source_error)


def _design(dictionary, selections, targets, states, *, observed_samples):
    result = np.zeros((observed_samples, 2*dictionary.matrix.shape[1]))
    for selection, target, state in zip(selections, targets, states, strict=True):
        contribution, _ = selection_design(dictionary, selection.onsets,
            selection.membership[:, target], origin_sample=0, observed_samples=observed_samples, state=state)
        result += contribution
    return result


@dataclass(frozen=True)
class FiniteCHistoryModel:
    """One persisted finite joint C posterior; no moment-matched replacement."""

    dictionary: ResponseDictionary
    prior: PhysicalCoefficientPrior
    noise: TemporalNoise
    innovation_covariance: np.ndarray
    protocol: FiniteCProtocol
    prior_factor: np.ndarray
    prior_factor_source_error: np.ndarray
    posteriors: tuple[GaussianQRPosteriorState, ...]
    components: tuple[dict, ...]
    binding: dict

    def __post_init__(self):
        fitted = self.binding["fit_authority"]
        selections, states = fitted["C_selections"], self.protocol.states
        expected = tuple(product(*(tuple(product(s["allowed_target_indices"], range(len(states))))
                                   for s in selections)))
        actual = tuple(tuple(zip(c["target_indices"], c["response_state_indices"], strict=True))
                       for c in self.components)
        if (self.binding["schema"] != _SCHEMA or actual != expected or not self.posteriors
                or len(self.components) != self.binding["positive_components"]
                or len(self.posteriors) != self.binding["unique_Gaussian_posteriors"]
                or self.protocol.digest != fitted["protocol_digest"]
                or self.dictionary.digest() != fitted["dictionary_digest"]
                or semantic_sha256(self.prior.record()) != fitted["physical_prior_digest"]
                or self.noise.identity_digest != fitted["temporal_noise_digest"]
                or array_digest(self.innovation_covariance) != fitted["innovation_covariance_sha256"]):
            raise ValueError("finite C producer changed its complete component law or fixed parameter owners")
        terms = []
        for c in self.components:
            index = c["posterior_index"]
            if type(index) is not int or not 0 <= index < len(self.posteriors):
                raise ValueError("finite C component names an absent Gaussian producer")
            prior = sum(np.log(states[i].probability)-np.log(len(s["allowed_target_indices"]))
                        for i, s in zip(c["response_state_indices"], selections, strict=True))
            if c["log_prior"] != prior:
                raise ValueError("finite C component prior differs from its complete positive protocol law")
            terms.append(prior+self.posteriors[index].log_marginal)
        terms = np.asarray(terms)
        expected_weights = terms-logsumexp(terms)
        if not np.array_equal(expected_weights, np.asarray([c["log_posterior_weight"] for c in self.components])):
            raise ValueError("finite C weights differ from actual component C evidence and protocol mass")
        if any(p.calibration_samples != fitted["fit_read_support"][1]
               or p.proper_dimensions != self.prior.mean.size
               or not np.array_equal(p.flat_transform, self.posteriors[0].flat_transform) for p in self.posteriors):
            raise ValueError("finite C Gaussian components do not share original record/prior/flat coordinates")
        factor, error = self.prior_factor, self.prior_factor_source_error
        width = self.prior.mean.size
        if (factor.shape != (width, width) or error.shape != factor.shape
                or not np.isfinite(factor).all() or not np.isfinite(error).all() or np.any(error < 0)):
            raise ValueError("saved producer proper-prior image/error do not match its physical coordinates")
        identity = np.eye(width)
        operands = np.abs(self.prior.precision_root)@np.abs(factor)
        bound = np.abs(self.prior.precision_root)@error+(width+2)*np.finfo(float).eps*(operands+identity)
        if np.any(np.abs(self.prior.precision_root@factor-identity) > bound):
            raise ValueError("saved producer proper-prior image differs from its original precision law")
        if any(p.chart_digest != array_digest(self.prior.mean, factor, p.flat_transform) for p in self.posteriors):
            raise ValueError("C posterior saved unit-latent image is not its original producer chart")

    @classmethod
    def fit(cls, authority):
        if not isinstance(authority, AuthorizedFiniteC):
            raise TypeError("finite C posterior is produced only from actual compiler-issued C authority")
        prior, dictionary, noise = authority.prior, authority.dictionary, authority.noise
        protocol, covariance = authority.protocol, authority.innovation_covariance
        primitive = _program(prior, dictionary, noise, covariance, authority.prefix.raw.samples,
                             protocol.record["conditioning_samples"])
        axes = [tuple(product(authority.allowed_candidates[s.selection_id], range(len(protocol.states))))
                for s in authority.selections]
        components, posteriors, cache = [], [], {}
        for indices in product(*axes):
            targets, state_ids = tuple(i[0] for i in indices), tuple(i[1] for i in indices)
            states = tuple(protocol.states[i] for i in state_ids)
            design = _design(dictionary, authority.selections, targets, states,
                             observed_samples=len(authority.prefix.raw.samples))
            identity = design.tobytes()
            if identity not in cache:
                cache[identity] = len(posteriors)
                posteriors.append(primitive.posterior_state(design))
            posterior_index = cache[identity]
            log_prior = sum(np.log(s.probability)-np.log(len(authority.allowed_candidates[selection.selection_id]))
                            for s, selection in zip(states, authority.selections, strict=True))
            components.append({"target_indices": targets, "response_state_indices": state_ids,
                "posterior_index": posterior_index, "log_prior": float(log_prior),
                "C_design_sha256": array_digest(design)})
        terms = np.asarray([c["log_prior"]+posteriors[c["posterior_index"]].log_marginal for c in components])
        anchor = logsumexp(terms)
        components = tuple(freeze_metadata({**c, "log_posterior_weight": float(term-anchor)})
                           for c, term in zip(components, terms, strict=True))
        binding = freeze_metadata({"schema": _SCHEMA, "fit_authority": authority.binding,
            "producer_source_sha256": IMPLEMENTATION_SHA256,
            "central_posterior_source_sha256": posteriors[0].resolution["source_sha256"],
            "posterior_program": "joint_finite_selection_states_shared_beta_and_record_drift_given_actual_C",
            "C_signal_log_marginal_given_authorized_C_labels": float(anchor), "positive_components": len(components),
            "unique_Gaussian_posteriors": len(posteriors), "moment_matching": False,
            "proper_prior_applications": 1, "flat_measure": "one_original_physical_record_da",
            "parameter_role": protocol.record["parameter_role"], "noise_fits": 0})
        return cls(dictionary, prior, noise, frozen_array(covariance), protocol,
                   frozen_array(primitive.context[0]), frozen_array(primitive.factor_source_error),
                   tuple(posteriors), components, binding)

    @property
    def calibration_samples(self):
        return self.posteriors[0].calibration_samples

    def record(self):
        return {"binding": self.binding, "protocol": self.protocol.record,
            "dictionary": self.dictionary.record(), "prior": self.prior.record(),
            "noise": self.noise.record(), "innovation_covariance_sha256": array_digest(self.innovation_covariance),
            "prior_image": {"program": "saved_producer_general_precision_inverse_and_defect_error/1",
                "factor_sha256": array_digest(self.prior_factor),
                "factor_source_error_sha256": array_digest(self.prior_factor_source_error)},
            "components": self.components,
            "posteriors": [{"root_rhs_flat_sha256": array_digest(p.precision_root, p.precision_response, p.flat_transform),
                "calibration_samples": p.calibration_samples, "proper_dimensions": p.proper_dimensions,
                "raw_prefix_digest": p.raw_prefix_digest, "chart_digest": p.chart_digest, "noise_digest": p.noise_digest,
                "log_marginal": p.log_marginal, "resolution": p.resolution} for p in self.posteriors]}

    def digest(self):
        return semantic_sha256(self.record())

    def save(self, path):
        metadata = self.record()
        metadata["model_digest"] = self.digest()
        arrays = {"prior_mean": self.prior.mean, "prior_precision_root": self.prior.precision_root,
            "prior_factor": self.prior_factor, "prior_factor_source_error": self.prior_factor_source_error,
            "innovation_covariance": self.innovation_covariance,
            "protocol_bytes": np.frombuffer(self.protocol.plan_bytes, dtype=np.uint8),
            "posterior_roots": np.stack([p.precision_root for p in self.posteriors]),
            "posterior_responses": np.stack([p.precision_response for p in self.posteriors]),
            "flat_transforms": np.stack([p.flat_transform for p in self.posteriors]),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))}
        path = Path(path)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return _sha(path.read_bytes())

    @classmethod
    def load(cls, path, *, expected_sha256):
        payload = Path(path).read_bytes()
        if _sha(payload) != expected_sha256:
            raise ValueError("finite C model differs from externally admitted original producer bytes")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            dictionary = ResponseDictionary.from_record(metadata["dictionary"])
            p = metadata["prior"]
            prior = PhysicalCoefficientPrior(archive["prior_mean"], archive["prior_precision_root"],
                dictionary.matrix, NoiseGeometry(**p["geometry"]), p["prior_id"], p["provenance"])
            noise = TemporalNoise.from_record(metadata["noise"])
            protocol = FiniteCProtocol(tuple(ResponseState(**s) for s in metadata["protocol"]["response_states"]),
                archive["protocol_bytes"].tobytes(), freeze_metadata(metadata["protocol"]))
            posteriors = tuple(GaussianQRPosteriorState(archive["posterior_roots"][i],
                archive["posterior_responses"][i], archive["flat_transforms"][i], p["calibration_samples"],
                p["proper_dimensions"], p["raw_prefix_digest"], p["chart_digest"], p["noise_digest"],
                p["log_marginal"], p["resolution"])
                for i, p in enumerate(metadata["posteriors"]))
            result = cls(dictionary, prior, noise, frozen_array(archive["innovation_covariance"]), protocol,
                frozen_array(archive["prior_factor"]), frozen_array(archive["prior_factor_source_error"]),
                posteriors, tuple(freeze_metadata(c) for c in metadata["components"]), freeze_metadata(metadata["binding"]))
        if result.digest() != metadata["model_digest"] or _sha(protocol.plan_bytes) != protocol.record["plan_sha256"]:
            raise ValueError("finite C posterior, original parameter authority or plan state changed")
        return result


@dataclass(frozen=True, init=False)
class AuthorizedFiniteCHistory:
    model_digest: str
    view: object
    samples: np.ndarray
    selections: tuple
    conditioning_samples: int
    binding: dict


def _issue_c_latent_history_scope(compiler, view, *, prefix, model, scoring_law=None):
    from data.bnci2014_008_candidate import BNCI2014_008_GRID_SHAPE
    from data.candidate_task import MembershipDecision
    from data.preprocess import prefix_raw_scopes
    from transfer.ensemble_views import PredictionDecisionView, calibration_prefix_identity

    if not isinstance(model, FiniteCHistoryModel) or not isinstance(view, PredictionDecisionView):
        raise TypeError("finite history requires actual C model and original physical current view")
    prefix = _assert_original(compiler, prefix)
    scoring_law = model.protocol.record["default_scoring_law"] if scoring_law is None else scoring_law
    if scoring_law not in model.protocol.record["scoring_laws"]:
        raise ValueError("finite history scoring law was not predeclared by its producer protocol")
    fitted = model.binding["fit_authority"]
    if (fitted["prefix_identity"] != calibration_prefix_identity(prefix)
            or fitted["C_input_sha256"] != array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask)):
        raise ValueError("finite history differs from actual C producer authorization")
    target, raw = compiler._arguments["target"], compiler._record
    physical = view.physical_record()
    keys = ("participant_key", "source_sha256", "source_relative_path", "source_dataset_id",
            "source_subject_id", "sample_rate_hz", "reference", "signal_unit", "channel_names", "channel_types")
    if any(semantic_sha256(physical[k]) != semantic_sha256(target[k]) for k in keys):
        raise ValueError("finite history native participant/reference/layout/clock differs from original authority")
    if not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0):
        raise HistoryUnavailableError("actual_current_schedule_incomplete", failure_kind="incomplete")
    start, stop = view.raw_scope
    if view.available_sample != stop or start < model.calibration_samples:
        raise ValueError("finite history current score overlaps C or differs from actual availability")
    _, test_scope = prefix_raw_scopes(compiler._raw, compiler._events,
        compiler._arguments["authorization_preprocessing"], calibration_events=0,
        maximum_calibration_events=compiler._arguments["maximum"])
    if not test_scope.start_sample <= start < stop <= test_scope.stop_sample:
        raise ValueError("finite history requested score exceeds original held support")
    current_rows = np.flatnonzero(raw.selection_id == view.decision_id)
    member = MembershipDecision.row_column(raw.candidate_id[current_rows], n_rows=BNCI2014_008_GRID_SHAPE[0],
        n_columns=BNCI2014_008_GRID_SHAPE[1],
        event_ids=tuple(f"{target['source_dataset_id']}:{raw.subject_id}:{int(r)}" for r in current_rows),
        repetition_indices=raw.repetition_index[current_rows], repetitions=view.decision.repetitions)
    if (member.record() != view.decision.record() or not np.array_equal(view.onsets+start, raw.flash_sample[current_rows])
            or view.decision.candidate_vocabulary != tuple(target["candidate_vocabulary"])):
        raise ValueError("finite history current clocks/membership differ from original event ledger")
    expected = np.asarray(raw.eeg_uv[start:stop], dtype=float)*1e-6
    if view.samples is not None and not np.array_equal(view.samples, expected):
        raise ValueError("finite history current samples differ from actual original source")
    allowed_overlap = {"previous_unknown_decision_overlaps_finite_response_support",
                       "next_unknown_decision_overlaps_finite_response_support"}
    if view.samples is None and view.unavailable_reason not in allowed_overlap:
        raise HistoryUnavailableError(view.unavailable_reason or "actual_current_samples_unavailable")
    selections, allowed = _selections(compiler, prefix, stop)
    c_names = tuple(s["selection_id"] for s in fitted["C_selections"])
    future = tuple(s for s in selections if s.selection_id not in c_names)
    if view.decision_id not in {s.selection_id for s in future}:
        raise ValueError("current held target cannot also be a fitted C target")
    leading = model.protocol.record["conditioning_samples"]
    conditioning = start+leading
    if not model.calibration_samples < conditioning < stop:
        raise HistoryUnavailableError("actual_future_prefix_must_leave_observed_score_tail")
    count = len(model.components)*prod(len(s.candidate_ids)*len(model.protocol.states) for s in future)
    if count > model.protocol.record["maximum_prediction_states"]:
        raise HistoryUnavailableError(f"reachable_future_state_resource_policy:{count}")
    length = len(model.dictionary.matrix)
    ledger = [{"selection_id": s.selection_id, "actual_onset_sample": int(onset),
        "finite_response_support": [int(onset), int(onset)+length],
        "retained_read_support": [int(onset), min(int(onset)+length, stop)],
        "right_censored": int(onset)+length > stop} for s in selections for onset in s.onsets]
    current_censored = any(int(onset)+length > stop for onset in raw.flash_sample[current_rows])
    binding = freeze_metadata({"schema": "n2p3_authorized_finite_c_history/1", "source_owner": "CalibrationPrefixCompiler",
        "original_source_sha256": target["source_sha256"], "model_digest": model.digest(),
        "prefix_identity": fitted["prefix_identity"], "current": view.record(),
        "fit_read_support": [0, model.calibration_samples], "fit_label_event_indices": [0, prefix.budget],
        "conditioning_read_support": [model.calibration_samples, conditioning], "score_support": [conditioning, stop],
        "full_read_support": [0, stop], "available_sample": stop,
        "conditioning_program": "all_actual_past_unlabelled_native_context_same_record_no_AR_restart",
        "scoring_law": scoring_law, "current_candidate_prior": "predeclared_independent_uniform",
        "conditioning_samples": leading, "current_first_onset_sample": int(raw.flash_sample[current_rows[0]]),
        "current_ERP_can_affect_prefix": int(raw.flash_sample[current_rows[0]]) < conditioning,
        "future_label_access": False, "truth_consumed": False, "clock_kind": fitted["clock_kind"],
        "original_baseline_route": view.unavailable_reason,
        "reachable_positive_states": count, "actual_C_target_support": allowed,
        "full_actual_event_ledger": ledger, "current_schedule_complete": True,
        "current_response_right_censored": current_censored})
    result = object.__new__(AuthorizedFiniteCHistory)
    for key, value in {"model_digest": model.digest(), "view": view,
        "samples": frozen_array(np.asarray(raw.eeg_uv[:stop], dtype=float)*1e-6), "selections": selections,
        "conditioning_samples": conditioning, "binding": binding}.items():
        object.__setattr__(result, key, value)
    return result


def score_finite_c_history(model, capability):
    """Read-only actual caller: C state weights update on the future prefix."""
    if not isinstance(capability, AuthorizedFiniteCHistory) or capability.model_digest != model.digest():
        raise ValueError("finite continuation requires actual unchanged producer/history authority")
    c_names = tuple(s["selection_id"] for s in model.binding["fit_authority"]["C_selections"])
    c_selections = tuple(next(s for s in capability.selections if s.selection_id == name) for name in c_names)
    future = tuple(s for s in capability.selections if s.selection_id not in c_names)
    current_axis = tuple(s.selection_id for s in future).index(capability.view.decision_id)
    states = model.protocol.states
    leading = model.protocol.record["conditioning_samples"]
    primitive = _program(model.prior, model.dictionary, model.noise, model.innovation_covariance,
        capability.samples, leading, transform=model.posteriors[0].flat_transform,
        factor=model.prior_factor, factor_source_error=model.prior_factor_source_error)
    axes = [tuple(product(range(len(s.candidate_ids)), range(len(states)))) for s in future]
    indices, current, logs, pairs, c_errors = [], [], [], [], []
    cache = {}
    for component_id, component in enumerate(model.components):
        posterior = model.posteriors[component["posterior_index"]]
        c_design = _design(model.dictionary, c_selections, component["target_indices"],
            tuple(states[i] for i in component["response_state_indices"]), observed_samples=len(capability.samples))
        for state in product(*axes):
            targets, response = tuple(x[0] for x in state), tuple(x[1] for x in state)
            design = c_design+_design(model.dictionary, future, targets, tuple(states[i] for i in response),
                                      observed_samples=len(capability.samples))
            key = (component["posterior_index"], design.tobytes())
            if key not in cache:
                cache[key] = primitive.continue_posterior(design, posterior,
                    conditioning_samples=capability.conditioning_samples)
            pairs.append(cache[key])
            c_errors.append(posterior.resolution["C"]["absolute_log_density_error_estimate"])
            indices.append((component_id, *targets, *response))
            current.append(targets[current_axis])
            logs.append(component["log_posterior_weight"]+sum(np.log(states[i].probability)-np.log(len(s.candidate_ids))
                for i, s in zip(response, future, strict=True)))
    evidence = finite_conditional_mixture(capability.view.decision.candidate_vocabulary, indices, current,
        logs, pairs, tail_scalar_rows=(len(capability.samples)-capability.conditioning_samples)*capability.samples.shape[1],
        c_logmass_errors=c_errors, scoring_law=capability.binding["scoring_law"])
    binding = freeze_metadata({"schema": "n2p3_finite_c_history_prediction/1", "model_digest": model.digest(),
        "capability_digest": semantic_sha256(capability.binding), "consumer_source_sha256": IMPLEMENTATION_SHA256,
        "state_indices": indices, "C_log_weights": evidence.state_c_log_weights.tolist(),
        "prefix_log_weights": evidence.state_prefix_log_weights.tolist(),
        "full_log_weights": evidence.state_full_log_weights.tolist(),
        "state_weight_conditioning": "each_requested_current_q",
        "whole_observed_joint_state_log_weights": (evidence.state_full_log_weights
            +evidence.log_whole_observed_current_probability[evidence.current_candidate_indices]).tolist(),
        "log_evidence": evidence.conditional_log_likelihood.tolist(),
        "scoring_law": evidence.program,
        "log_prefix_evidence_by_current": evidence.log_prefix_evidence_by_current.tolist(),
        "log_prefix_current_probability": evidence.log_prefix_current_probability.tolist(),
        "log_whole_observed_current_probability": evidence.log_whole_observed_current_probability.tolist(),
        "log_probability": evidence.log_candidate_probability.tolist(), "mixture_resolution": evidence.mixture_resolution,
        "component_measure": "normalized_joint_C_posterior_same_record_ordinary_increment_density",
        "current_q_scoring_measure": ("same_full_prefix_increment_ratio_fixed_current_prior"
            if evidence.program == "fixed_current_prior_conditional_tail" else "whole_observed_full_increment_current_q_posterior"),
        "current_schedule_complete": capability.binding["current_schedule_complete"],
        "current_response_right_censored": capability.binding["current_response_right_censored"],
        "C_refits": 0, "noise_fits": 0, "new_prior_applications": 0, "AR_restarts_at_selection_boundaries": 0,
        "unique_numerical_components": len(cache), "evaluated_positive_states": len(indices)})
    return evidence, binding
