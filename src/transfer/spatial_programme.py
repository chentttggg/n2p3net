"""Actual source/C builders and a frozen-noise physical ERP response programme.

The existing continuous fitter owns the response-dependent noise estimation.
Its complete producer state stays a named dependency. This programme fits the
new physical response once under that frozen noise and scores only unlabeled
PredictionDecisionView inputs; it is an explicit plugin, not joint MLE.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from data.candidate_task import MembershipDecision
from data.identity import DatasetIdentityTable, assert_target_identity_excluded, origin_subject_key
from data.study import study_ids_for_identity
from research.execution import atomic_write_json
from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    ContinuousDecisionResult,
    _event_design,
)
from transfer.ensemble_views import CalibrationSelection, PredictionDecisionView
from transfer.noise_model import (
    array_digest,
    finite_real_array,
    freeze_metadata,
    frozen_array,
    metadata_digest,
)
from transfer.response_dictionary import ResponseDictionary
from transfer.spatial_signal_prior import (
    ERPChannelEstimate,
    ERPPhysicalEffect,
    SourceGroupPolicy,
    SourceGroupPrior,
    SpatialLayout,
    SpatialLikelihoodDependence,
    SpatialObservation,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _layout(receipt):
    geometry = {key: receipt[key] for key in ("channel_names", "channel_types", "reference", "signal_unit",
                                             "source_dataset_id", "source_subject_id", "participant_key")}
    return SpatialLayout(tuple(receipt["channel_names"]), receipt["reference"], metadata_digest(geometry),
                         tuple(receipt["channel_types"]), receipt["signal_unit"])


def _observation(layout, mode, reference):
    if mode == "native":
        return SpatialObservation.native(layout)
    if mode == "car":
        return SpatialObservation.average_reference(layout)
    return SpatialObservation.measured_reference(layout, reference)


def _temporal_logdet(noise, rows):
    if noise.startup_mode == "stationary_innovations":
        return float(np.log(np.diag(noise.startup_whitening_factor)[:min(noise.order, rows)]).sum())
    if noise.startup_mode == "stationary":
        return -.5 * float(np.linalg.slogdet(noise.startup_covariance[:min(noise.order, rows), :min(noise.order, rows)])[1])
    raise ValueError("physical response programme requires a complete temporal noise map")


def _fit_controller_receipt(record, receipt):
    """Admit the two original fit capabilities without inventing a decision."""
    if "label_use" in receipt:
        use = receipt["label_use"]["raw"]
        if (not record.right_censored or not use["right_censored"]
                or use["used_labels"] != len(record.target_mask)
                or use["requested_labels"] != len(record.target_mask)
                or list(use["read_sample_support"]) != list(receipt["raw_scope"])
                or receipt["raw_input_digest"] != array_digest(record.samples, record.onsets, record.target_mask)):
            raise ValueError("physical prefix controller differs from its original authorized raw input")
        return {**receipt, "schema": "n2p3_spatial_prefix_fit_controller/1", "onsets": record.onsets.tolist()}
    if (receipt["schema"] != "n2p3_prediction_decision_physical_support/1"
            or not np.array_equal(record.onsets, receipt["onsets"]) or record.right_censored):
        raise ValueError("physical group controller differs from its original complete decision")
    return receipt


def _response_block(samples, design, observation, baseline, receipt, *, scope, original_record=None):
    """The same full-record W acts once on actual Y/D/Z, before separation."""
    times = np.arange(len(samples))
    prefix = baseline.conditioning_samples
    trend = (2 * times - (prefix - 1)) / (prefix - 1) if prefix else np.linspace(-1., 1., len(times))
    nuisance = np.column_stack((np.ones(len(times)), trend))
    transformed = baseline.noise.whiten(np.column_stack((samples, design, nuisance)))
    channels, width = len(observation.layout.channels), design.shape[1]
    authority = {"record_id": receipt["record_id"], "fit_scope": scope,
        "source_sha256": receipt["source_sha256"], "raw_sample_support": list(receipt["raw_scope"]),
        "dictionary_sha256": baseline.dictionary.digest(),
        "temporal_program_sha256": baseline.noise.record()["artifact_sha256"], "original_view": receipt}
    if scope != "read_only_prediction":
        if not isinstance(original_record, CalibrationRecord) or original_record.record_id != receipt["record_id"]:
            raise TypeError("physical fit whitening requires its actual original labeled record")
        authority["original_raw_fit_input_sha256"] = array_digest(original_record.samples, original_record.onsets,
                                                                  original_record.target_mask)
        authority["original_fit_labels_sha256"] = array_digest(original_record.target_mask)
        authority["original_fit_onsets_sha256"] = array_digest(original_record.onsets)
        authority["original_fit_target_count"] = int(np.sum(original_record.target_mask))
        authority["right_censored"] = original_record.right_censored
    return SpatialResponseBlock(observation.project(transformed[:, :channels]), transformed[:, channels:channels+width],
        observation, authority, transformed[:, channels+width:], _temporal_logdet(baseline.noise, len(transformed)))


def _noise_dependency_bytes(model):
    """Use the original continuous producer codec; no state is reconstructed."""
    return model.to_bytes()


def declared_spatial_effect(layout, dictionary, *, observation, reference_channel=None, waveform_weights=None):
    """Predeclare the common ERP functional using geometry, never target EEG."""
    if not isinstance(layout, SpatialLayout) or not isinstance(dictionary, ResponseDictionary):
        raise TypeError("common source effect requires owned physical geometry and response dictionary")
    width = 2*dictionary.matrix.shape[1]
    prior = SpatialSignalPrior(layout, np.zeros((width, len(layout.channels))), np.eye(width),
        np.eye(len(layout.channels)),
        {"dictionary_sha256": dictionary.digest(), "actual_basis_sha256": array_digest(dictionary.matrix),
         "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": dictionary.sample_rate_hz,
         "operator_identity": "native_unfiltered_volts/1", "coefficient_groups": ["common", "target_increment"]},
        {"role": "predeclared", "authority_sha256": metadata_digest({"layout": layout.record(), "dictionary": dictionary.digest()})})
    count = len(dictionary.matrix)
    weights = np.full(count, 1/count) if waveform_weights is None else waveform_weights
    return ERPPhysicalEffect.from_prior(prior, basis=dictionary.matrix, waveform_weights=weights,
        grid_seconds=np.arange(count)/dictionary.sample_rate_hz,
        observation=_observation(layout, observation, reference_channel))


@dataclass(frozen=True)
class SpatialSignalProgram:
    native_signal_variance: float
    observation: str = "native"
    reference_channel: str | None = None
    temporal_covariance: np.ndarray | None = field(default=None, repr=False, compare=False)
    source_group: SourceGroupPrior | None = field(default=None, repr=False, compare=False)
    completion: str = "separable"

    def __post_init__(self):
        if (isinstance(self.native_signal_variance, bool) or not np.isscalar(self.native_signal_variance)
                or not np.isfinite(self.native_signal_variance)
                or self.native_signal_variance <= 0 or self.observation not in {"native", "car", "measured_reference"}
                or self.completion not in {"mean_only", "separable"}
                or self.observation == "measured_reference" and not self.reference_channel):
            raise ValueError("physical signal programme needs explicit variance, observation and completion")
        if self.temporal_covariance is not None:
            value = finite_real_array(self.temporal_covariance, name="physical temporal signal covariance")
            if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
                raise ValueError("physical signal programme needs a finite temporal covariance")
            object.__setattr__(self, "temporal_covariance", frozen_array(value))
        if self.source_group is not None and not isinstance(self.source_group, SourceGroupPrior):
            raise TypeError("source signal programme requires an admitted source-group owner")
        if self.source_group is not None:
            _admit_source_cohort(self.source_group)

    def record(self):
        return {"schema": "n2p3_spatial_signal_programme/1", "native_signal_variance": self.native_signal_variance,
            "observation": self.observation, "reference_channel": self.reference_channel,
            "temporal_covariance": None if self.temporal_covariance is None else self.temporal_covariance.tolist(),
            "source_group": None if self.source_group is None else self.source_group.record(),
            "completion": self.completion, "noise_programme": "original_continuous_fit_then_freeze",
            "response_programme": "single_general_physical_prior_refit_given_original_noise"}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        if (values.pop("schema") != "n2p3_spatial_signal_programme/1"
                or values.pop("noise_programme") != "original_continuous_fit_then_freeze"
                or values.pop("response_programme") != "single_general_physical_prior_refit_given_original_noise"):
            raise ValueError("physical signal programme roles differ from their original declaration")
        group = values["source_group"]
        values["source_group"] = None if group is None else SourceGroupPrior.from_record(group,
            expected_artifact_sha256=group["artifact_sha256"])
        return cls(**values)

    def reference_prior(self, layout, baseline):
        p, channels = 2*baseline.basis.shape[1], len(layout.channels)
        temporal = np.eye(p) if self.temporal_covariance is None else self.temporal_covariance
        return SpatialSignalPrior(layout, np.zeros((p, channels)), temporal,
            self.native_signal_variance*np.eye(channels),
            {"dictionary_sha256": baseline.dictionary.digest(), "actual_basis_sha256": array_digest(baseline.basis),
             "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": baseline.sfreq,
             "operator_identity": baseline.operator_identity, "coefficient_groups": ["common", "target_increment"]},
            {"role": "predeclared", "authority_sha256": metadata_digest(self.record())})

    def prior(self, layout, baseline, participant):
        prior = self.reference_prior(layout, baseline)
        if self.source_group is not None:
            effect = ERPPhysicalEffect.from_prior(prior, basis=baseline.basis,
                waveform_weights=self.source_group.effect.waveform_weights,
                grid_seconds=self.source_group.effect.grid_seconds,
                observation=SpatialObservation.from_record({**self.source_group.effect.observation.record(),
                    "layout": layout.record()}))
            prior = self.source_group.spatial_prior(prior, effect=effect, completion=self.completion,
                                                    target_participant_key=participant)
        return prior

    def validate_prior(self, prior, baseline):
        if self.source_group is None:
            self._validate_reference(prior, baseline)
        else:
            if prior.source["source_learning"]["completion"] != self.completion:
                raise ValueError("physical response prior differs from its declared source completion")
            witness = prior.source["source_learning"]["completion_witness"]
            reference = SpatialSignalPrior.from_record(witness["reference_prior"])
            self._validate_reference(reference, baseline)
            self.source_group.validate_spatial_prior(prior, reference=reference)

    def _validate_reference(self, prior, baseline):
        width = 2*baseline.basis.shape[1]
        temporal = np.eye(width) if self.temporal_covariance is None else self.temporal_covariance
        binding = {"dictionary_sha256": baseline.dictionary.digest(), "actual_basis_sha256": array_digest(baseline.basis),
            "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": baseline.sfreq,
            "operator_identity": baseline.operator_identity, "coefficient_groups": ["common", "target_increment"]}
        if (prior.layout.channels != baseline.channels or prior.mean.shape != (width, len(baseline.channels)) or np.any(prior.mean != 0)
                or not np.array_equal(prior.temporal_covariance, temporal)
                or not np.array_equal(prior.signal_covariance, self.native_signal_variance*np.eye(len(baseline.channels)))
                or metadata_digest(prior.design_binding) != metadata_digest(binding)
                or prior.source.get("role") != "predeclared"
                or prior.source.get("authority_sha256") != metadata_digest(self.record())):
            raise ValueError("physical response prior differs from its actual primitive signal programme")


@dataclass(frozen=True)
class SpatialDecisionModel:
    programme: SpatialSignalProgram
    baseline: ContinuousDecisionModel
    posterior: SpatialSignalPosterior
    baseline_content: bytes = field(repr=False, compare=False)
    provenance: dict

    def __post_init__(self):
        if not isinstance(self.programme, SpatialSignalProgram) or not isinstance(self.baseline, ContinuousDecisionModel):
            raise TypeError("physical response model requires owned programme and original noise dependency")
        if (self.baseline.noise is None or self.baseline.noise.startup_mode == "conditional_prefix"
                or self.posterior.noise_binding["original_continuous_model_sha256"] != self.baseline.digest()
                or self.posterior.noise_binding["authority_sha256"] != self.baseline.digest()
                or self.posterior.noise_binding["programme"] != "original_fit_noise_fixed_single_response_refit"
                or self.posterior.noise_binding["original_continuous_file_sha256"] != hashlib.sha256(self.baseline_content).hexdigest()
                or not np.array_equal(self.posterior.native_noise_covariance, self.baseline.innovation_covariance)):
            raise ValueError("physical response posterior differs from its original frozen noise programme")
        if metadata_digest(self.provenance["signal_programme"]) != metadata_digest(self.programme.record()):
            raise ValueError("physical response model declaration differs from its numerical programme")
        group = self.programme.source_group
        if (self.provenance["source_group_artifact_sha256"] != (None if group is None else group.record()["artifact_sha256"])
                or self.provenance["source_cohort_authority_sha256"] != (None if group is None else
                    group.fit_binding["source_authority"]["authority_sha256"])):
            raise ValueError("physical response model source influence differs from its original cohort programme")
        if any(record["authorization"]["temporal_program_sha256"] != self.baseline.noise.record()["artifact_sha256"]
               for record in self.posterior.fit_records):
            raise ValueError("physical response records differ from the original fitted temporal noise")
        self.programme.validate_prior(self.posterior.prior, self.baseline)
        if (self.provenance["physical_response_artifact_sha256"] != self.posterior.record()["artifact_sha256"]
                or self.provenance["response_fit_passes_after_noise_freeze"] != 1
                or self.provenance["noise_refitted_after_freeze"] is not False
                or self.provenance["joint_spatial_noise_response_MLE_claimed"] is not False
                or self.provenance["kernel_uncertainty"] != "general_vec_Gaussian_given_original_fitted_noise"
                or metadata_digest(self.provenance["records"]) != metadata_digest(self.baseline.provenance["records"])
                or self.provenance["fit_scope"] != self.baseline.provenance["fit_scope"]):
            raise ValueError("physical response programme differs from its actual fixed-noise producer roles")
        if len(self.posterior.fit_records) != len(self.baseline.provenance["records"]):
            raise ValueError("physical response record roster differs from its original noise fit")
        for physical, original in zip(self.posterior.fit_records, self.baseline.provenance["records"], strict=True):
            authority = physical["authorization"]
            view = authority["original_view"]
            if (authority["record_id"] != original["record_id"] or authority["fit_scope"] != self.provenance["fit_scope"]
                    or authority["source_sha256"] != view["source_sha256"]
                    or list(authority["raw_sample_support"]) != list(view["raw_scope"])
                    or original["n_samples"] != view["raw_scope"][1]-view["raw_scope"][0]
                    or original["n_events"] != len(view["onsets"])
                    or authority["original_fit_onsets_sha256"] != array_digest(np.asarray(view["onsets"], dtype=np.int64))
                    or original["n_targets"] != authority["original_fit_target_count"]
                    or original["right_censored"] != authority["right_censored"]
                    or authority["original_raw_fit_input_sha256"] != original["raw_input_sha256"]):
                raise ValueError("physical response source/C records differ from original controller/noise authority")
            if view["schema"] == "n2p3_spatial_prefix_fit_controller/1":
                use = view["label_use"]["raw"]
                if (view["raw_input_digest"] != authority["original_raw_fit_input_sha256"]
                        or not authority["right_censored"] or not use["right_censored"]
                        or use["used_labels"] != original["n_events"]
                        or use["requested_labels"] != original["n_events"]
                        or list(use["read_sample_support"]) != list(view["raw_scope"])):
                    raise ValueError("physical prefix fit differs from the original controller label/read authority")
            if metadata_digest(_layout(view).record()) != metadata_digest(self.posterior.prior.layout.record()):
                raise ValueError("physical response layout differs from actual original controller authority")
            if self.programme.source_group is not None:
                _assert_forecast_target(self.programme.source_group, view)
                if self.posterior.prior.source["source_learning"]["target_participant_key"] != view["participant_key"]:
                    raise ValueError("source prior target differs from actual target C participant")
        expected_kind = {"native": "native_EEG", "car": "average_actual_EEG", "measured_reference": "measured_reference"}[self.programme.observation]
        for factor in self.posterior.structural_factors:
            observed = factor.observation
            if (observed.declaration["kind"] != expected_kind
                    or any(value != 1. for value in observed.declaration["row_gains"])
                    or expected_kind == "measured_reference" and observed.declaration["reference_channel"] != self.programme.reference_channel):
                raise ValueError("physical response observation differs from its declared signal programme")
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))

    @property
    def dictionary(self):
        return self.baseline.dictionary

    @property
    def basis(self):
        return self.baseline.basis

    @property
    def sfreq(self):
        return self.baseline.sfreq

    @property
    def channels(self):
        return self.baseline.channels

    @property
    def operator_identity(self):
        return self.baseline.operator_identity

    @property
    def noise_order(self):
        return self.baseline.noise_order

    @property
    def conditioning_samples(self):
        return self.baseline.conditioning_samples

    @property
    def kernel_prior(self):
        return self.baseline.kernel_prior

    @classmethod
    def fit(cls, records, receipts, programme, *, sfreq, channels, candidate_ids, operator_identity,
            dictionary, fit_scope="target_calibration", **noise_options):
        records, receipts = tuple(records), tuple(receipts)
        if noise_options.get("noise_order") is None:
            raise ValueError("physical response programme requires an explicit complete TemporalNoise order")
        if len(records) != len(receipts) or not records:
            raise ValueError("physical response fit needs each actual record and its original controller receipt")
        receipts = tuple(_fit_controller_receipt(record, receipt) for record, receipt in zip(records, receipts, strict=True))
        for record, receipt in zip(records, receipts, strict=True):
            if (record.record_id != receipt["record_id"] or len(record.samples) != receipt["raw_scope"][1]-receipt["raw_scope"][0]
                    or tuple(receipt["channel_names"]) != tuple(channels) or receipt["sample_rate_hz"] != sfreq):
                raise ValueError("physical response records differ from actual sampling/support authority")
        layouts = [_layout(receipt) for receipt in receipts]
        if any(layout.record() != layouts[0].record() for layout in layouts):
            raise ValueError("one physical response fit requires the same measured layout/participant")
        if programme.source_group is not None:
            for receipt in receipts:
                _assert_forecast_target(programme.source_group, receipt)
        baseline = ContinuousDecisionModel.fit(records, sfreq=sfreq, channels=channels, candidate_ids=candidate_ids,
            operator_identity=operator_identity, dictionary=dictionary, fit_scope=fit_scope, **noise_options)
        content = _noise_dependency_bytes(baseline)
        prior = programme.prior(layouts[0], baseline, receipts[0]["participant_key"])
        observation = _observation(layouts[0], programme.observation, programme.reference_channel)
        blocks = [_response_block(record.samples, _event_design(record.onsets, record.target_mask,
            len(record.samples), baseline.basis), observation, baseline, receipt, scope=fit_scope, original_record=record)
            for record, receipt in zip(records, receipts, strict=True)]
        posterior = SpatialSignalPosterior.fit(prior, blocks, native_noise_covariance=baseline.innovation_covariance,
            noise_binding={"authority_sha256": baseline.digest(), "original_continuous_model_sha256": baseline.digest(),
                "original_continuous_file_sha256": hashlib.sha256(content).hexdigest(),
                "programme": "original_fit_noise_fixed_single_response_refit"})
        provenance = {**baseline.provenance, "signal_programme": programme.record(),
            "source_group_artifact_sha256": None if programme.source_group is None else programme.source_group.record()["artifact_sha256"],
            "source_cohort_authority_sha256": None if programme.source_group is None else
                programme.source_group.fit_binding["source_authority"]["authority_sha256"],
            "physical_response_artifact_sha256": posterior.record()["artifact_sha256"],
            "kernel_uncertainty": "general_vec_Gaussian_given_original_fitted_noise",
            "response_fit_passes_after_noise_freeze": 1, "noise_refitted_after_freeze": False,
            "joint_spatial_noise_response_MLE_claimed": False}
        return cls(programme, baseline, posterior, content, provenance)

    def score_view(self, view):
        if not isinstance(view, PredictionDecisionView) or view.samples is None:
            raise TypeError("physical response scoring consumes actual unlabeled prediction views")
        layout = self.posterior.prior.layout
        if (view.channel_names != layout.channels or view.reference != layout.acquisition_reference
                or view.signal_unit != layout.signal_unit or view.sample_rate_hz != self.sfreq
                or view.decision.candidate_vocabulary != self.baseline.candidate_ids):
            raise ValueError("physical response prediction differs from its frozen observation/task")
        observation = self.posterior.structural_factors[0].observation
        evidence = []
        for labels in view.decision.membership.T:
            block = _response_block(view.samples, _event_design(view.onsets, labels, len(view.samples), self.basis),
                observation, self.baseline, view.record(), scope="read_only_prediction")
            if self.conditioning_samples:
                count = self.conditioning_samples
                auth = json.loads(json.dumps(block.authorization))
                auth["raw_sample_support"][1] = auth["raw_sample_support"][0] + count
                prefix = SpatialResponseBlock(block.samples[:count], block.design[:count], observation,
                    auth, block.nuisance[:count], _temporal_logdet(self.baseline.noise, count))
                evidence.append(self.posterior.conditional_log_density(block, prefix))
            else:
                evidence.append(self.posterior.log_density(block))
        evidence = np.asarray(evidence)
        probability = np.exp(evidence-logsumexp(evidence))
        maxima = np.flatnonzero(evidence == evidence.max())
        winner = view.decision.candidate_vocabulary[int(maxima[0])] if len(maxima) == 1 else None
        return ContinuousDecisionResult(view.decision.candidate_vocabulary, frozen_array(evidence), frozen_array(probability),
            winner, "unique_maximum" if winner else "tie", len(view.samples), view.raw_input_digest, self.conditioning_samples)

    def record(self):
        payload = {"schema": "n2p3_spatial_decision_programme/1", "implementation_sha256": IMPLEMENTATION_SHA256,
            "signal_programme": self.programme.record(), "original_noise_dependency": base64.b64encode(self.baseline_content).decode("ascii"),
            "posterior": self.posterior.record(), "provenance": self.provenance}
        return {**payload, "artifact_sha256": metadata_digest(payload)}

    def digest(self):
        return self.record()["artifact_sha256"]

    def save(self, path):
        return atomic_write_json(path, self.record())

    @classmethod
    def load(cls, path):
        content = path.read() if hasattr(path, "read") else Path(path).read_bytes()
        saved = json.loads(content)
        if (saved["schema"] != "n2p3_spatial_decision_programme/1"
                or saved["implementation_sha256"] != IMPLEMENTATION_SHA256
                or metadata_digest({key:value for key,value in saved.items() if key != "artifact_sha256"}) != saved["artifact_sha256"]):
            raise ValueError("physical response programme artifact differs from its original producer")
        dependency = base64.b64decode(saved["original_noise_dependency"], validate=True)
        model = cls(SpatialSignalProgram.from_record(saved["signal_programme"]),
            ContinuousDecisionModel.load(io.BytesIO(dependency)), SpatialSignalPosterior.from_record(saved["posterior"]),
            dependency, saved["provenance"])
        if metadata_digest(model.record()) != metadata_digest(saved):
            raise ValueError("physical response programme replay differs from original state")
        return model


def _identity_for_view(identity, view):
    origin = origin_subject_key(view.source_dataset_id, view.source_subject_id)
    matching = [record for record in identity.records if origin in record.origin_subject_keys]
    if len(matching) != 1 or view.participant_key not in (*matching[0].origin_subject_keys, *matching[0].global_person_keys):
        raise ValueError("actual raw source participant differs from shared identity authority")
    return matching[0]


def _assert_forecast_target(group, receipt):
    authority = group.fit_binding["source_authority"]
    target = DatasetIdentityTable.from_payload(json.loads(json.dumps(authority["target_identity"])))
    origin = origin_subject_key(receipt["source_dataset_id"], receipt["source_subject_id"])
    if not any(origin in record.origin_subject_keys and receipt["participant_key"] in
               (*record.origin_subject_keys, *record.global_person_keys) for record in target.records):
        raise ValueError("source forecast target differs from the independently excluded target identity")


def _admit_source_cohort(group):
    """One new capability join for builder and saved source-programme ingress."""
    authority = json.loads(json.dumps(group.fit_binding["source_authority"]))
    if authority.get("schema") != "n2p3_spatial_source_cohort/1":
        raise ValueError("source signal programme requires actual shared-identity/study cohort authority")
    fields = ("source_identity", "source_study_membership", "target_identity", "target_study_membership",
              "outer_excluded_studies", "participant_units", "original_source_records", "source_programme_digests",
              "source_programme_receipts", "source_measurements")
    cohort = {key: authority[key] for key in fields}
    if authority["authority_sha256"] != metadata_digest(cohort):
        raise ValueError("source cohort authority differs from its actual controller/measurement programme")
    identity = DatasetIdentityTable.from_payload(cohort["source_identity"])
    target = DatasetIdentityTable.from_payload(cohort["target_identity"])
    source_studies = study_ids_for_identity(identity, cohort["source_study_membership"])
    target_studies = study_ids_for_identity(target, cohort["target_study_membership"])
    if set(source_studies) & set(target_studies) or tuple(cohort["outer_excluded_studies"]) != target_studies:
        raise ValueError("source cohort includes an independently excluded outer study")
    for record in target.records:
        assert_target_identity_excluded(identity, record)
    units = []
    controller_records = cohort["original_source_records"]
    measurements = cohort["source_measurements"]
    if (len(controller_records) != len(measurements) or len(measurements) != len(group.measurements)
            or len(cohort["source_programme_digests"]) != len(measurements)
            or len(cohort["source_programme_receipts"]) != len(measurements)
            or authority["source_fit_counts"] != {"participant_programmes": len(measurements),
                "original_continuous_fits": len(measurements), "physical_response_refits_given_original_noise": len(measurements)}):
        raise ValueError("source cohort changed the requested real-participant programme roster")
    for original, measurement, programme_digest, programme_receipt, binding, mean, covariance in zip(controller_records, measurements,
        cohort["source_programme_digests"], cohort["source_programme_receipts"],
        group.fit_binding["source_estimates"], group.measurements, group.measurement_covariances, strict=True):
        receipt = measurement["receipt"]
        actual_records = receipt["actual_C"]
        if (measurement["artifact_sha256"] != metadata_digest({key:value for key,value in measurement.items() if key != "artifact_sha256"})
                or measurement["artifact_sha256"] != binding["artifact_sha256"]
                or len(original) != len(actual_records) or not original
                or receipt["participant_key"] != binding["participant_key"] or receipt["fit_scope"] != "source_training"
                or programme_digest != programme_receipt["model_digest"]
                or programme_receipt["posterior_artifact_sha256"] != receipt["posterior_artifact_sha256"]
                or programme_receipt["posterior_artifact_sha256"] != binding["posterior_artifact_sha256"]
                or programme_receipt["temporal_noise_sha256"] != receipt["temporal_noise_sha256"]
                or programme_receipt["native_noise_sha256"] != receipt["native_noise_sha256"]
                or programme_receipt["original_continuous_model_sha256"] != receipt["original_noise_programme"]["original_continuous_model_sha256"]
                or programme_receipt["original_continuous_file_sha256"] != receipt["original_noise_programme"]["original_continuous_file_sha256"]
                or programme_receipt["spatial_implementation_sha256"] != receipt["producer_implementation_sha256"]
                or programme_receipt["spatial_implementation_sha256"] != binding["producer_implementation_sha256"]
                or metadata_digest(measurement["effect"]["definition"]) != metadata_digest(group.effect.definition)
                or not np.array_equal(mean, measurement["mean"])):
            raise ValueError("source cohort measurement differs from its actual participant/physical effect")
        factor = np.asarray(measurement["covariance_factor"])
        SourceGroupPrior._generated_relation(covariance, factor@factor.T, np.abs(factor)@np.abs(factor.T),
            2*factor.shape[1], "source measurement covariance")
        participant_units = []
        for record, actual in zip(original, actual_records, strict=True):
            view = record["view"]
            decision = MembershipDecision.from_record(view["decision"])
            target = decision.candidate_vocabulary.index(record["target_candidate"])
            labels = decision.membership[:, target]
            if (record["schema"] != "n2p3_calibration_selection/1"
                    or record["authorized_label_count"] != len(labels)
                    or record["labels_sha256"] != array_digest(labels)
                    or record["labels_sha256"] != actual["authorization"]["original_fit_labels_sha256"]
                    or int(labels.sum()) != actual["authorization"]["original_fit_target_count"]):
                raise ValueError("source cohort calibration labels differ from actual original response fit")
            origin = origin_subject_key(view["source_dataset_id"], view["source_subject_id"])
            matching = [person for person in identity.records if origin in person.origin_subject_keys]
            if (len(matching) != 1 or view["participant_key"] not in (*matching[0].origin_subject_keys, *matching[0].global_person_keys)
                    or metadata_digest(view) != metadata_digest(actual["authorization"]["original_view"])
                    or view["participant_key"] != receipt["participant_key"]
                    or actual["authorization"]["record_id"] != view["record_id"]
                    or actual["authorization"]["source_sha256"] != view["source_sha256"]
                    or list(actual["authorization"]["raw_sample_support"]) != view["raw_scope"]
                    or actual["authorization"]["fit_scope"] != "source_training"):
                raise ValueError("source cohort raw controller records differ from shared identity/fit authority")
            participant_units.append(matching[0].authority_key("source_or_global"))
        if len(set(participant_units)) != 1:
            raise ValueError("source population measurement mixed independent participants")
        units.append(participant_units[0])
    if tuple(units) != tuple(cohort["participant_units"]) or set(units) != set(identity.authority_keys()) or len(set(units)) != len(units):
        raise ValueError("source cohort participant-unit roster differs from shared identity authority")


def fit_spatial_source_group(participants, *, identity, study_membership, target_identity, target_membership,
                             programme, dictionary, common_effect, policy, **noise_options):
    """Actual source records → original fitted noise → BLUE → participant group law."""
    if not isinstance(identity, DatasetIdentityTable) or not isinstance(target_identity, DatasetIdentityTable):
        raise TypeError("spatial source builder requires shared source/target identity owners")
    if programme.source_group is not None or not isinstance(dictionary, ResponseDictionary) or not isinstance(policy, SourceGroupPolicy):
        raise ValueError("source builder requires a predeclared physical reference and external group policy")
    source_studies = study_ids_for_identity(identity, study_membership)
    target_studies = study_ids_for_identity(target_identity, target_membership)
    if set(source_studies) & set(target_studies):
        raise ValueError("spatial source programme includes the outer target study")
    for target in target_identity.records:
        assert_target_identity_excluded(identity, target)
    participants = tuple(tuple(groups) for groups in participants)
    if not participants or any(not groups or any(not isinstance(group, CalibrationSelection) for group in groups)
                               for groups in participants):
        raise TypeError("spatial source builder consumes authorized selection records per real participant")
    keys, source_models, estimates = [], [], []
    for groups in participants:
        actual = [_identity_for_view(identity, group.view) for group in groups]
        key = actual[0].authority_key("source_or_global")
        if any(record.authority_key("source_or_global") != key for record in actual) or key in keys:
            raise ValueError("source population units must be distinct participants, not duplicated records")
        keys.append(key)
        first = groups[0].view
        records = [CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels) for group in groups]
        model = SpatialDecisionModel.fit(records, [group.view.record() for group in groups], programme,
            sfreq=first.sample_rate_hz, channels=first.channel_names, candidate_ids=first.decision.candidate_vocabulary,
            operator_identity="native_unfiltered_volts/1", dictionary=dictionary, fit_scope="source_training", **noise_options)
        source_effect = ERPPhysicalEffect.from_prior(model.posterior.prior, basis=model.basis,
            waveform_weights=common_effect.waveform_weights, grid_seconds=common_effect.grid_seconds,
            observation=SpatialObservation.native(model.posterior.prior.layout))
        dependency = SpatialLikelihoodDependence.independent(model.posterior,
            authority_sha256=metadata_digest({"original_model": model.digest(), "law": "independent_record_errors"}))
        measurement = ERPChannelEstimate.from_transported_posterior(model.posterior, source_effect=source_effect,
            destination_effect=common_effect, dependence=dependency, temporal_noise=model.baseline.noise)
        source_models.append(model)
        estimates.append(measurement)
    cohort = {"source_identity": identity.payload(), "source_study_membership": study_membership,
        "target_identity": target_identity.payload(), "target_study_membership": target_membership,
        "outer_excluded_studies": target_studies, "participant_units": keys,
        "original_source_records": [[group.record() for group in groups] for groups in participants],
        "source_programme_digests": [model.digest() for model in source_models],
        "source_programme_receipts": [{"model_digest": model.digest(),
            "posterior_artifact_sha256": measurement.receipt["posterior_artifact_sha256"],
            "original_continuous_model_sha256": model.baseline.digest(),
            "original_continuous_file_sha256": hashlib.sha256(model.baseline_content).hexdigest(),
            "temporal_noise_sha256": measurement.receipt["temporal_noise_sha256"],
            "native_noise_sha256": measurement.receipt["native_noise_sha256"],
            "spatial_implementation_sha256": measurement.receipt["producer_implementation_sha256"],
            "source_builder_implementation_sha256": IMPLEMENTATION_SHA256}
            for model, measurement in zip(source_models, estimates, strict=True)],
        "source_measurements": [measurement.record() for measurement in estimates]}
    authority = {"schema": "n2p3_spatial_source_cohort/1", "fit_scope": "source_training", "independent_unit": "participant",
        "participant_keys": [item.receipt["participant_key"] for item in estimates],
        "source_fit_counts": {"participant_programmes": len(source_models), "original_continuous_fits": len(source_models),
                              "physical_response_refits_given_original_noise": len(source_models)},
        **cohort, "authority_sha256": metadata_digest(cohort)}
    group = SourceGroupPrior.fit(estimates, policy=policy, source_authority=authority)
    _admit_source_cohort(group)
    return group, tuple(source_models), tuple(estimates)
