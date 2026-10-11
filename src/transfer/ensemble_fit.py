"""Fit real calibration experts and freeze their label-free candidate scoring."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from data.identity import DatasetIdentityTable, source_dataset_ids_from_identity
from models.candidate_evidence import (
    CandidateEvidence,
    CandidateExpertIdentity,
    membership_decision_scores,
)
from research.cloud_execution import worker_compute_phase
from research.contracts import semantic_sha256
from research.execution import (
    ExpectedSubjectError,
    SubjectFailureCode,
    UnidentifiableFitError,
    atomic_write_json,
)
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.calibration_state import FrozenTrialPredictor
from transfer.checkpoint import ScopedSourceCheckpointPreparation
from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
)
from transfer.ensemble import EnsembleProgram, EnsembleScorer
from transfer.ensemble_views import (
    CalibrationPrefix,
    CalibrationSelection,
    PredictionDecisionView,
    SelectionEpochProfile,
    build_calibration_epoch_dataset,
)
from transfer.kernel_prior import KernelPriorSpec, resolve_kernel_prior
from transfer.noise_model import array_digest, freeze_metadata
from transfer.recipes import TargetRecipe
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec

FITTED_ENSEMBLE_SCHEMA = "n2p3_fitted_candidate_ensemble/2"
GROUPED_PROBABILITY_OOF_SCHEMA = "n2p3_complete_group_probability_oof/1"


@dataclass(frozen=True)
class FrozenSourceCheckpoint:
    """One immutable set of source bytes reused throughout fit and OOF."""

    content: bytes

    def __post_init__(self):
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("frozen source checkpoint requires nonempty immutable bytes")

    @property
    def digest(self):
        return hashlib.sha256(self.content).hexdigest()

    @classmethod
    def from_path(cls, path):
        return cls(Path(path).read_bytes())


def _temperature_nll(log_temperature, log_probability, targets):
    values = log_probability / np.exp(log_temperature)
    return float(np.mean(logsumexp(values, axis=1) - values[np.arange(len(values)), targets]))


def temperature_from_oof(log_probability, targets):
    """Replay the original finite meta learner on captured C-only probabilities.

    No EEG fitter, future evidence, new state, or optimization policy is involved.
    """
    solution = minimize_scalar(_temperature_nll, args=(log_probability, targets),
                               bounds=(np.log(.05), np.log(20.)), method="bounded")
    if not solution.success or not np.isfinite(solution.fun):
        raise ExpectedSubjectError(SubjectFailureCode.NONFINITE_OPTIMIZATION, stage="temperature",
            detail="grouped temperature fit did not produce a finite declared calibration")
    return float(np.exp(solution.x))


def _native_observation_contract(view):
    return (view.sample_rate_hz, view.channel_names, view.channel_types, view.reference, view.signal_unit)


def _native_observation_contract_record(record):
    return {key: record[key] for key in
            ("sample_rate_hz", "channel_names", "channel_types", "reference", "signal_unit")}


@dataclass(frozen=True)
class CalibrationExpertProgram:
    expert_id: str
    family: str
    recipe: TargetRecipe | None = None
    profile: SelectionEpochProfile | None = None
    noise_order: int = 8
    ridge: float | None = None
    response_seconds: float | None = None
    n_basis: int | None = None
    conditioning_samples: int = 0
    noise_estimator: str = "burg"
    kernel_prior: KernelPriorSpec | None = None
    dictionary: ResponseDictionarySpec | None = None
    _prior_spec: KernelPriorSpec | None = field(init=False, repr=False, compare=False)
    _dictionary_spec: ResponseDictionarySpec | None = field(init=False, repr=False, compare=False)
    spatial: dict | None = None

    def __post_init__(self):
        if not self.expert_id or self.family not in {"trial", "continuous_ar"}:
            raise ValueError("calibration expert requires an explicit ID and supported family")
        if self.family == "trial" and (self.recipe is None or self.profile is None):
            raise ValueError("trial expert requires one target recipe and actual epoch profile")
        if self.spatial is not None:
            object.__setattr__(self, "spatial", freeze_metadata(self.spatial))
        if self.family == "continuous_ar" and (self.recipe is not None or self.profile is not None):
            raise ValueError("continuous expert consumes native raw, without an epoch recipe or profile")
        if self.family == "trial":
            if self.spatial is not None:
                raise ValueError("trial spatial programme belongs to its epoch profile")
            if self.kernel_prior is not None or self.dictionary is not None:
                raise ValueError("continuous kernel priors/dictionaries cannot be assigned to trial experts")
            object.__setattr__(self, "ridge", 1. if self.ridge is None else self.ridge)
            object.__setattr__(self, "_prior_spec", None)
            object.__setattr__(self, "_dictionary_spec", None)
            object.__setattr__(self, "response_seconds", 1.2 if self.response_seconds is None else self.response_seconds)
            object.__setattr__(self, "n_basis", 8 if self.n_basis is None else self.n_basis)
        else:
            if self.dictionary is not None:
                if (not isinstance(self.dictionary, ResponseDictionarySpec)
                        or (self.n_basis is not None and self.n_basis != self.dictionary.n_basis)
                        or (self.response_seconds is not None and self.response_seconds != self.dictionary.response_seconds)):
                    raise ValueError("owned dictionary conflicts with independent n_basis/response_seconds")
                dictionary_spec = self.dictionary
            else:
                dictionary_spec = ResponseDictionarySpec("gaussian", 1.2 if self.response_seconds is None else self.response_seconds,
                                                        8 if self.n_basis is None else self.n_basis)
            object.__setattr__(self, "_dictionary_spec", dictionary_spec)
            object.__setattr__(self, "n_basis", dictionary_spec.n_basis)
            object.__setattr__(self, "response_seconds", dictionary_spec.response_seconds)
            prior = resolve_kernel_prior(self.kernel_prior, self.ridge)
            if prior.uses_calibration_reference and not self.conditioning_samples:
                raise ValueError("calibration prior expert requires the fixed-prefix program")
            object.__setattr__(self, "_prior_spec", prior)
            object.__setattr__(self, "kernel_prior", prior if prior.uses_calibration_reference else None)
            object.__setattr__(self, "ridge", prior.precision)
        if (type(self.conditioning_samples) is not int or self.conditioning_samples < 0
                or self.noise_estimator not in {"burg", "conditional_profile"}):
            raise ValueError("expert requires a declared conditioning prefix and noise estimator")
        if self.family == "trial" and (self.conditioning_samples or self.noise_estimator != "burg"):
            raise ValueError("conditional raw likelihood settings cannot be assigned to a trial expert")
        if self.family == "continuous_ar" and (
            type(self.noise_order) is not int or not 0 <= self.noise_order <= 128
            or (self.conditioning_samples and self.conditioning_samples < max(2, self.noise_order))
            or (not self.conditioning_samples and self.noise_estimator != "burg")
        ):
            raise ValueError("raw conditioning prefix must cover its AR order and noise estimator")

    @property
    def prior_spec(self):
        return self._prior_spec

    @property
    def dictionary_spec(self):
        return self._dictionary_spec

    def record(self):
        record = {"expert_id": self.expert_id, "family": self.family,
                "recipe": None if self.recipe is None else self.recipe.record(),
                "profile": None if self.profile is None else self.profile.record(),
                "noise_order": self.noise_order, "ridge": self.ridge,
                "response_seconds": self.response_seconds, "n_basis": self.n_basis}
        if self.conditioning_samples:
            record.update(conditioning_samples=self.conditioning_samples, noise_estimator=self.noise_estimator)
        if self.kernel_prior is not None:
            record.pop("ridge")
            record["kernel_prior"] = self.kernel_prior.record()
        if self.dictionary is not None:
            record.pop("response_seconds")
            record.pop("n_basis")
            record["dictionary"] = self.dictionary.record()
        if self.spatial is not None:
            record["spatial"] = self.spatial
        return record

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        values["recipe"] = None if values["recipe"] is None else TargetRecipe(**values["recipe"])
        values["profile"] = None if values["profile"] is None else SelectionEpochProfile.from_record(values["profile"])
        if "kernel_prior" in values:
            values["kernel_prior"] = KernelPriorSpec.from_record(values["kernel_prior"])
        if "dictionary" in values:
            values["dictionary"] = ResponseDictionarySpec.from_record(values["dictionary"])
        return cls(**values)


@dataclass(frozen=True)
class _FittedExpert:
    program: CalibrationExpertProgram
    predictor: FrozenTrialPredictor | ContinuousDecisionModel | None
    model_digest: str | None
    fit_digest: str
    source_digest: str | None
    logit_offset: float
    unavailable_reason: str | None = None
    unavailable_kind: str = "not_available"

    def __post_init__(self):
        if self.predictor is None:
            if not self.unavailable_reason or self.model_digest is not None:
                raise ValueError("unavailable experts have a reason and no fitted numerical model")
        elif self.program.family == "trial" and "selection_epoch_capability" in self.predictor.record:
            if self.predictor.record["selection_epoch_capability"]["profile_digest"] != self.program.profile.digest:
                raise ValueError("source inference epoch programme differs from its actual frozen capability")
        elif self.program.family == "continuous_ar":
            model = self.predictor
            if self.program.spatial is not None:
                from transfer.ensemble_views import project_raw_observations

                receipts = model.provenance.get("raw_spatial_execution", ())
                if not receipts:
                    raise ValueError("projected raw expert requires its actual spatial execution receipt")
                for receipt in receipts:
                    if semantic_sha256(receipt["declaration"]) != semantic_sha256(self.program.spatial):
                        raise ValueError("raw spatial programme differs from actual fitted projection")
                    _, channels, operator, expected = project_raw_observations(np.zeros((1, len(receipt["native_channels"]))),
                        receipt["native_channels"], receipt["native_reference"], self.program.spatial,
                        sample_rate_hz=model.sfreq)
                    if (model.channels != channels or model.operator_identity != operator
                            or receipt["matrix"] != expected["matrix"] or receipt["observed_slots"] != expected["observed_slots"]):
                        raise ValueError("raw frozen numerical model differs from its actual spatial programme")
            dictionary = model.dictionary
            if (not isinstance(dictionary, ResponseDictionary)
                    or dictionary.spec != self.program.dictionary_spec
                    or dictionary.sample_rate_hz != model.sfreq):
                raise ValueError("frozen continuous expert requires its actual owned physical dictionary")
            actual_prior = (KernelPriorSpec.isotropic_ridge(model.provenance["ridge"]) if model.kernel_prior is None
                            else model.kernel_prior.spec)
            if (model.noise_order != self.program.noise_order
                    or model.conditioning_samples != self.program.conditioning_samples
                    or actual_prior != self.program.prior_spec
                    or model.provenance.get("noise_estimator", "burg") != self.program.noise_estimator
                    or not np.array_equal(model.basis, dictionary.matrix)):
                raise ValueError("frozen continuous recipe differs from its actual numerical model")

    def unavailable(self, view, reason, failure_kind="incomplete"):
        counts = tuple(int(value) for value in view.decision.membership.sum(axis=0))
        observed = tuple(int(value) for value in view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0))
        digest = semantic_sha256({"program": self.program.record(), "raw_input": view.raw_input_digest,
                                  "unavailable_reason": reason})
        identity = CandidateExpertIdentity(self.program.expert_id, digest, self.model_digest,
            self.source_digest, self.fit_digest, failure_kind, False)
        return CandidateEvidence(view.decision.candidate_vocabulary, None, view.participant_key,
            view.decision_id, view.physical_support_digest, (identity,), view.decision.repetitions,
            counts, observed, failure_reason=reason, failure_kind=failure_kind)

    def score(self, view: PredictionDecisionView, *, _epoch_cache=None):
        if self.predictor is None:
            return self.unavailable(view, self.unavailable_reason, self.unavailable_kind)
        if view.samples is None:
            return self.unavailable(view, view.unavailable_reason)
        if not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0):
            return self.unavailable(view, "incomplete_physical_decision")
        if self.program.family == "trial":
            if _epoch_cache is None:
                epochs, mask, receipt = self.program.profile.transform(view)
            else:
                # This cache exists only inside one current-view score call.
                # Fitted and separately loaded ensembles each compute their
                # own transforms; no cross-view or programme cache is retained.
                key = (self.program.profile.digest, view.source_sha256,
                       view.physical_support_digest, view.raw_input_digest)
                if key not in _epoch_cache:
                    _epoch_cache[key] = self.program.profile.transform(view)
                epochs, mask, receipt = _epoch_cache[key]
            logits = self.predictor.predict_logit(epochs, mask)
            numerical_digest = receipt["numerical_input_digest"]
            scores = membership_decision_scores(logits, view.decision, logit_offset=self.logit_offset)
            identity = CandidateExpertIdentity(self.program.expert_id, numerical_digest,
                self.model_digest, self.source_digest, self.fit_digest, scores.score_kind, False)
            return CandidateEvidence.from_membership_scores(scores, view.decision,
                participant_key=view.participant_key, decision_id=view.decision_id,
                physical_support_digest=view.physical_support_digest, expert_identities=(identity,))
        from transfer.ensemble_views import project_raw_observations

        samples, channels, operator, spatial_receipt = project_raw_observations(view.samples, view.channel_names,
            view.reference, self.program.spatial, sample_rate_hz=view.sample_rate_hz)
        try:
            result = self.predictor.score(samples, view.onsets, view.decision.membership,
                operator_identity=operator, sfreq=view.sample_rate_hz,
                channels=channels, candidate_ids=view.decision.candidate_vocabulary,
                record_id=view.record_id)
        except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
            failure = {"stage": "read_only_prediction", "type": type(error).__name__, "message": str(error)}
            if hasattr(error, "record"):
                failure["numerical_evidence"] = error.record()
            return self.unavailable(view, json.dumps(failure, sort_keys=True, allow_nan=False), "not_available")
        numerical_digest = view.raw_input_digest
        if spatial_receipt is not None:
            numerical_digest = semantic_sha256({"raw_input": view.raw_input_digest, "spatial": spatial_receipt})
        score_kind = "continuous_candidate_log_evidence"
        if self.program.conditioning_samples:
            numerical_digest = semantic_sha256({"raw_input": numerical_digest,
                "conditioning_sample_support": [0, self.program.conditioning_samples],
                "score_sample_support": [self.program.conditioning_samples, len(view.samples)]})
            score_kind = "continuous_conditional_tail_log_density_fixed_candidate_prior/1"
        identity = CandidateExpertIdentity(self.program.expert_id, numerical_digest,
            self.model_digest, self.source_digest, self.fit_digest, score_kind, False)
        counts = tuple(int(value) for value in view.decision.membership.sum(axis=0))
        observed = tuple(int(value) for value in view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0))
        return CandidateEvidence.from_log_evidence(result.log_evidence,
            candidate_vocabulary=view.decision.candidate_vocabulary, participant_key=view.participant_key,
            decision_id=view.decision_id, physical_support_digest=view.physical_support_digest,
            expert_identities=(identity,), evidence_level=view.decision.repetitions,
            expected_event_counts=counts, observed_event_counts=observed)


def _model_fit_binding(predictor, family):
    return (predictor.record["ensemble_fit_binding"] if family == "trial" else
            predictor.provenance["ensemble_fit_binding"])


def _with_model_fit_binding(predictor, family, binding):
    return (predictor.with_fit_binding(binding) if family == "trial" else
            replace(predictor, provenance={**predictor.provenance, "ensemble_fit_binding": binding}))


def _expert_state_record(expert):
    return {"program": expert.program.record(), "model_digest": expert.model_digest,
            "fit_digest": expert.fit_digest, "source_digest": expert.source_digest,
            "logit_offset": expert.logit_offset, "unavailable_reason": expert.unavailable_reason,
            "unavailable_kind": expert.unavailable_kind}


def _attach_oof_probability(fitted, temperature_record, oof_record):
    """Bind the actual captured scores to existing final numerical producers."""
    digest = semantic_sha256(oof_record)
    experts = []
    for expert in fitted._experts:
        if expert.predictor is None:
            experts.append(expert)
            continue
        binding = {**_model_fit_binding(expert.predictor, expert.program.family),
                   "OOF_probability_sha256": digest, "OOF_base_model_digest": expert.model_digest}
        predictor = _with_model_fit_binding(expert.predictor, expert.program.family, binding)
        experts.append(replace(expert, predictor=predictor, model_digest=predictor.digest()))
    return FittedEnsemble(fitted.scorer, tuple(experts), fitted._training_receipts, temperature_record,
                          target_contract=fitted._target_contract, oof_probability_record=oof_record)


class FittedEnsemble:
    """Own expert state, fit authority and one frozen mixture behind score(view)."""

    def __init__(self, scorer, experts, training_receipts, temperature_record=None, *, target_contract=None,
                 oof_probability_record=None):
        self._scorer = scorer
        self._experts = tuple(experts)
        self._training_receipts = tuple(freeze_metadata(row) for row in training_receipts)
        self._target_contract = (None if target_contract is None else freeze_metadata(target_contract))
        self._participant_key = (self._training_receipts[0]["participant_key"] if target_contract is None else
                                 target_contract["participant_key"])
        if tuple(expert.program.expert_id for expert in self._experts) != scorer.program.expert_ids:
            raise ValueError("frozen expert states differ from the declared roster")
        self._temperature_record = freeze_metadata({"strategy": "fixed", "temperatures": {name: 1. for name in scorer.program.expert_ids}}
                                   if temperature_record is None else temperature_record)
        temperatures = self._temperature_record["temperatures"]
        if set(temperatures) != set(scorer.program.expert_ids) or any(
            not np.isfinite(value) or value <= 0 for value in temperatures.values()
        ):
            raise ValueError("frozen temperatures must cover the roster and be positive finite scalars")
        strategy = self._temperature_record["strategy"]
        if strategy == "fixed":
            if any(value != 1. for value in temperatures.values()):
                raise ValueError("fixed temperature program requires unit temperatures")
        elif strategy == "grouped_temperature":
            if self._temperature_record.get("temperature_bounds") != (.05, 20.) or any(
                not .05 <= value <= 20. for value in temperatures.values()
            ):
                raise ValueError("grouped temperature program differs from its declared bounds")
            calibrated_ids = tuple(self._temperature_record.get("temperature_expert_ids", scorer.program.expert_ids))
            if (not calibrated_ids or len(set(calibrated_ids)) != len(calibrated_ids)
                    or not set(calibrated_ids) <= set(scorer.program.expert_ids)
                    or any(temperatures[name] != 1. for name in set(scorer.program.expert_ids) - set(calibrated_ids))):
                raise ValueError("grouped temperature programmes require explicit calibrated experts and fixed controls")
            if "temperature_expert_ids" in self._temperature_record and (
                set(self._temperature_record["fixed_control_expert_ids"]) != set(scorer.program.expert_ids) - set(calibrated_ids)
                or any(set(row["expert_fit_digests"]) != set(calibrated_ids) for row in self._temperature_record["OOF_groups"])):
                raise ValueError("temperature subset differs from actual grouped OOF producers")
        else:
            raise ValueError("unknown declared ensemble calibration strategy")
        self._oof_probability_record = (None if oof_probability_record is None else
                                        freeze_metadata(oof_probability_record))
        if self._oof_probability_record is not None:
            self._validate_oof_binding()

    def _validate_oof_binding(self):
        record = self._oof_probability_record
        if (record["schema"] != GROUPED_PROBABILITY_OOF_SCHEMA
                or semantic_sha256(record["final_program"]) != self.scorer.program.digest()
                or record["final_training_receipts_sha256"] != semantic_sha256(self._training_receipts)
                or record["final_target_contract_sha256"] != semantic_sha256(self._target_contract)):
            raise ValueError("OOF probability evidence differs from the actual final fit authority")
        native = self._training_receipts[0] if self._target_contract is None else self._target_contract
        if (record["native_observation"] != _native_observation_contract_record(native)
                or record["actual_K"] != (sum(len(row["onsets"]) for row in self._training_receipts)
                    if self._target_contract is None else self._training_receipts[0]["label_use"]["raw"]["requested_labels"])):
            raise ValueError("OOF probability evidence differs from actual K or native observation")
        if self.temperature_record["strategy"] == "grouped_temperature" and (
                semantic_sha256(record["temperature_calibration"]) != semantic_sha256(self.temperature_record)):
            raise ValueError("OOF probability evidence differs from the actual temperature programme")
        expected = {entry["program"]["expert_id"]: entry for entry in record["final_base_experts"]}
        digest = semantic_sha256(record)
        for expert in self._experts:
            entry = expected[expert.program.expert_id]
            if (semantic_sha256(entry["program"]) != semantic_sha256(expert.program.record())
                    or entry["fit_digest"] != expert.fit_digest or entry["source_digest"] != expert.source_digest
                    or entry["logit_offset"] != expert.logit_offset
                    or entry["unavailable_reason"] != expert.unavailable_reason
                    or entry["unavailable_kind"] != expert.unavailable_kind):
                raise ValueError("OOF probability evidence differs from the actual final expert programme")
            if expert.predictor is not None:
                binding = _model_fit_binding(expert.predictor, expert.program.family)
                if (binding.get("OOF_probability_sha256") != digest
                        or binding.get("OOF_base_model_digest") != entry["model_digest"]):
                    raise ValueError("OOF probability evidence differs from its actual numerical model binding")

    @property
    def oof_probability_record(self):
        """Complete C-only cross-fit evidence; it is not nested selection or efficacy."""
        return self._oof_probability_record

    @property
    def temperature_record(self):
        return self._temperature_record

    @property
    def scorer(self):
        return self._scorer

    @property
    def participant_key(self):
        return self._participant_key

    @property
    def fit_receipts(self):
        return self._training_receipts

    @property
    def fit_available(self):
        return all(expert.predictor is not None for expert in self._experts)

    @property
    def expert_fit_available(self):
        return {expert.program.expert_id: expert.predictor is not None for expert in self._experts}

    def _score_raw_experts(self, view: PredictionDecisionView):
        if view.participant_key != self.participant_key:
            raise ValueError("personal ensemble cannot score another participant identity")
        task = (self._training_receipts[0]["decision"] if self._target_contract is None else self._target_contract)
        if tuple(view.decision.candidate_vocabulary) != tuple(task["candidate_vocabulary"]):
            raise ValueError("ensemble candidate vocabulary/order differs from the frozen task")
        for expert in self._experts:
            if expert.program.family == "trial":
                expert.program.profile.validate_observation(view)
        if any(expert.program.family == "continuous_ar" for expert in self._experts):
            first = self._training_receipts[0] if self._target_contract is None else self._target_contract
            expected = (first["sample_rate_hz"], tuple(first["channel_names"]), tuple(first["channel_types"]),
                        first["reference"], first["signal_unit"])
            if _native_observation_contract(view) != expected:
                raise ValueError("continuous ensemble observation layout/reference differs from calibration")
        for receipt in self._training_receipts:
            if view.record_id == receipt["record_id"] or view.raw_input_digest == receipt["raw_input_digest"]:
                raise ValueError("ensemble prediction repeats a calibration observation")
            if view.samples is not None and view.source_sha256 == receipt["source_sha256"] and (
                view.raw_scope[0] < receipt["raw_scope"][1] and view.raw_scope[1] > receipt["raw_scope"][0]
            ):
                raise ValueError("ensemble prediction raw support overlaps calibration")
        epoch_cache = {}
        scored = {expert.program.expert_id: expert.score(view, _epoch_cache=epoch_cache)
                  for expert in self._experts}
        return scored

    def _apply_probability_program(self, scored):
        if self.temperature_record["strategy"] == "fixed":
            return scored
        calibrated = {}
        calibration_digest = semantic_sha256(self.temperature_record)
        for name, item in scored.items():
            if (not item.probability_available or name not in self.temperature_record.get(
                "temperature_expert_ids", self.scorer.program.expert_ids)):
                calibrated[name] = item
                continue
            identities = tuple(replace(identity, probability_calibrated=True,
                fit_digest=semantic_sha256({"expert_fit": identity.fit_digest, "calibration": calibration_digest}),
                score_kind=identity.score_kind + ":posterior_power_temperature/1") for identity in item.expert_identities)
            values = item.log_probability / self.temperature_record["temperatures"][name]
            calibrated[name] = replace(item, log_probability=values - logsumexp(values), expert_identities=identities)
        return calibrated

    def score_experts(self, view: PredictionDecisionView):
        return self._apply_probability_program(self._score_raw_experts(view))

    @classmethod
    def score_programs(cls, programs, view: PredictionDecisionView):
        """Score one current view from genuinely shared final expert objects.

        Fitted and reloaded program groups invoke this independently. Raw
        evidence lives only in this call; each probability program applies its
        original transform and keeps unavailable experts in the full roster.
        """
        if not programs or set(programs) - {"fixed", "grouped_temperature"}:
            raise ValueError("shared scoring requires declared probability programs")
        available = []
        for name, value in programs.items():
            if isinstance(value, ExpectedSubjectError):
                continue
            if not isinstance(value, cls) or value.temperature_record["strategy"] != name:
                raise ValueError("shared probability name differs from its actual program")
            available.append(value)
        if not available:
            return dict(programs)
        first = available[0]
        for value in available[1:]:
            if (value._experts is not first._experts or value.scorer.program != first.scorer.program
                    or value._training_receipts != first._training_receipts
                    or value._target_contract != first._target_contract):
                raise ValueError("probability programs must share actual final experts and fit authority")
        scored = first._score_raw_experts(view)
        return {name: value if isinstance(value, ExpectedSubjectError)
                else value._apply_probability_program(scored) for name, value in programs.items()}

    def score(self, view: PredictionDecisionView):
        return self.scorer.score(self.score_experts(view))

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        entries = []
        for index, expert in enumerate(self._experts):
            if expert.predictor is None:
                entries.append({"program": expert.program.record(), "model": None,
                    "model_digest": None, "fit_digest": expert.fit_digest,
                    "source_digest": expert.source_digest, "logit_offset": expert.logit_offset,
                    "unavailable_reason": expert.unavailable_reason, "unavailable_kind": expert.unavailable_kind})
                continue
            filename = f"expert_{index:03d}." + ("pt" if expert.program.family == "trial" else "npz")
            path = directory / filename
            expert.predictor.save(path)
            entries.append({"program": expert.program.record(), "model": filename,
                "model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_digest": expert.model_digest, "fit_digest": expert.fit_digest,
                "source_digest": expert.source_digest, "logit_offset": expert.logit_offset})
        record = {"schema": FITTED_ENSEMBLE_SCHEMA, "program": self.scorer.program.record(),
                  "experts": entries, "training_receipts": list(self._training_receipts),
                  "temperature_calibration": self.temperature_record}
        if self._target_contract is not None:
            record["target_contract"] = self._target_contract
        if self.oof_probability_record is not None:
            record["OOF_probability"] = self.oof_probability_record
        atomic_write_json(directory / "ensemble.json", {"record": record, "record_sha256": semantic_sha256(record)})
        return directory

    @classmethod
    def save_programs(cls, directory, programs, *, participant_key):
        """One numerical package, with separate declared probability programs."""
        directory = Path(directory)
        fitted = next((value for value in programs.values() if isinstance(value, cls)), None)
        if fitted is None:
            directory.mkdir(parents=True, exist_ok=False)
            base_digest = None
        else:
            fitted.save(directory)
            base_digest = hashlib.sha256((directory / "ensemble.json").read_bytes()).hexdigest()
        records = {}
        for name, value in programs.items():
            if name not in {"fixed", "grouped_temperature"}:
                raise ValueError("unknown declared ensemble calibration strategy")
            if isinstance(value, ExpectedSubjectError):
                records[name] = {"status": "unavailable", "failure": value.record(subject=participant_key)}
            else:
                if value._experts is not fitted._experts or value.scorer.program != fitted.scorer.program:
                    raise ValueError("temperature programs must share the actual all-C final expert states")
                if semantic_sha256(value.oof_probability_record) != semantic_sha256(fitted.oof_probability_record):
                    raise ValueError("temperature programs must share the actual captured OOF probabilities")
                if name != value.temperature_record["strategy"]:
                    raise ValueError("temperature program name differs from its actual strategy")
                records[name] = {"status": "completed", "temperature_calibration": value.temperature_record}
        record = {"schema": "n2p3_prefix_temperature_programs/1", "base_package_sha256": base_digest,
                  "participant_key": participant_key,
                  "programs": records}
        atomic_write_json(directory / "temperature_programs.json", {"record": record, "record_sha256": semantic_sha256(record)})
        return directory

    @classmethod
    def load_programs(cls, directory):
        directory = Path(directory)
        payload = json.loads((directory / "temperature_programs.json").read_bytes())
        record = payload["record"]
        if record["schema"] != "n2p3_prefix_temperature_programs/1" or semantic_sha256(record) != payload["record_sha256"]:
            raise ValueError("frozen temperature program manifest changed")
        if not record["programs"] or set(record["programs"]) - {"fixed", "grouped_temperature"}:
            raise ValueError("unknown declared ensemble calibration strategy")
        for name, value in record["programs"].items():
            if value["status"] not in {"completed", "unavailable"}:
                raise ValueError("temperature program requires an explicit completed or unavailable state")
            if value["status"] == "completed" and name != value["temperature_calibration"]["strategy"]:
                raise ValueError("temperature program name differs from its actual strategy")
        if record["base_package_sha256"] is None:
            if any(value["status"] != "unavailable" for value in record["programs"].values()):
                raise ValueError("available temperature programs require their actual final model package")
            return {}
        if hashlib.sha256((directory / "ensemble.json").read_bytes()).hexdigest() != record["base_package_sha256"]:
            raise ValueError("temperature programs differ from their actual final model package")
        fitted = cls.load(directory)
        if record["participant_key"] != fitted.participant_key:
            raise ValueError("temperature program participant differs from actual final model package")
        return {name: cls(fitted.scorer, fitted._experts, fitted._training_receipts,
            value["temperature_calibration"], target_contract=fitted._target_contract,
            oof_probability_record=fitted.oof_probability_record)
            for name, value in record["programs"].items() if value["status"] == "completed"}

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        payload = json.loads((directory / "ensemble.json").read_bytes())
        record = payload["record"]
        if semantic_sha256(record) != payload["record_sha256"] or record["schema"] != FITTED_ENSEMBLE_SCHEMA:
            raise ValueError("frozen ensemble manifest changed or has an unsupported schema")
        experts = []
        for entry in record["experts"]:
            program = CalibrationExpertProgram.from_record(entry["program"])
            if entry["model"] is None:
                experts.append(_FittedExpert(program, None, entry["model_digest"], entry["fit_digest"],
                    entry["source_digest"], entry["logit_offset"], entry["unavailable_reason"], entry["unavailable_kind"]))
                continue
            path = directory / entry["model"]
            if path.resolve().parent != directory.resolve():
                raise ValueError("frozen ensemble model path escapes its package")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["model_file_sha256"]:
                raise ValueError("frozen ensemble model bytes changed")
            predictor = (FrozenTrialPredictor.load(io.BytesIO(content)) if program.family == "trial"
                         else ContinuousDecisionModel.load(io.BytesIO(content)))
            if predictor.digest() != entry["model_digest"]:
                raise ValueError("frozen ensemble model state differs from its declared identity")
            binding = _model_fit_binding(predictor, program.family)
            expected_binding = {"program": program.record(), "training_receipts": record["training_receipts"],
                                "model_source_digest": entry["source_digest"], "fit_digest": entry["fit_digest"],
                                "logit_offset": entry["logit_offset"]}
            if "target_contract" in record:
                expected_binding["target_contract"] = record["target_contract"]
            if "OOF_probability" in record:
                oof = record["OOF_probability"]
                base = next(row for row in oof["final_base_experts"] if row["program"]["expert_id"] == program.expert_id)
                expected_binding.update(OOF_probability_sha256=semantic_sha256(oof),
                                        OOF_base_model_digest=base["model_digest"])
                original = _with_model_fit_binding(predictor, program.family,
                    {key: value for key, value in binding.items()
                     if key not in {"OOF_probability_sha256", "OOF_base_model_digest"}})
                if original.digest() != base["model_digest"]:
                    raise ValueError("OOF probability evidence differs from the actual final numerical state")
            if semantic_sha256(binding) != semantic_sha256(expected_binding):
                raise ValueError("frozen ensemble recipe and fit authority differ from the actual model")
            experts.append(_FittedExpert(program, predictor, entry["model_digest"], entry["fit_digest"],
                                         entry["source_digest"], entry["logit_offset"]))
        return cls(EnsembleScorer(EnsembleProgram.from_record(record["program"])), experts,
                   record["training_receipts"], record["temperature_calibration"],
                   target_contract=record.get("target_contract"),
                   oof_probability_record=record.get("OOF_probability"))


def validate_calibration_ensemble_groups(calibration, expert_programs):
    """Admit a complete physical roster before fitting or taking nested subsets."""
    if not calibration or any(not isinstance(group, CalibrationSelection) for group in calibration):
        raise ValueError("ensemble fitting requires authorized CalibrationSelection groups")
    participant = calibration[0].view.participant_key
    vocabulary = calibration[0].view.decision.candidate_vocabulary
    if any(group.view.participant_key != participant or group.view.decision.candidate_vocabulary != vocabulary
           or group.view.decision.protocol != calibration[0].view.decision.protocol
           for group in calibration):
        raise ValueError("ensemble calibration requires one participant and frozen candidate task")
    if any(group.view.samples is None or not group.view.decision.schedule_complete
           or np.any(group.view.decision.evidence_indices < 0)
           for group in calibration):
        raise ValueError("ensemble calibration requires complete observed physical groups")
    views = tuple(group.view for group in calibration)
    identities = DatasetIdentityTable.from_source_rows(
        [view.source_subject_id for view in views], [view.source_dataset_id for view in views])
    if any(identities.record_for(view.source_subject_id).authority_key() != participant for view in views):
        raise ValueError("calibration participant key disagrees with the shared identity contract")
    if len({view.decision_id for view in views}) != len(views):
        raise ValueError("calibration decision identities must be unique")
    events = [event for view in views for event in view.decision.event_ids]
    if len(set(events)) != len(events):
        raise ValueError("calibration event identities must be globally unique")
    if any(spec.family == "continuous_ar" for spec in expert_programs) and any(
        _native_observation_contract(group.view) != _native_observation_contract(calibration[0].view)
        for group in calibration[1:]
    ):
        raise ValueError("continuous calibration groups must share the native observation layout/reference")
    for index, left in enumerate(calibration):
        for right in calibration[index + 1:]:
            if left.view.record_id == right.view.record_id or left.view.raw_input_digest == right.view.raw_input_digest:
                raise ValueError("calibration groups repeat the same physical observation")
            if left.view.source_sha256 == right.view.source_sha256 and (
                left.view.raw_scope[0] < right.view.raw_scope[1] and right.view.raw_scope[0] < left.view.raw_scope[1]
            ):
                raise ValueError("calibration group raw scopes overlap")


def _calibration_expert_fit_digest(spec, fit_inputs, checkpoint_digest):
    """Bind the actual authorized labels/observations and source role once."""
    return semantic_sha256({"program": spec.record(), "authorized_calibration": fit_inputs,
        "source_checkpoint_sha256": checkpoint_digest, "checkpoint_role": (None if checkpoint_digest is None else
            "pretrained_weights" if spec.recipe.initialization == "source" else "architecture_only")})


def fit_calibration_ensemble(calibration, expert_programs, program: EnsembleProgram, *,
                             source_checkpoints=None, device="cpu", calibration_strategy="fixed", temperature_expert_ids=None,
                             _fixed_fit_cache=None, _producer_recorder=None, _producer_context=None,
                             _record_self=True, _source_preparation=None):
    """Fit only supplied authorized groups; no prediction data or truth is accepted."""
    calibration = tuple(calibration)
    expert_programs = tuple(expert_programs)
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    if calibration_strategy not in {"fixed", "grouped_temperature"}:
        raise ValueError("unknown declared ensemble calibration strategy")
    if calibration_strategy == "grouped_temperature" and len(calibration) < 3:
        raise ValueError("grouped probability calibration needs at least three complete groups")
    if tuple(spec.expert_id for spec in expert_programs) != program.expert_ids:
        raise ValueError("fit programs must match the frozen ensemble roster and order")
    validate_calibration_ensemble_groups(calibration, expert_programs)
    sources = {} if source_checkpoints is None else {
        name: value if isinstance(value, FrozenSourceCheckpoint) else FrozenSourceCheckpoint.from_path(value)
        for name, value in source_checkpoints.items()}
    if set(sources) - set(program.expert_ids):
        raise ValueError("source checkpoint declaration contains an undeclared expert")
    if any(spec.expert_id in sources and (spec.family != "trial"
               or spec.recipe.initialization not in {"source", "scratch"}) for spec in expert_programs):
        raise ValueError("source checkpoints belong only to source weights or matched scratch architecture")
    receipts = [dict(group.view.record()) for group in calibration]
    fit_inputs = [group.record() for group in calibration]
    cache_key = None if _fixed_fit_cache is None else semantic_sha256({
        "authorized_calibration": fit_inputs, "expert_programs": [spec.record() for spec in expert_programs],
        "ensemble_program": program.record(), "source_checkpoints": {name: source.digest for name, source in sources.items()},
        "device": str(device), "calibration_strategy": "fixed"})
    if cache_key is not None and cache_key in _fixed_fit_cache:
        fitted = _fixed_fit_cache[cache_key]
        if _producer_recorder is not None and _record_self:
            _producer_recorder.record_direct_fit(cache_key, fitted, expert_programs=expert_programs,
                program=program, training=calibration, context=_producer_context, cache_hit=True)
        return (fitted if calibration_strategy == "fixed" else
                _grouped_temperature(fitted, calibration, expert_programs, sources, device,
                    temperature_expert_ids, _fixed_fit_cache=_fixed_fit_cache,
                    _producer_recorder=_producer_recorder, _producer_context=_producer_context,
                    _source_preparation=preparation))
    experts = []
    native_fit_inputs = {}
    for spec in expert_programs:
        data = build_calibration_epoch_dataset(calibration, spec.profile) if spec.family == "trial" else None
        records = [CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
                   for group in calibration] if spec.family == "continuous_ar" else []
        experts.append(_fit_expert(spec, data, records, receipts[0], sources.get(spec.expert_id),
            receipts, fit_inputs, device=device, source_preparation=preparation))
        if _producer_recorder is not None:
            from research.prefix_inner_replay import actual_expert_fit_inputs

            native_fit_inputs[spec.expert_id] = actual_expert_fit_inputs(spec, data, records, receipts[0])
    fitted = FittedEnsemble(EnsembleScorer(program), experts, receipts)
    if _producer_recorder is not None:
        _producer_recorder.record_native_fit_inputs(fitted, native_fit_inputs)
    if cache_key is not None:
        _fixed_fit_cache[cache_key] = fitted
    if calibration_strategy == "fixed":
        if _producer_recorder is not None and _record_self:
            _producer_recorder.record_direct_fit(cache_key, fitted, expert_programs=expert_programs,
                program=program, training=calibration, context=_producer_context)
        return fitted
    return _grouped_temperature(fitted, calibration, expert_programs, sources, device, temperature_expert_ids,
                                _fixed_fit_cache=_fixed_fit_cache, _producer_recorder=_producer_recorder,
                                _producer_context=_producer_context, _source_preparation=preparation)


def _fit_expert(spec, data, raw_records, target, source, receipts, fit_inputs, *, device,
                source_only=False, target_contract=None, source_preparation=None):
    source_digest = checkpoint_digest = None
    if spec.family == "trial":
        source_trunk = source_stats = prepared = None
        if source is not None:
            preparation = source_preparation or ScopedSourceCheckpointPreparation()
            subject = data.identity_table.local_subject_ids[0]
            prepared = preparation.prepare(source.content, data, subject)
            target_membership = None
            if prepared.source_membership_present:
                from data.study import _from_metadata, dataset_study_membership

                if source_only:
                    from data.lineage import DataLineage

                    # K0 has physical metadata, never an EpochDataset or target
                    # observations. Use the same study compiler on that authority.
                    lineage = DataLineage.source(parameters={"source_dataset_ids":
                        list(source_dataset_ids_from_identity(data.identity_table)),
                        "target_contract": target_contract, "source_provenance": data.provenance})
                    target_membership = _from_metadata(data.identity_table, lineage, data.provenance)
                else:
                    target_membership = dataset_study_membership(data)
            prepared.assert_library_exclusion(target_membership=target_membership)
            subject = data.identity_table.local_subject_ids[0]
            source_trunk = prepared.trunk_for_target(subject)
            source_stats = prepared.input_statistics
            checkpoint_digest = source.digest
            if spec.recipe.initialization == "source":
                source_digest = checkpoint_digest
        if source_only:
            if prepared is None:
                raise ValueError("source-only K0 requires an admitted source checkpoint")
            predictor = FrozenTrialPredictor.from_source(prepared, subject)
            offset = 0.
        else:
            compute_device = (device if spec.recipe.initialization in
                {"source", "scratch", "source_feature_lda"} else "cpu")
            with worker_compute_phase(compute_device):
                fitted = fit_calibration_candidate(data, spec.recipe, np.ones(data.n_epochs, dtype=bool),
                    source_trunk=source_trunk, source_input_stats=source_stats, device=device)
            predictor = FrozenTrialPredictor.from_fitted(fitted)
            offset = 0.
    else:
        from transfer.ensemble_views import project_raw_observations

        rate = target["sample_rate_hz"]
        vocabulary = target["candidate_vocabulary"] if "candidate_vocabulary" in target else target["decision"]["candidate_vocabulary"]
        projected_records, spatial_receipts = [], []
        channels, operator = None, None
        for record in raw_records:
            samples, current_channels, current_operator, receipt = project_raw_observations(record.samples,
                target["channel_names"], target["reference"], spec.spatial, sample_rate_hz=rate)
            if channels is not None and (channels, operator) != (current_channels, current_operator):
                raise ValueError("raw C records require one actual observed spatial programme")
            channels, operator = current_channels, current_operator
            projected_records.append(replace(record, samples=samples))
            if receipt is not None:
                spatial_receipts.append({"record_id": record.record_id, **receipt})
        predictor = ContinuousDecisionModel.fit(projected_records, sfreq=rate, channels=channels,
            candidate_ids=vocabulary, operator_identity=operator, noise_order=spec.noise_order,
            **({"ridge": spec.ridge} if spec.prior_spec.kind == "isotropic_ridge" else {"kernel_prior": spec.kernel_prior}),
            conditioning_samples=spec.conditioning_samples, noise_estimator=spec.noise_estimator,
            dictionary=ResponseDictionary.compile(spec.dictionary_spec, sample_rate_hz=rate))
        if spatial_receipts:
            predictor = replace(predictor, provenance={**predictor.provenance, "raw_spatial_execution": spatial_receipts})
        offset = 0.
    fit_digest = _calibration_expert_fit_digest(spec, fit_inputs, checkpoint_digest)
    binding = {"program": spec.record(), "training_receipts": receipts,
               "model_source_digest": source_digest, "fit_digest": fit_digest, "logit_offset": offset}
    if target_contract is not None:
        binding["target_contract"] = target_contract
    predictor = (predictor.with_fit_binding(binding) if spec.family == "trial" else
                 replace(predictor, provenance={**predictor.provenance, "ensemble_fit_binding": binding}))
    return _FittedExpert(spec, predictor, predictor.digest(), fit_digest, source_digest, offset)


def fit_calibration_prefix(prefix: CalibrationPrefix, expert_programs, program: EnsembleProgram, *,
                            source_checkpoints=None, device="cpu", calibration_strategy="fixed",
                            inner_replay_recorder=None, inner_replay_context=None, _source_preparation=None):
    """Fit all legitimate prefix rows; complete C groups serve only temperature."""
    if not isinstance(prefix, CalibrationPrefix):
        raise TypeError("prefix fitting requires an authorized CalibrationPrefix")
    prefix.validate()
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    specs = tuple(expert_programs)
    if tuple(spec.expert_id for spec in specs) != program.expert_ids:
        raise ValueError("prefix fit programs must match the complete frozen expert roster")
    if calibration_strategy not in {"fixed", "grouped_temperature"}:
        raise ValueError("unknown declared ensemble calibration strategy")
    if calibration_strategy == "grouped_temperature" and len(prefix.complete_groups) < 3:
        raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
            stage="temperature", detail="fewer than three complete C-only groups")
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    if set(sources) - set(program.expert_ids) or any(spec.expert_id in sources and
            (spec.family != "trial" or spec.recipe.initialization not in {"source", "scratch"}) for spec in specs):
        raise ValueError("prefix sources must belong to declared source or matched scratch experts")
    receipts = [] if prefix.raw is None else [{**prefix.target_contract, "record_id": prefix.raw.record_id,
        "raw_input_digest": array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask),
        "label_use": prefix.label_use}]
    experts = []
    for spec in specs:
        source_only = prefix.budget == 0 and spec.family == "trial" and spec.recipe.initialization == "source"
        data = (prefix.trial_target(spec.profile) if source_only else
                prefix.trial_datasets.get(spec.profile.digest) if spec.family == "trial" else None)
        reason = None
        if not source_only and (prefix.raw is None or spec.family == "trial" and data is None):
            reason = "K0_target_fitted_expert_unavailable" if prefix.budget == 0 else "no_complete_authorized_trial_rows"
        elif spec.family == "trial" and not source_only and (data.n_epochs < 3 or set(np.unique(data.y)) != {0, 1}):
            reason = "insufficient_authorized_trial_classes_or_rows"
        if reason:
            fit_digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(), "unavailable": reason})
            expert = _FittedExpert(spec, None, None, fit_digest, None, 0., reason)
        else:
            try:
                expert = _fit_expert(spec, data, [] if prefix.raw is None else [prefix.raw], prefix.target_contract,
                    sources.get(spec.expert_id), receipts, prefix.record(), device=device, source_only=source_only,
                    target_contract=prefix.target_contract, source_preparation=preparation)
            except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
                reason = f"{type(error).__name__}: {error}"
                fit_digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(), "fit_failure": reason})
                expert = _FittedExpert(spec, None, None, fit_digest, None, 0., reason, "fit_failure")
        experts.append(expert)
    fitted = FittedEnsemble(EnsembleScorer(program), experts, receipts, target_contract=prefix.target_contract)
    if calibration_strategy == "fixed":
        if inner_replay_recorder is not None:
            inner_replay_recorder.record_prefix_programs({"fixed": fitted}, prefix=prefix,
                expert_programs=specs, program=program, source_checkpoints=sources, context=inner_replay_context)
        return fitted
    grouped = _grouped_temperature(fitted, prefix.complete_groups, specs, sources, device, prefix=prefix,
                                  _fixed_fit_cache={} if inner_replay_recorder is not None else None,
                                  _producer_recorder=inner_replay_recorder,
                                  _producer_context=inner_replay_context, _source_preparation=preparation)
    if inner_replay_recorder is not None:
        inner_replay_recorder.record_prefix_programs({"grouped_temperature": grouped}, prefix=prefix,
            expert_programs=specs, program=program, source_checkpoints=sources, context=inner_replay_context)
    return grouped


def _grouped_temperature(fitted, calibration, expert_programs, sources, device, temperature_expert_ids=None, *, prefix=None,
                         _fixed_fit_cache=None, _producer_recorder=None, _producer_context=None, _source_preparation=None):
    program = fitted.scorer.program
    selected = program.expert_ids if temperature_expert_ids is None else tuple(temperature_expert_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(program.expert_ids):
        raise ValueError("temperature calibration requires an explicit nonempty subset of declared producers")
    by_expert = {name: [] for name in selected}
    inner_program = EnsembleProgram.equal_weights(program.name + ":temperature_training", selected)
    selected_specs = tuple(next(spec for spec in expert_programs if spec.expert_id == name) for name in selected)
    producers, oof_rows = [], []
    for held_index, held in enumerate(calibration):
        training = [group for index, group in enumerate(calibration) if index != held_index]
        context = {**(_producer_context or {}), "phase": "temperature_oof_inner",
                   "temperature_held_record_id": held.view.record_id}
        cache_key = None if _producer_recorder is None else semantic_sha256({
            "authorized_calibration": [group.record() for group in training],
            "expert_programs": [spec.record() for spec in selected_specs],
            "ensemble_program": inner_program.record(),
            "source_checkpoints": {name: source.digest for name, source in sources.items() if name in selected},
            "device": str(device), "calibration_strategy": "fixed"})
        try:
            inner = fit_calibration_ensemble(training, selected_specs, inner_program,
                source_checkpoints={name: source for name, source in sources.items() if name in selected},
                device=device, calibration_strategy="fixed", _fixed_fit_cache=_fixed_fit_cache,
                _producer_recorder=_producer_recorder, _producer_context=context, _source_preparation=_source_preparation)
        except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
            if _producer_recorder is not None:
                failure = (error if isinstance(error, ExpectedSubjectError) else ExpectedSubjectError(
                    SubjectFailureCode.NUMERICAL_FAILURE, stage="temperature_inner_fit", detail=str(error)))
                _producer_recorder.record_direct_fit(cache_key, failure, expert_programs=selected_specs,
                    program=inner_program, training=training, context=context)
            raise
        try:
            scores = inner.score_experts(held.view)
        except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
            if _producer_recorder is not None:
                failure = (error if isinstance(error, ExpectedSubjectError) else ExpectedSubjectError(
                    SubjectFailureCode.NUMERICAL_FAILURE, stage="temperature_inner_score", detail=str(error)))
                _producer_recorder.record_oof_score_failure(cache_key, held=held, training=training,
                                                          error=failure, context=context)
            raise
        if _producer_recorder is not None:
            _producer_recorder.record_oof_score(cache_key, held=held, training=training,
                                               scores=scores, context=context)
        target = held.view.decision.candidate_vocabulary.index(held.target_candidate)
        for name, item in scores.items():
            if not item.probability_available:
                raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="temperature",
                    detail=f"{name}: grouped temperature fit requires complete OOF probability evidence")
            by_expert[name].append((item.log_probability, target))
        producers.append({"held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "expert_fit_digests": {name: item.expert_identities[0].fit_digest for name, item in scores.items()}})
        oof_rows.append({"held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "training_receipts_sha256": semantic_sha256(inner.fit_receipts),
            "expert_evidence": {name: scores[name].record() for name in selected}})
    temperatures = {name: 1. for name in program.expert_ids}
    for name, rows in by_expert.items():
        log_probability = np.stack([row[0] for row in rows])
        targets = np.asarray([row[1] for row in rows], dtype=int)

        temperatures[name] = temperature_from_oof(log_probability, targets)
    temperature_record = {"strategy": "grouped_temperature", "temperatures": temperatures,
        "temperature_expert_ids": list(selected), "fixed_control_expert_ids": [name for name in program.expert_ids if name not in selected],
        "temperature_bounds": [.05, 20.], "score_transform": "posterior_power_temperature/1",
        "evidence_scope": "calibration_development", "OOF_groups": producers,
        "final_experts_refitted_on_all_authorized_groups": True,
        "held_prediction_labels_used_for_calibration": False}
    native = fitted._training_receipts[0] if fitted._target_contract is None else fitted._target_contract
    oof_record = {"schema": GROUPED_PROBABILITY_OOF_SCHEMA, "evidence_scope": "calibration_development",
        "captured_expert_ids": selected, "OOF_program": inner_program.record(),
        "OOF_expert_programs": [spec.record() for spec in selected_specs],
        "calibration_groups": [group.record() for group in calibration], "rows": oof_rows,
        "actual_K": sum(len(group.labels) for group in calibration) if prefix is None else prefix.budget,
        "complete_group_labels": sum(len(group.labels) for group in calibration),
        "fit_authority_sha256": semantic_sha256([group.record() for group in calibration]
                                                 if prefix is None else prefix.record()),
        "native_observation": _native_observation_contract_record(native),
        "source_checkpoint_sha256": {name: source.digest for name, source in sources.items()},
        "final_program": fitted.scorer.program.record(),
        "final_training_receipts_sha256": semantic_sha256(fitted.fit_receipts),
        "final_target_contract_sha256": semantic_sha256(fitted._target_contract),
        "final_base_experts": [_expert_state_record(expert) for expert in fitted._experts],
        "temperature_calibration": temperature_record, "selection_performed": False}
    return _attach_oof_probability(fitted, temperature_record, oof_record)


def fit_calibration_prefix_programs(prefix, expert_programs, program, *, strategies,
                                    source_checkpoints=None, device="cpu", temperature_expert_ids=None,
                                    inner_replay_recorder=None, inner_replay_context=None, _source_preparation=None):
    """Declared temperature programs share one actual all-C final expert fit."""
    if (not strategies or len(set(strategies)) != len(strategies)
            or set(strategies) - {"fixed", "grouped_temperature"}):
        raise ValueError("prefix temperature programs must be explicit and distinct")
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    result = {}
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    replay_cache = {} if inner_replay_recorder is not None else None
    eligible = len(prefix.complete_groups) >= 3
    fixed = (fit_calibration_prefix(prefix, expert_programs, program, source_checkpoints=sources, device=device,
                                    inner_replay_recorder=inner_replay_recorder,
                                    inner_replay_context=inner_replay_context, _source_preparation=preparation)
             if "fixed" in strategies or eligible else None)
    for strategy in strategies:
        if strategy == "fixed":
            result[strategy] = fixed
        elif not eligible:
            result[strategy] = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                stage="temperature", detail="fewer than three complete C-only groups")
        else:
            try:
                result[strategy] = _grouped_temperature(fixed, prefix.complete_groups, expert_programs, sources,
                                                       device, temperature_expert_ids, prefix=prefix,
                                                       _fixed_fit_cache=replay_cache,
                                                       _producer_recorder=inner_replay_recorder,
                                                       _producer_context=inner_replay_context, _source_preparation=preparation)
            except ExpectedSubjectError as error:
                result[strategy] = error
            except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                result[strategy] = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE,
                    stage="temperature", detail=f"{type(error).__name__}: {error}")
    grouped = result.get("grouped_temperature")
    if "fixed" in result and isinstance(grouped, FittedEnsemble):
        result["fixed"] = FittedEnsemble(fixed.scorer, grouped._experts, fixed._training_receipts,
            fixed.temperature_record, target_contract=fixed._target_contract,
            oof_probability_record=grouped.oof_probability_record)
    if inner_replay_recorder is not None:
        record = getattr(inner_replay_recorder, "record_prefix_programs", None)
        if record is None:
            raise TypeError("inner_replay_recorder must support prefix program capture")
        record(result, prefix=prefix, expert_programs=expert_programs, program=program,
               source_checkpoints=sources, context=inner_replay_context)
    return result
