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
from research.history_authority import HistoryUnavailableError
from research.joint_response_sampling import SamplingResolutionError
from transfer.beta_uniform_calibration import (
    BETA_UNIFORM_OOF_SCHEMA,
    BetaCalibrationResolutionError,
    beta_uniform_from_oof,
    validate_beta_uniform_prior,
)
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.calibration_state import FrozenTrialPredictor
from transfer.checkpoint import ScopedSourceCheckpointPreparation
from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
)
from transfer.ensemble import EnsembleProgram, EnsembleScorer, beta_uniform_evidence
from transfer.ensemble_views import (
    CalibrationPrefix,
    CalibrationSelection,
    PredictionDecisionView,
    SelectionEpochProfile,
    build_calibration_epoch_dataset,
)
from transfer.joint_flat_model import JointFlatModel
from transfer.joint_flat_programme import FrozenJointFlatSource, JointFlatProgram
from transfer.kernel_prior import KernelPriorSpec, resolve_kernel_prior
from transfer.noise_model import freeze_metadata, metadata_digest
from transfer.pool_calibration import (
    FORWARD_BETA_STRATEGY,
    POOL_OOF_SCHEMA,
    POOL_STRATEGIES,
    POOL_STRATEGY,
    TEMPERATURE_POOL_STRATEGY,
    uniform_contamination_from_oof,
)
from transfer.recipes import TargetRecipe
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec
from transfer.spatial_programme import SpatialDecisionModel, SpatialSignalProgram

FITTED_ENSEMBLE_SCHEMA = "n2p3_fitted_candidate_ensemble/2"
GROUPED_PROBABILITY_OOF_SCHEMA = "n2p3_complete_group_probability_oof/1"


def _beta_uniform_oof_state(record, program, prior):
    """Derive the full posterior from bound raw forecasts and authorized C groups.

    Saved graph/public-parent admission additionally restores these original
    groups and producer outputs through its actual input owner.
    """
    groups, rows = record["calibration_groups"], record["rows"]
    names = [group["view"]["record_id"] for group in groups]
    inner_program = EnsembleProgram.from_record(record["OOF_program"])
    authority = record["fit_authority"]
    if (record["schema"] != BETA_UNIFORM_OOF_SCHEMA or len(program.expert_ids) != 1
            or tuple(record["captured_expert_ids"]) != program.expert_ids
            or inner_program.expert_ids != program.expert_ids
            or semantic_sha256(record["OOF_expert_programs"]) != semantic_sha256(
                [entry["program"] for entry in record["final_base_experts"]])
            or record["OOF_method"] != "original_earlier_C_prefix_prequential"
            or semantic_sha256(record["final_program"]) != program.digest()
            or semantic_sha256(authority) != record["fit_authority_sha256"]
            or semantic_sha256(authority["temperature_group_records"]) != semantic_sha256(groups)
            or authority["budget"] != record["actual_K"]
            or semantic_sha256(authority["target_contract"]) != record["final_target_contract_sha256"]
            or len(set(names)) != len(names) or len(rows) != len(groups)
            or [row["held_record_id"] for row in rows] != names):
        raise ValueError("Beta OOF must preserve the full original ordered forward roster")
    declared_prior = validate_beta_uniform_prior(prior)
    forecasts = []
    earlier_k = -1
    for index, (row, group) in enumerate(zip(rows, groups, strict=True)):
        training = list(row["training_record_ids"])
        training_set = set(training)
        if (row["status"] != "scored" or row["expert_failures"] or row["fit_failure"] is not None
                or row["held_labels_used_for_fit"] is not False
                or len(training_set) != len(training)
                or training != [name for name in names[:index] if name in training_set]
                or type(row["earlier_actual_K"]) is not int
                or row["earlier_actual_K"] < 0 or row["earlier_actual_K"] < earlier_k
                or row["earlier_actual_K"] >= record["actual_K"]
                or set(row["expert_evidence"]) != set(program.expert_ids)
                or not row["earlier_prefix_identity"]):
            raise ValueError("Beta OOF forecast must be complete and use only its actual earlier C")
        earlier_k = row["earlier_actual_K"]
        item = CandidateEvidence.from_record(row["expert_evidence"][program.expert_ids[0]])
        view = group["view"]
        if (not item.probability_available or item.decision_id != view["decision_id"]
                or item.candidate_vocabulary != tuple(view["decision"]["candidate_vocabulary"])
                or item.physical_support_digest != view["physical_support_digest"]
                or item.participant_key != view["participant_key"]
                or len(item.expert_identities) != 1
                or item.expert_identities[0].expert_id != program.expert_ids[0]
                or item.expert_identities[0].fit_digest != row["expert_fit_digests"][program.expert_ids[0]]
                or item.expert_identities[0].probability_calibrated):
            raise ValueError("Beta OOF differs from its original raw complete candidate forecast")
        target = item.candidate_vocabulary.index(group["target_candidate"])
        value = float(item.log_probability[target])
        forecasts.append({"row_id": names[index], "candidate_count": len(item.candidate_vocabulary),
            "true_log_probability": None if np.isneginf(value) else value,
            "forecast_identity": {"raw_row_sha256": semantic_sha256(row),
                "group_sha256": semantic_sha256(group), "earlier_prefix_identity": row["earlier_prefix_identity"],
                "base_expert_identities": [identity.record() for identity in item.expert_identities]}})
    lineage = {key: record[key] for key in ("fit_authority_sha256", "native_observation",
        "source_checkpoint_sha256", "final_program", "final_training_receipts_sha256",
        "final_target_contract_sha256", "final_base_experts", "actual_K", "complete_group_labels")}
    expected = beta_uniform_from_oof(forecasts, prior=declared_prior, requested_row_ids=names, lineage=lineage)
    return expected


def validate_beta_uniform_oof(record, program, prior=None):
    """Rebuild the saved full state from original raw rows and C-group labels."""
    declared = validate_beta_uniform_prior(record["beta_calibration"]["prior"])
    if prior is not None and declared != validate_beta_uniform_prior(prior):
        raise ValueError("Beta posterior prior differs from its explicit programme declaration")
    expected = _beta_uniform_oof_state(record, program, declared)
    if semantic_sha256(expected) != semantic_sha256(record["beta_calibration"]):
        raise ValueError("full Beta posterior differs from the original OOF forecasts and C labels")
    return expected


def _validate_pool_oof(record, program):
    """Replay the scalar meta learner from its bound complete ordered OOF rows."""
    groups = record["calibration_groups"]
    rows = record["rows"]
    names = [group["view"]["record_id"] for group in groups]
    if (record["schema"] != POOL_OOF_SCHEMA or tuple(record["captured_expert_ids"]) != program.expert_ids
            or semantic_sha256(record["OOF_program"]) != program.digest()
            or len(groups) < (4 if record["pool_calibration"]["strategy"] == TEMPERATURE_POOL_STRATEGY else 2)
            or len(rows) != len(groups) or len(set(names)) != len(names)
            or [row["held_record_id"] for row in rows] != names):
        raise ValueError("pool OOF must preserve its complete original group and expert roster")
    scores, targets = [], []
    for index, (row, group) in enumerate(zip(rows, groups, strict=True)):
        training = [item["view"]["record_id"] for j, item in enumerate(groups) if j != index]
        if (list(row["training_record_ids"]) != training or row["risk_mass"] != 1.
                or row["phase"] != "pool_probability_oof_inner"):
            raise ValueError("pool OOF row changes its actual held/training risk authority")
        if record["pool_calibration"]["strategy"] == TEMPERATURE_POOL_STRATEGY:
            temperature = row["inner_temperature_calibration"]
            if (temperature["strategy"] != "grouped_temperature" or not row["inner_temperature_OOF_sha256"]
                    or any(item["held_record_id"] == row["held_record_id"]
                        or row["held_record_id"] in item["training_record_ids"] for item in temperature["OOF_groups"])):
                raise ValueError("pool OOF temperature learner consumed its epsilon-held selection")
        evidence = {name: CandidateEvidence.from_record(value) for name, value in row["expert_evidence"].items()}
        pooled = EnsembleScorer(program).score(evidence)
        if not pooled.probability_available or semantic_sha256(pooled.record()) != semantic_sha256(row["pool_evidence"]):
            raise ValueError("pool OOF differs from the original complete arithmetic expert probabilities")
        scores.append(pooled.log_probability)
        targets.append(pooled.candidate_vocabulary.index(group["target_candidate"]))
    strategy = record["pool_calibration"]["strategy"]
    if strategy not in POOL_STRATEGIES:
        raise ValueError("unknown complete-pool probability learner")
    expected = {**uniform_contamination_from_oof(np.asarray(scores), np.asarray(targets, dtype=int)),
        "program_sha256": program.digest(), "strategy": strategy,
        "base_probability_strategy": "grouped_temperature" if strategy == TEMPERATURE_POOL_STRATEGY else "fixed"}
    if semantic_sha256(expected) != semantic_sha256(record["pool_calibration"]):
        raise ValueError("saved epsilon is not the actual captured complete-pool OOF learning result")


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


def _prefix_sources(sources, specs):
    by_id = {spec.expert_id: spec for spec in specs}
    if sources is not None and (set(sources)-set(by_id)
            or any(by_id[name].source_checkpoint_role is None for name in sources)):
        raise ValueError('source checkpoints must belong to a declared source or matched scratch programme')
    if sources is not None and any(
            isinstance(value, FrozenJointFlatSource) and by_id[name].family != 'joint_flat'
            or isinstance(value, FrozenSourceCheckpoint) and by_id[name].family == 'joint_flat'
            for name, value in sources.items()):
        raise TypeError('frozen source checkpoint type differs from its declared numerical expert family')
    return {} if sources is None else {name: value if isinstance(value, (FrozenSourceCheckpoint, FrozenJointFlatSource))
        else FrozenJointFlatSource.from_path(value) if by_id[name].family == 'joint_flat'
        else FrozenSourceCheckpoint.from_path(value) for name, value in sources.items()}


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
    signal_prior: SpatialSignalProgram | None = None
    joint_flat: JointFlatProgram | None = None

    def __post_init__(self):
        if not self.expert_id or self.family not in {"trial", "continuous_ar", "joint_flat"}:
            raise ValueError("calibration expert requires an explicit ID and supported family")
        if self.family == "joint_flat":
            if (not isinstance(self.joint_flat, JointFlatProgram)
                    or any(value is not None for value in (self.recipe, self.profile, self.spatial,
                        self.signal_prior, self.kernel_prior, self.dictionary))):
                raise ValueError("joint history requires its own typed source/sampling/native programme")
            object.__setattr__(self, "_prior_spec", None)
            object.__setattr__(self, "_dictionary_spec", None)
            return
        if self.joint_flat is not None:
            raise ValueError("joint history programme belongs to the joint_flat family")
        if self.family == "trial" and (self.recipe is None or self.profile is None):
            raise ValueError("trial expert requires one target recipe and actual epoch profile")
        if self.spatial is not None:
            object.__setattr__(self, "spatial", freeze_metadata(self.spatial))
        if self.signal_prior is not None and (not isinstance(self.signal_prior, SpatialSignalProgram)
                or self.family != "continuous_ar" or self.spatial is not None):
            raise ValueError("physical signal prior requires its native raw continuous programme")
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
    def source_checkpoint_role(self):
        """One family owner for parameter sources, weights and matched architecture."""
        if self.family == 'joint_flat':
            return 'original_source_response_and_noise_parameters'
        if self.family == 'trial' and self.recipe.initialization == 'source':
            return 'pretrained_weights'
        if self.family == 'trial' and self.recipe.initialization == 'scratch':
            return 'architecture_only'
        return None

    @property
    def source_only_capable(self):
        return self.source_checkpoint_role in {'original_source_response_and_noise_parameters', 'pretrained_weights'}

    @property
    def dictionary_spec(self):
        return self._dictionary_spec

    def record(self):
        if self.family == "joint_flat":
            return {"expert_id": self.expert_id, "family": self.family, "joint_flat": self.joint_flat.record()}
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
        if self.signal_prior is not None:
            record["signal_prior"] = self.signal_prior.record()
        return record

    @classmethod
    def from_record(cls, record):
        if record["family"] == "joint_flat":
            return cls(record["expert_id"], "joint_flat", joint_flat=JointFlatProgram.from_record(record["joint_flat"]))
        values = dict(record)
        values["recipe"] = None if values["recipe"] is None else TargetRecipe(**values["recipe"])
        values["profile"] = None if values["profile"] is None else SelectionEpochProfile.from_record(values["profile"])
        if "kernel_prior" in values:
            values["kernel_prior"] = KernelPriorSpec.from_record(values["kernel_prior"])
        if "dictionary" in values:
            values["dictionary"] = ResponseDictionarySpec.from_record(values["dictionary"])
        if "signal_prior" in values:
            values["signal_prior"] = SpatialSignalProgram.from_record(values["signal_prior"])
        return cls(**values)


@dataclass(frozen=True)
class _FittedExpert:
    program: CalibrationExpertProgram
    predictor: FrozenTrialPredictor | ContinuousDecisionModel | SpatialDecisionModel | JointFlatModel | None
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
            if self.program.signal_prior is not None:
                if (not isinstance(model, SpatialDecisionModel)
                        or semantic_sha256(model.programme.record()) != semantic_sha256(self.program.signal_prior.record())):
                    raise ValueError("physical response expert differs from its actual signal programme")
                model = model.baseline
            elif isinstance(model, SpatialDecisionModel):
                raise ValueError("physical response model requires its declared signal programme")
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
        elif self.program.family == "joint_flat" and (not isinstance(self.predictor, JointFlatModel)
                or metadata_digest(self.predictor.programme.record()) != metadata_digest(self.program.joint_flat.record())):
            raise ValueError("joint expert differs from its actual saved source/sampling programme")

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

    def score(self, view: PredictionDecisionView, *, _epoch_cache=None, history_context=None,
              captured_history=None, history_recorder=None, history_capability_recorder=None,
              history_refusal_recorder=None):
        if self.predictor is None:
            return self.unavailable(view, self.unavailable_reason, self.unavailable_kind)
        if self.program.family == "joint_flat":
            allowed = {None, 'previous_unknown_decision_overlaps_finite_response_support',
                       'next_unknown_decision_overlaps_finite_response_support',
                       'previous_unknown_selection_overlaps_response', 'next_unknown_selection_overlaps_response'}
            if view.unavailable_reason not in allowed:
                return self.unavailable(view, view.unavailable_reason)
            if not view.decision.schedule_complete:
                return self.unavailable(view, 'incomplete_physical_decision')
            if history_context is None:
                raise ValueError("joint history expert requires its original full-record permission context")
            compiler, prefix = history_context
            absolute_first = int(view.onsets[0])+view.raw_scope[0]
            p_stop = absolute_first+self.program.joint_flat.conditioning_samples
            try:
                capability = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=p_stop)
                if history_capability_recorder is not None:
                    history_capability_recorder(self.program.expert_id, capability)
                from research.joint_readset_transition import joint_readset_relation

                relation = joint_readset_relation(compiler, capability, captured_history)
                result = self.predictor.score_history(view, capability, captured=captured_history,
                    **({'_readset_relation': relation} if relation is not None else {}))
            except HistoryUnavailableError as error:
                if history_refusal_recorder is not None:
                    from transfer.ensemble_views import _lookup_record_history_refusal

                    refusal = _lookup_record_history_refusal(compiler, view, prefix=prefix, conditioning_stop=p_stop)
                    if semantic_sha256(refusal.binding['error']) != semantic_sha256(error.record()):
                        raise ValueError('joint history refusal changed its actual original typed error') from error
                    history_refusal_recorder(self.program.expert_id, refusal)
                return self.unavailable(view, json.dumps(error.record(), sort_keys=True, allow_nan=False), error.failure_kind)
            except (UnidentifiableFitError, SamplingResolutionError, np.linalg.LinAlgError) as error:
                failure = {'stage': 'joint_history_read_only_prediction', 'type': type(error).__name__, 'message': str(error)}
                if hasattr(error, 'record'):
                    failure['numerical_evidence'] = error.record()
                return self.unavailable(view, json.dumps(failure, sort_keys=True, allow_nan=False), 'not_available')
            if history_recorder is not None:
                history_recorder(self.program.expert_id, capability, result.capture)
            identity = CandidateExpertIdentity(self.program.expert_id, result.input_digest,
                self.model_digest, self.source_digest, self.fit_digest,
                "joint_flat_same_record_"+self.program.joint_flat.scoring_law+"/1", False)
            counts = tuple(map(int, view.decision.membership.sum(axis=0)))
            return CandidateEvidence.from_log_evidence(result.log_evidence,
                candidate_vocabulary=view.decision.candidate_vocabulary, participant_key=view.participant_key,
                decision_id=view.decision_id, physical_support_digest=view.physical_support_digest,
                expert_identities=(identity,), evidence_level=view.decision.repetitions,
                expected_event_counts=counts, observed_event_counts=counts)
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
            if self.program.signal_prior is not None:
                result = self.predictor.score_view(view)
            else:
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
        if self.program.signal_prior is not None:
            score_kind = "frozen_noise_physical_response_log_evidence/1"
        if self.program.conditioning_samples:
            numerical_digest = semantic_sha256({"raw_input": numerical_digest,
                "conditioning_sample_support": [0, self.program.conditioning_samples],
                "score_sample_support": [self.program.conditioning_samples, len(view.samples)]})
            score_kind = ("frozen_noise_physical_response_conditional_tail/1" if self.program.signal_prior is not None
                          else "continuous_conditional_tail_log_density_fixed_candidate_prior/1")
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


def _attach_oof_probability(fitted, temperature_record, oof_record, *, _producer_recorder=None):
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
    result = FittedEnsemble(fitted.scorer, tuple(experts), fitted._training_receipts, temperature_record,
                          target_contract=fitted._target_contract, oof_probability_record=oof_record)
    if _producer_recorder is not None and id(fitted) in _producer_recorder.native_fit_inputs:
        _producer_recorder.record_native_fit_inputs(result, _producer_recorder.native_fit_inputs[id(fitted)])
    return result


def _attach_pool_probability(fitted, oof_record, learned):
    """Preserve temperature evidence; attach a separate complete-pool producer."""
    digest = semantic_sha256(oof_record)
    experts = []
    for expert in fitted._experts:
        if expert.predictor is None:
            experts.append(expert)
            continue
        binding = {**_model_fit_binding(expert.predictor, expert.program.family),
                   "pool_OOF_probability_sha256": digest, "pool_base_model_digest": expert.model_digest}
        predictor = _with_model_fit_binding(expert.predictor, expert.program.family, binding)
        experts.append(replace(expert, predictor=predictor, model_digest=predictor.digest()))
    scorer = EnsembleScorer(fitted.scorer.program, {**learned, "pool_OOF_probability_sha256": digest})
    return FittedEnsemble(scorer, tuple(experts), fitted._training_receipts, fitted.temperature_record,
        target_contract=fitted._target_contract, oof_probability_record=fitted.oof_probability_record,
        pool_oof_record=oof_record)


class FittedEnsemble:
    """Own expert state, fit authority and one frozen mixture behind score(view)."""

    def __init__(self, scorer, experts, training_receipts, temperature_record=None, *, target_contract=None,
                 oof_probability_record=None, pool_oof_record=None, beta_calibration=None, beta_oof_record=None):
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
        self._pool_oof_record = None if pool_oof_record is None else freeze_metadata(pool_oof_record)
        if self.scorer.pool_calibration is not None and self._pool_oof_record is None:
            raise ValueError("learned pool calibration requires its actual C-only pool OOF binding")
        if self._pool_oof_record is not None:
            self._validate_pool_binding()
        self._beta_calibration = None if beta_calibration is None else freeze_metadata(beta_calibration)
        self._beta_oof_record = None if beta_oof_record is None else freeze_metadata(beta_oof_record)
        if (beta_calibration is None) != (beta_oof_record is None):
            raise ValueError("Beta probability programme requires its complete original forward OOF state")
        if beta_calibration is not None:
            if (len(self._experts) != 1 or self._experts[0].program.family != "joint_flat"
                    or self.temperature_record["strategy"] != "fixed" or self.scorer.pool_calibration is not None
                    or self.oof_probability_record is not None or self.pool_oof_record is not None):
                raise ValueError("forward Beta calibration requires one original raw joint programme")
            record = self._beta_oof_record
            if (record["final_training_receipts_sha256"] != semantic_sha256(self.fit_receipts)
                    or record["final_target_contract_sha256"] != semantic_sha256(self._target_contract)
                    or semantic_sha256(record["final_base_experts"]) != semantic_sha256(
                        [_expert_state_record(expert) for expert in self._experts])
                    or semantic_sha256(record["beta_calibration"]) != semantic_sha256(self._beta_calibration)):
                raise ValueError("Beta calibration differs from its unchanged original final numerical producers")
            validate_beta_uniform_oof(record, self.scorer.program)
            expert = self._experts[0]
            if expert.predictor is not None and _calibration_expert_fit_digest(
                    expert.program, record["fit_authority"], expert.predictor.source.digest) != expert.fit_digest:
                raise ValueError("Beta calibration groups differ from the original numerical model's C fit authority")

    def _validate_oof_binding(self):
        record = self._oof_probability_record
        if (record["schema"] not in {GROUPED_PROBABILITY_OOF_SCHEMA, "n2p3_joint_flat_forward_probability_OOF/1"}
                or semantic_sha256(record["final_program"]) != self.scorer.program.digest()
                or record["final_training_receipts_sha256"] != semantic_sha256(self._training_receipts)
                or record["final_target_contract_sha256"] != semantic_sha256(self._target_contract)):
            raise ValueError("OOF probability evidence differs from the actual final fit authority")
        if record["schema"] == "n2p3_joint_flat_forward_probability_OOF/1":
            if (record['OOF_method'] != 'original_earlier_C_prefix_prequential'
                    or not any(expert.program.family == 'joint_flat' for expert in self._experts)
                    or len(record['rows']) != len(record['calibration_groups'])):
                raise ValueError('joint OOF must retain the original complete forward group roster')
            for row, group in zip(record['rows'], record['calibration_groups'], strict=True):
                if (row['held_record_id'] != group['view']['record_id']
                        or row['earlier_actual_K'] < 0 or row['held_labels_used_for_fit'] is not False):
                    raise ValueError('forward OOF changed the held prediction or earlier-C permission')
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

    def _validate_pool_binding(self):
        record, program = self._pool_oof_record, self.scorer.program
        if (semantic_sha256(record["final_program"]) != program.digest()
                or record["final_training_receipts_sha256"] != semantic_sha256(self._training_receipts)
                or record["final_target_contract_sha256"] != semantic_sha256(self._target_contract)):
            raise ValueError("pool OOF differs from its actual final fit authority")
        native = self._training_receipts[0] if self._target_contract is None else self._target_contract
        if (record["native_observation"] != _native_observation_contract_record(native)
                or record["actual_K"] != (sum(len(row["onsets"]) for row in self._training_receipts)
                    if self._target_contract is None else self._training_receipts[0]["label_use"]["raw"]["requested_labels"])):
            raise ValueError("pool OOF differs from actual K or native observation")
        digest = semantic_sha256(record)
        expected = {row["program"]["expert_id"]: row for row in record["final_base_experts"]}
        for expert in self._experts:
            entry = expected[expert.program.expert_id]
            if semantic_sha256({key: value for key, value in _expert_state_record(expert).items()
                    if key != "model_digest"}) != semantic_sha256({key: value for key, value in entry.items()
                    if key != "model_digest"}):
                raise ValueError("pool OOF differs from its actual final expert programme")
            if expert.predictor is not None:
                binding = _model_fit_binding(expert.predictor, expert.program.family)
                if (binding.get("pool_OOF_probability_sha256") != digest
                        or binding.get("pool_base_model_digest") != entry["model_digest"]):
                    raise ValueError("pool OOF differs from its original numerical model binding")
        if self.scorer.pool_calibration is not None:
            state = self.scorer.pool_calibration
            if (state["pool_OOF_probability_sha256"] != digest
                    or state["base_probability_strategy"] != self.temperature_record["strategy"]
                    or semantic_sha256({key: value for key, value in state.items()
                        if key != "pool_OOF_probability_sha256"}) != semantic_sha256(record["pool_calibration"])):
                raise ValueError("pool calibration differs from the actual OOF or base temperature programme")
        if record["pool_calibration"]["strategy"] == TEMPERATURE_POOL_STRATEGY and (
                self.oof_probability_record is None
                or record["base_temperature_OOF_sha256"] != semantic_sha256(self.oof_probability_record)):
            raise ValueError("pool calibration lost its actual final temperature OOF programme")
        _validate_pool_oof(record, program)

    @property
    def oof_probability_record(self):
        """Complete C-only cross-fit evidence; it is not nested selection or efficacy."""
        return self._oof_probability_record

    @property
    def pool_oof_record(self):
        return self._pool_oof_record

    @property
    def beta_calibration(self):
        return self._beta_calibration

    @property
    def beta_oof_record(self):
        return self._beta_oof_record

    @property
    def temperature_record(self):
        return self._temperature_record

    @property
    def probability_strategy(self):
        return (FORWARD_BETA_STRATEGY if self.beta_calibration is not None else
                self.scorer.pool_calibration["strategy"] if self.scorer.pool_calibration is not None
                else self.temperature_record["strategy"])

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

    def _score_raw_experts(self, view: PredictionDecisionView, *, history_context=None,
                          captured_histories=None, history_recorder=None, history_capability_recorder=None,
                          history_refusal_recorder=None):
        if view.participant_key != self.participant_key:
            raise ValueError("personal ensemble cannot score another participant identity")
        task = (self._training_receipts[0]["decision"] if self._target_contract is None else self._target_contract)
        if tuple(view.decision.candidate_vocabulary) != tuple(task["candidate_vocabulary"]):
            raise ValueError("ensemble candidate vocabulary/order differs from the frozen task")
        for expert in self._experts:
            if expert.program.family == "trial":
                expert.program.profile.validate_observation(view)
        if any(expert.program.family in {"continuous_ar", "joint_flat"} for expert in self._experts):
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
        scored = {expert.program.expert_id: expert.score(view, _epoch_cache=epoch_cache,
                    history_context=history_context,
                    captured_history=None if captured_histories is None else captured_histories.get(expert.program.expert_id),
                    history_recorder=history_recorder, history_capability_recorder=history_capability_recorder,
                    history_refusal_recorder=history_refusal_recorder)
                  for expert in self._experts}
        return scored

    def _apply_probability_program(self, scored):
        if self.beta_calibration is not None:
            return {name: beta_uniform_evidence(item, self.beta_calibration) for name, item in scored.items()}
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

    def score_experts(self, view: PredictionDecisionView, **history):
        return self._apply_probability_program(self._score_raw_experts(view, **history))

    @classmethod
    def score_programs(cls, programs, view: PredictionDecisionView, **history):
        """Score one current view from genuinely shared final expert objects.

        Fitted and reloaded program groups invoke this independently. Raw
        evidence lives only in this call; each probability program applies its
        original transform and keeps unavailable experts in the full roster.
        """
        if not programs or set(programs) - ({"fixed", "grouped_temperature", FORWARD_BETA_STRATEGY} | POOL_STRATEGIES):
            raise ValueError("shared scoring requires declared probability programs")
        available = []
        for name, value in programs.items():
            if isinstance(value, ExpectedSubjectError):
                continue
            if not isinstance(value, cls) or value.probability_strategy != name:
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
        scored = first._score_raw_experts(view, **history)
        return {name: value if isinstance(value, ExpectedSubjectError)
                else value._apply_probability_program(scored) for name, value in programs.items()}

    def score(self, view: PredictionDecisionView, **history):
        return self.scorer.score(self.score_experts(view, **history))

    def save(self, directory: str | Path, *, metadata_store=None):
        from research.source_metadata_store import current_source_metadata_store

        metadata_store = metadata_store or current_source_metadata_store()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        entries = []
        source_references = []
        for index, expert in enumerate(self._experts):
            if expert.predictor is None:
                entries.append({"program": expert.program.record(), "model": None,
                    "model_digest": None, "fit_digest": expert.fit_digest,
                    "source_digest": expert.source_digest, "logit_offset": expert.logit_offset,
                    "unavailable_reason": expert.unavailable_reason, "unavailable_kind": expert.unavailable_kind})
                continue
            filename = f"expert_{index:03d}." + ("pt" if expert.program.family == "trial" else
                "json" if expert.program.signal_prior is not None else "npz")
            path = directory / filename
            if expert.program.family == "trial":
                expert.predictor.save(path, metadata_store=metadata_store)
                if metadata_store is not None and expert.predictor.source_contract_reference is not None:
                    source_references.append(expert.predictor.source_contract_reference)
            else:
                expert.predictor.save(path)
            entries.append({"program": expert.program.record(), "model": filename,
                "model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_digest": expert.model_digest, "fit_digest": expert.fit_digest,
                "source_digest": expert.source_digest, "logit_offset": expert.logit_offset})
        record = {"schema": ("n2p3_fitted_candidate_ensemble/5" if self.beta_calibration is not None else
                             FITTED_ENSEMBLE_SCHEMA if self.pool_oof_record is None
                             else "n2p3_fitted_candidate_ensemble/3"), "program": self.scorer.program.record(),
                  "experts": entries, "training_receipts": list(self._training_receipts),
                  "temperature_calibration": self.temperature_record}
        if self._target_contract is not None:
            record["target_contract"] = self._target_contract
        if self.oof_probability_record is not None:
            record["OOF_probability"] = self.oof_probability_record
        if self.scorer.pool_calibration is not None:
            record["pool_calibration"] = self.scorer.pool_calibration
        if self.pool_oof_record is not None:
            record["pool_OOF_probability"] = self.pool_oof_record
        if self.beta_calibration is not None:
            record.update(beta_calibration=self.beta_calibration, beta_OOF_probability=self.beta_oof_record)
        if source_references:
            from research.source_metadata_store import source_dependency_descriptor

            record["source_metadata"] = source_dependency_descriptor(directory, metadata_store, source_references)
            record["scientific_schema"] = record["schema"]
            record["schema"] = "n2p3_fitted_candidate_ensemble/4"
        atomic_write_json(directory / "ensemble.json", {"record": record, "record_sha256": semantic_sha256(record)})
        return directory

    def export_standalone(self, directory):
        """Write complete numerical/source dependencies in an independent package."""
        from research.source_metadata_store import source_metadata_scope

        with source_metadata_scope(None):
            return self.save(directory)

    @classmethod
    def save_programs(cls, directory, programs, *, participant_key, metadata_store=None):
        """One numerical package, with separate declared probability programs."""
        directory = Path(directory)
        fitted = next((value for value in programs.values() if isinstance(value, cls)), None)
        if fitted is None:
            directory.mkdir(parents=True, exist_ok=False)
            base_digest = None
        else:
            fitted.save(directory, metadata_store=metadata_store)
            base_digest = hashlib.sha256((directory / "ensemble.json").read_bytes()).hexdigest()
        records = {}
        for name, value in programs.items():
            if name not in {"fixed", "grouped_temperature", FORWARD_BETA_STRATEGY} | POOL_STRATEGIES:
                raise ValueError("unknown declared ensemble calibration strategy")
            if isinstance(value, ExpectedSubjectError):
                records[name] = {"status": "unavailable", "failure": value.record(subject=participant_key)}
            else:
                if value._experts is not fitted._experts or value.scorer.program != fitted.scorer.program:
                    raise ValueError("temperature programs must share the actual all-C final expert states")
                if semantic_sha256(value.oof_probability_record) != semantic_sha256(fitted.oof_probability_record):
                    raise ValueError("temperature programs must share the actual captured OOF probabilities")
                if semantic_sha256(value.pool_oof_record) != semantic_sha256(fitted.pool_oof_record):
                    raise ValueError("pool programmes must share their actual captured exact-pool OOF")
                if name != value.probability_strategy:
                    raise ValueError("temperature program name differs from its actual strategy")
                records[name] = {"status": "completed", "temperature_calibration": value.temperature_record}
                if value.scorer.pool_calibration is not None:
                    records[name]["pool_calibration"] = value.scorer.pool_calibration
                if value.beta_calibration is not None:
                    records[name].update(beta_calibration=value.beta_calibration, beta_OOF_probability=value.beta_oof_record)
        has_pool = bool(set(programs) & POOL_STRATEGIES)
        record = {"schema": ("n2p3_prefix_probability_programs/3" if FORWARD_BETA_STRATEGY in programs else
                             "n2p3_prefix_probability_programs/2" if has_pool else "n2p3_prefix_temperature_programs/1"), "base_package_sha256": base_digest,
                  "participant_key": participant_key,
                  "programs": records}
        atomic_write_json(directory / "temperature_programs.json", {"record": record, "record_sha256": semantic_sha256(record)})
        return directory

    @classmethod
    def load_programs(cls, directory, *, metadata_store=None, _joint_source_preparation=None):
        return cls._admit_programs(directory, metadata_store=metadata_store,
                                   _joint_source_preparation=_joint_source_preparation)[1]

    @classmethod
    def _admit_programs(cls, directory, *, package_hashes=None, metadata_store=None, _metadata_admission=None,
                        _joint_source_preparation=None):
        """Read one exact package once; keep decoded states inside this ingress."""
        directory = Path(directory)
        temperature_content = (directory / "temperature_programs.json").read_bytes()
        if (package_hashes is not None and hashlib.sha256(temperature_content).hexdigest()
                != package_hashes["temperature_programs_json_sha256"]):
            raise ValueError("prefix temperature_programs.json differs from captured package")
        payload = json.loads(temperature_content)
        record = payload["record"]
        if record["schema"] not in {"n2p3_prefix_temperature_programs/1", "n2p3_prefix_probability_programs/2", "n2p3_prefix_probability_programs/3"} or semantic_sha256(record) != payload["record_sha256"]:
            raise ValueError("frozen temperature program manifest changed")
        if not record["programs"] or set(record["programs"]) - ({"fixed", "grouped_temperature", FORWARD_BETA_STRATEGY} | POOL_STRATEGIES):
            raise ValueError("unknown declared ensemble calibration strategy")
        if (record["schema"] == "n2p3_prefix_probability_programs/2") != bool(set(record["programs"]) & POOL_STRATEGIES):
            raise ValueError("pool probability strategies require their explicit package schema")
        if (record["schema"] == "n2p3_prefix_probability_programs/3") != (FORWARD_BETA_STRATEGY in record["programs"]):
            raise ValueError("forward Beta strategy requires its explicit package schema")
        for name, value in record["programs"].items():
            if value["status"] not in {"completed", "unavailable"}:
                raise ValueError("temperature program requires an explicit completed or unavailable state")
            if value["status"] == "completed" and name != (
                    FORWARD_BETA_STRATEGY if "beta_calibration" in value else
                    value["pool_calibration"]["strategy"] if "pool_calibration" in value
                    else value["temperature_calibration"]["strategy"]):
                raise ValueError("temperature program name differs from its actual strategy")
        if record["base_package_sha256"] is None:
            if any(value["status"] != "unavailable" for value in record["programs"].values()):
                raise ValueError("available temperature programs require their actual final model package")
            if package_hashes is not None and (package_hashes["ensemble_json_sha256"] is not None
                    or (directory / "ensemble.json").exists()):
                raise ValueError("prefix unavailable roster unexpectedly has an ensemble package")
            return record, {}
        ensemble_content = (directory / "ensemble.json").read_bytes()
        ensemble_digest = hashlib.sha256(ensemble_content).hexdigest()
        if package_hashes is not None and ensemble_digest != package_hashes["ensemble_json_sha256"]:
            raise ValueError("prefix ensemble package differs from captured numerical package")
        if ensemble_digest != record["base_package_sha256"]:
            raise ValueError("temperature programs differ from their actual final model package")
        fitted = cls._load_content(directory, ensemble_content, metadata_store=metadata_store,
                                   _metadata_admission=_metadata_admission,
                                   _joint_source_preparation=_joint_source_preparation)
        if record["participant_key"] != fitted.participant_key:
            raise ValueError("temperature program participant differs from actual final model package")
        return record, {name: cls(EnsembleScorer(fitted.scorer.program, value.get("pool_calibration")), fitted._experts, fitted._training_receipts,
            value["temperature_calibration"], target_contract=fitted._target_contract,
            oof_probability_record=fitted.oof_probability_record, pool_oof_record=fitted.pool_oof_record,
            beta_calibration=value.get("beta_calibration"), beta_oof_record=value.get("beta_OOF_probability"))
            for name, value in record["programs"].items() if value["status"] == "completed"}

    @classmethod
    def load(cls, directory: str | Path, *, metadata_store=None, _joint_source_preparation=None):
        directory = Path(directory)
        return cls._load_content(directory, (directory / "ensemble.json").read_bytes(), metadata_store=metadata_store,
                                 _joint_source_preparation=_joint_source_preparation)

    @classmethod
    def _load_content(cls, directory, content, *, metadata_store=None, _metadata_admission=None,
                      _joint_source_preparation=None):
        """Lower the same bytes whose outer package identity was admitted."""
        payload = json.loads(content)
        record = payload["record"]
        if semantic_sha256(record) != payload["record_sha256"] or record["schema"] not in {
            FITTED_ENSEMBLE_SCHEMA, "n2p3_fitted_candidate_ensemble/3", "n2p3_fitted_candidate_ensemble/4", "n2p3_fitted_candidate_ensemble/5"
        }:
            raise ValueError("frozen ensemble manifest changed or has an unsupported schema")
        from research.source_metadata_store import (
            _SourceMetadataAdmission,
            admit_source_dependencies,
        )

        if record["schema"] == "n2p3_fitted_candidate_ensemble/4":
            metadata_admission = admit_source_dependencies(record["source_metadata"], directory,
                store=metadata_store, admission=_metadata_admission)
            scientific_schema = record["scientific_schema"]
            if scientific_schema not in {FITTED_ENSEMBLE_SCHEMA, "n2p3_fitted_candidate_ensemble/3", "n2p3_fitted_candidate_ensemble/5"}:
                raise ValueError("source-reference package changed its original scientific schema")
        else:
            if "source_metadata" in record or "scientific_schema" in record:
                raise ValueError("legacy ensemble cannot claim shared source dependencies")
            metadata_admission = _metadata_admission or _SourceMetadataAdmission(store=metadata_store)
            scientific_schema = record["schema"]
        if (scientific_schema == "n2p3_fitted_candidate_ensemble/3") != ("pool_OOF_probability" in record):
            raise ValueError("pool probability state requires its explicit model schema")
        if (scientific_schema == "n2p3_fitted_candidate_ensemble/5") != ("beta_OOF_probability" in record):
            raise ValueError("Beta probability state requires its explicit model schema")
        experts = []
        actual_source_references = set()
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
            if program.family == "joint_flat":
                from transfer.joint_flat_model import JointFlatModel
                from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

                if _joint_source_preparation is None:
                    _joint_source_preparation = FrozenJointFlatSourcePreparation()
                predictor = JointFlatModel.load(io.BytesIO(content), source_preparation=_joint_source_preparation)
            else:
                predictor = (FrozenTrialPredictor._load_with_admission(io.BytesIO(content), metadata_admission) if program.family == "trial"
                             else SpatialDecisionModel.load(io.BytesIO(content)) if program.signal_prior is not None
                             else ContinuousDecisionModel.load(io.BytesIO(content)))
            if program.family == "trial" and predictor.source_contract_reference is not None:
                reference = predictor.source_contract_reference
                if entry["source_digest"] != reference["checkpoint_sha256"]:
                    raise ValueError("source-only expert origin differs from its complete source dependency")
                from data.identity import DatasetIdentityTable
                from research.frozen_metadata import plain_metadata

                target = plain_metadata(reference["target_binding"])
                identity = DatasetIdentityTable.from_payload(target["identity"])
                expected_targets = ({record["target_contract"]["participant_key"]} if "target_contract" in record
                    else {receipt["participant_key"] for receipt in record["training_receipts"]})
                if expected_targets != {identity.record_for(target["subject"]).authority_key()}:
                    raise ValueError("source dependency target differs from the actual ensemble fit authority")
                actual_source_references.add(semantic_sha256(reference))
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
                     if key not in {"OOF_probability_sha256", "OOF_base_model_digest",
                                    "pool_OOF_probability_sha256", "pool_base_model_digest"}})
                if original.digest() != base["model_digest"]:
                    raise ValueError("OOF probability evidence differs from the actual final numerical state")
            if "pool_OOF_probability" in record:
                pool = record["pool_OOF_probability"]
                base = next(row for row in pool["final_base_experts"] if row["program"]["expert_id"] == program.expert_id)
                expected_binding.update(pool_OOF_probability_sha256=semantic_sha256(pool),
                                        pool_base_model_digest=base["model_digest"])
                original = _with_model_fit_binding(predictor, program.family,
                    {key: value for key, value in binding.items()
                     if key not in {"pool_OOF_probability_sha256", "pool_base_model_digest"}})
                if original.digest() != base["model_digest"]:
                    raise ValueError("pool OOF differs from the actual final base numerical state")
            if semantic_sha256(binding) != semantic_sha256(expected_binding):
                raise ValueError("frozen ensemble recipe and fit authority differ from the actual model")
            experts.append(_FittedExpert(program, predictor, entry["model_digest"], entry["fit_digest"],
                                         entry["source_digest"], entry["logit_offset"]))
        if record["schema"] == "n2p3_fitted_candidate_ensemble/4" and actual_source_references != {
            semantic_sha256(reference) for reference in record["source_metadata"]["references"]
        }:
            raise ValueError("source dependency closure differs from the actual numerical experts")
        return cls(EnsembleScorer(EnsembleProgram.from_record(record["program"]), record.get("pool_calibration")), experts,
                   record["training_receipts"], record["temperature_calibration"],
                   target_contract=record.get("target_contract"),
                   oof_probability_record=record.get("OOF_probability"), pool_oof_record=record.get("pool_OOF_probability"),
                   beta_calibration=record.get("beta_calibration"), beta_oof_record=record.get("beta_OOF_probability"))


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
        "source_checkpoint_sha256": checkpoint_digest,
        "checkpoint_role": None if checkpoint_digest is None else spec.source_checkpoint_role})


def fit_calibration_ensemble(calibration, expert_programs, program: EnsembleProgram, *,
                             source_checkpoints=None, device="cpu", calibration_strategy="fixed", temperature_expert_ids=None,
                             _fixed_fit_cache=None, _producer_recorder=None, _producer_context=None,
                             _record_self=True, _source_preparation=None, beta_uniform_prior=None):
    """Fit only supplied authorized groups; no prediction data or truth is accepted."""
    calibration = tuple(calibration)
    expert_programs = tuple(expert_programs)
    if calibration_strategy == FORWARD_BETA_STRATEGY or beta_uniform_prior is not None:
        raise ValueError("forward Beta calibration requires the original authorized CalibrationPrefix and compiler")
    if any(spec.family == 'joint_flat' for spec in expert_programs):
        raise ValueError('joint-flat fitting requires the original record-aware CalibrationPrefix and its compiler')
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    if calibration_strategy not in {"fixed", "grouped_temperature"} | POOL_STRATEGIES:
        raise ValueError("unknown declared ensemble calibration strategy")
    if calibration_strategy in POOL_STRATEGIES:
        required = 4 if calibration_strategy == TEMPERATURE_POOL_STRATEGY else 2
        if len(calibration) < required:
            raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                stage="pool_probability", detail=f"fewer than {required} complete pool OOF groups")
        base_strategy = "grouped_temperature" if calibration_strategy == TEMPERATURE_POOL_STRATEGY else "fixed"
        fixed = fit_calibration_ensemble(calibration, expert_programs, program,
            source_checkpoints=source_checkpoints, device=device, calibration_strategy=base_strategy,
            temperature_expert_ids=temperature_expert_ids, _fixed_fit_cache=_fixed_fit_cache,
            _producer_recorder=_producer_recorder, _producer_context=_producer_context,
            _record_self=False, _source_preparation=preparation)
        return _grouped_pool_calibration(fixed, calibration, expert_programs, source_checkpoints, device,
            strategy=calibration_strategy, temperature_expert_ids=temperature_expert_ids,
            _fixed_fit_cache=_fixed_fit_cache, _producer_recorder=_producer_recorder,
            _producer_context=_producer_context, _source_preparation=preparation)
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
    if any(spec.expert_id in sources and spec.source_checkpoint_role is None for spec in expert_programs):
        raise ValueError("source checkpoints belong only to source weights or matched scratch architecture")
    receipts = [dict(group.view.record()) for group in calibration]
    fit_inputs = [group.record() for group in calibration]
    cache_key = None if _fixed_fit_cache is None else semantic_sha256({
        "authorized_calibration": fit_inputs, "expert_programs": [spec.record() for spec in expert_programs],
        "ensemble_program": program.record(), "source_checkpoints": {name: source.digest for name, source in sources.items()},
        "device": str(device), "calibration_strategy": "fixed"})
    if cache_key is not None and cache_key in _fixed_fit_cache:
        fitted = _fixed_fit_cache[cache_key]
        if _producer_recorder is not None and _record_self and calibration_strategy == "fixed":
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
                source_only=False, target_contract=None, source_preparation=None, prefix_compiler=None,
                original_prefix=None):
    source_digest = checkpoint_digest = None
    if spec.family == "joint_flat":
        if not isinstance(source, FrozenJointFlatSource) or prefix_compiler is None or original_prefix is None:
            raise TypeError("joint expert requires its original frozen source and prefix issuer")
        predictor = JointFlatModel.fit_prefix(source, spec.joint_flat, compiler=prefix_compiler, prefix=original_prefix)
        source_digest = checkpoint_digest = source.digest
        offset = 0.
    elif spec.family == "trial":
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
            # ``source_feature_lda`` uses the source trunk only to derive CPU
            # features before sklearn LDA; it has no CUDA fitter to admit.
            compute_device = (device if spec.recipe.initialization in {"source", "scratch"} else "cpu")
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
        options = {"sfreq": rate, "channels": channels, "candidate_ids": vocabulary, "operator_identity": operator,
            "noise_order": spec.noise_order, "conditioning_samples": spec.conditioning_samples,
            "noise_estimator": spec.noise_estimator,
            "dictionary": ResponseDictionary.compile(spec.dictionary_spec, sample_rate_hz=rate),
            **({"ridge": spec.ridge} if spec.prior_spec.kind == "isotropic_ridge" else {"kernel_prior": spec.kernel_prior})}
        predictor = (ContinuousDecisionModel.fit(projected_records, **options) if spec.signal_prior is None else
                     SpatialDecisionModel.fit(projected_records, receipts, spec.signal_prior, **options))
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
                            inner_replay_recorder=None, inner_replay_context=None, _source_preparation=None,
                            prefix_compiler=None, _prefix_fit_cache=None, _joint_history_captures=None,
                            beta_uniform_prior=None):
    """Fit all legitimate prefix rows; complete C groups serve only temperature."""
    if not isinstance(prefix, CalibrationPrefix):
        raise TypeError("prefix fitting requires an authorized CalibrationPrefix")
    prefix.validate()
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    specs = tuple(expert_programs)
    if tuple(spec.expert_id for spec in specs) != program.expert_ids:
        raise ValueError("prefix fit programs must match the complete frozen expert roster")
    if calibration_strategy not in {"fixed", "grouped_temperature", FORWARD_BETA_STRATEGY} | POOL_STRATEGIES:
        raise ValueError("unknown declared ensemble calibration strategy")
    if calibration_strategy in POOL_STRATEGIES or calibration_strategy == FORWARD_BETA_STRATEGY:
        return fit_calibration_prefix_programs(prefix, specs, program, strategies=(calibration_strategy,),
            source_checkpoints=source_checkpoints, device=device, inner_replay_recorder=inner_replay_recorder,
            inner_replay_context=inner_replay_context, _source_preparation=preparation,
            prefix_compiler=prefix_compiler, _prefix_fit_cache=_prefix_fit_cache,
            _joint_history_captures=_joint_history_captures,
            beta_uniform_prior=beta_uniform_prior)[calibration_strategy]
    if beta_uniform_prior is not None:
        raise ValueError("a Beta prior requires the explicit forward Beta strategy")
    if calibration_strategy == "grouped_temperature" and len(prefix.complete_groups) < 3:
        raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
            stage="temperature", detail="fewer than three complete C-only groups")
    sources = _prefix_sources(source_checkpoints, specs)
    if set(sources) - set(program.expert_ids) or any(spec.expert_id in sources and
            spec.source_checkpoint_role is None for spec in specs):
        raise ValueError("prefix sources must belong to declared source or matched scratch experts")
    receipts = prefix.fit_receipts()
    experts = []
    reused = False
    for spec in specs:
        source_only = prefix.budget == 0 and spec.source_only_capable
        data = ((prefix.trial_target(spec.profile) if source_only else
                 prefix.trial_datasets.get(spec.profile.digest)) if spec.family == "trial" else None)
        reason = None
        raw_records = prefix.calibration_records
        if not source_only and (not raw_records or spec.family == "trial" and data is None):
            reason = "K0_target_fitted_expert_unavailable" if prefix.budget == 0 else "no_complete_authorized_trial_rows"
        elif spec.family == "trial" and not source_only and (data.n_epochs < 3 or set(np.unique(data.y)) != {0, 1}):
            reason = "insufficient_authorized_trial_classes_or_rows"
        cache_key = (_calibration_expert_fit_digest(spec, prefix.record(), sources[spec.expert_id].digest)
            if spec.family == 'joint_flat' and spec.expert_id in sources and _prefix_fit_cache is not None else None)
        cached = None if cache_key is None else _prefix_fit_cache.get(cache_key)
        if cached is not None:
            if reason or not isinstance(cached, _FittedExpert) or cached.predictor is None:
                raise ValueError('cached joint primitive is not an available original C producer')
            binding = _model_fit_binding(cached.predictor, cached.program.family)
            if (cached.model_digest != cached.predictor.digest()
                    or {'OOF_probability_sha256', 'pool_OOF_probability_sha256'} & set(binding)):
                raise ValueError('cached joint primitive must retain its original uncalibrated base model digest')
            expert, reused = cached, True
        elif reason:
            fit_digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(), "unavailable": reason})
            expert = _FittedExpert(spec, None, None, fit_digest, None, 0., reason)
        else:
            try:
                expert = _fit_expert(spec, data, raw_records, prefix.target_contract,
                    sources.get(spec.expert_id), receipts, prefix.record(), device=device, source_only=source_only,
                    target_contract=prefix.target_contract, source_preparation=preparation,
                    prefix_compiler=prefix_compiler, original_prefix=prefix)
            except (ExpectedSubjectError, UnidentifiableFitError, SamplingResolutionError, np.linalg.LinAlgError) as error:
                reason = f"{type(error).__name__}: {error}"
                fit_digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(), "fit_failure": reason})
                expert = _FittedExpert(spec, None, None, fit_digest, None, 0., reason, "fit_failure")
        experts.append(expert)
    fitted = FittedEnsemble(EnsembleScorer(program), experts, receipts, target_contract=prefix.target_contract)
    if reused:
        from research.joint_flat_replay import _authority_record, _check_joint_authority

        if prefix_compiler is None:
            raise TypeError('cached joint primitive requires its original current prefix compiler')
        prefix_compiler.assert_original_prefix(prefix)
        # A lookup digest never replaces the actual native sample/label,
        # fitted producer, source, geometry and target-authority join.
        _check_joint_authority(_authority_record(prefix, specs, program, sources,
            inner_replay_context or {}), {'fixed': fitted})
    if _prefix_fit_cache is not None:
        for expert in experts:
            if expert.program.family == 'joint_flat' and expert.predictor is not None:
                # Store the base producer before any OOF probability binding.
                _prefix_fit_cache[expert.fit_digest] = expert
    if calibration_strategy == "fixed":
        if inner_replay_recorder is not None:
            inner_replay_recorder.record_prefix_programs({"fixed": fitted}, prefix=prefix,
                expert_programs=specs, program=program, source_checkpoints=sources, context=inner_replay_context)
        return fitted
    grouped = _grouped_temperature(fitted, prefix.complete_groups, specs, sources, device, prefix=prefix,
                                  _fixed_fit_cache={} if inner_replay_recorder is not None else None,
                                  _producer_recorder=inner_replay_recorder,
                                  _producer_context=inner_replay_context, _source_preparation=preparation,
                                  prefix_compiler=prefix_compiler, _prefix_fit_cache=_prefix_fit_cache,
                                  _joint_history_captures=_joint_history_captures)
    if inner_replay_recorder is not None:
        inner_replay_recorder.record_prefix_programs({"grouped_temperature": grouped}, prefix=prefix,
            expert_programs=specs, program=program, source_checkpoints=sources, context=inner_replay_context)
    return grouped


def _grouped_pool_calibration(fitted, calibration, expert_programs, source_checkpoints, device, *, prefix=None,
                              strategy=POOL_STRATEGY, temperature_expert_ids=None,
                              _fixed_fit_cache=None, _producer_recorder=None, _producer_context=None,
                              _source_preparation=None):
    """Learn epsilon from the complete exact pool's actual C-minus-held fits."""
    if any(spec.family == 'joint_flat' for spec in expert_programs):
        raise ValueError('joint-flat pool calibration requires its original forward nested programme')
    required = 4 if strategy == TEMPERATURE_POOL_STRATEGY else 2
    base_strategy = "grouped_temperature" if strategy == TEMPERATURE_POOL_STRATEGY else "fixed"
    if len(calibration) < required or fitted.temperature_record["strategy"] != base_strategy:
        raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
            stage="pool_probability", detail=f"{base_strategy} pool OOF requires {required} complete C groups")
    sources = {} if source_checkpoints is None else {
        name: value if isinstance(value, FrozenSourceCheckpoint) else FrozenSourceCheckpoint.from_path(value)
        for name, value in source_checkpoints.items()}
    cache = {} if _fixed_fit_cache is None else _fixed_fit_cache
    program, rows, values, targets = fitted.scorer.program, [], [], []
    for index, held in enumerate(calibration):
        training = [group for j, group in enumerate(calibration) if j != index]
        context = {**(_producer_context or {}), "phase": "pool_probability_oof_inner",
                   "pool_held_record_id": held.view.record_id}
        key = semantic_sha256({"authorized_calibration": [group.record() for group in training],
            "expert_programs": [spec.record() for spec in expert_programs], "ensemble_program": program.record(),
            "source_checkpoints": {name: source.digest for name, source in sources.items()},
            "device": str(device), "calibration_strategy": base_strategy,
            **({"temperature_expert_ids": temperature_expert_ids} if base_strategy != "fixed" else {})})
        try:
            inner = fit_calibration_ensemble(training, expert_programs, program, source_checkpoints=sources,
                device=device, calibration_strategy=base_strategy, temperature_expert_ids=temperature_expert_ids,
                _fixed_fit_cache=cache, _producer_recorder=_producer_recorder,
                _producer_context=context, _source_preparation=_source_preparation)
            if base_strategy != "fixed" and _producer_recorder is not None:
                _producer_recorder.record_direct_fit(key, inner, expert_programs=expert_programs,
                    program=program, training=training, context=context)
        except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
            failure = (error if isinstance(error, ExpectedSubjectError) else ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE, stage="pool_inner_fit", detail=str(error)))
            if _producer_recorder is not None:
                _producer_recorder.record_direct_fit(key, failure, expert_programs=expert_programs,
                    program=program, training=training, context=context)
            raise failure from error
        try:
            scores = inner.score_experts(held.view)
        except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
            failure = (error if isinstance(error, ExpectedSubjectError) else ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE, stage="pool_inner_score", detail=str(error)))
            if _producer_recorder is not None:
                _producer_recorder.record_oof_score_failure(key, held=held, training=training,
                                                           error=failure, context=context)
            raise failure from error
        if _producer_recorder is not None:
            _producer_recorder.record_oof_score(key, held=held, training=training, scores=scores, context=context)
        pooled = inner.scorer.score(scores)
        if not pooled.probability_available:
            raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="pool_probability",
                detail="complete requested pool OOF is unavailable: " + pooled.failure_reason)
        rows.append({"held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "training_receipts_sha256": semantic_sha256(inner.fit_receipts),
            "phase": "pool_probability_oof_inner", "risk_mass": 1.,
            "expert_evidence": {name: item.record() for name, item in scores.items()}, "pool_evidence": pooled.record()})
        if base_strategy != "fixed":
            rows[-1].update(inner_temperature_calibration=inner.temperature_record,
                            inner_temperature_OOF_sha256=semantic_sha256(inner.oof_probability_record))
        values.append(pooled.log_probability)
        targets.append(pooled.candidate_vocabulary.index(held.target_candidate))
    learned = {**uniform_contamination_from_oof(np.asarray(values), np.asarray(targets, dtype=int)),
        "program_sha256": program.digest(), "strategy": strategy, "base_probability_strategy": base_strategy}
    native = fitted._training_receipts[0] if fitted._target_contract is None else fitted._target_contract
    oof = {"schema": POOL_OOF_SCHEMA, "evidence_scope": "calibration_development",
        "captured_expert_ids": program.expert_ids, "OOF_program": program.record(),
        "OOF_expert_programs": [spec.record() for spec in expert_programs],
        "calibration_groups": [group.record() for group in calibration], "rows": rows,
        "actual_K": sum(len(group.labels) for group in calibration) if prefix is None else prefix.budget,
        "complete_group_labels": sum(len(group.labels) for group in calibration),
        "fit_authority_sha256": semantic_sha256([group.record() for group in calibration]
                                                 if prefix is None else prefix.record()),
        "native_observation": _native_observation_contract_record(native),
        "source_checkpoint_sha256": {name: source.digest for name, source in sources.items()},
        "final_program": program.record(), "final_training_receipts_sha256": semantic_sha256(fitted.fit_receipts),
        "final_target_contract_sha256": semantic_sha256(fitted._target_contract),
        "final_base_experts": [_expert_state_record(expert) for expert in fitted._experts],
        "pool_calibration": learned, "temperature_calibration": fitted.temperature_record,
        "selection_performed": False, "meta_training_rows_are_out_of_meta": False}
    if base_strategy != "fixed":
        oof["base_temperature_OOF_sha256"] = semantic_sha256(fitted.oof_probability_record)
    return _attach_pool_probability(fitted, oof, learned)


def _grouped_temperature(fitted, calibration, expert_programs, sources, device, temperature_expert_ids=None, *, prefix=None,
                         _fixed_fit_cache=None, _producer_recorder=None, _producer_context=None, _source_preparation=None,
                         prefix_compiler=None, _prefix_fit_cache=None, _joint_history_captures=None):
    if any(spec.family == "joint_flat" for spec in expert_programs):
        return _forward_joint_temperature(fitted, calibration, expert_programs, sources, device,
            prefix=prefix, prefix_compiler=prefix_compiler, temperature_expert_ids=temperature_expert_ids,
            producer_recorder=_producer_recorder, producer_context=_producer_context,
            prefix_fit_cache=_prefix_fit_cache, history_captures=_joint_history_captures)
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
    return _attach_oof_probability(fitted, temperature_record, oof_record, _producer_recorder=_producer_recorder)


def _collect_forward_joint_oof(fitted, calibration, expert_programs, sources, device, *, prefix,
        prefix_compiler, temperature_expert_ids, producer_recorder, producer_context, prefix_fit_cache=None,
        history_captures=None, programme_suffix="forward_temperature", failure_stage="forward_temperature"):
    """All held C groups scored by earlier original C, including lawful K0."""
    if prefix_compiler is None or prefix is None:
        raise TypeError('forward joint OOF requires its original C prefix compiler')
    selected = fitted.scorer.program.expert_ids if temperature_expert_ids is None else tuple(temperature_expert_ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(fitted.scorer.program.expert_ids):
        raise ValueError('forward temperature roster must be explicitly declared')
    selected_specs = tuple(next(spec for spec in expert_programs if spec.expert_id == name) for name in selected)
    inner_program = EnsembleProgram.equal_weights(fitted.scorer.program.name+':'+programme_suffix, selected)
    from transfer.ensemble_views import calibration_prefix_identity

    score_rows, producers, failed = [], [], []
    probabilities = {name: [] for name in selected}
    for held in calibration:
        earlier = prefix_compiler.before_group(prefix, held)
        fit_failure = None
        try:
            inner = fit_calibration_prefix(earlier, selected_specs, inner_program, source_checkpoints={
                name: sources[name] for name in selected if name in sources}, device=device,
                prefix_compiler=prefix_compiler, _prefix_fit_cache=prefix_fit_cache)
        except (ExpectedSubjectError, UnidentifiableFitError, SamplingResolutionError, np.linalg.LinAlgError) as error:
            inner = None
            fit_failure = (error if isinstance(error, ExpectedSubjectError) else ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE, stage='forward_inner_fit', detail=f'{type(error).__name__}: {error}'))
        captures, issued, refusals = {}, {}, {}
        actual_captures, readset_transitions = {}, {}
        def remember(name, capability, capture, captures=captures, actual_captures=actual_captures,
                     inner=inner, held=held, readset_transitions=readset_transitions):
            captures[name] = {'capability_sha256': capability.digest, 'captured_P_measure_sha256': capture.digest}
            actual_captures[name] = (capability, capture)
            from research.joint_readset_transition import joint_readset_record

            transition = joint_readset_record(prefix_compiler, capability, capture)
            if transition is not None:
                readset_transitions[name] = transition
            if history_captures is not None:
                producer = next(expert for expert in inner._experts if expert.program.expert_id == name)
                history_captures[(producer.model_digest, held.view.raw_input_digest,
                    held.view.physical_support_digest)] = capture
        def remember_permission(name, capability, issued=issued):
            issued[name] = capability
        def remember_refusal(name, refusal, refusals=refusals):
            refusals[name] = refusal
        if fit_failure is None:
            restored = {} if history_captures is None else {expert.program.expert_id: history_captures[key]
                for expert in inner._experts if (key := (expert.model_digest, held.view.raw_input_digest,
                    held.view.physical_support_digest)) in history_captures}
            scored = inner.score_experts(held.view, history_context=(prefix_compiler, earlier), history_recorder=remember,
                history_capability_recorder=remember_permission, history_refusal_recorder=remember_refusal,
                captured_histories=restored or None)
        else:
            from research.prefix_inner_replay import unavailable_prefix_expert_scores

            scored = unavailable_prefix_expert_scores(earlier.record(), [spec.record() for spec in selected_specs],
                'fixed', fit_failure.record(subject=held.view.participant_key), held.view)
        failures = {name: {'failure_reason': evidence.failure_reason, 'failure_kind': evidence.failure_kind}
                    for name, evidence in scored.items() if not evidence.probability_available}
        row = {'held_record_id': held.view.record_id, 'earlier_actual_K': earlier.budget,
            'status': 'unavailable' if failures else 'scored', 'expert_failures': failures,
            'fit_failure': None if fit_failure is None else fit_failure.record(subject=held.view.participant_key),
            'expert_fit_digests': {name: evidence.expert_identities[0].fit_digest for name, evidence in scored.items()},
            'earlier_prefix_identity': calibration_prefix_identity(earlier),
            'training_record_ids': [group.view.record_id for group in earlier.complete_groups],
            'training_receipts_sha256': None if inner is None else semantic_sha256(inner.fit_receipts),
            'held_labels_used_for_fit': False, 'original_prediction_captures': captures,
            'expert_evidence': {name: evidence.record() for name, evidence in scored.items()}}
        if refusals:
            row['original_history_refusals'] = {name: {'sha256': refusal.digest, 'binding': refusal.binding}
                                               for name, refusal in refusals.items()}
        if readset_transitions:
            row['original_readset_transitions'] = readset_transitions
        score_rows.append(row)
        producers.append({key: value for key, value in row.items() if key != 'expert_evidence'})
        if producer_recorder is not None:
            record = getattr(producer_recorder, 'record_joint_forward_oof', None)
            if record is None:
                raise TypeError('joint forward replay recorder must capture actual earlier models and P measures')
            record(earlier=earlier, held=held, fitted=inner, row=row, context=producer_context,
                   actual_captures=actual_captures, issued_capabilities=issued,
                   **({'issued_refusals': refusals} if refusals else {}))
        if failures:
            failed.append(row)
            continue
        target_index = held.view.decision.candidate_vocabulary.index(held.target_candidate)
        for name, evidence in scored.items():
            probabilities[name].append((evidence.log_probability, target_index))
    if failed:
        # A numerical failure cannot shorten the requested sequence or turn a
        # held selection into structural warm-up. Preserve every original
        # earlier model/capture/evidence row before declining the meta learner.
        ledger = [{key: row[key] for key in ('held_record_id', 'earlier_actual_K', 'status', 'expert_failures')}
                  for row in score_rows]
        raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage=failure_stage,
            detail='complete original forward OOF probabilities required: '+json.dumps(ledger, sort_keys=True))
    return score_rows, producers, probabilities, inner_program, selected_specs, selected


def _forward_joint_temperature(fitted, calibration, expert_programs, sources, device, *, prefix,
        prefix_compiler, temperature_expert_ids, producer_recorder, producer_context, prefix_fit_cache=None,
        history_captures=None):
    score_rows, producers, probabilities, inner_program, selected_specs, selected = _collect_forward_joint_oof(
        fitted, calibration, expert_programs, sources, device, prefix=prefix, prefix_compiler=prefix_compiler,
        temperature_expert_ids=temperature_expert_ids, producer_recorder=producer_recorder,
        producer_context=producer_context, prefix_fit_cache=prefix_fit_cache, history_captures=history_captures)
    temperatures = {name: 1. for name in fitted.scorer.program.expert_ids}
    for name, rows in probabilities.items():
        temperatures[name] = temperature_from_oof(np.stack([row[0] for row in rows]), np.asarray([row[1] for row in rows]))
    temperature = {'strategy': 'grouped_temperature', 'temperatures': temperatures,
        'temperature_expert_ids': list(selected), 'fixed_control_expert_ids': [name for name in temperatures if name not in selected],
        'temperature_bounds': [.05, 20.], 'score_transform': 'posterior_power_temperature/1',
        'evidence_scope': 'calibration_development', 'OOF_method': 'original_earlier_C_prefix_prequential',
        'OOF_groups': producers, 'final_experts_refitted_on_all_authorized_groups': True,
        'held_prediction_labels_used_for_calibration': False}
    record = {'schema': 'n2p3_joint_flat_forward_probability_OOF/1', 'evidence_scope': 'calibration_development',
        'OOF_method': 'original_earlier_C_prefix_prequential', 'captured_expert_ids': selected,
        'OOF_program': inner_program.record(), 'OOF_expert_programs': [spec.record() for spec in selected_specs],
        'calibration_groups': [group.record() for group in calibration], 'rows': score_rows,
        'actual_K': prefix.budget, 'complete_group_labels': sum(len(group.labels) for group in calibration),
        'fit_authority_sha256': semantic_sha256(prefix.record()),
        'native_observation': _native_observation_contract_record(prefix.target_contract),
        'source_checkpoint_sha256': {name: source.digest for name, source in sources.items()},
        'final_program': fitted.scorer.program.record(), 'final_training_receipts_sha256': semantic_sha256(fitted.fit_receipts),
        'final_target_contract_sha256': semantic_sha256(fitted._target_contract),
        'final_base_experts': [_expert_state_record(expert) for expert in fitted._experts],
        'temperature_calibration': temperature, 'selection_performed': False}
    return _attach_oof_probability(fitted, temperature, record)


def _forward_joint_beta(fitted, calibration, expert_programs, sources, device, *, prior, prefix,
        prefix_compiler, producer_recorder=None, producer_context=None, prefix_fit_cache=None,
        history_captures=None):
    """One new calibration consumer; preserve original numerical producers."""
    prior = validate_beta_uniform_prior(prior)
    score_rows, _, _, inner_program, selected_specs, selected = _collect_forward_joint_oof(
        fitted, calibration, expert_programs, sources, device, prefix=prefix, prefix_compiler=prefix_compiler,
        temperature_expert_ids=None, producer_recorder=producer_recorder, producer_context=producer_context,
        prefix_fit_cache=prefix_fit_cache, history_captures=history_captures, programme_suffix="forward_beta_base",
        failure_stage="forward_beta_probability")
    record = {"schema": BETA_UNIFORM_OOF_SCHEMA, "evidence_scope": "calibration_development",
        "OOF_method": "original_earlier_C_prefix_prequential", "captured_expert_ids": selected,
        "OOF_program": inner_program.record(), "OOF_expert_programs": [spec.record() for spec in selected_specs],
        "calibration_groups": [group.record() for group in calibration], "rows": score_rows,
        "actual_K": prefix.budget, "complete_group_labels": sum(len(group.labels) for group in calibration),
        "fit_authority_sha256": semantic_sha256(prefix.record()),
        "fit_authority": prefix.record(),
        "native_observation": _native_observation_contract_record(prefix.target_contract),
        "source_checkpoint_sha256": {name: source.digest for name, source in sources.items()},
        "final_program": fitted.scorer.program.record(),
        "final_training_receipts_sha256": semantic_sha256(fitted.fit_receipts),
        "final_target_contract_sha256": semantic_sha256(fitted._target_contract),
        "final_base_experts": [_expert_state_record(expert) for expert in fitted._experts],
        "selection_performed": False}
    record["beta_calibration"] = _beta_uniform_oof_state(record, fitted.scorer.program, prior)
    return FittedEnsemble(fitted.scorer, fitted._experts, fitted._training_receipts,
        fitted.temperature_record, target_contract=fitted._target_contract,
        beta_calibration=record["beta_calibration"], beta_oof_record=record)


def fit_calibration_prefix_programs(prefix, expert_programs, program, *, strategies,
                                    source_checkpoints=None, device="cpu", temperature_expert_ids=None,
                                    inner_replay_recorder=None, inner_replay_context=None, _source_preparation=None,
                                    prefix_compiler=None, _prefix_fit_cache=None, _joint_history_captures=None,
                                    beta_uniform_prior=None):
    """Declared temperature programs share one actual all-C final expert fit."""
    if (not strategies or len(set(strategies)) != len(strategies)
            or set(strategies) - ({"fixed", "grouped_temperature", FORWARD_BETA_STRATEGY} | POOL_STRATEGIES)):
        raise ValueError("prefix temperature programs must be explicit and distinct")
    if FORWARD_BETA_STRATEGY in strategies:
        beta_uniform_prior = validate_beta_uniform_prior(beta_uniform_prior)
        if (set(strategies) - {"fixed", FORWARD_BETA_STRATEGY} or len(expert_programs) != 1
                or expert_programs[0].family != "joint_flat" or temperature_expert_ids is not None):
            raise ValueError("forward Beta programme requires only fixed/single-joint raw base and no temperature subset")
        if prefix_compiler is None:
            raise TypeError("forward Beta calibration requires the original prefix compiler")
    elif beta_uniform_prior is not None:
        raise ValueError("a Beta prior requires the explicit forward Beta strategy")
    if any(spec.family == 'joint_flat' for spec in expert_programs) and set(strategies) & POOL_STRATEGIES:
        raise ValueError('joint-flat pool calibration requires a separately implemented forward nested programme; '
                         'the original C-minus-held pool learner is not an admissible joint-flat strategy')
    if len(set(strategies) & POOL_STRATEGIES) > 1 or POOL_STRATEGY in strategies and "grouped_temperature" in strategies:
        raise ValueError("different base-pool OOF learners require separate captured programme invocations")
    sources = _prefix_sources(source_checkpoints, expert_programs)
    result = {}
    preparation = _source_preparation or ScopedSourceCheckpointPreparation()
    replay_cache = {} if inner_replay_recorder is not None else None
    required = {"fixed": 0, "grouped_temperature": 3, POOL_STRATEGY: 2, TEMPERATURE_POOL_STRATEGY: 4,
                FORWARD_BETA_STRATEGY: 0}
    eligible = any(len(prefix.complete_groups) >= required[strategy] for strategy in strategies)
    fixed = (fit_calibration_prefix(prefix, expert_programs, program, source_checkpoints=sources, device=device,
                                    inner_replay_recorder=inner_replay_recorder,
                                    inner_replay_context=inner_replay_context, _source_preparation=preparation,
                                    prefix_compiler=prefix_compiler, _prefix_fit_cache=_prefix_fit_cache,
                                    _joint_history_captures=_joint_history_captures)
             if "fixed" in strategies or eligible else None)
    for strategy in strategies:
        if strategy == "fixed":
            result[strategy] = fixed
        elif len(prefix.complete_groups) < required[strategy]:
            result[strategy] = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                stage="pool_probability" if strategy in POOL_STRATEGIES else "temperature",
                detail=f"fewer than {required[strategy]} complete C-only groups")
        else:
            try:
                if strategy == FORWARD_BETA_STRATEGY:
                    result[strategy] = _forward_joint_beta(fixed, prefix.complete_groups, expert_programs, sources,
                        device, prior=beta_uniform_prior, prefix=prefix, prefix_compiler=prefix_compiler,
                        producer_recorder=inner_replay_recorder, producer_context=inner_replay_context,
                        prefix_fit_cache=_prefix_fit_cache, history_captures=_joint_history_captures)
                elif strategy in POOL_STRATEGIES:
                    base = fixed
                    if strategy == TEMPERATURE_POOL_STRATEGY:
                        base = result.get("grouped_temperature")
                        if not isinstance(base, FittedEnsemble):
                            base = _grouped_temperature(fixed, prefix.complete_groups, expert_programs, sources,
                                device, temperature_expert_ids, prefix=prefix, _fixed_fit_cache=replay_cache,
                                _producer_recorder=inner_replay_recorder, _producer_context=inner_replay_context,
                                _source_preparation=preparation, prefix_compiler=prefix_compiler)
                    result[strategy] = _grouped_pool_calibration(base, prefix.complete_groups, expert_programs,
                        sources, device, prefix=prefix, _fixed_fit_cache=replay_cache,
                        strategy=strategy, temperature_expert_ids=temperature_expert_ids,
                        _producer_recorder=inner_replay_recorder, _producer_context=inner_replay_context,
                        _source_preparation=preparation)
                else:
                    result[strategy] = _grouped_temperature(fixed, prefix.complete_groups, expert_programs, sources,
                                                       device, temperature_expert_ids, prefix=prefix,
                                                       _fixed_fit_cache=replay_cache,
                                                       _producer_recorder=inner_replay_recorder,
                                                       _producer_context=inner_replay_context, _source_preparation=preparation,
                                                       prefix_compiler=prefix_compiler, _prefix_fit_cache=_prefix_fit_cache,
                                                       _joint_history_captures=_joint_history_captures)
            except ExpectedSubjectError as error:
                result[strategy] = error
            except (UnidentifiableFitError, BetaCalibrationResolutionError, np.linalg.LinAlgError) as error:
                result[strategy] = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE,
                    stage="beta_probability" if strategy == FORWARD_BETA_STRATEGY else "temperature",
                    detail=f"{type(error).__name__}: {error}")
    grouped = result.get(TEMPERATURE_POOL_STRATEGY, result.get(POOL_STRATEGY, result.get("grouped_temperature")))
    if isinstance(grouped, FittedEnsemble):
        for name, value in tuple(result.items()):
            if name in {"fixed", "grouped_temperature"} and isinstance(value, FittedEnsemble):
                result[name] = FittedEnsemble(value.scorer, grouped._experts, value._training_receipts,
                    value.temperature_record, target_contract=value._target_contract,
                    oof_probability_record=grouped.oof_probability_record, pool_oof_record=grouped.pool_oof_record)
    if inner_replay_recorder is not None:
        record = getattr(inner_replay_recorder, "record_prefix_programs", None)
        if record is None:
            raise TypeError("inner_replay_recorder must support prefix program capture")
        record(result, prefix=prefix, expert_programs=expert_programs, program=program,
               source_checkpoints=sources, context=inner_replay_context)
    return result
