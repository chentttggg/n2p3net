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
from research.contracts import semantic_sha256
from research.execution import (
    ExpectedSubjectError,
    SubjectFailureCode,
    UnidentifiableFitError,
    atomic_write_json,
)
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.calibration_state import FrozenTrialPredictor
from transfer.checkpoint import PreparedTransferCheckpoint
from transfer.continuous_decision import (
    CalibrationRecord,
    ContinuousDecisionModel,
    smooth_erp_basis,
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

FITTED_ENSEMBLE_SCHEMA = "n2p3_fitted_candidate_ensemble/2"


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


def _predictor_digest(predictor):
    return predictor.digest()


def _native_observation_contract(view):
    return (view.sample_rate_hz, view.channel_names, view.channel_types, view.reference, view.signal_unit)


@dataclass(frozen=True)
class CalibrationExpertProgram:
    expert_id: str
    family: str
    recipe: TargetRecipe | None = None
    profile: SelectionEpochProfile | None = None
    noise_order: int = 8
    ridge: float | None = None
    response_seconds: float = 1.2
    n_basis: int = 8
    conditioning_samples: int = 0
    noise_estimator: str = "burg"
    kernel_prior: KernelPriorSpec | None = None
    _prior_spec: KernelPriorSpec | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.expert_id or self.family not in {"trial", "continuous_ar"}:
            raise ValueError("calibration expert requires an explicit ID and supported family")
        if self.family == "trial" and (self.recipe is None or self.profile is None):
            raise ValueError("trial expert requires one target recipe and actual epoch profile")
        if self.family == "continuous_ar" and (self.recipe is not None or self.profile is not None):
            raise ValueError("continuous expert consumes native raw, without an epoch recipe or profile")
        if self.family == "trial":
            if self.kernel_prior is not None:
                raise ValueError("continuous kernel priors cannot be assigned to trial experts")
            object.__setattr__(self, "ridge", 1. if self.ridge is None else self.ridge)
            object.__setattr__(self, "_prior_spec", None)
        else:
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
        return record

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        values["recipe"] = None if values["recipe"] is None else TargetRecipe(**values["recipe"])
        values["profile"] = None if values["profile"] is None else SelectionEpochProfile.from_record(values["profile"])
        if "kernel_prior" in values:
            values["kernel_prior"] = KernelPriorSpec.from_record(values["kernel_prior"])
        return cls(**values)


@dataclass
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
        elif self.program.family == "continuous_ar":
            model = self.predictor
            expected_basis = smooth_erp_basis(model.sfreq, response_seconds=self.program.response_seconds,
                                              n_basis=self.program.n_basis)
            actual_prior = (KernelPriorSpec.isotropic_ridge(model.provenance["ridge"]) if model.kernel_prior is None
                            else model.kernel_prior.spec)
            if (model.noise_order != self.program.noise_order
                    or model.conditioning_samples != self.program.conditioning_samples
                    or actual_prior != self.program.prior_spec
                    or model.provenance.get("noise_estimator", "burg") != self.program.noise_estimator
                    or not np.array_equal(model.basis, expected_basis)):
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

    def score(self, view: PredictionDecisionView):
        if self.predictor is None:
            return self.unavailable(view, self.unavailable_reason, self.unavailable_kind)
        if view.samples is None:
            return self.unavailable(view, view.unavailable_reason)
        if not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0):
            return self.unavailable(view, "incomplete_physical_decision")
        if self.program.family == "trial":
            epochs, mask, receipt = self.program.profile.transform(view)
            logits = self.predictor.predict_logit(epochs, mask)
            numerical_digest = receipt["numerical_input_digest"]
            scores = membership_decision_scores(logits, view.decision, logit_offset=self.logit_offset)
            identity = CandidateExpertIdentity(self.program.expert_id, numerical_digest,
                self.model_digest, self.source_digest, self.fit_digest, scores.score_kind, False)
            return CandidateEvidence.from_membership_scores(scores, view.decision,
                participant_key=view.participant_key, decision_id=view.decision_id,
                physical_support_digest=view.physical_support_digest, expert_identities=(identity,))
        result = self.predictor.score(view.samples, view.onsets, view.decision.membership,
            operator_identity="native_unfiltered_volts/1", sfreq=view.sample_rate_hz,
            channels=view.channel_names, candidate_ids=view.decision.candidate_vocabulary,
            record_id=view.record_id)
        numerical_digest = view.raw_input_digest
        score_kind = "continuous_candidate_log_evidence"
        if self.program.conditioning_samples:
            numerical_digest = semantic_sha256({"raw_input": view.raw_input_digest,
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


class FittedEnsemble:
    """Own expert state, fit authority and one frozen mixture behind score(view)."""

    def __init__(self, scorer, experts, training_receipts, temperature_record=None, *, target_contract=None):
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
        else:
            raise ValueError("unknown declared ensemble calibration strategy")

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

    def score_experts(self, view: PredictionDecisionView):
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
        scored = {expert.program.expert_id: expert.score(view) for expert in self._experts}
        if self.temperature_record["strategy"] == "fixed":
            return scored
        calibrated = {}
        calibration_digest = semantic_sha256(self.temperature_record)
        for name, item in scored.items():
            if not item.probability_available:
                calibrated[name] = item
                continue
            identities = tuple(replace(identity, probability_calibrated=True,
                fit_digest=semantic_sha256({"expert_fit": identity.fit_digest, "calibration": calibration_digest}),
                score_kind=identity.score_kind + ":posterior_power_temperature/1") for identity in item.expert_identities)
            values = item.log_probability / self.temperature_record["temperatures"][name]
            calibrated[name] = replace(item, log_probability=values - logsumexp(values), expert_identities=identities)
        return calibrated

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
            value["temperature_calibration"], target_contract=fitted._target_contract)
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
            if _predictor_digest(predictor) != entry["model_digest"]:
                raise ValueError("frozen ensemble model state differs from its declared identity")
            binding = (predictor.record["ensemble_fit_binding"] if program.family == "trial"
                       else predictor.provenance["ensemble_fit_binding"])
            expected_binding = {"program": program.record(), "training_receipts": record["training_receipts"],
                                "model_source_digest": entry["source_digest"], "fit_digest": entry["fit_digest"],
                                "logit_offset": entry["logit_offset"]}
            if "target_contract" in record:
                expected_binding["target_contract"] = record["target_contract"]
            if semantic_sha256(binding) != semantic_sha256(expected_binding):
                raise ValueError("frozen ensemble recipe and fit authority differ from the actual model")
            experts.append(_FittedExpert(program, predictor, entry["model_digest"], entry["fit_digest"],
                                         entry["source_digest"], entry["logit_offset"]))
        return cls(EnsembleScorer(EnsembleProgram.from_record(record["program"])), experts,
                   record["training_receipts"], record["temperature_calibration"],
                   target_contract=record.get("target_contract"))


def fit_calibration_ensemble(calibration, expert_programs, program: EnsembleProgram, *,
                             source_checkpoints=None, device="cpu", calibration_strategy="fixed"):
    """Fit only supplied authorized groups; no prediction data or truth is accepted."""
    calibration = tuple(calibration)
    expert_programs = tuple(expert_programs)
    if not calibration or any(not isinstance(group, CalibrationSelection) for group in calibration):
        raise ValueError("ensemble fitting requires authorized CalibrationSelection groups")
    if calibration_strategy not in {"fixed", "grouped_temperature"}:
        raise ValueError("unknown declared ensemble calibration strategy")
    if calibration_strategy == "grouped_temperature" and len(calibration) < 3:
        raise ValueError("grouped probability calibration needs at least three complete groups")
    if tuple(spec.expert_id for spec in expert_programs) != program.expert_ids:
        raise ValueError("fit programs must match the frozen ensemble roster and order")
    participant = calibration[0].view.participant_key
    vocabulary = calibration[0].view.decision.candidate_vocabulary
    if any(group.view.participant_key != participant or group.view.decision.candidate_vocabulary != vocabulary
           for group in calibration):
        raise ValueError("ensemble calibration requires one participant and frozen candidate task")
    if any(not group.view.decision.schedule_complete or np.any(group.view.decision.evidence_indices < 0)
           for group in calibration):
        raise ValueError("ensemble calibration requires complete observed physical groups")
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
    experts = []
    for spec in expert_programs:
        data = build_calibration_epoch_dataset(calibration, spec.profile) if spec.family == "trial" else None
        records = [CalibrationRecord(group.view.record_id, group.view.samples, group.view.onsets, group.labels)
                   for group in calibration] if spec.family == "continuous_ar" else []
        experts.append(_fit_expert(spec, data, records, receipts[0], sources.get(spec.expert_id),
            receipts, fit_inputs, device=device))
    fitted = FittedEnsemble(EnsembleScorer(program), experts, receipts)
    if calibration_strategy == "fixed":
        return fitted
    return _grouped_temperature(fitted, calibration, expert_programs, sources, device)


def _fit_expert(spec, data, raw_records, target, source, receipts, fit_inputs, *, device,
                source_only=False, target_contract=None):
    source_digest = checkpoint_digest = None
    if spec.family == "trial":
        source_trunk = source_stats = prepared = None
        if source is not None:
            prepared = PreparedTransferCheckpoint(io.BytesIO(source.content), data)
            source_identity = DatasetIdentityTable.from_payload(prepared.payload["training_identity_ledger"])
            if set(source_dataset_ids_from_identity(source_identity)) & set(source_dataset_ids_from_identity(data.identity_table)):
                raise ValueError("ensemble source must exclude the complete target study")
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
            fitted = fit_calibration_candidate(data, spec.recipe, np.ones(data.n_epochs, dtype=bool),
                source_trunk=source_trunk, source_input_stats=source_stats, device=device)
            predictor = FrozenTrialPredictor.from_fitted(fitted)
            prior = float(data.y.mean())
            offset = (0. if fitted.adapter is not None else float(np.log(prior / (1 - prior))))
    else:
        rate = target["sample_rate_hz"]
        vocabulary = target["candidate_vocabulary"] if "candidate_vocabulary" in target else target["decision"]["candidate_vocabulary"]
        predictor = ContinuousDecisionModel.fit(raw_records, sfreq=rate, channels=target["channel_names"],
            candidate_ids=vocabulary, operator_identity="native_unfiltered_volts/1", noise_order=spec.noise_order,
            **({"ridge": spec.ridge} if spec.prior_spec.kind == "isotropic_ridge" else {"kernel_prior": spec.kernel_prior}),
            conditioning_samples=spec.conditioning_samples, noise_estimator=spec.noise_estimator,
            basis=smooth_erp_basis(rate, response_seconds=spec.response_seconds, n_basis=spec.n_basis))
        offset = 0.
    fit_digest = semantic_sha256({"program": spec.record(), "authorized_calibration": fit_inputs,
        "source_checkpoint_sha256": checkpoint_digest, "checkpoint_role": (None if checkpoint_digest is None else
            "pretrained_weights" if spec.recipe.initialization == "source" else "architecture_only")})
    binding = {"program": spec.record(), "training_receipts": receipts,
               "model_source_digest": source_digest, "fit_digest": fit_digest, "logit_offset": offset}
    if target_contract is not None:
        binding["target_contract"] = target_contract
    predictor = (predictor.with_fit_binding(binding) if spec.family == "trial" else
                 replace(predictor, provenance={**predictor.provenance, "ensemble_fit_binding": binding}))
    return _FittedExpert(spec, predictor, _predictor_digest(predictor), fit_digest, source_digest, offset)


def fit_calibration_prefix(prefix: CalibrationPrefix, expert_programs, program: EnsembleProgram, *,
                            source_checkpoints=None, device="cpu", calibration_strategy="fixed"):
    """Fit all legitimate prefix rows; complete C groups serve only temperature."""
    if not isinstance(prefix, CalibrationPrefix):
        raise TypeError("prefix fitting requires an authorized CalibrationPrefix")
    prefix.validate()
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
                    target_contract=prefix.target_contract)
            except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
                reason = f"{type(error).__name__}: {error}"
                fit_digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(), "fit_failure": reason})
                expert = _FittedExpert(spec, None, None, fit_digest, None, 0., reason, "fit_failure")
        experts.append(expert)
    fitted = FittedEnsemble(EnsembleScorer(program), experts, receipts, target_contract=prefix.target_contract)
    if calibration_strategy == "fixed":
        return fitted
    return _grouped_temperature(fitted, prefix.complete_groups, specs, sources, device)


def _grouped_temperature(fitted, calibration, expert_programs, sources, device):
    program = fitted.scorer.program
    by_expert = {name: [] for name in program.expert_ids}
    producers = []
    for held_index, held in enumerate(calibration):
        training = [group for index, group in enumerate(calibration) if index != held_index]
        inner = fit_calibration_ensemble(training, expert_programs, program,
            source_checkpoints=sources, device=device, calibration_strategy="fixed")
        scores = inner.score_experts(held.view)
        target = held.view.decision.candidate_vocabulary.index(held.target_candidate)
        for name, item in scores.items():
            if not item.probability_available:
                raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="temperature",
                    detail=f"{name}: grouped temperature fit requires complete OOF probability evidence")
            by_expert[name].append((item.log_probability, target))
        producers.append({"held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "expert_fit_digests": {name: item.expert_identities[0].fit_digest for name, item in scores.items()}})
    temperatures = {}
    for name, rows in by_expert.items():
        log_probability = np.stack([row[0] for row in rows])
        targets = np.asarray([row[1] for row in rows], dtype=int)

        solution = minimize_scalar(_temperature_nll, args=(log_probability, targets),
                                   bounds=(np.log(.05), np.log(20.)), method="bounded")
        if not solution.success or not np.isfinite(solution.fun):
            raise ExpectedSubjectError(SubjectFailureCode.NONFINITE_OPTIMIZATION, stage="temperature",
                detail="grouped temperature fit did not produce a finite declared calibration")
        temperatures[name] = float(np.exp(solution.x))
    temperature_record = {"strategy": "grouped_temperature", "temperatures": temperatures,
        "temperature_bounds": [.05, 20.], "score_transform": "posterior_power_temperature/1",
        "evidence_scope": "calibration_development", "OOF_groups": producers,
        "final_experts_refitted_on_all_authorized_groups": True,
        "held_prediction_labels_used_for_calibration": False}
    return FittedEnsemble(fitted.scorer, fitted._experts, fitted._training_receipts, temperature_record,
                          target_contract=fitted._target_contract)


def fit_calibration_prefix_programs(prefix, expert_programs, program, *, strategies,
                                    source_checkpoints=None, device="cpu"):
    """Declared temperature programs share one actual all-C final expert fit."""
    if (not strategies or len(set(strategies)) != len(strategies)
            or set(strategies) - {"fixed", "grouped_temperature"}):
        raise ValueError("prefix temperature programs must be explicit and distinct")
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    result = {}
    eligible = len(prefix.complete_groups) >= 3
    fixed = (fit_calibration_prefix(prefix, expert_programs, program, source_checkpoints=sources, device=device)
             if "fixed" in strategies or eligible else None)
    for strategy in strategies:
        if strategy == "fixed":
            result[strategy] = fixed
        elif not eligible:
            result[strategy] = ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                stage="temperature", detail="fewer than three complete C-only groups")
        else:
            try:
                result[strategy] = _grouped_temperature(fixed, prefix.complete_groups, expert_programs, sources, device)
            except ExpectedSubjectError as error:
                result[strategy] = error
            except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
                result[strategy] = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE,
                    stage="temperature", detail=f"{type(error).__name__}: {error}")
    return result
