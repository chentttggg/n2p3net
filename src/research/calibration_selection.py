"""Strict group-nested selection of declared calibration probability programs.

The selector sees only its supplied C-side groups. An outer held group is scored
after selection/refit; its truth reaches the original evaluator only afterwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import DecisionPlanContract, DecisionPlanEntry, semantic_sha256
from research.execution import (
    ExpectedSubjectError,
    SubjectFailureCode,
    UnidentifiableFitError,
    atomic_write_json,
)
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    FrozenSourceCheckpoint,
    fit_calibration_ensemble,
)
from transfer.ensemble_views import CalibrationSelection
from transfer.noise_model import freeze_metadata
from transfer.outcomes import CandidateCoverage, DecisionKey


@dataclass(frozen=True)
class CalibrationProgramCandidate:
    name: str
    expert_programs: tuple[CalibrationExpertProgram, ...]
    ensemble_program: EnsembleProgram
    strategy: str = "fixed"
    temperature_expert_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "expert_programs", tuple(self.expert_programs))
        if not self.name or self.name != self.name.strip():
            raise ValueError("calibration candidate needs an exact name")
        if tuple(spec.expert_id for spec in self.expert_programs) != self.ensemble_program.expert_ids:
            raise ValueError("candidate specs must match its complete frozen ensemble roster")
        if self.strategy not in {"fixed", "grouped_temperature"}:
            raise ValueError("nested selection supports declared fixed/grouped-temperature programs only")
        if self.temperature_expert_ids is not None:
            selected = tuple(self.temperature_expert_ids)
            if (self.strategy != "grouped_temperature" or not selected or len(set(selected)) != len(selected)
                    or not set(selected) <= set(self.ensemble_program.expert_ids)):
                raise ValueError("temperature subset must be explicit within the grouped candidate roster")
            object.__setattr__(self, "temperature_expert_ids", selected)

    def record(self):
        return {"name": self.name, "expert_programs": [spec.record() for spec in self.expert_programs],
                "ensemble_program": self.ensemble_program.record(), "strategy": self.strategy,
                "temperature_expert_ids": self.temperature_expert_ids}

    @classmethod
    def from_record(cls, record):
        return cls(record["name"], tuple(CalibrationExpertProgram.from_record(spec) for spec in record["expert_programs"]),
            EnsembleProgram.from_record(record["ensemble_program"]), record["strategy"], record["temperature_expert_ids"])


def _scope(groups):
    return {"actual_group_labels": sum(len(group.labels) for group in groups),
            "group_records": [group.record() for group in groups]}


def _plan(groups):
    return DecisionPlanContract(semantic_sha256([group.view.raw_input_digest for group in groups]),
        semantic_sha256([group.view.record() for group in groups]), (groups[0].view.participant_key,),
        tuple(DecisionPlanEntry(group.view.participant_key, group.view.decision_id, group.target_candidate) for group in groups))


def _metrics(groups, evidence):
    vocabulary = groups[0].view.decision.candidate_vocabulary
    coverage = {DecisionKey(group.view.participant_key, group.view.decision_id): CandidateCoverage.from_mappings(
        dict(zip(vocabulary, map(int, group.view.decision.membership.sum(axis=0)), strict=True)), {}) for group in groups}
    return evaluate_candidate_evidence({0: evidence}, decision_plan=_plan(groups), seeds=(0,),
        candidate_vocabulary=vocabulary, evidence_level=groups[0].view.decision.repetitions, requested_coverage=coverage)


def _final_identity(fitted):
    return {"program": fitted.scorer.program.record(), "training_receipts": fitted.fit_receipts,
        "temperature_calibration": fitted.temperature_record,
        "experts": [{"program": expert.program.record(), "model_digest": expert.model_digest,
            "fit_digest": expert.fit_digest, "source_digest": expert.source_digest,
            "unavailable_reason": expert.unavailable_reason} for expert in fitted._experts],
        "OOF_probability_sha256": semantic_sha256(fitted.oof_probability_record)}


def _failure_evidence(group, candidate, key, error):
    """Keep expected per-fold failure kinds through the original metric join."""
    counts = tuple(map(int, group.view.decision.membership.sum(axis=0)))
    kind = ("not_available" if error.code == SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS else "fit_failure")
    identities = tuple(CandidateExpertIdentity(name, group.view.raw_input_digest, None, None, key,
        f"nested_program_failure:{error.code.value}", False) for name in candidate.ensemble_program.expert_ids)
    return CandidateEvidence(group.view.decision.candidate_vocabulary, None,
        group.view.participant_key, group.view.decision_id, group.view.physical_support_digest,
        identities, group.view.decision.repetitions, counts, counts, failure_kind=kind, failure_reason=str(error))


class _SubsetFits:
    """Run-local memoization of exact immutable C subsets and declared programs."""

    def __init__(self, sources, device):
        self.sources, self.device, self.cache = freeze_metadata(sources), device, {}
        self._fixed_fit_cache = {}
        self.fit_calls = self.cache_hits = 0

    def fit(self, groups, candidate):
        sources = {name: self.sources[name] for name in candidate.ensemble_program.expert_ids if name in self.sources}
        key = semantic_sha256({"scope": _scope(groups), "candidate": candidate.record(),
            "sources": {name: source.digest for name, source in sources.items()}, "device": self.device})
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key], key
        self.fit_calls += 1
        try:
            if candidate.strategy == "grouped_temperature" and len(groups) < 3:
                raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                    stage="nested_temperature", detail="fewer than three complete training groups")
            fitted = fit_calibration_ensemble(groups, candidate.expert_programs, candidate.ensemble_program,
                source_checkpoints=sources, device=self.device, calibration_strategy=candidate.strategy,
                temperature_expert_ids=candidate.temperature_expert_ids, _fixed_fit_cache=self._fixed_fit_cache)
        except ExpectedSubjectError as error:
            fitted = error
        except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
            fitted = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="nested_fit", detail=str(error))
        self.cache[key] = fitted
        return fitted, key


def _selection(groups, candidates, baseline_name, fits):
    scores, traces = {}, {}
    for candidate in candidates:
        predictions, rows = [], []
        for index, held in enumerate(groups):
            training = groups[:index]+groups[index+1:]
            fitted, key = fits.fit(training, candidate)
            row = {"held_record_id": held.view.record_id, "fit_scope": _scope(training), "fit_key": key}
            if isinstance(fitted, ExpectedSubjectError):
                row["status"] = "unavailable"
                row["failure"] = fitted.record(subject=held.view.participant_key)
                failed = _failure_evidence(held, candidate, key, fitted)
                predictions.append(failed)
                row["evidence"] = failed.record()
            else:
                evidence = fitted.score(held.view)
                predictions.append(evidence)
                row.update(status="scored", producer=_final_identity(fitted), evidence=evidence.record())
            rows.append(row)
        scores[candidate.name] = _metrics(groups, predictions)
        traces[candidate.name] = rows
    selected, decision = select_calibration_candidate(scores, baseline_name=baseline_name,
                                                       tie_order=tuple(candidate.name for candidate in candidates))
    return next(candidate for candidate in candidates if candidate.name == selected), {
        "training_scope": _scope(groups), "decision": decision, "candidate_selection_rows": traces}


def select_calibration_candidate(metrics, *, baseline_name, tie_order):
    """Predeclared complete-support hit/NLL/Brier selection, with baseline retained."""
    if set(metrics) != set(tie_order) or len(set(tie_order)) != len(tie_order) or baseline_name not in metrics:
        raise ValueError("selection metrics must cover the frozen complete candidate roster")
    baseline = metrics[baseline_name]
    eligible, comparisons = [baseline_name], {}
    baseline_complete = baseline.probability_coverage == 1.
    for name in tie_order:
        if name == baseline_name:
            continue
        candidate = metrics[name]
        paired = compare_candidate_probability_metrics(baseline, candidate)
        complete = baseline_complete and candidate.probability_coverage == 1.
        brier_ok = (complete and candidate.available_probability_brier is not None
                    and baseline.available_probability_brier is not None
                    and candidate.available_probability_brier <= baseline.available_probability_brier)
        allowed = complete and paired.development_gate_passed and brier_ok
        comparisons[name] = {"paired": paired.record(), "complete_requested_probabilities": complete,
                             "brier_not_increased": brier_ok, "eligible": allowed}
        if allowed:
            eligible.append(name)
    if not baseline_complete:
        selected = baseline_name
    else:
        selected = min(eligible, key=lambda name: (-metrics[name].requested_participant_operational_hit_rate,
            metrics[name].available_probability_nll, metrics[name].available_probability_brier, tie_order.index(name)))
    return selected, {"selected_candidate": selected, "baseline": baseline_name, "tie_order": tie_order,
        "policy": "complete_requested_support_hit_then_nll_then_brier_baseline_guard/1",
        "baseline_complete_probability_support": baseline_complete,
        "candidate_metrics": {name: metrics[name].record() for name in tie_order}, "comparisons": comparisons,
        "promotion_authorized": False}


class FittedCalibrationSelection:
    """One original final model package plus immutable nested selection receipts."""

    def __init__(self, final, record):
        self._final = final
        self._record = freeze_metadata(record)
        if isinstance(final, ExpectedSubjectError):
            if final.record(subject=self.record["participant_key"]) != self.record["final_failure"]:
                raise ValueError("nested final failure differs from its requested receipt")
            return
        if semantic_sha256(_final_identity(final)) != self.record["final_producer_sha256"]:
            raise ValueError("nested selection differs from its actual final model/program/temperature")
        selected = CalibrationProgramCandidate.from_record(self.record["selected_candidate"])
        if (selected.ensemble_program != final.scorer.program
                or selected.strategy != final.temperature_record["strategy"]
                or [spec.record() for spec in selected.expert_programs] != [expert.program.record() for expert in final._experts]):
            raise ValueError("selected declaration differs from the actual final fitted programme")

    @property
    def record(self):
        return self._record

    @property
    def final(self):
        return self._final

    def score(self, view):
        if isinstance(self.final, ExpectedSubjectError):
            raise self.final
        return self.final.score(view)

    def save(self, directory):
        if isinstance(self.final, ExpectedSubjectError):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            base_sha = None
        else:
            directory = self.final.save(directory)
            base_sha = hashlib.sha256((directory/"ensemble.json").read_bytes()).hexdigest()
        atomic_write_json(directory/"calibration_selection.json", {"record": self.record,
            "record_sha256": semantic_sha256(self.record), "final_package_sha256": base_sha})
        return directory

    @classmethod
    def load(cls, directory, *, expected_selection_sha256):
        directory = Path(directory)
        payload = json.loads((directory/"calibration_selection.json").read_bytes())
        digest = semantic_sha256(payload["record"])
        if digest != payload["record_sha256"] or digest != expected_selection_sha256:
            raise ValueError("nested selection receipt or final original package identity changed")
        if payload["final_package_sha256"] is None:
            failure = payload["record"]["final_failure"]
            return cls(ExpectedSubjectError(failure["error_code"], stage=failure["stage"], detail=failure["detail"]), payload["record"])
        if hashlib.sha256((directory/"ensemble.json").read_bytes()).hexdigest() != payload["final_package_sha256"]:
            raise ValueError("nested selection receipt or final original package identity changed")
        return cls(FittedEnsemble.load(directory), payload["record"])


def fit_nested_calibration_selection(calibration, candidates, *, baseline_name, source_checkpoints=None, device="cpu"):
    """Evaluate the complete selector outside each held group, then fit it on all C."""
    groups, candidates = tuple(calibration), tuple(candidates)
    if len(groups) < 3 or any(not isinstance(group, CalibrationSelection) for group in groups):
        raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
            stage="nested_selection", detail="at least three authorized complete groups required")
    names = tuple(candidate.name for candidate in candidates)
    if not candidates or len(set(names)) != len(names) or baseline_name not in names:
        raise ValueError("nested selection requires unique declared candidates and a baseline")
    first = groups[0].view
    if any(group.view.participant_key != first.participant_key or
           group.view.decision.candidate_vocabulary != first.decision.candidate_vocabulary or
           group.view.decision.repetitions != first.decision.repetitions for group in groups):
        raise ValueError("nested calibration requires one participant, task and evidence level")
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    declared_experts = {spec.expert_id for candidate in candidates for spec in candidate.expert_programs}
    if set(sources)-declared_experts:
        raise ValueError("source declaration contains an unrequested candidate expert")
    fits = _SubsetFits(sources, device)
    outer_rows, outer_predictions = [], []
    for index, held in enumerate(groups):
        training = groups[:index]+groups[index+1:]
        selected, selection = _selection(training, candidates, baseline_name, fits)
        fitted, key = fits.fit(training, selected)
        row = {"held_record_id": held.view.record_id, "selection": selection,
               "selected_candidate": selected.record(), "refit_scope": _scope(training), "refit_key": key}
        if isinstance(fitted, ExpectedSubjectError):
            row.update(status="unavailable", failure=fitted.record(subject=held.view.participant_key))
            failed = _failure_evidence(held, selected, key, fitted)
            outer_predictions.append(failed)
            row["evidence"] = failed.record()
        else:
            evidence = fitted.score(held.view)
            outer_predictions.append(evidence)
            row.update(status="scored", producer=_final_identity(fitted), evidence=evidence.record())
        outer_rows.append(row)
    selected, final_selection = _selection(groups, candidates, baseline_name, fits)
    final, key = fits.fit(groups, selected)
    record = {"schema": "n2p3_strict_nested_calibration_selection/1", "evidence_scope": "calibration_development",
        "participant_key": first.participant_key,
        "candidates": [candidate.record() for candidate in candidates], "baseline": baseline_name,
        "calibration_scope": _scope(groups), "source_checkpoint_sha256": {name: source.digest for name, source in sources.items()},
        "outer_requested": _plan(groups).record(), "outer_rows": outer_rows,
        "outer_metrics": _metrics(groups, outer_predictions).record(), "final_selection": final_selection,
        "selected_candidate": selected.record(), "final_refit_key": key, "final_refit_scope": _scope(groups),
        "final_producer_sha256": None if isinstance(final, ExpectedSubjectError) else semantic_sha256(_final_identity(final)),
        "final_failure": final.record(subject=first.participant_key) if isinstance(final, ExpectedSubjectError) else None,
        "fit_calls": fits.fit_calls, "exact_subset_cache_hits": fits.cache_hits,
        "actual_fixed_fit_states": len(fits._fixed_fit_cache),
        "outer_scores_used_for_final_selection": False, "inner_models_persisted": False,
        "independent_participant_count": 1, "formal_product_promotion": False}
    return FittedCalibrationSelection(final, record)
