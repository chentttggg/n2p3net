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
    _calibration_expert_fit_digest,
    _expert_state_record,
    fit_calibration_ensemble,
    validate_calibration_ensemble_groups,
)
from transfer.ensemble_views import (
    CalibrationPrefix,
    CalibrationPrefixCompiler,
    CalibrationSelection,
)
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
        "experts": [_expert_state_record(expert) for expert in fitted._experts],
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

    def __init__(self, sources, device, producer_recorder=None):
        self.sources, self.device, self.cache = freeze_metadata(sources), device, {}
        self._fixed_fit_cache = {}
        self.fit_calls = self.cache_hits = 0
        self.producer_recorder = producer_recorder
        self.producer_context = {"phase": "unclassified"}

    def set_context(self, **context):
        self.producer_context = dict(context)

    def fit(self, groups, candidate):
        sources = {name: self.sources[name] for name in candidate.ensemble_program.expert_ids if name in self.sources}
        key = semantic_sha256({"scope": _scope(groups), "candidate": candidate.record(),
            "sources": {name: source.digest for name, source in sources.items()}, "device": self.device})
        if key in self.cache:
            self.cache_hits += 1
            fitted = self.cache[key]
            if self.producer_recorder is not None:
                self.producer_recorder.record_fit(key, fitted, candidate=candidate, training=groups,
                                                  cache_hit=True, context=self.producer_context)
            return fitted, key
        self.fit_calls += 1
        try:
            if candidate.strategy == "grouped_temperature" and len(groups) < 3:
                raise ExpectedSubjectError(SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS,
                    stage="nested_temperature", detail="fewer than three complete training groups")
            producer_context = {**self.producer_context, "candidate_name": candidate.name,
                                "candidate_record": candidate.record()}
            fitted = fit_calibration_ensemble(groups, candidate.expert_programs, candidate.ensemble_program,
                source_checkpoints=sources, device=self.device, calibration_strategy=candidate.strategy,
                temperature_expert_ids=candidate.temperature_expert_ids, _fixed_fit_cache=self._fixed_fit_cache,
                _producer_recorder=self.producer_recorder, _producer_context=producer_context,
                _record_self=False)
        except ExpectedSubjectError as error:
            fitted = error
        except (UnidentifiableFitError, np.linalg.LinAlgError) as error:
            fitted = ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="nested_fit", detail=str(error))
        self.cache[key] = fitted
        if self.producer_recorder is not None:
            self.producer_recorder.record_fit(key, fitted, candidate=candidate, training=groups,
                                              cache_hit=False, context=self.producer_context)
        return fitted, key


def _selection(groups, candidates, baseline_name, fits):
    scores, traces = {}, {}
    for candidate in candidates:
        predictions, rows = [], []
        for index, held in enumerate(groups):
            training = groups[:index]+groups[index+1:]
            if hasattr(fits, "set_context"):
                context = dict(fits.producer_context)
                context["selection_held_record_id"] = held.view.record_id
                fits.set_context(**context)
            fitted, key = fits.fit(training, candidate)
            row = {"held_record_id": held.view.record_id, "fit_scope": _scope(training), "fit_key": key}
            if isinstance(fitted, ExpectedSubjectError):
                row["status"] = "unavailable"
                row["failure"] = fitted.record(subject=held.view.participant_key)
                failed = _failure_evidence(held, candidate, key, fitted)
                predictions.append(failed)
                row["evidence"] = failed.record()
                if getattr(fits, "producer_recorder", None) is not None:
                    fits.producer_recorder.record_selection_score(key, held=held, evidence=failed)
            else:
                evidence = fitted.score(held.view)
                predictions.append(evidence)
                row.update(status="scored", producer=_final_identity(fitted), evidence=evidence.record())
                if getattr(fits, "producer_recorder", None) is not None:
                    fits.producer_recorder.record_selection_score(key, held=held, evidence=evidence)
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
        if (self.record["schema"] != "n2p3_strict_nested_calibration_selection/1"
                or self.record["evidence_scope"] != "calibration_development"):
            raise ValueError("unsupported nested calibration selection receipt")
        selected = CalibrationProgramCandidate.from_record(self.record["selected_candidate"])
        declared = {candidate["name"]: candidate for candidate in self.record["candidates"]}
        if (len(declared) != len(self.record["candidates"]) or selected.name not in declared
                or semantic_sha256(selected.record()) != semantic_sha256(declared[selected.name])
                or selected.name != self.record["final_selection"]["decision"]["selected_candidate"]):
            raise ValueError("selected declaration differs from the frozen selection programme")
        scope = self.record["calibration_scope"]
        if (semantic_sha256(scope) != semantic_sha256(self.record["final_refit_scope"])
                or semantic_sha256(scope) != semantic_sha256(self.record["final_selection"]["training_scope"])):
            raise ValueError("nested final refit authority differs from its calibration scope")
        if isinstance(final, ExpectedSubjectError):
            if (final.record(subject=self.record["participant_key"]) != self.record["final_failure"]
                    or self.record["final_producer_sha256"] is not None):
                raise ValueError("nested final failure differs from its requested receipt")
            return
        if (self.record["final_failure"] is not None
                or semantic_sha256(_final_identity(final)) != self.record["final_producer_sha256"]):
            raise ValueError("nested selection differs from its actual final model/program/temperature")
        if (selected.ensemble_program != final.scorer.program
                or selected.strategy != final.temperature_record["strategy"]
                or [spec.record() for spec in selected.expert_programs] != [expert.program.record() for expert in final._experts]):
            raise ValueError("selected declaration differs from the actual final fitted programme")
        if selected.strategy == "grouped_temperature" and tuple(final.temperature_record.get(
            "temperature_expert_ids", final.scorer.program.expert_ids)) != (
                selected.ensemble_program.expert_ids if selected.temperature_expert_ids is None
                else selected.temperature_expert_ids):
            raise ValueError("selected temperature subset differs from the actual final programme")
        fit_inputs = scope["group_records"]
        if (final.participant_key != self.record["participant_key"]
                or semantic_sha256(final.fit_receipts) != semantic_sha256([group["view"] for group in fit_inputs])
                or scope["actual_group_labels"] != sum(group["authorized_label_count"] for group in fit_inputs)):
            raise ValueError("nested final model authority differs from its actual calibration scope")
        for expert in final._experts:
            checkpoint_digest = self.record["source_checkpoint_sha256"].get(expert.program.expert_id)
            if _calibration_expert_fit_digest(expert.program, fit_inputs, checkpoint_digest) != expert.fit_digest:
                raise ValueError("nested final label/source authority differs from the actual fitted model")

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


def fit_nested_calibration_selection(calibration, candidates, *, baseline_name, source_checkpoints=None, device="cpu",
                                     inner_replay_recorder=None):
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
    validate_calibration_ensemble_groups(groups, tuple(spec for candidate in candidates for spec in candidate.expert_programs))
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    declared_experts = {spec.expert_id for candidate in candidates for spec in candidate.expert_programs}
    if set(sources)-declared_experts:
        raise ValueError("source declaration contains an unrequested candidate expert")
    fits = _SubsetFits(sources, device, producer_recorder=inner_replay_recorder)
    outer_rows, outer_predictions = [], []
    for index, held in enumerate(groups):
        training = groups[:index]+groups[index+1:]
        fits.set_context(phase="outer_selection_inner", outer_held_record_id=held.view.record_id)
        selected, selection = _selection(training, candidates, baseline_name, fits)
        fits.set_context(phase="outer_refit", outer_held_record_id=held.view.record_id)
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
    fits.set_context(phase="final_selection_inner")
    selected, final_selection = _selection(groups, candidates, baseline_name, fits)
    fits.set_context(phase="final_refit")
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


class _PrefixFits:
    """Same-run whole prefix programmes, each with its real inner recorder."""

    def __init__(self, sources, device, context):
        from transfer.ensemble_fit import ScopedSourceCheckpointPreparation

        self.sources, self.device, self.context = sources, device, dict(context)
        self.preparation = ScopedSourceCheckpointPreparation()
        self.nodes, self.uses, self.score_rows = {}, {}, {}
        self.fit_calls = self.cache_hits = 0

    def fit(self, prefix, candidate, *, held=None, use):
        from research.nested_selection_replay import InnerReplayRecorder
        from research.prefix_inner_replay import prefix_input_identity
        from transfer.ensemble_fit import fit_calibration_prefix_programs

        sources = {name: self.sources[name] for name in candidate.ensemble_program.expert_ids if name in self.sources}
        key = semantic_sha256({"prefix": prefix_input_identity(prefix), "candidate": candidate.record(),
            "sources": {name: source.digest for name, source in sources.items()}, "device": str(self.device)})
        self.uses.setdefault(key, []).append(dict(use))
        if key in self.nodes:
            self.cache_hits += 1
            return self.nodes[key]["programs"][candidate.strategy], key
        context = {**self.context, "phase": "whole_prefix_programme"}
        if held is not None:
            context["requested_Ks"] = [prefix.budget]
            context["requested_to_actual_K"] = {str(prefix.budget): prefix.budget}
            context["test_scope"] = {"start_sample": held.view.raw_scope[0], "stop_sample": held.view.raw_scope[1]}
            context["requested_decision_ids"] = [held.view.record_id]
        else:
            context.setdefault("requested_Ks", [prefix.budget])
            context.setdefault("requested_to_actual_K", {str(prefix.budget): prefix.budget})
        recorder = InnerReplayRecorder()
        programs = fit_calibration_prefix_programs(prefix, candidate.expert_programs, candidate.ensemble_program,
            strategies=(candidate.strategy,), source_checkpoints=sources, device=self.device,
            temperature_expert_ids=candidate.temperature_expert_ids,
            inner_replay_recorder=recorder, inner_replay_context=context, _source_preparation=self.preparation)
        self.fit_calls += 1
        self.nodes[key] = {"prefix": prefix, "candidate": candidate, "programs": programs,
            "sources": sources, "recorder": recorder, "context": context}
        return programs[candidate.strategy], key

    def score(self, key, view):
        from research.prefix_inner_replay import unavailable_prefix_expert_scores
        from transfer.ensemble import EnsembleScorer

        node = self.nodes[key]
        candidate = node["candidate"]
        cached = self.score_rows.setdefault(key, {})
        if view.record_id in cached:
            if cached[view.record_id]["view"].raw_input_digest != view.raw_input_digest:
                raise ValueError("same nested score identity has different actual observations")
            return cached[view.record_id]["evidence"]
        context = node["context"]
        if ("test_scope" not in context or "requested_decision_ids" not in context
                or view.record_id not in context["requested_decision_ids"]):
            raise ValueError("nested prefix scoring requires its predeclared complete held/future scope")
        shared = FittedEnsemble.score_programs(node["programs"], view)
        scored = shared[candidate.strategy]
        if isinstance(scored, ExpectedSubjectError):
            scored = unavailable_prefix_expert_scores(node["prefix"].record(),
                [spec.record() for spec in candidate.expert_programs], candidate.strategy,
                scored.record(subject=view.participant_key), view)
        node["recorder"].record_prefix_future_score(candidate.strategy, view=view, evidence=scored)
        evidence = EnsembleScorer(candidate.ensemble_program).score(scored)
        cached[view.record_id] = {"view": view, "evidence": evidence}
        return evidence


def _prefix_selection_scope(prefix):
    from research.prefix_inner_replay import prefix_input_identity

    return {"actual_K": prefix.budget, "input": prefix_input_identity(prefix)}


def _prefix_common_training_groups(candidates):
    """Structural schedule from declared learners, never observed success."""
    return max(3 if candidate.strategy == "grouped_temperature" else
        0 if all(spec.family == "trial" and spec.recipe.initialization == "source"
                 for spec in candidate.expert_programs) else 1 for candidate in candidates)


def _select_prefix_program(prefix, candidates, baseline_name, compiler, fits, minimum_rows, *, outer_held=None):
    groups = prefix.complete_groups
    required = _prefix_common_training_groups(candidates)
    admitted, excluded, scope_rows = [], [], []
    for held in groups:
        training = compiler.before_group(prefix, held)
        row = {"held_record_id": held.view.record_id, "training_scope": _prefix_selection_scope(training)}
        scope_rows.append(row)
        if len(training.complete_groups) < required:
            excluded.append({**row, "status": "structural_warmup", "minimum_training_groups": required})
        else:
            admitted.append((held, training))
    record = {"training_scope": _prefix_selection_scope(prefix),
        "minimum_selection_rows": minimum_rows, "minimum_training_groups": required,
        "requested_held_record_ids": [held.view.record_id for held, _ in admitted],
        "structural_scope_rows": scope_rows, "structural_warmup_rows": excluded, "candidate_selection_rows": {}}
    if len(admitted) < minimum_rows:
        selected = next(candidate for candidate in candidates if candidate.name == baseline_name)
        record["decision"] = {"selected_candidate": baseline_name,
            "policy": "predeclared_baseline_on_structurally_insufficient_selection_rows/1",
            "requested_rows": len(admitted), "minimum_selection_rows": minimum_rows,
            "candidate_metrics": {}, "promotion_authorized": False}
        return selected, record
    metrics = {}
    for candidate in candidates:
        evidence, rows = [], []
        for held, training in admitted:
            use = {"phase": "prefix_selection_inner", "outer_held_record_id": outer_held,
                "selection_held_record_id": held.view.record_id}
            fitted, key = fits.fit(training, candidate, held=held, use=use)
            value = fits.score(key, held.view)
            evidence.append(value)
            rows.append({"held_record_id": held.view.record_id, "fit_key": key,
                "training_scope": _prefix_selection_scope(training), "evidence": value.record(),
                "fit_program_completed": not isinstance(fitted, ExpectedSubjectError),
                "status": "scored" if value.probability_available else "unavailable"})
        metrics[candidate.name] = _metrics(tuple(held for held, _ in admitted), evidence)
        record["candidate_selection_rows"][candidate.name] = rows
    name, decision = select_calibration_candidate(metrics, baseline_name=baseline_name,
        tie_order=tuple(candidate.name for candidate in candidates))
    record["decision"] = decision
    return next(candidate for candidate in candidates if candidate.name == name), record


class FittedPrefixSelection:
    """Actual whole-prefix selector and its same-run model graph."""

    def __init__(self, final, record, *, fits=None):
        self._final, self._record, self._fits = final, freeze_metadata(record), fits
        if (record["schema"] != "n2p3_strict_nested_prefix_selection/1"
                or record["outer_scores_used_for_final_selection"] is not False):
            raise ValueError("unsupported whole-prefix selection receipt")
        selected = CalibrationProgramCandidate.from_record(record["selected_candidate"])
        declarations = {row["name"]: row for row in record["candidates"]}
        if (len(declarations) != len(record["candidates"]) or selected.name not in declarations
                or semantic_sha256(selected.record()) != semantic_sha256(declarations[selected.name])
                or selected.name != record["final_selection"]["decision"]["selected_candidate"]):
            raise ValueError("selected prefix programme differs from frozen candidate selection")
        if semantic_sha256(record["parent_scope"]) != semantic_sha256(record["final_refit_scope"]):
            raise ValueError("nested prefix final refit did not retain full original C")
        if isinstance(final, ExpectedSubjectError):
            if final.record(subject=record["participant_key"]) != record["final_failure"]:
                raise ValueError("nested prefix final failure changed")
        elif (record["final_failure"] is not None or final.scorer.program != selected.ensemble_program
                or final.temperature_record["strategy"] != selected.strategy
                or semantic_sha256(_final_identity(final)) != record["final_producer_sha256"]):
            raise ValueError("nested prefix actual final numerical programme changed")

    @property
    def record(self):
        return self._record

    @property
    def final(self):
        return self._final

    def score(self, view):
        if self._fits is None:
            if isinstance(self.final, ExpectedSubjectError):
                raise self.final
            return self.final.score(view)
        return self._fits.score(self.record["final_refit_key"], view)

    def save(self, directory):
        from research.prefix_selection_replay import persist_prefix_selection

        return persist_prefix_selection(self, directory)

    @classmethod
    def load(cls, directory, *, expected_selection_sha256):
        from research.prefix_selection_replay import load_prefix_selection

        return load_prefix_selection(directory, expected_selection_sha256=expected_selection_sha256)


def fit_nested_prefix_selection(prefix, candidates, *, baseline_name, prefix_compiler,
                                minimum_selection_rows, source_checkpoints=None, device="cpu", context=None):
    """Run A(P) completely outside h/v, then refit selected A on all actual C.

    Each A retains the accepted single-prefix final fit and its explicitly
    declared complete-group LOO temperature learner.  Its internal dose need
    not equal final dose; both belong to this same frozen learning algorithm.
    """
    if not isinstance(prefix, CalibrationPrefix) or not isinstance(prefix_compiler, CalibrationPrefixCompiler):
        raise TypeError("whole-prefix selection needs original prefix/compiler authority")
    prefix_compiler.assert_original_prefix(prefix)
    if type(minimum_selection_rows) is not int or minimum_selection_rows < 1:
        raise ValueError("minimum selection rows must be an explicit positive structural policy")
    candidates = tuple(candidates)
    names = tuple(candidate.name for candidate in candidates)
    if not candidates or len(set(names)) != len(names) or baseline_name not in names:
        raise ValueError("prefix selection needs a unique frozen candidate roster and baseline")
    profiles = {spec.profile.digest for candidate in candidates for spec in candidate.expert_programs if spec.family == "trial"}
    if not profiles <= set(prefix.label_use["trial"]):
        raise ValueError("candidate profile is absent from original prefix compilation")
    sources = {} if source_checkpoints is None else {name: value if isinstance(value, FrozenSourceCheckpoint)
        else FrozenSourceCheckpoint.from_path(value) for name, value in source_checkpoints.items()}
    if set(sources) - {spec.expert_id for candidate in candidates for spec in candidate.expert_programs}:
        raise ValueError("prefix selector received an undeclared source expert")
    fits = _PrefixFits(sources, device, context or {})
    outer_rows, evidence = [], []
    for held in prefix.complete_groups:
        training = prefix_compiler.before_group(prefix, held)
        selected, selection = _select_prefix_program(training, candidates, baseline_name,
            prefix_compiler, fits, minimum_selection_rows, outer_held=held.view.record_id)
        fitted, key = fits.fit(training, selected, held=held,
            use={"phase": "prefix_outer_refit", "outer_held_record_id": held.view.record_id})
        value = fits.score(key, held.view)
        evidence.append(value)
        outer_rows.append({"held_record_id": held.view.record_id, "selected_candidate": selected.record(),
            "selection": selection, "refit_scope": _prefix_selection_scope(training), "refit_key": key,
            "fit_program_completed": not isinstance(fitted, ExpectedSubjectError),
            "status": "scored" if value.probability_available else "unavailable",
            "evidence": value.record()})
    selected, selection = _select_prefix_program(prefix, candidates, baseline_name,
        prefix_compiler, fits, minimum_selection_rows)
    final, key = fits.fit(prefix, selected, use={"phase": "prefix_final_refit"})
    record = {"schema": "n2p3_strict_nested_prefix_selection/1", "evidence_scope": "calibration_development",
        "participant_key": prefix.target_contract["participant_key"], "parent_scope": _prefix_selection_scope(prefix),
        "candidates": [candidate.record() for candidate in candidates], "baseline": baseline_name,
        "minimum_selection_rows": minimum_selection_rows, "source_checkpoint_sha256": {name: source.digest for name, source in sources.items()},
        "outer_requested": [group.record() for group in prefix.complete_groups], "outer_rows": outer_rows,
        "outer_metrics": None if not evidence else _metrics(prefix.complete_groups, evidence).record(),
        "final_selection": selection, "selected_candidate": selected.record(), "final_refit_key": key,
        "final_refit_scope": _prefix_selection_scope(prefix),
        "final_producer_sha256": None if isinstance(final, ExpectedSubjectError) else semantic_sha256(_final_identity(final)),
        "final_failure": final.record(subject=prefix.target_contract["participant_key"]) if isinstance(final, ExpectedSubjectError) else None,
        "fit_calls": fits.fit_calls, "exact_prefix_cache_hits": fits.cache_hits,
        "outer_scores_used_for_final_selection": False, "formal_product_promotion": False}
    return FittedPrefixSelection(final, record, fits=fits)
