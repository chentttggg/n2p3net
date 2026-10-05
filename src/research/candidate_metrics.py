"""One requested-outcome seam for candidate probability and operational evaluation.

Truth and requested participants belong to DecisionPlanContract, never to the
scored evidence. Replicates average within participant; they do not add people.
Conditional probability diagnostics and fixed paired risk retain their missing
ledger and cannot replace requested-participant operational hit.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from models.candidate_evidence import CandidateEvidence
from research.contracts import DecisionPlanContract
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionOutcomeAccounting,
    DecisionStatus,
    build_decision_outcome_accounting,
)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _risk_record(value: float | None) -> float | str | None:
    return "positive_infinity" if value == math.inf else "negative_infinity" if value == -math.inf else value


@dataclass(frozen=True)
class CandidateEvaluationCell:
    seed: int
    outcome: DecisionOutcome
    evidence: CandidateEvidence | None
    failure_kind: str | None
    nll: float | None
    brier: float | None

    @property
    def probability_available(self) -> bool:
        return self.nll is not None

    @property
    def operational_available(self) -> bool:
        return self.outcome.predicted_candidate is not None

    @property
    def correct(self) -> bool:
        return self.outcome.status == DecisionStatus.CORRECT


@dataclass(frozen=True)
class ParticipantCandidateMetrics:
    participant_key: str
    requested_decisions: int
    operational_hit_rate: float
    prediction_coverage: float
    probability_coverage: float
    available_probability_nll: float | None
    available_probability_brier: float | None
    available_probability_seed_count: int
    correct_count: int
    tie_count: int
    fit_failure_count: int
    not_available_count: int
    incomplete_count: int
    abstain_count: int
    selection_failure_reason: str | None

    def record(self) -> dict[str, object]:
        result = asdict(self)
        result["available_probability_nll"] = _risk_record(self.available_probability_nll)
        return result


@dataclass(frozen=True)
class CandidateProbabilityMetrics:
    decision_plan_digest: str
    candidate_vocabulary: tuple[str, ...]
    seeds: tuple[int, ...]
    evidence_level: int
    participants: tuple[ParticipantCandidateMetrics, ...]
    cells: tuple[CandidateEvaluationCell, ...]
    outcomes_by_seed: tuple[tuple[int, tuple[DecisionOutcome, ...]], ...]
    accounting_by_seed: tuple[tuple[int, DecisionOutcomeAccounting], ...]

    @property
    def requested_participant_count(self) -> int:
        return len(self.participants)

    @property
    def requested_participant_operational_hit_rate(self) -> float:
        return float(_mean([row.operational_hit_rate for row in self.participants]))

    @property
    def decision_micro_hit_rate(self) -> float:
        return sum(row.correct for row in self.cells) / len(self.cells) if self.cells else 0.

    @property
    def prediction_coverage(self) -> float:
        return float(_mean([row.prediction_coverage for row in self.participants]))

    @property
    def probability_coverage(self) -> float:
        return float(_mean([row.probability_coverage for row in self.participants]))

    @property
    def available_probability_nll(self) -> float | None:
        return _mean([row.available_probability_nll for row in self.participants
                      if row.available_probability_nll is not None])

    @property
    def available_probability_brier(self) -> float | None:
        return _mean([row.available_probability_brier for row in self.participants
                      if row.available_probability_brier is not None])

    @property
    def tie_count(self) -> int:
        return sum(row.tie_count for row in self.participants)

    @property
    def not_available_count(self) -> int:
        return sum(row.not_available_count for row in self.participants)

    @property
    def selection_failure_count(self) -> int:
        return sum(row.selection_failure_reason is not None for row in self.participants)

    def record(self) -> dict[str, object]:
        return {"schema": "n2p3_candidate_probability_metrics/1",
                "decision_plan_digest": self.decision_plan_digest,
                "candidate_vocabulary": list(self.candidate_vocabulary),
                "seeds": list(self.seeds), "evidence_level": self.evidence_level,
                "requested_participant_count": self.requested_participant_count,
                "requested_participant_operational_hit_rate": self.requested_participant_operational_hit_rate,
                "decision_micro_hit_rate": self.decision_micro_hit_rate,
                "prediction_coverage": self.prediction_coverage,
                "probability_coverage": self.probability_coverage,
                "available_probability_nll": _risk_record(self.available_probability_nll),
                "available_probability_brier": self.available_probability_brier,
                "probability_diagnostics_scope": "available probabilities, decisions then seeds within participant",
                "probability_participant_count": sum(row.available_probability_nll is not None for row in self.participants),
                "tie_count": self.tie_count, "not_available_count": self.not_available_count,
                "selection_failure_count": self.selection_failure_count,
                "participants": [row.record() for row in self.participants],
                "outcome_accounting_by_seed": {str(seed): accounting.to_record() for seed, accounting in self.accounting_by_seed},
                "outcomes_by_seed": {str(seed): [outcome.to_record() for outcome in outcomes]
                                     for seed, outcomes in self.outcomes_by_seed}}


def evaluate_candidate_evidence(
    evidence_by_seed: Mapping[int, Sequence[CandidateEvidence]],
    *,
    decision_plan: DecisionPlanContract,
    seeds: Sequence[int],
    candidate_vocabulary: Sequence[str],
    evidence_level: int,
    participant_failures_by_seed: Mapping[int, Mapping[str, str]] | None = None,
    requested_coverage: Mapping[DecisionKey, CandidateCoverage] | None = None,
) -> CandidateProbabilityMetrics:
    """Join label-free outputs to one externally frozen requested truth roster.

    Expected physical coverage can be supplied by the requested views. When it
    is absent, observed output from another seed provides that decision's
    declared schedule; a wholly absent decision retains explicitly unknown
    empty coverage. No final-choice schedule is inferred from the labels.
    ``participant_failures_by_seed`` records expected fit failures only. Contract
    and source-admission errors must propagate from their validation boundary.
    """
    if not isinstance(decision_plan, DecisionPlanContract):
        raise ValueError("evaluation requires a frozen DecisionPlanContract")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    seeds = tuple(sorted(int(seed) for seed in seeds))
    if set(evidence_by_seed) - set(seeds):
        raise ValueError("output seeds differ from the requested seed roster")
    if isinstance(candidate_vocabulary, (str, bytes)):
        raise ValueError("candidate vocabulary must be a sequence of exact identities")
    vocabulary = tuple(candidate_vocabulary)
    if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary) or any(
        not isinstance(value, str) or not value or value != value.strip() for value in vocabulary
    ):
        raise ValueError("candidate vocabulary requires unique exact string identities")
    if isinstance(evidence_level, bool) or not isinstance(evidence_level, Integral) or evidence_level < 1:
        raise ValueError("evidence_level must be a positive integer")
    entries = {DecisionKey(entry.participant_key, entry.decision_id): entry for entry in decision_plan.entries}
    if any(entry.target_candidate not in vocabulary for entry in entries.values()):
        raise ValueError("external truth lies outside the frozen candidate vocabulary")
    participant_failures = participant_failures_by_seed or {}
    if set(participant_failures) - set(seeds):
        raise ValueError("fit-failure seeds differ from the requested seed roster")
    for failures in participant_failures.values():
        if set(failures) - set(decision_plan.participant_keys) or any(
            not isinstance(reason, str) or not reason or reason != reason.strip() for reason in failures.values()
        ):
            raise ValueError("expected fit failures require requested participants and explicit reasons")
    coverage = dict(requested_coverage or {})
    if requested_coverage is not None and set(coverage) != set(entries):
        raise ValueError("requested coverage must identify every requested decision exactly")
    if any(not isinstance(value, CandidateCoverage) or
           set(dict(value.expected_event_counts)) != set(vocabulary) or
           any(count < 1 for _, count in value.expected_event_counts) for value in coverage.values()):
        raise ValueError("requested physical coverage must retain every candidate and positive expected counts")
    indexed: dict[tuple[int, DecisionKey], CandidateEvidence] = {}
    support: dict[DecisionKey, str] = {}
    for seed in seeds:
        for evidence in evidence_by_seed.get(seed, ()):
            if not isinstance(evidence, CandidateEvidence):
                raise ValueError("evaluation consumes validated CandidateEvidence")
            key = DecisionKey(evidence.participant_key, evidence.decision_id)
            cell_key = (seed, key)
            if key not in entries:
                raise ValueError("output contains an unrequested participant/decision")
            if cell_key in indexed:
                raise ValueError("duplicate output for one requested seed/decision")
            if evidence.candidate_vocabulary != vocabulary or evidence.evidence_level != evidence_level:
                raise ValueError("output vocabulary or evidence level differs from the requested task")
            if key in support and support[key] != evidence.physical_support_digest:
                raise ValueError("seed outputs disagree on physical support identity")
            support[key] = evidence.physical_support_digest
            expected = dict(zip(vocabulary, evidence.expected_event_counts, strict=True))
            if key in coverage and dict(coverage[key].expected_event_counts) != expected:
                raise ValueError("output coverage differs from the requested physical schedule")
            if key not in coverage:
                coverage[key] = CandidateCoverage.from_mappings(expected, {})
            if key.subject_id in participant_failures.get(seed, {}) and evidence.failure_kind != "fit_failure":
                raise ValueError("fit-failed participant cannot also publish nonfailed evidence")
            indexed[cell_key] = evidence
    cells: list[CandidateEvaluationCell] = []
    outcomes_by_seed = []
    accounting_by_seed = []
    for seed in seeds:
        outcomes = []
        for key, entry in entries.items():
            evidence = indexed.get((seed, key))
            reason = participant_failures.get(seed, {}).get(key.subject_id)
            expected = dict(coverage[key].expected_event_counts) if key in coverage else {}
            actual = (dict(zip(vocabulary, evidence.observed_event_counts, strict=True)) if evidence else {})
            candidate_coverage = CandidateCoverage.from_mappings(expected, actual)
            if evidence is None:
                failure_kind = "fit_failure" if reason else "not_available"
                status = DecisionStatus.FIT_FAILURE if reason else DecisionStatus.ABSTAIN
                reason = reason or "requested_output_not_available"
            elif evidence.probability_available:
                failure_kind = None
                status = (DecisionStatus.TIE if evidence.tie else DecisionStatus.CORRECT
                          if evidence.predicted == entry.target_candidate else DecisionStatus.INCORRECT)
                reason = None
            else:
                failure_kind = evidence.failure_kind
                status = {"fit_failure": DecisionStatus.FIT_FAILURE, "incomplete": DecisionStatus.INCOMPLETE,
                          "not_available": DecisionStatus.ABSTAIN, "abstain": DecisionStatus.ABSTAIN}[failure_kind]
                reason = evidence.failure_reason
            outcome = DecisionOutcome(key, int(evidence_level), status, candidate_coverage,
                target_candidate=entry.target_candidate, predicted_candidate=evidence.predicted if evidence else None,
                failure_reason=reason)
            nll = brier = None
            if evidence is not None and evidence.probability_available:
                target = vocabulary.index(entry.target_candidate)
                nll = float(-evidence.log_probability[target])
                squared = evidence.probability
                squared[target] -= 1.
                brier = float(squared @ squared)
            cells.append(CandidateEvaluationCell(seed, outcome, evidence, failure_kind, nll, brier))
            outcomes.append(outcome)
        outcomes = tuple(outcomes)
        outcomes_by_seed.append((seed, outcomes))
        accounting_by_seed.append((seed, build_decision_outcome_accounting(
            outcomes, requested_decisions=tuple(entries), evidence_levels=(int(evidence_level),))))
    selection_failures = {row.participant_key: row.reason for row in decision_plan.participant_selection_failures}
    participants = []
    for participant in decision_plan.requested_participant_keys:
        rows = [row for row in cells if row.outcome.key.subject_id == participant]
        requested = sum(key.subject_id == participant for key in entries)
        denominator = len(rows)
        nll_by_seed, brier_by_seed = [], []
        for seed in seeds:
            available = [row for row in rows if row.seed == seed and row.probability_available]
            if available:
                nll_by_seed.append(float(_mean([row.nll for row in available])))
                brier_by_seed.append(float(_mean([row.brier for row in available])))
        participants.append(ParticipantCandidateMetrics(
            participant, requested, sum(row.correct for row in rows) / denominator if denominator else 0.,
            sum(row.operational_available for row in rows) / denominator if denominator else 0.,
            sum(row.probability_available for row in rows) / denominator if denominator else 0.,
            _mean(nll_by_seed), _mean(brier_by_seed), len(nll_by_seed), sum(row.correct for row in rows),
            sum(row.outcome.status == DecisionStatus.TIE for row in rows),
            sum(row.failure_kind == "fit_failure" for row in rows),
            sum(row.failure_kind == "not_available" for row in rows),
            sum(row.failure_kind == "incomplete" for row in rows),
            sum(row.failure_kind == "abstain" for row in rows), selection_failures.get(participant)))
    return CandidateProbabilityMetrics(decision_plan.digest(), vocabulary, seeds, int(evidence_level),
        tuple(participants), tuple(cells), tuple(outcomes_by_seed), tuple(accounting_by_seed))


@dataclass(frozen=True)
class MissingCandidateProbability:
    seed: int
    key: DecisionKey
    baseline_reason: str | None
    candidate_reason: str | None


@dataclass(frozen=True)
class CandidateProbabilityComparison:
    delta_hit: float
    paired_baseline_nll: float | None
    paired_candidate_nll: float | None
    delta_paired_nll: float | None
    paired_probability_count: int
    missing_probability_ledger: tuple[MissingCandidateProbability, ...]
    rescued_count: int
    harmed_count: int
    lost_prediction_count: int
    lost_probability_count: int
    development_gate_passed: bool
    gate_reasons: tuple[str, ...]
    participant_delta_hit: tuple[tuple[str, float], ...]

    def record(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("paired_baseline_nll", "paired_candidate_nll", "delta_paired_nll"):
            result[name] = _risk_record(getattr(self, name))
        result["paired_probability_scope"] = "fixed seed/decision pairs, averaged within requested participant"
        result["missing_probability_ledger"] = [
            {"seed": row.seed, "participant_key": row.key.subject_id, "decision_id": row.key.decision_id,
             "baseline_reason": row.baseline_reason, "candidate_reason": row.candidate_reason}
            for row in self.missing_probability_ledger]
        return result


def compare_candidate_probability_metrics(
    baseline: CandidateProbabilityMetrics, candidate: CandidateProbabilityMetrics,
) -> CandidateProbabilityComparison:
    """Apply the declared development rule without deleting difficult outputs.

    Every requested participant/seed must contribute a paired probability to
    make the NLL gate available. Missing decisions remain in the ledger and in
    operational hit; baseline probability and prediction outputs must survive.
    This is a development point-estimate gate, not adult confirmation.
    """
    if baseline.decision_plan_digest != candidate.decision_plan_digest:
        raise ValueError("paired evaluation requires identical decision plan identity")
    if baseline.seeds != candidate.seeds or baseline.candidate_vocabulary != candidate.candidate_vocabulary or baseline.evidence_level != candidate.evidence_level:
        raise ValueError("paired evaluation seed/vocabulary/evidence support differs")
    first = {(row.seed, row.outcome.key): row for row in baseline.cells}
    second = {(row.seed, row.outcome.key): row for row in candidate.cells}
    paired: dict[tuple[str, int], list[tuple[float, float]]] = {}
    missing = []
    rescued = harmed = lost_prediction = lost_probability = 0
    for cell_key, before in first.items():
        after = second[cell_key]
        if before.evidence is not None and after.evidence is not None:
            if before.evidence.physical_support_digest != after.evidence.physical_support_digest:
                raise ValueError("paired decision physical support identity differs")
            if before.evidence.expected_event_counts != after.evidence.expected_event_counts:
                raise ValueError("paired decision physical coverage support differs")
        rescued += not before.correct and after.correct
        harmed += before.correct and not after.correct
        lost_prediction += before.operational_available and not after.operational_available
        lost_probability += before.probability_available and not after.probability_available
        if before.probability_available and after.probability_available:
            pair_key = (before.outcome.key.subject_id, before.seed)
            paired.setdefault(pair_key, []).append((before.nll, after.nll))
        else:
            missing.append(MissingCandidateProbability(before.seed, before.outcome.key,
                None if before.probability_available else before.outcome.failure_reason,
                None if after.probability_available else after.outcome.failure_reason))
    participant_keys = tuple(row.participant_key for row in baseline.participants)
    complete_support = all((participant, seed) in paired for participant in participant_keys for seed in baseline.seeds)
    before_risk, after_risk = [], []
    for participant in participant_keys:
        seed_risks = [np.mean(paired[(participant, seed)], axis=0) for seed in baseline.seeds
                      if (participant, seed) in paired]
        if seed_risks:
            risks = np.mean(seed_risks, axis=0)
            before_risk.append(float(risks[0]))
            after_risk.append(float(risks[1]))
    baseline_nll, candidate_nll = _mean(before_risk), _mean(after_risk)
    delta_nll = None
    if complete_support and not (baseline_nll == math.inf and candidate_nll == math.inf):
        delta_nll = candidate_nll - baseline_nll
    delta_hit = candidate.requested_participant_operational_hit_rate - baseline.requested_participant_operational_hit_rate
    reasons = []
    if delta_hit < 0:
        reasons.append("hit_decreased")
    if delta_nll is None:
        reasons.append("paired_nll_unavailable")
    elif delta_nll > 0:
        reasons.append("nll_increased")
    if lost_prediction:
        reasons.append("baseline_prediction_output_lost")
    if lost_probability:
        reasons.append("baseline_probability_output_lost")
    before_failures = Counter((row.seed, row.outcome.key.subject_id) for row in baseline.cells if row.failure_kind is not None)
    after_failures = Counter((row.seed, row.outcome.key.subject_id) for row in candidate.cells if row.failure_kind is not None)
    if any(after_failures[key] > before_failures[key] for key in after_failures):
        reasons.append("participant_failure_increased")
    participant_delta = tuple((before.participant_key, after.operational_hit_rate - before.operational_hit_rate)
                              for before, after in zip(baseline.participants, candidate.participants, strict=True))
    return CandidateProbabilityComparison(delta_hit, baseline_nll, candidate_nll, delta_nll,
        sum(len(rows) for rows in paired.values()), tuple(missing), int(rescued), int(harmed),
        int(lost_prediction), int(lost_probability), not reasons, tuple(dict.fromkeys(reasons)), participant_delta)
