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
from dataclasses import asdict, dataclass, field
from numbers import Integral

import numpy as np

from models.candidate_evidence import CandidateEvidence
from research.contracts import DecisionPlanContract, semantic_sha256
from transfer.outcomes import (
    CandidateCoverage,
    DecisionKey,
    DecisionOutcome,
    DecisionOutcomeAccounting,
    DecisionStatus,
    build_decision_outcome_accounting,
    normalize_requested_evidence_levels,
    requested_evidence_levels_record,
)


def _mean(values: Sequence[float | None]) -> float | None:
    """Average a fixed roster; missing units cannot silently change its denominator."""
    return float(np.mean(values)) if values and all(value is not None for value in values) else None


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
    evidence_level: int | None
    participants: tuple[ParticipantCandidateMetrics, ...]
    cells: tuple[CandidateEvaluationCell, ...]
    outcomes_by_seed: tuple[tuple[int, tuple[DecisionOutcome, ...]], ...]
    accounting_by_seed: tuple[tuple[int, DecisionOutcomeAccounting], ...]
    requested_evidence_levels: tuple[tuple[DecisionKey, int], ...] | None = None
    requested_evidence_levels_digest: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.requested_evidence_levels is None:
            return
        if self.evidence_level is not None:
            raise ValueError("scalar evidence level and requested horizon plan are mutually exclusive")
        pairs = tuple(self.requested_evidence_levels)
        horizons = normalize_requested_evidence_levels(dict(pairs), requested_decisions=tuple(key for key, _ in pairs))
        object.__setattr__(self, "requested_evidence_levels", horizons)
        object.__setattr__(self, "requested_evidence_levels_digest", semantic_sha256({
            "schema": "n2p3_requested_evidence_levels/1", "decision_plan_digest": self.decision_plan_digest,
            "requested_evidence_levels": requested_evidence_levels_record(horizons)}))

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
        """Conditional decision NLL on the complete requested participant/seed roster."""
        return _mean([row.available_probability_nll for row in self.participants])

    @property
    def available_probability_brier(self) -> float | None:
        return _mean([row.available_probability_brier for row in self.participants])

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
        result = {"schema": "n2p3_candidate_probability_metrics/1",
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
                "probability_diagnostics_scope": "conditional on available decisions; all requested seeds and participants required",
                "probability_participant_count": sum(row.available_probability_seed_count > 0 for row in self.participants),
                "probability_complete_seed_participant_count": sum(row.available_probability_seed_count == len(self.seeds) for row in self.participants),
                "tie_count": self.tie_count, "not_available_count": self.not_available_count,
                "selection_failure_count": self.selection_failure_count,
                "participants": [row.record() for row in self.participants],
                "confidence_diagnostics": candidate_confidence_diagnostics(self),
                "outcome_accounting_by_seed": {str(seed): accounting.to_record() for seed, accounting in self.accounting_by_seed},
                "outcomes_by_seed": {str(seed): [outcome.to_record() for outcome in outcomes]
                                     for seed, outcomes in self.outcomes_by_seed}}
        if self.requested_evidence_levels is not None:
            result.pop("evidence_level")
            result.update(schema="n2p3_candidate_probability_metrics/2", endpoint="full_episode",
                requested_evidence_levels=requested_evidence_levels_record(self.requested_evidence_levels),
                requested_evidence_levels_digest=self.requested_evidence_levels_digest)
        return result


def evaluate_candidate_evidence(
    evidence_by_seed: Mapping[int, Sequence[CandidateEvidence]],
    *,
    decision_plan: DecisionPlanContract,
    seeds: Sequence[int],
    candidate_vocabulary: Sequence[str],
    evidence_level: int | None = None,
    requested_evidence_levels: Mapping[DecisionKey, int] | None = None,
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
    A full-episode request supplies one actual positive horizon per original
    decision instead of a scalar level; missing outputs retain that horizon.
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
    entries = {DecisionKey(entry.participant_key, entry.decision_id): entry for entry in decision_plan.entries}
    if (evidence_level is None) == (requested_evidence_levels is None):
        raise ValueError("provide exactly one of evidence_level or requested_evidence_levels")
    horizons = None
    if requested_evidence_levels is not None:
        horizons = normalize_requested_evidence_levels(requested_evidence_levels, requested_decisions=tuple(entries))
        levels_by_key = dict(horizons)
    else:
        if isinstance(evidence_level, bool) or not isinstance(evidence_level, Integral) or evidence_level < 1:
            raise ValueError("evidence_level must be a positive integer")
        evidence_level = int(evidence_level)
        levels_by_key = dict.fromkeys(entries, evidence_level)
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
            if evidence.candidate_vocabulary != vocabulary or evidence.evidence_level != levels_by_key[key]:
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
            outcome = DecisionOutcome(key, levels_by_key[key], status, candidate_coverage,
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
            outcomes, requested_decisions=tuple(entries),
            **({"evidence_levels": (evidence_level,)} if horizons is None
               else {"requested_evidence_levels": levels_by_key}))))
    selection_failures = {row.participant_key: row.reason for row in decision_plan.participant_selection_failures}
    participants = []
    for participant in decision_plan.requested_participant_keys:
        rows = [row for row in cells if row.outcome.key.subject_id == participant]
        requested = sum(key.subject_id == participant for key in entries)
        denominator = len(rows)
        nll_by_seed, brier_by_seed = [], []
        for seed in seeds:
            available = [row for row in rows if row.seed == seed and row.probability_available]
            nll_by_seed.append(_mean([row.nll for row in available]))
            brier_by_seed.append(_mean([row.brier for row in available]))
        participants.append(ParticipantCandidateMetrics(
            participant, requested, sum(row.correct for row in rows) / denominator if denominator else 0.,
            sum(row.operational_available for row in rows) / denominator if denominator else 0.,
            sum(row.probability_available for row in rows) / denominator if denominator else 0.,
            _mean(nll_by_seed), _mean(brier_by_seed), sum(value is not None for value in nll_by_seed), sum(row.correct for row in rows),
            sum(row.outcome.status == DecisionStatus.TIE for row in rows),
            sum(row.failure_kind == "fit_failure" for row in rows),
            sum(row.failure_kind == "not_available" for row in rows),
            sum(row.failure_kind == "incomplete" for row in rows),
            sum(row.failure_kind == "abstain" for row in rows), selection_failures.get(participant)))
    return CandidateProbabilityMetrics(decision_plan.digest(), vocabulary, seeds, evidence_level,
        tuple(participants), tuple(cells), tuple(outcomes_by_seed), tuple(accounting_by_seed), horizons)


def candidate_confidence_diagnostics(
    metrics: CandidateProbabilityMetrics, *,
    bin_edges: Sequence[float] = tuple(index / 10 for index in range(11)),
) -> dict[str, object]:
    """Describe available probabilities; never add a selection or promotion gate.

    Each seed averages its available complete-probability decisions; seeds then
    average within requested participant, followed by an equal participant
    average. An empty participant/seed makes the fixed-roster summary unavailable.
    Top-label calibration uses a diagnostic uniform draw from the evidence
    owner's tied maxima. This is expected correctness, not an EEG prediction.
    ECE is the absolute bin gap after that same hierarchy of probability masses.
    Wrong confidence conditions on operational unique winners that were wrong.
    No missing distribution is replaced by uniform and no true zero is clipped.
    """
    edges = tuple(float(value) for value in bin_edges)
    if (len(edges) < 2 or edges[0] != 0. or edges[-1] != 1.
            or not all(math.isfinite(value) for value in edges)
            or any(first >= second for first, second in zip(edges, edges[1:], strict=False))):
        raise ValueError("confidence bins must strictly partition [0,1]")
    count = len(edges) - 1
    uniform_nll = math.log(len(metrics.candidate_vocabulary))

    def summarize(units):
        complete = all(unit["fixed_roster_probability_units_complete"] for unit in units)
        fields = ("mean_max_confidence", "wrong_unique_winner_fraction",
                  "wrong_unique_winner_confidence_mass")
        result = {field: _mean([unit[field] for unit in units]) for field in fields}
        result.update(requested_cell_count=sum(unit["requested_cell_count"] for unit in units),
            missing_probability_count=sum(unit["missing_probability_count"] for unit in units),
            probability_count=sum(unit["probability_count"] for unit in units),
            tie_count=sum(unit["tie_count"] for unit in units),
            wrong_unique_winner_count=sum(unit["wrong_unique_winner_count"] for unit in units),
            fixed_roster_probability_units_complete=complete,
            observed_max_confidence=max((unit["observed_max_confidence"] for unit in units
                                         if unit["observed_max_confidence"] is not None), default=None),
            observed_wrong_unique_winner_max_confidence=max(
                (unit["observed_wrong_unique_winner_max_confidence"] for unit in units
                 if unit["observed_wrong_unique_winner_max_confidence"] is not None), default=None))
        fraction = result["wrong_unique_winner_fraction"]
        result["conditional_wrong_unique_winner_mean_confidence"] = (
            result["wrong_unique_winner_confidence_mass"] / fraction if fraction else None)
        bins = []
        for index in range(count):
            rows = [unit["calibration_table"][index] for unit in units]
            mass = _mean([row["hierarchy_mass"] for row in rows])
            confidence_mass = _mean([row["confidence_mass"] for row in rows])
            correctness_mass = _mean([row["correctness_mass"] for row in rows])
            bins.append({"lower": edges[index], "upper": edges[index + 1],
                "probability_count": sum(row["probability_count"] for row in rows),
                "hierarchy_mass": mass, "confidence_mass": confidence_mass,
                "correctness_mass": correctness_mass,
                "mean_confidence": confidence_mass / mass if mass else None,
                "expected_top_label_correctness": correctness_mass / mass if mass else None})
        result["calibration_table"] = bins
        result["binned_top_label_ece"] = (
            sum(abs(row["correctness_mass"] - row["confidence_mass"]) for row in bins)
            if complete else None)
        return result

    participants = []
    for participant in metrics.participants:
        seeds = []
        for seed in metrics.seeds:
            requested = [cell for cell in metrics.cells
                         if cell.outcome.key.subject_id == participant.participant_key and cell.seed == seed]
            available = [cell for cell in requested if cell.probability_available]
            denominator = len(available)
            bin_counts = np.zeros(count, dtype=np.int64)
            confidence_mass = np.zeros(count)
            correctness_mass = np.zeros(count)
            maxima, wrong = [], []
            for cell in available:
                evidence = cell.evidence
                top = evidence.top_candidate_indices
                probability = evidence.probability
                maximum = float(np.max(probability))
                confidence = float(np.mean(probability[list(top)]))
                truth_index = metrics.candidate_vocabulary.index(cell.outcome.target_candidate)
                correctness = float(truth_index in top) / len(top)
                # Endpoint roundoff accepted by CandidateEvidence stays unmodified.
                index = int(np.searchsorted(edges[1:-1], confidence, side="right"))
                bin_counts[index] += 1
                confidence_mass[index] += confidence
                correctness_mass[index] += correctness
                maxima.append(maximum)
                if cell.operational_available and not cell.correct:
                    wrong.append(maximum)
            table = [{"lower": edges[index], "upper": edges[index + 1],
                "probability_count": int(bin_counts[index]),
                "hierarchy_mass": float(bin_counts[index] / denominator) if denominator else None,
                "confidence_mass": float(confidence_mass[index] / denominator) if denominator else None,
                "correctness_mass": float(correctness_mass[index] / denominator) if denominator else None}
                for index in range(count)]
            seeds.append({"seed": seed, "requested_cell_count": len(requested),
                "probability_count": denominator,
                "fixed_roster_probability_units_complete": denominator > 0,
                "missing_probability_count": len(requested) - denominator,
                "unique_winner_count": sum(cell.operational_available for cell in available),
                "tie_count": sum(cell.evidence.tie for cell in available),
                "mean_max_confidence": _mean(maxima),
                "observed_max_confidence": max(maxima, default=None),
                "wrong_unique_winner_count": len(wrong),
                "wrong_unique_winner_fraction": len(wrong) / denominator if denominator else None,
                "wrong_unique_winner_confidence_mass": sum(wrong) / denominator if denominator else None,
                "observed_wrong_unique_winner_max_confidence": max(wrong, default=None),
                "wrong_unique_winner_confidences": sorted(wrong), "calibration_table": table,
                "conditional_wrong_unique_winner_mean_confidence": _mean(wrong),
                "binned_top_label_ece": sum(abs(float(correctness_mass[index] - confidence_mass[index]))
                    for index in range(count)) / denominator if denominator else None})
        summary = summarize(seeds)
        summary.update(participant_key=participant.participant_key,
            requested_decision_count=participant.requested_decisions,
            selection_failure_reason=participant.selection_failure_reason,
            prediction_coverage=participant.prediction_coverage,
            probability_coverage=participant.probability_coverage,
            available_probability_nll=_risk_record(participant.available_probability_nll),
            available_probability_brier=participant.available_probability_brier,
            nll_minus_same_candidate_count_uniform=_risk_record(
                participant.available_probability_nll - uniform_nll
                if participant.available_probability_nll is not None else None), seed_diagnostics=seeds)
        participants.append(summary)
    return {"schema": "n2p3_candidate_confidence_diagnostics/1", "diagnostic_only": True,
        "decision_plan_digest": metrics.decision_plan_digest,
        "candidate_count": len(metrics.candidate_vocabulary), "uniform_nll": uniform_nll,
        "seeds": list(metrics.seeds), "requested_participant_count": len(participants),
        "probability_coverage": metrics.probability_coverage, "prediction_coverage": metrics.prediction_coverage,
        "available_probability_nll": _risk_record(metrics.available_probability_nll),
        "available_probability_brier": metrics.available_probability_brier,
        "nll_minus_same_candidate_count_uniform": _risk_record(
            metrics.available_probability_nll - uniform_nll
            if metrics.available_probability_nll is not None else None),
        "aggregation": "available decisions within seed, then equal requested seeds within participant, then equal requested participants",
        "tie_calibration": "expected correctness and probability of a uniform diagnostic draw over owner-defined tied maxima; no operational winner",
        "bin_policy": "left closed/right open; final bin includes right endpoint and owner-accepted endpoint roundoff; probabilities unchanged",
        "ece_scope": "finite-sample descriptive binned top-label gap; bin-dependent; not a calibration certificate or uncertainty interval",
        "wrong_confidence_scope": "hierarchy-weighted confidence mass divided by hierarchy-weighted wrong unique-winner fraction; no-wrong units contribute zero masses; zero overall fraction yields null",
        "gate_action": "none; no threshold or selection is inferred from these diagnostics",
        **summarize(participants), "participants": participants}


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
    if (baseline.requested_evidence_levels != candidate.requested_evidence_levels
            or baseline.requested_evidence_levels_digest != candidate.requested_evidence_levels_digest):
        raise ValueError("paired evaluation requested horizon plan differs")
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
        before_seeds, after_seeds = [], []
        for seed in baseline.seeds:
            pairs = paired.get((participant, seed), ())
            before_seeds.append(_mean([pair[0] for pair in pairs]))
            after_seeds.append(_mean([pair[1] for pair in pairs]))
        before_risk.append(_mean(before_seeds))
        after_risk.append(_mean(after_seeds))
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
