"""Typed, auditable outcomes for candidate-set decisions.

The decision status is the primary record. Aggregate rates are derived from
these records, never used to reconstruct integer counts.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class DecisionStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIE = "tie"
    ABSTAIN = "abstain"
    INCOMPLETE = "incomplete"
    FIT_FAILURE = "fit_failure"


@dataclass(frozen=True, order=True)
class DecisionKey:
    subject_id: str
    decision_id: str

    def __post_init__(self) -> None:
        if not self.subject_id or not self.decision_id:
            raise ValueError("subject_id and decision_id must be non-empty.")


def _canonical_counts(
    counts: Mapping[str, int], *, name: str
) -> tuple[tuple[str, int], ...]:
    canonical: list[tuple[str, int]] = []
    for candidate, count in counts.items():
        candidate = str(candidate)
        if not candidate:
            raise ValueError(f"{name} contains an empty candidate identifier.")
        count = int(count)
        if count < 0:
            raise ValueError(f"{name}[{candidate!r}] must be non-negative.")
        canonical.append((candidate, count))
    return tuple(sorted(canonical))


@dataclass(frozen=True)
class CandidateCoverage:
    """Expected and observed event counts for every decision candidate."""

    expected_event_counts: tuple[tuple[str, int], ...]
    observed_event_counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_mappings(
        cls,
        expected_event_counts: Mapping[str, int],
        observed_event_counts: Mapping[str, int],
    ) -> CandidateCoverage:
        return cls(
            expected_event_counts=_canonical_counts(
                expected_event_counts, name="expected_event_counts"
            ),
            observed_event_counts=_canonical_counts(
                observed_event_counts, name="observed_event_counts"
            ),
        )

    def __post_init__(self) -> None:
        expected = dict(self.expected_event_counts)
        observed = dict(self.observed_event_counts)
        if len(expected) != len(self.expected_event_counts):
            raise ValueError("expected_event_counts contains duplicate candidates.")
        if len(observed) != len(self.observed_event_counts):
            raise ValueError("observed_event_counts contains duplicate candidates.")
        _canonical_counts(expected, name="expected_event_counts")
        _canonical_counts(observed, name="observed_event_counts")
        unexpected = sorted(set(observed) - set(expected))
        if unexpected:
            raise ValueError(f"observed_event_counts has unexpected candidates: {unexpected}")

    @property
    def expected_events(self) -> int:
        return sum(count for _, count in self.expected_event_counts)

    @property
    def observed_events(self) -> int:
        return sum(count for _, count in self.observed_event_counts)

    @property
    def missing_event_counts(self) -> dict[str, int]:
        observed = dict(self.observed_event_counts)
        return {
            candidate: expected - observed.get(candidate, 0)
            for candidate, expected in self.expected_event_counts
            if observed.get(candidate, 0) < expected
        }

    @property
    def excess_event_counts(self) -> dict[str, int]:
        expected = dict(self.expected_event_counts)
        return {
            candidate: observed - expected[candidate]
            for candidate, observed in self.observed_event_counts
            if observed > expected[candidate]
        }

    @property
    def complete(self) -> bool:
        return not self.missing_event_counts and not self.excess_event_counts

    def to_record(self) -> dict[str, object]:
        expected = dict(self.expected_event_counts)
        observed = dict(self.observed_event_counts)
        return {
            "expected_event_counts": expected,
            "observed_event_counts": observed,
            "expected_events": self.expected_events,
            "observed_events": self.observed_events,
            "observed_candidates": sum(count > 0 for count in observed.values()),
            "expected_candidates": len(expected),
            "missing_event_counts": self.missing_event_counts,
            "excess_event_counts": self.excess_event_counts,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class DecisionOutcome:
    key: DecisionKey
    evidence_level: int
    status: DecisionStatus
    coverage: CandidateCoverage
    target_candidate: str | None = None
    predicted_candidate: str | None = None
    failure_reason: str | None = None
    onset_start_s: float | None = None
    onset_end_s: float | None = None
    evidence_available_s: float | None = None

    def __post_init__(self) -> None:
        if self.evidence_level < 1:
            raise ValueError("evidence_level must be positive.")
        object.__setattr__(self, "status", DecisionStatus(self.status))
        if self.status in {DecisionStatus.CORRECT, DecisionStatus.INCORRECT}:
            if not self.coverage.complete:
                raise ValueError(f"{self.status.value} requires complete candidate coverage.")
            if self.target_candidate is None or self.predicted_candidate is None:
                raise ValueError(
                    f"{self.status.value} requires target_candidate and predicted_candidate."
                )
            if self.status == DecisionStatus.CORRECT and (
                self.predicted_candidate != self.target_candidate
            ):
                raise ValueError("correct outcome prediction must equal its target.")
            if self.status == DecisionStatus.INCORRECT and (
                self.predicted_candidate == self.target_candidate
            ):
                raise ValueError("incorrect outcome prediction must differ from its target.")
        if self.status in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}:
            if not self.failure_reason:
                raise ValueError(f"{self.status.value} requires failure_reason.")
        if self.status == DecisionStatus.TIE:
            if not self.coverage.complete:
                raise ValueError("tie outcomes require complete candidate coverage.")
            if self.target_candidate is None:
                raise ValueError("tie outcomes require target_candidate.")
            if self.predicted_candidate is not None:
                raise ValueError("tie outcomes cannot have a unique predicted_candidate.")
        if self.status in {DecisionStatus.INCOMPLETE, DecisionStatus.FIT_FAILURE}:
            if self.predicted_candidate is not None:
                raise ValueError(f"{self.status.value} cannot have predicted_candidate.")
        timings = (
            self.onset_start_s,
            self.onset_end_s,
            self.evidence_available_s,
        )
        if any(value is not None and not math.isfinite(value) for value in timings):
            raise ValueError("decision timings must be finite when present.")
        if (
            self.onset_start_s is not None
            and self.onset_end_s is not None
            and self.onset_end_s < self.onset_start_s
        ):
            raise ValueError("onset_end_s precedes onset_start_s.")
        if (
            self.onset_start_s is not None
            and self.evidence_available_s is not None
            and self.evidence_available_s < self.onset_start_s
        ):
            raise ValueError("evidence_available_s precedes onset_start_s.")

    def to_record(self) -> dict[str, object]:
        return {
            "subject": self.key.subject_id,
            "decision_id": self.key.decision_id,
            "evidence_level": self.evidence_level,
            "status": self.status.value,
            "target_candidate": self.target_candidate,
            "predicted_candidate": self.predicted_candidate,
            "failure_reason": self.failure_reason,
            "candidate_coverage": self.coverage.to_record(),
            "timing": {
                "onset_start_s": self.onset_start_s,
                "onset_end_s": self.onset_end_s,
                "evidence_available_s": self.evidence_available_s,
                "decision_latency_s": (
                    self.evidence_available_s - self.onset_start_s
                    if self.onset_start_s is not None
                    and self.evidence_available_s is not None
                    else None
                ),
            },
        }


@dataclass(frozen=True)
class EvidenceOutcomeCounts:
    evidence_level: int
    correct: int
    incorrect: int
    tie: int
    abstain: int
    incomplete: int
    fit_failure: int

    @property
    def accounted(self) -> int:
        return (
            self.correct
            + self.incorrect
            + self.tie
            + self.abstain
            + self.incomplete
            + self.fit_failure
        )

    def to_record(self, *, requested: int) -> dict[str, object]:
        if self.accounted != requested:
            raise ValueError(
                f"evidence level {self.evidence_level}: accounted={self.accounted} "
                f"does not equal requested={requested}."
            )
        return {
            "requested": requested,
            "accounted": self.accounted,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "tie": self.tie,
            "abstain": self.abstain,
            "incomplete": self.incomplete,
            "fit_failure": self.fit_failure,
            "operational_hit_rate": self.correct / requested if requested else 0.0,
        }


@dataclass(frozen=True)
class DecisionOutcomeAccounting:
    requested_decisions: int
    counts_by_evidence: tuple[EvidenceOutcomeCounts, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "requested": self.requested_decisions,
            "by_evidence_level": {
                str(counts.evidence_level): counts.to_record(
                    requested=self.requested_decisions
                )
                for counts in self.counts_by_evidence
            },
        }


def build_decision_outcome_accounting(
    outcomes: Sequence[DecisionOutcome],
    *,
    requested_decisions: Sequence[DecisionKey],
    evidence_levels: Sequence[int],
) -> DecisionOutcomeAccounting:
    """Validate the requested cross-product and count primary statuses."""

    requested = tuple(requested_decisions)
    levels = tuple(int(level) for level in evidence_levels)
    if len(set(requested)) != len(requested):
        raise ValueError("requested_decisions contains duplicates.")
    if not levels or any(level < 1 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("evidence_levels must be unique positive integers.")
    expected_keys = {(key, level) for key in requested for level in levels}
    observed_keys = [(outcome.key, outcome.evidence_level) for outcome in outcomes]
    duplicates = [key for key, count in Counter(observed_keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate decision outcomes: {duplicates}")
    observed_key_set = set(observed_keys)
    missing = sorted(expected_keys - observed_key_set)
    unexpected = sorted(observed_key_set - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"decision outcome accounting mismatch: missing={missing}, unexpected={unexpected}"
        )

    counts_by_evidence: list[EvidenceOutcomeCounts] = []
    for level in levels:
        status_counts = Counter(
            outcome.status for outcome in outcomes if outcome.evidence_level == level
        )
        counts = EvidenceOutcomeCounts(
            evidence_level=level,
            correct=status_counts[DecisionStatus.CORRECT],
            incorrect=status_counts[DecisionStatus.INCORRECT],
            tie=status_counts[DecisionStatus.TIE],
            abstain=status_counts[DecisionStatus.ABSTAIN],
            incomplete=status_counts[DecisionStatus.INCOMPLETE],
            fit_failure=status_counts[DecisionStatus.FIT_FAILURE],
        )
        if counts.accounted != len(requested):
            raise AssertionError("validated outcome keys produced invalid status accounting.")
        counts_by_evidence.append(counts)
    return DecisionOutcomeAccounting(
        requested_decisions=len(requested),
        counts_by_evidence=tuple(counts_by_evidence),
    )
