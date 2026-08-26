"""Selection-efficiency metrics for fixed-budget P300 decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


class _DecisionMetricLike(Protocol):
    evidence_budget: int | str
    aggregation: str
    hit_rate: float
    coverage: float
    n_covered: int
    budget_semantics: str


@dataclass(frozen=True)
class RepetitionEfficiencyPoint:
    repetitions: int
    accuracy: float
    error_rate: float
    coverage: float
    n_covered: int
    bits_per_selection: float
    itr_bits_per_minute: float


@dataclass(frozen=True)
class RepetitionEfficiencySummary:
    target_error_rate: float
    minimum_coverage: float
    repetitions_to_target_error: int | None
    n_choices: int
    repetition_duration_s: float | None
    aggregation: str
    budget_semantics: str
    points: list[RepetitionEfficiencyPoint] = field(default_factory=list)


def wolpaw_bits_per_selection(accuracy: float, n_choices: int) -> float:
    """Return the standard Wolpaw information per selection in bits."""

    if n_choices < 2:
        raise ValueError("n_choices must be at least two.")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0,1].")
    chance = 1.0 / n_choices
    if accuracy <= chance:
        return 0.0
    if accuracy >= 1.0:
        return math.log2(n_choices)
    incorrect = 1.0 - accuracy
    value = (
        math.log2(n_choices)
        + accuracy * math.log2(accuracy)
        + incorrect * math.log2(incorrect / (n_choices - 1))
    )
    return max(0.0, value)


def summarize_repetition_efficiency(
    decision_metrics: Mapping[str, _DecisionMetricLike],
    *,
    n_choices: int,
    target_error_rate: float = 0.05,
    minimum_coverage: float = 0.90,
    repetition_duration_s: float | None = None,
    aggregation: str | None = None,
    budget_semantics: str = "exact",
) -> RepetitionEfficiencySummary:
    """Summarize one explicitly identified accuracy/efficiency metric family.

    ``repetitions_to_target_error`` is descriptive: it is the smallest tested
    fixed budget whose observed error and coverage pass the supplied thresholds.
    It is not an adaptive stopping rule and must not be tuned on a held-out test
    fold. ``repetition_duration_s`` is the measured time for one complete pass
    across all candidates; ITR is NaN when that acquisition timing is unknown.
    """

    if not 0.0 <= target_error_rate < 1.0:
        raise ValueError("target_error_rate must be in [0,1).")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0,1].")
    if repetition_duration_s is not None:
        raise ValueError(
            "ITR cannot be derived as K * repetition_duration_s for ragged/rejected events; "
            "use online-causal time@T metrics with event-level availability timestamps."
        )

    fixed_k_semantics = budget_semantics in {"exact", "prefix_minK"}
    available_aggregations = {
        metric.aggregation
        for metric in decision_metrics.values()
        if fixed_k_semantics
        and isinstance(metric.evidence_budget, int)
        and metric.budget_semantics == budget_semantics
    }
    if aggregation is None:
        aggregation = "chain_llr" if "chain_llr" in available_aggregations else "llr"
    numeric = []
    for metric in decision_metrics.values():
        if (
            metric.aggregation != aggregation
            or not fixed_k_semantics
            or not isinstance(metric.evidence_budget, int)
            or metric.budget_semantics != budget_semantics
        ):
            continue
        numeric.append(metric)
    numeric.sort(key=lambda metric: int(metric.evidence_budget))

    points: list[RepetitionEfficiencyPoint] = []
    repetitions_to_target: int | None = None
    for metric in numeric:
        repetitions = int(metric.evidence_budget)
        accuracy = float(metric.hit_rate)
        if not math.isfinite(accuracy):
            bits = float("nan")
            error_rate = float("nan")
        else:
            bits = wolpaw_bits_per_selection(accuracy, n_choices)
            error_rate = 1.0 - accuracy
        itr = float("nan")
        point = RepetitionEfficiencyPoint(
            repetitions=repetitions,
            accuracy=accuracy,
            error_rate=error_rate,
            coverage=float(metric.coverage),
            n_covered=int(metric.n_covered),
            bits_per_selection=bits,
            itr_bits_per_minute=itr,
        )
        points.append(point)
        if (
            repetitions_to_target is None
            and math.isfinite(error_rate)
            and error_rate <= target_error_rate
            and point.coverage >= minimum_coverage
        ):
            repetitions_to_target = repetitions

    return RepetitionEfficiencySummary(
        target_error_rate=float(target_error_rate),
        minimum_coverage=float(minimum_coverage),
        repetitions_to_target_error=repetitions_to_target,
        n_choices=int(n_choices),
        repetition_duration_s=(
            float(repetition_duration_s) if repetition_duration_s is not None else None
        ),
        aggregation=aggregation,
        budget_semantics=budget_semantics,
        points=points,
    )
