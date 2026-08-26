from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from baselines.repetition_metrics import (
    summarize_repetition_efficiency,
    wolpaw_bits_per_selection,
)


@dataclass
class _Metric:
    evidence_budget: int | str
    aggregation: str
    hit_rate: float
    coverage: float
    n_covered: int
    budget_semantics: str = "exact"


def test_wolpaw_information_has_chance_and_perfect_endpoints() -> None:
    assert wolpaw_bits_per_selection(1.0 / 9.0, 9) == 0.0
    assert wolpaw_bits_per_selection(1.0, 9) == pytest.approx(math.log2(9))


def test_repetition_efficiency_reports_itr_and_fixed_error_budget() -> None:
    metrics = {
        "exact_llr@3": _Metric(3, "llr", 0.80, 1.0, 100),
        "exact_llr@5": _Metric(5, "llr", 0.96, 0.95, 95),
        "exact_llr@10": _Metric(10, "llr", 0.99, 0.80, 80),
        "exact_sum@5": _Metric(5, "sum", 1.0, 1.0, 100),
    }
    summary = summarize_repetition_efficiency(
        metrics,
        n_choices=9,
        target_error_rate=0.05,
        minimum_coverage=0.90,
    )

    assert summary.repetitions_to_target_error == 5
    assert [point.repetitions for point in summary.points] == [3, 5, 10]
    assert summary.budget_semantics == "exact"
    assert math.isnan(summary.points[1].itr_bits_per_minute)


def test_conditional_chain_curve_takes_priority_over_platt_llr() -> None:
    metrics = {
        "exact_llr@3": _Metric(3, "llr", 1.0, 1.0, 10),
        "exact_chain_llr@3": _Metric(3, "chain_llr", 0.8, 1.0, 10),
    }
    summary = summarize_repetition_efficiency(metrics, n_choices=9)

    assert summary.aggregation == "chain_llr"
    assert summary.points[0].accuracy == 0.8


def test_efficiency_can_be_bound_to_prefix_chain_primary_family() -> None:
    metrics = {
        "exact_llr@3": _Metric(3, "llr", 1.0, 1.0, 10),
        "prefix_minK_chain_llr@1": _Metric(
            1, "chain_llr", 0.7, 1.0, 10, budget_semantics="prefix_minK"
        ),
        "prefix_minK_chain_llr@3": _Metric(
            3, "chain_llr", 0.9, 0.9, 9, budget_semantics="prefix_minK"
        ),
    }

    summary = summarize_repetition_efficiency(
        metrics,
        n_choices=9,
        aggregation="chain_llr",
        budget_semantics="prefix_minK",
    )

    assert summary.aggregation == "chain_llr"
    assert summary.budget_semantics == "prefix_minK"
    assert [point.repetitions for point in summary.points] == [1, 3]


def test_non_repetition_primary_semantics_are_labeled_without_false_k_curve() -> None:
    metrics = {
        "flash_llr@9": _Metric(9, "llr", 0.8, 1.0, 10, budget_semantics="flash")
    }

    summary = summarize_repetition_efficiency(
        metrics,
        n_choices=9,
        aggregation="llr",
        budget_semantics="flash",
    )

    assert summary.aggregation == "llr"
    assert summary.budget_semantics == "flash"
    assert summary.points == []
    assert summary.repetitions_to_target_error is None


def test_itr_rejects_k_times_duration_shortcut() -> None:
    metrics = {"exact_llr@3": _Metric(3, "llr", 0.8, 1.0, 10)}
    with pytest.raises(ValueError, match="ragged/rejected"):
        summarize_repetition_efficiency(metrics, n_choices=9, repetition_duration_s=2.0)
