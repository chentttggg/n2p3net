from __future__ import annotations

from experiments.harness_s0.checks import (
    check_count_prior_cancellation,
    check_latency_gauge,
    check_monotone_rho_chain_rank,
    check_redundant_likelihood_fusion,
    check_reliability_prevalence_shift,
    check_soft_label_semantics,
    check_stopping_replay,
    run_all_checks,
)


def test_s0_soft_label_semantics() -> None:
    report = check_soft_label_semantics()
    assert report["passed"], report


def test_s0_count_prior_cancellation() -> None:
    report = check_count_prior_cancellation()
    assert report["passed"], report


def test_s0_monotone_rho_chain_rank() -> None:
    report = check_monotone_rho_chain_rank()
    assert report["passed"], report


def test_s0_redundant_likelihood_fusion() -> None:
    report = check_redundant_likelihood_fusion()
    assert report["passed"], report


def test_s0_latency_gauge() -> None:
    report = check_latency_gauge()
    assert report["passed"], report


def test_s0_reliability_prevalence_shift() -> None:
    report = check_reliability_prevalence_shift()
    assert report["passed"], report


def test_s0_stopping_replay() -> None:
    report = check_stopping_replay()
    assert report["passed"], report


def test_s0_all_checks_pass() -> None:
    report = run_all_checks()
    failed = {name: value for name, value in report.items() if not value["passed"]}
    assert not failed, failed
