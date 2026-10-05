"""Requested-participant risk remains paired through ties, missing fits and seeds."""

import json
from dataclasses import replace

import numpy as np
import pytest

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import DecisionPlanContract, DecisionPlanEntry, ParticipantSelectionFailure
from transfer.outcomes import CandidateCoverage, DecisionKey, DecisionStatus


def _plan(entries=(("p1", "d1", "a"), ("p2", "d2", "a")), failures=()):
    return DecisionPlanContract("0" * 64, "1" * 64,
        tuple(sorted({row[0] for row in entries} | {row[0] for row in failures})),
        tuple(DecisionPlanEntry(*row) for row in entries),
        tuple(ParticipantSelectionFailure(participant, "decision_selection", reason)
              for participant, reason in failures))


def _evidence(participant="p1", decision="d1", probability=(.8, .2), **changes):
    identity = CandidateExpertIdentity("test", "a" * 64, "b" * 64, None, "c" * 64,
                                      "test_llr", False)
    arguments = dict(candidate_vocabulary=("a", "b"), log_probability=np.log(probability),
        participant_key=participant, decision_id=decision, physical_support_digest="d" * 64,
        expert_identities=(identity,), evidence_level=2,
        expected_event_counts=(2, 2), observed_event_counts=(2, 2))
    arguments.update(changes)
    return CandidateEvidence(**arguments)


def _evaluate(by_seed, plan=None, seeds=(1,)):
    return evaluate_candidate_evidence(by_seed, decision_plan=plan or _plan(), seeds=seeds,
                                      candidate_vocabulary=("a", "b"), evidence_level=2)


def test_requested_participant_macro_differs_from_decision_micro():
    entries = tuple(("p1", f"d{i}", "a") for i in range(10)) + (("p2", "x", "a"),)
    evidence = [_evidence("p1", f"d{i}", (.8, .2) if i < 9 else (.2, .8)) for i in range(10)]
    evidence.append(_evidence("p2", "x", (.2, .8)))
    metrics = _evaluate({1: evidence}, _plan(entries))
    assert metrics.requested_participant_operational_hit_rate == pytest.approx(.45)
    assert metrics.decision_micro_hit_rate == pytest.approx(9 / 11)


def test_complete_probability_tie_retains_nll_but_has_no_operational_coverage():
    metrics = _evaluate({1: [_evidence(probability=(.5, .5)), _evidence("p2", "d2")]})
    assert metrics.prediction_coverage == .5
    assert metrics.probability_coverage == 1.
    assert metrics.tie_count == 1
    assert metrics.available_probability_nll == pytest.approx((-np.log(.5) - np.log(.8)) / 2)
    assert metrics.outcomes_by_seed[0][1][0].status == DecisionStatus.TIE


def test_missing_participant_and_selection_failure_remain_in_requested_macro():
    plan = _plan(failures=(("p3", "no legal independent decision"),))
    metrics = _evaluate({1: [_evidence()]}, plan)
    assert metrics.requested_participant_operational_hit_rate == pytest.approx(1 / 3)
    assert metrics.prediction_coverage == pytest.approx(1 / 3)
    assert metrics.not_available_count == 1
    assert metrics.selection_failure_count == 1
    assert {row.participant_key for row in metrics.participants} == {"p1", "p2", "p3"}


def test_seed_replicates_average_inside_participant_not_as_independent_people():
    plan = _plan(entries=(("p1", "d1", "a"),))
    rows = {1: [_evidence()], 2: [_evidence(probability=(.1, .9))], 3: []}
    metrics = _evaluate(rows, plan, seeds=(1, 2, 3))
    assert metrics.requested_participant_count == 1
    assert metrics.requested_participant_operational_hit_rate == pytest.approx(1 / 3)
    assert metrics.prediction_coverage == pytest.approx(2 / 3)


def test_log_domain_nll_distinguishes_underflow_and_true_zero():
    plan = _plan(entries=(("p1", "d1", "b"),))
    finite = _evidence(log_probability=np.array([0., -1000.]))
    zero = _evidence(log_probability=np.array([0., -np.inf]))
    assert _evaluate({1: [finite]}, plan).available_probability_nll == 1000.
    assert _evaluate({1: [zero]}, plan).available_probability_nll == np.inf


def test_brier_uses_complete_distribution_and_external_truth():
    metrics = _evaluate({1: [_evidence(), _evidence("p2", "d2", (.2, .8))]})
    assert metrics.available_probability_brier == pytest.approx((.08 + 1.28) / 2)


def test_paired_nll_support_is_probability_complete_including_ties():
    baseline = _evaluate({1: [_evidence(probability=(.5, .5)), _evidence("p2", "d2", (.6, .4))]})
    candidate = _evaluate({1: [_evidence(), _evidence("p2", "d2", (.9, .1))]})
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.paired_probability_count == 2
    assert comparison.rescued_count == 1
    assert comparison.harmed_count == 0
    assert comparison.delta_hit == .5
    assert comparison.delta_paired_nll < 0
    assert comparison.development_gate_passed


def test_selective_probability_dropout_cannot_pass_development_gate():
    baseline = _evaluate({1: [_evidence(), _evidence("p2", "d2", (.1, .9))]})
    candidate = _evaluate({1: [_evidence(probability=(.9, .1))]})
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.delta_paired_nll is None
    assert not comparison.development_gate_passed
    assert comparison.paired_probability_count == 1
    assert len(comparison.missing_probability_ledger) == 1
    assert "paired_nll_unavailable" in comparison.gate_reasons
    assert "baseline_prediction_output_lost" in comparison.gate_reasons


def test_no_probability_data_never_reports_zero_risk_or_gate_pass():
    empty = _evaluate({1: []})
    assert empty.available_probability_nll is None
    comparison = compare_candidate_probability_metrics(empty, empty)
    assert comparison.delta_paired_nll is None and not comparison.development_gate_passed


@pytest.mark.parametrize("mutation", ["cohort", "seed", "vocabulary", "physical_support"])
def test_paired_comparison_rejects_different_identity_or_requested_support(mutation):
    baseline = _evaluate({1: [_evidence(), _evidence("p2", "d2")]})
    if mutation == "cohort":
        other = _evaluate({1: [_evidence()]}, _plan(entries=(("p1", "d1", "a"),)))
    elif mutation == "seed":
        other = _evaluate({2: [_evidence(), _evidence("p2", "d2")]}, seeds=(2,))
    elif mutation == "vocabulary":
        evidence = [replace(_evidence(), candidate_vocabulary=("b", "a")),
                    replace(_evidence("p2", "d2"), candidate_vocabulary=("b", "a"))]
        other = evaluate_candidate_evidence({1: evidence}, decision_plan=_plan(), seeds=(1,),
                                             candidate_vocabulary=("b", "a"), evidence_level=2)
    else:
        other = _evaluate({1: [_evidence(physical_support_digest="e" * 64), _evidence("p2", "d2")]})
    with pytest.raises(ValueError, match="identity|support|plan|seed|vocabulary"):
        compare_candidate_probability_metrics(baseline, other)


def test_expert_input_transform_changes_are_allowed_when_physical_support_matches():
    baseline = _evaluate({1: [_evidence(), _evidence("p2", "d2")]})
    rows = [_evidence(probability=(.9, .1)), _evidence("p2", "d2", (.9, .1))]
    rows = [replace(row, expert_identities=(replace(row.expert_identities[0], numerical_input_digest="f" * 64),))
            for row in rows]
    assert compare_candidate_probability_metrics(baseline, _evaluate({1: rows})).development_gate_passed


@pytest.mark.parametrize("rows", [[_evidence(), _evidence()], [_evidence("other", "d1")]])
def test_duplicate_and_unrequested_output_rows_are_contract_errors(rows):
    with pytest.raises(ValueError, match="duplicate|requested"):
        _evaluate({1: rows})


def test_expected_fit_failure_retains_requested_schedule_and_separate_failure_status():
    coverage = {DecisionKey("p1", "d1"): CandidateCoverage.from_mappings({"a": 2, "b": 2}, {}),
                DecisionKey("p2", "d2"): CandidateCoverage.from_mappings({"a": 2, "b": 2}, {})}
    metrics = evaluate_candidate_evidence({1: [_evidence()]}, decision_plan=_plan(), seeds=(1,),
        candidate_vocabulary=("a", "b"), evidence_level=2,
        participant_failures_by_seed={1: {"p2": "authorized fit could not converge"}},
        requested_coverage=coverage)
    failed = metrics.outcomes_by_seed[0][1][1]
    assert failed.status == DecisionStatus.FIT_FAILURE
    assert failed.coverage.expected_events == 4
    assert failed.coverage.observed_events == 0
    assert metrics.participants[1].fit_failure_count == 1
    assert metrics.not_available_count == 0
    assert metrics.requested_participant_operational_hit_rate == .5


def test_probability_gate_requires_paired_support_for_each_requested_seed():
    plan = _plan(entries=(("p1", "d1", "a"),))
    baseline = _evaluate({1: [_evidence()], 2: []}, plan, seeds=(1, 2))
    candidate = _evaluate({1: [_evidence(probability=(.9, .1))], 2: []}, plan, seeds=(1, 2))
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.paired_probability_count == 1
    assert comparison.delta_paired_nll is None
    assert comparison.gate_reasons == ("paired_nll_unavailable",)


def test_two_infinite_paired_risks_have_no_defined_difference_or_gate_pass():
    plan = _plan(entries=(("p1", "d1", "b"),))
    baseline = _evaluate({1: [_evidence(log_probability=np.array([0., -np.inf]))]}, plan)
    comparison = compare_candidate_probability_metrics(baseline, baseline)
    assert comparison.paired_baseline_nll == np.inf
    assert comparison.delta_paired_nll is None and not comparison.development_gate_passed
    json.dumps(comparison.record(), allow_nan=False)
    json.dumps(baseline.record(), allow_nan=False)


def test_recovering_true_zero_probability_can_pass_when_hit_and_coverage_improve():
    plan = _plan(entries=(("p1", "d1", "b"),))
    baseline = _evaluate({1: [_evidence(log_probability=np.array([0., -np.inf]))]}, plan)
    candidate = _evaluate({1: [_evidence(probability=(.1, .9))]}, plan)
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.delta_paired_nll == -np.inf
    assert comparison.development_gate_passed


def test_rescue_and_harm_are_both_reported_when_net_hit_is_zero():
    baseline = _evaluate({1: [_evidence(probability=(.6, .4)), _evidence("p2", "d2", (.4, .6))]})
    candidate = _evaluate({1: [_evidence(probability=(.01, .99)), _evidence("p2", "d2", (.6, .4))]})
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.delta_hit == 0.
    assert comparison.rescued_count == comparison.harmed_count == 1
    assert comparison.delta_paired_nll > 0.
    assert comparison.participant_delta_hit == (("p1", -1.), ("p2", 1.))
    assert comparison.gate_reasons == ("nll_increased",)


def test_all_selection_failed_participants_remain_a_zero_hit_requested_roster():
    plan = _plan(entries=(), failures=(("p1", "no legal decision"), ("p2", "no legal decision")))
    metrics = _evaluate({1: []}, plan)
    assert metrics.requested_participant_count == 2
    assert metrics.requested_participant_operational_hit_rate == 0.
    assert metrics.available_probability_nll is None
    assert metrics.selection_failure_count == 2
    assert metrics.accounting_by_seed[0][1].requested_decisions == 0
    assert not compare_candidate_probability_metrics(metrics, metrics).development_gate_passed


def test_requested_physical_coverage_mismatch_is_not_swallowed_as_fit_failure():
    coverage = {DecisionKey("p1", "d1"): CandidateCoverage.from_mappings({"a": 2, "b": 3}, {}),
                DecisionKey("p2", "d2"): CandidateCoverage.from_mappings({"a": 2, "b": 2}, {})}
    with pytest.raises(ValueError, match="physical schedule"):
        evaluate_candidate_evidence({1: [_evidence()]}, decision_plan=_plan(), seeds=(1,),
            candidate_vocabulary=("a", "b"), evidence_level=2, requested_coverage=coverage)


@pytest.mark.parametrize("risk", ["available_probability_nll", "available_probability_brier"])
def test_requested_macro_probability_risk_is_unavailable_when_any_participant_has_no_probability(risk):
    metrics = _evaluate({1: [_evidence()]})
    assert metrics.probability_coverage == .5
    assert metrics.participants[0].available_probability_seed_count == 1
    assert metrics.participants[1].available_probability_seed_count == 0
    assert getattr(metrics, risk) is None
    assert metrics.record()[risk] is None


@pytest.mark.parametrize("risk", ["available_probability_nll", "available_probability_brier"])
def test_requested_macro_probability_risk_requires_all_requested_seeds_inside_each_participant(risk):
    plan = _plan(entries=(("p1", "d1", "a"),))
    metrics = _evaluate({1: [_evidence()], 2: []}, plan, seeds=(1, 2))
    assert metrics.probability_coverage == .5
    assert metrics.participants[0].available_probability_seed_count == 1
    assert getattr(metrics.participants[0], risk) is None
    assert getattr(metrics, risk) is None


def test_partial_decision_probability_coverage_keeps_conditional_risk_when_every_participant_seed_is_present():
    plan = _plan(entries=(("p1", "d1", "a"), ("p1", "d2", "a"),
                          ("p2", "d3", "a"), ("p2", "d4", "a")))
    metrics = _evaluate({1: [_evidence("p1", "d1", (.8, .2)), _evidence("p2", "d3", (.6, .4))],
                         2: [_evidence("p1", "d2", (.4, .6)), _evidence("p2", "d4", (.2, .8))]},
                        plan, seeds=(1, 2))
    assert metrics.probability_coverage == .5
    assert metrics.available_probability_nll == pytest.approx(-np.log([.8, .4, .6, .2]).mean())
    assert metrics.available_probability_brier == pytest.approx(np.mean([.08, .72, .32, 1.28]))
    assert all(row.available_probability_seed_count == 2 for row in metrics.participants)


@pytest.mark.parametrize("missing_unit", ["participant", "seed"])
def test_paired_macro_risks_do_not_silently_drop_an_entire_requested_probability_unit(missing_unit):
    if missing_unit == "participant":
        baseline = _evaluate({1: [_evidence(), _evidence("p2", "d2")]})
        candidate = _evaluate({1: [_evidence(probability=(.9, .1))]})
    else:
        plan = _plan(entries=(("p1", "d1", "a"),))
        baseline = _evaluate({1: [_evidence()], 2: [_evidence()]}, plan, seeds=(1, 2))
        candidate = _evaluate({1: [_evidence(probability=(.9, .1))], 2: []}, plan, seeds=(1, 2))
    comparison = compare_candidate_probability_metrics(baseline, candidate)
    assert comparison.paired_probability_count == 1
    assert len(comparison.missing_probability_ledger) == 1
    assert comparison.paired_baseline_nll is None
    assert comparison.paired_candidate_nll is None
    assert comparison.delta_paired_nll is None
    assert not comparison.development_gate_passed
