"""Independent finite examples for confidence diagnostics on requested outcomes."""

import json
import math

import numpy as np
import pytest

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research import candidate_metrics
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import DecisionPlanContract, DecisionPlanEntry, ParticipantSelectionFailure


def evidence(participant="p", decision="d", probability=(.8, .2), **changes):
    count = len(probability)
    identity = CandidateExpertIdentity("toy", "a" * 64, "b" * 64, None, "c" * 64, "toy", False)
    fields = dict(participant_key=participant, decision_id=decision,
        candidate_vocabulary=tuple("abc"[:count]), log_probability=np.log(probability),
        physical_support_digest="d" * 64, expert_identities=(identity,), evidence_level=1,
        expected_event_counts=(1,) * count, observed_event_counts=(1,) * count)
    fields.update(changes)
    return CandidateEvidence(**fields)


def evaluate(rows, entries=(("p", "d", "a"),), failures=(), seeds=(1,), count=2):
    roster = tuple(sorted({row[0] for row in entries} | {row[0] for row in failures}))
    plan = DecisionPlanContract("0" * 64, "1" * 64, roster,
        tuple(DecisionPlanEntry(*row) for row in entries),
        tuple(ParticipantSelectionFailure(participant, "decision_selection", reason)
              for participant, reason in failures))
    return evaluate_candidate_evidence(rows, decision_plan=plan, seeds=seeds,
        candidate_vocabulary=tuple("abc"[:count]), evidence_level=1)


def diagnostic(metrics, bins=None):
    if bins is None:
        return metrics.record()["confidence_diagnostics"]
    return candidate_metrics.candidate_confidence_diagnostics(metrics, bin_edges=bins)


def test_tied_complete_probability_calibrates_without_operational_prediction():
    metrics = evaluate({1: [evidence(probability=(.5, .5))]})
    report = diagnostic(metrics)
    assert report["probability_count"] == 1 and report["tie_count"] == 1
    assert report["prediction_coverage"] == 0 and report["probability_coverage"] == 1
    assert report["binned_top_label_ece"] == 0
    assert report["mean_max_confidence"] == .5
    assert report["wrong_unique_winner_count"] == 0
    assert report["conditional_wrong_unique_winner_mean_confidence"] is None
    assert report["nll_minus_same_candidate_count_uniform"] == 0
    occupied = [row for row in report["calibration_table"] if row["probability_count"]]
    assert occupied[0]["expected_top_label_correctness"] == .5


def test_tie_outside_truth_has_zero_expected_correctness_but_is_not_a_wrong_winner():
    row = evidence(probability=(.45, .45, .1))
    report = diagnostic(evaluate({1: [row]}, entries=(("p", "d", "c"),), count=3))
    assert report["binned_top_label_ece"] == pytest.approx(.45)
    assert report["wrong_unique_winner_count"] == 0
    assert row.top_candidate_indices == (0, 1)
    assert row.predicted is None


def test_owner_tie_tolerance_and_roundtrip_are_not_redecided_in_exp_domain():
    row = evidence(probability=(.5000000000001, .4999999999999))
    assert row.tie and row.top_candidate_indices == (0, 1)
    assert "top_candidate_indices" not in row.record()
    assert CandidateEvidence.from_record(row.record()).top_candidate_indices == (0, 1)
    report = diagnostic(evaluate({1: [row]}))
    assert report["binned_top_label_ece"] == pytest.approx(0., abs=1e-16)


def test_same_hit_can_have_very_different_wrong_confidence_without_a_new_gate():
    before = evaluate({1: [evidence(probability=(.4, .6))]})
    after = evaluate({1: [evidence(probability=(.01, .99))]})
    first, second = diagnostic(before), diagnostic(after)
    assert before.requested_participant_operational_hit_rate == after.requested_participant_operational_hit_rate == 0
    assert first["conditional_wrong_unique_winner_mean_confidence"] == pytest.approx(.6)
    assert second["conditional_wrong_unique_winner_mean_confidence"] == pytest.approx(.99)
    assert second["nll_minus_same_candidate_count_uniform"] > first["nll_minus_same_candidate_count_uniform"]
    assert second["gate_action"].startswith("none")
    assert compare_candidate_probability_metrics(before, after).gate_reasons == ("nll_increased",)


def test_log_domain_underflow_and_true_zero_remain_distinct_in_risk_contrast():
    finite = evaluate({1: [evidence(log_probability=np.array([0., -1000.]))]}, entries=(("p", "d", "b"),))
    zero = evaluate({1: [evidence(log_probability=np.array([0., -np.inf]))]}, entries=(("p", "d", "b"),))
    first, second = diagnostic(finite), diagnostic(zero)
    assert first["available_probability_nll"] == 1000
    assert first["nll_minus_same_candidate_count_uniform"] == 1000 - math.log(2)
    assert second["available_probability_nll"] == second["nll_minus_same_candidate_count_uniform"] == "positive_infinity"
    assert first["observed_wrong_unique_winner_max_confidence"] == second["observed_wrong_unique_winner_max_confidence"] == 1
    json.dumps(second, allow_nan=False)


@pytest.mark.parametrize("kind", ["fit_failure", "not_available", "incomplete", "abstain"])
def test_unavailable_evidence_retains_requested_denominator_and_is_never_uniform_filled(kind):
    failed = evidence("q", "x", log_probability=None, failure_kind=kind, failure_reason="explicit failure")
    metrics = evaluate({1: [evidence(), failed]}, entries=(("p", "d", "a"), ("q", "x", "a")))
    report = diagnostic(metrics)
    assert report["requested_cell_count"] == 2 and report["missing_probability_count"] == 1
    assert report["probability_count"] == 1 and report["probability_coverage"] == .5
    assert report["mean_max_confidence"] is None and report["binned_top_label_ece"] is None
    assert report["nll_minus_same_candidate_count_uniform"] is None
    assert report["participants"][1]["seed_diagnostics"][0]["probability_count"] == 0


def test_selection_failure_keeps_person_even_without_a_requested_decision():
    report = diagnostic(evaluate({1: [evidence()]}, failures=(("q", "no legal decision"),)))
    assert report["requested_participant_count"] == 2
    assert report["participants"][1]["selection_failure_reason"] == "no legal decision"
    assert report["participants"][1]["requested_cell_count"] == 0
    assert report["binned_top_label_ece"] is None


def test_unequal_decision_counts_and_seeds_average_inside_the_requested_person():
    entries = (("p", "a", "a"), ("p", "b", "a"), ("p", "c", "a"), ("q", "x", "a"))
    rows = {1: [evidence("p", "a", (.8, .2)), evidence("q", "x", (.1, .9))],
            2: [evidence("p", name, (.4, .6)) for name in "abc"] + [evidence("q", "x", (.7, .3))]}
    report = diagnostic(evaluate(rows, entries=entries, seeds=(1, 2)), bins=(0., 1.))
    assert report["mean_max_confidence"] == pytest.approx(.75)
    assert report["wrong_unique_winner_fraction"] == .5
    assert report["conditional_wrong_unique_winner_mean_confidence"] == pytest.approx(.75)
    assert report["binned_top_label_ece"] == pytest.approx(.25)
    assert report["probability_count"] == 6 and report["requested_cell_count"] == 8
    assert report["missing_probability_count"] == 2
    assert report["fixed_roster_probability_units_complete"]


def test_one_empty_requested_seed_makes_macro_unavailable_but_keeps_observed_max():
    report = diagnostic(evaluate({1: [evidence()], 2: []}, seeds=(1, 2)))
    assert report["mean_max_confidence"] is None and report["binned_top_label_ece"] is None
    assert report["observed_max_confidence"] == pytest.approx(.8)
    assert report["requested_cell_count"] == 2 and report["missing_probability_count"] == 1
    assert not report["fixed_roster_probability_units_complete"]


def test_uniform_baseline_uses_actual_candidate_count_and_never_crosses_tasks():
    binary = diagnostic(evaluate({1: [evidence(probability=(.5, .5))]}))
    ternary = diagnostic(evaluate({1: [evidence(probability=(1 / 3,) * 3)]}, count=3))
    assert binary["uniform_nll"] == math.log(2) and ternary["uniform_nll"] == math.log(3)
    assert binary["candidate_count"] == 2 and ternary["candidate_count"] == 3
    assert ternary["nll_minus_same_candidate_count_uniform"] == pytest.approx(0)


def test_finite_bin_ece_is_bin_dependent_and_does_not_certify_calibration():
    rows = [evidence("p", "d", (.6, .4)), evidence("p", "x", (.2, .8))]
    metrics = evaluate({1: rows}, entries=(("p", "d", "a"), ("p", "x", "a")))
    coarse, fine = diagnostic(metrics, bins=(0., 1.)), diagnostic(metrics, bins=(0., .7, 1.))
    assert coarse["binned_top_label_ece"] == pytest.approx(.2)
    assert fine["binned_top_label_ece"] == pytest.approx(.6)
    assert "not a calibration certificate" in fine["ece_scope"]
    assert "uncertainty_interval" not in fine


@pytest.mark.parametrize("bins", [(0., 0., 1.), (.1, 1.), (0., .9), (0., float("nan"), 1.)])
def test_invalid_bins_are_explicit_report_contract_errors(bins):
    with pytest.raises(ValueError, match="partition"):
        diagnostic(evaluate({1: [evidence()]}), bins=bins)


def test_actual_final_report_consumer_exports_diagnostics_without_touching_gate(tmp_path):
    from experiments.run_candidate_ensemble_study import _finalize_candidate_report
    from transfer.ensemble import EnsembleProgram

    row = evidence()
    plan = {"calibration_strategy": "fixed", "seeds": [1],
        "mixtures": [EnsembleProgram.equal_weights("toy", ("toy",)).record()],
        "comparison_role": "software_correctness", "ar_panel": None,
        "records": [], "baseline_mixture": "toy", "evidence_scope": "software_correctness"}
    participant = {"participant_key": "p", "requested": [{"participant_key": "p", "decision_id": "d", "target_candidate": "a"}],
        "coverage": [{"participant_key": "p", "decision_id": "d", "counts": {"a": 1, "b": 1}}],
        "candidate_vocabulary": ["a", "b"], "evidence_level": 1,
        "predictions": [{"mixture": "toy", "seed": 1, "evidence": row.record()}]}
    result = _finalize_candidate_report(plan, {"identity": "synthetic"}, [participant], tmp_path, persist=False)
    diagnostic_record = result["metrics"]["toy"]["confidence_diagnostics"]
    assert diagnostic_record["mean_max_confidence"] == pytest.approx(.8)
    assert diagnostic_record["gate_action"].startswith("none")
    assert result["contrasts"] == {}
    json.dumps(result, allow_nan=False)
