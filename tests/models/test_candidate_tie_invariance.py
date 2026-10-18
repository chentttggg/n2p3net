"""Contrast-floor and representable-shift counterexamples, without fitting."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from data.candidate_task import CandidateTaskContract, MembershipDecision
from models.candidate_evidence import (
    CandidateEvidence,
    CandidateExpertIdentity,
    candidate_evidence_scores,
    decide_by_group,
    membership_decision_scores,
)
from transfer.candidate_decision import decision_outcomes_at_repetition


def exact_top(scores):
    values = [Fraction.from_float(float(value)) for value in scores]
    maximum = max(values)
    tolerance = Fraction.from_float(1e-12)
    return tuple(index for index, value in enumerate(values) if maximum-value <= tolerance)


def direct(count=2):
    vocabulary = tuple(f'q{index}' for index in range(count))
    return MembershipDecision.direct(vocabulary, vocabulary,
        event_ids=tuple(f'e{index}' for index in range(count)),
        repetition_indices=np.zeros(count, dtype=np.int64), repetitions=1)


def identity(count=2):
    return dict(candidate_vocabulary=tuple(f'q{index}' for index in range(count)), participant_key='toy:p',
        decision_id='toy:selection', physical_support_digest='a'*64,
        expert_identities=(CandidateExpertIdentity('toy', 'b'*64, 'c'*64, None, 'd'*64, 'logit', False),),
        evidence_level=1, expected_event_counts=(1,)*count, observed_event_counts=(1,)*count)


def test_original_exact_common_shift_and_membership_offset_counterexamples():
    values, codes = np.array([0., 4e-12]), np.array([0, 1])
    for shift in (0., 10., -10.):
        result = candidate_evidence_scores(values+shift, codes, codes)
        assert result.predicted == 1
        grouped = decide_by_group(values+shift, codes, ['g', 'g'], codes, center_logits=False)
        assert grouped.predicted.tolist() == [1]
        score = membership_decision_scores(values, direct(), logit_offset=-shift)
        assert score.predicted == 'q1' and score.reason is None


@pytest.mark.parametrize('count', [9, 32, 36])
def test_dyadic_contrasts_preserved_by_shift_and_candidate_permutation(count):
    codes = np.arange(count)
    for gap, winner in ((2.**-36, 1), (2.**-41, None), (0., None)):
        values = np.r_[0., gap, np.repeat(-.25, count-2)]
        contrasts = values.max()-values
        for shift in (-1024., -10., 0., 10., 1024.):
            moved = values+shift
            np.testing.assert_array_equal(moved.max()-moved, contrasts)
            expected = exact_top(moved)
            assert (len(expected) == 1) == (winner is not None)
            assert candidate_evidence_scores(moved, codes, codes).predicted == winner
            permutation = np.arange(count)[::-1]
            assert candidate_evidence_scores(moved[permutation], codes[permutation], codes[permutation]).predicted == winner


def test_irrelevant_large_loser_does_not_change_the_top_pair_resolution():
    for loser in (-1., -1e100):
        values = np.array([0., 4e-12, loser])
        assert exact_top(values) == (1,)
        assert candidate_evidence_scores(values, [0, 1, 2], [0, 1, 2]).predicted == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_membership_uses_actual_representable_scores_at_ulp_boundaries(dtype):
    observed = []
    for gap in (float(np.nextafter(1e-12, np.inf)), 4e-12, 2.**-36, 1.):
        for offset in (0., -10., -1024., -(2.**54)):
            score = membership_decision_scores(torch.tensor([0., gap], dtype=dtype), direct(), logit_offset=offset)
            values = score.scores.detach().cpu().numpy()
            winners = exact_top(values)
            expected = f'q{winners[0]}' if len(winners) == 1 else None
            assert score.predicted == expected
            observed.append((offset, gap, len(winners)))
    assert any(offset == -(2.**54) and size == 2 for offset, _, size in observed)


def test_probability_publication_and_reload_use_the_same_contrast_floor():
    values = np.array([0., 1.3e-12])
    score = membership_decision_scores(values, direct())
    arguments = identity()
    for key in ('candidate_vocabulary', 'evidence_level', 'expected_event_counts', 'observed_event_counts'):
        arguments.pop(key)
    published = CandidateEvidence.from_membership_scores(score, direct(), **arguments)
    continuous = CandidateEvidence.from_log_evidence(values, **identity())
    for evidence in (published, continuous, CandidateEvidence.from_record(published.record())):
        assert evidence.predicted == 'q1' and evidence.status == 'available'
        assert evidence.probability_available and evidence.operational_available
        assert evidence.top_candidate_indices == (1,)
    for values in ([0., 0.], [0., 2.**-41]):
        tied = CandidateEvidence.from_log_evidence(values, **identity())
        assert tied.status == 'tie' and tied.predicted is None and tied.probability_available


def test_row_column_prefixes_keep_winner_coverage_and_timing_under_common_shifts():
    contract = CandidateTaskContract(dataset_id='toy', task_id='two_by_two',
        population={'label': 'mathematical fixture'}, evidence_scope={'stage': 'nonfit', 'product_confirmation': False},
        membership_kind='row_column', grid_shape=(2, 2), candidate_ids=(0, 1, 2, 3),
        row_candidate_ids=(0, 1), column_candidate_ids=(2, 3), target_representation='row_column_intersection')
    codes = np.tile(np.arange(4), 3)
    values = np.tile([0., 4e-12, 0., 1.], 3)
    arguments = (codes, np.ones(12, dtype=int), np.ones(12, dtype=int), ['g']*12, np.repeat(np.arange(3), 4))
    options = dict(contract=contract, max_repetitions=3, onset_times_s=np.arange(12.), evidence_available_times_s=np.arange(12.)+1.)
    for shift in (0., 10., -10.):
        outcomes = decision_outcomes_at_repetition(values+shift, *arguments, **options)
        assert [value.status.value for value in outcomes] == ['correct']*3
        assert [value.predicted_candidate for value in outcomes] == ['row:1|column:1']*3
        assert [value.evidence_available_s for value in outcomes] == [4., 8., 12.]
        assert all(value.coverage.complete for value in outcomes)


def test_missing_and_unidentifiable_evidence_keep_their_refusal_owners():
    missing = MembershipDecision.direct(('q0', 'q1'), ('q0', 'q1'), event_ids=('a', 'b'),
        repetition_indices=np.zeros(2, dtype=int), repetitions=1, evidence_indices=np.array([-1, -1]))
    score = membership_decision_scores([999., -999.], missing, prior=[.9, .1])
    assert score.predicted is None and score.reason == 'no_usable_evidence'
    assert candidate_evidence_scores([999.], [0], [0, 1]).predicted is None
