"""Unscored choices affect coverage, while accuracy counts actual choices."""
import pytest

from research.candidate_metrics import ScoredTargetAccuracy, scored_target_accuracy

VOCAB = tuple(map(str, range(1, 10)))


def test_unscored_decisions_do_not_become_incorrect_predictions():
    result = scored_target_accuracy(["7", None, "3", None], ["7", "9", "9", "7"], candidate_vocabulary=VOCAB)
    assert result.accuracy == .5
    assert result.coverage == .5
    assert result.record()["unscored_decisions"] == 2
    assert result.record()["incorrect_scored_decisions"] == 1


def test_no_scored_choices_has_undefined_accuracy_and_zero_coverage():
    result = scored_target_accuracy([None, None], ["7", "9"], candidate_vocabulary=VOCAB)
    assert result.accuracy is None
    assert result.coverage == 0


def test_empty_requested_roster_is_not_zero_accuracy_or_perfect_coverage():
    result = scored_target_accuracy([], [], candidate_vocabulary=VOCAB)
    assert result.accuracy is None and result.coverage is None


def test_three_seed_means_do_not_change_single_program_denominator():
    results = [scored_target_accuracy(p, ["7", "9"], candidate_vocabulary=VOCAB)
               for p in [["7", "9"], ["7", None], [None, "1"]]]
    assert [(r.correct_decisions, r.scored_decisions) for r in results] == [(2, 2), (1, 1), (0, 1)]


def test_invalid_identity_is_rejected_instead_of_hidden_from_accuracy():
    with pytest.raises(ValueError, match="prediction"):
        scored_target_accuracy(["10"], ["7"], candidate_vocabulary=VOCAB)
    with pytest.raises(ValueError, match="target"):
        scored_target_accuracy(["7"], ["10"], candidate_vocabulary=VOCAB)


@pytest.mark.parametrize("counts", [(1, 2, 1), (2, 1, 2), (-1, 0, 0), (True, 1, 1), (2., 1, 1)])
def test_invalid_decision_counts_fail(counts):
    with pytest.raises(ValueError, match="decision counts"):
        ScoredTargetAccuracy(*counts)
