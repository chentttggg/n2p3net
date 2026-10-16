"""Counterexample and contract tests for the unified candidate-evidence core.

The previous ``models.decision`` subject-digit path and the private
``transfer.evaluation._aggregate_scores`` implementation are deleted; this
suite pins the surviving unified contract (audit F-03): one scoring core,
exact-tie abstention, fail-closed vocabularies, both missing-candidate
policies, and the centering that removes constant classifier bias.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.candidate_evidence import (
    candidate_evidence_scores,
    count_tempered_evidence_scores,
    decide_by_group,
)

VOCAB = list(range(1, 10))


def test_argmax_picks_the_strongest_candidate() -> None:
    digits = np.repeat(VOCAB, 3)
    logits = np.zeros(len(digits))
    logits[digits == 5] = 3.0
    logits[digits == 1] = 1.0
    groups = np.zeros(len(digits), dtype=int)
    decision = decide_by_group(logits, digits, groups, VOCAB)
    assert decision.predicted[0] == 5
    uncentered = decide_by_group(logits, digits, groups, VOCAB, center_logits=False)
    assert uncentered.scores_by_group["0"][5] == pytest.approx(3.0)


def test_sum_accumulation_lets_frequent_candidates_win() -> None:
    digits = np.array([1] + [2] * 10)
    logits = np.array([5.0] + [1.0] * 10)
    decision = decide_by_group(
        logits, digits, np.zeros(11, dtype=int), VOCAB, center_logits=False, aggregation="sum"
    )
    assert decision.predicted[0] == 2


def test_center_logits_removes_constant_bias() -> None:
    rng = np.random.default_rng(0)
    true_digit = 3
    counts = rng.multinomial(200, np.ones(9) / 9)
    digits = np.concatenate([[d] * n for d, n in enumerate(counts, 1)])
    logits = np.where(digits == true_digit, 0.2, -0.2) + 5.0
    groups = np.zeros(len(digits), dtype=int)

    raw = decide_by_group(logits, digits, groups, VOCAB, center_logits=False, aggregation="sum")
    centered = decide_by_group(logits, digits, groups, VOCAB, center_logits=True, aggregation="sum")
    assert centered.predicted[0] == true_digit
    # Regression guard: without centering the constant bias must corrupt the
    # ranking through the c * n_d term.
    assert raw.predicted[0] != true_digit


@pytest.mark.parametrize(
    "reference_options,power,winner",
    [({}, 0.0, 2), ({"aggregation": "mean"}, 0.0, 2), ({"aggregation": "sum"}, 1.0, 1)],
    ids=["default-mean", "explicit-mean", "sum"],
)
def test_tempered_evidence_interpolates_mean_and_sum(reference_options, power, winner):
    data = ([0.5, 0.5, 0.5, 1.0], [1, 1, 1, 2], [0] * 4, VOCAB)
    reference = decide_by_group(*data, center_logits=False, **reference_options)
    tempered = decide_by_group(
        *data, center_logits=False, aggregation="tempered_evidence", evidence_count_power=power
    )
    assert reference.predicted[0] == tempered.predicted[0] == winner
    np.testing.assert_allclose(
        list(tempered.scores_by_group["0"].values()),
        list(reference.scores_by_group["0"].values()),
    )


def test_trial_weights_are_label_free_reliability():
    data = dict(
        logits=[2.0, 2.0, -5.0, 0.5],
        codes=[1, 1, 1, 2],
        group_ids=[0] * 4,
        vocabulary=VOCAB,
        center_logits=False,
        aggregation="tempered_evidence",
        evidence_count_power=0.0,
    )
    assert decide_by_group(**data).predicted[0] == 2
    assert decide_by_group(**data, trial_weights=[1.0, 1.0, 0.0, 1.0]).predicted[0] == 1


def test_exact_ties_abstain_instead_of_breaking_by_label() -> None:
    digits = np.array([1, 1, 2, 2, 3, 3])
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    decision = decide_by_group(logits, digits, np.zeros(6, dtype=int), VOCAB, center_logits=False)
    assert decision.predicted[0] is None


@pytest.mark.parametrize(
    "options,prediction", [({}, 5), ({"missing_candidate_policy": "abstain"}, None)]
)
def test_missing_candidates_follow_declared_policy(options, prediction):
    digits = np.repeat(np.arange(1, 9), 2)  # Candidate 9 is absent.
    logits = np.where(digits == 5, 5.0, 1.0)
    result = candidate_evidence_scores(logits, digits, VOCAB, **options)
    assert result.predicted == prediction
    assert result.scores[9] == -np.inf
    assert result.counts[9] == 0
    with pytest.raises(ValueError, match="missing_candidate_policy"):
        candidate_evidence_scores(logits, digits, VOCAB, missing_candidate_policy="ignore")


def test_multi_group_decisions_are_independent() -> None:
    digits = np.tile(np.repeat(VOCAB, 2), 2)
    groups = np.repeat([0, 1], len(VOCAB) * 2)
    logits = np.zeros(len(digits))
    logits[(groups == 0) & (digits == 5)] = 3.0
    logits[(groups == 1) & (digits == 8)] = 4.0
    decision = decide_by_group(logits, digits, groups, VOCAB)
    assert decision.predicted[0] == 5
    assert decision.predicted[1] == 8


def test_candidate_permutation_leaves_decisions_unchanged() -> None:
    """Relabeling candidates and truth simultaneously must not change hits."""

    rng = np.random.default_rng(9)
    digits = rng.integers(1, 10, size=300)
    logits = rng.standard_normal(300)
    groups = np.repeat(np.arange(5), 60)
    permutation = {d: 10 - d for d in VOCAB}
    relabeled = np.array([permutation[int(d)] for d in digits])

    original = decide_by_group(logits, digits, groups, VOCAB, center_logits=True)
    permuted = decide_by_group(
        logits, relabeled, groups, [permutation[d] for d in VOCAB], center_logits=True
    )
    for group_index, _group in enumerate(original.group_ids):
        left = original.predicted[group_index]
        right = permuted.predicted[group_index]
        assert (left is None) == (right is None)
        if left is not None:
            assert permutation[int(left)] == int(right)


def test_codes_outside_vocabulary_fail_closed() -> None:
    with pytest.raises(ValueError, match="fail closed"):
        candidate_evidence_scores([1.0, 2.0], [1, 42], [1, 2, 3])
    with pytest.raises(ValueError, match="fail closed"):
        decide_by_group([1.0, 2.0], [1, 42], [0, 0], [1, 2, 3])


def test_duplicate_vocabulary_raises() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        candidate_evidence_scores([1.0, 2.0], [1, 2], [1, 1, 2, 3])


def test_nan_logits_raise() -> None:
    with pytest.raises(ValueError):
        decide_by_group(
            [1.0, 0.5, float("nan"), 0.3, 0.8],
            [1, 2, 3, 4, 5],
            ["s0"] * 5,
            VOCAB,
        )


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        decide_by_group([1.0, 2.0], [1, 2], [0], VOCAB)


def test_invalid_aggregation_raises() -> None:
    with pytest.raises(ValueError):
        candidate_evidence_scores([1.0], [1], VOCAB, aggregation="avg")
    with pytest.raises(ValueError):
        decide_by_group([1.0, 2.0], [1, 2], [0, 0], VOCAB, aggregation="trim0.2")
    with pytest.raises(ValueError, match="count_power"):
        decide_by_group(
            [1.0, 0.0],
            [1, 2],
            [0, 0],
            VOCAB,
            aggregation="tempered_evidence",
            evidence_count_power=1.5,
        )


def test_trim_and_precision_aggregations() -> None:
    digits = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    logits = np.array([10.0, -10.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    trimmed = candidate_evidence_scores(logits, digits, VOCAB, aggregation="trim0.2")
    # Both extremes are trimmed per candidate: 1 keeps {1,1,1} -> 3; 2 keeps
    # {0.5 x3} -> 1.5.
    assert trimmed.scores[1] == pytest.approx(3.0)
    assert trimmed.predicted == 1
    # Inverse-variance weighting: candidate 1's second (high-value) trial is
    # a hundred times more reliable than its first, so the precision mean
    # tracks it, while the plain mean still prefers candidate 2.
    values = np.array([0.0, 2.0, 1.5, 1.5, 1.5])
    codes = np.array([1, 1, 2, 2, 2])
    precision = candidate_evidence_scores(
        values,
        codes,
        VOCAB,
        aggregation="precision",
        logit_variances=np.array([1.0, 0.01, 1.0, 1.0, 1.0]),
    )
    assert precision.scores[1] == pytest.approx(2.0 * 100.0 / 101.0, abs=1e-3)
    assert precision.predicted == 1
    assert candidate_evidence_scores(values, codes, VOCAB).predicted == 2
    with pytest.raises(ValueError, match="precision"):
        candidate_evidence_scores(logits, digits, VOCAB, aggregation="precision")


def test_count_tempered_evidence_scores_contract() -> None:
    scores = count_tempered_evidence_scores(
        np.asarray([[2.0, 4.0]]),
        np.asarray([[2.0, 4.0]]),
        np.asarray([[2.0, 4.0]]),
        count_power=0.5,
    )
    assert scores.shape == (1, 2)
    with pytest.raises(ValueError):
        count_tempered_evidence_scores(
            np.asarray([1.0]), np.asarray([1.0]), np.asarray([1.0]), count_power=2.0
        )
