from __future__ import annotations

import numpy as np
import torch

from models.reliability_v12 import (
    CleanProbabilityEstimator,
    FidelityEstimator,
    convert_prior_odds,
    digit_chain_scores_from_components,
    evaluate_clean_probability_chain_gate,
    evaluate_clean_probability_gate,
    evaluate_fidelity_gate,
)


def _fidelity_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean = rng.normal(0.0, 1.0, (n, 8))
    corrupt: list[np.ndarray] = []
    types = ["seen_a", "seen_b", "unseen_c", "unseen_d"]
    means = [3.8, 4.2, 4.5, 5.0]
    corruption_type = np.empty(5 * n, dtype=object)
    corruption_type[:n] = "clean"
    for index, (name, mean) in enumerate(zip(types, means, strict=True)):
        rows = rng.normal(mean, 1.0, (n, 8))
        corrupt.append(rows)
        corruption_type[(index + 1) * n : (index + 2) * n] = name
    q = np.concatenate([clean, *corrupt], axis=0)
    corruption_type = np.asarray(corruption_type)
    labels = np.concatenate((np.ones(n), np.zeros(4 * n)))
    return q, labels, corruption_type, np.arange(len(q)) % 4


def _chain_evidence(
    n_sequences: int, n_candidates: int = 9, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true = rng.integers(0, n_candidates, size=n_sequences)
    scores = rng.normal(0.0, 0.25, size=(n_sequences, n_candidates))
    scores[np.arange(n_sequences), true] += 2.0
    return scores, true


def test_fidelity_gate_on_unseen_corruption_types() -> None:
    rng = np.random.default_rng(0)
    q, labels, corruption_type, subject_ids = _fidelity_data(120, rng)
    estimator = FidelityEstimator(n_features=8, hidden_size=32)
    estimator.fit_normalizer(torch.from_numpy(q).float())

    seen = np.isin(corruption_type, ["seen_a", "seen_b"]) | (labels == 1)
    seen_q = torch.from_numpy(q[seen]).float()
    seen_labels = labels[seen]
    clean = seen_q[seen_labels == 1]
    corrupt = seen_q[seen_labels == 0]
    optimizer = torch.optim.Adam(estimator.parameters(), lr=0.02)
    for _ in range(120):
        optimizer.zero_grad()
        clean_scores = estimator.forward(clean)
        corrupt_scores = estimator.forward(corrupt)
        loss = estimator.margin_rank_loss(clean_scores, corrupt_scores)
        loss.backward()
        optimizer.step()

    unseen_types = {"unseen_c", "unseen_d"}
    gate_mask = (labels == 1) | np.isin(corruption_type, list(unseen_types))
    report = evaluate_fidelity_gate(
        estimator,
        torch.from_numpy(q[gate_mask]).float(),
        labels[gate_mask],
        corruption_type[gate_mask],
        subject_ids[gate_mask],
        unseen_types=unseen_types,
    )
    assert report.passed, report


def test_clean_probability_prior_shift_and_unseen_types() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    clean = rng.normal(0.0, 1.0, (n, 4))
    corrupt_seen = {
        "seen_a": rng.normal(0.8, 0.8, (n, 4)),
        "seen_b": rng.normal(1.0, 0.8, (n, 4)),
    }
    corrupt_unseen = {
        "unseen_c": rng.normal(1.1, 0.8, (n, 4)),
        "unseen_d": rng.normal(1.3, 0.8, (n, 4)),
    }

    def pool(corruptions, n_clean, n_corrupt_each):
        q = [clean[:n_clean]]
        labels = [np.ones(n_clean)]
        for value in corruptions:
            q.append(value[:n_corrupt_each])
            labels.append(np.zeros(n_corrupt_each))
        return np.concatenate(q), np.concatenate(labels)

    # Calibration at gate prevalence 0.2 (1 clean : 4 corrupt). Logistic fit
    # and isotonic calibration use disjoint halves of the calibration pool.
    cal_q, cal_y = pool(
        [corrupt_seen["seen_a"], corrupt_seen["seen_b"]], 200, 400
    )
    cal_order = rng.permutation(len(cal_q))
    cal_q = cal_q[cal_order]
    cal_y = cal_y[cal_order]
    half = len(cal_q) // 2
    estimator = CleanProbabilityEstimator().fit(cal_q[:half], cal_y[:half])
    estimator.fit_calibrator(cal_q[half:], cal_y[half:], calibration_prior=0.2)

    # Deployment prior 0.9.
    dep_q, dep_y = pool(
        [
            corrupt_seen["seen_a"],
            corrupt_seen["seen_b"],
            corrupt_unseen["unseen_c"],
            corrupt_unseen["unseen_d"],
        ],
        1800,
        50,
    )
    raw = estimator.predict_calibrated(dep_q)
    chain_scores, chain_true = _chain_evidence(len(dep_y))
    raw_gate = evaluate_clean_probability_gate(
        raw, dep_y, chain_scores=chain_scores, true_candidates=chain_true
    )
    assert raw_gate["passed"] is False, raw_gate
    assert raw_gate["chain"]["passed"] is True, raw_gate

    converted = convert_prior_odds(raw, calibration_prior=0.2, deployment_prior=0.9)
    converted_gate = evaluate_clean_probability_gate(
        converted, dep_y, chain_scores=chain_scores, true_candidates=chain_true
    )
    assert converted_gate["passed"] is True, converted_gate

    # Hard-label refit at the deployment prior is the production path.
    order = rng.permutation(len(dep_q))
    dep_q = dep_q[order]
    deployment_chain_scores, deployment_chain_true = _chain_evidence(
        len(dep_y) // 2, seed=12
    )
    dep_y = dep_y[order]
    deployment_estimator = CleanProbabilityEstimator().fit(dep_q[: len(dep_q) // 2], dep_y[: len(dep_q) // 2])
    deployment_probs = deployment_estimator.predict_proba(dep_q[len(dep_q) // 2 :])
    deployment_gate = evaluate_clean_probability_gate(
        deployment_probs,
        dep_y[len(dep_y) // 2 :],
        chain_scores=deployment_chain_scores,
        true_candidates=deployment_chain_true,
    )
    assert deployment_gate["passed"] is True, deployment_gate



def test_clean_probability_gate_requires_digit_chain_acceptance() -> None:
    """blueprint 4.2: rho Brier/ECE alone cannot enable clean_probability."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, 200)
    y = (p > 0.5).astype(int)
    report = evaluate_clean_probability_gate(p, y)
    assert report["passed"] is False
    assert report["failure"] == "missing_digit_chain_nll_or_true_candidates"
    assert report["chain"]["passed"] is False


def test_clean_probability_gate_fails_when_chain_is_uniform() -> None:
    """Rho calibration may pass while the digit chain has no skill."""
    y = np.asarray([0, 1] * 50)
    p = np.where(y == 1, 0.85, 0.15)
    report = evaluate_clean_probability_gate(
        p,
        y,
        chain_scores=np.zeros((len(y), 9)),
        true_candidates=np.arange(len(y)) % 9,
    )
    assert report["rho_passed"] is True
    assert report["passed"] is False
    assert report["chain"]["failure"] == "digit_chain_nll_worse_than_uniform"


def test_digit_chain_scores_match_blueprint_mixture_formula() -> None:
    rho = np.asarray([[0.9, 0.2], [1.0, 0.0]])
    clean = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    artifact = np.asarray(
        [[[-1.0, -2.0], [-3.0, -4.0]], [[-0.5, -1.5], [-2.5, -3.5]]]
    )
    lengths = np.asarray([2, 2])
    scores = digit_chain_scores_from_components(rho, clean, artifact, lengths=lengths)
    expected = np.zeros((2, 2))
    for n in range(2):
        for c in range(2):
            expected[n, c] = sum(
                np.log(rho[n, t] * np.exp(clean[n, t, c]) + (1 - rho[n, t]) * np.exp(artifact[n, t, c]))
                for t in range(2)
            )
    assert np.allclose(scores, expected)


def test_chain_gate_reports_calibration_rank_flips() -> None:
    """The gate must show decision agreement, not infer it from rho AUC."""
    scores, true = _chain_evidence(20)
    reference = scores.copy()
    true_index = int(true[0])
    other = (true_index + 1) % scores.shape[1]
    reference[0, true_index] = scores[0, other]
    reference[0, other] = scores[0, true_index]
    report = evaluate_clean_probability_chain_gate(
        scores, true, reference_chain_scores=reference
    )
    assert report.passed is True
    assert report.decision_reverified is True
    assert report.decision_agreement_rate is not None
    assert report.decision_agreement_rate < 1.0


def test_chain_gate_rejects_uniform_worse_chain() -> None:
    scores = np.zeros((6, 9))
    true = np.asarray([0, 1, 2, 3, 4, 5])
    report = evaluate_clean_probability_chain_gate(scores, true)
    assert report.passed is False
    assert report.failure == "digit_chain_nll_worse_than_uniform"

def test_prior_odds_conversion_formula() -> None:
    p = np.asarray([0.5, 0.8, 0.2])
    out = convert_prior_odds(p, calibration_prior=0.2, deployment_prior=0.9)
    expected_odds = p / (1 - p) * (0.9 / 0.1) * (0.8 / 0.2)
    expected = expected_odds / (1 + expected_odds)
    assert np.allclose(out, expected)
