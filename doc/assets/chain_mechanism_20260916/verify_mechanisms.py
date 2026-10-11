"""Deterministic mechanism checks; no model fitting or optimizer updates.

Run with the repository virtual environment. Synthetic counterexamples establish
possibility and invariance, never the effect size in recorded EEG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import sosfreqz
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from data.preprocess import design_continuous_sos  # noqa: E402
from models.candidate_evidence import membership_evidence  # noqa: E402
from models.parameter_groups import SourceParameterPrior  # noqa: E402


def check_decision(matrix, target, expected_ratio):
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    labels = matrix[:, target].long()
    weights = torch.tensor([1.0, float((labels == 0).sum() / labels.sum())])
    evidence = torch.zeros(len(matrix), dtype=torch.float64, requires_grad=True)
    logits = torch.stack((evidence * 0, evidence), dim=1)
    trial = F.cross_entropy(logits, labels, weight=weights.double())
    scores = membership_evidence(evidence, matrix)
    decision = F.cross_entropy(scores[None], torch.tensor([target]))
    gt = torch.autograd.grad(trial, evidence, retain_graph=True)[0]
    gd = torch.autograd.grad(decision, evidence)[0]
    torch.testing.assert_close(gd, gt * expected_ratio, atol=1e-12, rtol=1e-12)
    coefficient = float(gt.norm() / gd.norm())
    torch.testing.assert_close(gd * coefficient, gt, atol=1e-12, rtol=1e-12)

    probe = torch.linspace(-0.7, 1.1, len(matrix), requires_grad=True, dtype=torch.float64)
    score = membership_evidence(probe, matrix)
    loss = F.cross_entropy(score[None], torch.tensor([target]))
    gradient = torch.autograd.grad(loss, probe)[0]
    expected = matrix @ (score.softmax(0) - F.one_hot(torch.tensor(target), matrix.shape[1]))
    torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
    shifted = membership_evidence(probe.detach(), matrix, logit_offset=13.0)
    torch.testing.assert_close(score.softmax(0), shifted.softmax(0), atol=1e-12, rtol=1e-12)
    assert int(score.argmax()) == int((score / 10).argmax())
    return {"ratio_at_zero": expected_ratio, "geometry_loss_coefficient": coefficient}


def main():
    torch.set_num_threads(1)
    results = {}
    # Seven observed native potentials, six differences against channel 1 (Cz).
    keep = [0, 2, 3, 4, 5, 6]
    projection = np.eye(7)[keep] - np.eye(7)[[1]]
    delta = np.ones(7)
    assert np.array_equal(projection @ delta, np.zeros(6))
    covariance = projection @ projection.T
    np.testing.assert_allclose(np.diag(covariance), 2)
    np.testing.assert_allclose(covariance[0, 1:], 1)
    augmented = np.vstack((projection, np.eye(7)[1]))
    assert np.linalg.matrix_rank(augmented) == 7
    results["spatial_projection"] = {"rank": 6, "common_signal_remaining": 0.0,
                                     "native_information_for_unit_covariance": 7.0,
                                     "projected_information": 0.0}

    baseline = np.zeros(179)
    baseline[:26] = 1 / 26
    operator = np.eye(179) - np.outer(np.ones(179), baseline)
    np.testing.assert_allclose(operator @ operator, operator, atol=1e-15)
    output_covariance = operator @ operator.T
    np.testing.assert_allclose(output_covariance[40, 40], 1 + 1 / 26)
    np.testing.assert_allclose(output_covariance[40, 41], 1 / 26)
    results["baseline_white_noise"] = {"post_variance_ratio": float(output_covariance[40, 40])}

    sos = design_continuous_sos(250, l_freq=0.1, h_freq=30, order=4,
                                notch_freqs_hz=(50,), notch_quality_factor=30)
    frequencies = np.array([0.1, 0.2, 1, 10, 20, 30])
    _, response = sosfreqz(sos, worN=frequencies, fs=250)
    amplitude = np.abs(response) ** 2
    assert abs(amplitude[0] - 0.5) < 1e-4
    assert amplitude[1] > 0.99 and amplitude[3] > 0.999
    results["production_bidirectional_response"] = dict(
        zip(map(str, frequencies), amplitude.tolist(), strict=True)
    )

    direct = np.tile(np.eye(9), (5, 1))
    results["direct_9_R5"] = check_decision(direct, 0, 4 * 5 * 8 / 9)
    single_cycle = np.zeros((12, 36))
    for row in range(6):
        single_cycle[row, row * 6:(row + 1) * 6] = 1
    for column in range(6):
        single_cycle[6 + column, column::6] = 1
    matrix = np.tile(single_cycle, (10, 1))
    results["row_column_6x6_R10"] = check_decision(matrix, 0, 8 * 10 * 5 / 6)
    e = torch.linspace(-1, 1, 120, dtype=torch.float64)
    score = membership_evidence(e, matrix)
    parts = e.reshape(10, 12).sum(0)
    joint = F.cross_entropy(score[None], torch.tensor([0]))
    separate = F.cross_entropy(parts[:6][None], torch.tensor([0]))
    separate += F.cross_entropy(parts[6:][None], torch.tensor([0]))
    torch.testing.assert_close(joint, separate, atol=1e-12, rtol=1e-12)
    results["row_column_factorization"] = "PASS"

    gradient = torch.tensor([0.001, -0.02, 0.1], dtype=torch.float64)
    multiplier = 1 + 8 * 10 * 5 / 6
    normalized = gradient / (gradient.abs() + 1e-8)
    scaled = multiplier * gradient / (multiplier * gradient.abs() + 1e-8)
    discrepancy = float((scaled - normalized).abs().max())
    assert discrepancy < 1e-5
    results["adam_collinear_counterexample"] = {"max_update_difference_per_lr": discrepancy}

    # Equicorrelated noise has an offset common to all equal-exposure candidates.
    z = np.linspace(-2, 3, 45)
    precision = np.linalg.inv(np.eye(45) + 0.3 * np.ones((45, 45)))
    sums = direct.T @ z
    optimal = direct.T @ precision @ z
    optimal -= 0.5 * np.diag(direct.T @ precision @ direct)
    np.testing.assert_allclose(optimal - optimal.mean(), sums - sums.mean(), atol=1e-12)
    results["correlation_does_not_necessarily_change_order"] = "PASS"

    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.head.weight.fill_(1)

        def adaptation_parameter_groups(self):
            return {"head": tuple(self.head.parameters())}

    model = ExampleModel()
    prior = SourceParameterPrior(model)
    x = torch.tensor([[2.0]])
    old = model.head((x - 0) / 1)
    new = model.head((x - 1) / 2)
    assert not torch.equal(old, new)
    penalty = prior.loss()
    assert penalty.item() == 0
    assert torch.equal(torch.autograd.grad(penalty, model.head.weight)[0],
                       torch.zeros_like(model.head.weight))
    results["parameter_center_not_function_center"] = {"prior": 0, "old": old.item(), "new": new.item()}

    positives = np.array([10.0, 1.0])
    negatives = np.array([9.0, 0.0])
    auc = float((positives[:, None] > negatives).mean())
    permuted_auc = float((positives[::-1, None] > negatives).mean())
    assert auc == permuted_auc == 0.75
    assert (positives > negatives).mean() == 1
    assert (positives[::-1] > negatives).mean() == 0.5
    results["same_trial_auc_different_decisions"] = {"auc": auc, "hits": [1.0, 0.5]}

    document = {"scope": "synthetic mathematics and production operators only; no training",
                "status": "PASS", "checks": results}
    output = Path(__file__).with_name("verification.json")
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "groups": len(results), "output": str(output)}))


if __name__ == "__main__":
    main()
