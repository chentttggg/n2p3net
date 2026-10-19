"""Independent design oracles; no training data and no production writes."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    output = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    checks = {}

    # Quadratic local oracle: finite source precision permits every direction.
    hessian = np.diag([1.0, 4.0])
    optimum = np.array([3.0, -2.0])
    source = np.array([-1.0, 1.0])
    estimates = {}
    for precision in [0.0, 0.5, 5.0, 5000.0]:
        matrix = hessian + precision * np.eye(2)
        estimate = np.linalg.solve(matrix, hessian @ optimum + precision * source)
        gradient = hessian @ (estimate - optimum) + precision * (estimate - source)
        assert np.linalg.norm(gradient) < 1e-10
        estimates[str(precision)] = estimate.tolist()
    assert np.allclose(estimates["0.0"], optimum)
    assert np.linalg.norm(np.array(estimates["5000.0"]) - source) < 0.003
    checks["finite_prior_quadratic_solution"] = estimates

    # At the source initialization a quadratic prior has zero first-step gradient.
    # One explicit gradient step cannot learn its precision from the query loss.
    def outer_loss(precision: torch.Tensor, steps: int) -> torch.Tensor:
        center = torch.tensor(0.3, requires_grad=True)
        fast = center
        for _ in range(steps):
            loss = 0.5 * (fast - 2.0).square() + 0.5 * precision * (fast - center).square()
            (gradient,) = torch.autograd.grad(loss, fast, create_graph=True)
            fast = fast - 0.2 * gradient
        return 0.5 * (fast + 1.0).square()

    derivatives = {}
    for steps in [1, 2]:
        precision = torch.tensor(0.7, requires_grad=True)
        (gradient,) = torch.autograd.grad(outer_loss(precision, steps), precision)
        epsilon = 1e-5
        finite_difference = (
            outer_loss(torch.tensor(0.7 + epsilon), steps)
            - outer_loss(torch.tensor(0.7 - epsilon), steps)
        ) / (2.0 * epsilon)
        assert torch.allclose(gradient, finite_difference, atol=1e-9, rtol=1e-7)
        derivatives[str(steps)] = float(gradient)
    assert derivatives["1"] == 0.0
    assert abs(derivatives["2"]) > 0.01
    checks["precision_meta_gradient"] = derivatives

    # A mean-loss penalty of lambda corresponds to sum-loss precision N*lambda.
    data_mean, prior_center, precision = 2.0, -1.0, 3.0
    mean_penalty, count = precision / 20.0, 20
    by_sum = (count * data_mean + precision * prior_center) / (count + precision)
    by_mean = (data_mean + mean_penalty * prior_center) / (1.0 + mean_penalty)
    assert abs(by_sum - by_mean) < 1e-14
    checks["mean_sum_prior_scale"] = by_sum

    # A positive raw gradient dot product does not license a preconditioned update.
    main_gradient = np.array([1.0, 1.0])
    auxiliary_gradient = np.array([-1.9, 2.0])
    preconditioner = np.diag([10.0, 0.1])
    raw_dot = float(main_gradient @ auxiliary_gradient)
    main_change = float(-main_gradient @ preconditioner @ (main_gradient + auxiliary_gradient))
    assert raw_dot > 0.0 and main_change > 0.0
    checks["preconditioned_gradient_counterexample"] = {
        "raw_gradient_dot": raw_dot,
        "linear_main_loss_change_per_unit_step": main_change,
    }

    # Full 2^4 design estimates all interactions; source states share D on/off.
    factors = ["residual", "supcon", "conditional_mmd", "decision_loss"]
    rows = []
    for values in itertools.product([0, 1], repeat=4):
        row = dict(zip(factors, values, strict=True))
        row["treatment"] = "r{}s{}m{}d{}".format(*values)
        row["source_key"] = "r{}s{}m{}".format(*values[:3])
        row["decision_phase"] = "calibration_only"
        rows.append(row)
    columns = [np.ones(16)]
    encoded = 2 * np.array([[r[f] for f in factors] for r in rows]) - 1
    for degree in range(1, 5):
        for subset in itertools.combinations(range(4), degree):
            columns.append(np.prod(encoded[:, subset], axis=1))
    design = np.column_stack(columns)
    assert np.linalg.matrix_rank(design) == 16
    assert np.array_equal(design.T @ design, 16 * np.eye(16))
    assert len({r["source_key"] for r in rows}) == 8
    with (output / "factorial_design.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    interaction = 0.76 - 0.69 - 0.69 + 0.70
    assert math.isclose(interaction, 0.08)
    checks["factorial_design"] = {
        "rows": 16,
        "rank": int(np.linalg.matrix_rank(design)),
        "source_states": 8,
        "synthetic_positive_interaction": interaction,
    }

    # Given schedule, nine possible targets yield at most nine distinct label vectors.
    memberships = np.tile(np.eye(9, dtype=int), (5, 1))
    possible_labels = {tuple(memberships[:, target]) for target in range(9)}
    assert len(possible_labels) == 9
    checks["label_entropy_upper_bound_bits"] = math.log2(len(possible_labels))

    # Proposed pre-spatial masked BN oracle: all-observed agrees with PyTorch;
    # missing NaNs are removed before the temporal operation and cannot affect grads.
    torch.manual_seed(22)
    h = torch.randn(3, 4, 6, 8)
    affine = torch.randn(4)
    bias = torch.randn(4)

    def masked_normalize(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        mask = observed[:, None, :, None]
        clean = torch.where(mask, values, 0.0)
        count = observed.sum() * values.shape[-1]
        assert count > 1
        mean = clean.sum((0, 2, 3)) / count
        delta = torch.where(mask, clean - mean[None, :, None, None], 0.0)
        variance = delta.square().sum((0, 2, 3)) / count
        standardized = delta / torch.sqrt(variance[None, :, None, None] + 1e-5)
        return torch.where(
            mask, standardized * affine[None, :, None, None] + bias[None, :, None, None], 0.0
        )

    mask = torch.ones(3, 6, dtype=torch.bool)
    reference = torch.nn.functional.batch_norm(h, None, None, affine, bias, training=True)
    torch.testing.assert_close(masked_normalize(h, mask), reference, rtol=1e-12, atol=1e-12)
    mask[:, 1::2] = False
    raw = torch.randn(3, 1, 6, 8)
    outputs, gradients = [], []
    coefficient = torch.randn(3, 4, 6, 8)
    for fill in [float("nan"), 1e20]:
        corrupted = torch.where(mask[:, None, :, None], raw, fill)
        clean = torch.where(mask[:, None, :, None], corrupted, 0.0)
        weight = torch.tensor([0.2, -0.3, 0.5, 1.0]).reshape(4, 1, 1, 1).requires_grad_()
        features = torch.nn.functional.conv2d(clean, weight)
        normalized = masked_normalize(features, mask)
        (gradient,) = torch.autograd.grad((normalized * coefficient).sum(), weight)
        assert torch.isfinite(normalized).all() and torch.isfinite(gradient).all()
        outputs.append(normalized)
        gradients.append(gradient)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    checks["masked_normalization_oracle"] = {
        "all_observed_max_abs": float(
            (masked_normalize(h, torch.ones_like(mask)) - reference).abs().max()
        ),
        "missing_nan_large_value_outputs_and_gradients_equal": True,
        "scope": "independent formula oracle; production masked BN not implemented",
    }

    payload = {"status": "PASS", "oracle_groups": 7, "checks": checks}
    (output / "synthesis_math.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
