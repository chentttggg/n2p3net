"""Finite behavioral probes of real callers; these do not prove Python semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from models.candidate_evidence import membership_evidence  # noqa: E402
from transfer.binary import binary_time_split  # noqa: E402
from transfer.noise_model import JointNoiseModel, validate_covariance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = []
    rng = np.random.default_rng(20261016)

    # Paired identical variables give a maximally correlated valid covariance.
    draws = np.arange(-5, 6, dtype=float)[:, None]
    model = JointNoiseModel.fit(
        draws,
        draws,
        fit_scope="target_calibration",
        scope_id="synthetic_lean_pilot",
        draw_ids=[f"draw_{i}" for i in range(len(draws))],
        shrinkage=0,
    )
    full = model.project(np.ones((1, 1)), np.ones((1, 1))).covariance.item()
    omitted = float(np.trace(model.covariance))
    np.testing.assert_allclose(full, 2 * omitted, atol=1e-12)
    cases.append({"name": "correlated_cross_terms", "full": full, "omitted": omitted})
    max_error = 0.0
    for _ in range(32):
        a, b = rng.normal(size=(3, 1)), rng.normal(size=(3, 1))
        actual = model.project(a, b).covariance
        g, c, o = model.covariance[0, 0], model.covariance[0, 1], model.covariance[1, 1]
        expanded = g * a @ a.T + c * a @ b.T + c * b @ a.T + o * b @ b.T
        max_error = max(max_error, float(np.max(np.abs(actual - expanded))))
        np.testing.assert_allclose(actual, expanded, rtol=1e-12, atol=1e-12)
        assert np.linalg.eigvalsh(actual).min() >= -1e-12
    cases.append({"name": "joint_covariance_expansion", "draws": 32, "max_abs_error": max_error})

    # The real PSD premise is stronger than this floating-point admission gate.
    near_psd = validate_covariance(np.diag([1.0, -5e-11]), name="pilot near PSD")
    negative_variance = float(np.linalg.eigvalsh(near_psd).min())
    assert negative_variance < 0
    cases.append(
        {
            "name": "approximate_psd_gate_is_not_exact_psd",
            "accepted_min_eigenvalue": negative_variance,
            "interpretation": "numerical admission gap; exact PSD theorem does not certify this input",
        }
    )

    membership = np.array([[1, 0], [1, 0], [0, 1]])
    values = np.ones(3)
    scores = membership_evidence(values, membership)
    shifted = membership_evidence(values, membership, logit_offset=2)
    np.testing.assert_array_equal(scores, [2, 1])
    np.testing.assert_array_equal(shifted, [-2, -1])
    cases.append(
        {"name": "offset_rank_reversal", "ordinary": scores.tolist(), "shifted": shifted.tolist()}
    )

    available = np.array([True, False, True])
    missing = np.array([1.0, np.nan, 1.0])
    expected = membership_evidence(values, membership, available=available)
    np.testing.assert_array_equal(
        membership_evidence(missing, membership, available=available), expected
    )
    actual_torch = membership_evidence(
        torch.tensor(missing), membership, available=available
    ).numpy()
    np.testing.assert_array_equal(actual_torch, expected)
    cases.append({"name": "unavailable_nan_masked_numpy_and_torch", "score": expected.tolist()})

    balanced = np.array([[1, 0], [0, 1]])
    for offset in [-10.0, 0.0, 2.0, 10.0]:
        actual = membership_evidence(np.array([3.0, 1.0]), balanced, logit_offset=offset)
        assert actual[0] - actual[1] == 2.0
    cases.append({"name": "balanced_offset_difference", "offsets": 4})

    # Dyadic times exactly represent these examples in float64.
    onsets = np.arange(40) / 4.0
    availability = onsets + 1.0
    for maximum in range(1, 31):
        for k in range(maximum + 1):
            split = binary_time_split(
                onsets,
                availability,
                calibration_trials=k,
                maximum_calibration_trials=maximum,
                tmin_s=-0.25,
            )
            assert not np.intersect1d(split.calibration_rows, split.test_rows).size
            assert np.all(onsets[split.test_rows] - 0.25 >= availability[:maximum].max())
            np.testing.assert_array_equal(
                np.sort(np.concatenate((split.test_rows, split.purged_rows))),
                np.arange(maximum, len(onsets)),
            )
    cases.append(
        {"name": "binary_time_split_support_and_partition", "budget_pairs": sum(range(2, 32))}
    )
    files = [
        "src/transfer/noise_model.py",
        "src/models/candidate_evidence.py",
        "src/transfer/binary.py",
    ]
    result = {
        "scope": "finite Python/NumPy/Torch probes; not a formal refinement proof",
        "seed": 20261016,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in files},
        "cases": cases,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
