from __future__ import annotations

import numpy as np
import pytest

from data.euclidean_alignment import (
    EA_ALIGNMENT_SCHEMA,
    EuclideanAlignmentConfig,
    EuclideanAlignmentState,
    fit_euclidean_alignment,
    fit_per_domain_euclidean_alignment,
    transform_euclidean_alignment,
)


def _trials(scale: np.ndarray, n: int, t: int) -> np.ndarray:
    rng = np.random.default_rng(4)
    z = rng.normal(size=(n, scale.shape[0], t))
    return np.einsum("cp,npt->nct", scale, z)


def test_ea_reference_whitens_mean_trial_covariance() -> None:
    scale = np.array([[2.0, 0.4], [0.0, 1.5]])
    x = _trials(scale, 400, 32)
    state = fit_euclidean_alignment(
        x,
        fit_scope="source_rows",
        config=EuclideanAlignmentConfig(relative_eigenvalue_floor=1e-10, shrinkage=0.0),
    )
    y = transform_euclidean_alignment(x, state)
    covariance = np.mean(np.einsum("nct,nut->ncu", y, y), axis=0) / y.shape[2]
    assert np.allclose(covariance, np.eye(2), atol=1e-8)
    assert state.fit_scope == "source_rows"
    assert state.record()["schema"] == EA_ALIGNMENT_SCHEMA
    restored = EuclideanAlignmentState.from_record(state.record())
    np.testing.assert_allclose(
        transform_euclidean_alignment(x, restored),
        y,
        atol=1e-6,
    )


def test_ea_per_domain_fits_domain_specific_references() -> None:
    x_a = _trials(np.array([[3.0, 0.0], [0.0, 1.0]]), 120, 16)
    x_b = _trials(np.array([[1.0, 0.5], [0.0, 2.0]]), 120, 16)
    x = np.concatenate((x_a, x_b), axis=0)
    domains = np.asarray(["a"] * 120 + ["b"] * 120)
    states = fit_per_domain_euclidean_alignment(
        x,
        domains,
        config=EuclideanAlignmentConfig(relative_eigenvalue_floor=1e-10),
    )
    assert set(states) == {"a", "b"}
    for domain in ("a", "b"):
        mask = domains == domain
        y = transform_euclidean_alignment(x[mask], states[domain])
        covariance = np.mean(np.einsum("nct,nut->ncu", y, y), axis=0) / y.shape[2]
        assert np.allclose(covariance, np.eye(2), atol=1e-8)


def test_ea_rejects_condition_number_and_bad_scope() -> None:
    x = _trials(np.eye(2), 20, 8)
    with pytest.raises(ValueError, match="condition number"):
        fit_euclidean_alignment(
            x,
            fit_scope="target_calibration",
            config=EuclideanAlignmentConfig(
                relative_eigenvalue_floor=1e-12,
                max_condition_number=1.0 + 1e-9,
            ),
        )
    with pytest.raises(ValueError, match="per_domain"):
        fit_euclidean_alignment(x, fit_scope="per_domain")
    with pytest.raises(ValueError, match="fit_scope must be one of"):
        fit_euclidean_alignment(x, fit_scope="target_test")


def test_default_ea_preserves_whitening_at_physical_microvolt_scale() -> None:
    x = _trials(np.array([[2.0, 0.4], [0.0, 1.5]]) * 1e-5, 80, 64)
    state = fit_euclidean_alignment(x, fit_scope="source_rows")
    aligned = transform_euclidean_alignment(x, state).astype(np.float64)
    moment = np.einsum("nct,nut->cu", aligned, aligned) / (len(aligned) * aligned.shape[-1])
    np.testing.assert_allclose(moment, np.eye(2), atol=1e-7, rtol=1e-7)


def test_ea_default_regularization_is_invariant_to_common_positive_gain() -> None:
    x = _trials(np.array([[2.0, 0.4], [0.0, 1.5]]) * 1e-5, 80, 64)
    original = transform_euclidean_alignment(x, fit_euclidean_alignment(x, fit_scope="source_rows"))
    scaled = transform_euclidean_alignment(x * 1e6, fit_euclidean_alignment(x * 1e6, fit_scope="source_rows"))
    np.testing.assert_allclose(original, scaled, atol=1e-6, rtol=1e-6)

