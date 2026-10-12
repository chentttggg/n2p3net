from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.domain_alignment import (
    AcwConfig,
    AcwState,
    AjdCongruenceWhitening,
    apply_feature_lift,
    feature_lift_rank,
    flatten_feature_epochs,
)


def _domain_features(seed: int, covariance: np.ndarray, n_rows: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_rows, covariance.shape[0]))
    return raw @ np.linalg.cholesky(covariance).T


def test_apply_feature_lift_preserves_declared_shape_and_temporal_basis() -> None:
    epochs = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    spatial = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    temporal = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    lifted = apply_feature_lift(epochs, spatial, temporal_basis=temporal)

    assert lifted.shape == (2, 2, 2)
    np.testing.assert_array_equal(lifted, np.einsum("pc,nct->npt", spatial, epochs) @ temporal)
    assert feature_lift_rank(spatial) == 2
    np.testing.assert_array_equal(flatten_feature_epochs(lifted), lifted.reshape(2, -1))


def test_galia_aligns_shared_eigenbasis_and_round_trips_state() -> None:
    rng = np.random.default_rng(11)
    mixing = rng.normal(size=(3, 3))
    diagonal = (np.diag([1.0, 3.0, 8.0]), np.diag([2.0, 5.0, 11.0]))
    inverse = np.linalg.inv(mixing)
    covariances = tuple(inverse @ values @ inverse.T for values in diagonal)
    domains = {
        "source": _domain_features(1, covariances[0]),
        "target": _domain_features(2, covariances[1]),
    }

    state = AjdCongruenceWhitening(AcwConfig(regularization=1e-8)).fit(domains)
    transformed_covariances = []
    for index, name in enumerate(state.domain_order):
        transformed = state.transform(domains[name], name)
        transformed_covariances.append(np.cov(transformed, rowvar=False, bias=True))
        assert state.diagonal_scales[index].min() > 0.0
    assert all(np.isfinite(value).all() for value in transformed_covariances)
    assert 0.0 <= state.ajd_offdiag_residual < 0.1

    restored = AcwState.from_record(state.record())
    assert restored.state_digest == state.state_digest
    np.testing.assert_array_equal(
        restored.transform(domains["source"], "source"),
        state.transform(domains["source"], "source"),
    )


def test_galia_requires_equal_dimensions_and_rejects_bad_domain_weight_vocabulary() -> None:
    first = np.zeros((4, 3), dtype=np.float64)
    second = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="equal feature dimensions"):
        AjdCongruenceWhitening().fit({"a": first, "b": second})

    with pytest.raises(ValueError, match="exactly the frozen domain vocabulary"):
        AjdCongruenceWhitening().fit({"a": first, "b": first.copy()}, weights={"a": 1.0})


@pytest.mark.parametrize("method", ["rjd", "uwedge"])
def test_galia_rejects_explicit_weights_for_solvers_that_ignore_them(method) -> None:
    rng = np.random.default_rng(12)
    domains = {"a": rng.normal(size=(20, 3)), "b": rng.normal(size=(20, 3))}
    with pytest.raises(ValueError, match="require method='ajd_pham'"):
        AjdCongruenceWhitening(AcwConfig(method=method)).fit(
            domains,
            weights={"a": 0.25, "b": 0.75},
        )


def test_galia_fast_alignment_regularizes_source_weighted_mixture() -> None:
    # One rank-deficient source covariance would make the source-weighted
    # mixture condition explode without explicit regularization.
    first = np.column_stack((np.arange(8.0), np.zeros(8), np.ones(8)))
    second = np.column_stack((np.arange(8.0), np.ones(8), np.zeros(8)))
    state = AjdCongruenceWhitening(AcwConfig(regularization=1e-3)).fit(
        {"a": first, "b": second},
        weights={"a": 0.5, "b": 0.5},
    )
    fast = state.fast_alignment(
        second,
        domain_id="target",
        fit_scope="source_weighted",
    )
    assert np.isfinite(fast.covariance_condition_number)
    assert fast.covariance_condition_number < 1e6


def test_galia_fast_alignment_rejects_test_scope_and_detects_parent_state_mismatch() -> None:
    rng = np.random.default_rng(5)
    domains = {"a": rng.normal(size=(20, 2)), "b": rng.normal(size=(20, 2))}
    state = AjdCongruenceWhitening().fit(domains)

    with pytest.raises(ValueError, match="target test is forbidden"):
        state.fast_alignment(domains["a"], fit_scope="target_test")

    fast = state.fast_alignment(rng.normal(size=(20, 2)), fit_scope="target_calibration")
    changed = replace(fast, parent_state_digest="0" * 64)
    with pytest.raises(ValueError, match="different ACW state"):
        state.transform_fast(domains["a"], changed)


def test_galia_state_record_tampering_is_rejected() -> None:
    rng = np.random.default_rng(9)
    state = AjdCongruenceWhitening().fit(
        {"a": rng.normal(size=(10, 2)), "b": rng.normal(size=(10, 2))}
    )
    record = state.record()
    record["diagonal_scales"][0][0] *= 2.0
    with pytest.raises(ValueError, match="digest"):
        AcwState.from_record(record)
