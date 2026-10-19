from __future__ import annotations

import warnings

import numpy as np
import pytest

from baselines.structured_erp import ERP_SCHEMA, StructuredERP


def _loaded_xdawn(*, filters, prototypes):
    record = {
        "schema": ERP_SCHEMA,
        "family": "xdawn_erp",
        "bin_width": 1,
        "baseline_mode": "none",
        "nfilter": 1,
        "input_shape": [1, np.asarray(prototypes).shape[1]],
        "feature_order": "channel_prime",
        "discarded_tail_samples": 0,
        "observed_rank": 1,
        "effective_nfilter": 1,
    }
    state = {
        "filters": np.asarray(filters, dtype=float),
        "prototypes": np.asarray(prototypes, dtype=float),
        "reference": np.eye(4),
        "coefficient": np.arange(10, dtype=float),
        "intercept": np.asarray(.125),
    }
    return StructuredERP.from_state(record, state)


@pytest.mark.parametrize(("filters", "prototypes"), [
    (np.zeros((2, 1)), np.zeros((2, 4))),
    (np.zeros((2, 1)), np.array([[1., 1., 1., 1.], [3., 3., 3., 3.]])),
    (np.array([[1.], [.5]]), np.array([[1.], [3.]])),
])
def test_xdawn_reader_rejects_states_with_no_positive_definite_input(filters, prototypes):
    with pytest.raises(ValueError, match="xDAWN.*no.*positive definite"):
        _loaded_xdawn(filters=filters, prototypes=prototypes)


def test_loaded_xdawn_rejects_zero_augmented_covariance_before_matrix_log():
    # A finite, SPD reference does not make a future trial covariance SPD.
    # With real filters, the second trial alone has zero time variation.
    model = _loaded_xdawn(filters=[[1.], [.5]], prototypes=np.zeros((2, 4)))
    X = np.array([[[1., 0., -1., 0.]], [[2., 2., 2., 2.]]])
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(ValueError, match="xDAWN.*positive definite"):
            model.predict_logit(X)
    assert not [warning for warning in observed if issubclass(warning.category, RuntimeWarning)]


def test_loaded_xdawn_zero_prototypes_and_collinear_filters_can_have_valid_covariance():
    # OAS supplies positive shrinkage from existing trial energy, even though
    # the raw spatial projection has rank one and the templates contain no ERP.
    model = _loaded_xdawn(filters=[[1.], [.5]], prototypes=np.zeros((2, 4)))
    X = np.array([[[1., 0., -1., 0.]], [[0., 2., 0., -2.]]])
    scores = model.predict_logit(X)
    assert np.isfinite(scores).all()
    record, state = model.export_state()
    np.testing.assert_array_equal(StructuredERP.from_state(record, state).predict_logit(X), scores)


def test_loaded_xdawn_empty_batch_needs_no_covariance():
    model = _loaded_xdawn(filters=[[1.], [.5]], prototypes=np.zeros((2, 4)))
    scores = model.predict_logit(np.empty((0, 1, 4)))
    assert scores.shape == (0,)
    assert scores.dtype == np.float64


def test_xdawn_zero_filters_with_time_varying_prototypes_have_nonempty_domain():
    model = _loaded_xdawn(
        filters=np.zeros((2, 1)),
        prototypes=[[1., 0., -1., 0.], [0., 1., 0., -1.]],
    )
    assert np.isfinite(model.predict_logit(np.zeros((1, 1, 4)))).all()
