import numpy as np
import pytest
from scipy.signal import butter, sosfilt, sosfiltfilt

from data.finite_filter_operator import FiniteFilterOperator
from data.preprocess import design_continuous_sos, filter_bidirectional_segment


def _operator(padding):
    sos = design_continuous_sos(256., l_freq=.1, h_freq=30., order=4,
                              notch_freqs_hz=(50.,), notch_quality_factor=30.)
    return FiniteFilterOperator(tuple(map(tuple, sos)), padding)


@pytest.mark.parametrize("samples,padding", [(1, 0), (2, 2253), (19, 0), (19, 7), (257, 2253), (2700, 2253)])
def test_decomposition_reconstructs_production_and_independent_scipy(samples, padding):
    operator = _operator(padding)
    values = np.random.default_rng(samples+padding+11).normal(size=(samples, 3))
    sos = np.asarray(operator.sos)
    reference = filter_bidirectional_segment(sos, values.T, padding_limit=padding).data.T
    independently_executed = sosfiltfilt(sos, values, axis=0, padtype="odd", padlen=min(padding, samples-1))
    zero, basis, states = operator.decompose(values)
    np.testing.assert_allclose(operator.apply(values), reference, rtol=0, atol=0)
    np.testing.assert_allclose(reference, independently_executed, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(zero + basis @ states, reference, rtol=1e-9, atol=2e-9)
    assert basis.shape == (samples, 4*len(sos))
    assert states.shape == (4*len(sos), values.shape[1])


def test_zero_state_is_symmetric_positive_semidefinite_exact_causal_adjoint_product():
    sos = butter(3, [2., 25.], btype="bandpass", fs=256., output="sos")
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), 20)
    identity = np.eye(73)
    causal = sosfilt(sos, identity, axis=0)
    q = operator.zero_state(identity)
    np.testing.assert_allclose(q, causal.T @ causal, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(q, q.T, atol=2e-14)
    assert np.linalg.eigvalsh(q).min() > -1e-13
    x, y = np.random.default_rng(812).normal(size=(2, 73, 4))
    np.testing.assert_allclose(np.sum(x*operator.zero_state(y)),
                               np.sum(operator.zero_state(x)*y), atol=3e-13)


def test_dense_operator_and_same_operator_design_are_linear_without_extra_random_state():
    operator = _operator(25)
    rng = np.random.default_rng(907)
    design = rng.normal(size=(65, 7))
    kernel = rng.normal(size=(7, 3))
    matrix = operator.dense_matrix(65, byte_budget=65*65*8)
    np.testing.assert_allclose(matrix @ (design @ kernel), operator.apply(design @ kernel), atol=8e-12)
    np.testing.assert_allclose(operator.apply(design) @ kernel, operator.apply(design @ kernel), atol=8e-12)
    zero, basis, state = operator.decompose(design)
    np.testing.assert_allclose((zero+basis@state) @ kernel, operator.apply(design @ kernel), atol=8e-12)
    with pytest.raises(ValueError, match="budget"):
        operator.dense_matrix(66, byte_budget=65*65*8)
    assert operator.record()["state_role"] == "deterministic_function_of_observed_segment"


def test_finite_reflection_operator_need_not_be_symmetric_or_commute_with_crop():
    operator = _operator(90)
    matrix = operator.dense_matrix(101)
    assert np.linalg.norm(matrix-matrix.T) > .1
    values = np.random.default_rng(302).normal(size=(301, 2))
    cropped_after = operator.apply(values)[80:221]
    applied_after = operator.apply(values[80:221])
    assert np.linalg.norm(cropped_after-applied_after) > .1


def test_operator_rejects_unstable_sections_and_invalid_grids():
    with pytest.raises(ValueError, match="stable"):
        FiniteFilterOperator(((1., 0., 0., 1., -1.2, 0.),), 0)
    with pytest.raises(ValueError, match="normalized"):
        FiniteFilterOperator(((1., 0., 0., 2., -.4, 0.),), 0)
    with pytest.raises(ValueError, match="padding"):
        _operator(-1)
    operator = _operator(20)
    with pytest.raises(ValueError, match="finite"):
        operator.apply(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="positive"):
        operator.state_basis(0)
    with pytest.raises(ValueError, match="budget"):
        operator.dense_matrix(100000)
