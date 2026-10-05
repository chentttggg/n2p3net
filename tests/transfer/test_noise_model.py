import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel


def test_joint_noise_propagation_retains_cross_terms_and_reloads(tmp_path):
    rng = np.random.default_rng(18)
    residual = rng.normal(size=(2000, 2))
    state = (2 * residual[:, :1]) + rng.normal(size=(2000, 1)) * 0.1
    model = JointNoiseModel.fit(
        residual, state, fit_scope="source_training", scope_id="source-A",
        draw_ids=[f"draw-{i}" for i in range(len(residual))], shrinkage=0,
    )
    # Taking x0 - a/2 nearly cancels; omitting Omega predicts variance ~2.
    propagated = model.project(np.array([[1., 0.]]), np.array([[-0.5]]))
    assert propagated.covariance[0, 0] < 0.004
    assert np.linalg.eigvalsh(propagated.covariance).min() >= 0
    path = tmp_path / "noise.npz"
    model.save(path)
    reloaded = JointNoiseModel.load(path)
    np.testing.assert_array_equal(reloaded.covariance, model.covariance)
    np.testing.assert_array_equal(reloaded.project(np.eye(2), np.zeros((2, 1))).covariance,
                                  model.project(np.eye(2), np.zeros((2, 1))).covariance)
    with pytest.raises(ValueError, match="scope"):
        JointNoiseModel.fit(residual, state, fit_scope="test", scope_id="x",
                           draw_ids=[str(i) for i in range(len(residual))])


def test_constant_state_coordinates_do_not_change_active_shrinkage():
    rng = np.random.default_rng(21)
    raw = rng.normal(size=(30, 3))
    raw[:, 1] += raw[:, 0] * 2
    settings = dict(fit_scope="source_training", scope_id="S", draw_ids=list(map(str, range(30))))
    minimal = JointNoiseModel.fit(raw, np.empty((30, 0)), **settings)
    padded = JointNoiseModel.fit(raw, np.zeros((30, 4)), **settings)
    np.testing.assert_allclose(minimal.covariance, padded.covariance[:3, :3], atol=1e-14, rtol=0)
    assert np.count_nonzero(padded.covariance[3:]) == 0


def test_noise_fit_authority_is_deeply_frozen_and_cannot_be_resigned(tmp_path):
    raw = np.random.default_rng(5).normal(size=(10, 2))
    model = JointNoiseModel.fit(raw, np.empty((10, 0)), fit_scope="target_calibration",
                                scope_id="cal", draw_ids=list(map(str, range(10))))
    with pytest.raises(TypeError, match="frozen"):
        model.provenance["scope_id"] = "other"
    with pytest.raises((TypeError, AttributeError)):
        model.provenance["draw_ids"].clear()
    # A direct replacement bypasses Python's frozen dataclass, so consumers
    # must also verify the certificate rather than trust immutability alone.
    object.__setattr__(model, "provenance", {**model.provenance, "scope_id": "other"})
    with pytest.raises(ValueError, match="integrity"):
        model.project(np.eye(2), np.empty((2, 0)))
    with pytest.raises(ValueError, match="integrity"):
        model.save(tmp_path / "changed.npz")
