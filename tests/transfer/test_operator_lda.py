import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel
from transfer.operator_lda import OperatorAwareLDA


def test_operator_lda_uses_estimated_joint_covariance_and_frozen_replay(tmp_path):
    rng = np.random.default_rng(32)
    raw = rng.normal(size=(500, 2))
    state = 2 * raw[:, :1] + rng.normal(0, .1, (500, 1))
    noise = JointNoiseModel.fit(raw, state, fit_scope="source_training", scope_id="source-A",
                                draw_ids=[str(i) for i in range(500)], shrinkage=0)
    A, B = np.eye(2), np.array([[-.5], [0.]])
    y = np.arange(100) % 2
    X = np.column_stack([rng.normal(0, .05, 100), rng.normal(0, 1, 100)])
    X += (y - .5)[:, None]
    model = OperatorAwareLDA.fit(X, y, noise=noise, raw_operator=A, state_operator=B,
                                 calibration_ids=[f"cal-{i}" for i in range(100)],
                                 scope_id="calibration-A", input_identity="observed-filter-v1")
    # Correct Omega makes the first coordinate far more reliable than the second.
    scores = model.predict_logit(np.array([[.5, -3.], [-.5, 3.]]), input_identity="observed-filter-v1")
    assert scores[0] > 0 > scores[1]
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = OperatorAwareLDA.load(path)
    np.testing.assert_array_equal(loaded.predict_logit(X, input_identity="observed-filter-v1"),
                                  model.predict_logit(X, input_identity="observed-filter-v1"))
    with pytest.raises(ValueError, match="identity"):
        model.predict_logit(X, input_identity="other")


def test_operator_lda_does_not_invent_uncertainty_in_unobserved_directions():
    rng = np.random.default_rng(9)
    raw = np.column_stack([rng.normal(size=100), np.zeros(100)])
    noise = JointNoiseModel.fit(raw[:20], np.empty((20, 0)), fit_scope="target_calibration",
                                scope_id="C", draw_ids=[str(i) for i in range(20)])
    wider = JointNoiseModel.fit(raw, np.empty((100, 0)), fit_scope="target_calibration",
                                scope_id="C", draw_ids=[str(i) for i in range(100)])
    y = np.arange(20) % 2
    X = np.column_stack([rng.normal(size=20) + y, np.zeros(20)])
    model = OperatorAwareLDA.fit(X, y, noise=noise, raw_operator=np.eye(2),
                                 state_operator=np.empty((2, 0)), calibration_ids=list(map(str, range(20))),
                                 scope_id="C", input_identity="observed")
    assert model.covariance[1, 1] == 0
    with pytest.raises(ValueError, match="identifiable support"):
        model.predict_logit(np.array([[1., 1.]]), input_identity="observed")
    # Noise estimated from 80 rows outside the classifier's authorized calibration IDs.
    with pytest.raises(ValueError, match="noise draws exceed"):
        OperatorAwareLDA.fit(X, y, noise=wider, raw_operator=np.eye(2),
                             state_operator=np.empty((2, 0)), calibration_ids=list(map(str, range(20))),
                             scope_id="C", input_identity="observed")
    with pytest.raises(ValueError, match="exceeds"):
        OperatorAwareLDA.fit(X, y, noise=noise, raw_operator=np.eye(2),
                             state_operator=np.empty((2, 0)), calibration_ids=list(map(str, range(20))),
                             scope_id="other-person", input_identity="observed")


def test_operator_lda_preserves_nested_noise_authority_and_detects_derived_state_mutation(tmp_path):
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(20, 2))
    noise = JointNoiseModel.fit(raw, np.empty((20, 0)), fit_scope="source_training",
                                scope_id="S", draw_ids=list(map(str, range(20))))
    model = OperatorAwareLDA.fit(raw, np.arange(20) % 2, noise=noise, raw_operator=np.eye(2),
                                 state_operator=np.empty((2, 0)), calibration_ids=list(map(str, range(20))),
                                 scope_id="C", input_identity="features")
    with pytest.raises(TypeError, match="frozen"):
        model.provenance["noise_estimation_scope"]["fit_scope"] = "target_calibration"
    object.__setattr__(model, "weights", model.weights * 2)
    with pytest.raises(ValueError, match="integrity"):
        model.predict_logit(raw, input_identity="features")
    with pytest.raises(ValueError, match="integrity"):
        model.save(tmp_path / "changed.npz")


def test_operator_support_admission_is_per_sample_under_batching(tmp_path):
    rng = np.random.default_rng(231)
    raw = np.column_stack((rng.normal(size=32), np.zeros(32)))
    noise = JointNoiseModel.fit(raw, np.empty((32, 0)), fit_scope="source_training",
        scope_id="support-source", draw_ids=[f"source-{i}" for i in range(32)])
    labels = np.arange(12) % 2
    values = np.column_stack((rng.normal(size=12)+labels, np.zeros(12)))
    model = OperatorAwareLDA.fit(values, labels, noise=noise, raw_operator=np.eye(2),
        state_operator=np.empty((2, 0)), calibration_ids=[f"cal-{i}" for i in range(12)],
        scope_id="support-C", input_identity="support-features")
    center = model.class_means.mean(axis=0)
    bad = center+np.array([1., 1.])
    with pytest.raises(ValueError, match="identifiable support"):
        model.predict_logit(bad[None], input_identity="support-features")
    for magnitude in (1e8, 1e100):
        good = center+np.array([magnitude, 0.])
        for batch in (np.stack((bad, good)), np.stack((good, bad))):
            with pytest.raises(ValueError, match="identifiable support"):
                model.predict_logit(batch, input_identity="support-features")
    path = tmp_path/"row-support.npz"
    model.save(path)
    restored = OperatorAwareLDA.load(path)
    with pytest.raises(ValueError, match="identifiable support"):
        restored.predict_logit(np.stack((center+np.array([1e8, 0.]), bad)),
            input_identity="support-features")
    valid = np.stack((center+np.array([.3, 0.]), center+np.array([2., 0.])))
    np.testing.assert_array_equal(model.predict_logit(valid, input_identity="support-features"),
        np.concatenate([model.predict_logit(row[None], input_identity="support-features") for row in valid]))


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_operator_support_preserves_relative_geometry_at_finite_scales(magnitude):
    rng = np.random.default_rng(231)
    raw = np.column_stack((rng.normal(size=32), np.zeros(32)))
    noise = JointNoiseModel.fit(raw, np.empty((32, 0)), fit_scope="source_training",
        scope_id="support-source", draw_ids=[f"source-{i}" for i in range(32)])
    labels = np.arange(12) % 2
    values = np.column_stack((rng.normal(size=12)+labels, np.zeros(12)))
    model = OperatorAwareLDA.fit(values, labels, noise=noise, raw_operator=np.eye(2),
        state_operator=np.empty((2, 0)), calibration_ids=[f"cal-{i}" for i in range(12)],
        scope_id="support-C", input_identity="support-features")
    center = model.class_means.mean(axis=0)
    with pytest.raises(ValueError, match="identifiable support"):
        model.predict_logit((center+np.array([magnitude, magnitude]))[None],
            input_identity="support-features")
    valid = center+np.array([magnitude, 0.])
    assert np.isfinite(model.predict_logit(valid[None], input_identity="support-features")).all()
