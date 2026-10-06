import json

import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel, array_digest, metadata_digest
from transfer.operator_lda import ConditioningObservationSupport, OperatorAwareLDA


def calibration():
    rng = np.random.default_rng(61)
    y = np.arange(120) % 2
    raw = rng.normal(size=(120, 2)) + y[:, None]*np.array([.2, -.1])
    response = .7*raw[:, :1] + .4*y[:, None] + rng.normal(size=(120, 1))*.5
    residual_raw = raw - np.stack([raw[y == q].mean(axis=0) for q in (0, 1)])[y]
    residual_state = response - np.stack([response[y == q].mean(axis=0) for q in (0, 1)])[y]
    noise = JointNoiseModel.fit(residual_raw, residual_state, fit_scope="target_calibration", scope_id="C",
                              draw_ids=[f"C-{i}" for i in range(120)], shrinkage=0.)
    return raw, response, y, noise


def support():
    return ConditioningObservationSupport((0, 2), (2, 3), 3)


def fit():
    raw, response, y, noise = calibration()
    model = OperatorAwareLDA.fit(response, y, noise=noise, raw_operator=np.zeros((1, 2)), state_operator=np.eye(1),
        conditioning_features=raw, observation_support=support(), calibration_ids=[f"C-{i}" for i in range(120)],
        scope_id="C", input_identity="observed-continuation", ridge_relative=0.)
    return raw, response, y, noise, model


def test_conditional_operator_real_score_matches_dense_gaussian_candidate_mean_and_schur():
    raw, response, y, noise, model = fit()
    sx, cross, gamma = noise.covariance[:2, :2], noise.covariance[:2, 2:], noise.covariance[2:, 2:]
    gain = np.linalg.solve(sx, cross).T
    covariance = gamma-gain@cross
    mx = np.stack([raw[y == q].mean(axis=0) for q in (0, 1)])
    my = np.stack([response[y == q].mean(axis=0) for q in (0, 1)])
    expected = []
    for x, value in zip(raw[:8], response[:8], strict=True):
        candidate = my+(x-mx)@gain.T
        log_density = [-.5*float((value-m)@np.linalg.solve(covariance, value-m)) for m in candidate]
        expected.append(log_density[1]-log_density[0])
    np.testing.assert_allclose(model.predict_logit(response[:8], observed_raw=raw[:8], input_identity="observed-continuation"), expected, atol=2e-14)
    np.testing.assert_allclose(model.covariance, covariance, atol=2e-15)
    assert model.provenance["conditioning_program"] == "conditional_tail_with_fixed_class_prior"


def test_conditioner_is_required_and_deterministic_complete_raw_transform_cannot_claim_extra_observation():
    raw, response, y, noise, model = fit()
    with pytest.raises(ValueError, match="observed raw"):
        model.predict_logit(response, input_identity="observed-continuation")
    with pytest.raises(ValueError, match="outside|independent|external"):
        ConditioningObservationSupport((0, 3), (1, 3), 3)
    with pytest.raises(ValueError, match="available|support"):
        ConditioningObservationSupport((0, 2), (2, 4), 3)
    with pytest.raises(ValueError, match="support"):
        OperatorAwareLDA.fit(response, y, noise=noise, raw_operator=np.zeros((1, 2)), state_operator=np.eye(1),
            conditioning_features=raw, calibration_ids=[f"C-{i}" for i in range(120)], scope_id="C", input_identity="x")


def test_conditional_operator_replay_owns_actual_program_support_and_readonly_state(tmp_path):
    raw, response, _, _, model = fit()
    path = tmp_path/"model.npz"
    before = model.predict_logit(response, observed_raw=raw, input_identity="observed-continuation")
    model.save(path)
    loaded = OperatorAwareLDA.load(path)
    np.testing.assert_array_equal(loaded.predict_logit(response, observed_raw=raw, input_identity="observed-continuation"), before)
    assert loaded.observation_support == model.observation_support
    with pytest.raises(ValueError):
        loaded.conditioning_class_means.setflags(write=True)
    with pytest.raises(ValueError):
        loaded.conditioning_gain.setflags(write=True)


def test_overlap_response_range_stays_singular_and_null_class_contrast_is_rejected():
    raw, response, y, noise = calibration()
    A, B = np.array([[1., 0.], [-.3, .2]]), np.ones((2, 1))
    features = raw @ A.T+response @ B.T
    settings = dict(noise=noise, raw_operator=A, state_operator=B, conditioning_features=raw,
        observation_support=ConditioningObservationSupport((0, 2), (1, 3), 3),
        calibration_ids=[f"C-{i}" for i in range(120)], scope_id="C", input_identity="overlap", ridge_relative=0.)
    model = OperatorAwareLDA.fit(features, y, **settings)
    assert np.linalg.matrix_rank(model.identifiable_projector) == 1
    np.testing.assert_allclose(model.covariance, model.covariance[:1, :1]*np.ones((2, 2)), atol=2e-15)
    attacked = features.copy()
    attacked[y == 1] += np.array([.01, -.01])
    with pytest.raises(ValueError, match="contrast outside"):
        OperatorAwareLDA.fit(attacked, y, **settings)
    with pytest.raises(ValueError, match="identifiable support"):
        model.predict_logit(features[:1]+[.01, -.01], observed_raw=raw[:1], input_identity="overlap")


@pytest.mark.parametrize("changed", ["conditioning_features", "features"])
def test_conditional_fit_rejects_complex_arrays_before_fitting_any_means(changed):
    raw, response, y, noise = calibration()
    values = {"features": response, "conditioning_features": raw}
    values[changed] = values[changed].astype(complex)+1j
    with pytest.raises(ValueError, match="finite real"):
        OperatorAwareLDA.fit(values["features"], y, noise=noise, raw_operator=np.zeros((1, 2)), state_operator=np.eye(1),
            conditioning_features=values["conditioning_features"], observation_support=support(),
            calibration_ids=[f"C-{i}" for i in range(120)], scope_id="C", input_identity="observed-continuation")


@pytest.mark.parametrize("changed", ["observed_raw", "features"])
def test_conditional_predict_rejects_complex_arrays_before_scoring(changed):
    raw, response, _, _, model = fit()
    values = {"features": response[:2], "observed_raw": raw[:2]}
    values[changed] = values[changed].astype(complex)+1j
    with pytest.raises(ValueError, match="finite real"):
        model.predict_logit(values["features"], observed_raw=values["observed_raw"], input_identity="observed-continuation")


@pytest.mark.parametrize("changed", ["gain", "covariance", "producer", "downgrade"])
def test_saved_conditional_operator_cannot_be_resigned_to_another_numeric_or_producer_program(tmp_path, changed):
    *_, model = fit()
    path = tmp_path/"model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    if changed in {"gain", "covariance"}:
        key = "conditioning_gain" if changed == "gain" else "conditional_covariance"
        arrays[key] = arrays[key]+.1
        keys = ["class_means", "joint_mean", "joint_covariance", "raw_operator", "state_operator",
                "conditioning_class_means", "conditioning_gain", "conditional_covariance"]
        metadata["arrays_sha256"] = array_digest(*(arrays[k] for k in keys))
    elif changed == "producer":
        metadata["provenance"]["implementation_sha256"] = "f"*64
    else:
        metadata["schema"] = "n2p3_operator_aware_lda/1"
    metadata["artifact_sha256"] = metadata_digest({k: v for k, v in metadata.items() if k != "artifact_sha256"})
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    bad = tmp_path/"bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="conditional|producer|payload"):
        OperatorAwareLDA.load(bad)
