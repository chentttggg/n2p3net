import json

import numpy as np
import pytest

from transfer.noise_model import JointNoiseModel, array_digest, metadata_digest


def fitted(raw, state):
    return JointNoiseModel.fit(raw, state, fit_scope="target_calibration", scope_id="C",
                              draw_ids=[f"C-{i}" for i in range(len(raw))], shrinkage=0.)


def paired():
    h = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    return h[:, :1] + 7., .8*h[:, :1] + .6*h[:, 1:] + 3.


def test_condition_uses_full_candidate_means_and_schur_not_unconditional_state_mean():
    raw, state = paired()
    model = fitted(raw, state)
    result = model.condition(np.array([[.2]]), raw_means=np.array([[.4], [-.4]]),
                             state_means=np.array([[.3], [-.8]]))
    np.testing.assert_allclose(result.mean, [[[.14], [-.32]]], atol=2e-15)
    np.testing.assert_allclose(result.covariance, [[.36]], atol=2e-15)
    assert result.rank_ledger["raw_rank"] == 1
    # Adding model.mean here would count a noise mean already in full expectations.
    assert result.mean[0, 0, 0] != pytest.approx(.14 + model.mean[-1])
    with pytest.raises(ValueError):
        result.mean.setflags(write=True)


def test_singular_conditioning_solves_declared_support_and_rejects_all_candidate_null_contrasts():
    raw, state = paired()
    centered = raw-7
    model = fitted(np.column_stack((centered[:, 0], 2*centered[:, 0])), state-3)
    result = model.condition([[.2, .4]], raw_means=[[0., 0.]], state_means=[[0.]])
    np.testing.assert_allclose(result.mean, [[[.16]]], atol=2e-15)
    np.testing.assert_allclose(result.covariance, [[.36]], atol=2e-15)
    assert result.rank_ledger["raw_rank"] == 1
    with pytest.raises(ValueError, match="conditioning support"):
        model.condition([[.2, .5]], raw_means=[[0., 0.]], state_means=[[0.]])
    with pytest.raises(ValueError, match="conditioning support"):
        model.condition([[.2, .4]], raw_means=[[0., 0.], [0., .1]], state_means=[[0.], [0.]])


def test_deterministic_raw_state_relation_has_zero_schur_and_no_fabricated_variance():
    raw, _ = paired()
    model = fitted(raw, 2*raw)
    result = model.condition([[8.]], raw_means=[[7.]], state_means=[[14.]])
    np.testing.assert_allclose(result.mean, [[[16.]]], atol=2e-14)
    np.testing.assert_array_equal(result.covariance, [[0.]])
    assert result.rank_ledger["conditional_state_rank"] == 0


def test_constant_conditioner_is_dirac_and_resolvable_independent_small_voltage_direction_is_retained():
    _, state = paired()
    constant = fitted(np.zeros((4, 2)), state)
    result = constant.condition([[0., 0.]], raw_means=[[0., 0.]], state_means=[[3.]])
    np.testing.assert_allclose(result.covariance, [[1.]], atol=2e-15)
    assert result.rank_ledger["raw_rank"] == 0
    with pytest.raises(ValueError, match="conditioning support"):
        constant.condition([[0., 1e-12]], raw_means=[[0., 0.]], state_means=[[3.]])
    h = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    weak = fitted(h*np.array([1., 1e-10]), h[:, :1])
    assert weak.condition([[0., 0.]], raw_means=[[0., 0.]], state_means=[[0.]]).rank_ledger["raw_rank"] == 2


@pytest.mark.parametrize("changed", ["raw_values", "raw_means", "state_means"])
def test_condition_rejects_complex_observations_or_expectations_before_lossy_cast(changed):
    model = fitted(*paired())
    values = {"raw_values": np.array([[8.]]), "raw_means": np.array([[7.]]), "state_means": np.array([[3.]])}
    values[changed] = values[changed].astype(complex)+1j
    with pytest.raises(ValueError, match="finite real"):
        model.condition(values["raw_values"], raw_means=values["raw_means"], state_means=values["state_means"])


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_physical_unit_changes_conditioned_means_and_covariance_consistently(scale):
    raw, state = paired()
    first, changed = fitted(raw, state), fitted(scale*raw, scale*state)
    a = first.condition([[8.]], raw_means=[[7.]], state_means=[[3.]])
    b = changed.condition([[scale*8]], raw_means=[[scale*7]], state_means=[[scale*3]])
    np.testing.assert_allclose(b.mean/scale, a.mean, rtol=3e-14)
    np.testing.assert_allclose(b.covariance/scale**2, a.covariance, rtol=3e-14)


def test_conditional_numerical_state_and_producer_are_saved_verified_and_readonly(tmp_path):
    raw, state = paired()
    model = fitted(raw, state)
    path = tmp_path/"noise.npz"
    model.save(path)
    reloaded = JointNoiseModel.load(path)
    a = model.condition([[8.]], raw_means=[[7.]], state_means=[[3.]])
    b = reloaded.condition([[8.]], raw_means=[[7.]], state_means=[[3.]])
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.covariance, b.covariance)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    assert "conditional_state_covariance" in arrays
    meta = json.loads(str(arrays["metadata"]))
    meta["provenance"]["implementation_sha256"] = "0"*64
    meta["artifact_sha256"] = metadata_digest({k: v for k, v in meta.items() if k != "artifact_sha256"})
    arrays["metadata"] = np.asarray(json.dumps(meta))
    bad = tmp_path/"wrong_producer.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="producer|implementation"):
        JointNoiseModel.load(bad)
    # An artifact checksum cannot justify changing only the actual Schur payload.
    arrays["metadata"] = np.asarray(json.dumps({**model._metadata(), "artifact_sha256": metadata_digest(model._metadata())}))
    arrays["conditional_state_covariance"] = arrays["conditional_state_covariance"] + 1.
    np.savez_compressed(tmp_path/"wrong_schur.npz", **arrays)
    with pytest.raises(ValueError, match="conditional|state"):
        JointNoiseModel.load(tmp_path/"wrong_schur.npz")
    assert array_digest(model.mean, model.covariance) == model._metadata()["state_sha256"]
