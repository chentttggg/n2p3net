import json

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from transfer.temporal_noise import TemporalNoise


@pytest.mark.parametrize("coefficients", [[], [.7], [1.6, -.81]])
def test_whitening_matches_independent_stationary_dense_covariance(coefficients):
    model = TemporalNoise.from_parameters(coefficients, innovation_variance=2.4,
                                         reference_id="independent_impulse_covariance")
    impulse = np.zeros(20000)
    impulse[0] = 1
    response = lfilter([1.], np.r_[1., -np.asarray(coefficients)], impulse)
    covariance = toeplitz([response[:len(response)-lag] @ response[lag:] for lag in range(75)])
    whiten = model.whiten(np.eye(75))
    np.testing.assert_allclose(whiten @ covariance @ whiten.T, np.eye(75), atol=2e-12)
    x = np.random.default_rng(342).normal(size=(75, 3))
    np.testing.assert_allclose(model.whiten(x).T @ model.whiten(x),
                               x.T @ np.linalg.solve(covariance, x), atol=3e-11)
    # Unit innovation convention: the scalar fit estimate never divides data again.
    other_scale = TemporalNoise.from_parameters(coefficients, innovation_variance=19., reference_id="same_AR")
    np.testing.assert_array_equal(model.whiten(x), other_scale.whiten(x))


def test_burg_fits_independent_records_without_join_lags():
    rng = np.random.default_rng(4895)
    records = [lfilter([1.], [1., -1.6, .81], rng.normal(size=(5000, 2)), axis=0)[300:] for _ in range(3)]
    model = TemporalNoise.fit(records, order=2, record_ids=["a", "b", "c"], fit_scope="target_calibration")
    np.testing.assert_allclose(model.coefficients, [1.6, -.81], atol=.02)
    assert model.pole_radius < 1
    assert abs(model.innovation_variance - 1) < .03
    assert model.provenance["available_lag_rows"] == sum(len(x)-2 for x in records)
    permuted = TemporalNoise.fit(records[::-1], order=2, record_ids=["c", "b", "a"], fit_scope="target_calibration")
    np.testing.assert_allclose(model.coefficients, permuted.coefficients, atol=2e-15)
    # Exact AR1 aggregate independently computed per physical record, including discontinuous edges.
    records = [rng.normal(size=(120, 1)) for _ in range(4)]
    for i, record in enumerate(records):
        record[[0, -1]] = [[200 * (-1)**i], [200 * (-1)**i]]
    expected = 2 * sum(np.sum(x[1:] * x[:-1]) for x in records) / sum(
        np.sum(x[1:]**2 + x[:-1]**2) for x in records)
    model = TemporalNoise.fit(records, order=1, record_ids=["a", "b", "c", "d"], fit_scope="source_training")
    np.testing.assert_allclose(model.coefficients, [expected], atol=1e-15)
    incorrectly_joined = TemporalNoise.fit([np.vstack(records)], order=1, record_ids=["fake_join"], fit_scope="source_training")
    assert abs(model.coefficients[0] - incorrectly_joined.coefficients[0]) > .2


def test_numeric_replay_freezes_state_and_provenance(tmp_path):
    data = np.random.default_rng(88).normal(size=(1200, 2))
    model = TemporalNoise.fit([data], order=8, record_ids=["C-record"], fit_scope="target_calibration")
    path = tmp_path / "noise.npz"
    model.save(path)
    loaded = TemporalNoise.load(path)
    probe = np.random.default_rng(64).normal(size=(300, 4))
    np.testing.assert_array_equal(model.whiten(probe), loaded.whiten(probe))
    embedded = TemporalNoise.from_record(json.loads(json.dumps(model.record())))
    np.testing.assert_array_equal(model.whiten(probe), embedded.whiten(probe))
    assert data.flags.writeable
    with pytest.raises(ValueError):
        model.coefficients.setflags(write=True)
    with pytest.raises(TypeError):
        model.provenance["records"][0]["record_id"] = "altered"
    changed = json.loads(json.dumps(model.record()))
    changed["innovation_variance"] *= 2
    with pytest.raises(ValueError, match="digest"):
        TemporalNoise.from_record(changed)


def test_fail_closed_for_nonidentifiable_or_unstable_noise():
    with pytest.raises(ValueError, match="stabl"):
        TemporalNoise.from_parameters([1.1], innovation_variance=1, reference_id="unstable")
    with pytest.raises(ValueError, match="degenerate|stabl"):
        TemporalNoise.fit([np.ones((300, 2))], order=1, record_ids=["flat"], fit_scope="target_calibration")
    with pytest.raises(ValueError, match="lag rows|sample"):
        TemporalNoise.fit([np.random.default_rng(1).normal(size=(100, 2))], order=32,
                          record_ids=["short"], fit_scope="target_calibration")
    with pytest.raises(ValueError, match="scope"):
        TemporalNoise.fit([np.ones((300, 2))], order=1, record_ids=["T"], fit_scope="target_test")
    model = TemporalNoise.from_parameters([.8], innovation_variance=1, reference_id="valid")
    with pytest.raises(ValueError, match="finite"):
        model.whiten(np.array([[1.], [np.nan]]))
    with pytest.raises(ValueError, match="samples"):
        TemporalNoise.from_parameters([.8, -.2], innovation_variance=1, reference_id="valid").whiten(np.ones((1, 2)))


def test_forced_frozen_state_change_rejected_and_resigned_empty_ledger_rejected():
    from transfer.noise_model import metadata_digest

    model = TemporalNoise.fit([np.random.default_rng(66).normal(size=(100, 1))], order=1,
                              record_ids=["C1"], fit_scope="target_calibration")
    record = json.loads(json.dumps(model.record()))
    record["provenance"]["records"] = []
    record.pop("artifact_sha256")
    record["artifact_sha256"] = metadata_digest(record)
    with pytest.raises(ValueError, match="ledger|records"):
        TemporalNoise.from_record(record)
    object.__setattr__(model, "innovation_variance", 200.)
    with pytest.raises(ValueError, match="integrity"):
        model.whiten(np.ones((10, 1)))


def test_high_order_keeps_startup_shape_and_is_invariant_to_global_units():
    rng = np.random.default_rng(84)
    values = [lfilter([1.], [1., -1.1, .4], rng.normal(size=(2500, 3)), axis=0)[200:] for _ in range(2)]
    settings = dict(order=32, record_ids=["one", "two"], fit_scope="source_training")
    model = TemporalNoise.fit(values, **settings)
    volts = TemporalNoise.fit([x*1e-6 for x in values], **settings)
    np.testing.assert_allclose(model.coefficients, volts.coefficients, atol=3e-14)
    assert model.startup_covariance.shape == (32, 32)
    assert np.all(np.abs(model.reflection_coefficients) < 1)
    np.testing.assert_allclose(model.innovation_variance*1e-12, volts.innovation_variance, rtol=1e-13)
    probe = rng.normal(size=(300, 8))
    np.testing.assert_allclose(model.whiten(probe)*1e-6, volts.whiten(probe*1e-6), atol=3e-19)


def test_overwriting_private_startup_factor_cannot_silently_change_density():
    model = TemporalNoise.from_parameters([.8], innovation_variance=1, reference_id="counterexample")
    object.__setattr__(model, "_startup_root", np.ones((1, 1)))
    with pytest.raises(ValueError, match="integrity"):
        model.whiten(np.ones((10, 1)))


@pytest.mark.parametrize("coefficients", [[.7], [1.6, -.81]])
def test_conditional_prefix_matches_dense_conditional_gaussian(coefficients):
    p, n = len(coefficients), 65
    conditional = TemporalNoise.from_parameters(coefficients, innovation_variance=2., reference_id="conditional_reference",
                                                startup_mode="conditional_prefix")
    stationary = TemporalNoise.from_parameters(coefficients, innovation_variance=2., reference_id="stationary_reference")
    impulse = np.zeros(20000)
    impulse[0] = 1
    h = lfilter([1.], np.r_[1., -np.asarray(coefficients)], impulse)
    covariance = toeplitz([h[:len(h)-lag]@h[lag:] for lag in range(n)])
    regression = np.linalg.solve(covariance[:p, :p], covariance[:p, p:]).T
    conditioned_covariance = covariance[p:, p:]-regression@covariance[:p, p:]
    observations = np.random.default_rng(724).normal(size=(n, 4))
    conditioned = observations[p:]-regression@observations[:p]
    innovation = conditional.whiten(observations)
    assert innovation.shape == (n-p, 4)
    np.testing.assert_allclose(innovation.T@innovation, conditioned.T@np.linalg.solve(conditioned_covariance, conditioned), atol=2e-10)
    np.testing.assert_allclose(innovation, stationary.whiten(observations)[p:], atol=0, rtol=0)
    np.testing.assert_allclose(conditional.marginal_variance_per_unit_innovation,
                               stationary.marginal_variance_per_unit_innovation, rtol=2e-13)


def test_conditional_mode_does_not_require_ill_conditioned_stationary_matrix_but_requires_stability(tmp_path):
    coefficients = -np.poly([.99]*4)[1:]
    with pytest.raises(ValueError, match="numerically unidentifiable"):
        TemporalNoise.from_parameters(coefficients, innovation_variance=1., reference_id="ill_conditioned_stationary")
    model = TemporalNoise.from_parameters(coefficients, innovation_variance=1., reference_id="known_prefix",
                                          startup_mode="conditional_prefix")
    assert model.startup_covariance.shape == (0, 0) and model.startup_condition is None
    assert model.marginal_variance_per_unit_innovation > 1e10
    x = np.random.default_rng(67).normal(size=(100, 2))
    path = tmp_path/"conditional.npz"
    model.save(path)
    loaded = TemporalNoise.load(path)
    np.testing.assert_array_equal(model.whiten(x), loaded.whiten(x))
    assert loaded.record()["schema"] == "n2p3_temporal_noise/5"
    assert loaded.record()["startup_mode"] == "conditional_prefix"
    assert loaded.record()["conditioning_prefix_samples"] == 4
    with pytest.raises(ValueError, match="stabl"):
        TemporalNoise.from_parameters([1.01], innovation_variance=1., reference_id="unstable_prefix",
                                      startup_mode="conditional_prefix")
    with pytest.raises(ValueError, match="samples"):
        model.whiten(x[:4])


def test_conditional_fit_records_actual_full_inputs_without_changing_stationary_schema():
    data = np.random.default_rng(264).normal(size=(1200, 3))
    settings = dict(order=8, record_ids=["C"], fit_scope="target_calibration")
    marginal = TemporalNoise.fit([data], **settings)
    conditional = TemporalNoise.fit([data], **settings, startup_mode="conditional_prefix")
    np.testing.assert_array_equal(marginal.coefficients, conditional.coefficients)
    np.testing.assert_array_equal(marginal.whiten(data)[8:], conditional.whiten(data))
    assert marginal.provenance == conditional.provenance
    assert marginal.record()["schema"] == "n2p3_temporal_noise/5"
    assert marginal.record()["startup_mode"] == "stationary"
    assert conditional.provenance["records"][0]["n_samples"] == 1200
    with pytest.raises(ValueError, match="startup"):
        TemporalNoise.fit([data], **settings, startup_mode="drop_bad_prefix")


def test_marginal_scale_from_reflections_matches_lyapunov_for_high_order():
    data = lfilter([1.], [1., -1.5, .75], np.random.default_rng(610).normal(size=(8000, 2)), axis=0)[200:]
    model = TemporalNoise.fit([data], order=32, record_ids=["calibration"], fit_scope="target_calibration")
    conditional = TemporalNoise.from_parameters(model.coefficients, innovation_variance=model.innovation_variance,
                                                reference_id="scalar_scale_reference", startup_mode="conditional_prefix")
    np.testing.assert_allclose(conditional.marginal_variance_per_unit_innovation,
                               model.marginal_variance_per_unit_innovation, rtol=2e-11)


@pytest.mark.parametrize("coefficients", [[], [.7], [1.6, -.81]])
def test_stationary_lattice_matches_independent_dense_covariance_and_preserves_prefix(coefficients):
    n = 70
    model = TemporalNoise.from_parameters(coefficients, innovation_variance=2., reference_id="dense_lattice",
                                          startup_mode="stationary_innovations")
    impulse = np.zeros(20000)
    impulse[0] = 1
    h = lfilter([1.], np.r_[1., -np.asarray(coefficients)], impulse)
    covariance = toeplitz([h[:len(h)-lag]@h[lag:] for lag in range(n)])
    operator = model.whiten(np.eye(n))
    np.testing.assert_allclose(operator@covariance@operator.T, np.eye(n), atol=3e-12)
    if model.order:
        np.testing.assert_allclose(operator[:model.order, :model.order], model.startup_whitening_factor, atol=3e-15)
    old = TemporalNoise.from_parameters(coefficients, innovation_variance=2., reference_id="same_AR")
    np.testing.assert_allclose(operator, old.whiten(np.eye(n)), atol=3e-14)
    assert operator.shape == (n, n)
    assert model.record()["schema"] == "n2p3_temporal_noise/5"


def test_actual_burg_reflections_are_saved_without_unstable_inverse_stepdown(monkeypatch, tmp_path):
    import transfer.temporal_noise as module

    def forbidden(*args):
        raise AssertionError("the fitted Burg lattice must not invert its expanded polynomial")

    monkeypatch.setattr(module, "_reflections", forbidden)
    data = lfilter([1.], [1., -1.5, .75], np.random.default_rng(55).normal(size=(2000, 2)), axis=0)[100:]
    model = TemporalNoise.fit([data], order=8, record_ids=["C-only"], fit_scope="target_calibration",
                              startup_mode="stationary_innovations")
    assert model.record()["reflection_origin"] == "actual_Burg_recursion"
    assert model.provenance["records"][0]["n_samples"] == len(data)
    path = tmp_path/"lattice.npz"
    model.save(path)
    loaded = TemporalNoise.load(path)
    np.testing.assert_array_equal(loaded.reflection_coefficients, model.reflection_coefficients)
    np.testing.assert_array_equal(loaded.whiten(data), model.whiten(data))
    np.testing.assert_allclose(model.innovation_variance, np.mean(model.whiten(data)[8:]**2), rtol=3e-15)


def test_stable_parcor_does_not_trust_false_expanded_polynomial_roots(monkeypatch):
    def forbidden(*args):
        raise AssertionError("lattice must not construct stationary covariance")

    monkeypatch.setattr("scipy.linalg.solve_discrete_lyapunov", forbidden)
    # Independently obtained at 100 decimal digits from eight repeated .99 poles.
    parcor = np.array([-.9999961149865101, .9999901108404076, -.9999801038304161,
                       .999961519006537, -.9999206301468053, .9997979222646003,
                       -.9989401968753808, .9227446944279201])
    model = TemporalNoise.from_reflections(parcor, innovation_variance=1., reference_id="100digit_repeated_poles")
    assert model.pole_radius > 1  # Expanded float polynomial diagnosis is numerically misleading.
    assert np.all(np.abs(model.reflection_coefficients) < 1)
    assert model.startup_covariance.shape == (0, 0)
    assert model.startup_factor_condition > 1e15
    assert model.record()["floating_precision_condition_indicator"] > 1
    assert model.record()["stability_certificate"] == "actual_PARCOR_strictly_inside_unit_interval"
    assert np.isfinite(model.whiten(np.random.default_rng(56).normal(size=(100, 2)))).all()
    with pytest.raises(ValueError, match="stable"):
        TemporalNoise.from_reflections([.2, 1.], innovation_variance=1., reference_id="not_stable")


def test_lattice_serialized_sequence_and_factor_integrity_are_checked():
    from transfer.noise_model import metadata_digest

    model = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="integrity")
    record = json.loads(json.dumps(model.record()))
    record["reflection_coefficients"][0] -= .03
    record.pop("artifact_sha256")
    record["artifact_sha256"] = metadata_digest(record)
    with pytest.raises(ValueError, match="disagree"):
        TemporalNoise.from_record(record)
    object.__setattr__(model, "startup_whitening_factor", np.eye(2))
    with pytest.raises(ValueError, match="integrity"):
        model.whiten(np.ones((10, 1)))
