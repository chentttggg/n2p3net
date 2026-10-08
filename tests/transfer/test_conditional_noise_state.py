import copy
import json

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from transfer.noise_model import array_digest, metadata_digest
from transfer.temporal_noise import TemporalNoise


@pytest.fixture
def training():
    rng = np.random.default_rng(21405)
    records = tuple(lfilter([1.], [1., -.6, .1], rng.normal(size=(n, 3)), axis=0)
                    for n in (400, 520))
    initializer = TemporalNoise.fit(
        records, order=2, record_ids=("authorized-C-1", "authorized-C-2"),
        fit_scope="target_calibration", startup_mode="stationary_innovations",
    )
    return records, initializer


def optimization(strategy="conditional_profile", **changes):
    result = dict(strategy=strategy, initial_objective=50., final_objective=45.,
                  converged=True, iterations=3, evaluations=17, eta_max=5.,
                  status=0, message="CONVERGENCE")
    if strategy == "burg_only":
        result.update(final_objective=50., iterations=0, evaluations=1,
                      message="retained actual segmented Burg shape")
    result.update(changes)
    return result


def fitted(training, *, reflections=None, **changes):
    records, initializer = training
    if reflections is None:
        reflections = initializer.reflection_coefficients * .9
    arguments = dict(innovation_variance=7.3, conditioning_samples=32,
                     objective_rows=sum(len(x)-32 for x in records),
                     optimization=optimization(), training_input_sha256=array_digest(*records))
    arguments.update(changes)
    return TemporalNoise.from_conditional_fit(initializer, reflections, **arguments)


def resign(record):
    result = json.loads(json.dumps(record))
    result.pop("artifact_sha256")
    result["artifact_sha256"] = metadata_digest(result)
    return result


def test_conditional_state_keeps_actual_full_burg_initialization_and_unit_shape(training):
    records, initializer = training
    reflections = initializer.reflection_coefficients * .9
    state = fitted(training, reflections=reflections)
    record = state.record()
    assert record["schema"] == "n2p3_temporal_noise/5"
    assert record["startup_mode"] == "stationary_innovations"
    assert record["reflection_origin"] == "conditional_profile_optimized_PARCOR"
    assert record["stability_certificate"] == "actual_PARCOR_strictly_inside_unit_interval"
    assert record["provenance"]["method"] == "fixed_prefix_conditional_profile_after_segmented_burg"
    assert json.loads(json.dumps(state.provenance["initialization"])) == json.loads(json.dumps(initializer.record()))
    assert state.provenance["records"] == initializer.provenance["records"]
    assert state.provenance["n_columns"] == 3
    assert state.provenance["conditioning_samples"] == 32
    assert state.provenance["objective_rows"] == 856
    assert state.provenance["available_lag_rows"] == 916
    assert state.provenance["training_input_sha256"] == array_digest(*records)
    assert tuple(x["input_sha256"] for x in state.provenance["records"]) == tuple(
        array_digest(x) for x in records)
    # AR2 step-up computed independently, preserving the actual reflection sign convention.
    expected_ar = np.array([-reflections[0]*(1+reflections[1]), -reflections[1]])
    np.testing.assert_array_equal(state.coefficients, expected_ar)
    expected = records[0][2:] - expected_ar[0]*records[0][1:-1] - expected_ar[1]*records[0][:-2]
    np.testing.assert_allclose(state.whiten(records[0])[2:], expected, atol=5e-16)
    other_scale = fitted(training, reflections=reflections, innovation_variance=91.)
    np.testing.assert_array_equal(state.whiten(records[0]), other_scale.whiten(records[0]))
    # Full W remains a stationary unit-innovation operator; the continuous owner
    # alone conditions its predictive model on the declared fixed prefix.
    n = 70
    impulse = np.zeros(20000)
    impulse[0] = 1
    response = lfilter([1.], np.r_[1., -expected_ar], impulse)
    covariance = toeplitz([response[:-lag or None] @ response[lag:] for lag in range(n)])
    operator = state.whiten(np.eye(n))
    np.testing.assert_allclose(operator @ covariance @ operator.T, np.eye(n), atol=2e-14)


def test_conditional_save_json_reload_freeze_and_distinct_burg_origin(training, tmp_path):
    records, initializer = training
    state = fitted(training, reflections=initializer.reflection_coefficients,
                   optimization=optimization("burg_only"))
    assert state.record()["reflection_origin"] == "actual_Burg_recursion_retained"
    path = tmp_path / "conditional_state.npz"
    state.save(path)
    reloaded = TemporalNoise.load(path)
    embedded = TemporalNoise.from_record(json.loads(json.dumps(state.record())))
    for restored in (reloaded, embedded):
        assert restored.record() == state.record()
        np.testing.assert_array_equal(restored.whiten(records[1]), state.whiten(records[1]))
    with pytest.raises(TypeError):
        state.provenance["optimization"]["final_objective"] = 0.
    with pytest.raises(ValueError):
        state.reflection_coefficients.setflags(write=True)
    # An immutable record exposes full initialization without allowing caller mutation.
    with pytest.raises(TypeError):
        state.record()["provenance"]["initialization"]["provenance"]["records"][0]["n_samples"] = 1


@pytest.mark.parametrize("mode", ["stationary", "conditional_prefix"])
def test_initializer_requires_actual_stationary_burg_lattice(training, mode):
    records, initializer = training
    other = TemporalNoise.fit(records, order=2, record_ids=("one", "two"),
                              fit_scope="source_training", startup_mode=mode)
    with pytest.raises(ValueError, match="initializ|Burg|stationary"):
        fitted((records, other), reflections=initializer.reflection_coefficients)


def test_initializer_cannot_be_analytic_recursive_or_forged_scope(training):
    records, initializer = training
    analytic = TemporalNoise.from_reflections(initializer.reflection_coefficients,
                                             innovation_variance=1., reference_id="not-fitted")
    with pytest.raises(ValueError, match="initializ|Burg|scope"):
        fitted((records, analytic))
    with pytest.raises(ValueError, match="initializ|Burg"):
        fitted((records, fitted(training)))
    record = fitted(training).record()
    forged = json.loads(json.dumps(record))
    forged["provenance"]["fit_scope"] = "target_test"
    with pytest.raises(ValueError, match="scope|ledger"):
        TemporalNoise.from_record(resign(forged))
    forged = json.loads(json.dumps(record))
    initial = forged["provenance"]["initialization"]
    initial["provenance"]["fit_scope"] = "target_test"
    forged["provenance"]["initialization"] = resign(initial)
    with pytest.raises(ValueError, match="scope|initializ|ledger"):
        TemporalNoise.from_record(resign(forged))


@pytest.mark.parametrize("changes", [
    {"conditioning_samples": True}, {"conditioning_samples": 1},
    {"conditioning_samples": 32.}, {"conditioning_samples": 400},
    {"objective_rows": 855}, {"objective_rows": True}, {"objective_rows": 0},
    {"training_input_sha256": "a"*63}, {"training_input_sha256": "A"*64},
])
def test_conditional_tail_support_is_the_actual_per_record_fixed_prefix(training, changes):
    with pytest.raises(ValueError, match="conditioning|objective|SHA-256|tail"):
        fitted(training, **changes)


@pytest.mark.parametrize("field,value", [
    ("records", [{"record_id": "different", "n_samples": 920, "input_sha256": "f"*64}]),
    ("n_columns", 4), ("fit_scope", "source_training"),
    ("implementation_sha256", "f"*64), ("conditioning_samples", 33),
])
def test_resigned_conditional_record_rechecks_the_initializer_relation(training, field, value):
    changed = json.loads(json.dumps(fitted(training).record()))
    changed["provenance"][field] = value
    with pytest.raises(ValueError, match="initializ|column|scope|implementation|lag|objective"):
        TemporalNoise.from_record(resign(changed))


@pytest.mark.parametrize("changes", [
    {"strategy": "secret_optimizer"}, {"initial_objective": np.inf},
    {"final_objective": np.nan}, {"final_objective": 51.},
    {"converged": False}, {"converged": 1}, {"iterations": -1},
    {"iterations": True}, {"evaluations": 0}, {"eta_max": 0.},
    {"eta_max": True}, {"status": "success"}, {"message": ""},
])
def test_invalid_optimization_cannot_be_published_as_a_fitted_state(training, changes):
    with pytest.raises(ValueError, match="optimization|objective|converged|eta"):
        fitted(training, optimization=optimization(**changes))


def test_burg_only_requires_actual_unchanged_parcor_and_objective(training):
    _, initializer = training
    with pytest.raises(ValueError, match="Burg|reflection|retained"):
        fitted(training, optimization=optimization("burg_only"))
    with pytest.raises(ValueError, match="objective|Burg|retained"):
        fitted(training, reflections=initializer.reflection_coefficients,
               optimization=optimization("burg_only", final_objective=49.))
    # eta_max constrains an optimization, never clips or vetoes retained Burg shape.
    state = fitted(training, reflections=initializer.reflection_coefficients,
                   optimization=optimization("burg_only", eta_max=.01))
    np.testing.assert_array_equal(state.reflection_coefficients, initializer.reflection_coefficients)
    with pytest.raises(ValueError, match="eta|domain"):
        fitted(training, optimization=optimization(eta_max=.01))


def test_order_zero_has_no_fake_shape_parameters():
    records = (np.random.default_rng(703).normal(size=(120, 2)),)
    initializer = TemporalNoise.fit(records, order=0, record_ids=("C0",),
                                    fit_scope="source_training", startup_mode="stationary_innovations")
    state = fitted((records, initializer), optimization=optimization(
        initial_objective=50., final_objective=50., iterations=0, evaluations=1,
        message="no AR shape parameters"))
    assert state.order == 0
    assert state.provenance["objective_rows"] == 88
    np.testing.assert_array_equal(state.whiten(records[0]), records[0])
    assert TemporalNoise.from_record(state.record()).record() == state.record()


def test_resigned_parcor_or_origin_change_cannot_change_what_will_execute(training):
    _, initializer = training
    state = fitted(training)
    changed = json.loads(json.dumps(state.record()))
    changed["reflection_coefficients"][0] *= .5
    with pytest.raises(ValueError, match="disagree"):
        TemporalNoise.from_record(resign(changed))
    changed = json.loads(json.dumps(state.record()))
    changed["reflection_origin"] = "actual_Burg_recursion"
    with pytest.raises(ValueError, match="replay"):
        TemporalNoise.from_record(resign(changed))
    retained = fitted(training, reflections=initializer.reflection_coefficients,
                      optimization=optimization("burg_only"))
    changed = json.loads(json.dumps(retained.record()))
    # Replacing both coherent numeric representations still cannot claim Burg retention.
    k = np.asarray(changed["reflection_coefficients"])*.9
    changed["reflection_coefficients"] = k.tolist()
    changed["coefficients"] = [-k[0]*(1+k[1]), -k[1]]
    with pytest.raises(ValueError, match="Burg|retained"):
        TemporalNoise.from_record(resign(changed))


def test_conditional_schema_cannot_be_relabelled_as_legacy_semantics(training):
    changed = json.loads(json.dumps(fitted(training).record()))
    changed["schema"] = "n2p3_temporal_noise/3"
    with pytest.raises(ValueError, match="contract"):
        TemporalNoise.from_record(resign(changed))
    _, initializer = training
    changed = json.loads(json.dumps(initializer.record()))
    changed["schema"] = "n2p3_temporal_noise/4"
    with pytest.raises(ValueError, match="contract"):
        TemporalNoise.from_record(resign(changed))


def test_conditional_stability_replays_actual_optimized_lattice_not_inverse_roots(monkeypatch):
    import transfer.temporal_noise as module

    data = (np.random.default_rng(875).normal(size=(1300, 2)),)
    initializer = TemporalNoise.fit(data, order=8, record_ids=("full-C-residual",),
                                    fit_scope="target_calibration", startup_mode="stationary_innovations")

    def forbidden(*args):
        raise AssertionError("conditional state must preserve actual PARCOR without inverse polynomial algebra")

    monkeypatch.setattr(module, "_reflections", forbidden)
    monkeypatch.setattr("scipy.linalg.solve_discrete_lyapunov", forbidden)
    parcor = np.array([-.9999961149865101, .9999901108404076, -.9999801038304161,
                       .999961519006537, -.9999206301468053, .9997979222646003,
                       -.9989401968753808, .9227446944279201])
    state = fitted((data, initializer), reflections=parcor, optimization=optimization(eta_max=8.))
    assert state.pole_radius > 1
    assert state.record()["reflection_origin"] == "conditional_profile_optimized_PARCOR"
    assert state.record()["expanded_polynomial_radius_role"] == "producer_diagnostic_not_compiled_state_authority"
    assert state.startup_covariance.shape == (0, 0)
    loaded = TemporalNoise.from_record(state.record())
    np.testing.assert_array_equal(loaded.reflection_coefficients, parcor)
    np.testing.assert_array_equal(loaded.whiten(data[0]), state.whiten(data[0]))


def test_conditional_optimization_cannot_hide_extra_or_missing_information(training):
    incomplete = optimization()
    incomplete.pop("status")
    with pytest.raises(ValueError, match="optimization"):
        fitted(training, optimization=incomplete)
    with pytest.raises(ValueError, match="optimization"):
        fitted(training, optimization=optimization(oracle_labels=[1, 0]))


@pytest.mark.parametrize("kind", ["coefficients", "reflections", "observations", "conditional", "variance"])
def test_complex_state_ingress_never_discards_imaginary_evidence(training, kind):
    _, initializer = training
    with pytest.raises(ValueError, match="real|variance"):
        if kind == "coefficients":
            TemporalNoise.from_parameters(np.array([.5+1j]), innovation_variance=1., reference_id="complex")
        elif kind == "reflections":
            TemporalNoise.from_reflections(np.array([.5+1j]), innovation_variance=1., reference_id="complex")
        elif kind == "observations":
            initializer.whiten(np.ones((100, 3), dtype=complex))
        elif kind == "conditional":
            fitted(training, reflections=initializer.reflection_coefficients.astype(complex)+1j)
        else:
            fitted(training, innovation_variance=1.+1j)


def test_legacy_schema_replay_does_not_acquire_conditional_training_meanings(training):
    records, _ = training
    for mode in ("stationary", "conditional_prefix", "stationary_innovations"):
        state = TemporalNoise.fit(records, order=2, record_ids=("one", "two"),
                                  fit_scope="source_training", startup_mode=mode)
        record = state.record()
        assert record["schema"] == "n2p3_temporal_noise/5"
        assert record["startup_mode"] == mode
        assert "conditioning_samples" not in record["provenance"]
        assert "initialization" not in record["provenance"]
        restored = TemporalNoise.from_record(copy.deepcopy(record))
        np.testing.assert_array_equal(restored.whiten(records[0]), state.whiten(records[0]))
