"""Frozen state restore, original producer identity and non-mirror tamper gates."""
import copy
import json
from decimal import ROUND_DOWN, localcontext

import numpy as np
import pytest

import transfer.temporal_noise as module
from tests.transfer.test_conditional_noise_state import fitted, optimization
from transfer.noise_model import array_digest, metadata_digest
from transfer.temporal_noise import TemporalNoise


def _rehash(record):
    state = record["compiled_state"]
    arrays = [np.asarray(state["arrays"][key], dtype=np.float64).reshape(state["shapes"][key])
              for key in module._ARRAY_FIELDS]
    state["state_sha256"] = array_digest(np.asarray(record["coefficients"]),
        np.asarray(record["reflection_coefficients"]), *arrays)
    record["artifact_sha256"] = metadata_digest({key: value for key, value in record.items() if key != "artifact_sha256"})
    return record


@pytest.mark.parametrize("mode", ["stationary", "conditional_prefix", "stationary_innovations"])
@pytest.mark.parametrize("order", [0, 2])
def test_load_consumes_saved_state_without_constructor_or_startup_compiler(mode, order, tmp_path, monkeypatch):
    coefficients = [] if order == 0 else [.6, -.1]
    state = TemporalNoise.from_parameters(coefficients, innovation_variance=3., reference_id="portable", startup_mode=mode)
    record = state.record()
    probe = np.random.default_rng(616).normal(size=(37, 2))
    original = state.whiten(probe)
    path = tmp_path/"state.npz"
    state.save(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen load or whitening recompiled startup state")

    for name in ("_startup", "_innovation_factor", "_reflections"):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(TemporalNoise, "__post_init__", forbidden)
    monkeypatch.setattr("scipy.linalg.solve_discrete_lyapunov", forbidden)
    for name in ("eigvals", "cond", "cholesky"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(np, "exp", forbidden)
    restored = TemporalNoise.load(path)
    assert restored.record() == record
    assert restored.identity_digest == state.identity_digest
    np.testing.assert_array_equal(restored.whiten(probe), original)
    for key, value in restored._arrays().items():
        np.testing.assert_array_equal(value, state._arrays()[key])
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_fixed_scalar_compile_ignores_callers_decimal_context():
    original = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="context")
    with localcontext() as context:
        context.prec, context.rounding, context.Emax, context.Emin = 7, ROUND_DOWN, 9, -9
        changed = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="context")
    assert original.record() == changed.record()


def test_parameter_stepdown_ignores_callers_decimal_context():
    original = TemporalNoise.from_parameters([.6, -.1], innovation_variance=1., reference_id="exact_polynomial")
    with localcontext() as context:
        context.prec, context.rounding, context.Emax, context.Emin = 7, ROUND_DOWN, 9, -9
        changed = TemporalNoise.from_parameters([.6, -.1], innovation_variance=1., reference_id="exact_polynomial")
    assert original.record() == changed.record()


def test_readonly_array_header_cannot_reinterpret_compiled_noise_without_guard_failure():
    state = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="header_integrity")
    # Bytes are immutable, but NumPy still allows dtype/header mutation.
    state.reflection_coefficients.dtype = np.int64
    with pytest.raises(ValueError, match="integrity"):
        state.whiten(np.ones((10, 1)))


def test_resigned_small_stationary_pivot_cannot_hide_in_a_large_covariance():
    # AR1 represented with an explicit zero second coefficient has a large
    # marginal variance and a unit second prefix prediction-error variance.
    state = TemporalNoise.from_parameters([.999999, 0.], innovation_variance=1., reference_id="small_density")
    record = state.record()
    root = np.asarray(record["compiled_state"]["arrays"]["startup_root"])
    assert root[1, 1] == 1.
    root[1, 1] *= 2.
    record["compiled_state"]["arrays"]["startup_root"] = root.tolist()
    record["compiled_state"]["arrays"]["startup_covariance"] = (root@root.T).tolist()
    # The actual prefix density direction is changed by four in variance even
    # though its contribution is small compared with the marginal C entries.
    assert root[1, 1]**2 == 4.
    with pytest.raises(ValueError, match="equation|density"):
        TemporalNoise.from_record(_rehash(record))


@pytest.mark.parametrize("key,index", [("startup_scales", (0,)), ("startup_log_variances", (0,)),
    ("startup_innovation_variances", (0,)), ("startup_whitening_factor", (1, 0))])
def test_coordinated_array_hash_changes_cannot_replace_actual_lattice_expression(key, index):
    record = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="tamper").record()
    array = np.asarray(record["compiled_state"]["arrays"][key])
    array[index] = np.nextafter(array[index], np.inf)
    record["compiled_state"]["arrays"][key] = array.tolist()
    with pytest.raises(ValueError, match="expression|recursion"):
        TemporalNoise.from_record(_rehash(record))


@pytest.mark.parametrize("key", ["startup_covariance", "startup_root"])
def test_stationary_equations_reject_resigned_numeric_state_without_recompiling(key):
    record = TemporalNoise.from_parameters([.7], innovation_variance=1., reference_id="AR1_unit").record()
    record["compiled_state"]["arrays"][key][0][0] *= 1.1
    with pytest.raises(ValueError, match="relation|equation"):
        TemporalNoise.from_record(_rehash(record))


def test_preserved_producer_diagnostics_do_not_drive_whitening_or_get_recomputed():
    state = TemporalNoise.from_reflections([-.8, .4], innovation_variance=1., reference_id="producer")
    record = json.loads(json.dumps(state.record()))
    record["pole_radius"] = float(np.nextafter(record["pole_radius"], np.inf))
    record["startup_factor_condition"] = float(np.nextafter(record["startup_factor_condition"], np.inf))
    record["floating_precision_condition_indicator"] = np.finfo(float).eps*record["startup_factor_condition"]
    record["compiler"]["source_sha256"] = "a"*64
    restored = TemporalNoise.from_record(_rehash(record))
    assert json.loads(json.dumps(restored.record())) == record
    probe = np.random.default_rng(617).normal(size=(31, 2))
    np.testing.assert_array_equal(restored.whiten(probe), state.whiten(probe))


def test_conditional_nested_initializer_load_does_not_recompile_actual_burg_state(monkeypatch):
    records = (np.random.default_rng(618).normal(size=(130, 2)),)
    initializer = TemporalNoise.fit(records, order=2, record_ids=("C",),
        fit_scope="target_calibration", startup_mode="stationary_innovations")
    state = fitted((records, initializer), reflections=initializer.reflection_coefficients,
                   optimization=optimization("burg_only"))
    record = state.record()

    def forbidden(*args, **kwargs):
        raise AssertionError("nested initializer recompiled actual Burg numerical state")

    monkeypatch.setattr(TemporalNoise, "__post_init__", forbidden)
    restored = TemporalNoise.from_record(record)
    assert restored.record() == record
    np.testing.assert_array_equal(restored.whiten(records[0]), state.whiten(records[0]))


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_new_loader_never_migrates_or_relabels_legacy_transport(schema):
    record = TemporalNoise.from_parameters([.4], innovation_variance=1., reference_id="no_alias").record()
    record["schema"] = f"n2p3_temporal_noise/{schema}"
    with pytest.raises(ValueError, match="original execution source"):
        TemporalNoise.from_record(_rehash(record))


def test_same_class_initializer_binding_rejects_cross_producer_mutation():
    records = (np.random.default_rng(619).normal(size=(130, 2)),)
    initializer = TemporalNoise.fit(records, order=2, record_ids=("C",),
        fit_scope="target_calibration", startup_mode="stationary_innovations")
    state = fitted((records, initializer))
    record = json.loads(json.dumps(copy.deepcopy(state.record())))
    record["provenance"]["implementation_sha256"] = "b"*64
    with pytest.raises(ValueError, match="scope|initialization"):
        TemporalNoise.from_record(_rehash(record))


def test_real_continuous_model_load_and_unlabelled_score_use_saved_temporal_state(tmp_path, monkeypatch):
    from tests.transfer.test_continuous_dictionary import inputs
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec

    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 3), sample_rate_hz=64.)
    model = ContinuousDecisionModel.fit(inputs(), dictionary=dictionary, sfreq=64., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-fixture", noise_order=1, conditioning_samples=32)
    path = tmp_path/"model.npz"
    model.save(path)
    probe = np.random.default_rng(620).normal(size=(172, 2))
    onsets, membership = np.array([40, 80, 120]), np.array([[1, 0], [0, 1], [1, 0]])
    arguments = dict(sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
                     operator_identity="native-fixture", record_id="held-no-labels")
    before = model.score(probe, onsets, membership, **arguments)

    def forbidden(*args, **kwargs):
        raise AssertionError("actual continuous load recompiled saved temporal state")

    monkeypatch.setattr(TemporalNoise, "__post_init__", forbidden)
    monkeypatch.setattr(module, "_startup", forbidden)
    monkeypatch.setattr(module, "_innovation_factor", forbidden)
    loaded = ContinuousDecisionModel.load(path)
    assert loaded.noise.record() == model.noise.record()
    np.testing.assert_array_equal(loaded.score(probe, onsets, membership, **arguments).probabilities, before.probabilities)


def test_finite_payload_with_overflowing_factor_relations_is_rejected_before_nan_comparisons():
    # Stable PARCOR gives expanded coefficients larger than two. Both Gram and
    # AR companion products overflow, so Inf/NaN comparison alone is no gate.
    reflections = np.array([.9, .9, .9])
    coefficients = module._parcor_coefficients(reflections)
    record = TemporalNoise.from_parameters(coefficients, innovation_variance=1., reference_id="finite_overflow").record()
    record["compiled_state"]["arrays"]["startup_covariance"] = (np.eye(3)*1e308).tolist()
    record["compiled_state"]["arrays"]["startup_root"] = (np.eye(3)*1e200).tolist()
    record["marginal_variance_per_unit_innovation"] = 1e308
    with pytest.raises(ValueError, match="not numerically representable"):
        TemporalNoise.from_record(_rehash(record))
