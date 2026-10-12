"""Fixed cubic dictionary expression admission and real existing consumers."""
import json
from decimal import ROUND_DOWN, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy.interpolate import BSpline

import transfer.response_dictionary as dictionary_owner
from research.joint_response_scale import NumericalResolutionError
from research.latent_response_reference import ResponseState, analytic_basis, selection_design
from tests.transfer.test_continuous_dictionary import inputs
from tests.transfer.test_ensemble_fit import selections
from transfer.continuous_decision import ContinuousDecisionModel, _event_design
from transfer.dictionary_artifact import ARRAY_KEYS, DictionaryArtifactError
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.kernel_prior import KernelPriorSpec
from transfer.noise_model import array_digest
from transfer.predictive_prior import PhysicalResponseMetric
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def dictionary(endpoint="free", *, width=5, rate=64., seconds=.5):
    return ResponseDictionary.compile(ResponseDictionarySpec("bspline", seconds, width, endpoint), sample_rate_hz=rate)


def test_minimum_cubic_space_has_independent_exact_bernstein_values_and_function_endpoints():
    actual = dictionary(width=4, rate=10., seconds=.9)
    expected = []
    for sample in range(9):
        q = Fraction(sample, 9)
        expected.append([float((1-q)**3), float(3*q*(1-q)**2), float(3*q*q*(1-q)), float(q**3)])
    np.testing.assert_array_equal(actual.matrix, expected)
    np.testing.assert_array_equal(actual.endpoint_rows, [[1., 0., 0., 0.], [0., 0., 0., 1.]])
    assert actual.actual_seconds == .9 and actual.times[-1] == .8
    assert not np.array_equal(actual.matrix[-1], actual.endpoint_rows[-1])
    assert actual.record()["schema"] == "n2p3_response_dictionary/3"
    assert actual.metadata()["compiler"]["semantic_id"] == "n2p3_physical_cubic_bspline_rational_binary64/1"
    with pytest.raises(ValueError, match="four|cubic"):
        dictionary(width=3)


def test_cubic_uniform_knots_and_hard_chart_match_independent_scipy_and_physical_metric():
    free = dictionary(width=8, rate=256., seconds=1.2)
    hard = dictionary("hard_zero", width=8, rate=256., seconds=1.2)
    knots = np.r_[np.zeros(4), np.arange(1, 5)/5, np.ones(4)]
    oracle = BSpline(knots, np.eye(8), 3)
    np.testing.assert_allclose(free.matrix, oracle(np.arange(307)/307), atol=4e-16, rtol=0)
    np.testing.assert_array_equal(hard.matrix, free.matrix[:, 1:-1])
    np.testing.assert_array_equal(hard.constraint_map, np.eye(8)[:, 1:-1])
    assert hard.matrix.shape == (307, 6)
    assert np.array_equal(hard.endpoint_rows@hard.constraint_map, np.zeros((2, 6)))
    # Value constraints do not impose slope zero; first surviving spline has
    # nonzero physical slope at0, so it is a distinct function-space question.
    assert oracle(0., nu=1)[1] != 0.
    metric = PhysicalResponseMetric.from_basis(hard.matrix, sample_rate_hz=256.,
        binding={"basis_sha256": array_digest(hard.matrix), "scope": "software_physical_metric"})
    assert metric.precision_root.shape == (12, 12)


@pytest.mark.parametrize("endpoint", ["free", "hard_zero"])
def test_cubic_saved_state_restores_without_compile_or_platform_interpolation(endpoint, monkeypatch):
    original = dictionary(endpoint)
    standard = original.record()
    with localcontext() as context:
        context.prec, context.rounding = 7, ROUND_DOWN
        alternate = dictionary(endpoint)
    assert alternate.record() == standard
    standard["implementation_sha256"], standard["compiler"]["source_sha256"] = "a"*64, "b"*64

    def forbidden(*args, **kwargs):
        raise AssertionError("restore must retain saved physical scalars")

    monkeypatch.setattr(dictionary_owner, "compile_state", forbidden)
    monkeypatch.setattr(ResponseDictionary, "compile", forbidden)
    restored = ResponseDictionary.from_record(standard)
    assert restored.record() == standard
    for key in ARRAY_KEYS:
        assert getattr(restored, key).tobytes() == getattr(original, key).tobytes()


@pytest.mark.parametrize("key,index", [("times", (3,)), ("ambient_matrix", (4, 1)),
    ("matrix", (4, 1)), ("endpoint_rows", (1, 2)), ("constraint_map", (0, 1))])
def test_rehashed_cubic_payload_ulp_tampering_is_rejected_by_expression_relation(key, index):
    record = dictionary("hard_zero").record()
    arrays = {name: np.asarray(record[name]) for name in ARRAY_KEYS}
    arrays[key][index] = np.nextafter(arrays[key][index], np.inf)
    record.update({name: value.tolist() for name, value in arrays.items()})
    record["state_sha256"] = array_digest(*(arrays[name] for name in ARRAY_KEYS))
    with pytest.raises(DictionaryArtifactError) as error:
        ResponseDictionary.from_record(record)
    assert error.value.reason == "physical_expression_relation_mismatch"
    assert error.value.field == key


@pytest.mark.parametrize("family,endpoint,digest", [
    ("gaussian", "free", "e1045eb312c9dbca36b8ff218a428235081ab10f8ab32fd1ebbcd53b761894c6"),
    ("cosine", "free", "e48f0ef3510ee3ae1041b1708631538ba87ab88f140d390651b0a7aece9d8b36"),
    ("cosine", "hard_zero", "25fc654cbbc43328a814b8a07db1e939b9f48a354b54400b2ab16c6e9fe530e2"),
])
def test_frozen_v2_numeric_program_remains_its_exact_arrays_and_schema(family, endpoint, digest):
    original = ResponseDictionary.compile(ResponseDictionarySpec(family, .5, 5, endpoint), sample_rate_hz=64.)
    assert original.record()["schema"] == "n2p3_response_dictionary/2"
    assert array_digest(*(getattr(original, key) for key in ARRAY_KEYS)) == digest
    assert ResponseDictionary.from_record(original.record()).record() == original.record()


def test_schema_family_and_knot_metadata_cannot_relabel_saved_numbers():
    record = dictionary().record()
    record["schema"] = "n2p3_response_dictionary/2"
    with pytest.raises(DictionaryArtifactError, match="schema.*family"):
        ResponseDictionary.from_record(record)
    record = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", .5, 5), sample_rate_hz=64.).record()
    record["schema"] = "n2p3_response_dictionary/3"
    with pytest.raises(DictionaryArtifactError, match="schema.*family"):
        ResponseDictionary.from_record(record)
    record = dictionary().record()
    record["generation"]["normalized_knots_exact_ratio"][4] = [1, 3]
    with pytest.raises(DictionaryArtifactError, match="metadata"):
        ResponseDictionary.from_record(record)
    record = dictionary().record()
    record["schema"] = []
    with pytest.raises(DictionaryArtifactError, match="unsupported_dictionary_schema"):
        ResponseDictionary.from_record(record)


def test_bspline_transformed_latent_reference_uses_exact_owner_and_zero_state_keeps_saved_rows():
    actual = dictionary()
    onsets, labels = np.array([7, 49]), np.array([0, 1])
    design, _ = selection_design(actual, onsets, labels, origin_sample=0, observed_samples=100,
        state=ResponseState(1., 1., 0., 1., 1.))
    np.testing.assert_array_equal(design, _event_design(onsets, labels, 100, actual.matrix))
    assert analytic_basis(actual, np.array([.1])).tobytes() == actual.evaluate([.1]).matrix.tobytes()
    shifted, ledger = selection_design(actual, onsets, labels, origin_sample=0, observed_samples=100,
        state=ResponseState(1., 1., .01, 1., 1.))
    assert np.isfinite(shifted).all() and not np.array_equal(shifted, design)
    assert ledger["events"][0]["response_evaluation"]["schema"] == "n2p3_response_dictionary_evaluation/1"
    # The exact finite-shift feature does not claim a new derivative protocol.
    with pytest.raises(NumericalResolutionError, match="off_grid.*unsupported"):
        analytic_basis(actual, np.array([.1]), derivative=1)


@pytest.mark.parametrize("endpoint,prior", [("free", KernelPriorSpec.calibration_g(4.)),
    ("hard_zero", KernelPriorSpec.calibration_g(4.)), ("hard_zero", None)])
def test_cubic_existing_continuous_fit_capacity_and_saved_prediction_consumer(endpoint, prior, tmp_path):
    actual = dictionary(endpoint)
    model = ContinuousDecisionModel.fit(inputs(), sfreq=64., channels=("a", "b"),
        candidate_ids=("one", "two"), operator_identity="native-fixture", dictionary=actual,
        noise_order=0, conditioning_samples=32, kernel_prior=prior)
    width = 2*actual.matrix.shape[1]
    geometry = model.dictionary_geometry
    assert geometry["kernel_dimensions"] == width and geometry["ambient_dimensions"] == 10
    embedding = np.kron(np.eye(2), actual.constraint_map)
    precision = np.eye(width) if prior is None else model.kernel_prior.precision_root.T@model.kernel_prior.precision_root
    np.testing.assert_allclose(embedding.T@np.asarray(geometry["ambient_prior_precision"])@embedding,
                               precision, atol=2e-12)
    information = np.asarray(model.provenance["kernel_prior_fit_statistics"]["information_root"])
    np.testing.assert_allclose(geometry["df_fit"], np.trace(information.T@information@model.kernel_row_covariance), atol=1e-12)
    path = tmp_path/"cubic.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
    assert metadata["schema"] == "n2p3_continuous_dictionary_model/2"
    assert metadata["dictionary"]["schema"] == "n2p3_response_dictionary/3"
    restored = ContinuousDecisionModel.load(path)
    samples = np.random.default_rng(2).normal(size=(172, 2))
    arguments = {"sfreq": 64., "channels": ("a", "b"), "candidate_ids": ("one", "two"),
        "operator_identity": "native-fixture", "record_id": "software_read_only"}
    onsets, membership = np.array([40, 80, 120]), np.array([[1, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(model.score(samples, onsets, membership, **arguments).probabilities,
                                  restored.score(samples, onsets, membership, **arguments).probabilities)


def test_cubic_real_grouped_oof_wire_and_package_have_correct_each_c_reference(tmp_path, monkeypatch):
    import transfer.continuous_decision as continuous

    spec = CalibrationExpertProgram("cubic_hard", "continuous_ar", noise_order=0, conditioning_samples=32,
        dictionary=ResponseDictionarySpec("bspline", .5, 5, "hard_zero"), kernel_prior=KernelPriorSpec.calibration_g(4.))
    assert CalibrationExpertProgram.from_record(spec.record()).dictionary == spec.dictionary
    groups, captured = selections(), []
    compiler = continuous.compile_kernel_prior

    def capture(*args, **kwargs):
        prior = compiler(*args, **kwargs)
        captured.append(prior)
        return prior

    monkeypatch.setattr(continuous, "compile_kernel_prior", capture)
    fitted = fit_calibration_ensemble(groups[:3], (spec,), EnsembleProgram.equal_weights("cubic", (spec.expert_id,)),
        calibration_strategy="grouped_temperature")
    assert len(captured) == 4
    for prior, row in zip(captured[1:], fitted.temperature_record["OOF_groups"], strict=True):
        assert prior.rank_ledger["observed_rows"] == 2*551
        assert row["held_record_id"] not in {item["record_id"] for item in prior.provenance["records"]}
    before = fitted.score(groups[3].view)
    package = fitted.save(tmp_path/"cubic_ensemble")
    restored = FittedEnsemble.load(package)
    assert restored.score(groups[3].view).record() == before.record()
    assert restored._experts[0].predictor.dictionary.record()["schema"] == "n2p3_response_dictionary/3"
