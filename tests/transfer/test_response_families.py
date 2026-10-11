"""New declared /4 families, physical constraints and existing actual consumers."""
from decimal import ROUND_DOWN, localcontext

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.linalg import block_diag

import transfer.response_dictionary as dictionary_owner
from research.joint_response_scale import NumericalResolutionError
from research.latent_response_reference import ResponseState, analytic_basis, selection_design
from tests.transfer.test_continuous_dictionary import inputs
from tests.transfer.test_ensemble_fit import selections
from transfer.continuous_decision import (
    ContinuousDecisionModel,
    _event_design,
    _exact_capacity_diagnostics,
)
from transfer.dictionary_artifact import ARRAY_KEYS, DictionaryArtifactError
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.kernel_prior import KernelPriorSpec, compile_kernel_prior
from transfer.noise_model import array_digest
from transfer.predictive_prior import PhysicalResponseMetric
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec


def dictionary(family, endpoint="free", *, rate=64., seconds=.5):
    width = 5 if family == "raised_cosine" else 9
    return ResponseDictionary.compile(ResponseDictionarySpec(family, seconds, width, endpoint), sample_rate_hz=rate)


def multiscale_change(dictionary):
    ratios = dictionary.metadata()["generation"]["coarse_in_fine_exact_ratio"]
    coarse = np.asarray([[numerator/denominator for numerator, denominator in row] for row in ratios])
    return np.column_stack((coarse, np.eye(9)[:, (2, 4, 6)]))


def test_raised_composed_rounding_and_exact_atom_cuts_do_not_sign_old_sequential_prototype():
    actual = ResponseDictionary.compile(ResponseDictionarySpec("raised_cosine", 11/16, 3), sample_rate_hz=16.)
    # Independent140digit composed-expression constant, phase9/11. The old
    # prototype RNcos then affineRN returned0.0793732335844094 instead.
    assert actual.matrix[10, 1] == 0.07937323358440941
    assert actual.matrix[10, 1] != 0.0793732335844094
    even = ResponseDictionary.compile(ResponseDictionarySpec("raised_cosine", .5, 3), sample_rate_hz=16.)
    np.testing.assert_array_equal(even.ambient_matrix[0], [1., 0., 0.])
    np.testing.assert_array_equal(even.ambient_matrix[4], [0., 1., 0.])
    np.testing.assert_array_equal(even.endpoint_rows[1], [0., 0., 1.])
    metadata = actual.metadata()
    assert metadata["schema"] == "n2p3_response_dictionary/4"
    assert "composed" in metadata["compiler"]["semantic_id"]
    assert "free_endpoint_values_may_jump" in metadata["generation"]["global_crop"]
    hard = dictionary("raised_cosine", "hard_zero")
    assert np.array_equal(hard.matrix, hard.ambient_matrix[:, 1:-1])
    assert np.array_equal(hard.endpoint_rows@hard.constraint_map, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ResponseDictionarySpec("raised_cosine", .5, 1)


def test_multiscale_is_independent_coarse_detail_and_has_its_own_hard_endpoint_chart():
    actual, hard = dictionary("multiscale_bspline"), dictionary("multiscale_bspline", "hard_zero")
    fine = ResponseDictionary.compile(ResponseDictionarySpec("bspline", .5, 9), sample_rate_hz=64.)
    change = multiscale_change(actual)
    assert np.linalg.matrix_rank(actual.matrix) == 9
    assert np.linalg.det(change) == pytest.approx(-117/1024)
    np.testing.assert_allclose(actual.matrix, fine.matrix@change, atol=2e-16, rtol=0)
    np.testing.assert_array_equal(hard.constraint_map, np.eye(9)[:, (1, 2, 3, 4, 6, 7, 8)])
    np.testing.assert_array_equal(hard.matrix, actual.matrix[:, (1, 2, 3, 4, 6, 7, 8)])
    assert hard.matrix.shape == (32, 7) and np.linalg.matrix_rank(hard.matrix) == 7
    assert np.array_equal(hard.endpoint_rows@hard.constraint_map, np.zeros((2, 7)))
    # Independent Scipy spline definition: only coarse6 plus fine2/4/6.
    coarse_knots = np.r_[np.zeros(4), 1/3, 2/3, np.ones(4)]
    fine_knots = np.r_[np.zeros(4), np.arange(1, 6)/6, np.ones(4)]
    grid = np.arange(32)/32
    expected = np.column_stack((BSpline(coarse_knots, np.eye(6), 3)(grid),
                                BSpline(fine_knots, np.eye(9), 3)(grid)[:, (2, 4, 6)]))
    np.testing.assert_allclose(actual.matrix, expected, atol=6e-16, rtol=0)
    with pytest.raises(ValueError, match="nine"):
        ResponseDictionarySpec("multiscale_bspline", .5, 10)


@pytest.mark.parametrize("endpoint", ["free", "hard_zero"])
def test_multiscale_gl_prior_metric_and_capacity_consume_actual_ambient_restriction(endpoint):
    actual = dictionary("multiscale_bspline", endpoint)
    fine = ResponseDictionary.compile(ResponseDictionarySpec("bspline", .5, 9, endpoint), sample_rate_hz=64.)
    full_change = multiscale_change(actual)
    change = fine.constraint_map.T@full_change@actual.constraint_map
    width = actual.matrix.shape[1]
    assert change.shape == (width, width) and np.linalg.matrix_rank(change) == width
    np.testing.assert_allclose(fine.matrix@change, actual.matrix, rtol=0, atol=2e-16)
    block_change = block_diag(change, change)
    compiled = []
    for state in (fine, actual):
        designs = [_event_design(np.array([8, 42, 86]), np.array([0, 1, index]), 160, state.matrix) for index in (0, 1)]
        compiled.append(compile_kernel_prior(KernelPriorSpec.calibration_g(4.), designs,
            record_ids=("synthetic_c0", "synthetic_c1"), fit_scope="target_calibration",
            binding={"basis_sha256": array_digest(state.matrix), "scope": "synthetic_design_geometry"}))
    first, second = compiled
    np.testing.assert_allclose(second.precision_root.T@second.precision_root,
        block_change.T@(first.precision_root.T@first.precision_root)@block_change, rtol=2e-13, atol=5e-13)
    np.testing.assert_allclose(first.row_covariance, block_change@second.row_covariance@block_change.T,
                               rtol=5e-12, atol=5e-12)
    for prior in compiled:
        df, _, energy = _exact_capacity_diagnostics(prior.reference_root, prior.reference_root, prior.precision_root)
        assert df == pytest.approx(2*width*4/5, abs=3e-13)
        assert energy == pytest.approx(2*width*4, abs=5e-12)
    metric = PhysicalResponseMetric.from_basis(fine.matrix, sample_rate_hz=64., binding={"scope": "synthetic_metric"})
    pulled = metric.pullback(block_change, binding={"scope": "synthetic_changed_chart"})
    np.testing.assert_allclose(pulled.precision_root.T@pulled.precision_root,
        block_change.T@(metric.precision_root.T@metric.precision_root)@block_change, atol=1e-15)


@pytest.mark.parametrize("family,endpoint", [(f, e) for f in ("raised_cosine", "multiscale_bspline") for e in ("free", "hard_zero")])
def test_new_families_restore_saved_numbers_and_sources_without_compile(family, endpoint, monkeypatch):
    actual = dictionary(family, endpoint)
    record = actual.record()
    with localcontext() as context:
        context.prec, context.rounding = 7, ROUND_DOWN
        assert dictionary(family, endpoint).record() == record
    record["implementation_sha256"], record["compiler"]["source_sha256"] = "a"*64, "b"*64

    def forbidden(*args, **kwargs):
        raise AssertionError("restore may not replace saved state by compile")

    monkeypatch.setattr(dictionary_owner, "compile_state", forbidden)
    restored = ResponseDictionary.from_record(record)
    assert restored.record() == record
    for key in ARRAY_KEYS:
        assert getattr(actual, key).tobytes() == getattr(restored, key).tobytes()


@pytest.mark.parametrize("family,key,index", [("raised_cosine", "ambient_matrix", (3, 1)),
    ("raised_cosine", "endpoint_rows", (1, 2)), ("multiscale_bspline", "constraint_map", (0, 0)),
    ("multiscale_bspline", "matrix", (4, 1))])
def test_new_family_rehashed_ulp_payload_is_rejected_by_own_physical_expression(family, key, index):
    record = dictionary(family, "hard_zero").record()
    arrays = {name: np.asarray(record[name]) for name in ARRAY_KEYS}
    arrays[key][index] = np.nextafter(arrays[key][index], np.inf)
    record.update({name: value.tolist() for name, value in arrays.items()})
    record["state_sha256"] = array_digest(*(arrays[name] for name in ARRAY_KEYS))
    with pytest.raises(DictionaryArtifactError, match="physical_expression"):
        ResponseDictionary.from_record(record)


def test_v4_family_semantics_cannot_relabel_v3_or_change_multiscale_chart():
    record = dictionary("raised_cosine").record()
    record["schema"] = "n2p3_response_dictionary/3"
    with pytest.raises(DictionaryArtifactError, match="schema.*family"):
        ResponseDictionary.from_record(record)
    cubic = ResponseDictionary.compile(ResponseDictionarySpec("bspline", .5, 5), sample_rate_hz=64.).record()
    cubic["schema"] = "n2p3_response_dictionary/4"
    with pytest.raises(DictionaryArtifactError, match="schema.*family"):
        ResponseDictionary.from_record(cubic)
    record = dictionary("multiscale_bspline").record()
    record["generation"]["fine_detail_indices"] = [1, 3, 5]
    with pytest.raises(DictionaryArtifactError, match="metadata"):
        ResponseDictionary.from_record(record)


@pytest.mark.parametrize("family", ["raised_cosine", "multiscale_bspline"])
def test_saved_native_states_work_but_new_family_offgrid_remains_typed(family):
    actual = dictionary(family)
    onsets, labels = np.array([7, 49]), np.array([0, 1])
    design, _ = selection_design(actual, onsets, labels, origin_sample=0, observed_samples=100,
        state=ResponseState(1., 1., 0., 1., 1.))
    np.testing.assert_array_equal(design, _event_design(onsets, labels, 100, actual.matrix))
    with pytest.raises(NumericalResolutionError, match="off_grid.*unsupported"):
        analytic_basis(actual, np.array([.1]))


@pytest.mark.parametrize("family,endpoint", [(f, e) for f in ("raised_cosine", "multiscale_bspline") for e in ("free", "hard_zero")])
def test_real_existing_synthetic_fit_prior_capacity_save_score_consume_v4(family, endpoint, tmp_path):
    actual = dictionary(family, endpoint)
    model = ContinuousDecisionModel.fit(inputs(), sfreq=64., channels=("a", "b"), candidate_ids=("one", "two"),
        operator_identity="native-fixture", dictionary=actual, noise_order=0, conditioning_samples=32,
        kernel_prior=KernelPriorSpec.calibration_g(4.))
    width = 2*actual.matrix.shape[1]
    prior = model.kernel_prior.precision_root.T@model.kernel_prior.precision_root
    embedding = np.kron(np.eye(2), actual.constraint_map)
    np.testing.assert_allclose(embedding.T@np.asarray(model.dictionary_geometry["ambient_prior_precision"])@embedding,
                               prior, atol=1e-12)
    assert model.dictionary_geometry["kernel_dimensions"] == width
    root = np.asarray(model.provenance["kernel_prior_fit_statistics"]["information_root"])
    np.testing.assert_allclose(model.dictionary_geometry["df_fit"], np.trace(root.T@root@model.kernel_row_covariance), atol=2e-12)
    path = tmp_path/"new_family.npz"
    model.save(path)
    restored = ContinuousDecisionModel.load(path)
    samples = np.random.default_rng(2).normal(size=(172, 2))
    arguments = {"sfreq": 64., "channels": ("a", "b"), "candidate_ids": ("one", "two"),
        "operator_identity": "native-fixture", "record_id": "synthetic_read_only"}
    onsets, membership = np.array([40, 80, 120]), np.array([[1, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(model.score(samples, onsets, membership, **arguments).probabilities,
                                  restored.score(samples, onsets, membership, **arguments).probabilities)
    assert restored.dictionary.record()["schema"] == "n2p3_response_dictionary/4"


@pytest.mark.parametrize("family", ["raised_cosine", "multiscale_bspline"])
def test_real_grouped_oof_and_ensemble_package_use_each_training_reference(family, tmp_path, monkeypatch):
    import transfer.continuous_decision as continuous

    spec = CalibrationExpertProgram(family, "continuous_ar", noise_order=0, conditioning_samples=32,
        dictionary=dictionary(family, "hard_zero", rate=256.).spec, kernel_prior=KernelPriorSpec.calibration_g(4.))
    assert CalibrationExpertProgram.from_record(spec.record()).dictionary == spec.dictionary
    groups, captured = selections(), []
    original = continuous.compile_kernel_prior

    def capture(*args, **kwargs):
        prior = original(*args, **kwargs)
        captured.append(prior)
        return prior

    monkeypatch.setattr(continuous, "compile_kernel_prior", capture)
    fitted = fit_calibration_ensemble(groups[:3], (spec,), EnsembleProgram.equal_weights("new", (family,)),
        calibration_strategy="grouped_temperature")
    assert len(captured) == 4
    for prior, row in zip(captured[1:], fitted.temperature_record["OOF_groups"], strict=True):
        assert prior.rank_ledger["observed_rows"] == 2*551
        assert row["held_record_id"] not in {item["record_id"] for item in prior.provenance["records"]}
    before = fitted.score(groups[3].view)
    package = fitted.save(tmp_path/family)
    assert FittedEnsemble.load(package).score(groups[3].view).record() == before.record()
