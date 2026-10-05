"""Public full-map diagnostics against independent finite filtering references."""
import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfiltfilt

from data.finite_filter_operator import FiniteFilterOperator

MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments/boundary_state_research.py"
SPEC = importlib.util.spec_from_file_location("boundary_full_map_research", MODULE_PATH)
research = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = research
SPEC.loader.exec_module(research)


def fixture():
    # Deliberately nonzero physical source origin. Every array is below 0.1MB.
    rng = np.random.default_rng(106620)
    context, short, output = (1000, 1072), (1010, 1060), (1020, 1045)
    operator = FiniteFilterOperator(((1., 0., 0., 1., -.4, 0.),), 3)
    x = rng.normal(scale=1e-6, size=(72, 2))
    correction = rng.normal(scale=.025, size=(50, 100))
    mapping = research.FrozenShortBoundaryMap(operator, correction, short, output, 64.,
        ("Fz", "Pz"), "right_ear", "V", "synthetic_calibration_parameters")
    time = np.arange(72)
    # Calibration protection dictionary and independent injections differ in
    # latency, width, sign and physical topography; no dictionary self-test.
    training = np.exp(-.5 * ((time - 34) / 7) ** 2)[:, None] * np.array([1., .3])
    rows = research.erp_row_basis(operator.apply(training[10:60])[10:35].ravel()[None])
    injections = {
        "early_distinct_topography": 1e-6 * np.exp(-.5 * ((time - 26.3) / 4) ** 2)[:, None] * np.array([-.2, 1.]),
        "late_broad_negative": -2e-6 * np.exp(-.5 * ((time - 44.2) / 10) ** 2)[:, None] * np.array([.7, -.5]),
    }
    return context, mapping, x, rows, injections


def evaluate(x, mapping, context, injections):
    return research.evaluate_complete_map(x, mapping, mapping.operator, context_scope=context,
        erp_injections=injections, injection_design_id="predeclared_independent_morphology_v1")


def analyze(x, mapping, context, rows, injections, *, penalties=(0.,)):
    return research.BoundaryStateAnalysis.compute(x, mapping.operator, mapping.operator, rows,
        short_scope=mapping.short_scope, output_scope=mapping.output_scope, sfreq=mapping.sfreq,
        penalty_grid=penalties, context_scope=context, frozen_map=mapping, erp_injections=injections,
        injection_design_id="predeclared_independent_morphology_v1")


def test_public_applier_matches_scipy_and_frozen_matrix_without_full_context_or_labels():
    context, mapping, x, _, _ = fixture()
    short = x[10:60]
    direct = sosfiltfilt(np.asarray(mapping.operator.sos), short, axis=0, padtype="odd", padlen=3)[10:35]
    expected = direct + (mapping.correction_matrix @ short.ravel()).reshape(25, 2)
    np.testing.assert_allclose(mapping.apply(short), expected, atol=1e-19, rtol=1e-12)
    with pytest.raises(ValueError, match="only samples"):
        mapping.apply(x)
    with pytest.raises(TypeError):
        mapping.apply(short, target=1)
    assert mapping.record()["short_scope"] == [1010, 1060]
    assert mapping.record()["flattening"] == "sample_then_physical_channel"
    assert mapping.record()["output_stage"] == "native_crop"
    assert context[0] != 0


def test_independent_erp_increments_match_direct_scipy_both_reference_definitions():
    context, mapping, x, _, injections = fixture()
    result = evaluate(x, mapping, context, injections)
    assert result["requested_injections"] == result["defined_short_relative_errors"] == result["defined_long_relative_errors"] == 2
    for row in result["erp_increments"]:
        h = injections[row["injection_id"]]
        short = sosfiltfilt(np.asarray(mapping.operator.sos), h[10:60], axis=0, padtype="odd", padlen=3)[10:35]
        long = sosfiltfilt(np.asarray(mapping.operator.sos), h, axis=0, padtype="odd", padlen=3)[20:45]
        actual = short + (mapping.correction_matrix @ h[10:60].ravel()).reshape(25, 2)
        for key, reference in (("short_reference", short), ("observed_long_reference", long)):
            expected = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
            assert row[key]["relative_error"] == pytest.approx(expected, abs=2e-12)
    assert not result["policy_selected"] and not result["production_or_public_performance_claim"]


def test_complete_white_noise_gain_uses_full_derivative_and_cancels_affine_offset():
    context, mapping, x, _, injections = fixture()
    offset = np.full(50, 20e-6)
    affine = research.FrozenShortBoundaryMap(mapping.operator, mapping.correction_matrix,
        mapping.short_scope, mapping.output_scope, mapping.sfreq, mapping.channel_names,
        mapping.reference, mapping.signal_unit, mapping.parameter_source_id, correction_offset=offset)
    gain = evaluate(x, affine, context, injections)["white_noise_gain"]
    rng = np.random.default_rng(501)
    probes = rng.normal(scale=1e-6, size=(128, 50, 2))
    baseline, actual = [], []
    for probe in probes:
        baseline.append(mapping.operator.apply(probe)[10:35])
        actual.append(affine.apply(x[10:60] + probe) - affine.apply(x[10:60]))
    empirical = np.sum(np.asarray(actual) ** 2) / np.sum(np.asarray(baseline) ** 2)
    assert empirical == pytest.approx(gain["mean_relative_to_short"], rel=.03)
    assert gain["mean_relative_to_short"] == pytest.approx(
        np.sum(affine.complete_matrix**2) / np.sum(affine.baseline_matrix**2), abs=1e-13)
    assert gain["worst_absolute"] == pytest.approx(np.linalg.norm(affine.complete_matrix, 2)**2)
    assert gain == evaluate(x, mapping, context, injections)["white_noise_gain"]
    assert not gain["offset_included_as_noise"]


def test_oracle_delta_is_not_the_deployed_short_only_prediction():
    context, mapping, x, rows, injections = fixture()
    zero = research.FrozenShortBoundaryMap(mapping.operator, np.zeros_like(mapping.correction_matrix),
        mapping.short_scope, mapping.output_scope, mapping.sfreq, mapping.channel_names,
        mapping.reference, mapping.signal_unit, "declared_no_correction_anchor")
    teacher = research.BoundaryStateAnalysis.compute(x, zero.operator, zero.operator, rows,
        short_scope=zero.short_scope, output_scope=zero.output_scope, sfreq=zero.sfreq,
        penalty_grid=(0.,), context_scope=context)
    deployed = analyze(x, zero, context, rows, injections)
    assert teacher.pareto_diagnostics[0]["reference_difference_mse"] == 0.
    assert teacher.pareto_diagnostics[0]["erp_increment_error"] is None
    actual_difference = zero.operator.apply(x)[20:45] - zero.operator.apply(x[10:60])[10:35]
    row = deployed.pareto_diagnostics[0]
    assert row["reference_difference_mse"] == pytest.approx(np.mean(actual_difference**2), rel=1e-10)
    assert row["reference_difference_mse"] > 0
    assert row["correction_role"] == "frozen_short_only_predictor"
    assert row["erp_increment_error"] is not None and row["white_noise_gain"] is not None
    assert deployed.grid_support["context_scope"] == [1000, 1072]
    assert not deployed.grid_support["observed_long_used_at_deployment"]


def test_identical_short_different_exterior_keeps_prediction_and_changes_teacher():
    context, mapping, x, rows, injections = fixture()
    alternate = x.copy()
    alternate[9] += (50e-6, -30e-6)
    np.testing.assert_array_equal(mapping.apply(x[10:60]), mapping.apply(alternate[10:60]))
    left, right = [analyze(value, mapping, context, rows, injections) for value in (x, alternate)]
    assert left.pareto_diagnostics[0]["reference_difference_mse"] != right.pareto_diagnostics[0]["reference_difference_mse"]
    assert left.pareto_diagnostics[0]["white_noise_gain"] == right.pareto_diagnostics[0]["white_noise_gain"]
    assert left.pareto_diagnostics[0]["erp_increment_error"] == right.pareto_diagnostics[0]["erp_increment_error"]


def test_external_only_injection_keeps_undefined_short_error_and_long_failure():
    context, mapping, x, _, _ = fixture()
    outside = np.zeros_like(x)
    outside[9] = (1e-6, -.5e-6)
    result = evaluate(x, mapping, context, {"external_only": outside})
    assert result["requested_injections"] == 1 and result["defined_short_relative_errors"] == 0
    row = result["erp_increments"][0]
    assert row["short_input_increment_norm"] == 0
    assert row["short_reference"]["relative_error"] is None
    assert row["short_reference"]["undefined_reason"] == "zero_reference_increment"
    assert row["short_reference"]["peak_latency_error_seconds"] == [None, None]
    assert row["observed_long_reference"]["peak_latency_error_seconds"] == [None, None]
    assert row["observed_long_reference"]["relative_error"] == pytest.approx(1.)


def test_stronger_output_protection_can_increase_complete_white_noise_gain():
    context, mapping, x, rows, injections = fixture()
    correction = -.5 * rows.T @ rows @ mapping.baseline_matrix
    canceller = research.FrozenShortBoundaryMap(mapping.operator, correction, mapping.short_scope,
        mapping.output_scope, mapping.sfreq, mapping.channel_names, mapping.reference, mapping.signal_unit,
        "known_synthetic_cancellation_counterexample")
    report = analyze(x, canceller, context, rows, injections, penalties=(0., 10.))
    first, protected = report.pareto_diagnostics
    assert protected["white_noise_gain"]["mean_relative_to_short"] > first["white_noise_gain"]["mean_relative_to_short"]
    assert protected["protection_map_condition"] == 11.
    assert protected["white_noise_gain"]["condition_in_identified_range"] != 11.


def test_input_hard_dictionary_does_not_protect_an_independent_erp_shape():
    context, mapping, x, _, injections = fixture()
    train_shape = np.zeros(100)
    train_shape[40] = 1.
    null = np.eye(100) - np.outer(train_shape, train_shape)
    correction = .8 * mapping.baseline_matrix @ null
    trained = research.FrozenShortBoundaryMap(mapping.operator, correction, mapping.short_scope,
        mapping.output_scope, mapping.sfreq, mapping.channel_names, mapping.reference, mapping.signal_unit,
        "input_hard_finite_dictionary_counterexample")
    np.testing.assert_array_equal(correction @ train_shape, np.zeros(50))
    result = evaluate(x, trained, context, injections)
    assert max(row["short_reference"]["relative_error"] for row in result["erp_increments"]) > .5


def test_supplied_map_and_erp_design_are_required_in_the_public_analysis_path():
    context, mapping, x, rows, injections = fixture()
    args = {"short_scope": mapping.short_scope, "output_scope": mapping.output_scope, "sfreq": mapping.sfreq,
            "penalty_grid": (0.,), "context_scope": context}
    with pytest.raises(ValueError, match="explicitly supplied"):
        research.BoundaryStateAnalysis.compute(x, mapping.operator, mapping.operator, rows, **args,
                                               erp_injections=injections)
    with pytest.raises(ValueError, match="independent ERP"):
        research.BoundaryStateAnalysis.compute(x, mapping.operator, mapping.operator, rows, **args,
                                               frozen_map=mapping)
    with pytest.raises(ValueError, match="explicit context"):
        args.pop("context_scope")
        research.BoundaryStateAnalysis.compute(x, mapping.operator, mapping.operator, rows, **args,
            frozen_map=mapping, erp_injections=injections, injection_design_id="separate")


def test_deeply_frozen_owned_parameters_survive_caller_and_output_mutation():
    _, mapping, x, _, _ = fixture()
    correction = mapping.correction_matrix.copy()
    offset = np.zeros(50)
    frozen = research.FrozenShortBoundaryMap(mapping.operator, correction, mapping.short_scope,
        mapping.output_scope, mapping.sfreq, list(mapping.channel_names), mapping.reference, mapping.signal_unit,
        "immutable_provided_calibration", correction_offset=offset)
    before, identity = frozen.apply(x[10:60]), frozen.digest
    correction[:] = 900.
    offset[:] = 99.
    returned = frozen.apply(x[10:60])
    returned[:] = 42.
    np.testing.assert_array_equal(frozen.apply(x[10:60]), before)
    assert frozen.digest == identity
    for value in (frozen.correction_matrix, frozen.correction_offset, frozen.baseline_matrix, frozen.complete_matrix):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        frozen.sfreq = 256.


def test_zero_baseline_or_increment_does_not_fabricate_relative_metrics():
    context, mapping, x, _, injections = fixture()
    zero_operator = FiniteFilterOperator(((0., 0., 0., 1., -.4, 0.),), 3)
    zero = research.FrozenShortBoundaryMap(zero_operator, np.zeros_like(mapping.correction_matrix),
        mapping.short_scope, mapping.output_scope, mapping.sfreq, mapping.channel_names, mapping.reference,
        mapping.signal_unit, "zero_transfer_counterexample")
    result = evaluate(x, zero, context, injections)
    assert result["white_noise_gain"]["mean_relative_to_short"] is None
    assert result["white_noise_gain"]["worst_absolute"] == 0
    assert result["white_noise_gain"]["rank"] == 0
    assert result["white_noise_gain"]["condition_in_identified_range"] is None
    assert result["defined_short_relative_errors"] == result["defined_long_relative_errors"] == 0


@pytest.mark.parametrize("bad", [np.full((50, 100), np.nan), np.full((50, 100), 1j), np.zeros((50, 144))])
def test_numeric_map_rejects_invalid_or_full_context_input_columns(bad):
    _, mapping, _, _, _ = fixture()
    with pytest.raises(ValueError):
        research.FrozenShortBoundaryMap(mapping.operator, bad, mapping.short_scope, mapping.output_scope,
            mapping.sfreq, mapping.channel_names, mapping.reference, mapping.signal_unit, "bad_numeric_map")


def test_invalid_context_injection_and_dense_budget_are_explicit():
    context, mapping, x, _, injections = fixture()
    with pytest.raises(ValueError, match="observed context support"):
        evaluate(x, mapping, (1011, 1083), injections)
    with pytest.raises(ValueError, match="same actually observed"):
        evaluate(x, mapping, context, {"wrong_grid": np.zeros((73, 2))})
    with pytest.raises(ValueError, match="byte budget"):
        research.FrozenShortBoundaryMap(mapping.operator, mapping.correction_matrix, mapping.short_scope,
            mapping.output_scope, mapping.sfreq, mapping.channel_names, mapping.reference, mapping.signal_unit,
            "too_large", byte_budget=8)
