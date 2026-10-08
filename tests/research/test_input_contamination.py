"""Actual harmonic refit, independent finite references and switch counterexamples."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import sosfiltfilt

from data.finite_filter_operator import FiniteFilterOperator
from data.harmonic import HarmonicConfig, clean_harmonic_segments
from research.boundary_observability import NativeBoundaryGrid
from research.input_contamination import (
    ContaminationInput,
    ContaminationProgram,
    audit_input_contamination,
)


def _input(samples, *, segments=None):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = ("Fz", "Pz")[:samples.shape[1]]
    grid = NativeBoundaryGrid((1000, 1000 + len(samples)), 64., channels,
        "synthetic_declared_reference", "a" * 64, "synthetic_not_public_or_ADC")
    return ContaminationInput(grid, samples,
        (grid.context_scope,) if segments is None else segments, grid.context_scope[1],
        "synthetic_math", "synthetic_math", "predeclared_synthetic_native_grid")


def _config(**changes):
    return HarmonicConfig(frequency_hz=16., search_half_width_hz=.02, window_seconds=1.,
        overlap_fraction=0., robust_iterations=1, envelope_order=0,
        protected_low_frequency_hz=4., huber_multiplier=100.,
        min_line_to_residual_rms=changes.get("min_line_to_residual_rms", 1.))


def _switch_fixture():
    # A protected/line-orthogonal residual makes the predeclared ratio gate
    # cross between amplitudes 1.3/sqrt(2)<1 and 1.5/sqrt(2)>1. The core cleaner,
    # frequency optimization and last actual solve still run unmocked.
    time = np.arange(64) / 64.
    line = np.sin(2 * np.pi * 16 * time)
    low = np.column_stack([np.cos(np.pi * (np.arange(64) + .5) * k / 64) for k in range(9)]
                          + [time - time.mean(), line, np.cos(2 * np.pi * 16 * time)])
    residual = np.random.default_rng(1).normal(size=64)
    residual -= low @ np.linalg.lstsq(low, residual, rcond=None)[0]
    residual /= np.sqrt(np.mean(residual**2))
    return _input((residual + 1.3 * line) * 1e-6), (.2e-6 * line)[:, None]


def test_existing_operator_cli_runs_complete_adaptive_audit_with_exact_declared_layout(tmp_path):
    from experiments.run_operator_decision_study import main

    observation, h = _switch_fixture()
    np.savez(tmp_path / "input.npz", X=observation.samples.T, erp=h,
             draws=np.stack((h, -h)))
    plan = {"input": "input.npz", "sfreq": 64., "native_grid": observation.grid.record(),
        "segments": [[0, len(h)]], "available_at_sample": observation.available_at_sample,
        "scope_role": "synthetic_math", "input_level": "synthetic_math",
        "observation_identity": observation.observation_identity, "config": asdict(_config()),
        "challenges": {"independent_line": {"kind": "line", "key": "erp"}},
        "challenge_design_identity": "predeclared_line_challenge", "perturbations_key": "draws",
        "perturbation_design_identity": "two_symmetric_line_draws"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    assert main(["audit-clean", "--plan", str(path), "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_bytes())
    report = result["audit_report"]
    assert report["requested_challenges"] == report["completed_challenges"] == 1
    assert report["empirical_output_moments"]["completed_draws"] == 2
    assert report["execution"]["input"]["native_grid"] == observation.grid.record()
    assert report["challenges"][0]["base_to_injected_state"]["n_application_switches"] == 1
    assert json.loads((output / "execution.json").read_bytes())["status"] == "completed"


def test_actual_gate_switch_rejects_the_frozen_no_cleaning_approximation():
    observation, h = _switch_fixture()
    program = ContaminationProgram(_config())
    report = audit_input_contamination(observation, program,
        {"independent_small_line": {"kind": "line", "values": h}},
        challenge_design_identity="predeclared_gate_counterexample").report
    row = report["challenges"][0]
    assert report["execution"]["harmonic_windows"][0]["applied"] is False
    assert row["injected_windows"][0]["applied"] is True
    assert row["base_to_injected_state"]["n_application_switches"] == 1
    # Freezing the base applied=False gives T_frozen=I and would certify zero
    # distortion. Actual complete refit removes the old line too: error >7xh.
    assert row["complete_increment_metrics"]["relative_error"] > 7.
    assert row["isolated_pipeline_increment_metrics"]["energy_fraction_remaining"] < 1e-10
    assert row["refit_vs_isolated_increment_relative_norm"] > 6.
    direct = [clean_harmonic_segments(values.T, 64., [(0, 64)], _config()).cleaned.T
              for values in (observation.samples, observation.samples + h)]
    expected = np.linalg.norm(direct[1] - direct[0] - h) / np.linalg.norm(h)
    assert row["complete_increment_metrics"]["relative_error"] == pytest.approx(expected, rel=1e-12)
    assert not report["parameters_selected_from_audit"]


def test_erp_isolated_and_frozen_checks_can_pass_while_actual_refit_damages_increment():
    observation, _ = _switch_fixture()
    time = np.arange(64) / 64.
    # Separate protected-morphology challenge, chosen in a synthetic development
    # counterexample, never from any participant/held labels or protocol score.
    base = observation.samples + (.06e-6 * np.sin(2 * np.pi * 16 * time))[:, None]
    observation = replace(observation, samples=base)
    h = (2e-6 * np.exp(-.5 * ((time - .4) / .03)**2))[:, None]
    row = audit_input_contamination(observation, ContaminationProgram(_config()),
        {"independent_gaussian_ERP": {"kind": "ERP", "values": h}},
        challenge_design_identity="synthetic_early_narrow_ERP_switch_challenge").report["challenges"][0]
    # Cleaning h alone does nothing and frozen base applied=False is I. Actual
    # T(x+h)-T(x) switches and removes line from the original contaminated base.
    assert row["isolated_pipeline_increment_metrics"]["relative_error"] == 0.
    assert row["base_to_injected_state"]["n_application_switches"] == 1
    assert row["complete_increment_metrics"]["relative_error"] > 2.


def test_finite_program_matches_actual_scipy_increment_and_empirical_covariance():
    rng = np.random.default_rng(240)
    observation = _input(rng.normal(scale=1e-6, size=(64, 2)))
    operator = FiniteFilterOperator(((1., 0., 0., 1., -.5, 0.),), 2)
    program = ContaminationProgram(finite_operator=operator)
    time = np.arange(64)
    h = np.exp(-.5 * ((time - 24.3) / 3.1)**2)[:, None] * np.array([1e-6, -2e-6])
    draws = rng.normal(scale=2e-7, size=(6, 64, 2))
    result = audit_input_contamination(observation, program,
        {"independent_ERP": {"kind": "ERP", "values": h}}, challenge_design_identity="morphology_not_fitted",
        perturbations=draws, perturbation_design_identity="six_explicit_iid_white_draws_seed240")
    expected_draws = np.stack([sosfiltfilt(np.asarray(operator.sos), value, axis=0, padtype="odd", padlen=2)
                               for value in draws])
    expected_mean = expected_draws.mean(axis=0)
    np.testing.assert_allclose(result.mean_output_increment, expected_mean, atol=1e-20)
    expected_factor = (expected_draws - expected_mean).reshape(6, -1) / np.sqrt(5)
    np.testing.assert_allclose(result.output_covariance_factor.T @ result.output_covariance_factor,
                               expected_factor.T @ expected_factor, atol=1e-25)
    row = result.report["challenges"][0]
    assert row["complete_increment_metrics"]["relative_error"] < 1e-13
    assert row["refit_vs_isolated_increment_relative_norm"] < 1e-13
    assert not result.report["empirical_output_moments"]["fixed_F_covariance_or_Gaussian_likelihood_claimed"]


def test_fixed_filter_synchronizes_mean_design_covariance_and_prior_not_only_data():
    # Independent Gaussian oracle includes uncertain response coefficients.
    # A known invertible F lowers white-noise RMS yet preserves log Bayes factor
    # when it also transforms mean/design and the prior-predictive covariance.
    operator = FiniteFilterOperator(((.5, 0., 0., 1., -.5, 0.),), 2)
    length = 18
    F = sosfiltfilt(np.asarray(operator.sos), np.eye(length), axis=0, padtype="odd", padlen=2)
    time = np.arange(length)
    designs = [np.exp(-.5 * ((time - peak) / 2.)**2)[:, None] for peak in (6., 10.)]
    x = designs[0][:, 0] + np.random.default_rng(41).normal(scale=.2, size=length)
    noise, prior = .2**2 * np.eye(length), np.array([[.25]])

    def density(samples, mean, covariance):
        residual = samples - mean
        return -.5 * (np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual))

    raw = [density(x, D[:, 0], noise + D @ prior @ D.T) for D in designs]
    correct = [density(F @ x, (F @ D)[:, 0], F @ noise @ F.T + (F @ D) @ prior @ (F @ D).T)
               for D in designs]
    wrong = [density(F @ x, D[:, 0], noise + D @ prior @ D.T) for D in designs]
    assert np.trace(F @ noise @ F.T) < np.trace(noise)
    assert correct[0] - correct[1] == pytest.approx(raw[0] - raw[1], abs=1e-11)
    assert abs((wrong[0] - wrong[1]) - (raw[0] - raw[1])) > .1


def test_real_harmonic_plus_finite_composition_is_not_a_duplicate_cleaner():
    observation, h = _switch_fixture()
    operator = FiniteFilterOperator(((1., 0., 0., 1., -.25, 0.),), 2)
    program = ContaminationProgram(_config(), operator)
    actual, receipt = program.apply(observation)
    direct = clean_harmonic_segments(observation.samples.T, 64., [(0, 64)], _config()).cleaned.T
    expected = sosfiltfilt(np.asarray(operator.sos), direct, axis=0, padtype="odd", padlen=2)
    np.testing.assert_array_equal(actual, expected)
    assert receipt["complete_read_scope"] == (1000, 1064)
    assert receipt["finite_segment_read_scopes"] == ((1000, 1064),)
    assert receipt["harmonic_windows"][0]["absolute_read_scope"] == (1000, 1064)
    assert receipt["harmonic_windows"][0]["available_at_sample"] == 1064
    assert receipt["earliest_whole_output_sample"] == 1064
    with pytest.raises(TypeError):
        program.apply(observation, labels=[1])
    assert h.shape == observation.samples.shape


def test_actual_segments_do_not_read_another_segment_and_retain_physical_availability():
    first, _ = _switch_fixture()
    samples = np.vstack((first.samples, first.samples * 3))
    observation = _input(samples, segments=((1000, 1064), (1064, 1128)))
    program = ContaminationProgram(_config(), FiniteFilterOperator(((1., 0., 0., 1., -.25, 0.),), 2))
    output, receipt = program.apply(observation)
    alternate = samples.copy()
    alternate[64:] *= 100
    changed, _ = program.apply(replace(observation, samples=alternate))
    np.testing.assert_array_equal(output[:64], changed[:64])
    assert receipt["earliest_whole_output_sample"] == 1128
    assert receipt["harmonic_windows"][1]["absolute_read_scope"] == (1064, 1128)


def test_complete_refit_noise_moments_include_switches_and_bias_not_fixed_f():
    observation, h = _switch_fixture()
    draws = np.stack((h, -h, 2 * h, -2 * h))
    result = audit_input_contamination(observation, ContaminationProgram(_config()),
        {"ERP_control": {"kind": "ERP", "values": np.zeros_like(h)}},
        challenge_design_identity="zero_control_still_requested", perturbations=draws,
        perturbation_design_identity="symmetric_explicit_four_line_draws")
    moments = result.report["empirical_output_moments"]
    assert moments["requested_draws"] == moments["completed_draws"] == 4
    assert moments["input_mean_increment_norm_volts"] == 0
    assert moments["output_mean_increment_norm_volts"] > 1e-7
    assert sum(row["n_application_switches"] for row in moments["state_transitions"]) == 2
    assert result.report["requested_challenges"] == 1
    assert result.report["challenges"][0]["complete_increment_metrics"]["relative_error"] is None
    assert not moments["asymptotic_or_population_gain_claimed"]


def test_zero_noise_energy_keeps_denominator_and_undefined_relative_readout():
    observation, h = _switch_fixture()
    result = audit_input_contamination(observation, ContaminationProgram(),
        {"glitch": {"kind": "glitch", "values": h}}, challenge_design_identity="declared_impulse_control",
        perturbations=np.zeros((3, *h.shape)), perturbation_design_identity="three_actual_zero_draws")
    moments = result.report["empirical_output_moments"]
    assert moments["requested_draws"] == moments["completed_draws"] == moments["zero_energy_draws"] == 3
    assert moments["output_second_moment_relative_to_input"] is None
    assert moments["output_second_moment_relative_to_finite_reference"] is None
    assert moments["sample_covariance_trace_volts_squared"] == 0


def test_result_state_is_owned_and_saved_once_without_copying_raw_or_labels(tmp_path):
    observation, h = _switch_fixture()
    result = audit_input_contamination(observation, ContaminationProgram(),
        {"ERP": {"kind": "ERP", "values": h}}, challenge_design_identity="fixed_synthetic")
    destination = result.save(tmp_path / "diagnostic")
    with np.load(destination / "moments.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"mean_output_increment", "output_covariance_factor"}
        np.testing.assert_array_equal(archive["mean_output_increment"], result.mean_output_increment)
    saved = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert saved["execution"]["input"]["input_level"] == "synthetic_math"
    assert not saved["held_truth_used"] and not saved["product_or_public_performance_claimed"]
    with pytest.raises(FileExistsError):
        result.save(destination)
    for array in (observation.samples, result.mean_output_increment, result.output_covariance_factor):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.report["status"] = "forged"


def test_canonical_view_adapter_retains_real_identity_time_and_eeg_roles():
    observation, _ = _switch_fixture()
    # Adapter checks its physical record and consumes only the same public
    # fields as an admitted PredictionDecisionView. No model/noise is imported.
    view = SimpleNamespace(samples=observation.samples, raw_scope=(1000, 1064), channel_names=("Fz",),
        channel_types=("EEG",), reference="right earlobe", sample_rate_hz=64., source_sha256="b" * 64,
        source_dataset_id="fixture", source_subject_id="one", source_relative_path="source.eeg",
        available_sample=1068, signal_unit="V", raw_input_digest="c" * 64,
        physical_record=lambda: {"schema": "n2p3_prediction_decision_physical_support/1"})
    actual = ContaminationInput.from_prediction_view(view, role="target_calibration",
                                                     input_level="distributed_processed_eeg")
    assert actual.available_at_sample == 1068
    assert actual.grid.reference == "right earlobe" and actual.grid.source_sha256 == "b" * 64
    assert actual.observation_identity == "c" * 64
    view.channel_types = ("EOG",)
    with pytest.raises(ValueError, match="canonical"):
        ContaminationInput.from_prediction_view(view, role="target_calibration", input_level="distributed_processed_eeg")


@pytest.mark.parametrize("bad", [np.full((64, 1), 1j), np.full((64, 1), np.nan), np.zeros((63, 1))])
def test_bad_numeric_increment_is_rejected_before_cleaning(bad):
    observation, _ = _switch_fixture()
    with pytest.raises(ValueError):
        audit_input_contamination(observation, ContaminationProgram(_config()),
            {"bad": {"kind": "ERP", "values": bad}}, challenge_design_identity="invalid")


def test_read_support_gaps_availability_draw_shape_and_budget_are_explicit():
    observation, h = _switch_fixture()
    with pytest.raises(ValueError, match="partition"):
        replace(observation, segments=((1000, 1030), (1031, 1064)))
    with pytest.raises(ValueError, match="availability"):
        replace(observation, available_at_sample=1063)
    with pytest.raises(ValueError, match="B>=2"):
        audit_input_contamination(observation, ContaminationProgram(),
            {"ERP": {"kind": "ERP", "values": h}}, challenge_design_identity="fixed",
            perturbations=np.zeros((1, 64, 1)), perturbation_design_identity="one_draw")
    with pytest.raises(ValueError, match="byte budget"):
        audit_input_contamination(observation, ContaminationProgram(),
            {"ERP": {"kind": "ERP", "values": h}}, challenge_design_identity="fixed", byte_budget=8)
