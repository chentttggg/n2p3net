"""Independent observability and actual finite native-T counterexamples."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfiltfilt

from data.finite_filter_operator import FiniteFilterOperator
from research.boundary_observability import (
    BoundaryIdentifiabilityError,
    BoundaryObservationRangeError,
    LinearFunctionalObservability,
    NativeAffineBoundaryMap,
    NativeBoundaryGrid,
    NativeBoundaryObservability,
)


def _linear(observation, L, **kwargs):
    return LinearFunctionalObservability(observation, L, "declared_latent_coordinates",
                                         "actual_noiseless_observation", "requested_functional", **kwargs)


def _native(*, all_observed=False, zero_operator=False):
    grid = NativeBoundaryGrid((1000, 1072), 64., ("Fz", "Pz"), "synthetic_declared_reference",
                              "a" * 64, "synthetic_native_grid_not_a_public_dataset")
    operator = FiniteFilterOperator(((0. if zero_operator else 1., 0., 0., 1., -.4, 0.),), 3)
    observations = ((1000, 1072),) if all_observed else ((1010, 1060),)
    problem = NativeBoundaryObservability(grid, operator, operator, (1010, 1060), (1020, 1045),
                                         observations, observations[-1][1])
    time = np.arange(72)
    x = np.column_stack((3e-6 * np.sin(.2 * time), -2e-6 * np.cos(.13 * time)))
    injections = {
        "early_positive_Pz": 1e-6 * np.exp(-.5 * ((time - 25.3) / 3.1)**2)[:, None] * np.array([-.2, 1.]),
        "late_negative_Fz": -2e-6 * np.exp(-.5 * ((time - 43.7) / 8.4)**2)[:, None] * np.array([1., .4]),
    }
    return problem, x, injections


def _direct(problem, full):
    return sosfiltfilt(np.asarray(problem.context_operator.sos), full, axis=0, padtype="odd", padlen=3)[20:45]


def test_unobserved_direction_has_typed_failure_even_with_arbitrarily_tight_prior():
    audit = _linear([[1., 0.]], [[0., 1.]])
    report = audit.report()
    # Independent Gaussian oracle: conditioning the first independent coordinate
    # leaves variance 1e-24 in the second. Tight prediction is still not observation.
    prior = np.diag([1., 1e-24])
    assert prior[1, 1] - prior[1, 0]**2 / prior[0, 0] == 1e-24
    assert report["status"] == "structurally_unidentifiable"
    assert report["witness"]["exactly_zero_stored_map_observation"]
    assert report["witness"]["functional_change_norm"] == 1.
    with pytest.raises(BoundaryIdentifiabilityError) as caught:
        audit.recover([2.])
    assert caught.value.report["prior_makes_unobserved_state_an_observation"] is False


def test_functional_can_be_recovered_without_recovering_the_whole_latent_state():
    identified = _linear([[1., 1.]], [[2., 2.]])
    np.testing.assert_allclose(identified.recover([3.]), [6.], rtol=1e-14)
    assert identified.report()["observation_rank"] == 1
    whole = _linear([[1., 1.]], np.eye(2))
    with pytest.raises(BoundaryIdentifiabilityError):
        whole.recover([3.])


def test_observed_state_is_confounded_with_an_unknown_nuisance_input():
    # y=u+a is compatible with (u,a)=(3,0) and (1,2). Observing y does not identify a.
    confounded = _linear([[1., 1.]], [[0., 1.]])
    with pytest.raises(BoundaryIdentifiabilityError):
        confounded.recover([3.])
    # A separately observed nuisance removes the ambiguity through the same API.
    resolved = _linear([[1., 1.], [1., 0.]], [[0., 1.]])
    np.testing.assert_allclose(resolved.recover([3., 1.]), [2.], rtol=1e-14)


def test_full_rank_does_not_hide_weak_conditioning_or_threshold_truncation():
    observation = np.array([[1., 1.], [1., 1. + 1e-8]])
    strong = _linear(observation, [[0., 1.]])
    assert strong.report()["observation_rank"] == 2
    assert strong.report()["condition_in_identified_range"] > 1e8
    weak = _linear([[1., 1.], [1., 1. + 1e-14]], [[0., 1.]], rank_relative_tolerance=1e-10)
    assert weak.report()["discarded_nonzero_singular_values"]
    assert weak.report()["status"] == "unresolved_at_declared_resolution"
    with pytest.raises(BoundaryIdentifiabilityError):
        weak.recover([2., 2.])


def test_impossible_noiseless_observation_is_not_projected_into_the_range():
    audit = _linear([[1., 0.], [0., 0.]], [[1., 0.]])
    np.testing.assert_array_equal(audit.recover([3., 0.]), [3.])
    with pytest.raises(BoundaryObservationRangeError):
        audit.recover([3., 1.])


def test_row_unit_rescaling_preserves_identification_and_functional_recovery():
    observation, L, z = np.array([[1., 1.], [1., -1.]]), np.array([[2., -.3]]), np.array([2., 5.])
    scale = np.diag([1e-6, 1e6])
    first, second = _linear(observation, L), _linear(scale @ observation, L)
    np.testing.assert_allclose(first.recover(observation @ z), L @ z, atol=1e-14)
    np.testing.assert_allclose(second.recover(scale @ observation @ z), L @ z, atol=1e-14)
    assert first.report()["observation_rank"] == second.report()["observation_rank"]


@pytest.mark.parametrize("scale", [1e-12, 1e-200, 1e200])
def test_each_requested_functional_has_its_own_identification_resolution(scale):
    audit = _linear([[1., 0.]], [[1., 0.], [0., scale]])
    assert audit.report()["status"] == "structurally_unidentifiable"
    assert audit.report()["witness"]["functional_row_index"] == 1
    with pytest.raises(BoundaryIdentifiabilityError):
        audit.recover([3.])


def test_recovery_does_not_materialize_an_overflowing_physical_unit_gain():
    audit = _linear([[1e-200, 0.]], [[1e200, 0.]])
    np.testing.assert_allclose(audit.recover([2e-200]), [2e200], rtol=2e-14)
    assert audit.report()["status"] == "identified_at_declared_resolution"


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_noiseless_range_gate_does_not_depend_on_observation_amplitude(scale):
    audit = _linear([[1., 0.], [0., 0.]], [[1., 0.]])
    with pytest.raises(BoundaryObservationRangeError):
        audit.recover([0., scale])


def test_native_same_short_and_different_exterior_is_an_actual_scipy_counterexample():
    problem, x, _ = _native()
    report, delta = problem.counterexample(x, channel_name="Pz")
    assert report["status"] == "verified" and report["same_legal_observation_bytes"]
    assert report["outside_sample"] not in range(1010, 1060)
    np.testing.assert_array_equal(problem.observe(x), problem.observe(x + delta))
    expected = _direct(problem, x + delta) - _direct(problem, x)
    assert report["actual_target_change_norm"] == pytest.approx(np.linalg.norm(expected), rel=1e-12)
    assert np.linalg.norm(expected) > 0
    with pytest.raises(BoundaryIdentifiabilityError) as caught:
        problem.recover(problem.observe(x))
    assert caught.value.report["status"] == "structurally_unidentifiable"
    assert caught.value.report["observation_rank"] == 100


def test_observed_context_recovers_same_grid_target_without_claiming_unknown_state():
    problem, x, _ = _native(all_observed=True)
    report = problem.report()
    assert report["status"] == "identified" and report["missing_native_samples"] == []
    np.testing.assert_allclose(problem.recover(problem.observe(x)), _direct(problem, x), atol=1e-19)
    assert report["physical_state_dimension_inferred"] is False
    assert report["context_teacher_available_at_sample"] == 1072


def test_deterministic_padding_has_identical_state_for_identical_short():
    problem, x, _ = _native()
    _, delta = problem.counterexample(x)
    first = problem.padding_state_report(x[10:60])
    second = problem.padding_state_report((x + delta)[10:60])
    assert first == second and first["state_dimension_per_channel"] == 4
    assert first["decomposition_residual_norm"] < 1e-18
    assert not first["conditional_on_short_extra_independent_covariance"]


def test_complete_t_increment_and_gain_include_the_whole_operator_and_cancel_offset():
    problem, x, injections = _native()
    rng = np.random.default_rng(20261008)
    K = rng.normal(scale=.015, size=(50, 100))
    mapping = NativeAffineBoundaryMap(problem, K, "synthetic_provided_K_not_fitted", np.full(50, 20e-6))
    report = mapping.measure(x, injections, injection_design_identity="independent_latencies_widths_topographies")
    for entry in report["erp_increments"]:
        h = injections[entry["injection_identity"]]
        short_reference = sosfiltfilt(np.asarray(problem.short_operator.sos), h[10:60], axis=0,
                                      padtype="odd", padlen=3)[10:35]
        full_increment = short_reference + (K @ h[10:60].ravel()).reshape(25, 2)
        expected = np.linalg.norm(full_increment - short_reference) / np.linalg.norm(short_reference)
        assert entry["short_reference"]["relative_error"] == pytest.approx(expected, abs=1e-12)
        long_reference = _direct(problem, h)
        expected = np.linalg.norm(full_increment - long_reference) / np.linalg.norm(long_reference)
        assert entry["context_reference"]["relative_error"] == pytest.approx(expected, abs=1e-12)
        assert entry["actual_vs_affine_derivative_roundoff_norm"] < 1e-18
    gain = report["white_noise_gain"]
    assert gain["mean_relative_to_short"] == pytest.approx(
        np.sum(mapping.complete_matrix**2) / np.sum(mapping.baseline_matrix**2))
    assert gain["worst_absolute"] == pytest.approx(np.linalg.norm(mapping.complete_matrix, 2)**2)
    zero_offset = NativeAffineBoundaryMap(problem, K, "same_K_no_offset")
    assert zero_offset.measure(x, injections, injection_design_identity="same_design")["white_noise_gain"] == gain
    assert not gain["offset_included_as_noise"]
    assert report["mapping"]["deployment_available_at_sample"] == 1060
    assert report["diagnostic_context_available_at_sample"] == 1072


def test_external_only_challenge_cannot_be_fixed_by_a_short_only_t():
    problem, x, _ = _native()
    _, delta = problem.counterexample(x)
    mapping = NativeAffineBoundaryMap(problem, np.zeros((50, 100)), "no_correction_anchor")
    report = mapping.measure(x, {"outside_only": delta}, injection_design_identity="explicit_outside_counterexample")
    row = report["erp_increments"][0]
    assert row["short_input_increment_norm"] == 0
    assert row["short_reference"]["relative_error"] is None
    assert row["context_reference"]["relative_error"] == 1.
    assert row["context_reference"]["peak_latency_error_seconds_per_channel"] == [None, None]
    assert report["requested_injections"] == 1 and report["defined_short_relative_errors"] == 0


def test_zero_filter_does_not_fabricate_relative_gain_or_reject_an_identified_zero_functional():
    problem, x, injections = _native(zero_operator=True)
    assert problem.report()["status"] == "identified"
    np.testing.assert_array_equal(problem.recover(problem.observe(x)), np.zeros((25, 2)))
    report = NativeAffineBoundaryMap(problem, np.zeros((50, 100)), "zero_transfer_counterexample").measure(
        x, injections, injection_design_identity="independent_zero_reference_challenge")
    assert report["white_noise_gain"]["mean_relative_to_short"] is None
    assert report["white_noise_gain"]["rank"] == 0
    assert report["white_noise_gain"]["condition_in_identified_range"] is None
    assert report["defined_short_relative_errors"] == report["defined_context_relative_errors"] == 0


def test_inputs_and_reports_are_immutable_and_serialize_without_nonfinite_null_surrogates():
    observation, L = np.array([[1., 0.]]), np.array([[1., 0.]])
    audit = _linear(observation, L)
    observation[:] = 999
    L[:] = 888
    np.testing.assert_array_equal(audit.recover([2.]), [2.])
    with pytest.raises(ValueError):
        audit.observation_matrix.setflags(write=True)
    with pytest.raises(TypeError):
        audit.report()["status"] = "forged"
    problem, x, injections = _native()
    correction = np.zeros((50, 100))
    mapping = NativeAffineBoundaryMap(problem, correction, "immutable_zero_anchor")
    correction[:] = 100
    expected = mapping.apply(x[10:60])
    for array in (mapping.correction_matrix, mapping.baseline_matrix, mapping.complete_matrix):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(mapping.apply(x[10:60]), expected)
    json.dumps(mapping.measure(x, injections, injection_design_identity="fixed"), allow_nan=False)


@pytest.mark.parametrize("bad", [np.array([[1j, 0]]), np.array([[np.nan, 1.]]), np.empty((0, 2))])
def test_invalid_observation_maps_are_rejected_without_lossy_complex_cast(bad):
    with pytest.raises(ValueError):
        _linear(bad, [[1., 0.]])


def test_native_scope_availability_layout_and_budget_failures_are_explicit():
    problem, x, _ = _native()
    with pytest.raises(ValueError, match="availability"):
        NativeBoundaryObservability(problem.grid, problem.short_operator, problem.context_operator,
                                    (1010, 1060), (1020, 1045), ((1010, 1060),), 1059)
    with pytest.raises(ValueError, match="included"):
        NativeBoundaryObservability(problem.grid, problem.short_operator, problem.context_operator,
                                    (1010, 1060), (1020, 1045), ((1020, 1060),), 1060)
    with pytest.raises(ValueError, match="byte budget"):
        NativeAffineBoundaryMap(problem, np.zeros((50, 100)), "too_large", byte_budget=8)
    mapping = NativeAffineBoundaryMap(problem, np.zeros((50, 100)), "actual_short_only")
    with pytest.raises(ValueError, match="native sample/channel grid"):
        mapping.apply(x)
    with pytest.raises(TypeError):
        mapping.apply(x[10:60], labels=[1])


def test_native_cli_rejects_event_authorization_outside_its_absolute_scope():
    script = Path(__file__).resolve().parents[2] / "experiments/audit_boundary_observability.py"
    specification = importlib.util.spec_from_file_location("b06_scope_cli", script)
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    scope = cli.RawSampleScope("explicit_C", 100, 200, (0, 1, 2))
    with pytest.raises(ValueError, match="outside their actual native"):
        cli._validate_calibration_events(scope, np.array([99, 125, 180]), np.array([1, 2]), 300)
    cli._validate_calibration_events(scope, np.array([100, 125, 180]), np.array([1, 2]), 300)


def test_native_cli_binds_archive_and_manifest_as_distinct_source_identities(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "experiments/audit_boundary_observability.py"
    specification = importlib.util.spec_from_file_location("b06_identity_cli", script)
    cli = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cli)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text("{}", encoding="utf-8")
    snapshot = {"archive_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    monkeypatch.setattr(cli, "_load_plan", lambda path: ({}, "c" * 64))
    monkeypatch.setattr(cli, "verified_executed_snapshot", lambda *args, **kwargs: snapshot)
    def executed(plan, path, destination, source_sha256, plan_sha256, **kwargs):
        assert source_sha256 == snapshot["archive_sha256"]
        assert kwargs["snapshot_manifest_sha256"] == snapshot["manifest_sha256"]
        return {"status": "completed", "observability": {"status": "identified"},
                "recovery": {"status": "identified_noiseless_map"}}
    monkeypatch.setattr(cli, "_run", executed)
    assert cli.main(["--plan", str(tmp_path / "plan.json"), "--snapshot", str(manifest),
                     "--output", str(tmp_path / "result")]) == 0
    execution = json.loads((tmp_path / "result/execution.json").read_bytes())
    assert execution["source_snapshot_sha256"] == snapshot["archive_sha256"]
    assert execution["source_snapshot_manifest_sha256"] == snapshot["manifest_sha256"]
