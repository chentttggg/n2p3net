"""Restore original output charts; fresh host diagnostics do not republish them."""
from __future__ import annotations

import copy

import numpy as np
import pytest

import transfer.spatial_signal_prior as spatial
from tests.transfer.test_spatial_exact_factors import generic_source
from transfer.spatial_signal_prior import SpatialLayout, SpatialObservation, SpatialSignalPosterior


def layout():
    return SpatialLayout(("Fz", "Cz", "Pz"), "actual-reference", "1" * 64, ("EEG",) * 3)


def drift_svd(monkeypatch):
    original = np.linalg.svd

    def shifted(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("compute_uv", True) is False:
            result = result.copy()
            result[0] = np.nextafter(result[0], np.inf)
        return result

    monkeypatch.setattr(np.linalg, "svd", shifted)


def test_host_svd_one_ulp_drift_preserves_original_numeric_observation(monkeypatch):
    observation = SpatialObservation(layout(), np.array([[1., .3, -.7], [.4, 1., -.2]]),
                                     ("actual-0", "actual-1"), "actual-output")
    saved = observation.record()
    drift_svd(monkeypatch)
    restored = SpatialObservation.from_record(saved)
    assert restored.record() == saved
    np.testing.assert_array_equal(restored.normalized_range_map, observation.normalized_range_map)
    np.testing.assert_array_equal(restored.normalized_support_basis, observation.normalized_support_basis)


@pytest.mark.parametrize("gains", [[1., -.5, 2.], [1., 1e20, 1e20], [1e-20, 1., 1.]])
def test_car_original_householder_chart_and_volume_roundtrip(gains, monkeypatch):
    observation = SpatialObservation.average_reference(layout()).with_output_gains(gains)
    saved = observation.record()
    drift_svd(monkeypatch)

    def no_householder(*args, **kwargs):
        raise AssertionError("load must retain the original Householder chart")

    monkeypatch.setattr(spatial, "_null_complement", no_householder)
    restored = SpatialObservation.from_record(saved)
    assert restored.record() == saved
    np.testing.assert_array_equal(restored.support_basis, observation.support_basis)
    np.testing.assert_array_equal(restored.normalized_matrix, observation.normalized_matrix)
    np.testing.assert_array_equal(restored.normalized_range_map, observation.normalized_range_map)
    assert restored.coordinate_log_volume == observation.coordinate_log_volume


@pytest.mark.parametrize("field", ["support_basis", "normalized_support_basis", "normalized_matrix",
                                   "normalized_range_map", "coordinate_log_volume", "equilibration_scales"])
def test_changed_observation_state_must_satisfy_its_actual_generation(field):
    observation = SpatialObservation.average_reference(layout()).with_output_gains([1., -.5, 2.])
    saved = copy.deepcopy(observation.record())
    if field in {"support_basis", "normalized_support_basis"}:
        # Preserve orthonormal columns, but rotate one into the actual null.
        q = np.asarray(saved[field])
        signs = np.sign(np.asarray(saved["declaration"]["row_gains"]))
        null = signs / np.sqrt(3) if field == "normalized_support_basis" else (
            signs * (observation.equilibration_scales.min() / observation.equilibration_scales))
        null /= np.linalg.norm(null)
        q[:, 0] = np.cos(.1) * q[:, 0] + np.sin(.1) * null
        saved[field] = q.tolist()
        reason = "declared reference null"
    elif field in {"normalized_matrix", "normalized_range_map"}:
        saved[field][0][0] += .05
        reason = "original.*generation"
    elif field == "coordinate_log_volume":
        saved[field] += .1
        reason = "coordinate volume"
    else:
        saved[field][0] *= 2
        reason = "row scales"
    with pytest.raises(ValueError, match=reason):
        SpatialObservation.from_record(saved)


def test_source_posterior_replay_preserves_state_under_host_diagnostic_drift(monkeypatch):
    posterior, *_ = generic_source()
    saved = posterior.record()
    drift_svd(monkeypatch)
    restored = SpatialSignalPosterior.from_record(saved)
    assert restored.record() == saved
    np.testing.assert_array_equal(restored.precision_root, posterior.precision_root)
    np.testing.assert_array_equal(restored.mean, posterior.mean)
