"""Exact joint temporal row-space and independent nuisance elimination."""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from tests.transfer.test_spatial_exact_factors import generic_source
from transfer.noise_model import metadata_digest
from transfer.spatial_signal_prior import (
    SpatialLayout,
    SpatialLikelihoodDependence,
    SpatialObservation,
    SpatialPriorIdentificationError,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    _exact_structural_null,
    _ExactRowSpace,
    _SpatialStructuralFactors,
)


def factor(design, nuisance=None, mapping=None):
    channels = 1 if mapping is None else np.shape(mapping)[1]
    layout = SpatialLayout(tuple(f"EEG-{i}" for i in range(channels)), "actual-ref", "a" * 64, ("EEG",) * channels)
    observation = SpatialObservation.native(layout) if mapping is None else SpatialObservation(layout, mapping,
        tuple(f"row-{i}" for i in range(len(mapping))), "explicit")
    return _SpatialStructuralFactors(observation, np.asarray(design, dtype=float),
        None if nuisance is None else np.asarray(nuisance, dtype=float))


def raw_physical_rows(factors, width):
    rows = []
    local_widths = [(0 if value.nuisance is None else value.nuisance.shape[1]) * len(value.spatial_map) for value in factors]
    offset = 0
    for value, local_width in zip(factors, local_widths, strict=True):
        operator = []
        for observed, a in enumerate(value.spatial_map):
            for time, d in enumerate(value.design):
                beta = [coefficient * Fraction.from_float(float(item)) for coefficient in a for item in d]
                nuisance = [Fraction(0)] * sum(local_widths)
                if local_width:
                    n = value.nuisance.shape[1]
                    nuisance[offset + observed*n:offset + (observed+1)*n] = [Fraction.from_float(float(item)) for item in value.nuisance[time]]
                operator.append(beta + nuisance)
        rows.extend(operator)
        offset += local_width
    # Independent pure-Fraction elimination of the full beta/local-U law.
    rank, pivots = 0, []
    for column in range(width + sum(local_widths)):
        selected = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [item/divisor for item in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][column]:
                multiple = rows[i][column]
                rows[i] = [left-multiple*right for left, right in zip(rows[i], rows[rank], strict=True)]
        pivots.append(column)
        rank += 1
    null = []
    for column in range(width + sum(local_widths)):
        if column in pivots:
            continue
        vector = [Fraction(0)] * (width + sum(local_widths))
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = -rows[row][column]
        if any(vector[:width]):
            null.append(tuple(vector[:width]))
    return null


@pytest.mark.parametrize("case", ["same_D_Z", "independent_record_U", "joint_records", "record_pairing", "rational_quotient"])
def test_compressed_kernel_matches_original_beta_and_independent_u(case):
    if case == "same_D_Z":
        values, width = [factor([[1.], [2.]], [[1.], [2.]])], 1
    elif case == "independent_record_U":
        values, width = [factor([[1.]], [[1.]]), factor([[2.]], [[1.]])], 1
    elif case == "joint_records":
        values, width = [factor([[1., 0.]]), factor([[0., 1.]])], 2
    elif case == "record_pairing":
        values, width = [factor([[1., 0.]], mapping=[[1., 0.]]), factor([[0., 1.]], mapping=[[0., 1.]])], 4
    else:
        values, width = [factor([[1., 3.], [2., 1.]], [[3.], [1.]])], 2
    null, _ = _exact_structural_null(values, width)
    oracle = raw_physical_rows(values, width)
    # Canonical beta basis can differ by sign from full-law free-U projection.
    # Exact annihilation plus equal dimensions certifies the same subspace.
    assert len(null) == len(oracle)
    for value in values:
        for a in value.spatial_map:
            for g in value.response_quotient:
                covector = [left*right for left in a for right in g]
                assert all(sum((left*right for left, right in zip(covector, vector, strict=True)), Fraction(0)) == 0
                           for vector in oracle)
    reversed_null, _ = _exact_structural_null(tuple(reversed(values)), width)
    assert null == reversed_null


def test_last_tiny_independent_temporal_row_is_not_skipped():
    design = np.ones((2400, 2))
    original = factor(design)
    assert len(_exact_structural_null((original,), 2)[0]) == 1
    design[-1, 1] += 2. ** -48
    changed = factor(design)
    assert changed.temporal_rowspace["rows_examined"] == len(design)
    assert not _exact_structural_null((changed,), 2)[0]
    assert changed.temporal_rowspace["full_column_span"] is True
    shuffled = factor(design[::-1])
    assert _exact_structural_null((shuffled,), 2)[0] == ()
    assert shuffled.temporal_rowspace["rows_examined"] == 2


def test_factorized_beta_width_has_no_old_512_column_theory_limit():
    value = factor(np.zeros((1, 513)))
    null, resolution = _exact_structural_null((value,), 513)
    assert len(null) == 513 and resolution["rank"] == 0


def test_empty_rowspace_still_admits_actual_null_certificate_storage(monkeypatch):
    monkeypatch.setattr("transfer.spatial_signal_prior.psutil.virtual_memory", lambda: SimpleNamespace(available=4096))
    space = _ExactRowSpace(64)
    assert not space.add([Fraction(0)] * 64)
    with pytest.raises(SpatialPriorIdentificationError, match="structural_rowspace_host_memory_unresolved"):
        space.null()


def test_actual_long_record_blue_and_original_save_load_exceed_old_dense_budget(tmp_path, monkeypatch):
    def no_diagnostic_null(self):
        raise AssertionError("real fit/BLUE/save/load must not materialize the diagnostic null basis")

    monkeypatch.setattr(SpatialSignalPosterior, "structural_null_basis", property(no_diagnostic_null))
    original = generic_source()
    prior, noise = original[0].prior, original[0].native_noise_covariance
    block = original[-1][0]
    repeats = 1000
    rows = len(block.samples) * repeats
    auth = dict(block.authorization)
    auth["raw_sample_support"] = [0, rows]
    long = SpatialResponseBlock(np.tile(block.samples, (repeats, 1)), np.tile(block.design, (repeats, 1)),
        block.observation, auth, np.tile(block.nuisance, (repeats, 1)))
    assert rows * (prior.mean.size + 2) > 250_000
    posterior = SpatialSignalPosterior.fit(prior, (long,), native_noise_covariance=noise,
        noise_binding=dict(original[0].noise_binding))
    dependence = SpatialLikelihoodDependence.independent(posterior, authority_sha256="e" * 64)
    measured = posterior.estimate_effect(original[1], dependence, temporal_noise=original[4])
    small = SpatialSignalPosterior.fit(prior, (block,), native_noise_covariance=noise,
        noise_binding=dict(original[0].noise_binding))
    control = small.estimate_effect(original[1], SpatialLikelihoodDependence.independent(small, authority_sha256="e" * 64),
        temporal_noise=original[4])
    np.testing.assert_allclose(measured.mean, control.mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(measured.covariance * repeats, control.covariance, rtol=1e-10, atol=1e-12)
    state = posterior.record()
    assert "structural_operators" not in state
    assert state["structural_factors"][0]["temporal_rowspace"]["joint_rank"] == 6
    assert state["structural_factors"][0]["temporal_rowspace"]["rows_examined"] == 6
    path = tmp_path / "original_long_source.json"
    posterior.save(path)
    measured.save(tmp_path / "actual_long_blue.json")
    restored = SpatialSignalPosterior.load(path, expected_artifact_sha256=state["artifact_sha256"])
    np.testing.assert_array_equal(restored.mean, posterior.mean)
    np.testing.assert_array_equal(restored.precision_root, posterior.precision_root)
    assert restored.structural_rowspace == posterior.structural_rowspace


def test_resigned_selected_rows_certificate_cannot_skip_an_actual_late_direction():
    posterior, *_ = generic_source()
    saved = posterior.record()
    certificate = saved["structural_factors"][0]["temporal_rowspace"]
    certificate["selected_original_rows"] = [0]
    saved["structural_factors"][0]["artifact_sha256"] = metadata_digest({key:value for key,value in saved["structural_factors"][0].items() if key!="artifact_sha256"})
    saved["artifact_sha256"] = metadata_digest({key:value for key,value in saved.items() if key!="artifact_sha256"})
    with pytest.raises(ValueError, match="physical C factors"):
        SpatialSignalPosterior.from_record(saved)
