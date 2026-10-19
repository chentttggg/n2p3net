"""Deterministic source probes for V01.a/V02.a/V02.b; no Python refinement claim.

Only synthetic, in-memory EEG is consumed. The receipt binds the implementation
actually imported, independent counterexamples, and measured residuals. Lean
proves exact real-valued specifications; these checks exercise finite examples
of the Python bridge and never turn a numerical tolerance into exact PSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

# Set before importing numerical libraries. This probe never starts training.
for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.signal import butter, sosfilt, sosfiltfilt  # noqa: E402

from data.dataset import build_subject  # noqa: E402
from data.epochs import PreprocessingSpec  # noqa: E402
from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from data.preprocess import (  # noqa: E402
    RawSampleScope,
    crop_raw_scope,
    design_continuous_sos,
    execute_preprocessing,
    filter_bidirectional_segment,
    prefix_raw_scopes,
    source_epoch_support,
)
from transfer.calibration_split import calibration_inner_split  # noqa: E402
from transfer.continuous_decision import _event_design, _validate_record  # noqa: E402
from transfer.exact_covariance import ExactPropagatedCovariance  # noqa: E402
from transfer.noise_model import JointNoiseModel, validate_covariance  # noqa: E402
from transfer.raw_continuous import (  # noqa: E402
    ContinuousInputPlan,
    _calibration_view,
)
from transfer.temporal_noise import TemporalNoise  # noqa: E402

SOURCE_PATHS = (
    "src/data/preprocess.py",
    "src/data/dataset.py",
    "src/data/source_records.py",
    "src/data/finite_filter_operator.py",
    "src/transfer/raw_continuous.py",
    "src/transfer/calibration_split.py",
    "src/transfer/calibration_candidates.py",
    "src/transfer/noise_model.py",
    "src/transfer/exact_covariance.py",
    "src/transfer/continuous_decision.py",
    "src/transfer/temporal_noise.py",
)


def source_digests():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def _fixture():
    values = np.random.default_rng(20261016).normal(0, 4e-6, (3, 6500))
    raw = mne.io.RawArray(
        values, mne.create_info(["Fz", "Cz", "Pz"], 250, "eeg"), first_samp=701, verbose=False
    )
    labels = np.arange(20) % 2
    events = np.column_stack((701 + (np.arange(20) + 2) * 250, np.zeros(20, int), labels + 1))
    spec = PreprocessingSpec(
        filter_phase="zero", zero_phase_edge_policy="mne_reflect_limited", notch_freqs_hz=(50.0,)
    )
    return raw, events, labels, spec


def raw_scope_noninterference():
    raw, events, labels, spec = _fixture()
    scope, heldout = prefix_raw_scopes(
        raw, events, spec, calibration_events=5, maximum_calibration_events=8
    )
    changed = raw.copy()
    changed._data[:, scope.stop_sample - raw.first_samp :] = np.nan
    changed_labels = labels.copy()
    changed_labels[5:] = 999
    first = build_subject(raw, events, labels, preprocessing=spec, raw_scope=scope)
    other = build_subject(changed, events, changed_labels, preprocessing=spec, raw_scope=scope)
    np.testing.assert_array_equal(first.data, other.data)
    np.testing.assert_array_equal(first.labels, labels[:5])
    np.testing.assert_array_equal(first.labels, other.labels)
    assert first.event_timeline.n_events == 5
    # Generic epoch prefixes admit a response tail as waveform context, while
    # explicit event rows alone control labels. Do not equate that scope with
    # the continuous classifier's stronger before-next-onset sample rule.
    assert events[5, 0] < scope.stop_sample
    changed = raw.copy()
    changed._data[:, : heldout.start_sample - raw.first_samp] = np.nan
    changed_labels = labels.copy()
    changed_labels[:8] = 999
    first = build_subject(raw, events, labels, preprocessing=spec, raw_scope=heldout)
    other = build_subject(changed, events, changed_labels, preprocessing=spec, raw_scope=heldout)
    np.testing.assert_array_equal(first.data, other.data)
    np.testing.assert_array_equal(first.labels, other.labels)
    assert first.event_timeline.event_ids[0].endswith(":8")
    return {
        "calibration": scope.record(),
        "heldout": heldout.record(),
        "outside_values": "NaN",
        "outside_labels": 999,
        "waveforms_and_labels_bitwise_equal": True,
        "next_unauthorized_event_onset_inside_authorized_epoch_tail": True,
    }


def prefix_geometry_and_nested_scope():
    raw, events, _, spec = _fixture()
    # The planner has access only to geometry; no EEG/labels are present here.
    geometry = SimpleNamespace(first_samp=raw.first_samp, n_times=raw.n_times, info=raw.info)
    empty, heldout = prefix_raw_scopes(
        geometry, events, spec, calibration_events=0, maximum_calibration_events=12
    )
    outer, other_heldout = prefix_raw_scopes(
        geometry, events, spec, calibration_events=10, maximum_calibration_events=12
    )
    assert empty is None and other_heldout == heldout
    inner, validation = prefix_raw_scopes(
        geometry, events, spec, calibration_events=6, parent_scope=outer
    )
    assert inner.start_sample == outer.start_sample
    assert inner.stop_sample == validation.start_sample
    assert validation.stop_sample == outer.stop_sample
    assert set(inner.event_indices).isdisjoint(validation.event_indices)
    changed = raw.copy()
    changed._data[:, inner.stop_sample - raw.first_samp :] += 0.001
    np.testing.assert_array_equal(
        crop_raw_scope(raw, inner).get_data(), crop_raw_scope(changed, inner).get_data()
    )
    original = execute_preprocessing(raw, events, spec, raw_scope=inner)
    updated = execute_preprocessing(changed, events, spec, raw_scope=inner)
    np.testing.assert_array_equal(original.data, updated.data)
    assert not np.array_equal(
        execute_preprocessing(raw, events, spec, raw_scope=outer).data,
        execute_preprocessing(changed, events, spec, raw_scope=outer).data,
    )
    return {
        "K0_fit_scope_absent": True,
        "common_heldout_across_K": True,
        "inner": inner.record(),
        "validation": validation.record(),
        "inner_unchanged_and_outer_changed": True,
    }


def closed_endpoint_and_exact_crop():
    raw, events, _, spec = _fixture()
    onset = int(events[0, 0])
    starts, stops = source_epoch_support(events, 250.0, spec)
    assert starts[0] == onset - 50 and stops[0] == onset + 301
    scope = RawSampleScope("closed_support", onset - 50, onset + 301, (0,))
    valid = execute_preprocessing(raw, events, spec, raw_scope=scope)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        invalid = execute_preprocessing(
            raw, events, spec, raw_scope=replace(scope, stop_sample=onset + 300)
        )
    assert valid.n_epochs == 1 and invalid.n_epochs == 0
    assert invalid.event_statuses.tolist() == ["boundary_dropped"]
    rates = (128.0, 127.999, 512.0)
    for rate in rates:
        data = np.arange(3 * 1500, dtype=float).reshape(3, 1500) * 1e-9
        measured = mne.io.RawArray(
            data, mne.create_info(["Fz", "Cz", "Pz"], rate, "eeg"), first_samp=701, verbose=False
        )
        bounded = RawSampleScope("exact_crop", 877, 1532, (0,))
        cropped = crop_raw_scope(measured, bounded)
        np.testing.assert_array_equal(cropped.get_data(), data[:, 176:831])
        assert cropped.first_samp == 877 and cropped.n_times == 655
    return {
        "source_epoch_half_open": [int(starts[0]), int(stops[0])],
        "one_sample_short_is_boundary_dropped": True,
        "exact_crop_rates_hz": list(rates),
    }


def continuous_onset_censor():
    rng = np.random.default_rng(20261017)
    record = SimpleNamespace(
        eeg_uv=rng.normal(size=(1800, 3)),
        flash_sample=100 + 80 * np.arange(16),
        raw_is_target=np.arange(16) % 3 == 0,
        subject_id="synthetic_only",
        source_sha256="a" * 64,
    )
    plan = ContinuousInputPlan()
    sos = butter(2, [1.0, 30.0], fs=256.0, btype="bandpass", output="sos")
    budget, next_onset = 8, int(record.flash_sample[8])
    first, receipt = _calibration_view(record, budget, next_onset + 200, plan, sos, 29)
    changed_values = record.eeg_uv.copy()
    changed_values[next_onset:] = np.nan
    changed_labels = record.raw_is_target.copy()
    changed_labels[budget:] = ~changed_labels[budget:]
    changed = SimpleNamespace(
        **{**vars(record), "eeg_uv": changed_values, "raw_is_target": changed_labels}
    )
    second, updated = _calibration_view(changed, budget, next_onset + 200, plan, sos, 29)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.target_mask, second.target_mask)
    assert receipt == updated and len(first.target_mask) == budget
    assert len(first.samples) == next_onset and receipt["last_included_sample"] == next_onset - 1
    assert receipt["right_censored_event_indices"] and first.right_censored
    width = round(plan.response_seconds * 256.0)
    _validate_record(first.samples, first.onsets, 3, width, allow_right_censor=True)
    try:
        _validate_record(first.samples, first.onsets, 3, width, allow_right_censor=False)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete tails accepted without explicit right censoring")
    try:
        _validate_record(first.samples, np.array([next_onset]), 3, width, allow_right_censor=True)
    except ValueError:
        pass
    else:
        raise AssertionError("onset at exclusive stop was accepted")
    basis = np.arange(1.0, 17.0)[:, None]
    design = _event_design(np.array([next_onset - 10]), np.array([True]), next_onset, basis)
    np.testing.assert_array_equal(design[-10:, 0], basis[:10, 0])
    np.testing.assert_array_equal(design[-10:, 1], basis[:10, 0])
    assert np.count_nonzero(design[:-10]) == 0
    return {
        "budget": budget,
        "exclusive_stop": next_onset,
        "right_censored_event_indices": receipt["right_censored_event_indices"],
        "observed_response_samples": receipt["observed_response_samples"],
        "next_onset_and_incomplete_tail_rejected_without_correct_permission": True,
        "outside_samples_and_labels_do_not_change_actual_calibration_view": True,
    }


def calibration_mask_noninterference():
    onset = np.arange(36, dtype=float) * 2
    labels = np.arange(36) % 2
    available = onset + 0.8
    recordings = np.repeat("recording", 36).astype(object)
    allowed = np.arange(36) < 24
    first = calibration_inner_split(
        labels, onset, available, recordings, tmin_s=-0.2, calibration_mask=allowed
    )
    assert first.legal
    labels[~allowed], onset[~allowed], available[~allowed] = 999, np.nan, np.inf
    recordings[~allowed] = "unauthorized_recording"
    second = calibration_inner_split(
        labels, onset, available, recordings, tmin_s=-0.2, calibration_mask=allowed
    )
    assert first.record() == second.record()
    for key in ("training_mask", "validation_mask", "purged_mask"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    return {"legal_authorized_split": first.record(), "outside_mask_changes_ignored": True}


def df2t(sos, values, initial_state):
    """Independent scalar direct-form-II-transposed recurrence, not scipy code."""
    output, state = np.empty_like(values), initial_state.copy()
    for time_index in range(values.shape[1]):
        sample = values[:, time_index].copy()
        for section_index, row in enumerate(sos):
            result = row[0] * sample + state[section_index, :, 0]
            first = row[1] * sample - row[4] * result + state[section_index, :, 1]
            second = row[2] * sample - row[5] * result
            state[section_index, :, 0], state[section_index, :, 1] = first, second
            sample = result
        output[:, time_index] = sample
    return output, state


def constant_input_state(sos):
    """Solve each section's constant-input equilibrium by its DC gain."""
    state = np.empty((len(sos), 2))
    level = 1.0
    for index, row in enumerate(sos):
        result = row[:3].sum() / row[3:].sum() * level
        state[index] = result - row[0] * level, row[2] * level - row[5] * result
        level = result
    return state


def independent_bidirectional(sos, values, padding):
    pad = min(padding, values.shape[1] - 1)
    initial = constant_input_state(sos)[:, None, :]
    left = 2 * values[:, :1] - values[:, 1 : pad + 1][:, ::-1]
    right = 2 * values[:, -1:] - values[:, -pad - 1 : -1][:, ::-1] if pad else values[:, :0]
    extended = np.column_stack((left, values, right))
    state = initial * extended[:, 0][None, :, None]
    forward, _ = df2t(sos, extended, state)
    state = initial * forward[:, -1][None, :, None]
    backward, _ = df2t(sos, forward[:, ::-1], state)
    whole = backward[:, ::-1]
    return whole[:, pad : pad + values.shape[1]]


def finite_sos_recurrence_and_decomposition():
    sections = {
        "identity": np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "two_nontrivial_sections": np.array(
            [[0.3, 0.2, 0.1, 1.0, -0.5, 0.1], [0.7, -0.1, 0.2, 1.0, -0.2, 0.05]]
        ),
        "bandpass_with_notch": design_continuous_sos(
            256.0, l_freq=0.1, h_freq=30.0, order=4, notch_freqs_hz=(50.0,)
        ),
    }
    cases = []
    for filter_name, sos in sections.items():
        for samples, padding in ((1, 0), (2, 99), (19, 0), (19, 7), (65, 99)):
            rng = np.random.default_rng(50 + samples + padding)
            values = rng.normal(size=(samples, 3))
            operator = FiniteFilterOperator(tuple(map(tuple, sos)), padding)
            actual = filter_bidirectional_segment(sos, values.T, padding_limit=padding)
            independent = independent_bidirectional(sos, values.T, padding).T
            scipy_reference = sosfiltfilt(
                sos, values, axis=0, padtype="odd", padlen=min(padding, samples - 1)
            )
            np.testing.assert_allclose(actual.data.T, independent, rtol=2e-9, atol=2e-9)
            np.testing.assert_allclose(actual.data.T, scipy_reference, rtol=1e-11, atol=1e-11)
            zero, basis, states = operator.decompose(values)
            reconstruction = zero + basis @ states
            np.testing.assert_allclose(reconstruction, actual.data.T, rtol=2e-9, atol=2e-9)
            # Independently bind the flattened Python coordinates to the
            # section/coordinate sum type used by the Lean matrix theorem.
            order = 2 * len(sos)
            independent_states = np.empty((2 * order, values.shape[1]))
            independent_basis = np.empty_like(basis)
            for section_index in range(len(sos)):
                for coordinate in range(2):
                    column = 2 * section_index + coordinate
                    independent_states[column] = actual.left_state[section_index, :, coordinate]
                    independent_states[order + column] = actual.right_state[
                        section_index, :, coordinate
                    ]
                    seed = np.zeros((len(sos), 1, 2))
                    seed[section_index, 0, coordinate] = 1.0
                    response, _ = df2t(sos, np.zeros((1, samples)), seed)
                    left_response, _ = df2t(sos, response[:, ::-1], np.zeros_like(seed))
                    independent_basis[:, column] = left_response[0, ::-1]
                    independent_basis[:, order + column] = response[0, ::-1]
            np.testing.assert_array_equal(states, independent_states)
            np.testing.assert_allclose(basis, independent_basis, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                operator.apply(1.3 * values + 0.7 * values[::-1]),
                1.3 * operator.apply(values) + 0.7 * operator.apply(values[::-1]),
                rtol=2e-9,
                atol=2e-9,
            )
            identity = np.eye(samples)
            causal = sosfilt(sos, identity, axis=0)
            np.testing.assert_allclose(
                operator.zero_state(identity), causal.T @ causal, rtol=1e-12, atol=1e-12
            )
            # Test arbitrary nonzero DF2T initial states separately from padding.
            initial = rng.normal(size=(len(sos), 3, 2))
            manual, manual_end = df2t(sos, values.T, initial)
            expected, expected_end = sosfilt(sos, values.T, axis=-1, zi=initial)
            np.testing.assert_allclose(manual, expected, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(manual_end, expected_end, rtol=1e-13, atol=1e-13)
            assert basis.shape == (samples, 4 * len(sos))
            cases.append(
                {
                    "filter": filter_name,
                    "sections": len(sos),
                    "samples": samples,
                    "padding_limit": padding,
                    "actual_padding": actual.padding_samples,
                    "state_basis_columns": basis.shape[1],
                    "state_basis_numerical_rank": int(np.linalg.matrix_rank(basis)),
                    "explicit_section_coordinate_order_verified": True,
                    "manual_recurrence_max_abs_error": float(
                        np.max(abs(independent - actual.data.T))
                    ),
                    "decomposition_max_abs_error": float(
                        np.max(abs(reconstruction - actual.data.T))
                    ),
                }
            )
    # Two explicitly false shortcuts must remain distinguishable from the result.
    operator = FiniteFilterOperator(tuple(map(tuple, sections["bandpass_with_notch"])), 90)
    matrix = operator.dense_matrix(101)
    asymmetry = float(np.linalg.norm(matrix - matrix.T))
    values = np.random.default_rng(302).normal(size=(301, 2))
    crop_commutator = float(
        np.linalg.norm(operator.apply(values)[80:221] - operator.apply(values[80:221]))
    )
    assert asymmetry > 0.1 and crop_commutator > 0.1
    assert any(case["state_basis_numerical_rank"] < case["state_basis_columns"] for case in cases)
    return {
        "cases": cases,
        "full_reflection_operator_asymmetry_norm": asymmetry,
        "filter_crop_commutator_norm": crop_commutator,
        "four_S_columns_do_not_claim_four_S_rank": True,
    }


def deterministic_state_joint_covariance():
    sos = butter(2, [2.0, 20.0], fs=64.0, btype="bandpass", output="sos")
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), 7)
    values = np.random.default_rng(650).normal(size=(17, 400))
    _, basis, states = operator.decompose(values)
    zero_matrix = operator.zero_state(np.eye(17))
    state_map = operator.decompose(np.eye(17))[2]
    np.testing.assert_allclose(states, state_map @ values, atol=1e-13, rtol=1e-13)
    model = JointNoiseModel.fit(
        values.T,
        states.T,
        fit_scope="source_training",
        scope_id="synthetic_only",
        draw_ids=[f"draw-{index}" for index in range(values.shape[1])],
        shrinkage=0.0,
    )
    propagated = model.project(zero_matrix, basis).covariance
    dense = operator.dense_matrix(17)
    expected = dense @ np.cov(values, bias=True) @ dense.T
    np.testing.assert_allclose(propagated, expected, atol=1e-13, rtol=1e-12)
    marginal_only = (
        zero_matrix @ model.covariance[:17, :17] @ zero_matrix.T
        + basis @ model.covariance[17:, 17:] @ basis.T
    )
    omitted_cross_error = float(np.linalg.norm(marginal_only - propagated))
    assert omitted_cross_error > 0.1
    # Stacked rows retain cross-epoch blocks; independent epochs are not assumed.
    stacked_zero, stacked_basis = zero_matrix[[0, 1, 1, 2]], basis[[0, 1, 1, 2]]
    stacked = model.project(stacked_zero, stacked_basis).covariance
    np.testing.assert_allclose(stacked[1], stacked[2], atol=1e-14)
    return {
        "actual_deterministic_states_used": True,
        "joint_propagation_max_abs_error": float(np.max(abs(propagated - expected))),
        "omitting_raw_state_cross_covariance_error_norm": omitted_cross_error,
        "shared_epoch_row_covariance_retained": True,
    }


def approximate_psd_admission_counterexample():
    # This input satisfies the runtime tolerance, but is not an exact covariance.
    covariance = np.diag([1.0, -5e-11])
    admitted = validate_covariance(covariance, name="synthetic tolerance counterexample")
    model = JointNoiseModel.fit(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([[0.0], [1.0], [2.0]]),
        fit_scope="source_training",
        scope_id="synthetic_only",
        draw_ids=["a", "b", "c"],
        shrinkage=0.0,
    )
    model = replace(model, covariance=admitted)
    scale = 1e6
    projected = model.project(np.zeros((1, 1)), np.array([[scale]])).covariance
    epsilon = 5e-11
    amplified_bound = epsilon * scale**2
    value = float(projected[0, 0])
    np.testing.assert_allclose(value, -amplified_bound, rtol=1e-15)
    assert value < 0
    for ridge in (1.0, amplified_bound):
        try:
            np.linalg.cholesky(projected + ridge * np.eye(1))
        except np.linalg.LinAlgError:
            pass
        else:
            raise AssertionError("insufficient positive ridge unexpectedly guarantees PD")
    np.linalg.cholesky(projected + (amplified_bound + 1.0) * np.eye(1))
    # For symmetric Σ >= -εI, test q(MΣMᵀ,v) >= -ε ||Mᵀv||².
    rng = np.random.default_rng(422)
    matrix = rng.normal(size=(5, 2))
    direction = rng.normal(size=5)
    error_lower_bound = -epsilon * np.linalg.norm(matrix.T @ direction) ** 2
    quadratic = float(direction @ matrix @ admitted @ matrix.T @ direction)
    assert quadratic >= error_lower_bound - 1e-12
    # Certified residual bounds are distinct from NumPy's unknown roundoff.
    # This deliberately injected symmetric residual has known operator norm.
    amplified = model.project(np.array([[1.0], [0.0]]), np.array([[0.0], [3.0]])).covariance
    eta = 0.25
    residual = -eta * np.eye(2)
    perturbed = amplified + residual
    combined_bound = epsilon * 3.0**2 + eta
    np.testing.assert_allclose(
        np.linalg.eigvalsh(perturbed).min(), -combined_bound, atol=1e-15, rtol=0
    )
    np.linalg.cholesky(perturbed + (combined_bound + 0.1) * np.eye(2))
    try:
        validate_covariance(np.diag([1.0, -2e-10]), name="outside tolerance")
    except ValueError:
        pass
    else:
        raise AssertionError("outside-tolerance indefinite matrix was accepted")
    return {
        "accepted_min_eigenvalue": float(np.linalg.eigvalsh(admitted).min()),
        "runtime_relative_tolerance": 1e-10,
        "projection_scale": scale,
        "projected_negative_variance": value,
        "sharp_amplified_lower_bound": -amplified_bound,
        "injected_symmetric_residual_norm": eta,
        "amplification_plus_residual_lower_bound": -combined_bound,
        "combined_strict_ridge_succeeds": True,
        "positive_ridge_1_and_equal_bound_insufficient": True,
        "strictly_greater_ridge_succeeds_for_witness": True,
        "claim": "runtime tolerance is approximate admission; no exact PSD or universal ridge guarantee",
    }


def exact_propagation_numerical_bridge():
    operator = FiniteFilterOperator(
        tuple(map(tuple, butter(2, [2.0, 15.0], fs=64.0, btype="bandpass", output="sos"))), 20
    )
    noise = TemporalNoise.from_parameters(
        [0.5], innovation_variance=0.8, reference_id="synthetic_only"
    )
    result = ExactPropagatedCovariance.compute(operator, noise, 40)
    dense = operator.dense_matrix(40)
    reference = dense @ result.K_ar @ dense.T
    np.testing.assert_allclose(result.K_propagated, reference, atol=1e-10, rtol=1e-10)
    regularized = result.K_propagated + result.regularization_ridge * np.eye(40)
    factor_residual = float(
        np.max(abs(result.cholesky_propagated @ result.cholesky_propagated.T - regularized))
    )
    np.testing.assert_allclose(
        result.cholesky_propagated @ result.cholesky_propagated.T,
        regularized,
        atol=1e-10,
        rtol=1e-10,
    )
    result.assert_integrity()
    return {
        "dense_covariance_max_abs_error": float(np.max(abs(reference - result.K_propagated))),
        "raw_min_eigenvalue": float(result.eigenvalues.min()),
        "ridge": float(result.regularization_ridge),
        "rank": int(result.numerical_rank),
        "null_dimension": int(result.null_dimension),
        "regularized_factor_max_abs_error": factor_residual,
        "factor_targets_ridge_regularized_matrix": True,
        "universal_float_roundoff_bound_claimed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "doc/assets/lean_first_slice_20261016/source_review.json",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    before = source_digests()
    probes = []
    for probe in (
        raw_scope_noninterference,
        prefix_geometry_and_nested_scope,
        closed_endpoint_and_exact_crop,
        continuous_onset_censor,
        calibration_mask_noninterference,
        finite_sos_recurrence_and_decomposition,
        deterministic_state_joint_covariance,
        approximate_psd_admission_counterexample,
        exact_propagation_numerical_bridge,
    ):
        probe_start = time.perf_counter()
        try:
            details = probe()
            probes.append(
                {
                    "id": probe.__name__,
                    "status": "passed",
                    "details": details,
                    "seconds": round(time.perf_counter() - probe_start, 6),
                }
            )
        except Exception as error:
            probes.append(
                {
                    "id": probe.__name__,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": round(time.perf_counter() - probe_start, 6),
                }
            )
    after = source_digests()
    stable = before == after
    passed = stable and all(probe["status"] == "passed" for probe in probes)
    receipt = {
        "schema": "n2p3_lean_first_slice_source_review/1",
        "evidence_scope": "software_correctness_synthetic_only",
        "formal_refinement_claimed": False,
        "training_launched": False,
        "original_data_read": False,
        "source_unchanged_during_probes": stable,
        "source_sha256": before,
        "source_sha256_after": after,
        "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "numerical_thread_environment": {
                key: os.environ[key]
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "probes": probes,
        "passed": passed,
        "seconds": round(time.perf_counter() - started, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "probes": len(probes),
                "seconds": receipt["seconds"],
                "output": str(args.output),
                "failures": [probe for probe in probes if probe["status"] != "passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
