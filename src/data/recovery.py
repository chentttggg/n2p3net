"""Calibration-frozen reset uncertainty evaluated in the actual epoch space."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from numbers import Real
from typing import Any

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from data.epochs import PreprocessingSpec
from data.preprocess import (
    continuous_filter_segments,
    design_continuous_sos,
    transform_epoch_samples,
)
from research.contracts import semantic_sha256


def filter_sos(sample_rate_hz: float, preprocessing: PreprocessingSpec) -> np.ndarray:
    preprocessing.validate()
    if preprocessing.filter_method != "iir" or preprocessing.filter_phase != "forward":
        raise ValueError("epoch state recovery requires forward IIR preprocessing")
    return design_continuous_sos(sample_rate_hz, l_freq=preprocessing.l_freq,
        h_freq=preprocessing.h_freq, order=preprocessing.filter_order,
        notch_freqs_hz=preprocessing.notch_freqs_hz,
        notch_quality_factor=preprocessing.notch_quality_factor)


def modal_system(sos):
    sos = np.asarray(sos, dtype=float)
    size = 2 * len(sos)
    transition, output = np.empty((size, size)), np.empty(size)
    for index, state in enumerate(np.eye(size)):
        value, after = sosfilt(sos, [0.], zi=state.reshape(-1, 2))
        transition[:, index], output[index] = after.ravel(), value[0]
    poles, vectors = np.linalg.eig(transition)
    order = np.lexsort((poles.imag, poles.real))
    poles, vectors = poles[order], vectors[:, order]
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    condition = float(np.linalg.cond(vectors))
    if not np.isfinite(condition) or condition > 1e8 or np.max(np.abs(poles)) >= 1:
        raise ValueError("reset recovery requires a stable, well-conditioned modal system")
    return poles, vectors, np.linalg.inv(vectors), output, condition


_EPOCH_BASIS_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=4, typed=True)
def epoch_linear_projection(source_rate: float, preprocessing: PreprocessingSpec) -> np.ndarray:
    """Freeze the shared linear epoch transform for repeated state-error bounds."""
    preprocessing.validate()
    if (isinstance(source_rate, bool) or not isinstance(source_rate, Real)
            or not np.isfinite(source_rate) or source_rate <= 0):
        raise ValueError("epoch projection requires a finite positive source sample rate")
    samples = (round(preprocessing.tmax_ms / 1000 * source_rate)
               - round(preprocessing.tmin_ms / 1000 * source_rate) + 1)
    # Check the executable transform even when there is no data to transform.
    transform_epoch_samples(np.empty((0, samples)), source_rate, preprocessing, require_linear=True)
    row_bytes = samples * np.dtype(np.float64).itemsize
    if row_bytes > _EPOCH_BASIS_BYTES:
        raise ValueError("source epoch exceeds the bounded linear projection basis budget")
    matrix = np.empty((samples, preprocessing.n_times))
    chunk = min(samples, _EPOCH_BASIS_BYTES // row_bytes)
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        basis = np.zeros((count, samples))
        basis[np.arange(count), start + np.arange(count)] = 1
        matrix[start:start + count] = transform_epoch_samples(
            basis, source_rate, preprocessing, require_linear=True)
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class EpochStateRecovery:
    sos: tuple[tuple[float, ...], ...]
    sample_rate_hz: float
    preprocessing: PreprocessingSpec
    channel_names: tuple[str, ...]
    modal_envelope: tuple[tuple[float, ...], ...]
    tolerance_v: float
    fit_provenance: Mapping[str, Any]
    serialized_mode_identity: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        if (not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
                or not np.isfinite(self.tolerance_v) or self.tolerance_v <= 0
                or not self.fit_provenance or not self.channel_names
                or len(set(self.channel_names)) != len(self.channel_names)):
            raise ValueError("epoch recovery requires channels, tolerance and fit provenance")
        sos = np.asarray(self.sos)
        expected = filter_sos(self.sample_rate_hz, self.preprocessing)
        if sos.shape != expected.shape or not np.allclose(sos, expected, rtol=1e-12, atol=0):
            raise ValueError("epoch recovery SOS does not match the preprocessing contract")
        envelope = np.asarray(self.modal_envelope)
        if (envelope.shape != (len(self.channel_names), 2 * len(sos))
                or not np.isfinite(envelope).all() or (envelope < 0).any()):
            raise ValueError("epoch recovery modal envelope has invalid shape or values")
        samples = (round(self.preprocessing.tmax_ms / 1000 * self.sample_rate_hz)
                   - round(self.preprocessing.tmin_ms / 1000 * self.sample_rate_hz) + 1)
        transform_epoch_samples(np.empty((0, samples)), self.sample_rate_hz,
                                self.preprocessing, require_linear=True)
        if self.serialized_mode_identity is not None:
            identity = np.asarray(self.serialized_mode_identity, dtype=float)
            poles = modal_system(self.sos)[0]
            expected_identity = np.column_stack((poles.real, poles.imag))
            if (identity.shape != expected_identity.shape or not np.isfinite(identity).all()
                    or not np.allclose(identity, expected_identity, rtol=0, atol=1e-12)):
                raise ValueError("serialized modal identity differs from the frozen filter")

    @classmethod
    def from_envelope(cls, sos, sample_rate_hz, preprocessing, channels, envelope, *,
                      tolerance_v, fit_provenance, mode_identity=None):
        return cls(tuple(map(tuple, np.asarray(sos, dtype=float))), float(sample_rate_hz),
            preprocessing, tuple(channels), tuple(map(tuple, np.asarray(envelope, dtype=float))),
            float(tolerance_v), dict(fit_provenance),
            None if mode_identity is None else tuple(map(tuple, mode_identity)))

    @cached_property
    def operators(self):
        poles, vectors, inverse, output, condition = modal_system(self.sos)
        samples = (round(self.preprocessing.tmax_ms / 1000 * self.sample_rate_hz)
                   - round(self.preprocessing.tmin_ms / 1000 * self.sample_rate_hz) + 1)
        observable = (output @ vectors)[:, None] * poles[:, None] ** np.arange(samples)
        epoch = (transform_epoch_samples(observable.real, self.sample_rate_hz,
                                        self.preprocessing, require_linear=True)
                 + 1j * transform_epoch_samples(observable.imag, self.sample_rate_hz,
                                               self.preprocessing, require_linear=True))
        safety = 1 + condition * np.finfo(float).eps * 100
        return np.abs(poles), inverse, np.abs(epoch) * safety

    def epoch_error_bound(self, age_samples: int) -> np.ndarray:
        if type(age_samples) is not int or age_samples < 0:
            raise ValueError("epoch age must be a nonnegative integer sample count")
        radii, _, basis = self.operators
        return ((np.asarray(self.modal_envelope) * radii ** age_samples) @ basis).max(axis=1)

    def recovery_samples(self) -> tuple[int, ...]:
        limit = int(self.sample_rate_hz * 3600)
        if np.any(self.epoch_error_bound(limit) > self.tolerance_v):
            raise ValueError("epoch recovery envelope has no finite release within one hour")
        result = []
        for channel in range(len(self.channel_names)):
            low, high = 0, limit
            while low < high:
                middle = (low + high) // 2
                if self.epoch_error_bound(middle)[channel] <= self.tolerance_v:
                    high = middle
                else:
                    low = middle + 1
            result.append(low)
        return tuple(result)

    def recovery_certificate(self) -> dict[str, Any]:
        """Describe frozen state-error release costs without changing the model.

        Minimality concerns this model's monotone bound, not all possible
        uncertainty models. Continuing input contamination is a separate term.
        """
        waits = self.recovery_samples()
        left = round(self.preprocessing.tmin_ms / 1000 * self.sample_rate_hz)
        right = round(self.preprocessing.tmax_ms / 1000 * self.sample_rate_hz)
        rows = []
        for channel, (name, wait) in enumerate(zip(self.channel_names, waits, strict=True)):
            released = float(self.epoch_error_bound(wait)[channel])
            previous = None if wait == 0 else float(self.epoch_error_bound(wait - 1)[channel])
            rows.append({
                "channel": name,
                "release_samples": wait,
                "release_seconds": wait / self.sample_rate_hz,
                "state_error_bound_at_release_v": released,
                "previous_sample": None if wait == 0 else wait - 1,
                "state_error_bound_at_previous_sample_v": previous,
                "release_meets_tolerance": released <= self.tolerance_v,
                "previous_sample_exceeds_tolerance": None if previous is None else previous > self.tolerance_v,
                "earliest_stimulus_onset_samples_after_reset": wait - left,
                "earliest_complete_epoch_samples_after_reset": wait + right - left,
            })
        return {
            "schema": "n2p3_epoch_recovery_certificate/1",
            "recovery_model_digest": semantic_sha256(self.record()),
            "state_error_bound": {
                "covered_term": "homogeneous_response_from_reset_initial_state_error",
                "uncertainty_set": "frozen_per_channel_modal_coordinate_magnitude_envelope",
                "norm": "maximum_absolute_final_preprocessed_epoch_error_v",
                "tolerance_v": self.tolerance_v,
                "release_rule": "first_epoch_start_age_with_frozen_monotone_bound_at_or_below_tolerance",
                "minimality": "minimal_sample_for_this_frozen_bound",
                "age_origin": "first_source_sample_after_filter_reset",
                "age_endpoint": "first_source_sample_of_epoch_not_stimulus_onset",
                "covers_all_later_epoch_starts": True,
            },
            "continuing_input_contamination": {
                "covered": False,
                "excluded_term": "forced_response_from_continuing_input_contamination",
                "required_evidence": "independent_raw_signal_quality_and_acquisition_support",
            },
            "epoch_source_support": {
                "sample_rate_hz": self.sample_rate_hz,
                "source_samples_including_both_endpoints": right - left + 1,
                "first_to_last_sample_seconds": (right - left) / self.sample_rate_hz,
                "stimulus_onset_offset_from_epoch_start_samples": -left,
                "availability": "source_support_lower_bound_only_delivery_and_quality_confirmation_still_required",
            },
            "conditions": [
                "actual_reset_state_error_lies_within_the_frozen_calibration_envelope",
                "executed_filter_and_final_epoch_operator_match_the_bound_model",
                "final_reference_and_frozen_eog_dependencies_are_projected_before_using_this_certificate",
                "acquisition_boundaries_channel_masks_and_raw_quality_gates_remain_enforced",
                "calibration_envelope_is_not_refitted_on_evaluation_or_future_recordings",
            ],
            "channels": rows,
        }

    def contains_state_error(self, error, channel):
        return bool(np.all(np.abs(self.operators[1] @ np.asarray(error))
                           <= np.asarray(self.modal_envelope)[channel]))

    def project(self, mixing, channels):
        mixing = np.asarray(mixing, dtype=float)
        if mixing.shape != (len(channels), len(self.channel_names)) or not np.isfinite(mixing).all():
            raise ValueError("recovery mixing must map named source channels to output channels")
        return self.from_envelope(self.sos, self.sample_rate_hz, self.preprocessing, channels,
            np.abs(mixing) @ np.asarray(self.modal_envelope), tolerance_v=self.tolerance_v,
            fit_provenance={**self.fit_provenance, "projection": mixing.tolist(),
                            "native_recovery_digest": semantic_sha256(self.record())},
            mode_identity=self.serialized_mode_identity)

    def record(self):
        # Preserve the attested identity across LAPACK rounding differences.
        if self.serialized_mode_identity is None:
            poles = modal_system(self.sos)[0]
            identity = [[float(value.real), float(value.imag)] for value in poles]
        else:
            identity = [list(value) for value in self.serialized_mode_identity]
        return {"schema": "n2p3_epoch_state_recovery/2", "sos": [list(row) for row in self.sos],
                "sample_rate_hz": self.sample_rate_hz, "preprocessing": self.preprocessing.record(),
                "channel_names": list(self.channel_names),
                "modal_envelope": [list(row) for row in self.modal_envelope],
                "tolerance_v": self.tolerance_v, "fit_provenance": dict(self.fit_provenance),
                "mode_order": "eigenvalue_real_then_imaginary_unit_vector_norm",
                "mode_identity": identity,
                "interpretation": "conditional_on_calibration_empirical_state_envelope",
                "operator": "actual_epoch_resample_crop_mean_baseline_modal_envelope"}


def epoch_recovery_from_record(record):
    from data.bidirectional import BIDIRECTIONAL_RECOVERY_SCHEMA, BidirectionalStateRecovery

    if record.get("schema") == BIDIRECTIONAL_RECOVERY_SCHEMA:
        model = BidirectionalStateRecovery.from_envelope(record["sos"], record["sample_rate_hz"],
            PreprocessingSpec(**record["preprocessing"]), record["channel_names"], record["modal_envelope"],
            tolerance_v=record["tolerance_v"], fit_provenance=record["fit_provenance"],
            mode_identity=record.get("mode_identity"))
        if semantic_sha256(model.record()) != semantic_sha256(record):
            raise ValueError("invalid bidirectional state recovery record")
        return model
    model = EpochStateRecovery.from_envelope(record["sos"], record["sample_rate_hz"],
        PreprocessingSpec(**record["preprocessing"]), record["channel_names"], record["modal_envelope"],
        tolerance_v=record["tolerance_v"], fit_provenance=record["fit_provenance"],
        mode_identity=record.get("mode_identity"))
    if semantic_sha256(model.record()) != semantic_sha256(record):
        raise ValueError("invalid epoch state recovery record")
    return model


def fit_epoch_state_recovery(raw, *, preprocessing, channels, fit_sample_mask,
                             tolerance_v, inflation, warmup_samples, probe_step_samples,
                             fit_provenance):
    if preprocessing.filter_phase == "zero":
        from data.bidirectional import fit_bidirectional_state_recovery

        return fit_bidirectional_state_recovery(raw, preprocessing=preprocessing, channels=channels,
            fit_sample_mask=fit_sample_mask, tolerance_v=tolerance_v, inflation=inflation,
            warmup_samples=warmup_samples, probe_step_samples=probe_step_samples, fit_provenance=fit_provenance)
    mask = np.asarray(fit_sample_mask)
    if (mask.dtype != bool or mask.shape != (raw.n_times,) or inflation < 1
            or not np.isfinite(inflation) or type(warmup_samples) is not int or warmup_samples < 0
            or type(probe_step_samples) is not int or probe_step_samples < 1):
        raise ValueError("invalid explicit state-envelope calibration scope or sampling parameters")
    sos = filter_sos(float(raw.info["sfreq"]), preprocessing)
    inverse = modal_system(sos)[2]
    initial = sosfilt_zi(sos)
    values = raw.get_data(picks=list(channels))
    bounds = np.zeros((len(channels), 2 * len(sos)))
    probes = 0
    for source_start, source_stop in continuous_filter_segments(raw):
        allowed = mask[source_start:source_stop]
        edges = np.flatnonzero(np.diff(np.r_[False, allowed, False])) + source_start
        for start, stop in edges.reshape(-1, 2):
            if stop - start <= warmup_samples:
                continue
            segment = values[:, start:stop] - values[:, start:start+1]
            state = np.zeros((len(sos), len(channels), 2))
            previous = 0
            for sample in range(warmup_samples, segment.shape[1], probe_step_samples):
                if sample > previous:
                    _, state = sosfilt(sos, segment[:, previous:sample], axis=-1, zi=state)
                error = initial[:, None, :] * segment[None, :, sample, None] - state
                coordinates = error.transpose(1, 0, 2).reshape(len(channels), -1) @ inverse.T
                bounds = np.maximum(bounds, np.abs(coordinates))
                previous = sample
                probes += 1
    if probes < 2 or not np.isfinite(bounds).all():
        raise ValueError("insufficient clean calibration history for a reset-state envelope")
    return EpochStateRecovery.from_envelope(sos, float(raw.info["sfreq"]), preprocessing,
        channels, bounds * inflation, tolerance_v=tolerance_v,
        fit_provenance={**fit_provenance, "probe_count": probes, "inflation": float(inflation),
                        "warmup_samples": warmup_samples, "probe_step_samples": probe_step_samples,
                        "fit_mask_digest": semantic_sha256(mask.tolist()),
                        "scope": "declared_clean_calibration_history_only", "labels_used": False})
