"""Finite-record double-pass boundary uncertainty in the final epoch space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import mne
import numpy as np
from scipy.signal import sosfilt

from data.epochs import PreprocessingSpec
from data.preprocess import (
    continuous_filter_plan,
    continuous_filter_segments,
    design_continuous_sos,
    filter_bidirectional_segment,
    transform_epoch_samples,
)
from data.recovery import epoch_linear_projection, modal_system
from research.contracts import semantic_sha256

BIDIRECTIONAL_RECOVERY_SCHEMA = "n2p3_bidirectional_state_recovery/1"


@lru_cache(maxsize=2)
def _boundary_basis(sos_tuple, length):
    sos = np.asarray(sos_tuple)
    poles, vectors, _, output, condition = modal_system(sos)
    observable = (output @ vectors)[:, None] * poles[:, None] ** np.arange(length)
    left = sosfilt(sos, observable[:, ::-1], axis=-1)[:, ::-1]
    result = np.concatenate((left, observable[:, ::-1]), axis=0)
    result *= 1 + condition * np.finfo(float).eps * 100
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BidirectionalStateRecovery:
    sos: tuple[tuple[float, ...], ...]
    sample_rate_hz: float
    preprocessing: PreprocessingSpec
    channel_names: tuple[str, ...]
    modal_envelope: tuple[tuple[float, ...], ...]
    tolerance_v: float
    fit_provenance: dict[str, Any]
    serialized_mode_identity: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        self.preprocessing.validate()
        if (self.preprocessing.filter_phase != "zero"
                or self.preprocessing.zero_phase_edge_policy != "mne_reflect_limited"):
            raise ValueError("bidirectional recovery requires declared reflect_limited SOS")
        if (not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0
                or not np.isfinite(self.tolerance_v) or self.tolerance_v <= 0
                or not self.fit_provenance or not self.channel_names
                or len(set(self.channel_names)) != len(self.channel_names)):
            raise ValueError("bidirectional recovery requires physical geometry and calibration provenance")
        spec = self.preprocessing
        expected = design_continuous_sos(self.sample_rate_hz, l_freq=spec.l_freq, h_freq=spec.h_freq,
            order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
        sos = np.asarray(self.sos)
        if sos.shape != expected.shape or not np.allclose(sos, expected, rtol=1e-12, atol=0):
            raise ValueError("bidirectional recovery SOS differs from the executed preprocessing")
        envelope = np.asarray(self.modal_envelope)
        if (envelope.shape != (len(self.channel_names), 4 * len(sos))
                or not np.isfinite(envelope).all() or (envelope < 0).any()):
            raise ValueError("two boundary envelopes must match channels and both SOS state spaces")
        transform_epoch_samples(np.empty((0, self.epoch_samples)), self.sample_rate_hz,
                                self.preprocessing, require_linear=True)
        poles = modal_system(self.sos)[0]
        if self.serialized_mode_identity is not None:
            identity = np.asarray(self.serialized_mode_identity)
            if identity.shape != (len(poles), 2) or not np.allclose(
                    identity, np.column_stack((poles.real, poles.imag)), rtol=0, atol=1e-12):
                raise ValueError("serialized bidirectional mode identity differs from the filter")

    @classmethod
    def from_envelope(cls, sos, sample_rate_hz, preprocessing, channels, envelope, *,
                      tolerance_v, fit_provenance, mode_identity=None):
        return cls(tuple(map(tuple, np.asarray(sos, dtype=float))), float(sample_rate_hz),
                   preprocessing, tuple(channels), tuple(map(tuple, np.asarray(envelope, dtype=float))),
                   float(tolerance_v), dict(fit_provenance),
                   None if mode_identity is None else tuple(map(tuple, mode_identity)))

    @cached_property
    def epoch_samples(self):
        return (round(self.preprocessing.tmax_ms / 1000 * self.sample_rate_hz)
                - round(self.preprocessing.tmin_ms / 1000 * self.sample_rate_hz) + 1)

    def epoch_error_bounds(self, segment_samples: int, starts) -> np.ndarray:
        starts = np.asarray(starts)
        if (type(segment_samples) is not int or segment_samples < self.epoch_samples
                or starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer)
                or np.any(starts < 0) or np.any(starts + self.epoch_samples > segment_samples)):
            raise ValueError("bidirectional epoch support must lie inside its actual clean segment")
        basis = _boundary_basis(self.sos, segment_samples)
        projection = epoch_linear_projection(self.sample_rate_hz, self.preprocessing)
        envelope = np.asarray(self.modal_envelope)
        bounds = np.empty((len(starts), len(self.channel_names)))
        for row, start in enumerate(starts):
            cropped = basis[:, start:start+self.epoch_samples]
            mapped = cropped.real @ projection + 1j * (cropped.imag @ projection)
            bounds[row] = (envelope @ np.abs(mapped)).max(axis=1)
        return bounds

    def project(self, mixing, channels):
        mixing = np.asarray(mixing, dtype=float)
        if mixing.shape != (len(channels), len(self.channel_names)) or not np.isfinite(mixing).all():
            raise ValueError("recovery mixing must map named native channels to final output channels")
        return self.from_envelope(self.sos, self.sample_rate_hz, self.preprocessing, channels,
            np.abs(mixing) @ np.asarray(self.modal_envelope), tolerance_v=self.tolerance_v,
            fit_provenance={**self.fit_provenance, "projection": mixing.tolist(),
                            "native_recovery_digest": semantic_sha256(self.record())},
            mode_identity=self.serialized_mode_identity)

    def assess_epochs(self, raw, onset_samples, *, boundary_action="all_edges"):
        if float(raw.info["sfreq"]) != self.sample_rate_hz:
            raise ValueError("bidirectional recovery raw sample rate differs from calibration")
        starts = np.asarray(onset_samples) + round(self.preprocessing.tmin_ms / 1000 * self.sample_rate_hz)
        bounds = np.full((len(starts), len(self.channel_names)), np.inf)
        hard_bounds = bounds.copy()
        if boundary_action not in {"all_edges", "artifact_edges"}:
            raise ValueError("unknown bidirectional boundary_action")
        spans = {}
        segment_records = {}
        for channel, name in enumerate(self.channel_names):
            plan = continuous_filter_plan(raw, name)
            segment_records[name] = [span.record() for span in plan]
            for span in plan:
                left, right = span.start_sample, span.stop_sample
                rows = np.flatnonzero((starts >= left) & (starts+self.epoch_samples <= right))
                if len(rows):
                    required = boundary_action == "all_edges" or span.requires_recovery
                    spans.setdefault((left, right), []).append((channel, rows, required))
        for (left, right), members in spans.items():
            # Share the expensive epoch operator across channels with the same
            # finite span, including final reference/EOG dependency intersections.
            rows = np.unique(np.concatenate([rows for _, rows, _ in members]))
            computed = self.epoch_error_bounds(int(right-left), starts[rows]-left)
            for channel, selected, required in members:
                measured = computed[np.searchsorted(rows, selected), channel]
                finite = np.isfinite(measured) & (measured >= 0)
                bounds[selected, channel] = np.where(finite, measured, np.inf)
                # One artifact/unknown edge requires BOTH coupled state terms.
                # Natural-edge diagnostics still cannot hide a failed computation.
                hard_bounds[selected, channel] = np.where(finite, measured if required else 0., np.inf)
        usable = hard_bounds <= self.tolerance_v
        record = self.recovery_certificate()
        record.update(channel_names=list(self.channel_names), scheduled_events=len(starts),
                      usable_by_channel=usable.sum(axis=0).tolist(),
                      unsupported_by_channel=(~np.isfinite(bounds)).sum(axis=0).tolist(),
                      boundary_action=boundary_action,
                      filter_segment_boundaries=segment_records,
                      natural_boundary_interpretation="declared_finite_record_operator; sensitivity_is_diagnostic",
                      hard_error_bounds_v=[[float(x) if np.isfinite(x) else None for x in row] for row in hard_bounds],
                      epoch_error_bounds_v=[[float(x) if np.isfinite(x) else None for x in row] for row in bounds])
        return usable, record

    def recovery_certificate(self):
        return {"schema": "n2p3_bidirectional_epoch_boundary_certificate/1",
                "recovery_model_digest": semantic_sha256(self.record()),
                "covered_term": "two_effective_boundary_state_errors_of_actual_finite_double_pass",
                "uncertainty_set": "calibration_frozen_left_and_right_modal_magnitude_boxes",
                "norm": "maximum_absolute_final_resampled_mean_baselined_epoch_error_v",
                "tolerance_v": self.tolerance_v, "release_rule": "per_epoch_actual_segment_length_and_position",
                "monotone_release_time_assumed": False, "continuing_input_contamination_covered": False,
                "reference": "longer_finite_clean_calibration_context",
                "condition": "actual_effective_boundary_errors_must_lie_in_frozen_empirical_boxes"}

    def record(self):
        poles = modal_system(self.sos)[0]
        identity = (np.column_stack((poles.real, poles.imag)).tolist() if self.serialized_mode_identity is None
                    else [list(row) for row in self.serialized_mode_identity])
        return dict(schema=BIDIRECTIONAL_RECOVERY_SCHEMA, sos=[list(row) for row in self.sos],
                    sample_rate_hz=self.sample_rate_hz, preprocessing=self.preprocessing.record(),
                    channel_names=list(self.channel_names), modal_envelope=[list(row) for row in self.modal_envelope],
                    tolerance_v=self.tolerance_v, fit_provenance=dict(self.fit_provenance), mode_identity=identity,
                    mode_order="left_then_right_eigenvalue_real_then_imaginary_unit_vector_norm",
                    interpretation="conditional_on_calibration_empirical_effective_boundary_state_envelope")


def _states_at(sos, values, initial, positions):
    result, state, previous = {}, initial.copy(), 0
    for sample in sorted(set(positions)):
        if sample > previous:
            _, state = sosfilt(sos, values[:, previous:sample], axis=-1, zi=state)
        result[sample] = state.copy()
        previous = sample
    return result


def fit_bidirectional_state_recovery(raw, *, preprocessing, channels, fit_sample_mask, tolerance_v,
                                     inflation, warmup_samples, probe_step_samples, fit_provenance):
    mask = np.asarray(fit_sample_mask)
    if (mask.dtype != bool or mask.shape != (raw.n_times,) or not np.isfinite(inflation) or inflation < 1
            or type(warmup_samples) is not int or warmup_samples < 0
            or type(probe_step_samples) is not int or probe_step_samples < 1):
        raise ValueError("invalid bidirectional calibration scope or sampling parameters")
    rate = float(raw.info["sfreq"])
    spec = preprocessing
    sos = design_continuous_sos(rate, l_freq=spec.l_freq, h_freq=spec.h_freq, order=spec.filter_order,
        notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    padding = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    inverse = modal_system(sos)[2]
    values = raw.get_data(picks=list(channels))
    bounds = np.zeros((len(channels), 4*len(sos)))
    epoch_width = round(spec.tmax_ms/1000*rate)-round(spec.tmin_ms/1000*rate)+1
    counts = {}
    for a, b in continuous_filter_segments(raw):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[a:b], False])) + a
        for begin, end in edges.reshape(-1, 2):
            size = int(end-begin)
            maximum = size-2*warmup_samples
            if maximum < epoch_width:
                continue
            segment = values[:, begin:end] - values[:, begin:begin+1]
            reference = filter_bidirectional_segment(sos, segment, padding_limit=padding)
            lengths = np.unique(np.rint(np.geomspace(epoch_width, maximum, 5)).astype(int))
            probes = [(left, left+int(length)) for length in lengths
                      for left in range(warmup_samples, size-warmup_samples-int(length)+1, probe_step_samples)]
            forward_states = _states_at(sos, segment, reference.left_state, [left for left, _ in probes])
            backward_states = _states_at(sos, reference.forward_core[:, ::-1], reference.right_state,
                                         [size-right for _, right in probes])
            for left, right in probes:
                local = filter_bidirectional_segment(sos, segment[:, left:right], padding_limit=padding)
                coordinates = []
                for error in (local.left_state-forward_states[left], local.right_state-backward_states[size-right]):
                    coordinates.append(error.transpose(1, 0, 2).reshape(len(channels), -1) @ inverse.T)
                bounds = np.maximum(bounds, np.abs(np.concatenate(coordinates, axis=1)))
                counts[str(right-left)] = counts.get(str(right-left), 0)+1
    if sum(counts.values()) < 2 or not np.isfinite(bounds).all():
        raise ValueError("insufficient clean calibration context for two-boundary recovery probes")
    return BidirectionalStateRecovery.from_envelope(sos, rate, spec, channels, bounds*inflation,
        tolerance_v=tolerance_v, fit_provenance={**fit_provenance, "probe_count": sum(counts.values()),
            "probe_lengths_and_counts": counts, "warmup_samples_each_side": warmup_samples,
            "probe_step_samples": probe_step_samples, "inflation": float(inflation),
            "fit_mask_digest": semantic_sha256(mask.tolist()), "labels_used": False,
            "reference": "same_filter_longer_finite_clean_calibration_context",
            "boundary_state_definition": "before_first_real_forward_and_last_real_backward_sample"})
