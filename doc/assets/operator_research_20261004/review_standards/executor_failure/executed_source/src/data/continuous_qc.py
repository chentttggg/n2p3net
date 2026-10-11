"""Frozen raw-domain anomaly detection with explicit filter-state uncertainty.

Thresholds are development parameters, not hardware saturation limits. Unknown
initial-state error never produces an automatic recovery claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import sosfilt

from data.bidirectional import BidirectionalStateRecovery
from data.channel import canonical_channel_name
from data.preprocess import continuous_filter_segments
from data.recovery import EpochStateRecovery, epoch_linear_projection, epoch_recovery_from_record
from research.contracts import semantic_sha256
from research.execution import atomic_write_json

CONTINUOUS_QC_SCHEMA = "n2p3_continuous_qc/3"
CONTINUOUS_QC_CONSTANT_SCHEMA = "n2p3_continuous_qc/4"


@dataclass(frozen=True)
class FilterStateRecovery:
    """Conditional bound on every later output, including the epoch operator.

    The state-error box and operator gain need external validation evidence;
    fitting raw amplitude statistics does not establish either bound.
    """

    sos: tuple[tuple[float, ...], ...]
    state_error_bound_v: tuple[tuple[float, ...], ...]
    tolerance_v: float
    epoch_operator_gain: float
    evidence: str
    input_reference: str | None = None
    ocular_model_digest: str | None = None

    def __post_init__(self):
        sos, bounds = np.asarray(self.sos), np.asarray(self.state_error_bound_v)
        if not self.evidence.strip():
            raise ValueError("filter-state recovery requires explicit bound evidence")
        if (sos.ndim != 2 or sos.shape[1] != 6 or bounds.shape != (len(sos), 2)
                or not np.isfinite(sos).all() or not np.isfinite(bounds).all() or (bounds < 0).any()
                or not np.all(sos[:, 3] == 1) or not np.isfinite(self.tolerance_v)
                or self.tolerance_v <= 0 or not np.isfinite(self.epoch_operator_gain)
                or self.epoch_operator_gain < 1):
            raise ValueError("invalid SOS state-error box, tolerance or epoch operator gain")

    def required_samples(self, max_samples: int = 1_000_000) -> int | None:
        sos = np.asarray(self.sos)
        size = 2 * len(sos)
        transition, output = np.empty((size, size)), np.empty(size)
        for index, basis in enumerate(np.eye(size)):
            response, state = sosfilt(sos, [0.0], zi=basis.reshape(-1, 2))
            transition[:, index], output[index] = state.ravel(), response[0]
        poles, vectors = np.linalg.eig(transition)
        # Defective/ill-conditioned systems are an unresolved certificate, not
        # a reason to replace an infinite-tail bound with a finite simulation.
        condition = np.linalg.cond(vectors)
        if condition > 1e8 or np.max(np.abs(poles)) >= 1:
            return None
        inverse = np.linalg.inv(vectors)
        coefficients = np.abs(output @ vectors) * (
            np.abs(inverse) @ np.asarray(self.state_error_bound_v).ravel())
        coefficients *= self.epoch_operator_gain * (1 + condition * np.finfo(float).eps * 100)
        radii = np.abs(poles)
        def within(sample):
            return float(coefficients @ (radii ** sample)) <= self.tolerance_v
        if not within(max_samples):
            return None
        low, high = 0, max_samples
        while low < high:
            middle = (low + high) // 2
            if within(middle):
                high = middle
            else:
                low = middle + 1
        return low


@lru_cache(maxsize=16, typed=True)
def epoch_operator_gain(sample_rate_hz: float, preprocessing) -> float:
    """Induced infinity norm of the shared linear epoch transform."""
    return float(np.abs(epoch_linear_projection(sample_rate_hz, preprocessing)).sum(axis=0).max())


def validate_recovery_contract(model, preprocessing, *, source_reference, ocular_model=None):
    if model.epoch_recovery is not None:
        if semantic_sha256(asdict(preprocessing)) != semantic_sha256(asdict(model.epoch_recovery.preprocessing)):
            raise ValueError("empirical recovery epoch operator differs from executed preprocessing")
        return
    recovery = model.recovery
    if preprocessing.filter_phase == "zero":
        raise ValueError("channel-local zero-phase QC requires fitted bidirectional boundary recovery")
    if recovery is None:
        return
    if preprocessing.filter_method != "iir" or preprocessing.filter_phase != "forward":
        raise ValueError("state recovery certificate requires the bound forward SOS filter")
    from data.recovery import filter_sos
    sos = filter_sos(model.sample_rate_hz, preprocessing)
    if np.asarray(recovery.sos).shape != sos.shape or not np.allclose(recovery.sos, sos, rtol=1e-12, atol=0):
        raise ValueError("recovery certificate SOS differs from the executed filter")
    if recovery.epoch_operator_gain < epoch_operator_gain(model.sample_rate_hz, preprocessing):
        raise ValueError("recovery certificate understates the executed epoch operator gain")
    reference = source_reference if ocular_model is None else ocular_model.source_reference
    ocular_digest = None if ocular_model is None else semantic_sha256(ocular_model.record())
    if recovery.input_reference != reference or recovery.ocular_model_digest != ocular_digest:
        raise ValueError("recovery certificate reference/ocular preprocessing differs from executed input")


@dataclass(frozen=True)
class ContinuousQCSpec:
    difference_z: float
    difference_floor_v: float
    settle_samples: int
    channel_local: bool = False
    reference_channel: str | None = None
    include_eog: bool = False
    minimum_channel_fraction: float = 1.0
    range_window_samples: int | None = None
    range_quantile: float = .995
    range_inflation: float = 2.0
    range_floor_v: float = 50e-6
    artifact_pre_guard_samples: int = 0
    flatline_min_duration_s: float | None = None
    range_action: str = "reject"
    boundary_action: str = "all_edges"

    def __post_init__(self) -> None:
        if self.range_action not in {"reject", "diagnostic"}:
            raise ValueError("range_action must be reject or diagnostic")
        if self.boundary_action not in {"all_edges", "artifact_edges"}:
            raise ValueError("boundary_action must be all_edges or artifact_edges")
        for value in (self.difference_z, self.difference_floor_v):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("continuous QC thresholds must be finite and positive")
        if type(self.settle_samples) is not int or self.settle_samples < 1:
            raise ValueError("settle_samples must be a positive integer")
        if type(self.channel_local) is not bool or type(self.include_eog) is not bool:
            raise ValueError("QC channel and EOG switches must be boolean")
        if (type(self.artifact_pre_guard_samples) is not int or self.artifact_pre_guard_samples < 0
                or self.artifact_pre_guard_samples and not self.channel_local):
            raise ValueError("artifact pre-guard requires nonnegative samples and channel-local QC")
        if not 0 < self.minimum_channel_fraction <= 1:
            raise ValueError("minimum_channel_fraction must be in (0, 1]")
        if self.reference_channel is not None and not self.channel_local:
            raise ValueError("QC-owned reference requires explicit channel-local processing")
        if self.reference_channel is not None:
            object.__setattr__(self, "reference_channel", canonical_channel_name(self.reference_channel))
        if self.range_window_samples is not None and (
            not self.channel_local or type(self.range_window_samples) is not int or self.range_window_samples < 2
            or not .5 < self.range_quantile < 1 or not np.isfinite(self.range_inflation)
            or self.range_inflation < 1 or not np.isfinite(self.range_floor_v) or self.range_floor_v <= 0
        ):
            raise ValueError("invalid continuous range-monitor calibration parameters")
        if self.flatline_min_duration_s is not None and (
            not self.channel_local or isinstance(self.flatline_min_duration_s, bool)
            or not isinstance(self.flatline_min_duration_s, (int, float))
            or not np.isfinite(self.flatline_min_duration_s) or self.flatline_min_duration_s <= 0
        ):
            raise ValueError("raw exact-constant detection requires a positive duration and channel-local QC")


@dataclass(frozen=True)
class ContinuousQCInterval:
    start_sample: int
    end_sample: int
    detected_sample: int
    confirmed_sample: int
    channels: tuple[str, ...]
    reason: str
    reset_filter: bool


@dataclass(frozen=True)
class ContinuousQCResult:
    n_samples: int
    sample_rate_hz: float
    intervals: tuple[ContinuousQCInterval, ...]
    model_digest: str
    channel_names: tuple[str, ...] = ()
    channel_local: bool = False
    detection_lookahead_samples: int = 0

    @property
    def valid_sample_mask(self) -> np.ndarray:
        mask = np.ones(self.n_samples, dtype=bool)
        for interval in self.intervals:
            mask[interval.start_sample:interval.end_sample] = False
        return mask

    def record(self) -> dict[str, Any]:
        return {"schema": "n2p3_continuous_qc_intervals/1", "n_samples": self.n_samples,
                "sample_rate_hz": self.sample_rate_hz, "model_digest": self.model_digest,
                "intervals": [asdict(interval) for interval in self.intervals],
                "sample_interval": "half_open", "signal_unit": "V",
                "detection": "raw_anomaly_with_backdated_support_complete_recording"
                if self.detection_lookahead_samples else "causal_difference_no_backdating",
                "detection_lookahead_samples": self.detection_lookahead_samples,
                "disposition": "channel_mask" if self.channel_local else "global_epoch_exclusion_with_channel_local_reasons",
                "channel_names": list(self.channel_names)}

    def epoch_channel_mask(self, samples, *, channels, tmin_s, tmax_s):
        starts = np.asarray(samples) + round(tmin_s * self.sample_rate_hz)
        stops = np.asarray(samples) + round(tmax_s * self.sample_rate_hz) + 1
        valid = np.broadcast_to((starts >= 0) & (stops <= self.n_samples), (len(channels), len(starts))).T.copy()
        for row in self.intervals:
            overlap = (starts < row.end_sample) & (stops > row.start_sample)
            affected = [index for index, name in enumerate(channels) if not self.channel_local
                        or canonical_channel_name(name) in {canonical_channel_name(value) for value in row.channels}]
            if affected:
                valid[np.ix_(overlap, affected)] = False
        return valid

    def annotate(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if raw.n_times != self.n_samples or raw.info["sfreq"] != self.sample_rate_hz:
            raise ValueError("continuous QC ledger does not match raw sample geometry")
        output = raw.copy()
        if self.intervals:
            descriptions = (["QUALITY_LOCAL_ARTIFACT" if row.reset_filter else "QUALITY_LOCAL_RECOVERY"
                             for row in self.intervals] if self.channel_local else
                            ["BAD_continuous_artifact" if row.reset_filter else "BAD_continuous_recovery"
                             for row in self.intervals])
            output.annotations.append(
                [row.start_sample / self.sample_rate_hz + raw.first_time for row in self.intervals],
                [(row.end_sample - row.start_sample) / self.sample_rate_hz for row in self.intervals],
                descriptions,
                ch_names=[row.channels for row in self.intervals] if self.channel_local else None,
            )
        return output


@dataclass(frozen=True)
class FittedContinuousQC:
    spec: ContinuousQCSpec
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    difference_center_v: tuple[float, ...]
    difference_scale_v: tuple[float, ...]
    fit_provenance: Mapping[str, Any]
    recovery: FilterStateRecovery | None = None
    epoch_recovery: EpochStateRecovery | BidirectionalStateRecovery | None = None
    range_threshold_v: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        channels = len(self.channel_names)
        if not channels or len(set(self.channel_names)) != channels:
            raise ValueError("continuous QC needs unique measured channels")
        center, scale = np.asarray(self.difference_center_v), np.asarray(self.difference_scale_v)
        if (center.shape != (channels,) or scale.shape != (channels,)
                or not np.isfinite(center).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0) or not np.isfinite(self.sample_rate_hz)
                or self.sample_rate_hz <= 0 or not self.fit_provenance):
            raise ValueError("continuous QC fitted values or provenance are invalid")
        if self.epoch_recovery is not None and (
            not self.spec.channel_local or self.recovery is not None
            or self.epoch_recovery.channel_names != self.channel_names
            or self.epoch_recovery.sample_rate_hz != self.sample_rate_hz
        ):
            raise ValueError("epoch recovery requires matching native channels and exclusive local state policy")
        if self.epoch_recovery is not None:
            child = self.epoch_recovery.fit_provenance
            parent = self.fit_provenance
            for key in ("participant_key", "source_reference"):
                if not parent.get(key) or child.get(key) != parent[key]:
                    raise ValueError("nested epoch recovery calibration authority/reference differs")
            if (not child.get("calibration_session_ids") or not parent.get("calibration_session_ids")
                    or not set(child["calibration_session_ids"]) <= set(parent["calibration_session_ids"])):
                raise ValueError("nested epoch recovery exceeds calibration sessions")
            ends = [scope.get("calibration_end_timestamp_s") for scope in (child, parent)]
            if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in ends) or ends[0] > ends[1]:
                raise ValueError("nested epoch recovery includes future calibration data")
        if self.spec.range_window_samples is not None:
            ranges = np.asarray(self.range_threshold_v, dtype=float)
            if ranges.shape != (channels,) or not np.isfinite(ranges).all() or (ranges <= 0).any():
                raise ValueError("range monitor thresholds must align with fitted channels")

    def record(self) -> dict[str, Any]:
        spec = asdict(self.spec)
        if self.spec.range_action == "reject":
            spec.pop("range_action")
        if self.spec.boundary_action == "all_edges":
            spec.pop("boundary_action")
        schema = CONTINUOUS_QC_CONSTANT_SCHEMA
        if self.spec.flatline_min_duration_s is None:
            spec.pop("flatline_min_duration_s")
            schema = CONTINUOUS_QC_SCHEMA
        return {"schema": schema, "spec": spec,
                "channel_names": list(self.channel_names), "sample_rate_hz": self.sample_rate_hz,
                "difference_center_v": list(self.difference_center_v),
                "difference_scale_v": list(self.difference_scale_v),
                "fit_provenance": dict(self.fit_provenance),
                "recovery_policy": ("calibration_empirical_bidirectional_boundary_envelope"
                    if self.epoch_recovery is not None and self.epoch_recovery.preprocessing.filter_phase == "zero" else
                    "calibration_empirical_epoch_modal_envelope") if self.epoch_recovery is not None else
                    "reject_unbounded_state_error" if self.recovery is None else "declared_state_box_modal_tail_bound",
                "recovery": None if self.recovery is None else asdict(self.recovery),
                "epoch_recovery": None if self.epoch_recovery is None else self.epoch_recovery.record(),
                "range_threshold_v": None if self.range_threshold_v is None else list(self.range_threshold_v),
                "detection_lookahead_samples": self.detection_lookahead_samples,
                "signal_unit": "V"}

    @property
    def detection_lookahead_samples(self):
        previous = (0 if self.spec.range_window_samples is None or self.spec.range_action == "diagnostic"
                    else self.spec.range_window_samples - 1) + self.spec.artifact_pre_guard_samples
        return max(previous, 0 if self.flatline_min_samples is None else self.flatline_min_samples - 1)

    @property
    def flatline_min_samples(self):
        duration = self.spec.flatline_min_duration_s
        # Both endpoints must have been observed for the declared elapsed time.
        return None if duration is None else int(np.ceil(duration * self.sample_rate_hz)) + 1

    def signal_diagnostics(self, raw, *, include_windows=False):
        """Describe continuing raw trends separately from artifact and recovery gates."""
        from data.slow_trend import analyze_slow_trends

        if float(raw.info["sfreq"]) != self.sample_rate_hz:
            raise ValueError("signal diagnostics sample rate differs from the frozen QC model")
        record = analyze_slow_trends(raw, channels=self.channel_names)
        record["quality_model_digest"] = semantic_sha256(self.record())
        record["continuing_input_quality_certified"] = False
        if self.spec.range_window_samples is not None and self.spec.range_action == "diagnostic":
            width = self.spec.range_window_samples
            values = raw.get_data(picks=list(self.channel_names))
            counts, windows = np.zeros(len(self.channel_names), dtype=int), 0
            for start, end in continuous_filter_segments(raw):
                if end-start >= width:
                    measured = trailing_range(values[:, start:end], width)[:, width-1:]
                    counts += (measured > np.asarray(self.range_threshold_v)[:, None]).sum(axis=1)
                    windows += measured.shape[1]
            record["raw_range_monitor"] = dict(action="diagnostic", window_samples=width,
                complete_windows=windows, exceeded_by_channel=counts.tolist(),
                channel_names=list(self.channel_names), threshold_v=list(self.range_threshold_v),
                interpretation="large_raw_range_may_be_removable_drift; not_a_standalone_rejection")
        if not include_windows:
            windows = record.pop("windows")
            record["window_metrics_sha256"] = semantic_sha256(windows)
            record["window_metrics_count"] = len(windows)
            record["window_metrics_storage"] = "reproducible_with_signal_diagnostics_include_windows"
        return record

    def detect(self, raw: mne.io.BaseRaw) -> ContinuousQCResult:
        if float(raw.info["sfreq"]) != self.sample_rate_hz:
            raise ValueError("continuous QC source sample rate differs from calibration")
        if any(name not in raw.ch_names for name in self.channel_names):
            raise ValueError("continuous QC measured channels differ from calibration")
        values = raw.get_data(picks=list(self.channel_names))
        if not np.isfinite(values).all():
            raise ValueError("nonfinite raw requires acquisition rejection before QC")
        center = np.asarray(self.difference_center_v)[:, None]
        limit = self.spec.difference_z * np.asarray(self.difference_scale_v)[:, None]
        intervals = []
        recovery_samples = None if self.recovery is None else self.recovery.required_samples()
        for start, end in continuous_filter_segments(raw):
            differences = np.diff(values[:, start:end], axis=1)
            bad = np.abs(differences - center) > limit
            if self.spec.channel_local:
                flags = np.pad(bad, ((0, 0), (1, 0)))
                width = self.spec.range_window_samples
                if width is not None and self.spec.range_action == "reject" and end - start >= width:
                    ranges = trailing_range(values[:, start:end], width)
                    range_bad = ranges > np.asarray(self.range_threshold_v)[:, None]
                    range_bad[:, :width-1] = False
                    flags |= maximum_filter1d(range_bad, size=width, axis=-1,
                        origin=-(width // 2), mode="constant", cval=0)
                for channel, name in enumerate(self.channel_names):
                    indices = np.flatnonzero(flags[channel]) + start
                    detected = _cluster_intervals(indices, end, self.spec.settle_samples, (name,),
                        "raw_difference_or_range_outlier" if width is not None and self.spec.range_action == "reject" else "raw_difference_outlier",
                        lookahead=0 if width is None or self.spec.range_action == "diagnostic" else width-1)
                    # Threshold crossing may follow the physical onset. This
                    # finite guard is explicit precaution, not an onset bound.
                    intervals.extend(ContinuousQCInterval(
                        max(start, row.start_sample - self.spec.artifact_pre_guard_samples),
                        row.end_sample, row.detected_sample, row.confirmed_sample,
                        row.channels, row.reason, row.reset_filter) for row in detected)
                    if self.flatline_min_samples is not None:
                        intervals.extend(_exact_constant_intervals(
                            values[channel, start:end], start, self.flatline_min_samples, name))
                continue
            indices = np.flatnonzero(bad.any(axis=0)) + start + 1
            # Every transient is recorded, even when an earlier uncertain state
            # has already made this acquisition segment unusable for inference.
            cursor = 0
            while cursor < len(indices):
                first = int(indices[cursor])
                last = first
                affected = np.zeros(len(self.channel_names), dtype=bool)
                while cursor < len(indices) and int(indices[cursor]) <= last + self.spec.settle_samples:
                    last = int(indices[cursor])
                    affected |= bad[:, last - start - 1]
                    cursor += 1
                confirmed = min(last + self.spec.settle_samples, end - 1)
                stop = confirmed + 1
                names = tuple(np.asarray(self.channel_names)[affected])
                intervals.append(ContinuousQCInterval(first, stop, first, confirmed, names,
                                                       "raw_difference_outlier", True))
                if stop < end:
                    recovery_end = end if recovery_samples is None else min(end, stop + recovery_samples)
                    if recovery_end > stop:
                        intervals.append(ContinuousQCInterval(stop, recovery_end, first, confirmed, names,
                            "filter_state_uncertainty_unbounded" if recovery_samples is None else "filter_state_bounded_recovery", False))
            if start > 0:
                recovery_end = end if recovery_samples is None else min(end, start + recovery_samples)
                if recovery_end > start:
                    intervals.append(ContinuousQCInterval(start, recovery_end, start, start,
                        self.channel_names, "filter_state_uncertainty_unbounded" if recovery_samples is None else "filter_state_bounded_recovery", False))
        result = ContinuousQCResult(int(raw.n_times), self.sample_rate_hz, tuple(intervals),
                                   semantic_sha256(self.record()), self.channel_names, self.spec.channel_local,
                                   self.detection_lookahead_samples)
        if self.spec.channel_local:
            return self.project_result(result, np.eye(len(self.channel_names)), self.channel_names,
                boundaries=[start for start, _ in continuous_filter_segments(raw) if start])
        return result

    def project_result(self, result, mixing, channels, *, boundaries=()):
        mixing = np.asarray(mixing, dtype=float)
        if mixing.shape != (len(channels), len(self.channel_names)) or not np.isfinite(mixing).all():
            raise ValueError("QC dependencies must map native channels to final signal channels")
        recovery = None if self.epoch_recovery is None else self.epoch_recovery.project(mixing, channels)
        bidirectional = recovery is not None and recovery.preprocessing.filter_phase == "zero"
        waits = None if recovery is None or bidirectional else recovery.recovery_samples()
        intervals = []
        exact_constant_enabled = self.spec.flatline_min_duration_s is not None
        boundary_samples = tuple(sorted(set(int(sample) for sample in boundaries)))
        for channel, name in enumerate(channels):
            if not mixing[channel].any():
                intervals.append(ContinuousQCInterval(0, result.n_samples, 0, 0, (name,),
                    "unfitted_signal_dependency", True))
                continue
            parents = {source for index, source in enumerate(self.channel_names) if mixing[channel, index] != 0}
            raw_intervals = sorted((row for row in result.intervals if row.reset_filter and parents.intersection(row.channels)),
                                   key=lambda row: (row.start_sample, row.end_sample))
            merged = []
            evidence = []
            for row in raw_intervals:
                if exact_constant_enabled:
                    evidence.append(ContinuousQCInterval(
                        row.start_sample, row.end_sample, row.detected_sample, row.confirmed_sample,
                        (name,), "raw_exact_constant" if row.reason == "raw_exact_constant"
                        else "raw_dependency_outlier", True))
                crosses_boundary = exact_constant_enabled and row.start_sample in boundary_samples
                if merged and row.start_sample <= merged[-1].end_sample and not crosses_boundary:
                    previous = merged.pop()
                    row = ContinuousQCInterval(previous.start_sample, max(previous.end_sample, row.end_sample),
                        min(previous.detected_sample, row.detected_sample), max(previous.confirmed_sample, row.confirmed_sample),
                        (name,), "raw_dependency_outlier", True)
                else:
                    row = ContinuousQCInterval(row.start_sample, row.end_sample, row.detected_sample,
                                              row.confirmed_sample, (name,), "raw_dependency_outlier", True)
                merged.append(row)
            # Keep v4 raw evidence reasons separate; merge only for recovery.
            intervals.extend(evidence if exact_constant_enabled else merged)
            if bidirectional:
                # Finite double-pass bounds depend on both segment ends. The
                # loader evaluates each actual epoch after final mixing/segmentation.
                continue
            resets = [(row.end_sample, row.confirmed_sample) for row in merged]
            resets.extend((int(sample), int(sample)) for sample in boundaries)
            for start, confirmation in resets:
                stop = result.n_samples if waits is None else min(result.n_samples, start + waits[channel])
                if exact_constant_enabled:
                    stop = min(stop, next((sample for sample in boundary_samples if sample > start), result.n_samples))
                if start < stop:
                    intervals.append(ContinuousQCInterval(start, stop, start, confirmation, (name,),
                        "empirical_epoch_state_recovery" if waits is not None else "filter_state_uncertainty_unbounded", False))
        return ContinuousQCResult(result.n_samples, result.sample_rate_hz, tuple(intervals),
                                  result.model_digest, tuple(channels), True, result.detection_lookahead_samples)


def trailing_range(values, width):
    kwargs = dict(size=width, axis=-1, origin=(width-1)//2, mode="nearest")
    return maximum_filter1d(values, **kwargs) - minimum_filter1d(values, **kwargs)


def _exact_constant_intervals(values, offset, minimum_samples, channel):
    """Identify exact raw-value runs, never pooled across acquisition segments."""
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [ContinuousQCInterval(
        int(offset + start), int(offset + stop), int(offset + start + minimum_samples - 1),
        int(offset + min(stop, len(values) - 1)), (channel,), "raw_exact_constant", True)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop - start >= minimum_samples]


def _cluster_intervals(indices, end, settle, names, reason, *, lookahead=0):
    result = []
    cursor = 0
    while cursor < len(indices):
        first = last = int(indices[cursor])
        while cursor < len(indices) and indices[cursor] <= last + settle:
            last = int(indices[cursor])
            cursor += 1
        stop = min(end, last + settle + 1)
        result.append(ContinuousQCInterval(first, stop, min(end-1, first+lookahead),
            min(end-1, stop-1+lookahead), names, reason, True))
    return result


def signal_dependency_matrix(native_channels, output_channels, *, reference_channel=None, ocular_model=None):
    """Linear EEG/EOG and measured-reference dependencies, excluding DC constants."""
    from data.ocular import SHARED_MEASURED_REFERENCE

    positions = {canonical_channel_name(name): index for index, name in enumerate(native_channels)}
    anchor = (ocular_model.reference_channel if ocular_model is not None else reference_channel)
    anchor = None if anchor is None else canonical_channel_name(anchor)
    matrix = np.zeros((len(output_channels), len(native_channels)))
    for row, channel in enumerate(output_channels):
        name = canonical_channel_name(channel)
        if anchor is not None and anchor not in positions:
            raise ValueError("QC did not fit every measured EEG/reference dependency")
        if name not in positions:
            continue
        matrix[row, positions[name]] = 1
        if anchor is not None:
            matrix[row, positions[anchor]] -= 1
        if ocular_model is not None:
            ocular_row = tuple(map(canonical_channel_name, ocular_model.eeg_channels)).index(name)
            for eye, coefficient in zip(ocular_model.eog_channels, ocular_model.effective_coefficients[ocular_row], strict=True):
                if coefficient == 0:
                    continue
                if canonical_channel_name(eye) not in positions:
                    raise ValueError("QC must include each EOG channel used by frozen regression")
                matrix[row, positions[canonical_channel_name(eye)]] -= coefficient
                if anchor is not None and ocular_model.eog_reference_policy == SHARED_MEASURED_REFERENCE:
                    matrix[row, positions[anchor]] += coefficient
    return matrix


def minimum_usable_channel_count(processing, n_channels):
    if processing.get("status") != "applied":
        return 1
    spec = processing.get("model", {}).get("spec", {})
    if spec.get("channel_local") is not True:
        return 1
    return int(np.ceil(n_channels * spec["minimum_channel_fraction"]))


def fit_continuous_qc(
    calibration_raw: mne.io.BaseRaw, *, spec: ContinuousQCSpec,
    fit_sample_mask: np.ndarray, fit_provenance: Mapping[str, Any],
    recovery: FilterStateRecovery | None = None,
    epoch_recovery: EpochStateRecovery | BidirectionalStateRecovery | None = None,
) -> FittedContinuousQC:
    mask = np.asarray(fit_sample_mask)
    if mask.dtype != bool or mask.shape != (calibration_raw.n_times,):
        raise ValueError("continuous QC requires an explicit aligned boolean fit_sample_mask")
    if not fit_provenance:
        raise ValueError("continuous QC requires calibration provenance")
    names = tuple(calibration_raw.ch_names[index] for index in mne.pick_types(
        calibration_raw.info, eeg=True, eog=spec.include_eog, exclude="bads"))
    values = calibration_raw.get_data(picks=list(names))
    pairs = mask[1:] & mask[:-1]
    clean = np.zeros(calibration_raw.n_times, dtype=bool)
    for start, end in continuous_filter_segments(calibration_raw):
        clean[start:end] = True
        if start:
            pairs[start - 1] = False
    pairs &= clean[1:] & clean[:-1]
    if int(pairs.sum()) < 2:
        raise ValueError("continuous QC needs at least two valid within-segment differences")
    differences = np.diff(values, axis=1)[:, pairs]
    if not np.isfinite(differences).all():
        raise ValueError("continuous QC fit samples must be finite")
    center = np.median(differences, axis=1)
    scale = np.maximum(1.4826 * np.median(np.abs(differences - center[:, None]), axis=1),
                       spec.difference_floor_v)
    range_threshold = None
    if spec.range_window_samples is not None:
        width = spec.range_window_samples
        windows = []
        for start, stop in continuous_filter_segments(calibration_raw):
            if stop - start < width:
                continue
            supported = np.convolve(mask[start:stop].astype(int), np.ones(width, dtype=int), mode="valid") == width
            if supported.any():
                windows.append(trailing_range(values[:, start:stop], width)[:, width-1:][:, supported])
        if not windows:
            raise ValueError("range calibration lacks a complete clean observation window")
        ranges = np.concatenate(windows, axis=1)
        range_threshold = tuple(np.maximum(spec.range_floor_v,
            np.quantile(ranges, spec.range_quantile, axis=1) * spec.range_inflation))
    return FittedContinuousQC(spec, names, float(calibration_raw.info["sfreq"]), tuple(center),
        tuple(scale), {**fit_provenance, "fit_samples": int(mask.sum()),
                      "fit_difference_pairs": int(pairs.sum()),
                      "fit_sample_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                      "fit_values_sha256": hashlib.sha256(differences.tobytes()).hexdigest(),
                      "estimator": "calibration_difference_median_MAD", "labels_used": False},
        recovery, epoch_recovery, range_threshold)


def continuous_qc_from_record(record: Mapping[str, Any]) -> FittedContinuousQC:
    schema = record.get("schema")
    if schema not in (CONTINUOUS_QC_SCHEMA, CONTINUOUS_QC_CONSTANT_SCHEMA):
        raise ValueError("unsupported continuous QC schema")
    if schema == CONTINUOUS_QC_SCHEMA and "flatline_min_duration_s" in record["spec"]:
        raise ValueError("continuous QC schema /3 does not define raw exact-constant detection")
    if schema == CONTINUOUS_QC_CONSTANT_SCHEMA and record["spec"].get("flatline_min_duration_s") is None:
        raise ValueError("continuous QC schema /4 requires explicit raw exact-constant detection")
    recovery = record.get("recovery")
    if recovery is not None:
        recovery = FilterStateRecovery(**{**recovery, "sos": tuple(map(tuple, recovery["sos"])),
                                         "state_error_bound_v": tuple(map(tuple, recovery["state_error_bound_v"]))})
    model = FittedContinuousQC(ContinuousQCSpec(**record["spec"]), tuple(record["channel_names"]),
        record["sample_rate_hz"], tuple(record["difference_center_v"]),
        tuple(record["difference_scale_v"]), record["fit_provenance"], recovery,
        None if record["epoch_recovery"] is None else epoch_recovery_from_record(record["epoch_recovery"]),
        None if record["range_threshold_v"] is None else tuple(record["range_threshold_v"]))
    if semantic_sha256(model.record()) != semantic_sha256(record):
        raise ValueError("continuous QC record has inconsistent fields")
    return model


def save_continuous_qc(path: str | Path, model: FittedContinuousQC) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    record = model.record()
    return atomic_write_json(destination, {"model": record, "digest": semantic_sha256(record)})


def load_continuous_qc(path: str | Path) -> FittedContinuousQC:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"model", "digest"} or payload["digest"] != semantic_sha256(payload["model"]):
        raise ValueError("continuous QC digest is invalid")
    return continuous_qc_from_record(payload["model"])


def validate_continuous_calibration_budget(processing, calibration_by_participant):
    if processing.get("status") != "applied":
        return
    record = processing.get("model")
    if not isinstance(record, Mapping) or processing.get("model_digest") != semantic_sha256(record):
        raise ValueError("continuous QC processing lacks a valid fitted-model record")
    model = continuous_qc_from_record(record)
    provenance = model.fit_provenance
    sessions = provenance.get("calibration_session_ids", [])
    if not sessions or not set(sessions) <= set(calibration_by_participant.get(provenance.get("participant_key"), ())):
        raise ValueError("continuous QC used sessions outside the declared calibration budget")
    if model.epoch_recovery is not None:
        state_scope = model.epoch_recovery.fit_provenance
        if (state_scope.get("participant_key") != provenance.get("participant_key")
                or not state_scope.get("calibration_session_ids")
                or not set(state_scope["calibration_session_ids"]) <= set(sessions)):
            raise ValueError("epoch state envelope exceeds the continuous QC calibration authority")
