"""Format-independent continuous EEG preprocessing with explicit physical contracts."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import mne
import numpy as np
import scipy
from scipy.signal import iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from data.channel import (
    DEFAULT_MONTAGE,
    CoordinateRegistrationSpec,
    canonical_channel_name,
    resolve_channel_layout,
)
from data.contract import (
    CAUSAL_IIR_INITIAL_STATE,
    DEFAULT_P300_DATA_CONTRACT,
    validate_notch_parameters,
    validate_zero_phase_edge_policy,
)

DEFAULT_REJECT_THRESHOLD: float | None = None
FILTER_SKIP_ANNOTATIONS = ("edge", "bad_acq_skip", "bad_continuous_artifact")
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RawSampleScope:
    """Independent finite raw support and its explicitly authorized event rows.

    Samples are absolute MNE indices and ``stop_sample`` is exclusive. Event
    indices refer to the original recording ledger, including boundary-purged
    requests. Tail context never authorizes additional labels automatically.
    """

    name: str
    start_sample: int
    stop_sample: int
    event_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name.strip()
                or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                       for value in (self.start_sample, self.stop_sample))
                or self.start_sample < 0 or self.stop_sample <= self.start_sample):
            raise ValueError("raw scope requires a name and increasing absolute integer sample bounds")
        if not isinstance(self.event_indices, (tuple, list)):
            raise ValueError("raw scope event_indices must be an explicit ordered sequence")
        indices = tuple(self.event_indices)
        if (not indices or any(not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
                               or index < 0 for index in indices)
                or any(left >= right for left, right in zip(indices[:-1], indices[1:], strict=True))):
            raise ValueError("raw scope event_indices must be nonempty, unique and increasing")
        object.__setattr__(self, "start_sample", int(self.start_sample))
        object.__setattr__(self, "stop_sample", int(self.stop_sample))
        object.__setattr__(self, "event_indices", tuple(map(int, indices)))

    def record(self) -> dict:
        return {"schema": "n2p3_raw_sample_scope/1", "name": self.name,
                "start_sample": self.start_sample, "stop_sample": self.stop_sample,
                "event_indices": list(self.event_indices)}

    @classmethod
    def from_record(cls, record: Mapping) -> RawSampleScope:
        if (not isinstance(record, Mapping)
                or set(record) != {"schema", "name", "start_sample", "stop_sample", "event_indices"}
                or record["schema"] != "n2p3_raw_sample_scope/1"):
            raise ValueError("unsupported or incomplete raw sample scope")
        return cls(**{key: value for key, value in record.items() if key != "schema"})

    def validate_source(self, raw: mne.io.BaseRaw, events: np.ndarray | None = None) -> None:
        if not int(raw.first_samp) <= self.start_sample < self.stop_sample <= int(raw.first_samp + raw.n_times):
            raise ValueError("raw scope lies outside the actual source recording")
        if events is not None and self.event_indices[-1] >= len(events):
            raise ValueError("raw scope event index lies outside the original event ledger")


def crop_raw_scope(raw: mne.io.BaseRaw, scope: RawSampleScope) -> mne.io.BaseRaw:
    """Return isolated original samples before filtering or fitting QC/EOG states."""
    if not isinstance(scope, RawSampleScope):
        raise TypeError("raw scope must be a RawSampleScope")
    scope.validate_source(raw)
    rate = float(raw.info["sfreq"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("raw scope requires a finite positive source sample rate")
    cropped = raw.copy().crop(
        tmin=(scope.start_sample - raw.first_samp) / rate,
        tmax=(scope.stop_sample - raw.first_samp - 1) / rate,
        include_tmax=True,
    )
    if cropped.first_samp != scope.start_sample or cropped.n_times != scope.stop_sample - scope.start_sample:
        raise RuntimeError("MNE raw crop did not preserve the declared exact source samples")
    return cropped


def source_epoch_support(events, source_rate_hz, preprocessing) -> tuple[np.ndarray, np.ndarray]:
    """Resolve existing closed-end epoch support as absolute half-open samples."""
    _validate_execution(preprocessing, source_rate_hz)
    events = np.asarray(events)
    if (events.ndim != 2 or events.shape[1] != 3 or not np.issubdtype(events.dtype, np.integer)
            or np.issubdtype(events.dtype, np.bool_)):
        raise ValueError("source support requires an integer MNE event array")
    start = round(preprocessing.tmin_ms / 1000 * source_rate_hz)
    stop = round(preprocessing.tmax_ms / 1000 * source_rate_hz) + 1
    return events[:, 0].astype(np.int64) + start, events[:, 0].astype(np.int64) + stop


def prefix_raw_scopes(raw, events, preprocessing, *, calibration_events: int,
                      maximum_calibration_events: int | None = None,
                      parent_scope: RawSampleScope | None = None) -> tuple[RawSampleScope | None, RawSampleScope]:
    """Plan positive-K or K0 support and a common held-out suffix without reading EEG.

    Reapplying this inside a complete calibration ``parent_scope`` constructs an
    independent inner train/validation split. Choosing K/fallbacks and labels is
    the caller's protocol responsibility; this function never extends a scope.
    """
    events = np.asarray(events)
    _, stops = source_epoch_support(events, float(raw.info["sfreq"]), preprocessing)
    if parent_scope is not None:
        parent_scope.validate_source(raw, events)
        indices = parent_scope.event_indices
        first, end, name = parent_scope.start_sample, parent_scope.stop_sample, parent_scope.name
    else:
        indices = tuple(range(len(events)))
        first, end, name = int(raw.first_samp), int(raw.first_samp + raw.n_times), "recording"
    maximum = calibration_events if maximum_calibration_events is None else maximum_calibration_events
    if (type(calibration_events) is not int or type(maximum) is not int
            or not 0 <= calibration_events <= maximum < len(indices) or maximum < 1):
        raise ValueError("prefix counts must declare 0 <= K <= Kmax < the authorized event count")
    if np.any(np.diff(events[np.asarray(indices), 0]) <= 0):
        raise ValueError("raw prefix planning requires unique chronological source events")
    heldout_start = int(stops[list(indices[:maximum])].max())
    if not first < heldout_start < end:
        raise ValueError("prefix allocation leaves no legal held-out raw support")
    calibration = None
    if calibration_events:
        calibration = RawSampleScope(name + ":calibration", first,
            int(stops[list(indices[:calibration_events])].max()), indices[:calibration_events])
    heldout = RawSampleScope(name + ":heldout", heldout_start, end, indices[maximum:])
    return calibration, heldout


@dataclass(frozen=True)
class PreprocessResult:
    data: np.ndarray
    channel_names: tuple[str, ...]
    channel_positions_m: np.ndarray
    channel_mask: np.ndarray
    position_mask: np.ndarray
    layout_source: str
    coordinate_registration: CoordinateRegistrationSpec
    sfreq: float
    tmin: float
    tmax: float
    dropped: np.ndarray
    event_indices: np.ndarray
    event_samples: np.ndarray
    event_times_s: np.ndarray
    evidence_available_times_s: np.ndarray
    event_statuses: np.ndarray
    event_status_details: np.ndarray
    event_evidence_indices: np.ndarray
    online_causal: bool
    causal_iir_initial_state: str
    source_epoch_start_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    source_epoch_end_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    timing_source: str = "source_mne_event_samples;source_grid_closed_epoch;conservative_support"
    execution_receipt: dict | None = None
    scheduled_event_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("PreprocessResult.data must be (N,C,T).")
        channels = self.data.shape[1]
        if len(self.channel_names) != channels:
            raise ValueError("channel_names do not match the data channel dimension.")
        if self.channel_positions_m.shape != (channels, 3):
            raise ValueError("channel_positions_m must be (C,3).")
        if self.channel_mask.shape != (channels,) or self.position_mask.shape != (channels,):
            raise ValueError("channel masks must be (C,).")
        if np.any(self.channel_mask & ~self.position_mask):
            raise ValueError("Every observed channel requires a physical position.")
        n_events = len(self.event_samples)
        if self.scheduled_event_indices.size == 0 and n_events:
            object.__setattr__(self, "scheduled_event_indices", np.arange(n_events, dtype=np.int64))
        scheduled = np.asarray(self.scheduled_event_indices)
        if (scheduled.shape != (n_events,) or not np.issubdtype(scheduled.dtype, np.integer)
                or np.issubdtype(scheduled.dtype, np.bool_) or np.any(scheduled < 0)
                or np.any(np.diff(scheduled) <= 0)):
            raise ValueError("scheduled event indices must preserve unique original row identities")
        event_fields = (
            self.event_times_s,
            self.evidence_available_times_s,
            self.event_statuses,
            self.event_status_details,
            self.event_evidence_indices,
        )
        if any(np.asarray(field).shape != (n_events,) for field in event_fields):
            raise ValueError("Every event-ledger field must have one row per scheduled event.")
        available = np.asarray(self.event_statuses).astype(str) == "available"
        evidence = np.asarray(self.event_evidence_indices, dtype=np.int64)
        if sorted(evidence[available].tolist()) != list(range(self.n_epochs)):
            raise ValueError("Available event rows must map bijectively to output epochs.")
        if np.any(evidence[~available] != -1):
            raise ValueError("Unavailable events cannot point to model evidence.")
        for samples in (self.source_epoch_start_samples, self.source_epoch_end_samples):
            if samples.size and samples.shape != (n_events,):
                raise ValueError("source epoch support must align with scheduled events")

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])

    @property
    def n_present(self) -> int:
        return int(self.channel_mask.sum())


def _eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, exclude=[])
    if len(picks) == 0:
        raise ValueError("The recording contains no channels typed as EEG.")
    return [raw.ch_names[index] for index in picks]


def map_channels(
    raw: mne.io.BaseRaw,
    channels: Sequence[str] | None = None,
    *,
    aliases: Mapping[str, str] | None = None,
    copy: bool = True,
) -> tuple[mne.io.BaseRaw, np.ndarray, tuple[str, ...]]:
    """Select/reorder an exact physical EEG layout by canonical channel name."""

    if copy:
        raw = raw.copy()
    eeg_names = _eeg_channel_names(raw)
    present: dict[str, str] = {}
    for original in eeg_names:
        canonical = canonical_channel_name(original, aliases=aliases)
        if canonical in present:
            raise ValueError(
                f"Channels {present[canonical]!r} and {original!r} both normalize to "
                f"{canonical!r}; provide an explicit alias policy."
            )
        present[canonical] = original

    if channels is None:
        target_names = tuple(canonical_channel_name(name, aliases=aliases) for name in eeg_names)
    else:
        target_names = tuple(canonical_channel_name(name, aliases=aliases) for name in channels)
        if len(set(target_names)) != len(target_names):
            raise ValueError(
                f"Requested channels are duplicated after normalization: {target_names}."
            )

    missing = [name for name in target_names if name not in present]
    if missing:
        raise ValueError(
            f"Requested EEG channels are absent: {missing}. available={list(present)}. "
            "Use the recording's native layout or resolve a common intersection across records; "
            "the standard ingress does not pad or substitute electrodes."
        )
    selected = [present[name] for name in target_names]
    rename = {present[name]: name for name in target_names}
    raw.pick(selected)
    raw.rename_channels(rename)
    return raw, np.ones(len(target_names), dtype=bool), target_names


def resample(
    raw: mne.io.BaseRaw,
    events: np.ndarray | None = None,
    sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
    *,
    copy: bool = False,
) -> tuple[mne.io.BaseRaw, np.ndarray | None]:
    """Resample continuous data and event samples through MNE's joint operation."""

    if sfreq <= 0:
        raise ValueError("sfreq must be positive.")
    if copy:
        raw = raw.copy()
    if np.isclose(float(raw.info["sfreq"]), sfreq):
        return raw, None if events is None else np.asarray(events, dtype=np.int64).copy()
    if events is None:
        raw.resample(sfreq, verbose=False)
        return raw, None
    raw, scaled = raw.resample(sfreq, events=np.asarray(events, dtype=np.int64), verbose=False)
    return raw, scaled


@dataclass(frozen=True)
class ContinuousFilterSegment:
    start_sample: int
    stop_sample: int
    left_causes: frozenset[str]
    right_causes: frozenset[str]

    @property
    def requires_recovery(self) -> bool:
        return bool((self.left_causes | self.right_causes) & {"artifact", "unclassified_reset"})

    def record(self) -> dict:
        return {"start_sample": self.start_sample, "stop_sample": self.stop_sample,
                "left_causes": sorted(self.left_causes), "right_causes": sorted(self.right_causes)}


def continuous_filter_plan(raw: mne.io.BaseRaw, channel: str | None = None) -> tuple[ContinuousFilterSegment, ...]:
    """Interpret actual reset support and retain the reasons for both boundaries.

    ``channel=None`` preserves MNE's legacy global skip-annotation behavior:
    channel-scoped BAD/EDGE annotations apply globally, while local-QC annotations
    belong only to the explicit per-channel backend. Mixed excluded components
    carry their complete cause union to both exposed boundaries.
    """
    changes: dict[int, int] = {0: 0, int(raw.n_times): 0}
    intervals, cuts = [], []
    rate = float(raw.info["sfreq"])
    for onset, duration, description, channels in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description,
        raw.annotations.ch_names, strict=True
    ):
        description = str(description).lower()
        local = description.startswith("quality_local_artifact")
        if local and not channels:
            raise ValueError("channel-local artifact annotations require explicit affected channels")
        if local and (channel is None or channel not in channels):
            continue
        if channels and channel is not None and channel not in channels:
            continue
        if not local and not description.startswith(FILTER_SKIP_ANNOTATIONS):
            continue
        cause = ("artifact" if local or description.startswith("bad_continuous_artifact") else
                 "acquisition_discontinuity" if description.startswith("bad_acq_skip")
                 or description in {"edge boundary", "edge gap"} else "unclassified_reset")
        start = int(np.clip(round((float(onset) - raw.first_time) * rate), 0, raw.n_times))
        stop = int(np.clip(round((float(onset) + float(duration) - raw.first_time) * rate),
                           0, raw.n_times))
        changes.setdefault(start, 0)
        changes.setdefault(stop, 0)
        if stop > start:
            changes[start] += 1
            changes[stop] -= 1
            intervals.append((start, stop, cause))
        else:
            cuts.append((start, cause))
    components = []
    for start, stop, cause in sorted(intervals):
        if components and start <= components[-1][1]:
            previous_start, previous_stop, causes = components[-1]
            components[-1] = (previous_start, max(previous_stop, stop), causes | {cause})
        else:
            components.append((start, stop, {cause}))
    boundary_causes = {0: {"record_start"}, int(raw.n_times): {"record_end"}}
    component_index = 0
    for sample, cause in sorted(cuts):
        while component_index < len(components) and components[component_index][1] < sample:
            component_index += 1
        if component_index < len(components) and components[component_index][0] <= sample:
            components[component_index][2].add(cause)
        else:
            boundary_causes.setdefault(sample, set()).add(cause)
    for start, stop, causes in components:
        boundary_causes.setdefault(start, set()).update(causes)
        boundary_causes.setdefault(stop, set()).update(causes)
    segments = []
    previous = 0
    active = 0
    for sample, change in sorted(changes.items()):
        if previous < sample and active == 0:
            segments.append(ContinuousFilterSegment(previous, sample,
                frozenset(boundary_causes[previous]), frozenset(boundary_causes[sample])))
        active += change
        previous = sample
    return tuple(segments)


def continuous_filter_segments(raw: mne.io.BaseRaw, channel: str | None = None) -> tuple[tuple[int, int], ...]:
    """Return the coordinate projection of the shared reset interpretation."""
    return tuple((span.start_sample, span.stop_sample) for span in continuous_filter_plan(raw, channel))


def design_continuous_sos(sample_rate_hz, *, l_freq, h_freq, order=4,
                          notch_freqs_hz=(), notch_quality_factor=30.0):
    """One executable SOS cascade shared by filtering and recovery certificates."""
    validate_notch_parameters(notch_freqs_hz, notch_quality_factor, sample_rate_hz=sample_rate_hz)
    sections = [tf2sos(*iirnotch(f, notch_quality_factor, fs=sample_rate_hz))
                for f in notch_freqs_hz]
    if l_freq is not None or h_freq is not None:
        if l_freq is not None and h_freq is not None:
            f_pass, btype = [float(l_freq), float(h_freq)], "bandpass"
        elif l_freq is not None:
            f_pass, btype = float(l_freq), "highpass"
        else:
            f_pass, btype = float(h_freq), "lowpass"
        sections.append(mne.filter.construct_iir_filter(
            dict(order=order, ftype="butter", output="sos"), f_pass=f_pass,
            sfreq=sample_rate_hz, btype=btype, phase="forward", verbose=False)["sos"])
    return np.concatenate(sections, axis=0) if sections else np.empty((0, 6), dtype=float)


@dataclass(frozen=True)
class BidirectionalSegment:
    data: np.ndarray
    forward_core: np.ndarray
    left_state: np.ndarray
    right_state: np.ndarray
    padding_samples: int


def filter_bidirectional_segment(sos, values, *, padding_limit: int) -> BidirectionalSegment:
    """Execute MNE reflect_limited SOS filtering and expose its two effective states.

    Values are (channels, samples); states are (sections, channels, 2).
    Padding is restricted to the segment, including the one-sample case.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("bidirectional segment requires finite (channels, samples) values")
    if not isinstance(padding_limit, (int, np.integer)) or padding_limit < 0:
        raise ValueError("padding_limit must be a nonnegative integer")
    pad = min(int(padding_limit), values.shape[1] - 1)
    initial = sosfilt_zi(sos)[:, None, :]
    left = 2 * values[:, :1] - values[:, 1:pad+1][:, ::-1]
    right = 2 * values[:, -1:] - values[:, -pad-1:-1][:, ::-1] if pad else values[:, :0]
    state = initial * (left[:, 0] if pad else values[:, 0])[None, :, None]
    if pad:
        _, state = sosfilt(sos, left, axis=-1, zi=state)
    left_state = state.copy()
    forward, state = sosfilt(sos, values, axis=-1, zi=state)
    if pad:
        tail, _ = sosfilt(sos, right, axis=-1, zi=state)
        state = initial * tail[:, -1][None, :, None]
        _, state = sosfilt(sos, tail[:, ::-1], axis=-1, zi=state)
    else:
        state = initial * forward[:, -1][None, :, None]
    right_state = state.copy()
    reverse, _ = sosfilt(sos, forward[:, ::-1], axis=-1, zi=state)
    return BidirectionalSegment(reverse[:, ::-1], forward, left_state, right_state, pad)


def filter_continuous(
    raw: mne.io.BaseRaw,
    l_freq: float | None = 0.1,
    h_freq: float | None = None,
    *,
    method: str = "iir",
    phase: str = "zero",
    causal_iir_initial_state: str = "not_applicable",
    order: int = 4,
    notch_freqs_hz: Sequence[float] = (),
    notch_quality_factor: float = 30.0,
    zero_phase_edge_policy: str = "legacy",
    copy: bool = False,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply a declared continuous-domain high/low/band-pass before epoching."""

    validate_zero_phase_edge_policy(zero_phase_edge_policy, method, phase)
    validate_notch_parameters(notch_freqs_hz, notch_quality_factor,
                              sample_rate_hz=float(raw.info["sfreq"]))
    if notch_freqs_hz and (method != "iir" or phase not in {"forward", "zero"}):
        raise ValueError("notch filtering requires IIR with forward or zero phase")
    if l_freq is None and h_freq is None and not notch_freqs_hz:
        return raw.copy() if copy else raw
    if l_freq is not None and l_freq <= 0:
        raise ValueError("l_freq must be positive or None.")
    if h_freq is not None and h_freq <= 0:
        raise ValueError("h_freq must be positive or None.")
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError("l_freq must be smaller than h_freq for band-pass filtering.")
    nyquist = float(raw.info["sfreq"]) / 2.0
    if h_freq is not None and h_freq >= nyquist:
        raise ValueError(f"h_freq must be below Nyquist ({nyquist:g} Hz).")
    if copy:
        raw = raw.copy()
    if phase == "forward" and causal_iir_initial_state != CAUSAL_IIR_INITIAL_STATE:
        raise ValueError(
            "forward IIR requires causal_iir_initial_state='steady_state_first_sample'."
        )
    if phase != "forward" and causal_iir_initial_state != "not_applicable":
        raise ValueError("non-causal filtering requires causal_iir_initial_state='not_applicable'.")
    if method == "iir" and (phase == "forward" or notch_freqs_hz or zero_phase_edge_policy != "legacy"):
        if phase != "forward" and zero_phase_edge_policy == "legacy" and any(str(value).lower().startswith("quality_local_artifact")
                                      for value in raw.annotations.description):
            raise ValueError("channel-local artifact state handling currently requires forward IIR")
        sos = design_continuous_sos(float(raw.info["sfreq"]), l_freq=l_freq, h_freq=h_freq,
            order=order, notch_freqs_hz=notch_freqs_hz, notch_quality_factor=notch_quality_factor)
        steady_state = sosfilt_zi(sos)
        zero_params = (mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)
                       if zero_phase_edge_policy == "mne_reflect_limited" else None)
        for name in _eeg_channel_names(raw):
            segments = continuous_filter_segments(raw, name)

            def causal_filter(channel: np.ndarray, segments=segments) -> np.ndarray:
                filtered = channel.copy()
                for start, stop in segments:
                    if phase == "forward":
                        initial_state = steady_state * float(channel[start])
                        filtered[start:stop], _ = sosfilt(sos, channel[start:stop], zi=initial_state)
                    elif zero_params is not None:
                        # MNE estimates padding from this entire cascade's impulse response,
                        # caps it at segment length - 1 and applies reflect_limited padding.
                        filtered[start:stop] = filter_bidirectional_segment(
                            sos, channel[None, start:stop], padding_limit=zero_params["padlen"]).data[0]
                    else:
                        padding = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
                        filtered[start:stop] = sosfiltfilt(sos, channel[start:stop], padlen=min(padding, stop - start - 1))
                return filtered

            raw.apply_function(causal_filter, picks=[name], channel_wise=True, verbose=verbose)
    else:
        if any(str(value).lower().startswith("quality_local_artifact") for value in raw.annotations.description):
            raise ValueError("channel-local artifact state handling currently requires forward IIR")
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            iir_params=dict(order=order, ftype="butter") if method == "iir" else None,
            phase=phase,
            skip_by_annotation=FILTER_SKIP_ANNOTATIONS,
            verbose=verbose,
        )
    return raw


def apply_trial_baseline(
    data: np.ndarray,
    *,
    sfreq: float,
    tmin: float,
    baseline_mode: str,
    trial_reference_window_ms: tuple[float, float] | None = None,
    trial_reference_center: str = "mean",
    trial_reference_scale: str = "none",
) -> np.ndarray:
    """Execute the declared per-trial, per-channel reference transform.

    Windows use the same half-open physical time axis as ``EpochDataset``.
    The transform is label-free and does not fit population statistics.
    """

    values = np.asarray(data)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("baseline input must be a floating-point (N,C,T) array.")
    if not np.isfinite(values).all():
        raise ValueError("baseline input contains NaN/inf.")
    if baseline_mode == "none":
        return values
    if baseline_mode not in {"mean_only", "trial_reference"}:
        raise ValueError(
            "Executable baseline_mode must be none, mean_only, or center-only trial_reference."
        )
    if trial_reference_center not in {"mean", "median"}:
        raise ValueError("trial_reference_center must be mean or median.")
    if trial_reference_scale not in {"none", "std", "mad"}:
        raise ValueError("trial_reference_scale must be none, std, or mad.")
    if trial_reference_scale != "none":
        raise ValueError("Baseline scale normalization would destroy the physical volts contract.")

    if baseline_mode == "mean_only":
        if tmin >= 0.0:
            raise ValueError(f"baseline_mode={baseline_mode!r} requires tmin < 0.")
        window_ms = (tmin * 1000.0, 0.0)
        center_mode = "mean"
    else:
        if trial_reference_window_ms is None:
            raise ValueError("trial_reference mode requires trial_reference_window_ms.")
        window_ms = tuple(float(value) for value in trial_reference_window_ms)
        center_mode = trial_reference_center

    start = int(np.ceil((window_ms[0] / 1000.0 - tmin) * sfreq - 1e-9))
    stop = int(np.ceil((window_ms[1] / 1000.0 - tmin) * sfreq - 1e-9))
    if start < 0 or stop > values.shape[2] or stop - start < 2:
        raise ValueError(
            "baseline reference window must contain at least two samples inside the epoch."
        )
    reference = values[:, :, start:stop].astype(np.float64, copy=False)
    if center_mode == "mean":
        center = reference.mean(axis=2, keepdims=True)
    else:
        center = np.median(reference, axis=2, keepdims=True)
    transformed = values.astype(np.float64, copy=False) - center
    return transformed.astype(values.dtype, copy=False)


def _transform_epoch_samples(
    values, source_rate_hz, *, sfreq, n_times, tmin, baseline_mode,
    trial_reference_window_ms, trial_reference_center, trial_reference_scale,
    resample_method, resample_npad, resample_window, resample_pad,
):
    """One numeric kernel for production epochs and conditional certificates."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim < 2 or not data.shape[-1] or not np.isfinite(data).all():
        raise ValueError("epoch samples require finite values and a nonempty last time axis")
    do_resample = not np.isclose(source_rate_hz, sfreq)
    effective_rate = float(sfreq if do_resample else source_rate_hz)
    resampled_n_times = (int(round(data.shape[-1] * float(sfreq) / source_rate_hz))
                         if do_resample else data.shape[-1])
    if resampled_n_times < 1 or (n_times is not None and resampled_n_times < n_times):
        raise ValueError("Epoching produced fewer samples than the physical contract requires")
    leading_shape = data.shape[:-1]
    if data.size == 0:
        return np.empty((*leading_shape, n_times or resampled_n_times), dtype=np.float64)
    if do_resample:
        data = mne.filter.resample(data, sfreq, source_rate_hz,
            method=resample_method, npad=resample_npad, window=resample_window,
            pad=resample_pad, verbose=False)
    if n_times is not None:
        data = data[..., :n_times]
    transformed = apply_trial_baseline(data.reshape(-1, 1, data.shape[-1]),
        sfreq=effective_rate, tmin=tmin, baseline_mode=baseline_mode,
        trial_reference_window_ms=trial_reference_window_ms,
        trial_reference_center=trial_reference_center, trial_reference_scale=trial_reference_scale)
    return transformed.reshape(*leading_shape, transformed.shape[-1])


def _validate_execution(preprocessing, source_rate_hz):
    from data.epochs import PreprocessingSpec

    if not isinstance(preprocessing, PreprocessingSpec):
        raise TypeError("execution requires a complete PreprocessingSpec")
    preprocessing.validate()
    if not np.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError("source sample rate must be finite and positive")
    if preprocessing.reject_threshold_v is not None:
        raise ValueError("Fixed absolute-voltage epoch rejection is retired; historical records are read-only")
    for frequency in (preprocessing.l_freq, preprocessing.h_freq):
        if frequency is not None and frequency >= source_rate_hz / 2:
            raise ValueError("filter cutoff must be below actual source Nyquist")
    validate_notch_parameters(preprocessing.notch_freqs_hz, preprocessing.notch_quality_factor,
                              sample_rate_hz=source_rate_hz)
    # Validate the actual cropped baseline grid before touching the raw object.
    effective_rate = source_rate_hz if np.isclose(source_rate_hz, preprocessing.sfreq) else preprocessing.sfreq
    apply_trial_baseline(np.zeros((1, 1, preprocessing.n_times)),
        sfreq=effective_rate, tmin=preprocessing.tmin_ms / 1000,
        baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale)


def _epoch_transform_options(preprocessing):
    return dict(sfreq=preprocessing.sfreq, n_times=preprocessing.n_times,
        tmin=preprocessing.tmin_ms / 1000, baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale,
        resample_method=preprocessing.resample_method, resample_npad=preprocessing.resample_npad,
        resample_window=preprocessing.resample_window, resample_pad=preprocessing.resample_pad)


def transform_epoch_samples(values, source_rate_hz, preprocessing, *, require_linear=False):
    """Apply the resolved epoch FFT, first-T crop and half-open baseline.

    Input axes precede a last source-time axis. Linear certificate callers must
    request the capability explicitly; a median trial reference cannot qualify.
    """
    _validate_execution(preprocessing, source_rate_hz)
    if (require_linear and preprocessing.baseline_mode == "trial_reference"
            and preprocessing.trial_reference_center != "mean"):
        raise ValueError("a linear epoch operator cannot use median trial_reference")
    return _transform_epoch_samples(values, source_rate_hz, **_epoch_transform_options(preprocessing))


def execute_preprocessing(raw, events, preprocessing, *, raw_scope: RawSampleScope | None = None,
                          **layout_options) -> PreprocessResult:
    """Validate and execute a frozen declaration, recording the actual operator.

    Adapters own identities, units, reference/QC fitting and delivery clocks.
    This boundary owns all numeric lowering and source-sample support.
    """
    from data.epochs import (
        PREPROCESSING_DECLARATION_SOURCE_SHA256,
        PREPROCESSING_OPERATOR_IDENTITY,
        _semantic_sha256,
    )

    source_rate = float(raw.info["sfreq"])
    _validate_execution(preprocessing, source_rate)
    events = np.asarray(events)
    source_epoch_support(events, source_rate, preprocessing)
    scope_context = None
    if raw_scope is not None:
        if not isinstance(raw_scope, RawSampleScope):
            raise TypeError("raw_scope must be a RawSampleScope")
        raw_scope.validate_source(raw, events)
        scope_context = {"declaration": raw_scope.record(), "parent_first_samp": int(raw.first_samp),
                         "parent_n_times": int(raw.n_times), "parent_event_count": len(events),
                         "operator_order": "crop_original_raw_before_filtering_and_epoching"}
        raw = crop_raw_scope(raw, raw_scope)
        events = events[list(raw_scope.event_indices)]
    allowed = {"channels", "montage", "positions_m", "coordinate_frame", "fiducials_m",
               "coordinate_source", "icp_target_m", "allow_spherical_fallback",
               "channel_aliases", "event_id", "copy", "verbose"}
    if set(layout_options) - allowed:
        raise ValueError(f"execution options cannot override the contract: {sorted(set(layout_options) - allowed)}")
    start = int(round(preprocessing.tmin_ms / 1000 * source_rate))
    end = int(round(preprocessing.tmax_ms / 1000 * source_rate))
    source_n_times = end - start + 1
    do_resample = not np.isclose(source_rate, preprocessing.sfreq)
    resampled_n_times = int(round(source_n_times * preprocessing.sfreq / source_rate)) if do_resample else source_n_times
    if source_n_times < 1 or resampled_n_times < preprocessing.n_times:
        raise ValueError("source epoch sample geometry cannot produce the requested model time axis")
    options = _epoch_transform_options(preprocessing)
    options.update(tmax=preprocessing.tmax_ms / 1000, l_freq=preprocessing.l_freq,
        h_freq=preprocessing.h_freq, filter_method=preprocessing.filter_method,
        filter_order=preprocessing.filter_order, filter_phase=preprocessing.filter_phase,
        causal_iir_initial_state=preprocessing.causal_iir_initial_state,
        notch_freqs_hz=preprocessing.notch_freqs_hz,
        notch_quality_factor=preprocessing.notch_quality_factor,
        zero_phase_edge_policy=preprocessing.zero_phase_edge_policy,
        resample_domain=preprocessing.resample_domain,
        reject_threshold=preprocessing.reject_threshold_v)
    # Capture original context before copy=False is allowed to select/rename channels.
    original_names = tuple(raw.ch_names)
    aliases = layout_options.get("channel_aliases")
    segments = {canonical_channel_name(name, aliases=aliases): continuous_filter_plan(raw, name)
                for name in _eeg_channel_names(raw)}
    if (preprocessing.filter_phase == "zero" and preprocessing.zero_phase_edge_policy == "legacy"
            and not preprocessing.notch_freqs_hz):
        # Raw.filter applies skip annotations globally, including annotations
        # carrying channel names. Record its actual support, not our local backend's.
        global_segments = continuous_filter_plan(raw)
        segments = {name: global_segments for name in segments}
    first_samp, raw_n_times = int(raw.first_samp), int(raw.n_times)
    result = preprocess(raw, events, **options, **layout_options)
    if raw_scope is not None:
        scheduled = np.asarray(raw_scope.event_indices, dtype=np.int64)
        result = replace(result, event_indices=scheduled[result.event_indices],
                         scheduled_event_indices=scheduled,
                         timing_source=result.timing_source + ";independent_raw_sample_scope")
    sos = design_continuous_sos(source_rate, l_freq=preprocessing.l_freq,
        h_freq=preprocessing.h_freq, order=preprocessing.filter_order,
        notch_freqs_hz=preprocessing.notch_freqs_hz,
        notch_quality_factor=preprocessing.notch_quality_factor)
    active_filter = bool(len(sos))
    if not active_filter:
        backend, padding_limit = "identity", 0
    elif preprocessing.filter_phase == "forward":
        backend, padding_limit = "scipy.sosfilt", 0
    elif preprocessing.zero_phase_edge_policy == "mne_reflect_limited" or not preprocessing.notch_freqs_hz:
        backend = "mne_reflect_limited"
        padding_limit = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    else:
        backend = "scipy.sosfiltfilt_odd"
        padding_limit = int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())))
    receipt = {
        "schema": "n2p3_preprocessing_execution/1",
        "declaration": preprocessing.record(),
        "semantic_identity": PREPROCESSING_OPERATOR_IDENTITY,
        "implementation": {"python": platform.python_version(), "mne": mne.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "source_sha256": {"preprocess.py": _LOADED_SOURCE_SHA256,
                              "epochs.py": PREPROCESSING_DECLARATION_SOURCE_SHA256}},
        "source_context": {"sample_rate_hz": source_rate, "first_samp": first_samp,
            "n_times": raw_n_times, "channel_names": list(original_names),
            "filter_segments": {name: [[span.start_sample, span.stop_sample] for span in segments[name]]
                                for name in result.channel_names},
            "filter_segment_boundaries": {name: [span.record() for span in segments[name]]
                                          for name in result.channel_names},
            **({"raw_scope": scope_context} if scope_context is not None else {})},
        "filter": {"backend": backend, "sos": sos.tolist(), "padding_limit_samples": padding_limit,
                   "initial_state": preprocessing.causal_iir_initial_state,
                   "segments_are_half_open_record_offsets": True},
        "epoch_grid": {"start_offset_samples": start, "end_offset_samples": end,
            "source_n_times": source_n_times, "resampled_n_times": resampled_n_times,
            "output_n_times": result.n_times, "requested_tmin_s": result.tmin,
            "actual_source_tmin_s": start / source_rate, "resampled": do_resample},
        "output": {"shape": list(result.data.shape), "dtype": str(result.data.dtype),
            "channel_names": list(result.channel_names),
            "scheduled_events": len(result.event_samples),
            "scheduled_event_indices_sha256": _semantic_sha256(result.scheduled_event_indices.tolist()),
            "event_samples_sha256": _semantic_sha256(result.event_samples.tolist()),
            "selected_event_indices_sha256": _semantic_sha256(result.event_indices.tolist())},
        "timing_source": result.timing_source,
    }
    # Normalize tuples/numpy scalars for a stable JSON/cache round trip.
    receipt = json.loads(json.dumps(receipt, default=lambda value: value.item()))
    receipt["receipt_sha256"] = _semantic_sha256(receipt)
    return replace(result, execution_receipt=receipt)


def preprocess(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
    l_freq: float | None = DEFAULT_P300_DATA_CONTRACT.l_freq,
    h_freq: float | None = DEFAULT_P300_DATA_CONTRACT.h_freq,
    tmin: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
    tmax: float = DEFAULT_P300_DATA_CONTRACT.tmax_ms / 1000.0,
    n_times: int | None = DEFAULT_P300_DATA_CONTRACT.n_times,
    reject_threshold: float | None = DEFAULT_REJECT_THRESHOLD,
    baseline: tuple[float, float] | None = None,
    baseline_mode: str = DEFAULT_P300_DATA_CONTRACT.baseline_mode,
    trial_reference_window_ms: tuple[float, float] | None = None,
    trial_reference_center: str = "mean",
    trial_reference_scale: str = "none",
    filter_method: str = DEFAULT_P300_DATA_CONTRACT.filter_method,
    filter_order: int = DEFAULT_P300_DATA_CONTRACT.filter_order,
    notch_freqs_hz: Sequence[float] = (),
    notch_quality_factor: float = 30.0,
    zero_phase_edge_policy: str = "legacy",
    filter_phase: str = DEFAULT_P300_DATA_CONTRACT.filter_phase,
    causal_iir_initial_state: str = DEFAULT_P300_DATA_CONTRACT.causal_iir_initial_state,
    resample_domain: str = DEFAULT_P300_DATA_CONTRACT.resample_domain,
    resample_method: str = DEFAULT_P300_DATA_CONTRACT.resample_method,
    resample_npad: str = DEFAULT_P300_DATA_CONTRACT.resample_npad,
    resample_window: str = DEFAULT_P300_DATA_CONTRACT.resample_window,
    resample_pad: str = DEFAULT_P300_DATA_CONTRACT.resample_pad,
    channels: Sequence[str] | None = None,
    montage: str | Path | mne.channels.DigMontage | None = DEFAULT_MONTAGE,
    positions_m: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
    coordinate_frame: str = "head",
    fiducials_m: Mapping[str, Sequence[float]] | None = None,
    coordinate_source: str | None = None,
    icp_target_m: Sequence[Sequence[float]] | np.ndarray | None = None,
    allow_spherical_fallback: bool = False,
    channel_aliases: Mapping[str, str] | None = None,
    event_id: Mapping[str, int] | Sequence[int] | None = None,
    copy: bool = True,
    verbose: bool = False,
) -> PreprocessResult:
    """Map/filter continuous EEG, epoch on source events, then derive model rate."""

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"events must be an MNE (n,3) array, got {events.shape}.")
    if not np.issubdtype(events.dtype, np.integer) or np.issubdtype(events.dtype, np.bool_):
        raise ValueError("events must have an integer dtype.")
    if not tmin < tmax:
        raise ValueError("tmin must be smaller than tmax.")
    if n_times is not None and n_times <= 0:
        raise ValueError("n_times must be positive or None.")
    if reject_threshold is not None:
        raise ValueError(
            "Fixed absolute-voltage epoch rejection is retired; use fold-local artifact QC."
        )
    if baseline is not None:
        raise ValueError(
            "MNE baseline= is not part of the executable contract; use baseline_mode instead."
        )
    validate_notch_parameters(notch_freqs_hz, notch_quality_factor,
                              sample_rate_hz=float(raw.info["sfreq"]))
    validate_zero_phase_edge_policy(zero_phase_edge_policy, filter_method, filter_phase)
    if resample_domain != "epoched":
        raise ValueError("Executable preprocessing requires resample_domain='epoched'.")
    if (
        filter_method != DEFAULT_P300_DATA_CONTRACT.filter_method
        or filter_order != DEFAULT_P300_DATA_CONTRACT.filter_order
        or filter_phase not in {"zero", "forward"}
    ):
        raise ValueError(
            "Executable preprocessing requires fourth-order IIR with "
            "phase='zero' or phase='forward'."
        )
    if (
        resample_method != DEFAULT_P300_DATA_CONTRACT.resample_method
        or resample_npad != DEFAULT_P300_DATA_CONTRACT.resample_npad
        or resample_window != DEFAULT_P300_DATA_CONTRACT.resample_window
        or resample_pad != DEFAULT_P300_DATA_CONTRACT.resample_pad
    ):
        raise ValueError(
            "Executable preprocessing requires FFT resampling with "
            "npad/window='auto' and pad='edge'."
        )
    if copy:
        raw = raw.copy()

    embedded_montage = raw.get_montage()
    raw, observed_mask, target_names = map_channels(
        raw,
        channels=channels,
        aliases=channel_aliases,
        copy=False,
    )
    # MNE readers default to preload=False. Every following transform requires
    # real samples in memory, so the preprocessing boundary owns this transition.
    raw.load_data()
    selected_data = raw.get_data()
    if not np.isfinite(selected_data).all():
        count = int(np.count_nonzero(~np.isfinite(selected_data)))
        raise ValueError(
            f"Selected EEG channels contain {count} non-finite samples; repair or reject "
            "the source recording explicitly. The standard ingress never imputes samples."
        )
    if l_freq is not None or h_freq is not None or notch_freqs_hz:
        raw = filter_continuous(
            raw,
            l_freq=l_freq,
            h_freq=h_freq,
            method=filter_method,
            order=filter_order,
            notch_freqs_hz=notch_freqs_hz,
            notch_quality_factor=notch_quality_factor,
            zero_phase_edge_policy=zero_phase_edge_policy,
            phase=filter_phase,
            causal_iir_initial_state=causal_iir_initial_state,
            copy=False,
            verbose=verbose,
        )

    if event_id is None:
        event_id = [int(value) for value in np.unique(events[:, 2])]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        on_missing="warn",
        verbose=verbose,
    )
    source_rate = float(epochs.info["sfreq"])
    start_offset = int(round(float(epochs.times[0]) * source_rate))
    end_offset = int(round(float(epochs.times[-1]) * source_rate))
    effective_rate = source_rate if np.isclose(source_rate, sfreq) else float(sfreq)
    data = _transform_epoch_samples(epochs.get_data(), source_rate,
        sfreq=sfreq, n_times=n_times, tmin=tmin, baseline_mode=baseline_mode,
        trial_reference_window_ms=trial_reference_window_ms,
        trial_reference_center=trial_reference_center, trial_reference_scale=trial_reference_scale,
        resample_method=resample_method, resample_npad=resample_npad,
        resample_window=resample_window, resample_pad=resample_pad)
    selection = epochs.selection.astype(np.int64, copy=True)
    event_samples = np.asarray(events[:, 0], dtype=np.int64).copy()
    event_times_s = event_samples.astype(float) / float(raw.info["sfreq"])
    event_statuses = np.repeat("acquisition_rejected", len(events)).astype("U32")
    event_status_details = np.repeat("epoch_not_selected", len(events)).astype("U128")
    for event_index, reasons in enumerate(epochs.drop_log):
        if reasons:
            normalized_reasons = tuple(str(reason) for reason in reasons)
            event_status_details[event_index] = ";".join(normalized_reasons)
            if any(reason in {"NO_DATA", "TOO_SHORT"} for reason in normalized_reasons):
                event_statuses[event_index] = "boundary_dropped"
    event_statuses[selection] = "available"
    event_status_details[selection] = ""
    dropped = np.array([], dtype=np.int64)
    event_indices = selection

    event_evidence_indices = np.full(len(events), -1, dtype=np.int64)
    event_evidence_indices[event_indices] = np.arange(len(event_indices), dtype=np.int64)
    evidence_available_times_s = np.full(len(events), np.nan, dtype=float)
    source_epoch_start_samples = event_samples + start_offset
    source_epoch_end_samples = event_samples + end_offset
    evidence_available_times_s[event_indices] = np.maximum(
        event_times_s[event_indices] + float(tmax),
        source_epoch_end_samples[event_indices] / source_rate)
    timing_source = "source_mne_event_samples;source_grid_closed_epoch;max_nominal_and_source_endpoint"
    if filter_phase == "zero" and (l_freq is not None or h_freq is not None or notch_freqs_hz):
        # Reverse filtering consumes the completed recording. Use a conservative
        # recording-end bound, including across independently filtered segments.
        evidence_available_times_s[event_indices] = np.maximum(
            evidence_available_times_s[event_indices], (raw.first_samp + raw.n_times) / source_rate)
        timing_source += ";complete_recording_end"

    explicit_positions = positions_m
    layout_montage = montage
    registration_source = coordinate_source if positions_m is not None else None
    use_embedded = isinstance(montage, str) and montage == "embedded"
    if positions_m is None and embedded_montage is not None and montage == DEFAULT_MONTAGE:
        # The default average-head template is a fallback, not an instruction
        # to discard subject digitization already carried by the recording.
        use_embedded = True
    if use_embedded:
        layout_montage = None if positions_m is not None else embedded_montage
        registration_source = coordinate_source or "individual_digitization"
    layout = resolve_channel_layout(
        target_names,
        positions_m=explicit_positions,
        montage=layout_montage,
        aliases=channel_aliases,
        allow_missing=False,
        coordinate_frame=coordinate_frame,
        fiducials_m=fiducials_m,
        coordinate_source=registration_source,
        icp_target_m=icp_target_m,
        allow_spherical_fallback=allow_spherical_fallback,
    )
    output = data.astype(np.float32, copy=False)

    return PreprocessResult(
        data=output,
        channel_names=layout.names,
        channel_positions_m=layout.positions_m,
        channel_mask=observed_mask,
        position_mask=layout.position_mask,
        layout_source=layout.source,
        coordinate_registration=layout.registration,
        sfreq=effective_rate,
        tmin=float(tmin),
        tmax=float(tmax),
        dropped=dropped,
        event_indices=event_indices,
        event_samples=event_samples,
        event_times_s=event_times_s,
        evidence_available_times_s=evidence_available_times_s,
        event_statuses=event_statuses,
        event_status_details=event_status_details,
        event_evidence_indices=event_evidence_indices,
        online_causal=filter_phase == "forward",
        causal_iir_initial_state=causal_iir_initial_state,
        source_epoch_start_samples=source_epoch_start_samples,
        source_epoch_end_samples=source_epoch_end_samples,
        timing_source=timing_source,
    )
