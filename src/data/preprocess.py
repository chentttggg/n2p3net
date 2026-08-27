"""Format-independent continuous EEG preprocessing with explicit physical contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from data.channel import (
    DEFAULT_MONTAGE,
    CoordinateRegistrationSpec,
    canonical_channel_name,
    resolve_channel_layout,
)

DEFAULT_REJECT_THRESHOLD: float | None = None


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


def _canonical(name: str) -> str:
    """Backward name for tests and small utilities; use canonical_channel_name in new code."""

    return canonical_channel_name(name)


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
    sfreq: float = 256.0,
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


def filter_continuous(
    raw: mne.io.BaseRaw,
    l_freq: float | None = 0.1,
    h_freq: float | None = None,
    *,
    method: str = "iir",
    phase: str = "zero",
    copy: bool = False,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply a declared continuous-domain high/low/band-pass before epoching."""

    if l_freq is None and h_freq is None:
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
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        iir_params=dict(order=4, ftype="butter") if method == "iir" else None,
        phase=phase,
        verbose=verbose,
    )
    return raw


def highpass(
    raw: mne.io.BaseRaw,
    l_freq: float = 0.1,
    *,
    method: str = "iir",
    phase: str = "zero",
    copy: bool = False,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Backward-compatible high-pass wrapper around :func:`filter_continuous`."""

    if l_freq <= 0:
        raise ValueError("l_freq must be positive.")
    return filter_continuous(
        raw,
        l_freq=l_freq,
        h_freq=None,
        method=method,
        phase=phase,
        copy=copy,
        verbose=verbose,
    )


def reject_epochs(
    data: np.ndarray,
    threshold: float | None = DEFAULT_REJECT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject the retired fixed-threshold artifact path explicitly."""

    del data
    if threshold is not None:
        raise ValueError(
            "Fixed absolute-voltage epoch rejection is retired; use fold-local artifact QC."
        )
    raise ValueError("reject_epochs is retired; cache all finite epochs for fold-local QC.")


def preprocess(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    sfreq: float = 256.0,
    l_freq: float | None = 0.1,
    h_freq: float | None = None,
    tmin: float = -0.2,
    tmax: float = 0.8,
    n_times: int | None = 256,
    reject_threshold: float | None = DEFAULT_REJECT_THRESHOLD,
    baseline: tuple[float, float] | None = None,
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
    """Map channels, resample, high-pass, epoch, and preserve provenance."""

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
    raw, events = resample(raw, events, sfreq=sfreq, copy=False)
    if l_freq is not None or h_freq is not None:
        raw = filter_continuous(
            raw,
            l_freq=l_freq,
            h_freq=h_freq,
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
        baseline=baseline,
        preload=True,
        on_missing="warn",
        verbose=verbose,
    )
    data = epochs.get_data()
    if not np.isfinite(data).all():
        count = int(np.count_nonzero(~np.isfinite(data)))
        raise ValueError(
            f"Epoching produced {count} non-finite samples; the standard ingress does not "
            "replace NaN or infinite values."
        )
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
    if n_times is not None:
        if data.shape[2] < n_times:
            raise ValueError(
                f"Epoching produced {data.shape[2]} samples, fewer than the physical contract "
                f"requires ({n_times})."
            )
        data = data[:, :, :n_times]

    dropped = np.array([], dtype=np.int64)
    event_indices = selection

    event_evidence_indices = np.full(len(events), -1, dtype=np.int64)
    event_evidence_indices[event_indices] = np.arange(len(event_indices), dtype=np.int64)
    evidence_available_times_s = np.full(len(events), np.nan, dtype=float)
    evidence_available_times_s[event_indices] = event_times_s[event_indices] + float(tmax)

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
        sfreq=float(raw.info["sfreq"]),
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
        online_causal=False,
    )
