"""Generic EEG file, event, and subject-record ingestion built on MNE."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from data.channel import DEFAULT_N_FREQS, CoordinateRegistrationSpec
from data.events import ScheduledEventTimeline, selection_group_id
from data.metadata import build_subject_embedding, normalize_sex
from data.preprocess import PreprocessResult, preprocess


def _normalize_label_map(label_map: Mapping[str, int]) -> dict[str, int]:
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in label_map.values()
    ):
        raise ValueError("label_map values must be integers; implicit truncation is forbidden.")
    return {str(key): int(value) for key, value in label_map.items()}


@dataclass(frozen=True)
class EEGRecord:
    """One continuous recording and its event-level supervision."""

    path: Path
    subject_id: str
    events: np.ndarray | None = None
    labels: np.ndarray | None = None
    event_id: Mapping[str, int] | None = None
    event_file: Path | None = None
    label_map: Mapping[str, int] | None = None
    age: float | None = None
    sex: str | int | None = None
    reader_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SubjectData:
    data: np.ndarray
    labels: np.ndarray | None
    channel_names: tuple[str, ...]
    channel_positions_m: np.ndarray
    channel_mask: np.ndarray
    E_chn: np.ndarray
    E_sub: np.ndarray
    age: float | None
    sex: str
    sfreq: float
    tmin: float
    tmax: float
    subject_id: str
    event_timeline: ScheduledEventTimeline
    coordinate_registration: CoordinateRegistrationSpec
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])


def read_raw(path: str | Path, *, preload: bool = False, **kwargs) -> mne.io.BaseRaw:
    """Read any continuous format supported by MNE's format dispatcher."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return mne.io.read_raw(path, preload=preload, verbose=False, **kwargs)
    except Exception as exc:  # noqa: BLE001 - preserve format-specific cause
        raise ValueError(
            f"MNE could not read continuous EEG source {path}. The format may require an "
            "optional dependency or a companion/header file."
        ) from exc


def read_epochs(path: str | Path, *, preload: bool = True, **kwargs) -> mne.BaseEpochs:
    """Read common pre-epoched FIF or EEGLAB SET files."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".set":
        return mne.io.read_epochs_eeglab(path, verbose=False, **kwargs)
    if suffix in {".fif", ".gz"}:
        return mne.read_epochs(path, preload=preload, verbose=False, **kwargs)
    raise ValueError(
        f"Unsupported epoched EEG file {path}. Use FIF/EEGLAB SET or convert to EpochDataset NPZ."
    )


def events_from_annotations(
    raw: mne.io.BaseRaw,
    *,
    event_id: Mapping[str, int] | None = None,
) -> np.ndarray:
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    return events.astype(np.int64, copy=False)


def annotation_events_and_labels(
    raw: mne.io.BaseRaw,
    label_map: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract annotation events and map descriptions to integer labels."""

    events, description_ids = mne.events_from_annotations(raw, verbose=False)
    normalized_map = _normalize_label_map(label_map)
    code_to_label = {
        code: normalized_map[description]
        for description, code in description_ids.items()
        if description in normalized_map
    }
    keep = np.asarray([code in code_to_label for code in events[:, 2]], dtype=bool)
    if not keep.any():
        raise ValueError(
            f"No annotations match label_map keys {sorted(normalized_map)}; "
            f"available={sorted(description_ids)}."
        )
    selected = events[keep].astype(np.int64, copy=False)
    labels = np.asarray([code_to_label[int(code)] for code in selected[:, 2]], dtype=np.int64)
    return selected, labels


def read_event_table(
    path: str | Path,
    *,
    sfreq: float,
    label_map: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read BIDS-like TSV/CSV/JSON or NumPy event arrays."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        events = np.load(path, allow_pickle=False)
        labels = None
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "events" not in archive.files:
                raise ValueError(f"{path} must contain an 'events' array.")
            events = np.asarray(archive["events"])
            labels = np.asarray(archive["labels"]) if "labels" in archive.files else None
    else:
        if suffix == ".tsv":
            table = pd.read_csv(path, sep="\t")
        elif suffix == ".csv":
            table = pd.read_csv(path)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            table = pd.DataFrame(payload["events"] if isinstance(payload, dict) else payload)
        else:
            raise ValueError("Event files must be TSV, CSV, JSON, NPY, or NPZ.")
        if "sample" in table:
            samples = table["sample"].to_numpy()
            if not np.issubdtype(samples.dtype, np.integer):
                raise ValueError(f"{path} sample values must have an integer dtype.")
            samples = samples.astype(np.int64, copy=False)
        elif "onset" in table:
            samples = np.rint(table["onset"].to_numpy(dtype=float) * sfreq).astype(np.int64)
        else:
            raise ValueError(f"{path} needs a 'sample' column or an 'onset' column in seconds.")
        if "event_id" in table:
            codes = table["event_id"].to_numpy()
            if not np.issubdtype(codes.dtype, np.integer):
                raise ValueError(f"{path} event_id values must have an integer dtype.")
            codes = codes.astype(np.int64, copy=False)
        else:
            codes = np.arange(1, len(table) + 1, dtype=np.int64)
        events = np.column_stack([samples, np.zeros(len(table), dtype=np.int64), codes])
        labels = None
        label_column = next(
            (name for name in ("label", "target", "trial_type") if name in table), None
        )
        if label_column is not None:
            values = table[label_column]
            if label_map:
                mapping = _normalize_label_map(label_map)
                unknown = sorted(set(values.astype(str)) - set(mapping))
                if unknown:
                    raise ValueError(f"Unmapped event labels in {path}: {unknown}.")
                labels = values.astype(str).map(mapping).to_numpy(dtype=np.int64)
            else:
                labels = values.to_numpy()
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"Event array must be (n,3), got {events.shape} from {path}.")
    if not np.issubdtype(events.dtype, np.integer) or np.issubdtype(events.dtype, np.bool_):
        raise ValueError(f"Event array from {path} must have an integer dtype.")
    events = events.astype(np.int64, copy=False)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(events):
            raise ValueError("Event labels must be one-dimensional and align with event rows.")
        if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
            raise ValueError("Event labels must have an integer dtype.")
        labels = labels.astype(np.int64, copy=False)
    return events, labels


def resolve_record_events(
    raw: mne.io.BaseRaw,
    record: EEGRecord,
) -> tuple[np.ndarray, np.ndarray | None]:
    if record.events is not None and record.event_file is not None:
        raise ValueError("EEGRecord cannot specify both events and event_file.")
    if record.events is not None:
        return np.asarray(record.events), record.labels
    if record.event_file is not None:
        events, labels = read_event_table(
            record.event_file,
            sfreq=float(raw.info["sfreq"]),
            label_map=record.label_map,
        )
        if record.labels is not None:
            if labels is not None:
                raise ValueError("Labels are present in both EEGRecord and its event file.")
            labels = np.asarray(record.labels)
        return events, labels
    if record.label_map is not None:
        return annotation_events_and_labels(raw, record.label_map)
    return events_from_annotations(raw, event_id=record.event_id), record.labels


def build_subject(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    labels: Sequence[int] | None = None,
    *,
    age: float | None = None,
    sex: str | int | None = None,
    subject_id: str = "anonymous_subject",
    metadata: Mapping[str, Any] | None = None,
    n_freqs: int = DEFAULT_N_FREQS,
    **preprocess_kwargs,
) -> SubjectData:
    result: PreprocessResult = preprocess(raw, events, **preprocess_kwargs)
    data = result.data
    aligned_labels: np.ndarray | None = None
    if labels is not None:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1 or not np.issubdtype(labels_array.dtype, np.integer):
            raise ValueError("labels must be a one-dimensional integer array.")
        if len(labels_array) != len(events):
            raise ValueError(
                f"labels/events length mismatch: {len(labels_array)} vs {len(events)}."
            )
        aligned_labels = labels_array[result.event_indices].astype(np.int64, copy=False)

    from data.channel import build_channel_identity

    identity = build_channel_identity(
        result.channel_names,
        channel_mask=result.channel_mask,
        positions_m=result.channel_positions_m,
        montage=None,
        n_freqs=n_freqs,
        allow_missing_positions=False,
    )
    subject = build_subject_embedding(age, sex, n_freqs=n_freqs)
    subject_metadata = dict(metadata or {})
    subject_metadata["acquisition_time_s"] = np.asarray(events)[result.event_indices, 0].astype(
        float
    ) / float(raw.info["sfreq"])
    dataset_id = str(subject_metadata.get("dataset_id", "manifest"))
    session_id = str(subject_metadata.get("session", ""))
    run_id = str(subject_metadata.get("run", ""))
    selection_id = str(subject_metadata.get("selection_id", subject_id))
    group_id = selection_group_id(dataset_id, subject_id, session_id, run_id, selection_id)
    event_timeline = ScheduledEventTimeline(
        event_ids=np.asarray(
            [
                f"{dataset_id}:{subject_id}:{session_id}:{run_id}:{index}"
                for index in range(len(events))
            ]
        ),
        group_ids=np.repeat(group_id, len(events)),
        subject_ids=np.repeat(str(subject_id), len(events)),
        stimulus_ids=np.asarray(events[:, 2], dtype=np.int64),
        onset_samples=result.event_samples,
        onset_times_s=result.event_times_s,
        evidence_available_times_s=result.evidence_available_times_s,
        evidence_indices=result.event_evidence_indices,
        statuses=result.event_statuses,
        status_details=result.event_status_details,
        dataset_ids=np.repeat(dataset_id, len(events)),
        session_ids=np.repeat(session_id, len(events)),
        run_ids=np.repeat(run_id, len(events)),
        selection_ids=np.repeat(selection_id, len(events)),
        complete=True,
        online_causal=result.online_causal,
        timing_source="resampled_mne_event_samples;epoch_right_edge",
    ).validate(n_epochs=len(data))
    return SubjectData(
        data=data,
        labels=aligned_labels,
        channel_names=result.channel_names,
        channel_positions_m=result.channel_positions_m,
        channel_mask=result.channel_mask,
        E_chn=identity.embedding,
        E_sub=subject.embedding,
        age=age,
        sex=normalize_sex(sex),
        sfreq=result.sfreq,
        tmin=result.tmin,
        tmax=result.tmax,
        subject_id=str(subject_id),
        event_timeline=event_timeline,
        coordinate_registration=result.coordinate_registration,
        metadata=subject_metadata,
    )


def load_dataset(
    records: Sequence[EEGRecord],
    *,
    n_freqs: int = DEFAULT_N_FREQS,
    preprocess_kwargs: Mapping[str, Any] | None = None,
) -> list[SubjectData]:
    """Load typed recording specifications without dataset-name branches."""

    kwargs = dict(preprocess_kwargs or {})
    output: list[SubjectData] = []
    for record in records:
        raw = read_raw(record.path, **dict(record.reader_kwargs))
        events, labels = resolve_record_events(raw, record)
        output.append(
            build_subject(
                raw,
                events,
                labels,
                age=record.age,
                sex=record.sex,
                subject_id=record.subject_id,
                metadata=record.metadata,
                n_freqs=n_freqs,
                **kwargs,
            )
        )
    return output
