"""Declarative raw-EEG manifest ingestion for arbitrary MNE-supported formats."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.channel import DEFAULT_MONTAGE, canonical_channel_name
from data.dataset import EEGRecord, SubjectData, load_dataset, read_raw
from data.epochs import EpochDataset, PreprocessingSpec, concatenate_epoch_datasets

MANIFEST_SCHEMA = "n2p3net_raw_manifest/1"
_MANIFEST_FIELDS = {
    "schema",
    "name",
    "records",
    "preprocessing",
    "channels",
    "montage",
    "channel_aliases",
    "layout_policy",
    "label_map",
}
_RECORD_FIELDS = {
    "path",
    "subject_id",
    "event_id",
    "event_file",
    "label_map",
    "age",
    "sex",
    "reader_kwargs",
    "metadata",
    "session",
    "run",
    "reference",
    "candidate_ids",
    "target_candidate_ids",
    "repetition_indices",
}


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    records: tuple[EEGRecord, ...]
    preprocessing: PreprocessingSpec
    channels: tuple[str, ...] | None = None
    montage: str | Path | None = DEFAULT_MONTAGE
    channel_aliases: Mapping[str, str] | None = None
    layout_policy: str = "intersection"
    source_path: Path | None = None

    def validate(self) -> None:
        self.preprocessing.validate()
        if not isinstance(self.name, str) or not self.name.strip() or not self.records:
            raise ValueError("A dataset manifest needs a name and at least one record.")
        if self.layout_policy not in {"intersection", "union", "strict"}:
            raise ValueError("layout_policy must be intersection, union, or strict.")
        if self.channels is not None:
            if not self.channels or any(
                not isinstance(channel, str) or not channel.strip() for channel in self.channels
            ):
                raise ValueError("Manifest channels must be non-empty strings.")
        if self.channel_aliases is not None and (
            not isinstance(self.channel_aliases, Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in self.channel_aliases.items()
            )
        ):
            raise ValueError("channel_aliases must map strings to strings.")
        subject_ids = [record.subject_id for record in self.records]
        if any(not subject_id for subject_id in subject_ids):
            raise ValueError("Every manifest record needs a non-empty subject_id.")


def _resolve_relative(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load a JSON manifest; paths are resolved relative to the manifest file."""

    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest {path} must contain a JSON object.")
    unknown_manifest_fields = set(payload) - _MANIFEST_FIELDS
    if unknown_manifest_fields:
        raise ValueError(
            f"Manifest {path} contains unknown fields {sorted(unknown_manifest_fields)}."
        )
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Manifest {path} must declare schema={MANIFEST_SCHEMA!r}, got {payload.get('schema')!r}."
        )
    base = path.parent
    preprocessing_payload = payload.get("preprocessing", {})
    if not isinstance(preprocessing_payload, dict):
        raise ValueError("Manifest preprocessing must be an object.")
    preprocessing = PreprocessingSpec(**preprocessing_payload)
    global_label_map = payload.get("label_map")
    if global_label_map is not None and not isinstance(global_label_map, dict):
        raise ValueError("Manifest label_map must be an object.")
    record_payloads = payload.get("records", [])
    if not isinstance(record_payloads, list):
        raise ValueError("Manifest records must be an array.")
    records: list[EEGRecord] = []
    for item in record_payloads:
        if not isinstance(item, dict):
            raise ValueError("Every manifest record must be an object.")
        unknown_record_fields = set(item) - _RECORD_FIELDS
        if unknown_record_fields:
            raise ValueError(
                f"Manifest record contains unknown fields {sorted(unknown_record_fields)}."
            )
        record_path = _resolve_relative(base, item.get("path"))
        if record_path is None:
            raise ValueError("Each manifest record requires path.")
        metadata_payload = item.get("metadata", {})
        reader_kwargs = item.get("reader_kwargs", {})
        if not isinstance(metadata_payload, dict) or not isinstance(reader_kwargs, dict):
            raise ValueError("Record metadata and reader_kwargs must be objects.")
        metadata = dict(metadata_payload)
        metadata["dataset_id"] = str(payload.get("name", ""))
        for field_name in ("session", "run", "reference"):
            if item.get(field_name) is not None:
                metadata[field_name] = item[field_name]
        records.append(
            EEGRecord(
                path=record_path,
                subject_id=str(item.get("subject_id", "")),
                event_id=item.get("event_id"),
                event_file=_resolve_relative(base, item.get("event_file")),
                label_map=item.get("label_map", global_label_map),
                age=item.get("age"),
                sex=item.get("sex"),
                reader_kwargs=reader_kwargs,
                metadata=metadata,
                candidate_ids=item.get("candidate_ids"),
                target_candidate_ids=item.get("target_candidate_ids"),
                repetition_indices=item.get("repetition_indices"),
            )
        )
    montage: str | Path | None = payload.get("montage", DEFAULT_MONTAGE)
    if isinstance(montage, str):
        possible_path = base / montage
        if possible_path.exists():
            montage = possible_path.resolve()
    manifest = DatasetManifest(
        name=str(payload.get("name", "")),
        records=tuple(records),
        preprocessing=preprocessing,
        channels=(tuple(payload["channels"]) if payload.get("channels") else None),
        montage=montage,
        channel_aliases=payload.get("channel_aliases"),
        layout_policy=payload.get("layout_policy", "intersection"),
        source_path=path,
    )
    manifest.validate()
    return manifest


def _record_eeg_names(record: EEGRecord, aliases: Mapping[str, str] | None) -> tuple[str, ...]:
    raw = read_raw(record.path, **dict(record.reader_kwargs))
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, exclude=[])
    names = tuple(canonical_channel_name(raw.ch_names[index], aliases=aliases) for index in picks)
    if len(set(names)) != len(names):
        raise ValueError(f"Record {record.path} has duplicate canonical EEG names {names}.")
    return names


def resolve_manifest_channels(manifest: DatasetManifest) -> tuple[str, ...]:
    """Resolve one fixed physical tensor layout for the entire training dataset."""

    if manifest.channels is not None:
        channels = tuple(
            canonical_channel_name(name, aliases=manifest.channel_aliases)
            for name in manifest.channels
        )
        if len(set(channels)) != len(channels):
            raise ValueError(
                f"Explicit manifest channels are duplicated after normalization: {channels}."
            )
        return channels
    layouts = [_record_eeg_names(record, manifest.channel_aliases) for record in manifest.records]
    first = layouts[0]
    if manifest.layout_policy == "strict":
        first_set = set(first)
        mismatched = [
            index for index, names in enumerate(layouts[1:], start=1) if set(names) != first_set
        ]
        if mismatched:
            raise ValueError(
                f"Strict layout policy found different EEG channel sets in records {mismatched}."
            )
        return first
    common = set(first)
    for names in layouts[1:]:
        common &= set(names)
    if manifest.layout_policy == "union":
        channels = tuple(dict.fromkeys(name for names in layouts for name in names))
    else:
        channels = tuple(name for name in first if name in common)
    if not channels:
        raise ValueError("Manifest records have no common EEG channels.")
    return channels


def _subject_epoch_dataset(
    subject: SubjectData,
    preprocessing: PreprocessingSpec,
    *,
    dataset_name: str,
) -> EpochDataset:
    metadata = {
        "subject": np.repeat(subject.subject_id, subject.n_epochs),
        "age": np.repeat(np.nan if subject.age is None else subject.age, subject.n_epochs),
        "sex": np.repeat(subject.sex, subject.n_epochs),
    }
    for key, value in subject.metadata.items():
        if np.isscalar(value) or value is None:
            metadata[str(key)] = np.repeat(value, subject.n_epochs)
        else:
            values = np.asarray(value)
            if values.ndim != 1 or len(values) != subject.n_epochs:
                raise ValueError(f"Subject metadata {key!r} must be scalar or align with epochs.")
            metadata[str(key)] = values
    dataset = EpochDataset(
        name=dataset_name,
        X=subject.data.astype(np.float32, copy=False),
        y=subject.labels,
        subject_ids=np.repeat(subject.subject_id, subject.n_epochs).astype(str),
        channel_names=subject.channel_names,
        channel_positions_m=subject.channel_positions_m,
        channel_mask=subject.channel_mask,
        preprocessing=preprocessing,
        event_timeline=subject.event_timeline,
        metadata=pd.DataFrame(metadata),
        provenance={
            "subject_id": subject.subject_id,
            "coordinate_registration": subject.coordinate_registration.record(),
        },
    )
    dataset.validate()
    return dataset


def build_manifest_dataset(manifest: DatasetManifest) -> EpochDataset:
    """Materialize a manifest into the universal EpochDataset contract."""

    manifest.validate()
    channels = resolve_manifest_channels(manifest)
    spec = manifest.preprocessing
    record_layouts = [
        _record_eeg_names(record, manifest.channel_aliases) for record in manifest.records
    ]
    subjects: list[SubjectData] = []
    for record, record_layout in zip(manifest.records, record_layouts, strict=True):
        available = set(record_layout)
        selected_channels = tuple(channel for channel in channels if channel in available)
        if not selected_channels:
            raise ValueError(
                f"Record {record.path} has no channels in the resolved layout {channels}."
            )
        subjects.extend(
            load_dataset(
                (record,),
                preprocess_kwargs={
                    "sfreq": spec.sfreq,
                    "l_freq": spec.l_freq,
                    "h_freq": spec.h_freq,
                    "tmin": spec.tmin_ms / 1000.0,
                    "tmax": spec.tmax_ms / 1000.0,
                    "n_times": spec.n_times,
                    "reject_threshold": spec.reject_threshold_v,
                    "baseline": None,
                    "channels": selected_channels,
                    "montage": manifest.montage,
                    "channel_aliases": manifest.channel_aliases,
                },
            )
        )

    position_samples: dict[str, list[np.ndarray]] = {channel: [] for channel in channels}
    for subject in subjects:
        for channel, position, observed in zip(
            subject.channel_names,
            subject.channel_positions_m,
            subject.channel_mask,
            strict=True,
        ):
            if observed:
                position_samples[channel].append(np.asarray(position, dtype=np.float64))
    missing_positions = [channel for channel, values in position_samples.items() if not values]
    if missing_positions:
        raise ValueError(f"Resolved channels lack registered positions: {missing_positions}.")
    positions = np.asarray(
        [np.mean(position_samples[channel], axis=0) for channel in channels],
        dtype=np.float32,
    )

    per_subject: list[EpochDataset] = []
    for subject in subjects:
        native = _subject_epoch_dataset(subject, spec, dataset_name=manifest.name)
        destination = {channel: index for index, channel in enumerate(channels)}
        aligned = np.zeros(
            (native.n_epochs, len(channels), native.n_times), dtype=np.float32
        )
        trial_mask = np.zeros((native.n_epochs, len(channels)), dtype=bool)
        for source_index, channel in enumerate(native.channel_names):
            target_index = destination[channel]
            aligned[:, target_index] = native.X[:, source_index]
            if native.channel_mask[source_index]:
                trial_mask[:, target_index] = True
        per_subject.append(
            EpochDataset(
                name=native.name,
                X=aligned,
                y=native.y,
                subject_ids=native.subject_ids,
                channel_names=channels,
                channel_positions_m=positions,
                channel_mask=np.ones(len(channels), dtype=bool),
                preprocessing=native.preprocessing,
                event_timeline=native.event_timeline,
                metadata=native.metadata,
                provenance=native.provenance,
                trial_channel_mask=(None if trial_mask.all() else trial_mask),
            )
        )
    return concatenate_epoch_datasets(
        per_subject,
        name=manifest.name,
        provenance={
            "source": "raw_manifest",
            "manifest_schema": MANIFEST_SCHEMA,
            "manifest_path": str(manifest.source_path) if manifest.source_path else None,
            "layout_policy": manifest.layout_policy,
            "montage": str(manifest.montage),
            "coordinate_registration": {
                "priority": [
                    "individual_digitization",
                    "device_montage",
                    "average_head_template",
                    "explicit_unit_sphere_fallback",
                ],
                "per_subject": {
                    subject.subject_id: subject.coordinate_registration.record()
                    for subject in subjects
                },
            },
        },
    )
