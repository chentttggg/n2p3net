"""Validated BIDS-EEG raw ingress independent of any one acquisition device."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BidsEEGInputContract:
    """Configurable BIDS fields required by one downstream EEG adapter."""

    minimum_bids_version: tuple[int, int, int]
    task_name: str
    stimulus_trial_type: str
    rest_trial_type: str
    candidate_column: str
    candidate_vocabulary: tuple[str, ...]
    coordinate_units: str = "m"
    raw_extensions: tuple[str, ...] = (".edf", ".bdf", ".vhdr", ".set")

    @property
    def required_event_columns(self) -> tuple[str, ...]:
        return (
            "onset",
            "duration",
            "sample",
            "trial_type",
            "trial_id",
            "block_id",
            "trial_index",
            self.candidate_column,
            "rest_segment_id",
        )


@dataclass(frozen=True)
class BidsStimulusEvent:
    onset_seconds: float
    duration_seconds: float
    sample: int
    candidate_id: str
    trial_id: str
    block_id: int
    trial_index: int
    row_number: int


@dataclass(frozen=True)
class BidsRestInterval:
    onset_seconds: float
    duration_seconds: float
    sample: int
    rest_segment_id: str
    block_id: int
    row_number: int

    @property
    def end_seconds(self) -> float:
        return self.onset_seconds + self.duration_seconds


@dataclass(frozen=True)
class ValidatedBidsEEGRecording:
    root: Path
    raw_path: Path
    eeg_json_path: Path
    channels_path: Path
    events_path: Path
    electrodes_path: Path
    coordsystem_path: Path
    sample_rate_hz: float
    duration_seconds: float
    channel_names: tuple[str, ...]
    channel_statuses: tuple[str, ...]
    channel_positions_m: tuple[tuple[float, float, float], ...]
    source_reference: str
    stimuli: tuple[BidsStimulusEvent, ...]
    rest_intervals: tuple[BidsRestInterval, ...]


_REQUIRED_PATH_FIELDS = (
    "dataset_description",
    "recording",
    "eeg_json",
    "channels_tsv",
    "events_tsv",
    "electrodes_tsv",
    "coordsystem_json",
)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"BIDS {label} is missing: {path.name}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"BIDS {label} is not readable valid JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"BIDS {label} must contain a JSON object.")
    return value


def _read_tsv(path: Path, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        stream = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"Cannot open BIDS {label}: {path}.") from exc
    with stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        if not fields or any(not field for field in fields):
            raise ValueError(f"BIDS {label} has no valid header.")
        if len(set(fields)) != len(fields):
            raise ValueError(f"BIDS {label} has duplicate columns.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"BIDS {label} contains no data rows.")
    return fields, rows


def _contained_files(root: Path, files: Mapping[str, Any]) -> dict[str, Path]:
    missing = [field for field in _REQUIRED_PATH_FIELDS if field not in files]
    if missing:
        raise ValueError(f"BIDS file manifest lacks fields: {missing}.")
    output: dict[str, Path] = {}
    for field in _REQUIRED_PATH_FIELDS:
        value = files[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"BIDS file field {field} must be a non-empty relative path.")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"BIDS file field {field} must stay inside the dataset root.")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"BIDS file field {field} escapes the dataset root.") from exc
        if not resolved.is_file():
            raise ValueError(f"BIDS file field {field} does not exist: {value}.")
        output[field] = resolved
    return output


def _finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"BIDS {field} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BIDS {field} must be a finite number.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"BIDS {field} must be finite and at least {minimum}.")
    return result


def _integer_text(value: str, field: str, *, minimum: int = 0) -> int:
    text = str(value).strip()
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError(f"BIDS {field} must be an integer.") from exc
    if str(result) != text or result < minimum:
        raise ValueError(f"BIDS {field} must be an integer at least {minimum}.")
    return result


def _version_tuple(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError("BIDSVersion must be a semantic version string.")
    parts = value.strip().split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("BIDSVersion must contain three integer components.")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def epoch_rest_overlap_mask(
    onset_seconds: Sequence[float],
    *,
    tmin_seconds: float,
    tmax_seconds: float,
    rest_intervals: Sequence[BidsRestInterval],
) -> tuple[bool, ...]:
    """Return half-open interval intersections between epochs and retained rests."""

    if not math.isfinite(tmin_seconds) or not math.isfinite(tmax_seconds) or tmin_seconds >= tmax_seconds:
        raise ValueError("Epoch bounds must be finite and strictly increasing.")
    output: list[bool] = []
    for onset in onset_seconds:
        value = _finite_number(onset, "event onset", minimum=0.0)
        epoch_start = value + tmin_seconds
        epoch_end = value + tmax_seconds
        output.append(
            any(epoch_start < rest.end_seconds and epoch_end > rest.onset_seconds for rest in rest_intervals)
        )
    return tuple(output)


def validate_bids_eeg_recording(
    dataset_root: str | Path,
    files: Mapping[str, Any],
    *,
    contract: BidsEEGInputContract,
) -> ValidatedBidsEEGRecording:
    """Validate BIDS metadata and tables before invoking an EEG binary reader."""

    root = Path(dataset_root).expanduser().resolve()
    paths = _contained_files(root, files)
    if paths["recording"].suffix.casefold() not in contract.raw_extensions:
        raise ValueError(f"Unsupported BIDS EEG recording extension: {paths['recording'].suffix!r}.")

    dataset_description = _read_json_object(paths["dataset_description"], "dataset_description")
    version = _version_tuple(dataset_description.get("BIDSVersion"))
    if version[0] != contract.minimum_bids_version[0] or version < contract.minimum_bids_version:
        raise ValueError(
            f"BIDSVersion {version} is incompatible with minimum {contract.minimum_bids_version}."
        )
    if dataset_description.get("DatasetType", "raw") != "raw":
        raise ValueError("BrainSync preprocessing requires a BIDS raw dataset.")

    eeg_json = _read_json_object(paths["eeg_json"], "EEG sidecar")
    if eeg_json.get("TaskName") != contract.task_name:
        raise ValueError(f"BIDS TaskName must be {contract.task_name!r}.")
    sample_rate = _finite_number(eeg_json.get("SamplingFrequency"), "SamplingFrequency", minimum=1e-12)
    duration = _finite_number(eeg_json.get("RecordingDuration"), "RecordingDuration", minimum=1e-12)
    reference = eeg_json.get("EEGReference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("BIDS EEGReference must be a non-empty string.")

    channel_fields, channel_rows = _read_tsv(paths["channels_tsv"], "channels.tsv")
    required_channel_fields = {"name", "type", "units", "sampling_frequency", "status"}
    if not required_channel_fields.issubset(channel_fields):
        raise ValueError(f"BIDS channels.tsv lacks columns {sorted(required_channel_fields - set(channel_fields))}.")
    channel_names = tuple(row["name"].strip() for row in channel_rows)
    if any(not value for value in channel_names) or len(set(value.casefold() for value in channel_names)) != len(channel_names):
        raise ValueError("BIDS channels.tsv names must be unique and non-empty.")
    if any(row["type"].strip().upper() != "EEG" for row in channel_rows):
        raise ValueError("BrainSync BIDS input can contain EEG channels only.")
    for row in channel_rows:
        row_rate = _finite_number(row["sampling_frequency"], "channels.sampling_frequency", minimum=1e-12)
        if not math.isclose(row_rate, sample_rate, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("BIDS channel and EEG sidecar sampling frequencies conflict.")
    channel_statuses = tuple(row["status"].strip().casefold() for row in channel_rows)
    if any(status not in {"good", "bad"} for status in channel_statuses):
        raise ValueError("BIDS channel status must be good or bad.")

    coordinate_json = _read_json_object(paths["coordsystem_json"], "coordsystem.json")
    if coordinate_json.get("EEGCoordinateUnits") != contract.coordinate_units:
        raise ValueError(f"BIDS EEG coordinates must use {contract.coordinate_units!r} units.")
    coordinate_system = coordinate_json.get("EEGCoordinateSystem")
    if not isinstance(coordinate_system, str) or not coordinate_system.strip():
        raise ValueError("BIDS EEGCoordinateSystem must be declared.")
    if coordinate_system == "Other" and not str(
        coordinate_json.get("EEGCoordinateSystemDescription", "")
    ).strip():
        raise ValueError("BIDS EEGCoordinateSystem=Other requires a description.")

    electrode_fields, electrode_rows = _read_tsv(paths["electrodes_tsv"], "electrodes.tsv")
    required_electrode_fields = {"name", "x", "y", "z"}
    if not required_electrode_fields.issubset(electrode_fields):
        raise ValueError(f"BIDS electrodes.tsv lacks columns {sorted(required_electrode_fields - set(electrode_fields))}.")
    if tuple(row["name"].strip() for row in electrode_rows) != channel_names:
        raise ValueError("BIDS electrodes.tsv must align exactly with channels.tsv names.")
    positions: list[tuple[float, float, float]] = []
    for row in electrode_rows:
        position = tuple(_finite_number(row[axis], f"electrodes.{axis}") for axis in ("x", "y", "z"))
        if math.sqrt(sum(value * value for value in position)) <= 0.0 or max(abs(value) for value in position) > 0.5:
            raise ValueError("BIDS electrode positions must be non-zero head coordinates in metres.")
        positions.append(position)

    event_fields, event_rows = _read_tsv(paths["events_tsv"], "events.tsv")
    missing_event_fields = set(contract.required_event_columns) - set(event_fields)
    if missing_event_fields:
        raise ValueError(f"BIDS events.tsv lacks columns {sorted(missing_event_fields)}.")
    stimuli: list[BidsStimulusEvent] = []
    rests: list[BidsRestInterval] = []
    previous_onset = -math.inf
    trial_ids: set[str] = set()
    schedule_ids: set[tuple[int, int]] = set()
    previous_schedule: tuple[int, int] | None = None
    rest_ids: set[str] = set()
    for row_number, row in enumerate(event_rows, start=2):
        onset = _finite_number(row["onset"], f"events.tsv row {row_number} onset", minimum=0.0)
        duration_value = _finite_number(row["duration"], f"events.tsv row {row_number} duration", minimum=0.0)
        sample = _integer_text(row["sample"], f"events.tsv row {row_number} sample")
        if onset < previous_onset:
            raise ValueError("BIDS events.tsv rows must be chronological.")
        previous_onset = onset
        if onset >= duration:
            raise ValueError("BIDS event onset lies outside RecordingDuration.")
        if onset + duration_value > duration + 1e-9:
            raise ValueError("BIDS event interval extends beyond RecordingDuration.")
        if abs(sample / sample_rate - onset) > (0.5 / sample_rate + 1e-9):
            raise ValueError("BIDS event sample and onset columns disagree by more than half a sample.")
        trial_type = row["trial_type"].strip()
        block_id = _integer_text(row["block_id"], f"events.tsv row {row_number} block_id", minimum=1)
        if trial_type == contract.stimulus_trial_type:
            trial_id = row["trial_id"].strip()
            trial_index = _integer_text(row["trial_index"], f"events.tsv row {row_number} trial_index", minimum=1)
            candidate = row[contract.candidate_column].strip()
            if not trial_id or trial_id == "n/a" or candidate not in contract.candidate_vocabulary:
                raise ValueError("BIDS stimulus row has an invalid trial or candidate identity.")
            if trial_id in trial_ids or (block_id, trial_index) in schedule_ids:
                raise ValueError("BIDS stimulus trial identities must be unique.")
            schedule = (block_id, trial_index)
            if previous_schedule is not None and schedule <= previous_schedule:
                raise ValueError("BIDS stimulus block/trial identities must follow schedule order.")
            trial_ids.add(trial_id)
            schedule_ids.add(schedule)
            previous_schedule = schedule
            stimuli.append(BidsStimulusEvent(onset, duration_value, sample, candidate, trial_id, block_id, trial_index, row_number))
        elif trial_type == contract.rest_trial_type:
            rest_id = row["rest_segment_id"].strip()
            if not rest_id or rest_id == "n/a" or rest_id in rest_ids or duration_value <= 0.0:
                raise ValueError("BIDS rest rows require unique IDs and positive durations.")
            rest_ids.add(rest_id)
            rests.append(BidsRestInterval(onset, duration_value, sample, rest_id, block_id, row_number))
        else:
            raise ValueError(f"Unsupported BIDS trial_type {trial_type!r} at row {row_number}.")
    if not stimuli:
        raise ValueError("BIDS events.tsv contains no stimulus rows.")
    ordered_rests = tuple(sorted(rests, key=lambda item: item.onset_seconds))
    for left, right in zip(ordered_rests, ordered_rests[1:], strict=False):
        if left.end_seconds > right.onset_seconds:
            raise ValueError("BIDS rest intervals must not overlap.")
    for stimulus in stimuli:
        if any(rest.onset_seconds <= stimulus.onset_seconds < rest.end_seconds for rest in ordered_rests):
            raise ValueError("BIDS stimulus onset cannot occur inside a rest interval.")

    return ValidatedBidsEEGRecording(
        root=root,
        raw_path=paths["recording"],
        eeg_json_path=paths["eeg_json"],
        channels_path=paths["channels_tsv"],
        events_path=paths["events_tsv"],
        electrodes_path=paths["electrodes_tsv"],
        coordsystem_path=paths["coordsystem_json"],
        sample_rate_hz=sample_rate,
        duration_seconds=duration,
        channel_names=channel_names,
        channel_statuses=channel_statuses,
        channel_positions_m=tuple(positions),
        source_reference=reference.strip(),
        stimuli=tuple(stimuli),
        rest_intervals=ordered_rests,
    )
