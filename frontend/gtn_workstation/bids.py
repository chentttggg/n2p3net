"""BIDS-EEG raw export for one BrainSync recording session."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import BRAIN_SYNC_STORAGE_CONTRACT, ChannelMontage, ExperimentConfig
from .recording_clock import stored_time
from .storage import atomic_write_json

BIDS_VERSION = BRAIN_SYNC_STORAGE_CONTRACT.bids_version
BIDS_TASK_NAME = BRAIN_SYNC_STORAGE_CONTRACT.bids_task_name
BIDS_COORDINATE_SPACE = "BrainSyncHead"


@dataclass(frozen=True)
class BidsSessionLayout:
    root: Path
    subject_label: str
    session_label: str
    eeg_dir: Path
    stem: str

    def raw_path(self, extension: str) -> Path:
        suffix = extension if extension.startswith(".") else f".{extension}"
        return self.eeg_dir / f"{self.stem}_eeg{suffix.lower()}"

    def sidecar_path(self, suffix: str, extension: str = ".json") -> Path:
        return self.eeg_dir / f"{self.stem}_{suffix}{extension}"

    def coordinate_path(self, suffix: str, extension: str) -> Path:
        stem = f"sub-{self.subject_label}_ses-{self.session_label}_space-{BIDS_COORDINATE_SPACE}"
        return self.eeg_dir / f"{stem}_{suffix}{extension}"


@dataclass(frozen=True)
class BidsRawInfo:
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    duration_seconds: float
    units: str


def _bids_label(value: str, fallback: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "", str(value).strip())
    return label or fallback


def make_bids_session_layout(
    root: str | Path,
    subject_id: str,
    session_id: str,
) -> BidsSessionLayout:
    root = Path(root).expanduser().resolve()
    subject_label = _bids_label(subject_id, "P001")
    session_label = _bids_label(session_id, "session")
    stem = f"sub-{subject_label}_ses-{session_label}_task-{BIDS_TASK_NAME}_run-01"
    eeg_dir = root / f"sub-{subject_label}" / f"ses-{session_label}" / "eeg"
    return BidsSessionLayout(root, subject_label, session_label, eeg_dir, stem)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _bids_unit(value: Any) -> str:
    unit = _text(value).replace("μ", "u").replace("µ", "u")
    return unit or "n/a"


def _participant_sex(value: str) -> str:
    value = str(value).strip().upper()
    return value if value in {"M", "F"} else "n/a"


def _participant_age(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "n/a"
    try:
        age = float(value)
    except ValueError:
        return "n/a"
    return f"{age:g}" if math.isfinite(age) and age >= 0.0 else "n/a"


def write_bids_dataset_metadata(
    layout: BidsSessionLayout,
    config: ExperimentConfig,
) -> None:
    """Create the minimal dataset-level BIDS metadata before recording starts."""

    layout.root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(layout.root / "dataset_description.json", {
        "Name": "BrainSync GTN EEG recording",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "Authors": ["BrainSync"],
        "GeneratedBy": [{
            "Name": "BrainSync GTN Workstation",
            "Description": "Continuous EEG acquisition and BIDS-EEG export",
        }],
    })
    atomic_write_json(layout.root / "participants.json", {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age at recording time", "Units": "years"},
        "sex": {"Description": "Biological sex; n/a means not supplied"},
    })
    rows = [
        ["participant_id", "age", "sex"],
        [
            f"sub-{layout.subject_label}",
            _participant_age(config.age),
            _participant_sex(config.sex),
        ],
    ]
    _write_tsv(layout.root / "participants.tsv", rows)
    _atomic_write_text(
        layout.root / ".bidsignore",
        "session.json\n"
        "events/\n"
        "analysis/\n"
        "sourcedata/\n",
    )


def inspect_eeg_recording(path: str | Path) -> BidsRawInfo:
    """Read physical channel metadata from a finalized EDF/BDF file."""

    try:
        import pyedflib
    except ImportError as exc:
        raise RuntimeError("BIDS raw export requires pyedflib.") from exc

    reader = pyedflib.EdfReader(str(Path(path)))
    try:
        labels = tuple(_text(value) for value in reader.getSignalLabels())
        eeg_labels = tuple(
            label for label in labels if label.casefold() != "edf annotations"
        )
        rates = tuple(
            float(reader.getSampleFrequency(index))
            for index, label in enumerate(labels)
            if label.casefold() != "edf annotations"
        )
        units = tuple(
            _bids_unit(reader.getPhysicalDimension(index))
            for index, label in enumerate(labels)
            if label.casefold() != "edf annotations"
        )
        if not eeg_labels or not rates or not all(math.isfinite(rate) and rate > 0 for rate in rates):
            raise ValueError("EDF/BDF contains no valid EEG channel metadata.")
        if not all(math.isclose(rate, rates[0], rel_tol=0.0, abs_tol=1e-6) for rate in rates[1:]):
            raise ValueError("BIDS EEG export requires one common sampling frequency.")
        if len(set(units)) != 1:
            raise ValueError("BIDS EEG export requires one common channel unit.")
        duration = float(reader.getFileDuration())
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("EDF/BDF contains no valid recording duration.")
        return BidsRawInfo(eeg_labels, rates[0], duration, units[0])
    finally:
        reader.close()


def write_bids_eeg_sidecars(
    layout: BidsSessionLayout,
    raw_info: BidsRawInfo,
    montage: ChannelMontage,
    config: ExperimentConfig,
    *,
    sdk_version: str,
    impedance_values: list[float],
    acquisition_pauses: list[dict[str, Any]] | None = None,
) -> None:
    """Write EEG/EOG JSON, channels, electrodes, and coordinate sidecars."""

    montage.validate()
    if tuple(label.casefold() for label in raw_info.channel_names) != tuple(
        label.casefold() for label in montage.eeg
    ):
        raise ValueError("BIDS raw channel order does not match the BrainSync montage.")
    eeg_json = {
        "TaskName": BIDS_TASK_NAME,
        "Manufacturer": "BrainSync",
        "ManufacturersModelName": "BrainSync EEG",
        "SoftwareVersions": sdk_version or "n/a",
        "SamplingFrequency": raw_info.sample_rate_hz,
        "PowerLineFrequency": "n/a",
        "SoftwareFilters": "n/a",
        "HardwareFilters": "n/a",
        "EEGReference": montage.ref_label,
        "EEGReferenceDetails": montage.reference_details,
        "EEGGround": montage.gnd_label,
        "EEGPlacementScheme": "10-20",
        "EEGChannelCount": len(montage.eeg_indices),
        "EOGChannelCount": len(montage.eog_indices),
        "EOGDerivation": montage.ocular_derivation,
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        "RecordingDuration": raw_info.duration_seconds,
        "RecordingType": "discontinuous" if acquisition_pauses else "continuous",
        "AcquisitionPauses": acquisition_pauses or [],
        "Instructions": config.instruction_text,
        "Description": (
            "Continuous BrainSync GTN recording with independent EOG on hardware channel 6; "
            "rest intervals are retained and any ocular correction belongs to declared model preprocessing."
        ),
    }
    atomic_write_json(layout.sidecar_path("eeg"), eeg_json)

    channels = [[
        "name", "type", "units", "sampling_frequency", "low_cutoff",
        "high_cutoff", "notch", "reference", "status", "status_description",
    ]]
    for index, label in enumerate(raw_info.channel_names):
        enabled = bool(montage.active_mask & (1 << index))
        channel_type = montage.channel_types[index].upper()
        channels.append([
            label, channel_type, raw_info.units, f"{raw_info.sample_rate_hz:g}",
            "n/a", "n/a", "n/a", montage.ocular_derivation.get(label, {}).get("reference", "n/a")
            if channel_type == "EOG" else montage.ref_label,
            "good" if enabled else "bad",
            (
                "Ocular sensor retained outside classifier EEG; derivation declared in EEG sidecar."
                if channel_type == "EOG" and enabled
                else "Acquisition channel enabled; artifact rejection is applied during preprocessing."
                if enabled
                else "Channel disabled in the acquisition montage."
            ),
        ])
    _write_tsv(layout.sidecar_path("channels", ".tsv"), channels)
    atomic_write_json(layout.sidecar_path("channels"), {
        "status": {"Description": "Acquisition status; post-hoc artifact QC is deferred."},
        "status_description": {"Description": "Reason for a non-good channel status."},
    })

    electrodes = [["name", "x", "y", "z", "type", "material", "impedance"]]
    for index, (label, position) in enumerate(zip(montage.eeg, montage.channel_positions_m, strict=True)):
        impedance = impedance_values[index] if index < len(impedance_values) else float("nan")
        electrodes.append([
            label,
            "n/a" if position is None else f"{float(position[0]):.9g}",
            "n/a" if position is None else f"{float(position[1]):.9g}",
            "n/a" if position is None else f"{float(position[2]):.9g}",
            montage.channel_types[index].upper(),
            "n/a",
            f"{impedance:g}" if math.isfinite(float(impedance)) else "n/a",
        ])
    _write_tsv(layout.coordinate_path("electrodes", ".tsv"), electrodes)
    atomic_write_json(layout.coordinate_path("coordsystem", ".json"), {
        "EEGCoordinateSystem": "Other",
        "EEGCoordinateUnits": "m",
        "EEGCoordinateSystemDescription": (
            "BrainSync montage coordinates in the MNE head frame. These are standard "
            "10-20 template positions and are not digitized per recording."
        ),
    })


def write_bids_events(
    layout: BidsSessionLayout,
    audit_events_path: str | Path,
    sample_rate_hz: float,
    recording_duration_seconds: float,
    *,
    pauses: list[dict[str, Any]] | None = None,
) -> Path:
    """Convert the append-only event ledger to a BIDS events.tsv table."""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("BIDS events require a positive finite sample rate.")
    if not math.isfinite(recording_duration_seconds) or recording_duration_seconds <= 0.0:
        raise ValueError("BIDS events require a positive finite recording duration.")

    markers: list[dict[str, Any]] = []
    with Path(audit_events_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("event") == "recording_marker":
                payload = record.get("payload")
                if isinstance(payload, dict):
                    markers.append(payload)

    onsets: dict[str, dict[str, Any]] = {}
    offsets: dict[str, dict[str, Any]] = {}
    invalidated: dict[str, dict[str, Any]] = {}
    rest_starts: dict[int, dict[str, Any]] = {}
    rest_ends: dict[int, dict[str, Any]] = {}
    previous_time = -math.inf
    for marker in markers:
        marker = dict(marker)
        if "source_sample" in marker:
            sample = marker["source_sample"]
            if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
                raise ValueError("Bound source_sample must be a non-negative integer.")
            marker["eeg_time_seconds"] = sample / sample_rate_hz
        elif pauses:
            marker["eeg_time_seconds"] = stored_time(
                float(marker.get("wall_time_seconds", marker["eeg_time_seconds"])),
                pauses, sample_rate_hz,
            )
        if marker.get("eeg_time_base") != BRAIN_SYNC_STORAGE_CONTRACT.raw_time_base:
            raise ValueError("Recording markers must use the continuous BIDS raw time base.")
        try:
            marker_time = float(marker["eeg_time_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Recording marker has no finite eeg_time_seconds.") from exc
        if not math.isfinite(marker_time) or marker_time < 0.0:
            raise ValueError("Recording marker has no finite non-negative EEG time.")
        if marker_time < previous_time:
            raise ValueError("Recording markers must be chronological in EEG time.")
        if marker_time > recording_duration_seconds or (marker_time == recording_duration_seconds and (not pauses or marker.get("kind") == "onset")):
            raise ValueError("Recording marker lies outside the continuous EEG recording.")
        previous_time = marker_time
        kind = marker.get("kind")
        if kind in {"onset", "offset"}:
            trial_id = str(marker.get("trial_id", "")).strip()
            if not trial_id:
                raise ValueError("Stimulus marker requires a trial_id.")
            target = onsets if kind == "onset" else offsets
            if trial_id in target:
                raise ValueError(f"Duplicate {kind} marker for trial {trial_id!r}.")
            target[trial_id] = marker
        elif kind == "invalidated":
            invalidated[str(marker["trial_id"])] = marker
        elif kind in {"rest_start", "rest_end"}:
            raw_block = marker.get("block_id")
            if isinstance(raw_block, bool) or not isinstance(raw_block, int) or raw_block < 1:
                raise ValueError("Rest marker requires a positive integer block_id.")
            target = rest_starts if kind == "rest_start" else rest_ends
            if raw_block in target:
                raise ValueError(f"Duplicate {kind} marker for block {raw_block}.")
            target[raw_block] = marker
        else:
            raise ValueError(f"Unsupported recording marker kind: {kind!r}.")
    if set(onsets) != set(offsets):
        raise ValueError("Every stimulus onset must have exactly one matching offset.")
    if set(rest_starts) != set(rest_ends):
        raise ValueError("Every rest start must have exactly one matching end.")
    rows: list[dict[str, str]] = []
    for trial_id, onset in onsets.items():
        offset = offsets[trial_id]
        interrupted = trial_id in invalidated or offset.get("interrupted") is True
        if trial_id in invalidated:
            offset = invalidated[trial_id]
        onset_seconds = float(onset["eeg_time_seconds"])
        duration_seconds = float(offset["eeg_time_seconds"]) - onset_seconds
        if duration_seconds <= 0.0 and interrupted:
            continue
        if duration_seconds <= 0.0:
            raise ValueError(f"Stimulus interval for trial {trial_id!r} is empty.")
        duration = f"{duration_seconds:.9g}"
        rows.append({
            "onset": f"{onset_seconds:.9g}",
            "duration": duration,
            "sample": str(round(onset_seconds * sample_rate_hz)),
            "trial_type": "interrupted_stimulus" if interrupted else BRAIN_SYNC_STORAGE_CONTRACT.stimulus_trial_type,
            "value": str(int(onset["digit"])),
            "event_kind": "interrupted_stimulus" if interrupted else "stimulus",
            "trial_id": trial_id,
            "block_id": str(int(onset["block_id"])),
            "trial_index": str(int(onset["trial_index"])),
            "digit": str(int(onset["digit"])),
            "rest_segment_id": "n/a",
        })
    for block_id, start in rest_starts.items():
        end = rest_ends[block_id]
        onset_seconds = float(start["eeg_time_seconds"])
        duration = max(0.0, float(end["eeg_time_seconds"]) - onset_seconds)
        if duration <= 0.0:
            raise ValueError(f"Rest interval for block {block_id} is empty.")
        rows.append({
            "onset": f"{onset_seconds:.9g}",
            "duration": f"{duration:.9g}",
            "sample": str(round(onset_seconds * sample_rate_hz)),
            "trial_type": BRAIN_SYNC_STORAGE_CONTRACT.rest_trial_type,
            "value": "rest",
            "event_kind": "rest",
            "trial_id": "n/a",
            "block_id": str(block_id),
            "trial_index": "n/a",
            "digit": "n/a",
            "rest_segment_id": f"block-{block_id}",
        })
    emitted_boundaries: set[int] = set()
    for pause in pauses or []:
        if pause["boundary_sample"] in emitted_boundaries:
            continue
        emitted_boundaries.add(pause["boundary_sample"])
        onset_seconds = float(pause["boundary_sample"]) / sample_rate_hz
        if onset_seconds >= recording_duration_seconds:
            continue
        rows.append({
            "onset": f"{onset_seconds:.9g}", "duration": "0",
            "sample": str(pause["boundary_sample"]), "trial_type": "acquisition_boundary",
            "value": "pause", "event_kind": "acquisition_boundary", "trial_id": "n/a",
            "block_id": "1", "trial_index": "n/a", "digit": "n/a",
            "rest_segment_id": f"pause-{pause['pause_id']}",
        })
    rows.sort(key=lambda row: (float(row["onset"]), row["event_kind"]))
    fields = [
        "onset", "duration", "sample", "trial_type", "value", "event_kind",
        "trial_id", "block_id", "trial_index", "digit", "rest_segment_id",
    ]
    _write_tsv(
        layout.sidecar_path("events", ".tsv"),
        [fields, *[[row[field] for field in fields] for row in rows]],
    )
    atomic_write_json(layout.sidecar_path("events"), {
        "onset": {
            "Description": "Event onset relative to the first sample of the continuous raw recording.",
            "Units": "s",
        },
        "duration": {"Description": "Event duration from onset.", "Units": "s"},
        "sample": {"Description": "Event onset in raw recording samples; first sample is 0."},
        "trial_type": {
            "Description": "Event category used by preprocessing.",
            "Levels": {
                BRAIN_SYNC_STORAGE_CONTRACT.stimulus_trial_type: "Displayed digit stimulus.",
                BRAIN_SYNC_STORAGE_CONTRACT.rest_trial_type: "Retained block-rest interval.",
                "interrupted_stimulus": "Interrupted attempt, excluded from valid trials.",
                "acquisition_boundary": "Pause omitted from samples; reset filters at this boundary.",
            },
        },
        "value": {"Description": "Stimulus digit or rest marker."},
        "event_kind": {"Description": "Stable event category for preprocessing."},
        "trial_id": {"Description": "BrainSync trial identifier; n/a for rest."},
        "block_id": {"Description": "BrainSync block identifier."},
        "trial_index": {"Description": "Trial index within block; n/a for rest."},
        "digit": {"Description": "Displayed digit; n/a for rest."},
        "rest_segment_id": {"Description": "Paired rest interval identifier; n/a for stimulus."},
    })
    return layout.sidecar_path("events", ".tsv")


def _write_tsv(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
