"""BrainSync GTN session ingestion into the universal EpochDataset contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.channel import canonical_channel_name
from data.contract import (
    DEFAULT_GTN_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
)
from data.dataset import build_subject, read_raw
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
)

BRAIN_SYNC_SESSION_SCHEMA_PREFIX = "brainsync-gtn-session/"
BRAIN_SYNC_MARKER_EVENT = "recording_marker"
BRAIN_SYNC_ONSET_KIND = "onset"
BRAIN_SYNC_OFFLINE_PREPROCESSING = PreprocessingSpec(
    name=DEFAULT_GTN_DATA_CONTRACT.name,
    sfreq=DEFAULT_GTN_DATA_CONTRACT.sample_rate_hz,
    l_freq=DEFAULT_GTN_DATA_CONTRACT.l_freq,
    h_freq=DEFAULT_GTN_DATA_CONTRACT.h_freq,
    tmin_ms=DEFAULT_GTN_DATA_CONTRACT.tmin_ms,
    tmax_ms=DEFAULT_GTN_DATA_CONTRACT.tmax_ms,
    n_times=DEFAULT_GTN_DATA_CONTRACT.n_times,
    baseline_mode=DEFAULT_GTN_DATA_CONTRACT.baseline_mode,
)
BRAIN_SYNC_CAUSAL_PREPROCESSING = PreprocessingSpec(
    name="brainsync_single_subject_causal_v1",
    sfreq=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.sample_rate_hz,
    l_freq=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.l_freq,
    h_freq=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.h_freq,
    tmin_ms=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.tmin_ms,
    tmax_ms=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.tmax_ms,
    n_times=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.n_times,
    baseline_mode=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.baseline_mode,
    signal_unit=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.signal_unit,
    filter_method=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.filter_method,
    filter_order=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.filter_order,
    filter_phase=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.filter_phase,
    causal_iir_initial_state=(
        SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.causal_iir_initial_state
    ),
    resample_domain=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.resample_domain,
    resample_method=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.resample_method,
    resample_npad=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.resample_npad,
    resample_window=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.resample_window,
    resample_pad=SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT.resample_pad,
)
# Chronological calibration/test analysis is the BrainSync default. The old
# offline profile remains explicit for non-causal sensitivity work.
BRAIN_SYNC_PREPROCESSING = BRAIN_SYNC_CAUSAL_PREPROCESSING


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"BrainSync session is missing {path.name}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"BrainSync session file {path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BrainSync session file {path} must contain a JSON object.")
    return payload


def _session_artifact(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BrainSync session is missing {name}.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"BrainSync {name} does not exist: {candidate}")
    return candidate


def _integer_digit(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"BrainSync {field_name} must be an integer digit.")
    try:
        digit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BrainSync {field_name} must be an integer digit.") from exc
    if not 1 <= digit <= 9:
        raise ValueError(f"BrainSync {field_name} must be in the 1-9 vocabulary.")
    return digit


def _session_target(
    session: dict[str, Any], marker_payloads: Sequence[dict[str, Any]]
) -> int | None:
    target_label = session.get("target_label")
    target = target_label.get("thought_digit") if isinstance(target_label, dict) else None
    experiment = session.get("experiment")
    if target is None and isinstance(experiment, dict):
        target = experiment.get("thought_digit")
    payload_targets = {
        _integer_digit(payload["target_digit"], "target_digit")
        for payload in marker_payloads
        if payload.get("target_digit") is not None
    }
    if target is not None:
        target = _integer_digit(target, "thought_digit")
        if payload_targets and payload_targets != {target}:
            raise ValueError(
                "BrainSync session target conflicts with per-decision marker targets."
            )
        return target
    if len(payload_targets) == 1:
        return next(iter(payload_targets))
    if not payload_targets:
        raise ValueError("BrainSync session has no confirmed thought digit for label derivation.")
    # Multiple targets are valid only when every marker declares its target and
    # an explicit selection/block boundary.
    return None


def _marker_selection(payload: dict[str, Any]) -> str:
    explicit = payload.get("selection_id")
    if explicit is not None and str(explicit).strip():
        return f"selection:{str(explicit).strip()}"
    block = payload.get("block_id")
    if block is not None and str(block).strip():
        return f"block:{str(block).strip()}"
    return "selection:0"


def _read_brainsync_markers(path: Path, *, target: int | None = None) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    str,
]:
    payloads: list[dict[str, Any]] = []
    time_fields: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid BrainSync event JSON at line {line_number}.") from exc
            if not isinstance(record, dict) or record.get("event") != BRAIN_SYNC_MARKER_EVENT:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("kind") != BRAIN_SYNC_ONSET_KIND:
                continue
            if payload.get("eeg_time_seconds") is not None:
                time_fields.add("eeg_time_seconds")
            elif payload.get("recording_relative_seconds") is not None:
                time_fields.add("recording_relative_seconds")
            else:
                raise ValueError(
                    f"BrainSync onset marker at line {line_number} has no EEG time field."
                )
            payloads.append(dict(payload))
    if not payloads:
        raise ValueError(f"BrainSync event file {path} contains no onset recording markers.")
    if len(time_fields) > 1:
        raise ValueError("BrainSync event file mixes current and legacy EEG time fields.")

    field = next(iter(time_fields))
    times: list[float] = []
    digits: list[int] = []
    labels: list[int] = []
    candidates: list[str] = []
    targets: list[str] = []
    repetitions: list[int] = []
    selections: list[str] = []
    counts: dict[tuple[str, int], int] = {}
    target_by_selection: dict[str, int] = {}
    for payload in payloads:
        try:
            time_seconds = float(payload[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BrainSync {field} values must be finite numbers.") from exc
        if not np.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError(f"BrainSync {field} values must be finite and non-negative.")
        digit = _integer_digit(payload.get("digit"), "digit")
        payload_target = (
            _integer_digit(payload["target_digit"], "target_digit")
            if payload.get("target_digit") is not None
            else None
        )
        if target is not None and payload_target is not None and payload_target != target:
            raise ValueError("BrainSync marker target conflicts with the session target.")
        actual_target = payload_target if payload_target is not None else target
        if actual_target is None:
            raise ValueError("BrainSync onset markers need a session target or target_digit.")
        selection = _marker_selection(payload)
        previous_target = target_by_selection.setdefault(selection, actual_target)
        if previous_target != actual_target:
            raise ValueError(
                f"BrainSync selection {selection!r} mixes multiple target digits."
            )
        times.append(time_seconds)
        digits.append(digit)
        labels.append(int(digit == actual_target))
        candidates.append(str(digit))
        targets.append(str(actual_target))
        selections.append(selection)
        count_key = (selection, digit)
        raw_repetition = payload.get("repetition_index", counts.get(count_key, 0))
        if isinstance(raw_repetition, bool):
            raise ValueError("BrainSync repetition_index must be an integer.")
        try:
            repetition = int(raw_repetition)
        except (TypeError, ValueError) as exc:
            raise ValueError("BrainSync repetition_index must be an integer.") from exc
        if repetition < 0:
            raise ValueError("BrainSync repetition_index must be non-negative.")
        repetitions.append(repetition)
        counts[count_key] = repetitions[-1] + 1
    if np.any(np.diff(np.asarray(times, dtype=float)) < 0.0):
        raise ValueError("BrainSync onset markers must be chronological in EEG time.")
    return (
        np.asarray(times, dtype=float),
        np.asarray(digits, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(candidates, dtype=str),
        np.asarray(targets, dtype=str),
        np.asarray(repetitions, dtype=np.int64),
        np.asarray(selections, dtype=str),
        payloads,
        field,
    )


def _positions_from_montage(montage: dict[str, Any], labels: Sequence[str]) -> dict[str, Any] | None:
    raw_positions = montage.get("channel_positions_m")
    if raw_positions is None:
        return None
    if isinstance(raw_positions, dict):
        return {str(name): value for name, value in raw_positions.items()}
    if not isinstance(raw_positions, (list, tuple)) or len(raw_positions) != len(labels):
        raise ValueError("BrainSync channel_positions_m must align with montage labels.")
    return {str(label): value for label, value in zip(labels, raw_positions, strict=True)}


def _age_value(session: dict[str, Any]) -> float | None:
    experiment = session.get("experiment")
    value = experiment.get("age") if isinstance(experiment, dict) else None
    if value is None or value == "":
        return None
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("BrainSync experiment.age must be numeric or empty.") from exc
    if not np.isfinite(age):
        raise ValueError("BrainSync experiment.age must be finite.")
    return age


def _session_start(session: dict[str, Any]) -> tuple[str | None, float | None]:
    recording = session.get("recording")
    value = session.get("started_utc")
    if value is None and isinstance(recording, dict):
        value = recording.get("started_utc")
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("BrainSync started_utc must be a non-empty ISO timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("BrainSync started_utc must be an ISO timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError("BrainSync started_utc must include a timezone.")
    return value, float(parsed.timestamp())


def load_brainsync_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
) -> EpochDataset:
    """Load a BrainSync raw session and apply the standard n2p3 preprocessing."""

    preprocessing.validate()
    root = Path(session_dir).expanduser().resolve()
    session = _load_json_object(root / "session.json")
    schema = session.get("schema")
    if not isinstance(schema, str) or not schema.startswith(BRAIN_SYNC_SESSION_SCHEMA_PREFIX):
        raise ValueError(f"Unsupported BrainSync session schema: {schema!r}.")
    recording = session.get("recording")
    if not isinstance(recording, dict):
        raise ValueError("BrainSync session.recording must be an object.")
    raw_path = _session_artifact(root, recording.get("path"), "recording.path")
    events_path = root / "events" / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"BrainSync event file does not exist: {events_path}")

    experiment = session.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("BrainSync session.experiment must be an object.")
    subject_id = str(experiment.get("subject_id", "")).strip()
    if not subject_id:
        raise ValueError("BrainSync experiment.subject_id must be non-empty.")
    session_id = str(session.get("session_id", root.name)).strip() or root.name
    session_started_utc, session_start_timestamp_s = _session_start(session)
    montage = session.get("montage")
    if not isinstance(montage, dict):
        montage = {}
    source_reference = str(montage.get("ref_label", "")).strip()
    if not source_reference:
        raise ValueError("BrainSync session.montage.ref_label must declare the EEG reference.")
    if (
        montage.get("coordinate_frame", "head") != "head"
        or montage.get("units", "m") != "m"
    ):
        raise ValueError("BrainSync montage coordinates must use the head frame and metres.")
    raw_labels = montage.get("labels") or session.get("channels")
    if not isinstance(raw_labels, (list, tuple)) or not raw_labels:
        raise ValueError("BrainSync session must provide montage.labels or channels.")
    labels = tuple(str(label).strip() for label in raw_labels)
    if any(not label for label in labels) or len(set(label.casefold() for label in labels)) != len(labels):
        raise ValueError("BrainSync channel labels must be unique non-empty strings.")
    selected_channels = tuple(channels) if channels is not None else labels
    if not selected_channels:
        raise ValueError("At least one BrainSync EEG channel is required.")
    positions = _positions_from_montage(montage, labels)

    raw = read_raw(raw_path, preload=False)
    marker_payloads = []
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                isinstance(record, dict)
                and record.get("event") == BRAIN_SYNC_MARKER_EVENT
                and isinstance(record.get("payload"), dict)
                and record["payload"].get("kind") == BRAIN_SYNC_ONSET_KIND
            ):
                marker_payloads.append(dict(record["payload"]))
    target = _session_target(session, marker_payloads)
    (
        times,
        digits,
        labels_array,
        candidates,
        targets,
        repetitions,
        marker_selections,
        payloads,
        time_field,
    ) = _read_brainsync_markers(events_path, target=target)
    source_samples = np.rint(times * float(raw.info["sfreq"])).astype(np.int64)
    events = np.column_stack(
        (source_samples, np.zeros(len(source_samples), dtype=np.int64), digits)
    )
    global_selections = np.asarray(
        [f"{session_id}:{value}" for value in marker_selections], dtype=str
    )
    subject = build_subject(
        raw,
        events,
        labels_array,
        age=_age_value(session),
        sex=experiment.get("sex"),
        subject_id=subject_id,
        metadata={
            "dataset_id": "BrainSync-GTN",
            "session": session_id,
            "run": session_id,
            "selection_id": session_id,
            "reference": montage.get("ref_label", ""),
        },
        candidate_ids=candidates,
        target_candidate_ids=targets,
        repetition_indices=repetitions,
        selection_ids=global_selections,
        sfreq=preprocessing.sfreq,
        l_freq=preprocessing.l_freq,
        h_freq=preprocessing.h_freq,
        tmin=preprocessing.tmin_ms / 1000.0,
        tmax=preprocessing.tmax_ms / 1000.0,
        n_times=preprocessing.n_times,
        reject_threshold=preprocessing.reject_threshold_v,
        baseline=None,
        baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale,
        filter_method=preprocessing.filter_method,
        filter_order=preprocessing.filter_order,
        filter_phase=preprocessing.filter_phase,
        causal_iir_initial_state=preprocessing.causal_iir_initial_state,
        resample_domain=preprocessing.resample_domain,
        resample_method=preprocessing.resample_method,
        resample_npad=preprocessing.resample_npad,
        resample_window=preprocessing.resample_window,
        resample_pad=preprocessing.resample_pad,
        channels=selected_channels,
        positions_m=positions,
        montage=None if positions is not None else "standard_1005",
        coordinate_source="brainsync_channel_montage" if positions is not None else None,
    )


    evidence = np.asarray(subject.event_timeline.evidence_indices, dtype=np.int64)
    available_rows = np.flatnonzero(evidence >= 0)
    available_rows = available_rows[np.argsort(evidence[available_rows], kind="stable")]
    metadata = pd.DataFrame(
        {
            "subject": np.repeat(subject_id, len(available_rows)),
            "session": np.repeat(session_id, len(available_rows)),
            "run": np.repeat(session_id, len(available_rows)),
            "selection_id": global_selections[available_rows],
            "candidate_id": candidates[available_rows],
            "target_candidate_id": targets[available_rows],
            "repetition_index": repetitions[available_rows],
            "trial_id": [str(payloads[index].get("trial_id", "")) for index in available_rows],
            "block_id": [payloads[index].get("block_id") for index in available_rows],
            "trial_index": [payloads[index].get("trial_index") for index in available_rows],
            "scheduled_event_index": available_rows,
            "session_started_utc": np.repeat(session_started_utc, len(available_rows)),
            "session_start_timestamp_s": np.repeat(
                session_start_timestamp_s, len(available_rows)
            ),
        }
    )
    active_mask = montage.get("active_mask", (1 << len(labels)) - 1)
    if isinstance(active_mask, bool) or not isinstance(active_mask, int):
        raise ValueError("BrainSync montage.active_mask must be an integer.")
    if not 0 < active_mask <= (1 << len(labels)) - 1:
        raise ValueError("BrainSync montage.active_mask must enable at least one declared channel.")
    active_by_label = {
        canonical_channel_name(label): bool(active_mask & (1 << index))
        for index, label in enumerate(labels)
    }
    output_active = np.asarray(
        [active_by_label.get(canonical_channel_name(label), False) for label in subject.channel_names],
        dtype=bool,
    )
    values = np.asarray(subject.data, dtype=np.float32).copy()
    if not output_active.all():
        values[:, ~output_active, :] = 0.0
    dataset = EpochDataset(
        name="BrainSync-GTN",
        X=values,
        y=subject.labels,
        subject_ids=np.repeat(subject_id, len(values)),
        channel_names=subject.channel_names,
        channel_positions_m=subject.channel_positions_m,
        channel_mask=output_active,
        preprocessing=preprocessing,
        event_timeline=subject.event_timeline,
        metadata=metadata,
        provenance={
            "source": "brainsync_session",
            "session_schema": schema,
            "session_dir": str(root),
            "recording_path": str(raw_path),
            "events_path": str(events_path),
            "event_time_field": time_field,
            "source_sample_rate_hz": float(raw.info["sfreq"]),
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "source_reference": source_reference,
            "signal_unit": preprocessing.signal_unit,
            "target_digits": sorted({int(value) for value in targets.tolist()}),
            "session_started_utc": session_started_utc,
            "session_start_timestamp_s": session_start_timestamp_s,
            "channel_positions_source": (
                "session.montage.channel_positions_m" if positions is not None else "standard_1005_fallback"
            ),
        },
        trial_channel_mask=(
            np.broadcast_to(output_active, values.shape[:2]).copy()
            if not output_active.all()
            else None
        ),
    )
    dataset.validate(require_labels=True)
    return dataset


def load_brainsync_sessions(
    session_dirs: Sequence[str | Path],
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
) -> EpochDataset:
    """Load and concatenate multiple target-changing BrainSync decisions."""

    roots = tuple(Path(path).expanduser().resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("session_dirs must contain unique BrainSync session directories.")
    datasets = [
        load_brainsync_session(root, preprocessing=preprocessing, channels=channels)
        for root in roots
    ]
    if len(datasets) > 1:
        starts = [dataset.provenance.get("session_start_timestamp_s") for dataset in datasets]
        if any(value is None for value in starts) or len(set(starts)) != len(starts):
            raise ValueError(
                "Multiple BrainSync sessions require unique timezone-aware started_utc values."
            )
        ordered = sorted(
            zip(datasets, roots, strict=True),
            key=lambda item: float(item[0].provenance["session_start_timestamp_s"]),
        )
        datasets = [item[0] for item in ordered]
        roots = tuple(item[1] for item in ordered)
    references = {str(dataset.provenance["source_reference"]) for dataset in datasets}
    if len(references) != 1:
        raise ValueError("BrainSync sessions must declare one common source reference.")
    source_rates = sorted(
        {float(dataset.provenance["source_sample_rate_hz"]) for dataset in datasets}
    )
    return concatenate_epoch_datasets(
        datasets,
        name="BrainSync-GTN-multisession",
        provenance={
            "source": "brainsync_sessions",
            "session_dirs": [str(root) for root in roots],
            "source_reference": next(iter(references)),
            "source_sample_rate_hz": source_rates[0] if len(source_rates) == 1 else source_rates,
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "signal_unit": preprocessing.signal_unit,
            "n_sessions": len(datasets),
        },
    )
