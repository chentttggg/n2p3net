"""BrainSync GTN session ingestion into the universal EpochDataset contract."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.brainsync_contract import (
    BRAIN_SYNC_ANALYSIS_TIME_BASE,
    BRAIN_SYNC_CHANNEL_COUNT,
    BRAIN_SYNC_MONTAGE_SCHEMA,
    BRAIN_SYNC_SESSION_SCHEMA,
    PopulationScopePolicy,
    ValidatedBrainSyncSession,
    derive_population_scope,
    session_population_rows,
    validate_analysis_ready_brainsync_session,
)
from data.channel import canonical_channel_name
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.dataset import build_subject, read_raw
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    preprocessing_spec_from_contract,
)

BRAIN_SYNC_PREPROCESSING = preprocessing_spec_from_contract(
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
)


def _positions_from_montage(
    montage: dict[str, Any], labels: Sequence[str]
) -> dict[str, Any]:
    raw_positions = montage.get("channel_positions_m")
    if not isinstance(raw_positions, (list, tuple)) or len(raw_positions) != len(labels):
        raise ValueError(
            "BrainSync v2 montage.channel_positions_m must align with montage.labels."
        )
    positions: dict[str, Any] = {}
    for label, raw_position in zip(labels, raw_positions, strict=True):
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
            raise ValueError("BrainSync channel positions must be three-dimensional.")
        if any(isinstance(value, bool) for value in raw_position):
            raise ValueError("BrainSync channel positions must contain finite numbers.")
        try:
            position = np.asarray(raw_position, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BrainSync channel positions must contain finite numbers."
            ) from exc
        if (
            not np.isfinite(position).all()
            or float(np.linalg.norm(position)) <= 0.0
            or float(np.max(np.abs(position))) > 0.5
        ):
            raise ValueError(
                "BrainSync channel positions must be non-zero finite head coordinates in metres."
            )
        positions[str(label)] = position.tolist()
    return positions


def load_brainsync_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
) -> EpochDataset:
    """Load a BrainSync raw session and apply the standard n2p3 preprocessing."""

    preprocessing.validate()
    validated = validate_analysis_ready_brainsync_session(session_dir)
    return _build_validated_brainsync_session(
        validated,
        preprocessing=preprocessing,
        channels=channels,
    )


def _build_validated_brainsync_session(
    validated: ValidatedBrainSyncSession,
    *,
    preprocessing: PreprocessingSpec,
    channels: Sequence[str] | None,
) -> EpochDataset:
    root = validated.root
    session = validated.manifest
    raw_path = validated.recording_path
    events_path = validated.events_path
    subject_id = validated.subject_id
    session_id = validated.session_id
    session_started_utc = validated.started_utc
    session_start_timestamp_s = validated.started_timestamp_s
    montage = session.get("montage")
    if not isinstance(montage, dict):
        raise ValueError("BrainSync session.montage must be an object.")
    if montage.get("schema") != BRAIN_SYNC_MONTAGE_SCHEMA:
        raise ValueError(
            f"BrainSync montage schema must be {BRAIN_SYNC_MONTAGE_SCHEMA!r}."
        )
    raw_reference = montage.get("ref_label")
    if not isinstance(raw_reference, str) or not raw_reference.strip():
        raise ValueError("BrainSync session.montage.ref_label must declare the EEG reference.")
    source_reference = raw_reference.strip()
    if (
        montage.get("coordinate_frame") != "head"
        or montage.get("units") != "m"
    ):
        raise ValueError("BrainSync montage coordinates must use the head frame and metres.")
    raw_labels = montage.get("labels")
    if not isinstance(raw_labels, (list, tuple)) or not raw_labels:
        raise ValueError("BrainSync v2 session must provide montage.labels.")
    labels = tuple(str(label).strip() for label in raw_labels)
    if len(labels) != BRAIN_SYNC_CHANNEL_COUNT:
        raise ValueError(
            f"BrainSync v2 montage must declare {BRAIN_SYNC_CHANNEL_COUNT} EEG channels."
        )
    if any(not label for label in labels) or len(set(label.casefold() for label in labels)) != len(labels):
        raise ValueError("BrainSync channel labels must be unique non-empty strings.")
    session_channels = session.get("channels")
    if not isinstance(session_channels, list) or tuple(session_channels) != labels:
        raise ValueError("BrainSync session.channels must exactly match montage.labels.")
    raw_ground = montage.get("gnd_label")
    ground_label = raw_ground.strip() if isinstance(raw_ground, str) else ""
    normalized_eeg = {label.casefold() for label in labels}
    if (
        not ground_label
        or ground_label.casefold() == source_reference.casefold()
        or ground_label.casefold() in normalized_eeg
        or source_reference.casefold() in normalized_eeg
    ):
        raise ValueError(
            "BrainSync montage REF/GND labels must be non-empty and distinct from EEG labels."
        )
    active_mask = montage.get("active_mask")
    if isinstance(active_mask, bool) or not isinstance(active_mask, int):
        raise ValueError("BrainSync montage.active_mask must be an integer.")
    if not 0 < active_mask <= (1 << len(labels)) - 1:
        raise ValueError("BrainSync montage.active_mask must enable declared channels only.")
    selected_channels = tuple(channels) if channels is not None else labels
    if not selected_channels:
        raise ValueError("At least one BrainSync EEG channel is required.")
    positions = _positions_from_montage(montage, labels)

    markers = validated.onset_markers
    times = np.asarray([marker.eeg_time_seconds for marker in markers], dtype=float)
    digits = np.asarray([marker.digit for marker in markers], dtype=np.int64)
    labels_array = (digits == validated.thought_digit).astype(np.int64)
    candidates = digits.astype(str)
    targets = np.repeat(str(validated.thought_digit), len(markers))
    counts: dict[int, int] = {}
    repetitions_list: list[int] = []
    for digit in digits.tolist():
        repetitions_list.append(counts.get(digit, 0))
        counts[digit] = counts.get(digit, 0) + 1
    repetitions = np.asarray(repetitions_list, dtype=np.int64)
    # BrainSync v2 confirms one thought digit after the full session. Blocks
    # are presentation/rest structure, never separate label-bearing decisions.
    decision_ids = np.repeat(session_id, len(markers)).astype(str)

    raw = read_raw(raw_path, preload=False)
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(
        validated.source_sample_rate_hz,
        source_rate,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "BrainSync recording sample rate conflicts with the finalized timeline."
        )
    source_samples = np.rint(times * source_rate).astype(np.int64)
    if len(np.unique(source_samples)) != len(source_samples):
        raise ValueError("BrainSync onset markers map to duplicate source EEG samples.")
    if np.any(source_samples < 0) or np.any(source_samples >= int(raw.n_times)):
        raise ValueError("BrainSync onset marker lies outside the finalized recording.")
    observed_duration = float(raw.n_times) / source_rate
    if not np.isclose(
        validated.output_duration_seconds,
        observed_duration,
        rtol=0.0,
        atol=max(1.0 / source_rate, 1e-3),
    ):
        raise ValueError(
            "BrainSync finalized timeline duration conflicts with the recording."
        )
    events = np.column_stack(
        (source_samples, np.zeros(len(source_samples), dtype=np.int64), digits)
    )
    subject = build_subject(
        raw,
        events,
        labels_array,
        age=validated.age_years,
        sex=validated.sex,
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
        selection_ids=decision_ids,
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
        montage=None,
        coordinate_source="brainsync_channel_montage",
    )


    evidence = np.asarray(subject.event_timeline.evidence_indices, dtype=np.int64)
    available_rows = np.flatnonzero(evidence >= 0)
    available_rows = available_rows[np.argsort(evidence[available_rows], kind="stable")]
    metadata = pd.DataFrame(
        {
            "subject": np.repeat(subject_id, len(available_rows)),
            "session": np.repeat(session_id, len(available_rows)),
            "run": np.repeat(session_id, len(available_rows)),
            "selection_id": decision_ids[available_rows],
            "candidate_id": candidates[available_rows],
            "target_candidate_id": targets[available_rows],
            "repetition_index": repetitions[available_rows],
            "trial_id": [markers[index].trial_id for index in available_rows],
            "block_id": [markers[index].block_id for index in available_rows],
            "trial_index": [markers[index].trial_index for index in available_rows],
            "scheduled_event_index": available_rows,
            "age_years": np.repeat(validated.age_years, len(available_rows)),
            "age_source": np.repeat("session.experiment.age", len(available_rows)),
            "session_started_utc": np.repeat(session_started_utc, len(available_rows)),
            "session_start_timestamp_s": np.repeat(
                session_start_timestamp_s, len(available_rows)
            ),
        }
    )
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
            "session_schema": BRAIN_SYNC_SESSION_SCHEMA,
            "session_dir": str(root),
            "recording_path": str(raw_path),
            "events_path": str(events_path),
            "event_time_field": "eeg_time_seconds",
            "event_time_base": BRAIN_SYNC_ANALYSIS_TIME_BASE,
            "source_sample_rate_hz": source_rate,
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "source_reference": source_reference,
            "signal_unit": preprocessing.signal_unit,
            "target_digits": [validated.thought_digit],
            "decision_unit": "session",
            "decision_id": session_id,
            "age_years": validated.age_years,
            "age_source": "session.experiment.age",
            "population_scope": derive_population_scope(
                (validated.subject_id,),
                (validated.age_years,),
                policy=PopulationScopePolicy.DESCRIPTIVE,
            ).to_dict(),
            "session_started_utc": session_started_utc,
            "session_start_timestamp_s": session_start_timestamp_s,
            "session_ended_utc": validated.ended_utc,
            "target_confirmed_utc": validated.target_confirmed_utc,
            "channel_positions_source": "session.montage.channel_positions_m",
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
    """Load multiple v2 sessions, each representing exactly one decision."""

    roots = tuple(Path(path).expanduser().resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("session_dirs must contain unique BrainSync session directories.")
    preprocessing.validate()
    validated_sessions = [
        validate_analysis_ready_brainsync_session(root) for root in roots
    ]
    session_ids = [session.session_id for session in validated_sessions]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("BrainSync session_id values must be globally unique in one cache.")
    recordings = [session.recording_path for session in validated_sessions]
    if len(set(recordings)) != len(recordings):
        raise ValueError("BrainSync sessions cannot reference the same recording artifact.")
    if len(validated_sessions) > 1:
        start_keys = [
            (session.subject_id, session.started_timestamp_s)
            for session in validated_sessions
        ]
        if len(set(start_keys)) != len(start_keys):
            raise ValueError(
                "A subject's BrainSync sessions require unique timezone-aware started_utc values."
            )
        ordered = sorted(
            zip(validated_sessions, roots, strict=True),
            key=lambda item: (item[0].started_timestamp_s, item[0].session_id),
        )
        validated_sessions = [item[0] for item in ordered]
        roots = tuple(item[1] for item in ordered)
    datasets = [
        _build_validated_brainsync_session(
            session,
            preprocessing=preprocessing,
            channels=channels,
        )
        for session in validated_sessions
    ]
    references = {str(dataset.provenance["source_reference"]) for dataset in datasets}
    if len(references) != 1:
        raise ValueError("BrainSync sessions must declare one common source reference.")
    source_rates = sorted(
        {float(dataset.provenance["source_sample_rate_hz"]) for dataset in datasets}
    )
    population_subjects, population_ages = session_population_rows(validated_sessions)
    population_scope = derive_population_scope(
        population_subjects,
        population_ages,
        policy=PopulationScopePolicy.DESCRIPTIVE,
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
            "session_schema": BRAIN_SYNC_SESSION_SCHEMA,
            "session_ids": [session.session_id for session in validated_sessions],
            "decision_unit": "session",
            "population_scope": population_scope.to_dict(),
        },
    )
