"""BrainSync GTN session ingestion into the universal EpochDataset contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.bids_eeg import epoch_rest_overlap_mask
from data.brainsync_contract import (
    BRAIN_SYNC_BIDS_CONTRACT,
    BRAIN_SYNC_SESSION_SCHEMA,
    PopulationScopePolicy,
    ValidatedBrainSyncRecording,
    ValidatedBrainSyncSession,
    derive_population_scope,
    session_population_rows,
    validate_brainsync_bids_recording,
    validate_brainsync_bids_session,
)
from data.channel import canonical_channel_name
from data.contract import BRAIN_SYNC_LAYOUT_SCHEMA, SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.dataset import build_subject, read_raw
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    materialize_dataset_identity,
    preprocessing_spec_from_contract,
)
from data.ocular import FittedEOGRegression
from research.contracts import semantic_sha256

BRAIN_SYNC_PREPROCESSING = preprocessing_spec_from_contract(
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
)


class InvalidSessionPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"


@dataclass(frozen=True)
class BrainSyncSessionFailure:
    session_dir: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BrainSyncLoadResult:
    dataset: EpochDataset
    failures: tuple[BrainSyncSessionFailure, ...]


def load_brainsync_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
) -> EpochDataset:
    """Load a BrainSync raw session and apply the standard n2p3 preprocessing."""

    preprocessing.validate()
    validated = validate_brainsync_bids_session(session_dir)
    return _build_validated_brainsync_session(
        validated,
        preprocessing=preprocessing,
        channels=channels,
        ocular_model=ocular_model,
    )


def _build_validated_brainsync_session(
    validated: ValidatedBrainSyncRecording,
    *,
    preprocessing: PreprocessingSpec,
    channels: Sequence[str] | None,
    ocular_model: FittedEOGRegression | None = None,
) -> EpochDataset:
    labelled = isinstance(validated, ValidatedBrainSyncSession)
    device = validated.manifest.get("device")
    transport = device.get("transport") if isinstance(device, dict) else None
    simulation_status = "declared_mock" if transport == "mock" else "not_declared"
    root = validated.root
    bids = validated.bids
    raw_path = bids.raw_path
    events_path = bids.events_path
    subject_id = validated.subject_id
    session_id = validated.session_id
    session_started_utc = validated.started_utc
    session_start_timestamp_s = validated.started_timestamp_s
    labels = bids.channel_names
    source_reference = bids.source_reference
    eeg_labels = tuple(
        name for name, kind in zip(labels, bids.channel_types, strict=True) if kind == "EEG"
    )
    selected_channels = tuple(channels) if channels is not None else eeg_labels
    if not selected_channels:
        raise ValueError("At least one BrainSync EEG channel is required.")
    if not set(map(canonical_channel_name, selected_channels)) <= set(
        map(canonical_channel_name, eeg_labels)
    ):
        raise ValueError(
            "BrainSync classifier inputs may contain declared EEG channels only, never EOG."
        )
    positions = {
        name: position
        for name, kind, position in zip(
            labels, bids.channel_types, bids.channel_positions_m, strict=True
        )
        if kind == "EEG"
    }

    markers = bids.stimuli
    times = np.asarray([marker.onset_seconds for marker in markers], dtype=float)
    digits = np.asarray([int(marker.candidate_id) for marker in markers], dtype=np.int64)
    labels_array = (digits == validated.thought_digit).astype(np.int64) if labelled else None
    candidates = digits.astype(str)
    targets = np.repeat(str(validated.thought_digit), len(markers)) if labelled else None
    counts: dict[int, int] = {}
    repetitions_list: list[int] = []
    for digit in digits.tolist():
        repetitions_list.append(counts.get(digit, 0))
        counts[digit] = counts.get(digit, 0) + 1
    repetitions = np.asarray(repetitions_list, dtype=np.int64)
    # BrainSync v4 confirms one thought digit after the full session. Blocks
    # are presentation/rest structure, never separate label-bearing decisions.
    decision_ids = np.repeat(session_id, len(markers)).astype(str)

    raw = read_raw(raw_path, preload=False)
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(
        bids.sample_rate_hz,
        source_rate,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("BrainSync BIDS sampling frequency conflicts with the EEG recording.")
    raw_names = tuple(canonical_channel_name(value) for value in raw.ch_names)
    expected_names = tuple(canonical_channel_name(value) for value in labels)
    if raw_names != expected_names:
        raise ValueError("BrainSync BIDS channels.tsv order conflicts with the EEG recording.")
    raw.set_channel_types(
        {name: kind.lower() for name, kind in zip(raw.ch_names, bids.channel_types, strict=True)}
    )
    raw.info["bads"] = [
        name
        for name, status in zip(raw.ch_names, bids.channel_statuses, strict=True)
        if status == "bad"
    ]
    if bids.acquisition_boundaries:
        raw.set_annotations(raw.annotations + mne.Annotations(
            onset=[*bids.acquisition_boundaries, *bids.acquisition_boundaries],
            duration=[0.0] * (2 * len(bids.acquisition_boundaries)),
            description=["BAD boundary"] * len(bids.acquisition_boundaries)
            + ["EDGE boundary"] * len(bids.acquisition_boundaries),
            orig_time=raw.annotations.orig_time,
        ))
    if ocular_model is not None:
        fit_sessions = ocular_model.fit_provenance.get("calibration_session_ids", [])
        fit_end = ocular_model.fit_provenance.get("calibration_end_timestamp_s")
        if not fit_sessions or not isinstance(fit_end, (int, float)) or not np.isfinite(fit_end):
            raise ValueError(
                "BrainSync ocular model requires explicit calibration sessions and end time."
            )
        if session_id not in fit_sessions and session_start_timestamp_s < fit_end:
            raise ValueError("ocular calibration includes future data relative to this session.")
        raw = ocular_model.apply(raw, source_reference=source_reference)
        source_reference = ocular_model.source_reference
        if ocular_model.reference_channel is not None:
            anchor = ocular_model.reference_channel
            if channels is None:
                selected_channels = tuple(
                    name for name in eeg_labels if canonical_channel_name(name) != anchor
                )
            elif anchor in set(map(canonical_channel_name, selected_channels)):
                raise ValueError("the ocular reference channel is not an independent model input.")
    source_samples = np.asarray([marker.sample for marker in markers], dtype=np.int64)
    mne_samples = source_samples + int(raw.first_samp)
    if len(np.unique(source_samples)) != len(source_samples):
        raise ValueError("BrainSync onset markers map to duplicate source EEG samples.")
    if np.any(source_samples < 0) or np.any(source_samples >= int(raw.n_times)):
        raise ValueError("BrainSync BIDS stimulus sample lies outside the recording.")
    observed_duration = float(raw.n_times) / source_rate
    if not np.isclose(
        bids.duration_seconds,
        observed_duration,
        rtol=0.0,
        atol=max(1.0 / source_rate, 1e-3),
    ):
        raise ValueError("BrainSync BIDS RecordingDuration conflicts with the EEG recording.")
    excluded_intervals = (*bids.rest_intervals, *bids.excluded_intervals)
    if excluded_intervals:
        rest_annotations = mne.Annotations(
            onset=[interval.onset_seconds for interval in excluded_intervals],
            duration=[interval.duration_seconds for interval in excluded_intervals],
            description=["BAD_brainsync_rest"] * len(excluded_intervals),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + rest_annotations)
    events = np.column_stack((mne_samples, np.zeros(len(source_samples), dtype=np.int64), digits))
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
            "reference": source_reference,
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
        coordinate_source="bids_electrodes_tsv",
    )

    rest_overlap = np.asarray(
        epoch_rest_overlap_mask(
            times,
            tmin_seconds=preprocessing.tmin_ms / 1000.0,
            tmax_seconds=preprocessing.tmax_ms / 1000.0,
            rest_intervals=excluded_intervals,
        ),
        dtype=bool,
    )
    if np.any(rest_overlap & (np.asarray(subject.event_timeline.evidence_indices) >= 0)):
        raise RuntimeError(
            "A rest-overlapping BrainSync epoch escaped BIDS preprocessing exclusion."
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
            "repetition_index": repetitions[available_rows],
            "trial_id": [markers[index].trial_id for index in available_rows],
            "block_id": [markers[index].block_id for index in available_rows],
            "trial_index": [markers[index].trial_index for index in available_rows],
            "scheduled_event_index": available_rows,
            "age_years": np.repeat(validated.age_years, len(available_rows)),
            "age_source": np.repeat("session.experiment.age", len(available_rows)),
            "session_started_utc": np.repeat(session_started_utc, len(available_rows)),
            "session_start_timestamp_s": np.repeat(session_start_timestamp_s, len(available_rows)),
            "session_end_timestamp_s": np.repeat(validated.ended_timestamp_s, len(available_rows)),
            "recording_available_timestamp_s": np.repeat(
                validated.ended_timestamp_s, len(available_rows)
            ),
            **(
                {
                    "target_candidate_id": targets[available_rows],
                    "target_confirmed_timestamp_s": np.repeat(
                        validated.target_confirmed_timestamp_s, len(available_rows)
                    ),
                    "calibration_available_timestamp_s": np.repeat(
                        validated.calibration_available_timestamp_s, len(available_rows)
                    ),
                }
                if labelled
                else {}
            ),
        }
    )
    active_by_label = {
        canonical_channel_name(label): status == "good"
        for label, status in zip(labels, bids.channel_statuses, strict=True)
    }
    output_active = np.asarray(
        [
            active_by_label.get(canonical_channel_name(label), False)
            for label in subject.channel_names
        ],
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
            "simulation_status": simulation_status,
            "simulation_status_source": "session.device.transport",
            "session_dir": str(root),
            "recording_path": str(raw_path),
            "events_path": str(events_path),
            "event_time_field": "BIDS events.tsv sample/onset",
            "event_time_base": validated.manifest["recording"]["timeline"]["time_base"],
            "source_sample_rate_hz": source_rate,
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "source_reference": source_reference,
            "acquisition_source_reference": bids.source_reference,
            "signal_unit": preprocessing.signal_unit,
            "acquisition_layout": {
                "schema": BRAIN_SYNC_LAYOUT_SCHEMA,
                "channel_names": list(labels),
                "channel_types": list(bids.channel_types),
                "model_eeg_channels": list(subject.channel_names),
            },
            "ocular_artifact_processing": {
                "status": "not_applied" if ocular_model is None else "applied",
                "eog_retained_in_raw": True,
                "eog_in_classifier": False,
                "eog_channels": [
                    name
                    for name, kind in zip(labels, bids.channel_types, strict=True)
                    if kind == "EOG"
                ],
                **(
                    {}
                    if ocular_model is None
                    else {
                        "model": ocular_model.record(),
                        "model_digest": semantic_sha256(ocular_model.record()),
                    }
                ),
            },
            "candidate_vocabulary": list(BRAIN_SYNC_BIDS_CONTRACT.candidate_vocabulary),
            "target_label_used": labelled,
            **({"target_digits": [validated.thought_digit]} if labelled else {}),
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
            "recording_available_timestamp_s": validated.ended_timestamp_s,
            **(
                {
                    "target_confirmed_utc": validated.target_confirmed_utc,
                    "calibration_available_timestamp_s": validated.calibration_available_timestamp_s,
                }
                if labelled
                else {}
            ),
            "channel_positions_source": "BIDS electrodes.tsv + coordsystem.json",
            "rest_policy": "continuous_filter_then_exclude_intersecting_epochs",
            "retained_rest_interval_count": len(bids.rest_intervals),
            "rest_overlapping_stimulus_count": int(np.count_nonzero(rest_overlap)),
        },
        trial_channel_mask=(
            np.broadcast_to(output_active, values.shape[:2]).copy()
            if not output_active.all()
            else None
        ),
    )
    dataset.validate(require_labels=labelled)
    if ocular_model is not None:
        participant = (
            materialize_dataset_identity(dataset)
            .record_for(subject_id)
            .authority_key("source_or_global")
        )
        if participant != ocular_model.participant_key:
            raise ValueError("ocular model calibration belongs to a different participant_key.")
    return dataset


def load_brainsync_inference_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
) -> EpochDataset:
    """Read completed raw as target-free evidence, including before label confirmation."""
    preprocessing.validate()
    return _build_validated_brainsync_session(
        validate_brainsync_bids_recording(session_dir),
        preprocessing=preprocessing,
        channels=channels,
        ocular_model=ocular_model,
    )


def load_brainsync_sessions(
    session_dirs: Sequence[str | Path],
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
) -> EpochDataset:
    """Load multiple v4 BIDS raw sessions, one decision per session."""

    roots = tuple(Path(path).expanduser().resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("session_dirs must contain unique BrainSync session directories.")
    preprocessing.validate()
    validated_sessions = [validate_brainsync_bids_session(root) for root in roots]
    validated_sessions, roots = _order_brainsync_sessions(validated_sessions, roots)
    datasets = [
        _build_validated_brainsync_session(
            session,
            preprocessing=preprocessing,
            channels=channels,
            ocular_model=ocular_model,
        )
        for session in validated_sessions
    ]
    return _merge_brainsync_sessions(
        validated_sessions,
        roots,
        datasets,
        preprocessing=preprocessing,
    )


def _order_brainsync_sessions(
    sessions: Sequence[ValidatedBrainSyncSession],
    roots: Sequence[Path],
) -> tuple[list[ValidatedBrainSyncSession], tuple[Path, ...]]:
    if not sessions or len(sessions) != len(roots):
        raise ValueError("BrainSync sessions and roots must be aligned and non-empty.")
    session_ids = [session.session_id for session in sessions]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("BrainSync session_id values must be globally unique in one cache.")
    recordings = [session.bids.raw_path for session in sessions]
    if len(set(recordings)) != len(recordings):
        raise ValueError("BrainSync sessions cannot reference the same recording artifact.")
    start_keys = [(session.subject_id, session.started_timestamp_s) for session in sessions]
    if len(set(start_keys)) != len(start_keys):
        raise ValueError(
            "A subject's BrainSync sessions require unique timezone-aware started_utc values."
        )
    ordered = sorted(
        zip(sessions, roots, strict=True),
        key=lambda item: (item[0].started_timestamp_s, item[0].session_id),
    )
    return [item[0] for item in ordered], tuple(item[1] for item in ordered)


def _merge_brainsync_sessions(
    validated_sessions: Sequence[ValidatedBrainSyncSession],
    roots: Sequence[Path],
    datasets: Sequence[EpochDataset],
    *,
    preprocessing: PreprocessingSpec,
) -> EpochDataset:
    simulation_statuses = {dataset.provenance["simulation_status"] for dataset in datasets}
    if len(simulation_statuses) != 1:
        raise ValueError(
            "BrainSync sessions cannot mix declared mock and undeclared simulation provenance."
        )
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
            "simulation_status": next(iter(simulation_statuses)),
            "simulation_status_source": "session.device.transport",
            "session_dirs": [str(root) for root in roots],
            "source_reference": next(iter(references)),
            "source_sample_rate_hz": source_rates[0] if len(source_rates) == 1 else source_rates,
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "signal_unit": preprocessing.signal_unit,
            "acquisition_layout": datasets[0].provenance["acquisition_layout"],
            "ocular_artifact_processing": datasets[0].provenance["ocular_artifact_processing"],
            "n_sessions": len(datasets),
            "session_schema": BRAIN_SYNC_SESSION_SCHEMA,
            "session_ids": [session.session_id for session in validated_sessions],
            "decision_unit": "session",
            "population_scope": population_scope.to_dict(),
        },
    )


def load_brainsync_sessions_resilient(
    session_dirs: Sequence[str | Path],
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
    invalid_session_policy: InvalidSessionPolicy | str = InvalidSessionPolicy.ERROR,
) -> BrainSyncLoadResult:
    """Load a batch with an explicit fail-fast or skip-and-report policy."""

    policy = InvalidSessionPolicy(invalid_session_policy)
    roots = tuple(Path(path).expanduser().resolve() for path in session_dirs)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("session_dirs must contain unique BrainSync session directories.")
    preprocessing.validate()
    if policy is InvalidSessionPolicy.ERROR:
        return BrainSyncLoadResult(
            load_brainsync_sessions(
                roots, preprocessing=preprocessing, channels=channels, ocular_model=ocular_model
            ),
            (),
        )
    accepted_roots: list[Path] = []
    accepted_sessions: list[ValidatedBrainSyncSession] = []
    accepted_datasets: list[EpochDataset] = []
    failures: list[BrainSyncSessionFailure] = []
    for root in roots:
        try:
            validated = validate_brainsync_bids_session(root)
            dataset = _build_validated_brainsync_session(
                validated,
                preprocessing=preprocessing,
                channels=channels,
                ocular_model=ocular_model,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(BrainSyncSessionFailure(str(root), type(exc).__name__, str(exc)))
        else:
            accepted_roots.append(root)
            accepted_sessions.append(validated)
            accepted_datasets.append(dataset)
    if not accepted_roots:
        raise ValueError("No BrainSync session passed the BIDS raw input contract.")
    dataset_by_root = dict(zip(accepted_roots, accepted_datasets, strict=True))
    accepted_sessions, ordered_roots = _order_brainsync_sessions(accepted_sessions, accepted_roots)
    accepted_datasets = [dataset_by_root[root] for root in ordered_roots]
    dataset = _merge_brainsync_sessions(
        accepted_sessions,
        ordered_roots,
        accepted_datasets,
        preprocessing=preprocessing,
    )
    dataset.provenance["ingress_policy"] = policy.value
    dataset.provenance["skipped_sessions"] = [
        {
            "session_dir": failure.session_dir,
            "error_type": failure.error_type,
            "message": failure.message,
        }
        for failure in failures
    ]
    dataset.validate(require_labels=True)
    return BrainSyncLoadResult(dataset, tuple(failures))
