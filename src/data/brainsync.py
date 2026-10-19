"""BrainSync GTN session ingestion into the universal EpochDataset contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from data.continuous_qc import (
    FittedContinuousQC,
    signal_dependency_matrix,
    validate_recovery_contract,
)
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT, BRAIN_SYNC_LAYOUT_SCHEMA
from data.dataset import build_subject, read_raw
from data.domain import adapt_observed_montage
from data.epochs import (
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    materialize_dataset_identity,
    preprocessing_spec_from_contract,
)
from data.identity import DatasetIdentityTable
from data.ocular import FittedEOGRegression
from data.preprocess import continuous_filter_segments
from data.qc_features import compute_epoch_qc_features
from research.contracts import semantic_sha256

BRAIN_SYNC_PREPROCESSING = preprocessing_spec_from_contract(
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
)


class InvalidSessionPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"


@dataclass(frozen=True)
class IndependentRecordingScope:
    """Declare complete calibration and held-out recordings without time ordering."""

    calibration_session_ids: tuple[str, ...]
    evaluation_session_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("calibration_session_ids", "evaluation_session_ids"):
            values = tuple(getattr(self, name))
            if (not values or any(not isinstance(value, str) or not value.strip() for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("independent recording scope requires unique named sessions")
            object.__setattr__(self, name, values)
        if set(self.calibration_session_ids) & set(self.evaluation_session_ids):
            raise ValueError("independent calibration and evaluation recordings must be disjoint")

    def validate_application(self, session_id, fit_sessions=()):
        if session_id not in self.calibration_session_ids + self.evaluation_session_ids:
            raise ValueError("recording is outside the declared independent scope")
        if not set(fit_sessions) <= set(self.calibration_session_ids):
            raise ValueError("raw preprocessing fit exceeds the declared independent calibration scope")

    def record(self):
        return {"mode": "independent_recording_holdout",
                "calibration_session_ids": list(self.calibration_session_ids),
                "evaluation_session_ids": list(self.evaluation_session_ids)}


@dataclass(frozen=True)
class BrainSyncSessionFailure:
    session_dir: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BrainSyncLoadResult:
    dataset: EpochDataset
    failures: tuple[BrainSyncSessionFailure, ...]


class BrainSyncNoUsableEvidenceError(ValueError):
    """A valid requested recording whose entire epoch support failed raw QC."""

    def __init__(self, validated, ledger):
        super().__init__("continuous QC left no usable evidence; retain the requested recording as abstention")
        self.session_id = validated.session_id
        self.participant_key = DatasetIdentityTable.from_source_rows(
            [validated.subject_id], ["BrainSync-GTN"]).record_for(validated.subject_id).authority_key("source_or_global")
        self.started_timestamp_s = validated.started_timestamp_s
        self.scheduled_trials = len(validated.bids.stimuli)
        self.ledger = ledger


def resolve_brainsync_reference(reference_channel, ocular_model, continuous_model, *, mode="auto"):
    """Resolve one explicit reference across the request and frozen raw operators."""
    if mode not in {"auto", "native"}:
        raise ValueError("unknown BrainSync reference mode")
    anchors = [canonical_channel_name(value) for value in (
        reference_channel,
        None if ocular_model is None else ocular_model.reference_channel,
        None if continuous_model is None else continuous_model.spec.reference_channel,
    ) if value is not None]
    if len(set(anchors)) > 1:
        raise ValueError("Requested, ocular and continuous-QC references must agree")
    if mode == "native" and anchors:
        raise ValueError("Native acquisition reference cannot use a measured-reference QC/EOG model or channel")
    reference = anchors[0] if anchors else None
    if ocular_model is not None and ocular_model.reference_channel != reference:
        raise ValueError("requested reference differs from the fitted EOG model reference")
    return reference


def apply_brainsync_reference(dataset, reference_channel, ocular_model, *, channels=None):
    """Select the final montage, applying a measured reference exactly once."""
    if reference_channel is None:
        return dataset
    anchor = canonical_channel_name(reference_channel)
    if ocular_model is not None:
        if ocular_model.reference_channel != anchor:
            raise ValueError("requested reference differs from the fitted EOG model reference")
        return dataset
    qc = dataset.provenance.get("continuous_artifact_processing", {}).get("model", {})
    qc_reference = qc.get("spec", {}).get("reference_channel")
    if qc_reference is not None:
        if anchor != qc_reference or dataset.provenance["source_reference"] != "measured_channel:" + anchor:
            raise ValueError("requested reference differs from the frozen continuous QC reference")
        if channels is not None and tuple(channels) != tuple(dataset.channel_names):
            raise ValueError("QC-owned final montage differs from the saved input request")
        return dataset
    selected = channels or tuple(name for name in dataset.channel_names if canonical_channel_name(name) != anchor)
    return adapt_observed_montage(dataset, selected, reference_channel=anchor)


def load_brainsync_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    recording_scope: IndependentRecordingScope | None = None,
) -> EpochDataset:
    """Load a BrainSync raw session and apply the standard n2p3 preprocessing."""

    preprocessing.validate()
    validated = validate_brainsync_bids_session(session_dir)
    return _build_validated_brainsync_session(
        validated,
        preprocessing=preprocessing,
        channels=channels,
        ocular_model=ocular_model,
        continuous_qc_model=continuous_qc_model,
        recording_scope=recording_scope,
    )


def _build_validated_brainsync_session(
    validated: ValidatedBrainSyncRecording,
    *,
    preprocessing: PreprocessingSpec,
    channels: Sequence[str] | None,
    ocular_model: FittedEOGRegression | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    recording_scope: IndependentRecordingScope | None = None,
) -> EpochDataset:
    if recording_scope is not None:
        recording_scope.validate_application(validated.session_id)
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
    raw_names = {canonical_channel_name(name): name for name in eeg_labels}
    selected_channels = tuple(raw_names[canonical_channel_name(name)] for name in selected_channels)
    if len(set(selected_channels)) != len(selected_channels):
        raise ValueError("BrainSync selected EEG channels must be unique")
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
    clock = validated.manifest["recording"].get("clock")
    acquired_samples = int(raw.n_times) if clock is None else clock["acquired_samples"]
    if acquired_samples < raw.n_times:
        raw.annotations.append(acquired_samples / source_rate,
            (raw.n_times - acquired_samples) / source_rate, "BAD_acq_skip_edf_padding")
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
    continuous_result = None
    signal_diagnostics = None
    local_qc = continuous_qc_model is not None and continuous_qc_model.spec.channel_local
    qc_anchor = None if not local_qc else continuous_qc_model.spec.reference_channel
    if ocular_model is not None and qc_anchor is not None and ocular_model.reference_channel != qc_anchor:
        raise ValueError("QC and ocular preprocessing declare different final references")
    global_resets = [start for start, _ in continuous_filter_segments(raw) if start]
    if continuous_qc_model is not None:
        _validate_raw_fit_scope(continuous_qc_model.fit_provenance, session_id, session_start_timestamp_s,
                                recording_scope=recording_scope)
        if continuous_qc_model.fit_provenance.get("source_reference") != source_reference:
            raise ValueError("continuous QC native reference differs from calibration")
        participant = DatasetIdentityTable.from_source_rows([subject_id], ["BrainSync-GTN"]).record_for(subject_id).authority_key("source_or_global")
        if participant != continuous_qc_model.fit_provenance.get("participant_key"):
            raise ValueError("continuous QC calibration belongs to a different participant_key")
        validate_recovery_contract(continuous_qc_model, preprocessing,
                                  source_reference=source_reference, ocular_model=ocular_model)
        continuous_result = continuous_qc_model.detect(raw)
        signal_diagnostics = continuous_qc_model.signal_diagnostics(raw)
        if not local_qc:
            raw = continuous_result.annotate(raw)
    if ocular_model is not None:
        participant = DatasetIdentityTable.from_source_rows(
            [subject_id], ["BrainSync-GTN"]).record_for(subject_id).authority_key("source_or_global")
        if participant != ocular_model.participant_key:
            raise ValueError("ocular model calibration belongs to a different participant_key.")
        fit_sessions = ocular_model.fit_provenance.get("calibration_session_ids", [])
        fit_end = ocular_model.fit_provenance.get("calibration_end_timestamp_s")
        if not fit_sessions or not isinstance(fit_end, (int, float)) or not np.isfinite(fit_end):
            raise ValueError(
                "BrainSync ocular model requires explicit calibration sessions and end time."
            )
        if recording_scope is not None:
            recording_scope.validate_application(session_id, fit_sessions)
        elif session_id not in fit_sessions and session_start_timestamp_s < fit_end:
            raise ValueError("ocular calibration includes future data relative to this session.")
        raw = ocular_model.apply(raw, source_reference=source_reference,
                                eog_derivation=bids.eog_derivation)
        source_reference = ocular_model.source_reference
        if ocular_model.reference_channel is not None:
            anchor = ocular_model.reference_channel
            if channels is None:
                selected_channels = tuple(
                    name for name in eeg_labels if canonical_channel_name(name) != anchor
                )
            elif anchor in set(map(canonical_channel_name, selected_channels)):
                raise ValueError("the ocular reference channel is not an independent model input.")
    elif qc_anchor is not None:
        matches = [name for name in eeg_labels if canonical_channel_name(name) == qc_anchor]
        if len(matches) != 1 or matches[0] in raw.info["bads"]:
            raise ValueError("QC-owned reference must be a usable measured EEG channel")
        raw = raw.copy().load_data()
        raw.set_eeg_reference([matches[0]], projection=False, verbose=False)
        raw.set_channel_types({matches[0]: "misc"}, on_unit_change="ignore")
        source_reference = "measured_channel:" + qc_anchor
        if channels is None:
            selected_channels = tuple(name for name in eeg_labels if canonical_channel_name(name) != qc_anchor)
        elif qc_anchor in set(map(canonical_channel_name, selected_channels)):
            raise ValueError("the QC reference channel is not an independent classifier input")
    if local_qc:
        mixing = signal_dependency_matrix(continuous_qc_model.channel_names, selected_channels,
                                           reference_channel=qc_anchor, ocular_model=ocular_model)
        continuous_result = continuous_qc_model.project_result(continuous_result, mixing, selected_channels,
                                                               boundaries=global_resets)
        raw = continuous_result.annotate(raw)
    source_samples = np.asarray([marker.sample for marker in markers], dtype=np.int64)
    mne_samples = source_samples + int(raw.first_samp)
    if len(np.unique(source_samples)) != len(source_samples):
        raise ValueError("BrainSync onset markers map to duplicate source EEG samples.")
    if np.any(source_samples < 0) or np.any(source_samples >= acquired_samples):
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
    if continuous_result is not None and not local_qc:
        rejected = ~continuous_result.valid_sample_mask
        for interval in excluded_intervals:
            start = int(np.floor(interval.onset_seconds * source_rate))
            stop = int(np.ceil((interval.onset_seconds + interval.duration_seconds) * source_rate))
            rejected[max(0, start):min(len(rejected), stop)] = True
        starts = source_samples + round(preprocessing.tmin_ms / 1000 * source_rate)
        stops = source_samples + round(preprocessing.tmax_ms / 1000 * source_rate) + 1
        if not any(0 <= start < stop <= raw.n_times and not rejected[start:stop].any()
                   for start, stop in zip(starts, stops, strict=True)):
            raise BrainSyncNoUsableEvidenceError(validated, {
                **continuous_result.record(), "signal_diagnostics": signal_diagnostics})
    channel_quality = None
    if local_qc:
        channel_quality = continuous_result.epoch_channel_mask(source_samples, channels=selected_channels,
            tmin_s=preprocessing.tmin_ms / 1000, tmax_s=preprocessing.tmax_ms / 1000)
        if preprocessing.filter_phase == "zero":
            recovery = continuous_qc_model.epoch_recovery.project(mixing, selected_channels)
            boundary_mask, boundary_record = recovery.assess_epochs(raw, source_samples,
                boundary_action=continuous_qc_model.spec.boundary_action)
            channel_quality &= boundary_mask
            signal_diagnostics["bidirectional_boundary_recovery"] = boundary_record
        active = {canonical_channel_name(name): status == "good"
                  for name, status in zip(labels, bids.channel_statuses, strict=True)}
        channel_quality &= np.array([active[canonical_channel_name(name)] for name in selected_channels])[None, :]
        minimum = int(np.ceil(len(selected_channels) * continuous_qc_model.spec.minimum_channel_fraction))
        if not (channel_quality.sum(axis=1) >= minimum).any():
            raise BrainSyncNoUsableEvidenceError(validated, {
                **continuous_result.record(), "signal_diagnostics": signal_diagnostics})
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
        preprocessing=preprocessing,
        channels=selected_channels,
        positions_m=positions,
        montage=None,
        coordinate_source="bids_electrodes_tsv",
    )
    continuous_trial_mask = None
    if channel_quality is not None:
        event_rows = np.flatnonzero(subject.event_timeline.evidence_indices >= 0)
        event_rows = event_rows[np.argsort(subject.event_timeline.evidence_indices[event_rows])]
        trial_masks = channel_quality[event_rows]
        retained = trial_masks.sum(axis=1) >= minimum
        if not retained.any():
            raise BrainSyncNoUsableEvidenceError(validated, continuous_result.record())
        timeline = subject.event_timeline
        evidence = np.array(timeline.evidence_indices, copy=True)
        statuses, details = np.array(timeline.statuses, dtype=object), np.array(timeline.status_details, dtype=object)
        available = np.array(timeline.evidence_available_times_s, copy=True)
        evidence[event_rows] = -1
        evidence[event_rows[retained]] = np.arange(retained.sum())
        statuses[event_rows[~retained]] = "artifact_rejected"
        details[event_rows[~retained]] = "continuous_quality_insufficient_observed_channels"
        available[event_rows[~retained]] = np.nan
        subject = replace(subject, data=subject.data[retained],
            labels=None if subject.labels is None else subject.labels[retained],
            event_timeline=replace(timeline, evidence_indices=evidence, statuses=statuses.astype(str),
                                   status_details=details.astype(str), evidence_available_times_s=available))
        continuous_trial_mask = trial_masks[retained]
    # Complete recordings are batch inputs. Future interval confirmation and
    # callback delivery cannot be represented as onset+tmax availability.
    if continuous_result is not None or validated.manifest["recording"].get("clock") is not None:
        available = np.array(subject.event_timeline.evidence_available_times_s, copy=True)
        finite = np.isfinite(available)
        available[finite] = np.maximum(available[finite], observed_duration)
        subject = replace(subject, event_timeline=replace(subject.event_timeline,
            evidence_available_times_s=available,
            timing_source=subject.event_timeline.timing_source + ";continuous_qc_complete_recording"))

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
    effective_mask = np.broadcast_to(output_active, values.shape[:2]).copy()
    if continuous_trial_mask is not None:
        effective_mask &= continuous_trial_mask
        np.copyto(values, 0., where=~effective_mask[:, :, None])
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
            **({"preprocessing_execution": subject.metadata["preprocessing_execution"]}
               if "preprocessing_execution" in subject.metadata else {}),
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
            "acquisition_reference_details": dict(bids.source_reference_details),
            "reference_correction": validated.manifest.get("reference_correction"),
            "signal_unit": preprocessing.signal_unit,
            "acquisition_layout": {
                "schema": BRAIN_SYNC_LAYOUT_SCHEMA,
                "channel_names": list(labels),
                "channel_types": list(bids.channel_types),
                "model_eeg_channels": list(subject.channel_names),
                "eog_derivation": dict(bids.eog_derivation),
            },
            "continuous_artifact_processing": {
                "status": "not_applied" if continuous_qc_model is None else "applied",
                **({} if continuous_qc_model is None else {
                    "model": continuous_qc_model.record(),
                    "model_digest": semantic_sha256(continuous_qc_model.record()),
                    "operator": "channel_local_final_signal_filter_reset_with_epoch_uncertainty" if local_qc else "clean_segment_filter_reset_with_state_uncertainty",
                    "availability": "complete_recording_batch",
                }),
            },
            "continuous_quality_ledger": None if continuous_result is None else {
                **continuous_result.record(), "signal_diagnostics": signal_diagnostics},
            "sample_clock": validated.manifest["recording"].get("clock", {"alignment_level": "legacy_host_only"}),
            "acquisition_runtime": validated.manifest.get("runtime", {}),
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
            effective_mask
            if not effective_mask.all()
            else None
        ),
        qc_features=compute_epoch_qc_features(values, channel_mask=output_active, trial_channel_mask=effective_mask)
        if local_qc else None,
    )
    dataset.validate(require_labels=labelled)
    if recording_scope is not None:
        dataset.provenance["raw_calibration_scope"] = recording_scope.record()
    if continuous_qc_model is not None:
        participant = materialize_dataset_identity(dataset).record_for(subject_id).authority_key("source_or_global")
        if participant != continuous_qc_model.fit_provenance.get("participant_key"):
            raise ValueError("continuous QC calibration belongs to a different participant_key")
    return dataset


def _validate_raw_fit_scope(provenance, session_id, started_timestamp_s, *, recording_scope=None):
    sessions = provenance.get("calibration_session_ids", [])
    end = provenance.get("calibration_end_timestamp_s")
    if not sessions or not isinstance(end, (int, float)) or not np.isfinite(end):
        raise ValueError("raw QC requires calibration session identities and end timestamp")
    if recording_scope is not None:
        recording_scope.validate_application(session_id, sessions)
    elif session_id not in sessions and started_timestamp_s < end:
        raise ValueError("raw QC calibration includes future data relative to this session")


def load_brainsync_inference_session(
    session_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    recording_scope: IndependentRecordingScope | None = None,
    recording_validator: Callable[[ValidatedBrainSyncRecording], None] | None = None,
) -> EpochDataset:
    """Read target-free evidence; apply caller admission before signal processing/QC."""
    preprocessing.validate()
    validated = validate_brainsync_bids_recording(session_dir)
    if recording_validator is not None:
        recording_validator(validated)
    return _build_validated_brainsync_session(
        validated,
        preprocessing=preprocessing,
        channels=channels,
        ocular_model=ocular_model,
        continuous_qc_model=continuous_qc_model,
        recording_scope=recording_scope,
    )


def load_brainsync_sessions(
    session_dirs: Sequence[str | Path],
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    recording_scope: IndependentRecordingScope | None = None,
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
            continuous_qc_model=continuous_qc_model,
            recording_scope=recording_scope,
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
            "acquisition_references": [
                {"session_id": session.session_id, "label": session.bids.source_reference,
                 "details": dict(session.bids.source_reference_details)}
                for session in validated_sessions
            ],
            "source_reference": next(iter(references)),
            "source_sample_rate_hz": source_rates[0] if len(source_rates) == 1 else source_rates,
            "target_sample_rate_hz": float(preprocessing.sfreq),
            "signal_unit": preprocessing.signal_unit,
            "acquisition_layout": datasets[0].provenance["acquisition_layout"],
            "ocular_artifact_processing": datasets[0].provenance["ocular_artifact_processing"],
            "continuous_artifact_processing": datasets[0].provenance["continuous_artifact_processing"],
            "continuous_quality_ledgers": [dataset.provenance["continuous_quality_ledger"] for dataset in datasets],
            "sample_clocks": [dataset.provenance["sample_clock"] for dataset in datasets],
            "acquisition_runtimes": [dataset.provenance["acquisition_runtime"] for dataset in datasets],
            "n_sessions": len(datasets),
            "session_schema": BRAIN_SYNC_SESSION_SCHEMA,
            "session_ids": [session.session_id for session in validated_sessions],
            "decision_unit": "session",
            "population_scope": population_scope.to_dict(),
            **({"raw_calibration_scope": datasets[0].provenance["raw_calibration_scope"]}
               if "raw_calibration_scope" in datasets[0].provenance else {}),
        },
    )


def load_brainsync_sessions_resilient(
    session_dirs: Sequence[str | Path],
    *,
    preprocessing: PreprocessingSpec = BRAIN_SYNC_PREPROCESSING,
    channels: Sequence[str] | None = None,
    ocular_model: FittedEOGRegression | None = None,
    continuous_qc_model: FittedContinuousQC | None = None,
    invalid_session_policy: InvalidSessionPolicy | str = InvalidSessionPolicy.ERROR,
    recording_scope: IndependentRecordingScope | None = None,
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
                roots, preprocessing=preprocessing, channels=channels, ocular_model=ocular_model,
                continuous_qc_model=continuous_qc_model,
                recording_scope=recording_scope,
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
                continuous_qc_model=continuous_qc_model,
                recording_scope=recording_scope,
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
