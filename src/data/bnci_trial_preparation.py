"""Independent original-MAT episode preparation for the BNCI trial programme.

The first five complete selections authorize calibration. Every later original
selection remains a requested R10 decision, including unavailable EEG evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.bnci2014_008_candidate import (
    BNCI2014_008_CANDIDATE_TASK_CONTRACT,
    BNCI2014_008_CHANNELS,
    BNCI2014_008_DATASET_ID,
    BNCI2014_008_REPETITIONS_PER_SELECTION,
    BNCI2014_008_SELECTION_COUNT,
    BNCI2014_008_SOURCE_GROUND,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    BNCI2014008CandidateRecord,
    bnci2014_008_mat_source_contract,
)
from data.candidate_task import MembershipDecision, validate_candidate_membership_metadata
from data.channel import build_channel_identity
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT, assert_p300_input_contract
from data.epochs import EpochDataset, PreprocessingSpec, load_epoch_dataset
from data.events import ScheduledEventTimeline
from data.identity import DatasetIdentityTable
from data.preprocess import RawSampleScope, execute_preprocessing, source_epoch_support

BNCI_TRIAL_CHANNELS = ("FZ", "P3", "PZ", "P4", "PO8", "OZ")
BNCI_TRIAL_REFERENCE = "measured_channel:CZ"
BNCI_TRIAL_CALIBRATION_EPISODES = 5
BNCI_TRIAL_PREPARATION_SCHEMA = "n2p3_bnci_trial_preparation/1"
BNCI_TRIAL_INPUT_PROFILE = replace(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT, baseline_mode="none")
_FUTURE_TARGET_FIELDS = frozenset({"target_row", "target_col", "is_target", "raw_is_target",
                                 "raw_target_label", "raw_stimulus_code"})


@dataclass(frozen=True)
class BNCITrialPreparedParticipant:
    calibration: EpochDataset
    future: EpochDataset
    future_decisions: tuple[MembershipDecision, ...]
    ledger: dict


def bnci_trial_episode_scopes(record: BNCI2014008CandidateRecord,
                             preprocessing: PreprocessingSpec) -> tuple[RawSampleScope, ...]:
    """Resolve exact native-sample support before any filtering or resampling."""
    events = np.column_stack((record.flash_sample, np.zeros(len(record.flash_sample), dtype=np.int64),
                              record.raw_stimulus_code))
    starts, stops = source_epoch_support(events, BNCI2014_008_SOURCE_SAMPLE_RATE_HZ, preprocessing)
    selections = tuple(dict.fromkeys(record.selection_id.tolist()))
    if len(selections) != BNCI2014_008_SELECTION_COUNT:
        raise ValueError("BNCI trial preparation requires all 35 original MAT selections")
    indices = [np.flatnonzero(record.selection_id == selection) for selection in selections]
    left = [max(0, int(starts[rows[0]])) for rows in indices]
    end = len(record.eeg_uv)
    return tuple(RawSampleScope(selection, start,
        min(end, int(stops[rows[-1]]), left[position + 1] if position + 1 < len(left) else end),
        tuple(rows.tolist())) for position, (selection, start, rows)
        in enumerate(zip(selections, left, indices, strict=True)))


def _project_native(record):
    native = tuple(name.upper() for name in BNCI2014_008_CHANNELS)
    matrix = np.zeros((len(BNCI_TRIAL_CHANNELS), len(native)), dtype=np.float64)
    for row, channel in enumerate(BNCI_TRIAL_CHANNELS):
        matrix[row, native.index(channel)] = 1.
        matrix[row, native.index("CZ")] -= 1.
    volts = record.eeg_uv.astype(np.float64, copy=False).T * 1e-6
    projected = matrix @ volts
    receipt = {"schema": "n2p3_bnci_trial_projection/1", "acquisition_reference": BNCI2014_008_SOURCE_REFERENCE,
        "source_reference": BNCI_TRIAL_REFERENCE, "reference_channel": "CZ", "reference_was_measured": True,
        "raw_channel_names": list(native), "output_channels": list(BNCI_TRIAL_CHANNELS),
        "matrix": matrix.tolist(), "observed": [True] * len(BNCI_TRIAL_CHANNELS),
        "arithmetic": "float64 volts; (Vc-Vright_ear)-(VCz-Vright_ear)=Vc-VCz before filtering"}
    return projected, receipt


def _build_role(record, raw, events, scopes, preprocessing, projection, *, labels):
    scheduled = np.concatenate([np.asarray(scope.event_indices) for scope in scopes])
    evidence = np.full(len(events), -1, dtype=np.int64)
    status = np.full(len(events), "boundary_dropped", dtype=object)
    details = np.full(len(events), "outside_role", dtype=object)
    availability = np.full(len(events), np.nan)
    blocks, ordered_blocks, receipts, timing = [], [], [], []
    offset = 0
    for scope in scopes:
        result = execute_preprocessing(raw, events, preprocessing, raw_scope=scope,
                                       channels=BNCI_TRIAL_CHANNELS, montage="standard_1005")
        rows = result.scheduled_event_indices
        mapped = result.event_evidence_indices
        available = mapped >= 0
        evidence[rows[available]] = offset + mapped[available]
        status[rows], details[rows] = result.event_statuses, result.event_status_details
        availability[rows] = result.evidence_available_times_s
        ordered_blocks.append(rows[available][np.argsort(mapped[available], kind="stable")])
        blocks.append(result.data)
        receipts.append(result.execution_receipt)
        timing.append(result.timing_source)
        offset += result.n_epochs
    ordered = np.concatenate(ordered_blocks)
    values = np.concatenate(blocks)
    starts, stops = source_epoch_support(events, BNCI2014_008_SOURCE_SAMPLE_RATE_HZ, preprocessing)
    metadata = pd.DataFrame({"subject": np.repeat(record.subject_id, offset),
        "source_dataset_id": np.repeat(BNCI2014_008_DATASET_ID, offset),
        "source_subject_id": np.repeat(record.subject_id, offset),
        "selection_id": record.selection_id[ordered], "selection_index": record.selection_index[ordered],
        "selection_repetitions": np.full(offset, BNCI2014_008_REPETITIONS_PER_SELECTION),
        "repetition_index": record.repetition_index[ordered], "candidate_id": record.candidate_id[ordered],
        "row_code": record.row_code[ordered], "col_code": record.col_code[ordered],
        "flash_sample_matlab_1based": record.flash_sample_matlab_1based[ordered],
        "acquisition_time_s": record.flash_sample[ordered] / BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        "source_epoch_start_sample": starts[ordered], "source_epoch_stop_sample": stops[ordered]})
    y = None
    if labels:
        y = record.raw_is_target[ordered].astype(np.int64)
        for field in ("target_row", "target_col", "raw_is_target", "raw_target_label"):
            metadata[field] = getattr(record, field)[ordered]
        validate_candidate_membership_metadata(metadata, BNCI2014_008_CANDIDATE_TASK_CONTRACT, labels=y)
    identity = build_channel_identity(BNCI_TRIAL_CHANNELS, channel_mask=np.ones(len(BNCI_TRIAL_CHANNELS), dtype=bool),
                                      montage="standard_1005", allow_missing_positions=False)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"{BNCI2014_008_DATASET_ID}:{record.subject_id}:{row}" for row in scheduled]),
        group_ids=record.selection_id[scheduled], subject_ids=np.repeat(record.subject_id, len(scheduled)),
        stimulus_ids=record.candidate_id[scheduled], onset_samples=record.flash_sample[scheduled],
        onset_times_s=record.flash_sample[scheduled] / BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        evidence_available_times_s=availability[scheduled], evidence_indices=evidence[scheduled],
        statuses=status[scheduled].astype(str), status_details=details[scheduled].astype(str),
        dataset_ids=np.repeat(BNCI2014_008_DATASET_ID, len(scheduled)), session_ids=np.repeat("0", len(scheduled)),
        run_ids=np.repeat("0", len(scheduled)), selection_ids=record.selection_id[scheduled],
        candidate_ids=record.candidate_id[scheduled].astype(str), target_candidate_ids=np.repeat("", len(scheduled)),
        repetition_indices=record.repetition_index[scheduled], complete=True, online_causal=False,
        timing_source="bnci_mat_original_selection;" + ";".join(dict.fromkeys(timing)))
    dataset = EpochDataset(name="BNCI2014-008-trial-calibration" if labels else "BNCI2014-008-trial-future",
        X=values, y=y, subject_ids=np.repeat(record.subject_id, offset), channel_names=identity.names,
        channel_positions_m=identity.coords, channel_mask=np.ones(len(BNCI_TRIAL_CHANNELS), dtype=bool),
        preprocessing=preprocessing, event_timeline=timeline, metadata=metadata,
        identity_table=DatasetIdentityTable.from_source_rows([record.subject_id], [BNCI2014_008_DATASET_ID]),
        provenance={"source": "bnci_horizon_008_2014_mat_level5", "source_relative_path": record.source_relative_path,
            "source_sha256": record.source_sha256, "source_file_sha256": record.source_sha256,
            "source_reference": BNCI_TRIAL_REFERENCE, "acquisition_reference": BNCI2014_008_SOURCE_REFERENCE,
            "source_ground": BNCI2014_008_SOURCE_GROUND, "source_sample_rate_hz": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
            "source_signal_unit": "uV", "signal_unit": "V", "model_input_sample_rate_hz": preprocessing.sfreq,
            "observed_spatial_projection": projection, "candidate_task_contract": BNCI2014_008_CANDIDATE_TASK_CONTRACT.record(),
            "mat_source_contract": bnci2014_008_mat_source_contract(), "raw_episode_scopes": [scope.record() for scope in scopes],
            "preprocessing_execution": {"schema": "n2p3_preprocessing_execution_bundle/1", "recordings": receipts,
                                        "unverified_inputs": 0}})
    dataset.validate(require_labels=labels)
    return dataset


def _decisions(future):
    timeline = future.event_timeline
    return tuple(MembershipDecision.row_column(timeline.stimulus_ids[rows], n_rows=6, n_columns=6,
        event_ids=tuple(timeline.event_ids[rows]), repetition_indices=timeline.repetition_indices[rows],
        repetitions=BNCI2014_008_REPETITIONS_PER_SELECTION, evidence_indices=timeline.evidence_indices[rows],
        timeline_complete=True) for selection in dict.fromkeys(timeline.selection_ids.tolist())
        for rows in [np.flatnonzero(timeline.selection_ids == selection)])


def build_bnci_trial_prepared_participant(record: BNCI2014008CandidateRecord, *,
                                        preprocessing: PreprocessingSpec) -> BNCITrialPreparedParticipant:
    """Prepare fixed C600 and all 30 original future R10 selections, fitting nothing."""
    if not isinstance(record, BNCI2014008CandidateRecord):
        raise TypeError("BNCI trial preparation requires its original verified MAT record")
    assert_p300_input_contract(preprocessing, BNCI_TRIAL_INPUT_PROFILE)
    scopes = bnci_trial_episode_scopes(record, preprocessing)
    projected, projection = _project_native(record)
    raw = mne.io.RawArray(projected, mne.create_info(list(BNCI_TRIAL_CHANNELS),
        BNCI2014_008_SOURCE_SAMPLE_RATE_HZ, ch_types="eeg"), verbose=False)
    events = np.column_stack((record.flash_sample, np.zeros(len(record.flash_sample), dtype=np.int64),
                              record.raw_stimulus_code))
    c_scopes, f_scopes = scopes[:BNCI_TRIAL_CALIBRATION_EPISODES], scopes[BNCI_TRIAL_CALIBRATION_EPISODES:]
    calibration = _build_role(record, raw, events, c_scopes, preprocessing, projection, labels=True)
    future = _build_role(record, raw, events, f_scopes, preprocessing, projection, labels=False)
    decisions = _decisions(future)
    truths, descriptors = {}, []
    for scope, decision in zip(f_scopes, decisions, strict=True):
        raw_rows = np.asarray(scope.event_indices)
        pairs = np.unique(np.column_stack((record.target_row[raw_rows], record.target_col[raw_rows])), axis=0)
        if pairs.shape != (1, 2):
            raise ValueError("original MAT target pair drifts inside a BNCI future selection")
        truths[scope.name] = int(pairs[0, 0] * 6 + pairs[0, 1])
        descriptors.append({"episode_id": scope.name, "scheduled_flashes": len(decision.event_ids),
            "retained_flashes": int(np.count_nonzero(decision.evidence_indices >= 0)),
            "complete_raw_repetitions": BNCI2014_008_REPETITIONS_PER_SELECTION, "primary_horizon": 10,
            "terminal_partial_flashes": 0, "terminal_censor": False, "membership_decision": decision.record()})
    ledger = {"schema": BNCI_TRIAL_PREPARATION_SCHEMA, "dataset_id": BNCI2014_008_DATASET_ID,
        "subject": record.subject_id, "source_sha256": record.source_sha256,
        "source_sample_rate_hz": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ, "model_input_sample_rate_hz": preprocessing.sfreq,
        "preprocessing": preprocessing.record(), "observed_spatial_projection": projection,
        "calibration_episode_count": BNCI_TRIAL_CALIBRATION_EPISODES,
        "calibration_episode_ids": [scope.name for scope in c_scopes], "actual_calibration_labels": calibration.n_epochs,
        "actual_calibration_class_counts": {str(label): int(np.count_nonzero(calibration.y == label)) for label in (0, 1)},
        "requested_calibration_flashes": calibration.event_timeline.n_events,
        "future_episode_ids": [scope.name for scope in f_scopes], "future_requested_flashes": future.event_timeline.n_events,
        "calibration_raw_scopes": [scope.record() for scope in c_scopes], "future_raw_scopes": [scope.record() for scope in f_scopes],
        "common_future_first_raw_sample": f_scopes[0].start_sample,
        "calibration_last_raw_sample_exclusive": c_scopes[-1].stop_sample,
        "future_decisions": descriptors, "parent_scoring_truth": truths,
        "parent_truth_authority": "original MAT selection target row/column pairs, before epoch evidence retention",
        "public_training_fits": 0}
    return BNCITrialPreparedParticipant(calibration, future, decisions, ledger)


def load_bnci_trial_prepared_participant(directory: str | Path) -> BNCITrialPreparedParticipant:
    """Load a sealed producer pair and recheck its physical decision descriptors."""
    directory = Path(directory)
    ledger = json.loads((directory / "preparation.json").read_bytes())
    if ledger["schema"] != BNCI_TRIAL_PREPARATION_SCHEMA:
        raise ValueError("unsupported BNCI trial preparation schema")
    for role in ("calibration", "future"):
        with (directory / f"{role}.npz").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != ledger[f"{role}_cache"]["sha256"]:
            raise ValueError(f"BNCI {role} cache differs from its sealed preparation")
    calibration = load_epoch_dataset(directory / "calibration.npz", require_labels=True)
    future = load_epoch_dataset(directory / "future.npz", require_labels=False)
    if (future.y is not None or _FUTURE_TARGET_FIELDS & set(future.metadata)
            or not set(future.event_timeline.stimulus_ids).issubset(range(12))
            or np.any(future.event_timeline.target_candidate_ids != "")):
        raise ValueError("BNCI future cache exposes target labels instead of physical flash sets")
    decisions = _decisions(future)
    if ([decision.record() for decision in decisions] != [row["membership_decision"] for row in ledger["future_decisions"]]
            or list(dict.fromkeys(future.event_timeline.selection_ids.tolist())) != ledger["future_episode_ids"]
            or set(ledger["parent_scoring_truth"]) != set(ledger["future_episode_ids"])
            or len(decisions) != BNCI2014_008_SELECTION_COUNT - BNCI_TRIAL_CALIBRATION_EPISODES
            or calibration.n_epochs != ledger["actual_calibration_labels"]):
        raise ValueError("BNCI prepared pair differs from its original physical decision ledger")
    return BNCITrialPreparedParticipant(calibration, future, decisions, ledger)
