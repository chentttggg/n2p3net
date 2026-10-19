"""BI2014a raw-CSV candidate recovery.

MOABB's generic BI2014a adapter collapses the 6x6 speller flash codes to
target/non-target. The raw CSV keeps two event columns: column 17 is the
flash group code and column 18 is the binary target/non-target label. Each
repetition contains exactly 12 flashes:

    5 x {20..25} non-target rows + 1 x {60..65} target row
    5 x {40..45} non-target cols + 1 x {80..85} target col

The target character is the intersection (target_row, target_col).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.bi2014a_schedule import BI2014A_FLASH_SCHEDULE, BIFlashLabelAudit
from data.candidate_task import (
    CandidateTaskContract,
    MembershipDecision,
    validate_candidate_membership_metadata,
)
from data.channel import build_channel_identity
from data.contract import (
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    assert_causal_p300_input_contract,
    assert_p300_input_contract,
)
from data.epochs import EpochDataset, PreprocessingSpec, load_epoch_dataset, select_scheduled_events
from data.events import ScheduledEventTimeline
from data.exclusions import select_allowed_subjects
from data.identity import DatasetIdentityTable
from data.preprocess import RawSampleScope, execute_preprocessing, source_epoch_support

BI2014A_SFREQ = 512.0
BI2014A_SOURCE_BIDIRECTIONAL_PROFILE = replace(
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    baseline_mode="none",
)


def assert_bi2014a_candidate_input_contract(preprocessing: PreprocessingSpec) -> None:
    """Validate the source-compatible BI profile, including its explicit no-baseline field."""
    if preprocessing.filter_phase == "forward":
        assert_causal_p300_input_contract(preprocessing)
    else:
        assert_p300_input_contract(preprocessing, BI2014A_SOURCE_BIDIRECTIONAL_PROFILE)
BI2014A_CHANNELS = (
    "Fp1",
    "Fp2",
    "F3",
    "AFz",
    "F4",
    "T7",
    "Cz",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
)
_FLASH_COLUMN = 17
_LABEL_COLUMN = 18
# The public BI2014a Event stream retains experiment/session and game-level
# control markers in addition to the 20..85 flash codes. Code 100 starts a new
# recorded game session; code 104 closes a level or its forced restart after
# the attempt budget. Either marker is a hard target-episode boundary even when
# the next level happens to use the same target row/column pair.
_SESSION_START_CODE = 100
_LEVEL_OR_RESTART_BOUNDARY_CODE = 104
_SELECTION_BOUNDARY_CODES = frozenset({_SESSION_START_CODE, _LEVEL_OR_RESTART_BOUNDARY_CODE})
BI2014A_CANDIDATE_TASK_CONTRACT = CandidateTaskContract(
    dataset_id="BI2014a",
    task_id="p300_row_column_speller_6x6",
    population={
        "label": "public_dataset_participants",
        "age_scope": "not_asserted_by_cache_producer",
    },
    evidence_scope={"stage": "development", "product_confirmation": False},
    membership_kind="row_column",
    grid_shape=(6, 6),
    candidate_ids=tuple(range(12)),
    row_candidate_ids=tuple(range(6)),
    column_candidate_ids=tuple(range(6, 12)),
    target_representation="row_column_intersection",
    raw_target_label_is_target={"1": False, "2": True},
)


@dataclass(frozen=True)
class BI2014ACandidateRecord:
    flash_sample: np.ndarray
    flash_code: np.ndarray
    target_label: np.ndarray
    is_target: np.ndarray
    row_code: np.ndarray
    col_code: np.ndarray
    target_row: np.ndarray
    target_col: np.ndarray
    selection_id: np.ndarray
    repetition_index: np.ndarray
    selection_boundary_reason: np.ndarray
    n_repetitions: int
    n_explicit_boundaries: int
    raw_label_audit: BIFlashLabelAudit
    scheduled_flash_sample: np.ndarray
    scheduled_flash_code: np.ndarray
    scheduled_target_label: np.ndarray
    scheduled_selection_id: np.ndarray
    scheduled_repetition_index: np.ndarray
    dropped_tail_flashes: int = 0


def _require_csv(subject_dir: Path) -> Path:
    subject_id = subject_dir.name.removeprefix("subject_")
    csv_path = subject_dir / f"subject_{subject_id}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} is required to recover BI2014a flash codes from MAT fallback."
        )
    return csv_path


def recover_bi2014a_candidates(
    subject_dir: str | Path,
) -> BI2014ACandidateRecord:
    """Parse one raw BI2014a CSV and validate its 6x6 flash schedule."""

    subject_dir = Path(subject_dir)
    csv_path = _require_csv(subject_dir)
    table = pd.read_csv(csv_path, header=None, usecols=[_FLASH_COLUMN, _LABEL_COLUMN])
    return _recover_candidate_table(subject_dir, table)


def _recover_candidate_table(subject_dir: Path, table: pd.DataFrame) -> BI2014ACandidateRecord:
    csv_path = _require_csv(subject_dir)
    if _LABEL_COLUMN not in table:
        raise ValueError(f"{csv_path} lacks the BI2014a event columns.")
    raw_flash_code = table[_FLASH_COLUMN].to_numpy(dtype=np.float64)
    raw_target_label = table[_LABEL_COLUMN].to_numpy(dtype=np.float64)
    return recover_bi2014a_schedule(subject_dir.name, raw_flash_code, raw_target_label,
                                   stage="raw_csv_before_preprocessing")


def recover_bi2014a_schedule(subject_name: str, raw_flash_code, raw_target_label, *, stage: str):
    """Recover the original schedule from either original table representation."""
    raw_flash_code = np.asarray(raw_flash_code, dtype=np.float64)
    raw_target_label = np.asarray(raw_target_label, dtype=np.float64)
    if raw_flash_code.ndim != 1 or raw_target_label.shape != raw_flash_code.shape:
        raise ValueError("BI original marker and label fields must share one native sample axis")
    possible_flash = np.isfinite(raw_flash_code) & (raw_flash_code >= 20) & (raw_flash_code <= 85)
    flash_samples = np.flatnonzero(possible_flash)
    if np.any(raw_flash_code[flash_samples] != np.floor(raw_flash_code[flash_samples])):
        raise ValueError(f"{subject_name}: non-integer BI flash code.")
    flash_code = np.zeros(len(raw_flash_code), dtype=np.int64)
    integer_event_codes = np.isfinite(raw_flash_code) & (raw_flash_code == np.floor(raw_flash_code))
    flash_code[integer_event_codes] = raw_flash_code[integer_event_codes].astype(np.int64)
    all_codes = flash_code[flash_samples]
    all_labels = raw_target_label[flash_samples]
    all_flash_samples = flash_samples.copy()
    raw_label_audit = BI2014A_FLASH_SCHEDULE.audit_raw_labels(
        all_codes,
        all_labels,
        flash_samples=flash_samples,
        stage=stage,
    )

    # Some public BI2014a files end with one or two trailing flashes that do
    # not form a complete 12-flash repetition. Keep only complete repetitions
    # and report the dropped tail in the audit.
    n_repetitions = len(flash_samples) // 12
    if n_repetitions < 1:
        raise ValueError(f"{subject_name}: fewer than one complete flash repetition.")
    complete_count = n_repetitions * 12
    dropped_tail = len(flash_samples) - complete_count
    flash_samples = flash_samples[:complete_count]
    codes = all_codes[:complete_count]
    labels = all_labels[:complete_count].astype(np.int64, copy=False)
    is_target = BI2014A_FLASH_SCHEDULE.target_mask(codes)
    expected_targets = 2 * n_repetitions
    expected_nontargets = 10 * n_repetitions
    if np.count_nonzero(is_target) != expected_targets:
        raise ValueError(
            f"{subject_name}: expected {expected_targets} target flashes, "
            f"got {np.count_nonzero(is_target)}."
        )
    if np.count_nonzero(~is_target) != expected_nontargets:
        raise ValueError(
            f"{subject_name}: expected {expected_nontargets} non-target flashes, "
            f"got {np.count_nonzero(~is_target)}."
        )

    row_code = np.full(len(flash_samples), -1, dtype=np.int64)
    col_code = np.full(len(flash_samples), -1, dtype=np.int64)
    target_row = np.full(n_repetitions, -1, dtype=np.int64)
    target_col = np.full(n_repetitions, -1, dtype=np.int64)
    selection_id = np.empty(len(flash_samples), dtype=object)
    repetition_index = np.empty(len(flash_samples), dtype=np.int64)
    selection_boundary_reason = np.empty(len(flash_samples), dtype=object)

    previous_pair: tuple[int, int] | None = None
    active_selection = 0
    active_repetition = -1
    explicit_boundary_count = 0

    for rep in range(n_repetitions):
        start = rep * 12
        stop = start + 12
        rep_codes = codes[start:stop]
        decoded_events = tuple(BI2014A_FLASH_SCHEDULE.decode(code) for code in rep_codes)
        row_events = tuple(event for event in decoded_events if event.axis == "row")
        col_events = tuple(event for event in decoded_events if event.axis == "column")
        target_row_events = tuple(event for event in row_events if event.is_target)
        target_col_events = tuple(event for event in col_events if event.is_target)
        valid_candidates = tuple(range(BI2014A_FLASH_SCHEDULE.grid_size))
        if (
            len(target_row_events) != 1
            or len(target_col_events) != 1
            or tuple(sorted(event.candidate_index for event in row_events)) != valid_candidates
            or tuple(sorted(event.candidate_index for event in col_events)) != valid_candidates
        ):
            raise ValueError(f"{subject_name}: repetition {rep} has invalid flash structure.")

        pair = (
            target_row_events[0].candidate_index,
            target_col_events[0].candidate_index,
        )
        span_start = int(flash_samples[start])
        span_stop = int(flash_samples[stop - 1]) + 1
        internal_markers = _SELECTION_BOUNDARY_CODES.intersection(
            flash_code[span_start:span_stop].tolist()
        )
        if internal_markers:
            raise ValueError(
                f"{subject_name}: repetition {rep} contains selection-boundary "
                f"markers {sorted(internal_markers)} inside its 12-flash block."
            )

        explicit_markers: tuple[int, ...] = ()
        if rep > 0:
            previous_flash = int(flash_samples[start - 1])
            current_flash = int(flash_samples[start])
            explicit_markers = tuple(
                sorted(
                    _SELECTION_BOUNDARY_CODES.intersection(
                        flash_code[previous_flash + 1 : current_flash].tolist()
                    )
                )
            )
        pair_changed = previous_pair is not None and previous_pair != pair
        starts_selection = previous_pair is None or pair_changed or bool(explicit_markers)
        if starts_selection:
            active_selection += 1
            active_repetition = 0
            explicit_boundary_count += int(bool(explicit_markers))
        else:
            active_repetition += 1
        reasons: list[str] = []
        if previous_pair is None:
            reasons.append("recording_start")
        if _SESSION_START_CODE in explicit_markers:
            reasons.append("raw_session_start_code_100")
        if _LEVEL_OR_RESTART_BOUNDARY_CODE in explicit_markers:
            reasons.append("raw_level_or_restart_code_104")
        if pair_changed:
            reasons.append("target_pair_change")
        boundary_reason = "+".join(reasons) if reasons else "same_selection_repetition"
        previous_pair = pair
        target_row[rep] = pair[0]
        target_col[rep] = pair[1]

        for local, global_idx in enumerate(range(start, stop)):
            decoded = decoded_events[local]
            if decoded.axis == "row":
                row_code[global_idx] = decoded.candidate_index
            else:
                col_code[global_idx] = decoded.candidate_index
            selection_id[global_idx] = f"{subject_name}:selection{active_selection}"
            repetition_index[global_idx] = active_repetition
            selection_boundary_reason[global_idx] = boundary_reason

    tail_selection = str(selection_id[-1])
    tail_repetition = active_repetition + 1
    if dropped_tail:
        markers = _SELECTION_BOUNDARY_CODES.intersection(
            flash_code[int(flash_samples[-1]) + 1:int(all_flash_samples[complete_count])].tolist()
        )
        partial_targets = [BI2014A_FLASH_SCHEDULE.decode(code) for code in all_codes[complete_count:]]
        changed = any(event.is_target and event.candidate_index != previous_pair[0 if event.axis == "row" else 1]
                      for event in partial_targets)
        if markers or changed:
            tail_selection = f"{subject_name}:selection{active_selection + 1}"
            tail_repetition = 0
    return BI2014ACandidateRecord(
        flash_sample=flash_samples,
        flash_code=codes,
        target_label=labels,
        is_target=is_target,
        row_code=row_code,
        col_code=col_code,
        target_row=np.repeat(target_row, 12),
        target_col=np.repeat(target_col, 12),
        selection_id=np.asarray(selection_id),
        repetition_index=repetition_index,
        selection_boundary_reason=np.asarray(selection_boundary_reason),
        n_repetitions=n_repetitions,
        n_explicit_boundaries=explicit_boundary_count,
        raw_label_audit=raw_label_audit,
        scheduled_flash_sample=all_flash_samples,
        scheduled_flash_code=all_codes,
        scheduled_target_label=all_labels.astype(np.int64),
        scheduled_selection_id=np.concatenate((selection_id, np.repeat(tail_selection, dropped_tail))),
        scheduled_repetition_index=np.concatenate((repetition_index, np.repeat(tail_repetition, dropped_tail))).astype(np.int64),
        dropped_tail_flashes=dropped_tail,
    )


def _episode_scopes(raw, events, recovered, preprocessing):
    starts, stops = source_epoch_support(events, BI2014A_SFREQ, preprocessing)
    episodes = tuple(dict.fromkeys(recovered.scheduled_selection_id.tolist()))
    indices = [np.flatnonzero(recovered.scheduled_selection_id == name) for name in episodes]
    left = [max(int(raw.first_samp), int(starts[rows[0]])) for rows in indices]
    end = int(raw.first_samp + raw.n_times)
    scopes = [RawSampleScope(name, start, min(end, int(stops[rows[-1]]),
              left[position + 1] if position + 1 < len(left) else end), tuple(rows.tolist()))
              for position, (name, rows, start) in enumerate(zip(episodes, indices, left, strict=True))]
    return scopes


def _observed_projection(eeg_v, output_channels, observed_raw_channels, reference_channel, source_reference):
    native = tuple(name.upper() for name in BI2014A_CHANNELS)
    output = tuple(name.upper() for name in output_channels)
    observed = tuple(name.upper() for name in observed_raw_channels)
    if (not output or len(set(output)) != len(output) or not observed
            or len(set(observed)) != len(observed) or set(observed) - set(native)
            or set(observed) - set(output)):
        raise ValueError("BI output slots and observed raw electrodes must be unique physical channels")
    reference = None if reference_channel is None else reference_channel.upper()
    if reference is not None and reference not in native:
        raise ValueError("the BI reference must be an actually measured raw channel")
    actual_reference = "right earlobe" if reference is None else f"measured_channel:{reference}"
    if source_reference not in {"right earlobe", actual_reference}:
        raise ValueError("declared BI reference differs from the actual native/reference operation")
    matrix = np.zeros((len(observed), len(native)), dtype=np.float64)
    for index, channel in enumerate(observed):
        matrix[index, native.index(channel)] = 1.
        if reference is not None:
            matrix[index, native.index(reference)] -= 1.
    mask = np.asarray([channel in observed for channel in output], dtype=bool)
    receipt = {"schema": "n2p3_bi2014a_observed_projection/1", "acquisition_reference": "right earlobe",
        "source_reference": actual_reference, "raw_channel_names": list(native),
        "observed_raw_channels": list(observed), "output_channels": list(output),
        "matrix": matrix.tolist(), "observed": mask.tolist(),
        "reference_channel": reference, "reference_was_measured": reference is not None,
        "arithmetic": "float64 volts; (Vc-Vright_ear)-(VCz-Vright_ear)=Vc-VCz" if reference == "CZ"
            else "float64 volts; actual declared pointwise linear reference", "missing_slots_are_observations": False}
    return matrix @ eeg_v, output, observed, mask, receipt


def build_bi2014a_subject_dataset(
    subject_dir: str | Path, *, preprocessing: PreprocessingSpec,
    output_channels: tuple[str, ...] = BI2014A_CHANNELS,
    observed_raw_channels: tuple[str, ...] = BI2014A_CHANNELS,
    reference_channel: str | None = None, source_reference: str = "right earlobe",
    independent_episode_scopes: bool | None = None,
) -> EpochDataset:
    """Execute native CSV data, keeping every scheduled event and its evidence map.

    Original causal calls keep their continuous recording execution. Source-compatible
    bidirectional calls crop independent episode support before filtering.
    """
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name.removeprefix("subject_")
    select_allowed_subjects("BI2014a", [int(subject_id)], explicit=True)
    table = pd.read_csv(_require_csv(subject_dir), header=None)
    recovered = _recover_candidate_table(subject_dir, table)
    eeg_v = table.iloc[:, 1:17].to_numpy(dtype=np.float64).T * 1e-6
    if not np.isfinite(eeg_v).all():
        raise ValueError(f"{subject_dir.name}: non-finite EEG samples")
    projected, output, observed, channel_mask, projection = _observed_projection(
        eeg_v, output_channels, observed_raw_channels, reference_channel, source_reference)
    raw = mne.io.RawArray(projected, mne.create_info(list(observed), BI2014A_SFREQ, ch_types="eeg"), verbose=False)
    events = np.column_stack((recovered.scheduled_flash_sample,
        np.zeros(len(recovered.scheduled_flash_sample), dtype=np.int64), recovered.scheduled_flash_code))
    scoped = preprocessing.filter_phase == "zero" if independent_episode_scopes is None else independent_episode_scopes
    if preprocessing.filter_phase == "zero" and not scoped:
        raise ValueError("bidirectional BI candidate filtering requires independent raw episode scopes")
    scopes = _episode_scopes(raw, events, recovered, preprocessing) if scoped else [None]
    count = len(events)
    evidence = np.full(count, -1, dtype=np.int64)
    statuses = np.full(count, "boundary_dropped", dtype=object)
    details = np.full(count, "terminal_incomplete_repetition_floor_trim", dtype=object)
    availability = np.full(count, np.nan)
    blocks, mapped, receipts, timing = [], [], [], []
    offset = 0
    scope_records = []
    for scope in scopes:
        authorized = np.arange(count) if scope is None else np.asarray(scope.event_indices)
        executable = authorized[authorized < len(recovered.flash_sample)]
        # The final incomplete repetition remains scheduled, with no invented epochs.
        if not len(executable):
            scope_records.append(scope.record())
            continue
        actual_scope = (None if scope is None else replace(scope, event_indices=tuple(executable.tolist())))
        # Causal historical execution retains the same complete-repetition event array.
        execution_events = events[:len(recovered.flash_sample)] if scope is None else events
        result = execute_preprocessing(raw, execution_events, preprocessing, raw_scope=actual_scope,
                                       channels=observed, montage="standard_1005")
        scheduled = np.asarray(result.scheduled_event_indices)
        local_evidence = np.asarray(result.event_evidence_indices)
        available = local_evidence >= 0
        evidence[scheduled[available]] = offset + local_evidence[available]
        statuses[scheduled] = result.event_statuses
        details[scheduled] = result.event_status_details
        availability[scheduled] = result.evidence_available_times_s
        ordered = scheduled[available][np.argsort(local_evidence[available], kind="stable")]
        blocks.append(result.data)
        mapped.append(ordered)
        offset += result.n_epochs
        receipts.append(result.execution_receipt)
        timing.append(result.timing_source)
        if scope is not None:
            scope_records.append({**scope.record(), "executable_event_indices": executable.tolist()})
    if not blocks or offset == 0:
        raise ValueError("BI2014a has no retained model evidence in its declared scopes")
    ordered = np.concatenate(mapped)
    values = np.concatenate(blocks)
    padded = np.zeros((offset, len(output), preprocessing.n_times), dtype=np.float32)
    padded[:, [output.index(channel) for channel in observed]] = values
    identity = build_channel_identity(output, channel_mask=channel_mask,
        montage="standard_1005", allow_missing_positions=True)
    labels = recovered.is_target[ordered].astype(np.int64)
    post_audit = BI2014A_FLASH_SCHEDULE.audit_raw_labels(recovered.flash_code[ordered],
        recovered.target_label[ordered], flash_samples=recovered.flash_sample[ordered],
        stage="retained_epochs_after_preprocessing")
    starts, stops = source_epoch_support(events, BI2014A_SFREQ, preprocessing)
    metadata = pd.DataFrame({"subject": np.repeat(subject_id, offset),
        "source_dataset_id": np.repeat("BI2014a", offset), "source_subject_id": np.repeat(subject_id, offset),
        "flash_sample": recovered.flash_sample[ordered], "flash_code": recovered.flash_code[ordered],
        "candidate_id": np.where(recovered.row_code[ordered] >= 0, recovered.row_code[ordered], 6 + recovered.col_code[ordered]),
        "raw_is_target": labels.astype(bool), "raw_target_label": recovered.target_label[ordered],
        "row_code": recovered.row_code[ordered], "col_code": recovered.col_code[ordered],
        "target_row": recovered.target_row[ordered], "target_col": recovered.target_col[ordered],
        "selection_id": recovered.selection_id[ordered], "repetition_index": recovered.repetition_index[ordered],
        "selection_boundary_reason": recovered.selection_boundary_reason[ordered],
        "acquisition_time_s": recovered.flash_sample[ordered] / BI2014A_SFREQ,
        "source_epoch_start_sample": starts[ordered], "source_epoch_stop_sample": stops[ordered]})
    timeline = ScheduledEventTimeline(event_ids=np.asarray([f"BI2014a:{subject_id}:{index}" for index in range(count)]),
        group_ids=recovered.scheduled_selection_id, subject_ids=np.repeat(subject_id, count),
        stimulus_ids=recovered.scheduled_flash_code, onset_samples=recovered.scheduled_flash_sample,
        onset_times_s=recovered.scheduled_flash_sample / BI2014A_SFREQ,
        evidence_available_times_s=availability, evidence_indices=evidence,
        statuses=statuses.astype(str), status_details=details.astype(str),
        dataset_ids=np.repeat("BI2014a", count), session_ids=np.repeat("0", count), run_ids=np.repeat("0", count),
        selection_ids=recovered.scheduled_selection_id,
        candidate_ids=np.asarray([BI2014A_FLASH_SCHEDULE.decode(code).candidate_key for code in recovered.scheduled_flash_code]),
        target_candidate_ids=np.repeat("", count), repetition_indices=recovered.scheduled_repetition_index,
        complete=True, online_causal=preprocessing.filter_phase == "forward",
        timing_source=";".join(dict.fromkeys(timing)))
    execution = (receipts[0] if len(receipts) == 1 else
        {"schema": "n2p3_preprocessing_execution_bundle/1", "recordings": receipts, "unverified_inputs": 0})
    with _require_csv(subject_dir).open("rb") as source_stream:
        source_sha = hashlib.file_digest(source_stream, "sha256").hexdigest()
    provenance = {"source": "bi2014a_raw_csv", "subject_dir": str(subject_dir.resolve()), "source_sha256": source_sha,
        "source_sample_rate_hz": BI2014A_SFREQ, "model_input_sample_rate_hz": preprocessing.sfreq,
        "source_reference": projection["source_reference"], "acquisition_reference": "right earlobe",
        "observed_spatial_projection": projection, "signal_unit": "V", "preprocessing_execution": execution,
        "candidate_task_contract": BI2014A_CANDIDATE_TASK_CONTRACT.record(), "raw_episode_scopes": scope_records,
        "flash_schedule": {"n_flashes": len(recovered.flash_sample), "scheduled_flashes": count,
            "n_targets": int(recovered.is_target.sum()), "n_nontargets": int((~recovered.is_target).sum()),
            "n_repetitions": recovered.n_repetitions, "dropped_tail_flashes": recovered.dropped_tail_flashes,
            "candidate_grid": "6x6", "label_source": "derived_from_flash_code", "raw_label_role": "per_event_cross_check_only",
            "raw_label_audit": recovered.raw_label_audit.to_record(BI2014A_FLASH_SCHEDULE),
            "post_preprocessing_label_audit": post_audit.to_record(BI2014A_FLASH_SCHEDULE),
            "selection_boundary_rule": "target_pair_change_or_raw_session_100_or_level_restart_104",
            "n_explicit_boundaries": recovered.n_explicit_boundaries}}
    dataset = EpochDataset(name="BI2014a-candidate", X=padded, y=labels,
        subject_ids=np.repeat(subject_id, offset), channel_names=identity.names, channel_positions_m=identity.coords,
        channel_mask=channel_mask, preprocessing=preprocessing, event_timeline=timeline, metadata=metadata,
        provenance=provenance, trial_channel_mask=np.broadcast_to(channel_mask, padded.shape[:2]).copy(),
        identity_table=DatasetIdentityTable.from_source_rows([subject_id], ["BI2014a"]))
    validate_candidate_membership_metadata(metadata, BI2014A_CANDIDATE_TASK_CONTRACT, labels=labels)
    dataset.validate(require_labels=True)
    return dataset


@dataclass(frozen=True)
class BI2014APreparedParticipant:
    calibration: EpochDataset
    future: EpochDataset
    future_decisions: tuple[MembershipDecision, ...]
    ledger: dict


def _prepared_future_decisions(future, maximum_horizon):
    """Compile membership and every outer field from label-free scheduled rows."""
    if type(maximum_horizon) is not int or maximum_horizon < 1:
        raise ValueError("BI physical decision ledger requires a positive horizon declaration")
    timeline = future.event_timeline
    if not timeline.complete:
        raise ValueError("BI physical decision ledger requires its known original schedule")
    decisions, descriptors = [], []
    for name in dict.fromkeys(timeline.selection_ids.tolist()):
        rows = np.flatnonzero(timeline.selection_ids == name)
        repetitions = int(timeline.repetition_indices[rows].max()) + 1
        decision = MembershipDecision.row_column(timeline.stimulus_ids[rows], n_rows=6, n_columns=6,
            event_ids=tuple(timeline.event_ids[rows]), repetition_indices=timeline.repetition_indices[rows],
            repetitions=repetitions, evidence_indices=timeline.evidence_indices[rows], timeline_complete=timeline.complete)
        full_depth = len(rows) // 12
        decisions.append(decision)
        descriptors.append({"episode_id": name, "scheduled_flashes": len(rows),
            "retained_flashes": int(np.count_nonzero(timeline.evidence_indices[rows] >= 0)),
            "complete_raw_repetitions": full_depth, "terminal_partial_flashes": len(rows) % 12,
            "terminal_censor": len(rows) % 12 != 0, "membership_decision": decision.record(),
            "horizons": [{"repetitions": horizon, "decision": decision.prefix(horizon).record()}
                         for horizon in range(1, min(full_depth, maximum_horizon) + 1)]})
    return tuple(decisions), descriptors


def build_bi2014a_prepared_participant(subject_dir: str | Path, *, preprocessing: PreprocessingSpec,
        calibration_episode_count: int = 5, output_channels: tuple[str, ...] = ("FZ", "P3", "PZ", "P4", "PO8", "OZ"),
        observed_raw_channels: tuple[str, ...] = ("P3", "PZ", "P4", "OZ"), reference_channel: str = "CZ",
        maximum_horizon: int = 8) -> BI2014APreparedParticipant:
    """Five declared complete raw episodes for C; every later episode stays requested."""
    if type(calibration_episode_count) is not int or calibration_episode_count < 1:
        raise ValueError("calibration_episode_count must be an explicit positive episode budget")
    if type(maximum_horizon) is not int or maximum_horizon < 1:
        raise ValueError("maximum_horizon must be a positive physical repetition depth")
    assert_bi2014a_candidate_input_contract(preprocessing)
    dataset = build_bi2014a_subject_dataset(subject_dir, preprocessing=preprocessing,
        output_channels=output_channels, observed_raw_channels=observed_raw_channels,
        reference_channel=reference_channel, independent_episode_scopes=True)
    timeline = dataset.event_timeline
    episodes = tuple(dict.fromkeys(timeline.selection_ids.tolist()))
    complete = [name for name in episodes if np.count_nonzero(timeline.selection_ids == name) % 12 == 0]
    if len(complete) < calibration_episode_count:
        raise ValueError("BI episode budget leaves no independent requested future episodes")
    calibration_ids = tuple(complete[:calibration_episode_count])
    last_calibration = episodes.index(calibration_ids[-1])
    future_ids = episodes[last_calibration + 1:]
    if not future_ids:
        raise ValueError("BI episode budget leaves no independent requested future episodes")
    c_mask = np.isin(timeline.selection_ids, calibration_ids)
    f_mask = np.isin(timeline.selection_ids, future_ids)
    calibration = select_scheduled_events(dataset, c_mask, reason="authorized_complete_raw_episode_calibration")
    future = select_scheduled_events(dataset, f_mask, reason="common_future_after_declared_episode_budget")
    truths = {}
    for name in future_ids:
        known = dataset.metadata[dataset.metadata.selection_id == name]
        target = None if known.empty else int(known.target_row.iloc[0] * 6 + known.target_col.iloc[0])
        truths[name] = target
    # Public CSV codes 60/80 encode truth. Future inference receives only the
    # physical row/column flash-set index; raw codes and label audits stay with
    # the scientific parent, not the inference dataset.
    physical_stimuli = np.asarray([BI2014A_FLASH_SCHEDULE.decode(code).candidate_index +
        (6 if BI2014A_FLASH_SCHEDULE.decode(code).axis == "column" else 0)
        for code in future.event_timeline.stimulus_ids], dtype=np.int64)
    future = replace(future, y=None,
        event_timeline=replace(future.event_timeline, stimulus_ids=physical_stimuli),
        metadata=future.metadata.drop(columns=["target_row", "target_col", "raw_target_label", "raw_is_target",
                                              "flash_code", "selection_boundary_reason"]),
        provenance={key: value for key, value in future.provenance.items() if key != "flash_schedule"})
    future.validate(require_labels=False)
    decisions, descriptors = _prepared_future_decisions(future, maximum_horizon)
    scopes = dataset.provenance["raw_episode_scopes"]
    c_scope = [row for row in scopes if row["name"] in calibration_ids]
    f_scope = [row for row in scopes if row["name"] in future_ids]
    ledger = {"schema": "n2p3_bi2014a_episode_preparation/1", "dataset_id": "BI2014a",
        "subject": str(dataset.subject_ids[0]), "source_sample_rate_hz": BI2014A_SFREQ,
        "model_input_sample_rate_hz": preprocessing.sfreq, "preprocessing": preprocessing.record(),
        "observed_spatial_projection": dataset.provenance["observed_spatial_projection"],
        "calibration_episode_count": calibration_episode_count, "calibration_episode_ids": list(calibration_ids),
        "maximum_horizon": maximum_horizon,
        "actual_calibration_labels": calibration.n_epochs, "requested_calibration_flashes": calibration.event_timeline.n_events,
        "future_episode_ids": list(future_ids), "future_requested_flashes": future.event_timeline.n_events,
        "calibration_raw_scopes": c_scope, "future_raw_scopes": f_scope,
        "common_future_first_raw_sample": f_scope[0]["start_sample"],
        "calibration_last_raw_sample_exclusive": c_scope[-1]["stop_sample"],
        "future_decisions": descriptors, "parent_scoring_truth": truths,
        "dropped_tail_flashes": dataset.provenance["flash_schedule"]["dropped_tail_flashes"],
        "raw_flash_label_audit": dataset.provenance["flash_schedule"]["raw_label_audit"]}
    return BI2014APreparedParticipant(calibration, future, tuple(decisions), ledger)


def load_bi2014a_prepared_participant(directory: str | Path) -> BI2014APreparedParticipant:
    """Load one sealed C/future pair, retaining its original preparation producer."""
    directory = Path(directory)
    ledger = json.loads((directory / "preparation.json").read_bytes())
    if ledger["schema"] != "n2p3_bi2014a_episode_preparation/1":
        raise ValueError("unsupported BI episode preparation schema")
    for role in ("calibration", "future"):
        with (directory / f"{role}.npz").open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != ledger[f"{role}_cache"]["sha256"]:
            raise ValueError(f"BI {role} cache differs from its sealed preparation")
    calibration = load_epoch_dataset(directory / "calibration.npz", require_labels=True)
    future = load_epoch_dataset(directory / "future.npz", require_labels=False)
    if (future.y is not None or not set(future.event_timeline.stimulus_ids).issubset(range(12))
            or np.any(future.event_timeline.target_candidate_ids != "")
            or {"flash_code", "raw_target_label", "target_row", "target_col", "raw_is_target"} & set(future.metadata)):
        raise ValueError("BI future cache exposes target labels instead of physical flash sets")
    # Older producers saved the actual horizon list rather than its ceiling.
    # Retain that declared extent while rebuilding every descriptor from the
    # numeric cache's actual timeline; no original cache or producer is changed.
    maximum_horizon = ledger.get("maximum_horizon")
    if maximum_horizon is None:
        maximum_horizon = max((row["repetitions"] for descriptor in ledger["future_decisions"]
                               for row in descriptor["horizons"]), default=1)
    decisions, descriptors = _prepared_future_decisions(future, maximum_horizon)
    c_ids = list(dict.fromkeys(calibration.event_timeline.selection_ids.tolist()))
    f_ids = list(dict.fromkeys(future.event_timeline.selection_ids.tolist()))
    if (descriptors != ledger["future_decisions"] or f_ids != ledger["future_episode_ids"]
            or set(ledger["parent_scoring_truth"]) != set(f_ids)
            or c_ids != ledger["calibration_episode_ids"] or len(c_ids) != ledger["calibration_episode_count"]
            or calibration.n_epochs != ledger["actual_calibration_labels"]
            or calibration.event_timeline.n_events != ledger["requested_calibration_flashes"]
            or future.event_timeline.n_events != ledger["future_requested_flashes"]):
        raise ValueError("BI prepared pair differs from its original physical decision ledger")
    return BI2014APreparedParticipant(calibration, future, decisions, ledger)
