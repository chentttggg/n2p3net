"""Strict BNCI2014-008 row/column candidate ingestion.

The public Level-5 MAT files contain continuous EEG and sample-wise stimulus
traces.  This module treats the MAT transition ledger as the acquisition
authority, validates the complete 35-selection schedule, and only then derives
model epochs.  Raw target labels are retained as an audit field; they are not
the source of the candidate-membership labels used for training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, BinaryIO

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from data.candidate_task import (
    CandidateTaskContract,
    validate_candidate_membership_metadata,
)
from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import ScheduledEventTimeline
from data.preprocess import preprocess
from data.raw_artifacts import RawArtifactAttestation

BNCI2014_008_DATASET_ID = "BNCI2014-008"
BNCI2014_008_SOURCE_SAMPLE_RATE_HZ = 256.0
BNCI2014_008_SOURCE_REFERENCE = "right earlobe"
BNCI2014_008_SOURCE_GROUND = "left mastoid"
BNCI2014_008_CHANNELS = ("Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8")
BNCI2014_008_SUBJECT_IDS = tuple(f"A{index:02d}" for index in range(1, 9))
BNCI2014_008_MAT_FIELDS = (
    "channels",
    "X",
    "y",
    "y_stim",
    "trial",
    "classes",
    "classes_stim",
    "gender",
    "age",
    "ALSfrs",
    "onsetALS",
)
BNCI2014_008_SAMPLE_COUNT = 347_704
BNCI2014_008_SELECTION_COUNT = 35
BNCI2014_008_REPETITIONS_PER_SELECTION = 10
BNCI2014_008_GRID_SHAPE = (6, 6)
BNCI2014_008_ROW_COUNT, BNCI2014_008_COLUMN_COUNT = BNCI2014_008_GRID_SHAPE
BNCI2014_008_CANDIDATE_COUNT = BNCI2014_008_ROW_COUNT + BNCI2014_008_COLUMN_COUNT
BNCI2014_008_FLASHES_PER_SELECTION = (
    BNCI2014_008_REPETITIONS_PER_SELECTION * BNCI2014_008_CANDIDATE_COUNT
)
BNCI2014_008_FLASH_COUNT = BNCI2014_008_SELECTION_COUNT * BNCI2014_008_FLASHES_PER_SELECTION
BNCI2014_008_FLASH_DURATION_SAMPLES = 32
BNCI2014_008_FLASH_ONSET_INTERVAL_SAMPLES = 64

_EXPECTED_CLASSES = ("NonTarget", "Target")
_EXPECTED_STIMULUS_CLASSES = tuple(
    [
        *(f"Row{index}" for index in range(1, BNCI2014_008_ROW_COUNT + 1)),
        *(f"Col{index}" for index in range(1, BNCI2014_008_COLUMN_COUNT + 1)),
    ]
)
_SUBJECT_FILE = re.compile(r"^A0[1-8]\.mat$")

BNCI2014_008_CANDIDATE_TASK_CONTRACT = CandidateTaskContract(
    dataset_id=BNCI2014_008_DATASET_ID,
    task_id="row_column_flash_membership",
    population={
        "label": "amyotrophic lateral sclerosis patients",
        "clinical_population": "amyotrophic lateral sclerosis",
        "source_subject_count": len(BNCI2014_008_SUBJECT_IDS),
    },
    evidence_scope={
        "stage": "public_processed_dataset_development",
        "product_confirmation": False,
    },
    membership_kind="row_column",
    grid_shape=BNCI2014_008_GRID_SHAPE,
    candidate_ids=tuple(range(BNCI2014_008_CANDIDATE_COUNT)),
    row_candidate_ids=tuple(range(BNCI2014_008_ROW_COUNT)),
    column_candidate_ids=tuple(range(BNCI2014_008_ROW_COUNT, BNCI2014_008_CANDIDATE_COUNT)),
    target_representation="row_column_intersection",
    raw_target_label_is_target={"1": False, "2": True},
)


@dataclass(frozen=True)
class BNCI2014008CandidateRecord:
    """Validated continuous signal and its derived candidate schedule."""

    source_relative_path: str
    source_sha256: str
    source_provenance: dict[str, object]
    subject_id: str
    eeg_uv: np.ndarray
    flash_sample: np.ndarray
    flash_sample_matlab_1based: np.ndarray
    raw_stimulus_code: np.ndarray
    raw_target_label: np.ndarray
    raw_is_target: np.ndarray
    candidate_id: np.ndarray
    row_code: np.ndarray
    col_code: np.ndarray
    target_row: np.ndarray
    target_col: np.ndarray
    selection_id: np.ndarray
    selection_index: np.ndarray
    repetition_index: np.ndarray
    demographics: dict[str, str]


def _plain_strings(value: object, *, field: str, count: int | None = None) -> tuple[str, ...]:
    """Decode a MATLAB char/cellstr field without accepting arbitrary objects."""

    raw = np.asarray(value)
    output: list[str] = []
    for item in raw.reshape(-1):
        candidate = np.asarray(item) if isinstance(item, np.ndarray) else item
        if isinstance(candidate, np.ndarray):
            if candidate.dtype.kind not in {"U", "S"} or candidate.size != 1:
                raise ValueError(f"MAT field {field!r} must contain only scalar strings.")
            text = str(candidate.reshape(-1)[0])
        elif isinstance(candidate, (str, np.str_)):
            text = str(candidate)
        elif isinstance(candidate, (bytes, np.bytes_)):
            text = bytes(candidate).decode("utf-8")
        else:
            raise ValueError(
                f"MAT field {field!r} contains unsupported object {type(candidate).__name__}."
            )
        if not text.strip():
            raise ValueError(f"MAT field {field!r} contains an empty string.")
        output.append(text.strip())
    if count is not None and len(output) != count:
        raise ValueError(f"MAT field {field!r} must contain {count} strings, got {len(output)}.")
    return tuple(output)


def _numeric_field(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...],
    kinds: frozenset[str],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"MAT field {field!r} must have shape {shape}, got {raw.shape}.")
    if raw.dtype.kind not in kinds or raw.dtype.kind in {"b", "c", "O", "V"}:
        raise ValueError(f"MAT field {field!r} has unsupported dtype {raw.dtype}.")
    if not np.isfinite(raw).all():
        raise ValueError(f"MAT field {field!r} contains non-finite values.")
    return raw


def _integer_values(value: np.ndarray, *, field: str) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64)
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"MAT field {field!r} must contain integer-valued samples.")
    return numeric.astype(np.int64)


def _read_level5_struct(handle: BinaryIO, *, source_name: str) -> np.void:
    inventory = whosmat(handle)
    if inventory != [("data", (1, 1), "struct")]:
        raise ValueError(
            f"{source_name} must contain exactly one 1x1 Level-5 struct named 'data'; "
            f"found {inventory!r}."
        )
    handle.seek(0)
    archive = loadmat(
        handle,
        variable_names=("data",),
        struct_as_record=True,
        squeeze_me=False,
        chars_as_strings=True,
        mat_dtype=True,
    )
    data = np.asarray(archive.get("data"))
    if data.shape != (1, 1) or data.dtype.names is None:
        raise ValueError(f"{source_name}: data must decode as one structured record.")
    fields = tuple(data.dtype.names)
    if set(fields) != set(BNCI2014_008_MAT_FIELDS) or len(fields) != len(BNCI2014_008_MAT_FIELDS):
        missing = sorted(set(BNCI2014_008_MAT_FIELDS) - set(fields))
        unknown = sorted(set(fields) - set(BNCI2014_008_MAT_FIELDS))
        raise ValueError(
            f"{source_name}: MAT fields disagree with the source contract; "
            f"missing={missing}, unknown={unknown}."
        )
    return data[0, 0]


def _validate_sample_traces(
    y: np.ndarray,
    y_stim: np.ndarray,
    *,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_y = _integer_values(y.reshape(-1), field="y")
    stimulus = _integer_values(y_stim.reshape(-1), field="y_stim")
    if set(np.unique(raw_y).tolist()) != {0, 1, 2}:
        raise ValueError(f"{source_name}: y must use exactly the raw codebook 0/1/2.")
    if set(np.unique(stimulus).tolist()) != set(range(13)):
        raise ValueError(f"{source_name}: y_stim must use exactly the raw codebook 0..12.")

    direct_changes = np.flatnonzero(
        (stimulus[:-1] != 0) & (stimulus[1:] != 0) & (stimulus[:-1] != stimulus[1:])
    )
    if len(direct_changes):
        raise ValueError(
            f"{source_name}: y_stim has a nonzero-to-nonzero transition at sample "
            f"{int(direct_changes[0] + 1)}."
        )
    onsets = np.flatnonzero((stimulus != 0) & np.r_[True, stimulus[:-1] == 0])
    offsets = np.flatnonzero((stimulus[:-1] != 0) & (stimulus[1:] == 0)) + 1
    if stimulus[-1] != 0:
        raise ValueError(f"{source_name}: y_stim ends inside a nonzero flash pulse.")
    if len(onsets) != BNCI2014_008_FLASH_COUNT or len(offsets) != len(onsets):
        raise ValueError(
            f"{source_name}: expected {BNCI2014_008_FLASH_COUNT} zero-to-nonzero flash "
            f"transitions, got {len(onsets)}."
        )
    durations = offsets - onsets
    if np.any(durations != BNCI2014_008_FLASH_DURATION_SAMPLES):
        bad = int(np.flatnonzero(durations != BNCI2014_008_FLASH_DURATION_SAMPLES)[0])
        raise ValueError(
            f"{source_name}: flash {bad} lasts {int(durations[bad])} samples; expected "
            f"{BNCI2014_008_FLASH_DURATION_SAMPLES}."
        )
    stimulated = stimulus != 0
    if np.any(raw_y[~stimulated] != 0) or np.any(~np.isin(raw_y[stimulated], (1, 2))):
        raise ValueError(f"{source_name}: y and y_stim disagree on flash-active samples.")
    onset_labels = raw_y[onsets]
    for index, (start, stop) in enumerate(zip(onsets, offsets, strict=True)):
        if not np.all(raw_y[start:stop] == onset_labels[index]):
            raise ValueError(f"{source_name}: y changes inside flash {index}.")
    return onsets.astype(np.int64), stimulus[onsets], onset_labels


def _decode_schedule(
    *,
    subject_id: str,
    trial_matlab_1based: np.ndarray,
    flash_sample: np.ndarray,
    raw_stimulus_code: np.ndarray,
    raw_target_label: np.ndarray,
    sample_count: int,
) -> dict[str, np.ndarray]:
    trial_raw = _integer_values(trial_matlab_1based.reshape(-1), field="trial")
    if np.any((trial_raw < 1) | (trial_raw > sample_count)) or np.any(np.diff(trial_raw) <= 0):
        raise ValueError("trial must contain increasing MATLAB 1-based sample indices.")
    trial_zero_based = trial_raw - 1
    candidate_id = raw_stimulus_code.astype(np.int64) - 1
    raw_is_target = raw_target_label == 2

    n_events = len(flash_sample)
    selection_index = np.full(n_events, -1, dtype=np.int64)
    repetition_index = np.full(n_events, -1, dtype=np.int64)
    target_row = np.full(n_events, -1, dtype=np.int64)
    target_col = np.full(n_events, -1, dtype=np.int64)
    selection_id = np.empty(n_events, dtype=object)

    boundaries = np.r_[trial_zero_based, sample_count]
    expected_vocabulary = np.arange(BNCI2014_008_CANDIDATE_COUNT, dtype=np.int64)
    for selection in range(BNCI2014_008_SELECTION_COUNT):
        rows = np.flatnonzero(
            (flash_sample >= boundaries[selection]) & (flash_sample < boundaries[selection + 1])
        )
        if len(rows) != BNCI2014_008_FLASHES_PER_SELECTION:
            raise ValueError(
                f"{subject_id}: selection {selection} must contain "
                f"{BNCI2014_008_FLASHES_PER_SELECTION} flashes, got {len(rows)}."
            )
        if int(flash_sample[rows[0]]) != int(trial_zero_based[selection]):
            raise ValueError(
                f"{subject_id}: trial[{selection}] is not the MATLAB 1-based index of its "
                "first flash onset."
            )
        intervals = np.diff(flash_sample[rows])
        if np.any(intervals != BNCI2014_008_FLASH_ONSET_INTERVAL_SAMPLES):
            bad = int(np.flatnonzero(intervals != BNCI2014_008_FLASH_ONSET_INTERVAL_SAMPLES)[0])
            raise ValueError(
                f"{subject_id}: selection {selection} flash interval {bad} is "
                f"{int(intervals[bad])} samples, expected "
                f"{BNCI2014_008_FLASH_ONSET_INTERVAL_SAMPLES}."
            )

        pair: tuple[int, int] | None = None
        for repetition in range(BNCI2014_008_REPETITIONS_PER_SELECTION):
            rep_rows = rows[
                repetition * BNCI2014_008_CANDIDATE_COUNT : (repetition + 1)
                * BNCI2014_008_CANDIDATE_COUNT
            ]
            rep_candidates = candidate_id[rep_rows]
            if not np.array_equal(np.sort(rep_candidates), expected_vocabulary):
                raise ValueError(
                    f"{subject_id}: selection {selection} repetition {repetition} does not "
                    f"cover each candidate 0..{BNCI2014_008_CANDIDATE_COUNT - 1} exactly once."
                )
            target_candidates = rep_candidates[raw_is_target[rep_rows]]
            target_rows = target_candidates[target_candidates < BNCI2014_008_ROW_COUNT]
            target_columns = (
                target_candidates[target_candidates >= BNCI2014_008_ROW_COUNT]
                - BNCI2014_008_ROW_COUNT
            )
            if len(target_rows) != 1 or len(target_columns) != 1:
                raise ValueError(
                    f"{subject_id}: selection {selection} repetition {repetition} needs "
                    "exactly one target row and one target column."
                )
            current = (int(target_rows[0]), int(target_columns[0]))
            if pair is None:
                pair = current
            elif pair != current:
                raise ValueError(
                    f"{subject_id}: target row/column drifts inside selection {selection}: "
                    f"{pair} -> {current}."
                )
            repetition_index[rep_rows] = repetition
        if pair is None:
            raise AssertionError("A validated selection must define one target pair.")
        selection_index[rows] = selection
        target_row[rows] = pair[0]
        target_col[rows] = pair[1]
        selection_id[rows] = f"{subject_id}:selection{selection:02d}"

    if np.any(selection_index < 0):
        raise AssertionError("Every flash must be assigned to exactly one selection.")
    row_code = np.where(candidate_id < BNCI2014_008_ROW_COUNT, candidate_id, -1).astype(np.int64)
    col_code = np.where(
        candidate_id >= BNCI2014_008_ROW_COUNT,
        candidate_id - BNCI2014_008_ROW_COUNT,
        -1,
    ).astype(np.int64)
    derived_target = (row_code == target_row) | (col_code == target_col)
    if not np.array_equal(derived_target, raw_is_target):
        raise ValueError("Raw y labels disagree with the stable row/column target pair.")
    return {
        "candidate_id": candidate_id,
        "raw_is_target": raw_is_target,
        "row_code": row_code,
        "col_code": col_code,
        "target_row": target_row,
        "target_col": target_col,
        "selection_id": np.asarray(selection_id, dtype=str),
        "selection_index": selection_index,
        "repetition_index": repetition_index,
    }


def load_bnci2014_008_candidate_record(
    source_relative_path: str,
    *,
    raw_artifact_attestation: RawArtifactAttestation,
) -> BNCI2014008CandidateRecord:
    """Load one official MAT only from its verified immutable snapshot."""

    if not isinstance(raw_artifact_attestation, RawArtifactAttestation):
        raise TypeError("raw_artifact_attestation must be verified.")
    raw_artifact_attestation.assert_dataset_class(BNCI2014_008_DATASET_ID)
    source_name = PurePosixPath(source_relative_path).name
    if not _SUBJECT_FILE.fullmatch(source_name):
        raise ValueError("BNCI2014-008 source files must be named A01.mat through A08.mat.")
    subject_id = PurePosixPath(source_name).stem
    snapshot = raw_artifact_attestation.snapshot_for(source_relative_path)
    with snapshot.open_verified() as handle:
        record = _read_level5_struct(handle, source_name=source_name)
    channels = _plain_strings(
        record["channels"], field="channels", count=len(BNCI2014_008_CHANNELS)
    )
    classes = _plain_strings(record["classes"], field="classes", count=2)
    stimulus_classes = _plain_strings(
        record["classes_stim"],
        field="classes_stim",
        count=BNCI2014_008_CANDIDATE_COUNT,
    )
    if channels != BNCI2014_008_CHANNELS:
        raise ValueError(f"{source_name}: channels must be {BNCI2014_008_CHANNELS!r}.")
    if classes != _EXPECTED_CLASSES or stimulus_classes != _EXPECTED_STIMULUS_CLASSES:
        raise ValueError(f"{source_name}: class strings disagree with the row/column codebook.")
    demographics = {
        field: _plain_strings(record[field], field=field, count=1)[0]
        for field in ("gender", "age", "ALSfrs", "onsetALS")
    }

    eeg_uv = _numeric_field(
        record["X"],
        field="X",
        shape=(BNCI2014_008_SAMPLE_COUNT, len(BNCI2014_008_CHANNELS)),
        kinds=frozenset({"f"}),
    )
    y = _numeric_field(
        record["y"],
        field="y",
        shape=(BNCI2014_008_SAMPLE_COUNT, 1),
        kinds=frozenset({"f", "i", "u"}),
    )
    y_stim = _numeric_field(
        record["y_stim"],
        field="y_stim",
        shape=(BNCI2014_008_SAMPLE_COUNT, 1),
        kinds=frozenset({"i", "u"}),
    )
    trial = _numeric_field(
        record["trial"],
        field="trial",
        shape=(1, BNCI2014_008_SELECTION_COUNT),
        kinds=frozenset({"f", "i", "u"}),
    )
    flash_sample, raw_stimulus_code, raw_target_label = _validate_sample_traces(
        y, y_stim, source_name=source_name
    )
    decoded = _decode_schedule(
        subject_id=subject_id,
        trial_matlab_1based=trial,
        flash_sample=flash_sample,
        raw_stimulus_code=raw_stimulus_code,
        raw_target_label=raw_target_label,
        sample_count=len(eeg_uv),
    )
    return BNCI2014008CandidateRecord(
        source_relative_path=source_relative_path,
        source_sha256=snapshot.sha256,
        source_provenance=raw_artifact_attestation.source_provenance_record(source_relative_path),
        subject_id=subject_id,
        eeg_uv=eeg_uv,
        flash_sample=flash_sample,
        flash_sample_matlab_1based=flash_sample + 1,
        raw_stimulus_code=raw_stimulus_code,
        raw_target_label=raw_target_label,
        raw_is_target=decoded["raw_is_target"],
        candidate_id=decoded["candidate_id"],
        row_code=decoded["row_code"],
        col_code=decoded["col_code"],
        target_row=decoded["target_row"],
        target_col=decoded["target_col"],
        selection_id=decoded["selection_id"],
        selection_index=decoded["selection_index"],
        repetition_index=decoded["repetition_index"],
        demographics=demographics,
    )


def bnci2014_008_mat_source_contract() -> dict[str, Any]:
    """Return the JSON-safe physical and indexing contract for provenance."""

    return {
        "schema": "bnci2014_008_mat_source/1",
        "container": "MATLAB Level 5",
        "top_level_variable": "data",
        "source_sample_rate_hz": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        "source_reference": BNCI2014_008_SOURCE_REFERENCE,
        "source_ground": BNCI2014_008_SOURCE_GROUND,
        "source_signal_unit": "uV",
        "physical_contract_authority": (
            "BNCI Horizon 008-2014 dataset documentation; sample rate, reference, "
            "ground, and signal unit are not embedded as MAT fields"
        ),
        "channels": list(BNCI2014_008_CHANNELS),
        "expected_subject_files": [f"{value}.mat" for value in BNCI2014_008_SUBJECT_IDS],
        "expected_shape": [BNCI2014_008_SAMPLE_COUNT, len(BNCI2014_008_CHANNELS)],
        "matlab_indexing": {
            "trial": "1-based sample index",
            "internal_flash_sample": "0-based sample index after explicit subtraction of one",
            "metadata_flash_sample_matlab_1based": "preserved original sample index",
        },
        "flash_transition_rule": "y_stim zero-to-nonzero transition",
        "flash_duration_samples": BNCI2014_008_FLASH_DURATION_SAMPLES,
        "flash_onset_interval_samples": BNCI2014_008_FLASH_ONSET_INTERVAL_SAMPLES,
        "selection_count": BNCI2014_008_SELECTION_COUNT,
        "flashes_per_selection": BNCI2014_008_FLASHES_PER_SELECTION,
    }


def discover_bnci2014_008_files(
    raw_artifact_attestation: RawArtifactAttestation,
) -> tuple[str, ...]:
    """Require the exact eight-file inventory from a verified BNCI manifest."""

    raw_artifact_attestation.assert_dataset_class(BNCI2014_008_DATASET_ID)
    manifest_paths = tuple(item.relative_path for item in raw_artifact_attestation.verified_files)
    by_name = {PurePosixPath(path).name: path for path in manifest_paths}
    expected_names = tuple(f"{subject}.mat" for subject in BNCI2014_008_SUBJECT_IDS)
    missing = [name for name in expected_names if name not in by_name]
    unexpected = sorted(name for name in by_name if name not in set(expected_names))
    duplicate_names = len(by_name) != len(manifest_paths)
    if missing or unexpected:
        raise ValueError(
            "BNCI2014-008 requires its exact eight-file manifest inventory; "
            f"missing={missing}, unexpected={unexpected}, duplicate_names={duplicate_names}."
        )
    if duplicate_names:
        raise ValueError("BNCI2014-008 manifest subject basenames must be unique.")
    return tuple(by_name[name] for name in expected_names)


def build_bnci2014_008_subject_dataset(
    record: BNCI2014008CandidateRecord,
    *,
    preprocessing: PreprocessingSpec,
) -> EpochDataset:
    """Build one strict causal-v3 EpochDataset/5 subject cache."""

    if not isinstance(record, BNCI2014008CandidateRecord):
        raise TypeError("record must be a verified BNCI2014008CandidateRecord.")
    recovered = record
    eeg_v = (recovered.eeg_uv.astype(np.float64, copy=False) * 1e-6).T
    info = mne.create_info(
        list(BNCI2014_008_CHANNELS),
        BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(eeg_v, info, verbose=False)
    events = np.column_stack(
        (
            recovered.flash_sample,
            np.zeros(BNCI2014_008_FLASH_COUNT, dtype=np.int64),
            recovered.raw_stimulus_code,
        )
    ).astype(np.int64, copy=False)
    result = preprocess(
        raw,
        events,
        sfreq=preprocessing.sfreq,
        l_freq=preprocessing.l_freq,
        h_freq=preprocessing.h_freq,
        tmin=preprocessing.tmin_ms / 1000.0,
        tmax=preprocessing.tmax_ms / 1000.0,
        n_times=preprocessing.n_times,
        reject_threshold=preprocessing.reject_threshold_v,
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
        channels=BNCI2014_008_CHANNELS,
        montage="standard_1005",
    )
    evidence = np.asarray(result.event_evidence_indices, dtype=np.int64)
    available = np.flatnonzero(evidence >= 0)
    ordered = available[np.argsort(evidence[available], kind="stable")]
    if len(ordered) != result.n_epochs:
        raise AssertionError("Preprocessing evidence mapping is not bijective.")
    derived_labels = recovered.raw_is_target[ordered].astype(np.int64)
    metadata = pd.DataFrame(
        {
            "subject": np.repeat(recovered.subject_id, result.n_epochs),
            "selection_id": recovered.selection_id[ordered],
            "selection_index": recovered.selection_index[ordered],
            "repetition_index": recovered.repetition_index[ordered],
            "candidate_id": recovered.candidate_id[ordered],
            "row_code": recovered.row_code[ordered],
            "col_code": recovered.col_code[ordered],
            "target_row": recovered.target_row[ordered],
            "target_col": recovered.target_col[ordered],
            "is_target": recovered.raw_is_target[ordered],
            "raw_is_target": recovered.raw_is_target[ordered],
            "raw_target_label": recovered.raw_target_label[ordered],
            "raw_stimulus_code": recovered.raw_stimulus_code[ordered],
            "flash_sample_matlab_1based": recovered.flash_sample_matlab_1based[ordered],
            "acquisition_time_s": result.event_times_s[ordered],
        }
    )
    validate_candidate_membership_metadata(
        metadata,
        BNCI2014_008_CANDIDATE_TASK_CONTRACT,
        labels=derived_labels,
    )
    identity = build_channel_identity(
        result.channel_names,
        channel_mask=np.ones(len(result.channel_names), dtype=bool),
        montage="standard_1005",
        allow_missing_positions=False,
    )
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray(
            [
                f"{BNCI2014_008_DATASET_ID}:{recovered.subject_id}:{index}"
                for index in range(BNCI2014_008_FLASH_COUNT)
            ]
        ),
        group_ids=recovered.selection_id,
        subject_ids=np.repeat(recovered.subject_id, BNCI2014_008_FLASH_COUNT),
        stimulus_ids=recovered.raw_stimulus_code.astype(np.int64),
        onset_samples=recovered.flash_sample.astype(np.int64),
        onset_times_s=result.event_times_s,
        evidence_available_times_s=result.evidence_available_times_s,
        evidence_indices=evidence,
        statuses=result.event_statuses,
        status_details=result.event_status_details,
        dataset_ids=np.repeat(BNCI2014_008_DATASET_ID, BNCI2014_008_FLASH_COUNT),
        session_ids=np.repeat("0", BNCI2014_008_FLASH_COUNT),
        run_ids=np.repeat("0", BNCI2014_008_FLASH_COUNT),
        selection_ids=recovered.selection_id,
        candidate_ids=recovered.candidate_id.astype(str),
        repetition_indices=recovered.repetition_index,
        complete=True,
        online_causal=result.online_causal,
        timing_source=(
            "bnci2014_008_mat_y_stim_zero_to_nonzero;"
            "matlab_trial_1based_to_internal_0based;epoched_resample"
        ),
    )
    dataset = EpochDataset(
        name="BNCI2014-008-candidate",
        X=result.data.astype(np.float32, copy=False),
        y=derived_labels,
        subject_ids=np.repeat(recovered.subject_id, result.n_epochs),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(result.n_channels, dtype=bool),
        preprocessing=preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "bnci_horizon_008_2014_mat_level5",
            "source_relative_path": recovered.source_relative_path,
            "source_file_sha256": recovered.source_sha256,
            **recovered.source_provenance,
            "source_reference": BNCI2014_008_SOURCE_REFERENCE,
            "source_ground": BNCI2014_008_SOURCE_GROUND,
            "source_sample_rate_hz": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
            "source_signal_unit": "uV",
            "signal_unit": "V",
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "mat_source_contract": bnci2014_008_mat_source_contract(),
            "candidate_task_contract": BNCI2014_008_CANDIDATE_TASK_CONTRACT.record(),
            "schedule_audit": {
                "n_selections": BNCI2014_008_SELECTION_COUNT,
                "n_repetitions_per_selection": BNCI2014_008_REPETITIONS_PER_SELECTION,
                "n_flash_onsets": BNCI2014_008_FLASH_COUNT,
                "candidate_coverage_per_repetition": list(range(BNCI2014_008_CANDIDATE_COUNT)),
                "raw_label_role": "cross_check_only",
                "training_label_source": "row_column_candidate_membership",
            },
            "demographics": recovered.demographics,
        },
    )
    dataset.validate(require_labels=True)
    return dataset
