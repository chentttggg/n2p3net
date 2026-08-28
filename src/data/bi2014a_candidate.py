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

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from data.channel import build_channel_identity
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import observed_only_timeline
from data.preprocess import preprocess

BI2014A_SFREQ = 512.0
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


@dataclass(frozen=True)
class BI2014ACandidateRecord:
    flash_sample: np.ndarray
    flash_code: np.ndarray
    target_label: np.ndarray
    row_code: np.ndarray
    col_code: np.ndarray
    target_row: np.ndarray
    target_col: np.ndarray
    selection_id: np.ndarray
    repetition_index: np.ndarray
    n_repetitions: int
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
    table = pd.read_csv(csv_path, header=None)
    if table.shape[1] <= _LABEL_COLUMN:
        raise ValueError(f"{csv_path} lacks the BI2014a event columns.")
    flash_code = table[_FLASH_COLUMN].to_numpy(dtype=np.float64).astype(np.int64)
    target_label = table[_LABEL_COLUMN].to_numpy(dtype=np.float64).astype(np.int64)
    flash_samples = np.flatnonzero((flash_code >= 20) & (flash_code <= 85))

    # Some public BI2014a files end with one or two trailing flashes that do
    # not form a complete 12-flash repetition. Keep only complete repetitions
    # and report the dropped tail in the audit.
    n_repetitions = len(flash_samples) // 12
    if n_repetitions < 1:
        raise ValueError(f"{subject_dir.name}: fewer than one complete flash repetition.")
    complete_count = n_repetitions * 12
    dropped_tail = len(flash_samples) - complete_count
    flash_samples = flash_samples[:complete_count]
    codes = flash_code[flash_samples]
    labels = target_label[flash_samples]
    expected_targets = 2 * n_repetitions
    expected_nontargets = 10 * n_repetitions
    if np.count_nonzero(labels == 2) != expected_targets:
        raise ValueError(
            f"{subject_dir.name}: expected {expected_targets} target flashes, "
            f"got {np.count_nonzero(labels == 2)}."
        )
    if np.count_nonzero(labels == 1) != expected_nontargets:
        raise ValueError(
            f"{subject_dir.name}: expected {expected_nontargets} non-target flashes, "
            f"got {np.count_nonzero(labels == 1)}."
        )

    row_code = np.full(len(flash_samples), -1, dtype=np.int64)
    col_code = np.full(len(flash_samples), -1, dtype=np.int64)
    target_row = np.full(n_repetitions, -1, dtype=np.int64)
    target_col = np.full(n_repetitions, -1, dtype=np.int64)
    selection_id = np.empty(len(flash_samples), dtype=object)
    repetition_index = np.empty(len(flash_samples), dtype=np.int64)

    previous_pair: tuple[int, int] | None = None
    active_selection = 0

    for rep in range(n_repetitions):
        start = rep * 12
        stop = start + 12
        rep_codes = codes[start:stop]
        t_row_codes = rep_codes[(rep_codes >= 60) & (rep_codes <= 65)]
        t_col_codes = rep_codes[(rep_codes >= 80) & (rep_codes <= 85)]
        n20 = np.count_nonzero((rep_codes >= 20) & (rep_codes <= 25))
        n40 = np.count_nonzero((rep_codes >= 40) & (rep_codes <= 45))
        if len(t_row_codes) != 1 or len(t_col_codes) != 1 or n20 != 5 or n40 != 5:
            raise ValueError(f"{subject_dir.name}: repetition {rep} has invalid flash structure.")

        pair = (int(t_row_codes[0] - 60), int(t_col_codes[0] - 80))
        if previous_pair != pair:
            active_selection += 1
            previous_pair = pair
        target_row[rep] = pair[0]
        target_col[rep] = pair[1]

        for local, global_idx in enumerate(range(start, stop)):
            code = int(rep_codes[local])
            if 20 <= code <= 25:
                row_code[global_idx] = code - 20
            elif 40 <= code <= 45:
                col_code[global_idx] = code - 40
            elif 60 <= code <= 65:
                row_code[global_idx] = code - 60
            elif 80 <= code <= 85:
                col_code[global_idx] = code - 80
            else:
                raise AssertionError(f"unexpected flash code {code}")
            selection_id[global_idx] = f"{subject_dir.name}:selection{active_selection}"
            repetition_index[global_idx] = int(np.count_nonzero(selection_id[start : global_idx + 1] == selection_id[global_idx])) - 1

    return BI2014ACandidateRecord(
        flash_sample=flash_samples,
        flash_code=codes,
        target_label=labels,
        row_code=row_code,
        col_code=col_code,
        target_row=np.repeat(target_row, 12),
        target_col=np.repeat(target_col, 12),
        selection_id=np.asarray(selection_id),
        repetition_index=repetition_index,
        n_repetitions=n_repetitions,
        dropped_tail_flashes=dropped_tail,
    )


def build_bi2014a_subject_dataset(
    subject_dir: str | Path,
    *,
    preprocessing: PreprocessingSpec,
) -> EpochDataset:
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name.removeprefix("subject_")
    recovered = recover_bi2014a_candidates(subject_dir)

    csv_path = _require_csv(subject_dir)
    table = pd.read_csv(csv_path, header=None)
    eeg_uv = table.iloc[:, 1:17].to_numpy(dtype=np.float64).T
    eeg_v = (eeg_uv * 1e-6).astype(np.float32)
    if not np.isfinite(eeg_v).all():
        raise ValueError(f"{subject_dir.name}: non-finite EEG samples.")
    info = mne.create_info(list(BI2014A_CHANNELS), BI2014A_SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(eeg_v, info, verbose=False)

    events = np.column_stack(
        [
            recovered.flash_sample.astype(np.int64),
            np.zeros(len(recovered.flash_sample), dtype=np.int64),
            recovered.flash_code.astype(np.int64),
        ]
    )
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
        resample_domain=preprocessing.resample_domain,
        resample_method=preprocessing.resample_method,
        resample_npad=preprocessing.resample_npad,
        resample_window=preprocessing.resample_window,
        resample_pad=preprocessing.resample_pad,
        channels=BI2014A_CHANNELS,
        montage="standard_1005",
    )
    evidence = np.asarray(result.event_evidence_indices, dtype=np.int64)
    available = np.flatnonzero(evidence >= 0)
    ordered = available[np.argsort(evidence[available], kind="stable")]
    if len(ordered) != result.n_epochs:
        raise AssertionError("preprocess evidence mapping is not bijective.")

    identity = build_channel_identity(
        result.channel_names,
        channel_mask=np.ones(len(result.channel_names), dtype=bool),
        montage="standard_1005",
        allow_missing_positions=False,
    )
    metadata = pd.DataFrame(
        {
            "subject": np.repeat(subject_id, result.n_epochs),
            "flash_sample": recovered.flash_sample[ordered],
            "flash_code": recovered.flash_code[ordered],
            "row_code": recovered.row_code[ordered],
            "col_code": recovered.col_code[ordered],
            "target_row": recovered.target_row[ordered],
            "target_col": recovered.target_col[ordered],
            "selection_id": recovered.selection_id[ordered],
            "repetition_index": recovered.repetition_index[ordered],
            "acquisition_time_s": result.event_times_s[ordered],
        }
    )
    timeline = observed_only_timeline(
        dataset_id=f"BI2014a-candidate:{subject_id}",
        subject_ids=np.repeat(subject_id, result.n_epochs),
        stimulus_ids=recovered.flash_code[ordered],
        onset_times_s=result.event_times_s[ordered],
        group_ids=np.asarray(recovered.selection_id[ordered]),
        online_causal=result.online_causal,
        timing_source="bi2014a_raw_csv_flash_codes;epoched_resample",
        selection_ids=np.asarray(recovered.selection_id[ordered]),
        session_ids=np.repeat("0", result.n_epochs),
        run_ids=np.repeat("0", result.n_epochs),
    )
    dataset = EpochDataset(
        name="BI2014a-candidate",
        X=result.data.astype(np.float32, copy=False),
        y=(recovered.target_label[ordered] == 2).astype(np.int64),
        subject_ids=np.repeat(subject_id, result.n_epochs).astype(str),
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=np.ones(result.n_channels, dtype=bool),
        preprocessing=preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "bi2014a_raw_csv",
            "subject_dir": str(subject_dir.resolve()),
            "source_sample_rate_hz": BI2014A_SFREQ,
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "source_reference": "right earlobe",
            "signal_unit": "V",
            "flash_schedule": {
                "n_flashes": int(len(recovered.flash_sample)),
                "n_targets": int(np.count_nonzero(recovered.target_label == 2)),
                "n_nontargets": int(np.count_nonzero(recovered.target_label == 1)),
                "n_repetitions": recovered.n_repetitions,
                "candidate_grid": "6x6",
            },
        },
    )
    dataset.validate(require_labels=True)
    return dataset
