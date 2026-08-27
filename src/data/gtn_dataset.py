"""Build the current epoch contract from GTN NIX experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.contract import DEFAULT_GTN_DATA_CONTRACT
from data.epochs import EpochDataset, PreprocessingSpec
from data.events import (
    ScheduledEventTimeline,
    candidate_repetition_indices,
    concatenate_event_timelines,
)
from data.gtn import GTNData, read_gtn_experiment
from data.preprocess import PreprocessResult, preprocess

GTN_CHANNELS = ("Fz", "Cz", "Pz")
GTN_LMBC_PREPROCESSING = PreprocessingSpec(
    name=DEFAULT_GTN_DATA_CONTRACT.name,
    sfreq=DEFAULT_GTN_DATA_CONTRACT.sample_rate_hz,
    l_freq=DEFAULT_GTN_DATA_CONTRACT.l_freq,
    h_freq=DEFAULT_GTN_DATA_CONTRACT.h_freq,
    tmin_ms=DEFAULT_GTN_DATA_CONTRACT.tmin_ms,
    tmax_ms=DEFAULT_GTN_DATA_CONTRACT.tmax_ms,
    n_times=DEFAULT_GTN_DATA_CONTRACT.n_times,
    baseline_mode=DEFAULT_GTN_DATA_CONTRACT.baseline_mode,
    reject_threshold_v=None,
)


@dataclass(frozen=True)
class PreparedGTNExperiment:
    """One GTN selection group after physical preprocessing."""

    experiment_name: str
    subject_id: str
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    timeline: ScheduledEventTimeline
    channel_names: tuple[str, ...]
    channel_positions_m: np.ndarray
    channel_mask: np.ndarray
    source_sample_rate_hz: float
    source_reference: str


def gtn_experiment_dirs(root: str | Path) -> tuple[Path, ...]:
    """Return the deterministic GTN experiment cohort under an EEGBase root."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"GTN root is not a directory: {root}")
    experiments = tuple(sorted(root.glob("Experiment_*_P3_Numbers")))
    if not experiments:
        raise FileNotFoundError(f"No GTN experiment directories found under {root}.")
    return experiments


def _prepare_gtn_experiment(
    experiment_dir: Path,
    gtn: GTNData,
    result: PreprocessResult,
) -> PreparedGTNExperiment:
    """Bind one scheduled GTN digit sequence to its available epoch evidence."""

    scheduled_digits = np.asarray(gtn.events[:, 2], dtype=np.int64)
    if scheduled_digits.ndim != 1 or len(scheduled_digits) != len(gtn.events):
        raise ValueError(f"{experiment_dir.name} has an invalid GTN stimulus sequence.")
    if set(np.unique(scheduled_digits).tolist()) != set(range(1, 10)):
        raise ValueError(f"{experiment_dir.name} does not contain the complete 1-9 candidate vocabulary.")
    if int(gtn.thought_number) not in set(scheduled_digits.tolist()):
        raise ValueError(f"{experiment_dir.name} thought number is absent from scheduled candidates.")

    selection_id = f"gtn:{experiment_dir.name}"
    subject_id = str(gtn.subject_id)
    event_count = len(scheduled_digits)
    group_ids = np.repeat(selection_id, event_count)
    candidates = scheduled_digits.astype(str)
    repetitions = candidate_repetition_indices(candidates, group_ids)
    timeline = ScheduledEventTimeline(
        event_ids=np.asarray([f"{selection_id}:{index}" for index in range(event_count)]),
        group_ids=group_ids,
        subject_ids=np.repeat(subject_id, event_count),
        stimulus_ids=scheduled_digits,
        onset_samples=np.asarray(result.event_samples, dtype=np.int64),
        onset_times_s=np.asarray(result.event_times_s, dtype=float),
        evidence_available_times_s=np.asarray(result.evidence_available_times_s, dtype=float),
        evidence_indices=np.asarray(result.event_evidence_indices, dtype=np.int64),
        statuses=np.asarray(result.event_statuses, dtype=str),
        status_details=np.asarray(result.event_status_details, dtype=str),
        dataset_ids=np.repeat("GTN", event_count),
        session_ids=np.repeat("", event_count),
        run_ids=np.repeat(experiment_dir.name, event_count),
        selection_ids=np.repeat(selection_id, event_count),
        complete=True,
        online_causal=bool(result.online_causal),
        timing_source=(
            "gtn_nix_source_event_samples;epoched_resample;"
            "epoch_right_edge;acausal_preprocessing"
        ),
        candidate_ids=candidates,
        target_candidate_ids=np.repeat(str(int(gtn.thought_number)), event_count),
        repetition_indices=repetitions,
    ).validate(n_epochs=len(result.data))
    available = np.asarray(result.event_indices, dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "subject": np.repeat(subject_id, len(available)),
            "run": np.repeat(experiment_dir.name, len(available)),
            "selection_id": np.repeat(selection_id, len(available)),
            "candidate_id": candidates[available],
            "target_candidate_id": np.repeat(str(int(gtn.thought_number)), len(available)),
            "repetition_index": repetitions[available],
            "scheduled_event_index": available,
        }
    )
    observed_digits = scheduled_digits[available]
    return PreparedGTNExperiment(
        experiment_name=experiment_dir.name,
        subject_id=subject_id,
        X=np.asarray(result.data, dtype=np.float32),
        y=(observed_digits == int(gtn.thought_number)).astype(np.int64),
        metadata=metadata,
        timeline=timeline,
        channel_names=tuple(result.channel_names),
        channel_positions_m=np.asarray(result.channel_positions_m, dtype=np.float32),
        channel_mask=np.asarray(result.channel_mask, dtype=bool),
        source_sample_rate_hz=float(gtn.metadata["sfreq"]),
        source_reference=str(gtn.metadata["source_reference"]),
    )


def _preprocess_gtn_experiment(
    experiment_dir: Path,
    *,
    preprocessing: PreprocessingSpec,
) -> PreparedGTNExperiment:
    gtn = read_gtn_experiment(experiment_dir)
    result = preprocess(
        gtn.raw,
        gtn.events,
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
        channels=GTN_CHANNELS,
    )
    return _prepare_gtn_experiment(experiment_dir, gtn, result)


def build_gtn_epoch_dataset(
    root: str | Path,
    *,
    preprocessing: PreprocessingSpec = GTN_LMBC_PREPROCESSING,
    experiment_dirs: Sequence[str | Path] | None = None,
    allow_skipped: bool = False,
) -> EpochDataset:
    """Build a full GTN 9-choice dataset with an explicit skipped-source ledger.

    ``allow_skipped`` is intentionally opt-in.  Some public GTN directories
    have missing thought metadata or duplicate NIX participant identities;
    accepting either condition changes the cohort and is recorded in provenance.
    """

    preprocessing.validate()
    source_dirs = (
        tuple(Path(path) for path in experiment_dirs)
        if experiment_dirs is not None
        else gtn_experiment_dirs(root)
    )
    if not source_dirs:
        raise ValueError("At least one GTN experiment directory is required.")
    prepared: list[PreparedGTNExperiment] = []
    skipped: list[str] = []
    seen_subjects: dict[str, str] = {}
    for experiment_dir in source_dirs:
        try:
            item = _preprocess_gtn_experiment(experiment_dir, preprocessing=preprocessing)
            previous = seen_subjects.get(item.subject_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate GTN subject identity {item.subject_id!r}; first seen in {previous}."
                )
            seen_subjects[item.subject_id] = item.experiment_name
            prepared.append(item)
        except Exception as exc:  # noqa: BLE001 - source defects become explicit cohort records.
            message = f"{experiment_dir.name}: {type(exc).__name__}: {exc}"
            if not allow_skipped:
                raise ValueError(f"GTN source preparation failed: {message}") from exc
            skipped.append(message)
    if not prepared:
        raise ValueError("No GTN experiments produced usable epochs.")

    reference = prepared[0]
    for item in prepared[1:]:
        if item.channel_names != reference.channel_names:
            raise ValueError("GTN experiments disagree on canonical channel names.")
        if not np.array_equal(item.channel_mask, reference.channel_mask):
            raise ValueError("GTN experiments disagree on channel availability.")
        if not np.allclose(item.channel_positions_m, reference.channel_positions_m):
            raise ValueError("GTN experiments disagree on channel positions.")
    dataset = EpochDataset(
        name="GTN-LMBC",
        X=np.concatenate([item.X for item in prepared], axis=0),
        y=np.concatenate([item.y for item in prepared], axis=0),
        subject_ids=np.concatenate(
            [np.repeat(item.subject_id, len(item.X)) for item in prepared], axis=0
        ),
        channel_names=reference.channel_names,
        channel_positions_m=reference.channel_positions_m,
        channel_mask=reference.channel_mask,
        preprocessing=preprocessing,
        event_timeline=concatenate_event_timelines([item.timeline for item in prepared]),
        metadata=pd.concat([item.metadata for item in prepared], ignore_index=True),
        provenance={
            "source": "gtn_nix",
            "source_root": str(Path(root)),
            "source_experiments_requested": [str(path) for path in source_dirs],
            "source_experiments_used": [item.experiment_name for item in prepared],
            "skipped_sources": skipped,
            "candidate_vocabulary": list(range(1, 10)),
            "source_sample_rate_hz": {
                item.experiment_name: item.source_sample_rate_hz for item in prepared
            },
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "source_reference": "not_uniformly_recorded_in_gtn_source_metadata",
            "source_reference_by_experiment": {
                item.experiment_name: item.source_reference for item in prepared
            },
            "signal_unit": preprocessing.signal_unit,
            "artifact_rejection": {"method": "fold_local_ptp_cv"},
        },
    )
    dataset.validate(require_labels=True)
    if not dataset.event_timeline.supports_full_candidate_chain:
        raise AssertionError("GTN preparation must produce a complete 9-choice candidate chain.")
    return dataset
