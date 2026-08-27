"""Generic MOABB P300 adapter producing the universal EpochDataset contract."""

from __future__ import annotations

from collections.abc import Sequence

import mne
import moabb
import numpy as np
import pandas as pd

from data.channel import DEFAULT_MONTAGE, build_channel_identity
from data.epochs import P300_PERFORMANCE_PREPROCESSING, EpochDataset, PreprocessingSpec
from data.events import observed_only_timeline, selection_group_id
from data.preprocess import apply_trial_baseline

CANDIDATE_METADATA_ALIASES = (
    "candidate_id",
    "stimulus_candidate_id",
    "command_id",
)
TARGET_CANDIDATE_METADATA_ALIASES = (
    "target_candidate_id",
    "selected_candidate_id",
    "intended_candidate_id",
)
REPETITION_METADATA_ALIASES = (
    "repetition_index",
    "repetition_id",
    "sequence_index",
)


def _optional_metadata_column(
    metadata: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    integer: bool = False,
) -> np.ndarray | None:
    present = [name for name in aliases if name in metadata]
    if not present:
        return None
    reference = metadata[present[0]].to_numpy()
    for name in present[1:]:
        if not np.array_equal(reference.astype(str), metadata[name].to_numpy().astype(str)):
            raise ValueError(f"Conflicting metadata aliases {present}.")
    if integer:
        if not np.issubdtype(reference.dtype, np.integer) or np.issubdtype(
            reference.dtype, np.bool_
        ):
            raise ValueError(f"Metadata column {present[0]!r} must contain integers.")
        return reference.astype(np.int64, copy=False)
    values = reference.astype(str)
    if np.any(np.char.strip(values) == ""):
        raise ValueError(f"Metadata column {present[0]!r} contains empty identifiers.")
    return values


def resolve_moabb_dataset(dataset_class: str):
    """Resolve any installed MOABB dataset class by public class name."""

    import moabb.datasets

    dataset_type = getattr(moabb.datasets, dataset_class, None)
    if dataset_type is None or not isinstance(dataset_type, type):
        available = sorted(
            name
            for name, value in vars(moabb.datasets).items()
            if isinstance(value, type) and not name.startswith("_")
        )
        raise ValueError(
            f"Unknown MOABB dataset class {dataset_class!r}. Installed public classes include "
            f"{available}."
        )
    return dataset_type()


def encode_binary_labels(values: np.ndarray, *, target_label: str = "Target") -> np.ndarray:
    labels = np.asarray(values)
    if np.issubdtype(labels.dtype, np.number):
        unique = set(np.unique(labels).tolist())
        if unique <= {0, 1}:
            return labels.astype(np.int64)
    normalized = np.char.lower(np.char.strip(labels.astype(str)))
    target = target_label.strip().lower()
    unique = set(normalized.tolist())
    if target not in unique or len(unique) != 2:
        raise ValueError(
            f"Binary P300 labels must contain target={target_label!r} and one non-target class; "
            f"got {sorted(unique)}."
        )
    return (normalized == target).astype(np.int64)


def prepare_moabb_p300(
    dataset_class: str,
    *,
    subjects: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    montage: str = DEFAULT_MONTAGE,
    preprocessing: PreprocessingSpec = P300_PERFORMANCE_PREPROCESSING,
    target_label: str = "Target",
) -> EpochDataset:
    """Download/process an installed MOABB P300 dataset without dataset-specific branches."""

    from moabb.paradigms import P300

    preprocessing.validate()
    if preprocessing.reject_threshold_v is not None:
        raise ValueError(
            "Fixed absolute-voltage artifact rejection is retired; set reject_threshold_v=None "
            "and use fold-local artifact QC during evaluation."
        )
    dataset = resolve_moabb_dataset(dataset_class)
    acquisition = getattr(getattr(dataset, "metadata", None), "acquisition", None)
    source_reference = str(getattr(acquisition, "reference", None) or "unspecified")
    source_sample_rate = getattr(acquisition, "sampling_rate", None)
    if source_reference == "unspecified" or source_sample_rate is None:
        raise ValueError(
            f"MOABB dataset {dataset_class} lacks acquisition reference/sample-rate metadata."
        )
    selected_subjects = list(subjects) if subjects is not None else list(dataset.subject_list)
    if not selected_subjects:
        raise ValueError("At least one MOABB subject must be selected.")
    if len(set(selected_subjects)) != len(selected_subjects):
        raise ValueError("MOABB subject selection must not contain duplicates.")
    unknown_subjects = set(selected_subjects) - set(dataset.subject_list)
    if unknown_subjects:
        raise ValueError(f"Unknown subjects for {dataset_class}: {sorted(unknown_subjects)}.")
    paradigm = P300(
        fmin=preprocessing.l_freq,
        fmax=preprocessing.h_freq,
        tmin=preprocessing.tmin_ms / 1000.0,
        tmax=preprocessing.tmax_ms / 1000.0,
        baseline=None,
        resample=None,
        channels=None if channels is None else list(channels),
    )
    epochs, raw_labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=selected_subjects,
        return_epochs=True,
    )
    expected_tmin = preprocessing.tmin_ms / 1000.0
    if not np.isclose(float(epochs.times[0]), expected_tmin, atol=0.5 / preprocessing.sfreq):
        raise ValueError(
            "MOABB epoch origin does not match the declared stimulus-relative time contract: "
            f"{float(epochs.times[0]):g}s vs {expected_tmin:g}s."
        )
    source_epoch_sfreq = float(epochs.info["sfreq"])
    source_epoch_events = np.asarray(epochs.events, dtype=np.int64).copy()
    if not np.isclose(source_epoch_sfreq, preprocessing.sfreq):
        epochs.resample(preprocessing.sfreq, verbose=False)
    X = epochs.get_data()
    raw_labels = np.asarray(raw_labels)
    if len(X) != len(raw_labels) or len(metadata) != len(X):
        raise ValueError("MOABB epochs, labels, and metadata are not row-aligned.")
    if "subject" not in metadata:
        raise ValueError("MOABB metadata does not contain a subject column required for LOSO.")
    requested_subject_keys = {str(subject) for subject in selected_subjects}
    returned_subject_keys = set(metadata["subject"].astype(str))
    missing_before_rejection = requested_subject_keys - returned_subject_keys
    if missing_before_rejection:
        raise ValueError(
            "MOABB returned no epochs for selected subjects "
            f"{sorted(missing_before_rejection)}."
        )
    if X.shape[2] < preprocessing.n_times:
        raise ValueError(
            f"MOABB returned {X.shape[2]} time samples; {preprocessing.n_times} are required."
        )
    X = X[:, :, : preprocessing.n_times].astype(np.float32, copy=False)
    X = apply_trial_baseline(
        X,
        sfreq=preprocessing.sfreq,
        tmin=preprocessing.tmin_ms / 1000.0,
        baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale,
    )
    keep = np.ones(len(X), dtype=bool)
    metadata = metadata.reset_index(drop=True).copy()
    channel_names = tuple(str(name) for name in epochs.ch_names)
    identity = build_channel_identity(
        channel_names,
        channel_mask=np.ones(len(channel_names), dtype=bool),
        montage=montage,
        allow_missing_positions=False,
    )
    metadata["acquisition_time_s"] = source_epoch_events[keep, 0].astype(float) / source_epoch_sfreq
    subjects_array = metadata["subject"].astype(str).to_numpy()
    sessions = (
        metadata["session"].astype(str).to_numpy()
        if "session" in metadata
        else [""] * len(metadata)
    )
    runs = metadata["run"].astype(str).to_numpy() if "run" in metadata else [""] * len(metadata)
    selections = (
        metadata["selection_id"].astype(str).to_numpy()
        if "selection_id" in metadata
        else subjects_array
    )
    groups = np.asarray(
        [
            selection_group_id(dataset_class, subject, session, run, selection)
            for subject, session, run, selection in zip(
                subjects_array, sessions, runs, selections, strict=True
            )
        ]
    )
    timeline = observed_only_timeline(
        dataset_id=dataset_class,
        subject_ids=subjects_array,
        stimulus_ids=source_epoch_events[keep, 2],
        onset_times_s=metadata["acquisition_time_s"].to_numpy(dtype=float),
        group_ids=groups,
        selection_ids=np.asarray(selections, dtype=str),
        session_ids=np.asarray(sessions, dtype=str),
        run_ids=np.asarray(runs, dtype=str),
        candidate_ids=_optional_metadata_column(metadata, CANDIDATE_METADATA_ALIASES),
        target_candidate_ids=_optional_metadata_column(
            metadata, TARGET_CANDIDATE_METADATA_ALIASES
        ),
        repetition_indices=_optional_metadata_column(
            metadata, REPETITION_METADATA_ALIASES, integer=True
        ),
        online_causal=False,
        timing_source=(
            "moabb_post_resample_epoch_events;observed_epochs_only;"
            "preprocessing_not_online_causal"
        ),
    )
    result = EpochDataset(
        name=dataset_class,
        X=X,
        y=encode_binary_labels(raw_labels, target_label=target_label),
        subject_ids=subjects_array,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=identity.mask,
        preprocessing=preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "moabb_p300",
            "dataset_class": dataset_class,
            "subjects": selected_subjects,
            "moabb_version": moabb.__version__,
            "mne_version": mne.__version__,
            "source_sample_rate_hz": source_sample_rate,
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "source_reference": source_reference,
            "source_ground": getattr(acquisition, "ground", None),
            "event_time_basis": "post_resample_epoch_event_samples",
            "grouping_fidelity": "subject_session_run_only; source blocks are not exposed",
            "signal_unit": preprocessing.signal_unit,
            "filter": {
                "method": preprocessing.filter_method,
                "order": preprocessing.filter_order,
                "phase": preprocessing.filter_phase,
            },
            "resample_domain": preprocessing.resample_domain,
            "resample": {
                "method": preprocessing.resample_method,
                "npad": preprocessing.resample_npad,
                "window": preprocessing.resample_window,
                "pad": preprocessing.resample_pad,
            },
            "montage": montage,
            "coordinate_registration": identity.registration.record(),
            "right_endpoint": "exclusive_crop",
            "epoch_baseline": {
                "mode": preprocessing.baseline_mode,
                "window_ms": (
                    list(preprocessing.trial_reference_window_ms)
                    if preprocessing.trial_reference_window_ms is not None
                    else [preprocessing.tmin_ms, 0.0]
                ),
                "center": preprocessing.trial_reference_center,
                "scale": (
                    "std"
                    if preprocessing.baseline_mode == "trial"
                    else preprocessing.trial_reference_scale
                ),
            },
            "artifact_rejection": {"method": "fold_local_ptp_cv", "n_before": int(len(keep))},
        },
    )
    result.validate(require_labels=True)
    return result
