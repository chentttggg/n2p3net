"""Generic MOABB P300 adapter producing the universal EpochDataset contract."""

from __future__ import annotations

from collections.abc import Sequence

import mne
import moabb
import numpy as np

from data.channel import DEFAULT_MONTAGE, build_channel_identity
from data.epochs import NEURAL_RIDE_V8_PREPROCESSING, EpochDataset, PreprocessingSpec
from data.events import observed_only_timeline, selection_group_id


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
    preprocessing: PreprocessingSpec = NEURAL_RIDE_V8_PREPROCESSING,
    target_label: str = "Target",
) -> EpochDataset:
    """Download/process an installed MOABB P300 dataset without dataset-specific branches."""

    from moabb.paradigms import P300

    preprocessing.validate()
    dataset = resolve_moabb_dataset(dataset_class)
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
        resample=preprocessing.sfreq,
        channels=None if channels is None else list(channels),
    )
    epochs, raw_labels, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=selected_subjects,
        return_epochs=True,
    )
    X = epochs.get_data()
    raw_labels = np.asarray(raw_labels)
    if len(X) != len(raw_labels) or len(metadata) != len(X):
        raise ValueError("MOABB epochs, labels, and metadata are not row-aligned.")
    if X.shape[2] < preprocessing.n_times:
        raise ValueError(
            f"MOABB returned {X.shape[2]} time samples; {preprocessing.n_times} are required."
        )
    X = X[:, :, : preprocessing.n_times].astype(np.float32, copy=False)
    keep = np.ones(len(X), dtype=bool)
    if preprocessing.reject_threshold_v is not None:
        keep = np.max(np.abs(X), axis=(1, 2)) <= preprocessing.reject_threshold_v
        if not bool(keep.any()):
            raise ValueError("Artifact rejection removed every MOABB epoch.")
        X = X[keep]
        raw_labels = raw_labels[keep]
        metadata = metadata.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    else:
        metadata = metadata.reset_index(drop=True).copy()
    channel_names = tuple(str(name) for name in epochs.ch_names)
    identity = build_channel_identity(
        channel_names,
        channel_mask=np.ones(len(channel_names), dtype=bool),
        montage=montage,
        allow_missing_positions=False,
    )
    if "subject" not in metadata:
        raise ValueError("MOABB metadata does not contain a subject column required for LOSO.")
    metadata["acquisition_time_s"] = np.asarray(epochs.events[keep, 0], dtype=float) / float(
        epochs.info["sfreq"]
    )
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
        stimulus_ids=np.asarray(epochs.events[keep, 2], dtype=np.int64),
        onset_times_s=metadata["acquisition_time_s"].to_numpy(dtype=float),
        group_ids=groups,
        online_causal=False,
        timing_source="moabb_observed_epochs_only;preprocessing_not_online_causal",
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
            "montage": montage,
            "coordinate_registration": identity.registration.record(),
            "right_endpoint": "exclusive_crop",
            "continuous_baseline_correction": None,
            "artifact_rejection": {
                "threshold_v": preprocessing.reject_threshold_v,
                "n_before": int(len(keep)),
                "n_rejected": int((~keep).sum()),
            },
        },
    )
    result.validate(require_labels=True)
    return result
