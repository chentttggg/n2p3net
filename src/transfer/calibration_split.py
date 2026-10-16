"""Shared chronological calibration-only validation for target experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from data.epochs import EpochDataset
from research.contracts import semantic_sha256
from transfer.binary import binary_time_split


@dataclass(frozen=True)
class CalibrationInnerSplit:
    training_mask: np.ndarray
    validation_mask: np.ndarray
    purged_mask: np.ndarray
    mode: str
    reason: str | None
    training_recordings: tuple[str, ...] = ()
    validation_recordings: tuple[str, ...] = ()

    @property
    def legal(self) -> bool:
        return self.reason is None

    def record(self) -> dict[str, object]:
        return {
            "mode": self.mode, "legal": self.legal, "reason": self.reason,
            "training_sessions": list(self.training_recordings),
            "validation_sessions": list(self.validation_recordings),
            **{f"{name}_rows": int(mask.sum()) for name, mask in (
                ("training", self.training_mask), ("validation", self.validation_mask),
                ("purged", self.purged_mask),
            )},
            "split_sha256": semantic_sha256({
                "train": np.flatnonzero(self.training_mask).tolist(),
                "validation": np.flatnonzero(self.validation_mask).tolist(),
                "purged": np.flatnonzero(self.purged_mask).tolist(),
            }),
        }


def calibration_inner_split(
    labels: np.ndarray, onset_times: np.ndarray, available_times: np.ndarray,
    recording_ids: np.ndarray, *, tmin_s: float,
    recording_order: Sequence[str] | None = None,
    recording_start_times: Mapping[str, float] | None = None,
    label_available_times: Mapping[str, float] | None = None,
    calibration_mask: np.ndarray | None = None,
) -> CalibrationInnerSplit:
    """Hold out the last fifth of recordings, or purge one recording's suffix.

    Invalid scientific splits are explicit non-selection outcomes; malformed
    array contracts raise. No information outside calibration enters a decision.
    """
    labels = np.asarray(labels)
    onset = np.asarray(onset_times, dtype=float)
    available = np.asarray(available_times, dtype=float)
    recordings = np.asarray(recording_ids).astype(str)
    if labels.ndim != 1 or any(value.shape != labels.shape for value in (onset, available, recordings)):
        raise ValueError("calibration labels, times and recording identities must align")
    allowed = np.ones(len(labels), dtype=bool) if calibration_mask is None else np.asarray(calibration_mask)
    if allowed.dtype != bool or allowed.shape != labels.shape:
        raise ValueError("calibration_mask must be an aligned boolean vector")
    train, validation, purged = (np.zeros(len(labels), dtype=bool) for _ in range(3))
    rows = np.flatnonzero(allowed)
    ids = tuple(dict.fromkeys(recordings[rows]))
    mode = "complete_recording_chronological_80_20" if len(ids) > 1 else "within_recording_purged_80_20"
    training_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()

    def result(reason=None):
        return CalibrationInnerSplit(train, validation, purged, mode, reason, training_ids, validation_ids)

    if len(rows) < 4:
        return result("too few authorized calibration rows for a legal inner split")
    if not np.isfinite(onset[rows]).all() or not np.isfinite(available[rows]).all():
        return result("calibration event times must be finite")
    for recording in ids:
        selected = rows[recordings[rows] == recording]
        if np.any(np.diff(onset[selected]) <= 0) or np.any(available[selected] < onset[selected]):
            return result("one recording requires increasing onsets and evidence available after onset")
    if len(ids) > 1:
        if any(set(np.unique(labels[allowed & (recordings == recording)])) != {0, 1} for recording in ids):
            return result("every complete recording must retain both binary classes")
        if recording_order is not None:
            order = tuple(str(item) for item in recording_order if str(item) in ids)
            if len(order) != len(set(order)) or set(order) != set(ids):
                raise ValueError("recording_order must identify every calibration recording exactly once")
        elif recording_start_times is not None and all(item in recording_start_times for item in ids):
            starts = [float(recording_start_times[item]) for item in ids]
            if not np.isfinite(starts).all() or len(set(starts)) != len(ids):
                return result("recording timestamps must establish an unambiguous chronology")
            order = tuple(sorted(ids, key=lambda item: recording_start_times[item]))
        else:
            return result("multiple recordings require declared chronology; identifiers are not clocks")
        count = max(1, int(np.ceil(0.2 * len(order))))
        training_ids, validation_ids = order[:-count], order[-count:]
        train[:] = allowed & np.isin(recordings, training_ids)
        validation[:] = allowed & np.isin(recordings, validation_ids)
        if label_available_times is not None and recording_start_times is not None:
            if max(label_available_times[item] for item in training_ids) > min(
                recording_start_times[item] for item in validation_ids
            ):
                return result("inner-training labels were unavailable before validation recording started")
    else:
        count = int(np.floor(0.8 * len(rows)))
        try:
            split = binary_time_split(
                onset[rows], available[rows], calibration_trials=count,
                maximum_calibration_trials=count, tmin_s=tmin_s,
            )
        except ValueError as error:
            return result(str(error))
        train[rows[split.calibration_rows]] = True
        validation[rows[split.test_rows]] = True
        purged[rows[split.purged_rows]] = True
        training_ids = validation_ids = ids
    if any(set(np.unique(labels[mask])) != {0, 1} for mask in (train, validation)):
        return result("inner train and validation must each retain both binary classes")
    return result()


def dataset_calibration_inner_split(
    dataset: EpochDataset, calibration_mask: np.ndarray | None = None, *,
    recording_order: Sequence[str] | Mapping | None = None,
) -> CalibrationInnerSplit:
    """Align production event chronology to epoch rows without inventing sessions."""
    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=int)
    retained = evidence >= 0
    if not np.array_equal(np.sort(evidence[retained]), np.arange(dataset.n_epochs)):
        raise ValueError("scheduled evidence must map one-to-one to dataset rows")
    onset = np.full(dataset.n_epochs, np.nan)
    available = np.full(dataset.n_epochs, np.nan)
    recordings = np.full(dataset.n_epochs, "", dtype=object)
    onset[evidence[retained]] = np.asarray(timeline.onset_times_s)[retained]
    available[evidence[retained]] = np.asarray(timeline.evidence_available_times_s)[retained]
    sessions = np.asarray(timeline.session_ids).astype(str)
    runs = np.asarray(timeline.run_ids).astype(str)
    allowed = np.ones(dataset.n_epochs, dtype=bool) if calibration_mask is None else np.asarray(calibration_mask)
    if allowed.dtype != bool or allowed.shape != (dataset.n_epochs,):
        raise ValueError("calibration_mask must be an aligned boolean vector")
    if isinstance(recording_order, Mapping):
        recording_order = dataset_calibration_recording_order(dataset, allowed, recording_order)
    row_sessions = np.full(dataset.n_epochs, "", dtype=object)
    row_sessions[evidence[retained]] = sessions[retained]
    row_runs = np.full(dataset.n_epochs, "", dtype=object)
    row_runs[evidence[retained]] = runs[retained]
    # A single declared session with several runs retains real recording IDs.
    if len(np.unique(row_sessions[allowed])) == 1 and len(np.unique(row_runs[allowed])) > 1:
        recordings[:] = row_runs
    else:
        recordings[:] = row_sessions
    if dataset.preprocessing.filter_phase == "zero":
        reason = None
        for recording in np.unique(recordings[allowed]):
            if not allowed[recordings == recording].all():
                reason = "zero-phase calibration cannot select a prefix of an already filtered recording"
        if len(np.unique(recordings[allowed])) < 2:
            reason = "zero-phase inner validation requires independent complete recordings"
        if reason is not None:
            empty = np.zeros(dataset.n_epochs, dtype=bool)
            return CalibrationInnerSplit(empty, empty.copy(), empty.copy(),
                "complete_recording_zero_phase", reason)
    starts, labels_ready = {}, {}
    for recording in np.unique(recordings[allowed]):
        metadata = dataset.metadata.loc[allowed & (recordings == recording)]
        for field, target in (("session_start_timestamp_s", starts),
                              ("calibration_available_timestamp_s", labels_ready)):
            if field in metadata:
                values = metadata[field].to_numpy(dtype=float)
                if len(values) and np.isfinite(values).all() and len(np.unique(values)) == 1:
                    target[str(recording)] = float(values[0])
    return calibration_inner_split(
        dataset.y, onset, available, recordings, tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        calibration_mask=allowed, recording_order=recording_order,
        recording_start_times=starts or None,
        label_available_times=labels_ready if len(labels_ready) == len(starts) else None,
    )


def dataset_calibration_recording_order(
    dataset: EpochDataset, calibration_mask: np.ndarray, protocol: Mapping,
) -> tuple[str, ...] | None:
    """Resolve only a protocol-declared order for the authorized physical axis."""
    allowed = np.asarray(calibration_mask)
    if allowed.dtype != bool or allowed.shape != (dataset.n_epochs,):
        raise ValueError("calibration_mask must be an aligned boolean vector")
    timeline = dataset.event_timeline
    evidence = np.asarray(timeline.evidence_indices, dtype=int)
    retained = evidence >= 0
    if not np.array_equal(np.sort(evidence[retained]), np.arange(dataset.n_epochs)):
        raise ValueError("scheduled evidence must map one-to-one to dataset rows")
    scoped = allowed[evidence[retained]]
    sessions = np.asarray(timeline.session_ids).astype(str)[retained][scoped]
    runs = np.asarray(timeline.run_ids).astype(str)[retained][scoped]
    field = "run_order" if len(np.unique(sessions)) == 1 and len(np.unique(runs)) > 1 else "session_order"
    order = protocol.get(field)
    if order is None:
        return None
    if isinstance(order, str) or not isinstance(order, Sequence):
        raise ValueError(f"{field} must be an explicit sequence of recording identities")
    return tuple(str(value) for value in order)
