"""Versioned, format-neutral epoched EEG dataset contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from io import StringIO
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.channel import canonical_channel_name
from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.events import (
    EVENT_TIMELINE_SCHEMA,
    LEGACY_EVENT_TIMELINE_SCHEMAS,
    ScheduledEventTimeline,
    concatenate_event_timelines,
)

EPOCH_DATASET_SCHEMA = "n2p3net_epoch_dataset/3"
LEGACY_EPOCH_DATASET_SCHEMAS = frozenset({"n2p3net_epoch_dataset/2"})
DEFAULT_SAMPLE_RATE_HZ = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz


@dataclass(frozen=True)
class PreprocessingSpec:
    name: str = DEFAULT_P300_DATA_CONTRACT.name
    sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz
    l_freq: float | None = DEFAULT_P300_DATA_CONTRACT.l_freq
    h_freq: float | None = DEFAULT_P300_DATA_CONTRACT.h_freq
    tmin_ms: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms
    tmax_ms: float = DEFAULT_P300_DATA_CONTRACT.tmax_ms
    n_times: int = DEFAULT_P300_DATA_CONTRACT.n_times
    baseline_mode: str = DEFAULT_P300_DATA_CONTRACT.baseline_mode
    trial_reference_window_ms: tuple[float, float] | None = None
    trial_reference_center: str = "mean"
    trial_reference_scale: str = "none"
    # Retained only to read historical cache records. New ingress rejects a
    # fixed threshold because artifact quality is fitted inside each outer fold.
    reject_threshold_v: float | None = None

    def __post_init__(self) -> None:
        # JSON round-trips tuples as lists; normalize valid windows so the
        # physical contract compares identically before and after saving.
        window = self.trial_reference_window_ms
        if isinstance(window, (tuple, list)) and len(window) == 2:
            object.__setattr__(
                self,
                "trial_reference_window_ms",
                (window[0], window[1]),
            )

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Preprocessing name must be non-empty.")
        for field_name in ("sfreq", "tmin_ms", "tmax_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Preprocessing {field_name} must be numeric.")
        if not np.isfinite((self.sfreq, self.tmin_ms, self.tmax_ms)).all():
            raise ValueError("Preprocessing sfreq/tmin_ms/tmax_ms must be finite.")
        if isinstance(self.n_times, bool) or not isinstance(self.n_times, (int, np.integer)):
            raise ValueError("Preprocessing n_times must be an integer.")
        if self.sfreq <= 0 or self.n_times <= 0:
            raise ValueError("Preprocessing sfreq and n_times must be positive.")
        if not self.tmin_ms < self.tmax_ms:
            raise ValueError("Preprocessing tmin_ms must be smaller than tmax_ms.")
        expected = (self.tmax_ms - self.tmin_ms) * self.sfreq / 1000.0
        expected_n_times = int(np.floor(expected + 1e-9))
        if self.n_times != expected_n_times:
            raise ValueError(
                "Physical time axis with an exclusive right endpoint requires "
                f"floor({expected:g})={expected_n_times} samples, not n_times={self.n_times}."
            )
        if self.baseline_mode not in {"trial", "mean_only", "none", "trial_reference"}:
            raise ValueError(
                "baseline_mode must be trial, mean_only, none, or trial_reference."
            )
        if self.baseline_mode == "trial" and self.tmin_ms >= 0:
            raise ValueError("trial baseline standardization requires pre-stimulus samples.")
        if self.trial_reference_center not in {"mean", "median"}:
            raise ValueError("trial_reference_center must be mean or median.")
        if self.trial_reference_scale not in {"none", "std", "mad"}:
            raise ValueError("trial_reference_scale must be none, std, or mad.")
        if self.baseline_mode == "trial_reference":
            window = self.trial_reference_window_ms
            if window is None or len(window) != 2:
                raise ValueError(
                    "trial_reference mode requires trial_reference_window_ms=(start,end)."
                )
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in window):
                raise ValueError("trial_reference_window_ms values must be numeric.")
            start_ms, end_ms = (float(window[0]), float(window[1]))
            if not np.isfinite([start_ms, end_ms]).all() or start_ms >= end_ms:
                raise ValueError("trial_reference_window_ms must be a finite increasing interval.")
            if start_ms < self.tmin_ms or end_ms > self.tmax_ms:
                raise ValueError(
                    "trial_reference_window_ms must lie inside the physical epoch time axis."
                )
            if (end_ms - start_ms) * self.sfreq / 1000.0 < 2.0:
                raise ValueError("trial_reference window must contain at least two samples.")
        for field_name in ("l_freq", "h_freq", "reject_threshold_v"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real)
            ):
                raise ValueError(f"{field_name} must be numeric or None.")
        if self.l_freq is not None and (not np.isfinite(self.l_freq) or self.l_freq <= 0):
            raise ValueError("l_freq must be finite and positive or None.")
        if self.h_freq is not None and (not np.isfinite(self.h_freq) or self.h_freq <= 0):
            raise ValueError("h_freq must be finite and positive or None.")
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError("l_freq must be smaller than h_freq.")
        nyquist = self.sfreq / 2.0
        if self.l_freq is not None and self.l_freq >= nyquist:
            raise ValueError("l_freq must be below the Nyquist frequency.")
        if self.h_freq is not None and self.h_freq >= nyquist:
            raise ValueError("h_freq must be below the Nyquist frequency.")
        if self.reject_threshold_v is not None and (
            not np.isfinite(self.reject_threshold_v) or self.reject_threshold_v <= 0.0
        ):
            raise ValueError("reject_threshold_v must be finite and positive or None.")


P300_PERFORMANCE_PREPROCESSING = PreprocessingSpec()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


@dataclass
class EpochDataset:
    """A model-ready EEG epoch tensor with a complete physical/provenance contract."""

    name: str
    X: np.ndarray
    y: np.ndarray | None
    subject_ids: np.ndarray
    channel_names: tuple[str, ...]
    channel_positions_m: np.ndarray
    channel_mask: np.ndarray
    preprocessing: PreprocessingSpec
    event_timeline: ScheduledEventTimeline
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict[str, Any] = field(default_factory=dict)
    trial_channel_mask: np.ndarray | None = None

    def validate(self, *, require_labels: bool = False) -> None:
        self.preprocessing.validate()
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EpochDataset.name must be a non-empty string.")
        X = np.asarray(self.X)
        if X.ndim != 3:
            raise ValueError(f"EpochDataset.X must be (N,C,T), got {X.shape}.")
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("EpochDataset.X must use a floating-point dtype.")
        n_epochs, n_channels, n_times = X.shape
        if n_epochs == 0 or n_channels == 0:
            raise ValueError("EpochDataset.X must contain at least one epoch and channel.")
        if n_times != self.preprocessing.n_times:
            raise ValueError(
                f"Dataset stores {n_times} samples but preprocessing requires "
                f"{self.preprocessing.n_times}."
            )
        canonical_names = tuple(canonical_channel_name(name) for name in self.channel_names)
        if len(canonical_names) != n_channels or len(set(canonical_names)) != n_channels:
            raise ValueError(
                "channel_names must identify unique physical electrodes and match X.shape[1]."
            )
        positions = np.asarray(self.channel_positions_m)
        if positions.shape != (n_channels, 3):
            raise ValueError("channel_positions_m must be (C,3).")
        if not np.issubdtype(positions.dtype, np.number):
            raise ValueError("channel_positions_m must be numeric.")
        channel_mask = np.asarray(self.channel_mask)
        if channel_mask.dtype != np.dtype(bool):
            raise ValueError("channel_mask must have boolean dtype.")
        if channel_mask.shape != (n_channels,) or not bool(channel_mask.any()):
            raise ValueError("channel_mask must identify at least one channel.")
        if self.trial_channel_mask is not None:
            trial_mask = np.asarray(self.trial_channel_mask)
            if trial_mask.dtype != np.dtype(bool):
                raise ValueError("trial_channel_mask must have boolean dtype.")
            if trial_mask.shape != (n_epochs, n_channels):
                raise ValueError("trial_channel_mask must be (N,C).")
            if not bool(trial_mask.any(axis=1).all()):
                raise ValueError("Every trial must retain at least one observed channel.")
            if np.any(trial_mask & ~channel_mask[None]):
                raise ValueError("trial_channel_mask cannot enable a permanently absent channel.")
        observed_positions = positions[channel_mask]
        if not np.isfinite(observed_positions).all() or np.any(
            np.linalg.norm(observed_positions, axis=1) <= 0.0
        ):
            raise ValueError("Observed channels require finite, non-zero physical coordinates.")
        if np.any(np.linalg.norm(observed_positions, axis=1) > 0.5):
            raise ValueError(
                "channel_positions_m must be registered head-frame coordinates in metres; "
                "values above 0.5 m indicate an invalid unit/frame contract."
            )
        subject_ids = np.asarray(self.subject_ids)
        if subject_ids.ndim != 1 or len(subject_ids) != n_epochs:
            raise ValueError("subject_ids must contain one value per epoch.")
        if np.any(np.char.strip(subject_ids.astype(str)) == ""):
            raise ValueError("subject_ids must be non-empty.")
        self.event_timeline.validate(n_epochs=n_epochs)
        event_subjects = np.asarray(self.event_timeline.subject_ids).astype(str)
        event_evidence = np.asarray(self.event_timeline.evidence_indices, dtype=np.int64)
        available = event_evidence >= 0
        aligned_subjects = np.empty(n_epochs, dtype=object)
        aligned_subjects[event_evidence[available]] = event_subjects[available]
        if not np.array_equal(
            aligned_subjects.astype(str), np.asarray(self.subject_ids).astype(str)
        ):
            raise ValueError("Scheduled-event evidence mapping disagrees with subject_ids.")
        if self.y is not None:
            labels = np.asarray(self.y)
            if labels.ndim != 1 or len(labels) != n_epochs:
                raise ValueError("y must be a one-dimensional array with one label per epoch.")
            if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
                raise ValueError("y must use an integer dtype.")
            if self.event_timeline.has_candidate_sets:
                evidence = np.asarray(self.event_timeline.evidence_indices, dtype=np.int64)
                available = evidence >= 0
                expected = (
                    np.asarray(self.event_timeline.candidate_ids).astype(str)[available]
                    == np.asarray(self.event_timeline.target_candidate_ids).astype(str)[available]
                ).astype(np.int64)
                aligned_expected = np.empty(n_epochs, dtype=np.int64)
                aligned_expected[evidence[available]] = expected
                if not np.array_equal(labels.astype(np.int64), aligned_expected):
                    raise ValueError(
                        "Candidate metadata requires y == (candidate_id == target_candidate_id)."
                    )
        if require_labels and self.y is None:
            raise ValueError("This operation requires labels.")
        if not np.isfinite(X).all():
            raise ValueError(
                "EpochDataset.X must be finite; represent absent channels as zero + mask."
            )
        if np.any(~channel_mask) and np.any(X[:, ~channel_mask, :] != 0.0):
            raise ValueError("Absent channels must be exactly zero wherever channel_mask is false.")
        if self.trial_channel_mask is not None and np.any(
            X[~np.asarray(self.trial_channel_mask)] != 0.0
        ):
            raise ValueError("Trial-masked channels must be exactly zero.")
        if not isinstance(self.metadata, pd.DataFrame):
            raise ValueError("metadata must be a pandas DataFrame.")
        if not self.metadata.empty and len(self.metadata) != n_epochs:
            raise ValueError("metadata must have one row per epoch or be empty.")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dictionary.")

    @property
    def n_epochs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])

    def record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": EPOCH_DATASET_SCHEMA,
            "name": self.name,
            "shape": list(self.X.shape),
            "channel_names": list(self.channel_names),
            "channel_mask": self.channel_mask.astype(bool).tolist(),
            "has_trial_channel_mask": self.trial_channel_mask is not None,
            "mean_observed_channel_fraction": float(
                np.asarray(self.trial_channel_mask, dtype=bool).mean()
                if self.trial_channel_mask is not None
                else self.channel_mask.mean()
            ),
            "preprocessing": asdict(self.preprocessing),
            "n_subjects": int(len(np.unique(self.subject_ids))),
            "events": {
                "schema": EVENT_TIMELINE_SCHEMA,
                "n_scheduled": self.event_timeline.n_events,
                "n_available": self.event_timeline.n_available,
                "complete": self.event_timeline.complete,
                "online_causal": self.event_timeline.online_causal,
                "has_candidate_ids": self.event_timeline.has_candidate_ids,
                "has_candidate_sets": self.event_timeline.has_candidate_sets,
                "has_repetition_structure": self.event_timeline.has_repetition_structure,
                "supports_full_candidate_chain": (
                    self.event_timeline.supports_full_candidate_chain
                ),
                "fingerprint": self.event_timeline.fingerprint(),
            },
            "provenance": self.provenance,
        }


def save_epoch_dataset(
    path: str | Path,
    dataset: EpochDataset,
    *,
    compressed: bool = True,
) -> Path:
    """Persist an EpochDataset without pickle-dependent object arrays."""

    dataset.validate()
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("EpochDataset cache paths must end in .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dataset.metadata
    if metadata.empty:
        metadata = pd.DataFrame({"subject": dataset.subject_ids.astype(str)})
    payload = {
        "schema": np.asarray(EPOCH_DATASET_SCHEMA),
        "name": np.asarray(dataset.name),
        "X": np.asarray(dataset.X, dtype=np.float32),
        "subject_ids": np.asarray(dataset.subject_ids, dtype=str),
        "channel_names": np.asarray(dataset.channel_names, dtype=str),
        "channel_positions_m": np.asarray(dataset.channel_positions_m, dtype=np.float32),
        "channel_mask": np.asarray(dataset.channel_mask, dtype=bool),
        "trial_channel_mask": (
            np.asarray(dataset.trial_channel_mask, dtype=bool)
            if dataset.trial_channel_mask is not None
            else np.empty((0, dataset.n_channels), dtype=bool)
        ),
        "event_schema": np.asarray(EVENT_TIMELINE_SCHEMA),
        "event_ids": np.asarray(dataset.event_timeline.event_ids, dtype=str),
        "event_group_ids": np.asarray(dataset.event_timeline.group_ids, dtype=str),
        "event_subject_ids": np.asarray(dataset.event_timeline.subject_ids, dtype=str),
        "event_stimulus_ids": np.asarray(dataset.event_timeline.stimulus_ids, dtype=np.int64),
        "event_onset_samples": np.asarray(dataset.event_timeline.onset_samples, dtype=np.int64),
        "event_onset_times_s": np.asarray(dataset.event_timeline.onset_times_s, dtype=np.float64),
        "event_evidence_available_times_s": np.asarray(
            dataset.event_timeline.evidence_available_times_s, dtype=np.float64
        ),
        "event_evidence_indices": np.asarray(
            dataset.event_timeline.evidence_indices, dtype=np.int64
        ),
        "event_statuses": np.asarray(dataset.event_timeline.statuses, dtype=str),
        "event_status_details": np.asarray(dataset.event_timeline.status_details, dtype=str),
        "event_dataset_ids": np.asarray(dataset.event_timeline.dataset_ids, dtype=str),
        "event_session_ids": np.asarray(dataset.event_timeline.session_ids, dtype=str),
        "event_run_ids": np.asarray(dataset.event_timeline.run_ids, dtype=str),
        "event_selection_ids": np.asarray(dataset.event_timeline.selection_ids, dtype=str),
        "event_candidate_ids": np.asarray(dataset.event_timeline.candidate_ids, dtype=str),
        "event_target_candidate_ids": np.asarray(
            dataset.event_timeline.target_candidate_ids, dtype=str
        ),
        "event_repetition_indices": np.asarray(
            dataset.event_timeline.repetition_indices, dtype=np.int64
        ),
        "event_complete": np.asarray(dataset.event_timeline.complete),
        "event_online_causal": np.asarray(dataset.event_timeline.online_causal),
        "event_timing_source": np.asarray(dataset.event_timeline.timing_source),
        "preprocessing_json": np.asarray(
            json.dumps(asdict(dataset.preprocessing), sort_keys=True, default=_json_default)
        ),
        "metadata_json": np.asarray(metadata.to_json(orient="table", index=False)),
        "provenance_json": np.asarray(
            json.dumps(dataset.provenance, sort_keys=True, default=_json_default)
        ),
        "has_labels": np.asarray(dataset.y is not None),
        "y": (
            np.asarray(dataset.y, dtype=np.int64)
            if dataset.y is not None
            else np.empty(0, dtype=np.int64)
        ),
    }
    writer = np.savez_compressed if compressed else np.savez
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    writer(temporary, **payload)
    temporary.replace(path)
    return path


def load_epoch_dataset(
    path: str | Path,
    *,
    expected_preprocessing: PreprocessingSpec | None = None,
    require_labels: bool = False,
) -> EpochDataset:
    """Load and fail closed on schema or physical-contract mismatch."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "name",
            "X",
            "subject_ids",
            "channel_names",
            "channel_positions_m",
            "channel_mask",
            "preprocessing_json",
            "metadata_json",
            "provenance_json",
            "has_labels",
            "y",
            "event_schema",
            "event_ids",
            "event_group_ids",
            "event_subject_ids",
            "event_stimulus_ids",
            "event_onset_samples",
            "event_onset_times_s",
            "event_evidence_available_times_s",
            "event_evidence_indices",
            "event_statuses",
            "event_status_details",
            "event_dataset_ids",
            "event_session_ids",
            "event_run_ids",
            "event_selection_ids",
            "event_complete",
            "event_online_causal",
            "event_timing_source",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks EpochDataset fields {sorted(missing)}.")
        schema = str(np.asarray(archive["schema"]).item())
        if schema not in {EPOCH_DATASET_SCHEMA, *LEGACY_EPOCH_DATASET_SCHEMAS}:
            raise ValueError(f"Unsupported EpochDataset schema {schema!r} in {path}.")
        event_schema = str(np.asarray(archive["event_schema"]).item())
        if event_schema not in {EVENT_TIMELINE_SCHEMA, *LEGACY_EVENT_TIMELINE_SCHEMAS}:
            raise ValueError(f"Unsupported event timeline schema {event_schema!r} in {path}.")
        if schema == EPOCH_DATASET_SCHEMA and event_schema != EVENT_TIMELINE_SCHEMA:
            raise ValueError(
                f"EpochDataset schema {schema!r} requires event schema "
                f"{EVENT_TIMELINE_SCHEMA!r}, got {event_schema!r}."
            )
        has_current_candidate_contract = (
            schema == EPOCH_DATASET_SCHEMA and event_schema == EVENT_TIMELINE_SCHEMA
        )
        candidate_fields = {
            "event_candidate_ids",
            "event_target_candidate_ids",
            "event_repetition_indices",
        }
        if schema == EPOCH_DATASET_SCHEMA:
            missing_candidate_fields = candidate_fields - set(archive.files)
            if missing_candidate_fields:
                raise ValueError(
                    f"{path} lacks schema-v3 candidate fields "
                    f"{sorted(missing_candidate_fields)}."
                )
        integer_fields = (
            "event_stimulus_ids",
            "event_onset_samples",
            "event_evidence_indices",
        )
        if "event_repetition_indices" in archive.files:
            integer_fields = (*integer_fields, "event_repetition_indices")
        if bool(np.asarray(archive["has_labels"]).item()):
            integer_fields = (*integer_fields, "y")
        for field_name in integer_fields:
            dtype = np.asarray(archive[field_name]).dtype
            if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                raise ValueError(f"{path} field {field_name} must have an integer dtype.")
        for field_name in ("event_complete", "event_online_causal", "has_labels"):
            if np.asarray(archive[field_name]).dtype != np.dtype(bool):
                raise ValueError(f"{path} field {field_name} must be a strict boolean.")
        if not np.issubdtype(np.asarray(archive["X"]).dtype, np.floating):
            raise ValueError(f"{path} field X must have a floating-point dtype.")
        if not np.issubdtype(np.asarray(archive["channel_positions_m"]).dtype, np.floating):
            raise ValueError(f"{path} field channel_positions_m must be floating-point.")
        if np.asarray(archive["channel_mask"]).dtype != np.dtype(bool):
            raise ValueError(f"{path} field channel_mask must be a strict boolean array.")
        if "trial_channel_mask" in archive.files:
            trial_mask_field = np.asarray(archive["trial_channel_mask"])
            if trial_mask_field.dtype != np.dtype(bool):
                raise ValueError(
                    f"{path} field trial_channel_mask must be a strict boolean array."
                )
        preprocessing = PreprocessingSpec(
            **json.loads(str(np.asarray(archive["preprocessing_json"]).item()))
        )
        has_labels = bool(np.asarray(archive["has_labels"]).item())
        metadata_json = str(np.asarray(archive["metadata_json"]).item())
        dataset = EpochDataset(
            name=str(np.asarray(archive["name"]).item()),
            X=np.asarray(archive["X"], dtype=np.float32),
            y=np.asarray(archive["y"], dtype=np.int64) if has_labels else None,
            subject_ids=np.asarray(archive["subject_ids"], dtype=str),
            channel_names=tuple(str(name) for name in archive["channel_names"]),
            channel_positions_m=np.asarray(archive["channel_positions_m"], dtype=np.float32),
            channel_mask=np.asarray(archive["channel_mask"], dtype=bool),
            preprocessing=preprocessing,
            event_timeline=ScheduledEventTimeline(
                event_ids=np.asarray(archive["event_ids"], dtype=str),
                group_ids=np.asarray(archive["event_group_ids"], dtype=str),
                subject_ids=np.asarray(archive["event_subject_ids"], dtype=str),
                stimulus_ids=np.asarray(archive["event_stimulus_ids"], dtype=np.int64),
                onset_samples=np.asarray(archive["event_onset_samples"], dtype=np.int64),
                onset_times_s=np.asarray(archive["event_onset_times_s"], dtype=np.float64),
                evidence_available_times_s=np.asarray(
                    archive["event_evidence_available_times_s"], dtype=np.float64
                ),
                evidence_indices=np.asarray(archive["event_evidence_indices"], dtype=np.int64),
                statuses=np.asarray(archive["event_statuses"], dtype=str),
                status_details=np.asarray(archive["event_status_details"], dtype=str),
                dataset_ids=np.asarray(archive["event_dataset_ids"], dtype=str),
                session_ids=np.asarray(archive["event_session_ids"], dtype=str),
                run_ids=np.asarray(archive["event_run_ids"], dtype=str),
                selection_ids=np.asarray(archive["event_selection_ids"], dtype=str),
                complete=bool(np.asarray(archive["event_complete"]).item()),
                online_causal=bool(np.asarray(archive["event_online_causal"]).item()),
                timing_source=str(np.asarray(archive["event_timing_source"]).item()),
                candidate_ids=(
                    np.asarray(archive["event_candidate_ids"], dtype=str)
                    if has_current_candidate_contract
                    else None
                ),
                target_candidate_ids=(
                    np.asarray(archive["event_target_candidate_ids"], dtype=str)
                    if has_current_candidate_contract
                    else None
                ),
                repetition_indices=(
                    np.asarray(archive["event_repetition_indices"], dtype=np.int64)
                    if has_current_candidate_contract
                    else None
                ),
            ),
            metadata=pd.read_json(StringIO(metadata_json), orient="table"),
            provenance=json.loads(str(np.asarray(archive["provenance_json"]).item())),
            trial_channel_mask=(
                np.asarray(archive["trial_channel_mask"], dtype=bool)
                if "trial_channel_mask" in archive.files
                and np.asarray(archive["trial_channel_mask"]).size > 0
                else None
            ),
        )
    dataset.validate(require_labels=require_labels)
    if expected_preprocessing is not None and dataset.preprocessing != expected_preprocessing:
        raise ValueError(
            f"{path} preprocessing {dataset.preprocessing} does not match the required "
            f"contract {expected_preprocessing}."
        )
    return dataset


def select_epoch_channels(
    dataset: EpochDataset,
    target_channels: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> EpochDataset:
    """Select an exact channel subset; never pad or substitute a different electrode."""

    dataset.validate()
    present = {
        canonical_channel_name(name, aliases=aliases): index
        for index, name in enumerate(dataset.channel_names)
    }
    targets = tuple(canonical_channel_name(name, aliases=aliases) for name in target_channels)
    missing = [name for name in targets if name not in present]
    if missing:
        raise ValueError(
            f"Dataset {dataset.name!r} cannot supply channels {missing}; available={list(present)}."
        )
    picks = np.asarray([present[name] for name in targets], dtype=np.int64)
    selected = EpochDataset(
        name=dataset.name,
        X=dataset.X[:, picks, :],
        y=dataset.y,
        subject_ids=dataset.subject_ids,
        channel_names=targets,
        channel_positions_m=dataset.channel_positions_m[picks],
        channel_mask=dataset.channel_mask[picks],
        preprocessing=dataset.preprocessing,
        event_timeline=dataset.event_timeline,
        metadata=dataset.metadata,
        provenance={**dataset.provenance, "selected_channels": list(targets)},
        trial_channel_mask=(
            dataset.trial_channel_mask[:, picks] if dataset.trial_channel_mask is not None else None
        ),
    )
    selected.validate()
    return selected


def concatenate_epoch_datasets(
    datasets: Sequence[EpochDataset],
    *,
    name: str,
    provenance: Mapping[str, Any] | None = None,
) -> EpochDataset:
    """Concatenate records only when their physical tensor contracts are identical."""

    if not datasets:
        raise ValueError("At least one EpochDataset is required.")
    first = datasets[0]
    first.validate()
    for dataset in datasets[1:]:
        dataset.validate()
        if (
            dataset.channel_names != first.channel_names
            or dataset.preprocessing != first.preprocessing
            or not np.array_equal(dataset.channel_mask, first.channel_mask)
            or not np.allclose(dataset.channel_positions_m, first.channel_positions_m, atol=1e-6)
        ):
            raise ValueError(
                "Cannot concatenate datasets with different physical channel/time contracts."
            )
        if (dataset.y is None) != (first.y is None):
            raise ValueError("Cannot mix labeled and unlabeled datasets.")
    metadata = (
        pd.concat([dataset.metadata for dataset in datasets], ignore_index=True)
        if all(not dataset.metadata.empty for dataset in datasets)
        else pd.DataFrame()
    )
    merged = EpochDataset(
        name=name,
        X=np.concatenate([dataset.X for dataset in datasets], axis=0),
        y=(
            np.concatenate([dataset.y for dataset in datasets], axis=0)
            if first.y is not None
            else None
        ),
        subject_ids=np.concatenate([dataset.subject_ids for dataset in datasets]),
        channel_names=first.channel_names,
        channel_positions_m=first.channel_positions_m,
        channel_mask=first.channel_mask,
        preprocessing=first.preprocessing,
        event_timeline=concatenate_event_timelines(
            [dataset.event_timeline for dataset in datasets]
        ),
        metadata=metadata,
        provenance=dict(provenance or {}),
        trial_channel_mask=(
            np.concatenate(
                [
                    (
                        dataset.trial_channel_mask
                        if dataset.trial_channel_mask is not None
                        else np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                    )
                    for dataset in datasets
                ],
                axis=0,
            )
            if any(dataset.trial_channel_mask is not None for dataset in datasets)
            else None
        ),
    )
    merged.validate()
    return merged
