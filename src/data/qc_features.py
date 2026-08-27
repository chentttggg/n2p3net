"""Cacheable, fold-independent EEG quality features.

The values in this module are derived from one already-preprocessed epoch and
its channel-availability mask only.  They are therefore safe to materialize in
an :class:`EpochDataset` cache.  Artifact thresholds and accept/reject
decisions deliberately remain in ``data.artifact`` and are fitted per outer
training fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

QC_FEATURE_SCHEMA = "n2p3net_epoch_qc_features/1"


@dataclass(frozen=True)
class EpochQCFeatures:
    """Fold-independent quality statistics aligned to ``(epoch, channel)``.

    ``epoch_scale_v`` is the median raw peak-to-peak amplitude over observed
    channels.  It retains physical recording-scale information which relative
    PTP intentionally removes.  The local PTP and standard-deviation features
    have volts as their source unit; the ratio is dimensionless.
    """

    relative_ptp: np.ndarray
    channel_std_v: np.ndarray
    epoch_scale_v: np.ndarray
    observed_mask: np.ndarray
    schema: str = QC_FEATURE_SCHEMA

    @property
    def n_epochs(self) -> int:
        return int(self.relative_ptp.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.relative_ptp.shape[1])

    def validate(self, *, n_epochs: int, n_channels: int) -> None:
        if self.schema != QC_FEATURE_SCHEMA:
            raise ValueError(f"Unsupported QC feature schema {self.schema!r}.")
        expected_2d = (n_epochs, n_channels)
        for name, value in (
            ("relative_ptp", self.relative_ptp),
            ("channel_std_v", self.channel_std_v),
            ("observed_mask", self.observed_mask),
        ):
            array = np.asarray(value)
            if array.shape != expected_2d:
                raise ValueError(f"QC feature {name} must have shape {expected_2d}.")
        if np.asarray(self.epoch_scale_v).shape != (n_epochs,):
            raise ValueError(f"QC feature epoch_scale_v must have shape ({n_epochs},).")
        if not np.issubdtype(np.asarray(self.relative_ptp).dtype, np.floating):
            raise ValueError("QC feature relative_ptp must be floating-point.")
        if not np.issubdtype(np.asarray(self.channel_std_v).dtype, np.floating):
            raise ValueError("QC feature channel_std_v must be floating-point.")
        if not np.issubdtype(np.asarray(self.epoch_scale_v).dtype, np.floating):
            raise ValueError("QC feature epoch_scale_v must be floating-point.")
        if np.asarray(self.observed_mask).dtype != np.dtype(bool):
            raise ValueError("QC feature observed_mask must be boolean.")
        if not bool(np.asarray(self.observed_mask).any(axis=1).all()):
            raise ValueError("QC feature observed_mask must retain one channel per epoch.")
        for name, value in (
            ("relative_ptp", self.relative_ptp),
            ("channel_std_v", self.channel_std_v),
            ("epoch_scale_v", self.epoch_scale_v),
        ):
            if not np.isfinite(np.asarray(value)).all() or bool((np.asarray(value) < 0.0).any()):
                raise ValueError(f"QC feature {name} must be finite and non-negative.")

    def subset(self, rows: np.ndarray) -> EpochQCFeatures:
        indices = np.asarray(rows)
        if indices.dtype == np.dtype(bool) and indices.shape != (self.n_epochs,):
            raise ValueError("Boolean QC feature rows must align with the epoch axis.")
        return EpochQCFeatures(
            relative_ptp=np.asarray(self.relative_ptp)[indices].copy(),
            channel_std_v=np.asarray(self.channel_std_v)[indices].copy(),
            epoch_scale_v=np.asarray(self.epoch_scale_v)[indices].copy(),
            observed_mask=np.asarray(self.observed_mask)[indices].copy(),
        )

    def select_channels(self, indices: Sequence[int]) -> EpochQCFeatures:
        channel_indices = np.asarray(indices, dtype=np.int64)
        if channel_indices.ndim != 1 or not len(channel_indices):
            raise ValueError("QC feature channel selection must be non-empty and one-dimensional.")
        if int(channel_indices.min()) < 0 or int(channel_indices.max()) >= self.n_channels:
            raise ValueError("QC feature channel selection is out of bounds.")
        return EpochQCFeatures(
            relative_ptp=np.asarray(self.relative_ptp)[:, channel_indices].copy(),
            channel_std_v=np.asarray(self.channel_std_v)[:, channel_indices].copy(),
            epoch_scale_v=np.asarray(self.epoch_scale_v).copy(),
            observed_mask=np.asarray(self.observed_mask)[:, channel_indices].copy(),
        )

    @classmethod
    def concatenate(cls, features: Sequence[EpochQCFeatures]) -> EpochQCFeatures:
        if not features:
            raise ValueError("At least one QC feature payload is required.")
        n_channels = features[0].n_channels
        for feature in features:
            feature.validate(n_epochs=feature.n_epochs, n_channels=n_channels)
        return cls(
            relative_ptp=np.concatenate([feature.relative_ptp for feature in features], axis=0),
            channel_std_v=np.concatenate([feature.channel_std_v for feature in features], axis=0),
            epoch_scale_v=np.concatenate([feature.epoch_scale_v for feature in features], axis=0),
            observed_mask=np.concatenate([feature.observed_mask for feature in features], axis=0),
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "relative_ptp": "dimensionless_within_epoch_observed_channel_median",
            "channel_std_v": "within_epoch_standard_deviation_v",
            "epoch_scale_v": "within_epoch_observed_channel_median_raw_ptp_v",
            "observed_mask": "effective_channel_availability",
        }


def effective_observed_mask(
    *,
    n_epochs: int,
    channel_mask: np.ndarray,
    trial_channel_mask: np.ndarray | None,
) -> np.ndarray:
    """Combine static and trial-level availability without enabling missing channels."""

    static = np.asarray(channel_mask, dtype=bool)
    if static.ndim != 1 or not bool(static.any()):
        raise ValueError("channel_mask must retain one channel.")
    observed = np.broadcast_to(static, (n_epochs, len(static))).copy()
    if trial_channel_mask is not None:
        trial = np.asarray(trial_channel_mask)
        if trial.dtype != np.dtype(bool) or trial.shape != observed.shape:
            raise ValueError("trial_channel_mask must be boolean and align with epochs/channels.")
        if bool((trial & ~observed).any()):
            raise ValueError("trial_channel_mask cannot enable a statically absent channel.")
        observed &= trial
    if not bool(observed.any(axis=1).all()):
        raise ValueError("Every epoch must retain one observed channel.")
    return observed


def compute_epoch_qc_features(
    X: np.ndarray,
    *,
    channel_mask: np.ndarray,
    trial_channel_mask: np.ndarray | None = None,
) -> EpochQCFeatures:
    """Compute cache-safe statistics from final preprocessed EEG epochs only."""

    data = np.asarray(X)
    if data.ndim != 3 or not len(data):
        raise ValueError("QC features require a non-empty (N,C,T) EEG tensor.")
    if not np.issubdtype(data.dtype, np.floating) or not np.isfinite(data).all():
        raise ValueError("QC features require finite floating-point EEG data.")
    observed = effective_observed_mask(
        n_epochs=len(data),
        channel_mask=channel_mask,
        trial_channel_mask=trial_channel_mask,
    )
    values = data.astype(np.float64, copy=False)
    raw_ptp = np.ptp(values, axis=2)
    epoch_scale = np.nanmedian(np.where(observed, raw_ptp, np.nan), axis=1)
    floor = np.finfo(np.float64).eps
    relative_ptp = raw_ptp / np.maximum(epoch_scale[:, None], floor)
    channel_std = np.std(values, axis=2)
    relative_ptp = np.where(observed, relative_ptp, 0.0).astype(np.float32)
    channel_std = np.where(observed, channel_std, 0.0).astype(np.float32)
    result = EpochQCFeatures(
        relative_ptp=relative_ptp,
        channel_std_v=channel_std,
        epoch_scale_v=np.asarray(epoch_scale, dtype=np.float32),
        observed_mask=observed,
    )
    result.validate(n_epochs=data.shape[0], n_channels=data.shape[1])
    return result
