"""Fold-local, auditable EEG epoch quality control.

This module deliberately does not implement a fixed voltage cutoff.  Its
thresholds are learned on the outer-training data only, separately for every
channel, and are then frozen before the held-out subject is inspected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FoldLocalArtifactPolicy:
    """Cross-validated local peak-to-peak policy inspired by Autoreject.

    ``max_bad_channel_fraction`` is the local-repair boundary: affected
    channels are masked; only training epochs exceeding this fraction are
    excluded.  Held-out epochs are never silently removed.
    """

    candidate_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975, 0.99)
    flat_quantile: float = 0.005
    max_bad_channel_fraction: float = 0.25
    cv_splits: int = 5
    min_clean_epochs: int = 4

    def validate(self) -> None:
        if not self.candidate_quantiles or any(
            not 0.5 < value < 1.0 for value in self.candidate_quantiles
        ):
            raise ValueError("candidate_quantiles must be non-empty values in (0.5, 1).")
        if tuple(sorted(set(self.candidate_quantiles))) != self.candidate_quantiles:
            raise ValueError("candidate_quantiles must be sorted and unique.")
        if not 0.0 <= self.flat_quantile < 0.5:
            raise ValueError("flat_quantile must be in [0, 0.5).")
        if not 0.0 <= self.max_bad_channel_fraction < 1.0:
            raise ValueError("max_bad_channel_fraction must be in [0, 1).")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least two.")
        if self.min_clean_epochs < 1:
            raise ValueError("min_clean_epochs must be positive.")

    def fit(
        self,
        X: np.ndarray,
        subject_ids: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> FoldLocalArtifactModel:
        self.validate()
        X = np.asarray(X)
        subject_ids = np.asarray(subject_ids)
        if X.ndim != 3 or len(X) < 2:
            raise ValueError("Artifact fitting requires at least two (N,C,T) training epochs.")
        if subject_ids.shape != (len(X),):
            raise ValueError("subject_ids must align with X for artifact fitting.")
        if not np.isfinite(X).all():
            raise ValueError("Artifact fitting does not accept NaN/inf samples.")
        observed = _observed_mask(X, trial_channel_mask)
        ptp = _relative_peak_to_peak(X, observed)
        std = np.std(X.astype(np.float64, copy=False), axis=2)
        chosen_quantiles = np.asarray(
            [self._choose_quantile(X, ptp, observed, subject_ids, channel) for channel in range(X.shape[1])],
            dtype=float,
        )
        ptp_thresholds = np.empty(X.shape[1], dtype=float)
        flat_thresholds = np.empty(X.shape[1], dtype=float)
        for channel in range(X.shape[1]):
            values = ptp[observed[:, channel], channel]
            scales = std[observed[:, channel], channel]
            if len(values) == 0:
                ptp_thresholds[channel] = np.inf
                flat_thresholds[channel] = -np.inf
            else:
                ptp_thresholds[channel] = float(np.quantile(values, chosen_quantiles[channel]))
                flat_thresholds[channel] = float(np.quantile(scales, self.flat_quantile))
        return FoldLocalArtifactModel(
            policy=self,
            ptp_thresholds=ptp_thresholds,
            flat_std_thresholds=flat_thresholds,
            selected_quantiles=chosen_quantiles,
            fit_n_epochs=int(len(X)),
            fit_subjects=tuple(sorted(np.unique(subject_ids.astype(str)).tolist())),
        )

    def _choose_quantile(
        self,
        X: np.ndarray,
        ptp: np.ndarray,
        observed: np.ndarray,
        subject_ids: np.ndarray,
        channel: int,
    ) -> float:
        """Choose a threshold quantile through subject-disjoint ERP stability CV."""

        subjects = np.unique(subject_ids.astype(str))
        if len(subjects) < 2:
            return self.candidate_quantiles[-1]
        folds = np.array_split(subjects, min(self.cv_splits, len(subjects)))
        errors = np.zeros(len(self.candidate_quantiles), dtype=float)
        counts = np.zeros(len(self.candidate_quantiles), dtype=np.int64)
        for held_out_subjects in folds:
            validation = np.isin(subject_ids.astype(str), held_out_subjects) & observed[:, channel]
            training = ~np.isin(subject_ids.astype(str), held_out_subjects) & observed[:, channel]
            if training.sum() < self.min_clean_epochs or validation.sum() < self.min_clean_epochs:
                continue
            template = np.median(X[training, channel, :], axis=0)
            training_ptp = ptp[training, channel]
            for index, quantile in enumerate(self.candidate_quantiles):
                threshold = float(np.quantile(training_ptp, quantile))
                clean_validation = validation & (ptp[:, channel] <= threshold)
                if clean_validation.sum() < self.min_clean_epochs:
                    continue
                mean_validation = np.mean(X[clean_validation, channel, :], axis=0)
                errors[index] += float(np.mean((mean_validation - template) ** 2))
                counts[index] += 1
        valid = counts > 0
        if not valid.any():
            return self.candidate_quantiles[-1]
        mean_errors = np.full(len(errors), np.inf)
        mean_errors[valid] = errors[valid] / counts[valid]
        # A tolerance tie-break keeps the least aggressive threshold, avoiding
        # unstable hard deletions when CV error is numerically indistinguishable.
        best = float(np.min(mean_errors))
        tolerance = best * 1e-6 + np.finfo(float).eps
        return self.candidate_quantiles[
            int(np.flatnonzero(mean_errors <= best + tolerance)[-1])
        ]


@dataclass(frozen=True)
class FoldLocalArtifactModel:
    """Frozen fold-local quality model and its serializable audit record."""

    policy: FoldLocalArtifactPolicy
    ptp_thresholds: np.ndarray
    flat_std_thresholds: np.ndarray
    selected_quantiles: np.ndarray
    fit_n_epochs: int
    fit_subjects: tuple[str, ...]

    def transform(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> ArtifactTransformResult:
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] != len(self.ptp_thresholds):
            raise ValueError("Artifact transform X must match the fitted (N,C,T) geometry.")
        if not np.isfinite(X).all():
            raise ValueError("Artifact transform does not accept NaN/inf samples.")
        observed = _observed_mask(X, trial_channel_mask)
        ptp = _relative_peak_to_peak(X, observed)
        std = np.std(X.astype(np.float64, copy=False), axis=2)
        high_amplitude = ptp > self.ptp_thresholds[None, :]
        flatline = std <= self.flat_std_thresholds[None, :]
        bad = observed & (high_amplitude | flatline)
        effective_mask = observed & ~bad
        bad_counts = bad.sum(axis=1)
        observed_counts = observed.sum(axis=1)
        drop = bad_counts > self.policy.max_bad_channel_fraction * observed_counts
        transformed = X.copy()
        transformed[~effective_mask] = 0.0
        return ArtifactTransformResult(
            X=transformed,
            trial_channel_mask=effective_mask,
            bad_channel_mask=bad,
            drop_epoch_mask=drop,
            all_channels_bad=~effective_mask.any(axis=1),
        )

    def record(self) -> dict[str, object]:
        return {
            "method": "fold_local_ptp_cv",
            "ptp_normalization": "within_epoch_observed_channel_median",
            "policy": asdict(self.policy),
            "fit_n_epochs": self.fit_n_epochs,
            "fit_subjects": list(self.fit_subjects),
            "ptp_thresholds": self.ptp_thresholds.tolist(),
            "flat_std_thresholds": self.flat_std_thresholds.tolist(),
            "selected_quantiles": self.selected_quantiles.tolist(),
        }


@dataclass(frozen=True)
class ArtifactTransformResult:
    X: np.ndarray
    trial_channel_mask: np.ndarray
    bad_channel_mask: np.ndarray
    drop_epoch_mask: np.ndarray
    all_channels_bad: np.ndarray

    def summary(self) -> dict[str, object]:
        observed = self.trial_channel_mask.shape[1]
        return {
            "n_epochs": int(len(self.X)),
            "mean_bad_channels": float(self.bad_channel_mask.sum(axis=1).mean()),
            "mean_bad_channel_fraction": float(self.bad_channel_mask.mean()),
            "n_epochs_over_bad_channel_limit": int(self.drop_epoch_mask.sum()),
            "n_all_channels_bad": int(self.all_channels_bad.sum()),
            "n_channels": int(observed),
        }


def apply_fold_local_artifact_policy(
    policy: FoldLocalArtifactPolicy,
    X: np.ndarray,
    subject_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    trial_channel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Fit on train only; mask local bad channels and drop only train epochs."""

    fitted = policy.fit(X[train_mask], np.asarray(subject_ids)[train_mask], _subset_mask(trial_channel_mask, train_mask))
    transformed = fitted.transform(X, trial_channel_mask)
    if transformed.all_channels_bad[test_mask].any():
        count = int(transformed.all_channels_bad[test_mask].sum())
        raise ValueError(
            f"Fold-local artifact policy masked every channel in {count} held-out epochs; "
            "the test denominator cannot be silently changed."
        )
    effective_train_mask = train_mask & ~transformed.drop_epoch_mask
    if not effective_train_mask.any():
        raise ValueError("Fold-local artifact policy removed every training epoch.")
    labels = np.asarray(subject_ids)
    remaining_subjects = np.unique(labels[effective_train_mask])
    if len(remaining_subjects) < 2:
        raise ValueError("Artifact policy left fewer than two training subjects for validation.")
    audit = fitted.record()
    audit["train"] = transformed_summary(transformed, train_mask)
    audit["test"] = transformed_summary(transformed, test_mask)
    return transformed.X, transformed.trial_channel_mask, effective_train_mask, audit


def transformed_summary(result: ArtifactTransformResult, rows: np.ndarray) -> dict[str, object]:
    subset = np.asarray(rows, dtype=bool)
    if not subset.any():
        return {"n_epochs": 0, "mean_bad_channels": 0.0, "mean_bad_channel_fraction": 0.0,
                "n_epochs_over_bad_channel_limit": 0, "n_all_channels_bad": 0}
    bad = result.bad_channel_mask[subset]
    return {
        "n_epochs": int(subset.sum()),
        "mean_bad_channels": float(bad.sum(axis=1).mean()),
        "mean_bad_channel_fraction": float(bad.mean()),
        "n_epochs_over_bad_channel_limit": int(result.drop_epoch_mask[subset].sum()),
        "n_all_channels_bad": int(result.all_channels_bad[subset].sum()),
    }


def _observed_mask(X: np.ndarray, trial_channel_mask: np.ndarray | None) -> np.ndarray:
    if trial_channel_mask is None:
        return np.ones(X.shape[:2], dtype=bool)
    observed = np.asarray(trial_channel_mask)
    if observed.dtype != np.dtype(bool) or observed.shape != X.shape[:2]:
        raise ValueError("trial_channel_mask must be boolean and match X.shape[:2].")
    return observed


def _subset_mask(mask: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
    return None if mask is None else np.asarray(mask)[rows]


def _relative_peak_to_peak(X: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Normalize each epoch's PTP by its observed-channel median.

    EEG recording gain and reference scale can differ substantially across
    held-out subjects. This preserves a channel-local PTP contrast while
    preventing an unseen subject's global scale from becoming a false artifact.
    """

    raw_ptp = np.ptp(X.astype(np.float64, copy=False), axis=2)
    scales = np.empty(len(X), dtype=float)
    for row, row_observed in enumerate(observed):
        values = raw_ptp[row, row_observed]
        if len(values) == 0:
            raise ValueError("Every epoch must contain at least one observed channel.")
        scales[row] = float(np.median(values))
    floor = np.finfo(float).eps
    return raw_ptp / np.maximum(scales[:, None], floor)
