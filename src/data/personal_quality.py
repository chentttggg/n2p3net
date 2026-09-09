"""Separate personal EEG distribution diagnostics from unusable signal evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from data.artifact import (
    ArtifactTransformResult,
    _fit_data_digest,
    _fit_groups,
    _observed_mask,
    _resolve_qc_features,
)
from research.contracts import semantic_sha256

PERSONAL_EPOCH_QUALITY_METHOD = "personal_epoch_signal_evidence"
PERSONAL_EPOCH_QUALITY_SCHEMA = "n2p3_personal_epoch_quality/1"
FEATURE_NAMES = ("log_ptp_v", "log_std_v", "log_normalized_difference_rms")


def _signal_features(X):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 2 or not np.isfinite(values).all():
        raise ValueError("personal quality requires finite (epochs, channels, time>=2) EEG")
    ptp = np.ptp(values, axis=2)
    std = np.std(values, axis=2)
    difference = np.sqrt(np.mean(np.diff(values, axis=2) ** 2, axis=2))
    roughness = np.divide(difference, std, out=np.zeros_like(std), where=std > 0)
    features = np.stack((ptp, std, roughness), axis=2)
    logs = np.full_like(features, np.nan)
    np.log(features, out=logs, where=features > 0)
    return ptp, logs


@dataclass(frozen=True)
class PersonalQualityTransformResult(ArtifactTransformResult):
    diagnostic_masks: Mapping[str, np.ndarray] = field(default_factory=dict)

    def summary(self):
        return {**super().summary(), "hard_rejection_basis": "exact_constant_epoch_signal",
                "diagnostic_channel_epochs": {
                    name: int(mask.sum()) for name, mask in self.diagnostic_masks.items()},
                "distribution_diagnostics_are_artifact_proof": False}


@dataclass(frozen=True)
class PersonalEpochQualityPolicy:
    method: str = PERSONAL_EPOCH_QUALITY_METHOD
    diagnostic_mad_z: float = 6.0
    minimum_group_epochs: int = 4
    max_bad_channel_fraction: float = 0.5

    def validate(self):
        if self.method != PERSONAL_EPOCH_QUALITY_METHOD:
            raise ValueError("unknown personal epoch quality method")
        if not np.isfinite(self.diagnostic_mad_z) or self.diagnostic_mad_z <= 0:
            raise ValueError("diagnostic MAD multiplier must be finite and positive")
        if type(self.minimum_group_epochs) is not int or self.minimum_group_epochs < 2:
            raise ValueError("personal quality requires at least two epochs per diagnostic group")
        if not np.isfinite(self.max_bad_channel_fraction) or not 0 <= self.max_bad_channel_fraction < 1:
            raise ValueError("invalid constant-channel fraction")

    def fit(self, X, subject_ids, trial_channel_mask=None, qc_features=None, *, fit_groups=None):
        self.validate()
        X = np.asarray(X)
        subjects = np.asarray(subject_ids)
        if X.ndim != 3 or len(X) < 2 or subjects.shape != (len(X),):
            raise ValueError("personal quality requires aligned calibration epochs and subjects")
        observed = _observed_mask(X, trial_channel_mask)
        features = _resolve_qc_features(X, observed, qc_features)
        groups = _fit_groups(subjects, fit_groups)
        _, logs = _signal_features(X)
        lower = np.full((X.shape[1], len(FEATURE_NAMES)), np.nan)
        upper = lower.copy()
        support = np.zeros_like(lower, dtype=np.int64)
        for group in np.unique(groups):
            rows = groups == group
            for channel in range(X.shape[1]):
                for feature in range(len(FEATURE_NAMES)):
                    selected = rows & observed[:, channel] & np.isfinite(logs[:, channel, feature])
                    if selected.sum() < self.minimum_group_epochs:
                        continue
                    values = logs[selected, channel, feature]
                    center = np.median(values)
                    spread = 1.4826 * np.median(np.abs(values - center))
                    lo, hi = center - self.diagnostic_mad_z * spread, center + self.diagnostic_mad_z * spread
                    lower[channel, feature] = np.fmin(lower[channel, feature], lo)
                    upper[channel, feature] = np.fmax(upper[channel, feature], hi)
                    support[channel, feature] += 1
        return PersonalEpochQualityModel(
            policy=self, diagnostic_log_lower=lower, diagnostic_log_upper=upper,
            diagnostic_group_counts=support, fit_observed_channel_mask=observed.any(axis=0),
            fit_n_epochs=len(X), fit_groups=tuple(sorted(np.unique(groups).tolist())),
            fit_subject_ids=tuple(sorted(np.unique(subjects.astype(str)).tolist())),
            fit_group_source="subject_ids" if fit_groups is None else "explicit_fit_groups",
            fit_data_digest=_fit_data_digest(X, subjects, groups, observed, features),
        )


@dataclass(frozen=True)
class PersonalEpochQualityModel:
    policy: PersonalEpochQualityPolicy
    diagnostic_log_lower: np.ndarray
    diagnostic_log_upper: np.ndarray
    diagnostic_group_counts: np.ndarray
    fit_observed_channel_mask: np.ndarray
    fit_n_epochs: int
    fit_groups: tuple[str, ...]
    fit_subject_ids: tuple[str, ...]
    fit_group_source: str
    fit_data_digest: str
    fit_provenance: Mapping[str, object] = field(default_factory=dict)

    @property
    def n_channels(self):
        return int(len(self.fit_observed_channel_mask))

    def _validate(self):
        self.policy.validate()
        shape = (self.n_channels, len(FEATURE_NAMES))
        if (self.n_channels < 1 or self.fit_n_epochs < 2
                or self.fit_observed_channel_mask.dtype != np.dtype(bool)
                or self.fit_observed_channel_mask.shape != (self.n_channels,)
                or self.diagnostic_log_lower.shape != shape or self.diagnostic_log_upper.shape != shape
                or self.diagnostic_group_counts.shape != shape
                or not np.issubdtype(self.diagnostic_group_counts.dtype, np.integer)
                or (self.diagnostic_group_counts < 0).any()):
            raise ValueError("invalid personal quality geometry or fit support")
        supported = self.diagnostic_group_counts > 0
        if (not np.isfinite(self.diagnostic_log_lower[supported]).all()
                or not np.isfinite(self.diagnostic_log_upper[supported]).all()
                or (self.diagnostic_log_lower[supported] > self.diagnostic_log_upper[supported]).any()
                or not np.isnan(self.diagnostic_log_lower[~supported]).all()
                or not np.isnan(self.diagnostic_log_upper[~supported]).all()):
            raise ValueError("personal diagnostic envelope differs from its declared support")
        if (not self.fit_groups or not self.fit_subject_ids or not self.fit_data_digest
                or self.fit_group_source not in {"subject_ids", "explicit_fit_groups"}):
            raise ValueError("personal quality lacks fit provenance")

    def transform(self, X, trial_channel_mask=None, qc_features=None, *, materialize_masked_data=True):
        self._validate()
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] != self.n_channels:
            raise ValueError("personal quality channel geometry differs from its calibration")
        observed = _observed_mask(X, trial_channel_mask)
        ptp, logs = _signal_features(X)
        constant = observed & (ptp == 0)
        effective = observed & ~constant
        # Amplitude and spectral composition do not identify artifacts on their own.
        supported = self.diagnostic_group_counts > 0
        high = observed[:, :, None] & supported[None] & (logs > self.diagnostic_log_upper[None])
        low = observed[:, :, None] & supported[None] & (logs < self.diagnostic_log_lower[None])
        diagnostics = {
            "amplitude_above_calibration": high[:, :, 0],
            "amplitude_below_calibration": low[:, :, 0],
            "standard_deviation_below_calibration": low[:, :, 1],
            "roughness_above_calibration": high[:, :, 2],
            "joint_amplitude_roughness": high[:, :, 0] & high[:, :, 2],
            "unseen_calibration_channel": observed & ~self.fit_observed_channel_mask[None],
        }
        drop = constant.sum(axis=1) > self.policy.max_bad_channel_fraction * observed.sum(axis=1)
        values = X
        if materialize_masked_data:
            values = X.copy()
            values[~effective] = 0
        return PersonalQualityTransformResult(
            X=values, trial_channel_mask=effective, bad_channel_mask=constant,
            drop_epoch_mask=drop, all_channels_bad=~effective.any(axis=1),
            global_epoch_scale_mask=np.zeros(len(X), dtype=bool), diagnostic_masks=diagnostics,
        )

    def record(self):
        self._validate()
        def finite_rows(values):
            return [[float(value) if np.isfinite(value) else None for value in row] for row in values]

        return {
            "schema": PERSONAL_EPOCH_QUALITY_SCHEMA, "method": PERSONAL_EPOCH_QUALITY_METHOD,
            "policy": asdict(self.policy), "n_channels": self.n_channels,
            "feature_names": list(FEATURE_NAMES),
            "diagnostic_log_lower": finite_rows(self.diagnostic_log_lower),
            "diagnostic_log_upper": finite_rows(self.diagnostic_log_upper),
            "diagnostic_group_counts": self.diagnostic_group_counts.tolist(),
            "diagnostic_scope": "calibration_group_envelope_no_false_positive_rate_claim",
            "hard_rejection_basis": "exact_constant_epoch_signal",
            "raw_quality_requirement": "preserve_upstream_observation_masks_and_recovery",
            "distribution_diagnostics_are_artifact_proof": False,
            "fit_observed_channel_mask": self.fit_observed_channel_mask.tolist(),
            "fit_n_epochs": self.fit_n_epochs, "fit_groups": list(self.fit_groups),
            "fit_subject_ids": list(self.fit_subject_ids), "fit_group_source": self.fit_group_source,
            "fit_data_digest": self.fit_data_digest, "fit_provenance": dict(self.fit_provenance),
        }


def personal_epoch_quality_from_record(record):
    if (not isinstance(record, Mapping) or record.get("schema") != PERSONAL_EPOCH_QUALITY_SCHEMA
            or record.get("method") != PERSONAL_EPOCH_QUALITY_METHOD):
        raise ValueError("unknown personal epoch quality schema or method")
    try:
        model = PersonalEpochQualityModel(
            policy=PersonalEpochQualityPolicy(**record["policy"]),
            diagnostic_log_lower=np.asarray(record["diagnostic_log_lower"], dtype=float),
            diagnostic_log_upper=np.asarray(record["diagnostic_log_upper"], dtype=float),
            diagnostic_group_counts=np.asarray(record["diagnostic_group_counts"]),
            fit_observed_channel_mask=np.asarray(record["fit_observed_channel_mask"]),
            fit_n_epochs=record["fit_n_epochs"], fit_groups=tuple(record["fit_groups"]),
            fit_subject_ids=tuple(record["fit_subject_ids"]), fit_group_source=record["fit_group_source"],
            fit_data_digest=record["fit_data_digest"], fit_provenance=record["fit_provenance"],
        )
        if semantic_sha256(model.record()) != semantic_sha256(record):
            raise ValueError("personal epoch quality record is inconsistent")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid personal epoch quality record: {error}") from error
    return model
