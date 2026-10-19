"""Calibration-fitted, pointwise EOG regression for EEG raw recordings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
from sklearn.linear_model import LinearRegression

from data.channel import canonical_channel_name
from research.contracts import semantic_sha256
from research.execution import atomic_write_json

OCULAR_MODEL_SCHEMA = "n2p3_calibration_eog_regression/3"
OCULAR_PRESERVED_EOG_SCHEMA = "n2p3_calibration_eog_regression/4"
OCULAR_STRENGTH_SCHEMA = "n2p3_calibration_eog_regression/5"
SHARED_MEASURED_REFERENCE = "shared_measured_reference"
PRESERVE_ACQUIRED_EOG = "preserve_acquired_eog"
EOG_REFERENCE_POLICIES = (SHARED_MEASURED_REFERENCE, PRESERVE_ACQUIRED_EOG)


def validate_eog_derivation(derivation, channels, source_reference, *, rereference=False):
    """A single HEOG sensor shares a reference only when explicitly declared."""
    if not derivation:
        if rereference:
            raise ValueError("ocular rereferencing requires explicit EOG derivation")
        return
    if set(derivation) != set(channels):
        raise ValueError("EOG derivation must identify each typed EOG channel")
    for record in derivation.values():
        if (not isinstance(record, Mapping) or record.get("kind") not in {"single_ended", "bipolar"}
                or not str(record.get("placement", "")).strip()
                or not str(record.get("reference", "")).strip()):
            raise ValueError("EOG derivation requires kind, placement and reference")
        if rereference and (record["kind"] != "single_ended" or record["reference"] != source_reference):
            raise ValueError("EOG derivation does not declare the shared EEG acquisition reference")


def _reference_raw(raw: mne.io.BaseRaw, reference_channel: str | None, *,
                   eog_reference_policy: str = SHARED_MEASURED_REFERENCE) -> mne.io.BaseRaw:
    if eog_reference_policy not in EOG_REFERENCE_POLICIES:
        raise ValueError("unsupported EOG reference policy")
    output = raw.copy().load_data()
    if reference_channel is None:
        return output
    eeg, eog = _channel_roles(output)
    matches = [name for name in eeg if canonical_channel_name(name) == reference_channel]
    if len(matches) != 1 or matches[0] in output.info["bads"]:
        raise ValueError("ocular reference must be one usable measured EEG channel.")
    anchor = matches[0]
    waveform = output.get_data(picks=[anchor])[0]
    output.set_eeg_reference(ref_channels=[anchor], projection=False, verbose=False)
    if eog_reference_policy == SHARED_MEASURED_REFERENCE:
        output.apply_function(
            lambda values: values - waveform[None, :], picks=list(eog), channel_wise=False
        )
    output.set_channel_types({anchor: "misc"}, on_unit_change="ignore")
    return output


def _channel_roles(raw: mne.io.BaseRaw) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kinds = raw.get_channel_types()
    eeg = tuple(name for name, kind in zip(raw.ch_names, kinds, strict=True) if kind == "eeg")
    eog = tuple(name for name, kind in zip(raw.ch_names, kinds, strict=True) if kind == "eog")
    if not eeg or not eog:
        raise ValueError("ocular regression requires independently typed EEG and EOG channels.")
    if set(eog) & set(raw.info["bads"]):
        raise ValueError("bad EOG channels cannot supply an artifact correction signal.")
    return eeg, eog


@dataclass(frozen=True)
class FittedEOGRegression:
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    calibration_eog_mean_v: tuple[float, ...]
    participant_key: str
    source_reference: str
    fit_provenance: Mapping[str, Any]
    reference_channel: str | None = None
    eog_derivation: Mapping[str, Any] = field(default_factory=dict)
    eog_reference_policy: str = SHARED_MEASURED_REFERENCE
    correction_strength: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.correction_strength, bool) or not np.isscalar(self.correction_strength)
                or not np.isfinite(self.correction_strength) or not 0 <= self.correction_strength <= 1):
            raise ValueError("ocular correction_strength must be finite and between zero and one")
        if self.eog_reference_policy not in EOG_REFERENCE_POLICIES:
            raise ValueError("unsupported EOG reference policy")
        if self.eog_reference_policy == PRESERVE_ACQUIRED_EOG:
            acquisition_reference = self.fit_provenance.get("acquisition_source_reference")
            if not isinstance(acquisition_reference, str) or not acquisition_reference.strip():
                raise ValueError("preserved EOG requires its original acquisition source reference")
        validate_eog_derivation(self.eog_derivation, self.eog_channels,
            self.fit_provenance.get("acquisition_source_reference", self.source_reference),
            rereference=self.reference_channel is not None and self.eog_reference_policy == SHARED_MEASURED_REFERENCE)
        if self.reference_channel is not None:
            anchor = canonical_channel_name(self.reference_channel)
            object.__setattr__(self, "reference_channel", anchor)
            if self.source_reference != "measured_channel:" + anchor:
                raise ValueError("ocular reference preprocessing and final reference disagree.")
        combined = tuple(map(canonical_channel_name, self.eeg_channels + self.eog_channels))
        if not self.eeg_channels or not self.eog_channels or len(set(combined)) != len(combined):
            raise ValueError("ocular channel roles must be non-empty, unique and disjoint.")
        coefficients = np.asarray(self.coefficients, dtype=float)
        mean = np.asarray(self.calibration_eog_mean_v, dtype=float)
        if coefficients.shape != (len(self.eeg_channels), len(self.eog_channels)):
            raise ValueError("ocular coefficients disagree with channel roles.")
        if (
            mean.shape != (len(self.eog_channels),)
            or not np.isfinite(mean).all()
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("ocular model values must be finite and align with EOG channels.")
        if (
            not self.participant_key.strip()
            or not self.source_reference.strip()
            or not self.fit_provenance
        ):
            raise ValueError(
                "ocular models require participant authority, reference and fit provenance."
            )

    def record(self) -> dict[str, Any]:
        record = {
            "schema": OCULAR_MODEL_SCHEMA if self.eog_reference_policy == SHARED_MEASURED_REFERENCE
            else OCULAR_PRESERVED_EOG_SCHEMA,
            "eeg_channels": list(self.eeg_channels),
            "eog_channels": list(self.eog_channels),
            "coefficients": [list(row) for row in self.coefficients],
            "calibration_eog_mean_v": list(self.calibration_eog_mean_v),
            "participant_key": self.participant_key,
            "source_reference": self.source_reference,
            "fit_provenance": dict(self.fit_provenance),
            "application": "pointwise_subtraction_with_frozen_calibration_center",
            "signal_unit": "V",
            "reference_changed": self.reference_channel is not None,
            "reference_channel": self.reference_channel,
            "eog_derivation": dict(self.eog_derivation),
        }
        if self.eog_reference_policy != SHARED_MEASURED_REFERENCE:
            record["eog_reference_policy"] = self.eog_reference_policy
        if self.correction_strength != 1.0:
            record.update(schema=OCULAR_STRENGTH_SCHEMA, correction_strength=float(self.correction_strength),
                          eog_reference_policy=self.eog_reference_policy)
        return record

    @property
    def effective_coefficients(self) -> np.ndarray:
        """The applied map; fitted coefficients remain the original calibration estimate."""
        return float(self.correction_strength) * np.asarray(self.coefficients)

    def apply(self, raw: mne.io.BaseRaw, *, source_reference: str,
              eog_derivation: Mapping[str, Any] | None = None) -> mne.io.BaseRaw:
        if self.eog_reference_policy == PRESERVE_ACQUIRED_EOG:
            if source_reference != self.fit_provenance["acquisition_source_reference"]:
                raise ValueError("preserved EOG acquisition reference differs from calibration")
            if dict(eog_derivation or {}) != dict(self.eog_derivation):
                raise ValueError("recording EOG derivation differs from the frozen ocular model")
        if self.eog_derivation and eog_derivation != self.eog_derivation:
            raise ValueError("recording EOG derivation differs from the frozen ocular model")
        validate_eog_derivation(self.eog_derivation, self.eog_channels, source_reference,
                               rereference=self.reference_channel is not None and self.eog_reference_policy == SHARED_MEASURED_REFERENCE)
        output = _reference_raw(raw, self.reference_channel, eog_reference_policy=self.eog_reference_policy)
        eeg, eog = _channel_roles(output)
        if tuple(map(canonical_channel_name, eeg)) != tuple(
            map(canonical_channel_name, self.eeg_channels)
        ) or tuple(map(canonical_channel_name, eog)) != tuple(
            map(canonical_channel_name, self.eog_channels)
        ):
            raise ValueError("ocular model EEG/EOG channel order differs from the recording.")
        effective_reference = (
            source_reference
            if self.reference_channel is None
            else "measured_channel:" + self.reference_channel
        )
        if effective_reference != self.source_reference:
            raise ValueError("ocular model and recording references differ.")
        if self.correction_strength == 0:
            return output
        artifact = output.get_data(picks=list(eog))
        if not np.isfinite(artifact).all():
            raise ValueError("ocular inference requires finite EOG samples.")
        artifact -= np.asarray(self.calibration_eog_mean_v)[:, None]
        correction = self.effective_coefficients @ artifact
        output.apply_function(
            lambda signal: signal - correction, picks=list(eeg), channel_wise=False
        )
        return output


def fit_eog_regression(
    calibration_raw: mne.io.BaseRaw,
    *,
    participant_key: str,
    source_reference: str,
    fit_provenance: Mapping[str, Any],
    reference_channel: str | None = None,
    fit_sample_mask: np.ndarray | None = None,
    eog_derivation: Mapping[str, Any] | None = None,
    eog_reference_policy: str = SHARED_MEASURED_REFERENCE,
) -> FittedEOGRegression:
    """Fit only the explicitly supplied calibration raw; never choose EEG labels."""
    anchor = None if reference_channel is None else canonical_channel_name(reference_channel)
    if eog_reference_policy not in EOG_REFERENCE_POLICIES:
        raise ValueError("unsupported EOG reference policy")
    _, original_eog = _channel_roles(calibration_raw)
    validate_eog_derivation(eog_derivation, original_eog, source_reference,
                           rereference=anchor is not None and eog_reference_policy == SHARED_MEASURED_REFERENCE)
    mask = np.ones(calibration_raw.n_times, dtype=bool) if fit_sample_mask is None else np.asarray(fit_sample_mask)
    if mask.dtype != bool or mask.shape != (calibration_raw.n_times,) or int(mask.sum()) < 2:
        raise ValueError("ocular fit_sample_mask must select at least two aligned calibration samples")
    original_reference = source_reference
    calibration_raw = _reference_raw(calibration_raw, anchor, eog_reference_policy=eog_reference_policy)
    source_reference = original_reference if anchor is None else "measured_channel:" + anchor
    eeg, eog = _channel_roles(calibration_raw)
    if set(eeg) & set(calibration_raw.info["bads"]):
        raise ValueError(
            "calibration EEG channels must be usable before fitting ocular coefficients."
        )
    artifact = calibration_raw.get_data(picks=list(eog)).T[mask]
    signal = calibration_raw.get_data(picks=list(eeg)).T[mask]
    if not np.isfinite(artifact).all() or not np.isfinite(signal).all():
        raise ValueError("ocular calibration must be finite.")
    center = artifact.mean(axis=0)
    model = LinearRegression(fit_intercept=False).fit(
        artifact - center, signal - signal.mean(axis=0)
    )
    if model.rank_ != len(eog):
        raise ValueError("ocular calibration has flat or linearly dependent EOG channels.")
    return FittedEOGRegression(
        eeg_channels=eeg,
        eog_channels=eog,
        coefficients=tuple(tuple(float(value) for value in row) for row in model.coef_),
        calibration_eog_mean_v=tuple(float(value) for value in center),
        participant_key=participant_key,
        source_reference=source_reference,
        reference_channel=anchor,
        eog_derivation=dict(eog_derivation or {}),
        eog_reference_policy=eog_reference_policy,
        fit_provenance={
            **fit_provenance,
            "samples": len(artifact),
            "fit_sample_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "fit_sample_mask_explicit": fit_sample_mask is not None,
            "fit_signal_sha256": hashlib.sha256(signal.tobytes() + artifact.tobytes()).hexdigest(),
            "sample_rate_hz": float(calibration_raw.info["sfreq"]),
            "estimator": "sklearn.LinearRegression_centered_calibration_only",
            "acquisition_source_reference": original_reference,
            "reference_order": ("eeg_reference_before_regression_eog_preserved_as_acquired"
                if eog_reference_policy == PRESERVE_ACQUIRED_EOG else "common_measured_reference_before_regression")
            if anchor else "native_reference",
            "shared_eeg_eog_acquisition_reference": None
            if not eog_derivation and eog_reference_policy == PRESERVE_ACQUIRED_EOG else bool(eog_derivation) and all(
                row["kind"] == "single_ended" and row["reference"] == original_reference
                for row in eog_derivation.values()),
        },
    )


def save_eog_regression(path: str | Path, model: FittedEOGRegression) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    record = model.record()
    return atomic_write_json(destination, {"model": record, "digest": semantic_sha256(record)})


def load_eog_regression(path: str | Path) -> FittedEOGRegression:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"model", "digest"}
        or not isinstance(payload["model"], dict)
    ):
        raise ValueError("ocular file must contain a fitted model and its digest.")
    record = payload["model"]
    if payload["digest"] != semantic_sha256(record):
        raise ValueError("ocular model schema or digest is invalid.")
    return eog_regression_from_record(record)


def eog_regression_from_record(record: Mapping[str, Any]) -> FittedEOGRegression:
    """Restore an embedded fitted model whose enclosing artifact binds its digest."""
    if not isinstance(record, Mapping) or record.get("schema") not in (
        OCULAR_MODEL_SCHEMA, OCULAR_PRESERVED_EOG_SCHEMA, OCULAR_STRENGTH_SCHEMA
    ):
        raise ValueError("ocular model schema is invalid.")
    if record["schema"] == OCULAR_MODEL_SCHEMA and "eog_reference_policy" in record:
        raise ValueError("ocular schema /3 does not define an EOG reference policy field")
    if record["schema"] == OCULAR_PRESERVED_EOG_SCHEMA and record.get("eog_reference_policy") != PRESERVE_ACQUIRED_EOG:
        raise ValueError("ocular schema /4 requires preserve_acquired_eog policy")
    if record["schema"] == OCULAR_STRENGTH_SCHEMA and (
            "correction_strength" not in record or "eog_reference_policy" not in record):
        raise ValueError("ocular schema /5 requires explicit correction strength and reference policy")
    model = FittedEOGRegression(
        eeg_channels=tuple(record["eeg_channels"]),
        eog_channels=tuple(record["eog_channels"]),
        coefficients=tuple(tuple(row) for row in record["coefficients"]),
        calibration_eog_mean_v=tuple(record["calibration_eog_mean_v"]),
        participant_key=record["participant_key"],
        source_reference=record["source_reference"],
        fit_provenance=record["fit_provenance"],
        reference_channel=record["reference_channel"],
        eog_derivation=record["eog_derivation"],
        eog_reference_policy=record.get("eog_reference_policy", SHARED_MEASURED_REFERENCE),
        correction_strength=record.get("correction_strength", 1.0),
    )
    if model.record() != record:
        raise ValueError("ocular model record has unsupported or inconsistent fields.")
    return model


def validate_eog_calibration_budget(
    processing: Mapping[str, Any],
    calibration_by_participant: Mapping[str, tuple[str, ...]],
) -> None:
    """Reject preprocessing that used sessions outside the evaluator's budget."""
    if processing.get("status") != "applied":
        return
    record = processing.get("model")
    if not isinstance(record, Mapping) or processing.get("model_digest") != semantic_sha256(record):
        raise ValueError("ocular processing lacks a valid fitted-model record.")
    participant = record.get("participant_key")
    fit_sessions = record.get("fit_provenance", {}).get("calibration_session_ids", [])
    allowed = calibration_by_participant.get(participant, ())
    if not fit_sessions or not set(fit_sessions) <= set(allowed):
        raise ValueError(
            "ocular preprocessing used sessions outside the declared calibration budget."
        )
    from data.continuous_qc import validate_continuous_calibration_budget
    validate_continuous_calibration_budget(
        record.get("fit_provenance", {}).get("continuous_artifact_processing", {"status": "not_applied"}),
        calibration_by_participant)
