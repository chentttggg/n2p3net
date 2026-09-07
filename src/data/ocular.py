"""Calibration-fitted, pointwise EOG regression for EEG raw recordings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from sklearn.linear_model import LinearRegression

from data.channel import canonical_channel_name
from research.contracts import semantic_sha256
from research.execution import atomic_write_json

OCULAR_MODEL_SCHEMA = "n2p3_calibration_eog_regression/2"


def _reference_raw(raw: mne.io.BaseRaw, reference_channel: str | None) -> mne.io.BaseRaw:
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

    def __post_init__(self) -> None:
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
        return {
            "schema": OCULAR_MODEL_SCHEMA,
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
        }

    def apply(self, raw: mne.io.BaseRaw, *, source_reference: str) -> mne.io.BaseRaw:
        output = _reference_raw(raw, self.reference_channel)
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
        artifact = output.get_data(picks=list(eog))
        if not np.isfinite(artifact).all():
            raise ValueError("ocular inference requires finite EOG samples.")
        artifact -= np.asarray(self.calibration_eog_mean_v)[:, None]
        correction = np.asarray(self.coefficients) @ artifact
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
) -> FittedEOGRegression:
    """Fit only the explicitly supplied calibration raw; never choose EEG labels."""
    anchor = None if reference_channel is None else canonical_channel_name(reference_channel)
    original_reference = source_reference
    calibration_raw = _reference_raw(calibration_raw, anchor)
    source_reference = original_reference if anchor is None else "measured_channel:" + anchor
    eeg, eog = _channel_roles(calibration_raw)
    if set(eeg) & set(calibration_raw.info["bads"]):
        raise ValueError(
            "calibration EEG channels must be usable before fitting ocular coefficients."
        )
    artifact = calibration_raw.get_data(picks=list(eog)).T
    signal = calibration_raw.get_data(picks=list(eeg)).T
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
        fit_provenance={
            **fit_provenance,
            "samples": len(artifact),
            "sample_rate_hz": float(calibration_raw.info["sfreq"]),
            "estimator": "sklearn.LinearRegression_centered_calibration_only",
            "acquisition_source_reference": original_reference,
            "reference_order": "common_measured_reference_before_regression"
            if anchor
            else "native_reference",
            "shared_eeg_eog_acquisition_reference": True,
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
    if payload["digest"] != semantic_sha256(record) or record.get("schema") != OCULAR_MODEL_SCHEMA:
        raise ValueError("ocular model schema or digest is invalid.")
    return eog_regression_from_record(record)


def eog_regression_from_record(record: Mapping[str, Any]) -> FittedEOGRegression:
    """Restore an embedded fitted model whose enclosing artifact binds its digest."""
    if not isinstance(record, Mapping) or record.get("schema") != OCULAR_MODEL_SCHEMA:
        raise ValueError("ocular model schema is invalid.")
    model = FittedEOGRegression(
        eeg_channels=tuple(record["eeg_channels"]),
        eog_channels=tuple(record["eog_channels"]),
        coefficients=tuple(tuple(row) for row in record["coefficients"]),
        calibration_eog_mean_v=tuple(record["calibration_eog_mean_v"]),
        participant_key=record["participant_key"],
        source_reference=record["source_reference"],
        fit_provenance=record["fit_provenance"],
        reference_channel=record["reference_channel"],
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
