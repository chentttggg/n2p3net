"""Persistent sidecars for fold-local artifact models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data.artifact import FoldLocalArtifactModel, FoldLocalArtifactPolicy

SIDECAR_SCHEMA = "n2p3net_fold_artifact_models/1"


def _mask_digest(mask: np.ndarray) -> str:
    normalized = np.asarray(mask, dtype=bool)
    payload = normalized.shape[0].to_bytes(8, "little") + np.packbits(normalized).tobytes()
    return hashlib.sha256(payload).hexdigest()


def fold_artifact_fingerprint(
    *,
    cache_sha256: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    policy: FoldLocalArtifactPolicy,
) -> str:
    """Hash only cache identity, executable folds, and policy parameters."""

    cache_sha256 = str(cache_sha256).strip().lower()
    if len(cache_sha256) != 64 or any(char not in "0123456789abcdef" for char in cache_sha256):
        raise ValueError("cache_sha256 must be one lowercase SHA-256 digest.")
    policy.validate()
    if not folds:
        raise ValueError("folds must not be empty.")
    payload = {
        "cache_sha256": cache_sha256,
        "folds": [
            {
                "train": _mask_digest(train),
                "test": _mask_digest(test),
            }
            for train, test in folds
        ],
        "policy": asdict(policy),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_fold_artifact_sidecar_path(cache_path: str | Path, fingerprint: str) -> Path:
    cache = Path(cache_path)
    return cache.with_name(f"{cache.name}.fold-qc-{fingerprint[:16]}.json")


def _model_from_record(record: dict[str, Any]) -> FoldLocalArtifactModel:
    policy_data = dict(record["policy"])
    policy_data["candidate_quantiles"] = tuple(policy_data["candidate_quantiles"])
    policy_data["candidate_bad_channel_fractions"] = tuple(
        policy_data["candidate_bad_channel_fractions"]
    )
    policy = FoldLocalArtifactPolicy(**policy_data)
    policy.validate()
    calibration = record["global_scale_calibration"]
    model = FoldLocalArtifactModel(
        policy=policy,
        ptp_thresholds=np.asarray(record["ptp_thresholds"], dtype=float),
        flat_std_thresholds=np.asarray(record["flat_std_thresholds"], dtype=float),
        selected_quantiles=np.asarray(record["selected_quantiles"], dtype=float),
        selected_bad_channel_fraction=float(record["selected_bad_channel_fraction"]),
        global_scale_log_center=float(calibration["log_center"]),
        global_scale_log_robust_std=float(calibration["log_robust_std"]),
        fit_n_epochs=int(record["fit_n_epochs"]),
        fit_groups=tuple(str(group) for group in record["fit_groups"]),
    )
    shape = model.ptp_thresholds.shape
    if (
        model.ptp_thresholds.ndim != 1
        or model.flat_std_thresholds.shape != shape
        or model.selected_quantiles.shape != shape
        or model.fit_n_epochs < 1
        or not model.fit_groups
    ):
        raise ValueError("Fold-QC sidecar contains an invalid artifact model geometry.")
    return model


def load_fold_artifact_sidecar(
    path: str | Path,
    *,
    expected_fingerprint: str,
    expected_fold_count: int,
) -> dict[int, FoldLocalArtifactModel] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SIDECAR_SCHEMA:
            return None
        if payload.get("fingerprint") != expected_fingerprint:
            return None
        records = payload["models"]
        if set(records) != {str(index) for index in range(expected_fold_count)}:
            return None
        return {int(index): _model_from_record(record) for index, record in records.items()}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_fold_artifact_sidecar(
    path: str | Path,
    *,
    fingerprint: str,
    cache_sha256: str,
    policy: FoldLocalArtifactPolicy,
    models: dict[int, FoldLocalArtifactModel],
) -> Path:
    path = Path(path)
    expected = set(range(len(models)))
    if set(models) != expected:
        raise ValueError("Fold-QC models must use contiguous zero-based fold indices.")
    payload = {
        "schema": SIDECAR_SCHEMA,
        "fingerprint": fingerprint,
        "cache_sha256": cache_sha256,
        "policy": asdict(policy),
        "fold_count": len(models),
        "created_utc": datetime.now(UTC).isoformat(),
        "models": {str(index): model.record() for index, model in sorted(models.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
