"""Auditable loading and validation for N2P3 transfer checkpoints.

Transfer checkpoints are executable model contracts, not just collections of
weights.  This module keeps architecture reconstruction, target-subject
exclusion, physical-shape checks, and input-statistics handling in one place so
the GTN and BI runners cannot silently disagree about what was pretrained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data.identity import (
    DatasetIdentityTable,
    IdentityExclusionPolicy,
    assert_target_identity_excluded,
)
from models.n2p3net import (
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)
from research.contracts import TrainingRunContract, semantic_sha256

CHECKPOINT_SCHEMA = "n2p3_transfer_checkpoint/1"


def load_checkpoint_payload(path: str | Path) -> dict[str, object]:
    """Load a checkpoint payload and require the mapping format used by runners."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a mapping checkpoint payload.")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is unsupported; regenerate the checkpoint.")
    return payload


def checkpoint_training_contract(
    payload: Mapping[str, object],
) -> TrainingRunContract:
    """Validate and return the semantic training contract bound to a checkpoint."""

    record = payload.get("training_contract")
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint lacks a training_contract mapping.")
    try:
        contract = TrainingRunContract(**dict(record))
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint training_contract is invalid: {error}") from error
    if payload.get("training_contract_digest") != contract.digest():
        raise ValueError("checkpoint training_contract_digest disagrees with its record.")
    return contract


def _architecture_from_payload(
    payload: Mapping[str, object],
) -> tuple[N2P3ArchitectureConfig, str]:
    """Resolve the complete architecture record required by the current schema."""

    record = payload.get("architecture")
    if not isinstance(record, Mapping):
        raise ValueError("checkpoint architecture must be a mapping.")
    pooling = record.get("pooling_mode")
    if not isinstance(pooling, str) or pooling not in POOLING_MODES:
        raise ValueError(f"checkpoint pooling_mode must be one of {sorted(POOLING_MODES)}.")

    values: dict[str, object] = {}
    aliases = {
        "temporal_filters": "st_temporal_filters",
        "temporal_kernel_size": "st_temporal_kernel_samples",
        "st_temporal_dilation": "st_temporal_dilation",
        "spatial_depth_multiplier": "spatial_depth_multiplier",
        "st_pool_size": "st_pool_size",
        "mst_kernel_sizes": "mst_kernel_samples",
        "mst_dilations": "mst_dilations",
        "mst_features_per_scale": "mst_features_per_scale",
        "mst_pool_size": "mst_pool_size",
        "dropout": "dropout",
        "spatial_max_norm": "spatial_max_norm",
    }
    if pooling == "quadratic_full_unfold":
        aliases["interaction_rank"] = "interaction_rank"
    if pooling == "mlp_full_unfold":
        aliases["mlp_hidden_features"] = "mlp_hidden_features"
    missing = sorted(source for source in aliases.values() if source not in record)
    if missing:
        raise ValueError(f"checkpoint architecture lacks current fields {missing}.")
    for destination, source in aliases.items():
        values[destination] = record[source]

    if "mst_kernel_sizes" in values:
        values["mst_kernel_sizes"] = tuple(int(item) for item in values["mst_kernel_sizes"])
    if "mst_dilations" in values:
        values["mst_dilations"] = tuple(int(item) for item in values["mst_dilations"])
    try:
        architecture = N2P3ArchitectureConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint architecture is invalid: {exc}") from exc

    return architecture, pooling


def _validate_metadata(
    payload: Mapping[str, object],
    dataset: object,
    *,
    target_subject: str | None,
    identity_exclusion_policy: IdentityExclusionPolicy,
) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is unsupported; regenerate the checkpoint.")
    state = payload.get("trunk_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint lacks a non-empty trunk_state_dict mapping.")

    training_contract = checkpoint_training_contract(payload)
    if payload.get("source_cache_sha256") != training_contract.source_cache_sha256:
        raise ValueError(
            "checkpoint source_cache_sha256 disagrees with its training contract."
        )
    ledger_payload = payload.get("training_identity_ledger")
    if not isinstance(ledger_payload, Mapping):
        raise ValueError(
            "pretraining checkpoint lacks a structured training_identity_ledger."
        )
    training_identity_ledger = DatasetIdentityTable.from_payload(ledger_payload)
    declared_ledger_digest = payload.get("training_identity_ledger_digest")
    if declared_ledger_digest != training_identity_ledger.digest():
        raise ValueError("checkpoint training identity ledger digest disagrees with its records.")
    if set(training_contract.training_participant_keys) != set(
        training_identity_ledger.authority_keys()
    ):
        raise ValueError(
            "training_contract participant keys disagree with the identity ledger."
        )

    architecture_record = payload.get("architecture")
    if not isinstance(architecture_record, Mapping) or semantic_sha256(
        architecture_record
    ) != semantic_sha256(training_contract.architecture):
        raise ValueError("checkpoint architecture disagrees with its training contract.")
    contract_preprocessing = training_contract.preprocessing
    input_signature = {
        "epoch": payload.get("input_preprocessing"),
        "channel_names": payload.get("input_channel_names"),
        "source_reference": payload.get("input_source_reference"),
    }
    if semantic_sha256(input_signature) != semantic_sha256(contract_preprocessing):
        raise ValueError(
            "checkpoint input signature disagrees with its training contract."
        )
    checkpoint_classifier_is_trained(payload)
    checkpoint_input_stats(payload, int(dataset.n_channels))

    if target_subject is not None:
        target = str(target_subject)
        target_identity_table = getattr(dataset, "identity_table", None)
        if not isinstance(target_identity_table, DatasetIdentityTable):
            raise ValueError(
                "target dataset lacks a structured identity table; regenerate its cache."
            )
        target_identity = target_identity_table.record_for(target)
        assert_target_identity_excluded(
            training_identity_ledger,
            target_identity,
            policy=identity_exclusion_policy,
        )

    expected = {
        "n_channels": getattr(dataset, "n_channels", None),
        "n_times": getattr(dataset, "n_times", None),
    }
    for key, value in expected.items():
        declared = payload.get(key)
        if declared is None:
            raise ValueError(f"checkpoint lacks required {key}.")
        if int(declared) != int(value):
            raise ValueError(
                f"checkpoint {key}={declared!r} does not match target dataset {value!r}."
            )
    declared_rate = payload.get("input_sample_rate_hz")
    if declared_rate is None:
        raise ValueError("checkpoint lacks required input_sample_rate_hz.")
    if not np.isclose(
        float(declared_rate), float(dataset.preprocessing.sfreq), rtol=0.0, atol=1e-9
    ):
        raise ValueError("checkpoint input_sample_rate_hz does not match the target dataset.")
    declared_tmin = payload.get("input_tmin_s")
    if declared_tmin is None:
        raise ValueError("checkpoint lacks required input_tmin_s.")
    if not np.isclose(
        float(declared_tmin), float(dataset.preprocessing.tmin_ms) / 1000.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("checkpoint input_tmin_s does not match the target dataset.")
    declared_channels = payload.get("input_channel_names")
    declared_preprocessing = payload.get("input_preprocessing")
    declared_reference = payload.get("input_source_reference")
    if declared_channels is None or declared_preprocessing is None:
        raise ValueError(
            "checkpoint lacks the ordered channel and preprocessing signature; regenerate it."
        )
    if not isinstance(declared_channels, list) or not all(
        isinstance(value, str) for value in declared_channels
    ):
        raise ValueError("checkpoint input_channel_names must be a list of strings.")
    if not isinstance(declared_preprocessing, Mapping):
        raise ValueError("checkpoint input_preprocessing must be a mapping.")
    target_channels = list(getattr(dataset, "channel_names", ()))
    target_preprocessing = asdict(dataset.preprocessing)
    target_reference = getattr(dataset, "provenance", {}).get("source_reference")
    input_mismatches = []
    if declared_channels != target_channels:
        input_mismatches.append("channel_names/order")
    if dict(declared_preprocessing) != target_preprocessing:
        input_mismatches.append("preprocessing")
    if declared_reference != target_reference:
        input_mismatches.append("source_reference")
    if input_mismatches:
        raise ValueError(
            "checkpoint input domain differs from the target in "
            + ", ".join(input_mismatches)
            + "; use an explicit stem/re-reference/domain-shift adapter."
        )


def checkpoint_input_stats(
    payload: Mapping[str, object],
    n_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return required channel statistics in ``(1,C,1)`` form."""

    mean_value = payload.get("input_mean")
    std_value = payload.get("input_std")
    if mean_value is None or std_value is None:
        raise ValueError("checkpoint requires explicit input_mean and input_std.")

    def normalize(value: object, name: str, *, positive: bool = False) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 3 and array.shape == (1, n_channels, 1):
            array = array.reshape(n_channels)
        elif array.ndim != 1 or array.shape != (n_channels,):
            raise ValueError(
                f"checkpoint {name} must have shape ({n_channels},) or (1,{n_channels},1)."
            )
        if not np.isfinite(array).all() or (positive and np.any(array <= 0.0)):
            raise ValueError(f"checkpoint {name} contains invalid values.")
        return array.reshape(1, n_channels, 1)

    return normalize(mean_value, "input_mean"), normalize(std_value, "input_std", positive=True)


def checkpoint_classifier_is_trained(payload: Mapping[str, object]) -> bool:
    """Whether the checkpoint classifier received supervised target labels."""

    declared = payload.get("classifier_trained")
    if not isinstance(declared, (bool, np.bool_)):
        raise ValueError("checkpoint classifier_trained must be an explicit boolean.")
    return bool(declared)


def checkpoint_scores_to_llr(
    payload: Mapping[str, object], logits: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a positive-slope source calibration without changing score order.

    A weighted binary cross-entropy classifier estimates a posterior shifted by
    ``log(pos_weight)``.  Removing that shift and the source class prior yields
    an LLR.  An optional source-validation temperature is constrained positive,
    so equal-count candidate argmax decisions are exactly invariant.
    """

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("checkpoint logits must be a finite one-dimensional array.")
    obsolete = [
        key for key in ("training_pos_weight", "training_prior") if key in payload
    ]
    if obsolete:
        raise ValueError(
            f"checkpoint uses obsolete top-level calibration fields {obsolete}; "
            "store calibration only in source_calibration."
        )
    calibration = payload.get("source_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("checkpoint requires a source_calibration mapping.")
    required = ("pos_weight", "train_prior", "temperature", "source")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(f"checkpoint source_calibration lacks required keys {missing}.")
    numeric_values = [calibration[key] for key in required[:3]]
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in numeric_values
    ):
        raise ValueError("checkpoint source_calibration numeric fields are invalid.")
    pos_weight, train_prior, temperature = map(float, numeric_values)
    source = calibration["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("checkpoint source_calibration source must be a non-empty string.")
    if not np.isfinite([pos_weight, train_prior, temperature]).all():
        raise ValueError("checkpoint calibration contains NaN/inf.")
    if pos_weight <= 0.0 or not 0.0 < train_prior < 1.0 or temperature <= 0.0:
        raise ValueError("checkpoint calibration has invalid weight/prior/temperature.")
    offset = float(np.log(pos_weight) + np.log(train_prior / (1.0 - train_prior)))
    return (values - offset) / temperature, {
        "mode": "weighted_ce_llr",
        "source": source,
        "pos_weight": pos_weight,
        "train_prior": train_prior,
        "temperature": temperature,
        "offset": offset,
        "order_preserving": True,
    }


def predict_n2p3_checkpoint(
    trunk: N2P3Net,
    X: np.ndarray,
    *,
    input_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    trial_channel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run a small target suffix through a pretrained classifier unchanged."""

    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (trunk.n_channels, trunk.n_times):
        raise ValueError(
            f"checkpoint input must be (N,{trunk.n_channels},{trunk.n_times})."
        )
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("checkpoint input must be finite and non-empty.")
    if trial_channel_mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    else:
        mask = np.asarray(trial_channel_mask)
        if mask.dtype != np.dtype(bool) or mask.shape != values.shape[:2]:
            raise ValueError("trial_channel_mask must be boolean and match X.shape[:2].")
        if not bool(mask.any(axis=1).all()):
            raise ValueError("Every checkpoint trial must retain one observed channel.")
    mean, std = input_stats
    standardized = ((values - mean) / std).astype(np.float32, copy=False)
    np.copyto(standardized, 0.0, where=~mask[:, :, None])
    trunk = trunk.to(device)
    trunk.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(standardized)).to(device)
        output = trunk(tensor)
    return (output[:, 1] - output[:, 0]).detach().cpu().numpy().astype(np.float64)


def load_n2p3_trunk_checkpoint(
    path: str | Path | Mapping[str, object],
    dataset: object,
    *,
    target_subject: str | None = None,
    identity_exclusion_policy: IdentityExclusionPolicy = "source_or_global",
) -> tuple[N2P3Net, dict[str, object]]:
    """Load one target-excluded trunk using its declared architecture contract."""

    payload = dict(path) if isinstance(path, Mapping) else load_checkpoint_payload(path)
    _validate_metadata(
        payload,
        dataset,
        target_subject=target_subject,
        identity_exclusion_policy=identity_exclusion_policy,
    )
    architecture, pooling = _architecture_from_payload(payload)
    trunk = N2P3Net(
        int(dataset.n_channels),
        n_times=int(dataset.n_times),
        sfreq=float(dataset.preprocessing.sfreq),
        tmin_s=float(dataset.preprocessing.tmin_ms) / 1000.0,
        pooling_mode=pooling,
        **architecture.model_kwargs(),
    )
    state = payload["trunk_state_dict"]
    try:
        missing, unexpected = trunk.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint weights do not match its declared architecture: {exc}") from exc
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the target trunk: missing={missing} unexpected={unexpected}."
        )
    return trunk, payload


def checkpoint_architecture_record(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a normalized architecture summary for run ledgers and diagnostics."""

    architecture, pooling = _architecture_from_payload(payload)
    required = ("input_tmin_s", "input_sample_rate_hz", "n_times")
    missing = [field for field in required if payload.get(field) is None]
    if missing:
        raise ValueError(f"checkpoint lacks required input geometry {missing}.")
    return N2P3Net.default_architecture_record(
        pooling_mode=pooling,
        tmin_s=float(payload["input_tmin_s"]),
        sfreq=float(payload["input_sample_rate_hz"]),
        n_times=int(payload["n_times"]),
        architecture=architecture,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "checkpoint_architecture_record",
    "checkpoint_classifier_is_trained",
    "checkpoint_input_stats",
    "checkpoint_training_contract",
    "checkpoint_scores_to_llr",
    "load_checkpoint_payload",
    "load_n2p3_trunk_checkpoint",
    "predict_n2p3_checkpoint",
]
