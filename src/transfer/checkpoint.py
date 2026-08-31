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

from models.n2p3net import (
    BROAD_REFERENCE_ST_TEMPORAL_KERNEL_SIZE,
    POOLING_MODES,
    N2P3ArchitectureConfig,
    N2P3Net,
)


def load_checkpoint_payload(path: str | Path) -> dict[str, object]:
    """Load a checkpoint payload and require the mapping format used by runners."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a mapping checkpoint payload.")
    return payload


def _subject_from_key(value: str) -> str:
    """Extract the subject component from the historical ``dataset\\0subject`` key."""

    return value.split("\0", 1)[-1]


def _architecture_from_payload(
    payload: Mapping[str, object],
) -> tuple[N2P3ArchitectureConfig, str]:
    """Resolve an architecture record, retaining a conservative legacy fallback."""

    record = payload.get("architecture")
    if record is not None and not isinstance(record, Mapping):
        raise ValueError("checkpoint architecture must be a mapping when present.")
    config = payload.get("config")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping when present.")

    values: dict[str, object] = {}
    if isinstance(record, Mapping):
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
            "interaction_rank": "interaction_rank",
            "mlp_hidden_features": "mlp_hidden_features",
        }
        for destination, source in aliases.items():
            if source in record:
                values[destination] = record[source]
    if isinstance(config, Mapping):
        # Supervised source pretraining historically stored these fields under
        # config before the full architecture record was added.
        for key in (
            "temporal_filters",
            "temporal_kernel_size",
            "st_temporal_dilation",
            "spatial_depth_multiplier",
            "st_pool_size",
            "mst_kernel_sizes",
            "mst_dilations",
            "mst_features_per_scale",
            "mst_pool_size",
            "dropout",
            "spatial_max_norm",
            "interaction_rank",
            "mlp_hidden_features",
        ):
            if key in config:
                values[key] = config[key]

    if "mst_kernel_sizes" in values:
        values["mst_kernel_sizes"] = tuple(int(item) for item in values["mst_kernel_sizes"])
    if "mst_dilations" in values:
        values["mst_dilations"] = tuple(int(item) for item in values["mst_dilations"])
    if "temporal_kernel_size" not in values:
        # Checkpoints without an explicit kernel declaration predate the K35 default.
        values["temporal_kernel_size"] = BROAD_REFERENCE_ST_TEMPORAL_KERNEL_SIZE
    try:
        architecture = N2P3ArchitectureConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint architecture is invalid: {exc}") from exc

    pooling = None
    if isinstance(record, Mapping):
        pooling = record.get("pooling_mode")
    if pooling is None and isinstance(config, Mapping):
        pooling = config.get("pooling_mode")
    if pooling is None:
        pooling = "ms_flatten"
    if not isinstance(pooling, str) or pooling not in POOLING_MODES:
        raise ValueError(f"checkpoint pooling_mode must be one of {sorted(POOLING_MODES)}.")
    return architecture, pooling


def _validate_metadata(
    payload: Mapping[str, object],
    dataset: object,
    *,
    target_subject: str | None,
    target_cache_sha256: str | None,
    allow_input_domain_shift: bool,
) -> None:
    state = payload.get("trunk_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint lacks a non-empty trunk_state_dict mapping.")

    keys = payload.get("training_subject_keys")
    if not isinstance(keys, list) or not keys or not all(isinstance(value, str) for value in keys):
        raise ValueError("pretraining checkpoint lacks auditable training_subject_keys.")
    if len(set(keys)) != len(keys):
        raise ValueError("training_subject_keys must be unique.")

    training_subjects = payload.get("training_subjects")
    if training_subjects is not None:
        if not isinstance(training_subjects, list) or not all(
            isinstance(value, str) for value in training_subjects
        ):
            raise ValueError("checkpoint training_subjects must be a list of strings.")
        key_subjects = {_subject_from_key(value) for value in keys}
        if key_subjects != set(training_subjects):
            raise ValueError("training_subject_keys and training_subjects disagree.")

    if target_subject is not None:
        target = str(target_subject)
        dataset_name = str(getattr(dataset, "name", ""))
        target_key = f"{dataset_name}\0{target}"
        source_name = payload.get("source_dataset_name")
        same_dataset = source_name == dataset_name
        global_participant_keys = payload.get("training_global_participant_keys", [])
        if not isinstance(global_participant_keys, list) or not all(
            isinstance(value, str) for value in global_participant_keys
        ):
            raise ValueError("training_global_participant_keys must be a list of strings.")
        target_global_key = getattr(dataset, "provenance", {}).get("global_participant_key")
        overlaps_same_dataset = target_key in set(keys) or (
            same_dataset and target in {_subject_from_key(value) for value in keys}
        )
        cache_subject_keys = payload.get("training_cache_subject_keys", [])
        if not isinstance(cache_subject_keys, list) or not all(
            isinstance(value, str) for value in cache_subject_keys
        ):
            raise ValueError("training_cache_subject_keys must be a list of strings.")
        overlaps_same_cache = bool(
            target_cache_sha256
            and f"{target_cache_sha256}\0{target}" in set(cache_subject_keys)
        )
        overlaps_global_identity = (
            isinstance(target_global_key, str)
            and target_global_key
            and target_global_key in set(global_participant_keys)
        )
        if overlaps_same_dataset or overlaps_same_cache or overlaps_global_identity:
            raise ValueError(
                f"pretraining checkpoint includes target subject {target!r}; "
                "use a leave-target-out checkpoint."
            )
        holdout = payload.get("holdout_subjects")
        if same_dataset and holdout is not None:
            if not isinstance(holdout, list) or target not in {str(value) for value in holdout}:
                raise ValueError(
                    "checkpoint holdout_subjects does not declare the requested target subject."
                )

    expected = {
        "n_channels": getattr(dataset, "n_channels", None),
        "n_times": getattr(dataset, "n_times", None),
    }
    for key, value in expected.items():
        declared = payload.get(key)
        if declared is not None and int(declared) != int(value):
            raise ValueError(
                f"checkpoint {key}={declared!r} does not match target dataset {value!r}."
            )
    declared_rate = payload.get("input_sample_rate_hz")
    if declared_rate is not None and not np.isclose(
        float(declared_rate), float(dataset.preprocessing.sfreq), rtol=0.0, atol=1e-9
    ):
        raise ValueError("checkpoint input_sample_rate_hz does not match the target dataset.")
    declared_tmin = payload.get("input_tmin_s")
    if declared_tmin is not None and not np.isclose(
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
    if input_mismatches and not allow_input_domain_shift:
        raise ValueError(
            "checkpoint input domain differs from the target in "
            + ", ".join(input_mismatches)
            + "; use an explicit stem/re-reference/domain-shift adapter."
        )


def checkpoint_input_stats(
    payload: Mapping[str, object],
    n_channels: int,
    *,
    required: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return channel statistics in ``(1,C,1)`` form, or an explicit identity."""

    mean_value = payload.get("input_mean")
    std_value = payload.get("input_std")
    if (mean_value is None) != (std_value is None):
        raise ValueError("checkpoint input_mean and input_std must be supplied together.")
    if mean_value is None:
        if bool(payload.get("standardized", False)):
            if required:
                raise ValueError(
                    "standardized checkpoint lacks input_mean/input_std; normalization is not auditable."
                )
            return None
        return (
            np.zeros((1, int(n_channels), 1), dtype=np.float32),
            np.ones((1, int(n_channels), 1), dtype=np.float32),
        )

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
    if declared is not None:
        if not isinstance(declared, (bool, np.bool_)):
            raise ValueError("checkpoint classifier_trained must be boolean.")
        return bool(declared)
    config = payload.get("config")
    return bool(
        isinstance(config, Mapping)
        and isinstance(config.get("training"), str)
        and "supervised" in str(config["training"]).lower()
    )


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
    calibration = payload.get("source_calibration")
    if calibration is not None and not isinstance(calibration, Mapping):
        raise ValueError("source_calibration must be a mapping when present.")

    pos_weight = payload.get("training_pos_weight")
    train_prior = payload.get("training_prior")
    temperature = 1.0
    source = "source_weighted_ce_analytic"
    if isinstance(calibration, Mapping):
        pos_weight = calibration.get("pos_weight", pos_weight)
        train_prior = calibration.get("train_prior", train_prior)
        temperature = float(calibration.get("temperature", 1.0))
        source = str(calibration.get("source", "source_validation_temperature"))
    if pos_weight is None or train_prior is None:
        # Historical supervised checkpoints did not persist optimizer-prior
        # metadata.  Identity preserves candidate ranking, but it is explicitly
        # not advertised as a calibrated LLR.
        return values.copy(), {
            "mode": "legacy_rank_score",
            "source": "checkpoint_missing_weighted_ce_metadata",
            "order_preserving": True,
        }
    pos_weight = float(pos_weight)
    train_prior = float(train_prior)
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
    target_cache_sha256: str | None = None,
    allow_input_domain_shift: bool = False,
) -> tuple[N2P3Net, dict[str, object]]:
    """Load one target-excluded trunk using its declared architecture contract."""

    payload = dict(path) if isinstance(path, Mapping) else load_checkpoint_payload(path)
    _validate_metadata(
        payload,
        dataset,
        target_subject=target_subject,
        target_cache_sha256=target_cache_sha256,
        allow_input_domain_shift=allow_input_domain_shift,
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
    return N2P3Net.default_architecture_record(
        pooling_mode=pooling,
        tmin_s=float(payload.get("input_tmin_s", -0.2)),
        sfreq=float(payload.get("input_sample_rate_hz", 128.0)),
        n_times=int(payload.get("n_times", 128)),
        architecture=architecture,
    )


__all__ = [
    "checkpoint_architecture_record",
    "checkpoint_classifier_is_trained",
    "checkpoint_input_stats",
    "checkpoint_scores_to_llr",
    "load_checkpoint_payload",
    "load_n2p3_trunk_checkpoint",
    "predict_n2p3_checkpoint",
]
