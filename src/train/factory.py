"""Construct performance candidates from the universal epoch contract."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from baselines.classic import SWLDA, TemplateMatching, WindowLogisticRegression
from baselines.deep import DEEP_MODEL_NAMES, DeepBaseline, DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from baselines.riemann import XdawnRiemann
from data.epochs import EpochDataset

BINARY_MODEL_NAMES = (
    "swdla",
    "window_lr",
    "template",
    "xdawn_rg",
    "n2p3net",
    *DEEP_MODEL_NAMES,
)


def _validate_binary_dataset(dataset: EpochDataset) -> None:
    if not isinstance(dataset, EpochDataset):
        raise TypeError("Binary model construction requires an EpochDataset instance.")
    dataset.validate(require_labels=True)
    if set(np.unique(dataset.y).tolist()) != {0, 1}:
        raise ValueError("Binary oddball training requires labels {0,1}.")


def _deep_config(
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    validation_subject_fraction: float,
    overrides: dict[str, Any] | None,
) -> DeepConfig:
    values: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "val_subject_frac": validation_subject_fraction,
    }
    values.update(overrides or {})
    return DeepConfig(**values)


def build_binary_model(
    model_name: str,
    dataset: EpochDataset,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    validation_subject_fraction: float = 0.1,
    deep_config_overrides: dict[str, Any] | None = None,
    device: torch.device | None = None,
):
    """Return one candidate using only the data and performance contracts."""

    _validate_binary_dataset(dataset)
    key = str(model_name).strip().lower()
    sfreq = float(dataset.preprocessing.sfreq)
    tmin = float(dataset.preprocessing.tmin_ms) / 1000.0
    common_mask = np.asarray(dataset.channel_mask, dtype=bool)
    if key == "swdla":
        return SWLDA(sfreq=sfreq, tmin=tmin)
    if key == "window_lr":
        return WindowLogisticRegression(sfreq=sfreq, tmin=tmin)
    if key == "template":
        return TemplateMatching(sfreq=sfreq, tmin=tmin, window_ms=(250.0, 600.0))
    if key == "xdawn_rg":
        return XdawnRiemann(nfilter=min(4, dataset.n_channels))
    config = _deep_config(
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        validation_subject_fraction=validation_subject_fraction,
        overrides=deep_config_overrides,
    )
    if key == "n2p3net":
        return N2P3NetBaseline(
            dataset.n_channels,
            dataset.n_times,
            sfreq,
            config=config,
            device=device,
            channel_mask=common_mask,
        )
    if key in DEEP_MODEL_NAMES:
        return DeepBaseline(
            key,
            n_chans=dataset.n_channels,
            n_times=dataset.n_times,
            sfreq=sfreq,
            config=config,
            device=device,
            channel_mask=common_mask,
        )
    raise ValueError(f"Unknown binary model {model_name!r}; choose from {BINARY_MODEL_NAMES}.")


def describe_binary_model(model_name: str, model: object) -> dict[str, Any]:
    """Return a stable, model-neutral manifest entry."""

    key = str(model_name).strip().lower()
    if key in {"swdla", "window_lr", "template", "xdawn_rg"}:
        return {"name": key, "class": type(model).__name__}
    config = getattr(model, "cfg", None)
    return {
        "name": key,
        "class": type(model).__name__,
        "config": asdict(config) if config is not None else {},
        "n_chans": int(model.n_chans),
        "n_times": int(model.n_times),
        "sfreq": float(model.sfreq),
    }
