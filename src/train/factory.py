"""Dataset-independent construction of canonical Neural-RIDE adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from baselines.deep import DEEP_MODEL_NAMES, DeepBaseline, DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from data.channel import build_channel_identity
from data.epochs import EpochDataset
from models.component_window import PCW_CANONICAL_DTAU_BOUNDS
from models.erp_calibration import FoldERPCalibrator
from models.time_axis import EpochTimeAxis
from train.recipe import BINARY_ODDBALL_TASK, NEURAL_RIDE_V12, NeuralRideRecipe

BINARY_MODEL_NAMES = ("n2p3net", *DEEP_MODEL_NAMES)


def _validate_binary_dataset(dataset: EpochDataset) -> None:
    if not isinstance(dataset, EpochDataset):
        raise TypeError("Binary model construction requires an EpochDataset instance.")
    dataset.validate(require_labels=True)
    labels = set(np.unique(dataset.y).tolist())
    if labels != {0, 1}:
        raise ValueError(f"Binary oddball training requires labels {{0,1}}, got {sorted(labels)}.")


def describe_binary_model(
    model_name: str,
    model: object,
    recipe: NeuralRideRecipe = NEURAL_RIDE_V12,
) -> dict[str, Any]:
    """Return an auditable, model-neutral protocol record for a binary adapter.

    Runners should record this description instead of importing an adapter
    class or branching on dataset-specific model details. Architecture-specific
    fields stay in the factory boundary where the corresponding model is built.
    """

    key = str(model_name).strip().lower()
    if key == "n2p3net":
        return {
            "name": recipe.name,
            "task": asdict(BINARY_ODDBALL_TASK),
            "model_kwargs": dict(getattr(model, "model_kwargs", {})),
            "trainer_kwargs": dict(getattr(model, "trainer_kwargs", {})),
        }
    if key in DEEP_MODEL_NAMES:
        config = getattr(model, "cfg", None)
        return {
            "name": f"deep_{key}",
            "model": key,
            "config": asdict(config) if config is not None else {},
            "n_chans": int(model.n_chans),
            "n_times": int(model.n_times),
            "sfreq": float(model.sfreq),
        }
    raise ValueError(f"Unknown binary model {model_name!r}; choose from {BINARY_MODEL_NAMES}.")


def build_binary_neural_ride_adapter(
    dataset: EpochDataset,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    validation_subject_fraction: float = 0.1,
    fold_erp_calibration: bool = True,
    frozen_erp_prior: Mapping[str, Any] | None = None,
    trainer_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    recipe: NeuralRideRecipe = NEURAL_RIDE_V12,
) -> N2P3NetBaseline:
    """Build one binary-oddball adapter from physical dataset metadata only."""

    _validate_binary_dataset(dataset)
    if fold_erp_calibration and frozen_erp_prior is not None:
        raise ValueError("fold ERP calibration and a frozen ERP prior are mutually exclusive.")
    identity = build_channel_identity(
        dataset.channel_names,
        channel_mask=dataset.channel_mask,
        positions_m=dataset.channel_positions_m,
        montage=None,
        allow_missing_positions=True,
    )
    active_mask = np.asarray(dataset.channel_mask, dtype=bool)
    invalid_active = active_mask & ~identity.mask
    if invalid_active.any():
        missing = [
            name
            for name, invalid in zip(dataset.channel_names, invalid_active, strict=True)
            if invalid
        ]
        raise ValueError(f"Active dataset channels lack valid physical identity: {missing}.")
    profile = dataset.preprocessing
    model_kwargs = recipe.model_kwargs(
        n_channels=dataset.n_channels,
        channel_names=identity.names,
        tmin_ms=profile.tmin_ms,
        tmax_ms=profile.tmax_ms,
        sfreq=profile.sfreq,
        n_time=profile.n_times,
        baseline_mode=profile.baseline_mode,
        trial_reference_window_ms=profile.trial_reference_window_ms,
        trial_reference_center=profile.trial_reference_center,
        trial_reference_scale=profile.trial_reference_scale,
        channel_positions_m=tuple(tuple(float(value) for value in row) for row in identity.coords),
        overrides=model_overrides,
    )
    if frozen_erp_prior is not None:
        if frozen_erp_prior.get("calibration_scope") != "independent_development":
            raise ValueError(
                "A frozen ERP prior must declare calibration_scope=independent_development."
            )
        for key in ("tau0_ms", "tau0_bounds", "sigma_bounds"):
            if key not in frozen_erp_prior:
                raise ValueError(f"Frozen ERP prior lacks {key!r}.")
        model_kwargs.update(
            tau0_ms=tuple(float(value) for value in frozen_erp_prior["tau0_ms"]),
            tau0_bounds=tuple(
                tuple(float(value) for value in bounds)
                for bounds in frozen_erp_prior["tau0_bounds"]
            ),
            sigma_bounds=tuple(
                tuple(float(value) for value in bounds)
                for bounds in frozen_erp_prior["sigma_bounds"]
            ),
            dtau_bounds=tuple(
                tuple(float(value) for value in bounds)
                for bounds in frozen_erp_prior.get("dtau_bounds", PCW_CANONICAL_DTAU_BOUNDS)
            ),
        )
    trainer_config = recipe.trainer_config(
        BINARY_ODDBALL_TASK,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        overrides=trainer_overrides,
    )
    return N2P3NetBaseline(
        model_kwargs=model_kwargs,
        trainer_kwargs=asdict(trainer_config),
        E_chn=torch.from_numpy(identity.embedding),
        channel_mask=torch.from_numpy(np.asarray(dataset.channel_mask, dtype=bool)),
        val_subject_frac=validation_subject_fraction,
        erp_calibrator=(
            FoldERPCalibrator(
                EpochTimeAxis(
                    profile.tmin_ms,
                    profile.tmax_ms,
                    profile.sfreq,
                    profile.n_times,
                ),
                identity.names,
                channel_mask=tuple(active_mask.tolist()),
            )
            if fold_erp_calibration
            else None
        ),
        device=device,
    )


def build_binary_model(
    model_name: str,
    dataset: EpochDataset,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    validation_subject_fraction: float = 0.1,
    fold_erp_calibration: bool = True,
    frozen_erp_prior: Mapping[str, Any] | None = None,
    trainer_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    deep_config_overrides: Mapping[str, Any] | None = None,
    device: torch.device | None = None,
    recipe: NeuralRideRecipe = NEURAL_RIDE_V12,
):
    """Build any registered binary model from the universal epoch contract.

    Dataset runners select a model name and provide dataset metadata; all
    architecture-specific construction stays here. This keeps new epoch
    datasets from growing another copy of the model-selection branch.
    """

    _validate_binary_dataset(dataset)
    key = str(model_name).strip().lower()
    if key == "n2p3net":
        return build_binary_neural_ride_adapter(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            validation_subject_fraction=validation_subject_fraction,
            fold_erp_calibration=fold_erp_calibration,
            frozen_erp_prior=frozen_erp_prior,
            trainer_overrides=trainer_overrides,
            model_overrides=model_overrides,
            device=device,
            recipe=recipe,
        )
    if key not in DEEP_MODEL_NAMES:
        raise ValueError(f"Unknown binary model {model_name!r}; choose from {BINARY_MODEL_NAMES}.")

    config_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "val_subject_frac": validation_subject_fraction,
    }
    config_kwargs.update(dict(deep_config_overrides or {}))
    return DeepBaseline(
        key,
        n_chans=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        channel_mask=(
            None
            if getattr(dataset, "channel_mask", None) is None
            else np.asarray(dataset.channel_mask, dtype=bool)
        ),
        config=DeepConfig(**config_kwargs),
        device=device,
    )
