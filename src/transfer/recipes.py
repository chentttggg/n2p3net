"""Declarative target treatments shared by public and product calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import ceil

import numpy as np

from transfer.subject_adapter import SubjectAdapterConfig


@dataclass(frozen=True)
class TargetRecipe:
    name: str
    initialization: str = "source"
    input_statistics: str = "target_prefix"
    bn_buffers: str = "online"
    bn_affine: str = "trainable"
    weights: str = "non_bn"
    dropout: str = "checkpoint"
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    max_updates: int | None = None
    seed: int = 0
    refit_full_prefix: bool = False

    def __post_init__(self) -> None:
        options = {
            "initialization": {"source", "scratch", "waveform_lda", "source_feature_lda"},
            "input_statistics": {"source", "target_prefix"},
            "bn_buffers": {"source", "calibration", "online"},
            "bn_affine": {"frozen", "trainable"},
            "weights": {"none", "head", "non_bn"},
            "dropout": {"off", "checkpoint"},
        }
        for name, values in options.items():
            if getattr(self, name) not in values:
                raise ValueError(f"unsupported target recipe {name}: {getattr(self, name)!r}")
        if not self.name.strip():
            raise ValueError("target recipe requires a name")
        if self.initialization == "scratch" and self.input_statistics != "target_prefix":
            raise ValueError("scratch must fit its own calibration input statistics")
        if self.initialization in {"waveform_lda", "source_feature_lda"} and (
            self.weights != "none" or self.bn_affine != "frozen" or self.dropout != "off"
        ):
            raise ValueError("LDA feature transforms must have frozen parameters and dropout off")
        if self.weights == "none" and (
            self.bn_affine != "frozen" or self.bn_buffers == "online" or self.dropout != "off"
        ):
            raise ValueError("statistics-only treatments cannot update affine or online buffers")
        self.adapter_config()

    def record(self) -> dict[str, object]:
        return asdict(self)

    def adapter_config(
        self, *, input_mean: np.ndarray | None = None, input_std: np.ndarray | None = None
    ) -> SubjectAdapterConfig:
        return SubjectAdapterConfig(
            head_kind="full_fine" if self.weights == "non_bn" else "classifier_fine",
            input_statistics=self.input_statistics,
            input_mean=input_mean,
            input_std=input_std,
            batchnorm_buffers=self.bn_buffers,
            freeze_batchnorm_stats=self.bn_buffers != "online",
            batchnorm_affine=self.bn_affine,
            trainable_weights=self.weights,
            dropout_mode=self.dropout,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            max_updates=self.max_updates,
            seed=self.seed,
            val_group_fraction=None,
            refit_full_prefix=self.refit_full_prefix,
        )


def actual_batch_recipes(recipes: tuple[TargetRecipe, ...], n_fit: int) -> tuple[TargetRecipe, ...]:
    """Keep the first requested cap for otherwise identical actual treatments."""
    if n_fit < 1:
        raise ValueError("n_fit must be positive")
    unique: dict[tuple[tuple[str, object], ...], TargetRecipe] = {}
    for recipe in recipes:
        fields = recipe.record()
        fields.pop("name")
        fields["batch_size"] = min(recipe.batch_size, n_fit)
        unique.setdefault(tuple(fields.items()), recipe)
    return tuple(unique.values())


def target_batch_budget(
    n_fit: int, batch_size: int, *, epochs: int = 60, max_updates: int | None = None
) -> dict[str, object]:
    """Expected ordinary-CE exposure, including a possible partial final traversal."""
    if n_fit < 1 or batch_size < 1 or epochs < 1 or (max_updates is not None and max_updates < 1):
        raise ValueError("row count, batch, epochs and updates must be positive")
    per_epoch = ceil(n_fit / batch_size)
    updates = epochs * per_epoch if max_updates is None else max_updates
    full, partial = divmod(updates, per_epoch)
    return {
        "n_fit": n_fit,
        "requested_batch_size": batch_size,
        "effective_batch_size": min(batch_size, n_fit),
        "budget_mode": "epochs" if max_updates is None else "updates",
        "updates": updates,
        "draws": full * n_fit + min(partial * batch_size, n_fit),
        "complete_epochs": full,
        "partial_epoch_updates": partial,
        "reference_updates": epochs * ceil(n_fit / 128),
    }


def batch_size_aliases(
    n_fit: int, batch_size: int, *, batch_sizes: tuple[int, ...] = (32, 128, 512)
) -> tuple[int, ...]:
    if n_fit < 1 or batch_size < 1 or any(value < 1 for value in batch_sizes):
        raise ValueError("row counts and batches must be positive")
    return tuple(value for value in batch_sizes if min(value, n_fit) == min(batch_size, n_fit))


def b0_target_recipes(
    n_fit: int,
    *,
    seed: int = 0,
    budget_mode: str | None = None,
    epochs: int = 60,
    batch_sizes: tuple[int, ...] = (32, 128, 512),
) -> tuple[TargetRecipe, ...]:
    """Paired 60-pass and fixed-update comparisons; no data fitting occurs here."""
    if n_fit < 1:
        raise ValueError("n_fit must be positive")
    if budget_mode not in {None, "epochs", "updates"}:
        raise ValueError("budget_mode must be epochs, updates or None for both")
    return actual_batch_recipes(
        tuple(
            TargetRecipe(
                name=f"b0_{budget}_b{batch}_{initialization}",
                initialization=initialization,
                batch_size=batch,
                epochs=epochs,
                max_updates=epochs * ceil(n_fit / 128) if budget == "updates" else None,
                seed=seed,
            )
            for budget, batch, initialization in product(
                ("epochs", "updates") if budget_mode is None else (budget_mode,),
                batch_sizes,
                ("source", "scratch"),
            )
        ),
        n_fit,
    )


def compact_target_recipes(*, batch_size: int = 512, seed: int = 20260906) -> tuple[TargetRecipe, ...]:
    """Three positive-K baselines and three one-factor transfer ablations."""
    if type(batch_size) is not int or batch_size != 512:
        raise ValueError("the compact matrix requires the user-fixed batch cap 512")
    base = TargetRecipe(name="compact_full_fine", batch_size=batch_size, seed=seed)
    return (
        replace(
            base, name="compact_waveform_lda", initialization="waveform_lda",
            weights="none", bn_buffers="source", bn_affine="frozen", dropout="off",
        ),
        replace(base, name="compact_scratch", initialization="scratch"),
        base,
        replace(base, name="compact_source_statistics", input_statistics="source"),
        replace(base, name="compact_frozen_bn_buffers", bn_buffers="source"),
        # Only classifier parameters train; online BN buffers still adapt.
        replace(base, name="compact_head_only", weights="head", bn_affine="frozen"),
    )


def e1_target_recipes(*, batch_size: int = 32, seed: int = 0) -> tuple[TargetRecipe, ...]:
    """All positive-K E1 scores; I0B0 statistics-only reuses the separate K0 score."""
    base = TargetRecipe(name="e1_base", batch_size=batch_size, seed=seed)
    recipes = [
        replace(
            base,
            name=f"e1_i{i}b{b}w{w}",
            input_statistics="target_prefix" if i else "source",
            bn_buffers="calibration" if b else "source",
            bn_affine="frozen",
            weights="non_bn" if w else "head",
            dropout="off",
        )
        for i, b, w in product((0, 1), repeat=3)
    ]
    for i, b in ((0, 1), (1, 0), (1, 1)):
        recipes.append(
            replace(
                base,
                name=f"e1_stats_i{i}b{b}",
                input_statistics="target_prefix" if i else "source",
                bn_buffers="calibration" if b else "source",
                bn_affine="frozen",
                weights="none",
                dropout="off",
            )
        )
    recipes.extend(
        replace(
            base, name=f"e1_affine_b{b}", bn_buffers="calibration" if b else "source", dropout="off"
        )
        for b in (0, 1)
    )
    recipes.extend(
        (
            replace(base, name="e1_online_bn", dropout="off"),
            replace(base, name="e1_dropout"),
            replace(
                base, name="e1_product_statistics", input_statistics="source", bn_buffers="source"
            ),
            replace(base, name="e1_scratch_no_dropout", initialization="scratch", dropout="off"),
            replace(base, name="e1_scratch_dropout", initialization="scratch"),
            replace(
                base,
                name="e1_waveform_lda",
                initialization="waveform_lda",
                weights="none",
                bn_buffers="source",
                bn_affine="frozen",
                dropout="off",
            ),
        )
    )
    for i, b in ((0, 0), (1, 1)):
        recipes.append(
            replace(
                base,
                name=f"e1_feature_lda_i{i}b{b}",
                initialization="source_feature_lda",
                input_statistics="target_prefix" if i else "source",
                bn_buffers="calibration" if b else "source",
                bn_affine="frozen",
                weights="none",
                dropout="off",
            )
        )
    if batch_size != 32:
        recipes.extend(
            (
                replace(base, name="e1_history_source_b32", batch_size=32),
                replace(
                    base, name="e1_history_scratch_b32", initialization="scratch", batch_size=32
                ),
            )
        )
    return tuple(recipes)
