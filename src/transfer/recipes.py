"""Declarative target treatments shared by public and product calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from math import ceil

import numpy as np

from models.adapters import FeatureTransferConfig
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
    target_adapter_config: FeatureTransferConfig | dict | None = None
    candidate_loss_weight: float | None = None
    candidate_loss_temperature: float = 1.0
    source_prior_weight: float = 0.0

    def __post_init__(self) -> None:
        adapter = self.target_adapter_config
        if isinstance(adapter, dict):
            adapter = FeatureTransferConfig(**adapter)
        if self.weights == "adapter":
            adapter = FeatureTransferConfig() if adapter is None else adapter
            if not isinstance(adapter, FeatureTransferConfig) or (
                adapter.source_domain_count or adapter.private_domain_count
            ):
                raise ValueError("target residual recipes require an unindexed FeatureTransferConfig")
        elif adapter is not None:
            raise ValueError("target_adapter_config requires adapter weights")
        object.__setattr__(self, "target_adapter_config", adapter)
        for name in ("candidate_loss_weight", "candidate_loss_temperature", "source_prior_weight"):
            value = getattr(self, name)
            if name == "candidate_loss_weight" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar")
        if self.candidate_loss_temperature <= 0:
            raise ValueError("candidate_loss_temperature must be positive")
        options = {
            "initialization": {"source", "scratch", "waveform_lda", "source_feature_lda"},
            "input_statistics": {"source", "target_prefix"},
            "bn_buffers": {"source", "calibration", "online"},
            "bn_affine": {"frozen", "trainable"},
            "weights": {"none", "head", "non_bn", "spatial", "temporal", "adapter"},
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
        if self.initialization in {"waveform_lda", "source_feature_lda"} and (
            self.candidate_loss_weight is not None or self.source_prior_weight != 0
        ):
            raise ValueError("LDA does not optimize decision or source-prior losses")
        if self.weights == "none" and (
            self.bn_affine != "frozen" or self.bn_buffers == "online" or self.dropout != "off"
        ):
            raise ValueError("statistics-only treatments cannot update affine or online buffers")
        self.adapter_config()

    def record(self) -> dict[str, object]:
        record = asdict(self)
        if self.target_adapter_config is None:
            record.pop("target_adapter_config")
        if self.candidate_loss_weight is None:
            record.pop("candidate_loss_weight")
            if self.candidate_loss_temperature == 1.0:
                record.pop("candidate_loss_temperature")
        if self.source_prior_weight == 0:
            record.pop("source_prior_weight")
        return record

    def adapter_config(
        self, *, input_mean: np.ndarray | None = None, input_std: np.ndarray | None = None
    ) -> SubjectAdapterConfig:
        return SubjectAdapterConfig(
            head_kind=("adapter" if self.weights == "adapter" else
                       "full_fine" if self.weights in {"non_bn", "spatial", "temporal"}
                       else "classifier_fine"),
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
            candidate_loss_weight=0. if self.candidate_loss_weight is None else self.candidate_loss_weight,
            candidate_loss_temperature=self.candidate_loss_temperature,
            source_prior_weight=self.source_prior_weight,
        )


def actual_batch_recipes(recipes: tuple[TargetRecipe, ...], n_fit: int) -> tuple[TargetRecipe, ...]:
    """Keep the first requested cap for otherwise identical actual treatments."""
    if n_fit < 1:
        raise ValueError("n_fit must be positive")
    unique: dict[str, TargetRecipe] = {}
    for recipe in recipes:
        fields = recipe.record()
        fields.pop("name")
        fields["batch_size"] = min(recipe.batch_size, n_fit)
        unique.setdefault(json.dumps(fields, sort_keys=True), recipe)
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


def p1_target_recipes(
    *, batch_size: int = 512, seed: int = 0, max_updates: int = 200
) -> tuple[TargetRecipe, ...]:
    """Spatial x temporal adaptation, with independent BN and strong controls."""
    base = TargetRecipe("p1_full_fine", batch_size=batch_size, seed=seed,
                        max_updates=max_updates)
    modes = {(0, 0): "head", (1, 0): "spatial", (0, 1): "temporal", (1, 1): "non_bn"}
    factorial = tuple(replace(base, name=f"p1_s{s}t{t}", weights=mode,
                              bn_affine="frozen", dropout="off")
                      for (s, t), mode in modes.items())
    return (base, replace(base, name="p1_scratch", initialization="scratch"), *factorial,
            replace(base, name="p1_affine_head", weights="head", dropout="off"),
            replace(base, name="p1_residual_head", weights="adapter",
                    bn_affine="frozen", dropout="off"),
            replace(base, name="p1_waveform_lda", initialization="waveform_lda",
                    weights="none", bn_buffers="source", bn_affine="frozen",
                    dropout="off", max_updates=None))


def p2_target_recipes(*, batch_size: int = 512, seed: int = 0) -> tuple[TargetRecipe, ...]:
    """Matched full/scratch/LDA controls for each declared source objective."""
    return tuple(replace(recipe, name=recipe.name.replace("p1_", "p2_", 1))
                 for recipe in p1_target_recipes(batch_size=batch_size, seed=seed)
                 if recipe.name in {"p1_full_fine", "p1_scratch", "p1_waveform_lda"})


def unified_target_recipes(
    *, membership_available: bool, source_prior_weight: float,
    batch_size: int = 512, seed: int = 0, max_updates: int | None = None,
    decision_weight: float = 1., epochs: int = 90,
) -> tuple[TargetRecipe, ...]:
    """Two adaptation packages, supervision-matched scratch, and waveform LDA."""
    if not isinstance(membership_available, bool):
        raise ValueError("membership_available must be an explicit boolean admission")
    if isinstance(decision_weight, bool) or not np.isfinite(decision_weight) or decision_weight <= 0:
        raise ValueError("unified decision_weight must be finite and positive")
    base = TargetRecipe("unified_q0", batch_size=batch_size, seed=seed, max_updates=max_updates, epochs=epochs,
                        input_statistics="target_prefix", bn_buffers="calibration",
                        bn_affine="trainable", dropout="off", source_prior_weight=0.,
                        candidate_loss_weight=0. if membership_available else None)
    joint = replace(base, name="unified_q1", source_prior_weight=source_prior_weight,
                    candidate_loss_weight=decision_weight if membership_available else None)
    ce_control = (replace(base, name="unified_scratch_ce", initialization="scratch"),) if membership_available else ()
    return (base, joint, replace(joint, name="unified_scratch", initialization="scratch", source_prior_weight=0.),
            *ce_control,
            TargetRecipe("unified_waveform_lda", initialization="waveform_lda", weights="none",
                         bn_buffers="source", bn_affine="frozen", dropout="off", epochs=1, batch_size=batch_size, seed=0))


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
