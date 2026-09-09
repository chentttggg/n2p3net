"""Calibration-only epoch selection with independent full-calibration refitting."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from transfer.calibration_candidates import CalibrationCandidateFitCache, fit_calibration_candidate
from transfer.calibration_split import dataset_calibration_inner_split
from transfer.recipes import TargetRecipe, target_batch_budget


@dataclass(frozen=True)
class EpochSelectionPolicy:
    mode: str = "fixed"
    patience: int = 6
    min_delta: float = 1e-6

    def __post_init__(self):
        if self.mode not in {"fixed", "full_trajectory", "patience"}:
            raise ValueError("unsupported target epoch selection mode")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("epoch selection patience must be a positive integer")
        if isinstance(self.min_delta, bool) or not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("epoch selection min_delta must be finite and nonnegative")

    def record(self):
        return asdict(self)


def epoch_selection_recipes(*, batch_size=512, seed=20260906, epochs=60):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("target epochs must be a positive integer")
    return {
        "full": TargetRecipe("es_full_high", batch_size=batch_size, seed=seed, epochs=epochs),
        "head": TargetRecipe("es_head_high", weights="head", bn_affine="frozen", bn_buffers="source",
                             batch_size=batch_size, seed=seed, epochs=epochs),
        "scratch": TargetRecipe("es_scratch_high", initialization="scratch", batch_size=batch_size, seed=seed,
                                epochs=epochs),
    }


def epoch_selection_budget(recipe, policy, *, calibration_rows, inner_rows, inner_legal):
    selecting = policy.mode != "fixed" and inner_legal
    refit = target_batch_budget(calibration_rows, recipe.batch_size, epochs=recipe.epochs)
    inner = target_batch_budget(inner_rows, recipe.batch_size, epochs=recipe.epochs) if selecting else {"updates": 0, "draws": 0}
    return {"fit_calls_upper_bound": 2 if selecting else 1,
        "inner_epochs_upper_bound": recipe.epochs if selecting else 0,
        "full_calibration_epochs_upper_bound": recipe.epochs,
        "optimizer_updates_upper_bound": refit["updates"] + inner["updates"],
        "trial_draws_upper_bound": refit["draws"] + inner["draws"],
        "includes_inner_and_full_calibration": True}


def fit_with_epoch_selection(
    dataset, recipe: TargetRecipe, calibration_mask: np.ndarray, *, policy: EpochSelectionPolicy,
    source_trunk=None, source_input_stats=None, device="cpu", recording_order=None,
    fit_cache: CalibrationCandidateFitCache | None = None,
):
    """Select only on calibration, then discard inner state and refit the winner."""
    if recipe.initialization not in {"source", "scratch"} or recipe.weights == "none":
        raise ValueError("epoch selection requires a trainable target neural model")
    if recipe.max_updates is not None or recipe.refit_full_prefix:
        raise ValueError("epoch selection owns full refitting and requires an epoch budget")
    if not isinstance(policy, EpochSelectionPolicy):
        raise ValueError("an explicit EpochSelectionPolicy is required")
    allowed = np.asarray(calibration_mask)
    if allowed.dtype != bool or allowed.shape != (dataset.n_epochs,) or not allowed.any():
        raise ValueError("calibration_mask must explicitly identify authorized rows")
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    fit = fit_calibration_candidate if fit_cache is None else fit_cache.fit
    common = dict(source_trunk=source_trunk, source_input_stats=source_input_stats, device=device)
    record = {"schema": "n2p3_target_epoch_selection/1", "policy": policy.record(),
        "requested_recipe": recipe.record(), "effective_mode": policy.mode,
        "monitor": "validation_cross_entropy", "selection_evidence": False,
        "fallback": None, "inner_epochs_run": 0, "inner_history": {},
        "inner_wall_seconds": 0.0, "selected_epochs": recipe.epochs,
        "final_test_used_for_selection": False, "refit_initialization": recipe.initialization}
    if policy.mode != "fixed":
        split = dataset_calibration_inner_split(dataset, allowed, recording_order=recording_order)
        record["inner_split"] = split.record()
        if split.legal:
            inner_started = time.perf_counter()
            inner = fit(dataset, recipe, split.training_mask, **common,
                validation_mask=split.validation_mask,
                validation_behavior="select_epoch_full_trajectory" if policy.mode == "full_trajectory" else "select_epoch",
                early_stop_patience=policy.patience, early_stop_min_delta=policy.min_delta,
                **({"phase": "inner"} if fit_cache is not None else {}))
            if cuda:
                torch.cuda.synchronize(torch.device(device))
            history = inner.fit_history
            selected = history["best_epoch"]
            if selected is None or not 1 <= selected <= recipe.epochs:
                raise ValueError("epoch selection produced no valid training epoch")
            record.update(selection_evidence=True, selected_epochs=selected,
                inner_epochs_run=len(history["train_losses"]), inner_history=history,
                inner_wall_seconds=time.perf_counter() - inner_started)
            del inner
        else:
            record.update(effective_mode="fixed", fallback="fixed_requested_recipe", fallback_reason=split.reason)
    chosen = replace(recipe, epochs=record["selected_epochs"])
    refit_started = time.perf_counter()
    fitted = fit(dataset, chosen, allowed, **common,
                 **({"phase": "full_prefix"} if fit_cache is not None else {}))
    if cuda:
        torch.cuda.synchronize(torch.device(device))
    record.update(refit_recipe=chosen.record(), refit_history=fitted.fit_history,
        refit_wall_seconds=time.perf_counter() - refit_started,
        total_wall_seconds=time.perf_counter() - started,
        stopped_early=record["selection_evidence"] and record["inner_epochs_run"] < recipe.epochs)
    histories = [history for history in (record["inner_history"], fitted.fit_history) if history]
    for name, field in (("optimizer_updates", "updates"), ("trial_draws", "draws")):
        record["total_" + name] = sum(history["fit_budget"][field] for history in histories)
        record["physical_" + name] = sum(history["fit_budget"][field] for history in histories
            if history.get("fit_reuse", {}).get("fit_executed", True))
    record["fit_execution"] = {"requested_fits": len(histories),
        "fit_calls": sum(history.get("fit_reuse", {}).get("fit_executed", True) for history in histories),
        "cache_hits": sum(history.get("fit_reuse", {}).get("cache_hit", False) for history in histories)}
    return fitted, record
