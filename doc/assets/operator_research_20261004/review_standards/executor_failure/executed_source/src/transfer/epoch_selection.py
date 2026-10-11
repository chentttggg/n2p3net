"""Calibration-only epoch selection with independent full-calibration refitting."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from train.early_stopping import DEFAULT_EARLY_STOP_PATIENCE, PatienceStopper
from transfer.calibration_candidates import CalibrationCandidateFitCache, fit_calibration_candidate
from transfer.calibration_split import dataset_calibration_inner_split
from transfer.recipes import TargetRecipe, target_batch_budget


@dataclass(frozen=True)
class IndependentEpochSelectionData:
    dataset: object
    training_mask: np.ndarray
    validation_mask: np.ndarray
    evidence: dict

    @classmethod
    def from_views(cls, training, validation, calibration, *, channel_names=None):
        from data.epochs import (
            concatenate_epoch_datasets,
            select_epoch_channels,
        )
        from data.preprocess import RawSampleScope
        from data.source_records import target_raw_view_identity
        from research.contracts import semantic_sha256

        def supports(view):
            from data.epochs import validate_preprocessing_execution

            declared = view.provenance.get("target_raw_record_scopes")
            if declared is None:
                return None
            receipts = validate_preprocessing_execution(view.provenance.get("preprocessing_execution"), view.preprocessing)
            receipt_by_hash = {receipt["receipt_sha256"]: receipt for receipt in receipts}
            records = {}
            for entry in declared:
                receipt = receipt_by_hash.get(entry["receipt_sha256"])
                scope = RawSampleScope.from_record(entry["raw_scope"])
                if receipt is None or receipt["source_context"].get("raw_scope", {}).get("declaration") != scope.record():
                    raise ValueError("independent raw record support differs from actual preprocessing receipt")
                key = semantic_sha256(entry["raw_identity"])
                records.setdefault(key, []).append(scope)
            if {entry["receipt_sha256"] for entry in declared} != set(receipt_by_hash):
                raise ValueError("independent raw record supports must cover every preprocessing receipt")
            return records

        parent_records, training_records, validation_records = (supports(view) for view in (calibration, training, validation))
        if parent_records is None and training_records is None and validation_records is None:
            parent = RawSampleScope.from_record(calibration.provenance["source_preparation"]["raw_scope"])
            a = RawSampleScope.from_record(training.provenance["source_preparation"]["raw_scope"])
            b = RawSampleScope.from_record(validation.provenance["source_preparation"]["raw_scope"])
            parent_records, training_records, validation_records = ({"single": [parent]}, {"single": [a]}, {"single": [b]})
        elif any(value is None for value in (parent_records, training_records, validation_records)):
            raise ValueError("independent raw supports require consistent physical record identity")
        if set(training_records) | set(validation_records) != set(parent_records):
            raise ValueError("independent validation raw recordings must equal authorized calibration recordings")
        for key, parents in parent_records.items():
            if len(parents) != 1:
                raise ValueError("calibration must declare one contiguous scope per physical recording")
            parent = parents[0]
            parts = sorted(training_records.get(key, []) + validation_records.get(key, []), key=lambda value: value.start_sample)
            if (parts[0].start_sample != parent.start_sample or parts[-1].stop_sample != parent.stop_sample
                    or any(left.stop_sample != right.start_sample for left, right in zip(parts, parts[1:], strict=False))
                    or tuple(index for part in parts for index in part.event_indices) != parent.event_indices):
                raise ValueError("independent validation raw support must partition calibration")
        authority = calibration.identity_table.authority_keys()
        for view in (training, validation):
            if (view.identity_table.authority_keys() != authority or view.preprocessing != calibration.preprocessing
                    or view.channel_names != calibration.channel_names or set(np.unique(view.y)) != {0, 1}):
                raise ValueError("independent validation requires matching participants/contracts and both classes")
        shared = {}
        for field in ("source_reference", "event_contract", "candidate_task_contract", "candidate_vocabulary", "session_schema"):
            value = calibration.provenance.get(field)
            if any(view.provenance.get(field) != value for view in (training, validation)):
                raise ValueError(f"independent validation physical contracts differ: {field}")
            if value is not None:
                shared[field] = value
        combined = concatenate_epoch_datasets((training, validation), name="independent_inner_validation", provenance=shared)
        mask = np.arange(combined.n_epochs) < training.n_epochs
        evidence = {"legal": True, "kind": "independent_raw_training_and_validation",
            "training_raw_scopes": {key: [value.record() for value in rows] for key, rows in training_records.items()},
            "validation_raw_scopes": {key: [value.record() for value in rows] for key, rows in validation_records.items()},
            "parent_raw_scopes": {key: [value.record() for value in rows] for key, rows in parent_records.items()},
            "training_input": target_raw_view_identity(training),
            "validation_input": target_raw_view_identity(validation),
            "training_rows": int(mask.sum()), "validation_rows": int((~mask).sum())}
        if channel_names is not None:
            combined = select_epoch_channels(combined, channel_names)
        return cls(combined, mask, ~mask, evidence)


@dataclass(frozen=True)
class EpochSelectionPolicy:
    mode: str = "fixed"
    patience: int = DEFAULT_EARLY_STOP_PATIENCE
    min_delta: float = 1e-6
    require_validation: bool = False

    def __post_init__(self):
        if self.mode not in {"fixed", "full_trajectory", "patience"}:
            raise ValueError("unsupported target epoch selection mode")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("epoch selection patience must be a positive integer")
        if isinstance(self.min_delta, bool) or not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("epoch selection min_delta must be finite and nonnegative")
        PatienceStopper(self.patience, self.min_delta)
        if type(self.require_validation) is not bool:
            raise ValueError("require_validation must be boolean")

    def record(self):
        record = asdict(self)
        if not self.require_validation:
            record.pop("require_validation")
        return record


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
    selection_data: IndependentEpochSelectionData | None = None,
    epoch_quality_policy=None, epoch_quality_model=None,
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
    common = dict(source_trunk=source_trunk, source_input_stats=source_input_stats, device=device,
                  epoch_quality_policy=epoch_quality_policy, epoch_quality_model=epoch_quality_model)
    record = {"schema": "n2p3_target_epoch_selection/1", "policy": policy.record(),
        "requested_recipe": recipe.record(), "effective_mode": policy.mode,
        "monitor": "validation_cross_entropy", "selection_evidence": False,
        "fallback": None, "inner_epochs_run": 0, "inner_history": {},
        "inner_wall_seconds": 0.0, "selected_epochs": recipe.epochs,
        "final_test_used_for_selection": False, "refit_initialization": recipe.initialization}
    if policy.mode != "fixed":
        if selection_data is None:
            split = dataset_calibration_inner_split(dataset, allowed, recording_order=recording_order)
            inner_dataset = dataset
            record["inner_split"] = split.record()
        else:
            from types import SimpleNamespace

            inner_dataset = selection_data.dataset
            parent_timeline, inner_timeline = dataset.event_timeline, inner_dataset.event_timeline
            authorized = set(parent_timeline.event_ids[np.isin(parent_timeline.evidence_indices,
                                                               np.flatnonzero(allowed))].tolist())
            consumed = set(inner_timeline.event_ids[inner_timeline.evidence_indices >= 0].tolist())
            if not consumed <= authorized:
                raise ValueError("independent early-stopping data escape authorized calibration events")
            split = SimpleNamespace(legal=True, training_mask=selection_data.training_mask,
                                    validation_mask=selection_data.validation_mask)
            record["inner_split"] = selection_data.evidence
        if split.legal:
            inner_started = time.perf_counter()
            inner = fit(inner_dataset, recipe, split.training_mask, **common,
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
            if policy.require_validation:
                raise ValueError(f"global early stopping requires a legal calibration split: {split.reason}")
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
    record["reached_epoch_limit"] = record["selection_evidence"] and record["inner_epochs_run"] >= recipe.epochs
    record["stop_reason"] = record["inner_history"].get("stopping", {}).get(
        "stop_reason", "patience" if record["stopped_early"] else "epoch_limit"
    ) if record["selection_evidence"] else "fixed_budget"
    histories = [history for history in (record["inner_history"], fitted.fit_history) if history]
    for name, field in (("optimizer_updates", "updates"), ("trial_draws", "draws")):
        record["total_" + name] = sum(history["fit_budget"][field] for history in histories)
        record["physical_" + name] = sum(history["fit_budget"][field] for history in histories
            if history.get("fit_reuse", {}).get("fit_executed", True))
    record["fit_execution"] = {"requested_fits": len(histories),
        "fit_calls": sum(history.get("fit_reuse", {}).get("fit_executed", True) for history in histories),
        "cache_hits": sum(history.get("fit_reuse", {}).get("cache_hit", False) for history in histories)}
    return fitted, record
