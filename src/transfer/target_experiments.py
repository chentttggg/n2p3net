"""Public target experiments with calibration-only selection and explicit costs."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.contracts import semantic_sha256
from transfer.result_reuse import (
    ResultIntegrityError,
    read_result_artifacts,
    read_result_record,
    seal_result_record,
)


def user_fixed_target_batch_rule(*, batch_size: int = 512) -> dict:
    """Record an explicit prescribed cap without claiming validation selection."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("user-fixed batch must be a positive integer")
    return dict(schema="n2p3_target_batch_rule/1", batch_size=batch_size,
                policy="user_fixed", provenance="user_explicit", selection_evidence=False,
                selection_scope="user_prescribed_no_score_selection",
                evidence_level="prescribed_configuration", target_device_performance_confirmed=False)


def operator_fixed_target_batch_rule(*, batch_size: int, rationale: str) -> dict:
    """Declare a run configuration without attributing its exact cap to the user."""
    if type(batch_size) is not int or batch_size < 1 or not rationale.strip():
        raise ValueError("operator-fixed batch needs a positive cap and rationale")
    return dict(schema="n2p3_target_batch_rule/1", batch_size=batch_size,
                policy="operator_fixed", provenance="run_configuration", selection_evidence=False,
                selection_scope="predeclared_no_score_selection", rationale=rationale,
                evidence_level="prescribed_configuration", target_device_performance_confirmed=False)


def resolve_target_batch_rule(
    plan: dict, *, rule_path: Path | None = None, requested_batch: int | None = None,
    dry_run: bool = False,
) -> tuple[int, dict]:
    """Validate a B0 selection or explicitly prescribed run configuration."""
    path = rule_path or plan.get("batch_rule_file")
    rule = json.loads(Path(path).read_text(encoding="utf-8")) if path else plan.get("batch_rule")
    if rule is None:
        rule = freeze_target_batch([])
    if (not isinstance(rule, dict) or rule.get("schema") != "n2p3_target_batch_rule/1"
            or type(rule.get("batch_size")) is not int or rule["batch_size"] < 1
            or type(rule.get("selection_evidence")) is not bool):
        raise ValueError("a versioned valid target batch rule is required")
    user_fixed = rule.get("policy") == "user_fixed"
    operator_fixed = rule.get("policy") == "operator_fixed"
    if operator_fixed and (
        rule.get("provenance") != "run_configuration" or rule["selection_evidence"]
        or rule.get("selection_scope") != "predeclared_no_score_selection"
        or not isinstance(rule.get("rationale"), str) or not rule["rationale"].strip()
    ):
        raise ValueError("operator-fixed batch requires rationale and no selection claim")
    if user_fixed and (rule.get("provenance") != "user_explicit" or rule["selection_evidence"]
                       or rule.get("selection_scope") != "user_prescribed_no_score_selection"):
        raise ValueError("user-fixed batch requires explicit provenance and no selection claim")
    if not (user_fixed or operator_fixed) and not rule["selection_evidence"] and rule["batch_size"] != 32:
        raise ValueError("without selection evidence the predeclared fallback is batch32")
    if rule["selection_evidence"] and (rule.get("cohort_validation") != "predeclared_complete"
                                       or not rule.get("batch_cohort_digest")):
        raise ValueError("selected B0 rule requires complete predeclared cohort evidence")
    batch = requested_batch if requested_batch is not None else plan.get("target_batch_size", rule["batch_size"])
    if type(batch) is not int or batch < 1:
        raise ValueError("requested target batch must be a positive integer")
    if batch != rule["batch_size"] and (user_fixed or operator_fixed or not dry_run):
        rule_kind = "user-fixed" if user_fixed else "operator-fixed" if operator_fixed else "B0"
        raise ValueError(f"target batch differs from the frozen {rule_kind} rule")
    return batch, rule


def build_target_tasks(
    dataset, splits: dict, *, stage: str, batch_size: int = 32, seed: int = 20260906,
    recording_order=None, include_historical_controls: bool = True,
    epoch_selection: dict | None = None,
) -> dict:
    """Materialize target work without fitting statistics, models or selectors."""
    from transfer.calibration_split import dataset_calibration_inner_split
    from transfer.recipes import (
        b0_target_recipes,
        batch_size_aliases,
        compact_target_recipes,
        e1_target_recipes,
        target_batch_budget,
    )

    if stage not in {"B0", "B0_score", "E1", "E3", "compact", "ES"}:
        raise ValueError("unsupported target training stage")
    if epoch_selection is not None and stage != "ES":
        raise ValueError("epoch selection parameters require the ES stage")
    if stage == "ES":
        from transfer.epoch_selection import (
            EpochSelectionPolicy,
            epoch_selection_budget,
            epoch_selection_recipes,
        )

        policy_parameters = dict(epoch_selection or {})
        epochs = policy_parameters.pop("epochs", 60)
        treatments = policy_parameters.pop("treatments", ["full", "head", "scratch"])
        modes = policy_parameters.pop("modes", ["fixed", "full_trajectory", "patience"])
        include_baselines = policy_parameters.pop("include_baselines", True)
        es_recipes = epoch_selection_recipes(batch_size=batch_size, seed=seed, epochs=epochs)
        for name, values, allowed in (("treatments", treatments, set(es_recipes)),
                                      ("modes", modes, {"fixed", "full_trajectory", "patience"})):
            if (not isinstance(values, (list, tuple)) or not values
                    or any(not isinstance(value, str) for value in values)
                    or len(set(values)) != len(values) or set(values) - allowed):
                raise ValueError(f"ES {name} must be a nonempty unique list of supported values")
        if type(include_baselines) is not bool:
            raise ValueError("ES include_baselines must be boolean")
        if set(policy_parameters) - {"patience", "min_delta"}:
            raise ValueError("ES permits only predeclared patience and min_delta parameters")
        EpochSelectionPolicy(**policy_parameters)
    if type(include_historical_controls) is not bool:
        raise ValueError("include_historical_controls must be boolean")
    if stage == "compact":
        compact_target_recipes(batch_size=batch_size, seed=seed)
    tasks, scopes, exclusions = [], {}, []
    for subject, (rows, subject_splits) in splits.items():
        participant = dataset.identity_table.record_for(subject).authority_key()
        for budget, split in subject_splits.items():
            key = f"{subject}:{budget}"
            calibration = np.asarray(rows)[split.calibration_rows]
            test = np.asarray(rows)[split.test_rows]
            allowed = np.zeros(dataset.n_epochs, dtype=bool)
            allowed[calibration] = True
            scope = dict(subject=subject, participant_key=participant, K=budget,
                         calibration_rows=calibration.tolist(), test_rows=test.tolist(),
                         purged_rows=np.asarray(rows)[split.purged_rows].tolist())
            scopes[key] = scope
            common = dict(scope_id=key, participant_key=participant, subject=subject, K=budget)
            if budget == 0:
                if stage != "B0" and (stage != "ES" or include_baselines):
                    tasks.append({**common, "arm": "zero_shot", "recipe": None,
                                  "training_rows": 0, "validation_rows": 0})
                continue
            inner = dataset_calibration_inner_split(dataset, allowed, recording_order=recording_order)
            scope.update(inner_split=inner.record(),
                         inner_training_rows=np.flatnonzero(inner.training_mask).tolist(),
                         inner_validation_rows=np.flatnonzero(inner.validation_mask).tolist())
            if stage == "B0" and not inner.legal:
                exclusions.append({**common, "status": "no_legal_inner_split", "reason": inner.reason})
                continue
            n_fit = int(inner.training_mask.sum()) if stage == "B0" else len(calibration)
            if stage in {"B0", "B0_score"}:
                recipes = b0_target_recipes(n_fit=n_fit, seed=seed)
            elif stage == "E1":
                recipes = e1_target_recipes(batch_size=batch_size, seed=seed)
                if not include_historical_controls:
                    recipes = tuple(recipe for recipe in recipes if recipe.batch_size == batch_size)
            elif stage == "compact":
                recipes = compact_target_recipes(batch_size=batch_size, seed=seed)
            elif stage == "ES":
                from transfer.calibration_candidates import waveform_lda_recipe

                for treatment in treatments:
                    recipe = es_recipes[treatment]
                    for mode in modes:
                        policy = EpochSelectionPolicy(mode, **policy_parameters)
                        tasks.append({**common, "arm": f"es_{treatment}_{mode}", "treatment": treatment,
                            "recipe": recipe.record(), "epoch_selection": policy.record(),
                            "training_rows": n_fit,
                            "validation_rows": int(inner.validation_mask.sum()) if mode != "fixed" and inner.legal else 0,
                            "budget": target_batch_budget(n_fit, recipe.batch_size, epochs=recipe.epochs),
                            "selection_budget": epoch_selection_budget(recipe, policy, calibration_rows=n_fit,
                                inner_rows=int(inner.training_mask.sum()), inner_legal=inner.legal)})
                if include_baselines:
                    lda = waveform_lda_recipe(batch_size=batch_size, seed=seed)
                    tasks.append({**common, "arm": "es_waveform_lda", "recipe": lda.record(),
                        "training_rows": n_fit, "validation_rows": 0, "budget": {"updates": 0, "draws": 0}})
                continue
            else:
                from transfer.calibration_candidates import calibration_candidate_recipes

                for mode in ("legacy_product", "transfer_only", "no_transfer", "five_candidates"):
                    if mode == "legacy_product" and not include_historical_controls:
                        continue
                    candidates = calibration_candidate_recipes(mode, batch_size=batch_size, seed=seed)
                    tasks.append({**common, "arm": mode, "selector_mode": mode,
                                  "recipes": [recipe.record() for recipe in candidates],
                                  "training_rows": n_fit,
                                  "validation_rows": int(inner.validation_mask.sum())})
                continue
            for recipe in recipes:
                task = {**common, "arm": recipe.name, "recipe": recipe.record(),
                        "training_rows": n_fit,
                        "validation_rows": int(inner.validation_mask.sum()) if stage == "B0" else 0}
                task["budget"] = (target_batch_budget(n_fit, recipe.batch_size, epochs=recipe.epochs,
                                                      max_updates=recipe.max_updates)
                                  if recipe.weights != "none" else {"updates": 0, "draws": 0})
                if stage in {"B0", "B0_score"}:
                    task.update(requested_batch_aliases=list(batch_size_aliases(n_fit, recipe.batch_size)),
                                budget_mode="updates" if recipe.max_updates is not None else "epochs")
                tasks.append(task)
    return dict(schema="n2p3_target_tasks/1", stage=stage, tasks=tasks, scopes=scopes,
                exclusions=exclusions, task_count=len(tasks),
                fit_calls=sum(task.get("selection_budget", {}).get("fit_calls_upper_bound", task["arm"] != "zero_shot")
                              for task in tasks),
                dry_run_fits_data=False, final_test_used_for_selection=False,
                requested_batch_size=batch_size, target_seed=seed,
                include_historical_controls=include_historical_controls,
                cost_estimate={"wall_seconds": None, "peak_device_bytes": None,
                               "status": "requires_representative_device_timing"})


def apply_correctness_budget(manifest: dict, *, epochs: int) -> None:
    """Bound a declared software probe while keeping every treatment observable."""
    from math import ceil

    from transfer.recipes import TargetRecipe, target_batch_budget

    if type(epochs) is not int or epochs < 1:
        raise ValueError("software correctness epochs must be a positive integer")
    for task in manifest["tasks"]:
        records = task.get("recipes", [task.get("recipe")])
        for record in records:
            if record is None:
                continue
            record["epochs"] = min(record["epochs"], epochs)
            if record["max_updates"] is not None:
                record["max_updates"] = epochs * ceil(task["training_rows"] / 128)
            recipe = TargetRecipe(**record)
            if "recipe" in task:
                task["budget"] = (target_batch_budget(task["training_rows"], recipe.batch_size,
                    epochs=recipe.epochs, max_updates=recipe.max_updates) if recipe.weights != "none"
                    else {"updates": 0, "draws": 0})
                if "epoch_selection" in task:
                    from transfer.epoch_selection import (
                        EpochSelectionPolicy,
                        epoch_selection_budget,
                    )

                    scope = manifest["scopes"][task["scope_id"]]
                    task["selection_budget"] = epoch_selection_budget(recipe, EpochSelectionPolicy(**task["epoch_selection"]),
                        calibration_rows=task["training_rows"], inner_rows=len(scope["inner_training_rows"]),
                        inner_legal=scope["inner_split"]["legal"])
    manifest.update(evidence_scope="software_correctness", correctness_target_epochs=epochs)


def execute_target_tasks(
    dataset, payload: dict, manifest: dict, *, output: Path, context: dict,
    device: str, recording_order=None,
) -> dict:
    """Execute the declared target matrix; B0 only scores inner validation."""
    import torch
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    from research.evaluation import (
        build_evaluation_run_contract,
        scratch_model_origin,
    )
    from research.execution import ExpectedSubjectError, atomic_write_json
    from transfer.binary import matched_split_digests
    from transfer.calibration_candidates import (
        CalibrationCandidateFitCache,
        fit_calibration_candidate,
        select_calibration_candidate,
    )
    from transfer.checkpoint import (
        PreparedTransferCheckpoint,
        predict_n2p3_checkpoint,
    )
    from transfer.recipes import TargetRecipe

    reused_cells = {}
    for record_path in context.get("plan", {}).get("reuse_target_records", []):
        previous_path = Path(record_path)
        previous = read_result_record(previous_path)
        if (previous.get("status") != "completed" or previous.get("checkpoint_sha256") != context["checkpoint_sha256"]
                or previous.get("target_cache_sha256") != context["target_cache_sha256"]
                or previous.get("task_manifest", {}).get("stage") != "compact"):
            raise ValueError("target reuse requires a completed compact run with matching source and target bytes")
        if previous.get("environment") != context.get("environment"):
            raise ValueError("target reuse requires the same recorded execution environment")
        for old_index, old_cell in enumerate(previous["cells"]):
            if old_cell.get("status") != "completed" or old_cell.get("score_scope") != "final_test":
                continue
            saved_cell = read_result_record(previous_path.parent / f"{old_index:05d}.json")
            if saved_cell != old_cell:
                raise ResultIntegrityError("result integrity: cell disagrees with its completed run")
            if semantic_sha256(old_cell["evaluation_contract"]) != old_cell["evaluation_contract_digest"]:
                raise ResultIntegrityError("result integrity: evaluation contract digest disagrees with its cell")
            treatment = None if old_cell.get("recipe") is None else {
                key: value for key, value in old_cell["recipe"].items() if key != "name"}
            key = semantic_sha256({"person": old_cell["participant_key"], "recipe": treatment,
                "split": {field: old_cell[field] for field in ("calibration_rows_sha256", "test_rows_sha256")}})
            reused_cells[key] = (previous_path.parent, old_index, old_cell)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {**context, "schema": "n2p3_target_stage_run/1", "status": "running",
              "task_manifest": manifest, "cells": [], "batch_selection_rows": []}
    atomic_write_json(output / "tasks.json", manifest)
    atomic_write_json(output / "record.json", report)
    observed = (np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask)
    prepared = PreparedTransferCheckpoint(payload, dataset)
    stats = prepared.input_statistics
    source_origin = prepared.model_origin(context["checkpoint_sha256"])
    current_subject, base = None, None
    selector_scope, selector_cache = None, None
    for index, task in enumerate(manifest["tasks"]):
        if task["scope_id"] != selector_scope:
            if selector_cache is not None:
                selector_cache.clear()
            selector_scope = task["scope_id"]
            selector_cache = CalibrationCandidateFitCache() if "selector_mode" in task or "epoch_selection" in task else None
        if task["subject"] != current_subject:
            base = prepared.trunk_for_target(task["subject"])
            current_subject = task["subject"]
        scope = manifest["scopes"][task["scope_id"]]
        is_inner = manifest["stage"] == "B0"
        train = np.asarray(scope["inner_training_rows"] if is_inner else scope["calibration_rows"], dtype=int)
        test = np.asarray(scope["inner_validation_rows"] if is_inner else scope["test_rows"], dtype=int)
        training_mask = np.zeros(dataset.n_epochs, dtype=bool)
        training_mask[train] = True
        fit_counts_before = selector_cache.record() if selector_cache is not None else None
        started = time.perf_counter()
        cell = {**task, **matched_split_digests(train, test), "study": context["target_study"],
                "score_scope": "calibration_inner_validation" if is_inner else "final_test",
                "source_seed": context["source_seed"], "target_seed": manifest["target_seed"],
                "evidence_scope": context.get("evidence_scope", "development"),
                "calibration_positive": int(dataset.y[train].sum()), "test_rows": len(test)}
        recipe = None if task.get("recipe") is None else TargetRecipe(**task["recipe"])
        fitted = None
        reuse_key = semantic_sha256({"person": task["participant_key"],
            "recipe": None if recipe is None else {key: value for key, value in recipe.record().items() if key != "name"},
            "split": matched_split_digests(train, test)})
        imported = reused_cells.get(reuse_key) if not is_inner and "selector_mode" not in task and "epoch_selection" not in task else None
        try:
            if imported is not None:
                directory, old_index, old_cell = imported
                data, state_record = read_result_artifacts(old_cell, directory)
                if (not np.array_equal(data.get("train_rows"), train) or not np.array_equal(data.get("test_rows"), test)
                        or not np.array_equal(data.get("y"), dataset.y[test])):
                    raise ResultIntegrityError("result integrity: imported prediction rows or labels changed")
                scores = data["scores"]
                if recipe is not None:
                    if state_record is None:
                        raise ResultIntegrityError("result integrity: fitted target model is missing")
                    if _tensor_state_digest(state_record["state"]) != old_cell["fitted_state_sha256"]:
                        raise ResultIntegrityError("result integrity: imported target fitted state changed")
                    state_record["recipe"] = recipe.record()
                    torch.save(state_record, output / f"{index:05d}.pt")
                    cell.update(fit_history=state_record["fit_history"], fitted_state_sha256=old_cell["fitted_state_sha256"])
                cell["prediction_reuse"] = {"origin_record": str(directory / "record.json"),
                    "origin_cell": old_index, "origin_contract_digest": old_cell["evaluation_contract_digest"],
                    "origin_result_integrity": old_cell["result_integrity"],
                    "fit_executed": False}
            elif task["arm"] == "zero_shot":
                scores = predict_n2p3_checkpoint(base, dataset.X[test], input_stats=stats,
                                                 device=torch.device(device), trial_channel_mask=observed[test])
            elif "selector_mode" in task:
                recipes = tuple(TargetRecipe(**item) for item in task["recipes"])
                fitted, selection = select_calibration_candidate(
                    dataset, recipes, calibration_mask=training_mask, source_trunk=base,
                    source_input_stats=stats, device=device, protocol="binary_auc",
                    recording_order=recording_order, selector_mode=task["selector_mode"],
                    fit_cache=selector_cache,
                )
                recipe = fitted.recipe
                cell["selection"] = selection
                scores = fitted.predict_logit(dataset.X[test], observed[test])
            elif "epoch_selection" in task:
                from transfer.epoch_selection import EpochSelectionPolicy, fit_with_epoch_selection

                fitted, selection = fit_with_epoch_selection(dataset, recipe, training_mask,
                    policy=EpochSelectionPolicy(**task["epoch_selection"]), source_trunk=base,
                    source_input_stats=stats, device=device, recording_order=recording_order, fit_cache=selector_cache)
                recipe = fitted.recipe
                cell["epoch_selection_result"] = selection
                cell.update(requested_recipe=task["recipe"], recipe=recipe.record())
                scores = fitted.predict_logit(dataset.X[test], observed[test])
            else:
                diagnostic_mask = np.zeros(dataset.n_epochs, dtype=bool) if is_inner else None
                if diagnostic_mask is not None:
                    diagnostic_mask[test] = True
                fitted = fit_calibration_candidate(dataset, recipe, training_mask, source_trunk=base,
                                                   source_input_stats=stats, device=device,
                                                   validation_mask=diagnostic_mask,
                                                   validation_behavior="diagnostics_only" if is_inner else "select_epoch")
                scores = fitted.predict_logit(dataset.X[test], observed[test])
            if set(np.unique(dataset.y[test])) != {0, 1}:
                raise ValueError("evaluation requires both binary classes; no test rows may be borrowed")
            if fitted is not None:
                cell["fit_history"] = fitted.fit_history
                state = fitted.state
                torch.save({"recipe": recipe.record(), "state": state,
                            "fit_history": fitted.fit_history,
                            **({"epoch_selection": cell["epoch_selection_result"]}
                               if "epoch_selection_result" in cell else {})}, output / f"{index:05d}.pt")
                cell["fitted_state_sha256"] = _tensor_state_digest(state)
            scores = np.asarray(scores, dtype=float)
            cell.update(status="completed", auc=float(roc_auc_score(dataset.y[test], scores)),
                        ap=float(average_precision_score(dataset.y[test], scores)),
                        bacc_at_raw_score_zero=float(balanced_accuracy_score(dataset.y[test], scores > 0)),
                        brier_raw_sigmoid=float(np.mean((expit(scores) - dataset.y[test]) ** 2)),
                        nll_raw_sigmoid=float(np.mean(np.logaddexp(0, scores) - dataset.y[test] * scores)),
                        probability_quality_scope="raw_sigmoid_scores_not_externally_calibrated")
            np.savez_compressed(output / f"{index:05d}.npz", train_rows=train, test_rows=test,
                                y=dataset.y[test], scores=scores)
        except (ValueError, ExpectedSubjectError) as error:
            cell.update(status="fit_failed", auc=None,
                        failure={"type": type(error).__name__, "message": str(error)})
        finally:
            if fit_counts_before is not None and ("selector_mode" in task or "epoch_selection" in task):
                counts = selector_cache.record()
                actual_fits = counts["fit_calls"] - fit_counts_before["fit_calls"]
                reused_fits = counts["cache_hits"] - fit_counts_before["cache_hits"]
                cell["fit_execution"] = {"requested_fits": actual_fits + reused_fits,
                                         "fit_calls": actual_fits, "cache_hits": reused_fits}
            elif manifest["stage"] == "ES" and task["arm"] != "zero_shot":
                cell["fit_execution"] = {"requested_fits": 1, "fit_calls": int(imported is None),
                                         "cache_hits": int(imported is not None)}
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            cell["elapsed_seconds"] = time.perf_counter() - started
        scratch = recipe is not None and recipe.initialization in {"scratch", "waveform_lda"}
        origin = (scratch_model_origin({"recipe": recipe.record(), "architecture": base.architecture_record()})
                  if scratch else source_origin)
        contract = build_evaluation_run_contract(
            arm_name=task["arm"], model_origin=origin, dataset=dataset,
            target_cache_sha256=context["target_cache_sha256"],
            source_snapshot_sha256=context["evaluation_snapshot"]["archive_sha256"],
            requested_subjects=[task["subject"]], identity_policy="source_or_global",
            target_protocol=context["target_protocol"],
            adaptation={"stage": manifest["stage"], "actual_K": task["K"],
                        "recipe": None if recipe is None else recipe.record(),
                        "selector_mode": task.get("selector_mode"), "score_scope": cell["score_scope"],
                        **({"epoch_selection": task["epoch_selection"]} if "epoch_selection" in task else {})},
            decision={"kind": "binary_no_candidate_claim"},
            evidence_scope={"stage": context.get("evidence_scope", "development"), "product_confirmation": False,
                            "heldout_study": context["target_study"],
                            "source_study_membership_digest": context.get("source_study_membership", {}).get("digest"),
                            "target_study_membership_digest": context.get("target_study_membership", {}).get("digest")},
        )
        cell.update(evaluation_contract=contract.record(), evaluation_contract_digest=contract.digest())
        artifacts = {}
        if cell["status"] == "completed":
            artifacts["prediction"] = output / f"{index:05d}.npz"
            if recipe is not None:
                artifacts["model"] = output / f"{index:05d}.pt"
        cell = seal_result_record(cell, files=artifacts)
        report["cells"].append(cell)
        if is_inner:
            for requested_batch in task["requested_batch_aliases"]:
                report["batch_selection_rows"].append({
                    "study": context["target_study"], "participant_key": task["participant_key"],
                    "K": task["K"], "batch_size": requested_batch, "arm": recipe.initialization,
                    "budget_mode": task["budget_mode"], "score": cell["auc"], "status": cell["status"],
                    "elapsed_seconds": cell["elapsed_seconds"],
                    "calibration_rows_sha256": cell["calibration_rows_sha256"],
                    "validation_rows_sha256": cell["test_rows_sha256"],
                    "reused_effective_batch": requested_batch != recipe.batch_size,
                })
        atomic_write_json(output / f"{index:05d}.json", cell)
        atomic_write_json(output / "record.json", report)
        del fitted
    if selector_cache is not None:
        selector_cache.clear()
    report.update(status="completed", completed_cells=sum(cell["status"] == "completed" for cell in report["cells"]),
                  failed_cells=sum(cell["status"] != "completed" for cell in report["cells"]))
    if manifest["stage"] in {"E3", "ES"}:
        report["fit_execution"] = {key: sum(
            cell.get("fit_execution", {}).get(key, 0) for cell in report["cells"])
            for key in ("requested_fits", "fit_calls", "cache_hits")}
    report = seal_result_record(report)
    atomic_write_json(output / "record.json", report)
    return report


def _tensor_state_digest(state: dict) -> str:
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_target_batch(rows: list[dict], *, candidates: tuple[int, ...] = (32, 128, 512)) -> dict:
    """Freeze B0 using complete paired inner scores, never final-test outcomes."""
    if not candidates or tuple(sorted(set(candidates))) != candidates or candidates[0] < 1:
        raise ValueError("batch candidates must be positive unique increasing values")
    selected = [row for row in rows if row.get("budget_mode") == "epochs"]
    lookup = {}
    cohort = set()
    digests = {}
    for row in selected:
        batch = row["batch_size"]
        if batch not in candidates or row["arm"] not in {"source", "scratch"}:
            raise ValueError("B0 requires declared batch candidates and source/scratch pairs")
        unit = (row["study"], row["participant_key"], row["K"])
        key = (batch, *unit, row["arm"])
        if key in lookup:
            raise ValueError("duplicate B0 participant/K/batch/arm outcome")
        current = (row.get("calibration_rows_sha256"), row.get("validation_rows_sha256"))
        if not all(current) or (unit in digests and digests[unit] != current):
            raise ValueError("B0 candidates must use the same training and validation rows")
        digests[unit] = current
        lookup[key] = row
        cohort.add(unit)
    records = []
    for batch in candidates:
        participant_scores = defaultdict(list)
        failed = 0
        seconds = 0.0
        for study, participant, budget in sorted(cohort):
            values = []
            for arm in ("source", "scratch"):
                row = lookup.get((batch, study, participant, budget, arm))
                if row is None or row["status"] != "completed":
                    failed += 1
                    continue
                score = row.get("score")
                elapsed = row.get("elapsed_seconds")
                if (score is None or elapsed is None or not np.isfinite([score, elapsed]).all()
                        or not 0 <= score <= 1 or elapsed < 0):
                    raise ValueError("completed B0 cells require finite AUC and measured time")
                values.append(score)
                seconds += elapsed
            if len(values) == 2:
                participant_scores[study, participant].append(float(np.mean(values)))
        study_scores = defaultdict(list)
        for (study, _), scores in participant_scores.items():
            study_scores[study].append(float(np.mean(scores)))
        score = (float(np.mean([np.mean(values) for values in study_scores.values()]))
                 if cohort and not failed else None)
        records.append(dict(batch_size=batch, score=score, failed_cells=failed,
                            requested_cells=2 * len(cohort), elapsed_seconds=seconds))
    eligible = [row for row in records if row["score"] is not None]
    best_score = max((row["score"] for row in eligible), default=None)
    tied = [row for row in eligible if best_score - row["score"] <= 1e-6]
    chosen = min(tied, key=lambda row: (row["elapsed_seconds"], row["batch_size"])) if tied else None
    result = dict(schema="n2p3_target_batch_rule/1", batch_size=chosen["batch_size"] if chosen else 32,
                  selection_evidence=chosen is not None,
                  selection_scope="calibration_inner_validation_only",
                  estimator="equal_arms_then_actual_budgets_then_participants_then_studies",
                  candidates=records, cohort=[list(unit) for unit in sorted(cohort)],
                  fallback_reason=None if chosen else "no_complete_legal_inner_selection_evidence",
                  input_results_sha256=semantic_sha256(rows),
                  evidence_level="public_development", target_device_performance_confirmed=False)
    return result
