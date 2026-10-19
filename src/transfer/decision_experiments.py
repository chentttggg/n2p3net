"""Predeclared prior/decision attribution on the existing scoped executor.

This changes a declared experimental matrix, not the product's default recipe.
The full-joint low-G cells remain explicit exploratory treatments; repeated
optimization steps are never reported as additional physical decisions.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from numbers import Real

from transfer.recipes import TargetRecipe, target_batch_budget


def expand_decision_diagnostics(manifest, declaration):
    """Return the complete matched matrix; the input scoped plan stays intact."""
    expected = {"schema", "full_joint_interpretation", "decision_weight"}
    if (not isinstance(declaration, dict) or set(declaration) != expected
            or declaration["schema"] != "n2p3_prior_decision_diagnostic/1"
            or declaration["full_joint_interpretation"] != "exploratory_not_low_G_sufficiency"):
        raise ValueError("decision diagnostics require an explicit exploratory design declaration")
    decision_weight = declaration["decision_weight"]
    if isinstance(decision_weight, bool) or not isinstance(decision_weight, Real) or not math.isfinite(decision_weight) or decision_weight <= 0:
        raise ValueError("decision diagnostic weight must be explicitly finite and positive")
    if manifest.get("stage") != "unified":
        raise ValueError("decision diagnostics require independent unified raw scopes")
    result = copy.deepcopy(manifest)
    by_scope = {}
    for task in manifest["tasks"]:
        by_scope.setdefault(task["scope_id"], {})[task["arm"]] = task
    tasks = []
    for key, originals in by_scope.items():
        if "zero_shot" in originals:
            tasks.append(copy.deepcopy(originals["zero_shot"]))
            continue
        base_task = originals["unified_q0"]
        base = TargetRecipe(**base_task["recipe"])
        joint = TargetRecipe(**originals["unified_q1"]["recipe"])
        membership = manifest["scopes"][key]["membership"]
        if membership["available"] and joint.candidate_loss_weight != decision_weight:
            raise ValueError("diagnostic decision weight differs from the source-scoped plan")
        # With no complete group, positive decision cells deliberately fail in
        # the fit owner and remain requested. Do not silently turn them into CE.
        recipes = [
            replace(base, name="diag_trial"),
            replace(base, name="diag_prior", source_prior_weight=joint.source_prior_weight),
            replace(base, name="diag_decision", candidate_loss_weight=decision_weight),
            replace(joint, name="diag_prior_decision", candidate_loss_weight=decision_weight),
            replace(base, name="diag_head_trial", weights="head", bn_affine="frozen"),
            replace(base, name="diag_head_decision", weights="head", bn_affine="frozen",
                    candidate_loss_weight=decision_weight),
            replace(base, name="diag_scratch_trial", initialization="scratch"),
            replace(base, name="diag_scratch_decision", initialization="scratch", candidate_loss_weight=decision_weight),
        ]
        support = {"physical_decisions": len(membership.get("groups", ())),
                   "prior_active": joint.source_prior_weight > 0,
                   "prior_contrast_degenerate": joint.source_prior_weight == 0,
                   "target_coverage": sorted({row["target_candidate"] for row in membership.get("groups", ())}),
                   "independent_decisions_created_by_repeated_updates": 0,
                   "full_joint_is_exploratory": True}
        for recipe in recipes:
            task = {**copy.deepcopy(base_task), "arm": recipe.name, "recipe": recipe.record(),
                    "adaptation_package": recipe.name, "decision_factor": int(bool(recipe.candidate_loss_weight)),
                    "decision_information": support}
            task["budget"] = target_batch_budget(task["training_rows"], recipe.batch_size,
                epochs=recipe.epochs, max_updates=recipe.max_updates,
                decision_rows_per_update=result["scopes"][key].get("decision_rows_per_update", {}).get("calibration", 0)
                if recipe.candidate_loss_weight is not None else 0)
            tasks.append(task)
        tasks.append(copy.deepcopy(originals["unified_waveform_lda"]))
    result.update(tasks=tasks, task_count=len(tasks), fit_calls=sum(task["arm"] != "zero_shot" for task in tasks),
                  decision_diagnostic_design=copy.deepcopy(declaration))
    return result
