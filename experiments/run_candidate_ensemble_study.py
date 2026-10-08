"""Execute declared candidate ensembles on calibration-group development folds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from data.bnci2014_008_candidate import BNCI2014_008_DATASET_ID  # noqa: E402
from data.epochs import parse_preprocessing_spec  # noqa: E402
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity  # noqa: E402
from research.candidate_metrics import (  # noqa: E402
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import (  # noqa: E402
    DecisionPlanContract,
    DecisionPlanEntry,
    semantic_sha256,
)
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import (  # noqa: E402
    ExpectedSubjectError,
    SubjectFailureCode,
    UnidentifiableFitError,
    atomic_write_json,
)
from research.source_catalog import SourceSeedCatalog, paired_pipeline_seeds  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    checkpoint_architecture_record,
    load_checkpoint_payload,
)
from transfer.ensemble import EnsembleProgram, EnsembleScorer  # noqa: E402
from transfer.ensemble_fit import (  # noqa: E402
    CalibrationExpertProgram,
    FittedEnsemble,
    FrozenSourceCheckpoint,
    fit_calibration_ensemble,
    fit_calibration_prefix_programs,
)
from transfer.ensemble_views import (  # noqa: E402
    prepare_bnci_calibration_groups,
    prepare_bnci_independent_prefixes,
)
from transfer.kernel_prior import KernelPriorRankError  # noqa: E402
from transfer.outcomes import CandidateCoverage, DecisionKey  # noqa: E402

SCHEMA = "n2p3_candidate_ensemble_study/1"
_WORKER_ABORT = None


def _initialize_worker_abort(signal):
    global _WORKER_ABORT
    _WORKER_ABORT = signal


def _abort_checkpoint():
    if _WORKER_ABORT is not None and _WORKER_ABORT.is_set():
        raise RuntimeError("candidate matrix stopped after a peer contract failure")


def _matched_source_controls(experts, source_snapshots):
    bindings = []
    resolved_sources = dict(source_snapshots)
    for source in experts:
        if source.family != "trial" or source.recipe.initialization != "source":
            continue
        if source.expert_id not in resolved_sources:
            raise ValueError("source expert lacks a declared checkpoint")
        recipe = source.recipe.record()
        recipe.pop("name")
        recipe.pop("initialization")
        if recipe["input_statistics"] == "source":
            recipe["input_statistics"] = "target_prefix"
        scratch = None
        waveform = None
        for control in experts:
            if control.family != "trial" or control.profile.digest != source.profile.digest:
                continue
            if control.recipe.initialization == "waveform_lda":
                waveform = control
            if control.recipe.initialization == "scratch":
                actual = control.recipe.record()
                actual.pop("name")
                actual.pop("initialization")
                if actual == recipe:
                    scratch = control
        if scratch is None or waveform is None:
            raise ValueError("full-fine requires matched scratch and waveform LDA controls before fitting")
        resolved_sources.setdefault(scratch.expert_id, resolved_sources[source.expert_id])
        first = checkpoint_architecture_record(load_checkpoint_payload(io.BytesIO(resolved_sources[source.expert_id].content)))
        second = checkpoint_architecture_record(load_checkpoint_payload(io.BytesIO(resolved_sources[scratch.expert_id].content)))
        if semantic_sha256(first) != semantic_sha256(second):
            raise ValueError("source and matched scratch must use the same actual checkpoint architecture")
        bindings.append({"source": source.expert_id, "scratch": scratch.expert_id,
                         "waveform_lda": waveform.expert_id, "architecture_digest": semantic_sha256(first)})
    return resolved_sources, bindings


def _executing_source():
    paths = [*ROOT.joinpath("src").rglob("*.py"), Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _resume_plan_contract(plan):
    return {key: value for key, value in plan.items()
            if key not in {"source_snapshot_manifest", "resume", "resume_binding"}}


def _bound_resume_json(path, expected_digest, name):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError(f"original resume {name} bytes changed")
    return json.loads(content)


def _prepare_package_resume(plan, expected_source):
    declaration = plan.get("resume")
    if declaration is None:
        return None
    if (plan["comparison_role"] != "fixed_prefix_ar_order" or plan["calibration_strategy"] != "fixed"
            or plan["source_checkpoints"] or any(row["family"] != "continuous_ar" for row in plan["experts"])):
        raise ValueError("package resume is limited to fixed-prefix raw experts and fixed calibration")
    required = {"schema", "original_run", "original_plan_sha256", "original_executed_source_sha256",
                "original_execution", "original_execution_sha256", "original_source_manifest",
                "original_source_manifest_sha256", "original_source_archive_sha256"}
    if set(declaration) != required or declaration["schema"] != "n2p3_m1_package_resume/1":
        raise ValueError("package resume requires an explicit complete original execution binding")
    original_run = Path(declaration["original_run"]).resolve()
    original_plan = _bound_resume_json(original_run / "plan.json", declaration["original_plan_sha256"], "plan")
    original_source = _bound_resume_json(original_run / "executed_source.json",
                                       declaration["original_executed_source_sha256"], "source")
    execution = _bound_resume_json(declaration["original_execution"], declaration["original_execution_sha256"], "execution")
    manifest = _bound_resume_json(declaration["original_source_manifest"],
                                 declaration["original_source_manifest_sha256"], "manifest")
    original_root = Path(execution["executed_root"]).resolve()
    argv = execution["argv"]
    if argv.count("--plan") != 1 or argv.count("--output") != 1:
        raise ValueError("original execution must declare its actual input plan and output directory")
    input_plan_path = Path(argv[argv.index("--plan") + 1]).resolve()
    output_path = Path(argv[argv.index("--output") + 1]).resolve()
    if (output_path != original_run
            or hashlib.sha256(input_plan_path.read_bytes()).hexdigest() != execution["plan_sha256"]):
        raise ValueError("original execution plan/output binding differs from the resumed run")
    if json.loads((original_run / "progress.json").read_bytes())["status"] != "failed":
        raise ValueError("package resume requires a terminated failed original parent run")
    if _resume_plan_contract(original_plan) != _resume_plan_contract(plan):
        raise ValueError("original resume plan differs in scientific, input or resource fields")
    if (execution["source"]["archive_sha256"] != declaration["original_source_archive_sha256"]
            or execution["source"]["manifest_sha256"] != declaration["original_source_manifest_sha256"]
            or manifest["archive_sha256"] != declaration["original_source_archive_sha256"]
            or Path(original_plan["source_snapshot_manifest"]).resolve()
            != Path(declaration["original_source_manifest"]).resolve()):
        raise ValueError("original execution and physical source freeze disagree")
    original_freeze = verified_executed_snapshot(declaration["original_source_manifest"], root=original_root)
    current_freeze = verified_executed_snapshot(plan["source_snapshot_manifest"], root=ROOT)
    runtime_keys = {path.relative_to(original_root).as_posix() for path in (original_root / "src").rglob("*.py")}
    runtime_keys.add("experiments/run_candidate_ensemble_study.py")
    if set(original_source) != runtime_keys or set(original_source) != set(expected_source):
        raise ValueError("resume execution source does not cover the complete actual runtime")
    if any(original_source[name] != manifest["files"].get(name) for name in original_source):
        raise ValueError("original execution source differs from its archived source")
    mathematical_source = {name: digest for name, digest in expected_source.items() if name.startswith("src/")}
    original_mathematical_source = {name: digest for name, digest in original_source.items() if name.startswith("src/")}
    if mathematical_source != original_mathematical_source:
        raise ValueError("resume requires every original src Python byte to remain unchanged")
    return {"schema": "n2p3_m1_package_resume_binding/1", "original_run": str(original_run),
            "original_execution_root": str(original_root), "original_execution_sha256": declaration["original_execution_sha256"],
            "original_plan_sha256": declaration["original_plan_sha256"],
            "original_executed_source_sha256": declaration["original_executed_source_sha256"],
            "original_source_digest": semantic_sha256(original_source), "original_source_freeze": original_freeze,
            "score_source_digest": semantic_sha256(expected_source), "score_source_freeze": current_freeze,
            "executor_change_scope": "runner_transport_and_package_resume_only; all_src_bytes_identical"}


def _reuse_frozen_package(binding, record_index, seed, fold, expert_index, spec, program, training, destination):
    if binding is None:
        return None, None
    original = (Path(binding["original_run"]) / f"participant_{record_index:03d}"
                / f"seed{seed}_fold{fold}_expert{expert_index}")
    manifest_path = original / "ensemble.json"
    if not manifest_path.is_file():
        return None, None
    manifest_bytes = manifest_path.read_bytes()
    payload = json.loads(manifest_bytes)
    record = payload["record"]
    if semantic_sha256(record) != payload["record_sha256"] or len(record["experts"]) != 1:
        raise ValueError("original resumed package manifest changed")
    entry = record["experts"][0]
    expected_fit = semantic_sha256({"program": spec.record(), "authorized_calibration": [group.record() for group in training],
                                   "source_checkpoint_sha256": None, "checkpoint_role": None})
    if (entry["program"] != spec.record() or record["program"] != program.record()
            or record["training_receipts"] != [group.view.record() for group in training]
            or entry["fit_digest"] != expected_fit or entry["source_digest"] is not None
            or record["temperature_calibration"] != {"strategy": "fixed", "temperatures": {spec.expert_id: 1.}}):
        raise ValueError("original resumed package program, labels or fit authority differs")
    filename = entry["model"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or Path(filename).is_absolute():
        raise ValueError("original resumed model must name one relative package member")
    model_path = original / filename
    if model_path.resolve().parent != original.resolve():
        raise ValueError("original resumed model path escapes its package")
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != entry["model_file_sha256"]:
        raise ValueError("original resumed model bytes changed")
    model_destination = destination / filename
    if model_destination.resolve().parent != destination.resolve():
        raise ValueError("resumed model destination escapes its package")
    destination.mkdir()
    (destination / "ensemble.json").write_bytes(manifest_bytes)
    model_destination.write_bytes(model_bytes)
    try:
        fitted = FittedEnsemble.load(destination)
    except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
        raise ValueError("original resumed frozen state is invalid") from error
    if fitted.scorer.program.record() != program.record():
        raise ValueError("original resumed actual program differs")
    origin = {"kind": "reused_package", "physical_fit_attempts": 0, "physical_successful_fits": 0,
              "reused_package_count": 1, "producer_source_digest": binding["original_source_digest"],
              "producer_executed_source_sha256": binding["original_executed_source_sha256"],
              "original_execution_sha256": binding["original_execution_sha256"],
              "producer_source_freeze": binding["original_source_freeze"], "original_package": str(original),
              "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "original_model_file_sha256": entry["model_file_sha256"]}
    return fitted, origin


def _fixed_prefix_ar_panel(experts, role):
    if role == "candidate_ensemble":
        return None
    if role != "fixed_prefix_ar_order":
        raise ValueError("unknown candidate comparison role")
    if any(spec.kernel_prior is not None and spec.kernel_prior.kind != "isotropic_ridge" for spec in experts):
        raise ValueError("fixed-prefix AR order comparison requires the same isotropic ridge prior")
    orders = (0, 1, 4, 8, 16, 32)
    estimators = ("burg", "conditional_profile")
    expected = {(order, estimator) for order in orders for estimator in estimators}
    actual = {(spec.noise_order, spec.noise_estimator) for spec in experts}
    matched = {(spec.conditioning_samples, spec.ridge, spec.response_seconds, spec.n_basis) for spec in experts}
    if (len(experts) != len(expected) or actual != expected or len(matched) != 1
            or any(spec.family != "continuous_ar" or spec.conditioning_samples != 32 for spec in experts)):
        raise ValueError("fixed-prefix AR panel requires all six orders and both estimators on the same prefix, basis and prior")
    first = experts[0]
    return {"schema": "n2p3_fixed_prefix_ar_panel/1", "orders": list(orders), "noise_estimators": list(estimators),
        "conditioning_samples": 32, "ridge": first.ridge, "response_seconds": first.response_seconds,
        "n_basis": first.n_basis, "kernel_information": "all_authorized_prefix_and_tail",
        "score_estimand": "conditional_tail_with_fixed_candidate_prior",
        "initialization": "full_record_segmented_Burg_with_order_dependent_lag_pairs",
        "conditional_objective": "common_tail_joint_shared_kernel_and_independent_diffuse_record_drift"}


def _calibration_prior_panel(experts):
    """Bind shape contrasts separately from changes in prior signal energy."""
    noise_programs = ((0, "burg"), (8, "burg"), (8, "conditional_profile"), (32, "conditional_profile"))
    multipliers = (1., 9., 99.)
    prior_cells = {("isotropic_ridge", 1.),
                   *((kind, g) for kind in ("calibration_g", "calibration_trace_isotropic") for g in multipliers)}
    expected = {(order, estimator, kind, value) for order, estimator in noise_programs for kind, value in prior_cells}
    if len(experts) != len(expected) or any(spec.family != "continuous_ar" for spec in experts):
        raise ValueError("calibration prior panel requires all declared raw prior/noise cells")
    cells = {(spec.noise_order, spec.noise_estimator, spec.prior_spec.kind,
              spec.prior_spec.precision if spec.prior_spec.kind == "isotropic_ridge" else spec.prior_spec.g): spec.expert_id
             for spec in experts}
    matched = {(spec.conditioning_samples, spec.response_seconds, spec.n_basis) for spec in experts}
    if (set(cells) != expected or len(matched) != 1 or any(spec.conditioning_samples != 32 for spec in experts)):
        raise ValueError("calibration prior panel requires matched fixed32, basis, response and complete prior/noise cells")
    first = experts[0]
    return {"schema": "n2p3_calibration_prior_panel/1", "noise_programs": [list(row) for row in noise_programs],
            "variance_multipliers": list(multipliers), "conditioning_samples": 32,
            "response_seconds": first.response_seconds, "n_basis": first.n_basis,
            "reference_information": "all_authorized_C_AR0_per_record_diffuse_affine",
            "energy_matching": "tr(G H_g)=k*g=tr(G H_isotropic); lambda_C=tr(G)/(k*g)",
            "coordinate_policy": "g_basis_invariant; trace_isotropic_fixed_declared_basis",
            "rank_policy": "same_full_rank_admissibility_for_g_and_trace_controls",
            "score_estimand": "conditional_tail_with_fixed_candidate_prior",
            "shape_contrasts": [{"noise_order": order, "noise_estimator": estimator, "g": g,
                "baseline": cells[(order, estimator, "calibration_trace_isotropic", g)],
                "alternative": cells[(order, estimator, "calibration_g", g)]}
                for order, estimator in noise_programs for g in multipliers]}


def _subject_impl(row, plan, directory, expected_source, source_snapshots):
    directory = Path(directory)
    directory.mkdir()
    _abort_checkpoint()
    if _executing_source() != expected_source:
        raise ValueError("candidate ensemble execution differs from its frozen source")
    if plan.get("evaluation_design") == "independent_prefix":
        return _independent_subject_impl(row, plan, directory, expected_source, source_snapshots)
    grouped = prepare_bnci_calibration_groups(row["path"], source_sha256=row["sha256"],
        authorized_labels=plan["authorized_labels"], group_count=plan["group_count"],
        response_seconds=plan["observation_response_seconds"], pre_roll_seconds=plan["pre_roll_seconds"])
    atomic_write_json(directory / "scopes.json", grouped.ledger)
    groups = grouped.groups
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    predictions = []
    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for seed in plan["seeds"]:
            specs = tuple(replace(spec, recipe=None if spec.recipe is None else replace(spec.recipe, seed=seed))
                          for spec in declared)
            for fold, held in enumerate(groups):
                training = [group for index, group in enumerate(groups) if index != fold]
                evidence, calibrations, origins = {}, {}, {}
                for expert_index, spec in enumerate(specs):
                    _abort_checkpoint()
                    one = EnsembleProgram.equal_weights(spec.expert_id, (spec.expert_id,))
                    sources = {spec.expert_id: source_snapshots[spec.expert_id]} if spec.expert_id in source_snapshots else {}
                    package = directory / f"seed{seed}_fold{fold}_expert{expert_index}"
                    origin = {"kind": "new_fit", "physical_fit_attempts": 0, "physical_successful_fits": 0,
                              "reused_package_count": 0, "producer_source_digest": semantic_sha256(expected_source)}
                    try:
                        fitted, reused_origin = _reuse_frozen_package(plan.get("resume_binding"),
                            plan["records"].index(row), seed, fold, expert_index, spec, one, training, package)
                        if fitted is None:
                            origin["physical_fit_attempts"] = 1
                            fitted = fit_calibration_ensemble(training, (spec,), one,
                                source_checkpoints=sources, device=plan["device"],
                                calibration_strategy=plan["calibration_strategy"])
                            origin["physical_successful_fits"] = 1
                            fitted.save(package)
                        else:
                            origin = reused_origin
                        value = fitted.score(held.view)
                        replay = FittedEnsemble.load(package).score(held.view)
                        if value.record() != replay.record():
                            raise ValueError("saved expert prediction differs from the fitted state")
                        evidence[spec.expert_id] = value
                        calibrations[spec.expert_id] = fitted.temperature_record
                    except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
                        view = held.view
                        counts = tuple(int(value) for value in view.decision.membership.sum(axis=0))
                        fit_digest = semantic_sha256({"declared_program": spec.record(), "training": [group.record() for group in training]})
                        identity = CandidateExpertIdentity(spec.expert_id, view.raw_input_digest,
                            None, None, fit_digest, "failed_declared_expert_fit", False)
                        evidence[spec.expert_id] = CandidateEvidence(view.decision.candidate_vocabulary,
                            None, view.participant_key, view.decision_id, view.physical_support_digest,
                            (identity,), view.decision.repetitions, counts, counts,
                            failure_reason=f"{type(error).__name__}: {error}", failure_kind="fit_failure")
                        calibrations[spec.expert_id] = {"strategy": plan["calibration_strategy"], "status": "fit_failure"}
                        origin["kind"] = "reused_score_failure" if origin["reused_package_count"] else "new_fit_failure"
                        if isinstance(error, KernelPriorRankError):
                            origin["failure_diagnostics"] = error.record()
                    origins[spec.expert_id] = origin
                for mixture in mixtures:
                    result = EnsembleScorer(mixture).score({name: evidence[name] for name in mixture.expert_ids})
                    predictions.append({"seed": seed, "mixture": mixture.name, "fold": fold,
                        "evidence": result.record(), "truth": held.target_candidate,
                        "continuous_score_support": {spec.expert_id: {
                            "planned_score_sample_support": [spec.conditioning_samples, len(held.view.samples)],
                            "score_executed": evidence[spec.expert_id].probability_available,
                            "read_sample_support": [0, len(held.view.samples)] if evidence[spec.expert_id].probability_available else None,
                            "conditioning_sample_support": [0, spec.conditioning_samples] if evidence[spec.expert_id].probability_available else None,
                            "score_sample_support": [spec.conditioning_samples, len(held.view.samples)] if evidence[spec.expert_id].probability_available else None,
                            "n_scored_samples": len(held.view.samples) - spec.conditioning_samples if evidence[spec.expert_id].probability_available else 0,
                            "noise_order": spec.noise_order, "noise_estimator": spec.noise_estimator}
                            for spec in specs if spec.family == "continuous_ar" and spec.expert_id in mixture.expert_ids},
                        "training_labels": sum(len(group.labels) for group in training),
                        "requested_authorized_labels": plan["authorized_labels"],
                        "saved_replay_exact": all(evidence[name].probability_available for name in mixture.expert_ids),
                         "temperature_calibration": {name: calibrations[name] for name in mixture.expert_ids}})
                    predictions[-1].update(fit_origin=(origins[mixture.expert_ids[0]] if len(mixture.expert_ids) == 1 else
                        {name: origins[name] for name in mixture.expert_ids}), score_source_digest=semantic_sha256(expected_source))
                atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions),
                    "seed": seed, "fold": fold, "status": "running"})
    requested = []
    coverage = []
    for group in groups:
        view = group.view
        requested.append(DecisionPlanEntry(view.participant_key, view.decision_id, group.target_candidate).record())
        counts = tuple(int(value) for value in view.decision.membership.sum(axis=0))
        coverage.append({"participant_key": view.participant_key, "decision_id": view.decision_id,
                         "counts": dict(zip(view.decision.candidate_vocabulary, counts, strict=True))})
    result = {"status": "completed", "predictions": predictions, "requested": requested,
              "coverage": coverage, "candidate_vocabulary": list(groups[0].view.decision.candidate_vocabulary),
              "evidence_level": groups[0].view.decision.repetitions, "participant_key": groups[0].view.participant_key}
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions), "status": "completed"})
    # The persisted record is the process wire format; fitted metadata stays
    # deeply immutable inside the worker and never enters pickle transport.
    return json.loads((directory / "result.json").read_bytes())


def _independent_subject_impl(row, plan, directory, expected_source, source_snapshots):
    """Fit C once per expert/seed/K; score the same future queue for every pool."""
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    views = prepare_bnci_independent_prefixes(row["path"], source_sha256=row["sha256"],
        budgets=plan["budgets"], profiles=[spec.profile for spec in declared if spec.family == "trial"],
        authorization_preprocessing=parse_preprocessing_spec(plan["authorization_preprocessing"]),
        response_seconds=plan["observation_response_seconds"], pre_roll_seconds=plan["pre_roll_seconds"])
    atomic_write_json(directory / "scopes.json", views.ledger)
    predictions, fits = [], []
    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for source_seed, seed in plan["seed_pairs"]:
            specs = tuple(replace(spec, recipe=None if spec.recipe is None else replace(spec.recipe, seed=seed))
                          for spec in declared)
            sources = {spec.expert_id: source_snapshots[(BNCI2014_008_DATASET_ID, source_seed)] for spec in specs
                if spec.family == "trial" and spec.recipe.initialization == "source"}
            sources, controls = _matched_source_controls(specs, sources)
            for budget, prefix in views.calibration.items():
                roster = EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in specs))
                programs = fit_calibration_prefix_programs(prefix, specs, roster,
                    strategies=plan["calibration_strategies"], source_checkpoints=sources, device=plan["device"])
                package = directory / f"source{source_seed}_target{seed}_K{budget}"
                FittedEnsemble.save_programs(package, programs, participant_key=prefix.target_contract["participant_key"])
                replay_programs = FittedEnsemble.load_programs(package)
                for strategy in plan["calibration_strategies"]:
                    _abort_checkpoint()
                    fitted = replay = None
                    result = programs[strategy]
                    if not isinstance(result, ExpectedSubjectError):
                        fitted = result
                        replay = replay_programs[strategy]
                        calibration = fitted.temperature_record
                    else:
                        calibration = {"strategy": strategy, "status": "unavailable",
                            "failure": result.record(subject=prefix.target_contract["participant_key"])}
                    for spec in specs:
                        use = prefix.label_use["raw"] if spec.family == "continuous_ar" else prefix.label_use["trial"][spec.profile.digest]
                        fits.append({"source_seed": source_seed, "target_seed": seed, "K": budget,
                            "requested_Ks": [int(key) for key, actual in views.ledger["requested_to_actual_K"].items() if actual == budget],
                            "calibration_strategy": strategy, "expert_id": spec.expert_id,
                            "package": package.name, "label_use": use,
                            "matched_source_controls": controls, "temperature_calibration": calibration,
                            "fit_available": fitted is not None and fitted.expert_fit_available[spec.expert_id],
                            "source_only": strategy == "fixed" and budget == 0 and spec.family == "trial"
                                and spec.recipe.initialization == "source"})
                    for view in views.decisions:
                        if fitted is not None:
                            evidence, reloaded = fitted.score_experts(view), replay.score_experts(view)
                            if any(evidence[name].record() != reloaded[name].record() for name in roster.expert_ids):
                                raise ValueError("saved independent-prefix expert replay differs from actual fit")
                        else:
                            evidence = {}
                            counts = tuple(int(count) for count in view.decision.membership.sum(axis=0))
                            observed = tuple(int(count) for count in view.decision.membership[view.decision.evidence_indices >= 0].sum(axis=0))
                            for spec in specs:
                                identity = CandidateExpertIdentity(spec.expert_id, view.raw_input_digest, None, None,
                                    semantic_sha256({"prefix": prefix.record(), "program": spec.record(), "strategy": strategy}),
                                    f"temperature:{result.code.value}", False)
                                evidence[spec.expert_id] = CandidateEvidence(view.decision.candidate_vocabulary, None,
                                    view.participant_key, view.decision_id, view.physical_support_digest, (identity,),
                                    view.decision.repetitions, counts, observed,
                                    failure_reason=str(result), failure_kind="not_available"
                                    if result.code == SubjectFailureCode.INSUFFICIENT_CALIBRATION_GROUPS else "fit_failure")
                        for mixture in mixtures:
                            value = EnsembleScorer(mixture).score({name: evidence[name] for name in mixture.expert_ids})
                            predictions.append({"source_seed": source_seed, "target_seed": seed, "seed": seed,
                                "K": budget, "calibration_strategy": strategy, "mixture": mixture.name, "evidence": value.record(),
                                "truth": views.scoring_truth[view.decision_id], "actual_authorized_labels": budget,
                                "requested_Ks": [int(key) for key, actual in views.ledger["requested_to_actual_K"].items() if actual == budget],
                                "saved_replay_exact": fitted is not None,
                                "temperature_calibration": calibration,
                                "prediction_read_support": {spec.expert_id: {
                                    "raw_scope": list(view.raw_scope) if view.samples is not None else None,
                                    "available_sample": view.available_sample, "profile": None if spec.profile is None else spec.profile.record(),
                                    "filtering_scope": "independent_test_decision" if spec.family == "trial" else "native_unfiltered_decision"}
                                    for spec in specs if spec.expert_id in mixture.expert_ids},
                                "score_source_digest": semantic_sha256(expected_source)})
                atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions),
                    "source_seed": source_seed, "target_seed": seed, "K": budget, "status": "running"})
    requested = [DecisionPlanEntry(view.participant_key, view.decision_id, views.scoring_truth[view.decision_id]).record()
                 for view in views.decisions]
    coverage = [{"participant_key": view.participant_key, "decision_id": view.decision_id,
        "counts": dict(zip(view.decision.candidate_vocabulary, map(int, view.decision.membership.sum(axis=0)), strict=True))}
        for view in views.decisions]
    result = {"status": "completed", "predictions": predictions, "fits": fits, "requested": requested,
        "requested_to_actual_K": views.ledger["requested_to_actual_K"], "budget_adjustments": views.ledger["budget_adjustments"],
        "coverage": coverage, "candidate_vocabulary": list(views.decisions[0].decision.candidate_vocabulary),
        "evidence_level": views.decisions[0].decision.repetitions, "participant_key": views.decisions[0].participant_key}
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions), "status": "completed"})
    return json.loads((directory / "result.json").read_bytes())


def _subject(row, plan, directory, expected_source, source_snapshots):
    directory = Path(directory)
    try:
        return _subject_impl(row, plan, directory, expected_source, source_snapshots)
    except BaseException as error:
        if _WORKER_ABORT is not None:
            _WORKER_ABORT.set()
        progress = directory / "progress.json"
        previous = json.loads(progress.read_bytes()) if progress.exists() else {"completed_prediction_cells": 0}
        atomic_write_json(progress, {**previous, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def run_study(plan_path: str | Path, output: str | Path):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    if plan["schema"] != SCHEMA or plan["evidence_scope"] not in {"software_correctness", "public_calibration_development"}:
        raise ValueError("candidate ensemble plan needs an explicit supported evidence scope")
    if plan["evidence_scope"] == "public_calibration_development":
        verified_executed_snapshot(plan["source_snapshot_manifest"], root=ROOT)
    independent = plan.get("evaluation_design") == "independent_prefix"
    pairs = paired_pipeline_seeds(plan["seed_pairs"]) if independent else None
    strategies = plan["calibration_strategies"] if independent else [plan["calibration_strategy"]]
    if (not isinstance(strategies, list) or not strategies or len(set(strategies)) != len(strategies)
            or set(strategies) - {"fixed", "grouped_temperature"}):
        raise ValueError("calibration programs must be explicit distinct fixed/grouped_temperature declarations")
    seeds = [pair[1] for pair in pairs] if independent else plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("target-only seeds must be explicit unique nonnegative integers")
    expert_programs = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    comparison_role = plan.get("comparison_role", "candidate_ensemble")
    prior_panel = _calibration_prior_panel(expert_programs) if comparison_role == "fixed_reference_kernel_prior" else None
    ar_panel = None if prior_panel is not None else _fixed_prefix_ar_panel(expert_programs, comparison_role)
    names = {spec.expert_id for spec in expert_programs}
    if len(names) != len(expert_programs):
        raise ValueError("candidate expert roster contains duplicate names")
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    if not mixtures or len({value.name for value in mixtures}) != len(mixtures) or any(
        not set(value.expert_ids) <= names for value in mixtures
    ):
        raise ValueError("candidate mixtures require unique names and declared expert producers")
    if plan["baseline_mixture"] not in {value.name for value in mixtures}:
        raise ValueError("baseline mixture is not declared")
    if prior_panel is not None and (plan["calibration_strategy"] != "fixed" or plan["source_checkpoints"]
            or {value.name: value.expert_ids for value in mixtures} != {name: (name,) for name in names}):
        raise ValueError("calibration prior panel requires fixed temperatures and one unmixed output per declared expert")
    if type(plan["workers"]) is not int or plan["workers"] < 1 or type(plan["cpu_threads_per_worker"]) is not int or plan["cpu_threads_per_worker"] < 1:
        raise ValueError("candidate execution workers and thread budgets must be explicit positive integers")
    # A caller may declare a fixed source for target-only seed probes; these are
    # never labeled as independent full source-to-target pipeline replicates.
    if set(plan["source_checkpoints"]) - names:
        raise ValueError("source checkpoint names differ from the expert roster")
    normalized = dict(plan)
    normalized.update(comparison_role=comparison_role, ar_panel=ar_panel)
    if prior_panel is not None:
        normalized["prior_panel"] = prior_panel
    normalized["records"] = [{**row, "path": str((plan_path.parent / row["path"]).resolve())}
                              for row in plan["records"]]
    normalized["source_checkpoints"] = {name: str((plan_path.parent / path).resolve())
                                        for name, path in plan["source_checkpoints"].items()}
    if not normalized["records"] or len({row["path"] for row in normalized["records"]}) != len(normalized["records"]):
        raise ValueError("candidate study requires unique nonempty physical source records")
    if len({row["sha256"] for row in normalized["records"]}) != len(normalized["records"]):
        raise ValueError("one original raw artifact cannot count as several independent participants")
    if independent:
        if plan["source_checkpoints"] or plan.get("resume") is not None:
            raise ValueError("independent prefixes use the outer-study/seed source catalog")
        catalog = SourceSeedCatalog(plan["source_seed_catalog"], base_directory=plan_path.parent,
                                    evidence_scope=plan["evidence_scope"])
        source_snapshots = {(entry["outer_target_study"], entry["source_seed"]):
            FrozenSourceCheckpoint(catalog.content(entry["outer_target_study"], entry["source_seed"]))
            for entry in catalog.record()}
        normalized.update(source_seed_catalog=catalog.record(), seed_pairs=[list(pair) for pair in pairs], seeds=seeds)
    else:
        source_snapshots = {name: FrozenSourceCheckpoint.from_path(path)
                            for name, path in normalized["source_checkpoints"].items()}
        source_snapshots, normalized["matched_controls"] = _matched_source_controls(expert_programs, source_snapshots)
        for binding in normalized["matched_controls"]:
            normalized["source_checkpoints"].setdefault(binding["scratch"], normalized["source_checkpoints"][binding["source"]])
        normalized["source_checkpoint_sha256"] = {name: snapshot.digest for name, snapshot in source_snapshots.items()}
    source = _executing_source()
    normalized["resume_binding"] = _prepare_package_resume(normalized, source)
    if normalized["resume_binding"] is not None and any(output.is_relative_to(Path(normalized["resume_binding"][name]))
            for name in ("original_run", "original_execution_root")):
        raise ValueError("resumed output must preserve the original run and execution checkout read-only")
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "plan.json", normalized)
    atomic_write_json(output / "executed_source.json", source)
    atomic_write_json(output / "progress.json", {"status": "running", "completed_subjects": 0})
    completed = []
    try:
        if plan["workers"] == 1:
            for index, row in enumerate(normalized["records"]):
                completed.append(_subject(row, normalized, output / f"participant_{index:03d}", source, source_snapshots))
        else:
            context = get_context("spawn")
            abort = context.Event()
            rows = iter(enumerate(normalized["records"]))
            with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=context,
                    initializer=_initialize_worker_abort, initargs=(abort,)) as pool:
                pending = set()
                try:
                    while True:
                        while len(pending) < plan["workers"]:
                            entry = next(rows, None)
                            if entry is None:
                                break
                            index, row = entry
                            pending.add(pool.submit(_subject, row, normalized,
                                output / f"participant_{index:03d}", source, source_snapshots))
                        if not pending:
                            break
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            completed.append(future.result())
                            atomic_write_json(output / "progress.json", {"status": "running", "completed_subjects": len(completed)})
                except BaseException:
                    abort.set()
                    for future in pending:
                        future.cancel()
                    raise
        completed.sort(key=lambda row: row["participant_key"])
        requested = tuple(DecisionPlanEntry(**entry) for row in completed for entry in row["requested"])
        roster = tuple(row["participant_key"] for row in completed)
        cache_identity = semantic_sha256(normalized["records"])
        decision_plan = DecisionPlanContract(cache_identity, semantic_sha256([row.record() for row in requested]), roster, requested)
        coverage = {DecisionKey(item["participant_key"], item["decision_id"]): CandidateCoverage.from_mappings(item["counts"], item["counts"])
                    for row in completed for item in row["coverage"]}
        vocabularies = {tuple(row["candidate_vocabulary"]) for row in completed}
        levels = {row["evidence_level"] for row in completed}
        if len(vocabularies) != 1 or len(levels) != 1:
            raise ValueError("candidate study mixes different physical vocabularies or evidence levels")
        metrics = {}
        budgets = sorted({cell["K"] for row in completed for cell in row["predictions"]}) if independent else [None]
        for budget in budgets:
            for strategy in strategies:
                for mixture in mixtures:
                    scores = {seed: [CandidateEvidence.from_record(cell["evidence"]) for row in completed
                        for cell in row["predictions"] if cell["mixture"] == mixture.name and cell["seed"] == seed
                        and (not independent or cell["K"] == budget and cell["calibration_strategy"] == strategy)] for seed in seeds}
                    name = f"{mixture.name}@{strategy}@K{budget}" if independent else mixture.name
                    metrics[name] = evaluate_candidate_evidence(scores, decision_plan=decision_plan,
                        seeds=seeds, candidate_vocabulary=next(iter(vocabularies)), evidence_level=next(iter(levels)),
                        requested_coverage=coverage)
        contrasts = {}
        for budget in budgets:
            for strategy in strategies:
                baseline_name = f"{plan['baseline_mixture']}@{strategy}@K{budget}" if independent else plan["baseline_mixture"]
                for mixture in mixtures:
                    name = f"{mixture.name}@{strategy}@K{budget}" if independent else mixture.name
                    if name != baseline_name:
                        contrasts[name] = compare_candidate_probability_metrics(metrics[baseline_name], metrics[name]).record()
        report = {"schema": "n2p3_candidate_ensemble_report/1", "status": "completed",
                  "evidence_scope": plan["evidence_scope"], "randomness_scope": "paired_source_target_pipeline" if independent else "target_only_fixed_source",
                  "comparison_role": comparison_role, "ar_panel": ar_panel,
                  "decision_plan": decision_plan.record(), "source_digest": semantic_sha256(source),
                  "participants": completed, "metrics": {name: value.record() for name, value in metrics.items()},
                  "contrasts": contrasts}
        if independent:
            report.update(evaluation_design="independent_prefix", seed_pairs=[list(pair) for pair in pairs],
                calibration_strategies=strategies,
                requested_budgets=plan["budgets"], actual_budgets=budgets,
                source_seed_catalog=normalized["source_seed_catalog"],
                source_replication="not_applicable_no_source_NN" if not source_snapshots else plan["evidence_scope"])
        if prior_panel is not None:
            report["prior_panel"] = prior_panel
            report["prior_shape_contrasts"] = [{**row,
                "comparison": compare_candidate_probability_metrics(metrics[row["baseline"]], metrics[row["alternative"]]).record()}
                for row in prior_panel["shape_contrasts"]]
        atomic_write_json(output / "result.json", report)
        atomic_write_json(output / "progress.json", {"status": "completed", "completed_subjects": len(completed)})
        return report
    except BaseException as error:
        atomic_write_json(output / "progress.json", {"status": "failed", "completed_subjects": len(completed),
                          "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(args.plan, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
