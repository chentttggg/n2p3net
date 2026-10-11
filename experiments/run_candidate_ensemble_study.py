"""Execute declared candidate ensembles on calibration-group development folds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from contextlib import nullcontext
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
    UnidentifiableFitError,
    atomic_write_json,
)
from research.nested_selection_replay import InnerReplayRecorder  # noqa: E402
from research.prefix_inner_replay import (  # noqa: E402
    persist_prefix_inner_replay,
    replay_prefix_inner_replay,
    unavailable_prefix_expert_scores,
)
from research.source_catalog import SourceSeedCatalog, paired_pipeline_seeds  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.checkpoint import (  # noqa: E402
    ScopedSourceCheckpointPreparation,
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
from transfer.noise_model import freeze_metadata  # noqa: E402
from transfer.outcomes import CandidateCoverage, DecisionKey  # noqa: E402

SCHEMA = "n2p3_candidate_ensemble_study/1"
_WORKER_ABORT = None


def _initialize_worker_abort(signal):
    global _WORKER_ABORT
    _WORKER_ABORT = signal


def _abort_checkpoint():
    if _WORKER_ABORT is not None and _WORKER_ABORT.is_set():
        raise RuntimeError("candidate matrix stopped after a peer contract failure")


def _file_receipt(path, base):
    raw_path = Path(path)
    path, base = raw_path.resolve(), Path(base).resolve()
    if not path.is_relative_to(base) or raw_path.is_symlink():
        raise ValueError("candidate artifact escapes its owned participant directory")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(base).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def _sealed_record(path, record):
    atomic_write_json(path, {"record": record, "record_sha256": semantic_sha256(record)})


def _read_sealed_record(path):
    payload = json.loads(Path(path).read_bytes())
    if semantic_sha256(payload["record"]) != payload["record_sha256"]:
        raise ValueError("candidate artifact receipt changed")
    return payload["record"]


def _plan_authority(plan):
    # The scientific plan is retained; native threads are supplied by the one
    # live quota owner and recorded separately in the participant receipt.
    return {key: value for key, value in plan.items() if key != "cpu_threads_per_worker"}


def _pipeline_requests(plan, record_index):
    independent = plan.get("evaluation_design") == "independent_prefix"
    strategies = plan["calibration_strategies"] if independent else [plan["calibration_strategy"]]
    pairs = plan["seed_pairs"] if independent else [(None, seed) for seed in plan["seeds"]]
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    applicable = plan.get("mixture_calibration_strategies", {})
    if plan.get("temperature_expert_ids") is not None:
        ids = set(plan["temperature_expert_ids"])
        applicable = {value.name: [strategy for strategy in strategies if strategy == "fixed"
            or set(value.expert_ids) & ids] for value in mixtures}
    row = plan["records"][record_index]
    return [{"record_index": record_index, "record_sha256": row["sha256"],
        "declared_subject": row.get("subject"), "source_seed": pair[0], "target_seed": pair[1],
        "requested_K": budget, "actual_K": None, "calibration_strategy": strategy, "mixture": value.name,
        "decision_identity": "unresolved", "physical_fit_attempts": None, "status": "not_started"}
        for pair in pairs for budget in (plan["budgets"] if independent else [None])
        for strategy in strategies for value in mixtures
        if strategy in applicable.get(value.name, strategies)]


def _resolved_pipeline_requests(plan, record_index, requested, scopes):
    return [{**request,
        "actual_K": scopes["requested_to_actual_K"][str(request["requested_K"])] if "requested_to_actual_K" in scopes else None,
        "requested": entry, "decision_identity": "resolved", "status": "requested"}
        for request in _pipeline_requests(plan, record_index) for entry in requested]


def _expected_fit_roster(plan, scopes):
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    independent = plan.get("evaluation_design") == "independent_prefix"
    pairs = plan["seed_pairs"] if independent else [(None, seed) for seed in plan["seeds"]]
    expected = {}
    for source_seed, seed in pairs:
        specs = tuple(replace(spec, recipe=None if spec.recipe is None else replace(spec.recipe, seed=seed)) for spec in declared)
        sources = dict(plan.get("source_checkpoint_sha256", {}))
        if independent:
            digests = {entry["source_seed"]: entry["checkpoint_sha256"] for entry in plan["source_seed_catalog"]
                       if entry["outer_target_study"] == BNCI2014_008_DATASET_ID}
            sources = {spec.expert_id: digests[source_seed] for spec in specs
                       if spec.family == "trial" and spec.recipe.initialization == "source"}
            for source in specs:
                if source.expert_id not in sources:
                    continue
                recipe = source.recipe.record()
                recipe.pop("name")
                recipe.pop("initialization")
                if recipe["input_statistics"] == "source":
                    recipe["input_statistics"] = "target_prefix"
                for control in specs:
                    if control.family != "trial" or control.recipe.initialization != "scratch" or control.profile.digest != source.profile.digest:
                        continue
                    candidate = control.recipe.record()
                    candidate.pop("name")
                    candidate.pop("initialization")
                    if recipe == candidate:
                        sources[control.expert_id] = sources[source.expert_id]
            for budget, prefix in scopes["calibration"].items():
                package = f"source{source_seed}_target{seed}_K{budget}"
                expected[package] = {"package": package, "source_seed": source_seed, "target_seed": seed, "K": int(budget),
                    "expert_programs": [spec.record() for spec in specs],
                    "ensemble_program": EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in specs)).record(),
                    "training_digest": semantic_sha256(prefix), "target_contract": prefix["target_contract"],
                    "source_checkpoint_sha256": sources}
        else:
            for fold in range(scopes["actual_group_count"]):
                for expert_index, spec in enumerate(specs):
                    package = f"seed{seed}_fold{fold}_expert{expert_index}"
                    expected[package] = {"package": package, "seed": seed, "fold": fold,
                        "expert_programs": [spec.record()],
                        "ensemble_program": EnsembleProgram.equal_weights(spec.expert_id, (spec.expert_id,)).record(),
                        "source_checkpoint_sha256": {spec.expert_id: sources[spec.expert_id]} if spec.expert_id in sources else {}}
    return expected


def _write_candidate_json(path, payload, input_admission=None):
    if input_admission is not None:
        return input_admission.write_json(path, payload)
    return atomic_write_json(path, payload)


def _participant_ledger(row, plan, directory, source, views, truths, scopes, fit_requests):
    requested = [DecisionPlanEntry(view.participant_key, view.decision_id, truths[view.decision_id]).record()
                 for view in views]
    coverage = [{"participant_key": view.participant_key, "decision_id": view.decision_id,
        "counts": dict(zip(view.decision.candidate_vocabulary, map(int, view.decision.membership.sum(axis=0)), strict=True))}
        for view in views]
    record_index = plan["records"].index(row)
    resolved = _resolved_pipeline_requests(plan, record_index, requested, scopes)
    ledger = {"schema": "n2p3_candidate_participant_ledger/1", "record_index": record_index, "record": row,
        "plan_digest": semantic_sha256(_plan_authority(plan)), "execution_source": source,
        "native_threads": plan["cpu_threads_per_worker"], "scopes_sha256": _file_receipt(directory / "scopes.json", directory)["sha256"],
        "requested": requested, "coverage": coverage, "candidate_vocabulary": list(views[0].decision.candidate_vocabulary),
        "evidence_level": views[0].decision.repetitions, "participant_key": views[0].participant_key,
        "pipeline_requests": resolved, "fit_requests": fit_requests}
    if "requested_to_actual_K" in scopes:
        ledger.update(requested_to_actual_K=scopes["requested_to_actual_K"], budget_adjustments=scopes["budget_adjustments"])
    _sealed_record(directory / "request_ledger.json", ledger)
    return ledger


def _fit_started(directory, request):
    _sealed_record(directory / "fit_receipts" / f"{request['package']}.started.json", request)


def _fit_saved(directory, request, *, origin=None):
    package = directory / request["package"]
    files = [_file_receipt(path, directory) for path in sorted(package.rglob("*")) if path.is_file()]
    record = {"schema": "n2p3_candidate_fit_receipt/1", "request": request,
        "status": "saved", "files": files, "origin": origin}
    _sealed_record(directory / "fit_receipts" / f"{request['package']}.saved.json", record)


def _prediction_saved(directory, prediction):
    # Each immutable probability output is committed separately. A native exit
    # cannot erase earlier cells or force quadratic whole-participant rewrites.
    identity = {key: prediction[key] for key in ("source_seed", "target_seed", "seed", "K", "fold",
        "calibration_strategy", "mixture") if key in prediction}
    identity.update(participant_key=prediction["evidence"]["participant_key"], decision_id=prediction["evidence"]["decision_id"])
    _sealed_record(directory / "prediction_receipts" / f"{semantic_sha256(identity)}.json", prediction)


def _fits_saved(directory, package, fits):
    _sealed_record(directory / "fit_receipts" / f"{package}.replayed.json",
                   {"package": package, "status": "replayed", "fits": fits})


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
    matched = {(spec.conditioning_samples, spec.ridge,
                semantic_sha256(None if spec.dictionary_spec is None else spec.dictionary_spec.record())) for spec in experts}
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
    matched = {(spec.conditioning_samples, semantic_sha256(spec.dictionary_spec.record())) for spec in experts}
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


def _dictionary_panel(experts):
    """An endpoint comparison uses one ambient cosine, with changed hard capacity."""
    expected = {("gaussian", "free"), ("cosine", "free"), ("cosine", "hard_zero")}
    if len(experts) != 3 or any(spec.family != "continuous_ar" for spec in experts):
        raise ValueError("dictionary endpoint panel requires exactly three declared raw experts")
    actual = {(spec.dictionary_spec.family, spec.dictionary_spec.endpoint) for spec in experts}
    matched = {(spec.conditioning_samples, spec.noise_order, spec.noise_estimator, spec.dictionary_spec.response_seconds,
                spec.dictionary_spec.n_basis, semantic_sha256(spec.prior_spec.record())) for spec in experts}
    if actual != expected or len(matched) != 1 or any(spec.conditioning_samples != 32 for spec in experts):
        raise ValueError("dictionary endpoints require matched native support, ambient dimension, fixed32, noise and prior programs")
    return {"schema": "n2p3_dictionary_endpoint_panel/1", "same_ambient_cosine_for_free_and_hard": True,
            "hard_endpoint": "kernel_of_E_at_0_and_actual_L_over_sample_rate",
            "capacity_matching_claimed": False, "hard_endpoint_changes_function_dimension": True,
            "score_estimand": "conditional_tail_with_fixed_candidate_prior",
            "reference_prior_energy_and_df_reported_separately": True,
            "arms": {spec.expert_id: spec.dictionary_spec.record() for spec in experts}}


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
    fit_requests = []
    for seed in plan["seeds"]:
        for fold in range(len(groups)):
            training = [group for index, group in enumerate(groups) if index != fold]
            for expert_index, expert in enumerate(declared):
                spec = replace(expert, recipe=None if expert.recipe is None else replace(expert.recipe, seed=seed))
                fit_requests.append({"package": f"seed{seed}_fold{fold}_expert{expert_index}",
                    "seed": seed, "fold": fold, "expert_programs": [spec.record()],
                    "ensemble_program": EnsembleProgram.equal_weights(spec.expert_id, (spec.expert_id,)).record(),
                    "training_digest": semantic_sha256([group.record() for group in training]),
                    "source_checkpoint_sha256": {spec.expert_id: source_snapshots[spec.expert_id].digest}
                        if spec.expert_id in source_snapshots else {}})
    ledger = _participant_ledger(row, plan, directory, expected_source, [group.view for group in groups],
        {group.view.decision_id: group.target_candidate for group in groups}, grouped.ledger, fit_requests)
    fit_requests = {request["package"]: request for request in fit_requests}
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
                    request = fit_requests[package.name]
                    _fit_started(directory, request)
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
                        _fit_saved(directory, request, origin=origin)
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
                        _sealed_record(directory / "fit_receipts" / f"{package.name}.failed.json",
                            {"request": request, "failure": f"{type(error).__name__}: {error}", "origin": origin})
                    origins[spec.expert_id] = origin
                    _fits_saved(directory, package.name, [{"package": package.name, "seed": seed, "fold": fold,
                        "expert_id": spec.expert_id, "fit_origin": origin, "temperature_calibration": calibrations[spec.expert_id]}])
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
                    _prediction_saved(directory, predictions[-1])
                atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions),
                    "seed": seed, "fold": fold, "status": "running"})
    result = {"status": "completed", "predictions": predictions, **{key: ledger[key] for key in
        ("requested", "coverage", "candidate_vocabulary", "evidence_level", "participant_key")}}
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
    paired_specs, paired_sources, paired_controls = {}, {}, {}
    for source_seed, seed in plan["seed_pairs"]:
        specs = tuple(replace(spec, recipe=None if spec.recipe is None else replace(spec.recipe, seed=seed))
                      for spec in declared)
        sources = {spec.expert_id: source_snapshots[(BNCI2014_008_DATASET_ID, source_seed)] for spec in specs
            if spec.family == "trial" and spec.recipe.initialization == "source"}
        paired_specs[source_seed] = specs
        paired_sources[source_seed], paired_controls[source_seed] = _matched_source_controls(specs, sources)
    fit_requests = [{"package": f"source{source_seed}_target{seed}_K{budget}",
        "source_seed": source_seed, "target_seed": seed, "K": budget,
        "expert_programs": [spec.record() for spec in paired_specs[source_seed]],
        "ensemble_program": EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in declared)).record(),
        "training_digest": semantic_sha256(prefix.record()), "target_contract": prefix.target_contract,
        "source_checkpoint_sha256": {name: checkpoint.digest for name, checkpoint in paired_sources[source_seed].items()}}
        for source_seed, seed in plan["seed_pairs"] for budget, prefix in views.calibration.items()]
    ledger = _participant_ledger(row, plan, directory, expected_source, views.decisions,
                                views.scoring_truth, views.ledger, fit_requests)
    fit_requests = {request["package"]: request for request in fit_requests}
    predictions, fits = [], []
    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for source_seed, seed in plan["seed_pairs"]:
            source_preparation = ScopedSourceCheckpointPreparation()
            specs, sources, controls = paired_specs[source_seed], paired_sources[source_seed], paired_controls[source_seed]
            for budget, prefix in views.calibration.items():
                roster = EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in specs))
                package = directory / f"source{source_seed}_target{seed}_K{budget}"
                request = fit_requests[package.name]
                _fit_started(directory, request)
                recorder = InnerReplayRecorder()
                prediction_start = len(predictions)
                prefix_context = {
                    "source_seed": source_seed,
                    "target_seed": seed,
                    "requested_Ks": [int(key) for key, actual in views.ledger["requested_to_actual_K"].items()
                                     if actual == budget],
                    "requested_to_actual_K": views.ledger["requested_to_actual_K"],
                    "test_scope": views.ledger["test_scope"],
                    "requested_decision_ids": [view.record_id for view in views.decisions],
                    "phase": "independent_prefix",
                }
                programs = fit_calibration_prefix_programs(prefix, specs, roster,
                    strategies=plan["calibration_strategies"], source_checkpoints=sources, device=plan["device"],
                    temperature_expert_ids=plan.get("temperature_expert_ids"),
                    inner_replay_recorder=recorder, inner_replay_context=prefix_context,
                    _source_preparation=source_preparation)
                FittedEnsemble.save_programs(package, programs, participant_key=prefix.target_contract["participant_key"])
                for strategy in plan["calibration_strategies"]:
                    _abort_checkpoint()
                    fitted = None
                    result = programs[strategy]
                    if not isinstance(result, ExpectedSubjectError):
                        fitted = result
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
                                and spec.recipe.initialization == "source",
                            "temperature_calibrated": strategy == "grouped_temperature" and spec.expert_id in
                                plan.get("temperature_expert_ids", roster.expert_ids)})
                predictions_by_strategy = {strategy: [] for strategy in plan["calibration_strategies"]}
                for view in views.decisions:
                    _abort_checkpoint()
                    shared = FittedEnsemble.score_programs(programs, view)
                    for strategy in plan["calibration_strategies"]:
                        result = programs[strategy]
                        fitted = None if isinstance(result, ExpectedSubjectError) else result
                        calibration = (fitted.temperature_record if fitted is not None else
                            {"strategy": strategy, "status": "unavailable",
                             "failure": result.record(subject=prefix.target_contract["participant_key"])})
                        if fitted is not None:
                            evidence = shared[strategy]
                        else:
                            evidence = unavailable_prefix_expert_scores(prefix.record(),
                                [spec.record() for spec in specs], strategy,
                                result.record(subject=prefix.target_contract["participant_key"]), view)
                        recorder.record_prefix_future_score(strategy, view=view, evidence=evidence)
                        for mixture in mixtures:
                            if strategy not in plan.get("mixture_calibration_strategies", {}).get(mixture.name, plan["calibration_strategies"]):
                                continue
                            value = EnsembleScorer(mixture).score({name: evidence[name] for name in mixture.expert_ids})
                            predictions_by_strategy[strategy].append({"source_seed": source_seed, "target_seed": seed, "seed": seed,
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
                # Preserve the established strategy-major terminal and seal
                # order while sharing raw expert scores only in this view.
                for strategy in plan["calibration_strategies"]:
                    predictions.extend(predictions_by_strategy[strategy])
                persist_prefix_inner_replay(prefix, programs, package, recorder=recorder,
                    source_checkpoints=sources, context=prefix_context, device=plan["device"])
                _fit_saved(directory, request)
                admitted = replay_prefix_inner_replay(package, prefix=prefix, future_views=views.decisions,
                    source_checkpoints=sources, context=prefix_context, device=plan["device"],
                    expected_record_sha256=recorder.prefix_receipt_sha256)
                if (len(admitted["scores"]) != len(views.decisions)
                        or admitted["exact_score_checks"] < len(recorder.prefix_future_scores)):
                    raise ValueError("prefix replay did not verify its complete requested future score roster")
                _fits_saved(directory, package.name, [value for value in fits if value["package"] == package.name])
                for prediction in predictions[prediction_start:]:
                    prediction["prefix_replay_receipt_sha256"] = recorder.prefix_receipt_sha256
                    _prediction_saved(directory, prediction)
                atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions),
                    "source_seed": source_seed, "target_seed": seed, "K": budget, "status": "running"})
    result = {"status": "completed", "predictions": predictions, "fits": fits, **{key: ledger[key] for key in
        ("requested", "requested_to_actual_K", "budget_adjustments", "coverage", "candidate_vocabulary", "evidence_level", "participant_key")}}
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


def candidate_participant_adapter(arguments):
    import os

    plan_path, output = Path(arguments["plan"]), Path(arguments["output"])
    # One participant retains the original whole seed/K/expert/program loop.
    # Native threads come from the shared phase admission, not the old plan's
    # standalone pool setting.
    native = int(os.environ["OMP_NUM_THREADS"])
    from research.source_execution import current_input_admission

    owner = current_input_admission()
    writer = owner.artifact_directory(output, arguments["output_estimate_bytes"]) if owner is not None else nullcontext()
    with writer:
        run_study(plan_path, output, _actor_index=arguments["record_index"], _native_threads=native)
    return str(output / f"participant_{arguments['record_index']:03d}" / "result.json")


def candidate_tasks(plan_path, output, resources, *, dependencies=(), task_prefix="candidate"):
    from research.cloud_execution import CloudTask, PhaseResources

    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    return [CloudTask(f"{task_prefix}:{index:03d}",
        "experiments.run_candidate_ensemble_study:candidate_participant_adapter",
        {"plan": str(plan_path), "output": str(output / "participant_groups" / f"participant_{index:03d}"),
         "record_index": index, "output_estimate_bytes": resources["target_worker_memory_bytes"]},
        str(output / "participant_groups" / f"participant_{index:03d}"),
        PhaseResources("calibration", resources["native_threads_per_worker"], resources["target_worker_memory_bytes"],
                       resources["native_threads_per_worker"], plan.get("device", "cpu")), tuple(dependencies))
        for index in range(len(plan["records"]))]


def _admit_candidate_participant(task, outcome, plan, source):
    """One parent ingress for both complete and native-interrupted actors."""
    group = Path(task.output).resolve()
    index = task.arguments["record_index"]
    directory = group / f"participant_{index:03d}"
    expected_path = directory / "result.json"
    if outcome.status == "completed" and Path(outcome.value).resolve() != expected_path:
        raise ValueError("candidate actor returned another participant's result path")
    if not (directory / "request_ledger.json").exists():
        if outcome.status == "completed":
            raise ValueError("completed candidate actor lacks its prefit request ledger")
        return None
    actor_plan = json.loads((group / "plan.json").read_bytes())
    actor_source = json.loads((group / "executed_source.json").read_bytes())
    ledger = _read_sealed_record(directory / "request_ledger.json")
    if (semantic_sha256(_plan_authority(actor_plan)) != semantic_sha256(_plan_authority(plan)) or actor_source != source
            or ledger["execution_source"] != source or ledger["record_index"] != index
            or ledger["record"] != plan["records"][index]
            or ledger["plan_digest"] != semantic_sha256(_plan_authority(plan))
            or ledger["native_threads"] != task.resources.native_threads
            or _file_receipt(directory / "scopes.json", directory)["sha256"] != ledger["scopes_sha256"]):
        raise ValueError("candidate participant source, plan or physical-record authority differs")
    scopes = json.loads((directory / "scopes.json").read_bytes())
    if scopes["source_sha256"] != plan["records"][index]["sha256"]:
        raise ValueError("candidate participant scopes differ from its actual raw record")
    if ledger["pipeline_requests"] != _resolved_pipeline_requests(plan, index, ledger["requested"], scopes):
        raise ValueError("candidate prefit request ledger omits or changes declared pipeline requests")
    requested_keys = {(entry["participant_key"], entry["decision_id"]) for entry in ledger["requested"]}
    scope_keys = {(view["participant_key"], view["decision_id"]) for view in scopes["test_views"]} if "test_views" in scopes else {
        (scopes["participant_key"], group["decision_id"]) for group in scopes["groups"]}
    coverage_keys = {(entry["participant_key"], entry["decision_id"]) for entry in ledger["coverage"]}
    if (requested_keys != scope_keys or coverage_keys != scope_keys or len(requested_keys) != len(ledger["requested"])
            or len(coverage_keys) != len(ledger["coverage"])):
        raise ValueError("candidate requested decisions or coverage omit actual prepared views")
    requests = {value["package"]: value for value in ledger["fit_requests"]}
    expected_requests = _expected_fit_roster(plan, scopes)
    if (len(requests) != len(ledger["fit_requests"]) or set(requests) != set(expected_requests)
            or any(semantic_sha256({key: request[key] for key in expected_requests[name]}) !=
                   semantic_sha256(expected_requests[name])
                   for name, request in requests.items())):
        raise ValueError("candidate fit ledger omits or changes declared physical package requests")
    seals, fits, in_flight, failed_fits, started = {}, [], [], [], set()
    prefix_receipts = {}
    for path in sorted((directory / "fit_receipts").glob("*.saved.json")):
        seal = _read_sealed_record(path)
        request = seal["request"]
        package_name = request["package"]
        if package_name not in requests or request != requests[package_name] or package_name in seals:
            raise ValueError("saved candidate package differs from its prefit authority")
        package = directory / package_name
        if not package.resolve().is_relative_to(directory.resolve()) or package.is_symlink():
            raise ValueError("saved candidate package escapes its participant")
        file_receipts = {receipt["path"]: receipt for receipt in seal["files"]}
        if len(file_receipts) != len(seal["files"]):
            raise ValueError("candidate package seal repeats an artifact")
        for receipt in file_receipts.values():
            if _file_receipt(directory / receipt["path"], directory) != receipt:
                raise ValueError("saved candidate model or probability manifest bytes changed")
        manifest = package / "ensemble.json"
        if manifest.exists():
            payload = json.loads(manifest.read_bytes())
            record = payload["record"]
            if (semantic_sha256(record) != payload["record_sha256"]
                    or semantic_sha256(record["program"]) != semantic_sha256(request["ensemble_program"])
                    or semantic_sha256([entry["program"] for entry in record["experts"]]) != semantic_sha256(request["expert_programs"])
                    or "target_contract" in request and semantic_sha256(record["target_contract"]) != semantic_sha256(request["target_contract"])):
                raise ValueError("candidate package program or fit scope differs from its prefit authority")
            for entry in record["experts"]:
                spec = entry["program"]
                checkpoint = request["source_checkpoint_sha256"].get(spec["expert_id"])
                weights_from_source = spec["family"] == "trial" and spec["recipe"]["initialization"] == "source"
                # The canonical fitter binds matched scratch checkpoints as
                # architecture_only input; source_digest names fitted weights.
                expected_origin = checkpoint if weights_from_source and entry["model"] is not None else None
                if entry["source_digest"] != expected_origin:
                    raise ValueError("candidate expert was fitted from another source checkpoint")
                if "K" in request:
                    prefix = scopes["calibration"][str(request["K"])]
                    if entry["model"] is None:
                        reason = "fit_failure" if entry["unavailable_kind"] == "fit_failure" else "unavailable"
                        fit = {"program": spec, "prefix": prefix, reason: entry["unavailable_reason"]}
                    else:
                        fit = {"program": spec, "authorized_calibration": prefix,
                            "source_checkpoint_sha256": checkpoint,
                            "checkpoint_role": None if checkpoint is None else "pretrained_weights" if weights_from_source else "architecture_only"}
                    if entry["fit_digest"] != semantic_sha256(fit):
                        raise ValueError("candidate expert fit digest differs from its authorized prefix or checkpoint role")
                model_receipt = file_receipts.get(f"{package_name}/{entry['model']}")
                if entry["model"] is not None and (model_receipt is None or model_receipt["sha256"] != entry["model_file_sha256"]):
                    raise ValueError("candidate package manifest names different model bytes")
            if "groups" in scopes:
                training = [group for fold, group in enumerate(scopes["groups"]) if fold != request["fold"]]
                if len(training) != len(record["training_receipts"]) or any(
                    any(actual[key] != expected[key] for key in ("decision_id", "raw_scope", "physical_support_digest", "raw_input_digest"))
                    for actual, expected in zip(record["training_receipts"], training, strict=True)):
                    raise ValueError("candidate held-fold package trained on different physical groups")
        temperatures = package / "temperature_programs.json"
        if temperatures.exists():
            payload = json.loads(temperatures.read_bytes())
            record = payload["record"]
            if (semantic_sha256(record) != payload["record_sha256"] or record["participant_key"] != ledger["participant_key"]
                    or record["base_package_sha256"] != file_receipts.get(f"{package_name}/ensemble.json", {}).get("sha256")
                    or set(record["programs"]) != set(plan["calibration_strategies"])):
                raise ValueError("candidate temperature programmes differ from their saved numerical package")
        if "K" in request:
            prefix_receipt = package / "prefix_inner_replay.json"
            sealed_prefix = file_receipts.get(f"{package_name}/prefix_inner_replay.json")
            if sealed_prefix is None or not prefix_receipt.is_file():
                raise ValueError("candidate prefix package lacks its required captured replay receipt")
            prefix_record = _read_sealed_record(prefix_receipt)
            expected_context = {
                "source_seed": request["source_seed"], "target_seed": request["target_seed"],
                "requested_Ks": [int(key) for key, actual in scopes["requested_to_actual_K"].items()
                                 if actual == request["K"]],
                "requested_to_actual_K": scopes["requested_to_actual_K"],
                "test_scope": scopes["test_scope"],
                "requested_decision_ids": [view["record_id"] for view in scopes["test_views"]],
                "phase": "independent_prefix",
            }
            if (prefix_record["prefix_record_sha256"] != request["training_digest"]
                    or semantic_sha256(prefix_record["expert_programs"]) != semantic_sha256(request["expert_programs"])
                    or semantic_sha256(prefix_record["ensemble_program"]) != semantic_sha256(request["ensemble_program"])
                    or prefix_record["source_checkpoint_digests"] != request["source_checkpoint_sha256"]):
                raise ValueError("candidate prefix replay receipt differs from its prefit request")
            prefix_digest = semantic_sha256(prefix_record)
            replay_prefix_inner_replay(package, context=expected_context,
                expected_record_sha256=prefix_digest, device=plan["device"])
            prefix_receipts[package_name] = {
                "record_sha256": prefix_digest,
                "future_rows": {(row["strategy"], row["input"]["view"]["decision_id"]): freeze_metadata(row)
                                for row in prefix_record["future_score_rows"]},
            }
        origin = seal["origin"]
        if origin is not None:
            producer = (plan["resume_binding"]["original_source_digest"] if origin["reused_package_count"]
                        else semantic_sha256(source))
            if origin["producer_source_digest"] != producer:
                raise ValueError("candidate fit receipt rewrites the model's original producer")
        seals[package_name] = seal
    replayed_packages = set()
    for path in sorted((directory / "fit_receipts").glob("*.replayed.json")):
        record = _read_sealed_record(path)
        if record["package"] not in requests:
            raise ValueError("candidate fit replay is not requested")
        request = requests[record["package"]]
        replayed_packages.add(record["package"])
        if "K" in request:
            prefix = scopes["calibration"][str(request["K"])]
            temperature = json.loads((directory / request["package"] / "temperature_programs.json").read_bytes())["record"]["programs"]
            manifest_path = directory / request["package"] / "ensemble.json"
            actual_experts = {entry["program"]["expert_id"]: entry for entry in
                json.loads(manifest_path.read_bytes())["record"]["experts"]} if manifest_path.exists() else {}
            fit_keys = set()
            for value in record["fits"]:
                spec = next(spec for spec in request["expert_programs"] if spec["expert_id"] == value["expert_id"])
                key = (value["calibration_strategy"], value["expert_id"])
                program = temperature[value["calibration_strategy"]]
                calibration = (program["temperature_calibration"] if program["status"] == "completed" else
                    {"strategy": value["calibration_strategy"], "status": "unavailable", "failure": program["failure"]})
                label_use = prefix["label_use"]["raw"] if spec["family"] == "continuous_ar" else prefix["label_use"]["trial"][
                    CalibrationExpertProgram.from_record(spec).profile.digest]
                if (key in fit_keys or value["package"] != request["package"] or value["K"] != request["K"]
                        or value["source_seed"] != request["source_seed"] or value["target_seed"] != request["target_seed"]
                        or value["label_use"] != label_use or value["requested_Ks"] != [int(key) for key, actual in
                            scopes["requested_to_actual_K"].items() if actual == request["K"]]
                        or value["fit_available"] != (program["status"] == "completed" and actual_experts[value["expert_id"]]["model"] is not None)
                        or value["temperature_calibration"] != calibration):
                    raise ValueError("candidate fit replay ledger differs from its saved programme or actual C authority")
                fit_keys.add(key)
        fits.extend(record["fits"])
    for path in sorted((directory / "fit_receipts").glob("*.started.json")):
        request = _read_sealed_record(path)
        if request != requests.get(request["package"]):
            raise ValueError("candidate inflight fit differs from its request")
        started.add(request["package"])
        if request["package"] not in seals and not path.with_name(path.name.replace(".started.json", ".failed.json")).exists():
            in_flight.append({"request": request, "status": "interrupted_unknown", "physical_successful_fits": None})
    for path in sorted((directory / "fit_receipts").glob("*.failed.json")):
        record = _read_sealed_record(path)
        if record["request"] != requests.get(record["request"]["package"]):
            raise ValueError("candidate failed fit differs from its prefit request")
        failed_fits.append(record)
    predictions = [_read_sealed_record(path) for path in sorted((directory / "prediction_receipts").glob("*.json"))]
    requested_entries = {(entry["participant_key"], entry["decision_id"]): entry for entry in ledger["requested"]}
    coverage = {(entry["participant_key"], entry["decision_id"]): entry["counts"] for entry in ledger["coverage"]}
    support = {(view["participant_key"], view["decision_id"]): view["physical_support_digest"] for view in scopes.get("test_views", ())}
    if "groups" in scopes:
        support = {(scopes["participant_key"], group["decision_id"]): group["physical_support_digest"] for group in scopes["groups"]}
    applicable = {(value["source_seed"], value["target_seed"], value["actual_K"], value["calibration_strategy"],
                   value["mixture"], value["requested"]["decision_id"]) for value in ledger["pipeline_requests"]}
    identities, prediction_by_identity = set(), {}
    for prediction in predictions:
        evidence = CandidateEvidence.from_record(prediction["evidence"])
        key = (evidence.participant_key, evidence.decision_id)
        strategy = prediction.get("calibration_strategy", plan.get("calibration_strategy"))
        identity = (prediction.get("source_seed"), prediction["seed"], prediction.get("K"), strategy,
                    prediction["mixture"], evidence.decision_id)
        if (identity not in applicable or identity in identities or key not in requested_entries
                or prediction["truth"] != requested_entries[key]["target_candidate"]
                or prediction["score_source_digest"] != semantic_sha256(source)
                or evidence.physical_support_digest != support[key]
                or list(evidence.candidate_vocabulary) != ledger["candidate_vocabulary"]
                or evidence.evidence_level != ledger["evidence_level"]
                or dict(zip(evidence.candidate_vocabulary, evidence.expected_event_counts, strict=True)) != coverage[key]):
            raise ValueError("candidate prediction differs from its prefit request or score source")
        if "K" in prediction:
            package = f"source{prediction['source_seed']}_target{prediction['seed']}_K{prediction['K']}"
            prefix_receipt = prefix_receipts.get(package)
            if prefix_receipt is None or prediction.get("prefix_replay_receipt_sha256") != prefix_receipt["record_sha256"]:
                raise ValueError("candidate prediction lacks its admitted prefix replay receipt")
            captured = prefix_receipt["future_rows"].get((strategy, evidence.decision_id))
            if captured is None:
                raise ValueError("candidate prediction lacks its captured requested future row")
            mixture = next(value for value in plan["mixtures"] if value["name"] == prediction["mixture"])
            original = EnsembleScorer(EnsembleProgram.from_record(mixture)).score({
                name: CandidateEvidence.from_record(captured["expert_evidence"][name]) for name in mixture["expert_ids"]})
            if semantic_sha256(original.record()) != semantic_sha256(evidence.record()):
                raise ValueError("candidate prediction probability/failure differs from its captured experts and mixture")
        identities.add(identity)
        prediction_by_identity[identity] = prediction
        if "K" in prediction:
            packages = [f"source{prediction['source_seed']}_target{prediction['seed']}_K{prediction['K']}"]
        else:
            expert_indexes = {value["expert_id"]: index for index, value in enumerate(plan["experts"])}
            packages = [f"seed{prediction['seed']}_fold{prediction['fold']}_expert{expert_indexes[value.expert_id]}"
                        for value in evidence.expert_identities]
        models = {}
        for package_name in packages:
            if package_name in seals and (directory / package_name / "ensemble.json").exists():
                record = json.loads((directory / package_name / "ensemble.json").read_bytes())["record"]
                models.update((entry["program"]["expert_id"], entry["model_digest"]) for entry in record["experts"])
        if evidence.probability_available and (not prediction["saved_replay_exact"] or any(
                models.get(value.expert_id) != value.model_digest for value in evidence.expert_identities)):
            raise ValueError("candidate probability output is not bound to its replayed saved models")
    metadata = {key: ledger[key] for key in ("requested", "coverage", "candidate_vocabulary", "evidence_level", "participant_key")}
    if "requested_to_actual_K" in ledger:
        metadata.update(requested_to_actual_K=ledger["requested_to_actual_K"], budget_adjustments=ledger["budget_adjustments"])
    pipeline = []
    for request in ledger["pipeline_requests"]:
        identity = (request["source_seed"], request["target_seed"], request["actual_K"], request["calibration_strategy"],
                    request["mixture"], request["requested"]["decision_id"])
        prediction = prediction_by_identity.get(identity)
        if "requested_to_actual_K" in ledger:
            request_packages = {f"source{request['source_seed']}_target{request['target_seed']}_K{request['actual_K']}"}
        else:
            fold = next(fold for fold, group in enumerate(scopes["groups"]) if group["decision_id"] == request["requested"]["decision_id"])
            mixture = next(value for value in plan["mixtures"] if value["name"] == request["mixture"])
            request_packages = {f"seed{request['target_seed']}_fold{fold}_expert{expert_index}"
                for expert_index, spec in enumerate(plan["experts"]) if spec["expert_id"] in mixture["expert_ids"]}
        pipeline.append({**request, "status": ("completed" if prediction["evidence"]["log_probability"] is not None
            else "output_unavailable") if prediction is not None else "interrupted_unknown" if request_packages & started else "not_started"})
    unsealed = [_file_receipt(path, directory) for package_name in requests if package_name not in seals
                for path in sorted((directory / package_name).glob("*")) if path.is_file()]
    result = {**metadata, "status": "partial", "predictions": predictions, "fits": fits,
        "artifact_receipts": list(seals.values()), "failed_fit_receipts": failed_fits, "inflight_fits": in_flight,
        "surviving_unsealed_artifacts": unsealed, "pipeline_requests": pipeline, "record_index": index}
    if outcome.status == "completed":
        persisted = json.loads(expected_path.read_bytes())
        fit_complete = replayed_packages == set(requests)
        if "requested_to_actual_K" in ledger:
            actual_fit_keys = {(value["package"], value["calibration_strategy"], value["expert_id"]) for value in fits}
            expected_fit_keys = {(request["package"], strategy, expert["expert_id"]) for request in requests.values()
                for strategy in plan["calibration_strategies"] for expert in request["expert_programs"]}
            fit_complete = fit_complete and actual_fit_keys == expected_fit_keys and len(actual_fit_keys) == len(fits)
        if (persisted["status"] != "completed" or any(persisted[key] != value for key, value in metadata.items())
                or sorted(map(semantic_sha256, persisted["predictions"])) != sorted(map(semantic_sha256, predictions))
                or "fits" in persisted and sorted(map(semantic_sha256, persisted["fits"])) != sorted(map(semantic_sha256, fits))
                or len(identities) != len(applicable) or not fit_complete):
            raise ValueError("completed candidate result differs from its sealed requests or predictions")
        result.update(persisted)
    return result


class CandidateCloudProgramme:
    """Scientific consumer hooks; the caller owns the single CloudExecutor."""

    def __init__(self, plan_path, output, resources, *, task_prefix, dependencies, input_admission=None):
        self.original_plan_path = Path(plan_path).resolve()
        self.output, self.resources = Path(output).resolve(), resources
        self.task_prefix, self.dependencies = task_prefix, tuple(dependencies)
        self.input_admission = input_admission
        self.base_plan = json.loads(self.original_plan_path.read_bytes())
        self.output.mkdir(parents=True, exist_ok=False)
        self.plan_path = self.output / "bound_plan.json"
        self.plan = {**self.base_plan,
            "records": [{**row, "path": str((self.original_plan_path.parent / row["path"]).resolve())}
                        for row in self.base_plan["records"]],
            "source_checkpoints": {name: str((self.original_plan_path.parent / path).resolve())
                                   for name, path in self.base_plan["source_checkpoints"].items()}}
        if self.plan.get("source_snapshot_manifest"):
            self.plan["source_snapshot_manifest"] = str((self.original_plan_path.parent / self.plan["source_snapshot_manifest"]).resolve())
        if self.plan.get("source_seed_catalog"):
            self.plan["source_seed_catalog"] = [{**entry, "path": str((self.original_plan_path.parent / entry["path"]).resolve())}
                                                for entry in self.plan["source_seed_catalog"]]
        self.source, self.normalized, self.admitted, self.admission_failures = None, None, {}, {}
        _write_candidate_json(self.output / "consumer_requests.json", {"schema": "n2p3_candidate_consumer_requests/1",
            "base_plan_sha256": _file_receipt(self.original_plan_path, self.original_plan_path.parent)["sha256"],
            "requests": [request for index in range(len(self.plan["records"])) for request in _pipeline_requests(self.plan, index)]}, input_admission)
        _write_candidate_json(self.plan_path, self.plan, input_admission)
        self.tasks = candidate_tasks(self.plan_path, self.output, resources, dependencies=self.dependencies, task_prefix=task_prefix)

    def bind_source(self, source_seed_catalog=None, source_snapshot=None):
        if source_seed_catalog is not None:
            self.plan["source_seed_catalog"] = source_seed_catalog
        if source_snapshot is not None:
            self.plan["source_snapshot_manifest"] = str(Path(source_snapshot).resolve())
        _write_candidate_json(self.plan_path, self.plan, self.input_admission)
        self.normalized, self.source, _ = _prepare_candidate_plan(self.plan_path, load_sources=False)
        _write_candidate_json(self.output / "plan.json", self.normalized, self.input_admission)
        _write_candidate_json(self.output / "executed_source.json", self.source, self.input_admission)
        return self.tasks

    def on_finished(self, outcome):
        tasks = {task.task_id: task for task in self.tasks}
        if outcome.task_id not in tasks:
            raise ValueError("candidate consumer received another programme's task")
        if self.normalized is None:
            if outcome.status != "blocked":
                raise ValueError("candidate consumer cannot execute before its actual source binding")
            return ()
        try:
            result = _admit_candidate_participant(tasks[outcome.task_id], outcome, self.normalized, self.source)
            if result is not None:
                self.admitted[outcome.task_id] = result
        except Exception as error:
            self.admission_failures[outcome.task_id] = f"{type(error).__name__}: {error}"
            raise
        _write_candidate_json(self.output / "progress.json", {"status": "running",
            "completed_subjects": sum(row["status"] == "completed" for row in self.admitted.values()),
            "resolved_subjects": len(self.admitted)}, self.input_admission)
        return ()

    def finalize(self, results, *, input_admission=None):
        owner = input_admission if input_admission is not None else self.input_admission
        outcomes = [results[task.task_id] for task in self.tasks]
        participants = [self.admitted[task.task_id] for task in self.tasks if task.task_id in self.admitted]
        failures = [{"task_id": value.task_id, "status": value.status, "failure": value.failure,
            "supplementary_failures": value.supplementary_failures} for value in outcomes if value.status != "completed"]
        if self.source is not None and _executing_source() != self.source:
            raise ValueError("candidate execution source changed during the programme")
        if failures:
            report = {"schema": "n2p3_candidate_ensemble_report/1", "status": "failed",
                "evidence_scope": self.plan["evidence_scope"], "participants": participants,
                "requested_records": self.plan["records"], "execution_failures": failures,
                "admission_failures": self.admission_failures,
                "requested_pipeline_cells": [request for task in self.tasks for request in
                    (self.admitted[task.task_id]["pipeline_requests"] if task.task_id in self.admitted else
                     [{**value, "status": results[task.task_id].status, "failure": results[task.task_id].failure}
                      for value in _pipeline_requests(self.plan, task.arguments["record_index"])])],
                "unresolved_record_indexes": [task.arguments["record_index"] for task in self.tasks if task.task_id not in self.admitted],
                "decision_denominator_complete": len(participants) == len(self.tasks),
                "source_digest": None if self.source is None else semantic_sha256(self.source)}
            if participants:
                measured = _finalize_candidate_report(self.normalized, self.source, participants, self.output,
                    status="failed", persist=False)
                report.update(decision_plan=measured["decision_plan"], metrics=measured["metrics"], contrasts=measured["contrasts"],
                    metrics_scope="all_resolved_requested_decisions" if report["decision_denominator_complete"] else "resolved_subset_with_unresolved_roster")
            _write_candidate_json(self.output / "result.json", report, owner)
            _write_candidate_json(self.output / "progress.json", {"status": "failed", "resolved_subjects": len(participants),
                "completed_subjects": sum(row["status"] == "completed" for row in participants)}, owner)
            return report
        if len(participants) != len(self.tasks):
            raise ValueError("candidate programme lacks independently admitted participant results")
        return _finalize_candidate_report(self.normalized, self.source, participants, self.output, input_admission=owner)


def build_candidate_cloud_tasks(plan_path, output, resources, *, source_seed_catalog=None, source_snapshot=None,
                                task_prefix="candidate", dependencies=(), input_admission=None):
    programme = CandidateCloudProgramme(plan_path, output, resources, task_prefix=task_prefix, dependencies=dependencies,
                                        input_admission=input_admission)
    needs_source = any(value["family"] == "trial" and value["recipe"]["initialization"] == "source"
                       for value in programme.plan["experts"])
    ready_source = source_seed_catalog is not None or not needs_source or programme.plan.get("source_seed_catalog")
    ready_freeze = programme.plan["evidence_scope"] == "software_correctness" or source_snapshot or programme.plan.get("source_snapshot_manifest")
    if ready_source and ready_freeze:
        programme.bind_source(source_seed_catalog, source_snapshot)
    return programme


def _prepare_candidate_plan(plan_path, *, native_threads=None, load_sources=True):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_bytes())
    if native_threads is not None:
        plan = {**plan, "cpu_threads_per_worker": native_threads}

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
    spatial = plan.get("raw_spatial_views", {})
    if set(spatial) - {value["expert_id"] for value in plan["experts"] if value["family"] == "continuous_ar"}:
        raise ValueError("raw spatial declarations must identify actual continuous expert producers")
    plan = {**plan, "experts": [{**value, **({"spatial": spatial[value["expert_id"]]} if value["expert_id"] in spatial else {})}
                                for value in plan["experts"]]}
    expert_programs = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    comparison_role = plan.get("comparison_role", "candidate_ensemble")
    prior_panel = _calibration_prior_panel(expert_programs) if comparison_role == "fixed_reference_kernel_prior" else None
    dictionary_panel = _dictionary_panel(expert_programs) if comparison_role == "fixed_dictionary_endpoints" else None
    ar_panel = None if prior_panel is not None or dictionary_panel is not None else _fixed_prefix_ar_panel(expert_programs, comparison_role)
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
    if plan.get("temperature_expert_ids") is not None:
        ids = plan["temperature_expert_ids"]
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= names:
            raise ValueError("temperature experts must be an explicit nonempty subset of the actual roster")
        plan["mixture_calibration_strategies"] = {mixture.name: [strategy for strategy in strategies
            if strategy == "fixed" or set(mixture.expert_ids) & set(ids)] for mixture in mixtures}
    if (prior_panel is not None or dictionary_panel is not None) and (plan["calibration_strategy"] != "fixed" or plan["source_checkpoints"]
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
    if dictionary_panel is not None:
        normalized["dictionary_panel"] = dictionary_panel
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
                                    evidence_scope=plan["evidence_scope"],
                                    expected_source_config=plan.get("expected_source_config"))
        if any(spec.family == "trial" and spec.recipe.initialization == "source" for spec in expert_programs):
            available = {(entry["outer_target_study"], entry["source_seed"]) for entry in catalog.record()}
            required = {(BNCI2014_008_DATASET_ID, pair[0]) for pair in pairs}
            if required != available:
                raise ValueError("candidate source catalog differs from declared outer-study/source-seed producers")
        source_snapshots = {(entry["outer_target_study"], entry["source_seed"]):
            FrozenSourceCheckpoint(catalog.content(entry["outer_target_study"], entry["source_seed"]))
            for entry in catalog.record() if load_sources and entry["outer_target_study"] == BNCI2014_008_DATASET_ID
            and entry["source_seed"] in {pair[0] for pair in pairs}}
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
    return normalized, source, source_snapshots


def run_study(plan_path: str | Path, output: str | Path, *, _actor_index=None, _native_threads=None,
              execution="cloud"):
    """Standalone cloud execution, or the explicit in-process scientific seam."""
    if execution not in {"cloud", "inline"}:
        raise ValueError("candidate execution must be cloud or inline")
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan, source, snapshots = _prepare_candidate_plan(plan_path,
        native_threads=_native_threads, load_sources=_actor_index is not None or execution == "inline")
    if plan["resume_binding"] is not None and any(output.is_relative_to(Path(plan["resume_binding"][name]))
            for name in ("original_run", "original_execution_root")):
        raise ValueError("resumed output must preserve the original run and execution checkout read-only")
    completed = []
    if _actor_index is not None or execution == "inline":
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / "plan.json", plan)
        atomic_write_json(output / "executed_source.json", source)
        atomic_write_json(output / "progress.json", {"status": "running", "completed_subjects": 0})
    try:
        if _actor_index is not None:
            row = plan["records"][_actor_index]
            return _subject(row, plan, output / f"participant_{_actor_index:03d}", source, snapshots)
        if execution == "inline":
            for index, row in enumerate(plan["records"]):
                completed.append(_subject(row, plan, output / f"participant_{index:03d}", source, snapshots))
            if _executing_source() != source:
                raise ValueError("candidate execution source changed during the programme")
            return _finalize_candidate_report(plan, source, completed, output)
        from research.cloud_execution import CloudExecutor
        from research.source_execution import (
            HostMemoryAdmission,
            SourceInputAdmission,
            resolve_source_resources,
        )

        resource_plan = {"source_workers": plan["workers"], "target_workers": plan["workers"],
            "cpu_budget": plan["workers"] * plan["cpu_threads_per_worker"] + 1,
            "source_worker_memory_gib": plan.get("target_worker_memory_gib", .25 if plan["evidence_scope"] == "software_correctness" else 4.),
            "target_worker_memory_gib": .25 if plan["evidence_scope"] == "software_correctness" else 4., **plan}
        resources = resolve_source_resources(resource_plan, len(plan["records"]), plan.get("device", "cpu"))
        context = get_context("spawn")
        with context.Manager() as manager:
            input_owner = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"],
                                               load_workers=plan.get("source_load_workers", resources["source_workers"]))
            admission = HostMemoryAdmission(manager, resources, input_admission=input_owner)
            programme = build_candidate_cloud_tasks(plan_path, output, resources, input_admission=input_owner)
            run = CloudExecutor(admission, max_workers=resources["target_workers"],
                diagnostic_directory=output / "worker_diagnostics",
                identity={"actual_source_root": str(ROOT), "plan_sha256": semantic_sha256(plan)}).run(
                    programme.tasks, on_finished=programme.on_finished)
            return programme.finalize(run.results, input_admission=input_owner)
    except BaseException as error:
        atomic_write_json(output / "progress.json", {"status": "failed", "completed_subjects": len(completed),
                          "error": f"{type(error).__name__}: {error}"})
        raise


def _finalize_candidate_report(plan, source, completed, output, *, status="completed", input_admission=None, persist=True):
    independent = plan.get("evaluation_design") == "independent_prefix"
    pairs = tuple(map(tuple, plan["seed_pairs"])) if independent else None
    strategies = plan["calibration_strategies"] if independent else [plan["calibration_strategy"]]
    seeds = plan["seeds"]
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    normalized = plan
    comparison_role, ar_panel = plan["comparison_role"], plan["ar_panel"]
    prior_panel, dictionary_panel = plan.get("prior_panel"), plan.get("dictionary_panel")
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
    budgets = sorted({budget for row in completed for budget in row["requested_to_actual_K"].values()}) if independent else [None]
    for budget in budgets:
        for strategy in strategies:
            for mixture in mixtures:
                if strategy not in normalized.get("mixture_calibration_strategies", {}).get(mixture.name, strategies):
                    continue
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
                if strategy not in normalized.get("mixture_calibration_strategies", {}).get(mixture.name, strategies):
                    continue
                name = f"{mixture.name}@{strategy}@K{budget}" if independent else mixture.name
                if name != baseline_name:
                    contrasts[name] = (compare_candidate_probability_metrics(metrics[baseline_name], metrics[name]).record()
                        if baseline_name in metrics else {"status": "not_applicable", "reason": "baseline_probability_program_not_declared"})
    report = {"schema": "n2p3_candidate_ensemble_report/1", "status": status,
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
            source_replication="not_applicable_no_source_NN" if not normalized.get("source_seed_catalog", normalized["source_checkpoints"]) else plan["evidence_scope"])
    if prior_panel is not None:
        report["prior_panel"] = prior_panel
        report["prior_shape_contrasts"] = [{**row,
            "comparison": compare_candidate_probability_metrics(metrics[row["baseline"]], metrics[row["alternative"]]).record()}
            for row in prior_panel["shape_contrasts"]]
    if dictionary_panel is not None:
        report["dictionary_panel"] = dictionary_panel
    if persist:
        _write_candidate_json(output / "result.json", report, input_admission)
        _write_candidate_json(output / "progress.json", {"status": status, "completed_subjects": len(completed)}, input_admission)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(args.plan, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
