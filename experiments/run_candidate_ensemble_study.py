"""Execute declared candidate ensembles on calibration-group development folds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
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
    prepare_dataset_independent_prefixes,
)
from transfer.kernel_prior import KernelPriorRankError  # noqa: E402
from transfer.noise_model import freeze_metadata  # noqa: E402
from transfer.outcomes import CandidateCoverage, DecisionKey  # noqa: E402
from transfer.pool_calibration import (  # noqa: E402
    POOL_STRATEGIES,
    POOL_STRATEGY,
    TEMPERATURE_POOL_STRATEGY,
)

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
        applicable = {value.name: [strategy for strategy in strategies if strategy in {"fixed", POOL_STRATEGY}
            or set(value.expert_ids) & ids] for value in mixtures}
    row = plan["records"][record_index]
    requests = [{"record_index": record_index, "record_sha256": row["sha256"],
        "declared_subject": row.get("subject"), "source_seed": pair[0], "target_seed": pair[1],
        "requested_K": budget, "actual_K": None, "calibration_strategy": strategy, "mixture": value.name,
        "decision_identity": "unresolved", "physical_fit_attempts": None, "status": "not_started"}
        for pair in pairs for budget in (plan["budgets"] if independent else [None])
        for strategy in strategies for value in mixtures
        if strategy in applicable.get(value.name, strategies)]
    if plan.get("prefix_selection") is not None:
        requests.extend({"record_index": record_index, "record_sha256": row["sha256"],
            "declared_subject": row.get("subject"), "source_seed": pair[0], "target_seed": pair[1],
            "requested_K": budget, "actual_K": None, "calibration_strategy": "nested_selection",
            "mixture": plan["prefix_selection"]["output_name"], "decision_identity": "unresolved",
            "physical_fit_attempts": None, "status": "not_started"}
            for pair in pairs for budget in plan["budgets"])
    return requests


def _prefix_selection_candidates(plan, seed):
    from research.calibration_selection import CalibrationProgramCandidate

    declared = {value["expert_id"]: CalibrationExpertProgram.from_record(value) for value in plan["experts"]}
    candidates = []
    for row in plan["prefix_selection"]["candidates"]:
        candidate = CalibrationProgramCandidate.from_record(row)
        specs = tuple(replace(declared[spec.expert_id], recipe=None if spec.recipe is None else
            replace(declared[spec.expert_id].recipe, seed=seed)) for spec in candidate.expert_programs)
        candidates.append(replace(candidate, expert_programs=specs))
    return tuple(candidates)


def _prefix_selection_requests(plan, scopes):
    if plan.get("prefix_selection") is None:
        return {}
    options = plan["prefix_selection"]
    requests = {}
    for base in _expected_fit_roster(plan, scopes).values():
        if "pool_probability_strategy" in base:
            continue
        candidates = _prefix_selection_candidates(plan, base["target_seed"])
        names = {spec.expert_id for candidate in candidates for spec in candidate.expert_programs}
        package = base["package"] + "_selection"
        requests[package] = {"package": package, "source_seed": base["source_seed"],
            "target_seed": base["target_seed"], "K": base["K"], "output_name": options["output_name"],
            "candidates": [candidate.record() for candidate in candidates], "baseline_name": options["baseline_name"],
            "minimum_selection_rows": options["minimum_selection_rows"],
            "training_digest": base["training_digest"], "target_contract": base["target_contract"],
            "source_checkpoint_sha256": {name: digest for name, digest in base["source_checkpoint_sha256"].items() if name in names}}
    return requests


def _resolved_pipeline_requests(plan, record_index, requested, scopes):
    return [{**request,
        "actual_K": scopes["requested_to_actual_K"][str(request["requested_K"])] if "requested_to_actual_K" in scopes else None,
        "requested": entry, "decision_identity": "resolved", "status": "requested"}
        for request in _pipeline_requests(plan, record_index) for entry in requested]


def _joint_source_entries(plan):
    """Read typed source entries without rewriting their original stage paths."""
    entries = {}
    for receipt in plan.get("joint_source_catalog", ()):
        path = Path(receipt["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
            raise ValueError("joint candidate source entry differs from its declared original bytes")
        entry = json.loads(content)
        source_id = receipt["source_id"]
        if source_id in entries or entry["source_id"] != source_id:
            raise ValueError("joint candidate source catalogue requires its unique actual source IDs")
        entries[source_id] = entry, path.parent
    return entries


def _joint_sources_for_target(plan, specs, target_contract, *, pipeline_pair, _source_preparation=None):
    """Resolve original joint parameters for one actual target; NN stays separate."""
    from research.joint_source_catalog import (
        JointFlatSourceCatalog,
        admit_huebner_joint_source_data,
        resolve_joint_candidate_sources,
    )
    from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

    joints = tuple(spec for spec in specs if spec.family == "joint_flat")
    if not joints:
        return {}, {}
    bindings = plan["joint_source_bindings"][target_contract["participant_key"]]
    if set(bindings) != {spec.expert_id for spec in joints}:
        raise ValueError("joint source bindings differ from the actual target expert roster")
    entries = _joint_source_entries(plan)
    preparation = _source_preparation or FrozenJointFlatSourcePreparation()
    sources, resolutions = {}, {}
    for source_id in dict.fromkeys(bindings.values()):
        entry, base = entries[source_id]
        selected = tuple(spec for spec in joints if bindings[spec.expert_id] == source_id)
        admissions = []
        if plan["evidence_scope"] == "public_calibration_development":
            declaration = plan["joint_source_data_admission"][source_id]
            if declaration["issuer"] != "Huebner2017_original_roster":
                raise ValueError("joint public source requires its registered original-data issuer")
            scopes = {spec.joint_flat.source_exclusion_scope for spec in selected}
            if len(scopes) != 1:
                raise ValueError("one joint source entry must retain one actual exclusion programme")
            admissions.append(admit_huebner_joint_source_data(entry, target_contract=target_contract,
                exclusion_scope=next(iter(scopes)), source_requests=declaration["source_requests"],
                authorization_preprocessing=parse_preprocessing_spec(declaration["authorization_preprocessing"]),
                response_seconds=declaration["response_seconds"], pre_roll_seconds=declaration["pre_roll_seconds"],
                base_directory=base, _source_preparation=preparation))
        catalog = JointFlatSourceCatalog([entry], base_directory=base,
            evidence_scope=plan["evidence_scope"], data_admissions=admissions, _source_preparation=preparation)
        actual, receipts = resolve_joint_candidate_sources(selected,
            {spec.expert_id: source_id for spec in selected}, catalog,
            target_contract=target_contract, pipeline_pair=pipeline_pair)
        sources.update(actual)
        resolutions.update(receipts)
    return sources, resolutions


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
                       if entry["outer_target_study"] == _target_study(plan)}
            sources = {spec.expert_id: digests[source_seed] for spec in specs
                       if spec.family == "trial" and spec.recipe.initialization == "source"}
            for source in specs:
                if source.family != "trial" or source.expert_id not in sources:
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
                joint_specs = tuple(spec for spec in specs if spec.family == "joint_flat")
                if joint_specs:
                    bindings = plan["joint_source_bindings"][prefix["target_contract"]["participant_key"]]
                    entries = _joint_source_entries(plan)
                    sources.update({spec.expert_id: entries[bindings[spec.expert_id]][0]["codec"]["sha256"]
                                    for spec in joint_specs})
                package = f"source{source_seed}_target{seed}_K{budget}"
                expected[package] = {"package": package, "source_seed": source_seed, "target_seed": seed, "K": int(budget),
                    "expert_programs": [spec.record() for spec in specs],
                    "ensemble_program": EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in specs)).record(),
                    "training_digest": semantic_sha256(prefix), "target_contract": prefix["target_contract"],
                    "source_checkpoint_sha256": sources}
                for mixture in plan["mixtures"]:
                    programme = EnsembleProgram.from_record(mixture)
                    selected = tuple(spec for name in programme.expert_ids for spec in specs if spec.expert_id == name)
                    for strategy in _ordinary_pool_strategies(plan, programme.name):
                        pool_package = _ordinary_pool_package(package, programme, strategy)
                        if pool_package in expected:
                            raise ValueError("distinct ordinary probability programmes share a package identity")
                        expected[pool_package] = {"package": pool_package, "source_seed": source_seed,
                            "target_seed": seed, "K": int(budget), "expert_programs": [spec.record() for spec in selected],
                            "ensemble_program": programme.record(), "training_digest": semantic_sha256(prefix),
                            "target_contract": prefix["target_contract"],
                            "source_checkpoint_sha256": {name: digest for name, digest in sources.items() if name in programme.expert_ids},
                            "pool_probability_strategy": strategy, "mixture_name": programme.name, "strategies": [strategy]}
        else:
            for fold in range(scopes["actual_group_count"]):
                for expert_index, spec in enumerate(specs):
                    package = f"seed{seed}_fold{fold}_expert{expert_index}"
                    expected[package] = {"package": package, "seed": seed, "fold": fold,
                        "expert_programs": [spec.record()],
                        "ensemble_program": EnsembleProgram.equal_weights(spec.expert_id, (spec.expert_id,)).record(),
                        "source_checkpoint_sha256": {spec.expert_id: sources[spec.expert_id]} if spec.expert_id in sources else {}}
    return expected


def _prefix_replay_source_artifacts(request, plan, source_artifacts):
    from research.prefix_replay_execution import _request_source_programs

    if not request["source_checkpoint_sha256"]:
        return {}
    specs = _request_source_programs({"original_request": request,
        "source_checkpoints": request["source_checkpoint_sha256"]})
    result = {}
    for name, digest in request["source_checkpoint_sha256"].items():
        key = _target_study(plan), request["source_seed"]
        if specs[name].family == "joint_flat":
            participant = request["target_contract"]["participant_key"]
            key = "joint_flat", plan["joint_source_bindings"][participant][name]
        receipt = source_artifacts[key]
        if digest != receipt["sha256"]:
            raise ValueError("replay source artifact differs from its original expert request")
        result[name] = receipt
    return result


def _ordinary_pool_strategies(plan, mixture_name):
    requested = plan.get("mixture_calibration_strategies", {}).get(mixture_name, plan["calibration_strategies"])
    return tuple(strategy for strategy in requested if strategy in POOL_STRATEGIES)


def _ordinary_pool_package(base, programme, strategy):
    return base + "_pool_" + semantic_sha256({"program": programme.record(), "strategy": strategy})[:20]


def _request_strategies(plan, request):
    return request.get("strategies", [strategy for strategy in plan["calibration_strategies"] if strategy not in POOL_STRATEGIES])


def _write_candidate_json(path, payload, input_admission=None):
    if input_admission is not None:
        return input_admission.write_json(path, payload)
    return atomic_write_json(path, payload)


def _prediction_read_support(spec, view, *, recorder=None):
    support = {"raw_scope": list(view.raw_scope) if view.samples is not None else None,
        "available_sample": view.available_sample, "profile": None if spec.profile is None else spec.profile.record(),
        "filtering_scope": "independent_test_decision" if spec.family == "trial" else "native_unfiltered_decision"}
    if spec.family != "joint_flat":
        return support
    row = recorder.joint_future[-1]
    if row["view"].record_id != view.record_id:
        raise ValueError("joint prediction read support differs from its actual recorded future request")
    capability = row["issued_capabilities"].get(spec.expert_id)
    refusal = row.get("issued_refusals", {}).get(spec.expert_id)
    support.update(raw_scope=None, current_view_scope=list(view.raw_scope), filtering_scope="native_record_public_history")
    if capability is not None:
        binding = capability.binding
        support.update(raw_scope=list(binding["read_sample_support"]),
            conditioning_sample_support=list(binding["conditioning_sample_support"]),
            score_sample_support=list(binding["score_sample_support"]), history_capability_sha256=capability.digest)
    elif refusal is not None:
        support["history_refusal_sha256"] = refusal.digest
    return support


def _participant_ledger(row, plan, directory, source, views, truths, scopes, fit_requests,
                        *, joint_source_resolutions=None):
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
    if joint_source_resolutions is not None:
        ledger["joint_source_resolutions"] = joint_source_resolutions
    if _target_study(plan) != BNCI2014_008_DATASET_ID:
        ledger["requested_evidence_levels"] = {view.decision_id: view.decision.repetitions for view in views}
    if "requested_to_actual_K" in scopes:
        ledger.update(requested_to_actual_K=scopes["requested_to_actual_K"], budget_adjustments=scopes["budget_adjustments"])
        if plan.get("prefix_selection") is not None:
            ledger["selection_fit_requests"] = list(_prefix_selection_requests(plan, scopes).values())
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
        ("requested", "coverage", "candidate_vocabulary", "evidence_level", "participant_key", *(["requested_evidence_levels"] if "requested_evidence_levels" in ledger else []))}}
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions), "status": "completed"})
    # The persisted record is the process wire format; fitted metadata stays
    # deeply immutable inside the worker and never enters pickle transport.
    return json.loads((directory / "result.json").read_bytes())


def _run_prefix_selection_outputs(plan, directory, expected_source, views, paired_sources):
    from research.calibration_selection import fit_nested_prefix_selection
    from research.prefix_selection_replay import replay_prefix_selection

    predictions, fits = [], []
    requests = _prefix_selection_requests(plan, views.ledger)
    for request in requests.values():
        _abort_checkpoint()
        package = directory / request["package"]
        _sealed_record(directory / "selection_fit_receipts" / f"{package.name}.started.json", request)
        try:
            prefix = views.calibration[request["K"]]
            candidates = _prefix_selection_candidates(plan, request["target_seed"])
            sources = {name: value for name, value in paired_sources[request["source_seed"]].items()
                       if name in request["source_checkpoint_sha256"]}
            context = {"source_seed": request["source_seed"], "target_seed": request["target_seed"],
                "requested_Ks": [int(key) for key, actual in views.ledger["requested_to_actual_K"].items() if actual == prefix.budget],
                "requested_to_actual_K": views.ledger["requested_to_actual_K"],
                "test_scope": views.ledger["test_scope"], "requested_decision_ids": [view.record_id for view in views.decisions]}
            selected = fit_nested_prefix_selection(prefix, candidates, baseline_name=request["baseline_name"],
                prefix_compiler=views.prefix_compiler, minimum_selection_rows=request["minimum_selection_rows"],
                source_checkpoints=sources, device=plan["device"], context=context)
            batch = []
            for view in views.decisions:
                evidence = selected.score(view)
                batch.append({"source_seed": request["source_seed"], "target_seed": request["target_seed"],
                    "seed": request["target_seed"], "K": prefix.budget, "calibration_strategy": "nested_selection",
                    "mixture": request["output_name"], "evidence": evidence.record(),
                    "truth": views.scoring_truth[view.decision_id], "actual_authorized_labels": prefix.budget,
                    "requested_Ks": context["requested_Ks"], "saved_replay_exact": True,
                    "selection_package": package.name, "selected_candidate": selected.record["selected_candidate"],
                    "score_source_digest": semantic_sha256(expected_source)})
            selected.save(package)
            files = [_file_receipt(path, directory) for path in sorted(package.rglob("*")) if path.is_file()]
            _sealed_record(directory / "selection_fit_receipts" / f"{package.name}.saved.json",
                {"request": request, "files": files, "graph_sha256": selected.graph_sha256})
            replay = replay_prefix_selection(package, prefix=prefix, prefix_compiler=views.prefix_compiler,
                future_views=views.decisions, source_checkpoints=sources,
                expected_selection_sha256=semantic_sha256(selected.record), expected_graph_sha256=selected.graph_sha256)
            if len(replay["scores"]) != len(views.decisions):
                raise ValueError("nested prefix graph did not replay its entire requested future queue")
            summary = {"package": package.name, "K": prefix.budget, "source_seed": request["source_seed"],
                "target_seed": request["target_seed"], "graph_sha256": selected.graph_sha256,
                "selection_sha256": semantic_sha256(selected.record), "producer_count": replay["producer_count"],
                "exact_score_checks": replay["exact_score_checks"], "selected_candidate": selected.record["selected_candidate"],
                "final_refit_scope": selected.record["final_refit_scope"]}
            _sealed_record(directory / "selection_fit_receipts" / f"{package.name}.replayed.json", summary)
            fits.append(summary)
            # The transaction includes every future capture and the entire actual
            # graph replay; only then can parent-visible prediction seals appear.
            for prediction in batch:
                prediction["prefix_selection_graph_sha256"] = selected.graph_sha256
                _prediction_saved(directory, prediction)
            predictions.extend(batch)
        except BaseException as error:
            _sealed_record(directory / "selection_fit_receipts" / f"{package.name}.failed.json",
                {"request": request, "error_type": type(error).__name__, "detail": str(error)})
            raise
    return predictions, fits


def _run_ordinary_pool_outputs(plan, directory, expected_source, views, requests, sources,
        source_seed, seed, prefix, controls, source_preparation, context):
    """Each declared mixture learns and saves its own exact-pool programme."""
    predictions, fits = [], []
    for request in requests.values():
        if ("pool_probability_strategy" not in request or request["source_seed"] != source_seed
                or request["target_seed"] != seed or request["K"] != prefix.budget):
            continue
        _abort_checkpoint()
        strategy = request["pool_probability_strategy"]
        specs = tuple(CalibrationExpertProgram.from_record(row) for row in request["expert_programs"])
        programme = EnsembleProgram.from_record(request["ensemble_program"])
        selected = plan.get("temperature_expert_ids")
        selected = None if selected is None else tuple(name for name in selected if name in programme.expert_ids)
        bound_sources = {name: sources[name] for name in request["source_checkpoint_sha256"]}
        package = directory / request["package"]
        _fit_started(directory, request)
        recorder = InnerReplayRecorder()
        programs = fit_calibration_prefix_programs(prefix, specs, programme, strategies=(strategy,),
            source_checkpoints=bound_sources, device=plan["device"], temperature_expert_ids=selected,
            inner_replay_recorder=recorder, inner_replay_context=context, _source_preparation=source_preparation)
        value = programs[strategy]
        fitted = None if isinstance(value, ExpectedSubjectError) else value
        calibration = (fitted.temperature_record if fitted is not None else
            {"strategy": strategy, "status": "unavailable", "failure": value.record(subject=prefix.target_contract["participant_key"])})
        pool = fitted.scorer.pool_calibration if fitted is not None else None
        for spec in specs:
            use = prefix.label_use["trial"][spec.profile.digest] if spec.family == "trial" else prefix.label_use["raw"]
            fits.append({"source_seed": source_seed, "target_seed": seed, "K": prefix.budget,
                "requested_Ks": context["requested_Ks"], "calibration_strategy": strategy, "expert_id": spec.expert_id,
                "package": package.name, "label_use": use, "matched_source_controls": controls,
                "temperature_calibration": calibration, "pool_calibration": pool,
                "fit_available": fitted is not None and fitted.expert_fit_available[spec.expert_id],
                "source_only": False, "temperature_calibrated": strategy == TEMPERATURE_POOL_STRATEGY
                    and spec.expert_id in (selected or programme.expert_ids)})
        for view in views.decisions:
            evidence = FittedEnsemble.score_programs(programs, view)[strategy]
            if isinstance(evidence, ExpectedSubjectError):
                evidence = unavailable_prefix_expert_scores(prefix.record(), [spec.record() for spec in specs],
                    strategy, value.record(subject=prefix.target_contract["participant_key"]), view)
            recorder.record_prefix_future_score(strategy, view=view, evidence=evidence)
            score = (fitted.scorer if fitted is not None else EnsembleScorer(programme)).score(evidence)
            predictions.append({"source_seed": source_seed, "target_seed": seed, "seed": seed,
                "K": prefix.budget, "calibration_strategy": strategy, "mixture": programme.name,
                "evidence": score.record(), "truth": views.scoring_truth[view.decision_id],
                "actual_authorized_labels": prefix.budget, "requested_Ks": context["requested_Ks"],
                "saved_replay_exact": fitted is not None, "temperature_calibration": calibration,
                "pool_calibration": pool, "probability_package": package.name,
                "prediction_read_support": {spec.expert_id: {"raw_scope": list(view.raw_scope) if view.samples is not None else None,
                    "available_sample": view.available_sample, "profile": None if spec.profile is None else spec.profile.record(),
                    "filtering_scope": "independent_test_decision" if spec.family == "trial" else "native_unfiltered_decision"} for spec in specs},
                "score_source_digest": semantic_sha256(expected_source)})
        persist_prefix_inner_replay(prefix, programs, package, recorder=recorder,
            source_checkpoints=bound_sources, context=context, device=plan["device"])
        _fit_saved(directory, request)
        replay = replay_prefix_inner_replay(package, prefix=prefix, future_views=views.decisions,
            source_checkpoints=bound_sources, context=context, device=plan["device"],
            expected_record_sha256=recorder.prefix_receipt_sha256)
        if len(replay["scores"]) != len(views.decisions):
            raise ValueError("ordinary probability pool lost its complete requested future replay")
        _fits_saved(directory, package.name, [row for row in fits if row["package"] == package.name])
        for prediction in predictions:
            if prediction["probability_package"] == package.name:
                prediction["prefix_replay_receipt_sha256"] = recorder.prefix_receipt_sha256
                _prediction_saved(directory, prediction)
    return predictions, fits


def _independent_subject_impl(row, plan, directory, expected_source, source_snapshots):
    """Fit C once per expert/seed/K; score the same future queue for every pool."""
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    joint_history = any(spec.family == "joint_flat" for spec in declared)
    mixtures = tuple(EnsembleProgram.from_record(value) for value in plan["mixtures"])
    views = prepare_dataset_independent_prefixes(_candidate_source_declaration(row),
        budgets=plan["budgets"], profiles=[spec.profile for spec in declared if spec.family == "trial"],
        authorization_preprocessing=parse_preprocessing_spec(plan["authorization_preprocessing"]),
        response_seconds=plan["observation_response_seconds"], pre_roll_seconds=plan["pre_roll_seconds"])
    atomic_write_json(directory / "scopes.json", views.ledger)
    paired_specs, paired_sources, paired_controls, joint_resolutions = {}, {}, {}, {}
    joint_preparation = None
    if joint_history:
        from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

        joint_preparation = FrozenJointFlatSourcePreparation()
    target_contract = next(iter(views.calibration.values())).target_contract
    for source_seed, seed in plan["seed_pairs"]:
        specs = tuple(replace(spec, recipe=None if spec.recipe is None else replace(spec.recipe, seed=seed))
                      for spec in declared)
        sources = {spec.expert_id: source_snapshots[(_target_study(plan), source_seed)] for spec in specs
            if spec.family == "trial" and spec.recipe.initialization == "source"}
        if joint_history:
            joint_sources, joint_resolutions[str(source_seed)] = _joint_sources_for_target(plan, specs,
                target_contract, pipeline_pair=(source_seed, seed), _source_preparation=joint_preparation)
            sources.update(joint_sources)
        paired_specs[source_seed] = specs
        paired_sources[source_seed], paired_controls[source_seed] = _matched_source_controls(specs, sources)
    fit_requests = list(_expected_fit_roster(plan, views.ledger).values())
    ledger = _participant_ledger(row, plan, directory, expected_source, views.decisions,
                                views.scoring_truth, views.ledger, fit_requests,
                                joint_source_resolutions=joint_resolutions if joint_history else None)
    fit_requests = {request["package"]: request for request in fit_requests}
    predictions, fits = [], []
    base_strategies = [strategy for strategy in plan["calibration_strategies"] if strategy not in POOL_STRATEGIES]
    with cpu_thread_budget(plan["cpu_threads_per_worker"]):
        for source_seed, seed in plan["seed_pairs"]:
            source_preparation = ScopedSourceCheckpointPreparation()
            specs, sources, controls = paired_specs[source_seed], paired_sources[source_seed], paired_controls[source_seed]
            for budget, prefix in views.calibration.items():
                roster = EnsembleProgram.equal_weights("prefix_fit", tuple(spec.expert_id for spec in specs))
                package = directory / f"source{source_seed}_target{seed}_K{budget}"
                request = fit_requests[package.name]
                _fit_started(directory, request)
                if joint_history:
                    from research.joint_flat_replay import JointFlatReplayRecorder

                    recorder = JointFlatReplayRecorder()
                else:
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
                    strategies=base_strategies, source_checkpoints=sources, device=plan["device"],
                    temperature_expert_ids=plan.get("temperature_expert_ids"),
                    inner_replay_recorder=recorder, inner_replay_context=prefix_context,
                    _source_preparation=source_preparation, prefix_compiler=views.prefix_compiler)
                if not joint_history:
                    FittedEnsemble.save_programs(package, programs, participant_key=prefix.target_contract["participant_key"])
                for strategy in base_strategies:
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
                        use = prefix.label_use["trial"][spec.profile.digest] if spec.family == "trial" else prefix.label_use["raw"]
                        fits.append({"source_seed": source_seed, "target_seed": seed, "K": budget,
                            "requested_Ks": [int(key) for key, actual in views.ledger["requested_to_actual_K"].items() if actual == budget],
                            "calibration_strategy": strategy, "expert_id": spec.expert_id,
                            "package": package.name, "label_use": use,
                            "matched_source_controls": controls, "temperature_calibration": calibration,
                            "fit_available": fitted is not None and fitted.expert_fit_available[spec.expert_id],
                            "source_only": strategy == "fixed" and budget == 0 and spec.source_only_capable,
                            "temperature_calibrated": strategy == "grouped_temperature" and spec.expert_id in
                                plan.get("temperature_expert_ids", roster.expert_ids)})
                predictions_by_strategy = {strategy: [] for strategy in base_strategies}
                for view in views.decisions:
                    _abort_checkpoint()
                    shared = (recorder.score_future(programs, view, compiler=views.prefix_compiler, prefix=prefix)
                              if joint_history else FittedEnsemble.score_programs(programs, view))
                    for strategy in base_strategies:
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
                        if not joint_history:
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
                                "prediction_read_support": {spec.expert_id: _prediction_read_support(spec, view, recorder=recorder)
                                    for spec in specs if spec.expert_id in mixture.expert_ids},
                                "score_source_digest": semantic_sha256(expected_source)})
                # Preserve the established strategy-major terminal and seal
                # order while sharing raw expert scores only in this view.
                for strategy in base_strategies:
                    predictions.extend(predictions_by_strategy[strategy])
                persist_prefix_inner_replay(prefix, programs, package, recorder=recorder,
                    source_checkpoints=sources, context=prefix_context, device=plan["device"])
                _fit_saved(directory, request)
                admitted = replay_prefix_inner_replay(package, prefix=prefix, future_views=views.decisions,
                    source_checkpoints=sources, context=prefix_context, device=plan["device"],
                    expected_record_sha256=recorder.prefix_receipt_sha256, prefix_compiler=views.prefix_compiler)
                if (len(admitted["scores"]) != len(views.decisions)
                        or admitted["exact_score_checks"] < len(recorder.prefix_future_scores)):
                    raise ValueError("prefix replay did not verify its complete requested future score roster")
                _fits_saved(directory, package.name, [value for value in fits if value["package"] == package.name])
                for prediction in predictions[prediction_start:]:
                    prediction["prefix_replay_receipt_sha256"] = recorder.prefix_receipt_sha256
                    _prediction_saved(directory, prediction)
                atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions),
                    "source_seed": source_seed, "target_seed": seed, "K": budget, "status": "running"})
                extra_predictions, extra_fits = _run_ordinary_pool_outputs(plan, directory, expected_source,
                    views, fit_requests, sources, source_seed, seed, prefix, controls, source_preparation, prefix_context)
                predictions.extend(extra_predictions)
                fits.extend(extra_fits)
    selection_fits = []
    if plan.get("prefix_selection") is not None:
        with cpu_thread_budget(plan["cpu_threads_per_worker"]):
            extra, selection_fits = _run_prefix_selection_outputs(plan, directory, expected_source, views, paired_sources)
        predictions.extend(extra)
    result = {"status": "completed", "predictions": predictions, "fits": fits, "selection_fits": selection_fits, **{key: ledger[key] for key in
        ("requested", "requested_to_actual_K", "budget_adjustments", "coverage", "candidate_vocabulary", "evidence_level", "participant_key", *(["requested_evidence_levels"] if "requested_evidence_levels" in ledger else []))}}
    if joint_history:
        result["joint_source_resolutions"] = ledger["joint_source_resolutions"]
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "progress.json", {"completed_prediction_cells": len(predictions), "status": "completed"})
    return json.loads((directory / "result.json").read_bytes())


def _subject(row, plan, directory, expected_source, source_snapshots, *, metadata_directory=None):
    directory = Path(directory)
    from research.source_metadata_store import SourceMetadataStore, source_metadata_scope

    try:
        with source_metadata_scope(SourceMetadataStore(metadata_directory or directory.parent / "source_metadata")):
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
        run_study(plan_path, output, _actor_index=arguments["record_index"], _native_threads=native,
                  _metadata_directory=arguments.get("metadata_directory"))
    return str(output / f"participant_{arguments['record_index']:03d}" / "result.json")


def candidate_tasks(plan_path, output, resources, *, dependencies=(), task_prefix="candidate", metadata_directory=None):
    from research.cloud_execution import CloudTask, target_phase_resources

    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_bytes())
    return [CloudTask(f"{task_prefix}:{index:03d}",
        "experiments.run_candidate_ensemble_study:candidate_participant_adapter",
        {"plan": str(plan_path), "output": str(output / "participant_groups" / f"participant_{index:03d}"),
         "record_index": index, "metadata_directory": str(metadata_directory or output / "source_metadata"),
         "output_estimate_bytes": min(2**30, resources["target_worker_memory_bytes"])},
        str(output / "participant_groups" / f"participant_{index:03d}"),
        target_phase_resources(resources, plan.get("device", "cpu")), tuple(dependencies))
        for index in range(len(plan["records"]))]


def _participant_metadata_receipts(task):
    group = Path(task.output).resolve()
    directory = group / f"participant_{task.arguments['record_index']:03d}"
    return [_file_receipt(path, group) for path in
        (group / "plan.json", group / "executed_source.json", directory / "request_ledger.json",
         directory / "scopes.json", directory / "result.json") if path.is_file()]


def _admission_task_descriptor(record):
    from research.cloud_execution import CloudTask, PhaseResources

    return CloudTask(record["task_id"], record["adapter"], record["arguments"], record["output"],
                     PhaseResources(**record["resources"]), tuple(record["depends_on"]))


def candidate_participant_admission_adapter(arguments):
    """An independent zero-fit numerical ingress, under the shared CPU owner."""
    import os

    import torch

    from research.cloud_execution import CloudResult
    from research.prefix_recovery import preserve_prefix_producer_output
    from research.source_execution import current_input_admission

    output = Path(arguments["output"]).resolve()
    request = arguments["request"]
    context_path = Path(arguments["context"]["path"])
    if _file_receipt(context_path, context_path.parent)["sha256"] != arguments["context"]["sha256"]:
        raise ValueError("candidate admission producer context bytes changed")
    context = _read_sealed_record(context_path)
    normalized, consumer_source, snapshots = _prepare_candidate_plan(arguments["plan"],
        native_threads=int(os.environ["OMP_NUM_THREADS"]), load_sources="reducer_outcomes" not in arguments)
    preserve_prefix_producer_output(normalized, output)
    if (semantic_sha256(consumer_source) != request["consumer_source_digest"]
            or semantic_sha256(_plan_authority(normalized)) != request["consumer_plan_digest"]
            or semantic_sha256(context["source"]) != request["producer_source_digest"]
            or semantic_sha256(_plan_authority(context["plan"])) != request["producer_plan_digest"]):
        raise ValueError("candidate admission consumer or original producer identity changed")
    original = _admission_task_descriptor(request["producer_task"])
    outcome = CloudResult(**request["fit_outcome"])
    if (int(os.environ["OMP_NUM_THREADS"]) != original.resources.native_threads
            or torch.get_num_threads() != original.resources.native_threads
            or torch.get_num_interop_threads() != 1):
        raise ValueError("candidate numerical admission must retain its original producer native programme")
    if _participant_metadata_receipts(original) != request["participant_metadata"]:
        raise ValueError("candidate admission original participant metadata changed before replay")
    replay = None
    if "reducer_outcomes" in arguments:
        from research.prefix_replay_execution import CompletedParticipantReplay

        authority = _candidate_participant_authority(original, outcome, context["plan"], context["source"])
        if authority is None:
            raise ValueError("candidate completed replay lacks its original participant/source anchor")
        expected = _prefix_replay_participant_anchor(original, context["plan"], context["source"], consumer_source, authority)
        actual = {key: arguments["replay_request"][key] for key in expected}
        for key in ("selection_packages", "ordinary_packages"):
            actual[key] = [{name: value for name, value in row.items() if name != "source_checkpoints"}
                           for row in actual[key]]
        if semantic_sha256(actual) != semantic_sha256(expected):
            raise ValueError("candidate completed replay changed its original participant/source anchor")
        replay = CompletedParticipantReplay.from_reduced(arguments["replay_request"], arguments["reducer_outcomes"])
    admitted = _admit_candidate_participant(original, outcome, context["plan"], context["source"],
        source_snapshots=snapshots, replay_admission=replay)
    if _participant_metadata_receipts(original) != request["participant_metadata"]:
        raise ValueError("candidate admission original participant metadata changed during replay")
    if admitted is not None:
        admitted["execution_binding"] = {"role": request["producer_role"],
            "producer_source_digest": request["producer_source_digest"],
            "producer_plan_digest": request["producer_plan_digest"],
            "producer_native_threads": original.resources.native_threads,
            "consumer_native_threads": int(os.environ["OMP_NUM_THREADS"]),
            "consumer_source_digest": request["consumer_source_digest"],
            "original_group": original.output}
    owner = current_input_admission()
    writer = owner.artifact_directory(output, arguments["output_estimate_bytes"]) if owner is not None else nullcontext()
    with writer:
        output.mkdir(parents=True, exist_ok=False)
        _sealed_record(output / "admission.json", {"schema": "n2p3_candidate_participant_admission/1",
            "request": request, "consumer_source": consumer_source, "participant": admitted})
    return _file_receipt(output / "admission.json", output)


def _admit_prefix_selection_packages(directory, group, plan, scopes, ledger, *, source_snapshots=None,
                                     replay_admission=None, _metadata_admission=None, _joint_source_preparation=None,
                                     _joint_sources=None, _prepared_views=None):
    from research.prefix_selection_replay import _SelectionReplayAdmission

    expected = _prefix_selection_requests(plan, scopes)
    request_rows = ledger.get("selection_fit_requests", [])
    actual = {request["package"]: request for request in request_rows}
    if len(actual) != len(request_rows) or semantic_sha256(actual) != semantic_sha256(expected):
        raise ValueError("nested prefix prefit ledger changed its declared complete programme requests")
    started = set()
    for path in sorted((directory / "selection_fit_receipts").glob("*.started.json")):
        request = _read_sealed_record(path)
        if semantic_sha256(request) != semantic_sha256(expected.get(request.get("package"))):
            raise ValueError("nested prefix started receipt differs from original programme request")
        started.add(request["package"])
    failed = {}
    for path in sorted((directory / "selection_fit_receipts").glob("*.failed.json")):
        receipt = _read_sealed_record(path)
        request = receipt["request"]
        if (semantic_sha256(request) != semantic_sha256(expected.get(request.get("package")))
                or request["package"] not in started):
            raise ValueError("nested prefix failed receipt lacks its original programme attempt")
        failed[request["package"]] = receipt
    admitted, summaries = {}, []
    paths = sorted((directory / "selection_fit_receipts").glob("*.saved.json"))
    if not paths:
        return admitted, summaries, started, failed
    # The normalized catalog is a producer record, not a second raw path
    # declaration. Reuse the exact bytes admitted by the canonical preparer.
    declared_sources = {(entry["outer_target_study"], entry["source_seed"]): entry["checkpoint_sha256"]
        for entry in plan["source_seed_catalog"] if entry["outer_target_study"] == _target_study(plan)
        and entry["source_seed"] in {pair[0] for pair in plan["seed_pairs"]}}
    prepared = {} if source_snapshots is None else source_snapshots
    if replay_admission is None and (set(prepared) != set(declared_sources) or any(not isinstance(prepared[key], FrozenSourceCheckpoint)
            or prepared[key].digest != digest for key, digest in declared_sources.items())):
        raise ValueError("nested prefix parent requires its declared canonical source snapshots")
    snapshots = {key[1]: snapshot for key, snapshot in prepared.items()}
    row = ledger["record"]
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    if replay_admission is None:
        views = _prepared_views
        if views is None:
            with cpu_thread_budget(1):
                views = prepare_dataset_independent_prefixes(_candidate_source_declaration(row), budgets=plan["budgets"],
                    profiles=[spec.profile for spec in declared if spec.family == "trial"],
                    authorization_preprocessing=parse_preprocessing_spec(plan["authorization_preprocessing"]),
                    response_seconds=plan["observation_response_seconds"], pre_roll_seconds=plan["pre_roll_seconds"])
        if semantic_sha256(views.ledger) != semantic_sha256(scopes):
            raise ValueError("nested prefix parent original source compilation differs from actor scopes")
    for path in paths:
        seal = _read_sealed_record(path)
        request = seal["request"]
        name = request["package"]
        if name not in expected or semantic_sha256(request) != semantic_sha256(expected[name]) or name in admitted:
            raise ValueError("nested prefix package differs from its prefit original programme")
        package = directory / name
        if package.is_symlink() or not package.resolve().is_relative_to(directory.resolve()):
            raise ValueError("nested prefix package escapes its participant")
        delivered = None if replay_admission is None else replay_admission.selection(name)
        if delivered is None:
            for receipt in seal["files"]:
                if _file_receipt(directory / receipt["path"], directory) != receipt:
                    raise ValueError("nested prefix model/graph byte seal changed")
            graph_owner = _SelectionReplayAdmission(package, expected_graph_sha256=seal["graph_sha256"],
                                                     _metadata_admission=_metadata_admission,
                                                     _joint_source_preparation=_joint_source_preparation)
            selection = graph_owner.graph["selection"]
            selection_digest = semantic_sha256(selection)
        else:
            if (delivered["graph_sha256"] != seal["graph_sha256"]
                    or semantic_sha256(delivered["byte_seal_files"]) != semantic_sha256(seal["files"])):
                raise ValueError("nested prefix private admission changed original saved byte seals")
            selection, selection_digest = delivered["selection"], delivered["selection_sha256"]
        if (semantic_sha256(selection["candidates"]) != semantic_sha256(request["candidates"])
                or selection["baseline"] != request["baseline_name"]
                or selection["minimum_selection_rows"] != request["minimum_selection_rows"]
                or selection["source_checkpoint_sha256"] != request["source_checkpoint_sha256"]
                or selection["parent_scope"]["input"]["prefix_record_sha256"] != request["training_digest"]):
            raise ValueError("nested prefix selected graph changes candidate/source/full-C authority")
        replay_seal = directory / "selection_fit_receipts" / f"{name}.replayed.json"
        if not replay_seal.exists():
            continue
        summary = _read_sealed_record(replay_seal)
        if (summary["graph_sha256"] != seal["graph_sha256"]
                or summary["selection_sha256"] != selection_digest
                or semantic_sha256(summary["selected_candidate"]) != semantic_sha256(selection["selected_candidate"])):
            raise ValueError("nested prefix saved replay summary changed its actual selected programme")
        if delivered is None:
            sources = {}
            for expert in request["source_checkpoint_sha256"]:
                spec = next(spec for spec in declared if spec.expert_id == expert)
                sources[expert] = (_joint_sources[request["source_seed"]][expert] if spec.family == "joint_flat"
                                   else snapshots[request["source_seed"]])
            with cpu_thread_budget(1):
                replay = graph_owner.replay(prefix=views.calibration[request["K"]],
                    prefix_compiler=views.prefix_compiler, future_views=views.decisions, source_checkpoints=sources)
        else:
            if semantic_sha256(summary) != semantic_sha256(delivered["original_summary"]):
                raise ValueError("nested prefix private admission changed original replay summary")
            replay = delivered["summary"]
        if (summary["producer_count"] != replay["producer_count"]
                or summary["exact_score_checks"] != replay["exact_score_checks"]):
            raise ValueError("nested prefix actual parent replay did not retain captured outputs")
        admitted[name] = {"graph_sha256": seal["graph_sha256"], "selection": selection,
            "future": {row["decision_id"]: row["strategies"][selection["selected_candidate"]["strategy"]]
                       for row in replay["scores"]}}
        summaries.append(summary)
    return admitted, summaries, started, failed


def _candidate_participant_authority(task, outcome, plan, source):
    """The small physical/execution ingress shared by fit and numerical admission."""
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
    if _target_study(plan) != BNCI2014_008_DATASET_ID:
        declaration = _candidate_source_declaration(plan["records"][index])
        if declaration["dataset_id"] != _target_study(plan) or scopes["dataset_id"] != declaration["dataset_id"]:
            raise ValueError("candidate participant differs from its admitted original dataset declaration")
    elif scopes["source_sha256"] != plan["records"][index]["sha256"]:
        raise ValueError("candidate participant scopes differ from its actual raw record")
    return group, index, directory, expected_path, ledger, scopes


def _prefix_replay_participant_anchor(task, plan, source, consumer_source, authority, *, source_root=None):
    """Derive replay identity from the original physical participant ingress."""
    from research.prefix_replay_execution import SCHEMA as REPLAY_SCHEMA
    from research.prefix_replay_execution import artifact_receipt

    _, _, directory, _, ledger, scopes = authority
    preparation_id = task.task_id + ":replay_prepare"
    declared = tuple(CalibrationExpertProgram.from_record(value) for value in plan["experts"])
    profiles = {spec.profile.digest: spec.profile for spec in declared if spec.family == "trial"}
    mat_path = Path(ledger["record"]["path"]).resolve()
    selections, ordinary = [], []
    for path in sorted((directory / "selection_fit_receipts").glob("*.saved.json")):
        seal = _read_sealed_record(path)
        replayed = path.with_name(path.name.replace(".saved.json", ".replayed.json"))
        selections.append({"package": seal["request"]["package"], "graph_sha256": seal["graph_sha256"],
            "original_request": seal["request"], "files": seal["files"], "saved_seal": artifact_receipt(path),
            "replayed_summary": artifact_receipt(replayed) if replayed.exists() else None})
    for path in sorted((directory / "fit_receipts").glob("*.saved.json")):
        seal = _read_sealed_record(path)
        request = seal["request"]
        prefix = _read_sealed_record(directory / request["package"] / "prefix_inner_replay.json")
        aliases = [int(key) for key, actual in scopes["requested_to_actual_K"].items() if actual == request["K"]]
        context = {"source_seed": request["source_seed"], "target_seed": request["target_seed"],
            "requested_Ks": aliases, "requested_to_actual_K": scopes["requested_to_actual_K"],
            "test_scope": scopes["test_scope"], "requested_decision_ids": [view["record_id"] for view in scopes["test_views"]],
            "phase": "independent_prefix"}
        ordinary.append({"package": request["package"], "prefix_record_sha256": semantic_sha256(prefix),
            "original_request": request, "context": context, "files": seal["files"], "saved_seal": artifact_receipt(path)})
    replay = {"schema": REPLAY_SCHEMA, "preparation_task_id": preparation_id, "participant_directory": str(directory),
        "producer_ledger": artifact_receipt(directory / "request_ledger.json"),
        "ordinary_native_threads": ledger["native_threads"],
        "mat": {"path": str(mat_path), "sha256": ledger["record"]["sha256"], "bytes": mat_path.stat().st_size},
        "scopes": {"path": str((directory / "scopes.json").resolve()), "sha256": ledger["scopes_sha256"],
                   "bytes": (directory / "scopes.json").stat().st_size},
        "producer_source": source, "consumer_source": consumer_source,
        "input_request": {"schema": "n2p3_prefix_replay_input_request/1", "preparation_task_id": preparation_id,
            "original_source_sha256": ledger["record"]["sha256"], "producer_source_sha256": source["src/transfer/ensemble_views.py"],
            "participant_key": ledger["participant_key"], "source_relative_path": Path(ledger["record"]["path"]).name,
            "requested_Ks": plan["budgets"], "actual_Ks": sorted(set(scopes["requested_to_actual_K"].values())),
            "profiles": [profile.record() for profile in profiles.values()],
            "authorization_preprocessing": plan["authorization_preprocessing"],
            "observation_response_seconds": plan["observation_response_seconds"], "pre_roll_seconds": plan["pre_roll_seconds"]},
        "selection_packages": selections, "ordinary_packages": ordinary}
    if _target_study(plan) != BNCI2014_008_DATASET_ID:
        replay["input_request"].update(schema="n2p3_prefix_replay_input_request/2",
            dataset_source=_candidate_source_declaration(ledger["record"]),
            original_source_sha256=scopes["source_sha256"],
            source_relative_path="ordered_original_candidate_acquisitions")
        names = ("data.candidate_task", "data.candidate_input", *(
            ("data.huebner2017_candidate",) if _target_study(plan) == "Huebner2017" else
            ("data.bi2014a_candidate", "data.bi2014a_recording")))
        replay["producer_input_modules"] = {name: artifact_receipt(Path(source_root or ROOT)/"src"/(name.replace(".", "/")+".py"))
                                             for name in names}
    return replay


def _validate_nested_prefix_prediction(prediction, evidence, selection_packages, output_name):
    """Match a saved prediction to its independently replayed selection output."""
    package = prediction.get("selection_package")
    selected = selection_packages.get(package)
    expected_package = f"source{prediction['source_seed']}_target{prediction['seed']}_K{prediction['K']}_selection"
    if (package != expected_package or selected is None
            or prediction.get("prefix_selection_graph_sha256") != selected["graph_sha256"]
            or semantic_sha256(prediction.get("selected_candidate")) != semantic_sha256(selected["selection"]["selected_candidate"])
            or semantic_sha256(evidence.record()) != semantic_sha256(selected["future"].get(evidence.decision_id))
            or prediction["mixture"] != output_name
            or prediction.get("saved_replay_exact") is not True):
        raise ValueError("nested prefix prediction differs from actual loaded graph/expert probability programme")


def _admit_candidate_participant(task, outcome, plan, source, *, source_snapshots=None, replay_admission=None):
    """One numerical ingress for both complete and native-interrupted actors."""
    authority = _candidate_participant_authority(task, outcome, plan, source)
    if authority is None:
        return None
    group, index, directory, expected_path, ledger, scopes = authority
    from research.source_metadata_store import (
        _SourceMetadataAdmission,
        admit_source_dependencies,
        collect_source_dependencies,
    )

    metadata_admission = _SourceMetadataAdmission()
    joint_source_preparation = None
    if any(spec["family"] == "joint_flat" for spec in plan["experts"]):
        from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

        joint_source_preparation = FrozenJointFlatSourcePreparation()
    joint_sources, joint_resolutions, joint_views = {}, {}, None
    if joint_source_preparation is not None:
        declared = tuple(CalibrationExpertProgram.from_record(row) for row in plan["experts"])
        target_contract = next(iter(scopes["calibration"].values()))["target_contract"]
        if replay_admission is None:
            with cpu_thread_budget(1):
                joint_views = prepare_dataset_independent_prefixes(_candidate_source_declaration(ledger["record"]),
                    budgets=plan["budgets"], profiles=[spec.profile for spec in declared if spec.family == "trial"],
                    authorization_preprocessing=parse_preprocessing_spec(plan["authorization_preprocessing"]),
                    response_seconds=plan["observation_response_seconds"], pre_roll_seconds=plan["pre_roll_seconds"])
            if semantic_sha256(joint_views.ledger) != semantic_sha256(scopes):
                raise ValueError("joint parent original input compilation differs from actor scopes")
            target_contract = next(iter(joint_views.calibration.values())).target_contract
        for source_seed, seed in plan["seed_pairs"]:
            resolved, joint_resolutions[str(source_seed)] = _joint_sources_for_target(plan, declared,
                target_contract, pipeline_pair=(source_seed, seed), _source_preparation=joint_source_preparation)
            joint_sources[source_seed] = resolved
        if semantic_sha256(ledger.get("joint_source_resolutions")) != semantic_sha256(joint_resolutions):
            raise ValueError("candidate source resolution differs from its actual original source and target admission")
    dependencies = collect_source_dependencies(directory,
        {path.parent for name in ("ensemble.json", "prefix_inner_replay.json") for path in directory.rglob(name)},
        include_joint_prefixes=True)
    if dependencies is not None:
        admit_source_dependencies(dependencies, directory, admission=metadata_admission)
    if replay_admission is not None:
        from research.prefix_replay_execution import CompletedParticipantReplay

        if not isinstance(replay_admission, CompletedParticipantReplay):
            raise TypeError("candidate admission requires the actual private completed replay owner")
    selection_packages, selection_fits, selection_started, selection_failed = _admit_prefix_selection_packages(
        directory, group, plan, scopes, ledger, source_snapshots=source_snapshots, replay_admission=replay_admission,
        _metadata_admission=metadata_admission, _joint_source_preparation=joint_source_preparation,
        _joint_sources=joint_sources, _prepared_views=joint_views)
    if semantic_sha256(ledger["pipeline_requests"]) != semantic_sha256(
            _resolved_pipeline_requests(plan, index, ledger["requested"], scopes)):
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
    prefix_receipts, package_records = {}, {}
    for path in sorted((directory / "fit_receipts").glob("*.saved.json")):
        seal = _read_sealed_record(path)
        request = seal["request"]
        package_name = request["package"]
        if (package_name not in requests or semantic_sha256(request) != semantic_sha256(requests[package_name])
                or package_name in seals):
            raise ValueError("saved candidate package differs from its prefit authority")
        package = directory / package_name
        if not package.resolve().is_relative_to(directory.resolve()) or package.is_symlink():
            raise ValueError("saved candidate package escapes its participant")
        file_receipts = {receipt["path"]: receipt for receipt in seal["files"]}
        if len(file_receipts) != len(seal["files"]):
            raise ValueError("candidate package seal repeats an artifact")
        delivered = None if replay_admission is None else replay_admission.ordinary(package_name)
        if delivered is None:
            for receipt in file_receipts.values():
                if _file_receipt(directory / receipt["path"], directory) != receipt:
                    raise ValueError("saved candidate model or probability manifest bytes changed")
        elif semantic_sha256(delivered["byte_seal_files"]) != semantic_sha256(seal["files"]):
            raise ValueError("candidate private prefix admission changed original saved byte seals")
        manifest = package / "ensemble.json"
        if manifest.exists():
            payload = json.loads(manifest.read_bytes())
            record = payload["record"]
            package_records[package_name] = record
            if (semantic_sha256(record) != payload["record_sha256"]
                    or semantic_sha256(record["program"]) != semantic_sha256(request["ensemble_program"])
                    or semantic_sha256([entry["program"] for entry in record["experts"]]) != semantic_sha256(request["expert_programs"])
                    or "target_contract" in request and semantic_sha256(record["target_contract"]) != semantic_sha256(request["target_contract"])):
                raise ValueError("candidate package program or fit scope differs from its prefit authority")
            for entry in record["experts"]:
                spec = entry["program"]
                checkpoint = request["source_checkpoint_sha256"].get(spec["expert_id"])
                program = CalibrationExpertProgram.from_record(spec)
                # The family owner distinguishes fitted source parameters,
                # pretrained weights and a matched scratch architecture.
                expected_origin = checkpoint if program.source_only_capable and entry["model"] is not None else None
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
                            "checkpoint_role": None if checkpoint is None else program.source_checkpoint_role}
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
                    or set(record["programs"]) != set(_request_strategies(plan, request))):
                raise ValueError("candidate temperature programmes differ from their saved numerical package")
            for strategy, programme in record["programs"].items():
                if strategy in {"grouped_temperature", TEMPERATURE_POOL_STRATEGY} and programme["status"] == "completed":
                    roster = tuple(request["ensemble_program"]["expert_ids"])
                    subset = plan.get("temperature_expert_ids")
                    declared = roster if subset is None else tuple(name for name in subset if name in roster)
                    actual = tuple(programme["temperature_calibration"].get("temperature_expert_ids", roster))
                    if actual != declared:
                        raise ValueError("candidate temperature subset differs from its actual requested programme")
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
            if delivered is None:
                original_inputs = {}
                if joint_views is not None:
                    original_sources = {}
                    for spec_record in request["expert_programs"]:
                        name = spec_record["expert_id"]
                        if name not in request["source_checkpoint_sha256"]:
                            continue
                        original_sources[name] = (joint_sources[request["source_seed"]][name]
                            if spec_record["family"] == "joint_flat" else
                            source_snapshots[(_target_study(plan), request["source_seed"])])
                    original_inputs = {"prefix": joint_views.calibration[request["K"]],
                        "future_views": joint_views.decisions, "prefix_compiler": joint_views.prefix_compiler,
                        "source_checkpoints": original_sources}
                replay_prefix_inner_replay(package, context=expected_context,
                    expected_record_sha256=prefix_digest, device=plan["device"],
                    _metadata_admission=metadata_admission, _joint_source_preparation=joint_source_preparation,
                    **original_inputs)
            elif (delivered["record_sha256"] != prefix_digest
                    or semantic_sha256(delivered["future_rows"]) != semantic_sha256(prefix_record["future_score_rows"])):
                raise ValueError("candidate private prefix admission changed captured future probabilities")
            prefix_receipts[package_name] = {
                "record_sha256": prefix_digest,
                "pool_calibration": {strategy: value.get("pool_calibration")
                                     for strategy, value in prefix_record["programs"].items()},
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
            actual_experts = {entry["program"]["expert_id"]: entry for entry in
                package_records.get(request["package"], {}).get("experts", ())}
            fit_keys = set()
            for value in record["fits"]:
                spec = next(spec for spec in request["expert_programs"] if spec["expert_id"] == value["expert_id"])
                key = (value["calibration_strategy"], value["expert_id"])
                program = temperature[value["calibration_strategy"]]
                calibration = (program["temperature_calibration"] if program["status"] == "completed" else
                    {"strategy": value["calibration_strategy"], "status": "unavailable", "failure": program["failure"]})
                label_use = (prefix["label_use"]["trial"][CalibrationExpertProgram.from_record(spec).profile.digest]
                             if spec["family"] == "trial" else prefix["label_use"]["raw"])
                if (key in fit_keys or value["package"] != request["package"] or value["K"] != request["K"]
                        or value["source_seed"] != request["source_seed"] or value["target_seed"] != request["target_seed"]
                        or value["label_use"] != label_use or value["requested_Ks"] != [int(key) for key, actual in
                            scopes["requested_to_actual_K"].items() if actual == request["K"]]
                        or value["fit_available"] != (program["status"] == "completed" and actual_experts[value["expert_id"]]["model"] is not None)
                        or value["temperature_calibration"] != calibration
                        or value.get("pool_calibration") != program.get("pool_calibration")):
                    raise ValueError("candidate fit replay ledger differs from its saved programme or actual C authority")
                fit_keys.add(key)
        fits.extend(record["fits"])
    for path in sorted((directory / "fit_receipts").glob("*.started.json")):
        request = _read_sealed_record(path)
        if semantic_sha256(request) != semantic_sha256(requests.get(request["package"])):
            raise ValueError("candidate inflight fit differs from its request")
        started.add(request["package"])
        if request["package"] not in seals and not path.with_name(path.name.replace(".started.json", ".failed.json")).exists():
            in_flight.append({"request": request, "status": "interrupted_unknown", "physical_successful_fits": None})
    for path in sorted((directory / "fit_receipts").glob("*.failed.json")):
        record = _read_sealed_record(path)
        if semantic_sha256(record["request"]) != semantic_sha256(requests.get(record["request"]["package"])):
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
                or evidence.evidence_level != ledger.get("requested_evidence_levels", {}).get(evidence.decision_id, ledger["evidence_level"])
                or dict(zip(evidence.candidate_vocabulary, evidence.expected_event_counts, strict=True)) != coverage[key]):
            raise ValueError("candidate prediction differs from its prefit request or score source")
        if strategy == "nested_selection":
            _validate_nested_prefix_prediction(prediction, evidence, selection_packages,
                                               plan["prefix_selection"]["output_name"])
            identities.add(identity)
            prediction_by_identity[identity] = prediction
            continue
        if "K" in prediction:
            package = f"source{prediction['source_seed']}_target{prediction['seed']}_K{prediction['K']}"
            mixture = next(value for value in plan["mixtures"] if value["name"] == prediction["mixture"])
            programme = EnsembleProgram.from_record(mixture)
            if strategy in POOL_STRATEGIES:
                package = _ordinary_pool_package(package, programme, strategy)
            prefix_receipt = prefix_receipts.get(package)
            if prefix_receipt is None or prediction.get("prefix_replay_receipt_sha256") != prefix_receipt["record_sha256"]:
                raise ValueError("candidate prediction lacks its admitted prefix replay receipt")
            captured = prefix_receipt["future_rows"].get((strategy, evidence.decision_id))
            if captured is None:
                raise ValueError("candidate prediction lacks its captured requested future row")
            pool = prefix_receipt["pool_calibration"].get(strategy)
            if strategy in POOL_STRATEGIES and (prediction.get("probability_package") != package
                    or semantic_sha256(prediction.get("pool_calibration")) != semantic_sha256(pool)):
                raise ValueError("candidate probability pool differs from its own admitted mixture programme")
            original = EnsembleScorer(programme, pool_calibration=pool).score({
                name: CandidateEvidence.from_record(captured["expert_evidence"][name]) for name in mixture["expert_ids"]})
            if semantic_sha256(original.record()) != semantic_sha256(evidence.record()):
                raise ValueError("candidate prediction probability/failure differs from its captured experts and mixture")
        identities.add(identity)
        prediction_by_identity[identity] = prediction
        if "K" in prediction:
            packages = [package]
        else:
            expert_indexes = {value["expert_id"]: index for index, value in enumerate(plan["experts"])}
            packages = [f"seed{prediction['seed']}_fold{prediction['fold']}_expert{expert_indexes[value.expert_id]}"
                        for value in evidence.expert_identities]
        models = {}
        for package_name in packages:
            if package_name in package_records:
                record = package_records[package_name]
                models.update((entry["program"]["expert_id"], entry["model_digest"]) for entry in record["experts"])
        if evidence.probability_available and (not prediction["saved_replay_exact"] or any(
                models.get(value.expert_id) != value.model_digest for value in evidence.expert_identities)):
            raise ValueError("candidate probability output is not bound to its replayed saved models")
    metadata = {key: ledger[key] for key in ("requested", "coverage", "candidate_vocabulary", "evidence_level", "participant_key", *(["requested_evidence_levels"] if "requested_evidence_levels" in ledger else []))}
    if joint_source_preparation is not None:
        metadata["joint_source_resolutions"] = joint_resolutions
    if "requested_to_actual_K" in ledger:
        metadata.update(requested_to_actual_K=ledger["requested_to_actual_K"], budget_adjustments=ledger["budget_adjustments"])
    pipeline = []
    for request in ledger["pipeline_requests"]:
        identity = (request["source_seed"], request["target_seed"], request["actual_K"], request["calibration_strategy"],
                    request["mixture"], request["requested"]["decision_id"])
        prediction = prediction_by_identity.get(identity)
        if request["calibration_strategy"] == "nested_selection":
            request_packages = {f"source{request['source_seed']}_target{request['target_seed']}_K{request['actual_K']}_selection"}
            request_started = selection_started
        elif "requested_to_actual_K" in ledger:
            package = f"source{request['source_seed']}_target{request['target_seed']}_K{request['actual_K']}"
            if request["calibration_strategy"] in POOL_STRATEGIES:
                programme = EnsembleProgram.from_record(next(value for value in plan["mixtures"] if value["name"] == request["mixture"]))
                package = _ordinary_pool_package(package, programme, request["calibration_strategy"])
            request_packages = {package}
            request_started = started
        else:
            fold = next(fold for fold, group in enumerate(scopes["groups"]) if group["decision_id"] == request["requested"]["decision_id"])
            mixture = next(value for value in plan["mixtures"] if value["name"] == request["mixture"])
            request_packages = {f"seed{request['target_seed']}_fold{fold}_expert{expert_index}"
                for expert_index, spec in enumerate(plan["experts"]) if spec["expert_id"] in mixture["expert_ids"]}
            request_started = started
        pipeline.append({**request, "status": ("completed" if prediction["evidence"]["log_probability"] is not None
            else "output_unavailable") if prediction is not None else "failed" if request_packages & set(selection_failed)
            else "interrupted_unknown" if request_packages & request_started else "not_started"})
    unsealed = [_file_receipt(path, directory) for package_name in requests if package_name not in seals
                for path in sorted((directory / package_name).glob("*")) if path.is_file()]
    result = {**metadata, "status": "partial", "predictions": predictions, "fits": fits,
        "selection_fits": selection_fits,
        "selection_inflight_programmes": sorted(selection_started - set(selection_packages) - set(selection_failed)),
        "selection_failed_programmes": list(selection_failed.values()),
        "artifact_receipts": list(seals.values()), "failed_fit_receipts": failed_fits, "inflight_fits": in_flight,
        "surviving_unsealed_artifacts": unsealed, "pipeline_requests": pipeline, "record_index": index}
    if outcome.status == "completed":
        persisted = json.loads(expected_path.read_bytes())
        fit_complete = replayed_packages == set(requests)
        if "requested_to_actual_K" in ledger:
            actual_fit_keys = {(value["package"], value["calibration_strategy"], value["expert_id"]) for value in fits}
            expected_fit_keys = {(request["package"], strategy, expert["expert_id"]) for request in requests.values()
                for strategy in _request_strategies(plan, request) for expert in request["expert_programs"]}
            fit_complete = fit_complete and actual_fit_keys == expected_fit_keys and len(actual_fit_keys) == len(fits)
            fit_complete = fit_complete and set(selection_packages) == set(_prefix_selection_requests(plan, scopes))
        if (persisted["status"] != "completed" or any(persisted[key] != value for key, value in metadata.items())
                or sorted(map(semantic_sha256, persisted["predictions"])) != sorted(map(semantic_sha256, predictions))
                or "fits" in persisted and sorted(map(semantic_sha256, persisted["fits"])) != sorted(map(semantic_sha256, fits))
                or sorted(map(semantic_sha256, persisted.get("selection_fits", []))) != sorted(map(semantic_sha256, selection_fits))
                or len(identities) != len(applicable) or not fit_complete):
            raise ValueError("completed candidate result differs from its sealed requests or predictions")
        result.update(persisted)
    # Keep the original dependency facts after the independent numerical gate;
    # finalization rechecks these same bytes without retaining decoded sources.
    source_artifacts = ({} if dependencies is None else {
        reference["checkpoint_sha256"]: {
            "path": str((directory / dependencies["directory"] / (reference["checkpoint_sha256"] + ".pt")).resolve()),
            "sha256": reference["checkpoint_sha256"], "bytes": reference["checkpoint_bytes"]}
        for reference in dependencies["references"]})
    result["source_metadata_artifacts"] = [source_artifacts[key] for key in sorted(source_artifacts)]
    return result


def _verify_candidate_source_artifacts(participants):
    """Retain each independently admitted original blob through finalization."""
    admitted = {}
    for participant in participants:
        for artifact in participant.get("source_metadata_artifacts", ()):
            path = Path(artifact["path"])
            if str(path) in admitted:
                if admitted[str(path)] != artifact:
                    raise ValueError("candidate source consumers disagree on their original blob")
                continue
            receipt = _file_receipt(path, path.parent)
            if receipt["sha256"] != artifact["sha256"] or receipt["bytes"] != artifact["bytes"]:
                raise ValueError("admitted candidate source bytes changed before programme finalization")
            admitted[str(path)] = artifact


class CandidateCloudProgramme:
    """Scientific consumer hooks; the caller owns the single CloudExecutor."""

    def __init__(self, plan_path, output, resources, *, task_prefix, dependencies, input_admission=None,
                 metadata_directory=None):
        self.original_plan_path = Path(plan_path).resolve()
        self.output, self.resources = Path(output).resolve(), resources
        self.metadata_directory = Path(metadata_directory or self.output / "source_metadata").resolve()
        self.task_prefix, self.dependencies = task_prefix, tuple(dependencies)
        self.input_admission = input_admission
        self.base_plan = json.loads(self.original_plan_path.read_bytes())
        from research.prefix_recovery import preserve_prefix_producer_output

        preserve_prefix_producer_output(self.base_plan, self.output)
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
        if self.plan.get("joint_source_catalog"):
            self.plan["joint_source_catalog"] = [{**entry,
                "path": str((self.original_plan_path.parent / entry["path"]).resolve())}
                for entry in self.plan["joint_source_catalog"]]
        self.source, self.normalized, self.admitted, self.admission_failures = None, None, {}, {}
        self.source_snapshots = {}
        self.recovery_binding = None
        self.evaluation_tasks, self.fit_outcomes = {}, {}
        self.replay_states, self.replay_task_roles = {}, {}
        self.replay_pool = None
        self.preparation_failures = {}
        self.source_artifacts = {}
        self.replay_enabled = self.plan.get("evaluation_design") == "independent_prefix" and input_admission is not None
        if self.replay_enabled:
            from research.prefix_replay_dispatch import PrefixReplayInputPool
            from research.prefix_replay_execution import validate_replay_reader_resources

            self.replay_pool = PrefixReplayInputPool(input_admission,
                release_backing_aliases=self._release_replay_backing_aliases,
                validate_reader=validate_replay_reader_resources)
        _write_candidate_json(self.output / "consumer_requests.json", {"schema": "n2p3_candidate_consumer_requests/1",
            "base_plan_sha256": _file_receipt(self.original_plan_path, self.original_plan_path.parent)["sha256"],
            "requests": [request for index in range(len(self.plan["records"])) for request in _pipeline_requests(self.plan, index)]}, input_admission)
        _write_candidate_json(self.plan_path, self.plan, input_admission)
        self.tasks = candidate_tasks(self.plan_path, self.output, resources, dependencies=self.dependencies,
                                     task_prefix=task_prefix, metadata_directory=self.metadata_directory)
        self.task_roster = tuple(self.tasks)
        self.execution_workers = (resources["max_cache_group_processes"] if self.replay_enabled
            else resources["target_workers"])

    def bind_source(self, source_seed_catalog=None, source_snapshot=None):
        if source_seed_catalog is not None:
            self.plan["source_seed_catalog"] = source_seed_catalog
        if source_snapshot is not None:
            self.plan["source_snapshot_manifest"] = str(Path(source_snapshot).resolve())
        _write_candidate_json(self.plan_path, self.plan, self.input_admission)
        self.normalized, self.source, self.source_snapshots = _prepare_candidate_plan(
            self.plan_path, load_sources=self.replay_enabled or self.plan.get("prefix_selection") is not None
            and self.plan.get("prefix_recovery") is None)
        _write_candidate_json(self.output / "plan.json", self.normalized, self.input_admission)
        _write_candidate_json(self.output / "executed_source.json", self.source, self.input_admission)
        if self.replay_enabled:
            self.current_context = self._write_admission_context("current", self.normalized, self.source)
        if self.plan.get("prefix_recovery") is not None:
            self._recover_participants()
        return self.tasks

    def _recover_participants(self):
        from research.prefix_recovery import load_prefix_recovery

        producer = load_prefix_recovery(self.normalized, self.source, output=self.output, consumer_root=ROOT)
        self.recovery_binding = producer.binding
        self.current_context = self._write_admission_context("current", self.normalized, self.source)
        original_context = self._write_admission_context("original", producer.plan, producer.source)
        recovery_tasks = []
        for participant in producer.participants:
            original, outcome = participant.task, participant.outcome
            if original.task_id != self.task_roster[participant.record_index].task_id:
                raise ValueError("prefix recovery producer task differs from its original logical roster")
            self.fit_outcomes[original.task_id] = outcome
            if self.replay_enabled:
                recovery_tasks.append(self._make_replay_preparation(original, outcome, original_context,
                    producer.plan, producer.source, "original_producer_independent_worker_replay",
                    source_root=producer.binding["original_source_root"], dependencies=self.dependencies))
            else:
                recovery_tasks.append(self._make_admission_task(original, outcome, original_context,
                    producer.plan, producer.source, "original_producer_independent_worker_replay",
                    dependencies=self.dependencies))
        recovered = set(producer.binding["reused_record_indexes"])
        self.tasks = recovery_tasks + [task for task in self.task_roster if task.arguments["record_index"] not in recovered]
        if self.plan["prefix_recovery"].get("mode") == "all_fits_zero_fit" and (
                recovered != set(range(len(self.task_roster)))
                or any(task.adapter.endswith(":candidate_participant_adapter") for task in self.tasks)):
            raise ValueError("all-fit prefix recovery must schedule zero new scientific fit actors")
        self.execution_workers = self.resources["max_cache_group_processes"]
        _write_candidate_json(self.output / "prefix_recovery_binding.json", self.recovery_binding, self.input_admission)
        self._write_admission_progress()

    def _write_admission_context(self, name, plan, source):
        path = self.output / f"admission_context_{name}.json"
        record = {"plan": plan, "source": source}
        _write_candidate_json(path, {"record": record, "record_sha256": semantic_sha256(record)}, self.input_admission)
        return {"path": str(path), "sha256": _file_receipt(path, self.output)["sha256"]}

    def _make_admission_task(self, producer_task, fit_outcome, context, plan, source, role, *, dependencies=()):
        from research.cloud_execution import CloudTask, PhaseResources

        logical_id = producer_task.task_id
        index = producer_task.arguments["record_index"]
        directory = self.output / "admission_groups" / f"participant_{index:03d}"
        request = {"logical_task_id": logical_id, "record_index": index, "producer_role": role,
            "producer_source_digest": semantic_sha256(source), "producer_plan_digest": semantic_sha256(_plan_authority(plan)),
            "consumer_source_digest": semantic_sha256(self.source),
            "consumer_plan_digest": semantic_sha256(_plan_authority(self.normalized)),
            "producer_task": asdict(producer_task), "fit_outcome": asdict(fit_outcome),
            "participant_metadata": _participant_metadata_receipts(producer_task)}
        task = CloudTask(logical_id + ":admission",
            "experiments.run_candidate_ensemble_study:candidate_participant_admission_adapter",
            {"plan": str(self.plan_path), "output": str(directory), "context": context, "request": request,
             "output_estimate_bytes": min(2**26, self.resources["target_worker_memory_bytes"])}, str(directory),
            PhaseResources("evaluation", producer_task.resources.native_threads + 1,
                self.resources["target_worker_memory_bytes"], producer_task.resources.native_threads,
                "cpu", interop_threads=1), tuple(dependencies))
        self.evaluation_tasks[task.task_id] = task
        return task

    def _replay_memory(self, graph=False):
        name = "prefix_replay_graph_memory_gib" if graph else "prefix_replay_worker_memory_gib"
        value = self.plan.get(name)
        if value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError("prefix replay memory budgets must be explicit positive GiB")
            return int(value * 2**30)
        if graph and self.plan["evidence_scope"] != "software_correctness":
            return 8 * 2**30
        return self.resources["target_worker_memory_bytes"]

    def _source_replay_artifacts(self):
        from research.prefix_replay_execution import artifact_receipt

        if self.source_artifacts:
            return self.source_artifacts
        directory = self.output / "replay_sources"
        directory.mkdir(exist_ok=True)
        for key, snapshot in self.source_snapshots.items():
            path = directory / (snapshot.digest + ".pt")
            if not path.exists():
                writer = (self.input_admission.artifact_writer(path, len(snapshot.content))
                    if self.input_admission is not None else nullcontext())
                with writer, path.open("xb") as stream:
                    stream.write(snapshot.content)
            receipt = artifact_receipt(path)
            if receipt["sha256"] != snapshot.digest:
                raise ValueError("replay control artifact changed its original canonical source bytes")
            self.source_artifacts[key] = receipt
        for source_id, (entry, base) in _joint_source_entries(self.plan).items():
            receipt = artifact_receipt((base / entry["codec"]["path"]).resolve())
            if (receipt["sha256"] != entry["codec"]["sha256"]
                    or receipt["bytes"] != entry["codec"]["bytes"]):
                raise ValueError("replay joint source artifact differs from its original catalogue codec")
            self.source_artifacts["joint_flat", source_id] = receipt
        return self.source_artifacts

    def _make_replay_preparation(self, producer_task, fit_outcome, context, plan, source, role,
                                 *, source_root, dependencies=()):
        from research.cloud_execution import CloudTask, PhaseResources
        from research.prefix_replay_execution import artifact_receipt

        authority = _candidate_participant_authority(producer_task, fit_outcome, plan, source)
        if authority is None:
            raise ValueError("replay preparation requires the original physical participant authority")
        _, index, _, _, _, _ = authority
        source_artifacts = self._source_replay_artifacts()

        request = _prefix_replay_participant_anchor(producer_task, plan, source, self.source, authority,
                                                   source_root=source_root)
        selection, ordinary = request["selection_packages"], request["ordinary_packages"]
        for row in selection + ordinary:
            row["source_checkpoints"] = _prefix_replay_source_artifacts(row["original_request"], plan, source_artifacts)
        if not selection and not ordinary:
            return self._make_admission_task(producer_task, fit_outcome, context, plan, source, role,
                dependencies=dependencies)
        logical_id = producer_task.task_id
        prep_id = request["preparation_task_id"]
        request["producer_module"] = artifact_receipt(Path(source_root) / "src/transfer/ensemble_views.py")
        task = CloudTask(prep_id, "research.prefix_replay_execution:prefix_replay_prepare_adapter",
            {"request": request, "output": str(self.output / "replay_groups" / f"participant_{index:03d}" / "prepare"),
             "output_estimate_bytes": max(2**26, sum(receipt["bytes"] for row in selection for receipt in row["files"]))},
            str(self.output / "replay_groups" / f"participant_{index:03d}" / "prepare"),
            PhaseResources("evaluation", 2, self._replay_memory(graph=True), 1, "cpu", interop_threads=1), tuple(dependencies))
        self.replay_states[prep_id] = {"logical_id": logical_id, "producer_task": producer_task,
            "fit_outcome": fit_outcome, "context": context, "plan": plan, "source": source, "role": role,
            "request": request, "tasks": {}, "worker_outcomes": {}, "reducer_outcomes": {}, "reducers": {}}
        self.replay_task_roles[prep_id] = prep_id, "prepare", None
        return task

    def _replay_task(self, state, task_id, adapter, arguments, *, graph=False, dependencies=()):
        from research.cloud_execution import CloudTask, PhaseResources
        from research.prefix_replay_execution import replay_native_threads

        native = replay_native_threads(state["request"], ordinary=adapter.startswith("ordinary_"))
        directory = self.output / "replay_groups" / f"participant_{state['producer_task'].arguments['record_index']:03d}" / task_id.rsplit(":", 1)[-1]
        return CloudTask(task_id, "research.prefix_replay_execution:" + adapter,
            {**state["reader_arguments"], **arguments, "output": str(directory),
             "output_estimate_bytes": 2**26}, str(directory),
            PhaseResources("evaluation", native + 1, self._replay_memory(graph=graph), native,
                "cpu", interop_threads=1), tuple(dependencies))

    def _initial_replay_readers(self, preparation_id, holder):
        state = self.replay_states[preparation_id]
        manifest = state["manifest"]
        common = {"request": state["request"], "shared_inputs": holder.descriptor,
            "input_request_sha256": manifest["input_request_sha256"], "capsule_sha256": manifest["capsule"]["sha256"]}
        state["reader_arguments"] = common
        configured = self.plan.get("prefix_replay_chunk_size", 4)
        if type(configured) is not int or configured < 1:
            raise ValueError("prefix replay chunk size must be a positive integer")
        node_count = sum(len(job["node_requests"]) for job in manifest["selection_jobs"])
        ready_slots = max(1, min(self.execution_workers,
            self.resources["cpu_budget"] - self.resources.get("coordinator_cpu_threads", 1)))
        node_slots = max(1, ready_slots - min(ready_slots - 1, len(manifest["ordinary_jobs"])))
        chunk_size = min(configured, max(1, node_count // node_slots))
        state["batching"] = {"nodes": node_count, "ordinary_ready": len(manifest["ordinary_jobs"]),
            "cpu_ready_slots": ready_slots, "node_slots": node_slots, "chunk_size": chunk_size}
        tasks, roles = [], {}
        for job in manifest["selection_jobs"]:
            key = "selection", job["package"]
            dependencies = []
            for index, start in enumerate(range(0, len(job["node_requests"]), chunk_size)):
                task_id = state["logical_id"] + f":node_{job['package']}_{index:03d}"
                task = self._replay_task(state, task_id, "selection_chunk_adapter",
                    {"package": job["package"], "node_requests": job["node_requests"][start:start + chunk_size]})
                tasks.append(task)
                roles[task_id] = preparation_id, "selection_worker", key
                dependencies.append(task_id)
            state["reducers"][key] = {"job": job, "dependencies": dependencies}
        for row in manifest["ordinary_jobs"]:
            key = "ordinary", row["package"]
            task_id = state["logical_id"] + ":ordinary_" + row["package"]
            task = self._replay_task(state, task_id, "ordinary_prefix_replay_adapter", {"package": row["package"]})
            tasks.append(task)
            roles[task_id] = preparation_id, "ordinary_worker", key
            state["reducers"][key] = {"dependencies": [task_id]}
        # Build facts survive a failed progress write; only returned tasks become actual executor work.
        state["tasks"].update((task.task_id, task) for task in tasks)
        self.replay_task_roles.update(roles)
        self._write_admission_progress()
        self.tasks.extend(tasks)
        return tuple(tasks)

    def _on_finished_preparation(self, preparation_id, outcome):
        from research.prefix_replay_execution import read_prefix_preparation

        state = self.replay_states[preparation_id]
        state["preparation_outcome"] = outcome
        if outcome.status != "completed":
            self.preparation_failures[preparation_id] = outcome.failure
            self._write_admission_progress()
            return ()
        task = next(task for task in self.tasks if task.task_id == preparation_id)
        state["manifest"] = read_prefix_preparation(task, outcome, state["request"])
        additions = self.replay_pool.queue_preparation(outcome, state["manifest"]["input_request"],
            lambda holder: self._initial_replay_readers(preparation_id, holder), capsule_key="capsule")
        self.preparation_failures.update(self.replay_pool.failures)
        self._write_admission_progress()
        return additions

    def _on_finished_replay_reader(self, preparation_id, role, key, outcome):
        state = self.replay_states[preparation_id]
        additions = []
        if role.endswith("worker"):
            state["worker_outcomes"][outcome.task_id] = outcome
            reducer = state["reducers"][key]
            if ("task_id" not in reducer
                    and all(task_id in state["worker_outcomes"] for task_id in reducer["dependencies"])):
                kind, package = key
                task_id = state["logical_id"] + ":reduce_" + kind + "_" + package
                facts = [(state["tasks"][task_id], state["worker_outcomes"][task_id]) for task_id in reducer["dependencies"]]
                arguments = ({"job": reducer["job"], "node_outcomes": facts} if kind == "selection"
                    else {"package": package, "ordinary_outcome": facts[0]})
                task = self._replay_task(state, task_id, kind + "_reducer_adapter", arguments,
                    graph=kind == "selection", dependencies=reducer["dependencies"])
                reducer["task_id"] = task_id
                state["tasks"][task_id] = task
                self.replay_task_roles[task_id] = preparation_id, "reducer", key
                additions.append(task)
        elif role == "reducer":
            state["reducer_outcomes"][key] = outcome
            if len(state["reducer_outcomes"]) == len(state["reducers"]):
                task = self._make_admission_task(state["producer_task"], state["fit_outcome"], state["context"],
                    state["plan"], state["source"], state["role"],
                    dependencies=[row["task_id"] for row in state["reducers"].values()])
                facts = [(state["tasks"][row["task_id"]], state["reducer_outcomes"][key])
                    for key, row in state["reducers"].items()]
                task = replace(task, arguments={**task.arguments, "replay_request": state["request"],
                    "reducer_outcomes": facts}, resources=replace(task.resources, memory_bytes=self._replay_memory(graph=True)))
                self.evaluation_tasks[task.task_id] = task
                self.replay_task_roles[task.task_id] = preparation_id, "final_admission", None
                state["tasks"][task.task_id] = task
                additions.append(task)
        elif role == "final_admission":
            self._on_finished_admission(outcome)
        # A successor owns the backing before the predecessor gives its lease back.
        if additions:
            self.replay_pool.add_readers(preparation_id, additions)
            self._write_admission_progress()
            self.tasks.extend(additions)
        unlocked = self.replay_pool.finished_reader(outcome)
        self._scrub_retired_replay_descriptors()
        self.preparation_failures.update(self.replay_pool.failures)
        return tuple(additions) + tuple(unlocked)

    def _scrub_retired_replay_descriptors(self):
        resident = set(self.replay_pool.record()["resident_preparations"])
        for preparation_id, state in self.replay_states.items():
            if preparation_id in resident or "reader_arguments" not in state:
                continue
            self._release_replay_backing_aliases(preparation_id)

    def _release_replay_backing_aliases(self, preparation_id):
        state = self.replay_states[preparation_id]
        if "reader_arguments" not in state:
            return
        descriptor = state["reader_arguments"]["shared_inputs"]
        if descriptor["backing"] == "immutable_bytes":
            # The pool invokes this after the last actual tree retirement and
            # before freeing its byte claim or adopting another preparation.
            # All parent task arguments share this same descriptor mapping.
            descriptor.pop("payload", None)
            descriptor.pop("capsule", None)
        state["input_descriptor_retired"] = True

    def _write_admission_progress(self):
        _write_candidate_json(self.output / "progress.json", {"status": "running",
            "fitted_subjects": sum(value.status == "completed" for value in self.fit_outcomes.values()),
            "completed_subjects": sum(row["status"] == "completed" for row in self.admitted.values()),
            "resolved_subjects": len(self.admitted),
            "pending_admission_subjects": [task.arguments["request"]["record_index"] for task in self.evaluation_tasks.values()
                if task.arguments["request"]["logical_task_id"] not in self.admitted]}, self.input_admission)

    def _on_finished_admission(self, outcome):
        task = self.evaluation_tasks[outcome.task_id]
        request = task.arguments["request"]
        logical_id = request["logical_task_id"]
        if outcome.status != "completed":
            self.admission_failures[logical_id] = outcome.failure
            self._write_admission_progress()
            return ()
        expected = Path(task.output) / "admission.json"
        if outcome.value != _file_receipt(expected, task.output):
            raise ValueError("candidate admission artifact differs from the independent worker's return digest")
        record = _read_sealed_record(expected)
        if (record["schema"] != "n2p3_candidate_participant_admission/1"
                or semantic_sha256(record["request"]) != semantic_sha256(request)
                or record["consumer_source"] != self.source
                or _participant_metadata_receipts(_admission_task_descriptor(request["producer_task"])) != request["participant_metadata"]):
            raise ValueError("candidate admission receipt rewrites its participant or producing execution")
        admitted = record["participant"]
        if admitted is not None:
            ledger = _read_sealed_record(Path(request["producer_task"]["output"]) /
                f"participant_{request['record_index']:03d}" / "request_ledger.json")
            if (admitted["record_index"] != request["record_index"]
                    or any(semantic_sha256(admitted[key]) != semantic_sha256(ledger[key]) for key in
                        ("requested", "coverage", "participant_key", "candidate_vocabulary", "evidence_level"))
                    or any(row["score_source_digest"] != request["producer_source_digest"] for row in admitted["predictions"])):
                raise ValueError("candidate admission result omits requested authority or changes original score producer")
            self.admitted[logical_id] = admitted
            if request["producer_role"] == "original_producer_independent_worker_replay":
                _write_candidate_json(self.output / f"recovered_participant_{request['record_index']:03d}.json",
                                      admitted["execution_binding"], self.input_admission)
        self._write_admission_progress()
        return ()

    def _on_finished_fit(self, outcome):
        tasks = {task.task_id: task for task in self.task_roster}
        task = tasks[outcome.task_id]
        self.fit_outcomes[outcome.task_id] = replace(outcome,
            worker=None if outcome.worker is None else dict(outcome.worker),
            supplementary_failures=list(outcome.supplementary_failures))
        authority = _candidate_participant_authority(task, outcome, self.normalized, self.source)
        # Complete fallible IO before publishing a task for the executor's ACK.
        self._write_admission_progress()
        if authority is None:
            return ()
        evaluation = (self._make_replay_preparation(task, self.fit_outcomes[outcome.task_id], self.current_context,
            self.normalized, self.source, "new_producer", source_root=ROOT)
            if self.replay_enabled else self._make_admission_task(task, outcome, self.current_context,
                self.normalized, self.source, "new_producer"))
        self.tasks.append(evaluation)
        return (evaluation,)

    def _unresolved_pipeline_outcome(self, logical_id, results):
        from research.cloud_execution import CloudResult

        evaluation = results.get(logical_id + ":admission")
        fitted = results.get(logical_id) or self.fit_outcomes.get(logical_id)
        for preparation_id, state in self.replay_states.items():
            if state["logical_id"] != logical_id:
                continue
            actual = [results[key] for key in [*state["tasks"], preparation_id] if key in results]
            for status in ("failed", "blocked", "not_admitted"):
                for value in actual:
                    if value.status == status:
                        return value
            if logical_id in self.admission_failures:
                return CloudResult(logical_id, "not_admitted", failure=self.admission_failures[logical_id])
        if fitted is not None and fitted.status != "completed":
            return fitted
        if evaluation is not None and evaluation.status != "completed":
            return evaluation
        return CloudResult(logical_id, "not_admitted", failure={"type": "MissingIndependentAdmission",
            "stage": "artifact_admission", "message": "fitted output has no accepted numerical admission"})

    def _unresolved_participant(self, task, results):
        actual_task, plan, source = task, self.normalized, self.source
        for state in self.replay_states.values():
            if state["logical_id"] == task.task_id:
                actual_task, plan, source = state["producer_task"], state["plan"], state["source"]
                break
        fitted = self.fit_outcomes.get(task.task_id) or results.get(task.task_id)
        if fitted is None:
            fitted = next((state["fit_outcome"] for state in self.replay_states.values()
                if state["logical_id"] == task.task_id), None)
        if fitted is None or plan is None:
            return None
        authority = _candidate_participant_authority(actual_task, fitted, plan, source)
        if authority is None:
            return None
        ledger = authority[-2]
        failure = self._unresolved_pipeline_outcome(task.task_id, results)
        metadata = {key: ledger[key] for key in ("requested", "coverage", "candidate_vocabulary", "evidence_level", "participant_key", *(["requested_evidence_levels"] if "requested_evidence_levels" in ledger else []))}
        metadata.update({key: ledger[key] for key in ("requested_to_actual_K", "budget_adjustments") if key in ledger})
        return {**metadata, "status": "unresolved_admission", "predictions": [], "fits": [],
            "selection_fits": [], "record_index": task.arguments["record_index"],
            "pipeline_requests": [{**request, "status": failure.status, "failure": failure.failure}
                for request in ledger["pipeline_requests"]], "admission_failure": failure.failure}

    def on_finished(self, outcome):
        if self.replay_enabled:
            if outcome.task_id in self.replay_task_roles:
                preparation_id, role, key = self.replay_task_roles[outcome.task_id]
                try:
                    return (self._on_finished_preparation(preparation_id, outcome) if role == "prepare"
                        else self._on_finished_replay_reader(preparation_id, role, key, outcome))
                except Exception as error:
                    state = self.replay_states[preparation_id]
                    self.admission_failures[state["logical_id"]] = {
                        "type": type(error).__name__, "message": str(error), "stage": "artifact_admission"}
                    raise
            if outcome.task_id in self.evaluation_tasks:
                return self._on_finished_admission(outcome)
            return self._on_finished_fit(outcome)
        if self.recovery_binding is not None:
            if outcome.task_id in self.evaluation_tasks:
                return self._on_finished_admission(outcome)
            return self._on_finished_fit(outcome)
        tasks = {task.task_id: task for task in self.tasks}
        if outcome.task_id not in tasks:
            raise ValueError("candidate consumer received another programme's task")
        if self.normalized is None:
            if outcome.status != "blocked":
                raise ValueError("candidate consumer cannot execute before its actual source binding")
            return ()
        try:
            result = _admit_candidate_participant(tasks[outcome.task_id], outcome, self.normalized, self.source,
                                                  source_snapshots=self.source_snapshots)
            if result is not None:
                self.admitted[outcome.task_id] = result
        except Exception as error:
            self.admission_failures[outcome.task_id] = f"{type(error).__name__}: {error}"
            raise
        _write_candidate_json(self.output / "progress.json", {"status": "running",
            "completed_subjects": sum(row["status"] == "completed" for row in self.admitted.values()),
            "resolved_subjects": len(self.admitted)}, self.input_admission)
        return ()

    def finalize(self, results, *, input_admission=None, cloud_run=None):
        from research.cloud_execution import CloudResult

        owner = input_admission if input_admission is not None else self.input_admission
        if self.replay_pool is not None:
            if cloud_run is not None:
                if cloud_run.results is not results:
                    raise ValueError("candidate replay cleanup requires the actual originating run results")
                self.replay_pool.retire_after_run(cloud_run)
                self._scrub_retired_replay_descriptors()
            elif self.replay_pool.record()["resident_preparations"] or self.replay_pool.record()["pending_preparations"]:
                raise ValueError("candidate replay cleanup requires its actual settled CloudRun")
            self.preparation_failures.update(self.replay_pool.failures)
        _verify_candidate_source_artifacts(self.admitted.values())
        outcomes = [results.get(task.task_id) or CloudResult(task.task_id, "not_started", failure={
            "type": "UnpublishedReplayTask", "stage": "artifact_admission",
            "message": "task construction or callback ACK failed before actual executor publication"}) for task in self.tasks]
        participants = [self.admitted[task.task_id] for task in self.task_roster if task.task_id in self.admitted]
        if self.replay_enabled:
            for task in self.task_roster:
                if task.task_id not in self.admitted:
                    unresolved = self._unresolved_participant(task, results)
                    if unresolved is not None:
                        participants.append(unresolved)
        participant_records = {row["record_index"]: row for row in participants}
        failures = [{"task_id": value.task_id, "status": value.status, "failure": value.failure,
            "supplementary_failures": value.supplementary_failures} for value in outcomes if value.status != "completed"]
        for task in self.task_roster:
            if task.task_id not in self.admitted:
                missing = self._unresolved_pipeline_outcome(task.task_id, results)
                if not any(row["task_id"] == missing.task_id for row in failures):
                    failures.append({"task_id": missing.task_id, "status": missing.status,
                        "failure": missing.failure, "supplementary_failures": missing.supplementary_failures})
        if self.source is not None and _executing_source() != self.source:
            raise ValueError("candidate execution source changed during the programme")
        if failures:
            report = {"schema": "n2p3_candidate_ensemble_report/1", "status": "failed",
                "evidence_scope": self.plan["evidence_scope"], "participants": participants,
                "requested_records": self.plan["records"], "execution_failures": failures,
                "admission_failures": self.admission_failures, "preparation_failures": self.preparation_failures,
                "requested_pipeline_cells": [request for task in self.task_roster for request in
                    (participant_records[task.arguments["record_index"]]["pipeline_requests"] if task.arguments["record_index"] in participant_records else
                     [{**value, "status": self._unresolved_pipeline_outcome(task.task_id, results).status,
                        "failure": self._unresolved_pipeline_outcome(task.task_id, results).failure}
                      for value in _pipeline_requests(self.plan, task.arguments["record_index"])])],
                "unresolved_record_indexes": [task.arguments["record_index"] for task in self.task_roster if task.task_id not in self.admitted],
                "decision_denominator_complete": len(participants) == len(self.task_roster),
                "source_digest": None if self.source is None else semantic_sha256(self.source)}
            if participants:
                measured = _finalize_candidate_report(self.normalized, self.source, participants, self.output,
                    status="failed", persist=False)
                report.update(decision_plan=measured["decision_plan"], metrics=measured["metrics"], contrasts=measured["contrasts"],
                    metrics_scope="all_resolved_requested_decisions" if report["decision_denominator_complete"] else "resolved_subset_with_unresolved_roster")
            if self.recovery_binding is not None:
                report.update(prefix_recovery=self.recovery_binding,
                    participant_execution={str(row["record_index"]): row["execution_binding"] for row in participants if "execution_binding" in row},
                    fit_execution={key: asdict(value) for key, value in self.fit_outcomes.items()},
                    admission_execution={key: asdict(results[key]) if key in results else {
                        "task_id": key, "status": "not_started", "worker": None} for key in self.evaluation_tasks})
            if self.replay_enabled:
                report.update(fit_execution={key: asdict(value) for key, value in self.fit_outcomes.items()},
                    replay_execution={key: asdict(results[key]) if key in results else {
                        "task_id": key, "status": "not_started", "worker": None} for key in self.replay_task_roles},
                    replay_inputs=self.replay_pool.record())
                report["replay_batching"] = {key: state["batching"] for key, state in self.replay_states.items() if "batching" in state}
            _write_candidate_json(self.output / "result.json", report, owner)
            _write_candidate_json(self.output / "progress.json", {"status": "failed", "resolved_subjects": len(participants),
                "completed_subjects": sum(row["status"] == "completed" for row in participants)}, owner)
            return report
        if len(participants) != len(self.task_roster):
            raise ValueError("candidate programme lacks independently admitted participant results")
        report = _finalize_candidate_report(self.normalized, self.source, participants, self.output, input_admission=owner,
            persist=self.recovery_binding is None, recovery_binding=self.recovery_binding)
        report["decision_denominator_complete"] = len(participants) == len(self.task_roster)
        if self.recovery_binding is not None:
            report.update(fit_execution={key: asdict(value) for key, value in self.fit_outcomes.items()},
                          admission_execution={key: asdict(results[key]) if key in results else {
                              "task_id": key, "status": "not_started", "worker": None} for key in self.evaluation_tasks})
            _write_candidate_json(self.output / "result.json", report, owner)
            _write_candidate_json(self.output / "progress.json", {"status": report["status"],
                "fitted_subjects": sum(value.status == "completed" for value in self.fit_outcomes.values()),
                "completed_subjects": sum(row["status"] == "completed" for row in participants),
                "resolved_subjects": len(participants)}, owner)
        if self.replay_enabled:
            report.update(fit_execution={key: asdict(value) for key, value in self.fit_outcomes.items()},
                replay_execution={key: asdict(results[key]) for key in self.replay_task_roles if key in results},
                replay_inputs=self.replay_pool.record())
            report["replay_batching"] = {key: state["batching"] for key, state in self.replay_states.items() if "batching" in state}
            _write_candidate_json(self.output / "result.json", report, owner)
        return report


def build_candidate_cloud_tasks(plan_path, output, resources, *, source_seed_catalog=None, source_snapshot=None,
                                task_prefix="candidate", dependencies=(), input_admission=None, metadata_directory=None):
    programme = CandidateCloudProgramme(plan_path, output, resources, task_prefix=task_prefix, dependencies=dependencies,
                                        input_admission=input_admission, metadata_directory=metadata_directory)
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
    supported = {"fixed", "grouped_temperature"} | (POOL_STRATEGIES if independent else set())
    if (not isinstance(strategies, list) or not strategies or len(set(strategies)) != len(strategies)
            or set(strategies) - supported or not set(strategies) & {"fixed", "grouped_temperature"}):
        raise ValueError("calibration programmes need distinct supported strategies and a declared base programme")
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
            if strategy in {"fixed", POOL_STRATEGY} or set(mixture.expert_ids) & set(ids)] for mixture in mixtures}
    if plan.get("prefix_selection") is not None:
        from research.calibration_selection import CalibrationProgramCandidate

        options = plan["prefix_selection"]
        if (not independent or options.get("schema") != "n2p3_prefix_selection_plan/1"
                or type(options.get("minimum_selection_rows")) is not int or options["minimum_selection_rows"] < 1
                or not isinstance(options.get("output_name"), str) or not options["output_name"]
                or options["output_name"] != options["output_name"].strip()
                or options["output_name"] in {mixture.name for mixture in mixtures}):
            raise ValueError("prefix selector needs an explicit independent output and positive structural row policy")
        candidates = tuple(CalibrationProgramCandidate.from_record(value) for value in options["candidates"])
        candidate_names = [candidate.name for candidate in candidates]
        declared = {spec.expert_id: spec for spec in expert_programs}
        if (not candidates or len(set(candidate_names)) != len(candidate_names)
                or options["baseline_name"] not in candidate_names):
            raise ValueError("prefix selector candidate roster/baseline must be complete and explicit")
        for candidate in candidates:
            for spec in candidate.expert_programs:
                if (spec.expert_id not in declared or semantic_sha256(spec.record()) !=
                        semantic_sha256(declared[spec.expert_id].record())):
                    raise ValueError("prefix candidate specs must match their explicit shared producer declarations")
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
    normalized["records"] = [{**row, **({"dataset_id": _target_study(plan)} if _target_study(plan) != BNCI2014_008_DATASET_ID else {}),
                              "path": str((plan_path.parent / row["path"]).resolve())}
                              for row in plan["records"]]
    normalized["source_checkpoints"] = {name: str((plan_path.parent / path).resolve())
                                        for name, path in plan["source_checkpoints"].items()}
    joint_specs = tuple(spec for spec in expert_programs if spec.family == "joint_flat")
    if joint_specs:
        if not independent:
            raise ValueError("joint candidates require their original independent-prefix compiler")
        if len(pairs) != 1:
            raise ValueError("joint programme uses one declared pipeline pair; independent repeats need distinct sampler programmes")
        normalized["joint_source_catalog"] = [{**row,
            "path": str((plan_path.parent / row["path"]).resolve())} for row in plan["joint_source_catalog"]]
        entries = _joint_source_entries(normalized)
        bindings = plan["joint_source_bindings"]
        if not bindings or any(set(row) != {spec.expert_id for spec in joint_specs} for row in bindings.values()):
            raise ValueError("joint programme needs an exact source binding for each actual target and joint expert")
        for participant, row in bindings.items():
            for source_id in row.values():
                if (source_id not in entries
                        or entries[source_id][0]["exclusion"]["target_participant_key"] != participant):
                    raise ValueError("joint source entry differs from its declared excluded target participant")
        for mixture in mixtures:
            if any(spec.expert_id in mixture.expert_ids for spec in joint_specs) and set(
                    plan.get("mixture_calibration_strategies", {}).get(mixture.name, strategies)) & POOL_STRATEGIES:
                raise ValueError("joint probability programmes use their declared forward temperature algorithm")
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
            required = {(_target_study(plan), pair[0]) for pair in pairs}
            if required != available:
                raise ValueError("candidate source catalog differs from declared outer-study/source-seed producers")
        source_snapshots = {(entry["outer_target_study"], entry["source_seed"]):
            FrozenSourceCheckpoint(catalog.content(entry["outer_target_study"], entry["source_seed"]))
            for entry in catalog.record() if load_sources and entry["outer_target_study"] == _target_study(plan)
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


def _target_study(plan):
    return plan.get("dataset_id", BNCI2014_008_DATASET_ID)


def _candidate_source_declaration(row):
    if row.get("dataset_id", BNCI2014_008_DATASET_ID) == BNCI2014_008_DATASET_ID:
        return row
    payload = Path(row["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("candidate participant original source declaration changed")
    declaration = json.loads(payload)
    if declaration["dataset_id"] != row["dataset_id"]:
        raise ValueError("candidate participant declaration belongs to another original study")
    return declaration


def run_study(plan_path: str | Path, output: str | Path, *, _actor_index=None, _native_threads=None,
              execution="cloud", _metadata_directory=None):
    """Standalone cloud execution, or the explicit in-process scientific seam."""
    if execution not in {"cloud", "inline"}:
        raise ValueError("candidate execution must be cloud or inline")
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    original_plan = json.loads(plan_path.read_bytes())
    declaration = original_plan.get("prefix_recovery")
    if declaration is not None and declaration.get("mode") == "all_fits_zero_fit" and _actor_index is not None:
        raise ValueError("all-fit prefix recovery forbids every new scientific fit actor")
    if declaration is not None and (execution != "cloud" or _actor_index in {
            row["record_index"] for row in declaration["participants"]}):
        raise ValueError("prefix recovery fits only remaining original participants through the shared cloud owner")
    if declaration is not None:
        from research.prefix_recovery import preserve_prefix_producer_output

        preserve_prefix_producer_output(original_plan, output)
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
            return _subject(row, plan, output / f"participant_{_actor_index:03d}", source, snapshots,
                            metadata_directory=_metadata_directory)
        if execution == "inline":
            for index, row in enumerate(plan["records"]):
                completed.append(_subject(row, plan, output / f"participant_{index:03d}", source, snapshots,
                    metadata_directory=_metadata_directory or output / "source_metadata"))
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
            run = CloudExecutor(admission, max_workers=programme.execution_workers,
                diagnostic_directory=output / "worker_diagnostics",
                identity={"actual_source_root": str(ROOT), "plan_sha256": semantic_sha256(plan)}).run(
                    programme.tasks, on_finished=programme.on_finished)
            return programme.finalize(run.results, input_admission=input_owner, cloud_run=run)
    except BaseException as error:
        atomic_write_json(output / "progress.json", {"status": "failed", "completed_subjects": len(completed),
                          "error": f"{type(error).__name__}: {error}"})
        raise


def _finalize_candidate_report(plan, source, completed, output, *, status="completed", input_admission=None, persist=True,
                               recovery_binding=None):
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
    by_key = {DecisionKey(row["participant_key"], entry["decision_id"]):
        row.get("requested_evidence_levels", {}).get(entry["decision_id"], row["evidence_level"])
        for row in completed for entry in row["requested"]}
    if len(vocabularies) != 1 or len(levels) != 1 and not all("requested_evidence_levels" in row for row in completed):
        raise ValueError("candidate study mixes different physical vocabularies or evidence levels")
    level_arguments = ({"requested_evidence_levels": by_key} if any("requested_evidence_levels" in row for row in completed)
                       else {"evidence_level": next(iter(levels))})
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
                    seeds=seeds, candidate_vocabulary=next(iter(vocabularies)), **level_arguments,
                    requested_coverage=coverage)
    if plan.get("prefix_selection") is not None:
        selected_output = plan["prefix_selection"]["output_name"]
        for budget in budgets:
            scores = {seed: [CandidateEvidence.from_record(cell["evidence"]) for row in completed
                for cell in row["predictions"] if cell["mixture"] == selected_output and cell["seed"] == seed
                and cell["K"] == budget and cell["calibration_strategy"] == "nested_selection"] for seed in seeds}
            metrics[f"{selected_output}@nested_selection@K{budget}"] = evaluate_candidate_evidence(scores,
                decision_plan=decision_plan, seeds=seeds, candidate_vocabulary=next(iter(vocabularies)),
                **level_arguments, requested_coverage=coverage)
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
    if plan.get("prefix_selection") is not None:
        for budget in budgets:
            name = f"{plan['prefix_selection']['output_name']}@nested_selection@K{budget}"
            baseline = f"{plan['baseline_mixture']}@fixed@K{budget}"
            contrasts[name] = (compare_candidate_probability_metrics(metrics[baseline], metrics[name]).record()
                if baseline in metrics else {"status": "not_applicable", "reason": "fixed_baseline_program_not_declared"})
    report = {"schema": "n2p3_candidate_ensemble_report/1", "status": status,
              "evidence_scope": plan["evidence_scope"], "randomness_scope": "paired_source_target_pipeline" if independent else "target_only_fixed_source",
              "comparison_role": comparison_role, "ar_panel": ar_panel,
              "decision_plan": decision_plan.record(), "source_digest": semantic_sha256(source),
              "participants": completed, "metrics": {name: value.record() for name, value in metrics.items()},
              "contrasts": contrasts}
    if plan.get("prefix_selection") is not None:
        report["prefix_selection"] = plan["prefix_selection"]
    if independent:
        report.update(evaluation_design="independent_prefix", seed_pairs=[list(pair) for pair in pairs],
            calibration_strategies=strategies,
            requested_budgets=plan["budgets"], actual_budgets=budgets,
            source_seed_catalog=normalized["source_seed_catalog"],
            source_replication="not_applicable_no_source_NN" if not normalized.get("source_seed_catalog", normalized["source_checkpoints"]) else plan["evidence_scope"])
        if normalized.get("joint_source_catalog"):
            report.update(joint_source_catalog=normalized["joint_source_catalog"],
                joint_source_randomness="deterministic_original_source_fit; actual source seed is null",
                source_replication="declared_NN_seed_producers_and_deterministic_joint_sources"
                    if normalized["source_seed_catalog"] else "deterministic_joint_sources")
    if prior_panel is not None:
        report["prior_panel"] = prior_panel
        report["prior_shape_contrasts"] = [{**row,
            "comparison": compare_candidate_probability_metrics(metrics[row["baseline"]], metrics[row["alternative"]]).record()}
            for row in prior_panel["shape_contrasts"]]
    if dictionary_panel is not None:
        report["dictionary_panel"] = dictionary_panel
    if recovery_binding is not None:
        report.update(prefix_recovery=recovery_binding, requested_records=plan["records"],
            decision_denominator_complete={row["record_index"] for row in completed} == set(range(len(plan["records"]))),
            participant_execution={str(row["record_index"]): row["execution_binding"] for row in completed})
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
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
