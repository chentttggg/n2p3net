"""A new stationary-law program over an unchanged frozen P1 fit/input panel.

Original strict loaders export the original response and noise primitives in
isolated processes. This owner recompiles only joint noise roots, then uses the
existing label-free scorer and requested-probability evaluator.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.audit_channel_noise_mechanism import (
    BoundInput,
    paired_input_digests,
    public_calibration_authority,
)
from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research import channel_noise_mechanism as noise_owner
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.contracts import DecisionPlanContract, semantic_sha256
from research.local_program_control import atomic_record, score_checkpoints
from transfer.ensemble_views import PredictionDecisionView
from transfer.noise_model import array_digest
from transfer.outcomes import CandidateCoverage, DecisionKey
from transfer.temporal_noise import TemporalNoise

PROGRAM = "frozen_P1_primitives_correct_stationary_law_paired/1"
ARMS = ("shared8", "private8")
EXPECTED_SCORER = "452b18bbe5ce3601f123c10638e2a5fdf6bfa351c0c5b038c240f09d2469f8b2"
EXPECTED_COMPILER = "621badbd538bd724d24f91129f0c226db1c7001db90eeb1f013005dd81710c5f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError("bound source/producer/input changed: "+path)


def _receipt(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def load_plan(plan_file):
    """Original native roster, current source and plan seal are one authority."""
    config = read(plan_file)
    seal = read(Path(plan_file).with_name("binding.json"))
    if seal["plan_file_sha256"] != sha(plan_file) or config["program"] != PROGRAM or config["seed"] != 20260918:
        raise ValueError("reviewed stationary program plan changed")
    verify(config["original_files"])
    verify(config["original_capsule_files"])
    verify(config["source_files"])
    native = read(config["original_plan"]["path"])
    if (sha(config["original_plan"]["path"]) != config["original_plan"]["sha256"]
            or config["decision_plan"] != native["decision_plan"]
            or list(config["participants"]) != [row["subject"] for row in native["participants"]]
            or len(config["participants"]) != 8):
        raise ValueError("stationary program dropped or changed original physical requested scope")
    expected = [{"subject": row["subject"], "arm": arm, "participant_key": row["participant_key"],
                 "decision_id": row["views"][5]["decision_id"]}
                for row in native["participants"] for arm in ARMS]
    if config["requested_cells"] != expected:
        raise ValueError("stationary program changed the complete declared cell roster")
    for item in native["participants"]:
        if config["participants"][item["subject"]]["native"] != item:
            raise ValueError("stationary program changed actual C/held input authority")
    return config


def prepare(original, capsule, baseline, source_root, output):
    """Only source/scope preparation; no strict model load, fit, compile or score."""
    original, capsule, source_root, output = map(lambda path: Path(path).resolve(), (original, capsule, source_root, output))
    native = read(original/"scientific_plan.json")
    expected = read(original/"scientific_plan.binding.json")
    if (sha(original/"scientific_plan.json") != expected["plan_file_sha256"]
            or native["source_capsule_sha256"] != sha(capsule/"source_manifest.json")
            or native["seed"] != 20260918 or len(native["requested_cells"]) != 16
            or [row["subject"] for row in native["participants"]] != [f"A{i:02d}" for i in range(1, 9)]):
        raise ValueError("stationary program requires the unchanged original eight-person P1 plan")
    if (sha(source_root/"src/research/channel_noise_mechanism.py") != EXPECTED_SCORER
            or sha(source_root/"src/research/stationary_noise_artifact.py") != EXPECTED_COMPILER):
        raise ValueError("new producer must use the reviewed actual scorer/compiler bytes")
    old_result = read(baseline)
    if (old_result["status"] != "completed" or old_result["decision_plan"] != native["decision_plan"]
            or len(old_result["cells"]) != 16):
        raise ValueError("old compiled-W reference does not contain the complete paired panel")
    capsule_files = {str(capsule/path): value["sha256"]
                     for path, value in read(capsule/"source_manifest.json")["files"].items()}
    verify(capsule_files)
    participants, requests, all_views = {}, [], []
    original_files = {str(original/"scientific_plan.json"): sha(original/"scientific_plan.json"),
                      str(Path(baseline).resolve()): sha(baseline)}
    for item in native["participants"]:
        subject, directory = item["subject"], original/"subjects"/item["subject"]
        receipt = read(directory/"receipt.json")
        if (receipt["status"] != "completed" or receipt["scientific_plan_digest"] != native["scientific_plan_digest"]
                or item["fit_labels"] != 600 or item["held_scorer_only_labels"] != 120):
            raise ValueError("original participant fit/held authority changed")
        files = {str(directory/path): value for path, value in receipt["files"].items()}
        files[str(directory/"receipt.json")] = sha(directory/"receipt.json")
        files[item["source"]["path"]] = item["source"]["sha256"]
        verify(files)
        manifest = directory/"input/input.json"
        payload, inputs = read(manifest), BoundInput(manifest)
        entry = payload["participants"][0]
        authority = public_calibration_authority(payload["source_records"], inputs)
        if (len(entry["training"]) != 5 or len(entry["held"]) != 1 or payload["program"]["seed"] != 20260918
                or entry["held"][0]["view"] != item["views"][5]):
            raise ValueError("original requested C/held roster changed")
        for role in ("training", "held"):
            for row in entry[role]:
                samples = inputs.arrays(row["arrays"], "samples")[0]
                view = PredictionDecisionView.from_record(row["view"], samples=samples)
                actual = authority[view.record_id]
                if actual.view.record() != view.record():
                    raise ValueError("saved views differ from original MAT authority")
                if role == "training" and (row["target_candidate"] != actual.target_candidate
                        or not np.array_equal(inputs.arrays(row["arrays"], "labels")[0], actual.labels)):
                    raise ValueError("C labels differ from original source")
                all_views.append(view)
        held = entry["held"][0]
        samples, designs = inputs.arrays(held["arrays"], "samples", "candidate_designs")
        noises = {arm: _receipt(directory/"audit"/f"noise_000_{arm}.json") for arm in ARMS}
        participants[subject] = {"original_files": files, "input_manifest": _receipt(manifest),
            "geometry": entry["geometry"], "response_model": entry["response"]["source_model"],
            "response_record": read(directory/"audit/report.json")["responses"][0],
            "noise_files": noises, "held": held, "native": item,
            "samples_sha256": array_digest(samples), "candidate_designs_sha256": array_digest(designs)}
        original_files.update(files)
        requests.extend({"subject": subject, "arm": arm, "participant_key": item["participant_key"],
                         "decision_id": held["view"]["decision_id"]} for arm in ARMS)
        inputs.verify_unchanged()
    plan = DecisionPlanContract.from_record(native["decision_plan"])
    if paired_input_digests(all_views) != (plan.target_cache_sha256, plan.target_identity_digest):
        raise ValueError("complete paired input digests differ from original requested plan")
    if output.exists():
        raise FileExistsError("prepare requires a new independent program directory")
    output.mkdir(parents=True)
    freeze = output/"source"
    shutil.copytree(source_root/"src", freeze/"src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (freeze/"experiments").mkdir()
    for name in ("run_stationary_channel_noise_panel.py", "export_frozen_p1_state.py", "audit_channel_noise_mechanism.py"):
        shutil.copyfile(source_root/"experiments"/name, freeze/"experiments"/name)
    source_files = {str(path): sha(path) for path in sorted(freeze.rglob("*.py"))}
    loader = {name: _receipt(capsule/path) for name, path in {
        "channel": "src/research/channel_noise_mechanism.py", "continuous": "src/transfer/continuous_decision.py",
        "temporal": "src/transfer/temporal_noise.py", "noise_contract": "src/transfer/noise_model.py"}.items()}
    config = {"schema": "p1_correct_stationary_program/1", "program": PROGRAM, "seed": native["seed"],
        "scope": "original_finite_C_held_development_new_stationary_law_not_product_or_adult",
        "original_plan": _receipt(original/"scientific_plan.json"), "original_capsule": str(capsule),
        "original_capsule_files": capsule_files, "original_loader_sources": loader,
        "old_compiled_W_result": _receipt(baseline), "original_files": original_files,
        "source_root": str(freeze), "source_files": source_files, "decision_plan": plan.record(),
        "requested_cells": requests, "participants": participants,
        "resources": {"max_workers": 2, "native_threads": 1, "short_original_load_threads": 2,
            "launch_cpu_threshold_percent": 60., "worker_cpu_threshold_percent": 72.,
            "resume_cpu_threshold_percent": 60., "required_quiet_samples": 3, "growth_safety_factor": 1.15,
            "memory_reserve_bytes": 768*1024**2, "worker_rss_estimate_bytes": 1800*1024**2},
        "fits_allowed": 0, "response_fits_allowed": 0, "scoring_status": "not_started"}
    atomic_record(output/"plan.json", config)
    atomic_record(output/"binding.json", {"schema": "p1_correct_stationary_prepared_binding/1",
        "plan_file_sha256": sha(output/"plan.json"), "scorer_sha256": EXPECTED_SCORER,
        "compiler_sha256": EXPECTED_COMPILER, "source_files": source_files,
        "requested_participants": 8, "requested_cells": 16, "strict_exports": 0,
        "root_compiles": 0, "fits": 0, "scores": 0})
    return config


def _export_state(config, subject, output):
    descriptor, output = config["participants"][subject], Path(output)
    exported = read(output/"export.json")
    script = Path(config["source_root"])/"experiments/export_frozen_p1_state.py"
    expected = {"subject": subject, "plan_file_sha256": sha(output.parent.parent/"plan.json"),
        "export_script_sha256": sha(script), "loader_sources": config["original_loader_sources"],
        "response": descriptor["response_record"], "response_arrays_sha256": sha(output/"response.npz"),
        "fit_calls": [], "score_calls": 0, "original_input_digest": descriptor["input_manifest"]["sha256"]}
    if any(exported.get(key) != value for key, value in expected.items()):
        raise ValueError("strict original export differs from current program authority")
    with np.load(output/"response.npz", allow_pickle=False) as archive:
        response = noise_owner.FrozenERPResponse(archive["mean"], archive["coefficient_factor"],
            noise_owner.NoiseGeometry(**descriptor["geometry"]), exported["response"]["source_model_sha256"],
            exported["response"]["response_id"], exported["response"]["provenance"])
    if json.loads(json.dumps(response.record())) != exported["response"]:
        raise ValueError("actual physical response factor was not retained")
    for arm, original in descriptor["noise_files"].items():
        value, native = exported["noises"][arm], read(original["path"])
        if value != {"state": native["state"], "state_sha256": native["state_sha256"], "original_file": original}:
            raise ValueError("exported noise primitives differ from original strict state")
    return exported, response


def _recompile_provenance(value):
    return value["state"]["provenance"] | {"stationary_recompilation": {
        "program": PROGRAM, "original_noise_file": value["original_file"],
        "original_noise_state_sha256": value["state_sha256"],
        "original_compiled_root_sha256": array_digest(np.asarray(value["state"]["compiled_prefix_root"])),
        "parameter_fit_repeated": False, "original_fit_provenance_retained": True}}


def compile_noise(exported, arm):
    value, state = exported["noises"][arm], exported["noises"][arm]["state"]
    if (state["conditioning_samples"] != 32 or state["family"] != {"shared8": "shared", "private8": "private"}[arm]
            or any(len(row) != 8 for row in state["reflections"])):
        raise ValueError("stationary recompile changes declared AR8/prefix32/family")
    originals = tuple(TemporalNoise.from_record(record) for record in state["compiled_temporal_records"])
    temporal = originals*len(state["geometry"]["channels"]) if state["family"] == "shared" else originals
    provenance = _recompile_provenance(value)
    noise = noise_owner.FittedChannelNoise(temporal, state["innovation_covariance"],
        noise_owner.NoiseGeometry(**state["geometry"]), state["conditioning_samples"], state["family"], provenance)
    actual = json.loads(json.dumps(noise.record()))
    if (actual["coefficients"] != state["coefficients"] or actual["reflections"] != state["reflections"]
            or actual["innovation_covariance"] != state["innovation_covariance"]
            or actual["compiled_temporal_records"] != state["compiled_temporal_records"]):
        raise ValueError("new noise compile changed fitted primitives")
    return noise


def worker(plan_file, subject, control):
    config, directory = load_plan(plan_file), Path(plan_file).parent
    verify(config["source_files"])
    verify(config["participants"][subject]["original_files"])
    descriptor = config["participants"][subject]
    def forbidden_fit(*args, **kwargs):
        raise RuntimeError("correct stationary program forbids any response or noise fit")
    noise_owner.FittedChannelNoise.fit = classmethod(forbidden_fit)
    noise_owner.FittedChannelNoise.fit_response_given_frozen_noise = forbidden_fit
    TemporalNoise.fit = classmethod(forbidden_fit)
    export_dir = directory/"exports"/subject
    if export_dir.exists() and not (export_dir/"export.json").exists():
        # Only this program's deterministic subject path. Preserve interrupted
        # bytes; the new original-loader attempt has its own output directory.
        export_dir.rename(export_dir.with_name(f"{subject}.unsealed.{time.time_ns()}"))
    if not export_dir.exists():
        control.checkpoint("original_strict_export")
        subprocess.run([sys.executable, str(Path(config["source_root"])/"experiments/export_frozen_p1_state.py"),
            "--plan", str(plan_file), "--subject", subject, "--output", str(export_dir)], check=True,
            cwd=config["original_capsule"])
    exported, response = _export_state(config, subject, export_dir)
    inputs = BoundInput(descriptor["input_manifest"]["path"])
    samples, designs = inputs.arrays(descriptor["held"]["arrays"], "samples", "candidate_designs")
    view = descriptor["held"]["view"]
    with threadpool_limits(limits=1):
        for arm in ARMS:
            cell = directory/"cells"/f"{subject}_{arm}"
            if (cell/"receipt.json").exists():
                admit_cell(config, subject, arm, directory)
                continue
            if cell.exists() and any(cell.iterdir()):
                cell.rename(cell.with_name(f"{subject}_{arm}.unsealed.{time.time_ns()}"))
            cell.mkdir(parents=True, exist_ok=True)
            completed, start = [], time.perf_counter()
            identity = {"schema": "p1_correct_stationary_cell/1", "subject": subject, "arm": arm,
                "plan_file_sha256": sha(plan_file), "original_noise_file": descriptor["noise_files"][arm],
                "original_response": descriptor["response_record"], "original_input": descriptor["input_manifest"],
                "export_sha256": sha(export_dir/"export.json"), "response_arrays_sha256": sha(export_dir/"response.npz"),
                "scorer_sha256": EXPECTED_SCORER, "compiler_sha256": EXPECTED_COMPILER,
                "fit_calls": 0, "response_fit_calls": 0,
                "runtime": {"python_executable": sys.executable, "python_version": sys.version,
                    "native_threadpools": threadpool_info()}}
            control.checkpoint("stationary_root_compile")
            try:
                noise = compile_noise(exported, arm)
                noise.save(cell/"noise.json")
                loaded = noise_owner.FittedChannelNoise.load(cell/"noise.json")
                if loaded.record() != noise.record():
                    raise ValueError("new saved stationary model did not retain its exact numerical state")
                identity.update(new_model_sha256=sha(cell/"noise.json"), new_model_state_sha256=noise._integrity)
                def advance(resolution, completed=completed, cell=cell):
                    completed.append({"candidate_index": len(completed), "resolution": resolution})
                    atomic_record(cell/"candidate_progress.json", completed)
                with score_checkpoints(noise_owner, control, advance):
                    score = loaded.score(samples, designs, response,
                        candidate_ids=tuple(descriptor["native"]["vocabulary"]), geometry=loaded.geometry,
                        record_id=view["record_id"], recording_id=view["source_sha256"], sample_start=view["raw_scope"][0])
                if score["scoring_implementation_sha256"] != EXPECTED_SCORER:
                    raise ValueError("actual scorer source changed")
                atomic_record(cell/"score.json", score)
                identity.update(status="scored", score_sha256=sha(cell/"score.json"))
            except (noise_owner.UnidentifiableFitError, np.linalg.LinAlgError) as error:
                identity.update(status="not_available", numerical_failure={"type": type(error).__name__, "detail": str(error)})
            verify(config["source_files"])
            verify(config["original_capsule_files"])
            verify(descriptor["original_files"])
            inputs.verify_unchanged()
            identity.update(completed_candidates=len(completed), elapsed_seconds=time.perf_counter()-start)
            roster = [{"index": index, "candidate": candidate,
                "status": "scored" if identity["status"] == "scored" or index < len(completed)
                else "numerically_unavailable" if index == len(completed) else "not_evaluated_after_numeric_failure"}
                for index, candidate in enumerate(descriptor["native"]["vocabulary"])]
            atomic_record(cell/"candidate_roster.json", roster)
            identity["candidate_roster_sha256"] = sha(cell/"candidate_roster.json")
            if (cell/"candidate_progress.json").exists():
                identity["candidate_progress_sha256"] = sha(cell/"candidate_progress.json")
            atomic_record(cell/"receipt.json", identity)
            admit_cell(config, subject, arm, directory)


def admit_cell(config, subject, arm, directory):
    cell, descriptor = Path(directory)/"cells"/f"{subject}_{arm}", config["participants"][subject]
    receipt = read(cell/"receipt.json")
    expected = {"schema": "p1_correct_stationary_cell/1", "subject": subject, "arm": arm,
        "plan_file_sha256": sha(Path(directory)/"plan.json"), "original_noise_file": descriptor["noise_files"][arm],
        "original_response": descriptor["response_record"], "original_input": descriptor["input_manifest"],
        "scorer_sha256": EXPECTED_SCORER, "compiler_sha256": EXPECTED_COMPILER, "fit_calls": 0, "response_fit_calls": 0}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("cell receipt differs from the original exact request and current producer tuple")
    export = Path(directory)/"exports"/subject
    exported, response = _export_state(config, subject, export)
    if receipt["export_sha256"] != sha(export/"export.json") or receipt["response_arrays_sha256"] != sha(export/"response.npz"):
        raise ValueError("cell original export changed")
    roster = read(cell/"candidate_roster.json")
    if (receipt["candidate_roster_sha256"] != sha(cell/"candidate_roster.json")
            or [row["candidate"] for row in roster] != descriptor["native"]["vocabulary"]
            or [row["index"] for row in roster] != list(range(len(roster)))):
        raise ValueError("candidate ledger lost the original complete vocabulary")
    if "candidate_progress_sha256" in receipt:
        progress = read(cell/"candidate_progress.json")
        if (receipt["candidate_progress_sha256"] != sha(cell/"candidate_progress.json")
                or [row["candidate_index"] for row in progress] != list(range(receipt["completed_candidates"]))):
            raise ValueError("candidate execution progress does not belong to its receipt")
    if "new_model_sha256" in receipt:
        if receipt["new_model_sha256"] != sha(cell/"noise.json"):
            raise ValueError("new stationary model bytes changed")
        # Descriptor/parameter admission is checked without fitting or scoring.
        noise = noise_owner.FittedChannelNoise.load(cell/"noise.json")
        state = exported["noises"][arm]["state"]
        actual = json.loads(json.dumps(noise.record()))
        if (noise._integrity != receipt["new_model_state_sha256"]
                or actual["compiled_temporal_records"] != state["compiled_temporal_records"]
                or actual["innovation_covariance"] != state["innovation_covariance"]
                or noise.geometry.record() != state["geometry"] or noise.family != state["family"]
                or noise.conditioning_samples != state["conditioning_samples"]
                or semantic_sha256(actual["provenance"]) != semantic_sha256(
                    _recompile_provenance(exported["noises"][arm]))):
            raise ValueError("new model differs from the declared original fitted primitives")
    if receipt["status"] == "scored":
        score = read(cell/"score.json")
        view = descriptor["held"]["view"]
        expected_score = {"noise_sha256": receipt["new_model_state_sha256"], "response": json.loads(json.dumps(response.record())),
            "samples_sha256": descriptor["samples_sha256"], "candidate_designs_sha256": descriptor["candidate_designs_sha256"],
            "scoring_implementation_sha256": EXPECTED_SCORER, "candidate_ids": descriptor["native"]["vocabulary"],
            "read_support": view["raw_scope"], "conditioning_support": [view["raw_scope"][0], view["raw_scope"][0]+32],
            "score_support": [view["raw_scope"][0]+32, view["raw_scope"][1]], "truth_used": False, "parameters_updated": False}
        if (sha(cell/"score.json") != receipt["score_sha256"]
                or receipt["completed_candidates"] != len(descriptor["native"]["vocabulary"])
                or any(score.get(key) != value for key, value in expected_score.items())):
            raise ValueError("score no longer belongs to its current model and unchanged inputs")
    elif receipt["status"] != "not_available" or receipt["numerical_failure"]["type"] not in {
            "UnidentifiableFitError", "DiffuseProjectionResolutionError", "LinAlgError"}:
        raise ValueError("execution interruption cannot become numerical evidence")
    return receipt


def reduce_program(plan_file):
    config, directory = load_plan(plan_file), Path(plan_file).parent
    verify(config["original_files"])
    verify(config["source_files"])
    plan = DecisionPlanContract.from_record(config["decision_plan"])
    rows, coverage, cells = {arm: [] for arm in ARMS}, {}, []
    for subject, descriptor in config["participants"].items():
        item, view = descriptor["native"], descriptor["held"]["view"]
        coverage[DecisionKey(item["participant_key"], view["decision_id"])] = CandidateCoverage.from_mappings(
            dict(zip(item["vocabulary"], item["expected_counts"], strict=True)),
            dict(zip(item["vocabulary"], item["observed_counts"], strict=True)))
        for arm in ARMS:
            receipt = admit_cell(config, subject, arm, directory)
            score = read(directory/"cells"/f"{subject}_{arm}"/"score.json") if receipt["status"] == "scored" else None
            identity = CandidateExpertIdentity(arm, view["raw_input_digest"], semantic_sha256(receipt), None,
                semantic_sha256({"original_noise": receipt["original_noise_file"], "response": receipt["original_response"]}),
                "correct_stationary_frozen_response_conditional_tail/1", False)
            evidence = CandidateEvidence(candidate_vocabulary=tuple(item["vocabulary"]),
                participant_key=item["participant_key"], decision_id=view["decision_id"],
                physical_support_digest=view["physical_support_digest"], expert_identities=(identity,),
                evidence_level=item["evidence_level"], expected_event_counts=tuple(item["expected_counts"]),
                observed_event_counts=tuple(item["observed_counts"]),
                log_probability=None if score is None else score["log_probability"],
                failure_kind="not_available" if score is None else None,
                failure_reason=receipt["numerical_failure"]["detail"] if score is None else None)
            rows[arm].append(evidence)
            cells.append({"subject": subject, "arm": arm, "receipt": _receipt(directory/"cells"/f"{subject}_{arm}"/"receipt.json"),
                          "evidence": evidence.record()})
    sample = next(iter(config["participants"].values()))["native"]
    def evaluate(values):
        return evaluate_candidate_evidence({config["seed"]: values}, decision_plan=plan, seeds=(config["seed"],),
            candidate_vocabulary=tuple(sample["vocabulary"]), evidence_level=sample["evidence_level"], requested_coverage=coverage)
    metrics = {arm: evaluate(values) for arm, values in rows.items()}
    old = read(config["old_compiled_W_result"]["path"])
    baseline = {arm: evaluate([CandidateEvidence.from_record(row["evidence"]) for row in old["cells"] if row["arm"] == arm]) for arm in ARMS}
    result = {"schema": "p1_correct_stationary_program_result/1", "status": "completed", "program": PROGRAM,
        "plan_file_sha256": sha(plan_file), "decision_plan": plan.record(), "requested_cells": 16,
        "cells": cells, "metrics": {arm: value.record() for arm, value in metrics.items()},
        "shared_private_comparison": compare_candidate_probability_metrics(metrics["shared8"], metrics["private8"]).record(),
        "law_change_comparisons": {arm: compare_candidate_probability_metrics(baseline[arm], metrics[arm]).record() for arm in ARMS},
        "fit_calls": 0, "response_fit_calls": 0, "original_compiled_W_resigned": False,
        "scope": config["scope"], "old_reference": config["old_compiled_W_result"]}
    atomic_record(directory/"result.json", result)
    return result
