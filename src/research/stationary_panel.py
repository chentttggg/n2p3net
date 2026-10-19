"""A new stationary-law program over an unchanged frozen P1 fit/input panel.

Original strict loaders export the original response and noise primitives in
isolated processes. This owner recompiles only joint noise roots, then uses the
existing label-free scorer and requested-probability evaluator.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from models.candidate_evidence import CandidateEvidence, CandidateExpertIdentity
from research import channel_noise_mechanism as noise_owner
from research.candidate_metrics import (
    compare_candidate_probability_metrics,
    evaluate_candidate_evidence,
)
from research.channel_noise_audit_inputs import (
    BoundInput,
    paired_input_digests,
    public_calibration_authority,
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
WINDOWS_EXPORT_PLAN_SHA256 = "0569777f5f99eaac753b0fb015ab0fcec761a2b4e9dcf14071044338f1f0485e"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path, *, input_owner=None):
    return json.loads(_input_path(path, input_owner).read_bytes())


def _input_path(path, input_owner):
    """Resolve IO only; literal producer descriptors stay unchanged in records."""
    return Path(path) if input_owner is None else Path(input_owner.resolve(path))


def verify(files, *, input_owner=None):
    for path, expected in files.items():
        if sha(_input_path(path, input_owner)) != expected:
            raise ValueError("bound source/producer/input changed: "+path)


def _receipt(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def original_export_plan(config, input_owner):
    """The real Windows export plan is independent of the new score-plan writer."""
    descriptor = config["original_export_plan"]
    path = _input_path(descriptor["path"], input_owner)
    if descriptor["sha256"] != WINDOWS_EXPORT_PLAN_SHA256 or sha(path) != WINDOWS_EXPORT_PLAN_SHA256:
        raise ValueError("stationary cloud must consume the immutable original Windows export plan")
    producer = read(path)
    for key in ("program", "seed", "scope", "original_plan", "original_capsule", "original_capsule_files",
                "original_loader_sources", "original_files", "old_compiled_W_result", "decision_plan",
                "requested_cells", "participants", "fits_allowed", "response_fits_allowed"):
        if config[key] != producer[key]:
            raise ValueError("new score plan changed the original Windows export-plan authority: "+key)
    return producer


def _held_authority(config, subject, input_owner):
    """Tie the complete held row and numeric descriptors to immutable input bytes."""
    descriptor = config["participants"][subject]
    manifest = descriptor["input_manifest"]
    path = _input_path(manifest["path"], input_owner)
    if (descriptor["original_files"].get(manifest["path"]) != manifest["sha256"]
            or sha(path) != manifest["sha256"]):
        raise ValueError("original participant input manifest differs from its frozen producer")
    payload = read(path)
    if len(payload["participants"]) != 1:
        raise ValueError("original input manifest lost its unique participant")
    entry = payload["participants"][0]
    if (len(entry["held"]) != 1 or len(entry["training"]) != 5
            or payload["program"]["seed"] != config["seed"]
            or descriptor["held"] != entry["held"][0]
            or entry["held"][0]["view"] != descriptor["native"]["views"][5]
            or entry["geometry"] != descriptor["geometry"]
            or entry["response"]["source_model"] != descriptor["response_model"]):
        raise ValueError("held row/physical view/arrays differ from the immutable original input manifest")
    return entry["held"][0]


def load_plan(plan_file, *, input_owner=None):
    """Original native roster, current source and plan seal are one authority."""
    config = read(plan_file)
    seal = read(Path(plan_file).with_name("binding.json"))
    if seal["plan_file_sha256"] != sha(plan_file) or config["program"] != PROGRAM or config["seed"] != 20260918:
        raise ValueError("reviewed stationary program plan changed")
    verify(config["original_files"], input_owner=input_owner)
    verify(config["original_capsule_files"], input_owner=input_owner)
    verify(config["source_files"], input_owner=input_owner)
    native = read(config["original_plan"]["path"], input_owner=input_owner)
    if (sha(_input_path(config["original_plan"]["path"], input_owner)) != config["original_plan"]["sha256"]
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
        _held_authority(config, item["subject"], input_owner)
    if input_owner is not None or "original_export_plan" in config or "cloud_execution" in config:
        original_export_plan(config, input_owner)
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
    for name in ("run_stationary_channel_noise_panel.py", "run_stationary_cloud_panel.py",
                 "export_frozen_p1_state.py", "audit_channel_noise_mechanism.py"):
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


def _export_state(config, subject, output, *, input_owner=None):
    descriptor, output = config["participants"][subject], Path(output)
    exported = read(output/"export.json")
    script = Path(config["source_root"])/"experiments/export_frozen_p1_state.py"
    expected = {"subject": subject, "plan_file_sha256": sha(output.parent.parent/"plan.json"),
        "export_script_sha256": sha(script), "loader_sources": config["original_loader_sources"],
        "response": descriptor["response_record"], "response_arrays_sha256": sha(output/"response.npz"),
        "fit_calls": [], "score_calls": 0, "original_input_digest": descriptor["input_manifest"]["sha256"]}
    if input_owner is not None:
        # A sealed Windows export belongs to its actual original-loader plan
        # and source. The new cloud score plan may consume it, never re-sign it.
        producer = original_export_plan(config, input_owner)
        authority = input_owner.export_authority(subject)
        if (sha(output/"export.json") != authority["export_json_sha256"]
                or sha(output/"response.npz") != authority["response_arrays_sha256"]
                or authority["loader_sources"] != producer["original_loader_sources"]
                or authority["plan_file_sha256"] != WINDOWS_EXPORT_PLAN_SHA256
                or producer["source_files"].get(str(PureWindowsPath(producer["source_root"])/
                    "experiments/export_frozen_p1_state.py")) != authority["export_script_sha256"]):
            raise ValueError("cross-host original export differs from its sealed Windows authority")
        expected.update({key: authority[key] for key in (
            "plan_file_sha256", "export_script_sha256", "loader_sources")})
    if any(exported.get(key) != value for key, value in expected.items()):
        raise ValueError("strict original export differs from current program authority")
    with np.load(output/"response.npz", allow_pickle=False) as archive:
        response = noise_owner.FrozenERPResponse(archive["mean"], archive["coefficient_factor"],
            noise_owner.NoiseGeometry(**descriptor["geometry"]), exported["response"]["source_model_sha256"],
            exported["response"]["response_id"], exported["response"]["provenance"])
    if json.loads(json.dumps(response.record())) != exported["response"]:
        raise ValueError("actual physical response factor was not retained")
    for arm, original in descriptor["noise_files"].items():
        value, native = exported["noises"][arm], read(original["path"], input_owner=input_owner)
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


def worker_native_threads(config):
    """Consume the declared plan value; there is no silent one-thread fallback."""
    native = config["resources"]["native_threads"]
    if type(native) is not int or native < 1:
        raise ValueError("stationary worker requires a positive declared native thread budget")
    return native


def cloud_input_owner(spec):
    """Cloud preparation owns exact-SHA relocation and sealed Windows exports."""
    module, factory = spec["factory"].split(":", 1)
    imported = importlib.import_module(module)
    if sha(imported.__file__) != spec["source_sha256"]:
        raise ValueError("cloud input-owner source differs from its declared freeze")
    return getattr(imported, factory)(spec["arguments"])


class LiveControl:
    """Mark the current admitted cloud actor phase; no second resource scheduler."""
    def __init__(self, task_id, progress):
        self.task_id, self.progress = task_id, progress

    def checkpoint(self, phase):
        import os

        from research.worker_diagnostics import worker_diagnostic_phase

        if os.environ.get("N2P3_CLOUD_TASK_ID") != self.task_id:
            raise RuntimeError("stationary checkpoint is outside its admitted cloud actor")
        worker_diagnostic_phase(phase, stationary_task_id=self.task_id)
        self.progress(phase)


def cloud_worker(arguments):
    """One subject owns its two sequential arms; no live EEG/model crosses IPC."""
    import os

    import torch

    if os.environ.get("N2P3_CLOUD_TASK_ID") != arguments["task_id"]:
        raise RuntimeError("stationary worker task identity differs from its actual cloud actor")
    inputs = cloud_input_owner(arguments["input_owner"])
    plan_file, subject = Path(arguments["plan_file"]), arguments["subject"]
    config = load_plan(plan_file, input_owner=inputs)
    if (worker_native_threads(config) != arguments["native_threads"]
            or torch.get_num_threads() != arguments["native_threads"]
            or torch.get_num_interop_threads() != arguments["interop_threads"]):
        raise ValueError("stationary actor differs from its actual native/inter-op reservation")
    def progress(phase):
        atomic_record(plan_file.parent/f"cloud_progress.{subject}.json", {
            "plan_file_sha256": sha(plan_file), "task_id": arguments["task_id"],
            "subject": subject, "phase": phase, "time": time.time(),
            "native_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()})
    worker(plan_file, subject, LiveControl(arguments["task_id"], progress), input_owner=inputs)
    return {"subject": subject, "plan_file_sha256": sha(plan_file), "arms": list(ARMS)}


def worker(plan_file, subject, control, *, input_owner=None):
    config, directory = load_plan(plan_file, input_owner=input_owner), Path(plan_file).parent
    native_threads = worker_native_threads(config)
    verify(config["source_files"], input_owner=input_owner)
    verify(config["participants"][subject]["original_files"], input_owner=input_owner)
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
        if input_owner is None:
            subprocess.run([sys.executable, str(Path(config["source_root"])/"experiments/export_frozen_p1_state.py"),
                "--plan", str(plan_file), "--subject", subject, "--output", str(export_dir)], check=True,
                cwd=config["original_capsule"])
        else:
            input_owner.strict_export(plan_file, subject, export_dir)
    exported, response = _export_state(config, subject, export_dir, input_owner=input_owner)
    # Plan JSON serializes paths as strings; BoundInput intentionally accepts
    # a Path because its root is the manifest's parent directory.
    inputs = (BoundInput(Path(descriptor["input_manifest"]["path"])) if input_owner is None
              else input_owner.bound_input(descriptor["input_manifest"]))
    samples, designs = inputs.arrays(descriptor["held"]["arrays"], "samples", "candidate_designs")
    held = _held_authority(config, subject, input_owner)
    actual_view = PredictionDecisionView.from_record(held["view"], samples=samples)
    if actual_view.record() != descriptor["native"]["views"][5]:
        raise ValueError("actual decoded held view differs from original native authority")
    if (array_digest(samples) != descriptor["samples_sha256"]
            or array_digest(designs) != descriptor["candidate_designs_sha256"]):
        raise ValueError("actual held arrays differ from the original frozen scientific input")
    view = descriptor["held"]["view"]
    with threadpool_limits(limits=native_threads):
        for arm in ARMS:
            cell = directory/"cells"/f"{subject}_{arm}"
            if (cell/"receipt.json").exists():
                admit_cell(config, subject, arm, directory, input_owner=input_owner)
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
                    "declared_native_threads": native_threads,
                    "native_threadpools": threadpool_info(),
                    "torch_native_threads": sys.modules["torch"].get_num_threads(),
                    "torch_interop_threads": sys.modules["torch"].get_num_interop_threads()}}
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
            verify(config["source_files"], input_owner=input_owner)
            verify(config["original_capsule_files"], input_owner=input_owner)
            verify(descriptor["original_files"], input_owner=input_owner)
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
            admit_cell(config, subject, arm, directory, input_owner=input_owner)


def admit_cell(config, subject, arm, directory, *, input_owner=None):
    cell, descriptor = Path(directory)/"cells"/f"{subject}_{arm}", config["participants"][subject]
    receipt = read(cell/"receipt.json")
    expected = {"schema": "p1_correct_stationary_cell/1", "subject": subject, "arm": arm,
        "plan_file_sha256": sha(Path(directory)/"plan.json"), "original_noise_file": descriptor["noise_files"][arm],
        "original_response": descriptor["response_record"], "original_input": descriptor["input_manifest"],
        "scorer_sha256": EXPECTED_SCORER, "compiler_sha256": EXPECTED_COMPILER, "fit_calls": 0, "response_fit_calls": 0}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("cell receipt differs from the original exact request and current producer tuple")
    export = Path(directory)/"exports"/subject
    exported, response = _export_state(config, subject, export, input_owner=input_owner)
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


def reduce_program(plan_file, *, input_owner=None):
    config, directory = load_plan(plan_file, input_owner=input_owner), Path(plan_file).parent
    verify(config["original_files"], input_owner=input_owner)
    verify(config["source_files"], input_owner=input_owner)
    plan = DecisionPlanContract.from_record(config["decision_plan"])
    rows, coverage, cells = {arm: [] for arm in ARMS}, {}, []
    for subject, descriptor in config["participants"].items():
        item, view = descriptor["native"], descriptor["held"]["view"]
        coverage[DecisionKey(item["participant_key"], view["decision_id"])] = CandidateCoverage.from_mappings(
            dict(zip(item["vocabulary"], item["expected_counts"], strict=True)),
            dict(zip(item["vocabulary"], item["observed_counts"], strict=True)))
        for arm in ARMS:
            receipt = admit_cell(config, subject, arm, directory, input_owner=input_owner)
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
    old = read(config["old_compiled_W_result"]["path"], input_owner=input_owner)
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
