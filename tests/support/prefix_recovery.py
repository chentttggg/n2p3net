"""Real same-run prefix packages and their bounded historical recovery facts."""
import hashlib
import io
import json
import tarfile
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

import experiments.run_candidate_ensemble_study as runner
from research.calibration_selection import CalibrationProgramCandidate
from research.cloud_execution import CloudExecutor, CloudResult
from research.contracts import semantic_sha256
from research.source_execution import HostMemoryAdmission, resolve_source_resources
from tests.support.checkpoints import write_checkpoint
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.calibration_candidates import waveform_lda_recipe
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.recipes import TargetRecipe


def resources(plan_path):
    plan=json.loads(plan_path.read_bytes())
    native = plan["cpu_threads_per_worker"]
    return resolve_source_resources({**plan,"cpu_budget":max(len(plan["records"])+1, native*len(plan["records"])+1),
        "source_workers":len(plan["records"]),"target_workers":len(plan["records"]),
        "source_worker_memory_gib":1.,"target_worker_memory_gib":1.},len(plan["records"]),"cpu")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def freeze_runtime(directory):
    """A real physical source freeze also works in an extracted, git-free test source."""
    directory.mkdir()
    paths = [*(runner.ROOT / "src").rglob("*.py"), *(runner.ROOT / "experiments").glob("*.py"),
             runner.ROOT / "pyproject.toml", runner.ROOT / "uv.lock"]
    files = {path.relative_to(runner.ROOT).as_posix(): path.read_bytes() for path in paths}
    archive = directory / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    return write_json(directory / "source.manifest.json", {"schema": "n2p3_source_freeze/1",
        "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_commit": "0" * 40, "working_tree_snapshot": True, "source_commit_role": "base_commit_only",
        "byte_size": archive.stat().st_size, "member_count": len(files),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}})


def completed_prefix_producer(root, *, neural=False, native_threads=1):
    with cpu_thread_budget(1):
        _, fields, path, preprocessing, experts, views = independent_prefix_case(root)
    second = root / "A02.mat"
    fields = {**fields, "X": np.random.default_rng(40).normal(0, 4, fields["X"].shape).astype(np.float32)}
    savemat(second, {"data": fields})
    raw = experts[0]
    ensemble = EnsembleProgram.equal_weights("raw", (raw.expert_id,))
    candidate = CalibrationProgramCandidate("raw_fixed", (raw,), ensemble)
    source_manifest = freeze_runtime(root / "source")
    plan = {"schema": runner.SCHEMA, "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[20260918, 20260918]],
        "records": [{"path": str(path), "sha256": views.ledger["source_sha256"]},
                    {"path": str(second), "sha256": hashlib.sha256(second.read_bytes()).hexdigest()}],
        "budgets": [0, 200], "experts": [raw.record()], "mixtures": [ensemble.record()],
        "baseline_mixture": "raw", "authorization_preprocessing": preprocessing.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2, "device": "cpu",
        "workers": 1, "cpu_threads_per_worker": native_threads, "source_checkpoints": {}, "source_seed_catalog": [],
        "source_snapshot_manifest": source_manifest["path"], "calibration_strategies": ["fixed", "grouped_temperature"],
        "prefix_selection": {"schema": "n2p3_prefix_selection_plan/1", "output_name": "selected",
            "baseline_name": candidate.name, "minimum_selection_rows": 3, "candidates": [candidate.record()]}}
    if neural:
        profile = experts[1].profile
        checkpoint = write_checkpoint(root / "source.pt", views.calibration[200].trial_datasets[profile.digest],
                                      source_snapshot_sha256="b" * 64)
        payload = torch.load(checkpoint, weights_only=True)
        payload["training_contract"]["seed"] = 20260918
        payload["training_contract"]["optimizer"]["config"] = {
            "pos_weight": payload["source_calibration"]["pos_weight"]}
        payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
        torch.save(payload, checkpoint)
        recipe = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=1,
                              batch_size=128, bn_buffers="calibration", dropout="off")
        nn = CalibrationExpertProgram("NN", "trial", recipe, profile)
        scratch = CalibrationExpertProgram("scratch", "trial",
            replace(recipe, name="scratch", initialization="scratch", input_statistics="target_prefix"), profile)
        waveform = CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(), profile)
        nn_pool = EnsembleProgram.equal_weights("NN", (nn.expert_id,))
        nn_candidate = CalibrationProgramCandidate("nn_fixed", (nn,), nn_pool)
        plan.update(experts=[raw.record(), nn.record(), scratch.record(), waveform.record()],
            mixtures=[ensemble.record(), nn_pool.record()],
            baseline_mixture="NN", source_seed_catalog=[{"outer_target_study": "BNCI2014-008",
                "source_seed": 20260918, "path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}}])
        plan["prefix_selection"].update(baseline_name=nn_candidate.name, candidates=[nn_candidate.record()])
    plan_receipt = write_json(root / "input_plan.json", plan)
    actual_resources = resources(Path(plan_receipt["path"]))
    programme = runner.build_candidate_cloud_tasks(plan_receipt["path"], root / "failed_parent", actual_resources)
    task = programme.tasks[0]
    with get_context("spawn").Manager() as manager:
        admission = HostMemoryAdmission(manager, actual_resources)
        executed = CloudExecutor(admission, max_workers=1,
            diagnostic_directory=programme.output / "worker_diagnostics",
            identity={"actual_source_root": str(runner.ROOT), "plan_sha256": semantic_sha256(programme.normalized)},
            poll_seconds=.02).run([task])
    completed = executed.results[task.task_id]
    assert completed.status == "completed", completed.failure
    parent_outcome = completed
    with cpu_thread_budget(native_threads):
        if not neural:
            programme.on_finished(completed)
        else:
            # The real failed programme has complete worker packages that its
            # parent could not admit. Preserve that boundary in this fixture.
            parent_outcome = CloudResult(task.task_id, "failed",
                failure={"type": "SoftwareFixtureOriginalParentAdmissionUnavailable", "stage": "artifact_admission"})
    failed = CloudResult(programme.tasks[1].task_id, "failed", failure={"type": "SoftwareFixtureInterruption"})
    report = programme.finalize({parent_outcome.task_id: parent_outcome, failed.task_id: failed})
    # Preserve the actually executed fit independently of a parent's admission
    # failure, matching the real split fit/replay producer report.
    report["fit_execution"] = {completed.task_id: asdict(completed)}
    write_json(programme.output / "result.json", report)
    source = json.loads((programme.output / "executed_source.json").read_bytes())
    request = write_json(root / "queue_request.json", {"schema": "n2p3_dataset_queue_request/2", "dataset_id": "software",
        "argv": ["python", str(runner.ROOT / "experiments/run_candidate_ensemble_study.py"),
                 "--plan", plan_receipt["path"], "--output", str(programme.output)],
        "cwd": str(runner.ROOT), "output_dir": str(programme.output), "plan": plan_receipt, "source": source_manifest})
    child = {"pid": 1, "create_time": 1., "process_group": 1}
    actual_exit = write_json(root / "queue_exit.json", {"schema": "n2p3_dataset_queue_exit/2", "dataset_id": "software",
        "request_sha256": request["sha256"], "child": child, "exit_code": 1, "delegated_process_exit_code": 1,
        "science_execution_started": True, "resource_tree": {"active_owned_processes_after_exit": 0}})
    terminal = write_json(root / "queue_terminal.json", {"schema": "n2p3_dataset_queue_terminal/2", "dataset_id": "software",
        "request_sha256": request["sha256"], "child": child, "exit_code": 1, "process_status": "failed",
        "external_exit_receipt_sha256": actual_exit["sha256"], "missing_result_files": [],
        "artifact_facts": [{"path": name, "sha256": hashlib.sha256((programme.output / name).read_bytes()).hexdigest()}
                           for name in ("plan.json", "result.json")]})
    transition = write_json(root / "transition.json", {"schema": "n2p3_prefix_reviewed_source_transition/1",
        "review_status": "accepted", "producer_source_digest": semantic_sha256(source),
        "consumer_source_digest": semantic_sha256(source), "changes": []})
    group = Path(task.output)
    participant = group / "participant_000"
    diagnostics = [path for path in (programme.output / "worker_diagnostics").rglob("*.state.json")
        if json.loads(path.read_bytes())["identity"].get("cloud_task_id") == task.task_id]
    assert len(diagnostics) == 1
    declaration = {"schema": "n2p3_prefix_participant_recovery/1", "original_run": str(programme.output),
        "original_request": request, "original_exit": actual_exit, "original_terminal": terminal,
        "original_executed_source_sha256": hashlib.sha256((programme.output / "executed_source.json").read_bytes()).hexdigest(),
        "original_source_archive_sha256": json.loads(Path(source_manifest["path"]).read_bytes())["archive_sha256"],
        "reviewed_transition": transition, "participants": [{"record_index": 0,
            "fit_diagnostic": {"path": str(diagnostics[0]),
                "sha256": hashlib.sha256(diagnostics[0].read_bytes()).hexdigest()},
            "request_ledger_sha256": hashlib.sha256((participant / "request_ledger.json").read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256((participant / "result.json").read_bytes()).hexdigest(),
            "actor_plan_sha256": hashlib.sha256((group / "plan.json").read_bytes()).hexdigest(),
            "actor_source_sha256": hashlib.sha256((group / "executed_source.json").read_bytes()).hexdigest()}]}
    return root, plan, programme, actual_resources, source, declaration
