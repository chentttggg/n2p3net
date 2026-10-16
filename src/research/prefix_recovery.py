"""Recover completed prefix participants through a new, independently bound consumer.

The original programme remains failed and read-only. This owner binds its real
queue exit and frozen producer, checks a reviewed source transition, and retains
the complete scientific programme. Numerical/model admission stays with the
candidate runner's existing parent ingress; this module never fits or scores.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from research.cloud_execution import CloudResult, CloudTask, PhaseResources
from research.contracts import semantic_sha256
from research.evaluation import verified_executed_snapshot

_RESOURCE_FIELDS = frozenset({
    "workers", "cpu_threads_per_worker", "cpu_budget", "gpu_workers", "source_workers",
    "source_load_workers", "target_workers", "source_worker_memory_gib", "shared_staging_memory_gib",
    "host_memory_fraction", "target_worker_memory_gib", "gpu_task_memory_gib",
    "target_native_threads_per_worker", "participant_output_estimate_bytes",
    "cloud_resource_floor",
    "prefix_replay_chunk_size", "prefix_replay_graph_memory_gib", "prefix_replay_worker_memory_gib",
})
_TRANSITION_ROLES = {
    "experiments/run_candidate_ensemble_study.py": "consumer_admission",
    "src/research/prefix_inner_replay.py": "load_replay_admission",
    "src/research/prefix_selection_replay.py": "load_replay_admission",
    "src/transfer/ensemble_fit.py": "load_replay_admission",
    "src/research/cloud_execution.py": "resource_transport",
    "src/research/prefix_recovery.py": "recovery_consumer",
    "src/research/source_execution.py": "resource_transport",
    "src/research/prefix_replay_dispatch.py": "resource_transport",
    "src/research/prefix_replay_execution.py": "load_replay_dispatch",
    "src/research/prefix_replay_inputs.py": "immutable_input_transport",
    "src/transfer/ensemble_views.py": "immutable_input_transport",
}


def prefix_programme_authority(plan):
    """Resource reservations and producer snapshots have independent bindings."""
    return {key: value for key, value in plan.items()
            if key not in _RESOURCE_FIELDS | {"source_snapshot_manifest", "prefix_recovery"}}


def _bound_json(receipt):
    path = Path(receipt["path"])
    if path.is_symlink():
        raise ValueError("prefix recovery receipt must be an original regular artifact")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
        raise ValueError("prefix recovery original artifact bytes changed")
    return json.loads(content)


def _runtime_source(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*(root / "src").rglob("*.py"), root / "experiments/run_candidate_ensemble_study.py"]}


def _original_bound_plan(request, run):
    """Rebuild the original standalone task input from its sealed CLI plan."""
    base = _bound_json(request["plan"])
    parent = Path(request["plan"]["path"]).resolve().parent
    expected = {**base,
        "records": [{**row, "path": str((parent / row["path"]).resolve())} for row in base["records"]],
        "source_checkpoints": {name: str((parent / path).resolve())
                               for name, path in base["source_checkpoints"].items()}}
    if expected.get("source_snapshot_manifest"):
        expected["source_snapshot_manifest"] = str((parent / expected["source_snapshot_manifest"]).resolve())
    if expected.get("source_seed_catalog"):
        expected["source_seed_catalog"] = [{**entry, "path": str((parent / entry["path"]).resolve())}
                                           for entry in expected["source_seed_catalog"]]
    if expected.get("joint_source_catalog"):
        expected["joint_source_catalog"] = [{**entry, "path": str((parent / entry["path"]).resolve())}
                                            for entry in expected["joint_source_catalog"]]
    if expected.get("joint_completed_primitives"):
        from research.joint_flat_replay import bind_completed_joint_primitive_locators

        expected["joint_completed_primitives"] = bind_completed_joint_primitive_locators(
            expected["joint_completed_primitives"], base_directory=parent)
    if expected.get('joint_completed_readset_transition'):
        from research.joint_readset_transition import bind_joint_readset_descriptor

        expected['joint_completed_readset_transition'] = bind_joint_readset_descriptor(
            expected['joint_completed_readset_transition'], base_directory=parent)
    path = run / "bound_plan.json"
    content = path.read_bytes()
    if json.loads(content) != expected:
        raise ValueError("prefix recovery original task plan differs from its sealed standalone input")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def preserve_prefix_producer_output(plan, output):
    """Check the producer boundary before a new consumer writes its first file."""
    declaration = plan.get("prefix_recovery")
    if declaration is None:
        return
    request = _bound_json(declaration["original_request"])
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(path).resolve())
           for path in (declaration["original_run"], request["cwd"])):
        raise ValueError("prefix recovery must preserve the original run and producer checkout read-only")


def _reviewed_transition(receipt, producer, consumer):
    transition = _bound_json(receipt)
    if (transition["schema"] != "n2p3_prefix_reviewed_source_transition/1"
            or transition["review_status"] != "accepted"
            or transition["producer_source_digest"] != semantic_sha256(producer)
            or transition["consumer_source_digest"] != semantic_sha256(consumer)):
        raise ValueError("prefix recovery source transition lacks its exact reviewed identities")
    expected = {name: {"path": name, "producer_sha256": producer.get(name),
                      "consumer_sha256": consumer.get(name), "role": _TRANSITION_ROLES.get(name)}
                for name in set(producer) | set(consumer) if producer.get(name) != consumer.get(name)}
    changes = {row["path"]: row for row in transition["changes"]}
    if (len(changes) != len(transition["changes"]) or changes != expected
            or any(row["role"] is None or row["consumer_sha256"] is None for row in expected.values())):
        raise ValueError("prefix recovery changes differ from the reviewed admission/resource modules")
    return transition


@dataclass(frozen=True)
class PrefixRecoveryParticipant:
    record_index: int
    group: Path
    native_threads: int
    task: CloudTask
    outcome: CloudResult


@dataclass(frozen=True)
class PrefixRecoveryProducer:
    plan: dict
    source: dict
    participants: tuple[PrefixRecoveryParticipant, ...]
    binding: dict


def _original_fit_execution(row, result, plan, source, *, run, producer_root, index, ledger):
    """Join original returned work to its original process and resource record."""
    receipt = row["fit_diagnostic"]
    diagnostic_path = Path(receipt["path"]).resolve()
    if not diagnostic_path.is_relative_to(run / "worker_diagnostics"):
        raise ValueError("prefix recovery fit diagnostic is outside its original programme")
    diagnostic = _bound_json(receipt)
    role, identity = diagnostic["role"], diagnostic["identity"]
    task_id = role["task_id"]
    fit = result.get("fit_execution", {}).get(task_id)
    if not isinstance(fit, dict):
        raise ValueError("prefix recovery lacks the original returned fit execution")
    group = run / "participant_groups" / f"participant_{index:03d}"
    expected_result = group / f"participant_{index:03d}" / "result.json"
    worker = fit.get("worker")
    if (fit["task_id"] != task_id or fit["status"] != "completed"
            or fit.get("failure") is not None or fit.get("supplementary_failures")
            or not isinstance(fit.get("value"), str) or Path(fit["value"]).resolve() != expected_result
            or not isinstance(worker, dict) or worker.get("exit_code") != 0
            or worker.get("alive") is not False
            or worker.get("process_tree_retirement", {}).get("active_owned_processes_after_exit") != 0):
        raise ValueError("prefix recovery requires the original completed fit and retired worker tree")
    resources = PhaseResources(**role["resources"])
    if (diagnostic["schema"] != "n2p3_source_worker_diagnostic/1"
            or diagnostic["status"] != "completed"
            or task_id != f"candidate:{index:03d}" or identity["cloud_task_id"] != task_id
            or identity["execution_backend"] != "isolated_cache_group_processes"
            or Path(identity["actual_source_root"]).resolve() != producer_root
            or identity["plan_sha256"] != semantic_sha256(plan)
            or diagnostic["pid"] != worker["pid"] or diagnostic["create_time"] != worker["create_time"]
            or Path(diagnostic["output"]).resolve() != group
            or role["adapter"] != "experiments.run_candidate_ensemble_study:candidate_participant_adapter"
            or role["phase"] != "calibration" or resources.phase != "calibration"
            or resources.device != plan["device"] or resources.native_threads != ledger["native_threads"]
            or type(resources.native_threads) is not int or resources.native_threads < 1
            or type(resources.cpu_threads) is not int or resources.cpu_threads < resources.native_threads
            or type(resources.memory_bytes) is not int or resources.memory_bytes <= 0
            or "native_threads" in diagnostic and diagnostic["native_threads"] != resources.native_threads
            or ledger["execution_source"] != source):
        raise ValueError("prefix recovery original fit task, process, native resources or source differs")
    if resources.interop_threads is not None and (
            type(resources.interop_threads) is not int or resources.interop_threads < 1
            or resources.native_threads + resources.interop_threads > resources.cpu_threads
            or diagnostic.get("interop_threads") != resources.interop_threads):
        raise ValueError("prefix recovery original fit inter-op resources differ")
    if resources.defer_gpu:
        reservation = diagnostic["actual_phase_reservation"]
        if (not resources.device.startswith("cuda") or resources.interop_threads is None
                or worker.get("target_gpu_claim_activated") is not True
                or reservation["task_id"] != task_id or reservation["phase"] != resources.phase
                or reservation["cpu_threads"] != resources.cpu_threads
                or reservation["memory_bytes"] != resources.memory_bytes or reservation["gpu_workers"] != 1):
            raise ValueError("prefix recovery original fit deferred GPU reservation differs")
    owned = worker["process_tree_retirement"]["owned_processes"]
    if not any(value["pid"] == worker["pid"] and value["create_time"] == worker["create_time"] for value in owned):
        raise ValueError("prefix recovery retirement does not contain its original fit worker identity")
    # The original standalone entrypoint has one deterministic task per record.
    # Its resources come from the recorded role, never the new consumer's quota.
    task = CloudTask(task_id, role["adapter"], {"plan": str(run / "bound_plan.json"),
        "output": str(group), "record_index": index,
        "output_estimate_bytes": min(2**30, resources.memory_bytes)}, str(group), resources)
    return task, CloudResult(**fit)


def _pipeline_roster(rows):
    return Counter(semantic_sha256({key: value for key, value in row.items()
        if key not in {"status", "failure"}}) for row in rows)


def _validate_zero_fit_roster(declaration, original_plan, original_result, indexes, participant_requests):
    if declaration.get("mode") != "all_fits_zero_fit":
        return
    original_requests = original_result["requested_pipeline_cells"]
    if (set(indexes) != set(range(len(original_plan["records"]))) or len(indexes) != len(set(indexes))
            or original_result.get("requested_records") != original_plan["records"]
            or not original_requests or len(original_requests) != declaration["requested_pipeline_cells"]
            or semantic_sha256(original_requests) != declaration["original_requested_pipeline_sha256"]
            or _pipeline_roster(original_requests) != _pipeline_roster(participant_requests)):
        raise ValueError("all-fit zero-fit recovery must retain every original requested pipeline cell")


def load_prefix_recovery(plan, consumer_source, *, output, consumer_root):
    """Admit an explicit completed subset without shrinking its original programme."""
    from experiments.run_candidate_ensemble_study import _candidate_participant_authority

    declaration = plan["prefix_recovery"]
    if (declaration["schema"] != "n2p3_prefix_participant_recovery/1"
            or plan.get("evaluation_design") != "independent_prefix"
            or plan.get("resume") is not None or plan.get("resume_binding") is not None):
        raise ValueError("prefix recovery is separate from the fixed-prefix M1 resume contract")
    run = Path(declaration["original_run"]).resolve()
    request = _bound_json(declaration["original_request"])
    terminal = _bound_json(declaration["original_terminal"])
    actual_exit = _bound_json(declaration["original_exit"])
    request_digest = declaration["original_request"]["sha256"]
    if (request["schema"] != "n2p3_dataset_queue_request/2"
            or terminal["schema"] != "n2p3_dataset_queue_terminal/2"
            or actual_exit["schema"] != "n2p3_dataset_queue_exit/2"
            or terminal["request_sha256"] != request_digest or actual_exit["request_sha256"] != request_digest
            or terminal["dataset_id"] != request["dataset_id"] or actual_exit["dataset_id"] != request["dataset_id"]
            or terminal["process_status"] != "failed" or terminal["exit_code"] == 0
            or terminal["exit_code"] != actual_exit["exit_code"]
            or terminal["external_exit_receipt_sha256"] != declaration["original_exit"]["sha256"]
            or terminal["child"] != actual_exit["child"]
            or not actual_exit["science_execution_started"]
            or actual_exit["delegated_process_exit_code"] != actual_exit["exit_code"]
            or actual_exit["resource_tree"]["active_owned_processes_after_exit"] != 0):
        raise ValueError("prefix recovery requires the retired original science process's actual failed exit")
    producer_root = Path(request["cwd"]).resolve()
    output = Path(output).resolve()
    if (Path(request["output_dir"]).resolve() != run or output.is_relative_to(run)
            or output.is_relative_to(producer_root)):
        raise ValueError("prefix recovery must preserve the original run and producer checkout read-only")
    argv = request["argv"]
    if (argv.count("--plan") != 1 or argv.count("--output") != 1
            or Path(argv[argv.index("--plan") + 1]).resolve() != Path(request["plan"]["path"]).resolve()
            or Path(argv[argv.index("--output") + 1]).resolve() != run
            or str(producer_root / "experiments/run_candidate_ensemble_study.py") not in argv):
        raise ValueError("prefix recovery original science entrypoint, plan or output differs")
    original_bound_plan = _original_bound_plan(request, run)
    manifest = _bound_json(request["source"])
    if manifest["archive_sha256"] != declaration["original_source_archive_sha256"]:
        raise ValueError("prefix recovery source archive differs from its original execution")
    verified_executed_snapshot(request["source"]["path"], root=producer_root)
    verified_executed_snapshot(plan["source_snapshot_manifest"], root=consumer_root)
    facts = {row["path"]: row for row in terminal["artifact_facts"]}
    if len(facts) != len(terminal["artifact_facts"]) or terminal["missing_result_files"]:
        raise ValueError("prefix recovery original terminal lacks unique complete parent artifact facts")
    original_plan = _bound_json({"path": str(run / "plan.json"), "sha256": facts["plan.json"]["sha256"]})
    original_result = _bound_json({"path": str(run / "result.json"), "sha256": facts["result.json"]["sha256"]})
    source = _bound_json({"path": str(run / "executed_source.json"),
                          "sha256": declaration["original_executed_source_sha256"]})
    if (original_result["status"] != "failed" or original_result["source_digest"] != semantic_sha256(source)
            or Path(original_plan["source_snapshot_manifest"]).resolve() != Path(request["source"]["path"]).resolve()
            or source != _runtime_source(producer_root)
            or consumer_source != _runtime_source(Path(consumer_root).resolve())
            or semantic_sha256(prefix_programme_authority(original_plan)) != semantic_sha256(prefix_programme_authority(plan))):
        raise ValueError("prefix recovery original source or complete scientific programme changed")
    transition = _reviewed_transition(declaration["reviewed_transition"], source, consumer_source)
    participants = []
    participant_requests = []
    indexes = [row["record_index"] for row in declaration["participants"]]
    if (not indexes or len(set(indexes)) != len(indexes)
            or any(type(index) is not int or not 0 <= index < len(plan["records"]) for index in indexes)):
        raise ValueError("prefix recovery participants must be distinct original record indexes")
    mode = declaration.get("mode", "completed_subset")
    if mode not in {"completed_subset", "all_fits_zero_fit"}:
        raise ValueError("prefix recovery mode must retain its declared fit boundary")
    if mode == "all_fits_zero_fit" and set(indexes) != set(range(len(plan["records"]))):
        raise ValueError("all-fit zero-fit recovery requires the complete original record roster")
    for row in declaration["participants"]:
        index = row["record_index"]
        group = run / "participant_groups" / f"participant_{index:03d}"
        directory = group / f"participant_{index:03d}"
        ledger_payload = _bound_json({"path": str(directory / "request_ledger.json"), "sha256": row["request_ledger_sha256"]})
        ledger = ledger_payload["record"]
        actor_result = _bound_json({"path": str(directory / "result.json"), "sha256": row["result_sha256"]})
        actor_source = _bound_json({"path": str(group / "executed_source.json"), "sha256": row["actor_source_sha256"]})
        actor_plan = _bound_json({"path": str(group / "plan.json"), "sha256": row["actor_plan_sha256"]})
        if (semantic_sha256(ledger) != ledger_payload["record_sha256"] or actor_result["status"] != "completed"
                or actor_source != source or ledger["execution_source"] != source
                or ledger["record_index"] != index or ledger["record"] != original_plan["records"][index]
                or ledger["plan_digest"] != semantic_sha256({key: value for key, value in original_plan.items()
                                                            if key != "cpu_threads_per_worker"})
                or semantic_sha256(prefix_programme_authority(actor_plan)) != semantic_sha256(prefix_programme_authority(original_plan))):
            raise ValueError("prefix recovery participant does not retain its complete original producer authority")
        task, outcome = _original_fit_execution(row, original_result, original_plan, source,
            run=run, producer_root=producer_root, index=index, ledger=ledger)
        # The public ingress owns both raw-MAT and record-declaration scope
        # identities, including their distinct declaration/acquisition digests.
        _candidate_participant_authority(task, outcome, original_plan, source)
        participants.append(PrefixRecoveryParticipant(index, group, ledger["native_threads"], task, outcome))
        participant_requests.extend(ledger["pipeline_requests"])
    original_requests = original_result.get("requested_pipeline_cells", [])
    _validate_zero_fit_roster(declaration, original_plan, original_result, indexes, participant_requests)
    binding = {"schema": "n2p3_prefix_participant_recovery_binding/1", "original_run": str(run),
        "original_source_root": str(producer_root),
        "original_exit": declaration["original_exit"], "original_terminal": declaration["original_terminal"],
        "original_bound_plan": original_bound_plan,
        "original_plan_sha256": facts["plan.json"]["sha256"], "producer_source_digest": semantic_sha256(source),
        "consumer_source_digest": semantic_sha256(consumer_source),
        "reviewed_transition": declaration["reviewed_transition"], "transition": transition,
        "reused_record_indexes": indexes, "new_record_indexes": [index for index in range(len(plan["records"])) if index not in indexes],
        "requested_records": plan["records"], "original_failed_status": original_result["status"]}
    binding.update(mode=mode, original_requested_pipeline_cells=len(original_requests),
        original_requested_pipeline_sha256=semantic_sha256(original_requests),
        recovered_fit_task_ids=[participant.task.task_id for participant in participants],
        recovered_fit_diagnostics=[row["fit_diagnostic"] for row in declaration["participants"]])
    return PrefixRecoveryProducer(original_plan, source, tuple(participants), binding)
