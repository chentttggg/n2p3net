"""Recover completed prefix participants through a new, independently bound consumer.

The original programme remains failed and read-only. This owner binds its real
queue exit and frozen producer, checks a reviewed source transition, and retains
the complete scientific programme. Numerical/model admission stays with the
candidate runner's existing parent ingress; this module never fits or scores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class PrefixRecoveryProducer:
    plan: dict
    source: dict
    participants: tuple[PrefixRecoveryParticipant, ...]
    binding: dict


def load_prefix_recovery(plan, consumer_source, *, output, consumer_root):
    """Admit an explicit completed subset without shrinking its original programme."""
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
    _bound_json(request["plan"])
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
    indexes = [row["record_index"] for row in declaration["participants"]]
    if (not indexes or len(set(indexes)) != len(indexes)
            or any(type(index) is not int or not 0 <= index < len(plan["records"]) for index in indexes)):
        raise ValueError("prefix recovery participants must be distinct original record indexes")
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
                or semantic_sha256(prefix_programme_authority(actor_plan)) != semantic_sha256(prefix_programme_authority(original_plan))):
            raise ValueError("prefix recovery participant does not retain its complete original producer authority")
        participants.append(PrefixRecoveryParticipant(index, group, ledger["native_threads"]))
    binding = {"schema": "n2p3_prefix_participant_recovery_binding/1", "original_run": str(run),
        "original_source_root": str(producer_root),
        "original_exit": declaration["original_exit"], "original_terminal": declaration["original_terminal"],
        "original_plan_sha256": facts["plan.json"]["sha256"], "producer_source_digest": semantic_sha256(source),
        "consumer_source_digest": semantic_sha256(consumer_source),
        "reviewed_transition": declaration["reviewed_transition"], "transition": transition,
        "reused_record_indexes": indexes, "new_record_indexes": [index for index in range(len(plan["records"])) if index not in indexes],
        "requested_records": plan["records"], "original_failed_status": original_result["status"]}
    return PrefixRecoveryProducer(original_plan, source, tuple(participants), binding)
