"""Real cloud process/lock/exit probes, without EEG or model fitting."""
from __future__ import annotations

import inspect
import json
import runpy
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

import research.dataset_queue as queue_module
from research.dataset_queue import (
    LAUNCH_SCHEMA,
    TERMINAL_SCHEMA,
    DatasetQueue,
    DatasetQueueEntry,
    DatasetQueueError,
    DatasetQueuePlan,
    QueueBusyError,
    plan_from_record,
    sha256_file,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="cloud queue uses Linux process ownership")
RUNNER = Path(__file__).resolve().parents[2] / "experiments/run_dataset_queue.py"
SCIENCE = '''
import json,os,pathlib,subprocess,sys,time
import psutil
name,mode=sys.argv[1:3]; output=pathlib.Path(sys.argv[3]); events=pathlib.Path(sys.argv[4])
output.mkdir(parents=True,exist_ok=False)
with events.open('a') as f:f.write(json.dumps({'dataset':name,'event':'start','time':time.time()})+'\\n')
(output/'started').write_text(str(os.getpid()))
if mode=='wait':
    while not (output.parent/'release').exists():time.sleep(.01)
if mode in {'orphan','orphan-group'}:
    child_code="import json,pathlib,sys,time; import psutil; p=psutil.Process(); pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'create_time':p.create_time()})); time.sleep(60)"
    subprocess.Popen([sys.executable,'-c',child_code,str(output/'descendant.json')],start_new_session=mode=='orphan')
    while not (output/'descendant.json').exists():time.sleep(.01)
    (output/'result.json').write_text(json.dumps({'dataset':name,'partial':True}))
    os._exit(7)
time.sleep(.05)
(output/'result.json').write_text(json.dumps({'dataset':name}))
with events.open('a') as f:f.write(json.dumps({'dataset':name,'event':'finish','time':time.time()})+'\\n')
sys.exit(7 if mode=='fail' else 0)
'''


def _wait(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("required process event was not observed")
        time.sleep(.02)


def _bound(path: Path, body: bytes) -> dict:
    path.write_bytes(body)
    return {"path": str(path), "sha256": sha256_file(path)}


def _fixture(tmp_path, modes=("fail", "ok")):
    source = _bound(tmp_path / "science.py", SCIENCE.encode())
    science_plan = _bound(tmp_path / "science_plan.json", b'{"science_owner":"fixture"}')
    root = tmp_path / "out"
    entries = tuple(DatasetQueueEntry(str(index),
        (sys.executable, source["path"], str(index), mode, str(root / str(index)), str(tmp_path / "events.jsonl")),
        root / str(index), source, science_plan, ("result.json",), cwd=tmp_path)
        for index, mode in enumerate(modes))
    return DatasetQueuePlan(entries), root


def _cli_plan(path: Path, plan: DatasetQueuePlan):
    record = {"schema": "n2p3_dataset_queue/1", "queue_id": plan.queue_id, "datasets": [
        {"dataset_id": entry.dataset_id, "argv": list(entry.command), "output_dir": entry.output_dir.name,
         "source": dict(entry.source), "plan": dict(entry.plan), "cwd": str(entry.cwd),
         "result_files": list(entry.result_files)} for entry in plan.entries]}
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _cli(plan_path: Path, root: Path):
    return subprocess.Popen([sys.executable, "-B", str(RUNNER), "--plan", str(plan_path),
        "--output", str(root), "--poll-seconds", ".02"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _collect(process):
    stdout, stderr = process.communicate(timeout=30)
    return process.returncode, stdout, stderr


def test_queue_is_sequential_and_continues_after_science_failure(tmp_path):
    plan, root = _fixture(tmp_path)
    result = DatasetQueue(plan, root, poll_seconds=.02).run()
    assert result["status"] == "failed" and result["orchestration_status"] == "all_entries_processed"
    assert [row["process_status"] for row in result["results"]] == ["failed", "completed"]
    assert [row["exit_code"] for row in result["results"]] == [7, 0]
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(row["dataset"], row["event"]) for row in events] == [
        ("0", "start"), ("0", "finish"), ("1", "start"), ("1", "finish")]
    for index, row in enumerate(result["results"]):
        control = root / ".dataset_queue" / f"entry_{index:03d}"
        actual = json.loads((control / "queue_exit.json").read_bytes())
        assert actual["observed_by"] == "independent_dataset_supervisor.wait"
        assert actual["resource_tree"]["active_owned_processes_after_exit"] == 0
        assert sha256_file(control / "queue_exit.json") == row["external_exit_receipt_sha256"]
        assert json.loads((root / str(index) / "result.json").read_bytes())["dataset"] == str(index)


def test_queue_rejects_binding_drift_without_touching_previous_dataset(tmp_path):
    plan, root = _fixture(tmp_path, ("ok",))
    DatasetQueue(plan, root).run()
    result_path = root / "0/result.json"
    original = result_path.read_bytes()
    result_path.write_bytes(b"tampered")
    with pytest.raises(DatasetQueueError, match="artifacts changed"):
        DatasetQueue(plan, root).run()
    result_path.write_bytes(original)
    Path(plan.entries[0].source["path"]).write_bytes(b"changed")
    with pytest.raises(DatasetQueueError, match="bound input changed"):
        DatasetQueue(plan, root).run()
    assert result_path.read_bytes() == original


def test_queue_does_not_restart_a_live_child_and_marks_dead_missing_exit_unknown(tmp_path):
    plan, root = _fixture(tmp_path, ("ok",))
    queue = DatasetQueue(plan, root)
    entry = plan.entries[0]
    control = queue._entry_dir(entry)
    _, digest = queue._bound_request(entry, control)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        identity = {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time()}
        (control / "queue_child.json").write_text(json.dumps({"schema": LAUNCH_SCHEMA,
            "dataset_id": entry.dataset_id, "request_sha256": digest,
            "child": identity, "process_group": child.pid}))
        with pytest.raises(QueueBusyError):
            queue._recover_existing(entry, control, digest)
    finally:
        child.terminate()
        child.wait(timeout=10)
    (control / "queue_dispatch.json").write_text(json.dumps({"schema": LAUNCH_SCHEMA,
        "dataset_id": entry.dataset_id, "request_sha256": digest, "command": ["never-started"]}))
    result = queue.run()
    assert result["results"][0]["process_status"] == "unknown"
    assert result["results"][0]["scientific_admission_status"] == "unknown_missing_external_exit"
    assert not entry.output_dir.exists()


def test_plan_decoder_keeps_order_and_structured_argv(tmp_path):
    plan, root = _fixture(tmp_path)
    path = _cli_plan(tmp_path / "queue.json", plan)
    decoded = plan_from_record(json.loads(path.read_bytes()), output_root=root)
    assert [entry.dataset_id for entry in decoded.entries] == ["0", "1"]
    assert decoded.entries[0].command == plan.entries[0].command


def test_parent_kill_keeps_supervisor_late_exit_and_resume_never_duplicates(tmp_path):
    plan, root = _fixture(tmp_path, ("wait", "ok"))
    path = _cli_plan(tmp_path / "queue.json", plan)
    parent = _cli(path, root)
    _wait(lambda: (root / "0/started").is_file())
    control = root / ".dataset_queue/entry_000"
    launch = json.loads((control / "queue_launch.json").read_bytes())
    child = json.loads((control / "queue_child.json").read_bytes())["child"]
    parent.kill()
    parent.wait(timeout=10)
    assert DatasetQueue._pid_alive(launch["supervisor"]) and DatasetQueue._pid_alive(child)
    resumed = _cli(path, root)
    try:
        time.sleep(.1)
        assert resumed.poll() is None and not (root / "1").exists()
        (root / "release").write_text("continue")
        code, _, stderr = _collect(resumed)
        assert code == 0, stderr
    finally:
        (root / "release").write_text("continue")
        if resumed.poll() is None:
            resumed.kill()
            resumed.wait()
    result = json.loads((root / "queue_terminal.json").read_bytes())
    assert [row["process_status"] for row in result["results"]] == ["completed", "completed"]
    assert result["results"][0]["child"]["pid"] == child["pid"]
    assert json.loads((control / "queue_exit.json").read_bytes())["supervisor"] == launch["supervisor"]
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(row["dataset"] == "0" and row["event"] == "start" for row in events) == 1


def test_two_concurrent_queue_parents_spawn_each_dataset_once(tmp_path):
    plan, root = _fixture(tmp_path, ("ok", "ok"))
    path = _cli_plan(tmp_path / "queue.json", plan)
    first, second = _cli(path, root), _cli(path, root)
    outcomes = [_collect(first), _collect(second)]
    assert all(code == 0 for code, _, _ in outcomes), outcomes
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(row["dataset"], row["event"]) for row in events] == [
        ("0", "start"), ("0", "finish"), ("1", "start"), ("1", "finish")]


@pytest.mark.parametrize("mode", ["orphan", "orphan-group"])
def test_native_leader_exit_retires_descendant_before_next_dataset(tmp_path, mode):
    plan, root = _fixture(tmp_path, (mode, "ok"))
    result = DatasetQueue(plan, root, poll_seconds=.02).run()
    assert [row["exit_code"] for row in result["results"]] == [7, 0]
    descendant = json.loads((root / "0/descendant.json").read_bytes())
    assert not DatasetQueue._pid_alive(descendant)
    tree = result["results"][0]["resource_tree"]
    assert tree["active_owned_processes_after_exit"] == 0
    assert descendant["pid"] in {identity["pid"] for identity in tree["owned_processes"]}
    expected_group = descendant["pid"] if mode == "orphan" else result["results"][0]["child"]["pid"]
    assert expected_group in {group["process_group"] for group in tree["groups"]}
    assert (root / "0/result.json").exists() and (root / "1/result.json").exists()


def test_zombie_is_dead_and_reused_pid_identity_is_not_owned():
    child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(0)"])
    identity = {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time()}
    try:
        _wait(lambda: psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE)
        assert not DatasetQueue._pid_alive(identity)
    finally:
        child.wait(timeout=10)
    current = psutil.Process()
    assert not DatasetQueue._pid_alive({"pid": current.pid, "create_time": current.create_time() - 1})


def test_external_exit_survives_terminal_publication_loss(tmp_path):
    plan, root = _fixture(tmp_path, ("ok",))
    result = DatasetQueue(plan, root).run()
    control = root / ".dataset_queue/entry_000"
    original = result["results"][0]
    (control / "queue_terminal.json").unlink()
    recovered = DatasetQueue(plan, root).run()["results"][0]
    assert recovered == original
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2


def test_forged_completed_terminal_cannot_override_actual_failed_exit(tmp_path):
    plan, root = _fixture(tmp_path, ("fail",))
    original = DatasetQueue(plan, root).run()["results"][0]
    control = root / ".dataset_queue/entry_000"
    actual_exit = (control / "queue_exit.json").read_bytes()
    forged = {**original, "process_status": "completed", "exit_code": 0,
              "scientific_admission_status": "delegated_to_science_owner", "failure": None}
    (control / "queue_terminal.json").write_text(json.dumps(forged))
    with pytest.raises(DatasetQueueError, match="terminal differs"):
        DatasetQueue(plan, root).run()
    assert (control / "queue_exit.json").read_bytes() == actual_exit
    assert json.loads(actual_exit)["exit_code"] == 7


def test_unknown_refreshes_surviving_artifacts_and_accepts_later_bound_exit(tmp_path):
    plan, root = _fixture(tmp_path, ("ok",))
    original = DatasetQueue(plan, root).run()["results"][0]
    control = root / ".dataset_queue/entry_000"
    actual_exit = (control / "queue_exit.json").read_bytes()
    (control / "queue_exit.json").unlink()
    unknown = {"schema": TERMINAL_SCHEMA, "dataset_id": "0",
               "request_sha256": original["request_sha256"], "process_status": "unknown",
               "artifact_facts": [], "missing_result_files": ["result.json"]}
    (control / "queue_terminal.json").write_text(json.dumps(unknown))
    recovered = DatasetQueue(plan, root).run()["results"][0]
    assert recovered["process_status"] == "unknown"
    assert recovered["artifact_facts"] == original["artifact_facts"]
    assert recovered["missing_result_files"] == []
    (control / "queue_exit.json").write_bytes(actual_exit)
    assert DatasetQueue(plan, root).run()["results"][0] == original
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_unknown_without_launch_evidence_never_dispatches(tmp_path):
    plan, root = _fixture(tmp_path, ("ok",))
    queue = DatasetQueue(plan, root)
    entry = plan.entries[0]
    control = queue._entry_dir(entry)
    _, digest = queue._bound_request(entry, control)
    (control / "queue_terminal.json").write_text(json.dumps({"schema": TERMINAL_SCHEMA,
        "dataset_id": "0", "request_sha256": digest, "process_status": "unknown"}))
    result = queue.run()["results"][0]
    assert result["process_status"] == "unknown"
    assert not (control / "queue_dispatch.json").exists()
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("relative", [".", ".dataset_queue", ".dataset_queue/library"])
def test_science_output_cannot_overlap_queue_control_namespace(tmp_path, relative):
    plan, root = _fixture(tmp_path, ("ok",))
    entry = replace(plan.entries[0], output_dir=root / relative)
    with pytest.raises(DatasetQueueError, match="distinct directory|control namespace"):
        DatasetQueue(DatasetQueuePlan((entry,)), root).run()
    assert not (tmp_path / "events.jsonl").exists()


def test_middle_exec_enoent_is_typed_without_inventing_science_exit_and_third_runs(tmp_path):
    plan, root = _fixture(tmp_path, ("ok", "ok", "ok"))
    middle = replace(plan.entries[1], command=(str(tmp_path / "missing-science-executable"),))
    plan = replace(plan, entries=(plan.entries[0], middle, plan.entries[2]))
    result = DatasetQueue(plan, root, poll_seconds=.02).run()
    assert [row["process_status"] for row in result["results"]] == ["completed", "failed", "completed"]
    failure = result["results"][1]
    assert failure["exit_code"] is None and failure["science_execution_started"] is False
    assert failure["delegated_process_exit_code"] == 1
    assert failure["failure"]["type"] == "FileNotFoundError" and failure["failure"]["errno"] == 2
    control = root / ".dataset_queue/entry_001"
    receipt = json.loads((control / "queue_exec_failure.json").read_bytes())
    assert receipt["child"]["pid"] == failure["child"]["pid"]
    assert receipt["child"]["create_time"] == failure["child"]["create_time"]
    assert receipt["observed_by"] == "registered_guardian.exec_exception"
    first = (root / "0/result.json").read_bytes()
    assert json.loads((root / "2/result.json").read_bytes())["dataset"] == "2"
    assert DatasetQueue(plan, root).run() == result
    assert (root / "0/result.json").read_bytes() == first
    events = [json.loads(row) for row in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(row["dataset"], row["event"]) for row in events] == [
        ("0", "start"), ("0", "finish"), ("2", "start"), ("2", "finish")]


@pytest.mark.parametrize("owner", ["supervisor", "child"])
def test_external_exit_identity_must_match_registered_processes(tmp_path, owner):
    plan, root = _fixture(tmp_path, ("ok",))
    DatasetQueue(plan, root).run()
    control = root / ".dataset_queue/entry_000"
    exit_path = control / "queue_exit.json"
    exit_record = json.loads(exit_path.read_bytes())
    exit_record[owner]["create_time"] += 1
    exit_path.write_text(json.dumps(exit_record))
    (control / "queue_terminal.json").unlink()
    with pytest.raises(DatasetQueueError, match="differs from registered"):
        DatasetQueue(plan, root).run()


def test_appending_library_keeps_original_process_and_science_bytes(tmp_path):
    full_plan, root = _fixture(tmp_path, ("ok", "fail", "ok"))
    prefix = replace(full_plan, entries=full_plan.entries[:2])
    original_result = DatasetQueue(prefix, root).run()
    files = [path for directory in (root / ".dataset_queue/entry_000", root / ".dataset_queue/entry_001",
                                    root / "0", root / "1") for path in directory.rglob("*") if path.is_file()]
    preserved = {path: path.read_bytes() for path in files}
    original_declaration = (root / ".dataset_queue/queue.json").read_bytes()
    result = DatasetQueue(full_plan, root).run()
    assert result["results"][:2] == original_result["results"]
    assert result["results"][2]["process_status"] == "completed"
    assert all(path.read_bytes() == body for path, body in preserved.items())
    assert (root / ".dataset_queue/queue.json").read_bytes() == original_declaration
    extensions = list((root / ".dataset_queue/declarations").glob("extension_*.json"))
    assert len(extensions) == 1
    assert json.loads(extensions[0].read_bytes())["previous_declaration_sha256"] == sha256_file(
        root / ".dataset_queue/queue.json")
    assert DatasetQueue(full_plan, root).run() == result
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 6
    with pytest.raises(DatasetQueueError, match="declaration prefix changed"):
        DatasetQueue(prefix, root).run()
    with pytest.raises(DatasetQueueError, match="declaration prefix changed"):
        DatasetQueue(replace(full_plan, entries=tuple(reversed(full_plan.entries))), root).run()
    changed = replace(full_plan.entries[0], command=(*full_plan.entries[0].command, "changed"))
    with pytest.raises(DatasetQueueError, match="declaration prefix changed"):
        DatasetQueue(replace(full_plan, entries=(changed, *full_plan.entries[1:])), root).run()
    assert all(path.read_bytes() == body for path, body in preserved.items())


def test_runtime_closure_binds_actual_decorated_lock_and_rejects_private_drift(tmp_path, monkeypatch):
    implementation = Path(inspect.unwrap(queue_module.process_file_lock).__code__.co_filename).resolve()
    assert implementation in {Path(binding["path"]) for binding in queue_module._runtime_files()}
    private_owner = tmp_path / "private_source_execution.py"
    original_body = implementation.read_bytes()
    private_owner.write_bytes(original_body)
    private_lock = runpy.run_path(str(private_owner))["process_file_lock"]
    monkeypatch.setattr(queue_module, "process_file_lock", private_lock)
    plan, root = _fixture(tmp_path, ("ok",))
    original_result = DatasetQueue(plan, root).run()
    control = root / ".dataset_queue/entry_000"
    request_path = control / "queue_request.json"
    original_request = request_path.read_bytes()
    binding = next(item for item in json.loads(original_request)["runtime_files"]
                   if item["path"] == str(private_owner))
    assert binding["sha256"] == sha256_file(private_owner)
    drifted = original_body.replace(b"fcntl.LOCK_EX", b"fcntl.LOCK_SH", 1)
    assert drifted != original_body
    private_owner.write_bytes(drifted)
    assert sha256_file(private_owner) != binding["sha256"]
    with pytest.raises(DatasetQueueError, match="declaration prefix changed"):
        DatasetQueue(plan, root).run()
    assert request_path.read_bytes() == original_request
    assert json.loads((root / "queue_terminal.json").read_bytes()) == original_result
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
