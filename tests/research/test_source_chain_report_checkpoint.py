"""Checkpoint closure, crash/restart and terminal denominator controls; no fits."""
import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from research import source_chain_report as owner


def module(name, relative):
    return importlib.import_module(relative.removeprefix("src/").removesuffix(".py").replace("/", "."))


def report():
    return {"schema": "n2p3_source_training_chain_run/1", "source_snapshot": {
        "archive_sha256": "a"*64, "producer": "original_producer_kept"},
        "requested_source_tasks": [{"key": f"t{i}", "status": "pending", "source_seed": 7} for i in range(3)],
        "tasks": [{"key": f"t{i}", "status": "planned", "config": {"seed": 7},
            "numeric": [1e-200, -0., i/7], "original_producer": "b"*64,
            "payload": "同一序列化数据"*1000} for i in range(3)],
        "selection": [], "preflight_failures": [], "outer_evaluation_jobs": []}


def test_only_changed_shards_and_index_are_written_and_final_legacy_snapshot_is_exact(tmp_path):
    path = tmp_path/"source_chain_record.json"
    value = report()
    first = owner.checkpoint_source_chain_report(path, value)
    assert not path.exists()
    assert owner.load_source_chain_report(path) == value
    index = json.loads((tmp_path/owner.INDEX_NAME).read_bytes())
    assert index["schema"] == owner.INDEX_SCHEMA
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*.json")}
    unchanged = owner.checkpoint_source_chain_report(path, value)
    assert unchanged["written_checkpoint_bytes"] == unchanged["written_payload_shards"] == 0
    assert before == {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.rglob("*.json")}
    value["tasks"][0]["status"] = "completed"
    changed = owner.checkpoint_source_chain_report(path, value)
    assert changed["written_payload_shards"] == 1
    after_index = json.loads((tmp_path/owner.INDEX_NAME).read_bytes())
    assert index["fields"]["tasks"]["items"][1:] == after_index["fields"]["tasks"]["items"][1:]
    assert changed["written_checkpoint_bytes"] < first["written_checkpoint_bytes"]
    owner.checkpoint_source_chain_report(path, value, materialize=True)
    assert json.loads(path.read_bytes()) == owner.load_source_chain_report(path) == value
    assert value["source_snapshot"]["producer"] == "original_producer_kept"


def test_crash_before_index_switch_keeps_previous_complete_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/"source_chain_record.json"
    previous = report()
    owner.checkpoint_source_chain_report(path, previous)
    original_index = (tmp_path/owner.INDEX_NAME).read_bytes()
    next_value = deepcopy(previous)
    next_value["tasks"][0]["status"] = "completed"
    atomic = owner._atomic_write_stream
    def fail_index(path, *args, **kwargs):
        if path.name == owner.INDEX_NAME:
            raise OSError("synthetic crash immediately before atomic index switch")
        return atomic(path, *args, **kwargs)
    monkeypatch.setattr(owner, "_atomic_write_stream", fail_index)
    with pytest.raises(OSError, match="before atomic index switch"):
        owner.checkpoint_source_chain_report(path, next_value)
    assert (tmp_path/owner.INDEX_NAME).read_bytes() == original_index
    assert owner.load_source_chain_report(path) == previous


def test_historical_whole_json_preserves_original_bytes_and_producer(tmp_path):
    path = tmp_path/"source_chain_record.json"
    value = report()
    raw = json.dumps(value, ensure_ascii=False, indent=3).encode("utf8")
    path.write_bytes(raw)
    assert owner.load_source_chain_report(path) == value
    assert path.read_bytes() == raw
    assert not (tmp_path/owner.INDEX_NAME).exists()


def inventory(root, destination, *, omit=None):
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*.json")}
    if omit is not None:
        files.pop(omit)
    value = {"schema": "n2p3_source_resume_inventory/1", "source_root": str(root),
        "source_snapshot_manifest": "externally_frozen_snapshot.json", "files": files}
    destination.write_text(json.dumps(value), encoding="utf8")
    return {"inventory": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def test_resume_reads_same_current_closure_and_rejects_missing_or_tampered_external_dependencies(tmp_path, monkeypatch):
    resume = module("report_resume_audit", "src/research/source_resume.py")
    monkeypatch.setattr(resume, "verified_source_snapshot", lambda path: {"archive_sha256": "a"*64})
    root = tmp_path/"run"
    value = report()
    value["tasks"][0]["status"] = "completed"
    owner.checkpoint_source_chain_report(root/"source_chain_record.json", value)
    declaration = inventory(root, tmp_path/"inventory.json")
    restored = resume.SourceChainResume(declaration)
    assert restored.record == value
    assert restored.record_path.name == owner.INDEX_NAME
    assert set(restored.tasks) == {"t0"}
    assert restored.record["source_snapshot"]["producer"] == "original_producer_kept"
    index = json.loads((root/owner.INDEX_NAME).read_bytes())
    dependency = index["fields"]["tasks"]["items"][0]
    path = root/dependency["file"]
    original = path.read_bytes()
    path.write_bytes(original+b"tampered")
    with pytest.raises(ValueError, match="changed"):
        resume.SourceChainResume(declaration)
    path.write_bytes(original)
    incomplete = inventory(root, tmp_path/"incomplete.json", omit=dependency["file"])
    with pytest.raises(ValueError, match="not present in the frozen inventory"):
        resume.SourceChainResume(incomplete)


def test_resume_merges_partial_group_through_same_verified_reader(tmp_path, monkeypatch):
    resume = module("report_group_resume_audit", "src/research/source_resume.py")
    monkeypatch.setattr(resume, "verified_source_snapshot", lambda path: {"archive_sha256": "a"*64})
    root, child = tmp_path/"run", tmp_path/"run/groups/one"
    child_value = report()
    child_value["tasks"][0]["status"] = "completed"
    owner.checkpoint_source_chain_report(child/"source_chain_record.json", child_value)
    parent = report()
    parent.update(tasks=[], groups=[{"directory": str(child)}])
    owner.checkpoint_source_chain_report(root/"source_chain_record.json", parent)
    restored = resume.SourceChainResume(inventory(root, tmp_path/"inventory.json"))
    assert len(restored.record["tasks"]) == 3
    assert set(restored.tasks) == {"t0"}


@pytest.mark.parametrize("stage,completed", [("progress", 1), ("finalize", 3)])
def test_execution_and_finalization_failures_publish_terminal_state_with_full_requested_denominator(
        tmp_path, monkeypatch, stage, completed):
    runner = module("report_terminal_runner", "experiments/run_source_training_chain.py")
    path = tmp_path/"run/source_chain_record.json"
    def body(*args, **kwargs):
        value = report()
        owner.checkpoint_source_chain_report(path, value)
        for row in value["tasks"][:completed]:
            row["status"] = "completed"
        if stage == "progress":
            owner.checkpoint_source_chain_report(path, value)
            raise RuntimeError("synthetic progress failure")
        runner._save_chain_report(path.parent, value, materialize=True)
    monkeypatch.setattr(runner, "_execute_source_chain_body", body)
    if stage == "finalize":
        original = owner.atomic_write_json
        def fail_snapshot(destination, *args, **kwargs):
            if destination.name == "source_chain_record.json":
                raise OSError("synthetic compatibility snapshot failure")
            return original(destination, *args, **kwargs)
        monkeypatch.setattr(owner, "atomic_write_json", fail_snapshot)
    with pytest.raises((RuntimeError, OSError), match="synthetic"):
        runner.execute_source_chain({}, output=path.parent, dry_run=False, device="cpu", stages=())
    restored = owner.load_source_chain_report(path)
    assert restored["status"] == "failed"
    assert len(restored["requested_source_tasks"]) == len(restored["tasks"]) == 3
    assert sum(row["status"] == "completed" for row in restored["tasks"]) == completed
    assert restored["fatal_error"]["stage"] == "source_execution_or_finalization"
    terminal = json.loads((path.parent/"source_chain_failure.json").read_bytes())
    assert terminal["requested_source_task_count"] == 3
    assert terminal["completed_task_count"] == completed


def test_rejected_existing_output_does_not_rewrite_a_previous_scientific_run(tmp_path, monkeypatch):
    runner = module("report_existing_runner", "experiments/run_source_training_chain.py")
    path = tmp_path/"source_chain_record.json"
    value = report()
    owner.checkpoint_source_chain_report(path, value, materialize=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    def reject(*args, **kwargs):
        raise FileExistsError("source chain requires a new output directory")
    monkeypatch.setattr(runner, "_execute_source_chain_body", reject)
    with pytest.raises(FileExistsError, match="new output directory"):
        runner.execute_source_chain({}, output=tmp_path, dry_run=False, device="cpu", stages=())
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_actual_source_worker_reads_current_index_and_original_snapshot_identity(tmp_path, monkeypatch):
    runner = module("report_actual_worker", "experiments/run_source_training_chain.py")
    value = report()
    observed = []
    monkeypatch.setattr(runner, "execute_source_chain", lambda plan, *, output, **kwargs:
        owner.checkpoint_source_chain_report(output/"source_chain_record.json", value))
    monkeypatch.setattr(runner, "_assert_source_bytes", lambda manifest, snapshot: observed.append(snapshot))
    plan = {"outer_folds": [{"name": "software-reader"}], "_phase_reserved": True,
        "_native_threads_per_worker": 1, "source_snapshot_manifest": "externally_fixed_original_snapshot"}
    result = runner._source_group_worker((plan, str(tmp_path), "cpu", ()))
    assert result == str(tmp_path/"source_chain_record.json")
    assert observed == [value["source_snapshot"]]
    assert not (tmp_path/"source_chain_record.json").exists()


def test_preparation_worker_materializes_one_compatible_final_snapshot(tmp_path, monkeypatch):
    runner = module("report_actual_preparation", "experiments/run_source_training_chain.py")
    value = report()
    monkeypatch.setattr(runner, "_prepare_outer_plans", lambda *args, **kwargs: {"ready": [{"job": "one"}]})
    result = runner._source_legacy_outer_preparation_worker((value, {}, str(tmp_path), "cpu"))
    assert Path(result["record"]).is_file()
    assert json.loads(Path(result["record"]).read_bytes()) == owner.load_source_chain_report(result["record"]) == value
    assert result["groups"] == [[{"job": "one"}]]


def test_current_index_does_not_substitute_for_a_different_sibling_report_name(tmp_path):
    original = tmp_path/"source_chain_record.json"
    value = report()
    owner.checkpoint_source_chain_report(original, value)
    foreign = tmp_path/"foreign.json"
    assert owner.source_chain_report_exists(foreign) is False
    with pytest.raises(FileNotFoundError):
        owner.load_source_chain_report(foreign)
    foreign.write_text(json.dumps({"producer": "independent_sibling"}), encoding="utf8")
    assert owner.load_source_chain_report(foreign) == {"producer": "independent_sibling"}
    assert owner.load_source_chain_report(tmp_path/owner.INDEX_NAME) == value
    assert owner.load_source_chain_report(original) == value
