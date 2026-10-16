"""Original worker facts and full requested cells survive zero-fit recovery."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from research.contracts import semantic_sha256
from research.prefix_recovery import (
    _original_bound_plan,
    _original_fit_execution,
    _validate_zero_fit_roster,
)


def _write_receipt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_original_task_bound_plan_is_rebuilt_from_its_sealed_standalone_input(tmp_path):
    original = {"records": [{"path": "raw/A01.mat", "sha256": "a" * 64}],
        "source_checkpoints": {"NN": "source.pt"}, "source_snapshot_manifest": "source/manifest.json",
        "source_seed_catalog": [{"path": "source.pt", "source_seed": 20261014}]}
    request = {"plan": _write_receipt(tmp_path / "input.json", original)}
    bound = {**original, "records": [{**original["records"][0], "path": str(tmp_path / "raw/A01.mat")}],
        "source_checkpoints": {"NN": str(tmp_path / "source.pt")},
        "source_snapshot_manifest": str(tmp_path / "source/manifest.json"),
        "source_seed_catalog": [{"path": str(tmp_path / "source.pt"), "source_seed": 20261014}]}
    run = tmp_path / "run"
    expected = _write_receipt(run / "bound_plan.json", bound)
    assert _original_bound_plan(request, run) == expected
    bound["records"][0]["path"] = str(tmp_path / "other.mat")
    _write_receipt(run / "bound_plan.json", bound)
    with pytest.raises(ValueError, match="original task plan"):
        _original_bound_plan(request, run)


@pytest.fixture
def original_worker_metadata(tmp_path):
    """Schema-only fixture, not a claim that a public fit was executed here."""
    run, producer_root = tmp_path / "run", tmp_path / "original_source"
    group = run / "participant_groups/participant_000"
    source = {"src/example.py": "a" * 64}
    plan = {"device": "cuda:0", "records": [{"sha256": "b" * 64}]}
    resources = {"phase": "calibration", "cpu_threads": 5, "memory_bytes": 16 * 2**30,
        "native_threads": 4, "device": "cuda:0", "interop_threads": 1, "defer_gpu": True}
    diagnostic = {"schema": "n2p3_source_worker_diagnostic/1", "status": "completed",
        "pid": 19805, "create_time": 1791978248.15,
        "identity": {"cloud_task_id": "candidate:000", "actual_source_root": str(producer_root),
            "execution_backend": "isolated_cache_group_processes", "plan_sha256": semantic_sha256(plan)},
        "output": str(group), "role": {"task_id": "candidate:000", "phase": "calibration",
            "adapter": "experiments.run_candidate_ensemble_study:candidate_participant_adapter",
            "resources": resources}, "native_threads": 4, "interop_threads": 1,
        "actual_phase_reservation": {"task_id": "candidate:000", "phase": "calibration",
            "cpu_threads": 5, "memory_bytes": 16 * 2**30, "gpu_workers": 1}}
    result = {"fit_execution": {"candidate:000": {"task_id": "candidate:000", "status": "completed",
        "value": str(group / "participant_000/result.json"), "failure": None, "supplementary_failures": [],
        "worker": {"pid": diagnostic["pid"], "create_time": diagnostic["create_time"],
            "exit_code": 0, "alive": False, "target_gpu_claim_activated": True,
            "process_tree_retirement": {"active_owned_processes_after_exit": 0,
                "owned_processes": [{"pid": diagnostic["pid"], "create_time": diagnostic["create_time"]}]}}}}}
    receipt_path = run / "worker_diagnostics/original_pool/worker.state.json"
    row = {"fit_diagnostic": _write_receipt(receipt_path, diagnostic)}
    return row, result, plan, source, {"run": run, "producer_root": producer_root, "index": 0,
        "ledger": {"native_threads": 4, "execution_source": source}}, diagnostic


def test_recovery_returns_the_original_fit_outcome_and_actual_gpu_resource_role(original_worker_metadata):
    row, result, plan, source, context, _ = original_worker_metadata
    task, outcome = _original_fit_execution(row, result, plan, source, **context)
    assert asdict(outcome) == result["fit_execution"]["candidate:000"]
    assert task.task_id == outcome.task_id == "candidate:000"
    assert task.resources.cpu_threads == 5 and task.resources.native_threads == 4
    assert task.resources.interop_threads == 1 and task.resources.defer_gpu
    assert task.resources.memory_bytes == 16 * 2**30 and task.resources.device == "cuda:0"
    assert task.arguments["plan"] == str(context["run"] / "bound_plan.json")
    assert task.arguments["output"] == task.output and task.arguments["record_index"] == 0
    assert outcome.worker["pid"] == 19805 and outcome.worker["exit_code"] == 0


@pytest.mark.parametrize("mutation", ["no_fit", "no_worker", "fit_failed", "wrong_return_path",
    "fit_task", "exit", "alive", "owned", "owned_birth", "diagnostic_pid", "diagnostic_birth",
    "diagnostic_task", "foreign_source", "foreign_plan", "native", "cpu", "memory", "interop",
    "gpu_claim", "gpu_reservation", "adapter", "role_output", "source", "receipt_bytes", "foreign_diagnostic"])
def test_recovery_rejects_unjoined_original_fit_provenance(original_worker_metadata, mutation, tmp_path):
    row, result, plan, source, context, diagnostic = deepcopy(original_worker_metadata)
    fit = result["fit_execution"]["candidate:000"]
    if mutation == "no_fit":
        result["fit_execution"] = {}
    elif mutation == "no_worker":
        fit["worker"] = None
    elif mutation == "fit_failed":
        fit["status"] = "failed"
    elif mutation == "wrong_return_path":
        fit["value"] = str(tmp_path / "other/result.json")
    elif mutation == "fit_task":
        fit["task_id"] = "candidate:001"
    elif mutation in {"exit", "alive", "owned", "owned_birth", "gpu_claim"}:
        if mutation == "exit":
            fit["worker"]["exit_code"] = 1
        elif mutation == "alive":
            fit["worker"]["alive"] = True
        elif mutation == "owned":
            fit["worker"]["process_tree_retirement"]["active_owned_processes_after_exit"] = 1
        elif mutation == "owned_birth":
            fit["worker"]["process_tree_retirement"]["owned_processes"][0]["create_time"] += 1
        else:
            fit["worker"]["target_gpu_claim_activated"] = False
    elif mutation == "diagnostic_pid":
        diagnostic["pid"] += 1
    elif mutation == "diagnostic_birth":
        diagnostic["create_time"] += 1
    elif mutation == "diagnostic_task":
        diagnostic["identity"]["cloud_task_id"] = "candidate:001"
    elif mutation == "foreign_source":
        diagnostic["identity"]["actual_source_root"] = str(tmp_path / "other_source")
    elif mutation == "foreign_plan":
        diagnostic["identity"]["plan_sha256"] = "c" * 64
    elif mutation == "native":
        diagnostic["role"]["resources"]["native_threads"] = 1
    elif mutation == "cpu":
        diagnostic["role"]["resources"]["cpu_threads"] = 3
    elif mutation == "memory":
        diagnostic["role"]["resources"]["memory_bytes"] += 1
    elif mutation == "interop":
        diagnostic["interop_threads"] = 2
    elif mutation == "gpu_reservation":
        diagnostic["actual_phase_reservation"]["task_id"] = "candidate:001"
    elif mutation == "adapter":
        diagnostic["role"]["adapter"] = "another:fit"
    elif mutation == "role_output":
        diagnostic["output"] = str(tmp_path / "other_output")
    elif mutation == "source":
        context["ledger"]["execution_source"] = {}
    elif mutation == "receipt_bytes":
        row["fit_diagnostic"]["sha256"] = "d" * 64
    else:
        row["fit_diagnostic"] = _write_receipt(tmp_path / "other/worker.state.json", diagnostic)
    if mutation not in {"receipt_bytes", "foreign_diagnostic"}:
        row["fit_diagnostic"] = _write_receipt(context["run"] / "worker_diagnostics/original_pool/worker.state.json", diagnostic)
    with pytest.raises(ValueError, match="prefix recovery"):
        _original_fit_execution(row, result, plan, source, **context)


def _full_requested_roster():
    records = [{"sha256": f"{index:064x}"} for index in range(8)]
    rows = [{"record_index": index, "request_id": request, "requested": {"decision_id": str(request)},
        "record_sha256": records[index]["sha256"], "status": "requested"}
        for index in range(8) for request in range(760)]
    original = {"requested_records": records, "requested_pipeline_cells": [
        {**row, "status": "failed", "failure": {"type": "OriginalReplayFailure"}} for row in rows]}
    declaration = {"mode": "all_fits_zero_fit", "requested_pipeline_cells": 6080,
        "original_requested_pipeline_sha256": semantic_sha256(original["requested_pipeline_cells"])}
    return declaration, {"records": records}, original, list(range(8)), rows


def test_all_fit_zero_fit_preserves_all_6080_cells_and_accepts_only_changed_admission_state():
    declaration, plan, original, indexes, rows = _full_requested_roster()
    _validate_zero_fit_roster(declaration, plan, original, indexes, rows)
    assert len(rows) == 6080


@pytest.mark.parametrize("mutation", ["missing_actor", "duplicate_actor", "missing_cell",
    "duplicate_replaces_missing_cell", "changed_cell", "changed_original", "wrong_count", "wrong_records"])
def test_all_fit_zero_fit_cannot_shrink_or_replace_the_original_6080_cell_programme(mutation):
    declaration, plan, original, indexes, rows = _full_requested_roster()
    if mutation == "missing_actor":
        indexes.pop()
    elif mutation == "duplicate_actor":
        indexes[-1] = 0
    elif mutation == "missing_cell":
        rows.pop()
    elif mutation == "duplicate_replaces_missing_cell":
        rows[-1] = rows[0]
    elif mutation == "changed_cell":
        rows[-1]["requested"] = {"decision_id": "replacement"}
    elif mutation == "changed_original":
        original["requested_pipeline_cells"][-1]["record_sha256"] = "foreign"
    elif mutation == "wrong_count":
        declaration["requested_pipeline_cells"] = 6079
    else:
        original["requested_records"] = list(reversed(original["requested_records"]))
    with pytest.raises(ValueError, match="every original requested pipeline cell"):
        _validate_zero_fit_roster(declaration, plan, original, indexes, rows)
