"""Actual spawned binary callers, partial artifacts and requested denominators."""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import pytest

from data.source_records import prepare_source_record_prefixes
from experiments import run_erp_comparator as runner
from research.cloud_execution import CloudExecutor, CloudResult
from research.execution import atomic_write_json
from research.source_execution import (
    HostMemoryAdmission,
    SourceInputAdmission,
    resolve_source_resources,
)
from tests.support.raw_prefix import write_real_record


def _native_fault(arguments):
    """Install the fault inside the actual spawned scientific process."""
    subject, mode = arguments["subject"], arguments["fault"]
    if mode == "before_preflight":
        os._exit(39)
    original_fit = runner.fit_calibration_candidate
    fits = 0

    def fit(*args, **kwargs):
        nonlocal fits
        fits += 1
        if fits == 2 and subject == "2":
            os._exit(41)
        return original_fit(*args, **kwargs)

    runner.fit_calibration_candidate = fit
    return runner.comparison_participant_adapter(arguments)


def _native_bad_output(arguments):
    original_write, tampered = runner.atomic_write_json, False

    def write(path, report, **kwargs):
        nonlocal tampered
        if (not tampered and report.get("schema") == "n2p3_erp_comparator_result/2"
                and all(cell["status"] == "completed" for cell in report["cells"])):
            tampered = True
            violation = arguments["violation"]
            if violation == "subject":
                report["plan"]["subjects"] = ["different"]
            elif violation == "partial_source":
                report["source_sha256"] = "f" * 64
            elif violation == "containment":
                report["cells"][0]["model_artifact"]["path"] = "../outside.pt"
            elif violation == "seed_grid":
                report["seed_pairs"].append([99, 98])
            elif violation == "sha":
                (Path(path).parent / report["cells"][0]["prediction_artifact"]["path"]).write_bytes(b"changed")
            original_write(path, report, **kwargs)
            if violation in {"partial_source", "containment"}:
                os._exit(43)
            return Path(path)
        return original_write(path, report, **kwargs)

    runner.atomic_write_json = write
    return runner.comparison_participant_adapter(arguments)


def _plan(tmp_path, *, subjects=("1", "2"), arms=("waveform_lda", "pooled_lda")):
    records = []
    for subject in subjects:
        directory = tmp_path / subject
        directory.mkdir()
        item, spec, _ = write_real_record(directory)
        item["subject"] = subject
        records.append(prepare_source_record_prefixes(item, root=directory, output=directory / "prepared",
            preprocessing=spec, budgets=(8,), minimum_test_trials=2, prepare_inner=False))
    collection = tmp_path / "prefix.json"
    atomic_write_json(collection, {"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": records[0]["preprocessing"], "preparation_method": records[0]["preparation_method"],
        "records": records})
    plan = {"schema": "n2p3_erp_comparator_plan/1", "evidence_scope": "software_correctness",
        "subjects": list(subjects), "budgets": [8], "arms": [{"name": arm, "family": arm,
        "prefix_collection": str(collection), "bin_width": 16} for arm in arms], "allowed_input_axes": [],
        "cpu_budget": 3, "target_workers": 2, "target_worker_memory_gib": .25, "analysis_seed": 20261008}
    path = tmp_path / "plan.json"
    atomic_write_json(path, plan)
    return path, plan


def _run(programme, *, tasks=None):
    resources = programme.resources
    with multiprocessing.get_context("spawn").Manager() as manager:
        inputs = SourceInputAdmission(manager, capacity_bytes=resources["shared_staging_memory_bytes"], load_workers=2)
        admission = HostMemoryAdmission(manager, resources, input_admission=inputs)
        initial = admission.record()
        run = CloudExecutor(admission, max_workers=2, diagnostic_directory=programme.output / "diagnostics",
            identity={"scope": "software_correctness_binary_native"}, poll_seconds=.05).run(
                tasks or programme.tasks, on_finished=programme.on_finished)
        report = programme.finalize(run.results, input_admission=inputs)
        assert admission.record() == initial
        leases = inputs.record()["active_leases"]
    return run, report, leases


def _programme(path, plan, output, *, dependencies=()):
    resources = resolve_source_resources({"source_worker_memory_gib": .25, **plan}, len(plan["subjects"]), "cpu")
    return runner.build_binary_cloud_tasks(path, output, resources, dependencies=dependencies)


def test_native_binary_partial_keeps_peer_seals_and_unknown_inflight_fit(tmp_path):
    path, plan = _plan(tmp_path)
    programme = _programme(path, plan, tmp_path / "run")
    tasks = [programme.tasks[0], replace(programme.tasks[1], adapter=__name__ + ":_native_fault",
        arguments={**programme.tasks[1].arguments, "fault": "during_second_fit"})]
    run, report, leases = _run(programme, tasks=tasks)
    assert run.results[programme.tasks[0].task_id].status == "completed"
    assert run.results[programme.tasks[1].task_id].worker["exit_code"] == 41
    assert report["status"] == "failed" and report["requested_cells"] == 4
    assert report["actual_fit_score_cells"] == 4
    assert report["requested_budget_coverage"] == [{"requested_K": 8, "requested_cells": 4,
        "completed_cells": 3, "protocol_unavailable_cells": 0, "failed_cells": 1}]
    assert report["physical_target_fits"] is None and report["physical_target_fits_lower_bound"] == 3
    assert len(report["physical_target_fits_unknown"]) == 1
    assert report["execution_source_status"] == "unchanged"
    assert report["statistics"]["analysis_seed"] == 20261008
    for cell in report["cells"]:
        if cell["status"] == "completed":
            assert (programme.output / cell["model_artifact"]["path"]).is_file()
            assert runner.file_sha256(programme.output / cell["prediction_artifact"]["path"]) == cell["prediction_artifact"]["sha256"]
    actor_lease = next(lease for lease in leases if lease["path"] == programme.tasks[1].output)
    actual = sum(path.stat().st_size for path in Path(programme.tasks[1].output).rglob("*") if path.is_file())
    assert actor_lease["disk_bytes"] == actual and actor_lease["committed"]


def test_native_all_failed_nonpaired_keeps_unresolved_roster_without_zero_fit_claim(tmp_path):
    path, plan = _plan(tmp_path, arms=("waveform_lda",))
    programme = _programme(path, plan, tmp_path / "run")
    assert len(json.loads((programme.output / "requested.json").read_bytes())["requests"]) == 2
    tasks = [replace(task, adapter=__name__ + ":_native_fault", arguments={**task.arguments, "fault": "before_preflight"})
             for task in programme.tasks]
    run, report, _ = _run(programme, tasks=tasks)
    assert all(outcome.worker["exit_code"] == 39 for outcome in run.results.values())
    assert report["requested_cells"] == 2 and report["actual_fit_score_cells"] == 0
    assert report["physical_target_fits"] is None and report["physical_target_fits_lower_bound"] == 0
    assert len(report["physical_target_fits_unknown"]) == 2
    assert report["statistics_status"] == "unresolved_participant_authority"
    assert report["execution_source_status"] == "unchanged"
    assert all(cell["actual_K"] is None and cell["participant_key"] is None for cell in report["cells"])


@pytest.mark.parametrize("violation", ["subject", "partial_source", "containment", "seed_grid", "sha"])
def test_actual_binary_output_admission_rejects_wrong_scientific_identity(tmp_path, violation):
    path, plan = _plan(tmp_path, subjects=("1",), arms=("waveform_lda",))
    programme = _programme(path, plan, tmp_path / "run")
    task = programme.tasks[0]
    task = replace(task, adapter=__name__ + ":_native_bad_output", arguments={**task.arguments, "violation": violation})
    run, report, _ = _run(programme, tasks=[task])
    assert run.status == report["status"] == "failed"
    assert programme.admission_failures and not programme.admitted
    assert report["requested_cells"] == 1 and report["actual_fit_score_cells"] == 0
    outcome = run.results[task.task_id]
    if violation in {"partial_source", "containment"}:
        assert outcome.worker["exit_code"] == 43 and outcome.failure["stage"] == "process_exit"
        assert outcome.supplementary_failures[0]["stage"] == "artifact_admission"
    else:
        assert outcome.failure["stage"] == "artifact_admission"


def test_predeclared_binary_source_dependency_failures_keep_all_requests(tmp_path):
    path, plan = _plan(tmp_path, arms=("waveform_lda",))
    programme = _programme(path, plan, tmp_path / "run", dependencies=("source:failed",))
    outcomes = {task.task_id: CloudResult(task.task_id, "blocked", failure={"stage": "dependency"}) for task in programme.tasks}
    for outcome in outcomes.values():
        programme.on_finished(outcome)
    report = programme.finalize(outcomes)
    assert report["requested_cells"] == 2 and report["actual_fit_score_cells"] == 0
    assert report["physical_target_fits"] == 0 and not report["physical_target_fits_unknown"]
    assert report["statistics_status"] == "unresolved_participant_authority"
    assert report["requested_budget_coverage"][0]["failed_cells"] == 2
