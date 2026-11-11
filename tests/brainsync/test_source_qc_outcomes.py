import copy

import pytest

from experiments.admit_source_qc_outcomes import NO_EVIDENCE, validate_roster


def roster():
    good = {"study": "S", "subject": "1", "run": "good"}
    bad = {"study": "S", "subject": "1", "run": "unscored"}
    method = {"artifact_preprocessing_policy": {"enabled": True, "continuous_qc": {"rule": "fixed"}}}
    return {"records": [good, bad]}, {"status": "failed", "preparation_method": method,
        "records": [{"record": good, "preparation": {"method": method}}],
        "errors": [{"record": bad, "error_type": "ValueError", "message": NO_EVIDENCE}]}


def test_explicit_zero_evidence_result_keeps_full_original_record_roster():
    manifest, preparation = roster()
    validate_roster(manifest, preparation)


@pytest.mark.parametrize("damage", ["ordinary_error", "missing_record", "duplicate_record", "basic", "all_unusable"])
def test_qc_outcomes_never_hide_missing_recording_or_real_execution_failure(damage):
    manifest, preparation = copy.deepcopy(roster())
    if damage == "ordinary_error":
        preparation["errors"][0]["message"] = "source hash changed"
    elif damage == "missing_record":
        manifest["records"].append({"study": "S", "subject": "2"})
    elif damage == "duplicate_record":
        preparation["errors"].append(preparation["errors"][0])
    elif damage == "basic":
        preparation["preparation_method"]["artifact_preprocessing_policy"]["enabled"] = False
    else:
        preparation["records"] = []
    with pytest.raises(ValueError):
        validate_roster(manifest, preparation)


def test_queue_failure_is_published_to_task_checkpoint_and_pause_is_preserved(tmp_path):
    import json

    from experiments.prepare_consistent_source_cohorts import publish_progress

    task = tmp_path / "task.json"
    task.write_text(json.dumps({"state": "source_preparation_running_GPU_unavailable"}))
    ledger = {"status": "preparation_failed", "completed": {}, "active": "QC:S"}
    publish_progress(tmp_path / "queue.json", ledger, task)
    assert json.loads(task.read_text())["state"] == "source_preparation_failed_GPU_unavailable"
    task.write_text(json.dumps({"state": "paused_by_user"}))
    ledger["status"] = "source_studies_prepared"
    publish_progress(tmp_path / "queue.json", ledger, task)
    assert json.loads(task.read_text())["state"] == "paused_by_user"
