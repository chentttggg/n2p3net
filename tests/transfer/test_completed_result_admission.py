"""Completed evidence admission through production publication and CLI resume."""

from __future__ import annotations

import copy
import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.brainsync import load_brainsync_session
from data.study import dataset_study_membership
from experiments import run_personal_eog_matrix as personal_cli
from experiments import run_public_pool_targets as public_cli
from experiments.run_brainsync_recovery import candidate_artifact
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from tests.support.brainsync import write_bids_session
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from tests.support.io import read_json, write_json
from train.runtime import cpu_thread_budget
from transfer.binary import binary_time_split
from transfer.calibrated import load_calibrated_predictor, save_calibrated_predictor
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.checkpoint import PreparedTransferCheckpoint
from transfer.recipes import TargetRecipe
from transfer.result_reuse import (
    ResultIntegrityError,
    admit_completed_result,
    seal_result_record,
)
from transfer.target_experiments import build_target_tasks, execute_target_tasks


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _environment():
    return {"torch": torch.__version__, "numpy": np.__version__, "device": "cuda", "threads_per_worker": 1}


def _admit(case, *, expected=None):
    return admit_completed_result(case.path, kind=case.kind, expected=case.expected if expected is None else expected)


def _copy_case(template, destination):
    shutil.copytree(template.path.parent, destination)
    return SimpleNamespace(**{**vars(template), "path": destination / template.path.name,
                              "expected": copy.deepcopy(template.expected)})


def _public_reseal(case, record):
    for index, cell in enumerate(record["cells"]):
        files = {role: case.path.parent / item["name"]
                 for role, item in cell["result_integrity"]["files"].items()}
        cell = seal_result_record(cell, files=files)
        record["cells"][index] = cell
        atomic_write_json(case.path.parent / f"{index:05d}.json", cell)
    atomic_write_json(case.path, seal_result_record(record))


def _personal_reseal(case, record):
    files = {role: case.path.parent / item["name"]
             for role, item in record["result_integrity"]["files"].items()}
    for item in record["predictions"]:
        item["prediction_sha256"] = _sha(files[f"prediction:{item['session']}"])
    if "model" in files:
        record["model_sha256"] = _sha(files["model"])
    atomic_write_json(case.path, seal_result_record(record, files=files))


def _change_arrays(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    change(arrays)
    np.savez_compressed(path, **arrays)


@pytest.fixture(scope="module")
def public_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("public-completed-evidence")
    data = binary_dataset()
    timeline = data.event_timeline
    split = binary_time_split(timeline.onset_times_s, timeline.evidence_available_times_s,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=data.preprocessing.tmin_ms / 1000)
    manifest = build_target_tasks(data, {"p1": (np.arange(data.n_epochs), {20: split})},
                                  stage="compact", batch_size=512)
    manifest["tasks"] = [task for task in manifest["tasks"] if task["arm"] == "compact_waveform_lda"]
    manifest["source_task"] = dict(id="source-test", family="Bpublic", target="BI2014a", pool="B0",
                                    dose="full", seed=1)
    source = write_checkpoint(root / "source.pt", data, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=False)
    context = dict(target_study="BI2014a", source_seed=1, checkpoint_sha256=_sha(source),
        target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
        target_protocol=public_cli.protocol_for("BI2014a"), evidence_scope="development",
        source_study_membership={"digest": "d" * 64}, target_study_membership=dataset_study_membership(data),
        environment=_environment(), plan={})
    directory = root / "result"
    with cpu_thread_budget(1):
        report = execute_target_tasks(data, payload, manifest, output=directory, context=context, device="cpu")
    assert report["completed_cells"] == 1 and report["failed_cells"] == 0
    return SimpleNamespace(path=directory / "record.json", expected={**context, "task_manifest": manifest},
                           kind="public_target", data=data, source=source)


@pytest.fixture
def public_case(public_template, tmp_path):
    return _copy_case(public_template, tmp_path / "public")


@pytest.fixture(scope="module")
def public_zero_template(public_template):
    manifest = copy.deepcopy(public_template.expected["task_manifest"])
    task = manifest["tasks"][0]
    task.update(arm="zero_shot", recipe=None, K=0)
    manifest["scopes"][task["scope_id"]].update(K=0, calibration_rows=[])
    expected = {**copy.deepcopy(public_template.expected), "task_manifest": manifest}
    context = {key: value for key, value in expected.items() if key != "task_manifest"}
    directory = public_template.path.parent.parent / "zero-result"
    with cpu_thread_budget(1):
        execute_target_tasks(public_template.data, torch.load(public_template.source, weights_only=False),
            manifest, output=directory, context=context, device="cpu")
    return SimpleNamespace(path=directory / "record.json", expected=expected, kind="public_target")


@pytest.fixture(scope="module")
def personal_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("personal-completed-evidence")
    calibration = write_bids_session(root / "calibration", session_id="cal", target=4,
        started_utc="2026-09-06T00:00:00Z", blocks=(tuple(range(1, 10)) * 3,))
    source = root / "source.pt"
    source.write_bytes(b"source bytes are bound, not executed by a personal LDA fixture")
    task = dict(id="personal-lda", group="C", budget=27, initialization="waveform_lda", seed=1,
        source_task="personal-source", channels=None, aliases=[], evaluation_sessions=["test-a", "test-b"],
        variant={"alpha": 0.0, "view": "pipeline", "actual_training_rows": 27,
                 "train": {"path": "frozen-train.npz", "sha256": "a" * 64},
                 "test": {"path": "frozen-test.npz", "sha256": "b" * 64}, "scoring_event_ids": None})
    job = dict(task=task, snapshot={"archive_sha256": "c" * 64}, checkpoint_sha256=_sha(source),
               environment=_environment(), plan_sha256="d" * 64)
    expected = personal_cli.personal_result_contract(job)
    directory = root / "result"
    directory.mkdir()
    with cpu_thread_budget(1):
        data = load_brainsync_session(calibration)
        fitted = fit_calibration_candidate(data, TargetRecipe(**expected["recipe"]),
                                           np.ones(data.n_epochs, bool), device="cpu")
        predictor = candidate_artifact(data, fitted, {}, {}, device="cpu", plan_sha256=job["plan_sha256"])
        predictor.metadata.update(matrix_task=task, experiment_source_snapshot=job["snapshot"])
        model = save_calibrated_predictor(directory / "model.pt", predictor)
        restored = load_calibrated_predictor(model)
        files, predictions, test_datasets = {"model": model}, [], {}
        for index, session in enumerate(task["evaluation_sessions"]):
            recording = write_bids_session(root / session, session_id=session, target=7,
                                           started_utc=f"2026-09-07T0{index}:00:00Z")
            test = load_brainsync_session(recording)
            test_datasets[session] = test
            scores = restored.predict_logit(test)
            timeline = test.event_timeline
            path = directory / f"{session}.npz"
            np.savez_compressed(path, scores=scores, raw_scores=scores, y=test.y,
                event_ids=timeline.event_ids, candidate_ids=timeline.candidate_ids,
                repetition_indices=timeline.repetition_indices, group_ids=timeline.group_ids)
            files[f"prediction:{session}"] = path
            predictions.append(dict(session=session, requested_events=len(scores), prediction_sha256=_sha(path)))
    record = {**expected, "status": "completed", "plan_sha256": job["plan_sha256"],
              "source_training_contract_digest": "e" * 64, "fit_history": fitted.fit_history,
              "actual_training_rows": data.n_epochs, "model_sha256": _sha(model), "model_reload_equal": True,
              "predictions": predictions}
    atomic_write_json(directory / "result.json", seal_result_record(record, files=files))
    return SimpleNamespace(path=directory / "result.json", expected=expected, kind="personal", source=source,
                           data=data, test_datasets=test_datasets)


@pytest.fixture
def personal_case(personal_template, tmp_path):
    return _copy_case(personal_template, tmp_path / "personal")


@pytest.fixture(scope="module")
def personal_source_template(personal_template, tmp_path_factory):
    root = tmp_path_factory.mktemp("personal-source-admission")
    case = _copy_case(personal_template, root / "result")
    record = read_json(case.path)
    task = {**copy.deepcopy(record["task"]), "id": "personal-source-fit", "initialization": "source"}
    source = write_checkpoint(root / "source.pt", personal_template.data, source_snapshot_sha256="f" * 64)
    job = dict(task=task, snapshot=record["source_snapshot"], checkpoint_sha256=_sha(source),
               environment=record["environment"], plan_sha256=record["plan_sha256"])
    expected = personal_cli.personal_result_contract(job)
    with cpu_thread_budget(1):
        prepared = PreparedTransferCheckpoint(source, personal_template.data)
        fitted = fit_calibration_candidate(personal_template.data, TargetRecipe(**expected["recipe"]),
            np.ones(personal_template.data.n_epochs, bool), source_trunk=prepared.trunk_for_target("P001"),
            source_input_stats=prepared.input_statistics, device="cpu")
        predictor = candidate_artifact(personal_template.data, fitted, prepared.payload,
            {"checkpoint_sha256": _sha(source)}, device="cpu", plan_sha256=record["plan_sha256"])
        predictor.metadata.update(matrix_task=task, experiment_source_snapshot=record["source_snapshot"])
        model = case.path.parent / "model.pt"
        model.unlink()
        save_calibrated_predictor(model, predictor)
        for session, test in personal_template.test_datasets.items():
            scores = predictor.predict_logit(test)
            _change_arrays(case.path.parent / f"{session}.npz",
                           lambda arrays, scores=scores: arrays.update(scores=scores, raw_scores=scores))
    record.update(expected, fit_history=fitted.fit_history,
                  source_training_contract_digest=prepared.training_contract_digest)
    case.expected = expected
    _personal_reseal(case, record)
    return case


def _as_zero_calibration(case):
    record = read_json(case.path)
    task = record["task"]
    task.update(id="personal-zero", budget=0, initialization="zero_shot")
    case.expected = personal_cli.personal_result_contract(dict(task=task, snapshot=record["source_snapshot"],
        checkpoint_sha256=record["checkpoint_sha256"], environment=record["environment"]))
    record.update(case.expected, target_updates=0)
    for key in ("model_sha256", "model_reload_equal", "fit_history", "actual_training_rows"):
        record.pop(key)
    record["result_integrity"]["files"].pop("model")
    _personal_reseal(case, record)
    return case


@pytest.mark.parametrize("kind", ["public_case", "personal_case"])
def test_production_sealed_outputs_are_admitted(request, kind):
    case = request.getfixturevalue(kind)
    assert _admit(case) == read_json(case.path)


@pytest.mark.parametrize("artifact", ["model", "prediction", "cell", "record", "tasks"])
@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_public_rejects_missing_or_changed_published_artifacts(public_case, artifact, change):
    names = {"model": "00000.pt", "prediction": "00000.npz", "cell": "00000.json",
             "record": "record.json", "tasks": "tasks.json"}
    path = public_case.path.parent / names[artifact]
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed after publication")
    with pytest.raises(ResultIntegrityError):
        _admit(public_case)


@pytest.mark.parametrize("artifact", ["model.pt", "test-a.npz", "result.json"])
@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_personal_rejects_missing_or_changed_published_artifacts(personal_case, artifact, change):
    path = personal_case.path.parent / artifact
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed after publication")
    with pytest.raises(ResultIntegrityError):
        _admit(personal_case)


@pytest.mark.parametrize("kind,filename", [
    ("public_case", "record.json"), ("public_case", "00000.json"), ("personal_case", "result.json"),
])
def test_admission_does_not_invent_a_missing_original_seal(request, kind, filename):
    case = request.getfixturevalue(kind)
    path = case.path.parent / filename
    value = read_json(path)
    value.pop("result_integrity")
    write_json(path, value)
    before = path.read_bytes()
    with pytest.raises(ResultIntegrityError, match="unverified"):
        _admit(case)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["public_case", "personal_case"])
def test_caller_must_supply_the_complete_expected_task(request, kind):
    case = request.getfixturevalue(kind)
    with pytest.raises(ValueError, match="full task declaration"):
        _admit(case, expected={"checkpoint_sha256": case.expected["checkpoint_sha256"]})


@pytest.mark.parametrize("kind,field,value", [
    ("public_case", "source_seed", True), ("personal_case", "fit_performed", 1),
])
def test_task_matching_preserves_boolean_and_integer_contract_types(request, kind, field, value):
    case = request.getfixturevalue(kind)
    expected = copy.deepcopy(case.expected)
    expected[field] = value
    with pytest.raises(ResultIntegrityError):
        _admit(case, expected=expected)


@pytest.mark.parametrize("change", ["chunk", "recipe", "source_task", "environment"])
def test_public_resume_rejects_different_tasks_with_identical_source_and_cache(public_case, change):
    expected = copy.deepcopy(public_case.expected)
    manifest = expected["task_manifest"]
    if change == "chunk":
        manifest["tasks"][0]["subject"] = "another-person"
    elif change == "recipe":
        manifest["tasks"][0]["recipe"]["seed"] += 1
    elif change == "source_task":
        manifest["source_task"]["pool"] = "B7"
    else:
        expected["environment"]["threads_per_worker"] += 1
    with pytest.raises(ResultIntegrityError, match="differs"):
        _admit(public_case, expected=expected)


@pytest.mark.parametrize("change", ["task", "recipe", "scoring_events", "environment"])
def test_personal_resume_rejects_changed_treatment(personal_case, change):
    expected = copy.deepcopy(personal_case.expected)
    if change == "task":
        expected["task"]["seed"] += 1
    elif change == "recipe":
        expected["recipe"]["seed"] += 1
    elif change == "scoring_events":
        expected["task"]["variant"]["scoring_event_ids"] = ["another-event"]
    else:
        expected["environment"]["threads_per_worker"] += 1
    with pytest.raises(ResultIntegrityError, match="differs"):
        _admit(personal_case, expected=expected)


@pytest.mark.parametrize("change", ["missing_cell", "duplicate_cell", "wrong_count", "model_history", "model_state"])
def test_public_semantic_inconsistency_is_rejected_even_with_valid_file_hashes(public_case, change):
    record = read_json(public_case.path)
    if change == "missing_cell":
        record["cells"] = []
    elif change == "duplicate_cell":
        record["cells"].append(copy.deepcopy(record["cells"][0]))
    elif change == "wrong_count":
        record["failed_cells"] = 1
    else:
        model_path = public_case.path.parent / "00000.pt"
        model = torch.load(model_path, weights_only=False)
        if change == "model_history":
            model["fit_history"]["audit_changed"] = True
        else:
            model["state"]["intercept"] += 1
        torch.save(model, model_path)
    _public_reseal(public_case, record)
    with pytest.raises(ResultIntegrityError):
        _admit(public_case)


def test_completed_public_run_keeps_failed_cells_and_the_original_denominator(public_case):
    record = read_json(public_case.path)
    cell = record["cells"][0]
    cell.update(status="fit_failed", auc=None, failure={"type": "ValueError", "message": "no test classes"})
    cell["result_integrity"]["files"] = {}
    record.update(completed_cells=0, failed_cells=1)
    _public_reseal(public_case, record)
    admitted = _admit(public_case)
    assert admitted["completed_cells"] == 0 and admitted["failed_cells"] == len(admitted["cells"]) == 1
    assert (public_case.path.parent / "00000.pt").exists()


def test_public_zero_shot_is_complete_without_a_target_model(public_zero_template, tmp_path):
    case = _copy_case(public_zero_template, tmp_path / "zero")
    admitted = _admit(case)
    assert set(admitted["cells"][0]["result_integrity"]["files"]) == {"prediction"}
    extra = case.path.parent / "unexpected-model.pt"
    torch.save({"state": {}}, extra)
    cell = admitted["cells"][0]
    prediction = case.path.parent / cell["result_integrity"]["files"]["prediction"]["name"]
    admitted["cells"][0] = seal_result_record(cell, files={"prediction": prediction, "model": extra})
    _public_reseal(case, admitted)
    with pytest.raises(ResultIntegrityError, match="required artifacts"):
        _admit(case)


@pytest.mark.parametrize("zero", [False, True])
def test_personal_fitted_and_zero_calibration_have_different_required_models(personal_case, zero):
    if zero:
        _as_zero_calibration(personal_case)
        assert _admit(personal_case)["fit_performed"] is False
    else:
        record = read_json(personal_case.path)
        record["result_integrity"]["files"].pop("model")
        _personal_reseal(personal_case, record)
        with pytest.raises(ResultIntegrityError, match="required artifacts"):
            _admit(personal_case)


@pytest.mark.parametrize("all_missing", [False, True])
def test_personal_nan_evidence_remains_in_the_requested_denominator(personal_case, all_missing):
    record = read_json(personal_case.path)
    for item in record["predictions"]:
        def unavailable(arrays):
            indices = slice(None) if all_missing else slice(0, 2)
            arrays["scores"][indices] = np.nan
            arrays["raw_scores"][indices] = np.nan
        _change_arrays(personal_case.path.parent / f"{item['session']}.npz", unavailable)
    _personal_reseal(personal_case, record)
    admitted = _admit(personal_case)
    assert sum(item["requested_events"] for item in admitted["predictions"]) == 18


def test_personal_common_scoring_view_keeps_raw_scores_and_all_scheduled_events(personal_case):
    record = read_json(personal_case.path)
    scoring_ids = []
    for item in record["predictions"]:
        path = personal_case.path.parent / f"{item['session']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            scoring_ids.extend(archive["event_ids"][:3].tolist())
    record["task"]["variant"].update(view="common", scoring_event_ids=scoring_ids)
    for item in record["predictions"]:
        def select_common(arrays):
            arrays["scores"] = np.where(np.isin(arrays["event_ids"], scoring_ids), arrays["raw_scores"], np.nan)
        _change_arrays(personal_case.path.parent / f"{item['session']}.npz", select_common)
    model_path = personal_case.path.parent / "model.pt"
    payload = torch.load(model_path, weights_only=True)
    payload["metadata"]["matrix_task"] = record["task"]
    payload["metadata_digest"] = semantic_sha256(payload["metadata"])
    torch.save(payload, model_path)
    personal_case.expected = personal_cli.personal_result_contract(dict(
        task=record["task"], snapshot=record["source_snapshot"], checkpoint_sha256=record["checkpoint_sha256"],
        environment=record["environment"]))
    _personal_reseal(personal_case, record)
    assert sum(item["requested_events"] for item in _admit(personal_case)["predictions"]) == 18


@pytest.mark.parametrize("change", ["infinite", "wrong_shape", "duplicate_event", "cross_session_event",
                                    "wrong_scoring_view", "negative_repetition"])
def test_personal_semantic_prediction_failures_are_rejected_after_sealing(personal_case, change):
    record = read_json(personal_case.path)
    first = personal_case.path.parent / "test-a.npz"
    def malformed(arrays):
        if change == "infinite":
            arrays["scores"][0] = arrays["raw_scores"][0] = np.inf
        elif change == "wrong_shape":
            arrays["scores"] = arrays["scores"][:, None]
        elif change == "duplicate_event":
            arrays["event_ids"][1] = arrays["event_ids"][0]
        elif change == "cross_session_event":
            with np.load(personal_case.path.parent / "test-b.npz", allow_pickle=False) as other:
                arrays["event_ids"][0] = other["event_ids"][0]
        elif change == "wrong_scoring_view":
            arrays["scores"][0] += 1
        else:
            arrays["repetition_indices"][0] = -1
    _change_arrays(first, malformed)
    _personal_reseal(personal_case, record)
    with pytest.raises(ResultIntegrityError):
        _admit(personal_case)


@pytest.mark.parametrize("change", ["missing_session", "duplicate_session", "model_metadata"])
def test_personal_session_and_saved_model_declarations_must_be_complete(personal_case, change):
    record = read_json(personal_case.path)
    if change == "missing_session":
        record["predictions"].pop()
        record["result_integrity"]["files"].pop("prediction:test-b")
    elif change == "duplicate_session":
        record["predictions"].append(copy.deepcopy(record["predictions"][0]))
    else:
        model_path = personal_case.path.parent / "model.pt"
        payload = torch.load(model_path, weights_only=True)
        payload["metadata"]["matrix_task"]["seed"] += 1
        payload["metadata_digest"] = semantic_sha256(payload["metadata"])
        torch.save(payload, model_path)
    _personal_reseal(personal_case, record)
    with pytest.raises(ResultIntegrityError):
        _admit(personal_case)


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_source_initialized_personal_model_binds_original_source_contract(
    personal_source_template, tmp_path, mismatch,
):
    case = _copy_case(personal_source_template, tmp_path / "source-personal")
    if mismatch:
        record = read_json(case.path)
        record["source_training_contract_digest"] = "0" * 64
        _personal_reseal(case, record)
        with pytest.raises(ResultIntegrityError, match="source"):
            _admit(case)
    else:
        assert _admit(case)["task"]["initialization"] == "source"


@pytest.mark.parametrize("kind", ["public_case", "personal_case"])
def test_admission_decodes_the_verified_bytes_when_paths_are_replaced(request, kind, monkeypatch):
    from transfer import result_reuse

    case = request.getfixturevalue(kind)
    original = result_reuse.read_verified_bytes
    verified = []
    def replace_after_verification(path, digest):
        content = original(path, digest)
        Path(path).write_bytes(b"replacement must not be read by a decoder")
        verified.append(Path(path).suffix)
        return content
    monkeypatch.setattr(result_reuse, "read_verified_bytes", replace_after_verification)
    assert _admit(case)["status"] == "completed"
    assert ".npz" in verified and ".pt" in verified


class NoTrainingPool:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def submit(self, *_args, **_kwargs):
        raise AssertionError("completed-result CLI probes must never submit a training job")


@pytest.mark.parametrize("corrupt", [False, True])
def test_public_main_uses_real_admission_for_resume(public_template, tmp_path, monkeypatch, corrupt):
    output, sources, prepared = tmp_path / "output", tmp_path / "sources", tmp_path / "prepared"
    task = public_template.expected["task_manifest"]["source_task"]
    case = _copy_case(public_template, output / f"{task['id']}_chunk000")
    source_dir, prep_dir = sources / task["id"], prepared / task["target"]
    source_dir.mkdir(parents=True)
    prep_dir.mkdir(parents=True)
    checkpoint = source_dir / "source.pt"
    shutil.copyfile(public_template.source, checkpoint)
    write_json(source_dir / "result.json", {"status": "completed", "task": task, "trajectory": {"checkpoints": {"4000": str(checkpoint)}}})
    write_json(prep_dir / "preparation.json", {"status": "completed", "cache": "stub-cache",
                                               "cache_sha256": case.expected["target_cache_sha256"]})
    if corrupt:
        (case.path.parent / "00000.npz").unlink()
    monkeypatch.setattr(public_cli, "verified_executed_snapshot", lambda *_a, **_k: case.expected["evaluation_snapshot"])
    monkeypatch.setattr(public_cli, "target_data", lambda *_a: public_template.data)
    monkeypatch.setattr(public_cli, "source_payload", lambda *_a: {"source_study_membership": case.expected["source_study_membership"]})
    monkeypatch.setattr(public_cli, "source_matrix_tasks", lambda: [task])
    monkeypatch.setattr(public_cli, "build_manifest", lambda *_a, **_k: case.expected["task_manifest"])
    monkeypatch.setattr(public_cli, "ProcessPoolExecutor", NoTrainingPool)
    monkeypatch.setattr(public_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sys, "argv", ["public", "--prepared", str(prepared), "--sources", str(sources),
        "--dose-sources", str(sources), "--output", str(output), "--snapshot", "stub"])
    if corrupt:
        with pytest.raises(ResultIntegrityError):
            public_cli.main()
    else:
        public_cli.main()
        assert read_json(output / "progress.json")["results"][0]["reused"] is True


@pytest.mark.parametrize("corrupt", [False, True])
def test_personal_main_selective_resume_preserves_original_plan_identity(personal_template, tmp_path, monkeypatch, corrupt):
    output, sources = tmp_path / "output", tmp_path / "sources"
    task = personal_template.expected["task"]
    case = _copy_case(personal_template, output / task["id"])
    source_dir = sources / task["source_task"]
    source_dir.mkdir(parents=True)
    checkpoint = source_dir / "source.pt"
    shutil.copyfile(personal_template.source, checkpoint)
    write_json(source_dir / "result.json", {"status": "completed", "trajectory": {"checkpoints": {"4000": str(checkpoint)}}})
    prepared = tmp_path / "prepared.json"
    write_json(prepared, {"status": "completed"})
    other = {**copy.deepcopy(task), "id": "not-selected"}
    original_plan = {"tasks": [task, other], "snapshot": case.expected["source_snapshot"],
                     "preparation_sha256": _sha(prepared)}
    atomic_write_json(output / "plan.json", original_plan)
    original_plan_bytes = (output / "plan.json").read_bytes()
    original = read_json(case.path)
    original["plan_sha256"] = semantic_sha256(original_plan)
    model_path = case.path.parent / "model.pt"
    model = torch.load(model_path, weights_only=True)
    model["metadata"]["selection"]["plan_sha256"] = original["plan_sha256"]
    model["metadata_digest"] = semantic_sha256(model["metadata"])
    torch.save(model, model_path)
    _personal_reseal(case, original)
    original = read_json(case.path)
    if corrupt:
        (case.path.parent / "model.pt").unlink()
    monkeypatch.setattr(personal_cli, "verified_executed_snapshot", lambda *_a, **_k: case.expected["source_snapshot"])
    monkeypatch.setattr(personal_cli, "personal_tasks", lambda _plan: [task, other])
    monkeypatch.setattr(personal_cli, "ProcessPoolExecutor", NoTrainingPool)
    monkeypatch.setattr(personal_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(sys, "argv", ["personal", "--prepared", str(prepared), "--sources", str(sources),
        "--output", str(output), "--snapshot", "stub", "--task", task["id"]])
    if corrupt:
        with pytest.raises(ResultIntegrityError):
            personal_cli.main()
    else:
        personal_cli.main()
        progress = read_json(output / "progress.json")
        plan = read_json(progress["plan_path"])
        assert len(plan["tasks"]) == 1 and semantic_sha256(plan) != original["plan_sha256"]
        assert (output / "plan.json").read_bytes() == original_plan_bytes
        assert progress["results"] == [{"id": task["id"], "status": "completed", "reused": True}]
        assert read_json(case.path) == original
