from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.execution import atomic_write_json
from transfer.result_reuse import (
    ResultIntegrityError,
    TargetResultReuse,
    read_result_artifacts,
    read_result_record,
    seal_result_record,
    target_result_key,
)


@pytest.mark.parametrize("alteration", ["model", "prediction", "cell"])
def test_result_cache_isolates_models_and_requires_complete_treatment_key(tmp_path, alteration):
    options = dict(dataset_sha256="a" * 64, participant_key="authority:p1",
                   split={"calibration": "b" * 64, "test": "c" * 64}, architecture={"channels": 3},
                   treatment={"arm": "target_only", "seed": 1}, batch_size=512,
                   source_state=None, snapshot_sha256="d" * 64, device="cpu")
    key = target_result_key(**options)
    store = TargetResultReuse()
    model = tmp_path / "model.pt"
    torch.save({"weights": torch.ones(2)}, model)
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, scores=np.array([0.1, 0.9]))
    cell = seal_result_record(
        {"status": "completed", "evaluation_contract_digest": "e" * 64,
         "fit_history": {"updates": 60}}, files={"prediction": prediction, "model": model})
    atomic_write_json(prediction.with_suffix(".json"), cell)
    store.put(key, scores=np.array([0.1, 0.9]),
              cell=cell, prediction_path=prediction, model_path=model)
    first = store.get(key)
    first["scores"][0] = -100
    first["fit_history"]["updates"] = 0
    assert store.get(key)["fit_history"]["updates"] == 60
    assert store.get(key)["scores"][0] == 0.1
    for field, value in (("dataset_sha256", "f" * 64), ("participant_key", "authority:p2"),
                         ("batch_size", 128), ("source_state", "other"),
                         ("treatment", {"arm": "target_only", "seed": 2}),
                         ("split", {"calibration": "other", "test": "c" * 64})):
        assert store.get(target_result_key(**{**options, field: value})) is None
    if alteration == "model":
        torch.save({"weights": torch.zeros(2)}, model)
    elif alteration == "prediction":
        np.savez_compressed(prediction, scores=np.array([-0.1, -0.9]))
    else:
        atomic_write_json(prediction.with_suffix(".json"), {**cell, "status": "failed"})
    with pytest.raises(ResultIntegrityError, match="integrity"):
        store.get(key)


def test_verified_result_decodes_verified_bytes_after_path_replacement(tmp_path, monkeypatch):
    from transfer import result_reuse

    prediction = tmp_path / "prediction.npz"
    model = tmp_path / "model.pt"
    np.savez_compressed(prediction, scores=np.array([0.1, 0.9]))
    torch.save({"weights": torch.ones(2)}, model)
    record = seal_result_record({"status": "completed"}, files={"prediction": prediction, "model": model})
    original = result_reuse.read_verified_bytes

    def replace_after_read(path, expected_sha256):
        content = original(path, expected_sha256)
        if path == prediction:
            np.savez_compressed(path, scores=np.array([-0.1, -0.9]))
        elif path == model:
            torch.save({"weights": torch.zeros(2)}, path)
        return content

    monkeypatch.setattr(result_reuse, "read_verified_bytes", replace_after_read)
    arrays, payload = read_result_artifacts(record, tmp_path)
    np.testing.assert_array_equal(arrays["scores"], [0.1, 0.9])
    torch.testing.assert_close(payload["weights"], torch.ones(2))


def test_result_cache_cannot_bind_a_record_replaced_during_admission(tmp_path, monkeypatch):
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, scores=np.array([0.1, 0.9]))
    cell = seal_result_record({"status": "completed", "evaluation_contract_digest": "a" * 64,
                              "fit_history": {"updates": 60}}, files={"prediction": prediction})
    record_path = prediction.with_suffix(".json")
    atomic_write_json(record_path, cell)
    replacement = seal_result_record({**cell, "fit_history": {"updates": 999}},
                                     files={"prediction": prediction})
    original = Path.read_bytes

    def replace_during_artifact_read(path):
        content = original(path)
        if path == prediction:
            atomic_write_json(record_path, replacement)
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_during_artifact_read)
    store = TargetResultReuse()
    store.put("key", scores=np.array([0.1, 0.9]), cell=cell, prediction_path=prediction)
    with pytest.raises(ResultIntegrityError, match="integrity"):
        store.get("key")


def test_nonfinite_result_metadata_is_an_integrity_error(tmp_path):
    record = seal_result_record({"status": "completed", "auc": 0.8})
    record["auc"] = float("nan")
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ResultIntegrityError, match="integrity"):
        read_result_record(path)


@pytest.mark.parametrize("scores,fields", [
    (np.ones((2, 1)), {}), (np.ones(1), {"y": np.array([0, 1])}),
    (np.ones(2), {"test_rows": np.array([0])}), (np.array([]), {}),
])
def test_sealed_malformed_predictions_are_integrity_errors(tmp_path, scores, fields):
    path = tmp_path / "prediction.npz"
    np.savez_compressed(path, scores=scores, **fields)
    record = seal_result_record({"status": "completed"}, files={"prediction": path})
    with pytest.raises(ResultIntegrityError, match="integrity"):
        read_result_artifacts(record, tmp_path)


def test_fixed_batch_target_manifest_omits_historical_batches():
    from tests.test_target_experiments import binary_dataset
    from transfer.binary import binary_time_split
    from transfer.target_experiments import build_target_tasks

    dataset = binary_dataset()
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    scopes = {"p1": (np.arange(40), {20: split})}
    e1 = build_target_tasks(dataset, scopes, stage="E1", batch_size=512, include_historical_controls=False)
    assert len(e1["tasks"]) == 21
    assert all(task["recipe"]["batch_size"] == 512 for task in e1["tasks"])
    e3 = build_target_tasks(dataset, scopes, stage="E3", batch_size=512, include_historical_controls=False)
    assert {task["selector_mode"] for task in e3["tasks"]} == {"five_candidates", "transfer_only", "no_transfer"}
    assert all(recipe["batch_size"] == 512 for task in e3["tasks"] for recipe in task["recipes"])


@pytest.mark.parametrize("alteration", [
    None, "scores", "y", "train_rows", "test_rows", "model_state", "model_history",
    "cell", "record", "missing_prediction", "unverified_record",
])
def test_target_stage_reuses_verified_compact_fit_with_current_contract(tmp_path, monkeypatch, alteration):
    import hashlib

    from tests.test_brainsync import _write_current_checkpoint
    from tests.test_target_experiments import binary_dataset
    from transfer.binary import binary_time_split
    from transfer.target_experiments import build_target_tasks, execute_target_tasks

    dataset = binary_dataset()
    path = _write_current_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(path, weights_only=False)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    scopes = {"p1": (np.arange(40), {20: split})}
    context = dict(target_study="synthetic_p300", source_seed=1,
                   checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
                   target_protocol={"kind": "single_recording_prefix"})
    original = build_target_tasks(dataset, scopes, stage="compact", batch_size=512)
    original["tasks"] = [task for task in original["tasks"] if task["arm"] == "compact_waveform_lda"]
    first = execute_target_tasks(dataset, payload, original, output=tmp_path / "compact", context=context, device="cpu")
    if alteration in {"scores", "y", "train_rows", "test_rows"}:
        prediction_path = tmp_path / "compact/00000.npz"
        with np.load(prediction_path) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays[alteration] = -arrays[alteration] if alteration == "scores" else arrays[alteration] + 1
        np.savez_compressed(prediction_path, **arrays)
    elif alteration in {"model_state", "model_history"}:
        model_path = tmp_path / "compact/00000.pt"
        model = torch.load(model_path, weights_only=False)
        if alteration == "model_state":
            model["state"]["intercept"] += 1
        else:
            model["fit_history"]["unexpected_change"] = True
        torch.save(model, model_path)
    elif alteration in {"cell", "record", "unverified_record"}:
        record_path = tmp_path / "compact" / ("00000.json" if alteration == "cell" else "record.json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if alteration == "unverified_record":
            record.pop("result_integrity", None)
        else:
            record["unexpected_change"] = True
        record_path.write_text(json.dumps(record), encoding="utf-8")
    elif alteration == "missing_prediction":
        (tmp_path / "compact/00000.npz").unlink()
    later = build_target_tasks(dataset, scopes, stage="E1", batch_size=512)
    later["tasks"] = [task for task in later["tasks"] if task["arm"] == "e1_waveform_lda"]
    def forbidden(*args, **kwargs):
        raise AssertionError("a matched compact LDA was fitted again")
    monkeypatch.setattr("transfer.calibration_candidates.fit_calibration_candidate", forbidden)
    context["plan"] = {"reuse_target_records": [str(tmp_path / "compact/record.json")]}
    if alteration is not None:
        with pytest.raises(RuntimeError, match="integrity|unverified"):
            execute_target_tasks(dataset, payload, later, output=tmp_path / "e1", context=context, device="cpu")
        return
    second = execute_target_tasks(dataset, payload, later, output=tmp_path / "e1", context=context, device="cpu")
    assert second["cells"][0]["status"] == "completed"
    assert second["cells"][0]["fitted_state_sha256"] == first["cells"][0]["fitted_state_sha256"]
    assert second["cells"][0]["prediction_reuse"]["fit_executed"] is False
    assert second["cells"][0]["evaluation_contract_digest"] != first["cells"][0]["evaluation_contract_digest"]


def test_spawned_e2_scoring_matches_sequential_contracts_and_predictions(tmp_path):
    from baselines.deep import DeepConfig
    from experiments.run_source_training_chain import (
        _score_e2_candidates,
        fit_or_reuse_source_trajectory,
    )
    from tests.test_source_training_chain import _source_dataset
    from transfer.recipes import TargetRecipe

    dataset = _source_dataset()
    path = tmp_path / "trajectory"
    fit_or_reuse_source_trajectory(registry={}, dataset=dataset, rows=np.arange(4),
        config=DeepConfig(fixed_updates=2, snapshot_updates=(1, 2), batch_size=4,
                          sampling="uniform_rows", compile_mode=None, val_group_frac=None),
        destination=path, task_id="test", source_cache_sha256="a" * 64,
        source_snapshot_sha256="b" * 64, excluded_source_datasets=["outer"],
        device="cpu", evidence_scope="software_correctness")
    cohort = [{"subject": "source::s3", "participant_key": dataset.identity_table.record_for("source::s3").authority_key(),
               "domain": "source", "actual_K": 0, "train_rows": [], "test_rows": [4, 5], "purged_rows": []}]
    common = dict(cache_sha256="a" * 64, snapshot_sha256="b" * 64,
                  recipe=TargetRecipe("fixed", epochs=1, batch_size=512),
                  device="cpu", evidence_scope="software_correctness")
    sequential = _score_e2_candidates(dataset, path, cohort, (1, 2), workers=1,
        progress_path=path / "unused.json", output=path, **common)
    expected = [np.load(row["prediction_path"])["scores"].copy() for row in sequential]
    parallel = _score_e2_candidates(dataset, path, cohort, (1, 2), workers=2,
        progress_path=path / "progress.json", output=path, **common)
    assert len(parallel) == 2
    for first, second, scores in zip(sequential, parallel, expected, strict=True):
        assert first["status"] == second["status"] == "completed"
        assert first["evaluation_contract_digest"] == second["evaluation_contract_digest"]
        np.testing.assert_allclose(scores, np.load(second["prediction_path"])["scores"], rtol=1e-6, atol=1e-7)
    progress = json.loads((path / "progress.json").read_text())
    assert progress["completed_cells"] == progress["requested_cells"] == 2
    assert progress["failed_cells"] == 0
    with pytest.raises(ValueError, match="available CPU budget"):
        _score_e2_candidates(copy.deepcopy(dataset), path, cohort, (1, 2), workers=0,
                            progress_path=path / "invalid.json", output=path, **common)
