"""Independent source programs use real finite preprocessing and per-seed fits."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.source_records import (
    file_sha256,
    prepare_source_record,
)
from tests.support.snapshots import write_source_snapshot
from tests.support.source_programs import independent_source_plan as _plan
from tests.support.source_programs import prepared_finite_source as _prepared_source


def test_independent_program_preflights_actual_finite_input_and_three_seeds(tmp_path):
    from experiments.run_source_training_chain import execute_source_chain

    cache, spec, _ = _prepared_source(tmp_path)
    snapshot, _, _ = write_source_snapshot(tmp_path)
    plan = _plan(cache, spec, snapshot)
    report = execute_source_chain(plan, output=tmp_path / "dry", dry_run=True,
        device="cpu", stages=("P2_source",))
    assert report["source_program"] == "independent_selection_refit"
    assert report["task_count"] == 3
    assert [task["config"]["seed"] for task in report["tasks"]] == [11, 12, 13]
    assert all(task["config"]["global_early_stopping"] == plan["global_early_stopping"]
               for task in report["tasks"])
    assert not report["fitting_performed"]
    assert not list((tmp_path / "dry").rglob("*.pt"))
    assert not report["outer_evaluation_jobs"]


@pytest.mark.parametrize("change", ["no_profile", "shared_reference", "no_stopping", "partial_stopping", "auxiliary", "crossed"])
def test_independent_program_rejects_other_scientific_programs_before_loading(tmp_path, change):
    from experiments.run_source_training_chain import execute_source_chain

    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   baseline_mode="none")
    plan = _plan(tmp_path / "not_loaded.npz", spec, tmp_path / "not_loaded.json")
    if change == "no_profile":
        plan.pop("source_preprocessing")
    elif change == "shared_reference":
        plan["unified_training_budget"] = {}
    elif change == "no_stopping":
        plan.pop("global_early_stopping")
    elif change == "partial_stopping":
        plan["global_early_stopping"] = {"patience": 1}
    elif change == "auxiliary":
        plan["source_objectives"] = {"supcon": {"supcon_weight": .05}}
    else:
        plan["paired_source_target_seeds"] = False
    with pytest.raises(ValueError, match="independent"):
        execute_source_chain(plan, output=tmp_path / "unused", dry_run=True,
            device="cpu", stages=("P2_source",))


def test_native_cli_saves_actual_three_seed_producers_and_catalog(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    import torch

    from data.epochs import load_epoch_dataset
    from experiments.freeze_source_snapshot import freeze_source
    from research.source_catalog import SourceSeedCatalog
    from research.source_selection import fit_source_trajectory
    from train.early_stopping import GlobalEarlyStoppingPolicy
    from train.runtime import cpu_thread_budget

    cache, spec, _ = _prepared_source(tmp_path)
    snapshot = freeze_source(tmp_path / "snapshot", materialize=True)
    plan = _plan(cache, spec, snapshot)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable,
        str(snapshot.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(plan_path), "--output", str(output), "--stages", "P2_source", "--device", "cpu"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "source_chain_record.json").read_text())
    assert report["source_program"] == "independent_selection_refit"
    assert report["status"] == "completed"
    assert report["task_count"] == report["requested_source_task_count"] == 3
    assert report["completed_physical_fit_count"] == 6
    assert report["source_failure_count"] == 0
    assert not report["outer_evaluation_jobs"]
    entries = report["source_seed_catalog"]
    catalog = SourceSeedCatalog(entries, base_directory=output, evidence_scope="software_correctness")
    assert len(catalog.record()) == 3
    assert all(item["producer"]["selection"] == "per_source_seed" for item in catalog.record())
    dataset = load_epoch_dataset(cache, validation="attested")
    assert dataset.n_times == 179
    for entry in entries:
        payload = torch.load(entry["path"], weights_only=False)
        selection = payload["source_epoch_selection"]
        assert selection["source_seed"] == entry["source_seed"]
        assert selection["refit_initialization"] == "fresh_same_seed"
        assert set(selection["inner_fit_ledger"]["training_participant_keys"]).isdisjoint(
            selection["validation_fit_ledger"]["training_participant_keys"])
        assert selection["full_refit_ledger_digest"] == report["tasks"][entries.index(entry)]["source_producer"]["fit_ledger_digest"]
        # An independently invoked full-source producer from the same seed
        # and numeric thread topology must reproduce the saved refit, without
        # retaining selected weights. Earlier CLI tests may change the parent
        # thread budget without changing this worker's declared budget.
        from baselines.deep import DeepConfig

        config = DeepConfig(fixed_updates=selection["refit_updates"], batch_size=512,
            seed=entry["source_seed"], val_group_frac=None, sampling="uniform_rows", compile_mode=None)
        with cpu_thread_budget(report["resource_policy"]["native_threads_per_worker"]):
            _, expected = fit_source_trajectory(dataset, np.arange(dataset.n_epochs), config=config,
                source_cache_sha256=payload["source_cache_sha256"],
                source_snapshot_sha256=payload["source_snapshot_sha256"],
                excluded_source_datasets=["BNCI2014-008"], device=torch.device("cpu"),
                evidence_scope="software_correctness")
        for name, value in payload["trunk_state_dict"].items():
            torch.testing.assert_close(value, expected[selection["refit_updates"]]["trunk_state_dict"][name],
                rtol=0, atol=0)
        assert payload["global_early_stopping"] == GlobalEarlyStoppingPolicy(**plan["global_early_stopping"]).record()
        assert Path(entry["path"]).is_file()


def test_native_shared_preparation_dag_preserves_full_source_and_actual_worker_budget(tmp_path):
    import subprocess
    import sys

    from experiments.freeze_source_snapshot import freeze_source
    from research.source_catalog import SourceSeedCatalog

    _, spec, manifest = _prepared_source(tmp_path)
    snapshot = freeze_source(tmp_path / "snapshot", materialize=True)
    plan = _plan("not_consumed", spec, snapshot)
    plan["outer_folds"][0].pop("source_cache")
    plan.update(cpu_budget=3, prepared_source_directory=str(tmp_path / "fresh_prepared"),
                source_preparation_manifests={"Huebner2017": str(manifest)})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable,
        str(snapshot.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(plan_path), "--output", str(output), "--stages", "P2_source", "--device", "cpu"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "source_chain_record.json").read_text())
    assert report["status"] == "completed" and report["requested_source_task_count"] == 3
    assert report["completed_physical_fit_count"] == 6 and report["source_failure_count"] == 0
    preparation = report["preparation"][0]
    assert preparation["record_workers"] == preparation["native_threads_per_record_worker"] == 1
    assert preparation["resource_reservation"]["cpu_threads"] == 2
    assert report["groups"][0]["progress"]["native_threads"] == 2
    assert all(point["reserved_cpu_threads"] <= 3 for point in report["resource_observations"])
    assert all(set(task["source_input_bindings"]) == {"Huebner2017"} for task in report["tasks"])
    prepared = json.loads((tmp_path / "fresh_prepared/Huebner2017/preparation.json").read_text())
    assert len(prepared["records"]) == 4 and prepared["n_epochs"] == 32
    assert prepared["preparation_method"]["preprocessing"] == spec.record()
    SourceSeedCatalog(report["source_seed_catalog"], base_directory=output, evidence_scope="software_correctness")


def test_selected_trajectory_reuse_keeps_original_producer_bytes_and_rebind_requires_fit(tmp_path):
    import torch

    from baselines.deep import DeepConfig
    from data.epochs import load_epoch_dataset, save_epoch_dataset
    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory
    from research.source_catalog import SourceSeedCatalog

    cache, spec, _ = _prepared_source(tmp_path)
    data = load_epoch_dataset(cache, validation="attested")
    snapshot, digest, _ = write_source_snapshot(tmp_path)
    plan = _plan(cache, spec, snapshot)
    config = DeepConfig(global_early_stopping=plan["global_early_stopping"], batch_size=512,
        seed=11, val_group_frac=None, sampling="uniform_rows", compile_mode=None)
    arguments = dict(registry={}, dataset=data, rows=np.arange(data.n_epochs), config=config,
        source_cache_sha256=file_sha256(cache), source_snapshot_sha256=digest,
        excluded_source_datasets=["BNCI2014-008"], device="cpu", evidence_scope="software_correctness")
    first = fit_or_reuse_source_trajectory(**arguments, destination=tmp_path / "first", task_id="first")
    second = fit_or_reuse_source_trajectory(**arguments, destination=tmp_path / "second", task_id="second")
    before, reused = next(iter(first["checkpoints"].values())), next(iter(second["checkpoints"].values()))
    assert file_sha256(before) == file_sha256(reused)
    assert first["physical_optimizer_updates"] > 0 and second["physical_optimizer_updates"] == 0
    entry = {"outer_target_study": "BNCI2014-008", "source_seed": 11, "path": reused,
             "sha256": file_sha256(reused), "procedure": {"kind": "selection_refit"}}
    catalog = SourceSeedCatalog([entry], base_directory=tmp_path, evidence_scope="software_correctness")
    assert catalog.record()[0]["source_selection"] == "per_source_seed"
    actual = torch.load(reused, weights_only=False)
    assert actual["training_contract"]["validation"]["strategy"] == "source_internal_early_stopping_then_full_refit"
    repacked = replace(data, name="same_numeric_source_new_container")
    changed = save_epoch_dataset(tmp_path / "repacked.npz", repacked)
    assert file_sha256(changed) != file_sha256(cache)
    third = fit_or_reuse_source_trajectory(**{**arguments, "source_cache_sha256": file_sha256(changed)},
        destination=tmp_path / "third", task_id="third")
    assert third["physical_optimizer_updates"] > 0
    assert not third["source_training_reuse"]["reused"]


def test_missing_preparation_keeps_every_requested_seed_and_failure(tmp_path):
    from experiments.freeze_source_snapshot import freeze_source
    from experiments.run_source_training_chain import execute_source_chain

    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   baseline_mode="none")
    snapshot = freeze_source(tmp_path / "snapshot")
    plan = _plan("not_consumed", spec, snapshot)
    plan["outer_folds"][0].pop("source_cache")
    plan["prepared_source_directory"] = str(tmp_path / "missing_prepared")
    with pytest.raises(RuntimeError, match="requested"):
        execute_source_chain(plan, output=tmp_path / "run", dry_run=False, device="cpu", stages=("P2_source",))
    report = json.loads((tmp_path / "run/source_chain_record.json").read_text())
    assert report["status"] == "failed" and not report["fitting_performed"]
    assert report["requested_source_task_count"] == report["source_failure_count"] == 3
    assert {request["source_seed"] for request in report["requested_source_tasks"]} == {11, 12, 13}
    assert all(request["status"] == "source_group_failure" for request in report["requested_source_tasks"])
    assert not report["source_seed_catalog"]


def test_declared_manifest_cannot_admit_a_different_completed_preparation(tmp_path):
    from data.source_admission import PreparedSourceInventory

    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   baseline_mode="none")
    directory = tmp_path / "Huebner2017"
    directory.mkdir()
    (directory / "preparation.json").write_text(json.dumps({"study": "Huebner2017", "status": "completed",
        "manifest_sha256": "a" * 64}), encoding="utf-8")
    state = PreparedSourceInventory(tmp_path, preprocessing=spec,
        expected_manifest_sha256={"Huebner2017": "b" * 64}).inspect(["Huebner2017"])
    assert not state["ready"] and not state["pending"]
    assert "requested source manifest" in state["failed"]["Huebner2017"]


def test_two_cpu_budget_records_unaffordable_preparation_without_waiting(tmp_path):
    from experiments.freeze_source_snapshot import freeze_source
    from experiments.run_source_training_chain import execute_source_chain

    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
                   baseline_mode="none")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": "Huebner2017",
        "preprocessing": spec.record(), "remote_root": str(tmp_path), "records": [{"study": "Huebner2017",
        "subject": "1", "path": "not_read.bdf", "sha256": "a" * 64, "sample_rate_hz": 250.,
        "reference": "native common", "session": "s1", "run": "r1"}]}), encoding="utf-8")
    snapshot = freeze_source(tmp_path / "snapshot")
    plan = _plan("not_consumed", spec, snapshot)
    plan["outer_folds"][0].pop("source_cache")
    plan.update(cpu_budget=2, prepared_source_directory=str(tmp_path / "fresh_prepared"),
                source_preparation_manifests={"Huebner2017": str(manifest)})
    with pytest.raises(RuntimeError, match="requested"):
        execute_source_chain(plan, output=tmp_path / "run", dry_run=False, device="cpu", stages=("P2_source",))
    report = json.loads((tmp_path / "run/source_chain_record.json").read_text())
    assert report["status"] == "failed" and report["source_failure_count"] == 3
    assert len(report["preparation"]) == 1 and report["preparation"][0]["status"] == "failed"
    assert "controller and a record worker" in report["preparation"][0]["failure"]["message"]
    assert not report["fitting_performed"] and not report["tasks"]
    assert not (tmp_path / "fresh_prepared").exists()


def test_failed_source_fit_keeps_planned_bound_out_of_actual_exposure(tmp_path, monkeypatch):
    from experiments import run_source_training_chain as runner
    from experiments.freeze_source_snapshot import freeze_source

    cache, spec, _ = _prepared_source(tmp_path)
    snapshot = freeze_source(tmp_path / "snapshot")
    plan = {**_plan(cache, spec, snapshot), "_source_group": True}

    def fail_before_fit(**arguments):
        raise RuntimeError("failure before source fitting")

    monkeypatch.setattr(runner, "fit_or_reuse_source_trajectory", fail_before_fit)
    report = runner.execute_source_chain(plan, output=tmp_path / "run", dry_run=False,
                                        device="cpu", stages=("P2_source",))
    assert report["source_failure_count"] == report["requested_source_task_count"] == 3
    assert report["optimizer_updates"] == report["trial_draws"] == 0
    assert report["optimizer_exposure_scope"] == "completed_source_procedures"
    assert set(report["unmeasured_source_fit_tasks"]) == {task["key"] for task in report["tasks"]}
    assert not report["fitting_performed"] and not report["source_seed_catalog"]
    for task in report["tasks"]:
        assert task["optimizer_updates"] is task["trial_draws"] is None
        assert task["requested_optimizer_updates_upper_bound"] > 0
        assert task["budget_kind"] == "actual_unmeasured_after_fit_failure"


def test_uncapped_source_fraction_resolves_actual_participant_authority():
    from baselines.validation import group_disjoint_validation_split
    from data.identity import DatasetIdentityTable
    from research.contracts import semantic_sha256
    from research.source_selection import resolve_independent_source_config

    identity = DatasetIdentityTable.from_source_rows([f"person{i}" for i in range(245)], ["GTN"] * 245)
    membership = {"schema": "n2p3_source_study_membership/1", "stream_to_study": {"GTN": "GTN"},
                  "evidence": [{"basis": "synthetic_test_fixture", "study_id": "GTN"}]}
    membership["digest"] = semantic_sha256(membership)
    actual = resolve_independent_source_config({"pos_weight": 1., "val_groups_min": 2,
        "val_groups_max": None}, identity, membership)
    assert actual["source_participant_counts"] == {"GTN": 245}
    assert actual["resolved"] == {"pos_weight": 1., "val_groups_min": 2, "val_groups_max": 245}
    split = group_disjoint_validation_split(np.arange(245), fraction=.2,
        min_groups=actual["resolved"]["val_groups_min"], max_groups=actual["resolved"]["val_groups_max"])
    assert split.n_validation_groups == 49 and split.train_mask.sum() == 196


def test_current_source_config_reaches_actual_inner_refit_and_frozen_catalog(tmp_path):
    import torch

    from baselines.deep import DeepConfig
    from data.epochs import load_epoch_dataset
    from data.study import dataset_study_membership
    from experiments.freeze_source_snapshot import freeze_source
    from experiments.run_source_training_chain import (
        execute_source_chain,
        fit_or_reuse_source_trajectory,
    )
    from research.evaluation import verified_source_snapshot
    from research.source_catalog import SourceSeedCatalog
    from research.source_selection import resolve_independent_source_config
    from transfer.checkpoint import checkpoint_scores_to_llr

    cache, spec, _ = _prepared_source(tmp_path)
    source = load_epoch_dataset(cache, validation="attested")
    snapshot = freeze_source(tmp_path / "snapshot")
    digest = verified_source_snapshot(snapshot)["archive_sha256"]
    declared = {"pos_weight": 1., "val_groups_min": 2, "val_groups_max": None}
    resolved = resolve_independent_source_config(declared, source.identity_table, dataset_study_membership(source))
    assert resolved["source_participant_counts"] == {"Huebner2017": 4}
    policy = _plan(cache, spec, snapshot)["global_early_stopping"]
    config = DeepConfig(global_early_stopping=policy, seed=11, batch_size=512, sampling="uniform_rows",
                        compile_mode=None, **resolved["resolved"])
    acquired = fit_or_reuse_source_trajectory(registry={}, dataset=source, rows=np.arange(source.n_epochs),
        config=config, destination=tmp_path / "current", task_id="current", source_cache_sha256=file_sha256(cache),
        source_snapshot_sha256=digest, excluded_source_datasets=["BNCI2014-008"], device="cpu",
        evidence_scope="software_correctness", source_program_config=resolved)
    checkpoint = next(iter(acquired["checkpoints"].values()))
    payload = torch.load(checkpoint, weights_only=False)
    actual = payload["training_contract"]["optimizer"]["config"]
    assert actual["pos_weight"] == payload["source_calibration"]["pos_weight"] == 1.
    assert actual["val_groups_min"] == 2 and actual["val_groups_max"] == 4
    assert payload["training_contract"]["objective"]["source_program_config"] == resolved
    selection = payload["source_epoch_selection"]
    selection_progress = [json.loads(line) for line in (tmp_path / "current/progress/selection/fold_0.jsonl").read_text().splitlines()]
    refit_progress = [json.loads(line) for line in (tmp_path / "current/progress/refit/fixed/fold_0.jsonl").read_text().splitlines()]
    assert selection_progress[-1]["optimizer_steps"] == selection["inner_history"]["optimizer_updates"]
    assert selection_progress[-1]["validation_checks"] == selection["validation_checks"]
    assert refit_progress[-1]["optimizer_steps"] == selection["refit_updates"]
    assert len(selection["validation_fit_ledger"]["training_participant_keys"]) == 2
    assert selection["refit_initialization"] == "fresh_same_seed"
    logits = np.asarray([0.])
    llr, calibration = checkpoint_scores_to_llr(payload, logits)
    assert calibration["pos_weight"] == 1. and calibration["offset"] == 0.
    np.testing.assert_array_equal(llr, logits)
    entry = {"outer_target_study": "BNCI2014-008", "source_seed": 11, "path": checkpoint,
             "sha256": file_sha256(checkpoint), "procedure": {"kind": "selection_refit"}}
    catalog = SourceSeedCatalog([entry], base_directory=tmp_path, evidence_scope="software_correctness",
                                expected_source_config=declared)
    assert catalog.record()[0]["producer"]["source_program_config"] == resolved
    with pytest.raises(ValueError, match="source_config"):
        SourceSeedCatalog([entry], base_directory=tmp_path, evidence_scope="software_correctness",
                          expected_source_config={**declared, "pos_weight": 8.})
    assert acquired["source_training_reuse"]["signature_record"]["source_program_config"] == resolved
    dry_plan = {**_plan(cache, spec, snapshot), "source_config": declared}
    planned = execute_source_chain(dry_plan, output=tmp_path / "dry_current", dry_run=True,
                                   device="cpu", stages=("P2_source",))
    assert planned["task_count"] == 3 and not planned["fitting_performed"]
    assert all(task["config"]["pos_weight"] == 1. and task["config"]["val_groups_max"] == 4
               and task["source_program_config"] == resolved for task in planned["tasks"])


@pytest.mark.parametrize("separate_working_directory", [False, True])
def test_real_multistudy_group_merge_keeps_actual_eeg_labels_and_complete_membership(tmp_path, separate_working_directory):
    import multiprocessing
    from pathlib import Path

    from scipy.io import savemat

    from data.epochs import load_epoch_dataset
    from experiments.run_source_training_chain import _prepare_source_group_inputs
    from research.source_catalog import SourceCatalog
    from research.source_execution import SourceInputAdmission
    from research.source_selection import source_identity_axes, validate_independent_source_input
    from tests.support.bnci import bnci_mat_fields

    huebner_cache, spec, _ = _prepared_source(tmp_path)
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(771).normal(0, 4, fields["X"].shape).astype(np.float32)
    path = tmp_path / "raw/A01.mat"
    savemat(path, {"data": fields})
    item = dict(study="BNCI2014-008", subject="A01", path=path.name, sha256=file_sha256(path),
                sample_rate_hz=256., reference="nose", session="0", run="0")
    bnci = prepare_source_record(item, root=path.parent, output=tmp_path / "bnci",
                                preprocessing=spec)
    manifest = path.parent / "bnci_manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": "BNCI2014-008",
        "remote_root": str(path.parent), "preprocessing": spec.record(), "records": [item]}), encoding="utf-8")
    huebner = json.loads((huebner_cache.parent / "preparation.json").read_text())
    sources = {"Huebner2017": huebner, "BNCI2014-008": {
        **bnci, "manifest_sha256": file_sha256(manifest), "preparation_method": bnci["preparation"]["method"]}}
    inputs = {study: {field: record[field] for field in (
        "cache", "cache_sha256", "cache_record_sha256", "manifest_sha256", "preparation_method")}
        for study, record in sources.items()}
    outer = {"name": "leave_BI", "source_studies": list(sources), "target_study": "BI2014a",
             "excluded_source_datasets": ["BI2014a"]}
    plan = {"outer_folds": [outer], "_prepared_source_inputs": inputs,
            "_native_threads_per_worker": 1}
    if separate_working_directory:
        plan["source_view_working_directory"] = str(tmp_path / "independent_working_cache")
    with multiprocessing.get_context("spawn").Manager() as manager:
        plan["_source_input_admission"] = SourceInputAdmission(manager, capacity_bytes=0, load_workers=1)
        resolved = _prepare_source_group_inputs(plan, tmp_path / "run/groups/00_leave_BI_six")
    if separate_working_directory:
        assert Path(resolved["outer_folds"][0]["source_cache"]).is_relative_to(tmp_path / "independent_working_cache")
    merged = load_epoch_dataset(resolved["outer_folds"][0]["source_cache"], validation="attested")
    validate_independent_source_input(merged, spec.record())
    SourceCatalog().validate_fold(outer, source_identity_axes(merged)[0])
    offset = 0
    for record in sources.values():
        original = load_epoch_dataset(record["cache"], validation="attested")
        np.testing.assert_array_equal(merged.X[offset:offset + original.n_epochs], original.X)
        np.testing.assert_array_equal(merged.y[offset:offset + original.n_epochs], original.y)
        offset += original.n_epochs
    assert merged.n_epochs == offset
    assert merged.provenance["event_contract"] == "binary_evidence"
    assert {record["namespace"] for record in merged.provenance["sources"]} == set(sources)
    assert Path(resolved["outer_folds"][0]["source_cache"]).is_file()
