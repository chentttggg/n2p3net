"""Actual legacy selector spawn shares the existing snapshot ledger; zero fits."""

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from data.epochs import select_scheduled_events
from research.source_execution import (
    SourceInputAdmission,
    current_input_admission,
    install_input_admission,
)
from tests.support.source_programs import source_early_stopping_data


def _snapshot_probe(arguments):
    admission = current_input_admission()
    if admission is None:
        raise AssertionError("spawned source-prior worker lost its input owner")
    path = Path(arguments["output"]) / hashlib.sha256(arguments["subject"].encode()).hexdigest()[:16]
    path.mkdir(parents=True, exist_ok=True)
    with admission.snapshot(path, 16):
        held = [row for row in admission.record()["active_leases"]
                if row.get("pid") == os.getpid() and row.get("kind") == "anonymous_snapshot"]
        assert len(held) == 1 and held[0]["task_id"] == "legacy-selector-probe"
        assert held[0]["disk_bytes"] == 16
        witness = {"pid": os.getpid(), "task_id": held[0]["task_id"], "lease": held[0]}
        (path / "owner.json").write_text(json.dumps(witness), encoding="utf-8")
        arguments["probe_barrier"].wait(timeout=20)
        if arguments["native_failure"]:
            os._exit(73)
    return [{"participant_key": arguments["participant_key"], "source_seed": arguments["source_seed"],
             "target_seed": arguments["target_seed"], "source_prior_weight": weight,
             "status": "completed", "auc": .5 + weight / 10}
            for weight in arguments["candidates"]]


@pytest.mark.parametrize("native_failure", [False, True], ids=["normal", "native_exit73"])
def test_legacy_source_prior_actual_spawn_retains_parent_snapshot_owner(tmp_path, monkeypatch, native_failure):
    from experiments import run_source_training_chain as runner

    dataset = source_early_stopping_data()
    people = list(dict.fromkeys(dataset.subject_ids))[:2]
    targets = {str(person): select_scheduled_events(dataset,
        dataset.event_timeline.subject_ids == person, reason="synthetic selector probe") for person in people}
    keys = [targets[str(person)].identity_table.authority_keys()[0] for person in people]
    manifest = tmp_path / "prefixes.json"
    manifest.write_text(json.dumps({"records": [{"subject": str(person)} for person in people]}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"selector probe; no neural checkpoint or fit")
    monkeypatch.setattr("data.source_records.load_target_prefixes", lambda path, subjects, **kwargs: {
        subject: {"test": targets[subject], "calibration": {200: targets[subject]}} for subject in subjects})
    monkeypatch.setattr("research.source_selection.source_prior_development_fold", lambda *args, **kwargs: {
        "train_rows": [0, 1], "validation_participant_keys": keys, "fit_ledger": {},
        "participant_domains": {key: "source" for key in keys}})
    monkeypatch.setattr(runner, "fit_or_reuse_source_trajectory", lambda **kwargs: {
        "checkpoints": {1: str(checkpoint)}, "physical_optimizer_updates": 0, "physical_trial_draws": 0,
        "source_training_reuse": {}, "fit_ledger": {}})
    monkeypatch.setattr("transfer.checkpoint.load_checkpoint_payload", lambda path: {"source_epoch_selection": {
        "selected_epochs": 1, "total_optimizer_updates": 0, "total_trial_draws": 0}})

    exits = []

    class WitnessPool(ProcessPoolExecutor):
        def submit(self, function, arguments):
            assert function is runner._score_source_prior_person
            return super().submit(_snapshot_probe, {**arguments, "probe_barrier": barrier,
                "native_failure": native_failure and arguments["subject"] == str(people[0])})

        def __exit__(self, *args):
            processes = tuple(self._processes.values())
            result = super().__exit__(*args)
            exits.extend(process.exitcode for process in processes)
            return result

    monkeypatch.setattr(runner, "ProcessPoolExecutor", WitnessPool)
    monkeypatch.setenv("N2P3_CLOUD_TASK_ID", "legacy-selector-probe")
    old_admission = current_input_admission()
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        admission = SourceInputAdmission(manager, capacity_bytes=2**20, load_workers=2)
        barrier = manager.Barrier(2)
        install_input_admission(admission)
        try:
            plan = {"target_workers": 2, "_native_threads_per_worker": 1,
                    "source_prior_selection": {"prefix_manifests": [str(manifest)], "candidates": [0., .1]},
                    "source_preprocessing": dataset.preprocessing.record(), "evidence_scope": "software_correctness"}
            def select():
                return runner._select_unified_source_prior(dataset,
                    {"name": "heldout", "excluded_source_datasets": ["outer"]},
                    layout="six", plan=plan, report={"tasks": []}, output=tmp_path / "selection",
                    source_sha="a" * 64, snapshot_sha="b" * 64, seed_pairs=((1, 1),), registry={}, device="cpu", dry_run=False)

            if native_failure:
                with pytest.raises(BrokenProcessPool):
                    select()
                assert 73 in exits
                remaining = [row for row in admission.record()["active_leases"] if row["kind"] == "anonymous_snapshot"]
                assert remaining and all(row["task_id"] == "legacy-selector-probe" for row in remaining)
                admission.retire_task("legacy-selector-probe")
            else:
                record = select()
                assert record["target_workers"] == 2 and len(record["outcomes"]) == 4
                assert exits == [0, 0]
            witnesses = list((tmp_path / "selection").rglob("owner.json"))
            assert len(witnesses) == 2
            assert len({json.loads(path.read_bytes())["pid"] for path in witnesses}) == 2
            assert all(json.loads(path.read_bytes())["task_id"] == "legacy-selector-probe" for path in witnesses)
            leases = admission.record()["active_leases"]
            assert not any(row["kind"] == "anonymous_snapshot" for row in leases)
            assert all(row["kind"] == "persistent_artifact" and row["committed"] for row in leases)
        finally:
            install_input_admission(old_admission)
