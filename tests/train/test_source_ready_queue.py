"""Real source callers preserve ready consumers, producer scope and native tails."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.freeze_source_snapshot import freeze_source
from experiments.run_source_training_chain import _admit_source_group, execute_source_chain
from research.cloud_execution import CloudExecutor
from research.source_selection import resolve_source_program
from tests.support.source_programs import independent_source_plan as _plan
from tests.support.source_programs import prepared_finite_source as _prepared_source


def _native_source(tmp_path, *, seeds=(11,)):
    cache, spec, _ = _prepared_source(tmp_path)
    snapshot = freeze_source(tmp_path / "snapshot", materialize=True)
    plan = _plan(cache, spec, snapshot)
    plan.update(source_seeds=list(seeds), target_seeds=list(seeds))
    plan_path = tmp_path / "source-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable,
        str(snapshot.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(plan_path), "--output", str(output), "--stages", "P2_source", "--device", "cpu"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return plan, output, json.loads((output / "source_chain_record.json").read_bytes())


@pytest.fixture(scope="module")
def single_research_source(tmp_path_factory):
    return _native_source(tmp_path_factory.mktemp("source-single"))


def test_native_single_seed_research_keeps_independent_selection_and_fresh_refit(single_research_source):
    plan, output, report = single_research_source
    assert report["status"] == "completed"
    assert report["requested_source_task_count"] == 1
    assert report["completed_physical_fit_count"] == report["completed_logical_fit_count"] == 2
    assert [entry["source_seed"] for entry in report["source_seed_catalog"]] == [11]
    task = report["tasks"][0]
    assert task["source_epoch_selection"]["refit_initialization"] == "fresh_same_seed"
    assert task["source_epoch_selection"]["source_seed"] == 11
    assert report["groups"][0]["source_artifact_admission"] == "completed_declared_producer_programme"
    assert report["execution_end_source_check"] == "actual_runtime_bytes_match_the_ingress_archive"
    assert not report.get("unadmitted_source_artifacts")
    assert (output / "source_seed_catalog.json").is_file()
    assert plan["source_seeds"] == plan["target_seeds"] == [11]


@pytest.mark.parametrize("change", ["scope", "stopping", "key_seed", "missing_catalog"])
def test_source_ready_gate_rejects_self_consistent_old_scope_or_policy(single_research_source, tmp_path, change):
    plan, output, report = single_research_source
    path = Path(report["groups"][0]["directory"]) / "source_chain_record.json"
    child = json.loads(path.read_bytes())
    program = resolve_source_program(plan, ("P2_source",))
    if change == "scope":
        child["evidence_scope"] = "development"
    elif change == "stopping":
        program = {**program, "stopping": {**program["stopping"], "min_delta": 101.}}
        child["source_program_record"] = program
        child["tasks"][0]["config"]["global_early_stopping"] = program["stopping"]
    elif change == "key_seed":
        child["tasks"][0]["config"]["seed"] = 12
    else:
        child["source_seed_catalog"] = []
    damaged = tmp_path / "source_chain_record.json"
    damaged.write_text(json.dumps(child), encoding="utf-8")
    with pytest.raises(ValueError, match="source|declared"):
        _admit_source_group(damaged, child_plan=plan, program=program, snapshot=report["source_snapshot"])
    # The unchanged physical producer remains available for later recovery.
    for entry in report["source_seed_catalog"]:
        assert Path(entry["path"]).is_file()
    assert (output / "source_seed_catalog.json").is_file()


def test_native_source_releases_current_binary_and_candidate_into_its_resource_owner(tmp_path):
    import hashlib

    import numpy as np
    from scipy.io import savemat

    from data.bnci2014_008_candidate import BNCI2014_008_SOURCE_REFERENCE
    from data.source_records import prepare_source_record_prefixes
    from tests.support.bnci import bnci_mat_fields
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.ensemble_views import SelectionEpochProfile
    from transfer.recipes import TargetRecipe

    cache, spec, _ = _prepared_source(tmp_path)
    raw = tmp_path / "A01.mat"
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(39).normal(0, 4, fields["X"].shape).astype(np.float32)
    savemat(raw, {"data": fields})
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    item = {"study": "BNCI2014-008", "subject": "A01", "path": raw.name, "sha256": raw_sha,
        "sample_rate_hz": 256., "reference": BNCI2014_008_SOURCE_REFERENCE, "session": "0", "run": "0"}
    prefix = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "target-prefix",
                                           preprocessing=spec, budgets=(200,), minimum_test_trials=200)
    collection = tmp_path / "target-prefix.json"
    collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": prefix["preprocessing"], "preparation_method": prefix["preparation_method"],
        "records": [prefix]}), encoding="utf-8")
    channels = ["Fz", "P3", "Pz", "P4", "PO8", "Oz"]
    expected = {"pos_weight": 8., "val_groups_min": 2, "val_groups_max": 12}
    pairs = [[11, 11]]
    binary_plan = {"schema": "n2p3_erp_comparator_plan/1", "evidence_scope": "software_correctness",
        "subjects": ["A01"], "budgets": [200], "seed_pairs": pairs, "source_seed_catalog": [],
        "source_snapshot": None, "expected_source_config": expected, "allowed_input_axes": [], "device": "cpu",
        "arms": [{"name": family, "family": family, "prefix_collection": str(collection), "channel_names": channels,
            **({"epochs": 1, "batch_size": 128} if family in {"full_fine", "scratch"} else {})}
            for family in ("waveform_lda", "full_fine", "scratch")], "contrasts": {}}
    binary_path = tmp_path / "binary-base.json"
    binary_path.write_text(json.dumps(binary_plan), encoding="utf-8")
    profile = SelectionEpochProfile.from_preprocessing(spec, source_sample_rate_hz=256., channel_names=channels,
        reference=BNCI2014_008_SOURCE_REFERENCE, reference_channel="Cz")
    source_recipe = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=1,
        batch_size=128, bn_buffers="calibration", dropout="off")
    experts = [CalibrationExpertProgram("NN", "trial", source_recipe, profile),
        CalibrationExpertProgram("scratch", "trial", replace(source_recipe, name="scratch", initialization="scratch",
            input_statistics="target_prefix"), profile),
        CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(), profile)]
    mixtures = [EnsembleProgram.equal_weights(name, (name,)).record() for name in ("NN", "scratch", "waveform")]
    candidate_plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(raw), "sha256": raw_sha}],
        "budgets": [200], "seed_pairs": pairs, "source_seed_catalog": [], "source_snapshot_manifest": None,
        "expected_source_config": expected, "source_checkpoints": {}, "experts": [expert.record() for expert in experts],
        "mixtures": mixtures, "baseline_mixture": "waveform", "authorization_preprocessing": spec.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2, "device": "cpu",
        "workers": 1, "cpu_threads_per_worker": 1, "calibration_strategies": ["fixed"]}
    candidate_path = tmp_path / "candidate-base.json"
    candidate_path.write_text(json.dumps(candidate_plan), encoding="utf-8")
    snapshot = freeze_source(tmp_path / "snapshot", materialize=True)
    plan = _plan(cache, spec, snapshot)
    plan.update(source_seeds=[11], target_seeds=[11], cpu_budget=3, source_workers=1, target_workers=2,
        ready_consumers=[{"kind": kind, "name": kind, "outer": "leave_BNCI", "plan": str(path),
                          "output": str(tmp_path / kind)} for kind, path in (("binary", binary_path), ("candidate", candidate_path))])
    source_plan_path = tmp_path / "source-plan.json"
    source_plan_path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run"
    completed = subprocess.run([sys.executable,
        str(snapshot.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(source_plan_path), "--output", str(output), "--stages", "P2_source", "--device", "cpu"],
        capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "source_chain_record.json").read_bytes())
    assert report["status"] == "completed" and len(report["consumers"]) == 2
    assert all(consumer["status"] == "completed" for consumer in report["consumers"])
    outcomes = report["cloud_execution"]["results"]
    assert len(outcomes) == 3 and all(result["status"] == "completed" for result in outcomes)
    assert len({result["worker"]["pid"] for result in outcomes}) == 3
    assert all(point["reserved_cpu_threads"] <= 3 and point["free_worker_memory_bytes"] >= 0
               for point in report["resource_observations"])
    assert all(result["worker"]["alive"] is False for result in outcomes)
    binary = json.loads((tmp_path / "binary/result.json").read_bytes())
    candidate = json.loads((tmp_path / "candidate/result.json").read_bytes())
    assert binary["requested_cells"] == 3 and binary["seed_pairs"] == pairs
    assert candidate["seed_pairs"] == pairs and len(candidate["participants"]) == 1
    assert all(cell["saved_replay_exact"] for cell in binary["cells"] if cell["status"] == "completed")
    assert (tmp_path / "binary/requested.json").is_file() and (tmp_path / "candidate/consumer_requests.json").is_file()


def _fatal_source_group_worker(arguments):
    """Inject native death in the spawned scientific body after one saved seed."""
    from experiments import run_source_training_chain as runner

    actual = runner.fit_or_reuse_source_trajectory
    calls = 0

    def fit_then_die(**kwargs):
        nonlocal calls
        if calls:
            os._exit(77)
        calls += 1
        return actual(**kwargs)

    runner.fit_or_reuse_source_trajectory = fit_then_die
    return runner._source_group_worker(arguments)


def test_native_source_tail_keeps_sealed_artifacts_peer_and_blocked_consumer_denominator(tmp_path, monkeypatch):
    cache, spec, _ = _prepared_source(tmp_path)
    snapshot = freeze_source(tmp_path / "snapshot")
    plan = _plan(cache, spec, snapshot)
    plan.update(cpu_budget=3, source_workers=2, target_workers=2)
    healthy = copy.deepcopy(plan["outer_folds"][0])
    fatal = {**healthy, "name": "leave_BI2015a", "target_study": "BI2015a", "excluded_source_datasets": ["BI2015a"]}
    plan["outer_folds"] = [healthy, fatal]
    collection = tmp_path / "target-declaration.json"
    collection.write_text(json.dumps({"records": [{"record": {"study": "BI2015a", "subject": "1"},
        "protocol": {"kind": "single_recording_prefix"}}]}), encoding="utf-8")
    consumer = tmp_path / "consumer-base.json"
    pairs = [[seed, seed] for seed in plan["source_seeds"]]
    consumer.write_text(json.dumps({"schema": "n2p3_erp_comparator_plan/1", "evidence_scope": "software_correctness",
        "subjects": ["1"], "budgets": [8, 200], "seed_pairs": pairs, "source_seed_catalog": [],
        "source_snapshot": None, "expected_source_config": plan["source_config"], "allowed_input_axes": [],
        "arms": [{"name": family, "family": family, "prefix_collection": str(collection),
            **({"epochs": 1, "batch_size": 32} if family in {"full_fine", "scratch"} else {})}
            for family in ("waveform_lda", "full_fine", "scratch")]}), encoding="utf-8")
    consumer_output = tmp_path / "consumer"
    plan["ready_consumers"] = [{"kind": "binary", "name": "binary-BI2015a", "outer": fatal["name"],
        "plan": str(consumer), "output": str(consumer_output)}]
    actual_run = CloudExecutor.run

    def spawn_with_one_native_failure(self, tasks, **kwargs):
        altered = [replace(task, adapter="tests.train.test_source_ready_queue:_fatal_source_group_worker")
            if task.resources.phase == "source" and task.arguments[0]["outer_folds"][0]["name"] == fatal["name"]
            else task for task in tasks]
        return actual_run(self, altered, **kwargs)

    monkeypatch.setattr(CloudExecutor, "run", spawn_with_one_native_failure)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="requested"):
        execute_source_chain(plan, output=output, dry_run=False, device="cpu", stages=("P2_source",))
    report = json.loads((output / "source_chain_record.json").read_bytes())
    by_outer = {entry["outer"]: entry for entry in report["groups"]}
    assert by_outer[healthy["name"]]["status"] == "completed"
    assert by_outer[fatal["name"]]["status"] == "failed"
    assert by_outer[fatal["name"]]["worker"]["exit_code"] == 77
    assert report["requested_source_task_count"] == 6 and report["source_failure_count"] == 3
    assert {entry["outer_target_study"] for entry in report["source_seed_catalog"]} == {healthy["target_study"]}
    preserved = report["unadmitted_source_artifacts"][0]
    assert len(preserved["source_seed_catalog"]) == 1
    assert Path(preserved["source_seed_catalog"][0]["path"]).is_file()
    requests = json.loads((consumer_output / "requested.json").read_bytes())["requests"]
    result = json.loads((consumer_output / "result.json").read_bytes())
    assert len(requests) == result["requested_cells"] == 18
    assert result["status"] == "failed" and result["physical_target_fits"] == 0
    assert all(row["fit_observation"] == "not_started" for row in result["cells"])
    consumer_outcomes = [row for row in report["cloud_execution"]["results"] if row["task_id"].startswith("consumer-")]
    assert consumer_outcomes and all(row["status"] == "blocked" and row["worker"] is None for row in consumer_outcomes)
    assert all(point["reserved_cpu_threads"] <= 3 for point in report["resource_observations"])
