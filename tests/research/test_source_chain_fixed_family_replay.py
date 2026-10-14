"""Actual source producer, four fixed prefix families and one complete replay DAG."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_SOURCE_REFERENCE
from experiments.freeze_source_snapshot import freeze_source
from research.calibration_selection import transfer_scratch_prefix_plan
from research.cloud_execution import assert_no_prior_owned_live
from research.contracts import semantic_sha256
from research.evaluation import verified_executed_snapshot
from tests.support.bnci import bnci_mat_fields
from tests.support.source_programs import independent_source_plan, prepared_finite_source
from transfer.checkpoint import load_checkpoint_payload
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, FrozenSourceCheckpoint
from transfer.ensemble_views import SelectionEpochProfile
from transfer.recipes import TargetRecipe, erp_target_recipe


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sealed(path):
    payload = _json(path)
    assert payload["record_sha256"] == semantic_sha256(payload["record"])
    return payload["record"]


def _run_measured(command, output):
    """Sample the full CLI tree, including launcher/compiler descendants."""
    peaks, samples, owned_peak, min_available = {}, 0, 0, psutil.virtual_memory().available
    started = time.monotonic()
    with (output / "source_cli.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            live = {}
            for item in processes:
                try:
                    live[item.pid] = (item.create_time(), item.ppid(), item.memory_info().rss)
                except psutil.NoSuchProcess:
                    continue
            samples += 1
            owned_peak = max(owned_peak, sum(row[2] for row in live.values()))
            min_available = min(min_available, psutil.virtual_memory().available)
            for pid, (birth, _, rss) in live.items():
                descendants = {pid}
                while True:
                    expanded = descendants | {key for key, row in live.items() if row[1] in descendants}
                    if expanded == descendants:
                        break
                    descendants = expanded
                peak = peaks.setdefault((pid, birth), {"pid": pid, "create_time": birth,
                    "self_peak_bytes": 0, "whole_owned_tree_peak_bytes": 0, "sample_count": 0})
                peak["self_peak_bytes"] = max(peak["self_peak_bytes"], rss)
                peak["whole_owned_tree_peak_bytes"] = max(peak["whole_owned_tree_peak_bytes"],
                    sum(live[key][2] for key in descendants))
                peak["sample_count"] += 1
            time.sleep(.02)
        exit_code = process.wait()
    measurement = {"schema": "n2p3_fixed_family_owned_tree_measurement/1", "sample_period_seconds": .02,
        "sample_count": samples, "whole_cli_tree_peak_bytes": owned_peak,
        "minimum_available_host_bytes": min_available, "wall_seconds": time.monotonic() - started,
        "actual_exit_code": exit_code, "entities": list(peaks.values()),
        "basis": "sampled simultaneous RSS of CLI parent and all recursive descendants; per-entity whole-tree peaks"}
    (output / "whole_owned_tree_measurement.json").write_text(json.dumps(measurement), encoding="utf-8")
    return exit_code, measurement


def test_actual_source_chain_four_family_replay_keeps_original_source_and_complete_dag(tmp_path):
    cache, preprocessing, _ = prepared_finite_source(tmp_path)
    raw = tmp_path / "A01.mat"
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(39).normal(0, 4, fields["X"].shape).astype(np.float32)
    savemat(raw, {"data": fields})
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    channels = ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
    profile = SelectionEpochProfile.from_preprocessing(preprocessing, source_sample_rate_hz=256.,
        channel_names=channels, reference=BNCI2014_008_SOURCE_REFERENCE, reference_channel="Cz")
    xdawn_profile = replace(profile, preprocessing=replace(preprocessing, baseline_mode="mean_only"))
    source_recipe = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=1,
        batch_size=128, bn_buffers="calibration", dropout="off", seed=21,
        trial_loss_reduction="calibration_balanced_fixed_mass")
    experts = (CalibrationExpertProgram("NN", "trial", source_recipe, profile),
        CalibrationExpertProgram("scratch", "trial", replace(source_recipe, name="scratch",
            initialization="scratch", input_statistics="target_prefix"), profile),
        CalibrationExpertProgram("xdawn", "trial", erp_target_recipe("xdawn_erp", bin_width=16, seed=21),
            xdawn_profile))
    mixtures = [EnsembleProgram.equal_weights(name, (name,)).record() for name in ("NN", "scratch", "xdawn")]
    expected = {"pos_weight": 8., "val_groups_min": 2, "val_groups_max": 12}
    candidate_plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(raw), "sha256": raw_sha}],
        "budgets": [0, 800], "seed_pairs": [[11, 21]], "source_seed_catalog": [], "source_snapshot_manifest": None,
        "expected_source_config": expected, "source_checkpoints": {}, "experts": [expert.record() for expert in experts],
        "mixtures": mixtures, "baseline_mixture": "NN", "authorization_preprocessing": preprocessing.record(),
        "observation_response_seconds": 308 / 256, "pre_roll_seconds": .2, "device": "cpu",
        "workers": 1, "cpu_threads_per_worker": 1, "calibration_strategies": ["fixed"],
        "prefix_replay_graph_memory_gib": 1.25, "prefix_replay_worker_memory_gib": 1.25}
    candidate_plan = transfer_scratch_prefix_plan(candidate_plan, source_expert_id="NN",
        scratch_expert_id="scratch", xdawn_expert_id="xdawn", baseline_name="scratch_nn_fixed",
        output_name="selected-family", minimum_selection_rows=1)
    candidate_path = tmp_path / "candidate-plan.json"
    candidate_path.write_text(json.dumps(candidate_plan), encoding="utf-8")
    snapshot = freeze_source(tmp_path / "snapshot", materialize=True)
    source_plan = independent_source_plan(cache, preprocessing, snapshot)
    source_plan.update(source_seeds=[11], target_seeds=[21], seed_pairs=[[11, 21]],
        cpu_budget=2, source_workers=1, target_workers=1,
        source_worker_memory_gib=1.25, target_worker_memory_gib=1.25, shared_staging_memory_gib=.125,
        ready_consumers=[{"kind": "candidate", "name": "fixed-family", "outer": "leave_BNCI",
            "plan": str(candidate_path), "output": str(tmp_path / "candidate")}])
    source_path = tmp_path / "source-plan.json"
    source_path.write_text(json.dumps(source_plan), encoding="utf-8")
    command = [sys.executable, "-B", str(snapshot.parent / "checkout/experiments/run_source_training_chain.py"),
        "--plan", str(source_path), "--output", str(tmp_path / "source-run"), "--stages", "P2_source", "--device", "cpu"]
    code, measurement = _run_measured(command, tmp_path)
    assert code == 0, (tmp_path / "source_cli.log").read_text(encoding="utf-8")
    report = _json(tmp_path / "source-run/source_chain_record.json")
    assert report["status"] == "completed" and report["cloud_execution"]["status"] == "completed"
    assert report["execution_end_source_check"] == "actual_runtime_bytes_match_the_ingress_archive"
    verified_executed_snapshot(snapshot, root=Path(__file__).resolve().parents[2])
    resources = report["resource_policy"]
    assert resources["native_threads_per_worker"] == 1 and resources["cpu_budget"] == 2
    assert resources["source_workers"] == resources["target_workers"] == 1
    assert resources["cloud_resource_floor"] is None

    consumer, = report["consumers"]
    result = _json(tmp_path / "candidate/result.json")
    assert consumer["status"] == result["status"] == "completed"
    participant, = result["participants"]
    assert result["decision_denominator_complete"] and participant["status"] == "completed"
    assert participant["selection_inflight_programmes"] == participant["selection_failed_programmes"] == []
    requested = participant["requested"]
    assert len(participant["pipeline_requests"]) == len(participant["predictions"]) == 8 * len(requested)
    assert {row["status"] for row in participant["pipeline_requests"]} <= {"completed", "output_unavailable"}
    missing_zero = [row for row in participant["pipeline_requests"]
        if row["actual_K"] == 0 and row["mixture"] in {"scratch", "xdawn", "selected-family"}]
    assert len(missing_zero) == 3 * len(requested) and all(row["status"] == "output_unavailable" for row in missing_zero)
    for metric in result["metrics"].values():
        assert [row["requested_decisions"] for row in metric["participants"]] == [len(requested)]
    ledger = _sealed(tmp_path / "candidate/participant_groups/participant_000/participant_000/request_ledger.json")
    scopes = _json(tmp_path / "candidate/participant_groups/participant_000/participant_000/scopes.json")
    assert ledger["requested"] == requested and len(requested) == len(scopes["test_views"])

    source_entry, = report["source_seed_catalog"]
    source_bytes = Path(source_entry["path"]).read_bytes()
    frozen = FrozenSourceCheckpoint(source_bytes)
    source_digest = source_entry["sha256"]
    assert frozen.content is source_bytes and frozen.digest == source_digest
    control, = (tmp_path / "candidate/replay_sources").glob("*.pt")
    assert control.read_bytes() == source_bytes and control.stem == source_digest
    source = load_checkpoint_payload(io.BytesIO(frozen.content))
    assert source["training_contract"]["seed"] == 11
    assert source["source_epoch_selection"]["refit_initialization"] == "fresh_same_seed"
    assert source["source_epoch_selection"]["outer_target_used_for_selection"] is False
    assert source["training_contract"]["objective"]["source_program_config"]["declared"] == expected
    assert report["completed_physical_fit_count"] == report["completed_logical_fit_count"] == 2
    directory = tmp_path / "candidate/participant_groups/participant_000/participant_000"
    graph = _sealed(directory / "source11_target21_K800_selection/prefix_selection.json")
    selection = graph["selection"]
    names = {"source_nn_fixed", "scratch_nn_fixed", "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"}
    assert {node["candidate"]["name"] for node in graph["nodes"]} == names
    final_trace = selection["final_selection"]
    assert len(final_trace["requested_held_record_ids"]) == 5
    assert set(final_trace["candidate_selection_rows"]) == names
    assert all([row["held_record_id"] for row in rows] == final_trace["requested_held_record_ids"]
        for rows in final_trace["candidate_selection_rows"].values())
    assert all(row["training_scope"]["actual_K"] < 800
        for rows in final_trace["candidate_selection_rows"].values() for row in rows)
    assert any(row["status"] == "scored" for rows in final_trace["candidate_selection_rows"].values() for row in rows)
    assert selection["outer_scores_used_for_final_selection"] is False
    assert selection["parent_scope"] == selection["final_refit_scope"]
    assert len(selection["outer_requested"]) == len(selection["outer_rows"]) == 6
    assert selection["source_checkpoint_sha256"] == {"NN": source_digest, "scratch": source_digest}
    fitted_names = set()
    for node in graph["nodes"]:
        if node["scope"]["actual_K"] != 120:
            continue
        package = directory / "source11_target21_K800_selection" / node["package"]
        fitted = FittedEnsemble.load(package)
        prefix_receipt = _sealed(package / "prefix_inner_replay.json")
        assert prefix_receipt["actual_source_seeds"] == {name: {"seed": 11, "digest": source_digest} for name in
            node["candidate"]["ensemble_program"]["expert_ids"] if name in {"NN", "scratch"}}
        for expert in fitted._experts:
            assert expert.predictor is not None
            native = expert.predictor.record
            fit_contract = native["class_calibration"]["fit_contract"]
            assert fit_contract["recipe"]["seed"] == expert.program.recipe.seed == 21
            assert semantic_sha256(fit_contract["recipe"]) == semantic_sha256(expert.program.recipe.record())
            if expert.program.expert_id in {"NN", "scratch"}:
                fitted_names.add(node["candidate"]["name"])
                assert semantic_sha256(native["architecture"]) == semantic_sha256(source["architecture"])
                assert native["class_calibration"]["origin"] == "target_calibration"
                assert fit_contract["class_risk"]["loss_reduction"] == "calibration_balanced_fixed_mass"
                expected_role = "pretrained_weights" if expert.program.expert_id == "NN" else "architecture_only"
                assert prefix_receipt["source_checkpoint_roles"][expert.program.expert_id] == {
                    "digest": source_digest, "role": expected_role}
                assert expert.source_digest == (source_digest if expert.program.expert_id == "NN" else None)
            else:
                assert expert.program.profile.digest == xdawn_profile.digest != profile.digest
                assert native["class_calibration"]["origin"] == "target_generative"
        assert prefix_receipt["actual_K"] == 120
    assert fitted_names == names

    outcomes = report["cloud_execution"]["results"]
    by_id = {row["task_id"]: row for row in outcomes}
    assert len(by_id) == len(outcomes)
    assert set(by_id) == set(consumer["requested_tasks"]) | {report["groups"][0]["task_id"]}
    assert len(set(consumer["requested_tasks"])) == len(consumer["requested_tasks"])
    assert len(consumer["participant_task_ids"]) == 1
    assert sum(key.endswith(":replay_prepare") for key in by_id) == 1
    assert any(":node_" in key for key in by_id) and any(":ordinary_" in key for key in by_id)
    assert sum(":reduce_selection_" in key for key in by_id) == 2
    assert sum(":reduce_ordinary_" in key for key in by_id) == 2
    assert sum(key.endswith(":admission") for key in by_id) == 1
    assert len(result["fit_execution"]) == 1 and result["replay_execution"]
    assert result["replay_inputs"] == {"pending_preparations": [], "resident_preparations": [],
        "active_readers": [], "failed_preparations": {}}
    assert report["source_input_admission"]["resident_input_bytes"] == 0
    assert report["source_input_admission"]["load_active"] == report["source_input_admission"]["nn_active"] == 0
    for outcome in outcomes:
        assert outcome["status"] == "completed" and outcome["failure"] is None
        worker = outcome["worker"]
        assert worker["exit_code"] == 0 and worker["alive"] is False
        assert worker["process_tree_retirement"]["active_owned_processes_after_exit"] == 0
    assert_no_prior_owned_live(report["worker_diagnostics"]["directory"])
    entities = {(row["pid"], row["create_time"]): row for row in measurement["entities"]}
    task_peaks = {key: entities.get((row["worker"]["pid"], row["worker"]["create_time"]))
        for key, row in by_id.items()}
    measurement["task_owned_tree_peaks"] = task_peaks
    measurement["task_sampling_misses"] = [key for key, row in task_peaks.items() if row is None]
    measurement["source_freeze"] = _json(snapshot)
    (tmp_path / "whole_owned_tree_measurement.json").write_text(json.dumps(measurement), encoding="utf-8")
    assert not measurement["task_sampling_misses"], "20ms sampling missed actual task identities; peaks are unmeasured"
    assert all(row["whole_owned_tree_peak_bytes"] <= int(1.25 * 2**30) for row in task_peaks.values())
    assert all(point["reserved_cpu_threads"] <= 2 and point["free_worker_memory_bytes"] >= 0
        for point in report["resource_observations"])
