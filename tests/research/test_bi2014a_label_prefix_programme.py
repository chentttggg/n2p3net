"""Original BI MAT label-K reaches the real quartet, capsule and public parent."""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from research.contracts import semantic_sha256
from tests.data.test_bi2014a_original_recording import source
from tests.research.test_transfer_scratch_prefix_family import declared_plan
from train.runtime import cpu_thread_budget
from transfer.ensemble_views import SelectionEpochProfile


def label_prefix_samples():
    depths = [1, 2, 1, 3, 2, 1, 2, 2]
    onsets, codes, boundaries = [], [], []
    position = 512
    for selection, depth in enumerate(depths):
        row, col = selection % 6, (selection + 2) % 6
        block = [60 + row, 80 + col, *[20 + i for i in range(6) if i != row],
                 *[40 + i for i in range(6) if i != col]]
        for _ in range(depth):
            for code in block:
                onsets.append(position)
                codes.append(code)
                position += 96
        boundaries.append(position - 20)
        position += 900
    values = np.zeros((position + 700, 19))
    values[:, 0] = np.arange(len(values)) / 512
    values[:, 1:17] = np.random.default_rng(31).normal(0, 3, (len(values), 16))
    values[onsets, 17] = codes
    values[onsets, 18] = np.where(np.asarray(codes) >= 60, 2, 1)
    values[boundaries, 17] = 104
    return values


def test_bi_mat_actual_quartet_saved_capsule_and_zero_fit_parent(tmp_path, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from experiments.run_bi2014a_label_prefix_study import (
        prepare_bi2014a_label_prefix_plan,
        run_bi2014a_label_prefix_study,
    )
    from research.cloud_execution import CloudResult
    from research.prefix_replay_inputs import (
        PrefixReplayInputHolder,
        open_prepared_inputs,
        read_scoring_truth,
    )
    from research.source_execution import SourceInputAdmission, _SourceInputState
    from transfer import ensemble_fit as fit_owner
    from transfer.ensemble_views import prepare_dataset_independent_prefixes

    raw, archive_sha = source(tmp_path, label_prefix_samples())
    declaration = {"dataset_id": "BI2014a", "source_format": "original_mat_zip", "record_sources": [
        {"path": str(raw), "sha256": archive_sha, "subject": "01", "member": "subject_01.mat"}]}
    source_entry_path = os.environ.get("N2P3_BI_SOURCE_ENTRY")
    if source_entry_path is None:
        pytest.skip("the actual BI-excluded source locator is supplied for this original-source vertical")
    entry = json.loads(Path(source_entry_path).read_bytes())
    assert entry["outer_target_study"] == "BI2014a"
    seed = entry["source_seed"]
    from experiments.prepare_bi2014a_candidate import source_profile_from_checkpoint

    preprocessing, channels, reference, source_profile_receipt = source_profile_from_checkpoint(
        entry["path"], expected_sha256=entry["sha256"])
    assert reference == "measured_channel:CZ"
    profile = SelectionEpochProfile.from_preprocessing(preprocessing, source_sample_rate_hz=512.,
        channel_names=channels, reference="right earlobe",
        reference_channel="CZ", allow_missing_channels=True)
    with cpu_thread_budget(1):
        views = prepare_dataset_independent_prefixes(declaration, budgets=(0, 48, 2000), profiles=(profile,),
            authorization_preprocessing=preprocessing, response_seconds=1.203125, pre_roll_seconds=.2)
    base = declared_plan(profile)
    from transfer.ensemble_fit import CalibrationExpertProgram, FrozenSourceCheckpoint

    base["experts"] = [replace(CalibrationExpertProgram.from_record(row),
        recipe=replace(CalibrationExpertProgram.from_record(row).recipe, seed=seed)).record() for row in base["experts"]]
    base.update(dataset_id="BI2014a", evidence_scope="software_correctness", budgets=[0, 48, 2000],
        seed_pairs=[[seed, seed]], source_seed_catalog=[entry],
        expected_source_config={"pos_weight": 1., "val_groups_min": 2, "val_groups_max": None},
        authorization_preprocessing=preprocessing.record(), observation_response_seconds=1.203125,
        pre_roll_seconds=.2, device="cpu", workers=1, cpu_threads_per_worker=1,
        source_checkpoints={}, calibration_strategies=["fixed"])
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    plan_path = tmp_path / "bi-label-K.json"
    plan = prepare_bi2014a_label_prefix_plan(base_path, plan_path, raw_root=tmp_path, subjects=[1],
        source_format="original_mat_zip", source_expert_id="pretrained-A", scratch_expert_id="random-B",
        xdawn_expert_id="spatial-C", baseline_name="source_nn_fixed", output_name="BI-fixed-quartet-selected",
        minimum_selection_rows=1)
    original_fit, fits = fit_owner._fit_expert, []

    def counted_fit(*args, **kwargs):
        fitted = original_fit(*args, **kwargs)
        fits.append({"expert_id": args[0].expert_id, "initialization": args[0].recipe.initialization,
                     "labels": 0 if kwargs.get("source_only", False) else args[1].n_epochs,
                     "source_only": kwargs.get("source_only", False)})
        return fitted

    monkeypatch.setattr(fit_owner, "_fit_expert", counted_fit)
    from transfer import ensemble_views as view_owner

    execution_gate, source_family_receipts = view_owner.validate_selection_checkpoint_execution, {}

    def recorded_execution_gate(*args, **kwargs):
        receipt = execution_gate(*args, **kwargs)
        for relation in receipt["source_sos_family_relations"]:
            source_family_receipts[semantic_sha256(relation)] = relation
        return receipt

    monkeypatch.setattr(view_owner, "validate_selection_checkpoint_execution", recorded_execution_gate)
    with cpu_thread_budget(1):
        report = run_bi2014a_label_prefix_study(plan_path, tmp_path / "run", execution="inline")
    assert report["status"] == "completed" and report["participants"][0]["participant_key"] == views.ledger["test_views"][0]["participant_key"]
    participant = report["participants"][0]
    assert set(row["initialization"] for row in fits) >= {"source", "scratch", "xdawn_erp"}
    actual_K = participant["requested_to_actual_K"]["2000"]
    assert 0 < actual_K < 2000 and actual_K in views.calibration
    selected = [row for row in participant["predictions"] if row["calibration_strategy"] == "nested_selection"]
    assert len(selected) == 3 * len(participant["requested"])
    assert all(row["saved_replay_exact"] for row in selected)
    expected_levels = {view.decision_id: view.decision.repetitions for view in views.decisions}
    assert all(row["evidence"]["evidence_level"] == expected_levels[row["evidence"]["decision_id"]]
               for row in selected)
    directory = tmp_path / "run/participant_000"
    from research.selection_graph_store import SelectionGraphStore

    package = f"source{seed}_target{seed}_K{actual_K}_selection"
    saved = runner._read_sealed_record(directory / "selection_fit_receipts" / f"{package}.saved.json")
    replayed = runner._read_sealed_record(directory / "selection_fit_receipts" / f"{package}.replayed.json")
    graph = SelectionGraphStore(directory / package / "prefix_selection.json",
        expected_graph_sha256=saved["graph_sha256"],
        expected_selection_sha256=replayed["selection_sha256"]).graph
    assert {node["candidate"]["name"] for node in graph["nodes"]} == {
        "source_nn_fixed", "scratch_nn_fixed", "source_nn_xdawn_fixed", "scratch_nn_xdawn_fixed"}

    def no_fit(*args, **kwargs):
        raise AssertionError("BI original model/capsule/public parent replay must not fit EEG")

    monkeypatch.setattr(fit_owner, "_fit_expert", no_fit)
    normalized = json.loads((tmp_path / "run/plan.json").read_bytes())
    executed = json.loads((tmp_path / "run/executed_source.json").read_bytes())
    task = SimpleNamespace(output=str(tmp_path / "run"), arguments={"record_index": 0}, resources=SimpleNamespace(native_threads=1))
    outcome = SimpleNamespace(status="completed", value=str(directory / "result.json"))
    snapshots = {("BI2014a", seed): FrozenSourceCheckpoint.from_path(entry["path"])}
    with cpu_thread_budget(1):
        admitted = runner._admit_candidate_participant(task, outcome, normalized, executed, source_snapshots=snapshots)
    assert admitted["status"] == "completed" and len(admitted["predictions"]) == len(participant["predictions"])
    compiler = views.prefix_compiler
    request = {"schema": "n2p3_prefix_replay_input_request/2", "preparation_task_id": "BI-original-MAT",
        "original_source_sha256": compiler._arguments["target"]["source_sha256"],
        "producer_source_sha256": compiler.history_owner_source,
        "participant_key": compiler._arguments["target"]["participant_key"],
        "source_relative_path": compiler._arguments["target"]["source_relative_path"],
        "requested_Ks": [0, 48, 2000], "actual_Ks": list(views.calibration),
        "profiles": [profile.record()], "authorization_preprocessing": preprocessing.record(),
        "observation_response_seconds": 1.203125, "pre_roll_seconds": .2, "dataset_source": declaration}
    control = tmp_path / "capsule-control.json"
    control.write_text(json.dumps({"request": request, "output": str(tmp_path / "capsule")}), encoding="utf-8")
    producer = tmp_path / "original-capsule-producer.py"
    producer.write_text("""import json, os, sys
from pathlib import Path
import psutil
from data.epochs import parse_preprocessing_spec
from research.prefix_replay_inputs import export_prepared_views
from train.runtime import cpu_thread_budget
from transfer import ensemble_fit
from transfer.ensemble_views import SelectionEpochProfile, prepare_dataset_independent_prefixes
control = json.loads(Path(sys.argv[1]).read_bytes())
request = control['request']
def no_fit(*args, **kwargs):
    raise AssertionError('original capsule preparation must not fit EEG')
ensemble_fit._fit_expert = no_fit
profiles = tuple(SelectionEpochProfile.from_record(p) for p in request['profiles'])
with cpu_thread_budget(1):
    views = prepare_dataset_independent_prefixes(request['dataset_source'], budgets=request['requested_Ks'],
        profiles=profiles, authorization_preprocessing=parse_preprocessing_spec(request['authorization_preprocessing']),
        response_seconds=request['observation_response_seconds'], pre_roll_seconds=request['pre_roll_seconds'])
    descriptor = export_prepared_views(views, profiles, Path(control['output']), request)
assert not psutil.Process().children(recursive=True)
Path(control['output']).joinpath('actual-preparation.json').write_text(json.dumps({
    'descriptor': descriptor, 'pid': os.getpid(), 'birth': psutil.Process().create_time(),
    'new_fits': 0, 'active_children': 0}), encoding='utf-8')
""", encoding="utf-8")
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [sys.executable, str(producer), str(control)]
    process = subprocess.Popen(command, env=environment)
    owner = psutil.Process(process.pid)
    observed = {}
    while process.poll() is None:
        try:
            children = [owner, *owner.children(recursive=True)]
        except psutil.Error:
            children = [owner]
        for child in children:
            try:
                observed[child.pid, child.create_time()] = child
            except psutil.Error:
                pass
        time.sleep(.05)
    returncode = process.wait()
    active_owned = []
    for (pid, birth), child in observed.items():
        try:
            if child.is_running() and child.create_time() == birth:
                active_owned.append({"pid": pid, "birth": birth})
        except psutil.Error:
            pass
    preparation = json.loads((tmp_path / "capsule/actual-preparation.json").read_bytes())
    capsule = preparation["descriptor"]
    assert returncode == 0 and preparation["new_fits"] == preparation["active_children"] == 0
    assert not active_owned and (preparation["pid"], preparation["birth"]) in observed
    capsule_path = Path(capsule["path"])
    document = json.loads(capsule_path.read_bytes())
    resident_bytes = (capsule_path.parent / document["blob"]["name"]).stat().st_size + capsule_path.stat().st_size
    admission = SourceInputAdmission(SimpleNamespace(SourceInputState=_SourceInputState),
                                     capacity_bytes=resident_bytes, load_workers=1)
    assert admission.record()["resident_input_bytes"] == 0
    assert admission.record()["free_staging_bytes"] == resident_bytes
    capsule_outcome = CloudResult(request["preparation_task_id"], "completed", value=capsule, worker={
        "backend": "observed_local_software_capsule", "executor": "local subprocess; no CloudExecutor",
        "alive": False, "exit_code": returncode, "command": command,
        "scientific_child": {"pid": preparation["pid"], "birth": preparation["birth"]},
        "driver_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
        "control_sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
        "process_tree_retirement": {"active_owned_processes_after_exit": len(active_owned)}})
    holder = PrefixReplayInputHolder.from_prepared(capsule, request_sha=semantic_sha256(request),
        prep_outcome=capsule_outcome, admission=admission)
    assert holder is not None
    assert admission.record()["resident_input_bytes"] == resident_bytes
    assert admission.record()["free_staging_bytes"] == 0
    try:
        with cpu_thread_budget(1), open_prepared_inputs(holder.descriptor, expected_request_sha=semantic_sha256(request),
                expected_capsule_sha=capsule["sha256"]) as borrowed:
            assert [view.record() for view in borrowed.views.decisions] == [view.record() for view in views.decisions]
            for budget in views.calibration:
                assert borrowed.views.calibration[budget].record() == views.calibration[budget].record()
            assert borrowed.views.scoring_truth == {}
            truth_path = tmp_path / "capsule/scoring_truth.json"
            truth = {"path": str(truth_path), "sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                     "bytes": truth_path.stat().st_size}
            assert read_scoring_truth(truth, capsule_sha=capsule["sha256"], request_sha=semantic_sha256(request),
                                     future_views=borrowed.views.decisions) == views.scoring_truth
    finally:
        holder.close()
    assert admission.record()["resident_input_bytes"] == 0
    assert admission.record()["free_staging_bytes"] == resident_bytes
    (tmp_path / "BI_label_K_readout.json").write_text(json.dumps({"status": admitted["status"],
        "requested": len(participant["requested"]), "predictions": len(participant["predictions"]),
        "requested_to_actual_K": participant["requested_to_actual_K"], "actual_software_expert_fit_calls": len(fits),
        "actual_fit_inputs": fits, "source_digest": semantic_sha256(executed), "original_archive_sha256": archive_sha,
        "graph_nodes": len(graph["nodes"]), "capsule": capsule, "parent_new_fits": 0,
        "reused_original_source": entry, "source_profile_receipt": source_profile_receipt, "new_source_fits": 0,
        "actual_source_sos_family_receipts": list(source_family_receipts.values())}), encoding="utf-8")
    receipt = next(path for path in (directory / "prediction_receipts").glob("*.json")
        if runner._read_sealed_record(path)["calibration_strategy"] == "nested_selection"
        and runner._read_sealed_record(path)["evidence"]["log_probability"] is not None)
    original = receipt.read_bytes()
    changed = runner._read_sealed_record(receipt)
    attack = np.arange(36, dtype=np.float64) / 36
    attack -= np.log(np.exp(attack).sum())
    changed["evidence"]["log_probability"] = attack.tolist()
    runner._sealed_record(receipt, changed)
    with cpu_thread_budget(1), pytest.raises(ValueError, match="nested prefix prediction"):
        runner._admit_candidate_participant(task, outcome, normalized, executed, source_snapshots=snapshots)
    receipt.write_bytes(original)
    assert json.loads(plan_path.read_bytes()) == plan
