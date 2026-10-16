"""Actual K0 packages, graph seals and isolated complete source closures."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch

from research.contracts import semantic_sha256
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import _PrefixPackageAdmission, persist_prefix_inner_replay
from research.source_metadata_store import (
    SourceMetadataStore,
    _SourceMetadataAdmission,
    admit_source_dependencies,
    export_source_package_closure,
    source_metadata_scope,
)
from tests.support.checkpoints import write_checkpoint
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    fit_calibration_prefix_programs,
)
from transfer.recipes import TargetRecipe


def _case(tmp_path):
    _, _, _, _, experts, views = independent_prefix_case(tmp_path)
    profile = experts[1].profile
    path = write_checkpoint(tmp_path / "source.pt", views.calibration[200].trial_datasets[profile.digest],
                            source_snapshot_sha256="a" * 64)
    payload = torch.load(path, weights_only=True)
    payload["training_contract"]["optimizer"]["config"] = {"pos_weight": 8.}
    payload["training_contract"]["validation"]["complete_fixture_rows"] = list(range(30000))
    payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
    torch.save(payload, path)
    spec = CalibrationExpertProgram("source", "trial", TargetRecipe("source", initialization="source",
        input_statistics="source", epochs=1, batch_size=512, dropout="off", seed=0), profile)
    return views, path, spec, EnsembleProgram.equal_weights("source-pool", ("source",))


def test_k0_prefix_package_closure_keeps_original_bytes_and_independent_replay(tmp_path):
    with cpu_thread_budget(1):
        views, path, spec, programme = _case(tmp_path)
        recorder = InnerReplayRecorder()
        context = {"source_seed": 0, "target_seed": 0, "requested_Ks": [0],
            "requested_to_actual_K": {"0": 0}, "test_scope": views.ledger["test_scope"],
            "requested_decision_ids": [view.record_id for view in views.decisions]}
        programs = fit_calibration_prefix_programs(views.calibration[0], (spec,), programme,
            strategies=("fixed",), source_checkpoints={"source": path}, inner_replay_recorder=recorder,
            inner_replay_context=context)
        for view in views.decisions:
            recorder.record_prefix_future_score("fixed", view=view, evidence=programs["fixed"].score_experts(view))
        store = SourceMetadataStore(tmp_path / "shared")
        package = tmp_path / "prefix"
        with source_metadata_scope(store):
            persist_prefix_inner_replay(views.calibration[0], programs, package, recorder=recorder)
        source_bytes = path.read_bytes()
        assert next(store.directory.glob("*.pt")).read_bytes() == source_bytes
        assert (package / "expert_000.pt").stat().st_size < len(source_bytes) / 2
        admitted = _PrefixPackageAdmission(package)
        assert admitted._metadata_admission.source_admissions == 1
        admitted.replay(future_views=views.decisions)
        loaded = FittedEnsemble.load(package)
        ensemble_path = package / "ensemble.json"
        original_ensemble = ensemble_path.read_bytes()
        changed = json.loads(original_ensemble)
        changed["record"]["source_metadata"]["references"].append({
            **changed["record"]["source_metadata"]["references"][0], "checkpoint_bytes": len(source_bytes) + 1})
        changed["record_sha256"] = semantic_sha256(changed["record"])
        ensemble_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="bytes or size"):
            FittedEnsemble.load(package)
        ensemble_path.write_bytes(original_ensemble)
        model_path = package / "expert_000.pt"
        original_model = model_path.read_bytes()
        changed_model = torch.load(model_path, weights_only=True)
        reference = changed_model["record"]["class_calibration"]["fit_contract"]["training_contract"]
        reference["target_binding"]["identity"]["records"][0]["origin_subject_keys"] = [
            reference["target_binding"]["identity"]["records"][0]["origin_subject_keys"][0] + "-foreign"]
        from data.identity import DatasetIdentityTable, ParticipantIdentityRecord

        reference["target_binding"]["identity"] = DatasetIdentityTable(tuple(
            ParticipantIdentityRecord.from_payload(row)
            for row in reference["target_binding"]["identity"]["records"])).payload()
        torch.save(changed_model, model_path)
        changed = json.loads(original_ensemble)
        changed["record"]["experts"][0]["model_file_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        changed["record"]["source_metadata"]["references"] = [reference]
        changed["record_sha256"] = semantic_sha256(changed["record"])
        ensemble_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="target differs.*fit authority"):
            FittedEnsemble.load(package)
        ensemble_path.write_bytes(original_ensemble)
        model_path.write_bytes(original_model)
        descriptor = json.loads(original_ensemble)["record"]["source_metadata"]
        transaction = _SourceMetadataAdmission()
        admit_source_dependencies(descriptor, package, admission=transaction)
        second_store = SourceMetadataStore(tmp_path / "foreign-store")
        second_store.publish(source_bytes)
        foreign = {**descriptor, "directory": "../foreign-store"}
        with pytest.raises(ValueError, match="cannot change its actual package store"):
            admit_source_dependencies(foreign, package, admission=transaction)
        standalone = tmp_path / "standalone"
        loaded.export_standalone(standalone)
        isolated = export_source_package_closure(package, tmp_path / "isolated")
        original_seal = hashlib.sha256((package / "prefix_inner_replay.json").read_bytes()).hexdigest()
        assert hashlib.sha256((isolated / "prefix_inner_replay.json").read_bytes()).hexdigest() == original_seal
        assert store.directory.is_relative_to(tmp_path.resolve())
        shutil.rmtree(store.directory)
        assert _PrefixPackageAdmission(isolated).replay(future_views=views.decisions)["status"] == "replayed"
        restored_standalone = FittedEnsemble.load(standalone)
        for view in views.decisions:
            assert semantic_sha256(restored_standalone.score(view).record()) == semantic_sha256(loaded.score(view).record())
        source_file = next((tmp_path / "isolated").rglob("*.pt"))
        assert source_file.is_relative_to(tmp_path.resolve())
        # Remove the actual source blob, not a numerical model.
        source_blob = next(p for p in (tmp_path / "isolated").rglob("*.pt") if p.name == hashlib.sha256(source_bytes).hexdigest() + ".pt")
        source_blob.unlink()
        with pytest.raises(FileNotFoundError):
            _PrefixPackageAdmission(isolated)


def test_selection_graph_source_closure_is_sealed_and_fresh_loadable(tmp_path):
    from research.calibration_selection import (
        CalibrationProgramCandidate,
        fit_nested_prefix_selection,
    )
    from research.prefix_selection_replay import _SelectionReplayAdmission

    with cpu_thread_budget(1):
        views, path, spec, programme = _case(tmp_path)
        candidate = CalibrationProgramCandidate("source", (spec,), programme)
        selection = fit_nested_prefix_selection(views.calibration[200], (candidate,),
            baseline_name="source", minimum_selection_rows=1, prefix_compiler=views.prefix_compiler,
            source_checkpoints={"source": path}, context={"source_seed": 0, "target_seed": 0,
                "test_scope": views.ledger["test_scope"], "requested_decision_ids": [view.record_id for view in views.decisions]})
        for view in views.decisions:
            selection.score(view)
        store = SourceMetadataStore(tmp_path / "shared")
        graph_dir = tmp_path / "graph"
        with source_metadata_scope(store):
            selection.save(graph_dir)
        payload = json.loads((graph_dir / "prefix_selection.json").read_bytes())
        assert payload["record"]["source_metadata"]["references"]
        original = _SelectionReplayAdmission(graph_dir, expected_graph_sha256=selection.graph_sha256)
        assert original._metadata_admission.source_admissions == 1
        assert original.load().record == selection.record
        isolated = export_source_package_closure(graph_dir, tmp_path / "isolated")
        assert store.directory.is_relative_to(tmp_path.resolve())
        shutil.rmtree(store.directory)
        restored = _SelectionReplayAdmission(isolated, expected_graph_sha256=selection.graph_sha256)
        assert restored.load().record == selection.record
        changed = json.loads((isolated / "prefix_selection.json").read_bytes())
        changed["record"].pop("source_metadata")
        changed["record_sha256"] = semantic_sha256(changed["record"])
        (isolated / "prefix_selection.json").write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="source closure"):
            _SelectionReplayAdmission(isolated)


def test_candidate_finalization_requires_original_admitted_source_blob(tmp_path):
    from experiments.run_candidate_ensemble_study import (
        CandidateCloudProgramme,
        _verify_candidate_source_artifacts,
    )

    blob = tmp_path / "original.pt"
    content = b"original independently admitted source artifact"
    blob.write_bytes(content)
    receipt = {"path": str(blob), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    participant = {"source_metadata_artifacts": [receipt]}
    _verify_candidate_source_artifacts((participant, participant))
    programme = CandidateCloudProgramme.__new__(CandidateCloudProgramme)
    programme.admitted = {"first": participant, "second": participant}
    programme.input_admission, programme.replay_pool = None, None
    blob.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError, match="source bytes changed.*finalization"):
        programme.finalize({})
    blob.unlink()
    with pytest.raises(FileNotFoundError):
        programme.finalize({})
