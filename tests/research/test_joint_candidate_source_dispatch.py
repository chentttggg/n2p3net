"""Join a real original-source entry to generic candidate requests without fits."""

import hashlib
import json

import pytest

from tests.research.test_joint_source_catalog import (
    _entry,
    _no_fit,
    _record_issuer,
)
from tests.research.test_joint_source_catalog import (
    original_source as _original_source,
)

original_source = _original_source


def _plan(fixture, directory):
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import CalibrationExpertProgram
    from transfer.joint_flat_programme import JointFlatProgram, JointFlatSamplingProgram

    entry = _entry(fixture)
    path = directory / "original-joint-entry.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    spec = CalibrationExpertProgram("joint", "joint_flat", joint_flat=JointFlatProgram(
        JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 1.2, 5., 5., .1), 4,
        "whole_observed_current_q"))
    plan = {"evaluation_design": "independent_prefix", "dataset_id": "tiny-software-study",
        "evidence_scope": "software_correctness", "seed_pairs": [[71, 83]],
        "calibration_strategies": ["fixed"],
        "experts": [spec.record()], "source_seed_catalog": [],
        "mixtures": [EnsembleProgram.equal_weights("joint-fixed", ("joint",)).record()],
        "joint_source_catalog": [{"source_id": entry["source_id"], "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        "joint_source_bindings": {fixture.target["participant_key"]: {"joint": entry["source_id"]}}}
    return plan, spec, path


def test_original_joint_source_reaches_generic_fit_roster_with_its_own_role(
        original_source, tmp_path, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from transfer import checkpoint
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.joint_flat_programme import FrozenJointFlatSource

    plan, spec, _ = _plan(original_source, tmp_path)
    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    monkeypatch.setattr(checkpoint, "load_checkpoint_payload", _no_fit)
    sources, receipts = runner._joint_sources_for_target(plan, (spec,), original_source.target,
                                                        pipeline_pair=(71, 83))
    assert isinstance(sources["joint"], FrozenJointFlatSource)
    assert sources["joint"].digest == original_source.source.digest
    assert receipts["joint"]["actual_source_seed"] is None
    assert tuple(receipts["joint"]["pipeline_pair"]) == (71, 83)
    assert receipts["joint"]["resolution_source_fits"] == receipts["joint"]["resolution_target_fits"] == 0
    target, _ = _record_issuer("target", seed=91)
    prefix = target.compile(0).record()
    requests = runner._expected_fit_roster(plan, {"calibration": {"0": prefix}})
    request = requests["source71_target83_K0"]
    assert request["source_checkpoint_sha256"] == {"joint": original_source.source.digest}
    assert spec.source_checkpoint_role == "original_source_response_and_noise_parameters"
    assert spec.source_only_capable


def test_generic_joint_source_resolution_rechecks_target_and_original_entry_bytes(
        original_source, tmp_path, monkeypatch):
    from experiments import run_candidate_ensemble_study as runner
    from transfer.continuous_decision import ContinuousDecisionModel

    plan, spec, path = _plan(original_source, tmp_path)
    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    changed = {**original_source.target, "reference": "another actual reference"}
    with pytest.raises(ValueError, match="physical observation"):
        runner._joint_sources_for_target(plan, (spec,), changed, pipeline_pair=(71, 83))
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="declared original bytes"):
        runner._joint_sources_for_target(plan, (spec,), original_source.target, pipeline_pair=(71, 83))


def test_bound_candidate_plan_preserves_relative_joint_locator_for_actor_and_recovery(tmp_path):
    from experiments.run_candidate_ensemble_study import CandidateCloudProgramme
    from research.prefix_recovery import _original_bound_plan
    from research.source_execution import resolve_source_resources
    from tests.support.candidate_programs import raw_plan

    path = raw_plan(tmp_path)
    entry_path = tmp_path / "joint.entry.json"
    entry_path.write_text("{}", encoding="utf-8")
    entry_sha = hashlib.sha256(entry_path.read_bytes()).hexdigest()
    plan = json.loads(path.read_bytes())
    plan["joint_source_catalog"] = [{"source_id": "original-source", "path": entry_path.name, "sha256": entry_sha}]
    path.write_text(json.dumps(plan), encoding="utf-8")
    resources = resolve_source_resources({**plan, "cpu_budget": 2, "source_workers": 1, "target_workers": 1,
        "source_worker_memory_gib": .25, "target_worker_memory_gib": .25}, 1, "cpu",
        cpu_count=2, free_host_bytes=8 * 2**30)
    programme = CandidateCloudProgramme(path, tmp_path / "another-directory/run", resources,
                                       task_prefix="locator", dependencies=())
    bound = json.loads(programme.plan_path.read_bytes())
    assert bound["joint_source_catalog"] == [{"source_id": "original-source", "path": str(entry_path), "sha256": entry_sha}]
    assert programme.base_plan["joint_source_catalog"][0]["path"] == entry_path.name
    receipt = _original_bound_plan({"plan": {"path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}, programme.output)
    assert receipt["sha256"] == hashlib.sha256(programme.plan_path.read_bytes()).hexdigest()


def test_selector_transport_uses_candidate_rosters_and_keeps_source_families_distinct(tmp_path):
    from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from experiments.run_candidate_ensemble_study import _prefix_replay_source_artifacts
    from research.prefix_replay_execution import artifact_receipt
    from tests.research.test_joint_flat_replay_contracts import _programme
    from tests.research.test_transfer_scratch_prefix_family import declared_plan
    from transfer.ensemble_views import SelectionEpochProfile

    profile = SelectionEpochProfile.from_preprocessing(
        preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS,
        reference=BNCI2014_008_SOURCE_REFERENCE)
    nn = declared_plan(profile)["experts"][:2]
    files = []
    for name in ("nn", "joint"):
        path = tmp_path / name
        path.write_bytes(("original " + name + " transport bytes").encode())
        files.append(artifact_receipt(path))
    nn_receipt, joint_receipt = files
    request = {"source_seed": 71, "target_contract": {"participant_key": "original-target"},
        "candidates": [{"expert_programs": [*nn, _programme().record()]}],
        "source_checkpoint_sha256": {nn[0]["expert_id"]: nn_receipt["sha256"],
            nn[1]["expert_id"]: nn_receipt["sha256"], "joint": joint_receipt["sha256"]}}
    plan = {"dataset_id": "actual-study", "joint_source_bindings": {"original-target": {"joint": "joint-source"}}}
    artifacts = {("actual-study", 71): nn_receipt, ("joint_flat", "joint-source"): joint_receipt}
    resolved = _prefix_replay_source_artifacts(request, plan, artifacts)
    assert resolved == {nn[0]["expert_id"]: nn_receipt, nn[1]["expert_id"]: nn_receipt, "joint": joint_receipt}
    with pytest.raises(ValueError, match="original expert request"):
        _prefix_replay_source_artifacts(request, plan, {**artifacts, ("joint_flat", "joint-source"): nn_receipt})
    assert _prefix_replay_source_artifacts({"source_checkpoint_sha256": {}, "candidates": []}, plan, {}) == {}


def test_joint_prediction_read_support_follows_real_history_or_real_refusal():
    from types import SimpleNamespace

    from experiments.run_candidate_ensemble_study import _prediction_read_support
    from research.history_authority import HistoryUnavailableError
    from tests.research.test_joint_flat_replay_contracts import _programme
    from tests.research.test_typed_prefix_replay_interfaces import _original_histories
    from transfer.ensemble_views import _lookup_record_history_refusal

    views, view, _, _ = _original_histories()
    compiler, prefix = views.prefix_compiler, views.calibration[0]
    stop = int(view.onsets[0]) + view.raw_scope[0] + 2
    capability = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
    recorder = SimpleNamespace(joint_future=[{"view": view, "issued_capabilities": {"joint": capability}}])
    support = _prediction_read_support(_programme(), view, recorder=recorder)
    assert support["raw_scope"] == list(capability.binding["read_sample_support"])
    assert support["raw_scope"] != support["current_view_scope"]
    assert support["conditioning_sample_support"] == [0, stop]
    assert support["score_sample_support"] == [stop, view.available_sample]
    views, view, _, _ = _original_histories(unresolved=True)
    compiler, prefix = views.prefix_compiler, views.calibration[0]
    stop = int(view.onsets[0]) + view.raw_scope[0] + 2
    with pytest.raises(HistoryUnavailableError):
        compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
    refusal = _lookup_record_history_refusal(compiler, view, prefix=prefix, conditioning_stop=stop)
    recorder = SimpleNamespace(joint_future=[{"view": view, "issued_capabilities": {}, "issued_refusals": {"joint": refusal}}])
    support = _prediction_read_support(_programme(), view, recorder=recorder)
    assert support["raw_scope"] is None
    assert support["history_refusal_sha256"] == refusal.digest
    assert "conditioning_sample_support" not in support and "score_sample_support" not in support
