"""Original source roles and C/P/T authority survive the shared replay boundary."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from research.contracts import semantic_sha256
from research.prefix_replay_execution import (
    _prepare_history_capabilities,
    _request_source_programs,
    _sealed,
    _sources,
    artifact_receipt,
)


def test_original_expert_role_dispatches_source_bytes_without_filename_inference(tmp_path, monkeypatch):
    from tests.research.test_joint_flat_replay_contracts import _programme
    from transfer.joint_flat_programme import FrozenJointFlatSourcePreparation

    source = tmp_path / "actual_joint_source.pt"
    source.write_bytes(b"exact source codec ingress")
    receipt, spec = artifact_receipt(source), _programme()
    row = {"original_request": {"expert_programs": [spec.record()]}, "source_checkpoints": {"joint": receipt}}
    calls, admitted = [], object()
    owner = FrozenJointFlatSourcePreparation()

    def prepare(content, *, expected_sha256):
        calls.append((content, expected_sha256))
        return admitted

    monkeypatch.setattr(owner, "prepare", prepare)
    assert _sources(row, joint_source_preparation=owner) == {"joint": admitted}
    assert calls == [(source.read_bytes(), receipt["sha256"])]
    row["source_checkpoints"] = {"undeclared": receipt}
    with pytest.raises(ValueError, match="original declared expert role"):
        _request_source_programs(row)
    from transfer.ensemble_fit import CalibrationExpertProgram

    row = {"original_request": {"candidates": [{"expert_programs": [spec.record()]},
        {"expert_programs": [CalibrationExpertProgram("joint", "continuous_ar").record()]}]},
        "source_checkpoints": {"joint": receipt}}
    with pytest.raises(ValueError, match="conflicting original expert roles"):
        _sources(row, joint_source_preparation=owner)
    assert len(calls) == 1


def _original_histories(*, unresolved=False):
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from tests.research.test_record_candidate_prefix import _adapter
    from transfer.ensemble_views import RecordCalibrationPrefixCompiler

    adapter, calls = _adapter()
    if unresolved:
        record = adapter.recordings[0]
        adapter = replace(adapter, recordings=(replace(record,
            native_onsets=np.array([-1, 1250, 5000, 5250], dtype=np.int64),
            unavailable_public_onsets=np.array([1000, 1250, 5000, 5250], dtype=np.int64)), adapter.recordings[1]))
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0,), profiles=(),
        authorization_preprocessing=spec, response_seconds=.2, pre_roll_seconds=.2)
    views = compiler.independent_views()
    view = next(view for view in views.decisions if view.source_sha256 == adapter.recordings[0].source_sha256
        and view.decision_id.endswith("selection:1"))
    return views, view, spec, calls


def _input_request(views, spec):
    from transfer import ensemble_views

    return {"schema": "n2p3_prefix_replay_input_request/2", "preparation_task_id": "actual-history-ingress",
        "original_source_sha256": views.ledger["source_sha256"],
        "producer_source_sha256": ensemble_views._SELECTION_EXECUTION_SOURCE_SHA256,
        "participant_key": views.prefix_compiler.target["participant_key"],
        "source_relative_path": views.prefix_compiler.target["source_relative_path"],
        "requested_Ks": [0], "actual_Ks": [0],
        "profiles": [], "authorization_preprocessing": spec.record(), "observation_response_seconds": .2,
        "pre_roll_seconds": .2, "dataset_source": {"explicit_software_fixture": True}}


@pytest.mark.parametrize("unresolved", (False, True))
def test_actual_issuer_history_or_refusal_crosses_capsule_without_new_permission(tmp_path, unresolved):
    from research.history_authority import HistoryUnavailableError
    from research.joint_flat_replay import JointFlatReplayRecorder
    from research.prefix_replay_inputs import _ArrayReader, _restore_views, export_prepared_views

    views, view, spec, calls = _original_histories(unresolved=unresolved)
    compiler, prefix = views.prefix_compiler, views.calibration[0]
    stop = int(view.onsets[0] + view.raw_scope[0] + 1)
    if unresolved:
        with pytest.raises(HistoryUnavailableError, match="unresolved_actual_event_clock"):
            compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
        refusal = next(iter(compiler._record_history_refusals.values()))
        permission = {"issued_history_refusals": {"joint": {"sha256": refusal.digest, "binding": refusal.binding}},
            "issued_history_permissions": {}}
    else:
        cap = compiler.record_read_scope(view, prefix=prefix, conditioning_stop=stop)
        permission = {"issued_history_permissions": {"joint": {"sha256": cap.digest, "binding": cap.binding}},
            "issued_history_refusals": {}}
    record = {"schema": "n2p3_joint_flat_prefix_inner_replay/1", "OOF": [], "future": [permission]}
    package = tmp_path / "original-node"
    _sealed(package / "prefix_inner_replay.json", record)
    scopes, refusals = _prepare_history_capabilities(views, (0,), ((package, semantic_sha256(record)),))
    assert (len(scopes), len(refusals)) == ((0, 1) if unresolved else (1, 0))
    # Exercise the recorder's authority transport port without fitting a model.
    recorder = JointFlatReplayRecorder()
    recorder.record_joint_forward_oof(earlier=prefix, held=SimpleNamespace(view=view), fitted=None,
        row={}, context={}, actual_captures={}, issued_capabilities={"joint": scopes[0]} if scopes else {},
        issued_refusals={"joint": refusals[0]} if refusals else {})
    request = _input_request(views, spec)
    capsule = export_prepared_views(views, (), tmp_path / "capsule", request,
        record_history_scopes=recorder.original_history_capabilities,
        record_history_refusals=recorder.original_history_refusals)
    document = json.loads((tmp_path / "capsule" / "capsule.json").read_bytes())
    assert artifact_receipt(capsule["path"])["sha256"] == capsule["sha256"]
    inputs = _restore_views(document, _ArrayReader((tmp_path / "capsule" / "arrays.bin").read_bytes(), document["arrays"]))
    original_reads = len(calls)
    restored = next(item for item in inputs.decisions if item.record_id == view.record_id)
    restored_prefix, prepared = inputs.calibration[0], inputs.prefix_compiler
    if unresolved:
        with pytest.raises(HistoryUnavailableError) as caught:
            prepared.record_read_scope(restored, prefix=restored_prefix, conditioning_stop=stop)
        assert caught.value.record() == refusal.binding["error"]
    else:
        restored_cap = prepared.record_read_scope(restored, prefix=restored_prefix, conditioning_stop=stop)
        assert restored_cap.digest == scopes[0].digest
    with pytest.raises(ValueError):
        prepared.record_read_scope(restored, prefix=restored_prefix, conditioning_stop=stop + 1)
    assert len(calls) == original_reads
    changed = json.loads(json.dumps(record))
    field = "issued_history_refusals" if unresolved else "issued_history_permissions"
    binding = changed["future"][0][field]["joint"]["binding"]
    (binding["request"] if unresolved else binding)["source_owner_sha256"] = "f" * 64
    changed["future"][0][field]["joint"]["sha256"] = semantic_sha256(binding)
    foreign = tmp_path / "foreign-node"
    _sealed(foreign / "prefix_inner_replay.json", changed)
    with pytest.raises(ValueError, match="different"):
        _prepare_history_capabilities(views, (0,), ((foreign, semantic_sha256(changed)),))
