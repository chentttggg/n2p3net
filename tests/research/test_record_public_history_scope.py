"""Original-record C/P/T permissions and prepared lookup; no fits or FFT."""
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from tests.research.test_record_candidate_prefix import _adapter
from transfer.ensemble_views import (
    RecordCalibrationPrefixCompiler,
    _PreparedRecordCalibrationPrefixCompiler,
    _record_history_key,
    calibration_prefix_identity,
)


def _issuer(adapter):
    return RecordCalibrationPrefixCompiler(adapter, budgets=(0, 3), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        response_seconds=.2, pre_roll_seconds=.2)


def test_record_public_history_keeps_actual_clock_c_mapping_and_partial_same_selection_permission():
    adapter, reads = _adapter()
    issuer = _issuer(adapter)
    prefix = issuer.compile(3)
    view = issuer.independent_views().decisions[0]
    assert view.samples is None and view.unavailable_reason == "response_overlaps_common_C_test_embargo"
    cap = issuer.record_read_scope(view, prefix=prefix, conditioning_stop=5300)
    from research.contracts import semantic_sha256

    assert cap.digest == semantic_sha256(cap.binding)
    assert cap.samples.shape == (5450, 2)
    assert reads[-1] == (adapter.recordings[0].record_id, 0, 5450)
    assert cap.binding["score_sample_support"] == (5300, 5450)
    assert cap.binding["C_record_to_original_record"] == {prefix.calibration_records[0].record_id: adapter.recordings[0].record_id}
    assert cap.binding["future_truth_consumed"] is False
    public = cap.binding["public_selections"]
    assert len(public) == 2 and len(public[-1]["onsets"]) == 2
    assert not any({"target", "truth", "binary_labels"}.intersection(selection) for selection in public)
    with pytest.raises(ValueError, match="C observations exceed"):
        issuer.record_read_scope(view, prefix=prefix, conditioning_stop=5199)
    altered = replace(view, onsets=view.onsets+1)
    with pytest.raises(ValueError, match="event-set/native-clock"):
        issuer.record_read_scope(altered, prefix=prefix, conditioning_stop=5300)


def test_prepared_lookup_requires_exact_original_prefix_pt_and_view():
    adapter, _ = _adapter()
    issuer = _issuer(adapter)
    views = issuer.independent_views()
    prefix, view = views.calibration[3], views.decisions[0]
    cap = issuer.record_read_scope(view, prefix=prefix, conditioning_stop=5300)
    prepared = _PreparedRecordCalibrationPrefixCompiler(views.calibration,
        identities={k: calibration_prefix_identity(v) for k, v in views.calibration.items()},
        producer_source_sha256=cap.binding["source_owner_sha256"], consumer_source_sha256="f"*64)
    with pytest.raises(ValueError, match="not issued"):
        prepared.record_read_scope(view, prefix=prefix, conditioning_stop=5300)
    prepared._record_history_scopes = {_record_history_key(cap.binding): cap}
    restored = prepared.record_read_scope(view, prefix=prefix, conditioning_stop=5300)
    np.testing.assert_array_equal(restored.samples, cap.samples)
    with pytest.raises(ValueError, match="not issued"):
        prepared.record_read_scope(view, prefix=prefix, conditioning_stop=5301)
    with pytest.raises(ValueError, match="physical view"):
        prepared.record_read_scope(replace(view, onsets=view.onsets+1), prefix=prefix, conditioning_stop=5300)


def test_explicit_future_record_roster_preserves_all_selections_and_refuses_c_overlap():
    adapter, _ = _adapter()
    future = replace(adapter, requested_future_record_ids=(adapter.recordings[1].record_id,))
    views = _issuer(future).independent_views()
    assert len(views.decisions) == 2
    assert views.ledger["test_budget"]["actual_test_trials"] == 4
    assert views.ledger["test_budget"]["separate_test_recordings"] is True
    assert all(v.source_sha256 == adapter.recordings[1].source_sha256 for v in views.decisions)
    assert views.calibration[3].calibration_records[0].record_id == adapter.recordings[0].record_id
    with pytest.raises(ValueError, match="overlaps global maximum C"):
        _issuer(replace(adapter, requested_future_record_ids=(adapter.recordings[0].record_id,)))
    with pytest.raises(ValueError, match="unique original acquisition"):
        replace(adapter, requested_future_record_ids=(adapter.recordings[1].record_id,)*2)


def test_original_mat_history_uses_explicit_physical_mapping_and_preserves_legacy_dose_id(tmp_path):
    from scipy.io import savemat

    from tests.support.bnci import bnci_mat_fields
    from transfer.ensemble_views import CalibrationPrefixCompiler, _bnci_prediction_view

    path = tmp_path/"A01.mat"
    savemat(path, {"data": bnci_mat_fields()})
    payload = path.read_bytes()
    issuer = CalibrationPrefixCompiler(payload, source_sha256=hashlib.sha256(payload).hexdigest(),
        source_relative_path=path.name, budgets=(0, 119), profiles=(),
        authorization_preprocessing=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT))
    prefix, record = issuer.compile(119), issuer._record
    rows = np.flatnonzero(record.selection_index == 1)
    start, stop = int(record.flash_sample[rows[0]])-52, int(record.flash_sample[rows[-1]])+308
    view = _bnci_prediction_view(record, rows, start, stop, available_sample=stop)
    cap = issuer.record_read_scope(view, prefix=prefix, conditioning_stop=int(record.flash_sample[rows[0]])+1)
    original_c_id = prefix.raw.record_id
    assert original_c_id in cap.binding["C_record_to_original_record"]
    assert cap.binding["C_record_to_original_record"][original_c_id] == cap.record_id
    assert cap.record_id != original_c_id and prefix.raw.record_id == original_c_id
    np.testing.assert_array_equal(cap.samples, np.asarray(record.eeg_uv[:stop], dtype=np.float64)*1e-6)
    assert cap.binding["future_truth_consumed"] is False
    physical = issuer.calibration_record_physical_map(prefix)
    assert physical["records"][original_c_id]["physical_record_id"] == cap.record_id
    assert physical["records"][original_c_id]["input_sha256"] == cap.binding["C_record_input_sha256"][original_c_id]


def test_input_export_borrows_longest_real_record_master_for_short_history_and_long_c(tmp_path):
    import json

    import transfer.ensemble_views as owner
    from research.prefix_replay_inputs import export_prepared_views

    adapter, _ = _adapter()
    processing = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    issuer = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 6), profiles=(),
        authorization_preprocessing=processing, response_seconds=.2, pre_roll_seconds=.2)
    views = issuer.independent_views()
    short = issuer._current_view(adapter.recordings[0], 0)
    cap = issuer.record_read_scope(short, prefix=views.calibration[0], conditioning_stop=1200)
    assert len(cap.samples) == 1450 < len(views.calibration[6].calibration_records[0].samples)
    request = {"schema": "n2p3_prefix_replay_input_request/1", "preparation_task_id": "actual-original-records",
        "original_source_sha256": views.ledger["source_sha256"],
        "producer_source_sha256": hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        "participant_key": adapter.participant_key, "source_relative_path": "ordered_original_candidate_acquisitions",
        "requested_Ks": [0, 6], "actual_Ks": [0, 6], "profiles": [],
        "authorization_preprocessing": processing.record(), "observation_response_seconds": .2, "pre_roll_seconds": .2}
    descriptor = export_prepared_views(views, (), tmp_path/"capsule", request, record_history_scopes=(cap,))
    document = json.loads(Path(descriptor["path"]).read_bytes())
    long_entry = document["arrays"][document["prefixes"]["6"]["native_records"][0]["samples"]["array"]]
    short_entry = document["arrays"][document["record_history_scopes"][0]["samples"]["array"]]
    assert long_entry["offset"] == short_entry["offset"] and long_entry["shape"][0] == 5450
