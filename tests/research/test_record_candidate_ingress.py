"""Concrete record-ingress ownership regressions; no model fitting or FFT."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.huebner2017_candidate import (
    HuebnerNativeSource,
    _file_identity,
    load_huebner_native_source,
)
from tests.research.test_record_candidate_prefix import _adapter
from transfer.ensemble_views import (
    CalibrationPrefix,
    RecordCalibrationPrefixCompiler,
    SelectionEpochProfile,
)


def _compiler(adapter, *, profiles=()):
    return RecordCalibrationPrefixCompiler(adapter, budgets=(0, 3), profiles=profiles,
        authorization_preprocessing=preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
        response_seconds=.2, pre_roll_seconds=.2)


def test_legacy_zero_cannot_borrow_new_record_inputs_and_actual_zeros_are_empty():
    adapter, _ = _adapter()
    issuer = _compiler(adapter)
    actual = issuer.compile(0)
    assert actual.calibration_records == ()
    target = json.loads(json.dumps(actual.target_contract))
    for field in ("authority_kind", "record_scopes", "source_record_roster"):
        target.pop(field)
    target["raw_scope"] = [0, 0]
    use = {"raw": {"requested_labels": 0, "used_labels": 0, "read_sample_support": None,
                   "used_label_indices": []}, "trial": {}}
    legacy = CalibrationPrefix(0, None, {}, (), target, use)
    assert legacy.calibration_records == ()
    positive = issuer.compile(1)
    with pytest.raises(ValueError, match="legacy prefix cannot authorize record-only"):
        replace(legacy, native_records=positive.native_records)
    with pytest.raises(ValueError, match="legacy prefix cannot authorize record-only"):
        replace(legacy, target_contract={**target, "record_scopes": {}})
    with pytest.raises(ValueError):
        replace(actual, native_records=positive.native_records,
                target_contract={**actual.target_contract, "record_scopes": positive.target_contract["record_scopes"]})


def test_invalid_reference_fails_before_any_native_signal_read_and_valid_zero_is_metadata_only():
    adapter, reads = _adapter()
    processing = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    bad = SelectionEpochProfile.from_preprocessing(processing, source_sample_rate_hz=1000.,
        channel_names=("FZ",), reference="nose", reference_channel="OZ", allow_missing_channels=True)
    with pytest.raises(ValueError, match="measured-reference channels absent"):
        _compiler(adapter, profiles=(bad,))
    assert reads == []
    valid = replace(bad, reference_channel="CZ")
    prefix = _compiler(adapter, profiles=(valid,)).compile(0)
    assert prefix.trial_target(valid).channel_names == ("FZ",)
    assert reads == [] and prefix.calibration_records == ()


def test_adapter_and_record_metadata_are_owned_and_public_records_are_independent_snapshots():
    adapter, _ = _adapter()
    source = {"receipt": {"counter": 1}}
    record = replace(adapter.recordings[0], source_receipt=source)
    artifact, contract = {"origin": {"label": "original"}}, {"protocol": {"label": "set"}}
    admitted = replace(adapter, recordings=(record, *adapter.recordings[1:]),
                       source_artifacts=(artifact,), candidate_task_contract=contract)
    digest = admitted.digest
    source["receipt"]["counter"] = 2
    artifact["origin"]["label"] = "changed"
    contract["protocol"]["label"] = "changed"
    assert record.source_receipt["receipt"]["counter"] == 1
    assert admitted.digest == digest
    public = admitted.record()
    public["source_artifacts"][0]["origin"]["label"] = "changed snapshot"
    public["candidate_task_contract"]["protocol"]["label"] = "changed snapshot"
    public["recordings"][0]["source_receipt"]["receipt"]["counter"] = 3
    assert admitted.digest == digest


def test_native_reader_binds_before_after_file_identity_and_closes_on_change(tmp_path, monkeypatch):
    import mne

    path = tmp_path/"actual-byte-fixture.bdf"
    path.write_bytes(b"one original tiny file")
    raw = SimpleNamespace(ch_names=("Fz", "Cz"), info={"sfreq": 1000.}, n_times=100, closed=False)
    def close():
        raw.closed = True
    raw.close = close
    raw.get_data = lambda **kw: np.zeros((2, kw["stop"]-kw["start"]))
    opened = []
    monkeypatch.setattr(mne.io, "read_raw_bdf", lambda *a, **kw: opened.append(True) or raw)
    original = HuebnerNativeSource("actual-record", "participant", "1", "0", "0", path,
        hashlib.sha256(path.read_bytes()).hexdigest(), ("Fz", "Cz"), None, np.array([20]), {},
        _admitted_file_identity=_file_identity(path.stat()))
    np.testing.assert_array_equal(original.read_native(0, 2), np.zeros((2, 2)))
    assert raw.closed and len(opened) == 1
    path.write_bytes(b"another changed file with different size")
    with pytest.raises(ValueError, match="after byte admission"):
        original.read_native(0, 2)
    assert len(opened) == 1
    current = replace(original, _admitted_file_identity=_file_identity(path.stat()))
    raw.closed = False
    def change_during_read(**kw):
        path.write_bytes(b"changed once more during actual read")
        return np.zeros((2, kw["stop"]-kw["start"]))
    raw.get_data = change_during_read
    with pytest.raises(ValueError, match="during native range read"):
        current.read_native(0, 2)
    assert raw.closed


def test_original_member_and_companion_identity_fail_before_eeg_hash(monkeypatch):
    inputs = Path("D:/BCI/worktrees/huebner-candidate/tmp/HUEBNER_VERTICAL_02/original_inputs.json")
    if not inputs.exists():
        pytest.skip("actual original Huebner source declaration unavailable")
    source = json.loads(inputs.read_bytes())["record_sources"][0]
    def forbid_hash(*a, **kw):
        raise AssertionError("invalid source identities reached EEG byte hashing")
    monkeypatch.setattr(hashlib, "file_digest", forbid_hash)
    with pytest.raises(ValueError, match="marker subject/session/run/member identity"):
        load_huebner_native_source({**source, "subject": "2"})
    attacked = {**source, "events": {**source["events"],
        "path": str(Path(source["events"]["path"]).with_name("sub-1_ses-0_task-p300_run-1_events.tsv"))}}
    with pytest.raises(ValueError, match="sidecar identity"):
        load_huebner_native_source(attacked)
