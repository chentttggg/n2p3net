from types import SimpleNamespace

import pytest

from data.brainsync import load_brainsync_session, resolve_brainsync_reference
from data.brainsync_contract import validate_brainsync_bids_session
from data.contract import brainsync_reference_record
from experiments.correct_brainsync_reference import correct_sessions
from tests.support.brainsync import write_bids_session
from tests.support.io import read_json, write_json


def _session(tmp_path):
    root = tmp_path / "recording"
    write_bids_session(root, session_id="reference-test", target=2, started_utc="2026-09-08T00:00:00Z")
    path = root / "session.json"
    value = read_json(path)
    value["montage"] = {"ref_label": "A2"}
    write_json(path, value)
    return root


def test_explicit_correction_preserves_edf_and_records_real_native_reference(tmp_path):
    root = _session(tmp_path)
    before = {path: path.read_bytes() for path in root.rglob("*.edf")}
    result = correct_sessions([root], previous="A2", authority="explicit_test_confirmation", output=tmp_path / "correction")
    assert len(result["files"]) == 2
    assert result["reference"] == brainsync_reference_record()
    assert all(path.read_bytes() == content for path, content in before.items())
    validated = validate_brainsync_bids_session(root)
    assert validated.bids.source_reference == "left earlobe"
    assert validated.bids.eog_derivation["EOG"]["reference"] == "left earlobe"
    dataset = load_brainsync_session(root)
    assert dataset.n_channels == 7
    assert dataset.provenance["source_reference"] == "left earlobe"
    assert dataset.provenance["acquisition_reference_details"] == brainsync_reference_record()
    with pytest.raises(ValueError, match="previous label"):
        correct_sessions([root], previous="A2", authority="again", output=tmp_path / "second")


def test_reference_correction_prevalidates_all_sessions_before_mutation(tmp_path):
    root = _session(tmp_path)
    before = (root / "session.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        correct_sessions([root, tmp_path / "missing"], previous="A2", authority="test", output=tmp_path / "correction")
    assert (root / "session.json").read_bytes() == before


def test_brainsync_rejects_montage_sidecar_reference_disagreement(tmp_path):
    root = _session(tmp_path)
    path = root / "session.json"
    payload = read_json(path)
    payload["montage"]["ref_label"] = "left earlobe"
    write_json(path, payload)
    with pytest.raises(ValueError, match="REF conflicts"):
        validate_brainsync_bids_session(root)


def test_brainsync_rejects_wrong_laterality_in_reference_details(tmp_path):
    root = _session(tmp_path)
    correct_sessions([root], previous="A2", authority="test", output=tmp_path / "correction")
    session_path = root / "session.json"
    session = read_json(session_path)
    session["montage"]["reference_details"]["side"] = "right"
    write_json(session_path, session)
    sidecar_path = root / session["recording"]["bids"]["eeg_json"]
    sidecar = read_json(sidecar_path)
    sidecar["EEGReferenceDetails"]["side"] = "right"
    write_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="left-earlobe contract"):
        validate_brainsync_bids_session(root)


def test_native_reference_cannot_silently_inherit_cz_quality_or_eog():
    cz_quality = SimpleNamespace(spec=SimpleNamespace(reference_channel="CZ"))
    cz_ocular = SimpleNamespace(reference_channel="CZ")
    assert resolve_brainsync_reference(None, None, None, mode="native") is None
    assert resolve_brainsync_reference("Cz", cz_ocular, cz_quality) == "CZ"
    with pytest.raises(ValueError, match="Native acquisition reference"):
        resolve_brainsync_reference(None, None, cz_quality, mode="native")
    with pytest.raises(ValueError, match="Native acquisition reference"):
        resolve_brainsync_reference(None, cz_ocular, None, mode="native")
    with pytest.raises(ValueError, match="must agree"):
        resolve_brainsync_reference("Pz", None, cz_quality)
