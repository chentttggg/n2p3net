import hashlib
import json
import shutil

import pytest

from data.brainsync import load_brainsync_inference_session
from data.brainsync_contract import (
    validate_brainsync_bids_recording,
    validate_brainsync_bids_session,
)
from research.recording_quality import create_recording_quality_report
from tests.support.brainsync import write_bids_session


def _exclude(monkeypatch, root):
    manifest = json.loads((root / "session.json").read_text())
    path = root / manifest["recording"]["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    from data import recording_review

    monkeypatch.setattr(recording_review, "recording_reviews", lambda: {digest: {
        "status": "excluded", "session_id": manifest["session_id"],
        "reason": "Owner excluded this recording after signal review", "authority": "test owner",
        "raw_sha256": digest,
    }})
    return digest


def test_exclusion_follows_actual_raw_bytes_through_relocation(monkeypatch, tmp_path):
    root = write_bids_session(tmp_path / "original", session_id="excluded", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    digest = _exclude(monkeypatch, root)
    moved = tmp_path / "renamed"
    shutil.copytree(root, moved)
    for loader in (validate_brainsync_bids_recording, validate_brainsync_bids_session,
                   load_brainsync_inference_session):
        with pytest.raises(ValueError, match="excluded.*Owner excluded"):
            loader(moved)
    assert hashlib.sha256((moved / json.loads((moved / "session.json").read_text())["recording"]["path"]).read_bytes()).hexdigest() == digest


def test_excluded_quality_report_keeps_requested_denominator(monkeypatch, tmp_path):
    root = write_bids_session(tmp_path / "input", session_id="excluded-report", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    _exclude(monkeypatch, root)
    report = create_recording_quality_report(root, tmp_path / "report")
    assert report["status"] == "excluded"
    assert report["counts"]["scheduled_trials"] == 9
    assert report["counts"]["preprocessed_epochs"] == 0
    assert report["recording_review"]["status"] == "excluded"
    assert "Owner excluded" in report["headline"]


def test_different_raw_recording_is_not_excluded_by_similar_directory_name(monkeypatch, tmp_path):
    bad = write_bids_session(tmp_path / "bad", session_id="one", target=4,
                             started_utc="2026-09-06T00:00:00Z")
    _exclude(monkeypatch, bad)
    other = write_bids_session(tmp_path / "P001_20261030_144501_01", session_id="another", target=4,
                               started_utc="2026-09-07T00:00:00Z")
    from data import recording_review

    monkeypatch.setattr(recording_review, "recording_reviews", lambda: {"f" * 64: {
        "status": "excluded", "reason": "different bytes", "raw_sha256": "f" * 64,
    }})
    validated = validate_brainsync_bids_recording(other)
    validated.bids.close()
