import hashlib
import json
from dataclasses import replace

import mne
import numpy as np
import pytest

from data.continuous_qc import ContinuousQCSpec, continuous_qc_from_record, fit_continuous_qc
from data.contract import BRAIN_SYNC_ACQUISITION_TYPES
from tests.support.brainsync import write_bids_session


def _profile(tmp_path):
    root = write_bids_session(tmp_path / "input", session_id="reviewed", target=4,
                              started_utc="2026-09-06T00:00:00Z")
    meta = json.loads((root / "session.json").read_text())
    path = root / meta["recording"]["path"]
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.set_channel_types(dict(zip(raw.ch_names, [t.lower() for t in BRAIN_SYNC_ACQUISITION_TYPES], strict=True)))
    model = fit_continuous_qc(raw,
        spec=ContinuousQCSpec(1e9, 1e-6, 25, channel_local=True, flatline_min_duration_s=5.),
        fit_sample_mask=np.ones(raw.n_times, bool),
        fit_provenance={"participant_key": "test:P001", "source_reference": "left earlobe"})
    return raw, model, path


def test_reviewed_interval_is_content_bound_and_survives_model_roundtrip(tmp_path):
    raw, model, path = _profile(tmp_path)
    try:
        from data.reviewed_intervals import ReviewedInterval, ReviewedRecording

        entry = ReviewedRecording(hashlib.sha256(path.read_bytes()).hexdigest(), raw.n_times,
                                  "reviewed", "explicit batch review", (ReviewedInterval(500, 750, ("Fz",), "reviewed_spike"),))
        model = replace(model, reviewed_recordings=(entry,))
        restored = continuous_qc_from_record(model.record())
        assert restored.record() == model.record()
        assert model.record()["schema"] == "n2p3_continuous_qc/5"
        result = restored.detect(raw)
        assert any(r.start_sample == 500 and r.end_sample == 750 and r.reason == "reviewed_spike" for r in result.intervals)
        masks = result.epoch_channel_mask([600], channels=model.channel_names, tmin_s=-.2, tmax_s=1.2)
        assert not masks[0, model.channel_names.index("Fz")]
        assert masks[0, model.channel_names.index("Pz")]
    finally:
        raw.close()


def test_reviewed_intervals_cannot_silently_apply_to_unidentified_memory_raw(tmp_path):
    raw, model, path = _profile(tmp_path)
    try:
        from data.reviewed_intervals import ReviewedInterval, ReviewedRecording

        entry = ReviewedRecording(hashlib.sha256(path.read_bytes()).hexdigest(), raw.n_times,
                                  "reviewed", "explicit batch review", (ReviewedInterval(500, 750, ("Fz",), "reviewed_spike"),))
        model = replace(model, reviewed_recordings=(entry,))
        memory = mne.io.RawArray(raw.get_data(), raw.info, verbose=False)
        with pytest.raises(ValueError, match="original.*file"):
            model.detect(memory)
    finally:
        raw.close()


def test_legacy_profile_record_stays_unchanged_without_reviewed_intervals(tmp_path):
    raw, model, _ = _profile(tmp_path)
    try:
        assert model.record()["schema"] == "n2p3_continuous_qc/4"
        assert "reviewed_recordings" not in model.record()
        assert continuous_qc_from_record(model.record()).record() == model.record()
    finally:
        raw.close()


def test_reviewed_only_roster_can_admit_no_interval_and_reject_unknown_content(tmp_path):
    from data.reviewed_intervals import ReviewedRecording

    raw, model, path = _profile(tmp_path)
    try:
        entry = ReviewedRecording(hashlib.sha256(path.read_bytes()).hexdigest(), raw.n_times,
                                  "reviewed", "explicit review with no artifact intervals", ())
        entries = [entry]
        model = replace(model, reviewed_recordings=entries, reviewed_only=True)
        entries.clear()
        restored = continuous_qc_from_record(model.record())
        assert len(restored.reviewed_recordings) == 1
        assert restored.detect(raw).record()["detection"] == "reviewed_intervals_and_exact_constants"
        wrong = replace(entry, raw_sha256="f" * 64)
        with pytest.raises(ValueError, match="outside.*roster"):
            replace(model, reviewed_recordings=(wrong,)).detect(raw)
        with pytest.raises(ValueError, match="exact-constant"):
            replace(model, spec=replace(model.spec, flatline_min_duration_s=None))
    finally:
        raw.close()
