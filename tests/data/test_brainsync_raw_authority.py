"""Canonical original-file admission, C capabilities and physical pause clocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np
import pytest

from data import brainsync_raw_authority as authority
from data.contract import brainsync_reference_record
from data.dataset import read_raw
from tests.support.brainsync import write_bids_session, write_recording_clock
from tests.support.io import read_json, read_tsv, write_json, write_tsv

SESSION_ID = "canonical-raw-authority"
BOUNDARY = 7625
ANCHOR_NS = 10**15
VOCABULARY = tuple(map(str, range(1, 10)))


@pytest.fixture
def recording(tmp_path):
    """Real synthetic EDF, two balanced blocks and an interrupted pause attempt."""
    root = write_bids_session(
        tmp_path / "recording", session_id=SESSION_ID, target=8,
        started_utc="2026-10-01T08:00:00Z",
        blocks=(tuple(range(1, 10)) * 5, tuple(range(9, 0, -1)) * 5),
    )
    manifest = read_json(root / "session.json")
    bids = manifest["recording"]["bids"]
    sidecar = read_json(root / bids["eeg_json"])
    reference = brainsync_reference_record()
    manifest["montage"] = {"ref_label": reference["label"], "reference_details": reference}
    sidecar.update(EEGReference=reference["label"], EEGReferenceDetails=reference)
    sidecar["EOGDerivation"]["EOG"]["reference"] = reference["label"]
    count = round(sidecar["RecordingDuration"] * 250)
    clock = write_recording_clock(root, acquired=count, serialized=count)["clock"]
    metadata_path = root / clock["metadata_path"]
    journal_path = root / clock["journal_path"]
    metadata, original = read_json(metadata_path), read_json(journal_path)
    batches = [
        {
            **original, "batch_id": index, "received_start": start,
            "stored_start": start, "n_received": stop - start, "n_stored": stop - start,
            "segment_id": index, "seq_num": original["seq_num"][start:stop],
            "delta_time_raw": original["delta_time_raw"][start:stop],
        }
        for index, (start, stop) in enumerate(((0, BOUNDARY), (BOUNDARY, count)))
    ]
    journal_path.write_text("".join(json.dumps(row) + "\n" for row in batches), encoding="utf-8")
    anchor = {"anchor_source": "first_batch_delivery_minus_nominal_span",
              "relative_rate_assumption": "nominal_sample_rate"}
    metadata.update(
        batch_count=2, journal_sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        segments=[
            {**anchor, "segment_id": 0, "stored_start": 0, "stored_stop": BOUNDARY,
             "first_sample_estimate_monotonic_ns": ANCHOR_NS,
             "closed_monotonic_ns": ANCHOR_NS + 30_500_000_000},
            {**anchor, "segment_id": 1, "stored_start": BOUNDARY, "stored_stop": count,
             "first_sample_estimate_monotonic_ns": ANCHOR_NS + 32_500_000_000},
        ],
    )
    write_json(metadata_path, metadata)
    manifest["recording"]["clock"] = {
        **metadata, "metadata_path": clock["metadata_path"], "journal_path": clock["journal_path"],
    }
    pauses = [{"pause_id": 1, "boundary_sample": BOUNDARY,
               "start_wall_seconds": 30.5, "end_wall_seconds": 32.5}]
    manifest["recording"]["timeline"].update(
        time_base="segmented_recording", acquisition_pauses=pauses, pause_samples_retained=False,
    )
    sidecar.update(RecordingType="discontinuous", AcquisitionPauses=pauses)
    write_json(root / bids["eeg_json"], sidecar)
    events = read_tsv(root / bids["events_tsv"])
    retry = next(row for row in events if row["trial_id"] == "B001-T021")
    retry["trial_id"] = "B001-T021-A002"
    interrupted = {
        **retry, "onset": "30.3", "sample": "7575", "duration": "0.2",
        "trial_type": "interrupted_stimulus", "event_kind": "interrupted_stimulus",
        "trial_id": "B001-T021-A001", "digit": "n/a", "value": "n/a",
    }
    boundary = dict.fromkeys(retry, "n/a")
    boundary.update(
        onset="30.5", sample=str(BOUNDARY), duration="0", trial_type="acquisition_boundary",
        event_kind="acquisition_boundary", value="pause", block_id="1",
    )
    events.extend((interrupted, boundary))
    events.sort(key=lambda row: float(row["onset"]))
    write_tsv(root / bids["events_tsv"], events)
    start = datetime.fromisoformat(manifest["started_utc"].replace("Z", "+00:00"))
    manifest["target_label"]["confirmed_utc"] = (start + timedelta(seconds=143)).isoformat()
    manifest["ended_utc"] = (start + timedelta(seconds=144)).isoformat()
    write_json(root / "session.json", manifest)
    return root


def _whole_record(issuer, count):
    return (
        issuer.issue_raw(0, prefix_stop=250, full_stop=BOUNDARY, available_stop=BOUNDARY),
        issuer.issue_raw(1, prefix_stop=7750, full_stop=count, available_stop=count),
    )


def test_c0_inference_reads_native_eeg_and_preserves_requests_without_truth(recording, monkeypatch):
    scope = authority.BrainSyncRecordingScope((), (SESSION_ID,))
    issuer = authority.open_brainsync_recording(recording, scope=scope)
    try:
        cap = issuer.issue_raw(0, prefix_stop=250, full_stop=1500, available_stop=1500)
        assert authority.require_raw_capability(cap) is cap
        assert cap.candidate_vocabulary == VOCABULARY
        assert cap.membership.shape == (4, 9)
        assert [row["visible_public_event_count"] for row in cap.public_binding["block_requests"]] == [4, 0]
        assert [row["requested_flash_count"] for row in cap.public_binding["block_requests"]] == [45, 45]
        assert cap.public_binding["public_physical_history"]["requested_completed_flashes"] == 90
        manifest = read_json(recording / "session.json")
        original = read_raw(recording / manifest["recording"]["path"], preload=False)
        try:
            decoded = original.get_data(start=0, stop=1500)
        finally:
            original.close()
        np.testing.assert_array_equal(cap.eeg, decoded[[0, 1, 2, 3, 4, 5, 7]].T)
        np.testing.assert_array_equal(cap.eog, decoded[6:7].T)
        with pytest.raises(ValueError, match="no C-label authority"):
            issuer.issue_labels(0, ready_at_timestamp_s=2e9)
        with pytest.raises(ValueError, match="not issued"):
            authority.require_raw_capability(replace(cap))
        before = authority.public_record(cap)
    finally:
        issuer.close()

    # Recording admission follows the actual unlabelled BIDS owner, even when
    # target fields cannot satisfy the separate confirmed-session validator.
    manifest["experiment"]["thought_digit"] = {"private": "unavailable"}
    manifest["target_label"] = {"status": "withheld", "thought_digit": "not provided"}
    write_json(recording / "session.json", manifest)
    issuer = authority.open_brainsync_recording(recording, scope=scope)
    try:
        cap = issuer.issue_raw(0, prefix_stop=250, full_stop=1500, available_stop=1500)
        assert authority.public_record(cap) == before
    finally:
        issuer.close()

    def forbidden_supervised_validation(*args, **kwargs):
        pytest.fail("an empty-C held recording reached supervised target validation")

    monkeypatch.setattr(authority, "validate_brainsync_bids_session", forbidden_supervised_validation)
    with pytest.raises(ValueError, match="calibration/inference role"):
        authority.open_brainsync_calibration(recording, scope=scope)


def test_c_bits_are_scoped_to_original_events_and_all_required_segments(recording):
    manifest = read_json(recording / "session.json")
    count = manifest["recording"]["clock"]["acquired_samples"]
    ready = datetime.fromisoformat(manifest["ended_utc"]).timestamp()
    scope = authority.BrainSyncRecordingScope((SESSION_ID,), ("independent-held-recording",))
    issuer = authority.open_brainsync_calibration(recording, scope=scope)
    try:
        raw0, raw1 = _whole_record(issuer, count)
        empty = issuer.issue_labels(0, ready_at_timestamp_s=ready)
        assert authority.allowed_candidates((raw0, raw1), empty)["allowed_candidates"] == VOCABULARY
        partial = issuer.issue_labels(3, ready_at_timestamp_s=ready)
        assert tuple(label.binary_label for label in partial.labels) == (0, 0, 0)
        constrained = authority.allowed_candidates(raw0, partial)
        assert constrained["allowed_candidates"] == tuple(map(str, range(4, 10)))
        assert constrained["labels_used"] == constrained["authorized_global_event_count"] == 3
        with pytest.raises(ValueError, match="not issued"):
            authority.require_calibration_labels(replace(partial))
        full = issuer.issue_labels(90, ready_at_timestamp_s=ready)
        with pytest.raises(ValueError, match="actual availability"):
            authority.allowed_candidates(raw0, full)
        complete = authority.allowed_candidates((raw0, raw1), full)
        assert complete["allowed_candidates"] == ("8",)
        assert complete["labels_used"] == complete["authorized_global_event_count"] == 90
        assert complete["requested_candidate_count"] == 9
    finally:
        issuer.close()


def test_physical_rows_and_elapsed_times_preserve_pause_and_unknown_attempt(recording):
    count = read_json(recording / "session.json")["recording"]["clock"]["acquired_samples"]
    scope = authority.BrainSyncRecordingScope((), (SESSION_ID,))
    issuer = authority.open_brainsync_recording(recording, scope=scope)
    try:
        baseline = issuer.issue_raw(0, prefix_stop=0, full_stop=250, available_stop=250)
        baseline_clock = authority.compile_physical_clock(baseline)
        assert baseline.events == baseline.public_binding["public_physical_history"]["exposures"] == ()
        assert baseline_clock.sample_time_seconds((0, 1, 249), clock_program="timeline_wall") == (
            Fraction(0), Fraction(1, 250), Fraction(249, 250),
        )
        before, after = _whole_record(issuer, count)
        before_history = before.public_binding["public_physical_history"]
        history = after.public_binding["public_physical_history"]
        assert (len(before.events), len(after.events)) == (20, 70)
        assert before_history["acquisition_pauses"] == ()
        assert len(before_history["exposures"]) == 21
        assert len(history["exposures"]) == 91
        interrupted = next(row for row in history["exposures"] if row["trial_id"] == "B001-T021-A001")
        assert interrupted["exposure_role"] == "interrupted_physical_stimulus"
        assert interrupted["membership"] is None
        assert interrupted["candidate_options"] == VOCABULARY
        assert interrupted["membership_role"] == "unknown_exposure_candidate_no_prior_assigned"
        assert history["unknown_exposure_count"] == 1
        assert history["prior_segment_responses_reset_to_zero"] is False
        assert history["cross_segment_noise_law_owned_by_model"] is True
        assert before.public_binding["target_lock_group"] == after.public_binding["target_lock_group"]
        prior = next(row for row in history["exposures"] if row["trial_id"] == "B001-T020")
        clock0, clock1 = map(authority.compile_physical_clock, (before, after))
        for program, origin in (("timeline_wall", Fraction(0)),
                                ("delivery_anchor", Fraction(ANCHOR_NS, 10**9))):
            assert clock1.sample_time_seconds(7750, clock_program=program) == origin + 33
            assert (clock1.sample_time_seconds(BOUNDARY, clock_program=program)
                    - clock0.sample_time_seconds(BOUNDARY - 1, clock_program=program)) == Fraction(501, 250)
            assert clock1.elapsed_seconds(7750, prior["event_id"], clock_program=program) == Fraction(7, 2)
            assert clock1.elapsed_seconds(7750, interrupted["event_id"], clock_program=program) == Fraction(27, 10)
            assert clock1.elapsed_seconds(7750, prior["event_id"], clock_program=program) == (
                clock1.sample_time_seconds(7750, clock_program=program)
                - clock0.sample_time_seconds(prior["sample"], clock_program=program)
            )
            assert clock1.sample_time_seconds(7750, clock_program=program, reference_origin=origin + 13) == 20
    finally:
        issuer.close()
