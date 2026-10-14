"""Original marker authority, native clock join and truth-free physical sets."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.candidate_task import CandidateTaskContract
from data.huebner2017_candidate import (
    BLANK_POSITIONS,
    CANDIDATE_TASK,
    SELECTABLE_POSITIONS,
    decode_original_markers,
    join_nemar_events,
    membership_decision,
    read_local_marker_member,
)

JOIN = Path("D:/BCI/worktrees/research-director/tmp/HUEBNER_PHYSICAL_MARKER_JOIN.json")


@pytest.fixture(scope="module")
def original_join():
    if not JOIN.exists():
        pytest.skip("original source join receipt unavailable on this host")
    receipts = json.loads(JOIN.read_text(encoding="utf-8"))
    values = []
    for r in receipts:
        payload = read_local_marker_member(r["original_zip"], r["member"])
        record = decode_original_markers(payload, source_authority=r["member"])
        events = Path(r["nemar_events_path"]).read_bytes()
        onsets, join = join_nemar_events(record, events, sample_rate_hz=1000.)
        values.append((record, events, onsets, join, r))
    return values


def test_actual_twelve_physical_symbols_32_choices_and_whole_selection_endpoint(original_join):
    record, _, _, _, _ = original_join[0]
    contract = CandidateTaskContract.from_record(CANDIDATE_TASK.record())
    assert contract.candidate_ids == SELECTABLE_POSITIONS and contract.blank_position_ids == BLANK_POSITIONS
    assert contract.membership_kind == "set_membership"
    for ordinal in range(7):
        decision = membership_decision(record, ordinal, event_prefix="original-Huebner")
        assert decision.membership.shape == (68, 32) and decision.repetitions == 1
        assert decision.schedule_complete and decision.protocol == "set_membership"
        np.testing.assert_array_equal(decision.membership.sum(axis=0), 16)
        ideal = decision.membership[:, decision.candidate_vocabulary.index(f"position:{record.target_positions[ordinal]}")]
        scores = decision.membership.T@ideal.astype(int)
        assert np.argmax(scores) == decision.candidate_vocabulary.index(f"position:{record.target_positions[ordinal]}")
        assert len({tuple(x) for x in decision.membership.T}) == 32


def test_run_specific_original_optical_join_does_not_guess_global_offset(original_join):
    assert [r[3]["clock_offset_samples"] for r in original_join] == [3825, 3756]
    record, events, _, _, _ = original_join[0]
    altered = events.decode("utf-8-sig").splitlines()
    fields = altered[1].split("\t")
    header = altered[0].split("\t")
    fields[header.index("sample")] = str(int(fields[header.index("sample")])+1)
    fields[header.index("onset")] = str(float(fields[header.index("onset")])+.001)
    altered[1] = "\t".join(fields)
    with pytest.raises(ValueError, match="one exact run-specific offset"):
        join_nemar_events(record, "\n".join(altered).encode(), sample_rate_hz=1000.)


def test_future_public_marker_record_strips_target_classes_and_keeps_visual_blank_positions(original_join):
    record = original_join[0][0]
    assert any(set(f.physical_positions)&set(BLANK_POSITIONS) for f in record.flashes)
    for flash in record.flashes:
        physical = flash.public_record()
        assert not {"target_positions", "truth_codes", "raw_target_label", "is_target", "code"}&set(physical)
        other = replace(flash, target_positions=())
        assert physical == other.public_record()


def test_missing_eeg_evidence_preserves_every_scheduled_requested_event(original_join):
    record = original_join[0][0]
    indices = np.arange(68, dtype=np.int64)
    indices[4] = -1
    decision = membership_decision(record, 0, event_prefix="original-Huebner", evidence_indices=indices)
    assert decision.schedule_complete and len(decision.event_ids) == 68
    assert np.all(decision.membership.sum(axis=0) == 16)
    assert np.count_nonzero(decision.evidence_indices < 0) == 1


def test_original_member_crc_failure_and_target_label_join_failure_are_explicit(original_join):
    record, events, _, _, r = original_join[0]
    damaged = {**r["member"], "sha256": "f"*64}
    with pytest.raises(ValueError, match="corrupt bytes"):
        read_local_marker_member(r["original_zip"], damaged)
    text = events.decode("utf-8")
    attacked = text.replace("NonTarget", "Target", 1)
    with pytest.raises(ValueError, match="target bit"):
        join_nemar_events(record, attacked.encode(), sample_rate_hz=1000.)
