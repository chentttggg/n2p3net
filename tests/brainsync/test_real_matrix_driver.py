"""Recording-wide retained rows must map to each original 45-flash block."""

from types import SimpleNamespace

import numpy as np

from experiments.run_brainsync_real_matrix import block_scores


def fixture(removed=()):
    candidates = np.tile(np.arange(1, 10).astype(str), 50)
    original_logits = np.full(450, -2.0)
    for block in range(10):
        target = str(block % 9 + 1)
        rows = np.arange(block * 45, (block + 1) * 45)
        original_logits[rows[candidates[rows] == target]] = 5.0
    retained = ~np.isin(np.arange(450), removed)
    mapping = np.full(450, -1, dtype=np.int64)
    mapping[retained] = np.arange(retained.sum())
    timeline = SimpleNamespace(
        n_events=450,
        group_ids=np.repeat("original_recording", 450),
        candidate_ids=candidates,
        evidence_indices=mapping,
        event_ids=np.array([f"original_{i}" for i in range(450)]),
    )
    dataset = SimpleNamespace(
        event_timeline=timeline, y=(candidates[retained] == "3").astype(int), provenance={}
    )
    entry = {"session_id": "session", "target": 3, "paths": {"basic": {"sha256": "a" * 64}}}
    return dataset, original_logits[retained], entry


def test_each_block_uses_its_own_recording_rows():
    dataset, logits, entry = fixture()
    result = block_scores(dataset, logits, entry)
    assert [b["prediction"] for b in result["blocks"]] == [str(i % 9 + 1) for i in range(10)]
    assert all(b["eligible"] and b["counts"] == [5] * 9 for b in result["blocks"])


def test_dropped_event_retains_request_and_does_not_shift_later_block():
    dataset, logits, entry = fixture(removed=(8, 53))
    result = block_scores(dataset, logits, entry)
    assert len(result["blocks"]) == 10
    assert result["requested_trials"] == 450
    assert not result["blocks"][0]["eligible"]
    assert result["blocks"][0]["counts"] == [5] * 8 + [4]
    assert not result["blocks"][1]["eligible"]
    assert [b["prediction"] for b in result["blocks"][2:]] == [str(i % 9 + 1) for i in range(2, 10)]


def test_nonfinite_model_score_is_a_missing_physical_event():
    dataset, logits, entry = fixture()
    logits[0] = np.nan
    result = block_scores(dataset, logits, entry)
    assert not result["blocks"][0]["eligible"]
    assert result["blocks"][0]["counts"][0] == 4
    assert result["available_trial_scores"] == 449


def test_evaluation_truth_cannot_change_candidate_scores():
    dataset, logits, entry = fixture()
    first = block_scores(dataset, logits, entry)
    second = block_scores(dataset, logits, {**entry, "target": 9})
    assert [b["scores"] for b in first["blocks"]] == [b["scores"] for b in second["blocks"]]
    assert [b["prediction"] for b in first["blocks"]] == [b["prediction"] for b in second["blocks"]]
