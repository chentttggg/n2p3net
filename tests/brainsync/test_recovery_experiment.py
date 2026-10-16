"""Recovered cycles keep the nine-candidate vocabulary and original cost."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.run_brainsync_recovery import (
    complete_cycle_diagnostic,
    quality_policy_from_plan,
    validate_evaluation_scope,
)


def _scheduled():
    return SimpleNamespace(event_timeline=SimpleNamespace(
        n_events=45, evidence_indices=np.arange(45), candidate_ids=np.tile(np.arange(1, 10), 5),
        target_candidate_ids=np.full(45, 4)))


def test_recovered_cycles_discard_entire_incomplete_permutation_and_keep_original_indices():
    dataset = _scheduled()
    logits = np.tile(np.array([0, 0, 0, 1., 0, 0, 0, 0, 0]), 5)
    logits[[0, 10, 20]] = np.nan
    logits[1] = 1000.
    record = complete_cycle_diagnostic(dataset, logits, minimum_cycles=2)
    block = record["blocks"][0]
    assert block["complete_cycle_indices"] == [3, 4]
    assert block["used_scheduled_event_indices"] == list(range(27, 45))
    assert block["prediction"] == 4
    assert record["requested_blocks"] == 1 and record["predicted_blocks"] == 1
    assert complete_cycle_diagnostic(dataset, logits, minimum_cycles=3)["predicted_blocks"] == 0
    permuted = _scheduled()
    permuted.event_timeline.target_candidate_ids[:] = 7
    other = complete_cycle_diagnostic(permuted, logits, minimum_cycles=2)["blocks"][0]
    assert other["prediction"] == block["prediction"] and not other["hit"]
    assert other["used_scheduled_event_indices"] == block["used_scheduled_event_indices"]


def test_recovered_cycle_ties_empty_evidence_and_invalid_schedule_never_guess():
    dataset = _scheduled()
    for logits in (np.zeros(45), np.full(45, np.nan)):
        record = complete_cycle_diagnostic(dataset, logits, minimum_cycles=2)
        assert record["predicted_blocks"] == 0 and record["requested_blocks"] == 1
        assert record["hit_including_abstentions"] == 0
    dataset.event_timeline.candidate_ids[0] = 2
    with pytest.raises(ValueError, match="balanced"):
        complete_cycle_diagnostic(dataset, np.zeros(45), minimum_cycles=2)


def test_json_quality_plan_preserves_frozen_grids_and_rejects_invalid_order_before_fitting():
    import json
    from dataclasses import asdict

    from data.artifact import FoldLocalArtifactPolicy

    expected = FoldLocalArtifactPolicy()
    record = json.loads(json.dumps(asdict(expected)))
    assert quality_policy_from_plan(record) == expected
    assert quality_policy_from_plan(None) is None
    record["candidate_quantiles"].reverse()
    with pytest.raises(ValueError, match="sorted and unique"):
        quality_policy_from_plan(record)


def test_independent_recording_holdout_uses_disjoint_scope_not_acquisition_order(monkeypatch):
    import experiments.run_brainsync_recovery as runner

    calibration = SimpleNamespace(sessions=[{"session_id": "r2", "calibration_available_timestamp_s": 20},
                                            {"session_id": "r4", "calibration_available_timestamp_s": 40}],
                                  event_timeline=SimpleNamespace(event_ids=["r2a", "r4a"]),
                                  participant="P001", contract="cz6")
    evaluation = SimpleNamespace(sessions=[{"session_id": "r3", "session_start_timestamp_s": 30}],
                                 event_timeline=SimpleNamespace(event_ids=["r3a"]),
                                 participant="P001", contract="cz6")
    monkeypatch.setattr(runner, "_session_scope", lambda d, **kw: d.sessions)
    monkeypatch.setattr(runner, "_participant", lambda d: d.participant)
    monkeypatch.setattr(runner, "_input_contract", lambda d: d.contract)
    with pytest.raises(ValueError, match="precedes"):
        validate_evaluation_scope(calibration, evaluation)
    record = validate_evaluation_scope(calibration, evaluation, mode="independent_recording_holdout")
    assert record["disjoint_sessions"] and not record["evaluation_after_calibration_ready"]
    with pytest.raises(ValueError, match="unknown"):
        validate_evaluation_scope(calibration, evaluation, mode="ignore_checks")
    evaluation.participant = "P002"
    with pytest.raises(ValueError, match="participant"):
        validate_evaluation_scope(calibration, evaluation, mode="independent_recording_holdout")
    evaluation.participant = "P001"
    evaluation.contract = "different"
    with pytest.raises(ValueError, match="contracts"):
        validate_evaluation_scope(calibration, evaluation, mode="independent_recording_holdout")
    evaluation.contract = "cz6"
    evaluation.event_timeline.event_ids = ["r2a"]
    with pytest.raises(ValueError, match="events"):
        validate_evaluation_scope(calibration, evaluation, mode="independent_recording_holdout")
    evaluation.event_timeline.event_ids = ["r3a"]
    evaluation.sessions[0]["session_id"] = "r4"
    with pytest.raises(ValueError, match="sessions"):
        validate_evaluation_scope(calibration, evaluation, mode="independent_recording_holdout")
