"""Actual trial fits, physical row mapping and interrupted arm reuse."""
from dataclasses import replace

import numpy as np
import pytest

from data.candidate_task import MembershipDecision
from models.candidate_evidence import MEMBERSHIP_SCORE_METHOD, CandidateExpertIdentity
from research.trial_candidate_execution import (
    TrialCandidateDecision,
    _evidence,
    _secondary_diagnostics,
    execute_trial_candidate_participant,
)
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset
from transfer.ensemble_fit import FrozenSourceCheckpoint
from transfer.recipes import TargetRecipe


def test_trial_scores_map_physical_global_rows_before_membership_sum():
    decision = MembershipDecision.direct(("left", "right", "left", "right"), ("left", "right"),
        event_ids=("a", "b", "c", "d"), repetition_indices=np.array([0, 0, 1, 1]), repetitions=2,
        evidence_indices=np.array([5, 3, 1, 4]))
    request = TrialCandidateDecision("person", "decision", decision, "a" * 64, 2)
    identity = CandidateExpertIdentity("expert", "b" * 64, "c" * 64, None, "d" * 64, MEMBERSHIP_SCORE_METHOD, False)
    logits = np.array([100., 2., 200., 3., 4., 5.])
    evidence = _evidence(request, logits, identity)
    np.testing.assert_allclose(evidence.probability, [.5, .5], rtol=0, atol=1e-15)
    missing = replace(request, decision=replace(decision, evidence_indices=np.array([5, -1, 1, 4])))
    assert _evidence(missing, logits, identity).failure_kind == "incomplete"
    censored = replace(request, unavailable_reason="terminal_partial_raw_episode")
    assert _evidence(censored, logits, identity).log_probability is None


def test_trial_real_three_arms_and_interrupted_exact_reuse(tmp_path, monkeypatch):
    import research.trial_candidate_execution as execution

    calibration = binary_dataset(count=40)
    future = replace(calibration, y=None, X=calibration.X + 1e-7)
    decision = MembershipDecision.direct(("left", "right"), ("left", "right"),
        event_ids=("e38", "e39"), repetition_indices=np.array([0, 0]), repetitions=1,
        evidence_indices=np.array([38, 39]))
    request = TrialCandidateDecision("person", "future", decision, "a" * 64, 1)
    source = FrozenSourceCheckpoint.from_path(write_checkpoint(tmp_path / "source.pt", calibration,
        source_snapshot_sha256="b" * 64))
    recipe = TargetRecipe("source_full_fine", epochs=1, batch_size=32,
        trial_loss_reduction="calibration_balanced_fixed_mass")
    recipes = (recipe, replace(recipe, name="matched_scratch", initialization="scratch"))
    arguments = dict(calibration=calibration, future=future, requests=(request,), source=source, recipes=recipes,
        local_subject=str(calibration.subject_ids[0]), output=tmp_path / "participant",
        source_snapshot_sha256="b" * 64, execution_source_sha256="c" * 64)
    actual_fit = execution.fit_calibration_candidate
    calls = []

    def interrupted_fit(*args, **kwargs):
        raise RuntimeError("actual interrupted target fit")

    monkeypatch.setattr(execution, "fit_calibration_candidate", interrupted_fit)
    with pytest.raises(RuntimeError, match="interrupted target"):
        execute_trial_candidate_participant(**arguments)
    zero_shot_bytes = (tmp_path / "participant/source_zero_shot.receipt.json").read_bytes()

    def observed_fit(*args, **kwargs):
        calls.append(args[1].initialization)
        return actual_fit(*args, **kwargs)

    monkeypatch.setattr(execution, "fit_calibration_candidate", observed_fit)
    result = execute_trial_candidate_participant(**arguments)
    assert calls == ["source", "scratch"]
    assert [row["status"] for row in result["arms"]] == ["completed"] * 3
    assert all(row["saved_replay_exact"] for row in result["arms"])
    assert (tmp_path / "participant/source_zero_shot.receipt.json").read_bytes() == zero_shot_bytes
    assert result["arms"][1]["fit"]["calibration_labels_digest"] == result["arms"][2]["fit"]["calibration_labels_digest"]
    assert result["arms"][2]["predictions"][0]["expert_identities"][0]["source_digest"] is None
    execute_trial_candidate_participant(**arguments)
    assert calls == ["source", "scratch"]
    changed = dict(arguments, future=replace(future, X=future.X * 2))
    with pytest.raises(ValueError, match="actual input/recipe/source"):
        execute_trial_candidate_participant(**changed)


def test_labelled_future_rejected_before_any_output(tmp_path):
    data = binary_dataset(count=40)
    with pytest.raises(ValueError, match="label-free"):
        execute_trial_candidate_participant(calibration=data, future=data, requests=(), source=None,
            recipes=(), local_subject="person", output=tmp_path / "output", source_snapshot_sha256="a" * 64,
            execution_source_sha256="b" * 64)
    assert not (tmp_path / "output").exists()


def test_terminal_partial_keeps_complete_main_prefix_and_null_censor_secondary():
    decision = MembershipDecision.row_column(np.array([0, 1, 2, 3, 0, 2]), n_rows=2, n_columns=2,
        event_ids=tuple(f"event{i}" for i in range(6)), repetition_indices=np.array([0, 0, 0, 0, 1, 1]),
        repetitions=2)
    request = TrialCandidateDecision("person", "original_episode", decision, "a" * 64, 1,
        None, (2, 2, 2, 2), 2)
    identity = CandidateExpertIdentity("expert", "b" * 64, "c" * 64, None, "d" * 64, MEMBERSHIP_SCORE_METHOD, False)
    logits = np.arange(6, dtype=float)
    main = _evidence(request, logits, identity)
    assert main.decision_id == "original_episode" and main.evidence_level == 1
    assert main.log_probability is not None and main.expected_event_counts == (2, 2, 2, 2)
    secondary = _secondary_diagnostics((request,), logits, identity)
    assert len(secondary) == 1 and secondary[0]["primary_evidence_level"] == 1
    assert secondary[0]["evidence"]["decision_id"] == main.decision_id
    assert secondary[0]["evidence"]["evidence_level"] == 2
    assert secondary[0]["evidence"]["log_probability"] is None
    assert secondary[0]["evidence"]["expected_event_counts"] == [4, 4, 4, 4]
