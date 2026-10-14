"""Dataset-independent requests retain physical depth and hide parent truth."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.candidate_task import MembershipDecision
from data.digests import file_sha256
from data.identity import DatasetIdentityTable
from research.trial_candidate_programme import _load_prepared, compile_participants
from tests.support.datasets import binary_dataset
from transfer.recipes import TargetRecipe


def _daily_plan(seed_pairs):
    recipe = TargetRecipe("source_full_fine", seed=20261014,
                          trial_loss_reduction="calibration_balanced_fixed_mass")
    return {"schema": "n2p3_trial_candidate_programme/1", "seed_pairs": seed_pairs,
            "recipes": [recipe.record(), replace(recipe, name="matched_scratch", initialization="scratch").record()],
            "execution_manifest": "frozen-source.json"}


@pytest.mark.parametrize("seed", [20260918, 20261014])
def test_declared_daily_seed_reaches_original_snapshot_admission(tmp_path, monkeypatch, seed):
    import research.trial_candidate_programme as programme

    plan = _daily_plan([[seed, seed]])
    plan["recipes"] = [{**row, "seed": seed} for row in plan["recipes"]]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    class OriginalSnapshotAdmissionReachedError(Exception):
        pass

    def original_admission(*args, **kwargs):
        raise OriginalSnapshotAdmissionReachedError

    monkeypatch.setattr(programme, "verified_executed_snapshot", original_admission)
    with pytest.raises(OriginalSnapshotAdmissionReachedError):
        programme.run_programme(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("pairs", [[], [[20261014, 20261014], [20261015, 20261015]],
                                   [[20260918, 20261014]], [[True, True]], [[-1, -1]], [[20261014]]])
def test_daily_seed_rejects_unpaired_or_multiple_programmes_before_snapshot(tmp_path, monkeypatch, pairs):
    import research.trial_candidate_programme as programme

    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_daily_plan(pairs)))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid seed declaration reached source admission")

    monkeypatch.setattr(programme, "verified_executed_snapshot", forbidden)
    with pytest.raises(ValueError, match="one-seed"):
        programme.run_programme(path, tmp_path / "output")


def test_new_daily_seed_keeps_recipe_and_actual_source_seed_admission(tmp_path, monkeypatch):
    import research.trial_candidate_execution as execution
    import research.trial_candidate_programme as programme
    from research.trial_candidate_execution import TrialCandidateDecision
    from tests.support.checkpoints import write_checkpoint
    from transfer.ensemble_fit import FrozenSourceCheckpoint

    plan = _daily_plan([[20261014, 20261014]])
    plan["recipes"][1]["seed"] = 20260918
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="same declared target procedure"):
        programme.run_programme(path, tmp_path / "recipe-mismatch")

    calibration = binary_dataset(count=40)
    future = replace(calibration, y=None)
    source = FrozenSourceCheckpoint.from_path(write_checkpoint(tmp_path / "source.pt", calibration,
                                                              source_snapshot_sha256="b" * 64))
    decision = MembershipDecision.direct(("left", "right"), ("left", "right"),
        event_ids=("e38", "e39"), repetition_indices=np.array([0, 0]), repetitions=1,
        evidence_indices=np.array([38, 39]))
    request = TrialCandidateDecision("person", "future", decision, "a" * 64, 1)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("wrong source seed reached target fitting")

    monkeypatch.setattr(execution, "fit_calibration_candidate", forbidden_fit)
    recipes = tuple(TargetRecipe(**row) for row in _daily_plan([[20261014, 20261014]])["recipes"])
    with pytest.raises(ValueError, match="declared source seed"):
        execution.execute_trial_candidate_participant(calibration=calibration, future=future,
            requests=(request,), source=source, recipes=recipes, local_subject=str(calibration.subject_ids[0]),
            output=tmp_path / "source-mismatch", source_snapshot_sha256="b" * 64,
            execution_source_sha256="c" * 64, source_seed=20261014)


@pytest.mark.parametrize("dataset_id,subject,depth,extra", [
    ("BNCI2014-008", "A01", 10, 0), ("BI2014a", "57", 1, 2),
])
def test_generic_requests_keep_original_keys_depth_output_identity_and_censor(
    tmp_path, monkeypatch, dataset_id, subject, depth, extra,
):
    import research.trial_candidate_programme as programme

    count = 12 * depth + extra
    stimuli = np.tile(np.arange(12), depth)
    if extra:
        stimuli = np.r_[stimuli, np.arange(extra)]
    decision = MembershipDecision.row_column(stimuli, n_rows=6, n_columns=6,
        event_ids=tuple(f"event:{index}" for index in range(count)),
        repetition_indices=np.arange(count) // 12, repetitions=depth + bool(extra))
    calibration = replace(binary_dataset(count=40), subject_ids=np.repeat(subject, 40),
        identity_table=DatasetIdentityTable.from_source_rows([subject], [dataset_id]))
    calibration.provenance["source_sha256"] = "a" * 64
    source = {"path": "source.pt", "sha256": "b" * 64}
    descriptor = {"episode_id": "original", "membership_decision": decision.record(),
        "complete_raw_repetitions": depth, "terminal_censor": bool(extra)}
    ledger = {"dataset_id": dataset_id, "subject": subject, "calibration_episode_count": 5,
        "source_profile_checkpoint": source, "future_decisions": [descriptor],
        "parent_scoring_truth": {"original": 0}, "calibration_cache": {"path": "calibration.npz"},
        "future_cache": {"path": "future.npz"}}
    prepared = SimpleNamespace(calibration=calibration, ledger=ledger)
    monkeypatch.setattr(programme, "_load_prepared", lambda directory, declared: prepared)
    directory = tmp_path / subject
    directory.mkdir()
    preparation = directory / "preparation.json"
    preparation.write_text("{}")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"status": "completed", "requested_subjects": [subject],
        "participants": [{"preparation": {"path": str(preparation), "sha256": file_sha256(preparation)}}]}))
    plan = {"dataset_id": dataset_id, "preparation_cohort": {"path": str(cohort), "sha256": file_sha256(cohort)},
        "calibration_episode_count": 5, "maximum_horizon": depth, "source_checkpoint": source,
        "recipes": [], "source_snapshot_sha256": "c" * 64, "device": "cpu",
        "seed_pairs": [[20260918, 20260918]]}
    actors, decisions, horizons, coverage, _ = compile_participants(plan, "d" * 64, tmp_path / "output")
    assert len(decisions.entries) == len(horizons) == len(coverage) == 1
    assert set(horizons.values()) == {depth}
    actor = actors[0]
    assert actor["dataset_id"] == dataset_id and actor["local_subject"] == subject
    assert Path(actor["output"]).name == subject
    assert actor["requests"][0]["expected_event_counts"] == (2 * depth,) * 36
    assert actor["requests"][0]["secondary_evidence_level"] == (depth + 1 if extra else None)
    assert "parent_scoring_truth" not in json.dumps(actor)


def test_unregistered_dataset_has_no_guessed_preparation_owner(tmp_path):
    with pytest.raises(ValueError, match="no admitted dataset"):
        _load_prepared(tmp_path, "unregistered")
