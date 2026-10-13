"""Requested calibration risk resists abstention and survives score replay."""

import copy
import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tests.support.datasets import binary_dataset
from transfer import calibration_candidates as candidates


def scoring_data():
    data = binary_dataset(6)
    data.y[:] = [0, 0, 0, 1, 1, 1]
    return data, np.ones(6, dtype=bool)


def test_abstention_cannot_turn_one_ninth_requested_auc_into_one():
    data, mask = scoring_data()
    partial = np.array([0., np.nan, np.nan, 1., np.nan, np.nan])
    result = candidates.score_calibration_candidate(data, partial, mask)
    assert result["validation_auc"] == 1.
    assert result["primary_score"] == pytest.approx(1 / 9)
    assert result["requested_validation_pairs"] == 9
    assert result["available_validation_pairs"] == 1
    complete = candidates.score_calibration_candidate(data, np.array([3., 0., 1., 2., .5, -.5]), mask)
    assert complete["primary_score"] == pytest.approx(1 / 3)
    assert complete["primary_score"] > result["primary_score"]


def test_all_missing_or_missing_class_stays_in_requested_denominator():
    data, mask = scoring_data()
    for values in (np.full(6, np.nan), np.array([0., 1., 2., np.nan, np.nan, np.nan])):
        result = candidates.score_calibration_candidate(data, values, mask)
        assert result["primary_score"] == 0
        assert result["validation_auc"] is None
        assert result["requested_validation_pairs"] == 9
        assert not result["selection_eligible"]


def test_every_deletion_loses_only_nonnegative_requested_pair_credit():
    data, mask = scoring_data()
    original = np.array([0., 2., 1., 1., -1., 3.])
    for bits in itertools.product((False, True), repeat=6):
        keep = np.array(bits)
        values = np.where(keep, original, np.nan)
        result = candidates.score_calibration_candidate(data, values, mask)
        for index in np.flatnonzero(keep):
            removed = values.copy()
            removed[index] = np.nan
            other = candidates.score_calibration_candidate(data, removed, mask)
            assert other["primary_score"] <= result["primary_score"]


def test_complete_auc_and_requested_auc_are_scale_and_permutation_invariant():
    data, mask = scoring_data()
    values = np.array([0., 2., 1., 1., -1., 3.])
    expected = roc_auc_score(data.y, values)
    for changed in (values, values * 1e200 + 1e200, values * 1e-200):
        result = candidates.score_calibration_candidate(data, changed, mask)
        assert result["primary_score"] == pytest.approx(expected)
        assert result["primary_score"] == pytest.approx(result["validation_auc"])
    permutation = np.array([4, 0, 2, 5, 3, 1])
    data.y[:] = data.y[permutation]
    assert candidates.score_calibration_candidate(data, values[permutation], mask)["primary_score"] == pytest.approx(expected)


def fake_inner_fits(monkeypatch, predictions, *, histories=None):
    real_fit = candidates.fit_calibration_candidate
    calls = []

    def fit(data, recipe, training, **kwargs):
        calls.append((recipe.name, np.flatnonzero(training).tolist()))
        if int(training.sum()) == data.n_epochs:
            return real_fit(data, recipe, training)
        history = {"fit_scope": {"training_rows": int(training.sum()),
                                  "validation_rows": len(predictions[recipe.name])}}
        if histories is not None:
            history.update(histories[recipe.name])
        return SimpleNamespace(predict_logit=lambda *args: predictions[recipe.name].copy(),
                               state={}, fit_history=history)

    monkeypatch.setattr(candidates, "fit_calibration_candidate", fit)
    return calls


def test_selector_uses_requested_risk_and_refits_authorized_full_scope(monkeypatch):
    data = binary_dataset(40)
    # Eight held rows alternate classes. The partial candidate's conditional
    # AUC is perfect, but it supplies only one of the sixteen requested pairs.
    predictions = {"complete": np.array([0., 1., 2., 3., 4., 5., 6., 7.]),
                   "partial": np.array([0., 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])}
    calls = fake_inner_fits(monkeypatch, predictions)
    recipes = tuple(replace(candidates.waveform_lda_recipe(), name=name) for name in predictions)
    fitted, record = candidates.select_calibration_candidate(data, recipes,
        calibration_mask=np.ones(40, bool), selector_mode="no_transfer")
    assert fitted.recipe.name == record["selected"]["candidate_id"] == "complete"
    assert calls[-1] == ("complete", list(range(40)))
    assert record["recipes"][1]["validation_auc"] == 1.
    assert record["recipes"][1]["primary_score"] == 1 / 16
    from transfer.calibration_candidates import replay_calibration_selection

    assert replay_calibration_selection(record, dataset=data)["candidate_id"] == "complete"
    changed = copy.deepcopy(data)
    changed.X[0] *= 3  # Outside held scoring scope; the fitted recipe is unchanged.
    assert replay_calibration_selection(record, dataset=changed)["candidate_id"] == "complete"
    changed.y[32] = 1 - changed.y[32]
    with pytest.raises(ValueError, match="scoring support"):
        replay_calibration_selection(record, dataset=changed)
    tampered = copy.deepcopy(record)
    tampered["recipes"][1]["primary_score"] = 1.
    with pytest.raises(ValueError, match="score replay"):
        replay_calibration_selection(tampered)
    tampered = copy.deepcopy(record)
    tampered["tie_rule"] = "conditional_auc"
    with pytest.raises(ValueError, match="tie rule"):
        replay_calibration_selection(tampered)


def test_conditional_auc_cannot_break_a_requested_risk_tie(monkeypatch):
    data = binary_dataset(40)
    # Both candidates receive zero pair credit. Completeness wins the tie.
    predictions = {"z_complete": np.tile([1., 0.], 4),
                   "a_partial": np.array([1., 0., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])}
    fake_inner_fits(monkeypatch, predictions)
    recipes = tuple(replace(candidates.waveform_lda_recipe(), name=name) for name in predictions)
    _, record = candidates.select_calibration_candidate(data, recipes,
        calibration_mask=np.ones(40, bool), selector_mode="no_transfer")
    assert record["selected"]["candidate_id"] == "z_complete"


def test_legacy_ce_uses_one_requested_measure_and_requires_complete_output(monkeypatch):
    data = binary_dataset(40)
    predictions = {"complete": np.tile([-1., 1.], 4), "poor_complete": np.tile([1., -1.], 4),
                   "partial": np.array([-10., 10., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])}
    histories = {name: {"best_epoch": 1, "best_val_loss": 99. if name == "complete" else 0.001}
                 for name in predictions}
    fake_inner_fits(monkeypatch, predictions, histories=histories)
    recipes = tuple(replace(candidates.waveform_lda_recipe(), name=name) for name in predictions)
    _, record = candidates.select_calibration_candidate(data, recipes,
        calibration_mask=np.ones(40, bool), selector_mode="legacy_product")
    assert record["selected"]["candidate_id"] == "complete"
    assert record["selected"]["validation_selection_ce"] == pytest.approx(np.logaddexp(0., -1.))
    assert not record["recipes"][2]["selection_eligible"]


def test_real_public_selector_persists_required_selection_and_admits_it(tmp_path):
    import hashlib

    import torch

    from data.study import dataset_study_membership
    from research.execution import atomic_write_json
    from tests.support.checkpoints import write_checkpoint
    from transfer.binary import binary_time_split
    from transfer.result_reuse import (
        ResultIntegrityError,
        admit_completed_result,
        seal_result_record,
    )
    from transfer.target_experiments import build_target_tasks, execute_target_tasks

    data = binary_dataset(40)
    source = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="a" * 64)
    timeline = data.event_timeline
    split = binary_time_split(timeline.onset_times_s, timeline.evidence_available_times_s,
        calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    manifest = build_target_tasks(data, {"p1": (np.arange(40), {20: split})}, stage="E3", batch_size=512)
    manifest["tasks"] = [task for task in manifest["tasks"] if task["selector_mode"] == "no_transfer"]
    manifest["tasks"][0]["recipes"] = [candidates.waveform_lda_recipe(batch_size=512).record()]
    context = dict(target_study="synthetic_p300", source_seed=1,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), target_cache_sha256="b" * 64,
        evaluation_snapshot={"archive_sha256": "c" * 64}, target_protocol={"kind": "single_recording_prefix"},
        environment={"device": "cpu", "threads_per_worker": 1},
        source_study_membership={"digest": "d" * 64}, target_study_membership=dataset_study_membership(data),
        evidence_scope="software_correctness")
    directory = tmp_path / "public"
    report = execute_target_tasks(data, torch.load(source, weights_only=True), manifest,
        output=directory, context=context, device="cpu")
    assert report["completed_cells"] == 1 and report["failed_cells"] == 0
    expected = {**context, "task_manifest": manifest}
    admit_completed_result(directory / "record.json", kind="public_target", expected=expected)
    cell = report["cells"][0]
    model_path = directory / "00000.pt"
    model = torch.load(model_path, weights_only=False)
    assert model["selection"] == cell["selection"]
    assert model["selection"]["refit_rows"] == 20
    altered = copy.deepcopy(cell["selection"])
    altered["selector_mode"] = "five_candidates"
    assert candidates.replay_calibration_selection(altered)["candidate_id"] == "waveform_lda"
    model["selection"] = cell["selection"] = altered
    torch.save(model, model_path)
    cell = seal_result_record(cell, files={"model": model_path, "prediction": directory / "00000.npz"})
    report["cells"][0] = cell
    atomic_write_json(directory / "00000.json", cell)
    atomic_write_json(directory / "record.json", seal_result_record(report))
    with pytest.raises(ResultIntegrityError, match="candidate selection"):
        admit_completed_result(directory / "record.json", kind="public_target", expected=expected)
    # Deleting the optional-looking fields and re-sealing every changed file
    # still fails the current run schema's required selector contract.
    del model["selection"]
    del cell["selection"]
    torch.save(model, model_path)
    cell = seal_result_record(cell, files={"model": model_path, "prediction": directory / "00000.npz"})
    report["cells"][0] = cell
    atomic_write_json(directory / "00000.json", cell)
    atomic_write_json(directory / "record.json", seal_result_record(report))
    with pytest.raises(ResultIntegrityError, match="candidate selection"):
        admit_completed_result(directory / "record.json", kind="public_target", expected=expected)


def test_r5_abstention_keeps_the_physical_block_denominator(tmp_path):
    from data.brainsync import load_brainsync_sessions
    from tests.support.brainsync import write_bids_session

    root = write_bids_session(tmp_path / "r5", session_id="cal", target=3,
        started_utc="2026-09-06T08:00:00Z", blocks=(tuple(range(1, 10)) * 5,) * 2)
    data = load_brainsync_sessions((root,))
    validation = np.ones(data.n_epochs, bool)
    values = data.y.astype(float)
    second = data.metadata["block_id"].to_numpy() == 2
    values[second] = -values[second]
    complete = candidates.score_calibration_candidate(data, values, validation, protocol="brainsync_operational_r5")
    values[second] = np.nan
    partial = candidates.score_calibration_candidate(data, values, validation, protocol="brainsync_operational_r5")
    assert complete["primary_score"] == partial["primary_score"] == .5
    assert partial["requested_pair_auc"] <= complete["requested_pair_auc"]
    assert partial["requested_operational_blocks"] == 2 and partial["complete_operational_blocks"] == 1
    assert partial["blocks"][1]["prediction"] is None and partial["blocks"][1]["hit"] == 0
