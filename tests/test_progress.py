"""Shared training-progress interface tests."""

from __future__ import annotations

import json

import numpy as np

from baselines.classic import Baseline
from baselines.deep import DeepBaseline
from baselines.evaluate import evaluate_binary, loso_folds
from tests.test_epochs import _dataset
from train.factory import BINARY_MODEL_NAMES, build_binary_model


def test_epoch_progress_sink_is_model_and_fold_agnostic(tmp_path):
    model = Baseline()
    model.configure_epoch_progress(tmp_path)
    model.configure_evaluation_fold(7)

    callback = model.epoch_progress_callback()
    assert callback is not None
    callback({"epoch": 1, "epoch_limit": 2, "task_val_auc": 0.75})

    rows = [json.loads(line) for line in (tmp_path / "fold_7.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "epoch"
    assert rows[0]["fold"] == 7
    assert rows[0]["task_val_auc"] == 0.75


def test_epoch_progress_event_cannot_override_reserved_fields(tmp_path):
    model = Baseline()
    model.configure_epoch_progress(tmp_path)
    model.configure_evaluation_fold(3)
    model.epoch_progress_callback()({"epoch": 0, "fold": 99, "type": "spoofed"})

    row = json.loads((tmp_path / "fold_3.jsonl").read_text(encoding="utf-8"))
    assert row["type"] == "epoch"
    assert row["fold"] == 3


def test_epoch_progress_truncates_stale_fold_history_on_rerun(tmp_path):
    first = Baseline()
    first.configure_epoch_progress(tmp_path)
    first.configure_evaluation_fold(2)
    first.epoch_progress_callback()({"epoch": 99, "task_val_auc": 0.1})

    rerun = Baseline()
    rerun.configure_epoch_progress(tmp_path)
    rerun.configure_evaluation_fold(2)
    callback = rerun.epoch_progress_callback()
    callback({"epoch": 1, "task_val_auc": 0.8})
    callback({"epoch": 2, "task_val_auc": 0.9})

    rows = [json.loads(line) for line in (tmp_path / "fold_2.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]


def test_epoch_progress_truncates_stale_history_before_first_epoch(tmp_path):
    target = tmp_path / "fold_2.jsonl"
    target.write_text('{"epoch": 99}\n', encoding="utf-8")
    model = Baseline()
    model.configure_epoch_progress(tmp_path)
    model.configure_evaluation_fold(2)

    callback = model.epoch_progress_callback()

    assert callback is not None
    assert target.read_text(encoding="utf-8") == ""


def test_binary_model_factory_routes_registered_deep_models_without_dataset_branches():
    dataset = _dataset()
    model = build_binary_model("EEGNet", dataset, epochs=1, batch_size=4)

    assert "n2p3net" in BINARY_MODEL_NAMES
    assert isinstance(model, DeepBaseline)
    assert model.model_name == "eegnet"


class _ProgressModel(Baseline):
    fit_accepts_subject_ids = True

    def fit(self, X, y, subject_ids=None):
        callback = self.epoch_progress_callback()
        assert callback is not None
        callback({"epoch": 1, "epoch_limit": 1, "task_val_auc": 0.5})
        self.calibration_logits_ = np.array([-1.0, 1.0])
        self.calibration_labels_ = np.array([0, 1])
        self.calibration_source_ = "subject_disjoint_validation"
        self.last_history = {
            "train_losses": [1.0],
            "val_losses": [1.0],
            "val_objective_losses": [1.0],
            "task_val_aucs": [0.5],
            "final_task_val_auc": 0.5,
        }
        return self

    def predict_logit(self, X):
        return np.asarray(X)[:, 0, 0]


def test_binary_evaluation_passes_explicit_fold_context_to_parallel_workers(tmp_path):
    X = np.zeros((6, 1, 1), dtype=np.float32)
    X[:, 0, 0] = [-3.0, 3.0, -2.0, 2.0, -1.0, 1.0]
    y = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.repeat(np.arange(3), 2)
    model = _ProgressModel()
    model.configure_epoch_progress(tmp_path)

    summary = evaluate_binary(
        model,
        X,
        y,
        subjects,
        loso_folds(subjects),
        n_jobs=2,
        fold_id_offset=5,
    )

    assert len(summary.per_fold) == 3
    for fold_id in (5, 6, 7):
        rows = [
            json.loads(line)
            for line in (tmp_path / f"fold_{fold_id}.jsonl").read_text().splitlines()
        ]
        assert rows[0]["fold"] == fold_id
        assert rows[0]["task_val_auc"] == 0.5


def test_binary_evaluation_forwards_declared_trial_channel_masks():
    class MaskAwareModel:
        fit_accepts_subject_ids = True
        fit_accepts_trial_channel_mask = True
        predict_accepts_trial_channel_mask = True
        fit_masks = []
        predict_masks = []

        def fit(self, X, y, subject_ids=None, trial_channel_mask=None):
            self.fit_masks.append(np.asarray(trial_channel_mask).copy())
            self.calibration_logits_ = np.array([-2.0, 2.0])
            self.calibration_labels_ = np.array([0, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X, trial_channel_mask=None):
            self.predict_masks.append(np.asarray(trial_channel_mask).copy())
            return np.asarray(X)[:, 0, 0]

    X = np.zeros((6, 2, 1), dtype=np.float32)
    X[:, 0, 0] = [-2.0, 2.0, -1.0, 1.0, -3.0, 3.0]
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    subjects = np.array([0, 0, 1, 1, 2, 2])
    trial_mask = np.ones((6, 2), dtype=bool)
    trial_mask[[0, 2, 4], 1] = False
    model = MaskAwareModel()

    summary = evaluate_binary(
        model,
        X,
        y,
        subjects,
        [(subjects != 2, subjects == 2)],
        trial_channel_mask=trial_mask,
        fold_protocol="partial_loso",
    )

    assert summary.per_fold[0].auc > 0.9
    assert np.array_equal(MaskAwareModel.fit_masks[-1], trial_mask[:4])
    assert np.array_equal(MaskAwareModel.predict_masks[-1], trial_mask[4:])
    assert not hasattr(model, "fit_mask")
