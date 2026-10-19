"""Fold reductions and actual result I/O with fixed outputs and zero fitting."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import baselines.evaluate as evaluation
from baselines.calibration import LogitCalibration
from data.qc_features import EpochQCFeatures
from experiments import run_eeg_loso as runner
from tests.support.datasets import epoch_dataset


def fixed_summary(monkeypatch, *, candidate=False, values=(1.0, float("nan"))):
    groups = np.repeat(["a", "b"], 2)
    folds = [(groups != group, groups == group) for group in ("a", "b")]
    results = [evaluation.BinaryFoldResult(value, value, 2, 0.0, "fixed", 0.0) for value in values]
    monkeypatch.setattr(evaluation, "_run_fold_tasks", lambda *args, **kwargs: (
        [(i, result, np.array([2.0, 0.0]), folds[i][1], None) for i, result in enumerate(results)],
        "fixed_outputs_zero_fit", 1, "none", 1, 0, 0,
    ))
    X, y = np.zeros((4, 1, 1)), np.tile([0, 1], 2)
    if candidate:
        return evaluation.evaluate_candidate_selection(object(), X, y, np.tile([0, 1], 2), groups,
            {"a":0,"b":0}, folds, [0, 1], fold_id_offset=4)
    return evaluation.evaluate_binary(object(), X, y, groups, folds, fold_id_offset=4)


@pytest.mark.parametrize("candidate", (False, True))
def test_available_subset_is_explicit_diagnostic_with_full_requested_denominator(monkeypatch, candidate):
    result = fixed_summary(monkeypatch, candidate=candidate)
    assert result.auc_mean == result.balanced_acc_mean == 1.0
    scope = result.binary_metric_scope
    assert scope["status"] == "incomplete"
    assert scope["requested_fold_ids"] == [4, 5]
    assert scope["requested_fold_count"] == scope["completed_fold_count"] == 2
    assert scope["failed_fold_count"] == 0 and scope["execution_failure_policy"] == "raise_without_summary"
    for name in ("auc", "balanced_acc"):
        assert scope[name] == {"status":"incomplete", "role":"available_subset_diagnostic",
            "available_fold_ids":[4], "available_fold_count":1, "unavailable_fold_ids":[5], "unavailable_fold_count":1}
    if candidate:
        assert result.hit_rate_mean == 1.0 and len(result.subject_records) == 2


@pytest.mark.parametrize("candidate", (False, True))
def test_all_unavailable_metrics_have_no_estimate_or_empty_mean_warning(monkeypatch, candidate, recwarn):
    result = fixed_summary(monkeypatch, candidate=candidate, values=(float("nan"), float("nan")))
    assert result.balanced_acc_mean is result.balanced_acc_std is result.auc_mean is None
    assert result.binary_metric_scope["status"] == "unavailable"
    assert result.binary_metric_scope["auc"]["unavailable_fold_ids"] == [4, 5]
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    json.dumps({"bacc":result.balanced_acc_mean,"auc":result.auc_mean,"scope":result.binary_metric_scope}, allow_nan=False)


def test_missing_result_cannot_reduce_the_requested_fold_denominator(monkeypatch):
    with pytest.raises(ValueError, match="every requested fold"):
        fixed_summary(monkeypatch, values=(1.0,))


@pytest.mark.parametrize("failed_execution", (False, True))
def test_cli_records_unavailable_binary_metrics_and_preserves_candidate_task(tmp_path, monkeypatch, capsys, failed_execution):
    dataset = epoch_dataset()
    dataset.qc_features = EpochQCFeatures(np.ones((6, 3)), np.ones((6, 3)), np.ones(6), np.ones((6, 3),dtype=bool))
    monkeypatch.setattr(runner,"load_epoch_dataset",lambda *args,**kwargs:dataset)
    monkeypatch.setattr(runner,"loaded_epoch_cache_attestation",lambda data:{"sha256":"synthetic"})
    monkeypatch.setattr(runner,"assert_p300_input_contract",lambda *args:None)
    monkeypatch.setattr(runner,"assert_p300_source_provenance",lambda *args:None)
    monkeypatch.setattr(runner,"resolve_fold_local_artifact_models",lambda *args,**kwargs:({}, {"fit_seconds":0.0}))
    monkeypatch.setattr(runner,"build_binary_model",lambda *args,**kwargs:SimpleNamespace(configure_epoch_progress=lambda output:None))
    monkeypatch.setattr(runner,"describe_binary_model",lambda *args:{"name":"fixed_outputs_zero_fit"})
    undefined = evaluation.BinaryFoldResult(float("nan"),float("nan"),2,0.0,"fixed",0.0)

    def fixed_outputs(model, X, y, fit_groups, folds, **kwargs):
        if failed_execution:
            raise RuntimeError("controlled fold execution failure")
        result = [(i,undefined,np.array([0.0,2.0]),test,None) for i,(_,test) in enumerate(folds)]
        if kwargs.get("on_fold_result"):
            for row in result:
                kwargs["on_fold_result"](row)
        return result,"fixed_outputs_zero_fit",1,"none",1,0,0

    monkeypatch.setattr(evaluation,"_run_fold_tasks",fixed_outputs)
    monkeypatch.setattr(sys,"argv",["run_eeg_loso","--dataset-cache","synthetic.npz","--models","window_lr",
        "--run-dir",str(tmp_path),"--run-name","availability","--device","cpu","--subjects","2","--max-folds","1"])
    output = tmp_path / "availability/window_lr"
    if failed_execution:
        with pytest.raises(RuntimeError, match="controlled fold execution failure"):
            runner.main()
        assert not (output / "record.json").exists()
        progress = [json.loads(line) for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
        assert progress[-1]["type"] == "failed" and not any(line.get("type") == "done" for line in progress)
        return
    runner.main()
    record = json.loads((output / "record.json").read_text(encoding="utf-8"))
    assert record["auc_mean"] is record["balanced_acc_mean"] is None
    assert record["binary_metric_scope"]["status"] == "unavailable"
    assert record["binary_metric_scope"]["requested_fold_count"] == 1
    assert record["primary_decision_hit_rate"] == 1.0
    assert record["per_fold"][0]["auc"] is record["per_fold"][0]["balanced_acc"] is None
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    fold, = [line for line in progress if line.get("type") == "fold"]
    assert fold["fold_auc"] is fold["fold_bacc"] is None
    assert progress[-1]["type"] == "done" and progress[-1]["binary_metric_status"] == "unavailable"
    assert "binary_metrics=unavailable" in capsys.readouterr().out


class FrozenOutputs:
    def __init__(self):
        self.fold_ids = []
        self.model_ = torch.nn.Identity()

    def configure_evaluation_fold(self, fold_id):
        self.fold_ids.append(fold_id)

    def predict_logit(self, X):
        assert len(self.fold_ids) == 1
        return np.array([-1.0, 1.0])[:len(X)]


def test_actual_single_class_fold_has_no_binary_estimate_without_training(monkeypatch):
    monkeypatch.setattr(evaluation,"_fit",lambda *args,**kwargs:None)
    monkeypatch.setattr(evaluation,"calibration_data_from_model",lambda model:(np.array([-1.,1.]),np.array([0,1]),"fixed"))
    monkeypatch.setattr(evaluation,"fit_logit_calibration",lambda *args,**kwargs:LogitCalibration(0.,1.,1.,0.,"fixed",2))
    X, y, groups = np.zeros((4,1,1)), np.array([0,1,0,0]), np.array(["train","train","test","test"])
    train, test = np.array([True,True,False,False]), np.array([False,False,True,True])
    result = evaluation.evaluate_binary(FrozenOutputs(),X,y,groups,[(train,test)],cpu_threads=1)
    assert np.isnan(result.per_fold[0].auc) and result.per_fold[0].n_test_trials == 2
    assert result.auc_mean is result.balanced_acc_mean is None
    assert result.binary_metric_scope["auc"]["unavailable_fold_ids"] == [0]
    assert result.binary_metric_scope["requested_fold_count"] == 1


def test_removed_export_env_has_no_io_and_fold_prototypes_remain_isolated(tmp_path, monkeypatch):
    export = tmp_path / "unexpected-export"
    monkeypatch.setenv("P300_SAVE_FOLD_MODELS_DIR",str(export))
    monkeypatch.setattr(evaluation,"_fit",lambda *args,**kwargs:None)
    monkeypatch.setattr(evaluation,"calibration_data_from_model",lambda model:(np.array([-1.,1.]),np.array([0,1]),"fixed"))
    calibration = LogitCalibration(0.0,1.0,1.0,0.0,"fixed",2)
    monkeypatch.setattr(evaluation,"fit_logit_calibration",lambda *args,**kwargs:calibration)
    prototype = FrozenOutputs()
    X, y, groups = np.zeros((4,1,1)), np.array([0,1,0,1]), np.array(["train","train","test","test"])
    train, test = np.array([True,True,False,False]), np.array([False,False,True,True])
    for fold_id in (1,2):
        result, _, _, _ = evaluation._fold_result(prototype,X,y,groups,train,test,None,None,None,fold_id=fold_id)
        assert result.auc == 1.0
    assert prototype.fold_ids == [] and not export.exists()
