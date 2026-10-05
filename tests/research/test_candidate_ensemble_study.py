"""The real MAT-to-ensemble execution path and frozen requested evaluation."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from experiments.run_candidate_ensemble_study import run_study
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.ensemble_views import SelectionEpochProfile
from transfer.recipes import TargetRecipe, erp_target_recipe


def study_plan(tmp_path):
    path = tmp_path / "A01.mat"
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(39).normal(0, 4, fields["X"].shape).astype(np.float32)
    savemat(path, {"data": fields})
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    kwargs = dict(source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS,
                  reference=BNCI2014_008_SOURCE_REFERENCE)
    none = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"), **kwargs)
    baseline = SelectionEpochProfile.from_preprocessing(spec, **kwargs)
    experts = [CalibrationExpertProgram("raw", "continuous_ar", noise_order=0, n_basis=2),
               CalibrationExpertProgram("toeplitz", "trial", erp_target_recipe("toeplitz_lda", bin_width=16), none),
               CalibrationExpertProgram("scratch", "trial", TargetRecipe("scratch", initialization="scratch",
                    epochs=1, batch_size=128, bn_buffers="calibration", dropout="off"), baseline)]
    mixtures = [EnsembleProgram.equal_weights("classic", ("raw", "toeplitz")),
                EnsembleProgram.equal_weights("with-scratch", ("raw", "toeplitz", "scratch"))]
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
            "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "authorized_labels": 800, "group_count": 2, "observation_response_seconds": 308 / 256,
            "pre_roll_seconds": .2, "experts": [row.record() for row in experts],
            "mixtures": [row.record() for row in mixtures], "baseline_mixture": "classic", "seeds": [23],
            "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
            "calibration_strategy": "fixed", "source_checkpoints": {}}
    output = tmp_path / "plan.json"
    output.write_text(json.dumps(plan), encoding="utf-8")
    return output


def test_mat_real_experts_saved_replay_and_requested_metrics_execute(tmp_path):
    report = run_study(study_plan(tmp_path), tmp_path / "result")
    assert report["status"] == "completed"
    assert report["randomness_scope"] == "target_only_fixed_source"
    assert len(report["participants"][0]["predictions"]) == 4
    assert all(row["saved_replay_exact"] for row in report["participants"][0]["predictions"])
    assert set(report["metrics"]) == {"classic", "with-scratch"}
    assert report["contrasts"]["with-scratch"]["paired_probability_count"] == 2
    assert "experiments/run_candidate_ensemble_study.py" in json.loads((tmp_path / "result/executed_source.json").read_bytes())
    assert json.loads((tmp_path / "result/progress.json").read_bytes())["status"] == "completed"


def test_expected_failure_keeps_requested_cells_and_only_affects_its_mixture(tmp_path, monkeypatch):
    import experiments.run_candidate_ensemble_study as runner
    from research.execution import ExpectedSubjectError, SubjectFailureCode

    actual = runner.fit_calibration_ensemble

    def fail_scratch(calibration, specs, program, **kwargs):
        if specs[0].expert_id == "scratch":
            raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="fit", detail="fixture")
        return actual(calibration, specs, program, **kwargs)

    monkeypatch.setattr(runner, "fit_calibration_ensemble", fail_scratch)
    report = run_study(study_plan(tmp_path), tmp_path / "result")
    rows = report["participants"][0]["predictions"]
    assert all(row["evidence"]["log_probability"] is not None for row in rows if row["mixture"] == "classic")
    assert all(row["evidence"]["failure_kind"] == "fit_failure" for row in rows if row["mixture"] == "with-scratch")
    assert report["metrics"]["with-scratch"]["requested_participant_operational_hit_rate"] == 0


def test_contract_error_is_recorded_failed_and_propagates(tmp_path, monkeypatch):
    import experiments.run_candidate_ensemble_study as runner

    def broken(*args, **kwargs):
        raise ValueError("untrusted source/input relationship")

    monkeypatch.setattr(runner, "fit_calibration_ensemble", broken)
    with pytest.raises(ValueError, match="untrusted source"):
        run_study(study_plan(tmp_path), tmp_path / "result")
    state = json.loads((tmp_path / "result/progress.json").read_bytes())
    assert state["status"] == "failed" and "untrusted source" in state["error"]
