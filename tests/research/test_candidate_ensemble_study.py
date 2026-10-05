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
from experiments.run_candidate_ensemble_study import (
    _fixed_prefix_ar_panel,
    _matched_source_controls,
    run_study,
)
from tests.support.bnci import bnci_mat_fields
from tests.support.checkpoints import write_checkpoint
from transfer.calibration_candidates import waveform_lda_recipe
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FrozenSourceCheckpoint
from transfer.ensemble_views import (
    SelectionEpochProfile,
    build_calibration_epoch_dataset,
    prepare_bnci_calibration_groups,
)
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


def test_mat_conditional_program_executes_actual_fit_and_preserves_scored_support(tmp_path):
    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    raw = CalibrationExpertProgram.from_record(plan["experts"][0])
    raw = replace(raw, conditioning_samples=32, noise_estimator="conditional_profile")
    plan["experts"] = [raw.record()]
    plan["mixtures"] = [EnsembleProgram.equal_weights("conditional", (raw.expert_id,)).record()]
    plan["baseline_mixture"] = "conditional"
    path.write_text(json.dumps(plan), encoding="utf-8")
    result = run_study(path, tmp_path / "conditional")
    rows = result["participants"][0]["predictions"]
    assert result["status"] == "completed" and len(rows) == 2
    for row in rows:
        support = row["continuous_score_support"]["raw"]
        assert support["conditioning_sample_support"] == [0, 32]
        assert support["score_sample_support"][0] == 32
        assert support["n_scored_samples"] == support["read_sample_support"][1] - 32
        assert row["saved_replay_exact"]
        assert row["evidence"]["log_probability"] is not None
        assert row["evidence"]["expert_identities"][0]["score_kind"].startswith("continuous_conditional_tail")


def fixed_prefix_panel_programs():
    return tuple(CalibrationExpertProgram(f"ar{order}_{strategy}", "continuous_ar",
        noise_order=order, conditioning_samples=32, noise_estimator=strategy)
        for strategy in ("burg", "conditional_profile") for order in (0, 1, 4, 8, 16, 32))


@pytest.mark.parametrize("change", ["missing_order", "prefix", "basis", "ridge", "estimator"])
def test_ar_order_comparison_panel_rejects_unmatched_or_incomplete_matrix(change):
    experts = list(fixed_prefix_panel_programs())
    if change == "missing_order":
        experts.pop()
    elif change == "prefix":
        experts[0] = replace(experts[0], conditioning_samples=33)
    elif change == "basis":
        experts[0] = replace(experts[0], n_basis=9)
    elif change == "ridge":
        experts[0] = replace(experts[0], ridge=2.)
    else:
        experts[-1] = replace(experts[-1], noise_estimator="burg")
    with pytest.raises(ValueError, match="all six orders"):
        _fixed_prefix_ar_panel(tuple(experts), "fixed_prefix_ar_order")


def test_ar_order_comparison_panel_declares_burg_initialization_and_conditional_program():
    panel = _fixed_prefix_ar_panel(fixed_prefix_panel_programs(), "fixed_prefix_ar_order")
    assert panel["orders"] == [0, 1, 4, 8, 16, 32]
    assert panel["noise_estimators"] == ["burg", "conditional_profile"]
    assert panel["kernel_information"] == "all_authorized_prefix_and_tail"
    assert panel["score_estimand"] == "conditional_tail_with_fixed_candidate_prior"


def test_unconverged_conditional_fit_keeps_requested_failures_and_zero_executed_tail(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import transfer.continuous_decision as continuous

    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    raw = replace(CalibrationExpertProgram.from_record(plan["experts"][0]),
                  noise_order=1, conditioning_samples=32, noise_estimator="conditional_profile")
    plan["experts"] = [raw.record()]
    plan["mixtures"] = [EnsembleProgram.equal_weights("conditional", (raw.expert_id,)).record()]
    plan["baseline_mixture"] = "conditional"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(continuous, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=False, fun=0., message="iteration limit"))
    result = run_study(path, tmp_path / "failed_fits")
    metric = result["metrics"]["conditional"]
    assert result["status"] == "completed" and metric["requested_participant_count"] == 1
    assert metric["requested_participant_operational_hit_rate"] == 0
    for row in result["participants"][0]["predictions"]:
        assert row["evidence"]["failure_kind"] == "fit_failure"
        support = row["continuous_score_support"]["raw"]
        assert support["planned_score_sample_support"][0] == 32
        assert support["n_scored_samples"] == 0 and not support["score_executed"]
        assert support["score_sample_support"] is None


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
    child = json.loads((tmp_path / "result/participant_000/progress.json").read_bytes())
    assert child["status"] == "failed" and "untrusted source" in child["error"]


def test_valid_flat_record_keeps_each_requested_expected_fit_failure(tmp_path):
    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    fields = bnci_mat_fields()
    savemat(tmp_path / "A01.mat", {"data": fields})
    plan["records"][0]["sha256"] = hashlib.sha256((tmp_path / "A01.mat").read_bytes()).hexdigest()
    plan["experts"] = plan["experts"][:2]
    plan["mixtures"] = [EnsembleProgram.equal_weights("raw-only", ("raw",)).record(),
                        EnsembleProgram.equal_weights("toeplitz-only", ("toeplitz",)).record()]
    plan["baseline_mixture"] = "raw-only"
    path.write_text(json.dumps(plan), encoding="utf-8")
    report = run_study(path, tmp_path / "result")
    assert report["status"] == "completed"
    for metric in report["metrics"].values():
        assert metric["requested_participant_count"] == 1
        assert metric["probability_coverage"] == 0
        assert metric["requested_participant_operational_hit_rate"] == 0


def test_matched_source_controls_bind_frozen_architecture_before_any_fit(tmp_path):
    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    groups = prepare_bnci_calibration_groups(plan["records"][0]["path"],
        source_sha256=plan["records"][0]["sha256"], response_seconds=308 / 256, group_count=2).groups
    scratch = CalibrationExpertProgram.from_record(plan["experts"][-1])
    source = replace(scratch, expert_id="source", recipe=replace(scratch.recipe, name="source", initialization="source"))
    data = build_calibration_epoch_dataset(groups[:1], source.profile)
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="c" * 64)
    snapshot = FrozenSourceCheckpoint.from_path(checkpoint)
    with pytest.raises(ValueError, match="matched scratch and waveform"):
        _matched_source_controls((source,), {"source": snapshot})
    waveform = CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(), source.profile)
    frozen, bindings = _matched_source_controls((source, scratch, waveform), {"source": snapshot})
    assert frozen["scratch"] is snapshot
    assert bindings[0]["source"] == "source" and bindings[0]["scratch"] == "scratch"
    checkpoint.write_bytes(b"replaced after freeze")
    assert snapshot.digest == hashlib.sha256(snapshot.content).hexdigest()
    assert snapshot.content != checkpoint.read_bytes()
