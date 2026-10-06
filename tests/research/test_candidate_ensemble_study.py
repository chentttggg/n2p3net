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


def package_resume_fixture(tmp_path):
    import io
    import tarfile

    import experiments.run_candidate_ensemble_study as runner

    plan_path = study_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    raw = replace(CalibrationExpertProgram.from_record(plan["experts"][0]),
                  expert_id="ar0_burg", conditioning_samples=32)
    plan.update(experts=[raw.record()], mixtures=[EnsembleProgram.equal_weights("ar0_burg", ("ar0_burg",)).record()],
                baseline_mixture="ar0_burg", comparison_role="fixed_prefix_ar_order")
    source = runner._executing_source()

    def physical_freeze(name, root):
        freeze = tmp_path / name
        freeze.mkdir()
        paths = [*(runner.ROOT / "src").rglob("*.py"), *(runner.ROOT / "experiments").glob("*.py"),
                 runner.ROOT / "pyproject.toml", runner.ROOT / "uv.lock"]
        content = {path.relative_to(runner.ROOT).as_posix(): path.read_bytes() for path in paths}
        archive = freeze / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for relative, data in content.items():
                info = tarfile.TarInfo(relative)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
                if root != runner.ROOT:
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        payload = {"schema": "n2p3_source_freeze/1", "source_commit": "a" * 40,
                   "archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                   "byte_size": archive.stat().st_size, "member_count": len(content),
                   "files": {path: hashlib.sha256(data).hexdigest() for path, data in content.items()}}
        path = freeze / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    old_root = tmp_path / "old_checkout"
    old_manifest, old_freeze = physical_freeze("old_freeze", old_root)
    new_manifest, _ = physical_freeze("new_freeze", runner.ROOT)
    old_plan = {**plan, "source_snapshot_manifest": str(old_manifest), "ar_panel": None,
                "matched_controls": [], "source_checkpoint_sha256": {}}
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    (old_run / "plan.json").write_text(json.dumps(old_plan), encoding="utf-8")
    (old_run / "executed_source.json").write_text(json.dumps(source), encoding="utf-8")
    runner._subject(old_plan["records"][0], old_plan, old_run / "participant_000", source, {})
    (old_run / "progress.json").write_text(json.dumps({"status": "failed", "completed_subjects": 0}), encoding="utf-8")
    input_plan = old_root / "input_plan.json"
    input_plan.write_text(json.dumps(old_plan), encoding="utf-8")
    execution_path = tmp_path / "original_execution.json"
    execution_path.write_text(json.dumps({"executed_root": str(old_root),
        "argv": ["python", "--plan", str(input_plan), "--output", str(old_run)],
        "plan_sha256": hashlib.sha256(input_plan.read_bytes()).hexdigest(),
        "source": {"archive_sha256": old_freeze["archive_sha256"],
                   "manifest_sha256": hashlib.sha256(old_manifest.read_bytes()).hexdigest()}}), encoding="utf-8")
    normalized = {**old_plan, "source_snapshot_manifest": str(new_manifest), "resume": {
        "schema": "n2p3_m1_package_resume/1", "original_run": str(old_run),
        "original_plan_sha256": hashlib.sha256((old_run / "plan.json").read_bytes()).hexdigest(),
        "original_executed_source_sha256": hashlib.sha256((old_run / "executed_source.json").read_bytes()).hexdigest(),
        "original_execution": str(execution_path), "original_execution_sha256": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
        "original_source_manifest": str(old_manifest), "original_source_manifest_sha256": hashlib.sha256(old_manifest.read_bytes()).hexdigest(),
        "original_source_archive_sha256": old_freeze["archive_sha256"]}}
    return runner, normalized, old_run, source


def test_fixed_prefix_resume_reuses_signed_packages_without_refitting(tmp_path, monkeypatch):
    runner, plan, old_run, source = package_resume_fixture(tmp_path)
    plan["resume_binding"] = runner._prepare_package_resume(plan, source)
    original = {path.relative_to(old_run).as_posix(): path.read_bytes() for path in old_run.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("reused package must not fit")

    monkeypatch.setattr(runner, "fit_calibration_ensemble", forbidden)
    result = runner._subject(plan["records"][0], plan, tmp_path / "new_participant", source, {})
    assert len(result["predictions"]) == 2
    for row in result["predictions"]:
        assert row["fit_origin"]["kind"] == "reused_package"
        assert row["fit_origin"]["physical_fit_attempts"] == row["fit_origin"]["physical_successful_fits"] == 0
        assert row["fit_origin"]["reused_package_count"] == 1
        assert row["fit_origin"]["producer_source_digest"] == runner.semantic_sha256(source)
        assert row["score_source_digest"] == runner.semantic_sha256(source)
        assert row["saved_replay_exact"]
    assert original == {path.relative_to(old_run).as_posix(): path.read_bytes() for path in old_run.rglob("*") if path.is_file()}


def test_fixed_prefix_resume_keeps_packages_when_subject_result_is_absent(tmp_path, monkeypatch):
    runner, plan, old_run, source = package_resume_fixture(tmp_path)
    (old_run / "participant_000/result.json").unlink()
    plan["resume_binding"] = runner._prepare_package_resume(plan, source)
    monkeypatch.setattr(runner, "fit_calibration_ensemble", lambda *args, **kwargs: pytest.fail("partial record package refit"))
    result = runner._subject(plan["records"][0], plan, tmp_path / "new_participant", source, {})
    assert all(row["fit_origin"]["reused_package_count"] == 1 for row in result["predictions"])


@pytest.mark.parametrize("fail", [False, True])
def test_fixed_prefix_resume_missing_manifest_executes_one_new_attempt(tmp_path, monkeypatch, fail):
    from research.execution import ExpectedSubjectError, SubjectFailureCode

    runner, plan, old_run, source = package_resume_fixture(tmp_path)
    (old_run / "participant_000/seed23_fold1_expert0/ensemble.json").unlink()
    plan["resume_binding"] = runner._prepare_package_resume(plan, source)
    real_fit = runner.fit_calibration_ensemble
    attempts = []

    def one_new_fit(*args, **kwargs):
        attempts.append(True)
        if fail:
            raise ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE, stage="fit", detail="software resume failure")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_calibration_ensemble", one_new_fit)
    result = runner._subject(plan["records"][0], plan, tmp_path / "new_participant", source, {})
    assert len(attempts) == 1 and len(result["predictions"]) == len(result["requested"]) == 2
    first, second = result["predictions"]
    assert first["fit_origin"]["kind"] == "reused_package"
    assert second["fit_origin"]["physical_fit_attempts"] == 1
    assert second["fit_origin"]["physical_successful_fits"] == int(not fail)
    assert second["fit_origin"]["reused_package_count"] == 0
    assert second["fit_origin"]["kind"] == ("new_fit_failure" if fail else "new_fit")
    assert second["saved_replay_exact"] is not fail
    if fail:
        assert second["evidence"]["failure_kind"] == "fit_failure"
        assert second["continuous_score_support"]["ar0_burg"]["n_scored_samples"] == 0


@pytest.mark.parametrize("change", ["plan", "src_bytes", "execution", "manifest", "scope", "model_bytes", "absolute_model_path"])
def test_fixed_prefix_resume_rejects_changed_authority_without_fallback(tmp_path, change):
    runner, plan, old_run, source = package_resume_fixture(tmp_path)
    if change == "plan":
        plan["authorized_labels"] = 801
    elif change == "src_bytes":
        source = {**source, "src/transfer/continuous_gradient.py": "f" * 64}
    elif change == "execution":
        plan["resume"]["original_execution_sha256"] = "f" * 64
    elif change == "manifest":
        plan["resume"]["original_source_manifest_sha256"] = "f" * 64
    else:
        package = old_run / "participant_000/seed23_fold0_expert0"
        if change == "model_bytes":
            path = package / "expert_000.npz"
            path.write_bytes(path.read_bytes() + b"invalid")
        else:
            path = package / "ensemble.json"
            payload = json.loads(path.read_bytes())
            if change == "absolute_model_path":
                model_path = package / payload["record"]["experts"][0]["model"]
                model_before = (model_path.read_bytes(), model_path.stat().st_mtime_ns)
                payload["record"]["experts"][0]["model"] = str(model_path.resolve())
            else:
                payload["record"]["training_receipts"][0]["participant_key"] = "other"
            payload["record_sha256"] = runner.semantic_sha256(payload["record"])
            path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        plan["resume_binding"] = runner._prepare_package_resume(plan, source)
        runner._subject(plan["records"][0], plan, tmp_path / "new_participant", source, {})
    if change == "absolute_model_path":
        assert (model_path.read_bytes(), model_path.stat().st_mtime_ns) == model_before


def test_calibration_g_prior_cannot_enter_same_ridge_ar_order_attribution():
    from transfer.kernel_prior import KernelPriorSpec

    experts = tuple(replace(spec, ridge=None, kernel_prior=KernelPriorSpec.calibration_g(9.))
                    for spec in fixed_prefix_panel_programs())
    with pytest.raises(ValueError, match="isotropic ridge"):
        _fixed_prefix_ar_panel(experts, "fixed_prefix_ar_order")


def test_g_prior_rank_failure_keeps_structured_evidence_and_requested_score_support(tmp_path, monkeypatch):
    import experiments.run_candidate_ensemble_study as runner
    from transfer.kernel_prior import KernelPriorRankError, KernelPriorSpec, compile_kernel_prior

    prior = KernelPriorSpec.calibration_g(9.)
    with pytest.raises(KernelPriorRankError) as captured:
        compile_kernel_prior(prior, [np.ones((43, 1))], record_ids=("confounded_C",),
                             fit_scope="target_calibration", binding={"scope": "software_fixture"})
    rank_error = captured.value

    def rank_failed(*args, **kwargs):
        raise rank_error

    monkeypatch.setattr(runner, "fit_calibration_ensemble", rank_failed)
    path = study_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    expert = replace(CalibrationExpertProgram.from_record(plan["experts"][0]),
                     conditioning_samples=32, ridge=None, kernel_prior=prior)
    plan.update(experts=[expert.record()], mixtures=[EnsembleProgram.equal_weights("g9", (expert.expert_id,)).record()],
                baseline_mixture="g9")
    path.write_text(json.dumps(plan), encoding="utf-8")
    report = run_study(path, tmp_path / "rank_failed")
    assert len(report["participants"][0]["predictions"]) == 2
    for cell in report["participants"][0]["predictions"]:
        assert cell["evidence"]["failure_kind"] == "fit_failure"
        assert cell["fit_origin"]["failure_diagnostics"]["rank_ledger"]["rank"] == 0
        assert cell["fit_origin"]["failure_diagnostics"]["prior_spec"]["g"] == 9.
        assert cell["continuous_score_support"][expert.expert_id]["n_scored_samples"] == 0
        assert cell["requested_authorized_labels"] == 800
