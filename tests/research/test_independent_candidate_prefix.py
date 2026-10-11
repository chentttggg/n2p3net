"""Independent C/test caller counterexamples, not C-inner development scores."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_SOURCE_REFERENCE
from experiments.run_candidate_ensemble_study import run_study
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, SubjectFailureCode, UnidentifiableFitError
from research.source_catalog import SourceSeedCatalog, paired_pipeline_seeds
from tests.support.checkpoints import write_checkpoint
from tests.support.independent_prefix import independent_prefix_case
from transfer.calibration_candidates import waveform_lda_recipe
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    fit_calibration_prefix,
    fit_calibration_prefix_programs,
)
from transfer.ensemble_views import SelectionEpochProfile, prepare_bnci_independent_prefixes
from transfer.recipes import TargetRecipe


@pytest.fixture(scope="module")
def independent_case(tmp_path_factory):
    return independent_prefix_case(tmp_path_factory.mktemp("independent-prefix"))


def test_one_unresolved_continuous_expert_keeps_trial_and_all_requested_decisions(independent_case, tmp_path, monkeypatch):
    from transfer.continuous_decision import ContinuousDecisionModel, KernelPosteriorResolutionError

    _, _, path, spec, experts, views = independent_case
    mixtures = (EnsembleProgram.equal_weights("trial", ("toeplitz",)),
                EnsembleProgram.equal_weights("strict", ("raw", "toeplitz")))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
        "evaluation_design": "independent_prefix", "seed_pairs": [[20260918, 20260918]],
        "records": [{"path": str(path), "sha256": views.ledger["source_sha256"]}],
        "budgets": [200], "experts": [expert.record() for expert in experts],
        "mixtures": [mixture.record() for mixture in mixtures], "baseline_mixture": "trial",
        "authorization_preprocessing": spec.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "source_checkpoints": {}, "source_seed_catalog": [],
        "calibration_strategies": ["fixed"]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def unresolved(*args, **kwargs):
        raise KernelPosteriorResolutionError({"unresolved_held_direction": True})

    monkeypatch.setattr(ContinuousDecisionModel, "score", unresolved)
    report = run_study(plan_path, tmp_path / "run", execution="inline")
    assert report["status"] == "completed"
    participant = report["participants"][0]
    trial = [cell for cell in participant["predictions"] if cell["mixture"] == "trial"]
    strict = [cell for cell in participant["predictions"] if cell["mixture"] == "strict"]
    assert len(trial) == len(strict) == len(participant["requested"])
    available = [cell for cell in trial if cell["evidence"]["log_probability"] is not None]
    assert available and all(cell["saved_replay_exact"] for cell in available)
    resolved_ids = {cell["evidence"]["decision_id"] for cell in available}
    for cell in strict:
        evidence = cell["evidence"]
        assert evidence["log_probability"] is None
        if evidence["decision_id"] in resolved_ids:
            assert evidence["failure_kind"] == "not_available"
            assert "unresolved_held_direction" in evidence["failure_reason"]
        assert {identity["expert_id"] for identity in evidence["expert_identities"]} == {"raw", "toeplitz"}
    assert set(report["metrics"]) == {"trial@fixed@K200", "strict@fixed@K200"}


def test_k200_uses_partial_selection_labels_and_future_changes_cannot_change_c(independent_case):
    root, fields, path, spec, experts, views = independent_case
    prefix = views.calibration[200]
    assert len(prefix.raw.target_mask) == 200 and prefix.raw.right_censored
    data = prefix.trial_datasets[experts[1].profile.digest]
    assert 120 < data.n_epochs < 200 and len(prefix.complete_groups) == 1
    receipt = prefix.label_use["trial"][experts[1].profile.digest]
    assert receipt["requested_labels"] == 200
    assert receipt["used_labels"] + len(receipt["unused_label_indices"]) == 200
    cutoff = prefix.target_contract["raw_scope"][1]
    assert prefix.raw.onsets[-1] < cutoff
    assert prefix.raw.samples.shape[0] == cutoff == receipt["read_sample_support"][1]
    with pytest.raises(ValueError):
        prefix.raw.samples[0, 0] = 1.
    with pytest.raises(ValueError):
        data.X[0, 0, 0] = 1.
    with pytest.raises(TypeError, match="frozen"):
        data.y = 1 - data.y
    mutated = data.metadata
    mutated.loc[0, "flash_sample_matlab_1based"] = -1
    assert data.metadata.loc[0, "flash_sample_matlab_1based"] > 0
    changed = dict(fields)
    changed["X"] = fields["X"].copy()
    changed["X"][cutoff:] += 100_000
    path = root / "future-changed" / "A01.mat"
    path.parent.mkdir()
    savemat(path, {"data": changed})
    after = prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budgets=(0, 200), profiles=(experts[1].profile,), authorization_preprocessing=spec).calibration[200]
    np.testing.assert_array_equal(after.raw.samples, prefix.raw.samples)
    np.testing.assert_array_equal(after.trial_datasets[experts[1].profile.digest].X, data.X)
    for view in views.decisions:
        if view.samples is not None:
            assert view.raw_scope[0] >= views.ledger["test_scope"]["start_sample"] > cutoff
            assert not hasattr(view, "target_candidate") and not hasattr(view, "labels")


def test_k0_fixed_roster_and_unavailable_packages_replay_without_renormalizing(independent_case, tmp_path):
    _, _, _, _, experts, views = independent_case
    program = EnsembleProgram.equal_weights("fixed", ("raw", "toeplitz"))
    fitted = fit_calibration_prefix(views.calibration[0], experts, program)
    before = fitted.score(views.decisions[-1])
    assert before.log_probability is None and before.failure_kind == "not_available"
    assert tuple(item.expert_id for item in before.expert_identities) == ("raw", "toeplitz")
    fitted.save(tmp_path / "K0")
    assert FittedEnsemble.load(tmp_path / "K0").score(views.decisions[-1]).record() == before.record()
    fitted = fit_calibration_prefix(views.calibration[200], experts, program)
    with pytest.raises(ValueError, match="overlaps calibration"):
        fitted.score(views.calibration[200].complete_groups[0].view)
    with pytest.raises(ExpectedSubjectError, match="fewer than three complete"):
        fit_calibration_prefix(views.calibration[200], experts, program, calibration_strategy="grouped_temperature")
    with pytest.raises(ValueError, match="authorized K/cutoff"):
        fit_calibration_prefix(replace(views.calibration[200], raw=replace(views.calibration[200].raw,
            onsets=np.arange(240), target_mask=np.zeros(240, dtype=bool))), experts, program)
    fitted.save(tmp_path / "K200")
    after = FittedEnsemble.load(tmp_path / "K200")
    assert after.score(views.decisions[-1]).record() == fitted.score(views.decisions[-1]).record()
    # Updating only the shell digest cannot reassign actual model fit authority.
    path = tmp_path / "K200" / "ensemble.json"
    package = json.loads(path.read_bytes())
    package["record"]["target_contract"]["raw_scope"][1] -= 1
    package["record_sha256"] = semantic_sha256(package["record"])
    path.write_text(json.dumps(package), encoding="utf-8")
    with pytest.raises(ValueError, match="fit authority"):
        FittedEnsemble.load(tmp_path / "K200")


def test_requested_budgets_share_actual_owner_fallback_without_duplicate_calibration(independent_case):
    _, _, path, spec, experts, _ = independent_case
    views = prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budgets=(0, 200, 4000, 5000), profiles=(experts[1].profile,), authorization_preprocessing=spec)
    assert views.ledger["requested_to_actual_K"] == {"0": 0, "200": 200, "4000": 3360, "5000": 3360}
    assert set(views.calibration) == {0, 200, 3360}
    assert len(views.calibration[3360].raw.target_mask) == 3360
    assert set(views.ledger["budget_adjustments"]) == {"4000", "5000"}
    assert views.ledger["test_scope"]["event_indices"][0] == 3360


def source_entries(root, data):
    entries = []
    for seed in (11, 12, 13):
        path = write_checkpoint(root / f"source{seed}.pt", data, source_snapshot_sha256="b" * 64)
        payload = torch.load(path, weights_only=True)
        payload["training_contract"]["seed"] = seed
        payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
        torch.save(payload, path)
        entries.append({"outer_target_study": "BNCI2014-008", "source_seed": seed,
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}})
    return entries


def test_catalog_rejects_seed_lies_target_study_and_shared_selection(independent_case, tmp_path):
    _, _, _, _, experts, views = independent_case
    entries = source_entries(tmp_path, views.calibration[200].trial_datasets[experts[1].profile.digest])
    catalog = SourceSeedCatalog(entries, base_directory=tmp_path, evidence_scope="software_correctness")
    assert len(catalog.record()) == 3
    with pytest.raises(ValueError, match="actual checkpoint seed"):
        SourceSeedCatalog([{**entries[0], "source_seed": 99}], base_directory=tmp_path, evidence_scope="software_correctness")
    with pytest.raises(ValueError, match="complete outer target"):
        SourceSeedCatalog([{**entries[0], "outer_target_study": "external-study"}], base_directory=tmp_path, evidence_scope="software_correctness")
    with pytest.raises(ValueError, match="software source fixtures"):
        SourceSeedCatalog(entries, base_directory=tmp_path, evidence_scope="public_calibration_development")
    with pytest.raises(ValueError, match="three distinct paired"):
        paired_pipeline_seeds([(11, 21), (12, 21), (13, 23)])
    record = {"source_seed": 11, "outer_target_study": "BNCI2014-008", "evidence_scope": "software_correctness",
        "selection": {"source_seed": 10, "status": "selected", "outcomes": [{"auc": .8}], "source_snapshot_sha256": "b" * 64},
        "refit": {"source_seed": 11, "status": "completed", "physical_optimizer_updates": 1,
                  "checkpoint_sha256": entries[0]["sha256"], "training_contract_digest": catalog.record()[0]["training_contract_digest"]}}
    path = tmp_path / "shared-development-selection.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    procedure = {"kind": "selection_refit", "record_path": str(path), "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="same-seed producer"):
        SourceSeedCatalog([{**entries[0], "procedure": procedure}], base_directory=tmp_path, evidence_scope="software_correctness")


def test_actual_caller_three_pairs_k0_nn_strict_four_and_partial_requested_denominator(independent_case, tmp_path):
    _, _, path, spec, experts, views = independent_case
    profile = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=("Fz", "P3", "Pz", "P4", "PO8", "Oz"),
        reference=BNCI2014_008_SOURCE_REFERENCE, reference_channel="Cz")
    source_views = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(0, 200), profiles=(profile,), authorization_preprocessing=spec)
    entries = source_entries(tmp_path, source_views.calibration[200].trial_datasets[profile.digest])
    source = TargetRecipe("NN", initialization="source", input_statistics="source", epochs=1,
        batch_size=128, bn_buffers="calibration", dropout="off")
    source_experts = (CalibrationExpertProgram("NN", "trial", source, profile),
        CalibrationExpertProgram("scratch", "trial", replace(source, name="scratch", initialization="scratch", input_statistics="target_prefix"), profile),
        CalibrationExpertProgram("waveform", "trial", waveform_lda_recipe(), profile))
    all_experts = (*experts, *source_experts)
    mixtures = (EnsembleProgram.equal_weights("classic", ("raw", "toeplitz")),
                EnsembleProgram.equal_weights("NN", ("NN",)),
                EnsembleProgram.equal_weights("strict_four", ("raw", "toeplitz", "NN", "waveform")))
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evaluation_design": "independent_prefix",
        "evidence_scope": "software_correctness", "records": [{"path": str(path), "sha256": views.ledger["source_sha256"]}],
        "budgets": [0, 200], "seed_pairs": [[11, 21], [12, 22], [13, 23]], "source_seed_catalog": entries,
        "source_checkpoints": {}, "experts": [expert.record() for expert in all_experts],
        "mixtures": [mixture.record() for mixture in mixtures], "baseline_mixture": "classic",
        "authorization_preprocessing": spec.record(), "observation_response_seconds": 308 / 256,
        "pre_roll_seconds": .2, "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
        "calibration_strategies": ["fixed", "grouped_temperature"]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report = run_study(plan_path, tmp_path / "run")
    participant = report["participants"][0]
    assert report["randomness_scope"] == "paired_source_target_pipeline"
    assert report["seed_pairs"] == [[11, 21], [12, 22], [13, 23]]
    assert len(participant["fits"]) == 3 * 2 * 2 * len(all_experts)
    packages = {fit["package"] for fit in participant["fits"]}
    assert len(packages) == 3 * 2
    for cell in participant["predictions"]:
        assert (cell["source_seed"], cell["target_seed"]) in ((11, 21), (12, 22), (13, 23))
        if cell["calibration_strategy"] == "grouped_temperature":
            assert not cell["saved_replay_exact"]
            assert cell["evidence"]["log_probability"] is None
        else:
            assert cell["saved_replay_exact"]
        if cell["K"] == 0 and cell["mixture"] == "strict_four":
            assert cell["evidence"]["log_probability"] is None
            assert len(cell["evidence"]["expert_identities"]) == 4
    K0 = [cell for cell in participant["predictions"] if cell["K"] == 0 and cell["mixture"] == "NN"
        and cell["calibration_strategy"] == "fixed"]
    assert any(cell["evidence"]["log_probability"] is not None for cell in K0)
    assert report["metrics"]["strict_four@fixed@K0"]["requested_participant_operational_hit_rate"] == 0
    assert len(participant["requested"]) == len(views.decisions)
    assert any(cell["evidence"]["failure_kind"] == "incomplete" for cell in participant["predictions"] if cell["K"] == 200)


def test_temperature_failure_keeps_fixed_output_and_binds_named_programs(independent_case, tmp_path, monkeypatch):
    import transfer.ensemble_fit as owner

    _, _, path, spec, experts, views = independent_case
    trial = experts[1]
    larger = prepare_bnci_independent_prefixes(path, source_sha256=views.ledger["source_sha256"],
        budgets=(0, 400), profiles=(trial.profile,), authorization_preprocessing=spec)
    program = EnsembleProgram.equal_weights("toeplitz", (trial.expert_id,))
    prefix = larger.calibration[400]
    fixed = fit_calibration_prefix(prefix, (trial,), program)
    monkeypatch.setattr(owner, "fit_calibration_prefix", lambda *args, **kwargs: fixed)

    for index, error in enumerate((ExpectedSubjectError(SubjectFailureCode.NUMERICAL_FAILURE,
        stage="temperature", detail="valid C-only inner fit failed"), UnidentifiableFitError("inner fit is singular"),
        np.linalg.LinAlgError("inner decomposition failed"))):
        def failed_temperature(*args, failure=error, **kwargs):
            raise failure

        monkeypatch.setattr(owner, "_grouped_temperature", failed_temperature)
        result = fit_calibration_prefix_programs(prefix, (trial,), program,
            strategies=("fixed", "grouped_temperature"))
        assert result["fixed"] is fixed
        assert result["grouped_temperature"].code == SubjectFailureCode.NUMERICAL_FAILURE
        package = tmp_path / f"programs-{index}"
        FittedEnsemble.save_programs(package, result, participant_key=fixed.participant_key)
        loaded = FittedEnsemble.load_programs(package)
        assert set(loaded) == {"fixed"}
        assert loaded["fixed"].score(larger.decisions[-1]).record() == fixed.score(larger.decisions[-1]).record()

    manifest = package / "temperature_programs.json"
    before = json.loads(manifest.read_bytes())
    numerical_sha = hashlib.sha256((package / "expert_000.pt").read_bytes()).hexdigest()
    for violation in ("fixed_scale", "wrong_name", "unknown_strategy"):
        payload = json.loads(json.dumps(before))
        if violation == "fixed_scale":
            payload["record"]["programs"]["fixed"]["temperature_calibration"]["temperatures"][trial.expert_id] = 2.
        elif violation == "wrong_name":
            payload["record"]["programs"]["grouped_temperature"] = payload["record"]["programs"].pop("fixed")
        else:
            payload["record"]["programs"]["fixed"]["temperature_calibration"]["strategy"] = "undeclared"
        payload["record_sha256"] = semantic_sha256(payload["record"])
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="temperature|strategy"):
            FittedEnsemble.load_programs(package)
        assert hashlib.sha256((package / "expert_000.pt").read_bytes()).hexdigest() == numerical_sha

    def contract_failure(*args, **kwargs):
        raise ValueError("invalid authority contract")

    monkeypatch.setattr(owner, "_grouped_temperature", contract_failure)
    with pytest.raises(ValueError, match="invalid authority"):
        fit_calibration_prefix_programs(prefix, (trial,), program, strategies=("fixed", "grouped_temperature"))
