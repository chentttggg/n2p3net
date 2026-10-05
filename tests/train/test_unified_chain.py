"""Executed raw receipts, source architecture identity and scoped unified targets."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from data.epochs import concatenate_epoch_datasets, load_epoch_dataset, save_epoch_dataset
from data.source_records import (
    SOURCE_EOG_PREPROCESSING,
    file_sha256,
    load_target_prefixes,
    prepare_source_record,
    prepare_source_record_prefixes,
)
from models.n2p3net import N2P3ArchitectureConfig
from research.source_matrix import (
    unified_source_conditions,
    validate_unified_source_factors,
    validate_unified_source_input,
)
from tests.support.signal import make_raw
from tests.support.snapshots import write_source_snapshot


def test_unified_source_factorial_is_complete_and_locations_follow_actual_residual():
    conditions = unified_source_conditions()
    assert len(conditions) == 2
    assert {tuple(row["source_factors"].values()) for row in conditions} == {(1, 0, 0), (1, 1, 1)}
    for row in conditions:
        factors, architecture, objective = row["source_factors"], row["architecture_configuration"], row["source_objective"]
        assert architecture["observed_channel_norm"]
        assert bool(architecture["feature_residual"]) == bool(factors["R"])
        if objective is None:
            assert factors["S"] == factors["M"] == 0
        else:
            assert objective["supcon_weight"] == .05 * factors["S"]
            assert objective["mmd_weight"] == .05 * factors["M"]
            assert objective["feature_location"] == ("post_residual" if factors["R"] else "post_mst")
    with pytest.raises(ValueError, match="positive"):
        unified_source_conditions(supcon_weight=0)


@pytest.fixture
def actual_raw_sources(tmp_path, monkeypatch):
    from data.domain import namespace_epoch_dataset

    raw = make_raw(sfreq=256, n_seconds=30, ch_names=("Fz", "Cz", "P3", "Pz", "P4", "PO8", "Oz"))
    labels = np.tile([0, 1], 12)
    events = np.column_stack(((np.arange(24) + 2) * 256, np.zeros(24, int), labels + 1))
    monkeypatch.setattr("data.source_records._read_record", lambda _item, _path: (raw.copy(), events.copy(), labels.copy()))
    source = tmp_path / "raw.bin"
    source.write_bytes(b"software correctness raw reader fixture")

    def item(study, subject):
        return dict(study=study, subject=subject, path=source.name, sha256=file_sha256(source),
                    sample_rate_hz=256, reference="nose", session="session", run="run")

    views = []
    for study in ("BI2013a", "BI2014a", "BI2015a"):
        for number in range(6):
            prepared = prepare_source_record(item(study, str(number + 2)), root=tmp_path, output=tmp_path / study)
            views.append(namespace_epoch_dataset(load_epoch_dataset(prepared["cache"], validation="attested"), study))
    dataset = concatenate_epoch_datasets(views, name="executed_source_views", provenance={
        "source_reference": "measured_channel:CZ",
        "source_domain_axis": {**views[0].provenance["source_domain_axis"],
                               "domains": ["BI2013a", "BI2014a", "BI2015a"]}})
    return dataset, item("BNCI2014-008", "4")


def test_unified_source_dry_run_uses_receipts_and_global_stopping(tmp_path, actual_raw_sources):
    from experiments.run_source_training_chain import execute_source_chain

    dataset, _ = actual_raw_sources
    validate_unified_source_input(dataset, SOURCE_EOG_PREPROCESSING.record())
    with pytest.raises(ValueError, match="differs"):
        validate_unified_source_input(dataset, replace(SOURCE_EOG_PREPROCESSING, h_freq=16).record())
    missing = replace(dataset, provenance={key: value for key, value in dataset.provenance.items()
                                           if key != "preprocessing_execution"})
    with pytest.raises(ValueError, match="execution evidence"):
        validate_unified_source_input(missing, SOURCE_EOG_PREPROCESSING.record())
    cache = save_epoch_dataset(tmp_path / "sources.npz", dataset)
    snapshot, _, _ = write_source_snapshot(tmp_path)
    plan = {"schema": "n2p3_source_training_chain/1", "source_snapshot_manifest": str(snapshot),
            "source_preprocessing": SOURCE_EOG_PREPROCESSING.record(), "source_seeds": [11, 12, 13],
            "target_seeds": [11, 12, 13], "outer_folds": [{"name": "heldout", "source_cache": str(cache),
                "excluded_source_datasets": ["BNCI2014-008"]}]}
    result = execute_source_chain(plan, output=tmp_path / "dry", dry_run=True, device="cpu", stages=("unified_source",))
    assert result["task_count"] == 6 and result["optimizer_updates"] == 6 * 12
    assert not result["fitting_performed"]
    assert all(task["config"]["fixed_updates"] == 12 for task in result["tasks"])
    assert all(task["config"]["global_early_stopping"] is None for task in result["tasks"])
    assert all(task["config"]["source_budget"] is None for task in result["tasks"])


def test_unified_source_checkpoint_and_independent_target_prefixes_execute(tmp_path, monkeypatch, actual_raw_sources):
    from experiments.run_domain_transfer_matrix import target_job_groups
    from experiments.run_source_training_chain import _evaluate_outer_group, _prepare_outer_plans
    from research.evaluation import verified_source_snapshot
    from research.source_selection import fit_source_trajectory
    from transfer.checkpoint import load_n2p3_trunk_checkpoint
    from transfer.target_experiments import build_unified_scoped_tasks, user_fixed_target_batch_rule

    dataset, target = actual_raw_sources
    snapshot, snapshot_sha, _ = write_source_snapshot(tmp_path)
    condition = unified_source_conditions()[-1]
    from train.early_stopping import GlobalEarlyStoppingPolicy, UnifiedTrainingBudget

    policy = GlobalEarlyStoppingPolicy(patience=1, min_delta=100., source_max_epochs=3, target_max_epochs=3)
    config = DeepConfig(global_early_stopping=policy, batch_size=64, compile_mode=None,
                        source_objective=condition["source_objective"])
    _, checkpoints = fit_source_trajectory(dataset, np.arange(dataset.n_epochs), config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256=snapshot_sha,
        excluded_source_datasets=[target["study"]], device=torch.device("cpu"),
        architecture=N2P3ArchitectureConfig(**condition["architecture_configuration"]), evidence_scope="software_correctness")
    payload = next(iter(checkpoints.values()))
    assert payload["source_epoch_selection"]["validation_checks"] == 2
    validate_unified_source_factors(payload, condition["source_factors"])
    with pytest.raises(ValueError, match="differ"):
        validate_unified_source_factors(payload, {"R": 0, "S": 1, "M": 1})
    assert payload["input_preprocessing_execution"] == dataset.provenance["preprocessing_execution"]
    entry = prepare_source_record_prefixes(target, root=tmp_path, output=tmp_path / "target", budgets=(0, 16),
                                           minimum_test_trials=4)
    path = tmp_path / "prefixes.json"
    path.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": entry["preprocessing"], "preparation_method": entry["preparation_method"],
        "records": [entry]}), encoding="utf-8")
    prefixes = load_target_prefixes(path)
    test = prefixes["4"]["test"]
    loaded, _ = load_n2p3_trunk_checkpoint(payload, test, target_subject="4")
    assert loaded.feature_residual_config is not None and loaded.observed_channel_norm
    manifest, views = build_unified_scoped_tasks(prefixes, requested_subjects=["4"],
        channel_names=test.channel_names, preprocessing=SOURCE_EOG_PREPROCESSING.record(), batch_size=16,
        seed=7, source_prior_weight=.01, unified_training_budget=UnifiedTrainingBudget(target_epochs=3).record())
    assert manifest["task_count"] == 5 and manifest["fit_calls"] == 4
    assert not views.selection
    assert all("epoch_selection" not in task for task in manifest["tasks"])
    for key, view in views.items():
        np.testing.assert_array_equal(view.X[manifest["scopes"][key]["test_rows"]], test.X)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(payload, checkpoint_path)
    plan = {"source_snapshot_manifest": str(snapshot), "source_preprocessing": SOURCE_EOG_PREPROCESSING.record(),
            "source_prior_weight": .01, "unified_training_budget": UnifiedTrainingBudget(target_epochs=3).record(), "evidence_scope": "software_correctness",
            "calibration_trials": [0, 16], "target_batch_size": 16,
            "batch_rule": user_fixed_target_batch_rule(batch_size=16),
            "outer_folds": [{"name": "heldout", "excluded_source_datasets": [target["study"]],
                "target_prefix_manifest": str(path), "target_protocol": {"kind": "single_recording_prefix"},
                "target_study": target["study"]}]}
    source_record = {"resource_policy": {"target_workers": 1},
        "tasks": [{"key": "heldout/six/source", "status": "completed", "checkpoints": {"2": str(checkpoint_path)}}],
        "outer_evaluation_jobs": [{"source_checkpoint": str(checkpoint_path), "outer": "heldout", "layout": "six",
            "source_seed": config.seed, "target_seeds": [7, 8], "source_arm": condition["source_arm"],
            "unified_training_budget": plan["unified_training_budget"],
            "source_factors": condition["source_factors"], "target_stage": "unified"}]}
    _prepare_outer_plans(source_record, plan, tmp_path / "plans", device="cpu", execute=False)
    groups, declarations = target_job_groups(source_record, plan, tmp_path, device="cpu", program="unified")
    assert len(groups) == 1 and len(declarations) == 2
    assert declarations[0]["source_factors"] == condition["source_factors"]
    # Archive verification is exercised; matching a fixture archive to a live checkout is outside this probe.
    monkeypatch.setattr("experiments.run_cross_dataset_transfer.verified_executed_snapshot",
                        lambda path, root: verified_source_snapshot(path))
    results = _evaluate_outer_group(groups[0])
    assert all(result["failed_cells"] == 0 for result in results)
    assert results[1]["reused_cells"] == 1  # One deterministic LDA, independent NN seeds.
    report = json.loads((__import__("pathlib").Path(results[0]["output"]) / "record.json").read_text())
    assert report["completed_cells"] == 5, report["cells"]
    assert report["failed_cells"] == 0
    assert {cell["evaluation_contract"]["target_cache_sha256"] for cell in report["cells"]} == {
        scope["input_sha256"] for scope in manifest["scopes"].values()}


def test_source_prior_selection_uses_equal_people_and_preserves_failed_cells():
    from research.source_selection import select_source_prior_weight

    requested = [("a", 1, 1), ("a", 2, 2), ("b", 1, 1)]
    rows = [{"participant_key": person, "source_seed": source, "target_seed": target,
             "source_prior_weight": weight, "status": "completed", "auc": auc}
            for weight, scores in ((.001, (.9, .9, .5)), (.01, (.7, .7, .7)))
            for (person, source, target), auc in zip(requested, scores, strict=True)]
    result = select_source_prior_weight(rows, requested_cells=requested, candidates=(.001, .01))
    assert result["selected_weight"] == .001  # Both person-equal means are .7; select the smaller lambda.
    rows[0].update(status="fit_failed", auc=None)
    assert select_source_prior_weight(rows, requested_cells=requested, candidates=(.001, .01))["selected_weight"] == .01
    rows[-1]["auc"] = None
    assert select_source_prior_weight(rows, requested_cells=requested, candidates=(.001, .01))["status"] == "pending"
    with pytest.raises(ValueError, match="duplicate"):
        select_source_prior_weight(rows + [rows[0]], requested_cells=requested, candidates=(.001, .01))


def test_source_prior_reference_uses_the_declared_residual_configuration(tmp_path, monkeypatch):
    from experiments.run_source_training_chain import _select_unified_source_prior
    from tests.support.datasets import source_dataset

    dataset = source_dataset()
    person = dataset.identity_table.authority_keys()[0]
    manifest = tmp_path / "prefixes.json"
    manifest.write_text(json.dumps({"records": [{"subject": "person"}]}), encoding="utf-8")
    monkeypatch.setattr("research.source_selection.source_prior_development_fold", lambda *args, **kwargs: {
        "train_rows": [0, 1], "validation_participant_keys": [person]})
    monkeypatch.setattr("data.source_records.load_target_prefixes", lambda path, **kwargs: {
        "person": {"test": dataset, "calibration": {200: dataset}}})
    residual = {"bottleneck_channels": 2, "kernel_size": 5, "max_residual": .1}
    captured = []

    class ReferenceReachedError(Exception):
        pass

    def capture(**kwargs):
        captured.append(kwargs["architecture"])
        raise ReferenceReachedError

    monkeypatch.setattr("experiments.run_source_training_chain.fit_or_reuse_source_trajectory", capture)
    plan = {"source_prior_selection": {"prefix_manifests": [str(manifest)]},
            "unified_source": {"feature_residual": residual}}
    with pytest.raises(ReferenceReachedError):
        _select_unified_source_prior(dataset, {"name": "heldout", "excluded_source_datasets": ["outer"]},
            layout="six", plan=plan, report={"tasks": []}, output=tmp_path / "selection",
            source_sha="a" * 64, snapshot_sha="b" * 64, seed_pairs=((1, 1),), registry={}, device="cpu", dry_run=False)
    assert captured[0].feature_residual == residual
    assert captured[0].observed_channel_norm


@pytest.mark.parametrize("deployment", [False, True])
def test_source_prior_development_excludes_query_people_before_source_fitting(tmp_path, actual_raw_sources, monkeypatch, deployment):
    from data.identity import origin_subject_key
    from experiments.run_source_training_chain import execute_source_chain
    from research.evaluation import verified_source_snapshot
    from research.source_selection import source_prior_development_fold
    from train.early_stopping import UnifiedTrainingBudget

    dataset, target = actual_raw_sources
    outer = {"name": "heldout", "excluded_source_datasets": [target["study"]]}
    if deployment:
        outer.update(role="deployment", excluded_source_datasets=[],
                     source_studies=["BI2013a", "BI2014a", "BI2015a"])
    fold = source_prior_development_fold(dataset, seed=7, people_per_domain=2, excluded_source_datasets=outer["excluded_source_datasets"])
    assert not set(fold["validation_participant_keys"]) & set(fold["fit_ledger"]["training_participant_keys"])
    paths = []
    for study in ("BI2013a", "BI2014a", "BI2015a"):
        for subject in ("2", "3", "4", "5", "6", "7"):
            if origin_subject_key(study, subject) not in fold["validation_participant_keys"]:
                continue
            item = {**target, "study": study, "subject": subject}
            entry = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / f"prefix_{study}_{subject}",
                                                   budgets=(0, 16), minimum_test_trials=4, prepare_inner=False)
            path = tmp_path / f"prefix_{study}_{subject}.json"
            path.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
                "preprocessing": entry["preprocessing"], "preparation_method": entry["preparation_method"],
                "records": [entry]}), encoding="utf-8")
            paths.append(str(path))
    policy = UnifiedTrainingBudget(reference_patience=1, reference_min_delta=100., reference_max_epochs=2, target_epochs=2, development_seed=7)
    plan = {"evidence_scope": "software_correctness", "compile_mode": None,
        "source_preprocessing": SOURCE_EOG_PREPROCESSING.record(), "target_batch_size": 32, "target_workers": 2,
        "unified_training_budget": policy.record(), "source_prior_selection": {
            "calibration_trials": 16, "prefix_manifests": paths, "candidates": [0., .01]}}
    cache = save_epoch_dataset(tmp_path / "sources.npz", dataset)
    snapshot, _, _ = write_source_snapshot(tmp_path)
    plan.update(schema="n2p3_source_training_chain/1", source_snapshot_manifest=str(snapshot),
        source_seeds=[7, 8], target_seeds=[7, 8], outer_folds=[{**outer, "source_cache": str(cache)}],
        _source_group=True, _defer_outer_plans=True)
    monkeypatch.setattr("research.evaluation.verified_executed_snapshot", lambda path, root: verified_source_snapshot(path))
    report = execute_source_chain(plan, output=tmp_path / "development", dry_run=False, device="cpu", stages=("unified_source",))
    record = report["source_prior_selection"][0]
    assert record["status"] == "selected", record["outcomes"]
    assert len(record["outcomes"]) == len(fold["validation_participant_keys"]) * 2
    assert record["target_cost"]["total_optimizer_updates"] > 0
    assert record["target_workers"] == 2
    assert len(report["tasks"]) == 5
    assert len(report["outer_evaluation_jobs"]) == (0 if deployment else 4)
    assert record["source_budget"]["selected_epochs"] == 1
    assert all(row["target_recipe"]["epochs"] == 2 and row["epoch_selection"]["mode"] == "fixed_full_calibration" for row in record["outcomes"])
    reference = next(task for task in report["tasks"] if task.get("role") == "source_prior_development")
    assert not set(reference["fit_ledger"]["training_participant_keys"]) & set(fold["validation_participant_keys"])
    for task in report["tasks"]:
        if task is reference:
            continue
        assert task["config"]["global_early_stopping"] is None
        assert task["config"]["fixed_updates"] == 1
        assert task["config"]["source_budget"] == record["source_budget"]
        assert set(task["fit_ledger"]["training_participant_keys"]) == set(dataset.identity_table.authority_keys())
        saved = torch.load(next(iter(task["checkpoints"].values())), weights_only=False)
        assert saved["source_budget"] == record["source_budget"]



def test_development_support_keeps_all_other_source_people(actual_raw_sources):
    from research.contracts import semantic_sha256
    from research.source_selection import source_identity_axes, source_prior_development_fold

    dataset, _ = actual_raw_sources
    domains, people, _ = source_identity_axes(dataset)
    args = dict(seed=20260918, people_per_domain=2, excluded_source_datasets=["held"])
    fold = source_prior_development_fold(dataset, development_domains=["BI2013a"], **args)
    train = np.asarray(fold["train_rows"])
    validation = np.asarray(fold["validation_rows"])
    assert set(domains[validation]) == {"BI2013a"}
    assert set(np.flatnonzero(domains != "BI2013a")) <= set(train)
    assert not set(people[train]) & set(people[validation])
    alternative = source_prior_development_fold(dataset, development_domains=["BI2014a"], **args)
    assert semantic_sha256(fold) != semantic_sha256(alternative)
    for bad in (["unknown"], [], ["BI2013a", "BI2013a"]):
        with pytest.raises(ValueError):
            source_prior_development_fold(dataset, development_domains=bad, **args)
