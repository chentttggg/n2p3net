from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from tests.support.datasets import epoch_dataset, source_dataset
from tests.support.snapshots import write_source_snapshot
from train.source_sampling import observed_source_statistics, source_sampling_probabilities


def test_observed_hierarchical_statistics_include_between_means() -> None:
    values = np.asarray([0, 0, 4, 10], dtype=np.float32)[:, None, None]
    mean, std, ledger = observed_source_statistics(
        values, np.ones((4, 1), bool),
        domain_ids=np.asarray(["a", "a", "a", "b"]),
        participant_keys=np.asarray(["a1", "a1", "a2", "b1"]),
        weighting="observed_domain_participant",
    )
    # Weights 1/8, 1/8, 1/4, 1/2 give E[X]=6 and E[X^2]=54.
    assert mean.item() == pytest.approx(6)
    assert std.item() == pytest.approx(np.sqrt(18) + 1e-6)
    assert ledger["channels"][0]["observed_domains"] == 2
    probability = source_sampling_probabilities(
        np.asarray(["a", "a", "a", "b"]),
        np.asarray(["a1", "a1", "a2", "b1"]),
        sampling="domain_participant",
    )
    assert probability == pytest.approx([0.125, 0.125, 0.25, 0.5])


def test_fixed_source_mode_rejects_unsupported_epoch_materialization() -> None:
    with pytest.raises(ValueError, match="materialize_shuffled_epoch"):
        DeepConfig(fixed_updates=1, materialize_shuffled_epoch=True)


def test_fixed_updates_fit_has_exact_draws_and_independent_snapshots() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(12, 3, 180)).astype(np.float32)
    config = DeepConfig(
        fixed_updates=3, snapshot_updates=(1, 3), batch_size=16,
        sampling="uniform_rows", compile_mode=None, val_group_frac=None,
    )
    model = N2P3NetBaseline(3, 180, 128, config=config, device=torch.device("cpu"))
    model.fit(values, np.tile([0, 1], 6))
    assert model.last_history["optimizer_updates"] == 3
    assert model.last_history["trial_draws"] == 48
    assert set(model.candidate_snapshots_) == {1, 3}
    before = model.candidate_snapshots_[1]["model_state_dict"]
    after = model.candidate_snapshots_[3]["model_state_dict"]
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert model.predict_logit(values).shape == (12,)
    from dataclasses import replace
    standalone = N2P3NetBaseline(3, 180, 128, config=replace(config, fixed_updates=1, snapshot_updates=()),
                                device=torch.device("cpu"))
    standalone.fit(values, np.tile([0, 1], 6))
    for key, value in before.items():
        torch.testing.assert_close(value, standalone.model_.state_dict()[key], rtol=0, atol=0)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ]
)
def source_failure_case(request):
    values = np.random.default_rng(4).normal(size=(12, 3, 180)).astype(np.float32)
    labels = np.arange(len(values)) % 2
    model = N2P3NetBaseline(
        values.shape[1],
        values.shape[2],
        128,
        device=torch.device(request.param),
        config=DeepConfig(
            fixed_updates=2,
            snapshot_updates=(1, 2),
            batch_size=4,
            sampling="uniform_rows",
            compile_mode=None,
            val_group_frac=None,
        ),
    )
    return model, values, labels


def _assert_source_failure_recovers(case):
    model, values, labels = case
    assert not model.candidate_snapshots_
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_logit(values)
    model.fit(values, labels)
    assert np.isfinite(model.predict_logit(values)).all()


def test_nonfinite_source_loss_invalidates_fitted_model(monkeypatch, source_failure_case):
    from torch.nn import functional as F

    from research.execution import ExpectedSubjectError

    model, values, labels = source_failure_case
    original_loss = F.cross_entropy
    factors = iter((1.0, float("nan")))

    def loss(*args, **kwargs):
        return original_loss(*args, **kwargs) * next(factors)

    with monkeypatch.context() as injected:
        injected.setattr(F, "cross_entropy", loss)
        with pytest.raises(ExpectedSubjectError, match="nonfinite_optimization"):
            model.fit(values, labels)
    _assert_source_failure_recovers(source_failure_case)


def test_nonfinite_source_gradient_is_detected_even_after_optimizer_sanitizes_it(
    monkeypatch, source_failure_case
):
    from torch.nn import functional as F

    from research.execution import ExpectedSubjectError

    model, values, labels = source_failure_case
    original_loss, original_step = F.cross_entropy, torch.optim.Adam.step
    factors = iter((1.0, float("nan")))

    class GradientFactor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, factor):
            ctx.factor = factor
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return gradient * ctx.factor, None

    def loss(*args, **kwargs):
        return GradientFactor.apply(original_loss(*args, **kwargs), next(factors))

    def step(optimizer, *args, **kwargs):
        # All parameters in this core participate in CE; keep the state finite
        # so only the gradient health check can detect the failed update.
        for parameter in model.model_.parameters():
            parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return original_step(optimizer, *args, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(F, "cross_entropy", loss)
        injected.setattr(torch.optim.Adam, "step", step)
        with pytest.raises(ExpectedSubjectError, match="nonfinite_optimization"):
            model.fit(values, labels)
    _assert_source_failure_recovers(source_failure_case)


def test_nonfinite_source_parameter_after_update_invalidates_fitted_model(
    monkeypatch, source_failure_case
):
    from research.execution import ExpectedSubjectError

    model, values, labels = source_failure_case
    original_step = torch.optim.Adam.step
    updates = 0

    def step(optimizer, *args, **kwargs):
        nonlocal updates
        updates += 1
        result = original_step(optimizer, *args, **kwargs)
        if updates == 2:
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    with monkeypatch.context() as injected:
        injected.setattr(torch.optim.Adam, "step", step)
        with pytest.raises(ExpectedSubjectError, match="nonfinite_optimization"):
            model.fit(values, labels)
    _assert_source_failure_recovers(source_failure_case)


def test_source_selection_keeps_fixed_failure_cohort_and_separate_zero_score() -> None:
    from research.source_selection import select_source_updates

    requested = [("a", "person", 0), ("a", "person", 200), ("a", "person", 800)]
    outcomes = []
    for update, zero, calibrated in [(500, 0.9, 0.6), (1000, 0.7, 0.8)]:
        for domain, person, budget in requested:
            outcomes.append({"updates": update, "domain": domain, "participant_key": person,
                             "actual_K": budget, "status": "completed", "source_ce": 1 / update,
                             "auc": zero if budget == 0 else calibrated})
    result = select_source_updates(outcomes, requested_cells=requested, candidates=(500, 1000))
    assert result["U_zero"] == 500
    assert result["U_cal"] == 1000
    outcomes[-1].update(status="fit_failure", auc=None)
    result = select_source_updates(outcomes, requested_cells=requested, candidates=(500, 1000))
    assert result["U_cal"] == 500
    assert result["curves"][1]["calibration_budget_policy_score"] is None


def test_source_fit_contract_rejects_outer_library_other_person_and_duplicate_namespace() -> None:
    from data.epochs import concatenate_epoch_datasets
    from research.source_selection import source_fit_ledger, source_identity_axes

    dataset = source_dataset()
    with pytest.raises(ValueError, match="excluded outer source library"):
        source_fit_ledger(dataset, np.asarray([0, 1]), excluded_source_datasets=["source_library"])
    duplicate = concatenate_epoch_datasets([dataset, source_dataset(namespace="renamed")], name="duplicate")
    with pytest.raises(ValueError, match="authority key"):
        source_identity_axes(duplicate)


def test_source_trajectory_checkpoint_loads_and_never_uses_held_out_statistics() -> None:
    from research.source_selection import fit_source_trajectory
    from transfer.checkpoint import (
        checkpoint_input_stats,
        load_n2p3_trunk_checkpoint,
        predict_n2p3_checkpoint,
    )

    dataset = source_dataset()
    rows = np.asarray([0, 1, 2, 3])
    config = DeepConfig(fixed_updates=1, snapshot_updates=(1,), batch_size=4,
                        sampling="uniform_rows", compile_mode=None, val_group_frac=None)
    model, payloads = fit_source_trajectory(dataset, rows, config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["target_library"], device=torch.device("cpu"))
    trunk, payload = load_n2p3_trunk_checkpoint(payloads[1], dataset, target_subject="source::s3")
    scores = predict_n2p3_checkpoint(trunk, dataset.X[4:],
        input_stats=checkpoint_input_stats(payload, dataset.n_channels), device=torch.device("cpu"))
    assert np.isfinite(scores).all()
    assert len(payload["training_contract"]["training_participant_keys"]) == 2
    assert payload["source_statistics"]["channels"][0]["observed_rows"] == 4
    expected_mean = model._input_mean.copy()
    dataset.X[4:] += 100
    other, _ = fit_source_trajectory(dataset, rows, config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["target_library"], device=torch.device("cpu"))
    np.testing.assert_array_equal(expected_mean, other._input_mean)
    for name, value in model.model_.state_dict().items():
        torch.testing.assert_close(value, other.model_.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("missing_channels", [True, False])
def test_source_checkpoint_reloads_the_executed_input_transform(tmp_path, standardize, missing_channels):
    from data.qc_features import compute_epoch_qc_features
    from research.source_selection import fit_source_trajectory
    from transfer.checkpoint import (
        checkpoint_input_stats,
        load_n2p3_trunk_checkpoint,
        predict_n2p3_checkpoint,
    )

    dataset = source_dataset()
    observed = np.ones(dataset.X.shape[:2], dtype=bool)
    if missing_channels:
        observed[::2, 0] = False
        dataset.X[~observed] = 0
        dataset.trial_channel_mask = observed
        dataset.qc_features = compute_epoch_qc_features(
            dataset.X, channel_mask=dataset.channel_mask, trial_channel_mask=observed)
    config = DeepConfig(fixed_updates=1, snapshot_updates=(1,), batch_size=4,
                        sampling="uniform_rows", compile_mode=None, val_group_frac=None,
                        standardize_input=standardize)
    model, payloads = fit_source_trajectory(dataset, np.arange(4), config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["target_library"], device=torch.device("cpu"))
    path = tmp_path / "source.pt"
    torch.save(payloads[1], path)
    trunk, payload = load_n2p3_trunk_checkpoint(path, dataset, target_subject="source::s3")
    actual = predict_n2p3_checkpoint(trunk, dataset.X[4:],
        input_stats=checkpoint_input_stats(payload, dataset.n_channels),
        trial_channel_mask=observed[4:], device=torch.device("cpu"))
    expected = model.predict_logit(dataset.X[4:], trial_channel_mask=observed[4:])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if not standardize:
        np.testing.assert_array_equal(payload["input_mean"], np.zeros(dataset.n_channels))
        np.testing.assert_array_equal(payload["input_std"], np.ones(dataset.n_channels))


def test_source_checkpoint_rejects_nonidentity_transform_declared_disabled():
    from research.source_selection import fit_source_trajectory
    from transfer.checkpoint import load_n2p3_trunk_checkpoint

    dataset = source_dataset()
    config = DeepConfig(fixed_updates=1, batch_size=4, compile_mode=None,
                        standardize_input=False, val_group_frac=None)
    _, payloads = fit_source_trajectory(dataset, np.arange(4), config=config,
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["target_library"], device=torch.device("cpu"))
    payloads[1]["input_mean"] = [0.1] * dataset.n_channels
    with pytest.raises(ValueError, match="disabled.*identity"):
        load_n2p3_trunk_checkpoint(payloads[1], dataset, target_subject="source::s3")


def test_source_trajectory_reuses_physical_snapshot_across_outer_roles(tmp_path) -> None:
    from dataclasses import replace

    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory

    dataset = source_dataset()
    registry = {}
    config = DeepConfig(fixed_updates=2, snapshot_updates=(1, 2), batch_size=4,
                        sampling="uniform_rows", compile_mode=None, val_group_frac=None)
    arguments = dict(registry=registry, dataset=dataset, rows=np.asarray([0, 1, 2, 3]),
                     source_snapshot_sha256="f" * 64, device="cpu", evidence_scope="software_correctness")
    first = fit_or_reuse_source_trajectory(**arguments, config=config, destination=tmp_path / "first",
        source_cache_sha256="a" * 64, excluded_source_datasets=["outer_a"], task_id="first")
    dataset.X[4:] += 100
    reused = fit_or_reuse_source_trajectory(**arguments, config=replace(config, fixed_updates=1, snapshot_updates=()),
        destination=tmp_path / "second", source_cache_sha256="b" * 64,
        excluded_source_datasets=["outer_b"], task_id="second")
    assert first["physical_optimizer_updates"] == 2
    assert reused["physical_optimizer_updates"] == 0
    assert reused["checkpoints"][1] != first["checkpoints"][1]
    assert reused["source_training_reuse"]["origin_task"] == "first"
    assert reused["source_training_reuse"]["origin_source_cache_sha256"] == "a" * 64
    assert reused["source_training_reuse"]["requested_source_cache_sha256"] == "b" * 64
    original = torch.load(first["checkpoints"][1], weights_only=False)
    consumer = torch.load(reused["checkpoints"][1], weights_only=False)
    assert consumer["training_contract"]["source_cache_sha256"] == "b" * 64
    assert original["training_contract"]["source_cache_sha256"] == "a" * 64
    assert consumer["training_contract"]["validation"]["strategy"] == "exact_fitted_source_trajectory_reuse"
    assert consumer["source_training_reuse"]["origin_checkpoint"]["path"] == first["checkpoints"][1]
    for key, value in original["trunk_state_dict"].items():
        torch.testing.assert_close(value, consumer["trunk_state_dict"][key], rtol=0, atol=0)
    from transfer.checkpoint import load_n2p3_trunk_checkpoint
    loaded, validated = load_n2p3_trunk_checkpoint(consumer, dataset, target_subject="source::s3")
    assert validated["training_contract"]["optimizer"]["physical_optimizer_updates"] == 0
    for key, value in original["trunk_state_dict"].items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)


def test_source_trajectory_reuse_rejects_different_rows_seed_and_corrupt_state(tmp_path) -> None:
    from dataclasses import replace
    from pathlib import Path

    from experiments.run_source_training_chain import fit_or_reuse_source_trajectory

    dataset = source_dataset()
    dataset.X.fill(0)
    registry = {}
    config = DeepConfig(fixed_updates=1, batch_size=4, sampling="uniform_rows", compile_mode=None)
    arguments = dict(registry=registry, dataset=dataset, source_cache_sha256="a" * 64,
                     source_snapshot_sha256="f" * 64, device="cpu", evidence_scope="software_correctness",
                     excluded_source_datasets=["outer"])
    original = fit_or_reuse_source_trajectory(**arguments, rows=np.asarray([0, 1, 2, 3]), config=config,
        destination=tmp_path / "original", task_id="original")
    for name, rows, recipe in (("rows", [2, 3, 4, 5], config),
                               ("seed", [0, 1, 2, 3], replace(config, seed=99))):
        result = fit_or_reuse_source_trajectory(**arguments, rows=np.asarray(rows), config=recipe,
            destination=tmp_path / name, task_id=name)
        assert result["physical_optimizer_updates"] == 1
        assert not result["source_training_reuse"]["reused"]
    from models.n2p3net import DEFAULT_N2P3_ARCHITECTURE

    for name, architecture in (
        ("mask", replace(DEFAULT_N2P3_ARCHITECTURE, observed_channel_norm=True)),
        ("residual", replace(DEFAULT_N2P3_ARCHITECTURE, observed_channel_norm=True,
                            feature_residual={"bottleneck_channels": 4, "kernel_size": 9, "max_residual": .25})),
    ):
        changed = fit_or_reuse_source_trajectory(**arguments, rows=np.asarray([0, 1, 2, 3]), config=config,
            destination=tmp_path / name, task_id=name, architecture=architecture)
        repeated = fit_or_reuse_source_trajectory(**arguments, rows=np.asarray([0, 1, 2, 3]), config=config,
            destination=tmp_path / (name + "_again"), task_id=name + "_again", architecture=architecture)
        assert changed["physical_optimizer_updates"] == 1
        assert repeated["physical_optimizer_updates"] == 0
    checkpoint = Path(original["checkpoints"][1])
    payload = torch.load(checkpoint, weights_only=False)
    first_parameter = next(iter(payload["trunk_state_dict"].values()))
    first_parameter.add_(1)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="checkpoint changed"):
        fit_or_reuse_source_trajectory(**arguments, rows=np.asarray([0, 1, 2, 3]), config=config,
            destination=tmp_path / "corrupted", task_id="corrupted")


def test_fixed_dose_failure_does_not_replace_people_or_resample_labels() -> None:
    from research.source_selection import source_fixed_dose, source_identity_axes

    dataset = source_dataset()
    _, people, _ = source_identity_axes(dataset)
    chronology = {person: np.flatnonzero(people == person).tolist() for person in np.unique(people)}
    result = source_fixed_dose(dataset, domain_ids=["source"], trials_per_participant=3,
        seed=1, chronological_rows=chronology, participants_per_domain=2)
    assert result["status"] == "invalid_design"
    assert len(result["requests"]) == len(result["failures"]) == 2
    assert result["selected_rows"] == []


def test_fixed_probability_risk_counts_class_and_domain_mass_exactly_once() -> None:
    values = np.random.default_rng(11).normal(size=(6, 3, 100)).astype(np.float32)
    domains = np.asarray(["a", "a", "a", "a", "b", "b"])
    people = np.asarray(["a1", "a1", "a2", "a2", "b1", "b1"])
    labels = np.asarray([0, 1, 0, 0, 0, 1])
    model = N2P3NetBaseline(3, 100, 100, device=torch.device("cpu"), config=DeepConfig(
        fixed_updates=1, batch_size=8, compile_mode=None, sampling="domain_participant"))
    model.fit(values, labels, source_domain_ids=domains, risk_unit_ids=people,
              source_domain_mass={"a": 0.5, "b": 0.5})
    ledger = model.sampling_ledger_
    # Each domain gets 1/2; each person in A gets 1/4. Positive class mass is 8.
    expected_risk_mass = np.asarray([1 / 36, 2 / 9, 1 / 8, 1 / 8, 1 / 18, 4 / 9])
    assert ledger["target_risk_mass"] == pytest.approx(expected_risk_mass)
    row_losses = np.asarray([0.1, 0.3, 0.6, 0.9, 1.2, 0.2])
    # Independent exhaustive expectation of the random-row estimator is 33/80.
    expected = 33 / 80
    estimate = np.dot(np.asarray(ledger["row_probability"]), np.asarray(ledger["importance_weight"]) * row_losses)
    assert estimate == pytest.approx(expected)


@pytest.mark.parametrize("libraries", [("a", "b", "c"), ("a", "b", "c", "e")])
def test_source_chain_dry_run_uses_production_caches_without_fitting(tmp_path, libraries) -> None:
    import pandas as pd

    from data.channel import build_channel_identity
    from data.contract import SOURCE_COHORT_DATA_CONTRACTS
    from data.domain import annotate_source_domain, namespace_epoch_dataset
    from data.epochs import (
        concatenate_epoch_datasets,
        preprocessing_spec_from_contract,
        save_epoch_dataset,
    )
    from data.events import observed_only_timeline
    from data.identity import DatasetIdentityTable
    from experiments.run_source_training_chain import execute_source_chain

    views = []
    for library in libraries:
        source = epoch_dataset()
        identity = build_channel_identity(("Fz", "P3", "Pz", "P4", "PO8", "Oz"), allow_missing_positions=False)
        source.channel_names = identity.names
        source.channel_positions_m = identity.coords
        source.channel_mask = np.ones(6, bool)
        source.preprocessing = preprocessing_spec_from_contract(SOURCE_COHORT_DATA_CONTRACTS["causal"])
        source.subject_ids = np.repeat(["s1", "s2", "s3", "s4"], 12)
        source.y = np.tile([0, 1], 24)
        source.metadata = pd.DataFrame({"subject": source.subject_ids, "run": ["1"] * 48})
        source.event_timeline = observed_only_timeline(
            dataset_id=library, subject_ids=source.subject_ids,
            stimulus_ids=np.arange(48), onset_times_s=np.arange(48) * 2.0,
            evidence_available_times_s=np.arange(48) * 2.0 + 1.2,
            group_ids=source.subject_ids, online_causal=True, timing_source="synthetic_schedule",
        )
        source.X = np.zeros((48, 6, source.preprocessing.n_times), np.float32)
        source.identity_table = DatasetIdentityTable.from_source_rows(source.subject_ids.tolist(), [library] * 48)
        source.provenance["source_reference"] = "measured_channel:CZ"
        views.append(annotate_source_domain(namespace_epoch_dataset(source, library), library))
    dataset = concatenate_epoch_datasets(views, name="source_chain_synthetic", provenance={
        "source_reference": "measured_channel:CZ",
        "source_domain_axis": {**views[0].provenance["source_domain_axis"], "domains": list(libraries)},
    })
    path = save_epoch_dataset(tmp_path / "source.npz", dataset)
    manifest, _, _ = write_source_snapshot(tmp_path)
    plan = {"schema": "n2p3_source_training_chain/1", "source_snapshot_manifest": str(manifest),
            "outer_folds": [{"name": "held_d", "source_cache": str(path), "excluded_source_datasets": ["d"],
                             **({"source_studies": list(libraries)} if len(libraries) != 3 else {}),
                             "source_protocols": {library: {"kind": "single_recording_prefix"} for library in libraries}}]}
    report = execute_source_chain(plan, output=tmp_path / "dry", dry_run=True, device="cpu", stages=("S1", "E2", "S2"))
    assert report["task_count"] == 10 + 2 * len(libraries)
    assert report["optimizer_updates"] == 32000 + (len(libraries) + 1) * 12000
    assert not report["fitting_performed"]
    assert not list((tmp_path / "dry").rglob("*.pt"))
    assert len([failure for failure in report["preflight_failures"] if failure.get("status") == "invalid_design"]) == len(libraries) + 1
    auxiliary = execute_source_chain(plan, output=tmp_path / "p2_dry", dry_run=True,
                                     device="cpu", stages=("P2_source",))
    assert auxiliary["task_count"] == 3
    assert auxiliary["optimizer_updates"] == 12000
    assert not auxiliary["fitting_performed"]
    assert all(job["target_stage"] == "P2" for job in auxiliary["outer_evaluation_jobs"])
    assert all(job["analysis_seed_pairs"] == [[20260906, 20260906]]
               for job in auxiliary["outer_evaluation_jobs"])
    assert all(job["analysis_source_arms"] == ["P2_source_ce", "P2_source_supcon", "P2_source_mmd"]
               for job in auxiliary["outer_evaluation_jobs"])


@pytest.mark.parametrize("weighting", ["observed_rows", "observed_domain_participant"])
def test_chunked_source_statistics_match_full_float64_oracle(weighting):
    rng = np.random.default_rng(91)
    values = rng.normal(size=(45000, 2, 3)).astype(np.float32)
    mask = rng.uniform(size=values.shape[:2]) > .2
    domains = np.asarray(["a", "b"])[np.arange(len(values)) % 2]
    people = np.asarray([f"p{i}" for i in range(8)])[np.arange(len(values)) % 8]
    mean, std, ledger = observed_source_statistics(values, mask, domain_ids=domains,
        participant_keys=people, weighting=weighting)
    for channel in range(2):
        rows = np.flatnonzero(mask[:, channel])
        weights = source_sampling_probabilities(domains[rows], people[rows],
            sampling="uniform_rows" if weighting == "observed_rows" else "domain_participant")
        full = values[rows, channel].astype(np.float64)
        expected_mean = float(weights @ full.mean(axis=1))
        expected_var = float(weights @ np.square(full - expected_mean).mean(axis=1))
        assert mean[0, channel, 0] == np.float32(expected_mean)
        assert std[0, channel, 0] == np.float32(np.sqrt(expected_var) + 1e-6)
        assert ledger["channels"][channel]["population_variance"] == expected_var
