"""General boundary counterexamples; no EEG inputs or training."""
from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.finite_filter_operator import FiniteFilterOperator
from research.contracts import TrainingRunContract, training_procedure_record
from transfer.exact_covariance import ExactPropagatedCovariance
from transfer.temporal_noise import TemporalNoise


def test_covariance_object_equality_is_a_boolean_and_state_is_explicit():
    operator = FiniteFilterOperator(((1., 0, 0, 1, 0, 0),), 0)
    noise = TemporalNoise.from_parameters((), innovation_variance=2., reference_id="equality")
    first = ExactPropagatedCovariance.compute(operator, noise, 4)
    second = ExactPropagatedCovariance.compute(operator, noise, 4)
    assert isinstance(first == second, bool)
    assert first == first
    assert "eigenvalues" in asdict(first)
    assert first._state_digest() == second._state_digest()


@pytest.mark.parametrize("count", [1.9, -.2, True])
def test_candidate_event_counts_are_exact_integers(count):
    from transfer.outcomes import CandidateCoverage

    with pytest.raises(ValueError, match="integer count"):
        CandidateCoverage.from_mappings({"1": count}, {"1": 0})


def test_mask_generator_does_not_return_an_incomplete_declared_mask(monkeypatch):
    from transfer.masking import MaskingConfig, make_temporal_mask

    def repeated(low, _high, _size, **_kwargs):
        return torch.tensor([low])
    monkeypatch.setattr(torch, "randint", repeated)
    with pytest.raises(RuntimeError, match="declared masked-sample count"):
        make_temporal_mask(48, config=MaskingConfig(mask_fraction=.5, min_block_samples=12, max_block_samples=12))


def test_mask_generator_preserves_the_existing_empty_mask_rejection():
    from transfer.masking import MaskingConfig, make_temporal_mask

    with pytest.raises(RuntimeError):
        make_temporal_mask(48, config=MaskingConfig(mask_fraction=.001))


@pytest.mark.parametrize("projection", ["legacy", "current"])
def test_analyzer_replays_the_saved_procedure_projection_without_relabelling(tmp_path, projection):
    from experiments.analyze_candidate_cross_decision import _parse_manifest
    from research.contracts import (
        LEGACY_SOURCE_PROCEDURE_SCHEMA,
        SOURCE_PROCEDURE_SCHEMA,
        semantic_sha256,
    )
    from tests.research.test_candidate_cross_decision_analysis import _make_fixture
    from tests.support.io import write_json

    path, manifest = _make_fixture(tmp_path)
    raw_training = next(iter(manifest["checkpoint_registry"].values()))["training_contract"]
    contract = TrainingRunContract(**raw_training)
    schema = LEGACY_SOURCE_PROCEDURE_SCHEMA if projection == "legacy" else SOURCE_PROCEDURE_SCHEMA
    procedure = training_procedure_record(contract, projection_schema=schema)
    assert ("schema" not in procedure) if projection == "legacy" else procedure["schema"] == SOURCE_PROCEDURE_SCHEMA
    for arm in manifest["arm_contracts"].values():
        arm["contract"]["source_training_procedure"] = procedure
        arm["digest"] = semantic_sha256(arm["contract"])
    write_json(path, manifest)
    original = path.read_bytes()
    loaded = _parse_manifest(path)
    assert path.read_bytes() == original
    assert all(arm.source_training_procedure == procedure for arm in loaded.arm_contracts.values())


def _contract(*, optimizer=None, objective=None):
    return TrainingRunContract(source_cache_sha256="a" * 64, source_identity_digest="b" * 64,
        source_snapshot_sha256="c" * 64, architecture={"kind": "fixed"}, preprocessing={"kind": "fixed"},
        optimizer=optimizer or {"name": "fixed"}, objective=objective or {"name": "CE"}, validation={"strategy": "fixed"},
        seed=7, training_participant_keys=("source-person",), holdout_participant_keys=())


@pytest.mark.parametrize("optimizer,field,new", [
    ({"name": "torch.optim.Adam", "config": {"lr": .001, "seed": 7}}, "config", {"lr": .002, "seed": 7}),
    ({"name": "torch.optim.AdamW", "learning_rate": .001}, "learning_rate", .002),
    ({"name": "torch.optim.SGD", "lr": .001, "momentum": .1}, "momentum", .9),
])
def test_training_procedure_retains_real_producer_controls(optimizer, field, new):
    base = _contract(optimizer=optimizer)
    changed = replace(base, optimizer={**optimizer, field: new})
    assert training_procedure_record(base) != training_procedure_record(changed)


def test_source_objective_mask_and_loss_are_controls_but_fitted_weights_are_outcomes():
    base = _contract(objective={"name": "masked_eeg_reconstruction", "mask": {"mask_fraction": .5},
                                "loss": {"spectral_weight": 1.}, "resolved_band_weights": [.2, .8]})
    changed = replace(base, objective={**base.objective, "mask": {"mask_fraction": .25}})
    assert training_procedure_record(base) != training_procedure_record(changed)
    fitted = replace(base, objective={**base.objective, "resolved_band_weights": [.4, .6]})
    assert training_procedure_record(base) == training_procedure_record(fitted)
    optimizer = {"name": "Adam", "config": {"lr": .1, "seed": 7}, "actual_updates": 3}
    repeated = {"name": "Adam", "config": {"lr": .1, "seed": 19}, "actual_updates": 8}
    assert training_procedure_record(_contract(optimizer=optimizer)) == training_procedure_record(_contract(optimizer=repeated))


def test_source_selected_refit_exposure_is_an_outcome_and_stopping_policy_is_a_control():
    policy = {"patience": 3, "source_max_epochs": 12}
    base = replace(_contract(optimizer={"name": "Adam", "config": {"lr": .001, "fixed_updates": 128,
                                                                    "global_early_stopping": None}}),
                   validation={"strategy": "source_internal_early_stopping_then_full_refit",
                               "early_stopping": {"policy": policy, "selected_epochs": 2}})
    changed = replace(base, optimizer={"name": "Adam", "config": {"lr": .001, "fixed_updates": 256,
                                                                  "global_early_stopping": None}},
                      validation={"strategy": "source_internal_early_stopping_then_full_refit",
                                  "early_stopping": {"policy": policy, "selected_epochs": 4}})
    assert training_procedure_record(base) == training_procedure_record(changed)
    different_policy = replace(base, validation={"strategy": "source_internal_early_stopping_then_full_refit",
        "early_stopping": {"policy": {**policy, "patience": 6}, "selected_epochs": 2}})
    assert training_procedure_record(base) != training_procedure_record(different_policy)
    fixed = replace(base, validation={"strategy": "fixed_source_updates"})
    different_fixed = replace(changed, validation={"strategy": "fixed_source_updates"})
    assert training_procedure_record(fixed) != training_procedure_record(different_fixed)


def test_source_program_rule_survives_population_changes_without_losing_actual_resolution():
    from data.identity import DatasetIdentityTable
    from research.contracts import semantic_sha256
    from research.source_selection import resolve_independent_source_config

    membership = {"schema": "n2p3_source_study_membership/1", "stream_to_study": {"S": "S"},
                  "evidence": [{"basis": "synthetic_fixture", "study_id": "S"}]}
    membership["digest"] = semantic_sha256(membership)
    def actual(count, declared):
        identity = DatasetIdentityTable.from_source_rows([str(i) for i in range(count)], ["S"] * count)
        program = resolve_independent_source_config(declared, identity, membership)
        return replace(_contract(optimizer={"name": "Adam", "config": {"lr": .001, **program["resolved"]}},
                                 objective={"name": "CE", "source_program_config": program}),
                       source_identity_digest=identity.digest(), training_participant_keys=identity.authority_keys())
    declared = {"pos_weight": 1., "val_groups_min": 2, "val_groups_max": None}
    first, other_population = actual(5, declared), actual(7, declared)
    assert training_procedure_record(first) == training_procedure_record(other_population)
    changed_rule = actual(5, {**declared, "val_groups_max": 3})
    assert training_procedure_record(first) != training_procedure_record(changed_rule)
    inconsistent = replace(first, optimizer={"name": "Adam", "config": {
        **first.optimizer["config"], "val_groups_max": 99}})
    with pytest.raises(ValueError, match="resolution differs"):
        training_procedure_record(inconsistent)


def test_classical_manifest_tracks_the_actual_constructor_hyperparameters():
    from baselines.classic import SWLDA, TemplateMatching, WindowLogisticRegression
    from baselines.riemann import XdawnRiemann
    from train.factory import describe_binary_model

    first = describe_binary_model("template", TemplateMatching(window_ms=(250., 500.)))
    second = describe_binary_model("template", TemplateMatching(window_ms=(300., 600.)))
    assert first != second
    assert first["hyperparameters"]["window_ms"] == (250., 500.)
    for name, model, field in (("swdla", SWLDA(), "max_features"),
                               ("window_lr", WindowLogisticRegression(), "window_ms"),
                               ("xdawn_rg", XdawnRiemann(nfilter=2), "nfilter")):
        assert describe_binary_model(name, model)["hyperparameters"][field] == getattr(model, field)


def test_dense_pds_empty_class_draws_consume_the_finite_budget(monkeypatch):
    import train.source_sampling as sampling

    class EmptyDraws:
        def __init__(self):
            self.draws = 0
        def choice(self, values, **kwargs):
            if "p" in kwargs:
                self.draws += 1
                if self.draws > 3:
                    raise AssertionError("empty class exhausted the declared proposal budget")
                return 0
            return np.asarray(values)[0]
        def permutation(self, values):
            return np.asarray(values)
    rng = EmptyDraws()
    monkeypatch.setattr(sampling, "_torch", None)
    monkeypatch.setattr(sampling.np.random, "default_rng", lambda _seed: rng)
    result = sampling.select_dense_pds_rows(np.array([[0.], [2.], [4.]]), np.zeros((1, 1)),
        n_select=3, labels=np.array([0, 1, 1]), seed=7, selection_scope="target_unlabelled_buffer",
        pds_radius=1., pds_mingling_mixture=(.9, .1), pds_max_attempts=3)
    assert rng.draws == result.pds_attempts == 3 and result.pds_fallback_count == 2
    assert result.pds_attempt_policy == "all_class_draws_including_empty/1"
    assert sampling.pds_selection_from_record(result.record()).record() == result.record()
    historical = result.record()
    historical.pop("pds_attempt_policy")
    historical["pds_attempts"] = max(historical["pds_attempts"], historical["pds_fallback_count"])
    assert sampling.pds_selection_from_record(historical).record() == historical


@pytest.mark.parametrize("where", ["allocation", "distance"])
def test_dense_pds_gpu_oom_reuses_selected_rows_and_continues_on_cpu(monkeypatch, where):
    import train.source_sampling as sampling

    source = np.arange(1., 13.).reshape(6, 2)
    arguments = dict(n_select=4, labels=np.array([0, 1] * 3), seed=17,
        selection_scope="target_unlabelled_buffer", pds_radius=.5, pds_max_attempts=8)
    with monkeypatch.context() as cpu:
        cpu.setattr(sampling, "_torch", None)
        expected = sampling.select_dense_pds_rows(source, np.zeros((1, 2)), **arguments)
    calls = []
    def allocate(value, **kwargs):
        if where == "allocation" and kwargs.get("dtype") == torch.float64:
            calls.append("allocation")
            raise RuntimeError("CUDA out of memory")
        return torch.as_tensor(value, **kwargs)
    def distance(*args, **kwargs):
        calls.append("distance")
        raise RuntimeError("CUDA out of memory")
    backend = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda _name: torch.device("cpu"),
        float64=torch.float64, long=torch.long, as_tensor=allocate, empty=torch.empty, cdist=distance)
    monkeypatch.setattr(sampling, "_torch", backend)
    actual = sampling.select_dense_pds_rows(source, np.zeros((1, 2)), **arguments)
    assert calls == [where]
    assert actual.pds_distance_backend == ("cpu_f64" if where == "allocation" else "gpu_f64_then_cpu_f64")
    np.testing.assert_array_equal(actual.selected_indices, expected.selected_indices)
    assert actual.pds_attempts == expected.pds_attempts


def test_quality_generator_identity_covers_the_numeric_modules():
    from research.recording_quality import generator_program_record

    record = generator_program_record()
    assert {"data.preprocess", "data.qc_features", "transfer.quality"} <= record["module_sha256"].keys()
    assert all(len(digest) == 64 for digest in record["module_sha256"].values())


def test_mask_probe_state_identity_preserves_bfloat16_bytes():
    from transfer.mask_probe import _state_digest

    state = {"weight": torch.tensor([1., 2.], dtype=torch.bfloat16)}
    holder = SimpleNamespace(state_dict=lambda: state)
    before = _state_digest(holder)
    assert len(before) == 64
    state["weight"][0] += 1
    assert _state_digest(holder) != before


@pytest.mark.parametrize("domain_rows", [4, 16])
def test_conditional_mmd_keeps_finite_gradients_for_resolved_small_bandwidth(domain_rows):
    from transfer.source_objective import SourceObjective, SourceObjectiveConfig

    count = domain_rows * 2
    labels = np.tile(np.repeat([0, 1], domain_rows // 2), 2)
    domains = np.repeat(["a", "b"], domain_rows)
    features = torch.tensor(np.column_stack((np.ones(count), np.arange(count) * 1e-21)),
                            dtype=torch.float32, requires_grad=True)
    objective = SourceObjective(SourceObjectiveConfig(mmd_weight=1.), labels=labels,
        domains=domains, participants=np.repeat(["p1", "p2"], domain_rows), device=torch.device("cpu"))
    actual, _ = objective.loss(features, torch.arange(count), feature_location="post_mst")
    actual.backward()
    assert torch.isfinite(actual) and torch.isfinite(features.grad).all()
    # Double-precision direct differences independently evaluate the same fixed
    # kernels. Bandwidths consume the actual detached float32 distance program.
    reference = features.detach().double().requires_grad_()
    z = torch.nn.functional.normalize(reference, dim=1)
    target_z = torch.nn.functional.normalize(features.detach(), dim=1)
    terms = []
    for label in (0, 1):
        first = torch.as_tensor((labels == label) & (domains == "a"))
        second = torch.as_tensor((labels == label) & (domains == "b"))
        distances = torch.pdist(target_z[first | second])
        sigma = distances[distances > 0].median().double()
        left, right = z[first], z[second]
        def kernel(a, b, sigma=sigma):
            return torch.exp(-((a[:, None] - b[None, :]) / sigma).square().sum(-1) / 2)
        within_left, within_right = kernel(left, left), kernel(right, right)
        terms.append((within_left.sum() - within_left.diag().sum()) / (len(left) * (len(left) - 1))
                     + (within_right.sum() - within_right.diag().sum()) / (len(right) * (len(right) - 1))
                     - 2 * kernel(left, right).mean())
    oracle = torch.stack(terms).mean()
    oracle.backward()
    assert float(actual.detach()) == pytest.approx(float(oracle.detach()), rel=2e-4)
    np.testing.assert_allclose(features.grad.numpy(), reference.grad.numpy(), rtol=2e-3, atol=1e-3)
