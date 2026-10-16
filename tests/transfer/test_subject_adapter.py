from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from baselines.validation import group_disjoint_validation_split
from data.candidate_task import MembershipDecision
from models.n2p3net import N2P3Net
from models.parameter_groups import SourceParameterPrior
from tests.support.checkpoints import (
    small_trunk,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def test_frozen_subject_adapter_predicts_without_mutating_trunk_batchnorm() -> None:
    rng = np.random.default_rng(0)
    trunk = small_trunk()
    before = {
        name: value.detach().clone()
        for name, value in trunk.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    }
    adapter = SubjectAdapter(
        trunk,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=3,
            batch_size=16,
            val_group_fraction=None,
        ),
    )
    X = rng.normal(size=(32, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 16)
    X[y == 1, 2, 60:80] += 0.5
    adapter.fit(X, y)
    logits = adapter.predict_logit(X)
    assert logits.shape == (32,)
    assert np.isfinite(logits).all()
    # Frozen-trunk adapter should train only the 2-class linear head.
    assert adapter.parameter_count() == 34
    after = trunk.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


@pytest.mark.parametrize("freeze_batchnorm", [True, False])
def test_full_fine_batchnorm_adaptation_is_an_explicit_arm(freeze_batchnorm) -> None:
    adapter = SubjectAdapter(
        small_trunk(),
        config=SubjectAdapterConfig(head_kind="full_fine", freeze_batchnorm_stats=freeze_batchnorm),
    )
    adapter._set_training_mode(trunk_trainable=True)
    states = [
        module.training
        for module in adapter.trunk.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert states
    assert states == [not freeze_batchnorm] * len(states)


def test_subject_adapter_statistics_exclude_validation_groups() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    groups = np.repeat(np.array(["g0", "g1", "g2", "g3"]), 6)
    config = SubjectAdapterConfig(
        head_kind="linear",
        epochs=1,
        batch_size=8,
        val_group_fraction=0.25,
        val_groups_min=1,
        val_groups_max=1,
        early_stop_patience=1,
    )
    split = group_disjoint_validation_split(
        groups,
        fraction=config.val_group_fraction,
        min_groups=config.val_groups_min,
        max_groups=config.val_groups_max,
        seed=config.seed,
    )
    adapter = SubjectAdapter(small_trunk(), config=config)
    adapter.fit(X, y, group_ids=groups)
    expected = X[split.train_mask].mean(axis=(0, 2), keepdims=True)
    assert np.allclose(adapter._input_mean, expected)
    assert isinstance(adapter.calibration_logits_, np.ndarray)


def test_chronological_refit_uses_every_prefix_row_after_epoch_selection() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(30, 3, 128)).astype(np.float32)
    y = np.tile(np.array([0, 0, 0, 0, 1], dtype=np.int64), 6)
    train = np.arange(len(X)) < 18
    validation = np.arange(len(X)) >= 22
    adapter = SubjectAdapter(
        small_trunk(),
        config=SubjectAdapterConfig(epochs=2, batch_size=16, early_stop_patience=2),
    )

    adapter.fit(
        X,
        y,
        training_mask=train,
        validation_mask=validation,
        prefix_mask=np.ones(len(X), dtype=bool),
    )

    assert adapter.last_history["refit_full_prefix"] is True
    assert 1 <= int(adapter.last_history["refit_epochs"]) <= 2
    np.testing.assert_allclose(
        adapter._input_mean,
        X.mean(axis=(0, 2), keepdims=True),
        rtol=1e-6,
        atol=1e-6,
    )


def test_chronological_refit_requires_declared_prefix_before_fitting() -> None:
    rng = np.random.default_rng(119)
    X = rng.normal(size=(42, 3, 128)).astype(np.float32)
    y = np.arange(42) % 2
    train = np.arange(42) < 18
    validation = (np.arange(42) >= 22) & (np.arange(42) < 30)
    adapter = SubjectAdapter(small_trunk(), config=SubjectAdapterConfig(epochs=1, batch_size=16))
    with pytest.raises(ValueError, match="prefix_mask"):
        adapter.fit(X, y, training_mask=train, validation_mask=validation)


@pytest.mark.parametrize(
    "statistics,provided_input_stats",
    [
        ("source", {"input_mean": [0.0, 0.0, 0.0], "input_std": [1.0, 1.0, 1.0]}),
        ("target_prefix", {}),
        ("shrinkage", {"input_mean": [0.0, 0.0, 0.0], "input_std": [1.0, 1.0, 1.0]}),
    ],
)
@pytest.mark.parametrize("freeze_bn", [True, False])
def test_refit_ignores_unassigned_test_values_and_labels(
    statistics, provided_input_stats, freeze_bn
) -> None:
    rng = np.random.default_rng(120)
    X = rng.normal(size=(42, 3, 128)).astype(np.float32)
    y = np.arange(42) % 2
    changed_X, changed_y = X.copy(), y.copy()
    changed_X[30:] += 500.0
    changed_y[30:] = 1 - changed_y[30:]
    prefix = np.arange(42) < 30
    train = np.arange(42) < 18
    validation = (np.arange(42) >= 22) & prefix
    config = SubjectAdapterConfig(
        head_kind="full_fine",
        epochs=2,
        batch_size=16,
        seed=13,
        freeze_batchnorm_stats=freeze_bn,
        input_statistics=statistics,
        **{
            key: np.asarray(value, dtype=np.float32)[None, :, None]
            for key, value in provided_input_stats.items()
        },
    )
    initial = small_trunk()
    first = SubjectAdapter(copy.deepcopy(initial), config=config)
    second = SubjectAdapter(copy.deepcopy(initial), config=config)
    for adapter, values, labels in ((first, X, y), (second, changed_X, changed_y)):
        adapter.fit(
            values, labels, training_mask=train, validation_mask=validation, prefix_mask=prefix
        )
        assert adapter.last_history["refit_full_prefix"]
        assert adapter.last_history["fit_scope"]["refit_rows"] == 30
        assert adapter.last_history["fit_scope"]["inner_excluded_prefix_rows"] == 4
        assert adapter.last_history["fit_scope"]["outside_prefix_rows"] == 12
        assert adapter.training_prior_ == y[:30].mean()
    for key, value in first.trunk.state_dict().items():
        assert torch.equal(value, second.trunk.state_dict()[key]), key
    np.testing.assert_array_equal(first._input_mean, second._input_mean)
    np.testing.assert_array_equal(first._input_std, second._input_std)
    np.testing.assert_array_equal(first.predict_logit(X[:6]), second.predict_logit(X[:6]))
    if statistics == "target_prefix":
        np.testing.assert_allclose(
            first._input_mean, X[:30].mean(axis=(0, 2), keepdims=True), atol=1e-6
        )


@pytest.mark.parametrize(
    "bad_mask",
    [np.ones(30), np.ones((30, 1), dtype=bool), np.ones(29, dtype=bool), np.zeros(30, dtype=bool)],
)
def test_refit_rejects_malformed_or_empty_prefix_scope(bad_mask) -> None:
    X = np.ones((30, 3, 128), dtype=np.float32)
    y = np.arange(30) % 2
    adapter = SubjectAdapter(small_trunk(), config=SubjectAdapterConfig(epochs=1))
    with pytest.raises(ValueError, match="prefix_mask"):
        adapter.fit(X, y, prefix_mask=bad_mask)


@pytest.mark.parametrize("outside_role", ["training_mask", "validation_mask"])
def test_inner_masks_cannot_authorize_rows_outside_prefix(outside_role) -> None:
    X = np.ones((40, 3, 128), dtype=np.float32)
    y = np.arange(40) % 2
    prefix = np.arange(40) < 30
    masks = {
        "training_mask": np.arange(len(X)) < 18,
        "validation_mask": (np.arange(len(X)) >= 22) & prefix,
    }
    masks[outside_role][35] = True
    adapter = SubjectAdapter(small_trunk(), config=SubjectAdapterConfig(epochs=1))
    with pytest.raises(ValueError, match="inside prefix_mask"):
        adapter.fit(X, y, prefix_mask=prefix, **masks)


def test_refit_slices_candidate_metadata_and_retains_calibration_embargo() -> None:
    from data.candidate_task import MembershipDecision

    rng = np.random.default_rng(121)
    X = rng.normal(size=(40, 3, 128)).astype(np.float32)
    codes = np.tile(np.arange(4), 10)
    y = np.isin(codes, [0, 3]).astype(np.int64)
    prefix = np.arange(40) < 32
    train = np.arange(40) < 16
    validation = (np.arange(40) >= 20) & prefix
    initial = small_trunk()
    config = SubjectAdapterConfig(
        head_kind="full_fine", epochs=1, batch_size=16, candidate_loss_weight=0.1
    )
    with_context = SubjectAdapter(copy.deepcopy(initial), config=config)
    calibration_only = SubjectAdapter(copy.deepcopy(initial), config=config)
    decisions = tuple(MembershipDecision.row_column(np.arange(4), n_rows=2, n_columns=2,
        event_ids=tuple(f"decision-{index}:event-{event}" for event in range(4)),
        repetition_indices=np.zeros(4, dtype=np.int64), repetitions=1,
        evidence_indices=np.arange(index * 4, (index + 1) * 4)) for index in range(8))
    with_context.fit(
        X,
        y,
        prefix_mask=prefix,
        training_mask=train,
        validation_mask=validation,
        decision_groups=decisions,
        decision_targets=(1,) * 8,
    )
    calibration_only.fit(
        X[:32],
        y[:32],
        prefix_mask=np.ones(32, dtype=bool),
        training_mask=train[:32],
        validation_mask=validation[:32],
        decision_groups=decisions,
        decision_targets=(1,) * 8,
    )
    assert with_context.last_history["refit_candidate_losses"]
    for key, value in with_context.trunk.state_dict().items():
        assert torch.equal(value, calibration_only.trunk.state_dict()[key]), key


@pytest.mark.parametrize("split_argument", ["group_ids", "repetition_indices"])
def test_automatic_inner_split_is_restricted_to_declared_prefix(split_argument) -> None:
    rng = np.random.default_rng(122)
    X = rng.normal(size=(40, 3, 128)).astype(np.float32)
    y = np.arange(40) % 2
    groups = np.repeat(np.arange(10), 4)
    config = SubjectAdapterConfig(
        head_kind="full_fine",
        epochs=1,
        batch_size=16,
        val_group_fraction=0.25,
        val_groups_min=2,
        val_groups_max=2,
    )
    initial = small_trunk()
    context = SubjectAdapter(copy.deepcopy(initial), config=config)
    prefix_only = SubjectAdapter(copy.deepcopy(initial), config=config)
    context.fit(X, y, prefix_mask=np.arange(40) < 32, **{split_argument: groups})
    prefix_only.fit(
        X[:32], y[:32], prefix_mask=np.ones(32, dtype=bool), **{split_argument: groups[:32]}
    )
    assert context.last_history["fit_scope"]["outside_prefix_rows"] == 8
    for key, value in context.trunk.state_dict().items():
        assert torch.equal(value, prefix_only.trunk.state_dict()[key]), key


def test_refit_entry_rejects_empty_scope_before_restoring_weights() -> None:
    adapter = SubjectAdapter(small_trunk(), config=SubjectAdapterConfig(epochs=1))
    with pytest.raises(ValueError, match="prefix_mask"):
        adapter._refit_full_prefix(
            np.zeros((8, 3, 128), dtype=np.float32),
            np.arange(8) % 2,
            np.ones((8, 3), dtype=bool),
            prefix_mask=np.zeros(8, dtype=bool),
            initial_state={},
            epochs=1,
            statistics_mode="source",
            decision_groups=None,
            decision_targets=None,
            decision_weights=None,
        )


def _joint_source():
    return N2P3Net(3, n_times=32, dropout=0, observed_channel_norm=True)


def _joint_inputs_and_groups():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(18, 3, 32)).astype(np.float32)
    groups = []
    targets = [0, 1, 2]
    y = []
    codes = ("0", "1", "2") * 2
    for index, target in enumerate(targets):
        group = MembershipDecision.direct(
            codes, ("0", "1", "2"),
            event_ids=tuple(f"g{index}/e{i}" for i in range(6)),
            repetition_indices=np.repeat(np.arange(2), 3), repetitions=2,
            evidence_indices=np.arange(index * 6, index * 6 + 6),
        )
        groups.append(group)
        y.extend(group.membership[:, target].astype(int))
    return x, np.asarray(y), groups, targets


def _joint_config(weight, prior_weight=0.):
    return SubjectAdapterConfig(
        head_kind="full_fine", trainable_weights="non_bn", batchnorm_affine="trainable",
        batchnorm_buffers="source", input_statistics="target_prefix", dropout_mode="off",
        max_updates=3, batch_size=5, val_group_fraction=None, refit_full_prefix=False,
        candidate_loss_weight=weight, source_prior_weight=prior_weight, seed=42,
    )


def test_source_prior_has_fixed_center_and_expected_restoring_gradient():
    trunk = _joint_source()
    prior = SourceParameterPrior(trunk)
    center = trunk.classifier.bias.detach().clone()
    record = prior.record()
    assert prior.loss().item() == 0.0
    with torch.no_grad():
        trunk.classifier.bias.add_(0.2)
    prior.loss().backward()
    assert torch.all(trunk.classifier.bias.grad > 0)
    torch.testing.assert_close(trunk.classifier.bias.grad, torch.full_like(center, 0.1))
    assert prior.loss() > 0
    assert prior.record()["scale_squared"]["classifier.bias"] == 1.0
    assert prior.record() == record
    with torch.no_grad():
        trunk.classifier.bias.copy_(center)
        for parameter in trunk.parameters():
            parameter.add_(.1)
    assert prior.record() == record and prior.loss() > 0
    expected = torch.stack([torch.stack([
        (parameter.float() - reference.float()).square().mean() / (2 * scale)
        for parameter, reference, scale in members]).mean() for members in prior.groups.values()]).sum()
    torch.testing.assert_close(prior.loss(), expected)
    actual_gradient = torch.autograd.grad(prior.loss(), tuple(trunk.parameters()))
    expected_gradient = torch.autograd.grad(expected, tuple(trunk.parameters()))
    for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(actual, expected)


def test_all_zero_source_weights_have_finite_prior_and_gradient():
    trunk = _joint_source()
    with torch.no_grad():
        for parameter in trunk.parameters():
            parameter.zero_()
    prior = SourceParameterPrior(trunk)
    with torch.no_grad():
        trunk.st_spatial.weight.add_(0.2)
    loss = prior.loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(trunk.st_spatial.weight.grad).all()
    assert trunk.st_spatial.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_prior_weight_rejected(weight):
    with pytest.raises(ValueError, match="source_prior_weight"):
        SubjectAdapterConfig(source_prior_weight=weight)


def test_joint_updates_preserve_workset_and_isolate_decision_and_source_prior_effects(monkeypatch):
    x, y, groups, targets = _joint_inputs_and_groups()
    torch.manual_seed(9)
    source = _joint_source()
    original = copy.deepcopy(source.state_dict())
    center = SourceParameterPrior(source).record()["center_sha256"]
    calls, traces, fitted = [], [], []
    original_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        calls.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    for decision_weight, prior_weight in ((0., 0.), (1., 0.), (1., .2)):
        adapter = SubjectAdapter(copy.deepcopy(source), config=_joint_config(decision_weight, prior_weight))
        trace = []
        forward = adapter._forward_head

        def capture(values, *, channel_mask=None, _forward=forward, _trace=trace):
            _trace.append(values.detach().clone())
            return _forward(values, channel_mask=channel_mask)

        monkeypatch.setattr(adapter, "_forward_head", capture)
        previous_steps = len(calls)
        adapter.fit(x, y, decision_groups=groups, decision_targets=targets)
        assert len(calls) - previous_steps == 3
        assert max(map(len, trace)) <= 5
        budget = adapter.last_history["fit_budget"]
        assert budget["updates"] == budget["candidate_updates"] == 3
        assert budget["candidate_draws"] == 18 and budget["draws"] == 33
        assert adapter.last_history["decision_training"]["eligible_draws"] == 3
        traces.append(trace)
        fitted.append(adapter)
    assert len({len(trace) for trace in traces}) == 1
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(*traces, strict=True))
    assert len({adapter.fitted_state_digest() for adapter in fitted}) == 3
    prior = fitted[-1].last_history["source_parameter_prior"]
    assert prior["weight"] == .2 and prior["final_unweighted_penalty"] > 0
    assert prior["center_sha256"] == center
    assert all(torch.equal(value, source.state_dict()[name]) for name, value in original.items())


def test_budget_crossing_group_does_not_read_unallocated_truth_or_eeg():
    x, y, groups, targets = _joint_inputs_and_groups()
    torch.manual_seed(3)
    source = N2P3Net(3, n_times=32, observed_channel_norm=True, dropout=0)
    prefix = np.arange(len(x)) < 8
    first = SubjectAdapter(copy.deepcopy(source), config=_joint_config(1)).fit(
        x, y, prefix_mask=prefix, decision_groups=groups, decision_targets=targets,
    )
    changed, labels = x.copy(), y.copy()
    changed[8:] *= 10000
    labels[8:] = 1 - labels[8:]
    second = SubjectAdapter(copy.deepcopy(source), config=_joint_config(1)).fit(
        changed, labels, prefix_mask=prefix, decision_groups=groups, decision_targets=[0, 2, 0],
    )
    assert first.fitted_state_digest() == second.fitted_state_digest()
    assert first.last_history["decision_training"]["budget_excluded_groups"] == 2


def test_missing_requested_decision_stays_in_risk_without_trainable_loss():
    x, y, groups, targets = _joint_inputs_and_groups()
    missing = replace(groups[0], evidence_indices=np.full(6, -1, dtype=np.int64))
    adapter = SubjectAdapter(N2P3Net(3, n_times=32, dropout=0), config=_joint_config(1)).fit(
        x, y, decision_groups=[missing], decision_targets=[targets[0]],
    )
    ledger = adapter.last_history["decision_training"]
    assert ledger["requested_groups"] == ledger["authorized_groups"] == 1
    assert ledger["eligible_draws"] == 0
    assert adapter.last_history["fit_budget"]["candidate_updates"] == 3
