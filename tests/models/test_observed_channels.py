"""Observed-electrode normalization and the shared residual graph."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def make_model(**kwargs):
    return N2P3Net(6, n_times=DEFAULT_P300_DATA_CONTRACT.n_times, dropout=0, **kwargs)


def test_all_observed_matches_original_forward_gradients_and_bn_state():
    torch.manual_seed(7)
    original = make_model().double()
    observed = make_model(observed_channel_norm=True).double()
    observed.load_state_dict(original.state_dict())
    x = torch.randn(3, 6, original.n_times, dtype=torch.float64)
    target = torch.randn(3, 2, dtype=torch.float64)
    first = original(x)
    second = observed(x, channel_mask=torch.ones(3, 6, dtype=torch.bool))
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)
    (first * target).sum().backward()
    (second * target).sum().backward()
    for left, right in zip(original.parameters(), observed.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-10)
    for name, value in original.named_buffers():
        torch.testing.assert_close(value, dict(observed.named_buffers())[name], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("training", [False, True])
def test_missing_nan_values_never_enter_forward_or_weight_gradients(training):
    torch.manual_seed(4)
    model = make_model(observed_channel_norm=True).double().train(training)
    reference = copy.deepcopy(model)
    x = torch.randn(4, 6, model.n_times, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, True, False, True]]).expand(4, -1)
    missing = torch.where(mask[:, :, None], x, float("nan")).requires_grad_()
    zero = torch.where(mask[:, :, None], x, 0.0).requires_grad_()
    actual, expected = model(missing, channel_mask=mask), reference(zero, channel_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.count_nonzero(missing.grad[~mask]) == 0
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_observed_temporal_bn_uses_only_measured_channel_samples():
    model = make_model(observed_channel_norm=True).double()
    model.st_temporal_norm.momentum = None
    x = torch.randn(2, 6, model.n_times, dtype=torch.float64)
    mask = torch.tensor([[True, False, False, True, False, True], [True] * 6])
    with torch.no_grad():
        values = model.st_temporal(x[mask][:, None, None, :])
        expected_mean = values.mean((0, 2, 3))
        expected_var = values.var((0, 2, 3), unbiased=True)
        model(x, channel_mask=mask)
    torch.testing.assert_close(model.st_temporal_norm.running_mean, expected_mean)
    torch.testing.assert_close(model.st_temporal_norm.running_var, expected_var)
    assert model.st_temporal_norm.num_batches_tracked == 1


def test_empty_observation_rejects_before_mutating_state():
    model = make_model(observed_channel_norm=True)
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="observed channel"):
        model(torch.randn(2, 6, model.n_times), channel_mask=torch.zeros(2, 6, dtype=torch.bool))
    for name, value in state.items():
        assert torch.equal(value, model.state_dict()[name])


def test_shared_residual_starts_as_core_and_is_owned_and_serialized():
    config = N2P3ArchitectureConfig(
        dropout=0, observed_channel_norm=True,
        feature_residual={"bottleneck_channels": 4, "kernel_size": 9, "max_residual": 0.25},
    )
    torch.manual_seed(10)
    plain = make_model(observed_channel_norm=True)
    torch.manual_seed(10)
    model = N2P3Net(6, n_times=plain.n_times, **config.model_kwargs())
    x = torch.randn(2, 6, plain.n_times)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert torch.equal(plain.eval()(x, channel_mask=mask), model.eval()(x, channel_mask=mask))
    assert sum(p.numel() for p in model.parameters()) == 1279
    assert "feature_residual" in model.adaptation_parameter_groups()
    record = model.architecture_record()
    assert record["channel_masking"] == "observed_temporal_bn/1"
    assert record["feature_residual"] == config.feature_residual
    output = model(x, channel_mask=mask)
    output[:, 1].sum().backward()
    assert model.feature_residual.gate.grad.abs() > 0


def test_personal_fit_reestimation_and_read_only_prediction_keep_the_mask():
    rng = np.random.default_rng(6)
    model = make_model(observed_channel_norm=True)
    x = rng.normal(size=(12, 6, model.n_times)).astype(np.float32)
    mask = np.ones(x.shape[:2], dtype=bool)
    mask[:, 1] = False
    contaminated = x.copy()
    contaminated[:, 1] = np.nan
    config = SubjectAdapterConfig(
        head_kind="full_fine", trainable_weights="non_bn", batchnorm_affine="trainable",
        batchnorm_buffers="calibration", input_statistics="target_prefix", dropout_mode="off",
        max_updates=2, batch_size=5, val_group_fraction=None, refit_full_prefix=False,
    )
    first = SubjectAdapter(copy.deepcopy(model), config=config).fit(
        contaminated, np.arange(12) % 2, trial_channel_mask=mask,
    )
    second = SubjectAdapter(copy.deepcopy(model), config=config).fit(
        x, np.arange(12) % 2, trial_channel_mask=mask,
    )
    assert first.fitted_state_digest() == second.fitted_state_digest()
    assert first.bn_reestimation_[0]["count"] == 12 * 5 * model.n_times
    state = first.fitted_state_digest()
    actual = first.predict_logit(contaminated, trial_channel_mask=mask)
    np.testing.assert_allclose(actual, second.predict_logit(x, trial_channel_mask=mask), atol=0)
    assert first.fitted_state_digest() == state
