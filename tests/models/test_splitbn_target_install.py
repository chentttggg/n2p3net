"""Synthetic target-buffer installation contracts; no model fit or load."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from models.adapters import FeatureDomainTransfer, FeatureTransferConfig
from models.n2p3net import N2P3Net
from models.splitbn import (
    SplitBN1d,
    install_splitbn_target_calibration_buffers,
    validate_splitbn_target_install_receipt,
)


class _Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = SplitBN1d(2, domain_count=2, target_buffer_source="target_calibration")
        self.second = SplitBN1d(2, domain_count=2, target_buffer_source="target_calibration")

    def forward_features(self, x, *, eval_buffer_source=None, channel_mask=None):
        x = torch.where(channel_mask[:, :, None], x, 0.0) if channel_mask is not None else x
        x = self.first(x, eval_buffer_source=eval_buffer_source)
        return self.second(3 * x + 5, eval_buffer_source=eval_buffer_source)


def _values():
    x = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3)
    mask = torch.ones((4, 2), dtype=torch.bool)
    return x, mask


def test_downstream_moments_use_installed_upstream_target_normalization():
    values, mask = _values()
    model = _Chain().double()
    model.first.running_mean[0].fill_(11)
    model.first.running_var[0].fill_(7)
    model.first.num_batches_tracked[0].fill_(1)
    receipt = install_splitbn_target_calibration_buffers(model, values, device="cpu", channel_masks=mask)
    raw = torch.where(mask[:, :, None], values, 0.0)
    expected = 3 * (raw - model.first.target_running_mean[None, :, None]) / (
        model.first.target_running_var[None, :, None] + model.first.eps
    ).sqrt() + 5
    for channel in range(2):
        actual = expected[mask[:, channel], channel, :]
        torch.testing.assert_close(model.second.target_running_mean[channel], actual.mean())
        torch.testing.assert_close(model.second.target_running_var[channel], actual.var(unbiased=False))
    assert receipt["first"]["execution_order"] == 0
    assert receipt["second"]["execution_order"] == 1
    assert receipt["second"]["execution_order_paths"] == ["first", "second"]
    source_chart = 3 * (values - 11) / (7 + model.first.eps) ** .5 + 5
    assert not torch.allclose(model.second.target_running_mean, source_chart.mean((0, 2)))


def test_partial_initialization_rejects_without_new_target_state():
    values, mask = _values()
    model = _Chain()
    model.first.install_target_calibration_buffer(torch.zeros(2), torch.ones(2), frozen=True)
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="partially initialized"):
        install_splitbn_target_calibration_buffers(model, values.float(), device="cpu", channel_masks=mask)
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])


def test_receipt_json_roundtrip_binds_scope_mask_and_target_statistics():
    values, mask = _values()
    model = _Chain().double()
    scope = {"phase": "initial_fit", "training_mask_sha256": "a" * 64}
    record = install_splitbn_target_calibration_buffers(
        model, values, device="cpu", channel_masks=mask, scope_record=scope)
    replay = json.loads(json.dumps(record, allow_nan=False))
    assert replay == record
    assert replay["first"]["scope"] == scope
    assert replay["first"]["channel_mask_sha256"] == replay["second"]["channel_mask_sha256"]
    assert len(replay["first"]["mean_digest"]) == 64
    assert replay["first"]["estimand"] == "sequential_target_calibration"
    copied = _Chain().double()
    copied.load_state_dict(model.state_dict())
    validate_splitbn_target_install_receipt(copied, replay, expected_scope=scope)
    with torch.no_grad():
        copied.second.target_running_mean.add_(1)
    with pytest.raises(ValueError, match="target buffer differs"):
        validate_splitbn_target_install_receipt(copied, replay, expected_scope=scope)


@pytest.mark.parametrize("wrapped", [False, True])
def test_actual_observed_core_packs_temporal_and_keeps_mixed_features_complete(wrapped):
    torch.manual_seed(503)
    core = N2P3Net(n_channels=3, n_times=64, temporal_filters=3, dropout=0,
                   observed_channel_norm=True, splitbn_domain_count=2,
                   splitbn_target_buffer_source="target_calibration").double()
    model = FeatureDomainTransfer(core, config=FeatureTransferConfig()).double() if wrapped else core
    reference = deepcopy(model)
    x = torch.randn(4, 3, 64, dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [True, True, True],
                         [False, True, False], [True, False, True]])
    missing = torch.where(mask[:, :, None], x, float("nan"))
    records = install_splitbn_target_calibration_buffers(model, missing, device="cpu", channel_masks=mask)
    expected = install_splitbn_target_calibration_buffers(reference, x, device="cpu", channel_masks=mask)
    assert records == expected
    path = "core." if wrapped else ""
    assert records[path + "st_temporal_norm"]["observation_counts"] == [int(mask.sum()) * 64] * 3
    assert records[path + "st_spatial_norm"]["observation_count"] == len(x) * 64
    assert records[path + "mst_branches.0.norm"]["observation_count"] == len(x) * (64 // core.st_pool_size)
    assert records[path + "st_temporal_norm"]["observation_support"] == "packed_measured_electrode_traces"
    for name, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
    restored = deepcopy(model)
    restored.load_state_dict(model.state_dict())
    validate_splitbn_target_install_receipt(restored, json.loads(json.dumps(records)))
    model.eval()
    restored.eval()
    assert torch.equal(model.forward_features(missing, channel_mask=mask, eval_buffer_source="target_calibration"),
                       restored.forward_features(missing, channel_mask=mask, eval_buffer_source="target_calibration"))


def test_actual_mask_transport_is_required_for_core_and_wrapper():
    core = N2P3Net(n_channels=3, n_times=64, dropout=0,
        observed_channel_norm=True, splitbn_domain_count=2,
        splitbn_target_buffer_source="target_calibration")
    for model in (core, FeatureDomainTransfer(core, config=FeatureTransferConfig())):
        with pytest.raises(ValueError, match="require channel_masks"):
            install_splitbn_target_calibration_buffers(model, torch.ones(4, 3, 64), device="cpu")


def test_execution_failure_rolls_back_partial_install_and_restores_each_module_mode():
    class Failure(_Chain):
        def forward_features(self, x, *, eval_buffer_source=None, channel_mask=None):
            x = self.first(x, eval_buffer_source=eval_buffer_source)
            raise RuntimeError("failure after installed first site")

    model = Failure().double().train()
    model.first.eval()
    before = deepcopy(model.state_dict())
    values, mask = _values()
    with pytest.raises(RuntimeError, match="after installed first"):
        install_splitbn_target_calibration_buffers(model, values, device="cpu", channel_masks=mask)
    assert model.training and not model.first.training and model.second.training
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name


def test_real_observed_population_is_invariant_to_batch_and_trial_order():
    torch.manual_seed(304)
    core = N2P3Net(3, n_times=64, dropout=0, observed_channel_norm=True,
        splitbn_domain_count=2, splitbn_target_buffer_source="target_calibration").double()
    reference = deepcopy(core)
    x = torch.randn(5, 3, 64, dtype=torch.float64)
    mask = torch.tensor([[True, False, False], [False, True, False], [True, True, False],
                         [True, False, True], [False, True, True]])
    order = torch.tensor([3, 1, 4, 0, 2])
    first = install_splitbn_target_calibration_buffers(core, x, device="cpu", batch_size=2, channel_masks=mask)
    second = install_splitbn_target_calibration_buffers(reference, x[order], device="cpu", batch_size=3,
                                                      channel_masks=mask[order])
    for path in first:
        assert first[path]["observation_counts"] == second[path]["observation_counts"]
        actual, expected = dict(core.named_modules())[path], dict(reference.named_modules())[path]
        torch.testing.assert_close(actual.target_running_mean, expected.target_running_mean, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.target_running_var, expected.target_running_var, atol=1e-12, rtol=1e-12)


def test_reused_site_is_rejected_instead_of_certifying_one_of_two_populations():
    class Reused(_Chain):
        def forward_features(self, x, *, eval_buffer_source=None, channel_mask=None):
            x = self.first(x, eval_buffer_source=eval_buffer_source)
            x = self.first(x + 7, eval_buffer_source=eval_buffer_source)
            return self.second(x, eval_buffer_source=eval_buffer_source)

    model = Reused().double()
    before = deepcopy(model.state_dict())
    values, mask = _values()
    with pytest.raises(RuntimeError, match="exactly once"):
        install_splitbn_target_calibration_buffers(model, values, device="cpu", channel_masks=mask)
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name]), name
