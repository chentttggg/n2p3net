"""Analytic observed populations through complete tiny fit and refit callers."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from models.splitbn import (
    SplitBN1d,
    SplitBN2d,
    _target_statistic_digest,
    validate_splitbn_target_install_receipt,
)
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def _core():
    core = N2P3Net(
        n_channels=2, n_times=8, sfreq=128., tmin_s=-.2,
        temporal_filters=1, temporal_kernel_size=3, spatial_depth_multiplier=1,
        st_pool_size=1, mst_kernel_sizes=(3,), mst_dilations=(1,),
        mst_features_per_scale=1, mst_pool_size=1, dropout=0.,
        pooling_mode="global_average", observed_channel_norm=True,
        splitbn_domain_count=2, splitbn_target_buffer_source="target_calibration")
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
        core.st_temporal.weight[0, 0, 0, 1] = 1
        core.st_spatial.weight.fill_(.5)
        for module in core.modules():
            if isinstance(module, (SplitBN1d, SplitBN2d)):
                module.weight.fill_(1)
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.num_batches_tracked.fill_(1)
    return core


@pytest.mark.parametrize("all_observed", [False, True])
def test_real_fit_refit_preserve_analytic_observed_moments_and_saved_masks(all_observed):
    core = _core()
    adapter = SubjectAdapter(core, config=SubjectAdapterConfig(
        head_kind="full_fine", epochs=1, batch_size=4, val_group_fraction=None,
        input_statistics="source", input_mean=np.zeros(2), input_std=np.ones(2),
        as_mmd_lambda0=.1, as_mmd_warmup_epochs=1, refit_full_prefix=True), device=torch.device("cpu"))
    inputs = np.zeros((4, 2, 8), dtype=np.float32)
    inputs[:, 0] = np.array([-1., 1.] * 4)
    masks = np.ones((4, 2), dtype=bool)
    if not all_observed:
        masks[:, 1] = False
        inputs[~masks] = np.nan
    labels = np.array([0, 1, 0, 1])
    source = np.zeros((4, 2, 8), dtype=np.float32)
    source[:, 0] = np.array([-1., 1.] * 4)
    adapter.fit(inputs, labels, trial_channel_mask=masks,
        validation_mask=np.array([False, False, True, True]), prefix_mask=np.ones(4, dtype=bool),
        source_X=source, source_labels=labels, source_domain_indices=np.array([0, 0, 1, 1]))
    receipts = adapter.last_history["splitbn_install_receipts"]
    assert receipts["initial"] is not None and receipts["refit"] is not None
    expected_variance = .5 if all_observed else 1.
    observed_channels = 2 if all_observed else 1
    for phase, rows, expected_phase in (("initial", 2, "initial_fit"), ("refit", 4, "refit_full_prefix")):
        temporal = receipts[phase]["st_temporal_norm"]
        assert temporal["scope"]["phase"] == expected_phase
        assert temporal["scope"]["row_count"] == rows
        assert temporal["observation_count"] == rows * observed_channels * 8
        assert temporal["sample_count"] == rows
        assert temporal["variance_digest"] == _target_statistic_digest(torch.tensor([expected_variance]))
        assert temporal["mean_digest"] == _target_statistic_digest(torch.tensor([0.]))
        assert temporal["observation_support"] == "packed_measured_electrode_traces"
    torch.testing.assert_close(core.st_temporal_norm.target_running_var,
                               torch.tensor([expected_variance]), rtol=0, atol=0)
    validate_splitbn_target_install_receipt(core, receipts["refit"])
    assert adapter.last_history["fit_budget"]["updates"] == 1
    assert adapter.last_history["refit_budget"]["updates"] == 1

    stream = io.BytesIO()
    torch.save({"state": core.state_dict(), "receipt": json.loads(json.dumps(receipts["refit"]))}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = _core()
    restored.load_state_dict(saved["state"])
    validate_splitbn_target_install_receipt(restored, saved["receipt"])
    core.eval()
    restored.eval()
    tensor = torch.from_numpy(inputs)
    observed = torch.from_numpy(masks)
    expected = core.forward_features(tensor, channel_mask=observed, eval_buffer_source="target_calibration")
    actual = restored.forward_features(tensor, channel_mask=observed, eval_buffer_source="target_calibration")
    assert torch.equal(actual, expected)
    if not all_observed:
        changed = tensor.clone()
        changed[:, 1] = 1e12
        assert torch.equal(actual, restored.forward_features(changed, channel_mask=observed,
                                                             eval_buffer_source="target_calibration"))
