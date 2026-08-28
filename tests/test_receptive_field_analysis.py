from __future__ import annotations

import numpy as np
import torch

from experiments.analyze_receptive_field import (
    ArchitectureSpec,
    branch_dependencies,
    fft_resampling_impulse_diagnostic,
    summarize_architecture,
)
from models.n2p3net import N2P3Net


def _summary(k0: int):
    return summarize_architecture(
        ArchitectureSpec("test", k0, 4, (5, 17), 8),
        n_times=128,
        sample_rate_hz=128.0,
        tmin_ms=-200.0,
    )[0]


def test_finite_support_matches_k65_closed_form_and_edge_clipping() -> None:
    short = branch_dependencies(
        128,
        temporal_kernel=65,
        pool_size=4,
        branch_kernel=5,
    )
    long = branch_dependencies(
        128,
        temporal_kernel=65,
        pool_size=4,
        branch_kernel=17,
    )

    assert np.flatnonzero(short[15]).tolist() == list(range(20, 104))
    assert np.flatnonzero(long[15]).tolist() == list(range(128))
    assert short.sum(axis=1).max() == 84
    assert long.sum(axis=1).max() == 128


def test_k35_restores_locality_but_long_branch_remains_broad() -> None:
    short = branch_dependencies(
        128,
        temporal_kernel=35,
        pool_size=4,
        branch_kernel=5,
    )
    long = branch_dependencies(
        128,
        temporal_kernel=35,
        pool_size=4,
        branch_kernel=17,
    )

    assert np.flatnonzero(short[15]).tolist() == list(range(35, 89))
    assert np.flatnonzero(long[15]).tolist() == list(range(11, 113))
    assert short.sum(axis=1).max() == 54
    assert long.sum(axis=1).max() == 102


def test_lmbc_named_reference_contains_poststimulus_cached_samples() -> None:
    k65 = _summary(65)["branches"]
    k35 = _summary(35)["branches"]

    assert k65["5"]["lmbc"]["reference"]["end_ms"] == 323.4375
    assert k65["17"]["lmbc"]["reference"]["end_ms"] == 510.9375
    assert k35["5"]["lmbc"]["reference"]["end_ms"] == 206.25
    assert k35["17"]["lmbc"]["reference"]["end_ms"] == 393.75
    assert all(
        item["contrast"]["whole_cached_epoch"]
        for item in k65["5"]["lmbc"]["candidates"]
    )
    assert all(
        item["contrast"]["whole_cached_epoch"]
        for item in k35["17"]["lmbc"]["candidates"]
    )


def test_ms_flatten_bins_overlap_in_raw_cached_time() -> None:
    summary = _summary(65)["branches"]
    short_bins = summary["5"]["ms_flatten_supports"]
    long_bins = summary["17"]["ms_flatten_supports"]

    assert [(item["start_sample"], item["end_sample"]) for item in short_bins] == [
        (0, 71),
        (0, 103),
        (24, 127),
        (56, 127),
    ]
    assert long_bins[1]["whole_cached_epoch"] is True
    assert long_bins[2]["whole_cached_epoch"] is True


def test_lmbc_logit_responds_to_raw_point_outside_all_named_windows() -> None:
    model = N2P3Net(
        1,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="latency_marginal_contrast",
        temporal_kernel_size=35,
        temporal_filters=1,
        spatial_depth_multiplier=1,
        mst_features_per_scale=1,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                module.weight.fill_(1.0)
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.weight.fill_(1.0)
                module.bias.zero_()
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
        model.classifier.weight[0].zero_()
        model.classifier.weight[1].fill_(1.0)
        model.classifier.bias.zero_()

    x = torch.zeros(1, 1, 128, requires_grad=True)
    logits = model(x)
    gradient = torch.autograd.grad(logits[0, 1] - logits[0, 0], x)[0][0, 0]

    # Sample 34 is +65.625 ms: outside [-200,0) and every declared
    # candidate window (the earliest begins at +150 ms).
    assert gradient[34].abs() > 0.0
    assert int(torch.count_nonzero(gradient)) == 128


def test_executable_fft_resampling_has_global_epoch_support() -> None:
    diagnostic = fft_resampling_impulse_diagnostic()

    assert diagnostic["source_samples"] == 513
    assert diagnostic["output_samples"] == 128
    assert diagnostic["nonzero_output_samples"] == 128
    assert diagnostic["output_samples_above_1e_6"] == 128
