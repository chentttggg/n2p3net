from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.n2p3net import LatencyMarginalContrastPool, MSFlattenPool, N2P3Net
from train import factory
from train.runtime import GpuPerformanceScheduler


def test_n2p3net_returns_binary_logits_and_has_compact_capacity() -> None:
    model = N2P3Net(n_channels=3, n_times=256, sfreq=256.0, tmin_s=-0.2)
    logits = model(torch.randn(4, 3, 256))

    assert logits.shape == (4, 2)
    assert model.parameter_count() == 1_054
    assert isinstance(model.pool, nn.ModuleList)
    assert len(model.pool) == 2
    assert all(isinstance(pool, LatencyMarginalContrastPool) for pool in model.pool)


def test_n2p3net_matches_the_baseline_ms_eegnet_parameterization() -> None:
    """Paper baseline: 8 temporal filters, D=2, MST kernels 5/17, K=2 per scale."""
    dataset_1_shape = N2P3Net(
        n_channels=8,
        n_times=140,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
    )
    dataset_23_shape = N2P3Net(
        n_channels=12,
        n_times=113,
        sfreq=125.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
    )

    assert dataset_1_shape.st_temporal.out_channels == 8
    assert dataset_1_shape.st_temporal.kernel_size == (1, 65)
    assert dataset_1_shape.st_spatial.groups == 8
    assert dataset_1_shape.st_spatial.out_channels == 16
    assert [branch.depthwise.kernel_size for branch in dataset_1_shape.mst_branches] == [
        (5,),
        (17,),
    ]
    assert [branch.pointwise.out_channels for branch in dataset_1_shape.mst_branches] == [2, 2]
    assert dataset_1_shape.parameter_count() == 1_154
    assert dataset_23_shape.parameter_count() == 1_210


def test_spatial_projection_is_frequency_conditioned_and_max_norm_bounded() -> None:
    model = N2P3Net(n_channels=3, n_times=256, sfreq=256.0, tmin_s=-0.2)
    with torch.no_grad():
        model.st_spatial.weight.fill_(10.0)

    effective = model.st_spatial.effective_weight().flatten(start_dim=1)
    norms = torch.linalg.vector_norm(effective, ord=2, dim=1)

    assert model.st_spatial.groups == model.temporal_filters
    assert torch.all(norms <= 1.0 + 1e-6)


def test_ms_flatten_preserves_position_that_global_mean_discards() -> None:
    """Counterexample: equal global energy at early/late bins has distinct MS features."""
    pool = MSFlattenPool(pool_size=8)
    early = torch.zeros(1, 1, 32)
    late = torch.zeros(1, 1, 32)
    early[:, :, 2] = 1.0
    late[:, :, 26] = 1.0

    assert torch.equal(early.mean(dim=2), late.mean(dim=2))
    assert not torch.equal(pool(early), pool(late))


def test_ms_flatten_requires_a_declared_physical_epoch_width() -> None:
    with pytest.raises(ValueError, match="requires n_times"):
        N2P3Net(n_channels=3, pooling_mode="ms_flatten")


def test_n2p3net_baseline_fits_with_subject_disjoint_validation() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 3, 64)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    X[y == 1, 2, 28:42] += 1.5
    subjects = np.repeat(np.arange(6), 4)
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=DeepConfig(epochs=1, batch_size=8, val_subject_frac=0.25, val_subjects_min=1),
        device=torch.device("cpu"),
    ).fit(X, y, subject_ids=subjects)

    assert baseline.predict_logit(X).shape == (len(X),)
    assert baseline.calibration_source_ == "subject_disjoint_validation"


def test_n2p3net_baseline_accepts_the_shared_runtime() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    baseline = N2P3NetBaseline(3, 64, 128.0, device=torch.device("cpu"), runtime=runtime)

    assert baseline.runtime is runtime


def _pool(*, channels: int = 3) -> LatencyMarginalContrastPool:
    return LatencyMarginalContrastPool(
        channels,
        n_times=120,
        sfreq=100.0,
        tmin_s=-0.2,
        evidence_window_ms=(250.0, 450.0),
        reference_window_ms=(-200.0, 0.0),
        latency_offsets_ms=(-100.0, 0.0, 100.0),
        temperature=0.25,
    )


def test_latency_pool_is_invariant_to_feature_level_common_offset() -> None:
    """A constant encoded background cancels against the reference mean exactly."""
    pool = _pool()
    features = torch.randn(2, 3, 120)
    common_offset = torch.tensor([2.0, -3.0, 0.5]).view(1, 3, 1)

    pooled = pool(features)
    shifted = pool(features + common_offset)

    torch.testing.assert_close(pooled, shifted)


def test_latency_pool_ignores_artifact_outside_reference_and_candidate_support() -> None:
    """Counterexample: a huge off-window peak cannot enter the LMBC summary."""
    pool = _pool()
    support = (pool.candidate_weights.sum(dim=0) > 0.0) | (pool.reference_weights > 0.0)
    outside_indices = torch.nonzero(~support).flatten()
    assert len(outside_indices) > 0
    outside = int(outside_indices[0])
    clean = torch.zeros(1, 3, 120)
    contaminated = clean.clone()
    contaminated[:, :, outside] = 1_000_000.0

    torch.testing.assert_close(pool(clean), pool(contaminated))


def test_latency_pool_softly_selects_the_matching_latency_window() -> None:
    pool = _pool(channels=1)
    target_index = int(torch.nonzero(pool.candidate_offsets_ms == 100.0).item())
    features = torch.zeros(1, 1, 120)
    features[:, :, pool.candidate_weights[target_index].bool()] = 1.0
    with torch.no_grad():
        pool.latency_query.fill_(1.0)

    _, attention = pool(features, return_attention=True)

    assert int(attention.argmax(dim=1).item()) == target_index


def test_latency_pool_fails_closed_without_pre_stimulus_reference() -> None:
    with pytest.raises(ValueError, match="pre-stimulus reference"):
        LatencyMarginalContrastPool(
            1,
            n_times=100,
            sfreq=100.0,
            tmin_s=0.0,
        )


def test_factory_propagates_dataset_physical_time_to_n2p3net(monkeypatch) -> None:
    """The factory must not discard tmin before constructing the model branch."""
    dataset = SimpleNamespace(
        preprocessing=SimpleNamespace(sfreq=128.0, tmin_ms=-150.0),
        n_channels=3,
        n_times=128,
        channel_mask=np.ones(3, dtype=bool),
    )
    monkeypatch.setattr(factory, "_validate_binary_dataset", lambda _: None)

    model = factory.build_binary_model(
        "n2p3net",
        dataset,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert model.tmin_s == -0.15
    assert model._make_model().tmin_s == -0.15
    architecture = factory.describe_binary_model("n2p3net", model)["architecture"]
    assert architecture["tmin_s"] == -0.15
    assert architecture["trunk"] == "ms_eegnet_style"
    assert architecture["mst_kernel_samples"] == [5, 17]


def test_factory_exposes_global_average_only_as_an_explicit_ablation(monkeypatch) -> None:
    dataset = SimpleNamespace(
        preprocessing=SimpleNamespace(sfreq=128.0, tmin_ms=-150.0),
        n_channels=3,
        n_times=128,
        channel_mask=np.ones(3, dtype=bool),
    )
    monkeypatch.setattr(factory, "_validate_binary_dataset", lambda _: None)

    model = factory.build_binary_model(
        "n2p3net",
        dataset,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
        n2p3net_pooling_mode="global_average",
    )

    assert model.pooling_mode == "global_average"
    architecture = factory.describe_binary_model("n2p3net", model)["architecture"]
    assert architecture["pooling_mode"] == "global_average"
    assert architecture["tmin_s"] == -0.15
    assert architecture["trunk"] == "ms_eegnet_style"
    assert "evidence_window_ms" not in architecture
