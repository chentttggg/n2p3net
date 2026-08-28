from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.n2p3net import (
    FullResolutionUnfold,
    LatencyMarginalContrastPool,
    MSFlattenPool,
    N2P3ArchitectureConfig,
    N2P3Net,
    ResidualFactorizedQuadraticClassifier,
    ResidualMLPClassifier,
    scale_architecture_preserving_spans,
    scale_odd_kernel_preserving_span,
    temporal_receptive_span_ms,
)
from train import factory
from train.runtime import GpuPerformanceScheduler


def test_n2p3net_returns_binary_logits_and_has_compact_capacity() -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=256,
        sfreq=256.0,
        tmin_s=-0.2,
        pooling_mode="latency_marginal_contrast",
    )
    logits = model(torch.randn(4, 3, 256))

    assert logits.shape == (4, 2)
    assert model.parameter_count() == 1_054
    assert isinstance(model.pool, nn.ModuleList)
    assert len(model.pool) == 2
    assert all(isinstance(pool, LatencyMarginalContrastPool) for pool in model.pool)


def test_n2p3net_default_is_the_promoted_ms_flatten_head() -> None:
    model = N2P3Net(n_channels=3, n_times=128, sfreq=128.0, tmin_s=-0.2)

    assert model.pooling_mode == "ms_flatten"
    assert isinstance(model.pool, MSFlattenPool)


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


def test_sampling_rate_scaling_preserves_endpoint_receptive_span() -> None:
    """Counterexample: 127 samples is not the exact 256 Hz match for 65 at 128 Hz."""

    assert temporal_receptive_span_ms(65, 128.0) == pytest.approx(500.0)
    assert scale_odd_kernel_preserving_span(
        65,
        source_sample_rate_hz=128.0,
        target_sample_rate_hz=256.0,
    ) == 129
    assert temporal_receptive_span_ms(127, 256.0) == pytest.approx(492.1875)
    assert temporal_receptive_span_ms(129, 256.0) == pytest.approx(500.0)
    assert scale_odd_kernel_preserving_span(
        5,
        source_sample_rate_hz=32.0,
        target_sample_rate_hz=64.0,
    ) == 9
    assert scale_odd_kernel_preserving_span(
        17,
        source_sample_rate_hz=32.0,
        target_sample_rate_hz=64.0,
    ) == 33


def test_architecture_scaling_preserves_both_temporal_feature_rates() -> None:
    architecture = scale_architecture_preserving_spans(
        N2P3ArchitectureConfig(),
        source_sample_rate_hz=128.0,
        target_sample_rate_hz=256.0,
    )

    assert architecture.temporal_kernel_size == 129
    assert architecture.mst_kernel_sizes == (9, 33)


def test_architecture_record_exposes_physical_receptive_spans() -> None:
    record = N2P3Net(
        n_channels=8,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
    ).architecture_record()

    assert record["input_sample_rate_hz"] == 128.0
    assert record["feature_sample_rate_hz"] == 32.0
    assert record["st_temporal_receptive_span_ms"] == pytest.approx(500.0)
    assert record["mst_receptive_span_ms"] == pytest.approx([125.0, 500.0])


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


def test_full_unfold_resolves_within_pool_collisions() -> None:
    """Counterexample: fixed average pooling aliases positions inside one bin."""

    pooled = MSFlattenPool(pool_size=8)
    unfolded = FullResolutionUnfold()
    left = torch.zeros(1, 1, 32)
    right = torch.zeros(1, 1, 32)
    left[:, :, 2] = 1.0
    right[:, :, 5] = 1.0

    torch.testing.assert_close(pooled(left), pooled(right))
    assert not torch.equal(unfolded(left), unfolded(right))
    delta = unfolded(left) - unfolded(right)
    separation = (delta * unfolded(left)).sum() - (delta * unfolded(right)).sum()
    torch.testing.assert_close(separation, delta.square().sum())
    assert float(separation) > 0.0


@pytest.mark.parametrize(
    "head",
    [
        ResidualFactorizedQuadraticClassifier(features=12, outputs=2, rank=4),
        ResidualMLPClassifier(features=12, outputs=2, hidden_features=8),
    ],
)
def test_residual_nonlinear_classifiers_start_as_exact_linear_unfold(head: nn.Module) -> None:
    x = torch.randn(5, 12, requires_grad=True)

    torch.testing.assert_close(head(x), head.linear(x))
    head(x).sum().backward()

    assert head.linear.weight.grad is not None
    residual_output = (
        head.quadratic_output
        if isinstance(head, ResidualFactorizedQuadraticClassifier)
        else head.nonlinear_output
    )
    assert residual_output.weight.grad is not None


def test_factorized_quadratic_resolves_linear_xnor_counterexample() -> None:
    """Four XNOR corners cannot be separated by an affine readout."""

    head = ResidualFactorizedQuadraticClassifier(features=2, outputs=2, rank=1)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.zero_()
        head.left.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.right.weight.copy_(torch.tensor([[0.0, 1.0]]))
        head.quadratic_output.weight.copy_(torch.tensor([[0.0], [1.0]]))
    x = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])

    predicted = head(x).argmax(dim=1)

    torch.testing.assert_close(predicted, torch.tensor([1, 0, 0, 1]))


@pytest.mark.parametrize(
    "pooling_mode",
    ["full_unfold", "mlp_full_unfold", "quadratic_full_unfold"],
)
def test_prior_free_unfold_models_execute_forward_backward(pooling_mode: str) -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode=pooling_mode,
    )
    x = torch.randn(4, 3, 128)
    loss = model(x).square().mean()

    loss.backward()
    record = model.architecture_record()

    assert record["unfold_time_samples"] == 32
    assert record["unfold_features"] == 128
    assert record["classifier_features"] == 128
    assert model.parameter_count() > 0
    if pooling_mode == "quadratic_full_unfold":
        assert record["interaction_rank"] == 8
        assert record["classifier"] == "residual_factorized_quadratic"
    elif pooling_mode == "mlp_full_unfold":
        assert record["mlp_hidden_features"] == 16
        assert record["classifier"] == "residual_mlp"
    else:
        assert record["classifier"] == "linear_full_resolution"


def test_unfold_architecture_record_uses_instance_geometry() -> None:
    model = N2P3Net(
        n_channels=3,
        n_times=120,
        st_pool_size=5,
        mst_kernel_sizes=(3, 7, 11),
        mst_features_per_scale=3,
        pooling_mode="full_unfold",
    )

    record = model.architecture_record()

    assert record["unfold_time_samples"] == 24
    assert record["unfold_features"] == 216
    assert record["classifier_features"] == 216
    assert record["feature_sample_rate_hz"] == pytest.approx(25.6)


def test_architecture_config_controls_the_complete_baseline_model() -> None:
    architecture = N2P3ArchitectureConfig(
        temporal_filters=6,
        temporal_kernel_size=53,
        spatial_depth_multiplier=3,
        st_pool_size=5,
        mst_kernel_sizes=(3, 13, 19),
        mst_features_per_scale=3,
        mst_pool_size=7,
        dropout=0.2,
        spatial_max_norm=0.8,
        interaction_rank=6,
        mlp_hidden_features=12,
    )
    baseline = N2P3NetBaseline(
        3,
        120,
        128.0,
        device=torch.device("cpu"),
        pooling_mode="full_unfold",
        architecture=architecture,
    )

    model = baseline._make_model()
    record = baseline.architecture_record()

    assert model.temporal_filters == 6
    assert model.temporal_kernel_size == 53
    assert model.spatial_depth_multiplier == 3
    assert model.st_pool_size == 5
    assert model.mst_kernel_sizes == (3, 13, 19)
    assert model.mst_features_per_scale == 3
    assert model.mst_pool_size == 7
    assert model.dropout_probability == pytest.approx(0.2)
    assert model.spatial_max_norm == pytest.approx(0.8)
    assert model.interaction_rank == 6
    assert model.mlp_hidden_features == 12
    assert record["unfold_time_samples"] == 24
    assert record["unfold_features"] == 216
    assert record["dropout"] == pytest.approx(0.2)
    assert record["spatial_max_norm"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temporal_kernel_size": 64}, "odd integer"),
        ({"mst_kernel_sizes": (5, 16)}, "odd kernels"),
        ({"dropout": 1.0}, "dropout"),
        ({"spatial_max_norm": 0.0}, "spatial_max_norm"),
    ],
)
def test_architecture_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        N2P3ArchitectureConfig(**kwargs)


def test_ms_flatten_requires_a_declared_physical_epoch_width() -> None:
    with pytest.raises(ValueError, match="requires n_times"):
        N2P3Net(n_channels=3, pooling_mode="ms_flatten")


def test_n2p3net_baseline_fits_with_group_disjoint_validation() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 3, 64)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    X[y == 1, 2, 28:42] += 1.5
    subjects = np.repeat(np.arange(6), 4)
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=DeepConfig(epochs=1, batch_size=8, val_group_frac=0.25, val_groups_min=1),
        device=torch.device("cpu"),
    ).fit(X, y, group_ids=subjects)

    assert baseline.predict_logit(X).shape == (len(X),)
    assert baseline.calibration_source_ == "group_disjoint_validation"


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
    assert architecture["pooling_mode"] == "ms_flatten"


def test_factory_propagates_the_n2p3_architecture_contract(monkeypatch) -> None:
    dataset = SimpleNamespace(
        preprocessing=SimpleNamespace(sfreq=128.0, tmin_ms=-200.0),
        n_channels=3,
        n_times=128,
        channel_mask=np.ones(3, dtype=bool),
    )
    architecture = N2P3ArchitectureConfig(
        temporal_filters=7,
        temporal_kernel_size=59,
        mst_kernel_sizes=(5, 15),
        mst_pool_size=9,
        dropout=0.3,
    )
    monkeypatch.setattr(factory, "_validate_binary_dataset", lambda _: None)

    model = factory.build_binary_model(
        "n2p3net_full_unfold",
        dataset,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
        n2p3net_architecture=architecture,
    )
    record = factory.describe_binary_model("n2p3net_full_unfold", model)["architecture"]

    assert model.architecture is architecture
    assert record["st_temporal_filters"] == 7
    assert record["st_temporal_kernel_samples"] == 59
    assert record["mst_kernel_samples"] == [5, 15]
    assert record["mst_pool_size"] == 9
    assert record["dropout"] == pytest.approx(0.3)


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


@pytest.mark.parametrize(
    ("model_name", "pooling_mode"),
    [
        ("n2p3net_lmbc", "latency_marginal_contrast"),
        ("n2p3net_global_average", "global_average"),
        ("ms_eegnet", "ms_flatten"),
        ("n2p3net_full_unfold", "full_unfold"),
        ("n2p3net_mlp_full_unfold", "mlp_full_unfold"),
        ("n2p3net_quadratic_full_unfold", "quadratic_full_unfold"),
    ],
)
def test_factory_named_ablation_models_lock_one_pooling_mode(
    monkeypatch, model_name: str, pooling_mode: str
) -> None:
    dataset = SimpleNamespace(
        preprocessing=SimpleNamespace(sfreq=128.0, tmin_ms=-200.0),
        n_channels=3,
        n_times=128,
        channel_mask=np.ones(3, dtype=bool),
    )
    monkeypatch.setattr(factory, "_validate_binary_dataset", lambda _: None)

    model = factory.build_binary_model(
        model_name,
        dataset,
        epochs=1,
        batch_size=4,
        device=torch.device("cpu"),
        n2p3net_pooling_mode="global_average",
    )
    record = factory.describe_binary_model(model_name, model)

    assert model.pooling_mode == pooling_mode
    assert record["name"] == model_name
    assert record["architecture"]["pooling_mode"] == pooling_mode
