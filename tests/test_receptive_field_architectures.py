from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.n2p3net import (
    RF_MECHANISM_ARCHITECTURES,
    N2P3ArchitectureConfig,
    N2P3Net,
    scale_architecture_preserving_spans,
    stacked_temporal_receptive_field_samples,
    temporal_receptive_span_ms,
)

EXPECTED_RECEPTIVE_FIELDS = {
    "A": (84, 132),
    "B": (54, 102),
    "C": (52, 100),
    "D": (84, 132),
    "E": (84, 132),
}


def _receptive_fields(architecture: N2P3ArchitectureConfig) -> tuple[int, ...]:
    return tuple(
        stacked_temporal_receptive_field_samples(
            architecture.temporal_kernel_size,
            architecture.st_pool_size,
            kernel,
            temporal_dilation=architecture.st_temporal_dilation,
            branch_dilation=dilation,
        )
        for kernel, dilation in zip(
            architecture.mst_kernel_sizes,
            architecture.mst_dilations,
            strict=True,
        )
    )


def _model(architecture: N2P3ArchitectureConfig) -> N2P3Net:
    return N2P3Net(
        n_channels=16,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="full_unfold",
        **architecture.model_kwargs(),
    )


def test_receptive_field_mechanism_arms_are_fully_auditable() -> None:
    assert tuple(RF_MECHANISM_ARCHITECTURES) == ("A", "B", "C", "D", "E")
    assert {
        name: _receptive_fields(architecture)
        for name, architecture in RF_MECHANISM_ARCHITECTURES.items()
    } == EXPECTED_RECEPTIVE_FIELDS

    assert RF_MECHANISM_ARCHITECTURES["A"].temporal_kernel_size == 65
    assert RF_MECHANISM_ARCHITECTURES["B"].temporal_kernel_size == 35
    assert RF_MECHANISM_ARCHITECTURES["C"].temporal_kernel_size == 33
    assert RF_MECHANISM_ARCHITECTURES["D"].st_temporal_dilation == 2
    assert RF_MECHANISM_ARCHITECTURES["E"].mst_kernel_sizes == (13, 25)
    assert all(
        architecture.mst_dilations == (1, 1) for architecture in RF_MECHANISM_ARCHITECTURES.values()
    )


@pytest.mark.parametrize("name", tuple(EXPECTED_RECEPTIVE_FIELDS))
def test_receptive_field_mechanism_arms_preserve_time_shape(name: str) -> None:
    architecture = RF_MECHANISM_ARCHITECTURES[name]
    model = _model(architecture)

    assert model.st_temporal.dilation == (1, architecture.st_temporal_dilation)
    assert model.st_temporal.padding == (
        0,
        architecture.st_temporal_dilation * (architecture.temporal_kernel_size - 1) // 2,
    )
    assert [branch.depthwise.dilation for branch in model.mst_branches] == [
        (dilation,) for dilation in architecture.mst_dilations
    ]
    assert [branch.depthwise.padding for branch in model.mst_branches] == [
        (dilation * (kernel - 1) // 2,)
        for kernel, dilation in zip(
            architecture.mst_kernel_sizes,
            architecture.mst_dilations,
            strict=True,
        )
    ]
    assert model.forward_features(torch.randn(2, 16, 128)).shape == (2, 4, 32)
    assert model(torch.randn(2, 16, 128)).shape == (2, 2)

    record = model.architecture_record()
    assert record["st_temporal_dilation"] == architecture.st_temporal_dilation
    assert record["mst_dilations"] == list(architecture.mst_dilations)
    assert record["mst_total_receptive_field_samples"] == list(EXPECTED_RECEPTIVE_FIELDS[name])


def test_receptive_field_controls_match_parameter_budgets() -> None:
    models = {
        name: _model(architecture) for name, architecture in RF_MECHANISM_ARCHITECTURES.items()
    }

    assert models["A"].parameter_count() == models["E"].parameter_count() == 1_506
    assert models["C"].parameter_count() == models["D"].parameter_count()
    provisional_default = N2P3Net(n_channels=16, n_times=128, pooling_mode="ms_flatten")
    assert provisional_default.parameter_count() == 1_042


def test_nondefault_mst_dilations_propagate_without_changing_time_shape() -> None:
    architecture = replace(
        RF_MECHANISM_ARCHITECTURES["C"],
        mst_dilations=(2, 3),
    )
    model = _model(architecture)

    assert model.forward_features(torch.randn(2, 16, 128)).shape == (2, 4, 32)
    assert model.architecture_record()["mst_total_receptive_field_samples"] == [68, 228]


def test_dilated_st_kernel_scaling_preserves_effective_span() -> None:
    source = RF_MECHANISM_ARCHITECTURES["D"]
    scaled = scale_architecture_preserving_spans(
        source,
        source_sample_rate_hz=128.0,
        target_sample_rate_hz=256.0,
    )

    assert scaled.temporal_kernel_size == 65
    assert scaled.st_temporal_dilation == 2
    assert scaled.mst_kernel_sizes == (9, 33)
    assert scaled.mst_dilations == (1, 1)
    assert temporal_receptive_span_ms(
        source.temporal_kernel_size,
        128.0,
        source.st_temporal_dilation,
    ) == pytest.approx(500.0)
    assert temporal_receptive_span_ms(
        scaled.temporal_kernel_size,
        256.0,
        scaled.st_temporal_dilation,
    ) == pytest.approx(500.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"st_temporal_dilation": 0},
        {"mst_dilations": (1,)},
        {"mst_dilations": (1, 0)},
    ],
)
def test_architecture_rejects_invalid_dilation_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="dilation"):
        N2P3ArchitectureConfig(**kwargs)
