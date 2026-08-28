"""Compact multi-scale P300 decoder with auditable alternative readouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from data.contract import DEFAULT_P300_DATA_CONTRACT

POOLING_MODES = frozenset(
    {
        "full_unfold",
        "mlp_full_unfold",
        "quadratic_full_unfold",
        "global_average",
        "ms_flatten",
        "latency_marginal_contrast",
    }
)

DEFAULT_ST_TEMPORAL_FILTERS = 8
DEFAULT_ST_TEMPORAL_KERNEL_SIZE = 65
DEFAULT_ST_TEMPORAL_DILATION = 1
DEFAULT_SPATIAL_DEPTH_MULTIPLIER = 2
DEFAULT_ST_POOL_SIZE = 4
DEFAULT_MST_KERNEL_SIZES = (5, 17)
DEFAULT_MST_DILATION = 1
DEFAULT_MST_FEATURES_PER_SCALE = 2
DEFAULT_MST_POOL_SIZE = 8
DEFAULT_N2P3_DROPOUT = 0.25
DEFAULT_SPATIAL_MAX_NORM = 1.0
DEFAULT_INTERACTION_RANK = 8
DEFAULT_MLP_HIDDEN_FEATURES = 16


@dataclass(frozen=True)
class N2P3ArchitectureConfig:
    """Complete, serializable N2P3-Net structure contract."""

    temporal_filters: int = DEFAULT_ST_TEMPORAL_FILTERS
    temporal_kernel_size: int = DEFAULT_ST_TEMPORAL_KERNEL_SIZE
    st_temporal_dilation: int = DEFAULT_ST_TEMPORAL_DILATION
    spatial_depth_multiplier: int = DEFAULT_SPATIAL_DEPTH_MULTIPLIER
    st_pool_size: int = DEFAULT_ST_POOL_SIZE
    mst_kernel_sizes: tuple[int, ...] = DEFAULT_MST_KERNEL_SIZES
    mst_dilations: tuple[int, ...] | None = None
    mst_features_per_scale: int = DEFAULT_MST_FEATURES_PER_SCALE
    mst_pool_size: int = DEFAULT_MST_POOL_SIZE
    dropout: float = DEFAULT_N2P3_DROPOUT
    spatial_max_norm: float = DEFAULT_SPATIAL_MAX_NORM
    interaction_rank: int = DEFAULT_INTERACTION_RANK
    mlp_hidden_features: int = DEFAULT_MLP_HIDDEN_FEATURES

    def __post_init__(self) -> None:
        kernels = tuple(int(kernel) for kernel in self.mst_kernel_sizes)
        dilations = (
            tuple(DEFAULT_MST_DILATION for _ in kernels)
            if self.mst_dilations is None
            else tuple(int(dilation) for dilation in self.mst_dilations)
        )
        object.__setattr__(self, "mst_kernel_sizes", kernels)
        object.__setattr__(self, "mst_dilations", dilations)
        if self.temporal_filters < 1 or self.spatial_depth_multiplier < 1:
            raise ValueError("temporal_filters and spatial_depth_multiplier must be positive.")
        if self.temporal_kernel_size < 3 or self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be an odd integer of at least three.")
        if self.st_temporal_dilation < 1:
            raise ValueError("st_temporal_dilation must be positive.")
        if not kernels or any(kernel < 3 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("mst_kernel_sizes must contain odd kernels of at least three samples.")
        if len(dilations) != len(kernels) or any(dilation < 1 for dilation in dilations):
            raise ValueError(
                "mst_dilations must contain one positive dilation per MST kernel."
            )
        if self.mst_features_per_scale < 1 or self.st_pool_size < 1 or self.mst_pool_size < 1:
            raise ValueError("feature and pool sizes must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.spatial_max_norm <= 0.0:
            raise ValueError("spatial_max_norm must be positive.")
        if self.interaction_rank < 1 or self.mlp_hidden_features < 1:
            raise ValueError("interaction_rank and mlp_hidden_features must be positive.")

    def model_kwargs(self) -> dict[str, object]:
        return {
            "temporal_filters": self.temporal_filters,
            "temporal_kernel_size": self.temporal_kernel_size,
            "st_temporal_dilation": self.st_temporal_dilation,
            "spatial_depth_multiplier": self.spatial_depth_multiplier,
            "st_pool_size": self.st_pool_size,
            "mst_kernel_sizes": self.mst_kernel_sizes,
            "mst_dilations": self.mst_dilations,
            "mst_features_per_scale": self.mst_features_per_scale,
            "mst_pool_size": self.mst_pool_size,
            "dropout": self.dropout,
            "spatial_max_norm": self.spatial_max_norm,
            "interaction_rank": self.interaction_rank,
            "mlp_hidden_features": self.mlp_hidden_features,
        }


DEFAULT_N2P3_ARCHITECTURE = N2P3ArchitectureConfig()
TUNED_FULL_UNFOLD_SOURCE_SAMPLE_RATE_HZ = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz
TUNED_FULL_UNFOLD_ARCHITECTURE = replace(
    DEFAULT_N2P3_ARCHITECTURE,
    temporal_kernel_size=35,
)
RF_MECHANISM_ARCHITECTURES: dict[str, N2P3ArchitectureConfig] = {
    "A": DEFAULT_N2P3_ARCHITECTURE,
    "B": TUNED_FULL_UNFOLD_ARCHITECTURE,
    "C": replace(DEFAULT_N2P3_ARCHITECTURE, temporal_kernel_size=33),
    "D": replace(
        DEFAULT_N2P3_ARCHITECTURE,
        temporal_kernel_size=33,
        st_temporal_dilation=2,
    ),
    "E": replace(
        DEFAULT_N2P3_ARCHITECTURE,
        temporal_kernel_size=33,
        mst_kernel_sizes=(13, 25),
    ),
}


def temporal_receptive_span_ms(
    kernel_samples: int,
    sample_rate_hz: float,
    dilation: int = 1,
) -> float:
    """Return the endpoint-to-endpoint span of an odd discrete kernel."""

    if (
        kernel_samples < 1
        or kernel_samples % 2 == 0
        or dilation < 1
        or not isfinite(sample_rate_hz)
        or sample_rate_hz <= 0.0
    ):
        raise ValueError(
            "kernel_samples must be positive and odd; dilation and sample_rate_hz "
            "must be positive."
        )
    return (kernel_samples - 1) * dilation * 1000.0 / sample_rate_hz


def stacked_temporal_receptive_field_samples(
    temporal_kernel_samples: int,
    pool_samples: int,
    branch_kernel_samples: int,
    *,
    temporal_dilation: int = 1,
    branch_dilation: int = 1,
) -> int:
    """Return the input-sample receptive field through ST conv, pool, and MST conv."""

    if (
        temporal_kernel_samples < 1
        or branch_kernel_samples < 1
        or pool_samples < 1
        or temporal_dilation < 1
        or branch_dilation < 1
    ):
        raise ValueError("temporal kernels, dilations, and pool_samples must be positive.")
    return (
        1
        + (temporal_kernel_samples - 1) * temporal_dilation
        + pool_samples
        - 1
        + (branch_kernel_samples - 1) * branch_dilation * pool_samples
    )


def scale_odd_kernel_preserving_span(
    kernel_samples: int,
    *,
    source_sample_rate_hz: float,
    target_sample_rate_hz: float,
    dilation: int = 1,
) -> int:
    """Scale an odd kernel by preserving its centered physical endpoint span."""

    source_span_ms = temporal_receptive_span_ms(
        kernel_samples,
        source_sample_rate_hz,
        dilation,
    )
    target_dilated_intervals = source_span_ms * target_sample_rate_hz / 1000.0
    target_kernel_intervals = target_dilated_intervals / dilation
    even_intervals = 2 * int(round(target_kernel_intervals / 2.0))
    return even_intervals + 1


def scale_architecture_preserving_spans(
    architecture: N2P3ArchitectureConfig,
    *,
    source_sample_rate_hz: float,
    target_sample_rate_hz: float,
) -> N2P3ArchitectureConfig:
    """Scale temporal kernels while retaining both trunk feature-rate spans."""

    source_feature_rate = source_sample_rate_hz / architecture.st_pool_size
    target_feature_rate = target_sample_rate_hz / architecture.st_pool_size
    return replace(
        architecture,
        temporal_kernel_size=scale_odd_kernel_preserving_span(
            architecture.temporal_kernel_size,
            source_sample_rate_hz=source_sample_rate_hz,
            target_sample_rate_hz=target_sample_rate_hz,
            dilation=architecture.st_temporal_dilation,
        ),
        mst_kernel_sizes=tuple(
            scale_odd_kernel_preserving_span(
                kernel,
                source_sample_rate_hz=source_feature_rate,
                target_sample_rate_hz=target_feature_rate,
                dilation=dilation,
            )
            for kernel, dilation in zip(
                architecture.mst_kernel_sizes,
                architecture.mst_dilations,
                strict=True,
            )
        ),
    )


class _MaxNormSpatialConv(nn.Conv2d):
    """Depthwise spatial projection with a differentiable effective max-norm."""

    def __init__(self, *args, max_norm: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive.")
        self.max_norm = float(max_norm)

    def effective_weight(self) -> torch.Tensor:
        flat = self.weight.flatten(start_dim=1)
        norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
        scales = torch.clamp(self.max_norm / norms, max=1.0).reshape(-1, 1, 1, 1)
        return self.weight * scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.effective_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _MSTBranch(nn.Module):
    """One compressed MS-EEGNet temporal scale after the shared ST block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        output_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, output_features, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(output_features)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.dropout(self.activation(self.norm(x)))


class MSFlattenPool(nn.Module):
    """The original MS-EEGNet temporal-position-preserving classification pool."""

    def __init__(self, pool_size: int = DEFAULT_MST_POOL_SIZE) -> None:
        super().__init__()
        if pool_size < 1:
            raise ValueError("pool_size must be positive.")
        self.pool_size = int(pool_size)
        self.pool = nn.AvgPool1d(self.pool_size, stride=self.pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected MST features (B,D,T), got {tuple(x.shape)}.")
        if x.shape[-1] < self.pool_size:
            raise ValueError("MS flatten pooling requires at least one complete temporal pool window.")
        return torch.flatten(self.pool(x), start_dim=1)


class FullResolutionUnfold(nn.Module):
    """Injectively expose every encoded feature/time coordinate to the head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected MST features (B,D,T), got {tuple(x.shape)}.")
        return torch.flatten(x, start_dim=1)


class ResidualFactorizedQuadraticClassifier(nn.Module):
    """Full linear readout plus a low-rank factorized quadratic residual."""

    def __init__(self, features: int, outputs: int = 2, rank: int = 8) -> None:
        super().__init__()
        if features < 1 or outputs < 1 or rank < 1:
            raise ValueError("features, outputs, and rank must be positive.")
        self.features = int(features)
        self.outputs = int(outputs)
        self.rank = int(rank)
        self.linear = nn.Linear(self.features, self.outputs)
        self.left = nn.Linear(self.features, self.rank, bias=False)
        self.right = nn.Linear(self.features, self.rank, bias=False)
        self.quadratic_output = nn.Linear(self.rank, self.outputs, bias=False)
        nn.init.zeros_(self.quadratic_output.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(
                f"Expected unfolded features (B,{self.features}), got {tuple(x.shape)}."
            )
        interaction = self.left(x) * self.right(x)
        return self.linear(x) + self.quadratic_output(interaction)


class ResidualMLPClassifier(nn.Module):
    """Parameter-matched nonlinear control for the factorized quadratic head."""

    def __init__(self, features: int, outputs: int = 2, hidden_features: int = 16) -> None:
        super().__init__()
        if features < 1 or outputs < 1 or hidden_features < 1:
            raise ValueError("features, outputs, and hidden_features must be positive.")
        self.features = int(features)
        self.outputs = int(outputs)
        self.hidden_features = int(hidden_features)
        self.linear = nn.Linear(self.features, self.outputs)
        self.hidden = nn.Linear(self.features, self.hidden_features)
        self.nonlinear_output = nn.Linear(self.hidden_features, self.outputs, bias=False)
        nn.init.zeros_(self.nonlinear_output.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(
                f"Expected unfolded features (B,{self.features}), got {tuple(x.shape)}."
            )
        return self.linear(x) + self.nonlinear_output(F.gelu(self.hidden(x)))


class LatencyMarginalContrastPool(nn.Module):
    """Pool one temporal scale over a small, physically bounded latency bank.

    Candidate summaries are referenced to the pre-stimulus interval. A learned
    query marginalizes latency inside the declared P300 window rather than
    allowing an unconstrained time attention to select arbitrary peaks.
    """

    def __init__(
        self,
        channels: int,
        *,
        n_times: int | None,
        sfreq: float,
        tmin_s: float,
        evidence_window_ms: Sequence[float] = (250.0, 600.0),
        reference_window_ms: Sequence[float] = (-200.0, 0.0),
        latency_offsets_ms: Sequence[float] = (-100.0, -50.0, 0.0, 50.0, 100.0),
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if n_times is not None and (isinstance(n_times, bool) or n_times < 1):
            raise ValueError("n_times must be positive or None.")
        numeric = torch.tensor([sfreq, tmin_s, temperature], dtype=torch.float64)
        if not bool(torch.isfinite(numeric).all()) or sfreq <= 0.0 or temperature <= 0.0:
            raise ValueError("sfreq and temperature must be finite and positive; tmin_s must be finite.")

        self.channels = int(channels)
        self.n_times = int(n_times) if n_times is not None else None
        self.sfreq = float(sfreq)
        self.tmin_s = float(tmin_s)
        self.evidence_window_ms = self._validate_window("evidence_window_ms", evidence_window_ms)
        self.reference_window_ms = self._validate_window("reference_window_ms", reference_window_ms)
        offsets = tuple(float(offset) for offset in latency_offsets_ms)
        if not offsets or not bool(torch.isfinite(torch.tensor(offsets)).all()):
            raise ValueError("latency_offsets_ms must contain finite offsets.")
        if len(set(offsets)) != len(offsets):
            raise ValueError("latency_offsets_ms must not contain duplicates.")
        self.latency_offsets_ms = offsets
        self.temperature = float(temperature)

        if self.n_times is None:
            candidate_weights = torch.empty((0, 0), dtype=torch.float32)
            reference_weights = torch.empty(0, dtype=torch.float32)
            active_offsets = torch.empty(0, dtype=torch.float32)
        else:
            candidate_weights, reference_weights, active_offsets = self._build_weights(
                self.n_times, device=torch.device("cpu")
            )
        self.register_buffer("candidate_weights", candidate_weights, persistent=False)
        self.register_buffer("reference_weights", reference_weights, persistent=False)
        self.register_buffer("candidate_offsets_ms", active_offsets, persistent=False)
        self.latency_query = nn.Parameter(torch.zeros(self.channels))

    @staticmethod
    def _validate_window(name: str, values: Sequence[float]) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        start, end = (float(values[0]), float(values[1]))
        if not bool(torch.isfinite(torch.tensor([start, end])).all()) or start >= end:
            raise ValueError(f"{name} must be a finite increasing interval.")
        return start, end

    def _build_weights(
        self,
        n_times: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        times_ms = self.tmin_s * 1000.0 + torch.arange(
            n_times, device=device, dtype=torch.float32
        ) * (1000.0 / self.sfreq)
        reference_start, reference_end = self.reference_window_ms
        reference_mask = (times_ms >= reference_start) & (times_ms < reference_end)
        if int(reference_mask.sum()) < 2:
            raise ValueError(
                "The physical epoch must contain at least two pre-stimulus reference samples; "
                "use global_average or ms_flatten only as explicit ablations."
            )

        evidence_start, evidence_end = self.evidence_window_ms
        weights: list[torch.Tensor] = []
        active_offsets: list[float] = []
        for offset in self.latency_offsets_ms:
            mask = (times_ms >= evidence_start + offset) & (times_ms < evidence_end + offset)
            count = int(mask.sum())
            if count >= 2:
                weights.append(mask.to(torch.float32) / count)
                active_offsets.append(offset)
        if not weights:
            raise ValueError(
                "The physical epoch contains no usable latency candidate in the declared evidence window."
            )
        reference_weights = reference_mask.to(torch.float32) / int(reference_mask.sum())
        return (
            torch.stack(weights),
            reference_weights,
            torch.tensor(active_offsets, device=device, dtype=torch.float32),
        )

    def _weights_for(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n_times is not None:
            if x.shape[-1] != self.n_times:
                raise ValueError(
                    f"Expected {self.n_times} time samples from the physical epoch contract, "
                    f"got {x.shape[-1]}."
                )
            return (
                self.candidate_weights.to(device=x.device, dtype=x.dtype),
                self.reference_weights.to(device=x.device, dtype=x.dtype),
                self.candidate_offsets_ms.to(device=x.device),
            )
        candidate_weights, reference_weights, active_offsets = self._build_weights(
            x.shape[-1], device=x.device
        )
        return candidate_weights.to(dtype=x.dtype), reference_weights.to(dtype=x.dtype), active_offsets

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"Expected encoded features (B,{self.channels},T), got {tuple(x.shape)}.")
        candidate_weights, reference_weights, _ = self._weights_for(x)
        candidate_means = torch.einsum("bct,kt->bkc", x, candidate_weights)
        reference_mean = torch.einsum("bct,t->bc", x, reference_weights)
        contrasts = candidate_means - reference_mean.unsqueeze(1)
        scores = torch.einsum("bkc,c->bk", contrasts, torch.tanh(self.latency_query))
        attention = torch.softmax(scores / (self.temperature * self.channels**0.5), dim=1)
        pooled = torch.einsum("bk,bkc->bc", attention, contrasts)
        if return_attention:
            return pooled, attention
        return pooled


class N2P3Net(nn.Module):
    """A compact MS-EEGNet-style trunk with interchangeable readout hypotheses."""

    def __init__(
        self,
        n_channels: int,
        *,
        temporal_filters: int = DEFAULT_ST_TEMPORAL_FILTERS,
        temporal_kernel_size: int = DEFAULT_ST_TEMPORAL_KERNEL_SIZE,
        st_temporal_dilation: int = DEFAULT_ST_TEMPORAL_DILATION,
        spatial_depth_multiplier: int = DEFAULT_SPATIAL_DEPTH_MULTIPLIER,
        st_pool_size: int = DEFAULT_ST_POOL_SIZE,
        mst_kernel_sizes: Sequence[int] = DEFAULT_MST_KERNEL_SIZES,
        mst_dilations: Sequence[int] | None = None,
        mst_features_per_scale: int = DEFAULT_MST_FEATURES_PER_SCALE,
        mst_pool_size: int = DEFAULT_MST_POOL_SIZE,
        dropout: float = DEFAULT_N2P3_DROPOUT,
        spatial_max_norm: float = DEFAULT_SPATIAL_MAX_NORM,
        n_times: int | None = None,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        tmin_s: float = DEFAULT_P300_DATA_CONTRACT.tmin_ms / 1000.0,
        pooling_mode: str = "ms_flatten",
        evidence_window_ms: Sequence[float] = (250.0, 600.0),
        reference_window_ms: Sequence[float] = (-200.0, 0.0),
        latency_offsets_ms: Sequence[float] = (-100.0, -50.0, 0.0, 50.0, 100.0),
        latency_temperature: float = 0.5,
        interaction_rank: int = DEFAULT_INTERACTION_RANK,
        mlp_hidden_features: int = DEFAULT_MLP_HIDDEN_FEATURES,
    ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be positive.")
        if temporal_filters < 1 or spatial_depth_multiplier < 1 or mst_features_per_scale < 1:
            raise ValueError("filter and feature counts must be positive.")
        if temporal_kernel_size < 3 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be an odd integer of at least three.")
        if st_temporal_dilation < 1:
            raise ValueError("st_temporal_dilation must be positive.")
        if not mst_kernel_sizes or any(kernel < 3 or kernel % 2 == 0 for kernel in mst_kernel_sizes):
            raise ValueError("mst_kernel_sizes must contain odd kernels of at least three samples.")
        resolved_mst_dilations = (
            tuple(DEFAULT_MST_DILATION for _ in mst_kernel_sizes)
            if mst_dilations is None
            else tuple(int(dilation) for dilation in mst_dilations)
        )
        if len(resolved_mst_dilations) != len(mst_kernel_sizes) or any(
            dilation < 1 for dilation in resolved_mst_dilations
        ):
            raise ValueError(
                "mst_dilations must contain one positive dilation per MST kernel."
            )
        if st_pool_size < 1 or mst_pool_size < 1:
            raise ValueError("pool sizes must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if spatial_max_norm <= 0.0:
            raise ValueError("spatial_max_norm must be positive.")
        if interaction_rank < 1 or mlp_hidden_features < 1:
            raise ValueError("interaction_rank and mlp_hidden_features must be positive.")
        if n_times is not None and (isinstance(n_times, bool) or n_times < 1):
            raise ValueError("n_times must be positive or None.")
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {sorted(POOLING_MODES)}.")
        if (
            pooling_mode
            in {
                "ms_flatten",
                "full_unfold",
                "mlp_full_unfold",
                "quadratic_full_unfold",
            }
            and n_times is None
        ):
            raise ValueError(f"{pooling_mode} pooling requires n_times.")

        self.n_channels = int(n_channels)
        self.n_times = int(n_times) if n_times is not None else None
        self.sfreq = float(sfreq)
        self.tmin_s = float(tmin_s)
        self.pooling_mode = pooling_mode
        self.temporal_filters = int(temporal_filters)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.st_temporal_dilation = int(st_temporal_dilation)
        self.spatial_depth_multiplier = int(spatial_depth_multiplier)
        self.st_pool_size = int(st_pool_size)
        self.mst_kernel_sizes = tuple(int(kernel) for kernel in mst_kernel_sizes)
        self.mst_dilations = resolved_mst_dilations
        self.mst_features_per_scale = int(mst_features_per_scale)
        self.mst_pool_size = int(mst_pool_size)
        self.dropout_probability = float(dropout)
        self.spatial_max_norm = float(spatial_max_norm)
        self.interaction_rank = int(interaction_rank)
        self.mlp_hidden_features = int(mlp_hidden_features)
        spatial_features = self.temporal_filters * self.spatial_depth_multiplier
        self.spatial_features = spatial_features

        self.st_temporal = nn.Conv2d(
            1,
            self.temporal_filters,
            kernel_size=(1, self.temporal_kernel_size),
            padding=(
                0,
                self.st_temporal_dilation * (self.temporal_kernel_size - 1) // 2,
            ),
            dilation=(1, self.st_temporal_dilation),
            bias=False,
        )
        self.st_temporal_norm = nn.BatchNorm2d(self.temporal_filters)
        self.st_spatial = _MaxNormSpatialConv(
            self.temporal_filters,
            spatial_features,
            kernel_size=(self.n_channels, 1),
            groups=self.temporal_filters,
            bias=False,
            max_norm=self.spatial_max_norm,
        )
        self.st_spatial_norm = nn.BatchNorm2d(spatial_features)
        self.st_activation = nn.ELU()
        self.st_pool = nn.AvgPool2d((1, self.st_pool_size), stride=(1, self.st_pool_size))
        self.st_dropout = nn.Dropout(self.dropout_probability)
        self.mst_branches = nn.ModuleList(
            _MSTBranch(
                spatial_features,
                kernel_size=kernel,
                dilation=dilation,
                output_features=self.mst_features_per_scale,
                dropout=self.dropout_probability,
            )
            for kernel, dilation in zip(
                self.mst_kernel_sizes,
                self.mst_dilations,
                strict=True,
            )
        )

        st_times = (
            self._pooled_time_samples(self.n_times, self.st_pool_size)
            if self.n_times is not None
            else None
        )
        feature_sfreq = self.sfreq / self.st_pool_size
        # AvgPool summarizes samples [j*s, ..., j*s+s-1] at their center.
        feature_tmin_s = self.tmin_s + (self.st_pool_size - 1) / (2.0 * self.sfreq)
        feature_channels = len(self.mst_branches) * self.mst_features_per_scale
        if self.pooling_mode == "global_average":
            self.pool: nn.Module = nn.AdaptiveAvgPool1d(1)
            classifier_features = feature_channels
        elif self.pooling_mode == "ms_flatten":
            self.pool = MSFlattenPool(self.mst_pool_size)
            assert st_times is not None
            classifier_features = feature_channels * self._pooled_time_samples(
                st_times, self.mst_pool_size
            )
        elif self.pooling_mode in {
            "full_unfold",
            "mlp_full_unfold",
            "quadratic_full_unfold",
        }:
            self.pool = FullResolutionUnfold()
            assert st_times is not None
            classifier_features = feature_channels * st_times
        else:
            self.pool = nn.ModuleList(
                [
                    LatencyMarginalContrastPool(
                        self.mst_features_per_scale,
                        n_times=st_times,
                        sfreq=feature_sfreq,
                        tmin_s=feature_tmin_s,
                        evidence_window_ms=evidence_window_ms,
                        reference_window_ms=reference_window_ms,
                        latency_offsets_ms=latency_offsets_ms,
                        temperature=latency_temperature,
                    )
                    for _ in self.mst_branches
                ]
            )
            classifier_features = feature_channels
        self.classifier_features = int(classifier_features)
        if self.pooling_mode == "quadratic_full_unfold":
            self.classifier: nn.Module = ResidualFactorizedQuadraticClassifier(
                self.classifier_features,
                rank=self.interaction_rank,
            )
        elif self.pooling_mode == "mlp_full_unfold":
            self.classifier = ResidualMLPClassifier(
                self.classifier_features,
                hidden_features=self.mlp_hidden_features,
            )
        else:
            self.classifier = nn.Linear(self.classifier_features, 2)

    @staticmethod
    def _pooled_time_samples(n_times: int, pool_size: int) -> int:
        output = (n_times - pool_size) // pool_size + 1
        if output < 1:
            raise ValueError("The physical epoch is shorter than the required pooling window.")
        return output

    @classmethod
    def default_architecture_record(
        cls,
        *,
        pooling_mode: str,
        tmin_s: float,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        n_times: int | None = DEFAULT_P300_DATA_CONTRACT.n_times,
        architecture: N2P3ArchitectureConfig = DEFAULT_N2P3_ARCHITECTURE,
    ) -> dict[str, object]:
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {sorted(POOLING_MODES)}.")
        if not isinstance(architecture, N2P3ArchitectureConfig):
            raise TypeError("architecture must be an N2P3ArchitectureConfig.")
        record: dict[str, object] = {
            "trunk": "ms_eegnet_style",
            "pooling_mode": pooling_mode,
            "tmin_s": float(tmin_s),
            "input_sample_rate_hz": float(sfreq),
            "feature_sample_rate_hz": float(sfreq) / architecture.st_pool_size,
            "st_temporal_filters": architecture.temporal_filters,
            "st_temporal_kernel_samples": architecture.temporal_kernel_size,
            "st_temporal_dilation": architecture.st_temporal_dilation,
            "st_temporal_receptive_span_ms": temporal_receptive_span_ms(
                architecture.temporal_kernel_size,
                float(sfreq),
                architecture.st_temporal_dilation,
            ),
            "spatial_depth_multiplier": architecture.spatial_depth_multiplier,
            "st_pool_size": architecture.st_pool_size,
            "mst_kernel_samples": list(architecture.mst_kernel_sizes),
            "mst_dilations": list(architecture.mst_dilations),
            "mst_receptive_span_ms": [
                temporal_receptive_span_ms(
                    kernel,
                    float(sfreq) / architecture.st_pool_size,
                    dilation,
                )
                for kernel, dilation in zip(
                    architecture.mst_kernel_sizes,
                    architecture.mst_dilations,
                    strict=True,
                )
            ],
            "mst_total_receptive_field_samples": [
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
            ],
            "mst_total_receptive_span_ms": [
                (
                    stacked_temporal_receptive_field_samples(
                        architecture.temporal_kernel_size,
                        architecture.st_pool_size,
                        kernel,
                        temporal_dilation=architecture.st_temporal_dilation,
                        branch_dilation=dilation,
                    )
                    - 1
                )
                * 1000.0
                / float(sfreq)
                for kernel, dilation in zip(
                    architecture.mst_kernel_sizes,
                    architecture.mst_dilations,
                    strict=True,
                )
            ],
            "mst_features_per_scale": architecture.mst_features_per_scale,
            "mst_pool_size": architecture.mst_pool_size,
            "dropout": architecture.dropout,
            "spatial_max_norm": architecture.spatial_max_norm,
        }
        if (
            pooling_mode
            in {
                "full_unfold",
                "mlp_full_unfold",
                "quadratic_full_unfold",
            }
            and n_times is not None
        ):
            st_times = cls._pooled_time_samples(int(n_times), architecture.st_pool_size)
            record.update(
                {
                    "unfold_time_samples": st_times,
                    "unfold_features": (
                        len(architecture.mst_kernel_sizes)
                        * architecture.mst_features_per_scale
                        * st_times
                    ),
                    "classifier": {
                        "full_unfold": "linear_full_resolution",
                        "mlp_full_unfold": "residual_mlp",
                        "quadratic_full_unfold": "residual_factorized_quadratic",
                    }[pooling_mode],
                }
            )
        if pooling_mode == "quadratic_full_unfold":
            record["interaction_rank"] = architecture.interaction_rank
        if pooling_mode == "mlp_full_unfold":
            record["mlp_hidden_features"] = architecture.mlp_hidden_features
        if pooling_mode == "latency_marginal_contrast":
            record.update(
                {
                    "evidence_window_ms": [250.0, 600.0],
                    "reference_window_ms": [-200.0, 0.0],
                    "latency_offsets_ms": [-100.0, -50.0, 0.0, 50.0, 100.0],
                    "latency_temperature": 0.5,
                }
            )
        return record

    def architecture_record(self) -> dict[str, object]:
        architecture = N2P3ArchitectureConfig(
            temporal_filters=self.temporal_filters,
            temporal_kernel_size=self.temporal_kernel_size,
            st_temporal_dilation=self.st_temporal_dilation,
            spatial_depth_multiplier=self.spatial_depth_multiplier,
            st_pool_size=self.st_pool_size,
            mst_kernel_sizes=self.mst_kernel_sizes,
            mst_dilations=self.mst_dilations,
            mst_features_per_scale=self.mst_features_per_scale,
            mst_pool_size=self.mst_pool_size,
            dropout=self.dropout_probability,
            spatial_max_norm=self.spatial_max_norm,
            interaction_rank=self.interaction_rank,
            mlp_hidden_features=self.mlp_hidden_features,
        )
        record = self.default_architecture_record(
            pooling_mode=self.pooling_mode,
            tmin_s=self.tmin_s,
            sfreq=self.sfreq,
            n_times=self.n_times,
            architecture=architecture,
        )
        record.update(
            {
                "st_temporal_filters": self.temporal_filters,
                "st_temporal_kernel_samples": self.temporal_kernel_size,
                "st_temporal_dilation": self.st_temporal_dilation,
                "st_temporal_receptive_span_ms": temporal_receptive_span_ms(
                    self.temporal_kernel_size,
                    self.sfreq,
                    self.st_temporal_dilation,
                ),
                "spatial_depth_multiplier": self.spatial_depth_multiplier,
                "st_pool_size": self.st_pool_size,
                "feature_sample_rate_hz": self.sfreq / self.st_pool_size,
                "mst_kernel_samples": list(self.mst_kernel_sizes),
                "mst_dilations": list(self.mst_dilations),
                "mst_receptive_span_ms": [
                    temporal_receptive_span_ms(
                        kernel,
                        self.sfreq / self.st_pool_size,
                        dilation,
                    )
                    for kernel, dilation in zip(
                        self.mst_kernel_sizes,
                        self.mst_dilations,
                        strict=True,
                    )
                ],
                "mst_total_receptive_field_samples": [
                    stacked_temporal_receptive_field_samples(
                        self.temporal_kernel_size,
                        self.st_pool_size,
                        kernel,
                        temporal_dilation=self.st_temporal_dilation,
                        branch_dilation=dilation,
                    )
                    for kernel, dilation in zip(
                        self.mst_kernel_sizes,
                        self.mst_dilations,
                        strict=True,
                    )
                ],
                "mst_total_receptive_span_ms": [
                    (
                        stacked_temporal_receptive_field_samples(
                            self.temporal_kernel_size,
                            self.st_pool_size,
                            kernel,
                            temporal_dilation=self.st_temporal_dilation,
                            branch_dilation=dilation,
                        )
                        - 1
                    )
                    * 1000.0
                    / self.sfreq
                    for kernel, dilation in zip(
                        self.mst_kernel_sizes,
                        self.mst_dilations,
                        strict=True,
                    )
                ],
                "mst_features_per_scale": self.mst_features_per_scale,
                "mst_pool_size": self.mst_pool_size,
                "classifier_features": self.classifier_features,
            }
        )
        if self.pooling_mode in {
            "full_unfold",
            "mlp_full_unfold",
            "quadratic_full_unfold",
        }:
            assert self.n_times is not None
            unfold_time_samples = self._pooled_time_samples(self.n_times, self.st_pool_size)
            record.update(
                {
                    "unfold_time_samples": unfold_time_samples,
                    "unfold_features": (
                        len(self.mst_branches) * self.mst_features_per_scale * unfold_time_samples
                    ),
                }
            )
        if self.pooling_mode == "latency_marginal_contrast":
            first_pool = self.pool[0]
            record.update(
                {
                    "evidence_window_ms": list(first_pool.evidence_window_ms),
                    "reference_window_ms": list(first_pool.reference_window_ms),
                    "latency_offsets_ms": list(first_pool.latency_offsets_ms),
                    "active_latency_offsets_ms": first_pool.candidate_offsets_ms.tolist(),
                    "latency_temperature": first_pool.temperature,
                }
            )
        return record

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected EEG input (B,{self.n_channels},T), got {tuple(x.shape)}."
            )
        if self.n_times is not None and x.shape[-1] != self.n_times:
            raise ValueError(
                f"Expected {self.n_times} time samples from the physical epoch contract, "
                f"got {x.shape[-1]}."
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated post-MST trunk features without the readout.

        ``forward`` remains unchanged in its public contract: it consumes these
        features through the configured pooling/classification head.
        """

        self._validate_input(x)
        x = x.unsqueeze(1)
        x = self.st_temporal_norm(self.st_temporal(x))
        x = self.st_spatial_norm(self.st_spatial(x))
        x = self.st_dropout(self.st_pool(self.st_activation(x))).squeeze(2)
        branch_features = [branch(x) for branch in self.mst_branches]
        return torch.cat(branch_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.pooling_mode == "global_average":
            pooled = self.pool(features).squeeze(-1)
        elif self.pooling_mode in {
            "ms_flatten",
            "full_unfold",
            "mlp_full_unfold",
            "quadratic_full_unfold",
        }:
            pooled = self.pool(features)
        else:
            branch_features = torch.split(features, self.mst_features_per_scale, dim=1)
            pooled = torch.cat(
                [pool(branch) for pool, branch in zip(self.pool, branch_features, strict=True)],
                dim=1,
            )
        return self.classifier(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
