"""Causal sensor dynamics for strict-past prequential density estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise temporal convolution that never reads future samples."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        if channels < 1 or kernel_size < 1 or dilation < 1:
            raise ValueError("channels, kernel_size and dilation must be positive.")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.pad(x, (self.left_padding, 0)))


class ChannelLayerNorm1d(nn.Module):
    """Layer-normalize channels independently at every time point."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalDepthwiseSeparableBlock(nn.Module):
    """One causal depthwise temporal + gated pointwise residual block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")
        self.depthwise = CausalDepthwiseConv1d(channels, kernel_size, dilation)
        self.norm = ChannelLayerNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x)
        h = self.norm(h)
        h = functional.glu(self.pointwise(h), dim=1)
        return x + self.dropout(h)


class CausalSensorEncoder(nn.Module):
    """Five-layer causal TCN with time-local normalization.

    The default receptive field is exactly
    ``1 + (kernel_size - 1) * sum(dilations) = 249`` samples, or
    972.7 ms at 256 Hz. Normalization is per time point, so causality is not
    invalidated by statistics pooled over future samples.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 32,
        kernel_size: int = 9,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        normalize_io: bool = True,
    ):
        super().__init__()
        if n_channels < 1 or d_model < 1:
            raise ValueError("n_channels and d_model must be positive.")
        if kernel_size < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("kernel_size and every dilation must be positive.")
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.normalize_io = bool(normalize_io)
        self.input_projection = nn.Conv1d(n_channels, d_model, kernel_size=1, bias=False)
        self.input_norm = ChannelLayerNorm1d(d_model) if self.normalize_io else None
        self.blocks = nn.ModuleList(
            CausalDepthwiseSeparableBlock(
                d_model,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in self.dilations
        )
        self.output_norm = ChannelLayerNorm1d(d_model) if self.normalize_io else None

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() != 3 or signal.shape[1] != self.n_channels:
            raise ValueError(f"signal must be (B,{self.n_channels},T), got {tuple(signal.shape)}.")
        h = self.input_projection(signal)
        if self.input_norm is not None:
            h = self.input_norm(h)
        h = functional.gelu(h)
        for block in self.blocks:
            h = block(h)
        if self.output_norm is not None:
            h = self.output_norm(h)
        return h.transpose(1, 2)


class CausalObservationEncoder(CausalSensorEncoder):
    """Causal TCN used only by the fixed-coordinate observation likelihood.

    ``normalize_io=False`` preserves baseline-standardized sensor magnitude,
    which is required for continuous conditional-mean prediction. Residual
    blocks still normalize their nonlinear correction locally while their
    identity paths carry the unnormalized signal scale.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 32,
        kernel_size: int = 9,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        normalize_io: bool = False,
    ):
        super().__init__(
            n_channels=n_channels,
            d_model=d_model,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            normalize_io=normalize_io,
        )


@dataclass
class CausalInnovationOutput:
    """Strict-past moments for full or label-selected class hypotheses.

    ``history_correction`` augments a fixed fold-local VAR prediction.
    ``log_variance_scale`` and ``factor_scale`` are dimensionless corrections
    to the VAR innovation scale, so the neural model starts at the strong
    linear baseline rather than relearning absolute sensor units.
    ``Sigma_y = diag(D_y) + U_y U_y^T``. Public inference emits both
    hypotheses; supervised training may emit only the true-label slot.
    """

    history_correction: torch.Tensor
    log_variance_scale: torch.Tensor
    factor_scale: torch.Tensor
    hypothesis_labels: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return int(self.factor_scale.shape[-1])


class CausalInnovationDecoder(nn.Module):
    """Decode ``p(x_t | x_<t, y)`` parameters from observation history.

    Encoder state at ``t`` contains ``x_t``. The one-sample shift below is part
    of the decoder contract, not an optional caller convention. Consequently
    all moments at ``t`` are functions only of ``x_<t``. The first prediction
    uses an all-zero boundary state.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        *,
        covariance_rank: int = 2,
        variance_floor: float = 1e-4,
    ):
        super().__init__()
        if covariance_rank not in (1, 2):
            raise ValueError("covariance_rank must be 1 or 2.")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive.")
        self.n_channels = int(n_channels)
        self.covariance_rank = int(covariance_rank)
        self.variance_floor = float(variance_floor)
        self.mean_projection = nn.Linear(d_model, n_channels)
        self.class_embedding = nn.Parameter(torch.zeros(2, d_model))
        self.diagonal_projection = nn.Linear(d_model, n_channels)
        self.factor_projection = nn.Linear(d_model, n_channels * covariance_rank)
        nn.init.zeros_(self.mean_projection.weight)
        nn.init.zeros_(self.mean_projection.bias)
        nn.init.zeros_(self.diagonal_projection.weight)
        nn.init.constant_(self.diagonal_projection.bias, 0.0)
        nn.init.normal_(self.factor_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.factor_projection.bias)

    def forward(
        self,
        features: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        hypothesis_labels: torch.Tensor | None = None,
    ) -> CausalInnovationOutput:
        if features.dim() != 4 or features.shape[1] not in (1, 2):
            raise ValueError(
                f"features must be (B,H,T,D) with H in (1,2), got {tuple(features.shape)}."
            )
        batch_size, n_hypotheses, n_time, _ = features.shape
        if n_hypotheses == 1:
            if hypothesis_labels is None:
                raise ValueError("Single-hypothesis decoding requires hypothesis_labels.")
            hypothesis_labels = hypothesis_labels.to(
                device=features.device, dtype=torch.long
            ).reshape(-1)
            if hypothesis_labels.numel() != batch_size or bool(
                ((hypothesis_labels < 0) | (hypothesis_labels > 1)).any()
            ):
                raise ValueError("hypothesis_labels must be one binary class index per trial.")
            class_embedding = self.class_embedding[hypothesis_labels, None, None]
        else:
            if hypothesis_labels is not None:
                raise ValueError("hypothesis_labels are only valid for single-hypothesis decoding.")
            class_embedding = self.class_embedding[None, :, None]
        past_features = torch.cat(
            (torch.zeros_like(features[:, :, :1]), features[:, :, :-1]), dim=2
        )
        history_correction = self.mean_projection(past_features).permute(0, 1, 3, 2)
        covariance_features = past_features + class_embedding
        log_variance_scale = self.diagonal_projection(covariance_features)
        factor_scale = self.factor_projection(covariance_features).reshape(
            batch_size, n_hypotheses, n_time, self.n_channels, self.covariance_rank
        )
        if channel_mask is not None:
            mask = channel_mask.to(device=features.device, dtype=features.dtype)
            if mask.shape == (self.n_channels,):
                mask = mask[None].expand(batch_size, -1)
            if mask.shape != (batch_size, self.n_channels):
                raise ValueError(
                    f"channel_mask must be (C,) or (B,C), got {tuple(channel_mask.shape)}."
                )
            history_correction = history_correction * mask[:, None, :, None]
            factor_scale = factor_scale * mask[:, None, None, :, None]
            log_variance_scale = log_variance_scale * mask[:, None, None, :]
        return CausalInnovationOutput(
            history_correction=history_correction,
            log_variance_scale=log_variance_scale,
            factor_scale=factor_scale,
            hypothesis_labels=hypothesis_labels,
        )
