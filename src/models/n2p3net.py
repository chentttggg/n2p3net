"""Compact performance-first neural model for oddball P300 detection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _TemporalResidualBlock(nn.Module):
    """A small dilated temporal block that preserves the epoch time axis."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.nn.functional.elu(self.depthwise(x))
        x = self.dropout(self.pointwise(x))
        return residual + x


class N2P3Net(nn.Module):
    """Multi-scale temporal CNN for a single target/non-target logit pair.

    The model retains the project's neural learning core: parallel ERP-scale
    temporal filters, learned spatial mixing, and a compact dilated temporal
    encoder. Its public output is intentionally limited to binary class logits.
    """

    def __init__(
        self,
        n_channels: int,
        *,
        temporal_kernels: Sequence[int] = (17, 33, 65, 129),
        filters_per_scale: int = 8,
        spatial_filters: int = 32,
        temporal_dilations: Sequence[int] = (1, 4, 16),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be positive.")
        if not temporal_kernels or any(kernel < 3 or kernel % 2 == 0 for kernel in temporal_kernels):
            raise ValueError("temporal_kernels must contain positive odd kernels of at least 3 samples.")
        if filters_per_scale < 1 or spatial_filters < 1:
            raise ValueError("filter counts must be positive.")
        if not temporal_dilations or any(dilation < 1 for dilation in temporal_dilations):
            raise ValueError("temporal_dilations must contain positive values.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_channels = int(n_channels)
        branch_channels = len(temporal_kernels) * int(filters_per_scale)
        self.temporal_branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    1,
                    filters_per_scale,
                    kernel_size=(1, kernel),
                    padding=(0, kernel // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(filters_per_scale),
                nn.ELU(),
            )
            for kernel in temporal_kernels
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                branch_channels,
                spatial_filters,
                kernel_size=(self.n_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.Sequential(
            *[
                _TemporalResidualBlock(spatial_filters, int(dilation), dropout)
                for dilation in temporal_dilations
            ]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spatial_filters, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected EEG input (B,{self.n_channels},T), got {tuple(x.shape)}."
            )
        x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.temporal_branches], dim=1)
        x = self.spatial(x).squeeze(2)
        return self.classifier(self.temporal_encoder(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
