"""Feature-domain transfer modules kept outside the plain N2P3Net core.

The core model is intentionally domain agnostic. This module owns a shared
task path, a private style path, and a target-only residual adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from models.n2p3net import N2P3Net

FEATURE_TRANSFER_SCHEMA = "n2p3_feature_domain_transfer/1"


@dataclass(frozen=True)
class FeatureTransferConfig:
    """Serializable configuration for the transfer wrapper."""

    invariant_bottleneck_channels: int = 4
    private_channels: int = 4
    target_bottleneck_channels: int = 4
    target_kernel_size: int = 9
    max_invariant_residual: float = 0.25
    max_target_residual: float = 0.25
    private_domain_count: int = 0

    def __post_init__(self) -> None:
        if (
            self.invariant_bottleneck_channels < 1
            or self.private_channels < 1
            or self.target_bottleneck_channels < 1
        ):
            raise ValueError("feature-transfer channel counts must be positive.")
        if self.target_kernel_size < 3 or self.target_kernel_size % 2 == 0:
            raise ValueError("target_kernel_size must be odd and at least three.")
        if not isfinite(self.max_invariant_residual) or self.max_invariant_residual <= 0.0:
            raise ValueError("max_invariant_residual must be finite and positive.")
        if not isfinite(self.max_target_residual) or self.max_target_residual <= 0.0:
            raise ValueError("max_target_residual must be finite and positive.")
        if self.private_domain_count < 0:
            raise ValueError("private_domain_count must be non-negative.")

    def record(self) -> dict[str, object]:
        return {
            "schema": FEATURE_TRANSFER_SCHEMA,
            "invariant_bottleneck_channels": self.invariant_bottleneck_channels,
            "private_channels": self.private_channels,
            "target_bottleneck_channels": self.target_bottleneck_channels,
            "target_kernel_size": self.target_kernel_size,
            "max_invariant_residual": self.max_invariant_residual,
            "max_target_residual": self.max_target_residual,
            "private_domain_count": self.private_domain_count,
        }


class FeatureResidualAdapter(nn.Module):
    """Identity-initialized bounded residual on structured ``(B,C,T)`` maps."""

    def __init__(
        self,
        channels: int,
        *,
        bottleneck_channels: int,
        kernel_size: int,
        max_residual: float,
    ) -> None:
        super().__init__()
        if channels < 1 or bottleneck_channels < 1:
            raise ValueError("residual adapter channels must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("residual adapter kernel_size must be positive and odd.")
        if not isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError("residual adapter max_residual must be finite and positive.")
        self.channels = int(channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.kernel_size = int(kernel_size)
        self.max_residual = float(max_residual)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.down = nn.Conv1d(channels, bottleneck_channels, kernel_size=1)
        self.up = nn.Conv1d(bottleneck_channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.up(F.elu(self.down(self.depthwise(features))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected feature map (B,{self.channels},T), got {tuple(features.shape)}."
            )
        scale = self.scale_tensor(features.dtype)
        return features + scale * self.residual(features)

    def scale_tensor(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the bounded scalar gate in the requested computation dtype."""

        return self.max_residual * torch.tanh(self.gate.to(dtype))

    def scale_value(self) -> float:
        return float(self.scale_tensor().detach().item())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class FeatureDomainTransfer(nn.Module):
    """Wrap a plain N2P3Net with invariant/private and target-adapter paths."""

    def __init__(self, core: N2P3Net, *, config: FeatureTransferConfig) -> None:
        super().__init__()
        if not isinstance(core, N2P3Net):
            raise TypeError("FeatureDomainTransfer requires a plain N2P3Net core.")
        if not isinstance(config, FeatureTransferConfig):
            raise TypeError("config must be a FeatureTransferConfig.")
        self.core = core
        self.config = config
        self.feature_channels = len(core.mst_branches) * core.mst_features_per_scale
        self.invariant_adapter = FeatureResidualAdapter(
            self.feature_channels,
            bottleneck_channels=config.invariant_bottleneck_channels,
            kernel_size=1,
            max_residual=config.max_invariant_residual,
        )
        self.private_projection = nn.Conv1d(
            self.feature_channels,
            config.private_channels,
            kernel_size=1,
            bias=True,
        )
        self.private_domain_head = (
            nn.Linear(config.private_channels, config.private_domain_count)
            if config.private_domain_count >= 2
            else None
        )
        self.target_adapter = FeatureResidualAdapter(
            self.feature_channels,
            bottleneck_channels=config.target_bottleneck_channels,
            kernel_size=config.target_kernel_size,
            max_residual=config.max_target_residual,
        )
        self._target_adapter_enabled = False
        self.set_target_adapter_trainable(False)

    @property
    def n_channels(self) -> int:
        return self.core.n_channels

    @property
    def n_times(self) -> int | None:
        return self.core.n_times

    @property
    def sfreq(self) -> float:
        return self.core.sfreq

    @property
    def tmin_s(self) -> float:
        return self.core.tmin_s

    @property
    def pooling_mode(self) -> str:
        return self.core.pooling_mode

    @property
    def classifier(self) -> nn.Module:
        return self.core.classifier

    @property
    def classifier_features(self) -> int:
        return self.core.classifier_features

    def set_target_adapter_trainable(self, enabled: bool) -> None:
        for parameter in self.target_adapter_parameters():
            parameter.requires_grad = bool(enabled)

    def set_target_adapter_enabled(self, enabled: bool) -> None:
        """Select the target residual path without changing parameter scopes."""

        self._target_adapter_enabled = bool(enabled)

    def target_adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.target_adapter.parameters())

    def feature_views(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.core.forward_features(x)
        invariant = self.invariant_adapter(base)
        private = F.elu(self.private_projection(base))
        private_summary = private.mean(dim=-1)
        enabled = (
            self._target_adapter_enabled
            if use_target_adapter is None
            else bool(use_target_adapter)
        )
        if enabled:
            adapted = self.target_adapter(invariant)
        else:
            adapted = invariant
        return adapted, invariant, private_summary

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
    ) -> torch.Tensor:
        """Return the structured feature map consumed by the classifier."""

        adapted, _, _ = self.feature_views(x, use_target_adapter=use_target_adapter)
        return adapted

    def forward_with_features(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, invariant_features, private_summary)``.

        The classifier consumes the target-adapted feature map internally;
        exposing the invariant map separately keeps source alignment from
        accidentally aligning a target-specific residual.
        """
        adapted, invariant, private_summary = self.feature_views(
            x,
            use_target_adapter=use_target_adapter,
        )
        logits = self.core.classifier(self.core.pool_features(adapted))
        return logits, invariant, private_summary

    def private_domain_logits(self, private_summary: torch.Tensor) -> torch.Tensor:
        if self.private_domain_head is None:
            raise RuntimeError("private domain head is disabled by the transfer contract.")
        return self.private_domain_head(private_summary)

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pool a structured feature map with the core's declared readout."""

        return self.core.pool_features(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]

    def architecture_record(self) -> dict[str, object]:
        return {
            **self.core.architecture_record(),
            "feature_transfer": self.feature_transfer_record(),
        }

    def feature_transfer_record(self) -> dict[str, object]:
        """Return only the wrapper contract for checkpoint/runner ledgers."""

        return {
            **self.config.record(),
            "feature_channels": self.feature_channels,
            "target_style_conditioning": "not_in_default_path",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = ["FEATURE_TRANSFER_SCHEMA", "FeatureDomainTransfer", "FeatureResidualAdapter", "FeatureTransferConfig"]
