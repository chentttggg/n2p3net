"""Discardable heads for masked P300 pretraining.

None of these modules is part of the deployed classifier. ``WaveDecoderHead``
is a small approximate inverse of the MS-EEGNet pooling chain, while
``SubjectProbeHead`` exists only to audit subject-identity leakage.
"""

from __future__ import annotations

import torch
from torch import nn


class WaveDecoderHead(nn.Module):
    """Reconstruct (B,C,T) from trunk features (B,S,T/4) and a time mask.

    The mask is pooled to the trunk time resolution and projected to the same
    feature width, then added to the features. This is intentionally small:
    pretraining decoders must not become the downstream model.
    """

    def __init__(
        self,
        *,
        trunk_channels: int,
        output_channels: int,
        st_pool_size: int = 4,
    ) -> None:
        super().__init__()
        if trunk_channels < 1 or output_channels < 1 or st_pool_size < 1:
            raise ValueError("decoder dimensions and pool size must be positive.")
        self.st_pool_size = int(st_pool_size)
        self.mask_pool = nn.AvgPool1d(self.st_pool_size, stride=self.st_pool_size)
        self.mask_projection = nn.Conv1d(1, trunk_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=self.st_pool_size, mode="linear")
        self.output_conv = nn.Conv1d(trunk_channels, output_channels, kernel_size=1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] * self.st_pool_size != mask.shape[-1]:
            raise ValueError(
                "features must be (B,S,T/4) and mask must be (B,T) or (T)."
            )
        if mask.ndim == 1:
            mask = mask[None]
        if mask.shape[0] == 1 and features.shape[0] != 1:
            mask = mask.expand(features.shape[0], -1)
        if mask.shape[0] != features.shape[0]:
            raise ValueError("batch dimensions of features and mask disagree.")
        pooled = self.mask_pool(mask[:, None, :])
        projected = self.mask_projection(pooled)
        conditioned = features + projected
        upsampled = self.upsample(conditioned)
        if upsampled.shape[-1] != mask.shape[-1]:
            upsampled = upsampled[..., : mask.shape[-1]]
        return self.output_conv(upsampled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SubjectProbeHead(nn.Module):
    """Audit-only linear probe for subject identity.

    It must never be attached to a deployed model. A low probe accuracy is not
    proof of subject-invariance, but a high probe accuracy is a hard warning.
    """

    def __init__(self, feature_dim: int, n_subjects: int) -> None:
        super().__init__()
        if feature_dim < 1 or n_subjects < 2:
            raise ValueError("subject probe requires feature_dim>=1 and n_subjects>=2.")
        self.norm = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, n_subjects)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            features = features.flatten(start_dim=1)
        if features.ndim != 2 or features.shape[1] != self.linear.in_features:
            raise ValueError(f"subject probe expects (B,{self.linear.in_features}) features.")
        return self.linear(self.norm(features))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
