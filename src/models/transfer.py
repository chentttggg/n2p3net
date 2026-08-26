"""Shared/private transfer modules with low-rank dataset adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(value, float(scale))


class LowRankDatasetAdapter(nn.Module):
    """Per-dataset rank-r residual adapter, initialized as an exact identity."""

    def __init__(self, d_model: int, n_datasets: int, rank: int = 8, scale: float = 1.0) -> None:
        super().__init__()
        if d_model < 1 or n_datasets < 1 or not 1 <= rank <= d_model:
            raise ValueError("Adapter requires d_model/n_datasets>0 and rank in [1,d_model].")
        self.d_model = int(d_model)
        self.n_datasets = int(n_datasets)
        self.rank = int(rank)
        self.scale = float(scale)
        self.down = nn.Parameter(torch.empty(n_datasets, d_model, rank))
        self.up = nn.Parameter(torch.zeros(n_datasets, rank, d_model))
        nn.init.normal_(self.down, std=1.0 / max(1, d_model) ** 0.5)

    def forward(self, features: torch.Tensor, dataset_id: torch.Tensor | None) -> torch.Tensor:
        if features.dim() != 3 or features.shape[-1] != self.d_model:
            raise ValueError(f"features must be (B,T,{self.d_model}).")
        if dataset_id is None:
            dataset_id = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
        dataset_id = dataset_id.to(device=features.device, dtype=torch.long).reshape(-1)
        if dataset_id.shape[0] != features.shape[0]:
            raise ValueError("dataset_id must contain one entry per batch row.")
        if bool(((dataset_id < 0) | (dataset_id >= self.n_datasets)).any()):
            raise ValueError("dataset_id is outside the configured adapter vocabulary.")
        down = self.down[dataset_id]
        up = self.up[dataset_id]
        residual = torch.einsum("btd,bdr,brf->btf", features, down, up)
        return features + self.scale * residual


@dataclass(frozen=True)
class SharedPrivateOutput:
    shared_sequence: torch.Tensor
    shared: torch.Tensor
    private: torch.Tensor
    domain_logits: torch.Tensor
    dataset_logits: torch.Tensor


class SharedPrivateEncoder(nn.Module):
    """Split adapted canonical tokens into task-shared and dataset-private codes."""

    def __init__(
        self,
        d_model: int,
        n_datasets: int,
        *,
        private_dim: int | None = None,
        grl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        private_dim = int(private_dim or max(8, d_model // 2))
        if d_model < 1 or n_datasets < 2 or private_dim < 1:
            raise ValueError("Shared/private transfer requires at least two datasets.")
        self.d_model = int(d_model)
        self.private_dim = private_dim
        self.grl_scale = float(grl_scale)
        self.shared_norm = nn.LayerNorm(d_model)
        self.private_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, private_dim),
            nn.GELU(),
        )
        self.domain_classifier = nn.Linear(d_model, n_datasets)
        self.dataset_classifier = nn.Linear(private_dim, n_datasets)

    def forward(self, features: torch.Tensor) -> SharedPrivateOutput:
        if features.dim() != 3 or features.shape[-1] != self.d_model:
            raise ValueError(f"features must be (B,T,{self.d_model}).")
        shared_sequence = self.shared_norm(features)
        shared = shared_sequence.mean(dim=1)
        private = self.private_encoder(features.mean(dim=1))
        return SharedPrivateOutput(
            shared_sequence=shared_sequence,
            shared=shared,
            private=private,
            domain_logits=self.domain_classifier(gradient_reverse(shared, self.grl_scale)),
            dataset_logits=self.dataset_classifier(private),
        )
