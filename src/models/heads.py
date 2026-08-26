"""PCW-constrained trial heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HeadsOutput:
    """Trial-level discriminative and interpretable outputs."""

    logit_target: torch.Tensor
    logit_pcw: torch.Tensor
    logit_early: torch.Tensor
    amplitude: torch.Tensor
    logit_aux: torch.Tensor | None = None

    @property
    def p_target(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_target)

    @property
    def p_early(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_early)

    @property
    def p_pcw(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_pcw)


class MultiTaskHeads(nn.Module):
    """Map the three explicit PCW component summaries to trial outputs.

    There is intentionally no unconstrained global bypass. Complementary
    background evidence is supplied only by the independently audited
    strict-past likelihood ratio outside this module.
    """

    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        if d_model < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("d_model must be positive and dropout must lie in [0,1).")
        self.dropout = float(dropout)
        head_a_in = 3 * d_model
        self.head_a = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(head_a_in, 1))
        self.head_b = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(d_model, 1))
        self.head_d = nn.Linear(d_model, 1)

    @property
    def head_pcw(self) -> nn.Module:
        return self.head_a

    def forward(self, H: torch.Tensor) -> HeadsOutput:
        if H.dim() != 3 or H.shape[1] != 3:
            raise ValueError(f"H must be (B,3,D), got {tuple(H.shape)}.")
        h_n2, h_p3a, h_p3b = H.unbind(dim=1)
        logit_pcw = self.head_a(torch.cat((h_n2, h_p3a, h_p3b), dim=-1))
        return HeadsOutput(
            logit_target=logit_pcw,
            logit_pcw=logit_pcw,
            logit_early=self.head_b(h_n2),
            amplitude=self.head_d(h_p3b),
        )


Z2_AUX_POOLS: tuple[str, ...] = ("global_pool", "maxmean", "attention")


class Z2AuxiliaryHead(nn.Module):
    """Research-only full-Z2 auxiliary trial head (E5 claim-gate contrast).

    Production recipes keep this module disabled: the formal PCW main
    classification consumes only ``H``. A named research recipe may enable it
    in ``add`` or ``replace`` mode to test whether the unconstrained sequence
    representation ``Z2`` carries discriminative information outside the three
    component windows. This head is a black-box pooling contrast and must
    never be described as a physiological component reader.
    """

    def __init__(
        self,
        d_model: int = 64,
        pool: str = "attention",
        dropout: float = 0.25,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive.")
        if pool not in Z2_AUX_POOLS:
            raise ValueError(f"pool must be one of {Z2_AUX_POOLS}, got {pool!r}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")
        self.pool = pool
        self.d_model = int(d_model)
        self.query: nn.Parameter | None
        if pool == "global_pool":
            input_dim = d_model
            self.query = None
        elif pool == "maxmean":
            input_dim = 2 * d_model
            self.query = None
        else:
            input_dim = d_model
            self.query = nn.Parameter(torch.randn(d_model) * 0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(input_dim, 1),
        )

    def _pool_features(self, Z2: torch.Tensor) -> torch.Tensor:
        if self.pool == "global_pool":
            return Z2.mean(dim=1)
        if self.pool == "maxmean":
            return torch.cat((Z2.max(dim=1).values, Z2.mean(dim=1)), dim=-1)
        query = self.query.to(dtype=Z2.dtype) if self.query is not None else None
        if query is None:
            raise RuntimeError("attention auxiliary head has no query.")
        scores = torch.einsum("btd,d->bt", Z2, query) / max(float(self.d_model) ** 0.5, 1.0)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", weights, Z2)

    def forward(self, Z2: torch.Tensor) -> torch.Tensor:
        if Z2.dim() != 3:
            raise ValueError(f"Z2 must be (B,T,D), got {tuple(Z2.shape)}.")
        if Z2.shape[-1] != self.d_model:
            raise ValueError(
                f"Z2 feature dim {Z2.shape[-1]} does not match d_model={self.d_model}."
            )
        return self.classifier(self._pool_features(Z2))
