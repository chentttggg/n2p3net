"""Identity-initialized residual feature adapters for cross-domain P300 transfer.

Mathematical contract
---------------------
The trunk produces structured features ``H = f_theta(X)`` of shape
``(B, C_f, T_f)`` after the MST branches and before pooling. Each domain
adapter learns a bounded, low-capacity residual

    A_eta(H) = W_up( ELU( W_down( DWConv(H) ) ) )

    H' = H + alpha * A_eta(H),    alpha = rho * tanh(a),    a = 0 at init.

``alpha = 0`` at initialization makes the adapted model exactly equal to the
source trunk (identity initialization), so adapter capacity can only act
after the data moves the gate; ``|alpha| < rho`` bounds the residual
amplitude by construction. The depthwise/pointwise bottleneck keeps the
parameter count small (a low-rank, spatiotemporally structured correction
that preserves the MST channel/time axes rather than flattening them).

The adapter carries no normalization states: no BatchNorm statistics are
estimated on target prefixes (constitution P4 and the 2026-09-02 alignment
audit), and the trunk's own source-frozen statistics remain the only
normalization in the adapted path.

``DomainAdapterBank`` routes rows to per-domain adapters with a fail-closed
vocabulary and reserves a target slot (``RESERVED_TARGET_DOMAIN``) for the
subject-level inner loop, which trains only the adapter while the trunk and
classifier stay frozen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

RESERVED_DOMAIN_PREFIX = "__"
RESERVED_TARGET_DOMAIN = "__target__"

DEFAULT_ADAPTER_BOTTLENECK_CHANNELS = 4
DEFAULT_ADAPTER_KERNEL_SIZE = 9
DEFAULT_ADAPTER_MAX_RESIDUAL = 0.5


@dataclass(frozen=True)
class FeatureAdapterConfig:
    """Serializable structure contract for one residual feature adapter."""

    bottleneck_channels: int = DEFAULT_ADAPTER_BOTTLENECK_CHANNELS
    kernel_size: int = DEFAULT_ADAPTER_KERNEL_SIZE
    max_residual: float = DEFAULT_ADAPTER_MAX_RESIDUAL

    def __post_init__(self) -> None:
        if self.bottleneck_channels < 1:
            raise ValueError("bottleneck_channels must be positive.")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least three.")
        if not isfinite(self.max_residual) or not 0.0 < self.max_residual:
            raise ValueError("max_residual must be finite and positive.")


class FeatureResidualAdapter(nn.Module):
    """A zero-initialized, amplitude-bounded residual on structured features.

    Parameters
    ----------
    channels : int
        ``C_f`` of the trunk feature map the adapter corrects.
    config : FeatureAdapterConfig
        Bottleneck width, temporal kernel, and residual bound ``rho``.
    """

    def __init__(self, channels: int, *, config: FeatureAdapterConfig) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        self.channels = int(channels)
        self.config = config
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.kernel_size,
            padding=(config.kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.down = nn.Conv1d(channels, config.bottleneck_channels, kernel_size=1)
        self.up = nn.Conv1d(config.bottleneck_channels, channels, kernel_size=1)
        # ReZero-style gate: tanh(0) = 0 makes the adapter an exact identity
        # at initialization while leaving a nonzero gradient path into the
        # gate, so capacity activates only when the data supports it.
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def residual(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(F.elu(self.down(self.depthwise(h))))

    def residual_scale(self) -> float:
        return float(self.config.max_residual * torch.tanh(self.gate.detach()).item())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or h.shape[1] != self.channels:
            raise ValueError(
                f"Expected trunk features (B,{self.channels},T), got {tuple(h.shape)}."
            )
        alpha = self.config.max_residual * torch.tanh(self.gate.to(h.dtype))
        return h + alpha * self.residual(h)

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DomainAdapterBank(nn.Module):
    """Route trunk-feature rows to per-domain residual adapters.

    ``forward`` semantics (fail-closed):

    - ``domain_ids=None`` returns the features unchanged. This is the explicit
      no-adaptation deployment path (zero-shot target inference), not an error.
    - Integer indices (``torch.Tensor``/sequence) must lie in ``[0, len(vocab))``.
    - String ids must be present in the frozen vocabulary.
    - Post-hoc ``register`` is only allowed for reserved names (the subject
      adapter target slot); the source vocabulary is frozen at construction.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        config: FeatureAdapterConfig,
        domains: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive.")
        if not isinstance(config, FeatureAdapterConfig):
            raise TypeError("config must be a FeatureAdapterConfig.")
        vocabulary = tuple(str(domain) for domain in domains)
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("adapter domains must be unique.")
        if any(not domain.strip() for domain in vocabulary):
            raise ValueError("adapter domains must be non-empty strings.")
        reserved = [d for d in vocabulary if d.startswith(RESERVED_DOMAIN_PREFIX)]
        if reserved:
            raise ValueError(
                f"adapter domain names starting with {RESERVED_DOMAIN_PREFIX!r} are "
                f"reserved for runtime slots: {reserved}."
            )
        self.feature_channels = int(feature_channels)
        self.adapter_config = config
        self.adapters = nn.ModuleDict(
            {domain: FeatureResidualAdapter(feature_channels, config=config)
             for domain in vocabulary}
        )

    @property
    def vocab(self) -> tuple[str, ...]:
        return tuple(self.adapters.keys())

    def domain_index(self, domain: str) -> int:
        vocabulary = self.vocab
        try:
            return vocabulary.index(str(domain))
        except ValueError as error:
            raise ValueError(
                f"domain {domain!r} is absent from the frozen adapter vocabulary "
                f"{list(vocabulary)}."
            ) from error

    def register(self, domain: str, adapter: FeatureResidualAdapter, *, reserved: bool = False) -> None:
        name = str(domain)
        if not name.strip():
            raise ValueError("adapter domain names must be non-empty.")
        if name in self.adapters:
            raise ValueError(f"adapter domain {name!r} is already registered.")
        if not name.startswith(RESERVED_DOMAIN_PREFIX):
            raise ValueError(
                "non-reserved adapter domains are frozen at construction; post-hoc "
                "register is only allowed for reserved runtime slots."
            )
        if not reserved:
            raise ValueError(
                f"adapter domain names starting with {RESERVED_DOMAIN_PREFIX!r} "
                "require reserved=True."
            )
        if adapter.channels != self.feature_channels:
            raise ValueError(
                f"adapter channel contract {adapter.channels} does not match the "
                f"bank feature channels {self.feature_channels}."
            )
        if adapter.config != self.adapter_config:
            raise ValueError("adapter configuration does not match the bank contract.")
        self.adapters[name] = adapter

    def _normalize_indices(
        self, domain_ids: torch.Tensor | Sequence[int] | Sequence[str], batch: int
    ) -> torch.Tensor:
        if isinstance(domain_ids, torch.Tensor):
            if domain_ids.numel() != batch or domain_ids.ndim > 1:
                raise ValueError(
                    f"domain_ids must provide one entry per row ({batch}), "
                    f"got {tuple(domain_ids.shape)}."
                )
            if domain_ids.dtype.is_floating_point:
                raise ValueError("integer domain indices are required.")
            indices = domain_ids.reshape(-1).to(torch.long)
        elif len(domain_ids) != batch:
            raise ValueError(
                f"domain_ids must provide one entry per row ({batch}), "
                f"got {len(domain_ids)}."
            )
        elif isinstance(domain_ids[0], str):
            indices = torch.tensor(
                [self.domain_index(domain) for domain in domain_ids], dtype=torch.long
            )
        else:
            indices = torch.as_tensor(domain_ids, dtype=torch.long).reshape(-1)
        if batch and bool((indices < 0).any()):
            raise ValueError("domain indices must be non-negative.")
        if batch and bool((indices >= len(self.adapters)).any()):
            raise ValueError(
                f"domain indices must lie in [0,{len(self.adapters)}) for vocabulary "
                f"{list(self.vocab)}."
            )
        return indices

    def forward(
        self,
        features: torch.Tensor,
        *,
        domain_ids: torch.Tensor | Sequence[int] | Sequence[str] | None = None,
    ) -> torch.Tensor:
        if domain_ids is None:
            return features
        if features.ndim != 3 or features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected trunk features (B,{self.feature_channels},T), "
                f"got {tuple(features.shape)}."
            )
        if features.shape[0] == 0:
            return features
        indices = self._normalize_indices(domain_ids, features.shape[0]).to(features.device)
        vocabulary = self.vocab
        unique = torch.unique(indices).tolist()
        if len(unique) == 1:
            return self.adapters[vocabulary[unique[0]]](features)
        # Mixed-domain batch: route each slice through its own adapter and
        # scatter the corrected rows back in place. index_put_ on a cloned
        # tensor keeps the operation differentiable without reordering rows.
        output = features.clone()
        for domain_index in unique:
            rows = indices == domain_index
            output[rows] = self.adapters[vocabulary[domain_index]](features[rows])
        return output

    def adapter_parameter_count(self) -> int:
        return sum(
            adapter.adapter_parameter_count() for adapter in self.adapters.values()
        )

    def record(self) -> dict[str, object]:
        return {
            "feature_channels": self.feature_channels,
            "adapter_config": {
                "bottleneck_channels": self.adapter_config.bottleneck_channels,
                "kernel_size": self.adapter_config.kernel_size,
                "max_residual": self.adapter_config.max_residual,
            },
            "domains": list(self.vocab),
            "adapter_parameter_count": self.adapter_parameter_count(),
        }
