"""Feature-domain transfer modules kept outside the plain N2P3Net core.

The core model is intentionally domain agnostic. This module owns a shared
task path, source-training residuals, a private style path, and a target-only
residual adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from models.n2p3net import N2P3Net

FEATURE_TRANSFER_SCHEMA = "n2p3_feature_domain_transfer/2"
MODEL_SOURCE_DOMAIN_AXIS_SCHEMA = "n2p3_model_source_domain_axis/1"
SOURCE_RESIDUAL_LEDGER_SCHEMA = "n2p3_source_residual_ledger/1"
SOURCE_RESIDUAL_EPS = 1e-6
SOURCE_RESIDUAL_R_MAX = 0.5
SOURCE_DROP_MODES = frozenset({"batch", "sample"})
_FEATURE_TRANSFER_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "invariant_bottleneck_channels",
        "private_channels",
        "target_bottleneck_channels",
        "target_kernel_size",
        "max_invariant_residual",
        "max_target_residual",
        "private_domain_count",
        "source_domain_count",
        "source_drop_p",
        "source_drop_mode",
    }
)
_FEATURE_TRANSFER_DERIVED_FIELDS = frozenset(
    {
        "feature_channels",
        "target_style_conditioning",
        "source_residual_routing",
        "source_residual_ledger_schema",
        "source_residual_ratio_contract",
    }
)


class SourceResidualContractError(RuntimeError):
    """Raised when a source residual makes the complete run invalid."""

    def __init__(self, record: dict[str, object]) -> None:
        self.record = record
        reason = record.get("failure_reason", "source residual contract violation")
        super().__init__(f"Source residual run is invalid: {reason}.")


def source_domain_axis_record(
    vocabulary: tuple[str, ...] | list[str],
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Freeze the shared model-side domain-index vocabulary and optional weights."""

    if isinstance(vocabulary, (str, bytes)):
        raise TypeError("source domain vocabulary must be a sequence of strings.")
    domains = tuple(vocabulary)
    if len(domains) < 2:
        raise ValueError("source domain vocabulary must contain at least two domains.")
    if any(
        not isinstance(domain, str)
        or not domain
        or domain != domain.strip()
        or "\0" in domain
        for domain in domains
    ):
        raise ValueError(
            "source domain vocabulary must contain exact non-empty strings without "
            "whitespace or NUL ambiguity."
        )
    if len(set(domains)) != len(domains):
        raise ValueError("source domain vocabulary entries must be unique.")
    weights_record: dict[str, float] | None = None
    if source_weights is not None:
        if not isinstance(source_weights, Mapping):
            raise TypeError("source_weights must be a domain-keyed mapping or None.")
        if set(source_weights) != set(domains):
            raise ValueError("source_weights keys must exactly match the domain vocabulary.")
        weights_record = {}
        for domain in domains:
            value = source_weights[domain]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("source_weights values must be numeric.")
            weight = float(value)
            if not isfinite(weight) or weight < 0.0:
                raise ValueError("source_weights values must be finite and non-negative.")
            weights_record[domain] = weight
        if not isclose(
            sum(weights_record.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("source_weights values must sum to one.")
    return {
        "schema": MODEL_SOURCE_DOMAIN_AXIS_SCHEMA,
        "vocabulary": list(domains),
        "domain_to_index": {
            domain: domain_index for domain_index, domain in enumerate(domains)
        },
        "source_weights_by_domain": weights_record,
    }


def source_domain_axis_from_record(
    record: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, float] | None]:
    """Parse a model-side axis only when it exactly round-trips canonically."""

    if not isinstance(record, Mapping):
        raise TypeError("source domain axis record must be a mapping.")
    expected_fields = {
        "schema",
        "vocabulary",
        "domain_to_index",
        "source_weights_by_domain",
    }
    missing = sorted(expected_fields - set(record))
    unknown = sorted(set(record) - expected_fields)
    if missing or unknown:
        raise ValueError(
            "source domain axis fields mismatch: "
            f"missing={missing} unknown={unknown}."
        )
    if record.get("schema") != MODEL_SOURCE_DOMAIN_AXIS_SCHEMA:
        raise ValueError("source domain axis schema is unsupported.")
    raw_vocabulary = record["vocabulary"]
    if not isinstance(raw_vocabulary, (list, tuple)):
        raise ValueError("source domain axis vocabulary must be a list.")
    raw_weights = record["source_weights_by_domain"]
    if raw_weights is not None and not isinstance(raw_weights, Mapping):
        raise ValueError("source_weights_by_domain must be a mapping or null.")
    try:
        canonical = source_domain_axis_record(
            tuple(raw_vocabulary),
            raw_weights,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source domain axis record is invalid: {exc}") from exc
    if dict(record) != canonical:
        raise ValueError("source domain axis record is non-canonical or inconsistent.")
    weights = canonical["source_weights_by_domain"]
    return tuple(canonical["vocabulary"]), None if weights is None else dict(weights)


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
    source_domain_count: int = 0
    source_drop_p: float = 0.3
    source_drop_mode: str = "sample"

    def __post_init__(self) -> None:
        for name in (
            "invariant_bottleneck_channels",
            "private_channels",
            "target_bottleneck_channels",
            "target_kernel_size",
            "private_domain_count",
            "source_domain_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))
        for name in ("max_invariant_residual", "max_target_residual", "source_drop_p"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be numeric.")
            object.__setattr__(self, name, float(value))
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
        if self.private_domain_count == 1 or self.private_domain_count < 0:
            raise ValueError("private_domain_count must be zero or at least two.")
        if self.source_domain_count == 1 or self.source_domain_count < 0:
            raise ValueError("source_domain_count must be zero or at least two.")
        if not isfinite(self.source_drop_p) or not (
            self.source_drop_p == 0.0 or 0.2 <= self.source_drop_p <= 0.5
        ):
            raise ValueError(
                "source_drop_p must be D0 p=0 or lie in the preregistered [0.2,0.5]."
            )
        if self.source_drop_mode not in SOURCE_DROP_MODES:
            raise ValueError(
                f"source_drop_mode must be one of {sorted(SOURCE_DROP_MODES)}."
            )
        if (
            self.private_domain_count > 0
            and self.source_domain_count > 0
            and self.private_domain_count != self.source_domain_count
        ):
            raise ValueError(
                "private_domain_count and source_domain_count must match when both "
                "domain-indexed mechanisms are enabled."
            )

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
            "source_domain_count": self.source_domain_count,
            "source_drop_p": self.source_drop_p,
            "source_drop_mode": self.source_drop_mode,
        }


def feature_transfer_config_from_record(
    record: Mapping[str, object],
) -> FeatureTransferConfig:
    """Parse all required v2 fields without compatibility defaults."""

    if not isinstance(record, Mapping):
        raise TypeError("feature transfer record must be a mapping.")
    missing = sorted(_FEATURE_TRANSFER_CONFIG_FIELDS - set(record))
    unknown = sorted(
        set(record)
        - _FEATURE_TRANSFER_CONFIG_FIELDS
        - _FEATURE_TRANSFER_DERIVED_FIELDS
    )
    if missing or unknown:
        raise ValueError(
            "feature transfer record fields mismatch: "
            f"missing={missing} unknown={unknown}."
        )
    if record.get("schema") != FEATURE_TRANSFER_SCHEMA:
        raise ValueError("feature transfer schema is unsupported.")
    try:
        config = FeatureTransferConfig(
            invariant_bottleneck_channels=record["invariant_bottleneck_channels"],
            private_channels=record["private_channels"],
            target_bottleneck_channels=record["target_bottleneck_channels"],
            target_kernel_size=record["target_kernel_size"],
            max_invariant_residual=record["max_invariant_residual"],
            max_target_residual=record["max_target_residual"],
            private_domain_count=record["private_domain_count"],
            source_domain_count=record["source_domain_count"],
            source_drop_p=record["source_drop_p"],
            source_drop_mode=record["source_drop_mode"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature transfer record is invalid: {exc}") from exc
    if {key: record[key] for key in _FEATURE_TRANSFER_CONFIG_FIELDS} != config.record():
        raise ValueError("feature transfer record changes canonical value types.")
    return config


def source_residual_ratio_contract() -> dict[str, object]:
    """Return the frozen sample-level amplitude and failure contract."""

    return {
        "norm": "l2",
        "reduction_axes": "all_non_batch_feature_axes",
        "denominator": "max(feature_l2,eps_res)",
        "zero_denominator_rule": "zero_over_zero=0_nonzero_over_zero=invalid",
        "eps_res": SOURCE_RESIDUAL_EPS,
        "r_max": SOURCE_RESIDUAL_R_MAX,
        "source_gate": "counterfactual_m=1",
        "quantiles": [0.5, 0.95, 0.99],
        "reported_statistics": ["mean", "q50", "q95", "q99", "max"],
        "failure_policy": "entire_run_invalid_no_sample_deletion",
    }


def summarize_source_residual_ratios(
    ratios: torch.Tensor,
    *,
    zero_denominator_nonzero_count: int = 0,
) -> dict[str, object]:
    """Summarize one complete forward without weakening the max gate."""

    if ratios.ndim != 1 or ratios.numel() == 0:
        raise ValueError("source residual ratios must be a non-empty vector.")
    detached = ratios.detach().to(dtype=torch.float32)
    if zero_denominator_nonzero_count < 0:
        raise ValueError("zero_denominator_nonzero_count must be non-negative.")
    nonfinite_count = int((~torch.isfinite(detached)).sum().item())
    finite = nonfinite_count == 0
    if finite:
        quantiles = torch.quantile(
            detached,
            detached.new_tensor([0.5, 0.95, 0.99]),
        )
        maximum = float(detached.max().item())
        summary: dict[str, object] = {
            "sample_count": int(detached.numel()),
            "mean": float(detached.mean().item()),
            "q50": float(quantiles[0].item()),
            "q95": float(quantiles[1].item()),
            "q99": float(quantiles[2].item()),
            "max": maximum,
        }
        valid = maximum <= SOURCE_RESIDUAL_R_MAX
        failure_reason = (
            None
            if valid
            else f"max ratio {maximum:.9g} exceeds r_max={SOURCE_RESIDUAL_R_MAX}"
        )
    else:
        summary = {
            "sample_count": int(detached.numel()),
            "mean": None,
            "q50": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
        valid = False
        failure_reason = (
            "non-zero residual has an exactly zero feature denominator"
            if zero_denominator_nonzero_count
            else "ratio contains NaN or infinity"
        )
    return {
        "schema": SOURCE_RESIDUAL_LEDGER_SCHEMA,
        **source_residual_ratio_contract(),
        **summary,
        "nonfinite_count": nonfinite_count,
        "zero_denominator_nonzero_count": zero_denominator_nonzero_count,
        "valid": valid,
        "status": "valid" if valid else "evaluation_failed",
        "failure_reason": failure_reason,
    }


def source_residual_loss_ledger(
    loss_m0: torch.Tensor | float,
    loss_m1: torch.Tensor | float,
    *,
    ratio_record: dict[str, object],
) -> dict[str, object]:
    """Build an auditable consensus/residual counterfactual loss record."""

    values: list[float] = []
    for name, value in (("loss_m0", loss_m0), ("loss_m1", loss_m1)):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar.")
            scalar = float(value.detach().float().item())
        else:
            scalar = float(value)
        if not isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        values.append(scalar)
    if ratio_record.get("schema") != SOURCE_RESIDUAL_LEDGER_SCHEMA:
        raise ValueError("ratio_record does not use the source residual ledger schema.")
    return {
        "schema": SOURCE_RESIDUAL_LEDGER_SCHEMA,
        "loss_m0": values[0],
        "loss_m1": values[1],
        "loss_delta_m1_minus_m0": values[1] - values[0],
        "ratio": dict(ratio_record),
    }


def feature_transfer_architecture_record(
    config: FeatureTransferConfig,
    *,
    feature_channels: int,
) -> dict[str, object]:
    """Build the one canonical checkpoint/runner wrapper contract."""

    if not isinstance(config, FeatureTransferConfig):
        raise TypeError("config must be a FeatureTransferConfig.")
    if (
        isinstance(feature_channels, bool)
        or not isinstance(feature_channels, Integral)
        or feature_channels < 1
    ):
        raise ValueError("feature_channels must be a positive integer.")
    return {
        **config.record(),
        "feature_channels": int(feature_channels),
        "target_style_conditioning": "not_in_default_path",
        "source_residual_routing": "explicit_source_training_only",
        "source_residual_ledger_schema": SOURCE_RESIDUAL_LEDGER_SCHEMA,
        "source_residual_ratio_contract": source_residual_ratio_contract(),
    }


class FeatureResidualAdapter(nn.Module):
    """Identity-initialized residual with a bounded scalar gate on ``(B,C,T)`` maps."""

    @staticmethod
    def validate_configuration(channels, *, bottleneck_channels, kernel_size, max_residual):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in (channels, bottleneck_channels)):
            raise ValueError("residual adapter channels must be positive integers.")
        if (isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral)
                or kernel_size < 1 or kernel_size % 2 == 0):
            raise ValueError("residual adapter kernel_size must be positive and odd.")
        if (isinstance(max_residual, bool) or not isinstance(max_residual, Real)
                or not isfinite(max_residual) or max_residual <= 0.0):
            raise ValueError("residual adapter max_residual must be finite and positive.")

    def __init__(
        self,
        channels: int,
        *,
        bottleneck_channels: int,
        kernel_size: int,
        max_residual: float,
    ) -> None:
        super().__init__()
        self.validate_configuration(channels, bottleneck_channels=bottleneck_channels,
                                    kernel_size=kernel_size, max_residual=max_residual)
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
    """Wrap a plain N2P3Net with source-training and target-only paths."""

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
        self.source_domain_residuals = nn.ModuleList(
            [
                FeatureResidualAdapter(
                    self.feature_channels,
                    bottleneck_channels=config.target_bottleneck_channels,
                    kernel_size=config.target_kernel_size,
                    max_residual=config.max_target_residual,
                )
                for _ in range(config.source_domain_count)
            ]
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

    def adaptation_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Expose core ownership while keeping source extensions distinct."""
        groups = self.core.adaptation_parameter_groups()
        groups["target_adapter"] = tuple(self.target_adapter.parameters())
        groups["source_extensions"] = tuple(
            p for module in (self.invariant_adapter, self.private_projection,
                             self.private_domain_head, self.source_domain_residuals)
            if module is not None for p in module.parameters()
        )
        values = [p for group in groups.values() for p in group]
        if len(values) != len({id(p) for p in values}) or (
            {id(p) for p in values} != {id(p) for p in self.parameters()}
        ):
            raise ValueError("parameter groups must own every model parameter exactly once")
        return groups

    def set_target_adapter_trainable(self, enabled: bool) -> None:
        for parameter in self.target_adapter_parameters():
            parameter.requires_grad = bool(enabled)

    def set_target_adapter_enabled(self, enabled: bool) -> None:
        """Select the target residual path without changing parameter scopes."""

        self._target_adapter_enabled = bool(enabled)

    def target_adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.target_adapter.parameters())

    @property
    def source_residual_enabled(self) -> bool:
        return bool(self.source_domain_residuals)

    def source_residual_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.source_domain_residuals.parameters())

    def source_residual_scale_record(self) -> list[dict[str, object]]:
        """Record every learned source gate without assigning domain labels."""

        return [
            {
                "domain_index": domain_index,
                "scale": adapter.scale_value(),
                "max_abs_scale": adapter.max_residual,
            }
            for domain_index, adapter in enumerate(self.source_domain_residuals)
        ]

    def _source_domain_indices(
        self,
        features: torch.Tensor,
        domain_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.source_residual_enabled:
            raise RuntimeError("source residuals are disabled by the transfer contract.")
        if domain_indices is None:
            raise ValueError("source residual routing requires explicit domain_indices.")
        if domain_indices.ndim != 1 or domain_indices.shape[0] != features.shape[0]:
            raise ValueError(
                "domain_indices must be a vector with one index per feature row."
            )
        if domain_indices.dtype == torch.bool or domain_indices.dtype.is_floating_point:
            raise ValueError("domain_indices must contain integer source-domain indices.")
        indices = domain_indices.to(device=features.device, dtype=torch.long)
        if bool((indices < 0).any().item()) or bool(
            (indices >= len(self.source_domain_residuals)).any().item()
        ):
            raise ValueError(
                "domain_indices must lie in "
                f"[0,{len(self.source_domain_residuals)}) for source residual routing."
            )
        return indices

    def sample_source_residual_mask(
        self,
        features: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``m ~ Bernoulli(1-p)`` under the declared mask granularity."""

        if features.ndim < 1 or features.shape[0] < 1:
            raise ValueError("source residual mask sampling requires a non-empty batch axis.")
        sample_count = features.shape[0] if self.config.source_drop_mode == "sample" else 1
        sampled = torch.rand(
            (sample_count, 1, 1),
            device=features.device,
            dtype=torch.float32,
            generator=generator,
        ) >= self.config.source_drop_p
        return sampled.to(dtype=features.dtype).expand(features.shape[0], 1, 1)

    def _source_residual_mask(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return self.sample_source_residual_mask(features)
        if mask.ndim == 1 and mask.shape == (features.shape[0],):
            mask = mask[:, None, None]
        if mask.shape not in {(1, 1, 1), (features.shape[0], 1, 1)}:
            raise ValueError(
                "source_residual_mask must have shape (B,), (B,1,1), or (1,1,1)."
            )
        if mask.dtype == torch.bool:
            normalized = mask.to(device=features.device, dtype=features.dtype)
        else:
            normalized = mask.to(device=features.device, dtype=features.dtype)
            if not bool(torch.logical_or(normalized == 0, normalized == 1).all().item()):
                raise ValueError("source_residual_mask values must be exactly zero or one.")
        return normalized.expand(features.shape[0], 1, 1)

    def source_residual_delta(
        self,
        features: torch.Tensor,
        domain_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the gated ``m=1`` residual for each source-domain row."""

        if features.ndim != 3 or features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected feature map (B,{self.feature_channels},T), "
                f"got {tuple(features.shape)}."
            )
        indices = self._source_domain_indices(features, domain_indices)
        delta = torch.zeros_like(features)
        for domain_index, adapter in enumerate(self.source_domain_residuals):
            rows = torch.nonzero(indices == domain_index, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            domain_features = features.index_select(0, rows)
            domain_delta = adapter.scale_tensor(features.dtype) * adapter.residual(
                domain_features
            )
            delta = delta.index_copy(0, rows, domain_delta)
        return delta

    def source_residual_ratios(
        self,
        features: torch.Tensor,
        domain_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the fixed per-sample ``m=1`` residual-to-feature L2 ratio."""

        delta = self.source_residual_delta(features, domain_indices)
        reduction_axes = tuple(range(1, features.ndim))
        numerator = torch.linalg.vector_norm(delta, dim=reduction_axes)
        feature_norm = torch.linalg.vector_norm(features, dim=reduction_axes)
        denominator = torch.maximum(
            feature_norm,
            feature_norm.new_full(feature_norm.shape, SOURCE_RESIDUAL_EPS),
        )
        ratios = numerator / denominator
        return torch.where(
            (feature_norm == 0) & (numerator > 0),
            ratios.new_full(ratios.shape, float("inf")),
            ratios,
        )

    @staticmethod
    def _source_residual_ratio_record(
        features: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, object]:
        reduction_axes = tuple(range(1, features.ndim))
        numerator = torch.linalg.vector_norm(delta, dim=reduction_axes)
        feature_norm = torch.linalg.vector_norm(features, dim=reduction_axes)
        denominator = torch.maximum(
            feature_norm,
            feature_norm.new_full(feature_norm.shape, SOURCE_RESIDUAL_EPS),
        )
        zero_denominator_nonzero = (feature_norm == 0) & (numerator > 0)
        ratios = torch.where(
            zero_denominator_nonzero,
            numerator.new_full(numerator.shape, float("inf")),
            numerator / denominator,
        )
        return summarize_source_residual_ratios(
            ratios,
            zero_denominator_nonzero_count=int(
                zero_denominator_nonzero.sum().item()
            ),
        )

    def source_residual_counterfactual_views(
        self,
        features: torch.Tensor,
        domain_indices: torch.Tensor | None,
        *,
        fail_closed: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Return consensus and ``m=1`` paths plus their amplitude audit."""

        delta = self.source_residual_delta(features, domain_indices)
        ratio_record = self._source_residual_ratio_record(features, delta)
        if fail_closed and not bool(ratio_record["valid"]):
            raise SourceResidualContractError(ratio_record)
        return features, features + delta, ratio_record

    def source_domain_view(
        self,
        features: torch.Tensor,
        domain_indices: torch.Tensor | None,
        *,
        source_residual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route source-training rows; eval is always the exact consensus path."""

        if not self.source_residual_enabled:
            raise RuntimeError("source residuals are disabled by the transfer contract.")
        if not self.training:
            return features
        if self._target_adapter_enabled:
            raise RuntimeError("target adaptation cannot route source-domain residuals.")
        delta = self.source_residual_delta(features, domain_indices)
        ratio_record = self._source_residual_ratio_record(features, delta)
        if not bool(ratio_record["valid"]):
            raise SourceResidualContractError(ratio_record)
        mask = self._source_residual_mask(features, source_residual_mask)
        return features + mask * delta

    def feature_views(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
        use_source_residual: bool = False,
        source_residual_mask: torch.Tensor | None = None,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core_domain_indices = domain_indices
        if self.core.splitbn_config is None and use_source_residual:
            # The same axis selects P1 residuals, but it is not a SplitBN
            # selector when the wrapped core uses ordinary BatchNorm.
            core_domain_indices = None
        base = self.core.forward_features(
            x,
            domain_indices=core_domain_indices,
            eval_buffer_source=eval_buffer_source,
            eval_buffer_index=eval_buffer_index,
            channel_mask=channel_mask,
        )
        invariant = self.invariant_adapter(base)
        private = F.elu(self.private_projection(base))
        private_summary = private.mean(dim=-1)
        enabled = (
            self._target_adapter_enabled
            if use_target_adapter is None
            else bool(use_target_adapter)
        )
        if enabled:
            if use_source_residual:
                raise ValueError(
                    "target and source residual paths cannot be enabled together."
                )
            adapted = self.target_adapter(invariant)
        elif use_source_residual:
            adapted = self.source_domain_view(
                invariant,
                domain_indices,
                source_residual_mask=source_residual_mask,
            )
        else:
            if source_residual_mask is not None:
                raise ValueError(
                    "source_residual_mask requires use_source_residual=True."
                )
            adapted = invariant
        return adapted, invariant, private_summary

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
        use_source_residual: bool = False,
        source_residual_mask: torch.Tensor | None = None,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the structured feature map consumed by the classifier."""

        adapted, _, _ = self.feature_views(
            x,
            use_target_adapter=use_target_adapter,
            use_source_residual=use_source_residual,
            source_residual_mask=source_residual_mask,
            domain_indices=domain_indices,
            eval_buffer_source=eval_buffer_source,
            eval_buffer_index=eval_buffer_index,
            channel_mask=channel_mask,
        )
        return adapted

    def forward_with_features(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
        use_source_residual: bool = False,
        source_residual_mask: torch.Tensor | None = None,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, invariant_features, private_summary)``.

        The classifier consumes the target-adapted feature map internally;
        exposing the invariant map separately keeps source alignment from
        accidentally aligning a target-specific residual.
        """
        adapted, invariant, private_summary = self.feature_views(
            x,
            use_target_adapter=use_target_adapter,
            use_source_residual=use_source_residual,
            source_residual_mask=source_residual_mask,
            domain_indices=domain_indices,
            eval_buffer_source=eval_buffer_source,
            eval_buffer_index=eval_buffer_index,
            channel_mask=channel_mask,
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        use_target_adapter: bool | None = None,
        use_source_residual: bool = False,
        source_residual_mask: torch.Tensor | None = None,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_features(
            x,
            use_target_adapter=use_target_adapter,
            use_source_residual=use_source_residual,
            source_residual_mask=source_residual_mask,
            domain_indices=domain_indices,
            eval_buffer_source=eval_buffer_source,
            eval_buffer_index=eval_buffer_index,
            channel_mask=channel_mask,
        )[0]

    def architecture_record(self) -> dict[str, object]:
        return {
            **self.core.architecture_record(),
            "feature_transfer": self.feature_transfer_record(),
        }

    def feature_transfer_record(self) -> dict[str, object]:
        """Return only the wrapper contract for checkpoint/runner ledgers."""

        return feature_transfer_architecture_record(
            self.config,
            feature_channels=self.feature_channels,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = [
    "FEATURE_TRANSFER_SCHEMA",
    "MODEL_SOURCE_DOMAIN_AXIS_SCHEMA",
    "SOURCE_RESIDUAL_EPS",
    "SOURCE_RESIDUAL_LEDGER_SCHEMA",
    "SOURCE_RESIDUAL_R_MAX",
    "FeatureDomainTransfer",
    "FeatureResidualAdapter",
    "FeatureTransferConfig",
    "SourceResidualContractError",
    "feature_transfer_architecture_record",
    "feature_transfer_config_from_record",
    "source_domain_axis_from_record",
    "source_domain_axis_record",
    "source_residual_loss_ledger",
    "source_residual_ratio_contract",
    "summarize_source_residual_ratios",
]
