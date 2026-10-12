"""Fail-closed split BatchNorm with explicit evaluation provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

NORMALIZATION_SCHEMA = "n2p3_normalization/1"
SPLITBN_TARGET_BUFFER_ESTIMANDS = {
    "pooled_source": "source_mixture",
    "source_weighted": "declared_source_composition",
    "target_calibration": "calibration_defined_target",
}
SPLITBN_TARGET_BUFFER_SOURCES = frozenset(SPLITBN_TARGET_BUFFER_ESTIMANDS)
SPLITBN_SOURCE_DOMAIN_BUFFER = "source_domain"


def _target_statistic_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

_NORMALIZATION_RECORD_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "domain_count",
        "buffer_source",
        "buffer_index",
        "source_weights",
        "estimand",
    }
)


def batchnorm_record() -> dict[str, object]:
    """Return the explicit architecture contract for ordinary pooled BN."""

    return {
        "schema": NORMALIZATION_SCHEMA,
        "mode": "batchnorm",
        "domain_count": None,
        "buffer_source": "pooled_training",
        "buffer_index": None,
        "source_weights": None,
        "estimand": "pooled_training_distribution",
    }


@dataclass(frozen=True)
class SplitBNConfig:
    """Serializable source-domain and unseen-target buffer contract."""

    domain_count: int
    target_buffer_source: str
    source_weights: tuple[float, ...] | None = None

    _SUM_TOLERANCE: ClassVar[float] = 1e-9

    def __post_init__(self) -> None:
        if (
            isinstance(self.domain_count, bool)
            or not isinstance(self.domain_count, Integral)
            or self.domain_count < 2
        ):
            raise ValueError("splitbn_domain_count must be an integer of at least two.")
        object.__setattr__(self, "domain_count", int(self.domain_count))
        if self.target_buffer_source not in SPLITBN_TARGET_BUFFER_SOURCES:
            raise ValueError(
                "splitbn_target_buffer_source must be one of "
                f"{sorted(SPLITBN_TARGET_BUFFER_SOURCES)}."
            )
        if self.source_weights is not None:
            raw_weights = tuple(self.source_weights)
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_weights):
                raise ValueError("splitbn_source_weights must contain numeric values.")
            weights = tuple(float(value) for value in raw_weights)
            object.__setattr__(self, "source_weights", weights)
        if self.target_buffer_source == "source_weighted":
            if self.source_weights is None:
                raise ValueError(
                    "source_weighted SplitBN requires explicit splitbn_source_weights."
                )
            if len(self.source_weights) != self.domain_count:
                raise ValueError(
                    "splitbn_source_weights must contain one weight per source domain."
                )
            if any(not isfinite(value) or value < 0.0 for value in self.source_weights):
                raise ValueError("splitbn_source_weights must be finite and non-negative.")
            if not isclose(
                sum(self.source_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=self._SUM_TOLERANCE,
            ):
                raise ValueError("splitbn_source_weights must sum to one.")
        elif self.source_weights is not None:
            raise ValueError("splitbn_source_weights are only valid with source_weighted buffers.")

    @property
    def estimand(self) -> str:
        return SPLITBN_TARGET_BUFFER_ESTIMANDS[self.target_buffer_source]

    def model_kwargs(self) -> dict[str, object]:
        return {
            "splitbn_domain_count": self.domain_count,
            "splitbn_target_buffer_source": self.target_buffer_source,
            "splitbn_source_weights": self.source_weights,
        }

    def record(self) -> dict[str, object]:
        return {
            "schema": NORMALIZATION_SCHEMA,
            "mode": "splitbn",
            "domain_count": self.domain_count,
            "buffer_source": self.target_buffer_source,
            "buffer_index": None,
            "source_weights": (None if self.source_weights is None else list(self.source_weights)),
            "estimand": self.estimand,
        }


def normalization_record(config: SplitBNConfig | None) -> dict[str, object]:
    """Serialize either normalization arm without implicit defaults."""

    return batchnorm_record() if config is None else config.record()


def resolve_splitbn_config(
    *,
    domain_count: int | None,
    target_buffer_source: str | None,
    source_weights: Sequence[float] | None,
) -> SplitBNConfig | None:
    """Resolve constructor arguments without inventing a target buffer policy."""

    if domain_count is None:
        if target_buffer_source is not None or source_weights is not None:
            raise ValueError("SplitBN buffer settings require an explicit domain count.")
        return None
    if target_buffer_source is None:
        raise ValueError("SplitBN requires a predeclared target buffer source.")
    return SplitBNConfig(
        domain_count=domain_count,
        target_buffer_source=target_buffer_source,
        source_weights=None if source_weights is None else tuple(source_weights),
    )


def splitbn_config_from_record(record: object) -> SplitBNConfig | None:
    """Parse a complete normalization record and reject semantic drift."""

    if not isinstance(record, Mapping):
        raise ValueError("checkpoint normalization must be a mapping.")
    record = dict(record)
    keys = frozenset(record)
    missing = sorted(_NORMALIZATION_RECORD_FIELDS - keys)
    unknown = sorted(keys - _NORMALIZATION_RECORD_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"checkpoint normalization fields mismatch: missing={missing} unknown={unknown}."
        )
    if record.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("checkpoint normalization schema is unsupported.")
    mode = record.get("mode")
    if mode == "batchnorm":
        if record != batchnorm_record():
            raise ValueError("checkpoint batchnorm contract contains inconsistent fields.")
        return None
    if mode != "splitbn":
        raise ValueError("checkpoint normalization mode must be batchnorm or splitbn.")
    if record.get("buffer_index") is not None:
        raise ValueError("unseen-target SplitBN buffers cannot predeclare a source-domain index.")
    weights = record.get("source_weights")
    if weights is not None and not isinstance(weights, (list, tuple)):
        raise ValueError("checkpoint SplitBN source_weights must be a list or null.")
    domain_count = record["domain_count"]
    if isinstance(domain_count, bool) or not isinstance(domain_count, Integral):
        raise ValueError("checkpoint SplitBN domain_count must be an integer.")
    try:
        config = SplitBNConfig(
            domain_count=int(domain_count),
            target_buffer_source=record["buffer_source"],
            source_weights=(None if weights is None else tuple(weights)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint SplitBN contract is invalid: {exc}") from exc
    if record != config.record():
        raise ValueError("checkpoint SplitBN estimand or buffer contract is inconsistent.")
    return config


class _SplitBatchNorm(nn.Module):
    """BatchNorm implementation with shared affine and isolated domain buffers."""

    _expected_dims: ClassVar[tuple[int, ...]]

    def __init__(
        self,
        num_features: int,
        *,
        domain_count: int,
        target_buffer_source: str,
        source_weights: tuple[float, ...] | None = None,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
    ) -> None:
        super().__init__()
        config = SplitBNConfig(
            domain_count=domain_count,
            target_buffer_source=target_buffer_source,
            source_weights=source_weights,
        )
        if (
            isinstance(num_features, bool)
            or not isinstance(num_features, Integral)
            or num_features < 1
        ):
            raise ValueError("num_features must be a positive integer.")
        if isinstance(eps, bool) or not isinstance(eps, Real) or not isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive.")
        if momentum is not None and (
            isinstance(momentum, bool)
            or not isinstance(momentum, Real)
            or not isfinite(momentum)
            or not 0.0 <= momentum <= 1.0
        ):
            raise ValueError("momentum must be None or finite in [0, 1].")
        self.num_features = int(num_features)
        self.domain_count = config.domain_count
        self.target_buffer_source = config.target_buffer_source
        self.source_weights = config.source_weights
        self.eps = float(eps)
        self.momentum = None if momentum is None else float(momentum)

        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.register_buffer("running_mean", torch.zeros(self.domain_count, self.num_features))
        self.register_buffer("running_var", torch.ones(self.domain_count, self.num_features))
        self.register_buffer(
            "num_batches_tracked", torch.zeros(self.domain_count, dtype=torch.long)
        )
        self.register_buffer("pooled_running_mean", torch.zeros(self.num_features))
        self.register_buffer("pooled_running_var", torch.ones(self.num_features))
        self.register_buffer("pooled_num_batches_tracked", torch.zeros((), dtype=torch.long))
        self.register_buffer("target_running_mean", torch.zeros(self.num_features))
        self.register_buffer("target_running_var", torch.ones(self.num_features))
        self.register_buffer("target_buffer_initialized", torch.zeros((), dtype=torch.bool))
        # Live source-stream training may suppress the pooled running-statistics
        # update so the predeclared pooled-source target buffer stays frozen while
        # the per-domain source buffers continue to adapt. The flag is a plain
        # runtime attribute (not a buffer) and never appears in ``state_dict``.
        self._freeze_pooled_buffer = False

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()
        self.pooled_running_mean.zero_()
        self.pooled_running_var.fill_(1.0)
        self.pooled_num_batches_tracked.zero_()
        self._clear_target_calibration_buffer()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _clear_target_calibration_buffer(self) -> None:
        """Remove target-derived state; ordinary forward never repopulates it."""

        self.target_running_mean.zero_()
        self.target_running_var.fill_(1.0)
        self.target_buffer_initialized.fill_(False)

    def install_target_calibration_buffer(
        self,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        *,
        frozen: bool,
    ) -> None:
        """Install statistics computed only from declared calibration observations."""

        if self.target_buffer_source != "target_calibration":
            raise RuntimeError(
                "target calibration statistics are outside this SplitBN buffer contract."
            )
        if frozen is not True:
            raise ValueError("target calibration buffers must be explicitly frozen.")
        if bool(self.target_buffer_initialized):
            raise RuntimeError(
                "target calibration buffer is already frozen and cannot be overwritten."
            )
        mean = torch.as_tensor(
            running_mean,
            device=self.target_running_mean.device,
            dtype=self.target_running_mean.dtype,
        )
        variance = torch.as_tensor(
            running_var,
            device=self.target_running_var.device,
            dtype=self.target_running_var.dtype,
        )
        expected = (self.num_features,)
        if mean.shape != expected or variance.shape != expected:
            raise ValueError(f"target calibration statistics must have shape {expected}.")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(variance).all()):
            raise ValueError("target calibration statistics must be finite.")
        if bool((variance < 0.0).any()):
            raise ValueError("target calibration variance must be non-negative.")
        with torch.no_grad():
            self.target_running_mean.copy_(mean)
            self.target_running_var.copy_(variance)
            self.target_buffer_initialized.fill_(True)

    def _check_input_dim(self, inputs: torch.Tensor) -> None:
        if inputs.ndim not in self._expected_dims:
            expected = " or ".join(str(value) for value in self._expected_dims)
            raise ValueError(f"expected {expected}D input, got {inputs.ndim}D input.")
        if inputs.shape[1] != self.num_features:
            raise ValueError(f"expected {self.num_features} features, got {inputs.shape[1]}.")

    def _validate_domain_indices(
        self,
        inputs: torch.Tensor,
        domain_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if domain_indices is None:
            raise ValueError("SplitBN requires explicit domain_indices.")
        if not isinstance(domain_indices, torch.Tensor):
            raise TypeError("domain_indices must be a torch.Tensor.")
        if domain_indices.ndim != 1 or domain_indices.shape[0] != inputs.shape[0]:
            raise ValueError("domain_indices must be one-dimensional and align with the batch.")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if domain_indices.dtype not in integer_dtypes:
            raise TypeError("domain_indices must use an integer dtype.")
        if domain_indices.device != inputs.device:
            raise ValueError("domain_indices must be on the same device as the input.")
        if bool((domain_indices < 0).any()) or bool((domain_indices >= self.domain_count).any()):
            raise ValueError("domain_indices contain an out-of-contract domain index.")
        return domain_indices.to(dtype=torch.long)

    def _momentum_factor(self, batches_tracked: torch.Tensor) -> float:
        batches_tracked.add_(1)
        if self.momentum is not None:
            return self.momentum
        return 1.0 / float(batches_tracked.item())

    def _update_pooled_buffer(self, inputs: torch.Tensor) -> None:
        with torch.no_grad():
            factor = self._momentum_factor(self.pooled_num_batches_tracked)
            F.batch_norm(
                inputs.detach(),
                self.pooled_running_mean,
                self.pooled_running_var,
                None,
                None,
                True,
                factor,
                self.eps,
            )

    def _validate_training_observations(
        self,
        inputs: torch.Tensor,
        domain_indices: torch.Tensor,
    ) -> None:
        """Reject batches that BatchNorm cannot normalize before mutating buffers.

        ``F.batch_norm`` requires more than one value per channel in training
        mode.  A mixed-domain batch can satisfy that globally while an active
        domain does not.  Check the per-domain effective observation count
        before updating either the domain or pooled running statistics so a
        rejected batch leaves the module state unchanged.
        """

        observations_per_row = 1
        # BatchNorm reduces every axis except batch and feature/channel.
        for size in inputs.shape[2:]:
            observations_per_row *= size
        for domain_index in range(self.domain_count):
            rows = int((domain_indices == domain_index).sum())
            if rows and rows * observations_per_row <= 1:
                raise ValueError(
                    "SplitBN training requires more than one value per channel "
                    f"for active domain {domain_index}."
                )

    def _normalize_domains(
        self,
        inputs: torch.Tensor,
        domain_indices: torch.Tensor,
        *,
        update: bool,
    ) -> torch.Tensor:
        output = torch.empty_like(inputs)
        for domain_index in range(self.domain_count):
            rows = torch.nonzero(domain_indices == domain_index, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if not update and int(self.num_batches_tracked[domain_index]) == 0:
                raise RuntimeError(
                    f"SplitBN source-domain buffer {domain_index} was never initialized."
                )
            selected = inputs.index_select(0, rows)
            if update:
                factor = self._momentum_factor(self.num_batches_tracked[domain_index])
            else:
                factor = 0.0
            normalized = F.batch_norm(
                selected,
                self.running_mean[domain_index],
                self.running_var[domain_index],
                self.weight,
                self.bias,
                update,
                factor,
                self.eps,
            )
            output.index_copy_(0, rows, normalized)
        return output

    def _weighted_source_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source_weights is None:
            raise RuntimeError("source_weighted SplitBN lacks declared source weights.")
        weights = self.running_mean.new_tensor(self.source_weights)
        active = weights > 0.0
        if bool((self.num_batches_tracked[active] == 0).any()):
            raise RuntimeError("a positively weighted SplitBN source buffer is uninitialized.")
        mean = torch.sum(weights[:, None] * self.running_mean, dim=0)
        second_moment = torch.sum(
            weights[:, None] * (self.running_var + self.running_mean.square()),
            dim=0,
        )
        variance = torch.clamp(second_moment - mean.square(), min=0.0)
        return mean, variance

    def _aggregate_eval_statistics(self, source: str) -> tuple[torch.Tensor, torch.Tensor]:
        if source != self.target_buffer_source:
            raise ValueError(
                "eval_buffer_source disagrees with the predeclared SplitBN target buffer."
            )
        if source == "pooled_source":
            if int(self.pooled_num_batches_tracked) == 0:
                raise RuntimeError("the pooled-source SplitBN buffer is uninitialized.")
            return self.pooled_running_mean, self.pooled_running_var
        if source == "source_weighted":
            return self._weighted_source_statistics()
        if not bool(self.target_buffer_initialized):
            raise RuntimeError(
                "target_calibration SplitBN buffer was not explicitly installed and frozen."
            )
        return self.target_running_mean, self.target_running_var

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
        eval_buffer_source: str | None = None,
        eval_buffer_index: int | None = None,
    ) -> torch.Tensor:
        self._check_input_dim(inputs)
        if eval_buffer_source is not None and not isinstance(eval_buffer_source, str):
            raise TypeError("eval_buffer_source must be a string or None.")
        if self.training:
            if eval_buffer_source is not None or eval_buffer_index is not None:
                raise ValueError("training SplitBN cannot consume evaluation buffer selectors.")
            indices = self._validate_domain_indices(inputs, domain_indices)
            if inputs.shape[0] == 0:
                raise ValueError("SplitBN training requires a non-empty batch.")
            self._validate_training_observations(inputs, indices)
            if not getattr(self, "_freeze_pooled_buffer", False):
                self._update_pooled_buffer(inputs)
            return self._normalize_domains(inputs, indices, update=True)

        if eval_buffer_source is None:
            raise ValueError("eval SplitBN requires an explicit eval_buffer_source.")
        if eval_buffer_source == SPLITBN_SOURCE_DOMAIN_BUFFER:
            if domain_indices is not None and eval_buffer_index is not None:
                raise ValueError(
                    "source-domain eval accepts domain_indices or eval_buffer_index, not both."
                )
            if domain_indices is not None:
                indices = self._validate_domain_indices(inputs, domain_indices)
            else:
                if eval_buffer_index is None:
                    raise ValueError(
                        "source-domain eval requires domain_indices or eval_buffer_index."
                    )
                if isinstance(eval_buffer_index, bool) or not isinstance(eval_buffer_index, int):
                    raise TypeError("eval_buffer_index must be an integer.")
                if not 0 <= eval_buffer_index < self.domain_count:
                    raise ValueError("eval_buffer_index is outside the source-domain contract.")
                indices = torch.full(
                    (inputs.shape[0],),
                    eval_buffer_index,
                    device=inputs.device,
                    dtype=torch.long,
                )
            return self._normalize_domains(inputs, indices, update=False)

        if domain_indices is not None or eval_buffer_index is not None:
            raise ValueError("aggregate target buffers do not accept source-domain indices.")
        if eval_buffer_source not in SPLITBN_TARGET_BUFFER_SOURCES:
            raise ValueError(
                "eval_buffer_source must select source_domain or a declared target buffer."
            )
        mean, variance = self._aggregate_eval_statistics(eval_buffer_source)
        return F.batch_norm(
            inputs,
            mean,
            variance,
            self.weight,
            self.bias,
            False,
            0.0,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, domain_count={self.domain_count}, eps={self.eps}, "
            f"momentum={self.momentum}, target_buffer_source="
            f"{self.target_buffer_source!r}"
        )


class SplitBN1d(_SplitBatchNorm):
    """Split BatchNorm for ``(B,C)`` or ``(B,C,T)`` feature tensors."""

    _expected_dims = (2, 3)


class SplitBN2d(_SplitBatchNorm):
    """Split BatchNorm for ``(B,C,H,W)`` feature tensors."""

    _expected_dims = (4,)



def install_splitbn_target_calibration_buffers(
    model: nn.Module,
    inputs: torch.Tensor | list[torch.Tensor],
    *,
    device: torch.device | str,
    batch_size: int = 256,
    eval_buffer_source: str = SPLITBN_SOURCE_DOMAIN_BUFFER,
    domain_index: int = 0,
    channel_masks: torch.Tensor | list[torch.Tensor] | None = None,
    scope_record: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Install target moments in execution order with masked site statistics.

    The observed N2P3Net temporal site receives packed measured traces; its
    spatial/MST sites receive complete mixed features. Moments and counts use
    those actual site inputs, never a guessed electrode-to-feature mapping.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(domain_index, bool) or not isinstance(domain_index, int) or domain_index < 0:
        raise ValueError("domain_index must be a non-negative integer.")
    if eval_buffer_source not in {SPLITBN_SOURCE_DOMAIN_BUFFER, "target_calibration"}:
        raise ValueError("unsupported SplitBN calibration evaluation buffer source.")
    splitbn_modules = [(path, module) for path, module in model.named_modules()
                       if isinstance(module, (SplitBN1d, SplitBN2d))]
    if not splitbn_modules:
        raise ValueError("model contains no SplitBN1d/SplitBN2d modules.")
    mismatched = [path for path, module in splitbn_modules
                  if module.target_buffer_source != "target_calibration"]
    if mismatched:
        raise ValueError("every SplitBN module must declare target_calibration: "+str(mismatched))
    if domain_index >= min(module.domain_count for _, module in splitbn_modules):
        raise ValueError("domain_index is outside the source-domain contract.")
    initialized = {path: bool(module.target_buffer_initialized) for path, module in splitbn_modules}
    any_initialized = any(initialized.values())
    if any_initialized and not all(initialized.values()):
        raise ValueError("SplitBN target calibration buffers are partially initialized.")
    raw_chunks = [inputs] if isinstance(inputs, torch.Tensor) else list(inputs)
    if not raw_chunks or any(not isinstance(chunk, torch.Tensor) or chunk.ndim < 2 or not chunk.shape[0]
                             for chunk in raw_chunks):
        raise ValueError("inputs must contain non-empty tensor calibration batches.")
    resolved_device = torch.device(device)
    chunks = [chunk.detach().to(resolved_device) for chunk in raw_chunks]
    core = getattr(model, "core", model)
    if channel_masks is None:
        if bool(getattr(core, "observed_channel_norm", False)):
            raise ValueError("observed-channel-norm models require channel_masks.")
        mask_chunks = [None] * len(chunks)
    else:
        supplied = [channel_masks] if isinstance(channel_masks, torch.Tensor) else list(channel_masks)
        if len(supplied) != len(chunks):
            raise ValueError("channel_masks must align with calibration batches.")
        mask_chunks = []
        for mask, chunk in zip(supplied, chunks, strict=True):
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or tuple(mask.shape) != tuple(chunk.shape[:2])
                    or not bool(mask.any(dim=1).all())):
                raise ValueError("channel_masks must be tensors shaped (batch, channels).")
            mask_chunks.append(mask.detach().to(device=resolved_device, dtype=torch.bool))
    mask_digest = hashlib.sha256()
    if not any(mask is not None for mask in mask_chunks):
        mask_digest.update(b"unmasked")
    else:
        for mask in mask_chunks:
            if mask is None:
                mask_digest.update(b"none")
                continue
            mask_digest.update(str(tuple(mask.shape)).encode("ascii"))
            mask_digest.update(mask.cpu().contiguous().numpy().tobytes())
    mask_digest_hex = mask_digest.hexdigest()
    scope = {} if scope_record is None else json.loads(json.dumps(dict(scope_record), allow_nan=False))
    scope_bytes = json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    scope_digest = hashlib.sha256(scope_bytes).hexdigest()
    scope_input = hashlib.sha256()
    for chunk, mask in zip(chunks, mask_chunks, strict=True):
        measured = chunk if mask is None else chunk[mask]
        scope_input.update(str(tuple(measured.shape)).encode("ascii"))
        scope_input.update(str(measured.dtype).encode("ascii"))
        scope_input.update(measured.cpu().contiguous().numpy().tobytes())
    forward = getattr(model, "forward_features", None)
    if not callable(forward):
        forward = model.forward
    means: dict[str, torch.Tensor] = {}
    second: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    execution_order: list[str] = []
    reached_this_pass: list[str] = []
    site_input_shapes: dict[str, list[int]] = {}

    def accumulate(path: str, module: nn.Module, value: torch.Tensor) -> None:
        if value.ndim < 2 or value.shape[1] != module.num_features:
            raise RuntimeError(f"SplitBN module {path!r} received malformed input {tuple(value.shape)}.")
        value = value.detach().double()
        site_input_shapes[path] = list(value.shape[1:])
        reduction = (0, *tuple(range(2, value.ndim)))
        count = torch.full((module.num_features,), value.numel() // module.num_features,
                           device=value.device, dtype=torch.float64)
        mean = value.mean(dim=reduction)
        variance = value.var(dim=reduction, unbiased=False)
        previous = counts.get(path)
        if previous is None:
            means[path], second[path], counts[path] = mean.clone(), variance * count, count.clone()
        else:
            total = previous + count
            delta = mean - means[path]
            means[path] = means[path] + delta * (count / total.clamp_min(1))
            second[path] = second[path] + variance * count + delta * (mean - means[path]) * count
            counts[path] = total

    class _SiteCapturedError(Exception):
        pass

    def calibration_pass(sites, kwargs_factory, *, stop_at_first: bool, collect: bool = True):
        reached: list[str] = []
        active_mask: torch.Tensor | None = None
        def hook(module, args, *, path):
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"SplitBN module {path!r} received no tensor input.")
            if collect:
                accumulate(path, module, args[0])
            reached.append(path)
            if stop_at_first:
                raise _SiteCapturedError
        handles = [module.register_forward_pre_hook(lambda m, a, path=path: hook(m, a, path=path))
                   for path, module in sites]
        rows = 0
        expected = sorted(path for path, _ in sites)
        try:
            for chunk, mask in zip(chunks, mask_chunks, strict=True):
                for begin in range(0, len(chunk), batch_size):
                    batch = chunk[begin:begin + batch_size]
                    active_mask = None if mask is None else mask[begin:begin + batch_size]
                    before = len(reached)
                    kwargs = kwargs_factory(batch)
                    if active_mask is not None:
                        kwargs["channel_mask"] = active_mask
                    try:
                        forward(batch, **kwargs)
                    except _SiteCapturedError:
                        pass
                    seen = reached[before:]
                    if stop_at_first and (len(seen) != 1 or seen[0] != reached[0]):
                        raise RuntimeError("SplitBN execution order differs between calibration batches.")
                    if not stop_at_first and sorted(seen) != expected:
                        raise RuntimeError("every SplitBN site must be reached exactly once per forward.")
                    rows += len(batch)
        finally:
            for handle in handles:
                handle.remove()
        reached_this_pass[:] = reached
        return rows

    def diagnostic_kwargs(batch):
        if eval_buffer_source == SPLITBN_SOURCE_DOMAIN_BUFFER:
            return {"eval_buffer_source": eval_buffer_source,
                    "domain_indices": torch.full((len(batch),), domain_index,
                                                   device=batch.device, dtype=torch.long)}
        return {"eval_buffer_source": eval_buffer_source}

    calibration_rows = 0
    training_modes = [(module, module.training) for module in model.modules()]
    original_target = [(module, module.target_running_mean.clone(), module.target_running_var.clone(),
                        module.target_buffer_initialized.clone()) for _, module in splitbn_modules]
    succeeded = False
    model.eval()
    try:
        with torch.no_grad():
            if any_initialized:
                calibration_rows = calibration_pass(splitbn_modules, diagnostic_kwargs, stop_at_first=False)
                estimand = f"diagnostic_{eval_buffer_source}"
            else:
                pending = dict(splitbn_modules)
                while pending:
                    calibration_rows = calibration_pass(
                        list(pending.items()), lambda _batch: {"eval_buffer_source": "target_calibration"},
                        stop_at_first=True)
                    path = reached_this_pass[0]
                    if path not in counts:
                        raise RuntimeError("sequential SplitBN hook did not capture the next site.")
                    module = pending.pop(path)
                    execution_order.append(path)
                    if bool((counts[path] <= 0).any()):
                        raise RuntimeError(f"SplitBN module {path!r} has an unobserved feature channel.")
                    module.install_target_calibration_buffer(
                        means[path].to(module.target_running_mean.device),
                        (second[path] / counts[path]).to(module.target_running_var.device),
                        frozen=True)
                estimand = "sequential_target_calibration"
                # With all sites installed, confirm the complete actual graph
                # reaches each site once. This pass installs no extra moments.
                calibration_pass(splitbn_modules,
                    lambda _batch: {"eval_buffer_source": "target_calibration"},
                    stop_at_first=False, collect=False)
        if any(path not in counts for path, _ in splitbn_modules):
            raise RuntimeError("a SplitBN site observed no calibration rows.")
        succeeded = True
    finally:
        if not succeeded:
            for module, mean, variance, initialized_flag in original_target:
                module.target_running_mean.copy_(mean)
                module.target_running_var.copy_(variance)
                module.target_buffer_initialized.copy_(initialized_flag)
        for module, training in training_modes:
            module.training = training

    result = {}
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, module in splitbn_modules:
        variance = second[path] / counts[path]
        result[path] = {
            "schema": "n2p3_splitbn_target_install/1",
            "program": "execution_order_target_calibration_actual_site_population/1",
            "record_kind": "diagnostic" if any_initialized else "installation",
            "implementation_sha256": implementation,
            "execution_order": (execution_order or [p for p, _ in splitbn_modules]).index(path),
            "estimand": estimand, "scope": scope, "scope_sha256": scope_digest,
            "channel_mask_sha256": mask_digest_hex, "sample_count": int(calibration_rows),
            "observed_input_sha256": scope_input.hexdigest(),
            "observation_count": int(counts[path][0]) if bool((counts[path] == counts[path][0]).all()) else None,
            "observation_counts": [int(x) for x in counts[path].tolist()],
            "mean_digest": _target_statistic_digest(module.target_running_mean),
            "variance_digest": _target_statistic_digest(module.target_running_var),
            "estimated_mean_digest": _target_statistic_digest(means[path]),
            "estimated_variance_digest": _target_statistic_digest(variance),
            "variance_convention": "population_ddof0",
            "shape": list(means[path].shape), "target_buffer_source": module.target_buffer_source,
            "execution_order_paths": execution_order or [p for p, _ in splitbn_modules],
            "observation_support": ("packed_measured_electrode_traces"
                if bool(getattr(core, "observed_channel_norm", False))
                and module is getattr(core, "st_temporal_norm", None)
                else "actual_pre_normalization_features"),
            "site_input_shape": site_input_shapes[path],
            "already_initialized": any_initialized,
        }
        result[path]["receipt_sha256"] = hashlib.sha256(json.dumps(
            result[path], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def validate_splitbn_target_install_receipt(
    model: nn.Module,
    receipt: Mapping[str, Mapping[str, object]],
    *,
    expected_scope: Mapping[str, object] | None = None,
) -> None:
    """Validate a saved installation receipt against actual loaded buffers.

    Source/calibration permission is supplied by the caller. An expected scope
    prevents a coherent replacement receipt from silently changing that scope.
    """
    modules = {path: module for path, module in model.named_modules()
               if isinstance(module, (SplitBN1d, SplitBN2d))}
    if not modules or set(receipt) != set(modules):
        raise ValueError("SplitBN install receipt differs from the actual normalization sites.")
    orders = []
    for path, module in modules.items():
        row = dict(receipt[path])
        identity = row.pop("receipt_sha256", None)
        if identity != hashlib.sha256(json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest():
            raise ValueError("SplitBN install receipt digest is inconsistent.")
        if row.get("schema") != "n2p3_splitbn_target_install/1":
            raise ValueError("SplitBN install receipt schema is unsupported.")
        if (row.get("program") != "execution_order_target_calibration_actual_site_population/1"
                or row.get("variance_convention") != "population_ddof0"
                or row.get("target_buffer_source") != module.target_buffer_source):
            raise ValueError("SplitBN install receipt changes its normalization estimand.")
        if (row.get("record_kind") != "installation" or row.get("already_initialized") is not False
                or row.get("estimand") != "sequential_target_calibration"):
            raise ValueError("a diagnostic receipt cannot certify a target buffer installation.")
        if expected_scope is not None and row.get("scope") != dict(expected_scope):
            raise ValueError("SplitBN install receipt has a different calibration scope.")
        scope_bytes = json.dumps(row.get("scope"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if row.get("scope_sha256") != hashlib.sha256(scope_bytes).hexdigest():
            raise ValueError("SplitBN install receipt scope digest is inconsistent.")
        if not bool(module.target_buffer_initialized):
            raise ValueError("SplitBN installed target buffer is absent.")
        for name, key in (("target_running_mean", "mean_digest"), ("target_running_var", "variance_digest")):
            tensor = getattr(module, name)
            if row.get(key) != _target_statistic_digest(tensor):
                raise ValueError("SplitBN target buffer differs from its saved installation receipt.")
        orders.append(row.get("execution_order"))
    if sorted(orders) != list(range(len(modules))):
        raise ValueError("SplitBN install receipt does not retain a complete execution order.")


__all__ = [
    "NORMALIZATION_SCHEMA",
    "SPLITBN_SOURCE_DOMAIN_BUFFER",
    "SPLITBN_TARGET_BUFFER_ESTIMANDS",
    "SPLITBN_TARGET_BUFFER_SOURCES",
    "SplitBN1d",
    "SplitBN2d",
    "SplitBNConfig",
    "batchnorm_record",
    "install_splitbn_target_calibration_buffers",
    "normalization_record",
    "resolve_splitbn_config",
    "splitbn_config_from_record",
    "validate_splitbn_target_install_receipt",
]
