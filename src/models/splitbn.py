"""Fail-closed split BatchNorm with explicit evaluation provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real
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
) -> dict[str, dict[str, float | int | str]]:
    """Compute and install frozen target-calibration buffers for every SplitBN site.

    The calibration forward runs in eval mode so neither source-domain nor target
    running statistics are mutated while the pre-normalization activations are
    captured. By default the forward normalizes with the source-domain buffer at
    ``domain_index`` (``eval_buffer_source="source_domain"``); pass
    ``eval_buffer_source="target_calibration"`` to normalize with target buffers
    when they have already been installed. If every target buffer is already
    frozen, the returned statistics are diagnostic only and the installed values
    are left untouched.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(domain_index, bool) or not isinstance(domain_index, int) or domain_index < 0:
        raise ValueError("domain_index must be a non-negative integer.")
    if not isinstance(eval_buffer_source, str):
        raise TypeError("eval_buffer_source must be a string.")
    if eval_buffer_source not in {SPLITBN_SOURCE_DOMAIN_BUFFER, "target_calibration"}:
        raise ValueError(
            "target-calibration helper supports eval_buffer_source in "
            f"{sorted({SPLITBN_SOURCE_DOMAIN_BUFFER, 'target_calibration'})}."
        )
    splitbn_modules = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, (SplitBN1d, SplitBN2d))
    ]
    if not splitbn_modules:
        raise ValueError("model contains no SplitBN1d/SplitBN2d modules.")
    mismatched = [
        path for path, module in splitbn_modules if module.target_buffer_source != "target_calibration"
    ]
    if mismatched:
        raise ValueError(
            "install_splitbn_target_calibration_buffers requires every SplitBN module "
            "to declare target_buffer_source='target_calibration'; found "
            f"{mismatched}."
        )
    if domain_index >= min(module.domain_count for _, module in splitbn_modules):
        raise ValueError("domain_index is outside the SplitBN source-domain contract.")
    initialized = {path: bool(module.target_buffer_initialized) for path, module in splitbn_modules}
    any_initialized = any(initialized.values())
    if any_initialized and not all(initialized.values()):
        raise ValueError("SplitBN target calibration buffers are partially initialized.")

    if isinstance(inputs, torch.Tensor):
        raw_chunks: list[torch.Tensor] = [inputs]
    else:
        try:
            raw_chunks = list(inputs)
        except TypeError as error:
            raise TypeError("inputs must be a torch.Tensor or a list of torch.Tensors.") from error
    if not raw_chunks:
        raise ValueError("inputs must contain at least one calibration batch.")
    for chunk_index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, torch.Tensor):
            raise TypeError(f"calibration batch {chunk_index} must be a torch.Tensor.")
        if chunk.ndim < 2:
            raise ValueError(
                f"calibration batch {chunk_index} must have a channel axis, got {tuple(chunk.shape)}."
            )
        if chunk.shape[0] == 0:
            raise ValueError(f"calibration batch {chunk_index} is empty.")
    resolved_device = torch.device(device)
    chunks = [chunk.detach().to(resolved_device) for chunk in raw_chunks]

    forward = getattr(model, "forward_features", None)
    if not callable(forward):
        forward = model.forward

    captured: dict[str, torch.Tensor | None] = {path: None for path, _ in splitbn_modules}

    def capture_hook(
        module: nn.Module,
        args: tuple[torch.Tensor, ...],
        *,
        module_path: str,
    ) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError(f"SplitBN module {module_path!r} received no tensor input.")
        captured[module_path] = args[0].detach()

    handles = [
        module.register_forward_pre_hook(
            lambda _module, _args, _path=path: capture_hook(
                _module, _args, module_path=_path
            )
        )
        for path, module in splitbn_modules
    ]
    means: dict[str, torch.Tensor] = {}
    variances: dict[str, torch.Tensor] = {}
    second_moments: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    calibration_rows = 0
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for chunk in chunks:
                for path in captured:
                    captured[path] = None
                for start in range(0, chunk.shape[0], batch_size):
                    batch = chunk[start : start + batch_size]
                    if eval_buffer_source == SPLITBN_SOURCE_DOMAIN_BUFFER:
                        domain_indices = torch.full(
                            (batch.shape[0],),
                            int(domain_index),
                            device=batch.device,
                            dtype=torch.long,
                        )
                        forward_kwargs: dict[str, object] = {
                            "eval_buffer_source": SPLITBN_SOURCE_DOMAIN_BUFFER,
                            "domain_indices": domain_indices,
                        }
                    else:
                        forward_kwargs = {"eval_buffer_source": eval_buffer_source}
                    forward(batch, **forward_kwargs)
                    calibration_rows += int(batch.shape[0])
                    for path, module in splitbn_modules:
                        value = captured[path]
                        if value is None:
                            raise RuntimeError(
                                f"SplitBN module {path!r} was not reached during the "
                                "calibration forward; the model architecture is unsupported."
                            )
                        if value.ndim < 2 or value.shape[1] != module.num_features:
                            raise RuntimeError(
                                f"SplitBN module {path!r} captured malformed pre-normalization "
                                f"input {tuple(value.shape)}."
                            )
                        reduction = (0, *tuple(range(2, value.ndim)))
                        count = value.numel() // int(module.num_features)
                        chunk_mean = value.double().mean(dim=reduction)
                        chunk_variance = value.double().var(dim=reduction, unbiased=False)
                        previous_count = counts.get(path, 0)
                        total_count = previous_count + count
                        if path not in means:
                            means[path] = chunk_mean.clone()
                            second_moments[path] = chunk_variance * count
                        else:
                            previous_mean = means[path]
                            delta = chunk_mean - previous_mean
                            updated_mean = previous_mean + delta * (count / total_count)
                            delta_after = chunk_mean - updated_mean
                            second_moments[path] = (
                                second_moments[path]
                                + chunk_variance * count
                                + delta * delta_after * count
                            )
                            means[path] = updated_mean
                        counts[path] = total_count
        for path, _module in splitbn_modules:
            count = counts.get(path, 0)
            if count <= 0:
                raise RuntimeError(f"SplitBN module {path!r} observed no calibration rows.")
            variances[path] = second_moments[path] / count
        if not any_initialized:
            for path, module in splitbn_modules:
                module.install_target_calibration_buffer(
                    means[path].to(
                        device=module.target_running_mean.device,
                        dtype=module.target_running_mean.dtype,
                    ),
                    variances[path].to(
                        device=module.target_running_var.device,
                        dtype=module.target_running_var.dtype,
                    ),
                    frozen=True,
                )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    def _tensor_digest(*tensors: torch.Tensor) -> str:
        digest = hashlib.sha256()
        for tensor in tensors:
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    return {
        path: {
            "mean_digest": _tensor_digest(means[path]),
            "variance_digest": _tensor_digest(variances[path]),
            "shape": list(means[path].shape),
            "sample_count": int(calibration_rows),
            "observation_count": int(counts[path]),
        }
        for path, _ in splitbn_modules
    }


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
]
