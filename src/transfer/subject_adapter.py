"""Subject-specific downstream adapters on a frozen or fine-tuned trunk.

The adapter intentionally mirrors the compact-baseline contract: ``fit(X, y)``
and ``predict_logit(X) -> (N,)``. It standardizes only from training rows and
can hold out complete temporal groups for early stopping and calibration.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import ceil, isfinite

import numpy as np
import torch
from torch import nn

from baselines.validation import group_disjoint_validation_split
from data.candidate_task import MembershipDecision
from models.adapters import FeatureDomainTransfer
from models.n2p3net import N2P3Net
from models.parameter_groups import SourceParameterPrior
from models.splitbn import (
    SplitBN1d,
    SplitBN2d,
    install_splitbn_target_calibration_buffers,
)
from research.execution import ExpectedSubjectError, SubjectFailureCode
from train.early_stopping import DEFAULT_EARLY_STOP_PATIENCE, PatienceStopper
from transfer.asmmd import asmmd_joint_loss
from transfer.decision_loss import membership_decision_loss
from transfer.within_subject import chronological_validation_split

ADAPTER_POOLING_MODES = frozenset(
    {
        "ms_flatten",
        "global_average",
        "full_unfold",
        "mlp_full_unfold",
        "quadratic_full_unfold",
    }
)


def _validated_prefix_mask(mask: np.ndarray, n_rows: int) -> np.ndarray:
    scope = np.asarray(mask)
    if scope.dtype != np.dtype(bool) or scope.shape != (n_rows,) or not bool(scope.any()):
        raise ValueError("prefix_mask must be a non-empty boolean vector aligned with X.")
    return scope.copy()


@dataclass(frozen=True)
class _AdaptationSource:
    labels: torch.Tensor
    logits: torch.Tensor | None = None
    inputs: torch.Tensor | None = None
    domain_indices: torch.Tensor | None = None


@dataclass
class SubjectAdapterConfig:
    head_kind: str = "linear"
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    val_group_fraction: float | None = 0.1
    val_groups_min: int = 2
    val_groups_max: int = 6
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    early_stop_min_delta: float = 1e-6
    refit_full_prefix: bool = True
    freeze_batchnorm_stats: bool = True
    input_statistics: str = "auto"
    target_stat_weight: float = 0.25
    candidate_loss_weight: float = 0.0
    candidate_loss_temperature: float = 1.0
    source_prior_weight: float = 0.0
    as_mmd_lambda0: float = 0.0
    as_mmd_warmup_epochs: int = 1
    as_mmd_target_weight_min: float = 1.0
    as_mmd_target_weight_max: float = 6.0
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    batchnorm_buffers: str | None = None
    batchnorm_affine: str | None = None
    trainable_weights: str | None = None
    dropout_mode: str = "legacy"
    max_updates: int | None = None
    bn_reestimation_batch_size: int = 128
    validation_behavior: str = "select_epoch"

    def __post_init__(self) -> None:
        if not isfinite(self.source_prior_weight) or self.source_prior_weight < 0:
            raise ValueError("source_prior_weight must be finite and nonnegative.")
        if self.head_kind not in {
            "linear",
            "mlp16",
            "classifier_fine",
            "full_fine",
            "adapter",
        }:
            raise ValueError(
                "head_kind must be linear, mlp16, classifier_fine, full_fine, or adapter."
            )
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0.0:
            raise ValueError("invalid optimization settings.")
        for name, values in {
            "batchnorm_buffers": {None, "source", "calibration", "online"},
            "batchnorm_affine": {None, "frozen", "trainable"},
            "trainable_weights": {None, "none", "head", "non_bn", "spatial", "temporal", "adapter"},
            "dropout_mode": {"legacy", "off", "checkpoint"},
            "validation_behavior": {"select_epoch", "select_epoch_full_trajectory", "diagnostics_only"},
        }.items():
            if getattr(self, name) not in values:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be positive or None")
        if self.max_updates is not None and (
            isinstance(self.max_updates, bool)
            or not isinstance(self.max_updates, (int, np.integer))
        ):
            raise ValueError("max_updates must be an integer")
        if self.bn_reestimation_batch_size < 1:
            raise ValueError("bn_reestimation_batch_size must be positive")
        if self.trainable_weights == "none" and (
            self.batchnorm_affine == "trainable" or self.batchnorm_buffers == "online"
        ):
            raise ValueError("statistics-only adaptation requires frozen affine and fixed buffers")
        if self.trainable_weights in {"spatial", "temporal", "adapter"} and (
            self.batchnorm_buffers is None or self.batchnorm_affine is None
            or self.dropout_mode == "legacy"
        ):
            raise ValueError("structural adaptation requires explicit BN and dropout policies")
        if self.trainable_weights == "adapter" and self.head_kind != "adapter":
            raise ValueError("adapter weights require head_kind='adapter'")
        if self.val_group_fraction is not None and not 0.0 < self.val_group_fraction < 1.0:
            raise ValueError("val_group_fraction must be in (0,1) or None.")
        if self.val_groups_min < 1 or self.val_groups_max < self.val_groups_min:
            raise ValueError("invalid validation group bounds.")
        if self.early_stop_patience < 1 or self.early_stop_min_delta < 0.0:
            raise ValueError("invalid early stopping settings.")
        if not isinstance(self.refit_full_prefix, bool):
            raise ValueError("refit_full_prefix must be boolean.")
        if not isinstance(self.freeze_batchnorm_stats, bool):
            raise ValueError("freeze_batchnorm_stats must be boolean.")
        if self.input_statistics not in {"auto", "source", "target_prefix", "shrinkage"}:
            raise ValueError("input_statistics must be auto, source, target_prefix, or shrinkage.")
        if not 0.0 <= self.target_stat_weight <= 1.0:
            raise ValueError("target_stat_weight must be in [0,1].")
        if not isfinite(self.candidate_loss_weight) or not 0.0 <= self.candidate_loss_weight:
            raise ValueError("candidate_loss_weight must be non-negative.")
        if not isfinite(self.candidate_loss_temperature) or not 0.0 < self.candidate_loss_temperature:
            raise ValueError("candidate_loss_temperature must be positive.")
        if not 0.0 <= self.as_mmd_lambda0:
            raise ValueError("as_mmd_lambda0 must be non-negative.")
        if self.as_mmd_lambda0 > 0.0 and self.batch_size < 2:
            raise ValueError("AS-MMD requires batch_size >= 2 for its unbiased MMD loss.")
        if self.as_mmd_warmup_epochs < 1:
            raise ValueError("as_mmd_warmup_epochs must be at least one.")
        if not 0.0 < self.as_mmd_target_weight_min <= self.as_mmd_target_weight_max:
            raise ValueError("invalid AS-MMD target-weight bounds.")


class SubjectAdapter:
    """Train a subject head on prefix trials of one target subject."""

    def __init__(
        self,
        trunk: N2P3Net | FeatureDomainTransfer,
        *,
        config: SubjectAdapterConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or SubjectAdapterConfig()
        self.device = torch.device("cpu") if device is None else device
        if trunk.n_times is None:
            raise ValueError("subject adapters require a trunk with fixed n_times.")
        if trunk.pooling_mode not in ADAPTER_POOLING_MODES:
            raise ValueError(
                "subject adapters require a single-tensor pooling readout; "
                f"got {trunk.pooling_mode!r}. latency_marginal_contrast is not supported "
                "because its pool is a per-branch ModuleList."
            )
        self.trunk = trunk.to(self.device)
        self.feature_dim = trunk.classifier_features
        # Inner-loop target adapter is a single explicit module owned by the
        # transfer wrapper. There is deliberately no source-domain vocabulary
        # or runtime registration here: an unseen target is not a source-domain
        # lookup problem.
        if self.config.head_kind == "adapter":
            if not isinstance(self.trunk, FeatureDomainTransfer):
                raise ValueError(
                    "head_kind='adapter' requires a FeatureDomainTransfer wrapper "
                    "with an explicit target_adapter."
                )
            self.trunk.set_target_adapter_enabled(True)
            self.trunk.set_target_adapter_trainable(False)
        else:
            if isinstance(self.trunk, FeatureDomainTransfer):
                self.trunk.set_target_adapter_enabled(False)
        self.head: nn.Module
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            if self.config.head_kind == "linear":
                self.head = nn.Linear(self.feature_dim, 2).to(self.device)
            elif self.config.head_kind == "mlp16":
                self.head = nn.Sequential(
                    nn.Linear(self.feature_dim, 16),
                    nn.GELU(),
                    nn.Linear(16, 2),
                ).to(self.device)
            elif self.config.head_kind == "classifier_fine":
                # Reuse the supervised source classifier. Replacing it with a
                # random linear head is the measured 45-trial failure mode.
                self.head = self.trunk.classifier
            else:
                self.head = None  # type: ignore[assignment]
        self._fitted = False
        self.calibration_logits_: np.ndarray | None = None
        self.calibration_labels_: np.ndarray | None = None
        self.calibration_source_ = None
        self.training_pos_weight_: float | None = None
        self.training_prior_: float | None = None
        self.last_history: dict[str, object] = {}
        self._input_mean_t: torch.Tensor | None = None
        self._input_std_t: torch.Tensor | None = None
        self._initial_state = self._clone_state()
        self._source_prior = (
            SourceParameterPrior(self.trunk) if self.config.source_prior_weight > 0 else None
        )
        self.bn_reestimation_: list[dict[str, object]] = []
        self._fit_budget: dict[str, object] = {}
        self._refit_budget: dict[str, object] = {}
        self.feature_cache_key_: str | None = None

    def _features(
        self,
        X: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Gradient enablement follows trainable parameters, not train/eval
        # mode: the adapter path keeps the trunk in eval mode (frozen source
        # statistics, no dropout) while still backpropagating into the target
        # adapter registered on the bank.
        trunk_requires_grad = any(parameter.requires_grad for parameter in self.trunk.parameters())
        if domain_indices is None:
            forward_kwargs: dict[str, object] = self._splitbn_eval_kwargs()
        else:
            # Live source-stream forward: explicit domain routing updates the
            # source-domain SplitBN buffers and must not consume a target
            # buffer selector.
            forward_kwargs = {"domain_indices": domain_indices}
        if channel_mask is not None:
            forward_kwargs["channel_mask"] = channel_mask
        with torch.set_grad_enabled(torch.is_grad_enabled() and trunk_requires_grad):
            if isinstance(self.trunk, FeatureDomainTransfer):
                features, _, _ = self.trunk.feature_views(
                    X,
                    use_target_adapter=self.config.head_kind == "adapter",
                    **forward_kwargs,
                )
                pooled = self.trunk.pool_features(features)
            else:
                features = self.trunk.forward_features(X, **forward_kwargs)
                pooled = self.trunk.pool_features(features)
            if pooled.ndim != 2 or pooled.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"trunk pooling must return (B,{self.feature_dim}) features, "
                    f"got {tuple(pooled.shape)}."
                )
            return pooled

    def _prepare_input(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1:] != (self.trunk.n_channels, self.trunk.n_times):
            raise ValueError(f"X must be (N,{self.trunk.n_channels},{self.trunk.n_times}).")
        if trial_channel_mask is None:
            mask = np.ones(X.shape[:2], dtype=bool)
        else:
            mask = np.asarray(trial_channel_mask)
            if mask.dtype != np.dtype(bool) or mask.shape != X.shape[:2]:
                raise ValueError("trial_channel_mask must be boolean and match X.shape[:2].")
            if not bool(mask.any(axis=1).all()):
                raise ValueError("Every trial must retain at least one observed channel.")
        if not np.isfinite(X[mask]).all():
            raise ValueError("Observed X contains NaN/inf.")
        if self._input_mean_t is None or self._input_std_t is None:
            raise RuntimeError("input preparation requires fitted input statistics.")
        mean = self._input_mean_t.detach().cpu().numpy()
        std = self._input_std_t.detach().cpu().numpy()
        prepared = ((X - mean) / std).astype(np.float32, copy=False)
        np.copyto(prepared, 0.0, where=~mask[:, :, None])
        return torch.from_numpy(np.ascontiguousarray(prepared)).to(self.device)

    @staticmethod
    def _masked_stats(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observed = np.asarray(mask, dtype=bool)[:, :, None]
        counts = np.asarray(mask, dtype=np.float64).sum(axis=0)[None, :, None] * X.shape[2]
        sums = np.sum(X, axis=(0, 2), where=observed, dtype=np.float64, keepdims=True)
        mean = np.divide(sums, np.maximum(counts, 1.0), out=np.zeros_like(sums), where=counts > 0)
        centered = np.where(observed, X - mean, 0.0).astype(np.float64, copy=False)
        variance = np.divide(
            np.sum(centered * centered, axis=(0, 2), keepdims=True),
            np.maximum(counts, 1.0),
        )
        std = np.where(counts > 0, np.sqrt(np.maximum(variance, 0.0)) + 1e-6, 1.0)
        return mean.astype(np.float32), std.astype(np.float32)

    def _validated_external_stats(
        self, value: np.ndarray, *, name: str, require_positive: bool = False
    ) -> np.ndarray:
        """Validate checkpoint input statistics against the physical channel axis."""

        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite numeric channel vector.") from error
        if array.ndim == 0 and self.trunk.n_channels == 1:
            array = array.reshape(1)
        elif array.ndim == 3 and array.shape == (1, self.trunk.n_channels, 1):
            array = array.reshape(self.trunk.n_channels)
        elif array.ndim != 1 or array.shape != (self.trunk.n_channels,):
            raise ValueError(
                f"{name} must have shape ({self.trunk.n_channels},) or "
                f"(1,{self.trunk.n_channels},1), got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/inf.")
        if require_positive and np.any(array <= 0.0):
            raise ValueError(f"{name} must contain strictly positive values.")
        return array.reshape(1, self.trunk.n_channels, 1)

    def _set_trainable(self, *, trunk_trainable: bool) -> None:
        if self.config.trainable_weights is not None:
            groups = self.trunk.adaptation_parameter_groups()
            mode = self.config.trainable_weights
            selected = {
                "none": set(), "head": {"readout"},
                "spatial": {"readout", "spatial"},
                "temporal": {"readout", "temporal"},
                "adapter": {"readout", "target_adapter"},
                "non_bn": set(groups) - {"normalization"},
            }[mode]
            if selected - set(groups):
                raise ValueError("requested adaptation group is absent from the model")
            if self.config.batchnorm_affine == "trainable":
                selected.add("normalization")
            for name, parameters in groups.items():
                for parameter in parameters:
                    parameter.requires_grad = name in selected
            if isinstance(self.trunk, FeatureDomainTransfer):
                self.trunk.set_target_adapter_enabled(mode == "adapter")
            if self.head is not None:
                for parameter in self.head.parameters():
                    parameter.requires_grad = self.config.trainable_weights != "none"
            return
        for parameter in self.trunk.parameters():
            parameter.requires_grad = trunk_trainable
        if self.config.head_kind == "adapter":
            if not isinstance(self.trunk, FeatureDomainTransfer):
                raise RuntimeError("adapter mode requires FeatureDomainTransfer.")
            self.trunk.set_target_adapter_enabled(True)
            self.trunk.set_target_adapter_trainable(True)
        elif isinstance(self.trunk, FeatureDomainTransfer):
            self.trunk.set_target_adapter_enabled(False)
            self.trunk.set_target_adapter_trainable(False)
        if self.head is not None:
            for parameter in self.head.parameters():
                parameter.requires_grad = True
        if self.config.batchnorm_affine is not None:
            for module in self.trunk.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    for parameter in module.parameters(recurse=False):
                        parameter.requires_grad = self.config.batchnorm_affine == "trainable"

    def _live_features_required(self) -> bool:
        if self.config.trainable_weights is None:
            return self.config.head_kind in {"full_fine", "adapter"}
        return (
            self.config.trainable_weights in {"non_bn", "spatial", "temporal", "adapter"}
            or self.config.batchnorm_affine == "trainable"
            or self.config.batchnorm_buffers == "online"
            or self.config.dropout_mode == "checkpoint"
        )

    def fitted_input_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self._input_mean_t is None or self._input_std_t is None:
            raise RuntimeError("input statistics require fit")
        return self._input_mean.copy(), self._input_std.copy()

    def input_statistics_record(self) -> dict[str, object]:
        mean, std = self.fitted_input_statistics()
        return {
            "source": self.input_statistics_source_,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(),
        }

    def fitted_state_digest(self) -> str:
        """Bind weights, all BN buffers and actual input statistics for derived features."""
        digest = hashlib.sha256()
        for scope, state in self._state_dict().items():
            for name, value in sorted(state.items()):
                digest.update(f"{scope}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(
                    value.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        for values in self.fitted_input_statistics():
            digest.update(values.tobytes())
        return digest.hexdigest()

    def _feature_cache_key(self, inputs: torch.Tensor, channel_mask: torch.Tensor | None = None) -> str:
        digest = hashlib.sha256(self.fitted_state_digest().encode("ascii"))
        digest.update(str(tuple(inputs.shape)).encode("ascii"))
        digest.update(inputs.detach().cpu().contiguous().numpy().tobytes())
        if channel_mask is not None and self._core_trunk().observed_channel_norm:
            digest.update(channel_mask.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _reestimate_batchnorm(
        self, inputs: torch.Tensor, channel_mask: torch.Tensor | None = None,
    ) -> list[dict[str, object]]:
        """Install exact population moments in execution order, with upstream BN frozen."""
        if self.config.batchnorm_buffers != "calibration":
            return []
        if self._splitbn_modules():
            raise ValueError("E1 calibration BN moments require ordinary BatchNorm, not SplitBN")
        self.trunk.eval()
        modules = {
            name: module
            for name, module in self.trunk.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        }
        order: list[str] = []

        def ordered(name):
            def hook(_module, _args):
                if name not in order:
                    order.append(name)

            return hook

        handles = [
            module.register_forward_pre_hook(ordered(name)) for name, module in modules.items()
        ]
        try:
            with torch.no_grad():
                self._features(inputs[:1], channel_mask=None if channel_mask is None else channel_mask[:1])
        finally:
            for handle in handles:
                handle.remove()
        records: list[dict[str, object]] = []
        with torch.no_grad():
            for name in order:
                module = modules[name]
                if not module.track_running_stats:
                    raise ValueError("calibration BN requires persistent running statistics")
                count = 0
                total = torch.zeros(module.num_features, dtype=torch.float64, device=self.device)
                squares = torch.zeros_like(total)

                def collect(_module, args):
                    nonlocal count, total, squares
                    values = args[0].detach().double().movedim(1, 0).flatten(1)
                    count += values.shape[1]
                    total += values.sum(1)
                    squares += values.square().sum(1)

                handle = module.register_forward_pre_hook(collect)
                try:
                    for start in range(0, len(inputs), self.config.bn_reestimation_batch_size):
                        stop = start + self.config.bn_reestimation_batch_size
                        self._features(inputs[start:stop], channel_mask=(
                            None if channel_mask is None else channel_mask[start:stop]))
                finally:
                    handle.remove()
                if count < 1:
                    raise RuntimeError(f"BN layer {name} received no calibration activations")
                mean = total / count
                variance = (squares / count - mean.square()).clamp_min(0.0)
                module.running_mean.copy_(mean.to(module.running_mean.dtype))
                module.running_var.copy_(variance.to(module.running_var.dtype))
                module.num_batches_tracked.zero_()
                mean_values = module.running_mean.cpu().numpy()
                variance_values = module.running_var.cpu().numpy()
                records.append(
                    {
                        "layer": name,
                        "count": count,
                        "calibration_rows": len(inputs),
                        "variance_convention": "population_ddof0",
                        "mean": mean_values.tolist(),
                        "variance": variance_values.tolist(),
                        "sha256": hashlib.sha256(
                            mean_values.tobytes() + variance_values.tobytes()
                        ).hexdigest(),
                        "num_batches_tracked": 0,
                    }
                )
        return records

    @property
    def training(self) -> bool:
        return self.trunk.training

    def _core_trunk(self) -> N2P3Net:
        return self.trunk.core if isinstance(self.trunk, FeatureDomainTransfer) else self.trunk

    def _splitbn_config(self):
        return getattr(self._core_trunk(), "splitbn_config", None)

    def _splitbn_modules(self) -> tuple[nn.Module, ...]:
        return tuple(
            module for module in self.trunk.modules() if isinstance(module, (SplitBN1d, SplitBN2d))
        )

    def _splitbn_eval_kwargs(self) -> dict[str, str]:
        """Return the checkpoint-declared unseen-target normalization selector."""

        splitbn_config = self._splitbn_config()
        if splitbn_config is None:
            return {}
        return {"eval_buffer_source": splitbn_config.target_buffer_source}

    def _splitbn_digest(self, *, target: bool) -> str | None:
        """Deterministic SHA-256 over the SplitBN buffers selected by *target*."""

        modules = self._splitbn_modules()
        if not modules:
            return None
        if target:
            names = (
                "pooled_running_mean",
                "pooled_running_var",
                "pooled_num_batches_tracked",
                "target_running_mean",
                "target_running_var",
                "target_buffer_initialized",
            )
        else:
            names = ("running_mean", "running_var", "num_batches_tracked")
        digest = hashlib.sha256()
        for path, module in self.trunk.named_modules():
            if not isinstance(module, (SplitBN1d, SplitBN2d)):
                continue
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            for name in names:
                tensor = getattr(module, name)
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(b"\0")
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                digest.update(b"\0")
        return digest.hexdigest()

    def _prepare_live_source_stream(
        self,
        source_X: np.ndarray,
        source_labels: np.ndarray,
        source_domain_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """Validate live source rows and return normalized-ready arrays plus stats."""

        splitbn_config = self._splitbn_config()
        if splitbn_config is None:
            raise ValueError("live AS-MMD requires a SplitBN checkpoint.")
        if self.config.head_kind != "full_fine":
            raise ValueError("live AS-MMD requires head_kind='full_fine'.")
        if self.config.input_mean is None or self.config.input_std is None:
            raise ValueError("live AS-MMD requires checkpoint input statistics.")
        source_X = np.asarray(source_X, dtype=np.float32)
        if source_X.ndim != 3 or source_X.shape[1:] != (
            self.trunk.n_channels,
            self.trunk.n_times,
        ):
            raise ValueError(
                "source_X must have shape "
                f"(N,{self.trunk.n_channels},{self.trunk.n_times}), got {source_X.shape}."
            )
        if not np.isfinite(source_X).all():
            raise ValueError("source_X contains NaN/inf.")
        if len(source_X) < 2:
            raise ValueError("live AS-MMD requires at least two source rows.")
        source_labels = np.asarray(source_labels).astype(np.int64)
        if source_labels.ndim != 1 or source_labels.shape != (len(source_X),):
            raise ValueError("source_labels must be a 1D vector aligned with source_X.")
        if bool(((source_labels < 0) | (source_labels > 1)).any()):
            raise ValueError("source_labels must be binary 0/1.")
        source_domain_indices = np.asarray(source_domain_indices)
        if source_domain_indices.dtype != np.int64:
            raise ValueError("source_domain_indices must use int64.")
        if source_domain_indices.ndim != 1 or source_domain_indices.shape != (len(source_X),):
            raise ValueError("source_domain_indices must be a 1D vector aligned with source_X.")
        domain_count = int(splitbn_config.domain_count)
        if bool(((source_domain_indices < 0) | (source_domain_indices >= domain_count)).any()):
            raise ValueError(
                "source_domain_indices must lie in [0, domain_count) for the "
                "checkpoint SplitBN contract."
            )
        source_mean = self._validated_external_stats(self.config.input_mean, name="input_mean")
        source_std = self._validated_external_stats(
            self.config.input_std, name="input_std", require_positive=True
        )
        return (
            source_X,
            source_labels,
            source_domain_indices,
            domain_count,
            source_mean,
            source_std,
        )

    def _forward_source_batch(
        self,
        X: torch.Tensor,
        domain_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward source rows while updating only per-domain SplitBN buffers.

        Live AS-MMD declares source-domain SplitBN updates as its adaptation
        mechanism. The predeclared target buffer must stay frozen, so the
        pooled running statistics are suppressed for ``pooled_source``
        checkpoints. Ordinary BatchNorm is kept in eval mode by
        ``_set_training_mode`` and is never touched here.
        """

        modules = self._splitbn_modules()
        if not modules:
            raise RuntimeError("live source-stream forward requires SplitBN modules.")
        previous_flags = [getattr(module, "_freeze_pooled_buffer", False) for module in modules]
        previous_modes = [module.training for module in modules]
        try:
            for module in modules:
                module._freeze_pooled_buffer = True
                module.train()
            return self._forward_head(X, domain_indices=domain_indices)
        finally:
            for module, previous_flag, previous_mode in zip(
                modules, previous_flags, previous_modes, strict=True
            ):
                module._freeze_pooled_buffer = previous_flag
                module.train(previous_mode)

    def _set_training_mode(self, *, trunk_trainable: bool, live_splitbn: bool = False) -> None:
        if self.config.batchnorm_buffers is not None:
            if live_splitbn or self._splitbn_modules():
                raise ValueError("explicit E1 BN policy is not a SplitBN training procedure")
            self.trunk.train(self._live_features_required())
            for module in self.trunk.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train(self.config.batchnorm_buffers == "online")
                elif isinstance(module, nn.modules.dropout._DropoutNd):
                    module.train(self.config.dropout_mode == "checkpoint")
            if self.head is not None:
                self.head.train(self.config.trainable_weights != "none")
            return
        if self.config.head_kind == "adapter":
            # Deterministic frozen trunk: source BatchNorm statistics and no
            # dropout. The target adapter has no train/eval-dependent layers,
            # so eval-mode forward with enabled gradients is exact.
            self.trunk.eval()
            return
        if live_splitbn:
            if not trunk_trainable:
                raise RuntimeError("live SplitBN training requires a trainable trunk.")
            # Live AS-MMD uses the predeclared target buffer for every target
            # batch and explicit source-domain updates for every source batch.
            # Ordinary BatchNorm therefore always stays in eval mode, while the
            # SplitBN modules are put in eval mode here and are only toggled to
            # train mode inside ``_forward_source_batch``. This is independent
            # of ``freeze_batchnorm_stats`` by design.
            self.trunk.train(True)
            for module in self.trunk.modules():
                if isinstance(
                    module,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        SplitBN1d,
                        SplitBN2d,
                    ),
                ):
                    module.eval()
            if self.head is not None:
                self.head.train()
            return
        self.trunk.train(trunk_trainable)
        splitbn_modules = (SplitBN1d, SplitBN2d)
        if (
            trunk_trainable
            and not self.config.freeze_batchnorm_stats
            and any(isinstance(module, splitbn_modules) for module in self.trunk.modules())
        ):
            raise ValueError(
                "full_fine cannot update SplitBN on an unseen target domain; "
                "use frozen predeclared target buffers or an explicit target-calibration "
                "buffer procedure."
            )
        if trunk_trainable and self.config.freeze_batchnorm_stats:
            for module in self.trunk.modules():
                if isinstance(
                    module,
                    (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, *splitbn_modules),
                ):
                    module.eval()
        if self.head is not None:
            self.head.train()
        if self.config.dropout_mode != "legacy":
            for module in self.trunk.modules():
                if isinstance(module, nn.modules.dropout._DropoutNd):
                    module.train(self.config.dropout_mode == "checkpoint")

    def _clear_failed_fit(self) -> None:
        self._fitted = False
        self.last_history = {}
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        self.training_pos_weight_ = None
        self.training_prior_ = None
        self._input_mean_t = self._input_std_t = None
        self._input_mean = self._input_std = None
        self._source_input_mean = self._source_input_std = None
        self.bn_reestimation_ = []
        self._fit_budget = {}
        self._refit_budget = {}
        self.feature_cache_key_ = None
        self.trunk.zero_grad(set_to_none=True)
        self.trunk.eval()
        if self.head is not None:
            self.head.zero_grad(set_to_none=True)
            self.head.eval()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        group_ids: np.ndarray | None = None,
        training_mask: np.ndarray | None = None,
        validation_mask: np.ndarray | None = None,
        prefix_mask: np.ndarray | None = None,
        repetition_indices: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
        decision_groups: Sequence[MembershipDecision] | None = None,
        decision_targets: Sequence[int] | None = None,
        decision_weights: np.ndarray | None = None,
        source_X: np.ndarray | None = None,
        source_domain_indices: np.ndarray | None = None,
        source_logits: np.ndarray | torch.Tensor | None = None,
        source_labels: np.ndarray | None = None,
    ) -> SubjectAdapter:
        """Fit within a calibration scope; chronological refit requires its explicit mask.

        The prefix can include inner embargo rows, but never an outer test row.
        Its membership must come from the caller's calibration split.
        """
        self._clear_failed_fit()
        try:
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y).astype(np.int64)
            if X.ndim != 3 or X.shape[1:] != (self.trunk.n_channels, self.trunk.n_times):
                raise ValueError(
                    f"X must be (N,{self.trunk.n_channels},{self.trunk.n_times}), got {X.shape}."
                )
            if y.ndim != 1 or len(y) != len(X) or set(np.unique(y).tolist()) != {0, 1}:
                raise ValueError("subject adapter requires binary labels {0,1} aligned with X.")
            if (decision_groups is None) != (decision_targets is None):
                raise ValueError("decision groups and targets must be supplied together")
            if decision_weights is not None and decision_groups is None:
                raise ValueError("decision weights require decision groups")
            if self.config.candidate_loss_weight > 0 and decision_groups is None:
                raise ValueError("candidate_loss_weight requires physical decision_groups")
            if decision_groups is not None and (
                self.config.batchnorm_buffers == "online"
                or (self.config.batchnorm_buffers is None and not self.config.freeze_batchnorm_stats)
            ):
                raise ValueError("joint decision batches require frozen support BN statistics")
            has_input_mean = self.config.input_mean is not None
            has_input_std = self.config.input_std is not None
            if has_input_mean != has_input_std:
                raise ValueError("input_mean and input_std must be supplied together.")
            supplied_splits = sum(value is not None for value in (group_ids, repetition_indices))
            if validation_mask is not None:
                supplied_splits += 1
            if supplied_splits > 1:
                raise ValueError(
                    "provide only one of group_ids, validation_mask, or repetition_indices."
                )
            if training_mask is not None and validation_mask is None:
                raise ValueError("training_mask requires validation_mask.")
            if trial_channel_mask is None:
                observed_mask = np.ones(X.shape[:2], dtype=bool)
            else:
                observed_mask = np.asarray(trial_channel_mask)
                if observed_mask.dtype != np.dtype(bool) or observed_mask.shape != X.shape[:2]:
                    raise ValueError("trial_channel_mask must be boolean and match X.shape[:2].")
                if not bool(observed_mask.any(axis=1).all()):
                    raise ValueError("Every trial must retain at least one observed channel.")
            if not np.isfinite(X[observed_mask]).all():
                raise ValueError("Observed X contains NaN/inf.")

            prefix_scope = (
                np.ones(len(X), dtype=bool)
                if prefix_mask is None
                else _validated_prefix_mask(prefix_mask, len(X))
            )
            prefix_rows = np.flatnonzero(prefix_scope)
            train_mask = prefix_scope.copy()
            val_mask = np.zeros(len(X), dtype=bool)
            validation_source = None
            if repetition_indices is not None:
                repetition_indices = np.asarray(repetition_indices)
                if repetition_indices.shape != (len(X),):
                    raise ValueError("repetition_indices must align with X.")
                chronological = chronological_validation_split(
                    repetition_indices[prefix_rows],
                    y[prefix_rows],
                    fraction=self.config.val_group_fraction,
                    min_repetitions=self.config.val_groups_min,
                    max_repetitions=self.config.val_groups_max,
                )
                train_mask[prefix_rows] = chronological.train_mask
                val_mask[prefix_rows] = chronological.validation_mask
                validation_source = "chronological_prefix_validation"
            elif validation_mask is not None:
                validation_mask = np.asarray(validation_mask)
                if validation_mask.dtype != np.dtype(bool) or validation_mask.shape != (len(X),):
                    raise ValueError("validation_mask must be boolean and align with X.")
                val_mask = validation_mask.copy()
                train_mask = (
                    prefix_scope & ~val_mask
                    if training_mask is None
                    else np.asarray(training_mask).copy()
                )
                if train_mask.dtype != np.dtype(bool) or train_mask.shape != (len(X),):
                    raise ValueError("training_mask must be boolean and align with X.")
                if bool((train_mask & val_mask).any()):
                    raise ValueError("training_mask and validation_mask must be disjoint.")
                validation_source = "chronological_prefix_validation"
                if not bool(train_mask.any()) or not bool(val_mask.any()):
                    raise ValueError("validation_mask must retain both training and validation rows.")
            elif group_ids is not None:
                group_ids = np.asarray(group_ids)
                if group_ids.shape != (len(X),):
                    raise ValueError("group_ids must align with X.")
                split = group_disjoint_validation_split(
                    group_ids[prefix_rows],
                    fraction=self.config.val_group_fraction,
                    min_groups=self.config.val_groups_min,
                    max_groups=self.config.val_groups_max,
                    seed=self.config.seed,
                )
                train_mask[prefix_rows] = split.train_mask
                val_mask[prefix_rows] = split.validation_mask
                validation_source = "group_disjoint_prefix_validation"
            if bool(((train_mask | val_mask) & ~prefix_scope).any()):
                raise ValueError("training_mask and validation_mask must stay inside prefix_mask.")
            if (
                validation_source == "chronological_prefix_validation"
                and self.config.refit_full_prefix
                and self.config.validation_behavior != "diagnostics_only"
                and bool(val_mask.any())
                and prefix_mask is None
            ):
                raise ValueError("chronological refit requires an explicit prefix_mask before fitting.")
            if len(np.unique(y[train_mask])) != 2:
                raise ExpectedSubjectError(
                    SubjectFailureCode.INSUFFICIENT_CLASSES,
                    stage="adapter_fit",
                    detail="training split does not contain both binary classes",
                )
            if (
                self.config.max_updates is not None
                and val_mask.any()
                and self.config.validation_behavior != "diagnostics_only"
            ):
                raise ValueError("fixed updates requires fixed-budget fitting without epoch selection")
            if self.config.trainable_weights is not None:
                self._load_state_dict(self._initial_state)
            fit_started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self._fit_budget = {}
            self._refit_budget = {}
            self.bn_reestimation_ = []
            self.feature_cache_key_ = None

            target_mean, target_std = self._masked_stats(X[train_mask], observed_mask[train_mask])
            statistics_mode = self.config.input_statistics
            if statistics_mode == "auto":
                statistics_mode = "source" if has_input_mean and has_input_std else "target_prefix"
            if statistics_mode in {"source", "shrinkage"} and not (has_input_mean and has_input_std):
                raise ValueError(
                    f"input_statistics={statistics_mode!r} requires checkpoint statistics."
                )
            if statistics_mode in {"source", "shrinkage"}:
                source_mean = self._validated_external_stats(self.config.input_mean, name="input_mean")
                source_std = self._validated_external_stats(
                    self.config.input_std, name="input_std", require_positive=True
                )
                self._source_input_mean = source_mean
                self._source_input_std = source_std
            if statistics_mode == "source":
                self._input_mean, self._input_std = source_mean, source_std
            elif statistics_mode == "target_prefix":
                self._input_mean, self._input_std = target_mean, target_std
            else:
                alpha = float(self.config.target_stat_weight)
                self._input_mean = (1.0 - alpha) * source_mean + alpha * target_mean
                variance = (1.0 - alpha) * (
                    source_std**2 + (source_mean - self._input_mean) ** 2
                ) + alpha * (target_std**2 + (target_mean - self._input_mean) ** 2)
                self._input_std = np.sqrt(np.maximum(variance, 1e-12))
            self.input_statistics_source_ = statistics_mode
            self._input_std = np.where(self._input_std < 1e-6, 1.0, self._input_std)
            self._input_mean_t = torch.as_tensor(self._input_mean, device=self.device)
            self._input_std_t = torch.as_tensor(self._input_std, device=self.device)
            standardized = ((X - self._input_mean) / self._input_std).astype(np.float32, copy=False)
            np.copyto(standardized, 0.0, where=~observed_mask[:, :, None])
            Xt = torch.from_numpy(np.ascontiguousarray(standardized)).to(self.device)
            yt = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
            channel_masks = torch.tensor(observed_mask, device=self.device)

            asmmd_active = bool(self.config.as_mmd_lambda0 > 0.0)
            as_mmd_input_mode = "none"
            source_logits_t: torch.Tensor | None = None
            source_labels_t: torch.Tensor | None = None
            source_Xt: torch.Tensor | None = None
            source_domain_t: torch.Tensor | None = None
            source_count = 0
            source_domain_count: int | None = None
            if asmmd_active:
                frozen_supplied = (
                    source_logits is not None
                    and source_labels is not None
                    and source_X is None
                    and source_domain_indices is None
                )
                live_supplied = (
                    source_logits is None
                    and source_labels is not None
                    and source_X is not None
                    and source_domain_indices is not None
                )
                if frozen_supplied == live_supplied:
                    raise ValueError(
                        "as_mmd_lambda0 > 0 requires exactly one AS-MMD input mode: "
                        "frozen source_logits/source_labels or live "
                        "source_X/source_labels/source_domain_indices."
                    )
                if frozen_supplied:
                    as_mmd_input_mode = "frozen_logits"
                    source_logits_t = (
                        torch.as_tensor(source_logits, device=self.device).float().detach()
                    )
                    source_labels_t = torch.as_tensor(source_labels, device=self.device).long()
                    if source_logits_t.ndim != 2 or source_labels_t.shape != (
                        source_logits_t.shape[0],
                    ):
                        raise ValueError(
                            "source_logits/source_labels must be aligned (N, classes)/(N,)."
                        )
                    if bool(((source_labels_t < 0) | (source_labels_t > 1)).any()):
                        raise ValueError("source_labels must be binary 0/1.")
                    source_count = source_logits_t.shape[0]
                else:
                    as_mmd_input_mode = "source_stream"
                    (
                        source_X,
                        source_labels,
                        source_domain_indices,
                        source_domain_count,
                        live_source_mean,
                        live_source_std,
                    ) = self._prepare_live_source_stream(
                        source_X,
                        source_labels,
                        source_domain_indices,
                    )
                    source_count = len(source_X)
                    standardized_source = ((source_X - live_source_mean) / live_source_std).astype(
                        np.float32, copy=False
                    )
                    source_Xt = torch.from_numpy(np.ascontiguousarray(standardized_source)).to(
                        self.device
                    )
                    source_labels_t = torch.from_numpy(np.ascontiguousarray(source_labels)).to(
                        self.device
                    )
                    source_domain_t = torch.from_numpy(np.ascontiguousarray(source_domain_indices)).to(
                        self.device
                    )

            adaptation_source = (
                _AdaptationSource(
                    labels=source_labels_t,
                    logits=source_logits_t,
                    inputs=source_Xt,
                    domain_indices=source_domain_t,
                )
                if asmmd_active
                else None
            )
            torch.manual_seed(self.config.seed)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(self.config.seed)
            trunk_trainable = self.config.head_kind == "full_fine"
            live_splitbn = as_mmd_input_mode == "source_stream"
            self._set_trainable(trunk_trainable=trunk_trainable)
            initial_state = self._clone_state()
            parameters = [p for p in self.trunk.parameters() if p.requires_grad]
            if self.head is not None:
                parameters.extend(p for p in self.head.parameters() if p.requires_grad)
            parameters = list({id(parameter): parameter for parameter in parameters}.values())
            if not parameters and self.config.trainable_weights != "none":
                raise RuntimeError("no trainable parameters in subject adapter.")
            optimizer = (
                torch.optim.AdamW(
                    parameters,
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay,
                )
                if parameters
                else None
            )
            train_counts = np.bincount(y[train_mask], minlength=2)
            pos_weight = float(train_counts[0] / max(train_counts[1], 1))
            self.training_pos_weight_ = pos_weight
            self.training_prior_ = float(y[train_mask].mean())
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=self.device))

            # Materialize train/validation row blocks once. The training loop then
            # uses device-side index_select with the same permutation values, so the
            # epoch batches are identical to the previous numpy-indexed path without
            # any per-batch host synchronization.
            n_train = int(train_mask.sum())
            train_rows = torch.as_tensor(
                np.flatnonzero(train_mask), dtype=torch.long, device=self.device
            )
            Xt_train = Xt.index_select(0, train_rows)
            yt_train = yt.index_select(0, train_rows)
            train_channels = channel_masks.index_select(0, train_rows)
            val_rows = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=self.device)
            Xt_val = Xt.index_select(0, val_rows) if len(val_rows) else Xt[:0]
            yt_val = yt.index_select(0, val_rows) if len(val_rows) else yt[:0]
            val_channels = channel_masks.index_select(0, val_rows)
            self.bn_reestimation_ = self._reestimate_batchnorm(Xt_train, train_channels)
            initial_bn_reestimation = self.bn_reestimation_

            splitbn_config = self._splitbn_config()
            if (
                live_splitbn
                and splitbn_config is not None
                and splitbn_config.target_buffer_source == "target_calibration"
                and any(
                    not bool(module.target_buffer_initialized) for module in self._splitbn_modules()
                )
            ):
                install_splitbn_target_calibration_buffers(
                    self.trunk,
                    Xt_train,
                    device=self.device,
                    eval_buffer_source="source_domain",
                    domain_index=0,
                )
            if (
                live_splitbn
                and splitbn_config is not None
                and splitbn_config.target_buffer_source == "target_calibration"
                and any(
                    not bool(module.target_buffer_initialized) for module in self._splitbn_modules()
                )
            ):
                raise RuntimeError("target_calibration SplitBN buffers were not installed and frozen.")
            splitbn_source_buffer_digest_before = self._splitbn_digest(target=False)
            splitbn_target_buffer_digest_before = self._splitbn_digest(target=True)

            train_inputs = Xt_train
            val_inputs = Xt_val
            # The adapter path cannot cache trunk features: they change with the
            # target adapter parameters on every optimizer step.
            cache_features = not self._live_features_required()
            if cache_features:
                # A frozen trunk is a fixed feature transform. Keeping it in train
                # mode would update BatchNorm and apply fresh dropout every epoch;
                # caching eval-mode features is both the intended contract and the
                # dominant launch reduction for subject-head adaptation.
                self.trunk.eval()
                self.feature_cache_key_ = self._feature_cache_key(Xt_train, train_channels)
                with torch.no_grad():
                    train_inputs = self._features(Xt_train, channel_mask=train_channels)
                    val_inputs = self._features(Xt_val, channel_mask=val_channels) if len(Xt_val) else train_inputs[:0]
                del Xt, Xt_train, Xt_val
            live_forward = self._live_features_required()
            initial_feature_cache_key = self.feature_cache_key_

            best_state = None
            best_epoch_count = None
            best_val = float("inf")
            stopper = PatienceStopper(self.config.early_stop_patience, self.config.early_stop_min_delta)
            train_losses: list[float] = []
            candidate_losses: list[dict[str, object]] = []
            asmmd_losses: list[dict[str, object]] = []
            decision_context = self._scoped_decisions(
                decision_groups, decision_targets, decision_weights, np.flatnonzero(train_mask), y,
            )
            decision_rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 1]))

            val_losses: list[float | None] = []
            validation_curve: list[dict[str, object]] = []
            budget = self._new_budget(n_train)
            self._fit_budget = budget
            for epoch_index in range(self._training_epoch_count(n_train) if parameters else 0):
                self._set_training_mode(
                    trunk_trainable=trunk_trainable,
                    live_splitbn=live_splitbn,
                )
                epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                epoch_draws = 0
                for rows in self._training_batches(n_train, epoch_index=epoch_index):
                    if (
                        self.config.max_updates is not None
                        and budget["updates"] >= self.config.max_updates
                    ):
                        break
                    xb = train_inputs.index_select(0, rows)
                    yb = yt_train.index_select(0, rows)
                    # Target batches always consume the predeclared target buffer.
                    # In live mode the trunk is in train mode for its parameters
                    # and dropout, while SplitBN modules stay in eval mode here and
                    # therefore do not update any running statistics.
                    logits, decision_loss, decision_diagnostics = self._joint_training_batch(
                        train_inputs, train_channels, rows, decision_context, decision_rng,
                        live_forward=live_forward,
                    )
                    loss, asmmd_diag = self._training_loss(
                        logits,
                        yb,
                        loss_fn=loss_fn,
                        source=adaptation_source,
                        epoch=epoch_index,
                        target_count=n_train,
                        pos_weight=pos_weight,
                    )
                    if asmmd_diag is not None:
                        asmmd_losses.append(asmmd_diag)
                    if self._source_prior is not None:
                        loss = loss + self.config.source_prior_weight * self._source_prior.loss()
                    loss = loss + self.config.candidate_loss_weight * decision_loss
                    if decision_diagnostics is not None:
                        candidate_losses.append(decision_diagnostics)
                    if not bool(torch.isfinite(loss)):
                        raise ExpectedSubjectError(
                            SubjectFailureCode.NONFINITE_OPTIMIZATION,
                            stage="adapter_fit",
                            detail="training loss became non-finite",
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().float() * len(xb)
                    self._record_batch(budget, rows, epoch_index=epoch_index)
                    if decision_diagnostics is not None:
                        budget["candidate_updates"] += 1
                        budget["candidate_draws"] += decision_diagnostics["draws"]
                        budget["draws"] += decision_diagnostics["draws"]
                    epoch_draws += len(xb)

                val_loss = None
                if val_mask.any():
                    self.trunk.eval()
                    if self.head is not None:
                        self.head.eval()
                    with torch.inference_mode():
                        logits = (
                            self._forward_head(val_inputs, channel_mask=val_channels) if live_forward else self.head(val_inputs)
                        )
                        val_loss_tensor = loss_fn(logits, yt_val).detach().float()
                    train_value, val_loss = (
                        torch.stack((epoch_loss / max(epoch_draws, 1), val_loss_tensor)).cpu().tolist()
                    )
                    train_losses.append(float(train_value))
                    val_loss = float(val_loss)
                    if self.config.validation_behavior == "diagnostics_only":
                        from sklearn.metrics import roc_auc_score

                        scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float64)
                        finite = np.isfinite(val_loss) and np.isfinite(scores).all()
                        two_classes = len(np.unique(y[val_mask])) == 2
                        loss_value = val_loss if finite else None
                        val_losses.append(loss_value)
                        validation_curve.append(
                            {
                                "epoch": epoch_index + 1,
                                "updates": budget["updates"],
                                "draws": budget["draws"],
                                "loss": loss_value,
                                "auc": float(roc_auc_score(y[val_mask], scores))
                                if finite and two_classes
                                else None,
                                "status": "finite" if finite else "nonfinite_validation_diagnostic",
                                "used_for_epoch_selection": False,
                            }
                        )
                    else:
                        if not np.isfinite(val_loss):
                            raise ExpectedSubjectError(
                                SubjectFailureCode.NONFINITE_OPTIMIZATION,
                                stage="adapter_validation",
                                detail="epoch selection loss became non-finite",
                            )
                        val_losses.append(val_loss)
                        validation_curve.append({"epoch": epoch_index + 1, "updates": budget["updates"],
                            "draws": budget["draws"], "loss": val_loss, "status": "finite",
                            "used_for_epoch_selection": True})
                        if stopper.observe(val_loss):
                            best_val = val_loss
                            best_epoch_count = epoch_index + 1
                            best_state = self._clone_state()
                        if stopper.exhausted and self.config.validation_behavior == "select_epoch":
                            break
                else:
                    train_losses.append(float(epoch_loss.cpu()) / max(epoch_draws, 1))

            if best_state is not None:
                self._load_state_dict(best_state)

            self.trunk.eval()
            if self.head is not None:
                self.head.eval()
            if val_mask.any():
                with torch.inference_mode():
                    logits = self._forward_head(val_inputs, channel_mask=val_channels) if live_forward else self.head(val_inputs)
                self.calibration_logits_ = (
                    (logits[:, 1] - logits[:, 0]).detach().cpu().numpy().astype(np.float64)
                )
                self.calibration_labels_ = y[np.flatnonzero(val_mask)]
                self.calibration_source_ = validation_source or "validation"
            else:
                self.calibration_logits_ = None
                self.calibration_labels_ = None
                self.calibration_source_ = None
            refit_losses: list[float] = []
            refit_candidate_losses: list[dict[str, object]] = []
            refit_asmmd_losses: list[dict[str, object]] = []
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._fit_budget["wall_seconds"] = time.perf_counter() - fit_started
            if (
                validation_source == "chronological_prefix_validation"
                and self.config.refit_full_prefix
                and best_epoch_count is not None
            ):
                refit_losses, refit_candidate_losses, refit_asmmd_losses = self._refit_full_prefix(
                    X,
                    y,
                    observed_mask,
                    prefix_mask=prefix_scope,
                    initial_state=initial_state,
                    epochs=best_epoch_count,
                    statistics_mode=statistics_mode,
                    decision_groups=decision_groups,
                    decision_targets=decision_targets,
                    decision_weights=decision_weights,
                    live_splitbn=live_splitbn,
                    adaptation_source=adaptation_source,
                )
            splitbn_source_buffer_digest_after = self._splitbn_digest(target=False)
            splitbn_target_buffer_digest_after = self._splitbn_digest(target=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.last_history = {
                "source_parameter_prior": (None if self._source_prior is None else {
                    **self._source_prior.record(), "weight": self.config.source_prior_weight,
                    "final_unweighted_penalty": float(self._source_prior.loss().detach().cpu()),
                }),
                "fit_budget": self._finalize_budget(self._fit_budget),
                "refit_budget": self._finalize_budget(self._refit_budget),
                "input_statistics": self.input_statistics_record(),
                "bn_reestimation": self.bn_reestimation_,
                "initial_bn_reestimation": initial_bn_reestimation,
                "feature_cache_key": self.feature_cache_key_,
                "initial_feature_cache_key": initial_feature_cache_key,
                "fitted_state_sha256": self.fitted_state_digest(),
                "trainable_parameter_names": [
                    name for name, p in self.trunk.named_parameters() if p.requires_grad
                ],
                "parameter_groups": {
                    name: {"parameters": sum(p.numel() for p in parameters),
                           "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad)}
                    for name, parameters in self.trunk.adaptation_parameter_groups().items()
                },
                "resources": {
                    "requested_device": str(self.device),
                    "effective_device": str(self.device),
                    "fallback": None,
                    "torch_threads": torch.get_num_threads(),
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else None,
                },
                "fit_wall_seconds": time.perf_counter() - fit_started,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "validation_behavior": self.config.validation_behavior,
                "validation_curve": validation_curve,
                "best_val_loss": best_val if best_val != float("inf") else None,
                "best_epoch": best_epoch_count,
                "stopping": stopper.record(
                    enabled=bool(val_mask.any()) and self.config.validation_behavior != "diagnostics_only",
                    stop_at_patience=self.config.validation_behavior == "select_epoch",
                ),
                "refit_full_prefix": bool(refit_losses),
                "refit_epochs": len(refit_losses),
                "refit_train_losses": refit_losses,
                "fit_scope": {
                    "input_rows": len(X),
                    "prefix_mask_explicit": prefix_mask is not None,
                    "prefix_rows": int(prefix_scope.sum()),
                    "prefix_mask_sha256": hashlib.sha256(prefix_scope.tobytes()).hexdigest(),
                    "training_rows": int(train_mask.sum()),
                    "validation_rows": int(val_mask.sum()),
                    "inner_excluded_prefix_rows": int((prefix_scope & ~(train_mask | val_mask)).sum()),
                    "outside_prefix_rows": int((~prefix_scope).sum()),
                    "refit_rows": int(prefix_scope.sum()) if refit_losses else 0,
                },
                "candidate_loss_active": self.config.candidate_loss_weight > 0 and decision_groups is not None,
                "decision_training": (None if decision_context is None else {
                    "score_kind": "membership_llr_sum/1",
                    "update_policy": "one_joint_step_row_and_complete_decision_streams",
                    "requested_groups": len(decision_groups),
                    "authorized_groups": len(decision_context[0]),
                    "budget_excluded_groups": len(decision_groups) - len(decision_context[0]),
                    "sampling_probability": decision_context[1].tolist(),
                    "eligible_draws": sum(value["eligible"] for value in candidate_losses),
                }),
                "candidate_loss_weight": self.config.candidate_loss_weight,
                "candidate_loss_temperature": self.config.candidate_loss_temperature,
                "candidate_losses": candidate_losses,
                "refit_candidate_losses": refit_candidate_losses,
                "as_mmd_active": bool(asmmd_active),
                "as_mmd_input_mode": as_mmd_input_mode,
                "as_mmd_losses": asmmd_losses,
                "refit_as_mmd_losses": refit_asmmd_losses,
                "as_mmd_lambda0": self.config.as_mmd_lambda0,
                "as_mmd_warmup_epochs": self.config.as_mmd_warmup_epochs,
                "source_stream_count": (
                    int(source_count) if as_mmd_input_mode == "source_stream" else None
                ),
                "source_domain_count": (
                    source_domain_count if as_mmd_input_mode == "source_stream" else None
                ),
                "splitbn_source_buffer_digest_before": splitbn_source_buffer_digest_before,
                "splitbn_source_buffer_digest_after": splitbn_source_buffer_digest_after,
                "splitbn_target_buffer_digest_before": splitbn_target_buffer_digest_before,
                "splitbn_target_buffer_digest_after": splitbn_target_buffer_digest_after,
            }
            self._fitted = True
            return self
        finally:
            if not self._fitted:
                self._clear_failed_fit()

    def _refit_full_prefix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        observed_mask: np.ndarray,
        *,
        prefix_mask: np.ndarray,
        initial_state: dict[str, dict[str, torch.Tensor]],
        epochs: int,
        statistics_mode: str,
        decision_groups: Sequence[MembershipDecision] | None,
        decision_targets: Sequence[int] | None,
        decision_weights: np.ndarray | None,
        live_splitbn: bool = False,
        adaptation_source: _AdaptationSource | None = None,
    ) -> tuple[list[float], list[dict[str, object]], list[dict[str, object]]]:
        """Refit only explicitly authorized calibration rows, including inner embargo."""

        scope = _validated_prefix_mask(prefix_mask, len(X))
        refit_started = time.perf_counter()
        decision_context = self._scoped_decisions(
            decision_groups, decision_targets, decision_weights, np.flatnonzero(scope), y,
        )
        decision_rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 1]))
        X, y, observed_mask = X[scope], y[scope], observed_mask[scope]
        self._load_state_dict(initial_state)
        if statistics_mode != "source":
            target_mean, target_std = self._masked_stats(X, observed_mask)
            if statistics_mode == "target_prefix":
                self._input_mean, self._input_std = target_mean, target_std
            else:
                alpha = float(self.config.target_stat_weight)
                source_mean = self._source_input_mean
                source_std = self._source_input_std
                self._input_mean = (1.0 - alpha) * source_mean + alpha * target_mean
                variance = (1.0 - alpha) * (
                    source_std**2 + (source_mean - self._input_mean) ** 2
                ) + alpha * (target_std**2 + (target_mean - self._input_mean) ** 2)
                self._input_std = np.sqrt(np.maximum(variance, 1e-12))
        self._input_mean_t = torch.as_tensor(self._input_mean, device=self.device)
        self._input_std_t = torch.as_tensor(self._input_std, device=self.device)
        standardized = ((X - self._input_mean) / self._input_std).astype(np.float32, copy=False)
        np.copyto(standardized, 0.0, where=~observed_mask[:, :, None])
        inputs = torch.from_numpy(np.ascontiguousarray(standardized)).to(self.device)
        labels = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
        channel_masks = torch.tensor(observed_mask, device=self.device)
        self.bn_reestimation_ = self._reestimate_batchnorm(inputs, channel_masks)

        splitbn_config = self._splitbn_config()
        if (
            live_splitbn
            and splitbn_config is not None
            and splitbn_config.target_buffer_source == "target_calibration"
            and any(
                not bool(module.target_buffer_initialized) for module in self._splitbn_modules()
            )
        ):
            # ``initial_state`` predates the frozen target-calibration install,
            # so the full-prefix refit must reinstall it before its first
            # target-buffer forward.
            install_splitbn_target_calibration_buffers(
                self.trunk,
                inputs,
                device=self.device,
                eval_buffer_source="source_domain",
                domain_index=0,
            )

        trunk_trainable = self.config.head_kind == "full_fine"
        self._set_trainable(trunk_trainable=trunk_trainable)
        if not self._live_features_required():
            self.trunk.eval()
            self.feature_cache_key_ = self._feature_cache_key(inputs, channel_masks)
            with torch.no_grad():
                inputs = self._features(inputs, channel_mask=channel_masks)
        live_forward = self._live_features_required()
        parameters = [parameter for parameter in self.trunk.parameters() if parameter.requires_grad]
        if self.head is not None:
            parameters.extend(
                parameter for parameter in self.head.parameters() if parameter.requires_grad
            )
        parameters = list({id(parameter): parameter for parameter in parameters}.values())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        counts = np.bincount(y, minlength=2)
        pos_weight = float(counts[0] / max(counts[1], 1))
        self.training_pos_weight_ = pos_weight
        self.training_prior_ = float(y.mean())
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=self.device))
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(self.config.seed)
        losses: list[float] = []
        candidate_losses: list[dict[str, object]] = []
        asmmd_losses: list[dict[str, object]] = []
        budget = self._new_budget(len(y))
        self._refit_budget = budget
        for epoch_index in range(int(epochs)):
            self._set_training_mode(
                trunk_trainable=trunk_trainable,
                live_splitbn=live_splitbn,
            )
            epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for rows in self._training_batches(len(y), epoch_index=epoch_index):
                xb = inputs.index_select(0, rows)
                yb = labels.index_select(0, rows)
                logits, decision_loss, decision_diagnostics = self._joint_training_batch(
                    inputs, channel_masks, rows, decision_context, decision_rng,
                    live_forward=live_forward,
                )
                loss, asmmd_diag = self._training_loss(
                    logits,
                    yb,
                    loss_fn=loss_fn,
                    source=adaptation_source,
                    epoch=epoch_index,
                    target_count=len(y),
                    pos_weight=pos_weight,
                )
                if asmmd_diag is not None:
                    asmmd_losses.append(asmmd_diag)
                if self._source_prior is not None:
                    loss = loss + self.config.source_prior_weight * self._source_prior.loss()
                loss = loss + self.config.candidate_loss_weight * decision_loss
                if decision_diagnostics is not None:
                    candidate_losses.append(decision_diagnostics)
                if not bool(torch.isfinite(loss)):
                    raise ExpectedSubjectError(
                        SubjectFailureCode.NONFINITE_OPTIMIZATION,
                        stage="adapter_refit",
                        detail="full-prefix refit loss became non-finite",
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().float() * len(xb)
                self._record_batch(budget, rows, epoch_index=epoch_index)
                if decision_diagnostics is not None:
                    budget["candidate_updates"] += 1
                    budget["candidate_draws"] += decision_diagnostics["draws"]
                    budget["draws"] += decision_diagnostics["draws"]
            losses.append(float(epoch_loss.cpu()) / len(y))
        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        budget["wall_seconds"] = time.perf_counter() - refit_started
        return losses, candidate_losses, asmmd_losses

    def _training_epoch_count(self, count: int) -> int:
        if self.config.max_updates is None:
            return self.config.epochs
        batches = ceil(count / self.config.batch_size)
        if self.config.as_mmd_lambda0 > 0 and count % self.config.batch_size == 1 and batches > 1:
            batches -= 1
        return ceil(self.config.max_updates / batches)

    def _new_budget(self, count: int) -> dict[str, object]:
        return {
            "n_fit": count,
            "requested_batch_size": self.config.batch_size,
            "effective_batch_size": min(self.config.batch_size, count),
            "budget_mode": "updates" if self.config.max_updates is not None else "epochs",
            "requested_epochs": self.config.epochs,
            "requested_updates": self.config.max_updates,
            "updates": 0,
            "draws": 0,
            "candidate_updates": 0,
            "candidate_draws": 0,
            "batch_sizes": [],
            "epoch_draws": [],
            "epoch_permutation_sha256": [],
            "batch_rows_sha256": [],
            "seed": self.config.seed,
            "order_algorithm": "numpy_pcg64_seedsequence_seed_epoch",
        }

    def _record_batch(
        self, budget: dict[str, object], rows: torch.Tensor, *, epoch_index: int
    ) -> None:
        count, row_digest = self._batch_index_records[id(rows)]
        budget["updates"] += 1
        budget["draws"] += count
        budget["batch_sizes"].append(count)
        budget["batch_rows_sha256"].append(row_digest)
        if len(budget["epoch_draws"]) <= epoch_index:
            budget["epoch_draws"].append(0)
            permutation = (
                np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch_index]))
                .permutation(budget["n_fit"])
                .astype(np.int64)
            )
            budget["epoch_permutation_sha256"].append(
                hashlib.sha256(permutation.tobytes()).hexdigest()
            )
        budget["epoch_draws"][epoch_index] += count

    @staticmethod
    def _finalize_budget(budget: dict[str, object]) -> dict[str, object]:
        if not budget:
            return {"updates": 0, "draws": 0, "batch_sizes": []}
        hashes = budget["batch_rows_sha256"]
        return {
            **budget,
            "batch_sequence_sha256": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
        }

    def _training_batches(self, count: int, *, epoch_index: int = 0) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, epoch_index]))
        permutation = rng.permutation(count).astype(np.int64)
        host_batches = tuple(
            permutation[start : start + self.config.batch_size]
            for start in range(0, count, self.config.batch_size)
        )
        # Unbiased MMD cannot estimate a singleton tail. Keep every row by
        # joining that tail to the previous batch in both selection and refit.
        if (
            self.config.as_mmd_lambda0 > 0.0
            and len(host_batches) > 1
            and len(host_batches[-1]) == 1
        ):
            host_batches = (*host_batches[:-2], np.concatenate(host_batches[-2:]))
        batches = tuple(torch.from_numpy(values).to(self.device) for values in host_batches)
        self._batch_index_records = {
            id(batch): (len(values), hashlib.sha256(values.tobytes()).hexdigest())
            for batch, values in zip(batches, host_batches, strict=True)
        }
        return batches

    def _training_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss_fn: nn.Module,
        source: _AdaptationSource | None,
        epoch: int,
        target_count: int,
        pos_weight: float,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        if source is None:
            if self.config.as_mmd_lambda0 > 0.0:
                raise RuntimeError("AS-MMD training requires its validated source context.")
            return loss_fn(logits, labels), None
        source_count = len(source.labels)
        rows = torch.randperm(source_count, device=self.device)[: self.config.batch_size]
        if source.inputs is not None:
            assert source.domain_indices is not None
            source_logits = self._forward_source_batch(
                source.inputs.index_select(0, rows), source.domain_indices.index_select(0, rows)
            )
        else:
            assert source.logits is not None
            source_logits = source.logits.index_select(0, rows)
        return asmmd_joint_loss(
            source_logits,
            source.labels.index_select(0, rows),
            logits,
            labels,
            epoch=epoch,
            warmup_epochs=self.config.as_mmd_warmup_epochs,
            base_weight=self.config.as_mmd_lambda0,
            source_count=source_count,
            target_count=target_count,
            target_weight_min=self.config.as_mmd_target_weight_min,
            target_weight_max=self.config.as_mmd_target_weight_max,
            pos_weight=pos_weight,
        )

    def _scoped_decisions(self, groups, targets, weights, rows, labels):
        if groups is None:
            return None
        groups, targets = tuple(groups), tuple(targets)
        if len(groups) != len(targets):
            raise ValueError("decision targets must align with groups")
        mass = np.ones(len(groups), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        if (mass.shape != (len(groups),) or not np.isfinite(mass).all() or np.any(mass < 0)
                or (len(mass) and mass.sum() <= 0)):
            raise ValueError("decision weights must be finite nonnegative requested-group masses")
        lookup = np.full(len(labels), -1, dtype=np.int64)
        lookup[rows] = np.arange(len(rows))
        scoped, retained = [], []
        for index, (group, target) in enumerate(zip(groups, targets, strict=True)):
            if not isinstance(group, MembershipDecision):
                raise ValueError("decision_groups require physical MembershipDecision records")
            observed = group.evidence_indices >= 0
            original = group.evidence_indices[observed]
            if len(original) and original.max() >= len(labels):
                raise ValueError("decision evidence indices exceed the supplied epoch matrix")
            if np.any(lookup[original] < 0):
                continue  # A group crossing a label-budget boundary is not an authorized request.
            group.validate_epoch_labels(labels, target)
            mapped = group.evidence_indices.copy()
            mapped[observed] = lookup[original]
            scoped.append((replace(group, evidence_indices=mapped), int(target)))
            retained.append(mass[index])
        probability = np.asarray(retained, dtype=float)
        if len(probability) and probability.sum() > 0:
            probability /= probability.sum()
        elif len(probability):
            raise ValueError("authorized decisions have no positive risk mass")
        return scoped, probability

    def _joint_training_batch(self, inputs, channel_masks, rows, decisions, rng, *, live_forward):
        if decisions is None:
            data = inputs.index_select(0, rows)
            logits = (self._forward_head(data, channel_mask=channel_masks.index_select(0, rows))
                      if live_forward else self.head(data))
            return logits, logits.sum() * 0.0, None
        selected = None
        extra = rows[:0]
        if decisions is not None and decisions[0]:
            groups, probability = decisions
            selected = groups[int(rng.choice(len(groups), p=probability))]
            group, _ = selected
            extra = torch.as_tensor(group.evidence_indices[group.evidence_indices >= 0].copy(),
                                    dtype=torch.long, device=self.device)
        workset = torch.cat((rows, extra))
        parts = []
        for chunk in workset.split(self.config.batch_size):
            data = inputs.index_select(0, chunk)
            parts.append(self._forward_head(data, channel_mask=channel_masks.index_select(0, chunk))
                         if live_forward else self.head(data))
        logits = torch.cat(parts)
        zero = logits.sum() * 0.0
        if selected is None:
            return logits[:len(rows)], zero, None
        group, target = selected
        observed = np.flatnonzero(group.evidence_indices >= 0)
        aligned = logits.new_zeros(len(group.event_ids)).index_copy(
            0, torch.as_tensor(observed, device=self.device),
            logits[len(rows):, 1] - logits[len(rows):, 0],
        )
        result = membership_decision_loss(aligned, group, target,
                                         temperature=self.config.candidate_loss_temperature)
        loss = zero if result.loss is None else result.loss
        diagnostics = {"draws": len(extra), "requested": 1, "eligible": int(result.loss is not None),
                       "status": result.status, "unweighted_loss": float(loss.detach().cpu()),
                       "score_kind": result.score_kind}
        return logits[:len(rows)], loss, diagnostics

    def _forward_head(
        self,
        X: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._features(X, domain_indices=domain_indices, channel_mask=channel_mask)
        if self.head is not None:
            return self.head(features)
        return self.trunk.classifier(features)

    def _state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        payload = {"trunk": self.trunk.state_dict()}
        if self.head is not None and self.config.head_kind != "classifier_fine":
            payload["head"] = self.head.state_dict()
        return payload

    def _clone_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            scope: {key: value.detach().clone() for key, value in state.items()}
            for scope, state in self._state_dict().items()
        }

    def _load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.trunk.load_state_dict(state["trunk"])
        if self.head is not None and "head" in state:
            self.head.load_state_dict(state["head"])

    def predict_features(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Expose fitted head inputs without updating normalization or BN state."""
        if not self._fitted:
            raise RuntimeError("fit before predict_features.")
        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        Xt = self._prepare_input(X, trial_channel_mask)
        with torch.inference_mode():
            features = self._features(Xt, channel_mask=(None if trial_channel_mask is None else
                                      torch.tensor(trial_channel_mask, device=self.device)))
        values = features.cpu().numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE,
                stage="adapter_predict_features",
                detail="adapter produced non-finite features",
            )
        return values

    def predict_logit(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit before predict_logit.")
        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        if self._input_mean_t is None or self._input_std_t is None:
            raise RuntimeError("predict_logit requires fitted input statistics.")
        Xt = self._prepare_input(X, trial_channel_mask)
        with torch.inference_mode():
            logits = self._forward_head(Xt, channel_mask=(None if trial_channel_mask is None else
                                        torch.tensor(trial_channel_mask, device=self.device)))
        values = (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE,
                stage="adapter_predict",
                detail="adapter produced non-finite logits",
            )
        return values

    def parameter_count(self) -> int:
        parameters = [parameter for parameter in self.trunk.parameters() if parameter.requires_grad]
        if self.head is not None:
            parameters.extend(
                parameter for parameter in self.head.parameters() if parameter.requires_grad
            )
        unique = {id(parameter): parameter for parameter in parameters}
        return sum(parameter.numel() for parameter in unique.values())

    def total_parameter_count(self) -> int:
        parameters = list(self.trunk.parameters())
        if self.head is not None:
            parameters.extend(self.head.parameters())
        unique = {id(parameter): parameter for parameter in parameters}
        return sum(parameter.numel() for parameter in unique.values())
