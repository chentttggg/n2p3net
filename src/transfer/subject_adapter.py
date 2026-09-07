"""Subject-specific downstream adapters on a frozen or fine-tuned trunk.

The adapter intentionally mirrors the compact-baseline contract: ``fit(X, y)``
and ``predict_logit(X) -> (N,)``. It standardizes only from training rows and
can hold out complete temporal groups for early stopping and calibration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from baselines.validation import group_disjoint_validation_split
from models.adapters import FeatureDomainTransfer
from models.n2p3net import N2P3Net
from models.splitbn import (
    SplitBN1d,
    SplitBN2d,
    install_splitbn_target_calibration_buffers,
)
from research.execution import ExpectedSubjectError, SubjectFailureCode
from transfer.asmmd import asmmd_joint_loss
from transfer.candidate_set_loss import row_column_candidate_loss
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
    early_stop_patience: int = 6
    early_stop_min_delta: float = 1e-6
    refit_full_prefix: bool = True
    freeze_batchnorm_stats: bool = True
    input_statistics: str = "auto"
    target_stat_weight: float = 0.25
    candidate_loss_weight: float = 0.0
    candidate_loss_temperature: float = 1.0
    as_mmd_lambda0: float = 0.0
    as_mmd_warmup_epochs: int = 1
    as_mmd_target_weight_min: float = 1.0
    as_mmd_target_weight_max: float = 6.0
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None

    def __post_init__(self) -> None:
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
        if not 0.0 <= self.candidate_loss_weight:
            raise ValueError("candidate_loss_weight must be non-negative.")
        if not 0.0 < self.candidate_loss_temperature:
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

    def _features(
        self,
        X: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
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
        with torch.set_grad_enabled(trunk_requires_grad):
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
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
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
        candidate_ids: np.ndarray | None = None,
        selection_ids: np.ndarray | None = None,
        target_rows: np.ndarray | None = None,
        target_columns: np.ndarray | None = None,
        candidate_loss_num_rows: int = 6,
        candidate_loss_num_columns: int = 6,
        source_X: np.ndarray | None = None,
        source_domain_indices: np.ndarray | None = None,
        source_logits: np.ndarray | torch.Tensor | None = None,
        source_labels: np.ndarray | None = None,
    ) -> SubjectAdapter:
        """Fit within a calibration scope; chronological refit requires its explicit mask.

        The prefix can include inner embargo rows, but never an outer test row.
        Its membership must come from the caller's calibration split.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim != 3 or X.shape[1:] != (self.trunk.n_channels, self.trunk.n_times):
            raise ValueError(
                f"X must be (N,{self.trunk.n_channels},{self.trunk.n_times}), got {X.shape}."
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        if y.ndim != 1 or len(y) != len(X) or set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("subject adapter requires binary labels {0,1} aligned with X.")
        candidate_metadata = (
            candidate_ids,
            selection_ids,
            target_rows,
            target_columns,
        )
        if self.config.candidate_loss_weight > 0.0:
            if any(value is None for value in candidate_metadata):
                raise ValueError(
                    "candidate_loss_weight > 0 requires candidate_ids, selection_ids, "
                    "target_rows, and target_columns."
                )
            candidate_ids = np.asarray(candidate_ids)
            selection_ids = np.asarray(selection_ids)
            target_rows = np.asarray(target_rows).astype(np.int64)
            target_columns = np.asarray(target_columns).astype(np.int64)
            if candidate_ids.ndim != 1 or candidate_ids.shape != (len(X),):
                raise ValueError("candidate_ids must align with X.")
            if selection_ids.ndim != 1 or selection_ids.shape != (len(X),):
                raise ValueError("selection_ids must align with X.")
            if target_rows.shape != (len(X),) or target_columns.shape != (len(X),):
                raise ValueError("target_rows/target_columns must align with X.")
            if candidate_loss_num_rows < 2 or candidate_loss_num_columns < 2:
                raise ValueError("candidate_loss_num_rows/columns must be at least two.")
            if not np.isfinite(target_rows).all() or not np.isfinite(target_columns).all():
                raise ValueError("target_rows/target_columns must be finite.")
        elif any(value is not None for value in candidate_metadata):
            raise ValueError("candidate metadata requires candidate_loss_weight > 0.")
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
            parameters.extend(self.head.parameters())
        parameters = list({id(parameter): parameter for parameter in parameters}.values())
        if not parameters:
            raise RuntimeError("no trainable parameters in subject adapter.")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
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
        val_rows = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=self.device)
        Xt_val = Xt.index_select(0, val_rows) if len(val_rows) else Xt[:0]
        yt_val = yt.index_select(0, val_rows) if len(val_rows) else yt[:0]

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
        cache_features = not trunk_trainable and self.config.head_kind != "adapter"
        if cache_features:
            # A frozen trunk is a fixed feature transform. Keeping it in train
            # mode would update BatchNorm and apply fresh dropout every epoch;
            # caching eval-mode features is both the intended contract and the
            # dominant launch reduction for subject-head adaptation.
            self.trunk.eval()
            with torch.no_grad():
                train_inputs = self._features(Xt_train)
                val_inputs = self._features(Xt_val) if len(Xt_val) else train_inputs[:0]
            del Xt, Xt_train, Xt_val
        live_forward = trunk_trainable or self.config.head_kind == "adapter"

        best_state = None
        best_epoch_count = None
        best_val = float("inf")
        patience = self.config.early_stop_patience
        train_losses: list[float] = []
        candidate_losses: list[dict[str, object]] = []
        asmmd_losses: list[dict[str, object]] = []
        candidate_active = self.config.candidate_loss_weight > 0.0
        train_row_indices = np.flatnonzero(train_mask)

        val_losses: list[float] = []
        for epoch_index in range(self.config.epochs):
            self._set_training_mode(
                trunk_trainable=trunk_trainable,
                live_splitbn=live_splitbn,
            )
            epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for rows in self._training_batches(n_train):
                xb = train_inputs.index_select(0, rows)
                yb = yt_train.index_select(0, rows)
                # Target batches always consume the predeclared target buffer.
                # In live mode the trunk is in train mode for its parameters
                # and dropout, while SplitBN modules stay in eval mode here and
                # therefore do not update any running statistics.
                logits = self._forward_head(xb) if live_forward else self.head(xb)
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

            if candidate_active:
                candidate_losses.append(
                    self._candidate_loss_update(
                        train_inputs,
                        row_indices=train_row_indices,
                        candidate_ids=candidate_ids,
                        selection_ids=selection_ids,
                        target_rows=target_rows,
                        target_columns=target_columns,
                        num_rows=candidate_loss_num_rows,
                        num_columns=candidate_loss_num_columns,
                        live_forward=live_forward,
                        optimizer=optimizer,
                        stage="adapter_fit_candidate_loss",
                    )
                )

            val_loss = None
            if val_mask.any():
                self.trunk.eval()
                if self.head is not None:
                    self.head.eval()
                with torch.inference_mode():
                    logits = (
                        self._forward_head(val_inputs) if live_forward else self.head(val_inputs)
                    )
                    val_loss_tensor = loss_fn(logits, yt_val).detach().float()
                train_value, val_loss = (
                    torch.stack((epoch_loss / max(n_train, 1), val_loss_tensor)).cpu().tolist()
                )
                train_losses.append(float(train_value))
                val_loss = float(val_loss)
                val_losses.append(val_loss)
                if val_loss < best_val - self.config.early_stop_min_delta:
                    best_val = val_loss
                    best_epoch_count = epoch_index + 1
                    patience = self.config.early_stop_patience
                    best_state = self._clone_state()
                else:
                    patience -= 1
                if patience <= 0:
                    break
            else:
                train_losses.append(float(epoch_loss.cpu()) / max(n_train, 1))

        if best_state is not None:
            self._load_state_dict(best_state)

        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        if val_mask.any():
            with torch.inference_mode():
                logits = self._forward_head(val_inputs) if live_forward else self.head(val_inputs)
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
                candidate_ids=candidate_ids,
                selection_ids=selection_ids,
                target_rows=target_rows,
                target_columns=target_columns,
                candidate_loss_num_rows=candidate_loss_num_rows,
                candidate_loss_num_columns=candidate_loss_num_columns,
                live_splitbn=live_splitbn,
                adaptation_source=adaptation_source,
            )
        splitbn_source_buffer_digest_after = self._splitbn_digest(target=False)
        splitbn_target_buffer_digest_after = self._splitbn_digest(target=True)
        self.last_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val if best_val != float("inf") else None,
            "best_epoch": best_epoch_count,
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
            "candidate_loss_active": bool(candidate_active),
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
        candidate_ids: np.ndarray | None,
        selection_ids: np.ndarray | None,
        target_rows: np.ndarray | None,
        target_columns: np.ndarray | None,
        candidate_loss_num_rows: int,
        candidate_loss_num_columns: int,
        live_splitbn: bool = False,
        adaptation_source: _AdaptationSource | None = None,
    ) -> tuple[list[float], list[dict[str, object]], list[dict[str, object]]]:
        """Refit only explicitly authorized calibration rows, including inner embargo."""

        scope = _validated_prefix_mask(prefix_mask, len(X))
        X, y, observed_mask = X[scope], y[scope], observed_mask[scope]
        candidate_ids, selection_ids, target_rows, target_columns = (
            None if values is None else np.asarray(values)[scope]
            for values in (candidate_ids, selection_ids, target_rows, target_columns)
        )
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
        if not trunk_trainable and self.config.head_kind != "adapter":
            self.trunk.eval()
            with torch.no_grad():
                inputs = self._features(inputs)
        live_forward = trunk_trainable or self.config.head_kind == "adapter"
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
        candidate_active = self.config.candidate_loss_weight > 0.0
        if candidate_active and any(
            value is None for value in (candidate_ids, selection_ids, target_rows, target_columns)
        ):
            raise ValueError("candidate-loss refit requires complete candidate metadata.")
        row_indices = np.arange(len(y), dtype=np.int64)
        for epoch_index in range(int(epochs)):
            self._set_training_mode(
                trunk_trainable=trunk_trainable,
                live_splitbn=live_splitbn,
            )
            epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for rows in self._training_batches(len(y)):
                xb = inputs.index_select(0, rows)
                yb = labels.index_select(0, rows)
                logits = self._forward_head(xb) if live_forward else self.head(xb)
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
            if candidate_active:
                candidate_losses.append(
                    self._candidate_loss_update(
                        inputs,
                        row_indices=row_indices,
                        candidate_ids=candidate_ids,
                        selection_ids=selection_ids,
                        target_rows=target_rows,
                        target_columns=target_columns,
                        num_rows=candidate_loss_num_rows,
                        num_columns=candidate_loss_num_columns,
                        live_forward=live_forward,
                        optimizer=optimizer,
                        stage="adapter_refit_candidate_loss",
                    )
                )
            losses.append(float(epoch_loss.cpu()) / len(y))
        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        return losses, candidate_losses, asmmd_losses

    def _training_batches(self, count: int) -> tuple[torch.Tensor, ...]:
        permutation = torch.randperm(count, device=self.device)
        batches = permutation.split(self.config.batch_size)
        # Unbiased MMD cannot estimate a singleton tail. Keep every row by
        # joining that tail to the previous batch in both selection and refit.
        if self.config.as_mmd_lambda0 > 0.0 and len(batches) > 1 and len(batches[-1]) == 1:
            batches = (*batches[:-2], torch.cat(batches[-2:]))
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

    def _candidate_loss_update(
        self,
        inputs: torch.Tensor,
        *,
        row_indices: np.ndarray,
        candidate_ids: np.ndarray,
        selection_ids: np.ndarray,
        target_rows: np.ndarray,
        target_columns: np.ndarray,
        num_rows: int,
        num_columns: int,
        live_forward: bool,
        optimizer: torch.optim.Optimizer,
        stage: str,
    ) -> dict[str, object]:
        """One full-batch row/column decision step on the selected rows."""

        if len(row_indices) == 0:
            raise ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE,
                stage=stage,
                detail="candidate loss received an empty training row set",
            )
        candidate_tensor = torch.as_tensor(
            np.asarray(candidate_ids[row_indices], dtype=np.int64),
            device=inputs.device,
        )
        target_row_tensor = torch.as_tensor(
            target_rows[row_indices].astype(np.int64),
            device=inputs.device,
        )
        target_column_tensor = torch.as_tensor(
            target_columns[row_indices].astype(np.int64),
            device=inputs.device,
        )
        selection_values = np.asarray(selection_ids[row_indices]).astype(str).tolist()
        logits = self._forward_head(inputs) if live_forward else self.head(inputs)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE,
                stage=stage,
                detail="candidate loss requires two-class logits",
            )
        evidence = logits[:, 1] - logits[:, 0]
        loss, diagnostics = row_column_candidate_loss(
            evidence,
            candidate_tensor,
            selection_values,
            target_row_tensor,
            target_column_tensor,
            num_rows=num_rows,
            num_columns=num_columns,
            temperature=self.config.candidate_loss_temperature,
        )
        if diagnostics.complete_count == 0:
            raise ExpectedSubjectError(
                SubjectFailureCode.NUMERICAL_FAILURE,
                stage=stage,
                detail=(
                    "candidate loss found no complete row/column calibration decision: "
                    f"{diagnostics.record()}"
                ),
            )
        weighted = self.config.candidate_loss_weight * loss
        if not bool(torch.isfinite(weighted)):
            raise ExpectedSubjectError(
                SubjectFailureCode.NONFINITE_OPTIMIZATION,
                stage=stage,
                detail="candidate decision loss became non-finite",
            )
        optimizer.zero_grad(set_to_none=True)
        weighted.backward()
        optimizer.step()
        return {
            "loss": float(weighted.detach().cpu()),
            "unweighted_loss": float(loss.detach().cpu()),
            **diagnostics.record(),
        }

    def _forward_head(
        self,
        X: torch.Tensor,
        *,
        domain_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._features(X, domain_indices=domain_indices)
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
            features = self._features(Xt)
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
            logits = self._forward_head(Xt)
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
