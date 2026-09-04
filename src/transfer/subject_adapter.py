"""Subject-specific downstream adapters on a frozen or fine-tuned trunk.

The adapter intentionally mirrors the compact-baseline contract: ``fit(X, y)``
and ``predict_logit(X) -> (N,)``. It standardizes only from training rows and
can hold out complete temporal groups for early stopping and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from baselines.validation import group_disjoint_validation_split
from models.adapters import FeatureDomainTransfer
from models.n2p3net import N2P3Net
from models.splitbn import SplitBN1d, SplitBN2d
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
            raise ValueError(
                "input_statistics must be auto, source, target_prefix, or shrinkage."
            )
        if not 0.0 <= self.target_stat_weight <= 1.0:
            raise ValueError("target_stat_weight must be in [0,1].")
        if not 0.0 <= self.candidate_loss_weight:
            raise ValueError("candidate_loss_weight must be non-negative.")
        if not 0.0 < self.candidate_loss_temperature:
            raise ValueError("candidate_loss_temperature must be positive.")
        if not 0.0 <= self.as_mmd_lambda0:
            raise ValueError("as_mmd_lambda0 must be non-negative.")
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

    def _features(self, X: torch.Tensor) -> torch.Tensor:
        # Gradient enablement follows trainable parameters, not train/eval
        # mode: the adapter path keeps the trunk in eval mode (frozen source
        # statistics, no dropout) while still backpropagating into the target
        # adapter registered on the bank.
        trunk_requires_grad = any(
            parameter.requires_grad for parameter in self.trunk.parameters()
        )
        splitbn_eval_kwargs = self._splitbn_eval_kwargs()
        with torch.set_grad_enabled(trunk_requires_grad):
            if isinstance(self.trunk, FeatureDomainTransfer):
                features, _, _ = self.trunk.feature_views(
                    X,
                    use_target_adapter=self.config.head_kind == "adapter",
                    **splitbn_eval_kwargs,
                )
                pooled = self.trunk.pool_features(features)
            else:
                features = self.trunk.forward_features(X, **splitbn_eval_kwargs)
                pooled = self.trunk.pool_features(features)
            # AdaptiveAvgPool1d(1) retains a singleton temporal axis, whereas
            # the other readouts already return (B,D).  Normalize the feature
            # shape at this boundary so every head sees the same contract.
            if self.trunk.pooling_mode == "global_average":
                if pooled.ndim != 3 or pooled.shape[-1] != 1:
                    raise RuntimeError(
                        "global_average pooling must return features shaped (B,D,1)."
                    )
                pooled = pooled.squeeze(-1)
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
    def _masked_stats(
        X: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def _splitbn_eval_kwargs(self) -> dict[str, str]:
        """Return the checkpoint-declared unseen-target normalization selector."""

        core = self.trunk.core if isinstance(self.trunk, FeatureDomainTransfer) else self.trunk
        splitbn_config = getattr(core, "splitbn_config", None)
        if splitbn_config is None:
            return {}
        return {"eval_buffer_source": splitbn_config.target_buffer_source}

    def _set_training_mode(self, *, trunk_trainable: bool) -> None:
        if self.config.head_kind == "adapter":
            # Deterministic frozen trunk: source BatchNorm statistics and no
            # dropout. The target adapter has no train/eval-dependent layers,
            # so eval-mode forward with enabled gradients is exact.
            self.trunk.eval()
            return
        self.trunk.train(trunk_trainable)
        splitbn_modules = (SplitBN1d, SplitBN2d)
        if trunk_trainable and not self.config.freeze_batchnorm_stats and any(
            isinstance(module, splitbn_modules) for module in self.trunk.modules()
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
        repetition_indices: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
        candidate_ids: np.ndarray | None = None,
        selection_ids: np.ndarray | None = None,
        target_rows: np.ndarray | None = None,
        target_columns: np.ndarray | None = None,
        candidate_loss_num_rows: int = 6,
        candidate_loss_num_columns: int = 6,
        source_logits: np.ndarray | torch.Tensor | None = None,
        source_labels: np.ndarray | None = None,
    ) -> SubjectAdapter:
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
            raise ValueError(
                "candidate metadata requires candidate_loss_weight > 0."
            )
        has_input_mean = self.config.input_mean is not None
        has_input_std = self.config.input_std is not None
        if has_input_mean != has_input_std:
            raise ValueError("input_mean and input_std must be supplied together.")
        supplied_splits = sum(
            value is not None for value in (group_ids, repetition_indices)
        )
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

        train_mask = np.ones(len(X), dtype=bool)
        val_mask = np.zeros(len(X), dtype=bool)
        validation_source = None
        if repetition_indices is not None:
            chronological = chronological_validation_split(
                repetition_indices,
                y,
                fraction=self.config.val_group_fraction,
                min_repetitions=self.config.val_groups_min,
                max_repetitions=self.config.val_groups_max,
            )
            train_mask, val_mask = (
                chronological.train_mask,
                chronological.validation_mask,
            )
            validation_source = "chronological_prefix_validation"
        elif validation_mask is not None:
            validation_mask = np.asarray(validation_mask)
            if validation_mask.dtype != np.dtype(bool) or validation_mask.shape != (len(X),):
                raise ValueError("validation_mask must be boolean and align with X.")
            val_mask = validation_mask.copy()
            train_mask = ~val_mask if training_mask is None else np.asarray(training_mask).copy()
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
                group_ids,
                fraction=self.config.val_group_fraction,
                min_groups=self.config.val_groups_min,
                max_groups=self.config.val_groups_max,
                seed=self.config.seed,
            )
            train_mask, val_mask = split.train_mask, split.validation_mask
            validation_source = "group_disjoint_prefix_validation"
        if len(np.unique(y[train_mask])) != 2:
            raise ExpectedSubjectError(
                SubjectFailureCode.INSUFFICIENT_CLASSES,
                stage="adapter_fit",
                detail="training split does not contain both binary classes",
            )

        target_mean, target_std = self._masked_stats(
            X[train_mask], observed_mask[train_mask]
        )
        statistics_mode = self.config.input_statistics
        if statistics_mode == "auto":
            statistics_mode = "source" if has_input_mean and has_input_std else "target_prefix"
        if statistics_mode in {"source", "shrinkage"} and not (
            has_input_mean and has_input_std
        ):
            raise ValueError(f"input_statistics={statistics_mode!r} requires checkpoint statistics.")
        if statistics_mode in {"source", "shrinkage"}:
            source_mean = self._validated_external_stats(
                self.config.input_mean, name="input_mean"
            )
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
        if asmmd_active:
            if source_logits is None or source_labels is None:
                raise ValueError("as_mmd_lambda0 > 0 requires source_logits and source_labels.")
            source_logits_t = torch.as_tensor(source_logits, device=self.device).float().detach()
            source_labels_t = torch.as_tensor(source_labels, device=self.device).long()
            if source_logits_t.ndim != 2 or source_labels_t.shape != (source_logits_t.shape[0],):
                raise ValueError("source_logits/source_labels must be aligned (N, classes)/(N,).")
            if bool(((source_labels_t < 0) | (source_labels_t > 1)).any()):
                raise ValueError("source_labels must be binary 0/1.")
            source_count = source_logits_t.shape[0]
        else:
            source_logits_t = source_labels_t = None
            source_count = 0

        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(self.config.seed)
        trunk_trainable = self.config.head_kind == "full_fine"
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
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], device=self.device)
        )

        # Materialize train/validation row blocks once. The training loop then
        # uses device-side index_select with the same permutation values, so the
        # epoch batches are identical to the previous numpy-indexed path without
        # any per-batch host synchronization.
        n_train = int(train_mask.sum())
        train_rows = torch.as_tensor(np.flatnonzero(train_mask), dtype=torch.long, device=self.device)
        Xt_train = Xt.index_select(0, train_rows)
        yt_train = yt.index_select(0, train_rows)
        val_rows = torch.as_tensor(np.flatnonzero(val_mask), dtype=torch.long, device=self.device)
        Xt_val = Xt.index_select(0, val_rows) if len(val_rows) else Xt[:0]
        yt_val = yt.index_select(0, val_rows) if len(val_rows) else yt[:0]

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
            self._set_training_mode(trunk_trainable=trunk_trainable)
            permutation = torch.randperm(n_train, device=self.device)
            epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for start in range(0, n_train, self.config.batch_size):
                rows = permutation[start : start + self.config.batch_size]
                xb = train_inputs.index_select(0, rows)
                yb = yt_train.index_select(0, rows)
                logits = self._forward_head(xb) if live_forward else self.head(xb)
                if asmmd_active:
                    assert source_logits_t is not None and source_labels_t is not None
                    source_rows = torch.randperm(source_count, device=self.device)[
                        : self.config.batch_size
                    ]
                    source_batch_logits = source_logits_t.index_select(0, source_rows)
                    source_batch_labels = source_labels_t.index_select(0, source_rows)
                    loss, asmmd_diag = asmmd_joint_loss(
                        source_batch_logits,
                        source_batch_labels,
                        logits,
                        yb,
                        epoch=epoch_index,
                        warmup_epochs=self.config.as_mmd_warmup_epochs,
                        base_weight=self.config.as_mmd_lambda0,
                        source_count=source_count,
                        target_count=n_train,
                        target_weight_min=self.config.as_mmd_target_weight_min,
                        target_weight_max=self.config.as_mmd_target_weight_max,
                        pos_weight=pos_weight,
                    )
                    asmmd_losses.append(asmmd_diag)
                else:
                    loss = loss_fn(logits, yb)
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
                        self._forward_head(val_inputs)
                        if live_forward
                        else self.head(val_inputs)
                    )
                    val_loss_tensor = loss_fn(logits, yt_val).detach().float()
                train_value, val_loss = torch.stack(
                    (epoch_loss / max(n_train, 1), val_loss_tensor)
                ).cpu().tolist()
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
                logits = (
                    self._forward_head(val_inputs)
                    if live_forward
                    else self.head(val_inputs)
                )
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
        if (
            validation_source == "chronological_prefix_validation"
            and self.config.refit_full_prefix
            and best_epoch_count is not None
        ):
            refit_losses, refit_candidate_losses = self._refit_full_prefix(
                X,
                y,
                observed_mask,
                initial_state=initial_state,
                epochs=best_epoch_count,
                statistics_mode=statistics_mode,
                candidate_ids=candidate_ids,
                selection_ids=selection_ids,
                target_rows=target_rows,
                target_columns=target_columns,
                candidate_loss_num_rows=candidate_loss_num_rows,
                candidate_loss_num_columns=candidate_loss_num_columns,
            )
        self.last_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val if best_val != float("inf") else None,
            "best_epoch": best_epoch_count,
            "refit_full_prefix": bool(refit_losses),
            "refit_epochs": len(refit_losses),
            "refit_train_losses": refit_losses,
            "candidate_loss_active": bool(candidate_active),
            "candidate_loss_weight": self.config.candidate_loss_weight,
            "candidate_loss_temperature": self.config.candidate_loss_temperature,
            "candidate_losses": candidate_losses,
            "refit_candidate_losses": refit_candidate_losses,
            "as_mmd_active": bool(asmmd_active),
            "as_mmd_losses": asmmd_losses,
            "as_mmd_lambda0": self.config.as_mmd_lambda0,
            "as_mmd_warmup_epochs": self.config.as_mmd_warmup_epochs,
        }
        self._fitted = True
        return self

    def _refit_full_prefix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        observed_mask: np.ndarray,
        *,
        initial_state: dict[str, dict[str, torch.Tensor]],
        epochs: int,
        statistics_mode: str,
        candidate_ids: np.ndarray | None,
        selection_ids: np.ndarray | None,
        target_rows: np.ndarray | None,
        target_columns: np.ndarray | None,
        candidate_loss_num_rows: int,
        candidate_loss_num_columns: int,
    ) -> tuple[list[float], list[dict[str, object]]]:
        """Refit from the same initialization on every chronological prefix row."""

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

        trunk_trainable = self.config.head_kind == "full_fine"
        self._set_trainable(trunk_trainable=trunk_trainable)
        if not trunk_trainable and self.config.head_kind != "adapter":
            self.trunk.eval()
            with torch.no_grad():
                inputs = self._features(inputs)
        live_forward = trunk_trainable or self.config.head_kind == "adapter"
        parameters = [parameter for parameter in self.trunk.parameters() if parameter.requires_grad]
        if self.head is not None:
            parameters.extend(parameter for parameter in self.head.parameters() if parameter.requires_grad)
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
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], device=self.device)
        )
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(self.config.seed)
        losses: list[float] = []
        candidate_losses: list[dict[str, object]] = []
        candidate_active = self.config.candidate_loss_weight > 0.0
        if candidate_active and any(
            value is None for value in (candidate_ids, selection_ids, target_rows, target_columns)
        ):
            raise ValueError("candidate-loss refit requires complete candidate metadata.")
        row_indices = np.arange(len(y), dtype=np.int64)
        for _ in range(int(epochs)):
            self._set_training_mode(trunk_trainable=trunk_trainable)
            permutation = torch.randperm(len(y), device=self.device)
            epoch_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for start in range(0, len(y), self.config.batch_size):
                rows = permutation[start : start + self.config.batch_size]
                xb = inputs.index_select(0, rows)
                yb = labels.index_select(0, rows)
                logits = self._forward_head(xb) if live_forward else self.head(xb)
                loss = loss_fn(logits, yb)
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
        return losses, candidate_losses

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


    def _forward_head(self, X: torch.Tensor) -> torch.Tensor:
        features = self._features(X)
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
            scope: {
                key: value.detach().clone() for key, value in state.items()
            }
            for scope, state in self._state_dict().items()
        }

    def _load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.trunk.load_state_dict(state["trunk"])
        if self.head is not None and "head" in state:
            self.head.load_state_dict(state["head"])

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
            parameters.extend(parameter for parameter in self.head.parameters() if parameter.requires_grad)
        unique = {id(parameter): parameter for parameter in parameters}
        return sum(parameter.numel() for parameter in unique.values())

    def total_parameter_count(self) -> int:
        parameters = list(self.trunk.parameters())
        if self.head is not None:
            parameters.extend(self.head.parameters())
        unique = {id(parameter): parameter for parameter in parameters}
        return sum(parameter.numel() for parameter in unique.values())
