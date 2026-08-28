"""Subject-specific downstream adapters on a frozen or fine-tuned trunk.

The adapter intentionally mirrors the compact-baseline contract: ``fit(X, y)``
and ``predict_logit(X) -> (N,)``. It standardizes only from training rows and
can hold out complete temporal groups for early stopping and calibration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from baselines.validation import group_disjoint_validation_split
from models.n2p3net import N2P3Net


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

    def __post_init__(self) -> None:
        if self.head_kind not in {"linear", "mlp16", "full_fine"}:
            raise ValueError("head_kind must be linear, mlp16, or full_fine.")
        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0.0:
            raise ValueError("invalid optimization settings.")
        if self.val_group_fraction is not None and not 0.0 < self.val_group_fraction < 1.0:
            raise ValueError("val_group_fraction must be in (0,1) or None.")
        if self.val_groups_min < 1 or self.val_groups_max < self.val_groups_min:
            raise ValueError("invalid validation group bounds.")
        if self.early_stop_patience < 1 or self.early_stop_min_delta < 0.0:
            raise ValueError("invalid early stopping settings.")


class SubjectAdapter:
    """Train a subject head on prefix trials of one target subject."""

    def __init__(
        self,
        trunk: N2P3Net,
        *,
        config: SubjectAdapterConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or SubjectAdapterConfig()
        self.device = torch.device("cpu") if device is None else device
        if trunk.n_times is None:
            raise ValueError("subject adapters require a trunk with fixed n_times.")
        if trunk.pooling_mode != "ms_flatten":
            raise ValueError("subject adapters currently require pooling_mode='ms_flatten'.")
        self.trunk = trunk.to(self.device)
        self.feature_dim = trunk.classifier_features
        self.head: nn.Module
        if self.config.head_kind == "linear":
            self.head = nn.Linear(self.feature_dim, 2)
        elif self.config.head_kind == "mlp16":
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, 16),
                nn.GELU(),
                nn.Linear(16, 2),
            )
        else:
            self.head = None  # type: ignore[assignment]
        self._fitted = False
        self.calibration_logits_: np.ndarray | None = None
        self.calibration_labels_: np.ndarray | None = None
        self.calibration_source_ = None
        self.last_history: dict[str, object] = {}

    def _features(self, X: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.training):
            features = self.trunk.forward_features(X)
            return self.trunk.pool(features)

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        if X.ndim != 3 or X.shape[1:] != (self.trunk.n_channels, self.trunk.n_times):
            raise ValueError(f"X must be (N,{self.trunk.n_channels},{self.trunk.n_times}).")
        return torch.from_numpy(np.ascontiguousarray(X)).to(self.device)

    def _set_trainable(self, *, trunk_trainable: bool) -> None:
        for parameter in self.trunk.parameters():
            parameter.requires_grad = trunk_trainable
        if self.head is not None:
            for parameter in self.head.parameters():
                parameter.requires_grad = True

    @property
    def training(self) -> bool:
        return self.trunk.training

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        group_ids: np.ndarray | None = None,
    ) -> SubjectAdapter:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if y.ndim != 1 or len(y) != len(X) or set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("subject adapter requires binary labels {0,1} aligned with X.")
        self._input_mean = X.reshape(X.shape[0], X.shape[1], -1).mean(axis=(0, 2), keepdims=True)
        self._input_std = X.reshape(X.shape[0], X.shape[1], -1).std(axis=(0, 2), keepdims=True)
        self._input_std = np.where(self._input_std < 1e-6, 1.0, self._input_std)
        Xt = torch.from_numpy(np.ascontiguousarray((X - self._input_mean) / self._input_std)).to(
            self.device
        )
        yt = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)

        train_mask = np.ones(len(X), dtype=bool)
        val_mask = np.zeros(len(X), dtype=bool)
        if group_ids is not None:
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
        if len(np.unique(y[train_mask])) != 2:
            raise ValueError("training split must contain both classes.")

        torch.manual_seed(self.config.seed)
        self._set_trainable(trunk_trainable=self.config.head_kind == "full_fine")
        parameters = [p for p in self.trunk.parameters() if p.requires_grad]
        if self.head is not None:
            parameters.extend(self.head.parameters())
        if not parameters:
            raise RuntimeError("no trainable parameters in subject adapter.")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        train_counts = np.bincount(y[train_mask], minlength=2)
        pos_weight = float(train_counts[0] / max(train_counts[1], 1))
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], device=self.device)
        )

        best_state = None
        best_val = float("inf")
        patience = self.config.early_stop_patience
        train_losses: list[float] = []
        val_losses: list[float] = []
        for _ in range(self.config.epochs):
            self.trunk.train()
            if self.head is not None:
                self.head.train()
            permutation = torch.randperm(int(train_mask.sum()), device=self.device)
            train_idx = np.flatnonzero(train_mask)
            epoch_loss = 0.0
            for start in range(0, len(train_idx), self.config.batch_size):
                idx = train_idx[permutation[start : start + self.config.batch_size].cpu().numpy()]
                logits = self._forward_head(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
            train_losses.append(epoch_loss / max(len(train_idx), 1))

            val_loss = None
            if val_mask.any():
                val_idx = np.flatnonzero(val_mask)
                self.trunk.eval()
                if self.head is not None:
                    self.head.eval()
                with torch.inference_mode():
                    logits = self._forward_head(Xt[val_idx])
                    val_loss = float(loss_fn(logits, yt[val_idx]))
                val_losses.append(val_loss)
                if val_loss < best_val - self.config.early_stop_min_delta:
                    best_val = val_loss
                    patience = self.config.early_stop_patience
                    best_state = copy.deepcopy(self._state_dict())
                else:
                    patience -= 1
                if patience <= 0:
                    break

        if best_state is not None:
            self._load_state_dict(best_state)

        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        if val_mask.any():
            val_idx = np.flatnonzero(val_mask)
            with torch.inference_mode():
                logits = self._forward_head(Xt[val_idx])
            self.calibration_logits_ = logits[:, 1] - logits[:, 0]
            self.calibration_labels_ = y[val_idx]
            self.calibration_source_ = "group_disjoint_prefix_validation"
        else:
            self.calibration_logits_ = None
            self.calibration_labels_ = None
            self.calibration_source_ = None
        self.last_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val if best_val != float("inf") else None,
        }
        self._fitted = True
        return self

    def _forward_head(self, X: torch.Tensor) -> torch.Tensor:
        features = self._features(X)
        if self.head is not None:
            return self.head(features)
        return self.trunk.classifier(features)

    def _state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        payload = {"trunk": self.trunk.state_dict()}
        if self.head is not None:
            payload["head"] = self.head.state_dict()
        return payload

    def _load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.trunk.load_state_dict(state["trunk"])
        if self.head is not None:
            self.head.load_state_dict(state["head"])

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit before predict_logit.")
        self.trunk.eval()
        if self.head is not None:
            self.head.eval()
        Xt = self._prepare_input(X)
        standardized = (Xt - torch.from_numpy(self._input_mean).to(self.device)) / torch.from_numpy(
            self._input_std
        ).to(self.device)
        with torch.inference_mode():
            logits = self._forward_head(standardized)
        return (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)

    def parameter_count(self) -> int:
        total = 0
        for parameter in self.trunk.parameters():
            if parameter.requires_grad:
                total += parameter.numel()
        if self.head is not None:
            total += sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return total
