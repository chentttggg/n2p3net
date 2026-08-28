"""模块：深度基线（Deep Baselines）。

职责（roadmap Phase 1 + constitution P8）：
    EEGNet / EEG-Inception(ERP) / EEG Conformer 三个深度基线，统一包装成 Baseline 接口
    （fit(X,y) / predict_logit(X)→(N,)），与 classic.py / riemann.py 同契约，供 evaluate.py
    在完全相同的三层协议下公平比较。输出逐试次 target 判别 logit（供 decision 层累加）。

明确「不做」：
    - 不复用 train/trainer.py（它是 N2P3-Net 专用：吃 N2P3NetOutput + compute_losses 多任务
      损失）。深度基线是「输入 (N,C,T) → 二分类 logits」的单任务模型，用一个自包含的轻量
      训练循环更直接、无耦合。
    - 不做架构超参搜索；epochs 是上限，模型选择与 N2P3-Net 共用被试隔离验证、patience 和
      最佳权重恢复。优化器仍固定为 Adam + pos_weight。

三思决策记录（供后续会话追溯）：
    D-deep-ce        用 n_outputs=2 + CrossEntropyLoss(weight=[1, pos_weight])，而非 n_outputs=1 +
                     BCE。理由：braindecode 三个模型统一支持 n_outputs=2（实测输出 (N,2) logits），
                     且 CrossEntropyLoss 的 weight 是 pos_weight 的最干净等价（加重 target 类）。
                     predict_logit 返回 logits[:,1]−logits[:,0]，即 target/non-target 的 log-odds，
                     与 heads 输出 logit 的语义一致（decision 层对数似然比累积）。
    D-deep-shape     braindecode 模型输入约定 (N, C, T)（channels-first），与 data 层输出一致，直接
                     喂无需转置。实测三个模型 n_outputs=2 均输出 (N,2)。
    D-deep-param     实测参数量：EEGNet 1,490 / EEGInceptionERP 26,622 / EEGConformer 255,106。
                     EEGConformer 远超 E4 的 80k——但 E4 约束的是 N2P3-Net 本体，深度基线是「对照
                     坐标系」：复现一个高容量基线恰好验证 D6「容量非瓶颈」的对照意义（若 Conformer
                     在数千试次上过拟合、反而不如 EEGNet，正是 D6 的实证）。docstring 如实标注。
    D-deep-amp       AMP 用 bf16，CUDA/XPU 启用、CPU 禁用（DP4），复用 train/device.get_device 与
                     device-portability §3 的 autocast 写法。CPU 测试时 enabled=False 等价 fp32。
    D-deep-cpu       Trainer/DeepBaseline 均接受可选 device（默认 get_device()），测试显式传 CPU
                     保证稳定与速度（device-portability 的 D-device-param 同款约定）。
    D-deep-seed      fit 起始 torch.manual_seed(seed) + 每 epoch shuffle 用 torch.randperm，保证
                     可复现。注意：deep fold 线程并行时 CUDA dropout 仍用进程级全局 RNG；
                      模型初始化与 shuffle 已确定化，但逐 bit 复现建议 --deep-jobs 1（audit P2）。
    D-deep-sfreq     EEGInceptionERP 的 scales_samples_s 是「秒」单位，需显式传 sfreq=128 与
                     n_times=128；EEGNet 用 kernel_length(样本)，其物理跨度必须随输入合同审计；
                     EEGConformer 无需 sfreq。
    D-deep-standard  训练前用训练集逐通道 mean/std 做 z-score（fit 内完成，predict 复用训练统计量）。
                     review v6 P1：V 单位输入（~1e-5–1e-4）会让 deep logit 坍缩成窄带
                     （实测 EEGNet AUC 0.744 但 bacc@0=0.500、命中率≈chance），z-score 后
                     AUC 0.797、top1 0.900。standardize_input=False 可作消融。

契约（输入 → 输出）：
    X ∈ R^{N×C×T}（float32，缺失通道须已填 0）+ y ∈ {0,1}^N → fit 后 predict_logit(X) ∈ R^N。
    ``channel_mask`` describes channels that are physically available in the
    dataset; ``trial_channel_mask`` can further mark channels missing in an
    individual trial. Both masks are optional for backwards compatibility.

依赖的决策：roadmap Phase 1、constitution P8/D6、device-portability.md（DP1–DP6）、
    train/device.py（get_device）、baselines/classic.Baseline。
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGConformer, EEGInceptionERP, EEGNet
from sklearn.metrics import roc_auc_score

from baselines.classic import Baseline
from baselines.validation import group_disjoint_validation_split
from data.contract import DEFAULT_P300_DATA_CONTRACT
from train.device import get_device
from train.runtime import (
    DEFAULT_COMPILE_MODE,
    DEFAULT_FUSED_ADAM,
    GpuPerformanceScheduler,
    MatrixBatchSource,
    is_oom_error,
    resolve_optimizer_execution,
)

# 模型名（lowercase）→ 构造类
_MODEL_FACTORIES = {
    "eegnet": EEGNet,
    "inception": EEGInceptionERP,
    "conformer": EEGConformer,
}
DEEP_MODEL_NAMES = tuple(_MODEL_FACTORIES)

# deep fold 线程并行时，torch.manual_seed 与模型初始化必须串行化，否则两个线程
# 会互相踩全局 RNG，导致初始化不可复现（review v6 性能项）。
_INIT_LOCK = threading.Lock()

DEFAULT_DEEP_EPOCHS = 32
DEFAULT_EARLY_STOP_PATIENCE = 6


@dataclass
class DeepConfig:
    """深度基线训练配置（DP5：batch_size 等经配置传入，不写死）。"""

    epochs: int = DEFAULT_DEEP_EPOCHS
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 8.0
    seed: int = 0
    standardize_input: bool = True
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    val_group_frac: float | None = 0.08
    val_groups_min: int = 2
    val_groups_max: int = 12
    early_stop_min_delta: float = 1e-6
    precision: str = "auto"
    fused_adam: bool = DEFAULT_FUSED_ADAM
    compile_mode: str | None = DEFAULT_COMPILE_MODE
    shuffle_each_epoch: bool = False
    max_update_batch_size: int | None = 512
    batch_memory_fraction: float = 0.55
    preload_memory_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("DeepConfig epochs and batch_size must be positive.")
        if self.lr <= 0.0 or self.weight_decay < 0.0 or self.pos_weight <= 0.0:
            raise ValueError("DeepConfig lr/pos_weight must be positive and weight_decay non-negative.")
        if self.early_stop_patience < 1 or self.early_stop_min_delta < 0.0:
            raise ValueError("DeepConfig early-stop patience must be positive and delta non-negative.")
        if self.val_groups_min < 1 or self.val_groups_max < self.val_groups_min:
            raise ValueError("DeepConfig validation group bounds are invalid.")
        if self.val_group_frac is not None and not 0.0 < self.val_group_frac < 1.0:
            raise ValueError("DeepConfig val_group_frac must be in (0,1) or None.")
        if self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("DeepConfig precision must be 'auto', 'bf16', or 'fp32'.")
        if not isinstance(self.fused_adam, bool):
            raise ValueError("DeepConfig fused_adam must be boolean.")
        if not isinstance(self.shuffle_each_epoch, bool):
            raise ValueError("DeepConfig shuffle_each_epoch must be boolean.")
        if self.compile_mode is not None and self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError(
                "DeepConfig compile_mode must be None, 'default', 'reduce-overhead', "
                "or 'max-autotune'."
            )
        if self.max_update_batch_size is not None and self.max_update_batch_size < 1:
            raise ValueError("DeepConfig max_update_batch_size must be positive or None.")
        if not 0.05 <= self.batch_memory_fraction < 1.0:
            raise ValueError("DeepConfig batch_memory_fraction must be in [0.05, 1).")
        if not 0.05 <= self.preload_memory_fraction < 1.0:
            raise ValueError("DeepConfig preload_memory_fraction must be in [0.05, 1).")


class DeepBaseline(Baseline):
    """深度基线统一包装（EEGNet / EEG-Inception / EEG Conformer）。

    Parameters
    ----------
    model_name : str
        "eegnet" / "inception" / "conformer"（大小写不敏感）。
    n_chans / n_times / sfreq : int / int / float
        通道数、时间点数、采样率（默认模型合同为 8 / 128 / 128）。
    config : DeepConfig | None
        训练配置（默认 DeepConfig()）。
    device : torch.device | None
        默认 get_device()；测试可显式传 CPU。
    """

    fit_accepts_group_ids = True
    fit_accepts_trial_channel_mask = True
    predict_accepts_trial_channel_mask = True
    accepts_unmaterialized_trial_channel_mask = True
    runtime_requires_exclusive_lease = True

    def __init__(
        self,
        model_name: str = "eegnet",
        n_chans: int = 8,
        n_times: int = DEFAULT_P300_DATA_CONTRACT.n_times,
        sfreq: float = DEFAULT_P300_DATA_CONTRACT.sample_rate_hz,
        config: DeepConfig | None = None,
        device: torch.device | None = None,
        runtime: GpuPerformanceScheduler | None = None,
        *,
        channel_mask: np.ndarray | None = None,
        pretrained_state_dict: dict | None = None,
        load_mapping: dict[str, str | None] | None = None,
        freeze_prefixes: Sequence[str] = (),
        strict_load: bool = False,
    ):
        key = model_name.lower()
        if key not in _MODEL_FACTORIES:
            raise ValueError(f"未知 model_name={model_name!r}，可选 {list(_MODEL_FACTORIES)}。")
        if n_chans < 1 or n_times < 1 or not np.isfinite(sfreq) or sfreq <= 0.0:
            raise ValueError("n_chans/n_times/sfreq must define a positive physical input.")
        self.model_name = key
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.cfg = config if config is not None else DeepConfig()
        requested_device = torch.device(device) if device is not None else get_device()
        if runtime is not None:
            if (
                runtime.device.type != requested_device.type
                or (
                    requested_device.index is not None
                    and runtime.device.index != requested_device.index
                )
            ):
                raise ValueError("runtime device must match the requested model device.")
            self.runtime = runtime
        else:
            self.runtime = GpuPerformanceScheduler(
                requested_device,
                precision=self.cfg.precision,
                batch_memory_fraction=self.cfg.batch_memory_fraction,
                preload_memory_fraction=self.cfg.preload_memory_fraction,
            )
        self.device = self.runtime.device
        self.optimizer_execution = resolve_optimizer_execution(
            self.device,
            fused_adam=self.cfg.fused_adam,
            compile_mode=self.cfg.compile_mode,
        )
        self.use_amp = self.runtime.precision.amp_enabled
        if channel_mask is None:
            self.channel_mask = np.ones(self.n_chans, dtype=bool)
        else:
            self.channel_mask = np.asarray(channel_mask, dtype=bool)
            if self.channel_mask.shape != (self.n_chans,) or not bool(self.channel_mask.any()):
                raise ValueError("channel_mask must be (n_chans,) and retain one channel.")
        # P9 辅助预训练接口（transfer_policy 方式 A/C）：加载报告 + 层冻结/映射
        self.pretrained_state_dict = pretrained_state_dict
        self.load_mapping: dict[str, str | None] = dict(load_mapping or {})
        self.freeze_prefixes = tuple(freeze_prefixes)
        self.strict_load = bool(strict_load)
        self.load_report: list[dict] = []
        self._fitted = False
        self.last_history: dict[str, list[float] | int | None] = {
            "train_losses": [],
            "val_losses": [],
            "best_epoch": None,
        }
        self._evaluation_fold_id: int | None = None
        self.last_val_groups: int | None = None
        self.calibration_logits_: np.ndarray | None = None
        self.calibration_labels_: np.ndarray | None = None
        self.calibration_source_: str | None = None
        self.last_runtime: dict[str, object] = {}

    # ---------------- 模型构造 ----------------

    def _make_model(self) -> nn.Module:
        """按 model_name 构造 braindecode 模型（n_outputs=2，D-deep-ce）。"""
        common = dict(n_chans=self.n_chans, n_outputs=2, n_times=self.n_times)
        if self.model_name == "eegnet":
            return EEGNet(**common, final_conv_length="auto")
        if self.model_name == "inception":
            return EEGInceptionERP(**common, sfreq=self.sfreq)  # D-deep-sfreq
        if self.model_name == "conformer":
            return EEGConformer(**common)
        raise AssertionError("unreachable")  # 构造前已校验 model_name

    def parameter_count(self) -> int:
        """Return trainable architecture parameters without allocating on the accelerator."""

        fitted = getattr(self, "model_", None)
        if fitted is not None:
            return sum(parameter.numel() for parameter in fitted.parameters() if parameter.requires_grad)
        with _INIT_LOCK:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.cfg.seed)
                candidate = self._make_model()
        return sum(
            parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
        )

    def _autocast_ctx(self):
        return self.runtime.autocast()

    # ---------------- P9 辅助预训练加载 / 冻结 / checkpoint ----------------

    def _apply_pretrained_state_dict(self) -> None:
        """按 transfer_policy 加载辅助预训练权重，并生成 load_report。

        规则：
        - load_mapping 形如 {source_key: target_key_or_None}；None 表示显式跳过该 source key；
        - 未出现在 load_mapping 的 key 按同名加载；
        - 形状不一致只记录并跳过（strict_load=True 时抛错）；
        - 所有加载事件写入 self.load_report，供实验记录。
        """
        self.load_report = []
        if not self.pretrained_state_dict:
            self.load_report.append({"event": "no_pretrained_state", "message": "未提供预训练权重"})
            return

        model_state = self.model_.state_dict()
        loaded: dict[str, torch.Tensor] = {}

        for src_key, src_value in self.pretrained_state_dict.items():
            dst_key = self.load_mapping.get(src_key, src_key)
            if dst_key is None:
                self.load_report.append({"event": "mapped_skip", "source_key": src_key})
                continue
            if dst_key not in model_state:
                self.load_report.append(
                    {"event": "missing_target", "source_key": src_key, "target_key": dst_key}
                )
                continue
            dst_value = model_state[dst_key]
            if tuple(dst_value.shape) != tuple(src_value.shape):
                self.load_report.append(
                    {
                        "event": "shape_mismatch",
                        "target_key": dst_key,
                        "source_shape": tuple(src_value.shape),
                        "target_shape": tuple(dst_value.shape),
                    }
                )
                continue
            loaded[dst_key] = src_value.to(device=dst_value.device, dtype=dst_value.dtype)
            self.load_report.append({"event": "loaded", "key": dst_key})

        # 从未出现在源 checkpoint 中的目标 key
        for key in model_state:
            if key not in loaded and key not in self.load_mapping.values():
                self.load_report.append({"event": "not_in_source", "key": key})

        has_blocking = any(
            e["event"] in ("missing_target", "shape_mismatch") for e in self.load_report
        )
        if self.strict_load and has_blocking:
            raise ValueError(
                "strict_load=True 但预训练权重存在 missing_target/shape_mismatch，详见 load_report。"
            )

        with torch.no_grad():
            for key, value in loaded.items():
                model_state[key].copy_(value)

    def _freeze_layers(self) -> None:
        """按 freeze_prefixes 冻结参数；空元组表示全模型可训练。"""
        for name, param in self.model_.named_parameters():
            if self.freeze_prefixes and any(
                name.startswith(prefix) for prefix in self.freeze_prefixes
            ):
                param.requires_grad = False

    def save_checkpoint(self, path: str | Path) -> Path:
        """保存 P9 预训练 checkpoint：state_dict + 构造元数据。"""
        if not hasattr(self, "model_") or self.model_ is None:
            raise RuntimeError("模型尚未构造，无法保存 checkpoint。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "n_chans": self.n_chans,
            "n_times": self.n_times,
            "sfreq": self.sfreq,
            "channel_mask": self.channel_mask.copy(),
            "config": self.cfg,
            "model_state_dict": self.model_.state_dict(),
        }
        torch.save(payload, path)
        return path

    @staticmethod
    def load_state_dict_file(path: str | Path) -> dict:
        """读取 checkpoint 的 state_dict（供其他模型作为 pretrained_state_dict 传入）。"""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        return payload

    # ---------------- 训练 ----------------

    def _effective_trial_channel_mask(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Validate and combine static and per-trial channel availability."""

        static = np.broadcast_to(self.channel_mask, X.shape[:2])
        if trial_channel_mask is None:
            effective = np.array(static, dtype=bool, copy=True)
        else:
            supplied = np.asarray(trial_channel_mask)
            if supplied.dtype != np.dtype(bool):
                raise ValueError("trial_channel_mask must have boolean dtype.")
            if supplied.shape != X.shape[:2]:
                raise ValueError("trial_channel_mask must have shape (N,C) matching X.")
            if bool((supplied & ~static).any()):
                raise ValueError("trial_channel_mask cannot enable a permanently absent channel.")
            effective = supplied & static
        if not bool(effective.any(axis=1).all()):
            raise ValueError("Every trial must retain at least one observed channel.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf.")
        return effective

    def _masked_input_stats(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel statistics using observed samples only."""

        observed = mask[:, :, None]
        counts = mask.sum(axis=0, dtype=np.float64)[None, :, None] * X.shape[2]
        denominator = np.maximum(counts, 1.0)
        sums = np.sum(
            X,
            axis=(0, 2),
            dtype=np.float64,
            keepdims=True,
            where=observed,
        )
        mean = np.divide(sums, denominator, out=np.zeros_like(sums), where=counts > 0.0)
        # NumPy performs the masked two-pass reduction internally, avoiding
        # several full-size float64 temporaries in every parallel fold.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.var(
                X,
                axis=(0, 2),
                dtype=np.float64,
                keepdims=True,
                where=observed,
                mean=mean,
            )
        std = np.where(
            counts > 0.0,
            np.sqrt(np.maximum(variance, 0.0)) + 1e-6,
            1.0,
        )
        return mean.astype(np.float32), std.astype(np.float32)

    def _prepare_input(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        prepared = np.empty(X.shape, dtype=np.float32)
        if self.cfg.standardize_input:
            np.subtract(X, self._input_mean, out=prepared, casting="unsafe")
            np.divide(prepared, self._input_std, out=prepared)
        else:
            np.copyto(prepared, X, casting="unsafe")
        # Standardization turns zero-filled missing channels into non-zero
        # values unless the mask is re-applied after the transform.
        np.copyto(prepared, 0.0, where=~mask[:, :, None])
        return prepared

    def _model_seed(self) -> int:
        return int(self.cfg.seed + (self._evaluation_fold_id or 0))

    def configure_runtime_worker_budget(self, worker_count: int) -> None:
        """Apply the parent scheduler's per-worker memory budget before fitting."""

        self.runtime.configure_shared_worker_budget(worker_count)

    def _initialize_model(self, seed: int) -> None:
        """Construct a fold-local model without leaking CPU RNG state to peer workers."""

        with _INIT_LOCK:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
                self.model_ = self._make_model().to(self.device)
            self._apply_pretrained_state_dict()
            self._freeze_layers()

    def _validation_pass(
        self,
        source: MatrixBatchSource,
        loss_fn: nn.Module | None,
        batch_size: int,
    ) -> tuple[float | None, np.ndarray, np.ndarray]:
        """Run a vectorized batch pass and return CPU scores/labels once."""

        score_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        total_loss: torch.Tensor | None = None
        n_rows = 0
        self.model_.eval()
        with torch.inference_mode():
            for xb, yb in source.batches(batch_size):
                assert yb is not None
                with self._autocast_ctx():
                    logits = self.model_(xb)
                    loss = None if loss_fn is None else loss_fn(logits, yb)
                score_parts.append((logits[:, 1] - logits[:, 0]).float())
                label_parts.append(yb)
                if loss is not None:
                    weighted_loss = loss.detach().float() * len(xb)
                    total_loss = (
                        weighted_loss if total_loss is None else total_loss + weighted_loss
                    )
                n_rows += len(xb)
        scores = torch.cat(score_parts).cpu().numpy().astype(np.float64)
        labels = torch.cat(label_parts).cpu().numpy().astype(np.int64)
        mean_loss = (
            None
            if total_loss is None
            else float(total_loss.cpu()) / max(n_rows, 1)
        )
        return (
            mean_loss,
            scores,
            labels,
        )

    def _validation_pass_with_oom_retry(
        self,
        source: MatrixBatchSource,
        loss_fn: nn.Module | None,
        batch_size: int,
    ) -> tuple[tuple[float | None, np.ndarray, np.ndarray], int]:
        """Retry an inference-only pass without restarting completed training."""

        active_batch_size = int(batch_size)
        while True:
            try:
                return self._validation_pass(source, loss_fn, active_batch_size), active_batch_size
            except RuntimeError as error:
                if not is_oom_error(error) or active_batch_size <= 1:
                    raise
                active_batch_size = max(1, active_batch_size // 2)
            # The exception variable and its traceback are cleared when the
            # handler exits, so failed-pass tensors can actually be reclaimed.
            self.runtime.release_temporary_memory()

    def _clear_failed_fit(self, *, release_memory: bool = True) -> None:
        self.model_ = None
        self._fitted = False
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        if release_memory:
            self.runtime.release_temporary_memory()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ids: np.ndarray | None = None,
        trial_channel_mask: np.ndarray | None = None,
    ) -> DeepBaseline:
        """Fit with a bounded OOM retry that never accumulates gradients."""

        # Dropping Python references returns prior tensors to the caching
        # allocator. Emptying the CUDA cache here would create a gap before
        # every healthy fold; reserve that expensive cleanup for actual OOMs.
        self._clear_failed_fit(release_memory=False)
        requested_batch_size = self.cfg.batch_size
        retries = 0
        while True:
            try:
                with self.runtime.lease():
                    self._fit_attempt(
                        X,
                        y,
                        group_ids=group_ids,
                        trial_channel_mask=trial_channel_mask,
                        requested_batch_size=requested_batch_size,
                    )
                self.last_runtime = {
                    "device": str(self.device),
                    "precision": self.runtime.precision.name,
                    "batch_size": self._active_batch_size,
                    "validation_batch_size": self._validation_batch_size,
                    "preloaded": self._preloaded_batches,
                    "shuffle_each_epoch": self._shuffle_each_epoch,
                    "transfer_fallback": self._transfer_fallback,
                    **self.optimizer_execution.record(),
                    "host_sync_policy": "epoch_boundary",
                    "oom_retries": retries,
                    "shared_worker_count": self.runtime.shared_worker_count,
                    "memory": self.runtime.memory_record(),
                }
                return self
            except RuntimeError as error:
                if not is_oom_error(error):
                    self._clear_failed_fit(release_memory=False)
                    raise
                attempted_batch_size = getattr(self, "_active_batch_size", requested_batch_size)
                if attempted_batch_size <= 1:
                    self._clear_failed_fit(release_memory=False)
                    raise RuntimeError(
                        "GPU OOM persists at batch_size=1. "
                        "Reduce model/data residency or select a larger device."
                    ) from error
                requested_batch_size = max(1, attempted_batch_size // 2)
                retries += 1
            except Exception:
                self._clear_failed_fit(release_memory=False)
                raise
            # Leave the OOM handler first so its traceback no longer owns the
            # failed model, optimizer, source matrices, or activation tensors.
            self._clear_failed_fit()

    def _fit_attempt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        group_ids: np.ndarray | None,
        trial_channel_mask: np.ndarray | None,
        requested_batch_size: int,
    ) -> None:
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("X must have a floating dtype.")
        X = X.astype(np.float32, copy=False)
        y_raw = np.asarray(y)
        if y_raw.ndim != 1 or len(y_raw) != len(X):
            raise ValueError("y must be a one-dimensional array aligned with X.")
        if not np.issubdtype(y_raw.dtype, np.integer) or set(np.unique(y_raw).tolist()) != {0, 1}:
            raise ValueError("Deep binary training requires integer labels {0,1}.")
        y = y_raw.astype(np.int64)
        if X.ndim != 3 or X.shape[1] != self.n_chans:
            raise ValueError(f"X 须为 (N,{self.n_chans},T)，得到 {X.shape}。")
        if X.shape[2] != self.n_times:
            raise ValueError(f"X 时间点数 {X.shape[2]} 与模型契约 n_times={self.n_times} 不一致。")
        trial_mask = self._effective_trial_channel_mask(X, trial_channel_mask)

        if group_ids is None:
            train_mask = np.ones(len(X), dtype=bool)
            val_mask = np.zeros(len(X), dtype=bool)
            self.last_val_groups = None
        else:
            group_ids = np.asarray(group_ids)
            if group_ids.shape != (len(X),):
                raise ValueError("group_ids must align with X.")
            split = group_disjoint_validation_split(
                group_ids,
                fraction=self.cfg.val_group_frac,
                min_groups=self.cfg.val_groups_min,
                max_groups=self.cfg.val_groups_max,
                seed=self.cfg.seed,
            )
            train_mask, val_mask = split.train_mask, split.validation_mask
            self.last_val_groups = split.n_validation_groups
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        train_channel_mask = trial_mask[train_mask]
        val_channel_mask = trial_mask[val_mask]
        unobserved_train_channels = self.channel_mask & ~train_channel_mask.any(axis=0)
        if bool(unobserved_train_channels.any()):
            missing = np.flatnonzero(unobserved_train_channels).tolist()
            raise ValueError(
                "Training split never observes active channels "
                f"{missing}; use an intersection layout or mark them permanently absent."
            )
        if set(np.unique(y_train).tolist()) != {0, 1}:
            raise ValueError("Deep training split must contain both binary classes.")
        if len(y_val) and set(np.unique(y_val).tolist()) != {0, 1}:
            raise ValueError("Deep validation split must contain both binary classes.")

        self._input_mean, self._input_std = self._masked_input_stats(X_train, train_channel_mask)
        X_train = self._prepare_input(X_train, train_channel_mask)
        if len(X_val):
            X_val = self._prepare_input(X_val, val_channel_mask)

        seed = self._model_seed()
        self._initialize_model(seed)
        trainable_params = [parameter for parameter in self.model_.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise RuntimeError("freeze_prefixes froze every parameter; no optimizer update is possible.")
        batch_size = self.runtime.choose_batch_size(
            requested_batch_size,
            X_train.shape[1:],
            model=self.model_,
            max_update_batch_size=self.cfg.max_update_batch_size,
        )
        self._active_batch_size = batch_size
        total_matrix_bytes = X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes
        preload = self.runtime.can_preload(total_matrix_bytes)
        train_source = MatrixBatchSource(
            torch.from_numpy(np.ascontiguousarray(X_train)),
            torch.from_numpy(np.ascontiguousarray(y_train)),
            self.runtime,
            preload=preload,
        )
        val_source = (
            MatrixBatchSource(
                torch.from_numpy(np.ascontiguousarray(X_val)),
                torch.from_numpy(np.ascontiguousarray(y_val)),
                self.runtime,
                preload=preload,
            )
            if len(X_val)
            else None
        )
        self._preloaded_batches = train_source.preloaded and (
            val_source is None or val_source.preloaded
        )
        self._transfer_fallback = train_source.transfer_fallback or bool(
            val_source is not None and val_source.transfer_fallback
        )
        # The one-shot epoch shuffle is an explicit performance experiment:
        # it uses one device gather per epoch instead of one per batch, but it
        # reserves an extra full training-matrix copy. Row order and label
        # alignment are identical to the per-batch path.
        self._shuffle_each_epoch = (
            self.cfg.shuffle_each_epoch
            and self._preloaded_batches
            and self.runtime.can_preload(train_source.nbytes)
        )
        train_source.shuffle_each_epoch = self._shuffle_each_epoch
        # Inference and validation do not retain activations or gradients.
        # After an OOM-induced training-batch reduction, give them their own
        # memory-safe batch budget so validation does not inherit batch_size=1.
        validation_batch_size = batch_size
        if val_source is not None and batch_size < self.cfg.batch_size:
            validation_batch_size = self.runtime.choose_batch_size(
                max(self.cfg.batch_size * 4, 256),
                X_train.shape[1:],
                model=self.model_,
            )
        self._validation_batch_size = validation_batch_size

        adam_kwargs: dict[str, object] = {
            "lr": self.cfg.lr,
            "weight_decay": self.cfg.weight_decay,
        }
        if self.optimizer_execution.fused_adam:
            adam_kwargs["fused"] = True
        if self.optimizer_execution.uses_cuda_graphs:
            adam_kwargs["capturable"] = True
        opt = torch.optim.Adam(trainable_params, **adam_kwargs)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, self.cfg.pos_weight], device=self.device)
        )
        perm_gen = train_source.make_generator(seed)
        train_losses: list[float] = []
        val_losses: list[float] = []
        task_val_aucs: list[float | None] = []
        best_state = None
        best_epoch = None
        best_val_loss = float("inf")
        patience_left = int(self.cfg.early_stop_patience)
        early_stop_triggered = False
        epoch_progress_callback = self.epoch_progress_callback()

        def train_step(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            return loss.detach()

        if self.optimizer_execution.compile_mode is not None:
            train_step = torch.compile(
                train_step,
                mode=self.optimizer_execution.compile_mode,
                fullgraph=False,
            )

        for epoch in range(self.cfg.epochs):
            self.model_.train()
            epoch_loss: torch.Tensor | None = None
            n_seen = 0
            for xb, yb in train_source.shuffled_batches(batch_size, perm_gen):
                assert yb is not None
                loss = train_step(xb, yb)
                weighted_loss = loss.float() * len(xb)
                epoch_loss = weighted_loss if epoch_loss is None else epoch_loss + weighted_loss
                n_seen += len(xb)
            if epoch_loss is None:
                raise RuntimeError("Deep training produced no optimizer batches.")
            train_losses.append(float(epoch_loss.cpu()) / max(n_seen, 1))
            mean_val = None
            task_val_auc = None
            will_early_stop = False

            if val_source is not None:
                (
                    (mean_val, val_scores, val_labels),
                    validation_batch_size,
                ) = self._validation_pass_with_oom_retry(
                    val_source,
                    loss_fn,
                    validation_batch_size,
                )
                self._validation_batch_size = validation_batch_size
                assert mean_val is not None
                val_losses.append(mean_val)
                if len(np.unique(val_labels)) == 2:
                    task_val_auc = float(roc_auc_score(val_labels, val_scores))
                task_val_aucs.append(task_val_auc)
                if mean_val < best_val_loss - self.cfg.early_stop_min_delta:
                    best_val_loss = mean_val
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.model_.state_dict().items()
                    }
                    patience_left = int(self.cfg.early_stop_patience)
                else:
                    patience_left -= 1
                will_early_stop = patience_left <= 0

            if epoch_progress_callback is not None:
                epoch_progress_callback(
                    {
                        "epoch": epoch + 1,
                        "epoch_limit": self.cfg.epochs,
                        "train_loss": train_losses[-1],
                        "train_loss_components": {},
                        "task_val_loss": mean_val,
                        "task_val_auc": task_val_auc,
                        "objective_val_loss": mean_val,
                        "phase": "joint",
                        "optimizer_steps": None,
                        "selection_active": mean_val is not None,
                        "patience_left": patience_left if mean_val is not None else None,
                        "early_stop_patience": self.cfg.early_stop_patience,
                        "best_epoch": best_epoch + 1 if best_epoch is not None else None,
                        "best_task_epoch": best_epoch + 1 if best_epoch is not None else None,
                        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                        "best_task_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                        "will_early_stop": will_early_stop,
                    }
                )
            if will_early_stop:
                early_stop_triggered = True
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        final_task_val_auc = None
        if val_source is not None:
            (
                (_, final_scores, final_labels),
                validation_batch_size,
            ) = self._validation_pass_with_oom_retry(
                val_source,
                None,
                validation_batch_size,
            )
            self._validation_batch_size = validation_batch_size
            if len(np.unique(final_labels)) == 2:
                final_task_val_auc = float(roc_auc_score(final_labels, final_scores))
            self.calibration_logits_ = final_scores
            self.calibration_labels_ = final_labels
            self.calibration_source_ = "group_disjoint_validation"
            self.model_.train()
        else:
            self.calibration_logits_ = None
            self.calibration_labels_ = None
            self.calibration_source_ = None
        self.last_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_objective_losses": list(val_losses),
            "task_val_aucs": task_val_aucs,
            "final_task_val_auc": final_task_val_auc,
            "phases": ["joint"] * len(train_losses),
            "best_epoch": best_epoch,
            "best_task_epoch": best_epoch,
            "best_task_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "task_patience_exhausted": early_stop_triggered,
        }
        self._fitted = True

    # ---------------- 预测 ----------------

    def predict_logit(
        self,
        X: np.ndarray,
        trial_channel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_logit。")
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError("X must have a floating dtype.")
        X = X.astype(np.float32, copy=False)
        if X.ndim != 3 or X.shape[1] != self.n_chans:
            raise ValueError(f"X 须为 (N,{self.n_chans},T)，得到 {X.shape}。")
        if X.shape[2] != self.n_times:
            raise ValueError(f"X 时间点数 {X.shape[2]} 与模型契约 n_times={self.n_times} 不一致。")
        if len(X) == 0:
            raise ValueError("predict_logit requires at least one trial.")
        mask = self._effective_trial_channel_mask(X, trial_channel_mask)
        X = self._prepare_input(X, mask)

        # D-deep-predict-chunks: bounded matrix batches keep output tensors on
        # CPU and avoid a large test-set allocation on the accelerator.
        with self.runtime.lease():
            source = MatrixBatchSource(
                torch.from_numpy(np.ascontiguousarray(X)),
                None,
                self.runtime,
                preload=self.runtime.can_preload(X.nbytes),
            )
            chunk_size = self.runtime.choose_batch_size(
                max(self.cfg.batch_size * 4, 256),
                X.shape[1:],
                model=self.model_,
            )
            self.model_.eval()
            out_chunks: list[torch.Tensor] = []
            with torch.inference_mode():
                for xb, _ in source.batches(chunk_size):
                    with self._autocast_ctx():
                        logits = self.model_(xb)
                    out_chunks.append((logits[:, 1] - logits[:, 0]).float())
            output = torch.cat(out_chunks).cpu().numpy().astype(np.float64)
        self.last_inference_runtime = {
            "device": str(self.device),
            "precision": self.runtime.precision.name,
            "batch_size": chunk_size,
            "preloaded": source.preloaded,
            "transfer_fallback": source.transfer_fallback,
            "host_sync_policy": "output_boundary",
            "shared_worker_count": self.runtime.shared_worker_count,
            "memory": self.runtime.memory_record(),
        }
        return output
