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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGConformer, EEGInceptionERP, EEGNet
from sklearn.metrics import roc_auc_score

from baselines.classic import Baseline
from baselines.validation import (
    domain_stratified_group_validation_split,
    group_disjoint_validation_split,
)
from data.contract import DEFAULT_P300_DATA_CONTRACT
from models.adapters import (
    SOURCE_RESIDUAL_LEDGER_SCHEMA,
    FeatureDomainTransfer,
    SourceResidualContractError,
    source_domain_axis_record,
    source_residual_loss_ledger,
    source_residual_ratio_contract,
)
from models.splitbn import SPLITBN_SOURCE_DOMAIN_BUFFER
from train.device import get_device
from train.runtime import (
    DEFAULT_COMPILE_MODE,
    DEFAULT_FUSED_ADAM,
    GpuPerformanceScheduler,
    MatrixBatchSource,
    is_oom_error,
    resolve_optimizer_execution,
)
from transfer.alignment import (
    conditional_domain_rbf_mmd_loss,
    conditional_mean_alignment_loss,
)
from transfer.decision_loss import (
    DECISION_TRAINING_LEDGER_SCHEMA,
    DecisionBatchPlan,
    DecisionLevelLoss,
    DecisionLossConfig,
    DirectDecisionContractError,
    DirectDecisionMetadata,
    build_decision_batch_plan,
    invalid_decision_ledger,
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


def resolve_source_risk_weights(
    labels: np.ndarray,
    domain_ids: np.ndarray,
    unit_ids: np.ndarray,
    *,
    domain_mass: Mapping[str, float],
    within_domain_unit: str,
    pos_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve fixed-row multipliers for a hierarchical multi-source CE risk."""

    y = np.asarray(labels, dtype=np.int64)
    domains = np.asarray(domain_ids).astype(str)
    units = np.asarray(unit_ids).astype(str)
    if y.ndim != 1 or domains.shape != y.shape or units.shape != y.shape:
        raise ValueError("labels, domain_ids, and risk unit ids must be aligned vectors.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("source risk requires both binary labels {0,1}.")
    if np.any(np.char.strip(domains) == "") or np.any(np.char.strip(units) == ""):
        raise ValueError("source risk domain and unit ids must be non-empty.")
    if within_domain_unit not in {"epoch", "participant"}:
        raise ValueError("within_domain_unit must be epoch or participant.")
    masses = {str(key).strip(): float(value) for key, value in domain_mass.items()}
    if (
        set(masses) != set(domains.tolist())
        or any(not key or not np.isfinite(value) or value <= 0.0 for key, value in masses.items())
        or not np.isclose(sum(masses.values()), 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError(
            "domain_mass must cover retained domains exactly with finite positive masses "
            "summing to one."
        )
    unit_domain: dict[str, str] = {}
    for unit_id, domain in zip(units, domains, strict=True):
        previous = unit_domain.setdefault(unit_id, domain)
        if previous != domain:
            raise ValueError(f"risk unit {unit_id!r} appears in multiple domains.")

    class_weight = np.where(y == 1, float(pos_weight), 1.0)
    total_class_mass = float(class_weight.sum())
    multipliers = np.empty(len(y), dtype=np.float64)
    domain_records: list[dict[str, object]] = []
    for domain, target_mass in sorted(masses.items()):
        domain_mask = domains == domain
        domain_units = (
            ["all_epochs"]
            if within_domain_unit == "epoch"
            else sorted(set(units[domain_mask].tolist()))
        )
        for unit_id in domain_units:
            unit_mask = domain_mask if unit_id == "all_epochs" else domain_mask & (units == unit_id)
            class_mass = float(class_weight[unit_mask].sum())
            multipliers[unit_mask] = (
                total_class_mass * target_mass / (len(domain_units) * class_mass)
            )
        achieved = float(
            np.sum(multipliers[domain_mask] * class_weight[domain_mask]) / total_class_mass
        )
        domain_records.append(
            {
                "domain_id": domain,
                "target_mass": target_mass,
                "achieved_class_weighted_mass": achieved,
                "rows": int(domain_mask.sum()),
                "units": len(domain_units),
                "label_counts": np.bincount(y[domain_mask], minlength=2).astype(int).tolist(),
            }
        )
    if not np.isclose(
        np.sum(multipliers * class_weight) / total_class_mass,
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("source risk weights do not preserve weighted-CE objective scale.")
    report = {
        "domain_mass": dict(sorted(masses.items())),
        "within_domain_unit": within_domain_unit,
        "domains": domain_records,
        "row_multiplier_min": float(multipliers.min()),
        "row_multiplier_max": float(multipliers.max()),
        "effective_positive_prior": float(np.sum(multipliers * y) / np.sum(multipliers)),
    }
    return multipliers.astype(np.float32), report


def _selection_row_weights(
    labels: np.ndarray,
    unit_ids: np.ndarray,
    *,
    within_domain_unit: str,
    pos_weight: float,
) -> np.ndarray:
    if within_domain_unit == "epoch":
        return np.ones(len(labels), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    units = np.asarray(unit_ids).astype(str)
    class_weight = np.where(y == 1, float(pos_weight), 1.0)
    total = float(class_weight.sum())
    unique_units = sorted(set(units.tolist()))
    output = np.empty(len(y), dtype=np.float64)
    for unit_id in unique_units:
        mask = units == unit_id
        output[mask] = total / (len(unique_units) * float(class_weight[mask].sum()))
    return output.astype(np.float32)


class _CrossEntropyObjective(nn.Module):
    """Weighted CE that preserves PyTorch's class-weight denominator."""

    def __init__(self, pos_weight: float, device: torch.device) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, float(pos_weight)], device=device),
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        row_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numerator, denominator = self.components(logits, labels, row_weights)
        return numerator / denominator

    def components(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        row_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return additive sufficient statistics for split-independent validation."""

        rows = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        if row_weights is not None:
            rows = rows * row_weights
        denominator = self.class_weights.index_select(0, labels).sum()
        return rows.sum(), denominator


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
    source_validation_all_domains: bool = False
    early_stop_min_delta: float = 1e-6
    precision: str = "auto"
    fused_adam: bool = DEFAULT_FUSED_ADAM
    compile_mode: str | None = DEFAULT_COMPILE_MODE
    materialize_shuffled_epoch: bool = False
    max_update_batch_size: int | None = 512
    batch_memory_fraction: float = 0.55
    preload_memory_fraction: float = 0.30

    def __post_init__(self) -> None:
        if not isinstance(self.source_validation_all_domains, bool):
            raise ValueError("source_validation_all_domains must be boolean")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("DeepConfig epochs and batch_size must be positive.")
        if self.lr <= 0.0 or self.weight_decay < 0.0 or self.pos_weight <= 0.0:
            raise ValueError(
                "DeepConfig lr/pos_weight must be positive and weight_decay non-negative."
            )
        if self.early_stop_patience < 1 or self.early_stop_min_delta < 0.0:
            raise ValueError(
                "DeepConfig early-stop patience must be positive and delta non-negative."
            )
        if self.val_groups_min < 1 or self.val_groups_max < self.val_groups_min:
            raise ValueError("DeepConfig validation group bounds are invalid.")
        if self.val_group_frac is not None and not 0.0 < self.val_group_frac < 1.0:
            raise ValueError("DeepConfig val_group_frac must be in (0,1) or None.")
        if self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("DeepConfig precision must be 'auto', 'bf16', or 'fp32'.")
        if not isinstance(self.fused_adam, bool):
            raise ValueError("DeepConfig fused_adam must be boolean.")
        if not isinstance(self.materialize_shuffled_epoch, bool):
            raise ValueError("DeepConfig materialize_shuffled_epoch must be boolean.")
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
    fit_accepts_input_stats_row_mask = True
    fit_accepts_source_risk = True
    fit_accepts_decision_metadata = True
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
            if runtime.device.type != requested_device.type or (
                requested_device.index is not None
                and runtime.device.index != requested_device.index
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
        self.last_history: dict[str, object] = {
            "train_losses": [],
            "val_losses": [],
            "best_epoch": None,
        }
        self._evaluation_fold_id: int | None = None
        self.last_val_groups: int | None = None
        self.calibration_logits_: np.ndarray | None = None
        self.calibration_labels_: np.ndarray | None = None
        self.calibration_source_: str | None = None
        self.training_pos_weight_: float | None = None
        self.training_prior_: float | None = None
        self.last_runtime: dict[str, object] = {}
        self.last_source_risk: dict[str, object] | None = None
        self._feature_alignment_weight = 0.0
        self._feature_alignment_mode = "mean"
        self._feature_alignment_min_cell_count = 8
        self._feature_alignment_inactive_policy = "record"

        self._feature_private_domain_weight = 0.0
        self._feature_orthogonality_weight = 0.0
        self._alignment_domain_vocabulary: tuple[str, ...] = ()
        self.last_feature_alignment: dict[str, object] | None = None
        self._source_residual_active = False
        self.last_source_residual: dict[str, object] | None = None
        self._decision_loss_config: DecisionLossConfig | None = None
        self._decision_context_active = False
        self.last_decision_loss: dict[str, object] | None = None

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
            return sum(
                parameter.numel() for parameter in fitted.parameters() if parameter.requires_grad
            )
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
        loss_fn: _CrossEntropyObjective | None,
        batch_size: int,
    ) -> tuple[float | None, np.ndarray, np.ndarray]:
        """Run a vectorized batch pass and return CPU scores/labels once."""

        score_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        total_loss: torch.Tensor | None = None
        n_rows = 0
        self.model_.eval()
        with torch.inference_mode():
            for xb, yb, row_weights, domain_batch in source.iter_batches(batch_size):
                assert yb is not None
                with self._autocast_ctx():
                    if getattr(self, "splitbn_config", None) is not None:
                        if domain_batch is None:
                            raise RuntimeError(
                                "SplitBN source validation lacks per-row domain indices."
                            )
                        logits = self.model_(
                            xb,
                            domain_indices=domain_batch,
                            eval_buffer_source=SPLITBN_SOURCE_DOMAIN_BUFFER,
                        )
                    else:
                        logits = self.model_(xb)
                    loss = None if loss_fn is None else loss_fn(logits, yb, row_weights)
                score_parts.append((logits[:, 1] - logits[:, 0]).float())
                label_parts.append(yb)
                if loss is not None:
                    weighted_loss = loss.detach().float() * len(xb)
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                n_rows += len(xb)
        scores = torch.cat(score_parts).cpu().numpy().astype(np.float64)
        labels = torch.cat(label_parts).cpu().numpy().astype(np.int64)
        mean_loss = None if total_loss is None else float(total_loss.cpu()) / max(n_rows, 1)
        return (
            mean_loss,
            scores,
            labels,
        )

    def _validation_pass_with_oom_retry(
        self,
        source: MatrixBatchSource,
        loss_fn: _CrossEntropyObjective | None,
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

    def _decision_validation_pass(
        self,
        source: MatrixBatchSource,
        loss_fn: _CrossEntropyObjective,
        decision_module: DecisionLevelLoss,
        decision_plan: DecisionBatchPlan,
        config: DecisionLossConfig,
    ) -> dict[str, object]:
        """Evaluate the exact registered P2 objective using additive denominators."""

        score_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        trial_numerator: torch.Tensor | None = None
        trial_denominator: torch.Tensor | None = None
        candidate_sum: torch.Tensor | None = None
        prefix_sum: torch.Tensor | None = None
        requested_decisions = 0
        batches = decision_plan.batches(seed=None)
        index_batches = tuple(
            torch.from_numpy(np.array(batch.row_indices, dtype=np.int64, copy=True))
            for batch in batches
        )
        self.model_.eval()
        with torch.inference_mode():
            for decision_batch, (xb, yb, row_weights, domain_batch) in zip(
                batches,
                source.iter_index_batches(index_batches),
                strict=True,
            ):
                assert yb is not None
                with self._autocast_ctx():
                    if getattr(self, "splitbn_config", None) is not None:
                        if domain_batch is None:
                            raise RuntimeError(
                                "SplitBN source validation lacks per-row domain indices."
                            )
                        logits = self.model_(
                            xb,
                            domain_indices=domain_batch,
                            eval_buffer_source=SPLITBN_SOURCE_DOMAIN_BUFFER,
                        )
                    else:
                        logits = self.model_(xb)
                    numerator, denominator = loss_fn.components(logits, yb, row_weights)
                    targets = self.runtime.to_device(
                        torch.from_numpy(
                            np.array(
                                decision_batch.target_indices,
                                dtype=np.int64,
                                copy=True,
                            )
                        )
                    )
                    candidate_loss, prefix_losses = decision_module.packed_loss(
                        logits[:, 1] - logits[:, 0],
                        targets,
                        max_repetitions=config.max_repetitions,
                        prefix_weights=config.prefix_weights,
                        validate=False,
                    )
                detached_numerator = numerator.detach().float()
                detached_denominator = denominator.detach().float()
                decision_count = len(decision_batch.decision_ids)
                detached_candidate_sum = candidate_loss.detach().float() * decision_count
                detached_prefix_sum = prefix_losses.detach().float() * decision_count
                trial_numerator = (
                    detached_numerator
                    if trial_numerator is None
                    else trial_numerator + detached_numerator
                )
                trial_denominator = (
                    detached_denominator
                    if trial_denominator is None
                    else trial_denominator + detached_denominator
                )
                candidate_sum = (
                    detached_candidate_sum
                    if candidate_sum is None
                    else candidate_sum + detached_candidate_sum
                )
                prefix_sum = (
                    detached_prefix_sum if prefix_sum is None else prefix_sum + detached_prefix_sum
                )
                requested_decisions += decision_count
                score_parts.append((logits[:, 1] - logits[:, 0]).detach().float())
                label_parts.append(yb.detach())
        if (
            trial_numerator is None
            or trial_denominator is None
            or candidate_sum is None
            or prefix_sum is None
            or requested_decisions != len(decision_plan.decision_ids)
        ):
            raise RuntimeError("P2 validation did not cover the registered decision plan.")
        trial_mean = trial_numerator / trial_denominator
        candidate_mean = candidate_sum / requested_decisions
        objective = trial_mean + config.lambda_candidate * candidate_mean
        return {
            "objective_loss": float(objective.cpu()),
            "trial_loss": float(trial_mean.cpu()),
            "candidate_loss": float(candidate_mean.cpu()),
            "candidate_loss_by_prefix": [
                float(value) for value in (prefix_sum / requested_decisions).cpu().tolist()
            ],
            "requested_decisions": requested_decisions,
            "scores": torch.cat(score_parts).cpu().numpy().astype(np.float64),
            "labels": torch.cat(label_parts).cpu().numpy().astype(np.int64),
        }

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
        input_stats_row_mask: np.ndarray | None = None,
        source_domain_ids: np.ndarray | None = None,
        source_domain_mass: Mapping[str, float] | None = None,
        risk_unit_ids: np.ndarray | None = None,
        risk_within_domain_unit: str = "participant",
        selection_domain: str | None = None,
        feature_alignment_weight: float = 0.0,
        feature_alignment_mode: str = "mean",
        feature_alignment_min_cell_count: int = 8,
        feature_alignment_inactive_policy: str = "record",
        feature_private_domain_weight: float = 0.0,
        feature_orthogonality_weight: float = 0.0,
        decision_metadata: DirectDecisionMetadata | None = None,
        decision_loss_config: DecisionLossConfig | None = None,
    ) -> DeepBaseline:
        """Fit with a bounded OOM retry that never accumulates gradients."""

        if not np.isfinite(feature_alignment_weight) or feature_alignment_weight < 0.0:
            raise ValueError("feature_alignment_weight must be finite and non-negative.")
        if feature_alignment_mode not in {"mean", "conditional_rbf_mmd"}:
            raise ValueError("feature_alignment_mode must be 'mean' or 'conditional_rbf_mmd'.")
        if (
            isinstance(feature_alignment_min_cell_count, bool)
            or feature_alignment_min_cell_count < 2
        ):
            raise ValueError("feature_alignment_min_cell_count must be at least two.")
        if feature_alignment_inactive_policy not in {"record", "fail_closed"}:
            raise ValueError("feature_alignment_inactive_policy must be 'record' or 'fail_closed'.")

        for name, value in (
            ("feature_private_domain_weight", feature_private_domain_weight),
            ("feature_orthogonality_weight", feature_orthogonality_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if decision_loss_config is not None and not isinstance(
            decision_loss_config, DecisionLossConfig
        ):
            raise TypeError("decision_loss_config must be DecisionLossConfig or None.")
        if decision_metadata is not None and not isinstance(
            decision_metadata, DirectDecisionMetadata
        ):
            raise TypeError("decision_metadata must be DirectDecisionMetadata or None.")
        if decision_metadata is not None and decision_loss_config is None:
            raise ValueError("decision_metadata requires an explicit DecisionLossConfig.")
        if decision_loss_config is not None and decision_metadata is None:
            ledger = invalid_decision_ledger(
                phase="fit",
                reason="decision_loss_config_requires_direct_decision_metadata",
                metadata=None,
                config=decision_loss_config,
            )
            self.last_decision_loss = ledger.record()
            raise DirectDecisionContractError(ledger)
        decision_context_active = bool(
            decision_loss_config is not None and decision_metadata is not None
        )
        if decision_context_active and source_domain_mass is not None:
            assert decision_loss_config is not None
            ledger = invalid_decision_ledger(
                phase="fit",
                reason=(
                    "row_weighted_source_risk_has_no_preregistered_decision_level_"
                    "domain_or_participant_weighting_estimand"
                ),
                metadata=decision_metadata,
                config=decision_loss_config,
            )
            self.last_decision_loss = ledger.record()
            raise DirectDecisionContractError(ledger)
        if (
            decision_loss_config is not None
            and decision_loss_config.active
            and any(
                value > 0.0
                for value in (
                    feature_alignment_weight,
                    feature_private_domain_weight,
                    feature_orthogonality_weight,
                )
            )
        ):
            raise ValueError(
                "active P2 admits only L_trial + lambda_candidate*L_candidate; "
                "feature auxiliary losses require a separate experiment arm."
            )
        transfer = getattr(self, "feature_transfer", None)
        splitbn_config = getattr(self, "splitbn_config", None)
        splitbn_active = splitbn_config is not None
        if splitbn_active:
            self.optimizer_execution = self.optimizer_execution.without_cuda_graphs(
                reason="splitbn_cuda_graph_domain_buffer_updates_not_preserved"
            )
        source_residual_active = bool(
            transfer is not None and int(getattr(transfer, "source_domain_count", 0)) >= 2
        )
        if feature_private_domain_weight > 0.0 and (
            transfer is None or int(getattr(transfer, "private_domain_count", 0)) < 2
        ):
            raise ValueError(
                "feature_private_domain_weight requires FeatureDomainTransfer "
                "with private_domain_count >= 2."
            )
        if feature_orthogonality_weight > 0.0 and transfer is None:
            raise ValueError("feature_orthogonality_weight requires FeatureDomainTransfer.")
        if source_residual_active and source_domain_ids is None:
            raise ValueError(
                "source residual training requires an explicit source_domain_ids axis."
            )
        if splitbn_active and source_domain_ids is None:
            raise ValueError("SplitBN training requires an explicit source_domain_ids axis.")
        indexed_domain_active = (
            source_residual_active or splitbn_active or feature_private_domain_weight > 0.0
        )
        declared_domain_vocabulary = getattr(self, "source_domain_vocabulary", None)
        if indexed_domain_active and declared_domain_vocabulary is None:
            raise ValueError(
                "indexed domain mechanisms require the baseline's explicit ordered "
                "source_domain_vocabulary."
            )
        if source_residual_active and not getattr(self, "supports_feature_transfer", False):
            raise ValueError("source residual training requires a FeatureDomainTransfer model.")
        if (
            feature_alignment_weight > 0.0
            or feature_private_domain_weight > 0.0
            or feature_orthogonality_weight > 0.0
        ):
            if source_domain_ids is None:
                raise ValueError(
                    "feature-transfer auxiliary losses require an explicit source-domain axis."
                )
            if not getattr(self, "supports_feature_alignment", False):
                raise ValueError(
                    "feature-transfer auxiliary losses require a model exposing structured trunk "
                    "features (N2P3NetBaseline)."
                )
        self._feature_alignment_weight = float(feature_alignment_weight)
        self._feature_alignment_mode = feature_alignment_mode
        self._feature_alignment_min_cell_count = int(feature_alignment_min_cell_count)
        self._feature_alignment_inactive_policy = feature_alignment_inactive_policy

        self._feature_private_domain_weight = float(feature_private_domain_weight)
        self._feature_orthogonality_weight = float(feature_orthogonality_weight)
        self._source_residual_active = source_residual_active
        self._decision_loss_config = decision_loss_config
        self._decision_context_active = decision_context_active
        self.last_decision_loss = {
            "schema": DECISION_TRAINING_LEDGER_SCHEMA,
            "status": "pending" if decision_context_active else "disabled",
            "valid": True,
            "reason": None,
            "config": (None if decision_loss_config is None else decision_loss_config.record()),
            "metadata": (
                None if decision_metadata is None else decision_metadata.record(include_rows=False)
            ),
            "loss_scope": "trial_plus_lambda_candidate",
            "trial_weight": 1.0,
            "candidate_component_active": bool(
                decision_loss_config is not None and decision_loss_config.active
            ),
        }
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
                        input_stats_row_mask=input_stats_row_mask,
                        source_domain_ids=source_domain_ids,
                        source_domain_mass=source_domain_mass,
                        risk_unit_ids=risk_unit_ids,
                        risk_within_domain_unit=risk_within_domain_unit,
                        selection_domain=selection_domain,
                        feature_alignment_mode=feature_alignment_mode,
                        feature_alignment_min_cell_count=feature_alignment_min_cell_count,
                        feature_alignment_inactive_policy=feature_alignment_inactive_policy,
                        decision_metadata=decision_metadata,
                        decision_loss_config=decision_loss_config,
                        requested_batch_size=requested_batch_size,
                    )
                self.last_runtime = {
                    "device": str(self.device),
                    "precision": self.runtime.precision.name,
                    "batch_size": self._active_batch_size,
                    "validation_batch_size": self._validation_batch_size,
                    "preloaded": self._preloaded_batches,
                    "materialize_shuffled_epoch": self._materialize_shuffled_epoch,
                    "training_order": "complete_decision_batches"
                    if self._decision_context_active
                    else "per_epoch_random_permutation",
                    "transfer_fallback": self._transfer_fallback,
                    **self.optimizer_execution.record(),
                    "host_sync_policy": "epoch_boundary",
                    "oom_retries": retries,
                    "decision_grouped_batches": self._decision_context_active,
                    "decision_batch_contract": (
                        None
                        if not self._decision_context_active or self.last_decision_loss is None
                        else self.last_decision_loss.get("training_batch_plan")
                    ),
                    "shared_worker_count": self.runtime.shared_worker_count,
                    "memory": self.runtime.memory_record(),
                }
                return self
            except RuntimeError as error:
                if not is_oom_error(error):
                    self._clear_failed_fit(release_memory=False)
                    raise
                if self._decision_context_active:
                    assert self._decision_loss_config is not None
                    ledger = invalid_decision_ledger(
                        phase="fit",
                        reason="oom_cannot_change_complete_decision_batch_contract",
                        metadata=decision_metadata,
                        config=self._decision_loss_config,
                        batch_plan=(
                            None
                            if self.last_decision_loss is None
                            else self.last_decision_loss.get("training_batch_plan")
                        ),
                    )
                    self.last_decision_loss = ledger.record()
                    self._clear_failed_fit(release_memory=False)
                    raise DirectDecisionContractError(ledger) from error
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
        input_stats_row_mask: np.ndarray | None,
        source_domain_ids: np.ndarray | None,
        source_domain_mass: Mapping[str, float] | None,
        risk_unit_ids: np.ndarray | None,
        risk_within_domain_unit: str,
        selection_domain: str | None,
        feature_alignment_mode: str,
        feature_alignment_min_cell_count: int,
        feature_alignment_inactive_policy: str,
        decision_metadata: DirectDecisionMetadata | None,
        decision_loss_config: DecisionLossConfig | None,
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
        decision_context_active = self._decision_context_active
        full_decision_ledger: dict[str, object] | None = None
        if decision_context_active:
            assert decision_metadata is not None and decision_loss_config is not None
            if decision_metadata.n_rows != len(X):
                ledger = invalid_decision_ledger(
                    phase="all_rows_preflight",
                    reason="decision_metadata_rows_do_not_align_with_X",
                    metadata=decision_metadata,
                    config=decision_loss_config,
                )
                self.last_decision_loss = ledger.record()
                raise DirectDecisionContractError(ledger)
            try:
                _, preflight = build_decision_batch_plan(
                    decision_metadata,
                    y,
                    decision_loss_config,
                    max_batch_rows=max(len(X), decision_loss_config.min_rows_per_batch),
                    phase="all_rows_preflight",
                )
            except DirectDecisionContractError as error:
                self.last_decision_loss = error.record
                raise
            full_decision_ledger = preflight.record()
        trial_mask = self._effective_trial_channel_mask(X, trial_channel_mask)
        if input_stats_row_mask is None:
            requested_stats_rows = np.ones(len(X), dtype=bool)
        else:
            requested_stats_rows = np.asarray(input_stats_row_mask, dtype=bool)
            if requested_stats_rows.shape != (len(X),):
                raise ValueError("input_stats_row_mask must align with X.")
            if not bool(requested_stats_rows.any()):
                raise ValueError("input_stats_row_mask must retain at least one row.")

        if selection_domain is not None and source_domain_ids is None:
            raise ValueError("selection_domain requires an explicit source_domain_ids axis.")
        if (source_domain_mass is None) != (risk_unit_ids is None):
            raise ValueError(
                "weighted source risk requires domain mass and risk unit ids together."
            )
        if source_domain_mass is not None and source_domain_ids is None:
            raise ValueError("weighted source risk requires an explicit source-domain axis.")
        risk_domains = None
        risk_units = None
        if source_domain_ids is not None:
            risk_domains = np.asarray(source_domain_ids).astype(str)
            if risk_domains.shape != (len(X),):
                raise ValueError("source_domain_ids must align with X.")
            if selection_domain is not None and str(selection_domain) not in set(
                risk_domains.tolist()
            ):
                raise ValueError("selection_domain is absent from source rows.")
        if risk_unit_ids is not None:
            risk_units = np.asarray(risk_unit_ids).astype(str)
            if risk_units.shape != (len(X),):
                raise ValueError("risk_unit_ids must align with X.")
        self.last_source_risk = None

        if decision_context_active and risk_domains is not None:
            assert decision_metadata is not None and decision_loss_config is not None
            domain_records: list[dict[str, object]] = []
            invalid_domain_decisions = False
            for decision_index, decision_id in enumerate(decision_metadata.requested_decision_ids):
                rows = decision_metadata.decision_indices == decision_index
                domains = sorted(set(risk_domains[rows].tolist()))
                domain_records.append({"decision_id": decision_id, "source_domains": domains})
                invalid_domain_decisions = invalid_domain_decisions or len(domains) != 1
            if invalid_domain_decisions:
                ledger = invalid_decision_ledger(
                    phase="source_domain_axis",
                    reason="each_direct_decision_must_belong_to_exactly_one_source_domain",
                    metadata=decision_metadata,
                    config=decision_loss_config,
                    decisions=domain_records,
                )
                self.last_decision_loss = ledger.record()
                raise DirectDecisionContractError(ledger)

        if group_ids is None:
            train_mask = np.ones(len(X), dtype=bool)
            val_mask = np.zeros(len(X), dtype=bool)
            self.last_val_groups = None
        else:
            group_ids = np.asarray(group_ids)
            if group_ids.shape != (len(X),):
                raise ValueError("group_ids must align with X.")
            if decision_context_active:
                assert decision_metadata is not None and decision_loss_config is not None
                validation_unit_records: list[dict[str, object]] = []
                invalid_validation_units = False
                for decision_index, decision_id in enumerate(
                    decision_metadata.requested_decision_ids
                ):
                    rows = decision_metadata.decision_indices == decision_index
                    units = sorted(set(group_ids[rows].astype(str).tolist()))
                    validation_unit_records.append(
                        {"decision_id": decision_id, "validation_units": units}
                    )
                    invalid_validation_units = invalid_validation_units or len(units) != 1
                if invalid_validation_units:
                    ledger = invalid_decision_ledger(
                        phase="validation_group_axis",
                        reason=("each_direct_decision_must_belong_to_exactly_one_validation_group"),
                        metadata=decision_metadata,
                        config=decision_loss_config,
                        decisions=validation_unit_records,
                    )
                    self.last_decision_loss = ledger.record()
                    raise DirectDecisionContractError(ledger)
            split_scope = (
                np.ones(len(X), dtype=bool)
                if selection_domain is None
                else risk_domains == str(selection_domain)
            )
            if self.cfg.source_validation_all_domains:
                if risk_domains is None or source_domain_mass is None or selection_domain is not None:
                    raise ValueError("all-domain validation requires domain risk and no single selection domain")
                split = domain_stratified_group_validation_split(
                    group_ids[split_scope], risk_domains[split_scope],
                    fraction=self.cfg.val_group_frac, min_groups=self.cfg.val_groups_min,
                    max_groups=self.cfg.val_groups_max, seed=self.cfg.seed,
                )
            else:
                split = group_disjoint_validation_split(
                    group_ids[split_scope],
                    fraction=self.cfg.val_group_frac,
                    min_groups=self.cfg.val_groups_min,
                    max_groups=self.cfg.val_groups_max,
                    seed=self.cfg.seed,
                )
            train_mask = np.ones(len(X), dtype=bool)
            val_mask = np.zeros(len(X), dtype=bool)
            train_mask[split_scope] = split.train_mask
            val_mask[split_scope] = split.validation_mask
            self.last_val_groups = split.n_validation_groups
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        train_decision_metadata: DirectDecisionMetadata | None = None
        val_decision_metadata: DirectDecisionMetadata | None = None
        if decision_context_active:
            assert decision_metadata is not None and decision_loss_config is not None
            train_decisions: list[str] = []
            val_decisions: list[str] = []
            split_records: list[dict[str, object]] = []
            split_invalid = False
            for decision_index, decision_id in enumerate(decision_metadata.requested_decision_ids):
                rows = decision_metadata.decision_indices == decision_index
                in_train = bool(np.any(rows & train_mask))
                in_validation = bool(np.any(rows & val_mask))
                if in_train and in_validation:
                    split_invalid = True
                    status = "split_across_train_and_validation"
                elif in_train:
                    train_decisions.append(decision_id)
                    status = "train"
                elif in_validation:
                    val_decisions.append(decision_id)
                    status = "validation"
                else:
                    split_invalid = True
                    status = "absent_after_split"
                split_records.append({"decision_id": decision_id, "split_status": status})
            if split_invalid:
                ledger = invalid_decision_ledger(
                    phase="train_validation_split",
                    reason="a_complete_decision_was_split_or_removed",
                    metadata=decision_metadata,
                    config=decision_loss_config,
                    decisions=split_records,
                )
                self.last_decision_loss = ledger.record()
                raise DirectDecisionContractError(ledger)
            train_decision_metadata = decision_metadata.subset_for_decisions(train_decisions)
            if val_decisions:
                val_decision_metadata = decision_metadata.subset_for_decisions(val_decisions)
        train_channel_mask = trial_mask[train_mask]
        val_channel_mask = trial_mask[val_mask]
        unobserved_train_channels = self.channel_mask & ~train_channel_mask.any(axis=0)
        if bool(unobserved_train_channels.any()):
            missing = np.flatnonzero(unobserved_train_channels).tolist()
            raise ValueError(
                "Training split never observes active channels "
                f"{missing}; use an intersection layout or mark them permanently absent."
            )
        stats_mask = train_mask & requested_stats_rows
        if not bool(stats_mask.any()):
            raise ValueError(
                "input_stats_row_mask retains no rows in the group-disjoint training split."
            )
        stats_channel_mask = trial_mask[stats_mask]
        unobserved_stats_channels = self.channel_mask & ~stats_channel_mask.any(axis=0)
        if bool(unobserved_stats_channels.any()):
            missing = np.flatnonzero(unobserved_stats_channels).tolist()
            raise ValueError(
                "Input-statistics rows never observe active channels "
                f"{missing}; choose a complete statistics domain."
            )
        if set(np.unique(y_train).tolist()) != {0, 1}:
            raise ValueError("Deep training split must contain both binary classes.")
        if len(y_val) and set(np.unique(y_val).tolist()) != {0, 1}:
            raise ValueError("Deep validation split must contain both binary classes.")
        train_row_weights = None
        val_row_weights = None
        self.training_pos_weight_ = float(self.cfg.pos_weight)
        if risk_domains is not None and risk_units is not None and source_domain_mass is not None:
            train_row_weights, train_risk_record = resolve_source_risk_weights(
                y_train,
                risk_domains[train_mask],
                risk_units[train_mask],
                domain_mass=source_domain_mass,
                within_domain_unit=risk_within_domain_unit,
                pos_weight=self.cfg.pos_weight,
            )
            validation_risk_record = None
            if len(y_val):
                if self.cfg.source_validation_all_domains:
                    val_row_weights, validation_risk_record = resolve_source_risk_weights(
                        y_val, risk_domains[val_mask], risk_units[val_mask],
                        domain_mass=source_domain_mass, within_domain_unit=risk_within_domain_unit,
                        pos_weight=self.cfg.pos_weight,
                    )
                else:
                    val_row_weights = _selection_row_weights(
                        y_val,
                        risk_units[val_mask],
                        within_domain_unit=risk_within_domain_unit,
                        pos_weight=self.cfg.pos_weight,
                    )
            self.training_prior_ = float(train_risk_record["effective_positive_prior"])
            self.last_source_risk = {
                "training": train_risk_record,
                "validation": validation_risk_record,
                "selection_scope": "all_domains" if self.cfg.source_validation_all_domains else "selected_domain",
                "selection_domain": selection_domain,
                "selection_rows": int(val_mask.sum()),
                "selection_units": int(len(set(risk_units[val_mask].tolist())))
                if bool(val_mask.any())
                else 0,
            }
        else:
            self.training_prior_ = float(y_train.mean())
            if risk_domains is not None:
                self.last_source_risk = {
                    "training": {
                        "mode": "natural_epoch_ce",
                        "rows": int(train_mask.sum()),
                    },
                    "selection_domain": selection_domain,
                    "selection_rows": int(val_mask.sum()),
                    "selection_units": int(len(set(group_ids[val_mask].tolist())))
                    if group_ids is not None and bool(val_mask.any())
                    else 0,
                }

        self._input_mean, self._input_std = self._masked_input_stats(
            X[stats_mask],
            stats_channel_mask,
        )
        # Domain ids remain metadata at inference. During an explicitly
        # configured source-training arm they also select the matching
        # drop-to-consensus residual; target/eval paths never use that route.
        alignment_weight = self._feature_alignment_weight
        private_weight = self._feature_private_domain_weight
        orthogonality_weight = self._feature_orthogonality_weight
        splitbn_config = getattr(self, "splitbn_config", None)
        splitbn_active = splitbn_config is not None
        source_residual_active = self._source_residual_active
        alignment_active = bool(alignment_weight > 0.0 and risk_domains is not None)
        private_active = bool(private_weight > 0.0 and risk_domains is not None)
        orthogonality_active = bool(orthogonality_weight > 0.0)
        feature_objective_active = (
            alignment_active or private_active or orthogonality_active or source_residual_active
        )
        domain_axis_active = feature_objective_active or splitbn_active
        if private_active and len(set(risk_domains.tolist())) < 2:
            raise ValueError("private domain supervision requires at least two source domains.")
        train_domain_tensor: torch.Tensor | None = None
        val_domain_tensor: torch.Tensor | None = None
        self._alignment_domain_vocabulary = ()
        if domain_axis_active:
            assert risk_domains is not None
            observed_domains = set(risk_domains.tolist())
            indexed_domain_active = source_residual_active or splitbn_active or private_active
            declared_domain_vocabulary = getattr(
                self,
                "source_domain_vocabulary",
                None,
            )
            if indexed_domain_active:
                if declared_domain_vocabulary is None:
                    raise ValueError(
                        "indexed domain mechanisms require the baseline's explicit "
                        "ordered source_domain_vocabulary."
                    )
                axis_record = source_domain_axis_record(tuple(declared_domain_vocabulary))
                domain_vocabulary = tuple(axis_record["vocabulary"])
                if observed_domains != set(domain_vocabulary):
                    raise ValueError(
                        "observed source domains must exactly match the declared "
                        "source_domain_vocabulary."
                    )
                if private_active and len(domain_vocabulary) != int(
                    self.feature_transfer.private_domain_count
                ):
                    raise ValueError(
                        "private_domain_count must exactly match the declared "
                        "source_domain_vocabulary."
                    )
            else:
                domain_vocabulary = tuple(sorted(observed_domains))
            if source_residual_active:
                configured_count = int(self.feature_transfer.source_domain_count)
                if len(domain_vocabulary) != configured_count:
                    raise ValueError(
                        "source_domain_count must exactly match the observed source-domain "
                        f"vocabulary ({configured_count} configured, "
                        f"{len(domain_vocabulary)} observed)."
                    )
                training_vocabulary = set(risk_domains[train_mask].tolist())
                if training_vocabulary != set(domain_vocabulary):
                    missing = sorted(set(domain_vocabulary) - training_vocabulary)
                    raise ValueError(
                        f"source residual training split omits configured domains {missing}."
                    )
            if splitbn_active:
                configured_count = int(splitbn_config.domain_count)
                if len(domain_vocabulary) != configured_count:
                    raise ValueError(
                        "SplitBN domain_count must exactly match the observed source-domain "
                        f"vocabulary ({configured_count} configured, "
                        f"{len(domain_vocabulary)} observed)."
                    )
                splitbn_vocabulary = getattr(
                    splitbn_config,
                    "domain_vocabulary",
                    domain_vocabulary,
                )
                if tuple(splitbn_vocabulary) != domain_vocabulary:
                    raise ValueError(
                        "SplitBN vocabulary disagrees with the baseline model-domain axis."
                    )
                training_vocabulary = set(risk_domains[train_mask].tolist())
                if training_vocabulary != set(domain_vocabulary):
                    missing = sorted(set(domain_vocabulary) - training_vocabulary)
                    raise ValueError(f"SplitBN training split omits configured domains {missing}.")
            domain_to_index = {domain: index for index, domain in enumerate(domain_vocabulary)}
            self._alignment_domain_vocabulary = domain_vocabulary
            train_domain_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    np.array(
                        [domain_to_index[domain] for domain in risk_domains[train_mask]],
                        dtype=np.int64,
                    )
                )
            )
            if len(X_val):
                val_domain_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        np.array(
                            [domain_to_index[domain] for domain in risk_domains[val_mask]],
                            dtype=np.int64,
                        )
                    )
                )
        self.last_feature_alignment = {
            "weight": alignment_weight,
            "mode": self._feature_alignment_mode,
            "min_cell_count": self._feature_alignment_min_cell_count,
            "inactive_policy": self._feature_alignment_inactive_policy,
            "inactive_batch_count": 0,
            "negative_mmd_batch_count": 0,
            "alignment_active": alignment_active,
            "private_domain_weight": private_weight,
            "private_domain_active": private_active,
            "orthogonality_weight": orthogonality_weight,
            "orthogonality_active": orthogonality_active,
            "domain_vocabulary": list(self._alignment_domain_vocabulary),
            "epoch_mean_squared_distance": [],
            "epoch_mmd_squared": [],
            "epoch_negative_mmd_terms": [],
            "epoch_private_domain_loss": [],
            "epoch_orthogonality_loss": [],
        }
        self.last_source_residual = {
            "schema": SOURCE_RESIDUAL_LEDGER_SCHEMA,
            "active": source_residual_active,
            "status": "active" if source_residual_active else "disabled",
            "valid": True,
            "domain_vocabulary": list(self._alignment_domain_vocabulary),
            "source_domain_count": (
                int(self.feature_transfer.source_domain_count) if source_residual_active else 0
            ),
            "source_drop_p": (
                float(self.feature_transfer.source_drop_p) if source_residual_active else None
            ),
            "source_drop_mode": (
                self.feature_transfer.source_drop_mode if source_residual_active else None
            ),
            "ratio_contract": source_residual_ratio_contract(),
            "epoch_loss_m0": [],
            "epoch_loss_m1": [],
            "epoch_loss_delta_m1_minus_m0": [],
            "epoch_trial_loss_m0": [],
            "epoch_trial_loss_m1": [],
            "epoch_trial_loss_delta_m1_minus_m0": [],
            "epoch_ratio_forwards": [],
            "epoch_domain_scales": [],
            "run_max_ratio": 0.0 if source_residual_active else None,
            "counterfactual_loss_scope": "trial_component",
            "counterfactual_excludes_candidate_component": True,
        }
        X_train = self._prepare_input(X_train, train_channel_mask)
        if len(X_val):
            X_val = self._prepare_input(X_val, val_channel_mask)

        seed = self._model_seed()
        self._initialize_model(seed)
        trainable_params = [
            parameter for parameter in self.model_.parameters() if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError(
                "freeze_prefixes froze every parameter; no optimizer update is possible."
            )
        batch_size = self.runtime.choose_batch_size(
            requested_batch_size,
            X_train.shape[1:],
            model=self.model_,
            max_update_batch_size=self.cfg.max_update_batch_size,
        )
        decision_module: DecisionLevelLoss | None = None
        train_decision_plan: DecisionBatchPlan | None = None
        val_decision_plan: DecisionBatchPlan | None = None
        train_decision_ledger: dict[str, object] | None = None
        val_decision_ledger: dict[str, object] | None = None
        if decision_context_active:
            assert decision_loss_config is not None
            assert train_decision_metadata is not None
            try:
                train_decision_plan, train_ledger = build_decision_batch_plan(
                    train_decision_metadata,
                    y_train,
                    decision_loss_config,
                    max_batch_rows=batch_size,
                    phase="training",
                )
            except DirectDecisionContractError as error:
                self.last_decision_loss = error.record
                raise
            train_decision_ledger = train_ledger.record()
            batch_size = train_decision_plan.max_batch_rows
            decision_module = DecisionLevelLoss(
                candidate_vocabulary=decision_loss_config.candidate_vocabulary,
                temperature=decision_loss_config.temperature,
            )
        self._active_batch_size = batch_size
        total_matrix_bytes = X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes
        if train_row_weights is not None:
            total_matrix_bytes += train_row_weights.nbytes
        if val_row_weights is not None:
            total_matrix_bytes += val_row_weights.nbytes
        preload = self.runtime.can_preload(total_matrix_bytes)
        train_source = MatrixBatchSource(
            torch.from_numpy(np.ascontiguousarray(X_train)),
            torch.from_numpy(np.ascontiguousarray(y_train)),
            self.runtime,
            preload=preload,
            row_weights=(
                None
                if train_row_weights is None
                else torch.from_numpy(np.ascontiguousarray(train_row_weights))
            ),
            row_domains=train_domain_tensor,
        )
        val_source = (
            MatrixBatchSource(
                torch.from_numpy(np.ascontiguousarray(X_val)),
                torch.from_numpy(np.ascontiguousarray(y_val)),
                self.runtime,
                preload=preload,
                row_weights=(
                    None
                    if val_row_weights is None
                    else torch.from_numpy(np.ascontiguousarray(val_row_weights))
                ),
                row_domains=val_domain_tensor,
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
        self._materialize_shuffled_epoch = (
            not decision_context_active
            and self.cfg.materialize_shuffled_epoch
            and self._preloaded_batches
            and self.runtime.can_preload(train_source.nbytes)
        )
        train_source.materialize_shuffled_epoch = self._materialize_shuffled_epoch
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
        if decision_context_active and val_source is not None:
            assert decision_loss_config is not None
            assert val_decision_metadata is not None
            try:
                val_decision_plan, val_ledger = build_decision_batch_plan(
                    val_decision_metadata,
                    y_val,
                    decision_loss_config,
                    max_batch_rows=validation_batch_size,
                    phase="validation",
                )
            except DirectDecisionContractError as error:
                self.last_decision_loss = error.record
                raise
            val_decision_ledger = val_ledger.record()
            validation_batch_size = val_decision_plan.max_batch_rows
        self._validation_batch_size = validation_batch_size

        if decision_context_active:
            assert self.last_decision_loss is not None
            self.last_decision_loss.update(
                {
                    "status": "valid",
                    "valid": True,
                    "all_rows_preflight": full_decision_ledger,
                    "training_preflight": train_decision_ledger,
                    "validation_preflight": val_decision_ledger,
                    "training_batch_plan": (
                        None if train_decision_plan is None else train_decision_plan.record()
                    ),
                    "validation_batch_plan": (
                        None if val_decision_plan is None else val_decision_plan.record()
                    ),
                    "requested_training_row_batch_size": int(self.cfg.batch_size),
                    "scheduler_training_row_budget": int(train_decision_plan.requested_max_rows),
                    "effective_training_row_batch_size": int(train_decision_plan.max_batch_rows),
                    "epoch_train_components": [],
                    "epoch_validation_components": [],
                    "early_stopping_objective": (
                        "trial_plus_lambda_candidate" if val_decision_plan is not None else None
                    ),
                    "p1_counterfactual_loss_scope": "trial_component",
                }
            )

        adam_kwargs: dict[str, object] = {
            "lr": self.cfg.lr,
            "weight_decay": self.cfg.weight_decay,
        }
        if self.optimizer_execution.fused_adam:
            adam_kwargs["fused"] = True
        if self.optimizer_execution.uses_cuda_graphs:
            adam_kwargs["capturable"] = True
        opt = torch.optim.Adam(trainable_params, **adam_kwargs)
        loss_fn = _CrossEntropyObjective(self.cfg.pos_weight, self.device)
        perm_gen = train_source.make_generator(seed)
        source_residual_mask_seed = seed + 1_000_003
        source_residual_generator = (
            train_source.make_generator(source_residual_mask_seed)
            if source_residual_active
            else None
        )
        if source_residual_active:
            assert self.last_source_residual is not None
            self.last_source_residual.update(
                {
                    "mask_rng": "explicit_torch_generator",
                    "mask_rng_seed": source_residual_mask_seed,
                }
            )
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_objective_losses: list[float] = []
        task_val_aucs: list[float | None] = []
        best_state = None
        best_epoch = None
        best_task_epoch = None
        best_val_loss = float("inf")
        best_task_val_loss = float("inf")
        patience_left = int(self.cfg.early_stop_patience)
        early_stop_triggered = False
        epoch_progress_callback = self.epoch_progress_callback()

        domain_count = len(getattr(self, "_alignment_domain_vocabulary", ()) or ())

        def _orthogonality_loss(
            invariant_features: torch.Tensor,
            private_summary: torch.Tensor,
        ) -> torch.Tensor:
            if invariant_features.ndim != 3 or private_summary.ndim != 2:
                raise RuntimeError("feature-transfer views have invalid ranks.")
            invariant_summary = invariant_features.mean(dim=-1)
            if invariant_summary.shape[0] < 2:
                return invariant_summary.new_zeros(())
            invariant_centered = invariant_summary - invariant_summary.mean(dim=0, keepdim=True)
            private_centered = private_summary - private_summary.mean(dim=0, keepdim=True)
            cross_cov = invariant_centered.transpose(0, 1) @ private_centered
            cross_cov = cross_cov / max(invariant_summary.shape[0] - 1, 1)
            return cross_cov.square().mean()

        def train_step(
            xb: torch.Tensor,
            yb: torch.Tensor,
            row_weights: torch.Tensor | None,
            domain_batch: torch.Tensor | None,
            decision_targets: torch.Tensor | None,
            source_residual_mask: torch.Tensor | None,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            dict[str, object] | None,
        ]:
            opt.zero_grad(set_to_none=True)
            alignment_diagnostics: dict[str, object] | None = None
            with self._autocast_ctx():
                if feature_objective_active:
                    assert domain_batch is not None
                    forwarded = self.model_.forward_with_features(
                        xb,
                        use_source_residual=source_residual_active,
                        source_residual_mask=source_residual_mask,
                        domain_indices=(
                            domain_batch if source_residual_active or splitbn_active else None
                        ),
                    )
                    if not isinstance(forwarded, tuple) or len(forwarded) < 2:
                        raise RuntimeError(
                            "feature-transfer model must return logits and adapted features."
                        )
                    logits, features = forwarded[:2]
                    trial_numerator, trial_denominator = loss_fn.components(logits, yb, row_weights)
                    trial_loss = trial_numerator / trial_denominator
                    loss = trial_loss
                    if alignment_active:
                        if self._feature_alignment_mode == "conditional_rbf_mmd":
                            alignment_loss, alignment_diagnostics = conditional_domain_rbf_mmd_loss(
                                features,
                                yb,
                                domain_batch,
                                domain_count,
                                min_cell_count=self._feature_alignment_min_cell_count,
                            )
                        else:
                            alignment_loss, alignment_diagnostics = conditional_mean_alignment_loss(
                                features,
                                yb,
                                domain_batch,
                                domain_count,
                                min_cell_count=self._feature_alignment_min_cell_count,
                            )
                        loss = loss + alignment_weight * alignment_loss
                        if self._feature_alignment_inactive_policy == "fail_closed" and bool(
                            alignment_diagnostics.get("alignment_inactive")
                        ):
                            raise RuntimeError(
                                "feature alignment became inactive in a training batch "
                                f"under fail_closed policy: {alignment_diagnostics}"
                            )

                    if private_active:
                        if len(forwarded) < 3 or not hasattr(self.model_, "private_domain_logits"):
                            raise RuntimeError(
                                "private domain loss requires FeatureDomainTransfer with a private head."
                            )
                        private_logits = self.model_.private_domain_logits(forwarded[2])
                        private_loss = F.cross_entropy(private_logits, domain_batch)
                        loss = loss + private_weight * private_loss
                        if alignment_diagnostics is None:
                            alignment_diagnostics = {}
                        alignment_diagnostics["private_domain_loss"] = float(private_loss.detach())
                    if orthogonality_active:
                        if len(forwarded) < 3:
                            raise RuntimeError(
                                "orthogonality loss requires FeatureDomainTransfer private features."
                            )
                        orthogonality_loss = _orthogonality_loss(features, forwarded[2])
                        loss = loss + orthogonality_weight * orthogonality_loss
                        if alignment_diagnostics is None:
                            alignment_diagnostics = {}
                        alignment_diagnostics["orthogonality_loss"] = float(
                            orthogonality_loss.detach()
                        )
                    if source_residual_active:
                        if not isinstance(self.model_, FeatureDomainTransfer):
                            raise RuntimeError(
                                "source residual training requires FeatureDomainTransfer."
                            )
                        with torch.no_grad():
                            consensus, residual, ratio_record = (
                                self.model_.source_residual_counterfactual_views(
                                    features.detach(),
                                    domain_batch,
                                )
                            )
                            logits_m0 = self.model_.classifier(self.model_.pool_features(consensus))
                            logits_m1 = self.model_.classifier(self.model_.pool_features(residual))
                            source_ledger = source_residual_loss_ledger(
                                loss_fn(logits_m0, yb, row_weights),
                                loss_fn(logits_m1, yb, row_weights),
                                ratio_record=ratio_record,
                            )
                            source_ledger["loss_scope"] = "trial_component"
                            source_ledger["trial_loss_m0"] = source_ledger["loss_m0"]
                            source_ledger["trial_loss_m1"] = source_ledger["loss_m1"]
                            source_ledger["trial_loss_delta_m1_minus_m0"] = source_ledger[
                                "loss_delta_m1_minus_m0"
                            ]
                        if alignment_diagnostics is None:
                            alignment_diagnostics = {}
                        alignment_diagnostics["source_residual"] = source_ledger
                else:
                    logits = (
                        self.model_(xb, domain_indices=domain_batch)
                        if splitbn_active
                        else self.model_(xb)
                    )
                    trial_numerator, trial_denominator = loss_fn.components(logits, yb, row_weights)
                    trial_loss = trial_numerator / trial_denominator
                    loss = trial_loss
                candidate_loss: torch.Tensor | None = None
                prefix_losses: torch.Tensor | None = None
                if decision_context_active:
                    assert decision_module is not None
                    assert decision_loss_config is not None
                    assert decision_targets is not None
                    candidate_loss, prefix_losses = decision_module.packed_loss(
                        logits[:, 1] - logits[:, 0],
                        decision_targets,
                        max_repetitions=decision_loss_config.max_repetitions,
                        prefix_weights=decision_loss_config.prefix_weights,
                        validate=False,
                    )
                    loss = loss + decision_loss_config.lambda_candidate * candidate_loss
            loss.backward()
            opt.step()
            if source_residual_active:
                if not isinstance(self.model_, FeatureDomainTransfer):
                    raise RuntimeError("source residual training requires FeatureDomainTransfer.")
                with torch.no_grad(), self._autocast_ctx():
                    # The pre-update ledger above diagnoses the actual loss
                    # point. Re-check after the optimizer step so the final
                    # batch cannot move a gate beyond r_max unnoticed.
                    self.model_.source_residual_counterfactual_views(
                        features.detach(),
                        domain_batch,
                    )
            return (
                loss.detach(),
                trial_numerator.detach(),
                trial_denominator.detach(),
                None if candidate_loss is None else candidate_loss.detach(),
                None if prefix_losses is None else prefix_losses.detach(),
                alignment_diagnostics,
            )

        if self.optimizer_execution.compile_mode is not None:
            train_step = torch.compile(
                train_step,
                mode=self.optimizer_execution.compile_mode,
                fullgraph=False,
            )

        def _training_batches(
            epoch: int,
        ):
            if train_decision_plan is None:
                for xb, yb, row_weights, domain_batch in train_source.iter_batches(
                    batch_size, generator=perm_gen
                ):
                    source_mask = None
                    if source_residual_active:
                        assert isinstance(self.model_, FeatureDomainTransfer)
                        assert source_residual_generator is not None
                        generator_device = getattr(
                            source_residual_generator,
                            "device",
                            torch.device("cpu"),
                        )
                        reference = (
                            xb
                            if torch.device(generator_device).type == xb.device.type
                            else torch.empty(
                                (len(xb),),
                                device=generator_device,
                                dtype=xb.dtype,
                            )
                        )
                        source_mask = self.model_.sample_source_residual_mask(
                            reference,
                            generator=source_residual_generator,
                        ).to(device=xb.device, dtype=xb.dtype)
                    yield xb, yb, row_weights, domain_batch, None, 0, source_mask
                return
            decision_batches = train_decision_plan.batches(seed=seed + epoch)
            index_batches = tuple(
                torch.from_numpy(np.array(batch.row_indices, dtype=np.int64, copy=True))
                for batch in decision_batches
            )
            for decision_batch, (xb, yb, row_weights, domain_batch) in zip(
                decision_batches,
                train_source.iter_index_batches(index_batches),
                strict=True,
            ):
                targets = self.runtime.to_device(
                    torch.from_numpy(
                        np.array(
                            decision_batch.target_indices,
                            dtype=np.int64,
                            copy=True,
                        )
                    )
                )
                source_mask = None
                if source_residual_active:
                    assert isinstance(self.model_, FeatureDomainTransfer)
                    assert source_residual_generator is not None
                    generator_device = getattr(
                        source_residual_generator,
                        "device",
                        torch.device("cpu"),
                    )
                    reference = (
                        xb
                        if torch.device(generator_device).type == xb.device.type
                        else torch.empty(
                            (len(xb),),
                            device=generator_device,
                            dtype=xb.dtype,
                        )
                    )
                    source_mask = self.model_.sample_source_residual_mask(
                        reference,
                        generator=source_residual_generator,
                    ).to(device=xb.device, dtype=xb.dtype)
                yield (
                    xb,
                    yb,
                    row_weights,
                    domain_batch,
                    targets,
                    len(decision_batch.decision_ids),
                    source_mask,
                )

        for epoch in range(self.cfg.epochs):
            self.model_.train()
            epoch_loss: torch.Tensor | None = None
            n_seen = 0
            epoch_trial_numerator: torch.Tensor | None = None
            epoch_trial_denominator: torch.Tensor | None = None
            epoch_candidate_sum: torch.Tensor | None = None
            epoch_candidate_prefix_sum: torch.Tensor | None = None
            epoch_decision_count = 0
            epoch_alignment_sum = 0.0
            epoch_mmd_sum = 0.0
            epoch_negative_mmd_terms = 0.0
            epoch_alignment_batches = 0
            epoch_private_sum = 0.0
            epoch_orthogonality_sum = 0.0
            epoch_source_loss_m0 = 0.0
            epoch_source_loss_m1 = 0.0
            epoch_source_rows = 0
            epoch_source_ratio_forwards: list[dict[str, object]] = []
            for (
                xb,
                yb,
                row_weights,
                domain_batch,
                decision_targets,
                batch_decision_count,
                source_residual_mask,
            ) in _training_batches(epoch):
                assert yb is not None
                try:
                    (
                        loss,
                        trial_numerator,
                        trial_denominator,
                        candidate_loss,
                        candidate_prefix_losses,
                        alignment_diagnostics,
                    ) = train_step(
                        xb,
                        yb,
                        row_weights,
                        domain_batch,
                        decision_targets,
                        source_residual_mask,
                    )
                    # CUDA-graph contract (compile mode "reduce-overhead"):
                    # outputs of a compiled region live in static buffers that
                    # the NEXT invocation overwrites. Cloning inside the
                    # compiled region only produces another static buffer —
                    # the snapshot must be taken here, in eager code, so every
                    # tensor that reaches cross-step accumulation owns its
                    # storage. clone() is value-exact on every device.
                    loss = loss.detach().clone()
                    trial_numerator = trial_numerator.detach().clone()
                    trial_denominator = trial_denominator.detach().clone()
                    candidate_loss = (
                        None if candidate_loss is None else candidate_loss.detach().clone()
                    )
                    candidate_prefix_losses = (
                        None
                        if candidate_prefix_losses is None
                        else candidate_prefix_losses.detach().clone()
                    )
                except SourceResidualContractError as error:
                    assert self.last_source_residual is not None
                    self.last_source_residual.update(
                        {
                            "status": "evaluation_failed",
                            "valid": False,
                            "failure": dict(error.record),
                        }
                    )
                    raise
                weighted_loss = loss.float() * len(xb)
                epoch_loss = weighted_loss if epoch_loss is None else epoch_loss + weighted_loss
                n_seen += len(xb)
                detached_trial_numerator = trial_numerator.float()
                detached_trial_denominator = trial_denominator.float()
                epoch_trial_numerator = (
                    detached_trial_numerator
                    if epoch_trial_numerator is None
                    else epoch_trial_numerator + detached_trial_numerator
                )
                epoch_trial_denominator = (
                    detached_trial_denominator
                    if epoch_trial_denominator is None
                    else epoch_trial_denominator + detached_trial_denominator
                )
                if decision_context_active:
                    if candidate_loss is None or candidate_prefix_losses is None:
                        raise RuntimeError("P2 training batch omitted candidate loss components.")
                    candidate_batch_sum = candidate_loss.float() * batch_decision_count
                    candidate_prefix_batch_sum = (
                        candidate_prefix_losses.float() * batch_decision_count
                    )
                    epoch_candidate_sum = (
                        candidate_batch_sum
                        if epoch_candidate_sum is None
                        else epoch_candidate_sum + candidate_batch_sum
                    )
                    epoch_candidate_prefix_sum = (
                        candidate_prefix_batch_sum
                        if epoch_candidate_prefix_sum is None
                        else epoch_candidate_prefix_sum + candidate_prefix_batch_sum
                    )
                    epoch_decision_count += batch_decision_count
                if alignment_diagnostics is not None:
                    if "mean_squared_distance" in alignment_diagnostics:
                        epoch_alignment_sum += float(alignment_diagnostics["mean_squared_distance"])
                    if "mmd_squared" in alignment_diagnostics:
                        epoch_mmd_sum += float(alignment_diagnostics["mmd_squared"])
                    negative_terms = float(alignment_diagnostics.get("negative_mmd_terms", 0.0))
                    if negative_terms > 0.0:
                        epoch_negative_mmd_terms += negative_terms
                        self.last_feature_alignment["negative_mmd_batch_count"] = (
                            int(self.last_feature_alignment.get("negative_mmd_batch_count", 0)) + 1
                        )

                    if alignment_diagnostics.get("alignment_inactive"):
                        self.last_feature_alignment["inactive_batch_count"] = (
                            int(self.last_feature_alignment.get("inactive_batch_count", 0)) + 1
                        )

                    if "private_domain_loss" in alignment_diagnostics:
                        epoch_private_sum += float(alignment_diagnostics["private_domain_loss"])
                    if "orthogonality_loss" in alignment_diagnostics:
                        epoch_orthogonality_sum += float(
                            alignment_diagnostics["orthogonality_loss"]
                        )
                    source_ledger = alignment_diagnostics.get("source_residual")
                    if source_ledger is not None:
                        if not isinstance(source_ledger, Mapping):
                            raise RuntimeError("source residual diagnostics must be a mapping.")
                        ratio_record = source_ledger.get("ratio")
                        if not isinstance(ratio_record, Mapping):
                            raise RuntimeError("source residual diagnostics lack ratio record.")
                        epoch_source_loss_m0 += float(source_ledger["loss_m0"]) * len(xb)
                        epoch_source_loss_m1 += float(source_ledger["loss_m1"]) * len(xb)
                        epoch_source_rows += len(xb)
                        epoch_source_ratio_forwards.append(dict(ratio_record))
                    epoch_alignment_batches += 1
            assert self.last_feature_alignment is not None
            self.last_feature_alignment["epoch_mean_squared_distance"].append(
                epoch_alignment_sum / epoch_alignment_batches if epoch_alignment_batches else None
            )
            self.last_feature_alignment["epoch_mmd_squared"].append(
                epoch_mmd_sum / epoch_alignment_batches
                if self._feature_alignment_mode == "conditional_rbf_mmd" and epoch_alignment_batches
                else None
            )
            self.last_feature_alignment["epoch_negative_mmd_terms"].append(
                epoch_negative_mmd_terms / epoch_alignment_batches
                if self._feature_alignment_mode == "conditional_rbf_mmd" and epoch_alignment_batches
                else None
            )
            self.last_feature_alignment["epoch_private_domain_loss"].append(
                epoch_private_sum / epoch_alignment_batches
                if private_active and epoch_alignment_batches
                else None
            )
            self.last_feature_alignment["epoch_orthogonality_loss"].append(
                epoch_orthogonality_sum / epoch_alignment_batches
                if orthogonality_active and epoch_alignment_batches
                else None
            )
            assert self.last_source_residual is not None
            if source_residual_active:
                if epoch_source_rows != n_seen or not epoch_source_ratio_forwards:
                    raise RuntimeError(
                        "source residual training failed to produce complete counterfactual ledger."
                    )
                mean_m0 = epoch_source_loss_m0 / epoch_source_rows
                mean_m1 = epoch_source_loss_m1 / epoch_source_rows
                self.last_source_residual["epoch_loss_m0"].append(mean_m0)
                self.last_source_residual["epoch_loss_m1"].append(mean_m1)
                self.last_source_residual["epoch_loss_delta_m1_minus_m0"].append(mean_m1 - mean_m0)
                self.last_source_residual["epoch_trial_loss_m0"].append(mean_m0)
                self.last_source_residual["epoch_trial_loss_m1"].append(mean_m1)
                self.last_source_residual["epoch_trial_loss_delta_m1_minus_m0"].append(
                    mean_m1 - mean_m0
                )
                self.last_source_residual["epoch_ratio_forwards"].append(
                    epoch_source_ratio_forwards
                )
                assert isinstance(self.model_, FeatureDomainTransfer)
                self.last_source_residual["epoch_domain_scales"].append(
                    self.model_.source_residual_scale_record()
                )
                epoch_max = max(
                    float(record["max"])
                    for record in epoch_source_ratio_forwards
                    if record["max"] is not None
                )
                self.last_source_residual["run_max_ratio"] = max(
                    float(self.last_source_residual["run_max_ratio"]),
                    epoch_max,
                )
            if epoch_loss is None:
                raise RuntimeError("Deep training produced no optimizer batches.")
            if epoch_trial_numerator is None or epoch_trial_denominator is None:
                raise RuntimeError("Deep training omitted trial-loss sufficient statistics.")
            if decision_context_active:
                assert decision_loss_config is not None
                assert train_decision_plan is not None
                if (
                    epoch_candidate_sum is None
                    or epoch_candidate_prefix_sum is None
                    or epoch_decision_count != len(train_decision_plan.decision_ids)
                ):
                    raise RuntimeError("P2 training did not cover every planned decision.")
                epoch_trial_mean = epoch_trial_numerator / epoch_trial_denominator
                epoch_candidate_mean = epoch_candidate_sum / epoch_decision_count
                epoch_objective = (
                    epoch_trial_mean + decision_loss_config.lambda_candidate * epoch_candidate_mean
                )
                train_losses.append(float(epoch_objective.cpu()))
                assert self.last_decision_loss is not None
                train_component = {
                    "epoch": epoch + 1,
                    "trial_component_loss": float(epoch_trial_mean.cpu()),
                    "candidate_component_loss": float(epoch_candidate_mean.cpu()),
                    "candidate_component_by_prefix": [
                        float(value)
                        for value in (epoch_candidate_prefix_sum / epoch_decision_count)
                        .cpu()
                        .tolist()
                    ],
                    "lambda_candidate": decision_loss_config.lambda_candidate,
                    "full_objective_loss": float(epoch_objective.cpu()),
                    "requested_decisions": epoch_decision_count,
                    "trial_denominator": float(epoch_trial_denominator.cpu()),
                }
                self.last_decision_loss["epoch_train_components"].append(train_component)
            else:
                train_component = None
                train_losses.append(float(epoch_loss.cpu()) / max(n_seen, 1))
            mean_val = None
            objective_val = None
            task_val_auc = None
            will_early_stop = False

            if val_source is not None:
                if decision_context_active:
                    assert decision_module is not None
                    assert decision_loss_config is not None
                    assert val_decision_plan is not None
                    validation_result = self._decision_validation_pass(
                        val_source,
                        loss_fn,
                        decision_module,
                        val_decision_plan,
                        decision_loss_config,
                    )
                    mean_val = float(validation_result["trial_loss"])
                    objective_val = float(validation_result["objective_loss"])
                    val_scores = np.asarray(validation_result["scores"])
                    val_labels = np.asarray(validation_result["labels"])
                    assert self.last_decision_loss is not None
                    validation_component = {
                        "epoch": epoch + 1,
                        "trial_component_loss": mean_val,
                        "candidate_component_loss": float(validation_result["candidate_loss"]),
                        "candidate_component_by_prefix": list(
                            validation_result["candidate_loss_by_prefix"]
                        ),
                        "lambda_candidate": decision_loss_config.lambda_candidate,
                        "full_objective_loss": objective_val,
                        "requested_decisions": int(validation_result["requested_decisions"]),
                    }
                    self.last_decision_loss["epoch_validation_components"].append(
                        validation_component
                    )
                else:
                    (
                        (mean_val, val_scores, val_labels),
                        validation_batch_size,
                    ) = self._validation_pass_with_oom_retry(
                        val_source,
                        loss_fn,
                        validation_batch_size,
                    )
                    self._validation_batch_size = validation_batch_size
                    objective_val = mean_val
                assert mean_val is not None
                assert objective_val is not None
                val_losses.append(mean_val)
                val_objective_losses.append(objective_val)
                if len(np.unique(val_labels)) == 2:
                    task_val_auc = float(roc_auc_score(val_labels, val_scores))
                task_val_aucs.append(task_val_auc)
                if mean_val < best_task_val_loss - self.cfg.early_stop_min_delta:
                    best_task_val_loss = mean_val
                    best_task_epoch = epoch
                if objective_val < best_val_loss - self.cfg.early_stop_min_delta:
                    best_val_loss = objective_val
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
                        "train_loss_components": (
                            {} if train_component is None else dict(train_component)
                        ),
                        "task_val_loss": mean_val,
                        "task_val_auc": task_val_auc,
                        "objective_val_loss": objective_val,
                        "phase": "joint",
                        "optimizer_steps": None,
                        "selection_active": mean_val is not None,
                        "patience_left": patience_left if mean_val is not None else None,
                        "early_stop_patience": self.cfg.early_stop_patience,
                        "best_epoch": best_epoch + 1 if best_epoch is not None else None,
                        "best_task_epoch": (
                            best_task_epoch + 1 if best_task_epoch is not None else None
                        ),
                        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                        "best_task_val_loss": (
                            best_task_val_loss if best_task_val_loss != float("inf") else None
                        ),
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
            if decision_context_active:
                assert decision_module is not None
                assert decision_loss_config is not None
                assert val_decision_plan is not None
                final_validation = self._decision_validation_pass(
                    val_source,
                    loss_fn,
                    decision_module,
                    val_decision_plan,
                    decision_loss_config,
                )
                final_scores = np.asarray(final_validation["scores"])
                final_labels = np.asarray(final_validation["labels"])
            else:
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
        assert self.last_source_residual is not None
        if source_residual_active:
            assert isinstance(self.model_, FeatureDomainTransfer)
            self.last_source_residual.update(
                {
                    "status": "valid",
                    "valid": True,
                    "final_domain_scales": self.model_.source_residual_scale_record(),
                }
            )
        if decision_context_active:
            assert self.last_decision_loss is not None
            self.last_decision_loss.update(
                {
                    "status": "valid",
                    "valid": True,
                    "fit_completed": True,
                    "selected_epoch": best_epoch,
                    "selected_objective": "trial_plus_lambda_candidate",
                    "oom_batch_repacking": "forbidden",
                }
            )
        self.last_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_objective_losses": val_objective_losses,
            "task_val_aucs": task_val_aucs,
            "final_task_val_auc": final_task_val_auc,
            "phases": ["joint"] * len(train_losses),
            "best_epoch": best_epoch,
            "best_task_epoch": best_task_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "best_task_val_loss": (
                best_task_val_loss if best_task_val_loss != float("inf") else None
            ),
            "task_patience_exhausted": early_stop_triggered,
            "feature_alignment": dict(self.last_feature_alignment or {}),
            "source_residual": dict(self.last_source_residual),
            # ``last_decision_loss`` is None exactly when the P2 decision loss is
            # disabled; serializing it as an empty dict would fabricate a P2
            # ledger and make the checkpoint unreadable by the candidate runner.
            "decision_loss": (
                None if self.last_decision_loss is None else dict(self.last_decision_loss)
            ),
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
                        splitbn_config = getattr(self, "splitbn_config", None)
                        if splitbn_config is None:
                            logits = self.model_(xb)
                        else:
                            logits = self.model_(
                                xb,
                                eval_buffer_source=splitbn_config.target_buffer_source,
                            )
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
