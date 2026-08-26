"""模块 #14：训练循环（Trainer）。

职责（CODING_WORKFLOW #14）：
    单受试（Phase 2）训练循环：epoch 迭代 + 数据增强 + AMP 混合精度 + 梯度累积 + early stop +
    多任务总损失（复用 train/losses.compute_losses）。**须遵守 device-portability.md 全部 DP 规则**。

明确「不做」：
    - 不构建 DataLoader（DataLoader 由调用方构造，本模块只消费 batch 流）。
    - λ2 网格搜索（λ2 是配置单值，网格由外部脚本跑多次）；本模块只训练单个 λ2 配置。
    - 跨域 MMD（Phase 3）：compute_losses 的 λ4 默认 0。

三思决策记录（供后续会话追溯）：
    D-device-detect  get_device() 按 CUDA→XPU→CPU 检测（DP2），禁止硬编码 .cuda()（DP1）；原生
                     torch.xpu（PyTorch≥2.5），不 import 已 EOL 的 IPEX（device-portability C1）。
    D-amp-bf16       AMP 用 bf16（DP4/C2）：CUDA/XPU 启用、CPU 禁用；bf16 指数位同 fp32，无需 GradScaler。
                     仅当 Phase 2 实测 bf16 精度不足（罕见）才回退 fp16+GradScaler（§8 补充注意 3）。
    D-accum          梯度累积（DP5）：loss/accum_steps 后 backward，每 accum_steps 步 optimizer.step，
                     物理 batch 过小时模拟大 batch。
      D-jit-default    2026-08-22 失败诊断方案 B：lambda_jit/jit_prob 默认 0（关闭自监督 jitter 一致性）。
                       实测 L_jit 不收敛（τ 跟踪 ±40ms 平移 RMS≈52ms），且向共享 encoder 注入与 P300
                       时间局域判别冲突的"时间不变性"梯度；接口保留，显式设置可复现旧实验。
    D-oom            OOM 捕获按 torch.OutOfMemoryError + RuntimeError 兜底（DP6），提示减 batch_size/
                     调 accum_steps（§6）。
    D-augment        数据增强仅训练期（train_step 内），验证/评估不增强。
    D-early-stop     early stop 用 val loss（若提供 val_loader），patience 个 epoch 不改善即停，存最佳权重。
    D-perf-sync      prevalidated loader 在构造时完成 finite 校验；Trainer 关闭逐 forward finite 检查，
                     训练 loss 只 epoch 末同步，AUC 只在训练后一次性计算（doc/performance_lessons.md）。
    D-echn-esub      E_chn/E_sub 作为 Trainer 固定属性（单被试 Phase 2 时二者固定）；跨被试 Phase 3
                     再扩展为 batch 级。None 时 forward 不融合。
    D-device-param   Trainer 接受可选 device（默认 get_device()），便于测试显式用 CPU 保证稳定。
    D-windows-numworkers  DataLoader 多进程在 Windows 用 spawn，须 `if __name__ == "__main__":` 包裹
                     入口（§8 补充注意 1）；本模块不建 loader，但提醒调用方。

契约（输入 → 输出）：
    model(N2P3Net) + TrainerConfig + train_loader(→(X,y)) [+ val_loader] → fit() 返回训练历史
    dict{train_losses, val_losses}；模型权重就地更新。

依赖的决策：device-portability.md（DP1–DP6）、blueprint §7/§8、train/losses.py、train/augment.py。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.channel import canonical_channel_name
from models.n2p3net import N2P3Net
from models.repetition import extract_quality_features
from models.repetition_v12 import AdditiveRepetitionEvidence
from train.augment import apply_augmentations, known_time_shift
from train.contracts import GenerativeProfile, ReconstructionProfile, TrialContext
from train.device import (
    empty_cache,
    get_device,
    optimize_device_for_training,
    print_device_memory,
)
from train.losses import (
    compute_losses,
    estimate_reconstruction_profile,
    gtn_multi_k_cross_entropy,
    repetition_multi_k_objective,
)
from train.prequential import estimate_generative_profile
from train.repetition_v12_objective import additive_repetition_multi_k_objective

COMPILE_MODES = ("eager", "default", "reduce-overhead", "max-autotune")
LR_SCHEDULES = ("constant", "cosine")


@dataclass
class TrainerConfig:
    """训练配置（DP5：batch_size 等经配置传入，不写死）。"""

    epochs: int = 50
    batch_size: int = 256
    accum_steps: int = 1
    compile_mode: str = "eager"
    lr: float = 1e-3
    lr_schedule: str = "cosine"
    lr_warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.10
    erp_decoder_lr_multiplier: float = 5.0
    weight_decay: float = 2.5e-5
    lambda2: float = 0.3
    lambda3: float = 0.0
    lambda_pcw: float = 0.0
    lambda_digit: float = 0.0
    lambda_conditional_nll: float = 0.0
    repetition_reliability_aux_weight: float = 1.0
    repetition_reliability_lr_multiplier: float = 10.0
    repetition_refit_epochs: int = 5
    repetition_v12: bool = False
    repetition_state_residual_l2_weight: float = 0.0
    repetition_v12_evidence_ks: tuple[int, ...] = (1, 3, 5)
    repetition_v12_evidence_weights: tuple[float, ...] = (0.34, 0.33, 0.33)
    auto_pos_weight: bool = True
    digit_evidence_ks: tuple[int, ...] = (1, 3, 5, 10, 15)
    digit_evidence_weights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.45)
    lambda_amp: float = 0.0
    amplitude_channel: str | None = "Pz"
    lambda_recon: float = 0.0
    recon_bands_hz: tuple[tuple[float, float], ...] = (
        (0.5, 2.0),
        (2.0, 4.0),
        (4.0, 8.0),
        (8.0, 15.0),
        (15.0, 30.0),
    )
    recon_prior_weights: tuple[float, ...] = (0.40, 0.35, 0.15, 0.07, 0.03)
    recon_profile_max_trials: int = 4096
    recon_bootstrap_samples: int = 64
    recon_split_half_repeats: int = 16
    recon_waveform_weight: float = 1.0
    recon_projection_weight: float = 1.0
    recon_nll_weight: float = 0.1
    lambda_innovation: float = 0.0
    innovation_score_interval_ms: tuple[float, float] = (0.0, 800.0)
    innovation_ar_order: int = 32
    lambda_morphology_l0: float = 0.0
    variance_warmup_epochs: int = 0
    variance_ramp_epochs: int = 0
    recon_erp_interval_ms: tuple[float, float] = (150.0, 900.0)
    lambda_jit: float = 0.0
    jit_max_ms: float = 40.0
    jit_prob: float = 0.0
    lambda4: float = 0.0
    mmd_bandwidth: float | None = None
    lambda_orth: float = 0.0
    lambda_adv: float = 0.0
    lambda_private: float = 0.0
    reconstruct_all_domains: bool = False
    main_domain: int = 0
    aux_domain: int = 1
    pos_weight: float = 8.0
    tau_scale_ms: float = 50.0
    early_stop_patience: int = 10
    augment: bool = True
    seed: int = 0
    track_pcw_gradients: bool = False
    epoch_trajectory_audit: bool = False
    recalibrate_batch_norm: bool = False


class Trainer:
    """训练循环（单受试 Phase 2）。

    Parameters
    ----------
    model : N2P3Net
        已构造的模型（将在 __init__ 内 .to(device)）。
    config : TrainerConfig
        训练配置。
    E_chn / E_sub : torch.Tensor | None
        固定通道坐标嵌入 / subject 元数据嵌入（单被试时固定；None 不融合）。
    channel_mask : torch.Tensor | None
        (C,) bool 缺失通道掩码（True=存在）。训练/验证 forward 与数据增强都会使用，
        缺失通道保持 0（review v6 P0-2）。
    device : torch.device | None
        默认 get_device()；测试可显式传 CPU。
    """

    def __init__(
        self,
        model: N2P3Net,
        config: TrainerConfig,
        E_chn: torch.Tensor | None = None,
        E_sub: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        fold_id: int | None = None,
    ):
        self.cfg = config
        self.fold_id = None if fold_id is None else int(fold_id)
        for name in ("epochs", "batch_size", "accum_steps", "early_stop_patience"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        positive_scalars = (
            "lr",
            "erp_decoder_lr_multiplier",
            "pos_weight",
            "tau_scale_ms",
            "repetition_reliability_lr_multiplier",
        )
        for name in positive_scalars:
            value = float(getattr(config, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative_scalars = (
            "weight_decay",
            "lambda2",
            "lambda3",
            "lambda_pcw",
            "lambda_digit",
            "lambda_conditional_nll",
            "repetition_reliability_aux_weight",
            "repetition_state_residual_l2_weight",
            "lambda_amp",
            "lambda_recon",
            "recon_waveform_weight",
            "recon_projection_weight",
            "recon_nll_weight",
            "lambda_innovation",
            "lambda_morphology_l0",
            "lambda_jit",
            "jit_max_ms",
            "lambda4",
            "lambda_orth",
            "lambda_adv",
            "lambda_private",
        )
        for name in nonnegative_scalars:
            value = float(getattr(config, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if config.mmd_bandwidth is not None and (
            not math.isfinite(float(config.mmd_bandwidth)) or config.mmd_bandwidth <= 0.0
        ):
            raise ValueError("mmd_bandwidth must be finite and positive or None.")
        if config.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}.")
        if config.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}.")
        if not 0.0 <= config.lr_warmup_fraction < 1.0:
            raise ValueError("lr_warmup_fraction must be in [0, 1).")
        if not 0.0 <= config.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1].")
        if config.erp_decoder_lr_multiplier <= 0.0:
            raise ValueError("erp_decoder_lr_multiplier must be positive.")
        if not 0.0 <= config.jit_prob <= 1.0:
            raise ValueError(f"jit_prob 须在 [0,1]，得到 {config.jit_prob}。")
        if (
            not config.digit_evidence_ks
            or tuple(sorted(set(config.digit_evidence_ks))) != config.digit_evidence_ks
            or config.digit_evidence_ks[0] < 1
        ):
            raise ValueError("digit_evidence_ks must be unique positive integers in order.")
        if (
            len(config.digit_evidence_ks) != len(config.digit_evidence_weights)
            or any(weight < 0.0 for weight in config.digit_evidence_weights)
            or sum(config.digit_evidence_weights) <= 0.0
        ):
            raise ValueError("digit_evidence_weights must match Ks and have positive mass.")
        if config.variance_warmup_epochs < 0 or config.variance_ramp_epochs < 0:
            raise ValueError("Variance warmup/ramp epochs cannot be negative.")
        if config.lambda_morphology_l0 < 0.0:
            raise ValueError("lambda_morphology_l0 must be non-negative.")
        if config.lambda_innovation < 0.0:
            raise ValueError("lambda_innovation must be non-negative.")
        if config.lambda_innovation > 0.0 and model.innovation_decoder is None:
            raise ValueError(
                "lambda_innovation>0 requires N2P3Net(use_innovation_likelihood=True)."
            )
        if config.lambda_recon > 0.0 and model.component_decoder is None:
            raise ValueError("lambda_recon>0 requires component_decoder=True.")
        if config.lambda_morphology_l0 > 0.0 and model.component_decoder is None:
            raise ValueError("lambda_morphology_l0>0 requires component_decoder=True.")
        if config.innovation_ar_order < 1:
            raise ValueError("innovation_ar_order must be positive.")
        if min(config.lambda_orth, config.lambda_adv, config.lambda_private) < 0.0:
            raise ValueError("Shared/private transfer loss weights must be non-negative.")
        if (
            config.lambda_conditional_nll < 0.0
            or config.repetition_refit_epochs < 0
            or config.repetition_reliability_aux_weight < 0.0
            or config.repetition_reliability_lr_multiplier <= 0.0
        ):
            raise ValueError("Conditional NLL weight/refit epochs cannot be negative.")
        if config.lambda_conditional_nll > 0.0 and model.repetition_evidence is None:
            raise ValueError(
                "lambda_conditional_nll>0 requires N2P3Net(use_repetition_evidence=True)."
            )
        if config.recon_profile_max_trials < 2:
            raise ValueError("Reconstruction profile trial limit must be at least two.")
        if config.recon_bootstrap_samples < 2 or config.recon_split_half_repeats < 0:
            raise ValueError("Invalid reconstruction bootstrap/split-half replicate counts.")
        if len(config.recon_bands_hz) != len(config.recon_prior_weights):
            raise ValueError("recon_bands_hz and recon_prior_weights lengths must match.")
        self.device = device if device is not None else get_device()
        self.use_amp = self.device.type in ("cuda", "xpu")  # DP4

        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        optimize_device_for_training(self.device)  # D-device-tune
        self.model = model.to(self.device)  # DP3
        if config.compile_mode != "eager":
            if not hasattr(self.model, "compile"):
                raise RuntimeError("The installed PyTorch does not provide nn.Module.compile().")
            compile_kwargs = (
                {} if config.compile_mode == "default" else {"mode": config.compile_mode}
            )
            self.model.compile(**compile_kwargs)
        self.active_domain_indices: tuple[int, ...] | None = None
        self.E_chn = E_chn.to(self.device) if E_chn is not None else None
        self.E_sub = E_sub.to(self.device) if E_sub is not None else None
        if channel_mask is not None and channel_mask.dtype != torch.bool:
            raise ValueError("Trainer channel_mask must have boolean dtype.")
        self.channel_mask = channel_mask.to(self.device) if channel_mask is not None else None
        if self.channel_mask is not None:
            if self.channel_mask.shape != (self.model.n_channels,):
                raise ValueError("Trainer channel_mask must be (C,).")
            if not bool(self.channel_mask.any()):
                raise ValueError("Trainer channel_mask must retain an observed channel.")
        channel_names = tuple(canonical_channel_name(name) for name in self.model.channel_names)
        amplitude_channel = (
            canonical_channel_name(config.amplitude_channel)
            if config.amplitude_channel is not None
            else None
        )
        if amplitude_channel is None or amplitude_channel not in channel_names:
            if config.lambda_amp > 0.0:
                raise ValueError(
                    "lambda_amp>0 requires amplitude_channel to identify a real model channel; "
                    f"requested {config.amplitude_channel!r}, available={channel_names}."
                )
            self.pz_channel = max(0, self.model.n_channels - 1)
        else:
            self.pz_channel = channel_names.index(amplitude_channel)

        # review v6 P1：τ0 是生理先验中心，不参与 AdamW weight decay（否则会被缓慢拉向 0）。
        decoder_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "component_decoder" in n
        ]
        decay_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and "tau0" not in n and "component_decoder" not in n
        ]
        no_decay_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "tau0" in n
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
                {
                    "params": decoder_params,
                    "weight_decay": config.weight_decay,
                    "lr": config.lr * config.erp_decoder_lr_multiplier,
                },
            ],
            lr=config.lr,
            fused=self.device.type == "cuda",
        )
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.optimizer_steps = 0
        self.planned_optimizer_steps = 0
        self._lr_history: list[float] = []
        # 确保初始 τ0 也在生理界内
        self.model.component_window.clamp_tau0_()
        self.best_val_loss = float("inf")
        self.best_state = None
        self.best_epoch: int | None = None
        # Strict-past research has two parameter-disjoint validation endpoints.
        # Keep the legacy names above as aliases for the task-selected checkpoint.
        self.best_task_val_loss = float("inf")
        self.best_task_state: dict[str, torch.Tensor] | None = None
        self.best_task_epoch: int | None = None
        self.best_density_nll = float("inf")
        self.best_density_state: dict[str, torch.Tensor] | None = None
        self.best_density_epoch: int | None = None
        # P9 梯度累积：当前 accum 周期内是否已有任何有效 backward。
        self._accum_has_grad = False
        self._pcw_tau_grad_norms: list[float] = []
        self._pcw_head_grad_norms: list[float] = []
        self._pcw_classifier_grad_norms: list[float] = []
        self._pcw_path_grad_norms: list[float] = []
        self._innovation_path_grad_norms: list[float] = []
        self._pcw_tau0_initial = self.model.component_window.tau0_bounded.detach().cpu().clone()
        self.reconstruction_profile: ReconstructionProfile | None = None
        self.generative_profile: GenerativeProfile | None = None
        self._variance_nll_weight_history: list[float] = []
        self._innovation_covariance_weight_history: list[float] = []
        self._active_recon_nll_weight = 0.0
        self._active_innovation_covariance_weight = 0.0
        self._loss_component_names = (
            "target",
            "pcw",
            "digit",
            "conditional_nll",
            "early",
            "tau",
            "amp",
            "recon",
            "recon_erp",
            "recon_erp_waveform",
            "recon_erp_projection",
            "recon_nll",
            "innovation_nll",
            "jit",
            "mmd",
            "morphology_l0",
            "orth",
            "domain",
            "private",
        )
        self._last_loss_vector = torch.zeros(
            len(self._loss_component_names), device=self.device, dtype=torch.float32
        )

        print_device_memory(self.device)

    def _autocast_ctx(self):
        return torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        )

    def _configure_lr_scheduler(self, total_steps: int) -> None:
        """Create a sample-budget scheduler before the first optimizer update."""

        total_steps = int(total_steps)
        if total_steps < 1:
            raise ValueError("Training requires at least one planned optimizer step.")
        self.planned_optimizer_steps = total_steps
        if self.cfg.lr_schedule == "constant":
            self.lr_scheduler = None
            return

        warmup_steps = int(math.ceil(total_steps * self.cfg.lr_warmup_fraction))
        decay_steps = max(total_steps - warmup_steps, 1)

        def lr_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (
                1.0
                if decay_steps == 1
                else min(1.0, max(0.0, (step - warmup_steps) / (decay_steps - 1)))
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

    def _optimizer_step(self) -> None:
        """Apply one audited main-model update and advance the step scheduler."""

        self._lr_history.append(float(self.optimizer.param_groups[0]["lr"]))
        self.optimizer.step()
        self.model.component_window.clamp_tau0_()
        self.optimizer_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    @staticmethod
    def _interleaved_update_sources(n_trial: int, n_set: int) -> Iterator[str]:
        """Distribute complete set batches across the trial stream deterministically."""

        if n_trial < 0 or n_set < 0 or n_trial + n_set == 0:
            raise ValueError("Interleaving requires at least one non-negative batch count.")
        set_emitted = 0
        total = n_trial + n_set
        for slot in range(total):
            target_set_count = ((slot + 1) * n_set) // total
            if target_set_count > set_emitted:
                set_emitted += 1
                yield "set"
            else:
                yield "trial"

    def _planned_steps_per_epoch(self, n_trial: int, n_set: int) -> int:
        """Upper-bound actual optimizer steps using the exact interleave schedule."""

        pending_trial = 0
        trial_index = 0
        steps = 0
        for source in self._interleaved_update_sources(n_trial, n_set):
            if source == "trial":
                pending_trial += 1
                trial_index += 1
                if trial_index % self.cfg.accum_steps == 0:
                    steps += 1
                    pending_trial = 0
            else:
                if pending_trial:
                    steps += 1
                    pending_trial = 0
                steps += 1
        return steps + int(pending_trial > 0)

    @torch.inference_mode()
    def _recalibrate_batch_norm(self, train_loader) -> int:
        """Re-estimate fold-local BN buffers without dropout or augmentation."""

        if not self.cfg.recalibrate_batch_norm:
            return 0
        batch_norms = [
            module
            for module in self.model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ]
        if not batch_norms:
            return 0

        original_momenta = {module: module.momentum for module in batch_norms}
        self.model.eval()
        for module in batch_norms:
            module.reset_running_stats()
            module.momentum = None
            module.train()

        n_batches = 0
        try:
            for batch in train_loader:
                context = self._unpack_batch(batch)
                non_blocking = self.device.type == "cuda" and context.X.is_pinned()
                context = context.to(self.device, non_blocking=non_blocking)
                effective_channel_mask = self._effective_channel_mask(
                    context.channel_mask,
                    batch_size=context.X.shape[0],
                    channels=context.X.shape[1],
                )
                with self._autocast_ctx():
                    self.model(
                        context.X,
                        self.E_chn,
                        self.E_sub,
                        channel_mask=effective_channel_mask,
                        domain_id=context.domain_id,
                        return_attention=False,
                        return_likelihood=False,
                    )
                n_batches += 1
        finally:
            for module, momentum in original_momenta.items():
                module.momentum = momentum
            self.model.eval()
        return n_batches

    def _phase_for_epoch(self, epoch: int) -> str:
        if not self._variance_schedule_active():
            return "joint"
        if epoch < self.cfg.variance_warmup_epochs:
            return "mean_warmup"
        if self._variance_progress_for_epoch(epoch) < 1.0:
            return "variance_ramp"
        return "joint"

    def _required_epoch_count(self) -> int:
        """Return the fixed training budget; audit data never changes epochs."""

        return int(self.cfg.epochs)

    def _first_joint_epoch(self) -> int:
        """Return the first zero-based epoch with full covariance weight."""

        if not self._variance_schedule_active():
            return 0
        return self.cfg.variance_warmup_epochs + max(self.cfg.variance_ramp_epochs - 1, 0)

    def _variance_schedule_active(self) -> bool:
        return self.cfg.lambda_recon > 0.0 or self.cfg.lambda_innovation > 0.0

    def _variance_progress_for_epoch(self, epoch: int) -> float:
        """Shared warmup/ramp for ERP and innovation covariance models."""

        if not self._variance_schedule_active():
            return 0.0
        if epoch < self.cfg.variance_warmup_epochs:
            return 0.0
        if self.cfg.variance_ramp_epochs <= 0:
            return 1.0
        ramp_step = epoch - self.cfg.variance_warmup_epochs + 1
        return min(1.0, max(0.0, ramp_step / self.cfg.variance_ramp_epochs))

    def _variance_nll_weight_for_epoch(self, epoch: int) -> float:
        """ERP variance-score weight using the shared variance schedule."""

        return float(self.cfg.recon_nll_weight) * self._variance_progress_for_epoch(epoch)

    @torch.no_grad()
    def _prepare_reconstruction_profile(self, reconstruction_context: TrialContext | None) -> None:
        if self.cfg.lambda_recon <= 0.0 and self.cfg.lambda_innovation <= 0.0:
            return
        if reconstruction_context is None:
            raise ValueError(
                "ERP or innovation training requires an explicit optimization-training "
                "TrialContext for fold-local profile estimation."
            )
        reconstruction_context.validate()
        # Normalize the statistics source to host memory before selecting the
        # profile sample.  This also covers callers that pass a GPU full_context.
        X = reconstruction_context.X.detach().cpu()
        y = reconstruction_context.y.detach().cpu()
        domain_id = (
            reconstruction_context.domain_id.detach().cpu()
            if reconstruction_context.domain_id is not None
            else None
        )
        context_channel_mask = (
            reconstruction_context.channel_mask.detach().cpu()
            if reconstruction_context.channel_mask is not None
            else None
        )
        valid = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        if domain_id is not None:
            valid &= domain_id.to(X.device) == self.cfg.main_domain
        effective_profile_mask = self._effective_channel_mask(
            context_channel_mask,
            batch_size=X.shape[0],
            channels=X.shape[1],
        )
        if effective_profile_mask is None:
            profile_mask = torch.ones(X.shape[1], device=self.device, dtype=torch.bool)
        elif effective_profile_mask.dim() == 1:
            profile_mask = effective_profile_mask
        else:
            profile_mask = effective_profile_mask[
                valid.to(device=effective_profile_mask.device)
            ].all(dim=0)
        if not bool(profile_mask.any()):
            raise ValueError(
                "Reconstruction profiling requires at least one channel observed in every "
                "eligible fold-training trial."
            )
        all_labels = y.reshape(-1).to(X.device)
        positive_indices = torch.nonzero(valid & (all_labels > 0.5), as_tuple=False).flatten()
        negative_indices = torch.nonzero(valid & (all_labels <= 0.5), as_tuple=False).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            raise ValueError("Reconstruction profiling requires both target classes.")
        max_per_class = max(1, int(self.cfg.recon_profile_max_trials) // 2)
        per_class = min(
            max_per_class,
            int(positive_indices.numel()),
            int(negative_indices.numel()),
        )
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        positive_order = torch.randperm(positive_indices.numel(), generator=gen, device="cpu")[
            :per_class
        ].to(X.device)
        negative_order = torch.randperm(negative_indices.numel(), generator=gen, device="cpu")[
            :per_class
        ].to(X.device)
        indices = torch.cat((positive_indices[positive_order], negative_indices[negative_order]))
        # Profile fitting is a no-grad fold-local statistics pass.  Keep its
        # large lagged/FFT work on the host: the innovation estimator creates
        # O(N * T * ar_order * C) temporary matrices, which otherwise causes a
        # multi-GB CUDA spike before the first training epoch.
        sample_context = TrialContext(
            X=X[indices],
            y=y[indices],
            domain_id=(
                domain_id[indices] if domain_id is not None else None
            ),
            channel_mask=(
                context_channel_mask[indices]
                if context_channel_mask is not None and context_channel_mask.dim() == 2
                else context_channel_mask
            ),
        )
        X_sample = sample_context.X
        y_sample = sample_context.y
        sample_channel_mask = self._effective_channel_mask(
            sample_context.channel_mask,
            batch_size=X_sample.shape[0],
            channels=X_sample.shape[1],
        )
        if sample_channel_mask is not None:
            sample_channel_mask = sample_channel_mask.to(
                device=X_sample.device, dtype=torch.bool
            )
        profile_mask = profile_mask.to(device=X_sample.device, dtype=torch.bool)
        if self.cfg.lambda_innovation > 0.0:
            fixed_observation = self.model.preprocess_likelihood_observation(
                X_sample,
                channel_mask=sample_channel_mask,
            )
            self.generative_profile = estimate_generative_profile(
                fixed_observation.cpu(),
                y_sample.cpu(),
                sfreq=float(self.model.sfreq),
                tmin_ms=float(self.model.tmin_ms),
                channel_mask=profile_mask.cpu(),
                score_interval_ms=self.cfg.innovation_score_interval_ms,
                ar_order=self.cfg.innovation_ar_order,
                covariance_rank=self.model.innovation_decoder.covariance_rank,
            ).to(self.device)
            self.generative_profile.validate(n_channels=X_sample.shape[1])
        if self.cfg.lambda_recon <= 0.0:
            return
        X0 = self.model.preprocess_likelihood_observation(
            X_sample,
            channel_mask=sample_channel_mask,
        )
        self.reconstruction_profile = estimate_reconstruction_profile(
            X0.cpu(),
            y_sample.cpu(),
            sfreq=float(self.model.sfreq),
            tmin_ms=float(self.model.tmin_ms),
            bands_hz=self.cfg.recon_bands_hz,
            prior_weights=self.cfg.recon_prior_weights,
            channel_mask=profile_mask.cpu(),
            erp_interval_ms=self.cfg.recon_erp_interval_ms,
            bootstrap_samples=self.cfg.recon_bootstrap_samples,
            split_half_repeats=self.cfg.recon_split_half_repeats,
            bootstrap_seed=self.cfg.seed,
        ).to(self.device)

    @torch.no_grad()
    def _prepare_repetition_calibration(self, reconstruction_context: TrialContext | None) -> None:
        evidence_model = self.model.repetition_evidence
        if self.cfg.lambda_conditional_nll <= 0.0:
            return
        if reconstruction_context is None:
            raise ValueError(
                "Conditional repetition modeling requires fold-training labels for "
                "the weighted-logit prior correction."
            )
        reconstruction_context.validate()
        labels = reconstruction_context.y.reshape(-1)
        valid = torch.ones(labels.numel(), dtype=torch.bool, device=labels.device)
        if reconstruction_context.domain_id is not None:
            valid &= reconstruction_context.domain_id.reshape(-1).to(labels.device) == (
                self.cfg.main_domain
            )
        labels = labels[valid]
        positives = int((labels > 0.5).sum())
        negatives = int((labels <= 0.5).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("Conditional evidence calibration requires both target classes.")
        train_prior = positives / (positives + negatives)
        evidence_model.set_evidence_calibration(
            pos_weight=self.cfg.pos_weight,
            train_prior=train_prior,
            temperature=1.0,
        )

    @torch.inference_mode()
    def _prepare_repetition_quality_normalizer(self, context: TrialContext | None) -> None:
        evidence_model = self.model.repetition_evidence
        if self.cfg.lambda_conditional_nll <= 0.0:
            return
        if evidence_model is None or context is None:
            raise ValueError("Conditional evidence requires optimization quality data.")
        context.validate()
        was_training = self.model.training
        self.model.eval()
        quality_batches: list[torch.Tensor] = []
        for start in range(0, context.X.shape[0], 256):
            stop = start + 256
            X = context.X[start:stop].to(self.device)
            domain_id = (
                context.domain_id[start:stop].to(self.device)
                if context.domain_id is not None
                else None
            )
            context_mask = (
                context.channel_mask[start:stop]
                if context.channel_mask is not None and context.channel_mask.dim() == 2
                else context.channel_mask
            )
            effective_channel_mask = self._effective_channel_mask(
                context_mask,
                batch_size=X.shape[0],
                channels=X.shape[1],
            )
            output = self.model(
                X,
                self.E_chn,
                self.E_sub,
                channel_mask=effective_channel_mask,
                domain_id=domain_id,
                return_attention=False,
                return_likelihood=False,
            )
            quality_batches.append(
                self._quality_features(X, output, domain_id, effective_channel_mask).float()
            )
        evidence_model.fit_quality_normalizer(torch.cat(quality_batches, dim=0))
        self.model.train(was_training)

    def _quality_features(
        self,
        X: torch.Tensor,
        output,
        domain_id: torch.Tensor | None,
        channel_mask: torch.Tensor | None = None,
        check_finite: bool = True,
    ) -> torch.Tensor:
        effective_mask = self.channel_mask if channel_mask is None else channel_mask
        return extract_quality_features(
            X.detach(),
            sfreq=float(self.model.sfreq),
            baseline_n=int(self.model.baseline_n),
            reference_slice=getattr(self.model, "trial_reference_slice", None),
            channel_mask=effective_mask,
            check_finite=check_finite,
        )

    @staticmethod
    def _unpack_batch(batch) -> TrialContext:
        if isinstance(batch, TrialContext):
            batch.validate()
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) in (2, 3):
            context = TrialContext(
                X=batch[0], y=batch[1], domain_id=batch[2] if len(batch) == 3 else None
            )
            context.validate()
            return context
        raise TypeError(
            "Training batches must be TrialContext or standard (X,y[,domain_id]) tuples."
        )

    def _effective_channel_mask(
        self,
        context_mask: torch.Tensor | None,
        *,
        batch_size: int,
        channels: int,
    ) -> torch.Tensor | None:
        """Intersect montage availability with per-trial observations."""

        static_mask = self.channel_mask
        if static_mask is not None and static_mask.shape != (channels,):
            raise ValueError("Trainer channel_mask does not match the current montage.")
        if context_mask is None:
            return static_mask

        dynamic_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if dynamic_mask.shape not in {(channels,), (batch_size, channels)}:
            raise ValueError("TrialContext channel_mask must be (C,) or (B,C).")
        if static_mask is not None:
            dynamic_mask = dynamic_mask & static_mask
        rows = (
            dynamic_mask[None].expand(batch_size, -1) if dynamic_mask.dim() == 1 else dynamic_mask
        )
        if not bool(rows.any(dim=1).all()):
            raise ValueError(
                "The montage/per-trial channel-mask intersection leaves an empty trial."
            )
        return dynamic_mask

    def _likelihood_channel_mask(
        self,
        observation_mask: torch.Tensor | None,
        *,
        batch_size: int,
        channels: int,
    ) -> torch.Tensor | None:
        """Restrict conditional histories to channels supported by the fold profile."""

        if self.generative_profile is None:
            return observation_mask
        profile_mask = self.generative_profile.channel_mask.to(self.device, dtype=torch.bool)
        if profile_mask.shape != (channels,):
            raise ValueError("Generative profile mask does not match the current montage.")
        if observation_mask is None:
            return profile_mask
        runtime = observation_mask.to(self.device, dtype=torch.bool)
        if runtime.shape == (channels,):
            return runtime & profile_mask
        if runtime.shape == (batch_size, channels):
            return runtime & profile_mask[None]
        raise ValueError("Likelihood channel mask must be (C,) or (B,C).")

    def _batch_reconstruction_weight(
        self,
        channel_mask: torch.Tensor | None,
        *,
        batch_size: int,
        channels: int,
    ) -> float:
        """Disable reconstruction rather than supervise unobserved sensor zeros."""

        weight = float(self.cfg.lambda_recon)
        if weight <= 0.0 or self.reconstruction_profile is None or channel_mask is None:
            return weight
        observed = channel_mask.to(device=self.device, dtype=torch.bool)
        if observed.shape == (channels,):
            observed = observed[None].expand(batch_size, -1)
        if observed.shape != (batch_size, channels):
            raise ValueError("Reconstruction channel mask must be (C,) or (B,C).")
        profile_channels = self.reconstruction_profile.channel_mask.to(
            device=self.device,
            dtype=torch.bool,
        )
        return weight if bool(observed[:, profile_channels].all()) else 0.0

    def _train_step(
        self,
        context: TrialContext,
        step: int,
        *,
        record_gradient_diagnostics: bool = False,
    ):
        """单步训练：增强 + 前向（AMP）+ 反向 + 梯度累积（D-accum）。

        domain_id : (B,) long 可选。Phase 3 辅助域对齐时使用；compute_losses 会按
        P9 只对 main_domain 样本计算 L_target/L_early/L_amp。
        """
        non_blocking = self.device.type == "cuda" and context.X.is_pinned()
        context = context.to(self.device, non_blocking=non_blocking)
        X, y, domain_id = context.X, context.y, context.domain_id
        effective_channel_mask = self._effective_channel_mask(
            context.channel_mask,
            batch_size=X.shape[0],
            channels=X.shape[1],
        )

        likelihood_X = X
        likelihood_channel_mask = self._likelihood_channel_mask(
            effective_channel_mask,
            batch_size=X.shape[0],
            channels=X.shape[1],
        )
        if self.cfg.augment:
            # channel_mask 透传：reference_jitter/gaussian_noise 不得污染缺失通道（review v6 P0-2）
            X, effective_channel_mask = apply_augmentations(
                X,
                channel_mask=effective_channel_mask,
                return_channel_mask=True,
            )

        with self._autocast_ctx():
            # Head-D only consumes attention when its physical amplitude loss is active.
            output = self.model(
                X,
                self.E_chn,
                self.E_sub,
                channel_mask=effective_channel_mask,
                domain_id=domain_id,
                return_attention=self.cfg.lambda_amp > 0.0,
                return_likelihood=self.cfg.lambda_innovation > 0.0,
                likelihood_input=likelihood_X,
                likelihood_channel_mask=likelihood_channel_mask,
                likelihood_class_means=(
                    self.generative_profile.class_means
                    if self.generative_profile is not None
                    else None
                ),
                **({"likelihood_labels": y} if self.cfg.lambda_innovation > 0.0 else {}),
            )

            # Phase 2 自监督 jitter 一致性：已知偏移的第二个 forward 只为 τ 尺度锚定。
            tau_shift = None
            shift_ms = None
            if self.cfg.lambda_jit > 0.0 and (
                self.cfg.jit_prob >= 1.0
                or torch.rand(1, device=X.device).item() < self.cfg.jit_prob
            ):
                sfreq = float(getattr(self.model, "sfreq", 256.0))
                max_shift = max(1, int(round(self.cfg.jit_max_ms / 1000.0 * sfreq)))
                mag = torch.randint(1, max_shift + 1, (X.shape[0],), device=X.device)
                sign = torch.randint(0, 2, (X.shape[0],), device=X.device) * 2 - 1
                shift_samples = mag * sign
                X_shift = known_time_shift(X, shift_samples)
                output_shift = self.model(
                    X_shift,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=domain_id,
                    return_attention=False,
                    return_heads=False,
                    return_likelihood=False,
                )
                tau_shift = output_shift.tau
                shift_ms = shift_samples.float() * (1000.0 / sfreq)

            losses = compute_losses(
                output,
                self.model.component_window.tau0_bounded,
                y,
                lambda2=self.cfg.lambda2,
                lambda3=self.cfg.lambda3,
                lambda_pcw=self.cfg.lambda_pcw,
                lambda_digit=0.0,
                set_metadata=None,
                digit_evidence_ks=self.cfg.digit_evidence_ks,
                digit_evidence_weights=self.cfg.digit_evidence_weights,
                lambda_amp=self.cfg.lambda_amp,
                lambda_recon=self._batch_reconstruction_weight(
                    effective_channel_mask,
                    batch_size=X.shape[0],
                    channels=X.shape[1],
                ),
                reconstruction_profile=self.reconstruction_profile,
                recon_waveform_weight=self.cfg.recon_waveform_weight,
                recon_projection_weight=self.cfg.recon_projection_weight,
                recon_nll_weight=self._active_recon_nll_weight,
                lambda_innovation=self.cfg.lambda_innovation,
                generative_profile=self.generative_profile,
                generative_profile_validated=True,
                innovation_covariance_weight=self._active_innovation_covariance_weight,
                lambda_morphology_l0=self.cfg.lambda_morphology_l0,
                lambda_jit=self.cfg.lambda_jit,
                tau_shift=tau_shift,
                shift_ms=shift_ms,
                lambda4=self.cfg.lambda4,
                mmd_bandwidth=self.cfg.mmd_bandwidth,
                lambda_orth=self.cfg.lambda_orth,
                lambda_adv=self.cfg.lambda_adv,
                lambda_private=self.cfg.lambda_private,
                reconstruct_all_domains=self.cfg.reconstruct_all_domains,
                main_domain=self.cfg.main_domain,
                aux_domain=self.cfg.aux_domain,
                pos_weight=self.cfg.pos_weight,
                tau_scale_ms=self.cfg.tau_scale_ms,
                z_features=output.features,
                domain_ids=domain_id,
                active_domain_indices=self.active_domain_indices,
                X=X,
                pz_channel=self.pz_channel,
            )
            loss = losses.total / self.cfg.accum_steps

        # P9：纯辅助域 batch 的总损失可为零且无计算图；跳过 backward。但若本累积周期
        # 已有前面 GTN batch 累积的梯度，到边界仍必须 step，否则会丢梯度（audit P1-4）。
        if loss.requires_grad:
            loss.backward()
            self._accum_has_grad = True
            if record_gradient_diagnostics:
                tau_grad = self.model.component_window.tau0.grad
                head_grads = [
                    p.grad.detach().norm()
                    for p in self.model.heads.parameters()
                    if p.grad is not None
                ]
                self._pcw_tau_grad_norms.append(
                    float(tau_grad.detach().norm()) if tau_grad is not None else 0.0
                )
                self._pcw_head_grad_norms.append(
                    float(torch.stack(head_grads).norm()) if head_grads else 0.0
                )
                pcw_grads = [
                    p.grad.detach().norm()
                    for p in self.model.heads.head_pcw.parameters()
                    if p.grad is not None
                ]
                self._pcw_classifier_grad_norms.append(
                    float(torch.stack(pcw_grads).norm()) if pcw_grads else 0.0
                )
                pcw_modules = [
                    self.model.tokenizer,
                    self.model.encoder,
                    self.model.component_window,
                    self.model.component_decoder,
                ]
                pcw_path_grads = [
                    parameter.grad.detach().norm()
                    for module in pcw_modules
                    if module is not None
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                innovation_modules = [
                    self.model.innovation_encoder,
                    self.model.innovation_decoder,
                ]
                innovation_grads = [
                    parameter.grad.detach().norm()
                    for module in innovation_modules
                    if module is not None
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                self._pcw_path_grad_norms.append(
                    float(torch.stack(pcw_path_grads).norm()) if pcw_path_grads else 0.0
                )
                self._innovation_path_grad_norms.append(
                    float(torch.stack(innovation_grads).norm()) if innovation_grads else 0.0
                )
        if (step + 1) % self.cfg.accum_steps == 0:
            if self._accum_has_grad:
                self._optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)  # D-zero-grad-none：释放梯度张量
            self._accum_has_grad = False
        # 返回设备端 0 维张量而非 float：fit 在 epoch 结束时只做一次 .item()，
        # 避免每个 batch 都触发 GPU→CPU 同步（D-loss-sync）。
        self._last_loss_vector = torch.stack(
            [
                (
                    getattr(losses, name).detach().float()
                    if getattr(losses, name) is not None
                    else torch.zeros((), device=self.device, dtype=torch.float32)
                )
                for name in self._loss_component_names
            ]
        )
        return losses.total.detach()

    def _set_train_step(
        self, context: TrialContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize only the deployment-aligned nested multi-K objective."""

        context = context.to(self.device)
        if context.set_metadata is None:
            raise ValueError("Set-objective batches require SetMetadata.")
        X = context.X
        effective_channel_mask = self._effective_channel_mask(
            context.channel_mask,
            batch_size=X.shape[0],
            channels=X.shape[1],
        )
        if self.cfg.augment:
            X, effective_channel_mask = apply_augmentations(
                X,
                channel_mask=effective_channel_mask,
                return_channel_mask=True,
            )
        # Flush a partial trial-loss accumulation before entering the distinct
        # set objective; otherwise zero_grad would discard its gradients.
        if self._accum_has_grad:
            self._optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_has_grad = False
        else:
            self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_ctx():
            output = self.model(
                X,
                self.E_chn,
                self.E_sub,
                channel_mask=effective_channel_mask,
                domain_id=context.domain_id,
                return_attention=False,
                return_likelihood=False,
            )
            if output.heads is None:
                raise RuntimeError("Set objective requires final logits.")
            if self.cfg.lambda_conditional_nll > 0.0:
                quality = self._quality_features(
                    X, output, context.domain_id, effective_channel_mask,
                    check_finite=not context.prevalidated,
                )
                if isinstance(self.model.repetition_evidence, AdditiveRepetitionEvidence):
                    digit_loss, conditional_nll, _, _ = additive_repetition_multi_k_objective(
                        output.heads.logit_target,
                        quality,
                        context.y,
                        context.set_metadata,
                        self.model.repetition_evidence,
                        evidence_ks=self.cfg.repetition_v12_evidence_ks,
                        evidence_weights=self.cfg.repetition_v12_evidence_weights,
                        state_residual_l2_weight=self.cfg.repetition_state_residual_l2_weight,
                        fidelity_aux_weight=self.cfg.repetition_reliability_aux_weight,
                    )
                else:
                    digit_loss, conditional_nll, _ = repetition_multi_k_objective(
                        output.heads.logit_target,
                        quality,
                        context.y,
                        context.set_metadata,
                        self.model.repetition_evidence,
                        evidence_ks=self.cfg.digit_evidence_ks,
                        evidence_weights=self.cfg.digit_evidence_weights,
                        reliability_aux_weight=self.cfg.repetition_reliability_aux_weight,
                    )
            else:
                digit_loss, _ = gtn_multi_k_cross_entropy(
                    output.heads.logit_target,
                    context.y,
                    context.set_metadata,
                    evidence_ks=self.cfg.digit_evidence_ks,
                    evidence_weights=self.cfg.digit_evidence_weights,
                )
                conditional_nll = digit_loss.detach() * 0.0
            weighted = (
                self.cfg.lambda_digit * digit_loss
                + self.cfg.lambda_conditional_nll * conditional_nll
            )
        weighted.backward()
        self._optimizer_step()
        return weighted.detach(), digit_loss.detach(), conditional_nll.detach()

    @torch.inference_mode()
    def _evaluate_with_task(
        self, loader, *, compute_auc: bool = True
    ) -> tuple[float, float, float, float | None]:
        """Evaluate objective, task, innovation loss, and optionally trial AUC."""
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        task_total = torch.zeros((), device=self.device, dtype=torch.float32)
        density_total = torch.zeros((), device=self.device, dtype=torch.float32)
        validation_logits: list[torch.Tensor] = []
        validation_labels: list[torch.Tensor] = []
        n = 0
        for batch in loader:
            context = self._unpack_batch(batch)
            non_blocking = self.device.type == "cuda" and context.X.is_pinned()
            context = context.to(self.device, non_blocking=non_blocking)
            X, y, domain_id = context.X, context.y, context.domain_id
            effective_channel_mask = self._effective_channel_mask(
                context.channel_mask,
                batch_size=X.shape[0],
                channels=X.shape[1],
            )
            likelihood_channel_mask = self._likelihood_channel_mask(
                effective_channel_mask,
                batch_size=X.shape[0],
                channels=X.shape[1],
            )
            with self._autocast_ctx():
                output = self.model(
                    X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=domain_id,
                    return_attention=True,
                    return_likelihood=self.cfg.lambda_innovation > 0.0,
                    likelihood_channel_mask=likelihood_channel_mask,
                    likelihood_class_means=(
                        self.generative_profile.class_means
                        if self.generative_profile is not None
                        else None
                    ),
                    **({"likelihood_labels": y} if self.cfg.lambda_innovation > 0.0 else {}),
                )
                losses = compute_losses(
                    output,
                    self.model.component_window.tau0_bounded,
                    y,
                    lambda2=self.cfg.lambda2,
                    lambda3=self.cfg.lambda3,
                    lambda_pcw=self.cfg.lambda_pcw,
                    lambda_digit=0.0,
                    set_metadata=None,
                    digit_evidence_ks=self.cfg.digit_evidence_ks,
                    digit_evidence_weights=self.cfg.digit_evidence_weights,
                    lambda_amp=self.cfg.lambda_amp,
                    lambda_recon=self._batch_reconstruction_weight(
                        effective_channel_mask,
                        batch_size=X.shape[0],
                        channels=X.shape[1],
                    ),
                    reconstruction_profile=self.reconstruction_profile,
                    recon_waveform_weight=self.cfg.recon_waveform_weight,
                    recon_projection_weight=self.cfg.recon_projection_weight,
                    recon_nll_weight=self._active_recon_nll_weight,
                    lambda_innovation=self.cfg.lambda_innovation,
                    generative_profile=self.generative_profile,
                    generative_profile_validated=True,
                    innovation_covariance_weight=self._active_innovation_covariance_weight,
                    lambda_morphology_l0=self.cfg.lambda_morphology_l0,
                    lambda4=self.cfg.lambda4,
                    mmd_bandwidth=self.cfg.mmd_bandwidth,
                    lambda_orth=self.cfg.lambda_orth,
                    lambda_adv=self.cfg.lambda_adv,
                    lambda_private=self.cfg.lambda_private,
                    reconstruct_all_domains=self.cfg.reconstruct_all_domains,
                    main_domain=self.cfg.main_domain,
                    aux_domain=self.cfg.aux_domain,
                    pos_weight=self.cfg.pos_weight,
                    tau_scale_ms=self.cfg.tau_scale_ms,
                    z_features=output.features,
                    domain_ids=domain_id,
                    active_domain_indices=self.active_domain_indices,
                    X=X,
                    pz_channel=self.pz_channel,
                )
                if output.heads is None:
                    raise RuntimeError("Task validation requires final trial logits.")
                logits = output.heads.logit_target
                labels = y.reshape_as(logits).to(dtype=logits.dtype)
                task_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=torch.as_tensor(
                        self.cfg.pos_weight,
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                )
            if compute_auc:
                validation_logits.append(logits.detach().float().reshape(-1))
            if compute_auc:
                validation_labels.append(labels.detach().float().reshape(-1))
            total += losses.total.detach().float() * X.shape[0]
            task_total += task_loss.detach().float() * X.shape[0]
            density_total += losses.innovation_nll.detach().float() * X.shape[0]
            n += X.shape[0]
        self.model.train()
        values = torch.stack((total, task_total, density_total)).div(max(n, 1)).cpu().tolist()
        auc: float | None = None
        if compute_auc and validation_labels:
            labels = torch.cat(validation_labels).cpu()
            logits = torch.cat(validation_logits).cpu()
            if len(torch.unique(labels)) == 2:
                auc = float(roc_auc_score(labels.numpy(), logits.numpy()))
        return float(values[0]), float(values[1]), float(values[2]), auc

    def _evaluate(self, loader) -> float:
        """Evaluate the complete optimization objective for diagnostics."""

        return self._evaluate_with_task(loader)[0]

    @torch.inference_mode()
    def _evaluate_auc(self, loader) -> float | None:
        """One lightweight validation forward for final AUC reporting.

        Kept separate from the per-epoch early-stopping loop so the hot loop
        never moves validation logits to CPU or runs sklearn on every epoch.
        """
        self.model.eval()
        logits_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for batch in loader:
            context = self._unpack_batch(batch)
            context = context.to(self.device, non_blocking=False)
            effective_channel_mask = self._effective_channel_mask(
                context.channel_mask,
                batch_size=context.X.shape[0],
                channels=context.X.shape[1],
            )
            with self._autocast_ctx():
                output = self.model(
                    context.X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=context.domain_id,
                    return_attention=False,
                    return_likelihood=False,
                )
            if output.heads is None:
                raise RuntimeError("Final AUC validation requires final logits.")
            logits_parts.append(output.heads.logit_target.detach().float().reshape(-1))
            label_parts.append(context.y.reshape(-1).to(dtype=torch.float32))
        self.model.train()
        if not logits_parts:
            return None
        logits = torch.cat(logits_parts).cpu().numpy()
        labels = torch.cat(label_parts).cpu().numpy()
        if len(np.unique(labels)) == 2:
            return float(roc_auc_score(labels, logits))
        return None

    @staticmethod
    def _is_density_state_key(key: str) -> bool:
        return key.startswith(("innovation_encoder.", "innovation_decoder."))

    def _capture_density_state(self) -> dict[str, torch.Tensor]:
        state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
            if self._is_density_state_key(key)
        }
        if not state:
            raise RuntimeError(
                "Strict-past density checkpointing found no innovation encoder/decoder state."
            )
        return state

    def _restore_selected_states(self) -> None:
        """Restore task-best state, then overlay the independently selected density state."""

        if self.best_task_state is not None:
            self.model.load_state_dict(self.best_task_state)
        if self.best_density_state is None:
            return
        current_state = self.model.state_dict()
        expected_keys = {key for key in current_state if self._is_density_state_key(key)}
        selected_keys = set(self.best_density_state)
        if not expected_keys or selected_keys != expected_keys:
            missing = sorted(expected_keys - selected_keys)
            unexpected = sorted(selected_keys - expected_keys)
            raise RuntimeError(
                "Density checkpoint does not match the model innovation state: "
                f"missing={missing}, unexpected={unexpected}."
            )
        with torch.no_grad():
            for key, value in self.best_density_state.items():
                current_state[key].copy_(value)

    @torch.inference_mode()
    def _evaluate_task(self, loader) -> float:
        """Evaluate only target classification for checkpoint selection."""

        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for batch in loader:
            context = self._unpack_batch(batch)
            non_blocking = self.device.type == "cuda" and context.X.is_pinned()
            context = context.to(self.device, non_blocking=non_blocking)
            effective_channel_mask = self._effective_channel_mask(
                context.channel_mask,
                batch_size=context.X.shape[0],
                channels=context.X.shape[1],
            )
            with self._autocast_ctx():
                output = self.model(
                    context.X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=context.domain_id,
                    return_attention=False,
                    return_likelihood=False,
                )
                if output.heads is None:
                    raise RuntimeError("Task validation requires final trial logits.")
                logits = output.heads.logit_target
                labels = context.y.reshape_as(logits).to(dtype=logits.dtype)
                pos_weight = torch.as_tensor(
                    self.cfg.pos_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                task_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                )
            total += task_loss.detach().float() * context.X.shape[0]
            n += context.X.shape[0]
        self.model.train()
        if n == 0:
            raise ValueError("Task validation loader produced no trials.")
        return (total / n).item()

    @torch.inference_mode()
    def _evaluate_set_with_task(self, loader) -> tuple[float, float]:
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        task_total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for batch in loader:
            context = self._unpack_batch(batch).to(self.device)
            if context.set_metadata is None:
                raise ValueError("Validation set-objective batches require SetMetadata.")
            effective_channel_mask = self._effective_channel_mask(
                context.channel_mask,
                batch_size=context.X.shape[0],
                channels=context.X.shape[1],
            )
            with self._autocast_ctx():
                output = self.model(
                    context.X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=context.domain_id,
                    return_attention=False,
                    return_likelihood=False,
                )
                if output.heads is None:
                    raise RuntimeError("Set validation requires final logits.")
                if self.cfg.lambda_conditional_nll > 0.0:
                    quality = self._quality_features(
                        context.X,
                        output,
                        context.domain_id,
                        effective_channel_mask,
                        check_finite=not context.prevalidated,
                    )
                    if isinstance(self.model.repetition_evidence, AdditiveRepetitionEvidence):
                        digit_loss, conditional_nll, _, _ = additive_repetition_multi_k_objective(
                            output.heads.logit_target,
                            quality,
                            context.y,
                            context.set_metadata,
                            self.model.repetition_evidence,
                            evidence_ks=self.cfg.repetition_v12_evidence_ks,
                            evidence_weights=self.cfg.repetition_v12_evidence_weights,
                            fidelity_aux_weight=0.0,
                            state_residual_l2_weight=0.0,
                        )
                    else:
                        digit_loss, conditional_nll, _ = repetition_multi_k_objective(
                            output.heads.logit_target,
                            quality,
                            context.y,
                            context.set_metadata,
                            self.model.repetition_evidence,
                            evidence_ks=self.cfg.digit_evidence_ks,
                            evidence_weights=self.cfg.digit_evidence_weights,
                            reliability_aux_weight=self.cfg.repetition_reliability_aux_weight,
                        )
                else:
                    digit_loss, _ = gtn_multi_k_cross_entropy(
                        output.heads.logit_target,
                        context.y,
                        context.set_metadata,
                        evidence_ks=self.cfg.digit_evidence_ks,
                        evidence_weights=self.cfg.digit_evidence_weights,
                    )
                    conditional_nll = digit_loss * 0.0
                set_loss = (
                    self.cfg.lambda_digit * digit_loss
                    + self.cfg.lambda_conditional_nll * conditional_nll
                )
            total += set_loss.detach().float()
            task_total += digit_loss.detach().float()
            n += 1
        self.model.train()
        if n == 0:
            raise ValueError("Set validation loader produced no complete batches.")
        values = torch.stack((total, task_total)).div(n).cpu().tolist()
        return float(values[0]), float(values[1])

    def _evaluate_set(self, loader) -> float:
        return self._evaluate_set_with_task(loader)[0]

    @torch.inference_mode()
    def _evaluate_set_task(self, loader) -> float:
        """Evaluate deployment-aligned digit CE without density auxiliaries."""

        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for batch in loader:
            context = self._unpack_batch(batch).to(self.device)
            if context.set_metadata is None:
                raise ValueError("Validation set-objective batches require SetMetadata.")
            effective_channel_mask = self._effective_channel_mask(
                context.channel_mask,
                batch_size=context.X.shape[0],
                channels=context.X.shape[1],
            )
            with self._autocast_ctx():
                output = self.model(
                    context.X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=effective_channel_mask,
                    domain_id=context.domain_id,
                    return_attention=False,
                    return_likelihood=False,
                )
                if output.heads is None:
                    raise RuntimeError("Set task validation requires final logits.")
                digit_loss, _ = gtn_multi_k_cross_entropy(
                    output.heads.logit_target,
                    context.y,
                    context.set_metadata,
                    evidence_ks=self.cfg.digit_evidence_ks,
                    evidence_weights=self.cfg.digit_evidence_weights,
                )
            total += digit_loss.detach().float()
            n += 1
        self.model.train()
        if n == 0:
            raise ValueError("Set task validation loader produced no complete batches.")
        return (total / n).item()

    def refit_repetition_evidence(self, loader, *, epochs: int | None = None) -> list[float]:
        """Refit only the density head after inner-validation temperature calibration."""

        evidence_model = self.model.repetition_evidence
        if evidence_model is None or self.cfg.lambda_conditional_nll <= 0.0:
            return []
        n_epochs = self.cfg.repetition_refit_epochs if epochs is None else int(epochs)
        if n_epochs <= 0:
            return []
        requires_grad = {
            name: parameter.requires_grad for name, parameter in self.model.named_parameters()
        }
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in evidence_model.parameters():
            parameter.requires_grad_(True)
        if isinstance(evidence_model, AdditiveRepetitionEvidence):
            evidence_model.prepare_state_residual_for_refit(gain=0.1)
            parameter_groups = [{"params": evidence_model.parameters(), "lr": self.cfg.lr}]
        else:
            reliability_parameters = list(evidence_model.reliability_net.parameters())
            reliability_ids = {id(parameter) for parameter in reliability_parameters}
            density_parameters = [
                parameter
                for parameter in evidence_model.parameters()
                if id(parameter) not in reliability_ids
            ]
            parameter_groups = [
                {"params": density_parameters, "lr": self.cfg.lr},
                {
                    "params": reliability_parameters,
                    "lr": self.cfg.lr * self.cfg.repetition_reliability_lr_multiplier,
                },
            ]
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.cfg.weight_decay)
        history: list[float] = []
        try:
            self.model.eval()
            evidence_model.train()
            for _ in range(n_epochs):
                total = torch.zeros((), device=self.device, dtype=torch.float32)
                n_batches = 0
                for batch in loader:
                    context = self._unpack_batch(batch).to(self.device)
                    if context.set_metadata is None:
                        raise ValueError("Repetition refit requires ordered SetMetadata.")
                    effective_channel_mask = self._effective_channel_mask(
                        context.channel_mask,
                        batch_size=context.X.shape[0],
                        channels=context.X.shape[1],
                    )
                    with torch.no_grad():
                        output = self.model(
                            context.X,
                            self.E_chn,
                            self.E_sub,
                            channel_mask=effective_channel_mask,
                            domain_id=context.domain_id,
                            return_attention=False,
                            return_likelihood=False,
                        )
                        if output.heads is None:
                            raise RuntimeError("Repetition refit requires final trial logits.")
                        logits = output.heads.logit_target.detach().clone()
                        quality = (
                            self._quality_features(
                                context.X,
                                output,
                                context.domain_id,
                                effective_channel_mask,
                                check_finite=not context.prevalidated,
                            )
                            .detach()
                            .clone()
                        )
                    if isinstance(evidence_model, AdditiveRepetitionEvidence):
                        digit_loss, conditional_nll, _, _ = additive_repetition_multi_k_objective(
                            logits,
                            quality,
                            context.y,
                            context.set_metadata,
                            evidence_model,
                            evidence_ks=self.cfg.repetition_v12_evidence_ks,
                            fidelity_aux_weight=self.cfg.repetition_reliability_aux_weight,
                            evidence_weights=self.cfg.repetition_v12_evidence_weights,
                            state_residual_l2_weight=self.cfg.repetition_state_residual_l2_weight,
                        )
                    else:
                        digit_loss, conditional_nll, _ = repetition_multi_k_objective(
                            logits,
                            quality,
                            context.y,
                            context.set_metadata,
                            evidence_model,
                            evidence_ks=self.cfg.digit_evidence_ks,
                            evidence_weights=self.cfg.digit_evidence_weights,
                            reliability_aux_weight=self.cfg.repetition_reliability_aux_weight,
                        )
                    loss = (
                        self.cfg.lambda_digit * digit_loss
                        + self.cfg.lambda_conditional_nll * conditional_nll
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += loss.detach().float()
                    n_batches += 1
                if n_batches == 0:
                    raise ValueError("Repetition refit loader produced no complete batches.")
                history.append(float((total / n_batches).cpu()))
        finally:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(requires_grad[name])
            self.model.train()
        return history

    def fit(
        self,
        train_loader,
        val_loader=None,
        *,
        train_set_loader=None,
        val_set_loader=None,
        reconstruction_context: TrialContext | None = None,
        on_epoch_end: Callable[[dict[str, object]], None] | None = None,
        on_epoch_checkpoint: Callable[
            [dict[str, object], dict[str, torch.Tensor]], None
        ]
        | None = None,
    ) -> dict:
        """训练主循环：epoch 迭代 + early stop + OOM 保护（D-oom）。

        train_loader/val_loader 提供 (X, y) 或 (X, y, domain_id) batch。返回
          {train_losses, val_losses}。domain_id 用于 Phase 3 P9 域隔离。
        """
        train_losses: list[float] = []
        train_loss_components: list[dict[str, float]] = []
        val_losses: list[float] = []
        val_objective_losses: list[float] = []
        val_innovation_nlls: list[float] = []
        task_val_aucs: list[float | None] = []
        phases: list[str] = []
        selection_epochs: list[int] = []
        epoch_update_counts: list[dict[str, int]] = []
        patience_left = self.cfg.early_stop_patience
        task_patience_exhausted = False
        fixed_research_budget = self.cfg.lambda_innovation > 0.0
        first_joint_epoch = self._first_joint_epoch()
        if val_loader is not None and self.cfg.epochs <= first_joint_epoch:
            raise ValueError(
                "Phase-aware early stopping requires at least one joint epoch after "
                f"variance scheduling; first_joint_epoch={first_joint_epoch}, "
                f"got epochs={self.cfg.epochs}."
            )
        set_objective_active = self.cfg.lambda_digit > 0.0 or self.cfg.lambda_conditional_nll > 0.0
        if set_objective_active and train_set_loader is None:
            raise ValueError(
                "Set objectives require an explicit GTN set loader in addition to "
                "the full-coverage trial loader."
            )
        if not set_objective_active and train_set_loader is not None:
            raise ValueError("A set loader was provided while set objectives are disabled.")
        if val_loader is not None and set_objective_active and val_set_loader is None:
            raise ValueError("Set-supervised early stopping requires a validation set loader.")
        fold_label = "serial" if self.fold_id is None else str(self.fold_id)
        print(f"[fold {fold_label}] fold-local setup start", flush=True)
        self._prepare_reconstruction_profile(reconstruction_context)
        repetition_context = (
            train_set_loader.full_context
            if train_set_loader is not None and hasattr(train_set_loader, "full_context")
            else reconstruction_context
        )
        self._prepare_repetition_quality_normalizer(repetition_context)
        self._prepare_repetition_calibration(repetition_context)
        print(f"[fold {fold_label}] fold-local setup done; epoch loop start", flush=True)
        finite_guard_loaders = [
            loader
            for loader in (train_loader, val_loader, train_set_loader, val_set_loader)
            if loader is not None
        ]
        if finite_guard_loaders and all(
            getattr(loader, "finite_validated", False) for loader in finite_guard_loaders
        ):
            # Preloaded loaders validated their complete tensors once at upload.
            # Skip the per-forward eager finite check so GPU kernels are not
            # interrupted by a device->host sync on every batch.
            self.model.validate_input_finite = False

        n_train_batches_per_epoch = len(train_loader)
        n_set_batches_per_epoch = len(train_set_loader) if train_set_loader is not None else 0
        planned_steps_per_epoch = self._planned_steps_per_epoch(
            n_train_batches_per_epoch,
            n_set_batches_per_epoch,
        )
        self._configure_lr_scheduler(self.cfg.epochs * planned_steps_per_epoch)
        maximum_epoch_count = self.cfg.epochs
        for epoch in range(maximum_epoch_count):
            variance_progress = self._variance_progress_for_epoch(epoch)
            self._active_recon_nll_weight = float(self.cfg.recon_nll_weight) * variance_progress
            self._active_innovation_covariance_weight = variance_progress
            epoch_limit = self._required_epoch_count()
            if epoch >= epoch_limit:
                break
            self._variance_nll_weight_history.append(self._active_recon_nll_weight)
            self._innovation_covariance_weight_history.append(
                self._active_innovation_covariance_weight
            )
            phase = self._phase_for_epoch(epoch)
            phases.append(phase)
            self.model.train()
            epoch_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            epoch_component_sum = torch.zeros_like(self._last_loss_vector)
            n_batches = 0
            set_epoch_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            set_digit_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            set_conditional_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            n_set_batches = 0
            trial_optimizer_steps = 0
            set_optimizer_steps = 0
            try:
                trial_iterator = iter(train_loader)
                set_iterator = iter(train_set_loader) if train_set_loader is not None else None
                for source in self._interleaved_update_sources(
                    n_train_batches_per_epoch,
                    n_set_batches_per_epoch,
                ):
                    steps_before = self.optimizer_steps
                    if source == "trial":
                        batch = next(trial_iterator)
                        context = self._unpack_batch(batch)
                        epoch_loss += self._train_step(
                            context,
                            n_batches,
                            record_gradient_diagnostics=(
                                self.cfg.track_pcw_gradients
                                and n_batches + 1 == n_train_batches_per_epoch
                            ),
                        ).float()
                        epoch_component_sum += self._last_loss_vector
                        n_batches += 1
                        trial_optimizer_steps += self.optimizer_steps - steps_before
                    else:
                        if set_iterator is None:
                            raise RuntimeError("Set interleave requested without a set iterator.")
                        set_batch = next(set_iterator)
                        weighted_set, digit_set, conditional_set = self._set_train_step(
                            self._unpack_batch(set_batch)
                        )
                        set_epoch_loss += weighted_set.float()
                        set_digit_sum += digit_set.float()
                        set_conditional_sum += conditional_set.float()
                        n_set_batches += 1
                        set_optimizer_steps += self.optimizer_steps - steps_before
            except torch.OutOfMemoryError:  # DP6：torch 2.x 统一 OOM
                raise RuntimeError(
                    f"显存溢出（OOM）：请减小 batch_size（当前 {self.cfg.batch_size}）、"
                    f"调大 accum_steps、或关闭其他占用显存的程序后重试。"
                ) from None
            except RuntimeError as e:  # 旧版兜底
                if "out of memory" in str(e).lower():
                    raise RuntimeError("显存溢出（OOM）：请减小 batch_size 后重试。") from None
                raise

            # 尾部梯度累积：n_batches % accum_steps ≠ 0 时补一次 step（review P3，防尾部梯度丢失）
            if self.cfg.accum_steps > 1 and self._accum_has_grad:
                steps_before = self.optimizer_steps
                self._optimizer_step()
                trial_optimizer_steps += self.optimizer_steps - steps_before
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_has_grad = False

            epoch_update_counts.append(
                {
                    "trial": int(trial_optimizer_steps),
                    "set": int(set_optimizer_steps),
                    "total": int(trial_optimizer_steps + set_optimizer_steps),
                }
            )

            # D-loss-sync：整个 epoch 只在结束时做一次设备→主机同步
            trial_mean = epoch_loss / max(n_batches, 1)
            component_mean = epoch_component_sum / max(n_batches, 1)
            if n_set_batches:
                trial_mean = trial_mean + set_epoch_loss / n_set_batches
                digit_idx = self._loss_component_names.index("digit")
                component_mean[digit_idx] = set_digit_sum / n_set_batches
                conditional_idx = self._loss_component_names.index("conditional_nll")
                component_mean[conditional_idx] = set_conditional_sum / n_set_batches
            epoch_summary = torch.cat([trial_mean.view(1), component_mean])
            summary_values = epoch_summary.cpu().tolist()
            train_loss = summary_values[0]
            train_losses.append(train_loss)
            component_values = summary_values[1:]
            train_loss_components.append(
                dict(zip(self._loss_component_names, component_values, strict=True))
            )

            task_val_loss: float | None = None
            objective_val_loss: float | None = None
            val_innovation_nll: float | None = None
            task_val_auc: float | None = None
            selection_active = False
            if val_loader is not None:
                (
                    val_objective_loss,
                    val_loss,
                    density_nll,
                    task_val_auc,
                ) = self._evaluate_with_task(val_loader, compute_auc=True)
                if val_set_loader is not None:
                    set_objective_loss, set_task_loss = self._evaluate_set_with_task(
                        val_set_loader
                    )
                    val_objective_loss += set_objective_loss
                    val_loss += self.cfg.lambda_digit * set_task_loss
                val_losses.append(val_loss)
                val_objective_losses.append(val_objective_loss)
                val_innovation_nlls.append(density_nll)
                task_val_aucs.append(task_val_auc)
                selection_active = phase == "joint"
                if selection_active:
                    selection_epochs.append(epoch)
                    if val_loss < self.best_val_loss - 1e-6:
                        self.best_val_loss = val_loss
                        self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        self.best_epoch = epoch
                        self.best_task_val_loss = self.best_val_loss
                        self.best_task_state = self.best_state
                        self.best_task_epoch = self.best_epoch
                        patience_left = self.cfg.early_stop_patience
                    else:
                        patience_left = max(0, patience_left - 1)
                        task_patience_exhausted |= patience_left <= 0
                    if (
                        fixed_research_budget
                        and density_nll < self.best_density_nll - 1e-8
                    ):
                        self.best_density_nll = density_nll
                        self.best_density_state = self._capture_density_state()
                        self.best_density_epoch = epoch
                task_val_loss = float(val_loss)
                objective_val_loss = float(val_objective_loss)
                val_innovation_nll = float(density_nll) if fixed_research_budget else None
            else:
                task_val_loss = None
                objective_val_loss = None
                val_innovation_nll = None
                task_val_auc = None

            if on_epoch_end is not None or self.cfg.epoch_trajectory_audit:
                event = {
                    "epoch": epoch + 1,
                    "epoch_limit": epoch_limit,
                    "train_loss": float(train_loss),
                    "train_loss_components": dict(train_loss_components[-1]),
                    "task_val_loss": task_val_loss,
                    "task_val_auc": task_val_auc,
                    "objective_val_loss": objective_val_loss,
                    "val_innovation_nll": val_innovation_nll,
                    "phase": phase,
                    "optimizer_steps": int(self.optimizer_steps),
                    "epoch_update_counts": dict(epoch_update_counts[-1]),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "selection_active": bool(selection_active),
                    "patience_left": int(patience_left) if val_loader is not None else None,
                    "early_stop_patience": int(self.cfg.early_stop_patience),
                    "best_epoch": self.best_epoch + 1 if self.best_epoch is not None else None,
                    "best_task_epoch": (
                        self.best_task_epoch + 1 if self.best_task_epoch is not None else None
                    ),
                    "best_density_epoch": (
                        self.best_density_epoch + 1 if self.best_density_epoch is not None else None
                    ),
                    "best_val_loss": (
                        float(self.best_val_loss)
                        if self.best_val_loss != float("inf")
                        else None
                    ),
                    "best_task_val_loss": (
                        float(self.best_task_val_loss)
                        if self.best_task_val_loss != float("inf")
                        else None
                    ),
                    "best_density_nll": (
                        float(self.best_density_nll)
                        if self.best_density_nll != float("inf")
                        else None
                    ),
                    "task_patience_exhausted": bool(task_patience_exhausted),
                    "will_early_stop": bool(
                        selection_active and patience_left <= 0 and not fixed_research_budget
                    ),
                }
                if self.cfg.epoch_trajectory_audit:
                    if on_epoch_checkpoint is None:
                        raise RuntimeError(
                            "epoch_trajectory_audit requires an epoch checkpoint sink."
                        )
                    # Capture the raw end-of-epoch state before best-state restore.
                    # CPU snapshots keep the audit from retaining GPU allocations.
                    epoch_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    on_epoch_checkpoint(dict(event), epoch_state)
                if on_epoch_end is not None:
                    try:
                        on_epoch_end(event)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[progress warning] epoch callback failed: {exc}", flush=True)

            if val_loader is not None:
                auc_text = "null" if task_val_auc is None else f"{task_val_auc:.4f}"
                print(
                    f"[epoch {epoch + 1}/{epoch_limit}] train_loss={train_loss:.4f} "
                    f"task_val_loss={task_val_loss:.4f} "
                    f"task_val_auc={auc_text} "
                    f"objective_val_loss={objective_val_loss:.4f} phase={phase}",
                    flush=True,
                )
                if selection_active and patience_left <= 0 and not fixed_research_budget:
                    print(
                        f"[early stop] val_loss {self.cfg.early_stop_patience} 轮未改善，停止。",
                        flush=True,
                    )
                    break
            else:
                print(f"[epoch {epoch + 1}/{epoch_limit}] train_loss={train_loss:.4f}", flush=True)

        self._restore_selected_states()
        bn_recalibration_batches = self._recalibrate_batch_norm(train_loader)
        final_task_auc = self._evaluate_auc(val_loader) if val_loader is not None else None

        empty_cache(self.device)
        history = {
            "train_losses": train_losses,
            "train_loss_components": train_loss_components,
            "val_losses": val_losses,
            "val_objective_losses": val_objective_losses,
            "val_innovation_nlls": val_innovation_nlls,
            "task_val_aucs": task_val_aucs,
            "final_task_val_auc": final_task_auc,
            "selection_metric": "target_bce_plus_weighted_digit_ce",
            "variance_nll_weight": self._variance_nll_weight_history,
            "innovation_covariance_weight": self._innovation_covariance_weight_history,
            "phases": phases,
            "selection_epochs": selection_epochs,
            "optimizer_steps": int(self.optimizer_steps),
            "planned_optimizer_steps": int(self.planned_optimizer_steps),
            "epoch_update_counts": epoch_update_counts,
            "learning_rates": self._lr_history,
            "lr_schedule": self.cfg.lr_schedule,
            "bn_recalibration_batches": int(bn_recalibration_batches),
            "best_epoch": self.best_epoch,
            "best_task_epoch": self.best_task_epoch,
            "best_density_epoch": self.best_density_epoch,
            "best_task_val_loss": (
                float(self.best_task_val_loss)
                if self.best_task_val_loss != float("inf")
                else None
            ),
            "best_density_nll": (
                float(self.best_density_nll)
                if self.best_density_nll != float("inf")
                else None
            ),
            "task_patience_exhausted": bool(task_patience_exhausted),
            "configured_epochs": self.cfg.epochs,
            "effective_epoch_limit": self._required_epoch_count(),
            "epoch_trajectory_audit": bool(self.cfg.epoch_trajectory_audit),
        }
        if self.reconstruction_profile is not None:
            profile = self.reconstruction_profile
            valid_target_variance = profile.evoked_target_variance[profile.channel_mask][
                :, profile.time_mask > 0.5
            ]
            history["reconstruction_profile"] = {
                "bands_hz": [list(b) for b in profile.bands_hz],
                "scales": profile.band_scales.detach().cpu().tolist(),
                "weights": profile.band_weights.detach().cpu().tolist(),
                "evoked_snr": profile.evoked_snr.detach().cpu().tolist(),
                "target_rate": float(profile.target_rate.detach().cpu()),
                "target_variance_mean": float(valid_target_variance.mean().detach().cpu()),
                "target_variance_max": float(valid_target_variance.max().detach().cpu()),
                "bootstrap_samples": profile.bootstrap_samples,
                "split_half_repeats": profile.split_half_repeats,
                "split_half_correlation": profile.split_half_correlation,
                "split_half_nrmse": profile.split_half_nrmse,
                "source_n_trials": profile.source_n_trials,
                "scope": profile.scope,
            }
        if self.generative_profile is not None:
            profile = self.generative_profile
            history["generative_profile"] = {
                "target_rate": float(profile.target_rate.detach().cpu()),
                "score_time_samples": int((profile.score_time_mask > 0.0).sum().cpu()),
                "source_n_trials": profile.source_n_trials,
                "scope": profile.scope,
                "ar_frobenius_norm": float(
                    torch.linalg.vector_norm(profile.ar_coefficients).detach().cpu()
                ),
                "ar_order": int(profile.ar_coefficients.shape[0]),
                "static_low_rank_factor_norm": (
                    float(profile.ar_low_rank_factor.norm().detach().cpu())
                    if profile.ar_low_rank_factor is not None
                    else None
                ),
            }
        if self.cfg.track_pcw_gradients:
            history["pcw_gradient_diagnostics"] = {
                "tau_gradient_norms": self._pcw_tau_grad_norms,
                "head_gradient_norms": self._pcw_head_grad_norms,
                "pcw_classifier_gradient_norms": self._pcw_classifier_grad_norms,
                "pcw_path_gradient_norms": self._pcw_path_grad_norms,
                "innovation_path_gradient_norms": self._innovation_path_grad_norms,
                "tau0_before_ms": self._pcw_tau0_initial.tolist(),
                "tau0_after_ms": self.model.component_window.tau0_bounded.detach().cpu().tolist(),
            }
        return history
