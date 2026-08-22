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

from dataclasses import dataclass
from typing import Optional

import torch

from models.n2p3net import N2P3Net
from train.augment import apply_augmentations, known_time_shift
from train.device import (
    empty_cache,
    get_device,
    optimize_device_for_training,
    print_device_memory,
)
from train.losses import compute_losses


@dataclass
class TrainerConfig:
    """训练配置（DP5：batch_size 等经配置传入，不写死）。"""

    epochs: int = 50
    batch_size: int = 32
    accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda2: float = 0.3
    lambda3: float = 1e-2
    lambda_amp: float = 1e-2
    lambda_jit: float = 0.0
    jit_max_ms: float = 40.0
    jit_prob: float = 0.0
    lambda4: float = 0.0
    mmd_bandwidth: Optional[float] = None
    main_domain: int = 0
    aux_domain: int = 1
    pos_weight: float = 8.0
    tau_scale_ms: float = 50.0
    early_stop_patience: int = 10
    augment: bool = True
    seed: int = 0


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
        E_chn: Optional[torch.Tensor] = None,
        E_sub: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        if not 0.0 <= config.jit_prob <= 1.0:
            raise ValueError(f"jit_prob 须在 [0,1]，得到 {config.jit_prob}。")
        self.device = device if device is not None else get_device()
        self.use_amp = self.device.type in ("cuda", "xpu")  # DP4

        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        optimize_device_for_training(self.device)  # D-device-tune
        self.model = model.to(self.device)  # DP3
        self.E_chn = E_chn.to(self.device) if E_chn is not None else None
        self.E_sub = E_sub.to(self.device) if E_sub is not None else None
        self.channel_mask = channel_mask.to(self.device) if channel_mask is not None else None

        # review v6 P1：τ0 是生理先验中心，不参与 AdamW weight decay（否则会被缓慢拉向 0）。
        decay_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "tau0" not in n
        ]
        no_decay_params = [
            p for n, p in model.named_parameters() if p.requires_grad and "tau0" in n
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.lr,
            fused=self.device.type == "cuda",
        )
        # 确保初始 τ0 也在生理界内
        self.model.component_window.clamp_tau0_()
        self.best_val_loss = float("inf")
        self.best_state = None
        # P9 梯度累积：当前 accum 周期内是否已有任何有效 backward。
        self._accum_has_grad = False

        print_device_memory(self.device)

    def _autocast_ctx(self):
        return torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        )

    def _train_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        step: int,
        domain_id: Optional[torch.Tensor] = None,
    ):
        """单步训练：增强 + 前向（AMP）+ 反向 + 梯度累积（D-accum）。

        domain_id : (B,) long 可选。Phase 3 辅助域对齐时使用；compute_losses 会按
        P9 只对 main_domain 样本计算 L_target/L_early/L_amp。
        """
        non_blocking = self.device.type == "cuda" and X.is_pinned()
        X = X.to(self.device, non_blocking=non_blocking)  # DP3 + pinned H2D overlap
        y = y.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
        if domain_id is not None:
            domain_id = domain_id.to(self.device, dtype=torch.long, non_blocking=non_blocking)

        if self.cfg.augment:
            # channel_mask 透传：reference_jitter/gaussian_noise 不得污染缺失通道（review v6 P0-2）
            X = apply_augmentations(X, channel_mask=self.channel_mask)

        # 3 导子集时 Pz 是最后一个通道（Fz/Cz/Pz）；标准 8 导时 Pz 是索引 3。
        pz_channel = 2 if X.shape[1] == 3 else 3

        with self._autocast_ctx():
            # return_attention=True 供 Head-D 物理幅值损失使用（review v6 P1）
            output = self.model(
                X,
                self.E_chn,
                self.E_sub,
                channel_mask=self.channel_mask,
                domain_id=domain_id,
                return_attention=True,
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
                    channel_mask=self.channel_mask,
                    domain_id=domain_id,
                    return_attention=False,
                      return_heads=False,
                )
                tau_shift = output_shift.tau
                shift_ms = shift_samples.float() * (1000.0 / sfreq)

            losses = compute_losses(
                output,
                self.model.component_window.tau0_bounded,
                y,
                lambda2=self.cfg.lambda2,
                lambda3=self.cfg.lambda3,
                lambda_amp=self.cfg.lambda_amp,
                lambda_jit=self.cfg.lambda_jit,
                tau_shift=tau_shift,
                shift_ms=shift_ms,
                lambda4=self.cfg.lambda4,
                mmd_bandwidth=self.cfg.mmd_bandwidth,
                main_domain=self.cfg.main_domain,
                aux_domain=self.cfg.aux_domain,
                pos_weight=self.cfg.pos_weight,
                tau_scale_ms=self.cfg.tau_scale_ms,
                z_features=output.features,
                domain_ids=domain_id,
                X=X,
                pz_channel=pz_channel,
            )
            loss = losses.total / self.cfg.accum_steps

        # P9：纯辅助域 batch 的总损失可为零且无计算图；跳过 backward。但若本累积周期
        # 已有前面 GTN batch 累积的梯度，到边界仍必须 step，否则会丢梯度（audit P1-4）。
        if loss.requires_grad:
            loss.backward()
            self._accum_has_grad = True
        if (step + 1) % self.cfg.accum_steps == 0:
            if self._accum_has_grad:
                self.optimizer.step()
                self.model.component_window.clamp_tau0_()  # review v6 P1：τ0 保持生理界内
            self.optimizer.zero_grad(set_to_none=True)  # D-zero-grad-none：释放梯度张量
            self._accum_has_grad = False
        # 返回设备端 0 维张量而非 float：fit 在 epoch 结束时只做一次 .item()，
        # 避免每个 batch 都触发 GPU→CPU 同步（D-loss-sync）。
        return losses.total.detach()

    @torch.inference_mode()
    def _evaluate(self, loader) -> float:
        """评估：平均 val loss（无增强、eval 模式；D-loss-sync 的评估端）。"""
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for batch in loader:
            X, y = batch[0], batch[1]
            domain_id = batch[2] if len(batch) == 3 else None
            non_blocking = self.device.type == "cuda" and X.is_pinned()
            X = X.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, dtype=torch.float32, non_blocking=non_blocking)
            if domain_id is not None:
                domain_id = domain_id.to(self.device, dtype=torch.long, non_blocking=non_blocking)
            pz_channel = 2 if X.shape[1] == 3 else 3
            with self._autocast_ctx():
                output = self.model(
                    X,
                    self.E_chn,
                    self.E_sub,
                    channel_mask=self.channel_mask,
                    domain_id=domain_id,
                    return_attention=True,
                )
                losses = compute_losses(
                    output,
                    self.model.component_window.tau0_bounded,
                    y,
                    lambda2=self.cfg.lambda2,
                    lambda3=self.cfg.lambda3,
                    lambda_amp=self.cfg.lambda_amp,
                    lambda4=self.cfg.lambda4,
                    mmd_bandwidth=self.cfg.mmd_bandwidth,
                    main_domain=self.cfg.main_domain,
                    aux_domain=self.cfg.aux_domain,
                    pos_weight=self.cfg.pos_weight,
                    tau_scale_ms=self.cfg.tau_scale_ms,
                    z_features=output.features,
                    domain_ids=domain_id,
                    X=X,
                    pz_channel=pz_channel,
                )
            total += losses.total.detach().float() * X.shape[0]
            n += X.shape[0]
        self.model.train()
        return (total / max(n, 1)).item()

    def fit(self, train_loader, val_loader=None) -> dict:
        """训练主循环：epoch 迭代 + early stop + OOM 保护（D-oom）。

        train_loader/val_loader 提供 (X, y) 或 (X, y, domain_id) batch。返回
          {train_losses, val_losses}。domain_id 用于 Phase 3 P9 域隔离。
        """
        train_losses: list[float] = []
        val_losses: list[float] = []
        patience_left = self.cfg.early_stop_patience

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            n_batches = 0
            try:
                for step, batch in enumerate(train_loader):
                    X, y = batch[0], batch[1]
                    domain_id = batch[2] if len(batch) == 3 else None
                    epoch_loss += self._train_step(X, y, step, domain_id=domain_id).float()
                    n_batches += 1
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
            if self.cfg.accum_steps > 1 and n_batches % self.cfg.accum_steps != 0:
                if self._accum_has_grad:
                    self.optimizer.step()
                    self.model.component_window.clamp_tau0_()  # review v6 P1
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_has_grad = False

            # D-loss-sync：整个 epoch 只在结束时做一次设备→主机同步
            train_loss = (epoch_loss / max(n_batches, 1)).item()
            train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                val_losses.append(val_loss)
                if val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.cfg.early_stop_patience
                else:
                    patience_left -= 1
                print(f"[epoch {epoch + 1}/{self.cfg.epochs}] train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f}")
                if patience_left <= 0:
                    print(f"[early stop] val_loss {self.cfg.early_stop_patience} 轮未改善，停止。")
                    break
            else:
                print(f"[epoch {epoch + 1}/{self.cfg.epochs}] train_loss={train_loss:.4f}")

        # 恢复最佳权重（若做过 early stop）
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        empty_cache(self.device)
        return {"train_losses": train_losses, "val_losses": val_losses}
