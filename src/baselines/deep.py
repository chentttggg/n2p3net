"""模块：深度基线（Deep Baselines）。

职责（roadmap Phase 1 + constitution P8）：
    EEGNet / EEG-Inception(ERP) / EEG Conformer 三个深度基线，统一包装成 Baseline 接口
    （fit(X,y) / predict_logit(X)→(N,)），与 classic.py / riemann.py 同契约，供 evaluate.py
    在完全相同的三层协议下公平比较。输出逐试次 target 判别 logit（供 decision 层累加）。

明确「不做」：
    - 不复用 train/trainer.py（它是 N2P3-Net 专用：吃 N2P3NetOutput + compute_losses 多任务
      损失）。深度基线是「输入 (N,C,T) → 二分类 logits」的单任务模型，用一个自包含的轻量
      训练循环更直接、无耦合。
    - 不做超参精调（早停/λ 网格）——基线复现用固定 epoch + Adam + pos_weight，保证可复现；
      精调留给 Phase 2 的 N2P3-Net 本身。

三思决策记录（供后续会话追溯）：
    D-deep-ce        用 n_outputs=2 + CrossEntropyLoss(weight=[1, pos_weight])，而非 n_outputs=1 +
                     BCE。理由：braindecode 三个模型统一支持 n_outputs=2（实测输出 (N,2) logits），
                     且 CrossEntropyLoss 的 weight 是 pos_weight 的最干净等价（加重 target 类）。
                     predict_logit 返回 logits[:,1]−logits[:,0]，即 target/non-target 的 log-odds，
                     与 heads 输出 logit 的语义一致（decision 层对数似然比累积）。
    D-deep-shape     braindecode 模型输入约定 (N, C, T)（channels-first），与 data 层输出一致，直接
                     喂无需转置。实测三个模型 n_outputs=2 均输出 (N,2)。
    D-deep-param     实测参数量：EEGNet 1,490 / EEGInceptionERP 26,622 / EEGConformer 255,106。
                     EEGConformer 远超 E4 的 50k——但 E4 约束的是 N2P3-Net 本体，深度基线是「对照
                     坐标系」：复现一个高容量基线恰好验证 D6「容量非瓶颈」的对照意义（若 Conformer
                     在数千试次上过拟合、反而不如 EEGNet，正是 D6 的实证）。docstring 如实标注。
    D-deep-amp       AMP 用 bf16，CUDA/XPU 启用、CPU 禁用（DP4），复用 train/device.get_device 与
                     device-portability §3 的 autocast 写法。CPU 测试时 enabled=False 等价 fp32。
    D-deep-cpu       Trainer/DeepBaseline 均接受可选 device（默认 get_device()），测试显式传 CPU
                     保证稳定与速度（device-portability 的 D-device-param 同款约定）。
    D-deep-seed      fit 起始 torch.manual_seed(seed) + 每 epoch shuffle 用 torch.randperm，保证
                     可复现。注意：deep fold 线程并行时 CUDA dropout 仍用进程级全局 RNG；
                      模型初始化与 shuffle 已确定化，但逐 bit 复现建议 --deep-jobs 1（audit P2）。
    D-deep-sfreq     EEGInceptionERP 的 scales_samples_s 是「秒」单位，需显式传 sfreq=256 与
                     n_times=256（默认 n_times=1000/sfreq=128，是 ~7.8s 窗口，与我们的 1s epoch
                     不符）；EEGNet 用 kernel_length(样本) 与 sfreq 无关；EEGConformer 无需 sfreq。
    D-deep-standard  训练前用训练集逐通道 mean/std 做 z-score（fit 内完成，predict 复用训练统计量）。
                     review v6 P1：V 单位输入（~1e-5–1e-4）会让 deep logit 坍缩成窄带
                     （实测 EEGNet AUC 0.744 但 bacc@0=0.500、命中率≈chance），z-score 后
                     AUC 0.797、top1 0.900。standardize_input=False 可作消融。

契约（输入 → 输出）：
    X ∈ R^{N×C×T}（float32，缺失通道须已填 0）+ y ∈ {0,1}^N → fit 后 predict_logit(X) ∈ R^N。

依赖的决策：roadmap Phase 1、constitution P8/D6、device-portability.md（DP1–DP6）、
    train/device.py（get_device）、baselines/classic.Baseline。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGConformer, EEGInceptionERP, EEGNet

from baselines.classic import Baseline
from train.device import get_device, optimize_device_for_training

# 模型名（lowercase）→ 构造类
_MODEL_FACTORIES = {
    "eegnet": EEGNet,
    "inception": EEGInceptionERP,
    "conformer": EEGConformer,
}

# deep fold 线程并行时，torch.manual_seed 与模型初始化必须串行化，否则两个线程
# 会互相踩全局 RNG，导致初始化不可复现（review v6 性能项）。
_INIT_LOCK = threading.Lock()


@dataclass
class DeepConfig:
    """深度基线训练配置（DP5：batch_size 等经配置传入，不写死）。"""

    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 8.0
    seed: int = 0
    standardize_input: bool = True


class DeepBaseline(Baseline):
    """深度基线统一包装（EEGNet / EEG-Inception / EEG Conformer）。

    Parameters
    ----------
    model_name : str
        "eegnet" / "inception" / "conformer"（大小写不敏感）。
    n_chans / n_times / sfreq : int / int / float
        通道数、时间点数、采样率（与 data 层契约一致：8 / 256 / 256）。
    config : DeepConfig | None
        训练配置（默认 DeepConfig()）。
    device : torch.device | None
        默认 get_device()；测试可显式传 CPU。
    """

    def __init__(
        self,
        model_name: str = "eegnet",
        n_chans: int = 8,
        n_times: int = 256,
        sfreq: float = 256.0,
        config: Optional[DeepConfig] = None,
        device: Optional[torch.device] = None,
        *,
        pretrained_state_dict: Optional[dict] = None,
        load_mapping: Optional[dict[str, Optional[str]]] = None,
        freeze_prefixes: Sequence[str] = (),
        strict_load: bool = False,
    ):
        key = model_name.lower()
        if key not in _MODEL_FACTORIES:
            raise ValueError(f"未知 model_name={model_name!r}，可选 {list(_MODEL_FACTORIES)}。")
        self.model_name = key
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.cfg = config if config is not None else DeepConfig()
        self.device = device if device is not None else get_device()
        self.use_amp = self.device.type in ("cuda", "xpu")  # DP4
        # P9 辅助预训练接口（transfer_policy 方式 A/C）：加载报告 + 层冻结/映射
        self.pretrained_state_dict = pretrained_state_dict
        self.load_mapping: dict[str, Optional[str]] = dict(load_mapping or {})
        self.freeze_prefixes = tuple(freeze_prefixes)
        self.strict_load = bool(strict_load)
        self.load_report: list[dict] = []
        self._fitted = False

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

    def _autocast_ctx(self):
        return torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        )

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
        src_keys = set(self.pretrained_state_dict.keys())
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
            loaded[dst_key] = src_value.to(
                device=dst_value.device, dtype=dst_value.dtype
            )
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

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
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
            "config": self.cfg,
            "model_state_dict": self.model_.state_dict(),
        }
        torch.save(payload, path)
        return path

    @staticmethod
    def load_state_dict_file(path: Union[str, Path]) -> dict:
        """读取 checkpoint 的 state_dict（供其他模型作为 pretrained_state_dict 传入）。"""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if "model_state_dict" in payload:
            return payload["model_state_dict"]
        return payload

    # ---------------- 训练 ----------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBaseline":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        if X.ndim != 3 or X.shape[1] != self.n_chans:
            raise ValueError(f"X 须为 (N,{self.n_chans},T)，得到 {X.shape}。")

        # 输入标准化（D-deep-standard）：统计量只来自训练 fold，predict 复用
        self._input_mean = X.mean(axis=(0, 2), keepdims=True)
        self._input_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        if self.cfg.standardize_input:
            X = (X - self._input_mean) / self._input_std

        # D-deep-seed：模型初始化/全局种子必须串行化，线程并行 fold 时避免互踩 RNG。
        with _INIT_LOCK:
            torch.manual_seed(self.cfg.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.cfg.seed)
            optimize_device_for_training(self.device)  # D-device-tune
            self.model_ = self._make_model().to(self.device)  # DP3
            # P9：先加载辅助预训练权重，再冻结指定前缀（transfer_policy 方式 A/C）。
            self._apply_pretrained_state_dict()
            self._freeze_layers()
            # 使用每 fold 私有的 CPU Generator，shuffle 顺序不依赖其他线程的 RNG 进度。
            perm_gen = torch.Generator()
            perm_gen.manual_seed(self.cfg.seed)
        trainable_params = [p for p in self.model_.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("freeze_prefixes 冻结了全部参数，没有可训练参数。")
        opt = torch.optim.Adam(
            trainable_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, self.cfg.pos_weight], device=self.device)
        )

        # D-deep-upload：一次性把训练张量放到目标设备，后续 batch 只做设备端索引切片，
        # 避免每个 batch 都从 CPU 做 H2D 拷贝。N2P3-Net 数据量（≤数千试次 × 8 × 256）远
        # 小于显存上限；若未来数据过大，可回退为逐 batch 上传。
        Xt = torch.from_numpy(X).to(self.device)  # (N, C, T)
        yt = torch.from_numpy(y).to(self.device)  # (N,)
        n = Xt.shape[0]

        try:
            for _ in range(self.cfg.epochs):
                self.model_.train()
                perm = torch.randperm(n, generator=perm_gen).to(self.device)
                for i in range(0, n, self.cfg.batch_size):
                    idx = perm[i : i + self.cfg.batch_size]
                    with self._autocast_ctx():
                        logits = self.model_(Xt[idx])
                        loss = loss_fn(logits, yt[idx])
                    opt.zero_grad(set_to_none=True)  # 释放梯度张量，减少显存碎片
                    loss.backward()
                    opt.step()
        except torch.OutOfMemoryError:  # DP6
            raise RuntimeError(
                f"显存溢出（OOM）：请减小 batch_size（当前 {self.cfg.batch_size}）后重试。"
            ) from None
        except RuntimeError as e:  # 旧版兜底
            if "out of memory" in str(e).lower():
                raise RuntimeError("显存溢出（OOM）：请减小 batch_size 后重试。") from None
            raise

        self._fitted = True
        return self

    # ---------------- 预测 ----------------

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_logit。")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1] != self.n_chans:
            raise ValueError(f"X 须为 (N,{self.n_chans},T)，得到 {X.shape}。")
        if X.shape[2] != self.n_times:
            raise ValueError(
                f"X 时间点数 {X.shape[2]} 与模型契约 n_times={self.n_times} 不一致。"
            )
        if self.cfg.standardize_input:
            X = (X - self._input_mean) / self._input_std

        # D-deep-predict-chunks：分块前向，既避免大测试集一次性占满显存，也避免在
        # GPU 上拼接全部 logits；每块只把 log-odds 转 float32 后立即搬回 CPU。
        Xt = torch.from_numpy(X)
        chunk_size = max(self.cfg.batch_size * 4, 256)
        self.model_.eval()
        out_chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, X.shape[0], chunk_size):
                xb = Xt[i : i + chunk_size].to(self.device)
                with self._autocast_ctx():
                    logits = self.model_(xb)  # (chunk, 2)
                # log-odds（target vs non-target），D-deep-ce
                # AMP 下 logits 可能是 bfloat16，先转 float32 再回 CPU（numpy 不支持 bfloat16）
                out_chunks.append((logits[:, 1] - logits[:, 0]).float().cpu())
        return torch.cat(out_chunks).numpy().astype(np.float64)
