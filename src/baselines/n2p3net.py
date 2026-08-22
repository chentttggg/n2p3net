"""N2P3Net 的 Baseline 适配器：把 Trainer 训练出的模型接到 evaluate() 协议。

GTN-N2P3Net 闭环使用同一 evaluate()，因此 hit rate / bacc / AUC / scores 的
口径与 7 个基线完全一致。每个 fit 都重新构造 N2P3Net，避免 LOSO fold 间权重泄漏。

GLM（2026-08-22）：被试级验证早停协议。evaluate() 会把训练 fold 的 subject_ids
传给本适配器（fit_accepts_subject_ids = True）；fit 按被试分组切出验证集，
Trainer 在 val loss 不再改善时早停并恢复最佳权重。动机（失败诊断 §2.4）：
固定 10ep 是「欠拟合赌博」——一折逐 epoch 曲线显示 held-out 指标在 epoch 10–11
见顶后崩塌（30ep bacc 0.68→0.55），无验证协议的固定 epoch 要么停在峰值前、
要么冲进过拟合区。被试级（而非试次级随机）切分保证同被试试次不跨 train/val，
验证损失才能诚实反映跨被试泛化。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from baselines.classic import Baseline
from models.n2p3net import N2P3Net
from train.device import get_device
from train.preloaded import PreloadedDataLoader
from train.trainer import Trainer, TrainerConfig


@dataclass
class N2P3NetBaselineConfig:
    """N2P3Net 进入 evaluate 协议的训练配置。"""

    model_kwargs: dict = field(default_factory=dict)
    trainer_kwargs: dict = field(default_factory=dict)


class N2P3NetBaseline(Baseline):
    """fit(X,y[,subject_ids]) / predict_logit(X) 适配器。

    Parameters
    ----------
    model_kwargs : dict
        透传给 N2P3Net 的构造参数。
    trainer_kwargs : dict
        透传给 TrainerConfig 的字段；epochs/batch_size/early_stop_patience 等在此设置。
    E_chn / channel_mask / device :
        与 Trainer 契约一致；GTN 3 导时使用 8 导零填充 mask。
    val_subject_frac : float | None
        被试级验证集比例（GLM 协议，默认 0.08）；None 关闭验证早停（旧固定 epoch 行为）。
    val_subjects_min / val_subjects_max : int
        验证被试数的下/上限（小池保护：太少验证信号不稳；太多挤占训练数据）。
    """

    # evaluate() 检测此属性决定是否传 subject_ids（GLM：被试级验证早停）。
    fit_accepts_subject_ids = True

    def __init__(
        self,
        model_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        E_chn: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        val_subject_frac: Optional[float] = 0.08,
        val_subjects_min: int = 2,
        val_subjects_max: int = 12,
    ):
        self.model_kwargs = dict(model_kwargs or {})
        self.trainer_kwargs = dict(trainer_kwargs or {})
        self.device = device if device is not None else get_device()
        self.E_chn = E_chn.to(self.device) if E_chn is not None else None
        self.channel_mask = channel_mask.to(self.device) if channel_mask is not None else None
        self.val_subject_frac = val_subject_frac
        self.val_subjects_min = int(val_subjects_min)
        self.val_subjects_max = int(val_subjects_max)
        self._fitted = False
        # 实验记录：每个 fold 的训练耗时 / 显存峰值 / 最后一个 fold 的 history。
        self.fit_durations: list[float] = []
        self.fit_peak_memory_mb: list[float] = []
        self.last_history: Optional[dict] = None
        # GLM：每 fold 实际用到的训练/验证被试数（记录协议是否生效）。
        self.last_val_subjects: Optional[int] = None

    def _split_val_subjects(
        self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """按被试分组切验证集；返回 (X_train, y_train, X_val, y_val, n_val_subjects)。

        验证被试由 cfg.seed 决定（确定性）；被试数 = clamp(round(frac·N), min, max)。
        训练被试数 < 4 时不足以分组，返回整集训练（n_val_subjects=0）。
        """
        cfg_seed = int(self.trainer_kwargs.get("seed", 0))
        uniq = np.unique(subject_ids)
        if self.val_subject_frac is None or len(uniq) < 4:
            return X, y, X[:0], y[:0], 0
        k = int(round(self.val_subject_frac * len(uniq)))
        k = max(self.val_subjects_min, min(self.val_subjects_max, k))
        k = min(k, len(uniq) - 2)  # 训练侧至少保留 2 名被试
        rng = np.random.default_rng(cfg_seed)
        val_subjects = set(rng.choice(uniq, size=k, replace=False).tolist())
        val_mask = np.isin(subject_ids, list(val_subjects))
        return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask], k

    def _fit_common(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "N2P3NetBaseline":
        """构造 model/trainer/loader 并跑 fit（fit 主路径与域对齐适配器共用）。

        X_val/y_val 给定时启用验证早停（Trainer 恢复 val loss 最佳权重）。
        """
        model = N2P3Net(**self.model_kwargs)
        cfg = TrainerConfig(**self.trainer_kwargs)
        trainer = Trainer(
            model,
            cfg,
            E_chn=self.E_chn,
            channel_mask=self.channel_mask,
            device=self.device,
        )

        loader = PreloadedDataLoader(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed,
            device=self.device,
        )
        val_loader = None
        if X_val is not None and len(X_val) > 0:
            val_loader = PreloadedDataLoader(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val),
                batch_size=cfg.batch_size,
                shuffle=False,
                seed=cfg.seed,
                device=self.device,
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        history = trainer.fit(loader, val_loader=val_loader)
        self.fit_durations.append(time.perf_counter() - t0)
        if self.device.type == "cuda":
            self.fit_peak_memory_mb.append(torch.cuda.max_memory_allocated() / 1e6)
        else:
            self.fit_peak_memory_mb.append(float("nan"))
        self.last_history = history

        self.model_ = model
        self._fitted = True
        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, subject_ids: Optional[np.ndarray] = None
    ) -> "N2P3NetBaseline":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")
        if subject_ids is None:
            self.last_val_subjects = None
            return self._fit_common(X, y)
        subject_ids = np.asarray(subject_ids)
        X_train, y_train, X_val, y_val, k = self._split_val_subjects(X, y, subject_ids)
        self.last_val_subjects = int(k)
        return self._fit_common(X_train, y_train, X_val, y_val)

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_logit。")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"X 须为 (N,C,T)，得到 {X.shape}。")

        Xt = torch.from_numpy(X).to(self.device)
        chunk_size = 256
        self.model_.eval()
        out: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, Xt.shape[0], chunk_size):
                xb = Xt[i : i + chunk_size]
                output = self.model_(xb, E_chn=self.E_chn, channel_mask=self.channel_mask)
                if output.heads is None:
                    raise RuntimeError("模型 forward 未返回 heads；请勿设置 return_heads=False。")
                out.append(output.heads.logit_target.float().cpu())
        return torch.cat(out).squeeze(-1).numpy().astype(np.float64)

    def predict_full(self, X: np.ndarray):
        """返回 (logits, tau, sigma)，用于 Phase 2 成分记录。

        只对最后一个 fit 的模型生效；供实验记录使用，不作为 evaluate 主路径。
        """
        if not self._fitted:
            raise RuntimeError("请先 fit 再 predict_full。")
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.from_numpy(X).to(self.device)
        logits_list, tau_list, sigma_list = [], [], []
        self.model_.eval()
        with torch.inference_mode():
            for i in range(0, Xt.shape[0], 256):
                xb = Xt[i : i + 256]
                output = self.model_(xb, E_chn=self.E_chn, channel_mask=self.channel_mask)
                if output.heads is None:
                    raise RuntimeError("模型 forward 未返回 heads。")
                logits_list.append(output.heads.logit_target.float().cpu())
                tau_list.append(output.tau.float().cpu())
                sigma_list.append(output.sigma.float().cpu())
        logits = torch.cat(logits_list).squeeze(-1).numpy().astype(np.float64)
        tau = torch.cat(tau_list).numpy()
        sigma = sigma_list[0].numpy() if sigma_list else np.zeros((3, 2))
        return logits, tau, sigma


class N2P3NetDomainBaseline(N2P3NetBaseline):
    """transfer_policy 方式 B（T3）：辅助域只进域条件仿射 + RBF-MMD，GTN 独占分类头。

    每个 fold 把 GTN 训练集与辅助域试次拼成 (X, y, domain_id) 流：
    domain_id 0 = GTN（main_domain），1 = 辅助域（aux_domain）。
    Trainer/compute_losses 按 P9 只对 main_domain 计算 L_target/L_early/L_amp/L_tau/L_jit；
    L_MMD 在 batch 内两个域之间计算。评估时 predict_logit 仍只吃 GTN 测试 fold。
    """

    def __init__(
        self,
        *args,
        aux_X: Optional[np.ndarray] = None,
        aux_y: Optional[np.ndarray] = None,
        lambda4: float = 0.1,
        mmd_bandwidth: Optional[float] = 5.0,
        aux_subsample: Optional[int] = None,
        aux_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_X = np.asarray(aux_X, dtype=np.float32) if aux_X is not None else None
        self.aux_y = np.asarray(aux_y, dtype=np.int64) if aux_y is not None else None
        self.lambda4 = float(lambda4)
        self.mmd_bandwidth = mmd_bandwidth
        self.aux_subsample = aux_subsample
        self.aux_seed = int(aux_seed)
        if self.aux_X is not None:
            if self.aux_y is None or len(self.aux_X) != len(self.aux_y):
                raise ValueError("aux_X/aux_y 长度须一致。")
            if self.aux_X.ndim != 3:
                raise ValueError(f"aux_X 须为 (N,C,T)，得到 {self.aux_X.shape}。")
            if self.aux_subsample is not None and self.aux_subsample < len(self.aux_X):
                rng = np.random.default_rng(self.aux_seed)
                idx = rng.choice(len(self.aux_X), size=self.aux_subsample, replace=False)
                self.aux_X = self.aux_X[idx]
                self.aux_y = self.aux_y[idx]

    def fit(
        self, X: np.ndarray, y: np.ndarray, subject_ids: Optional[np.ndarray] = None
    ) -> "N2P3NetDomainBaseline":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if self.aux_X is None:
            super().fit(X, y, subject_ids=subject_ids)
            return self
        if X.shape[1:] != self.aux_X.shape[1:]:
            raise ValueError(
                f"GTN X 与 aux_X 的 (C,T) 须一致，得到 {X.shape[1:]} vs {self.aux_X.shape[1:]}。"
            )

        # GLM：GTN 训练集按被试分组切验证集（验证只含 GTN 试次，早停信号干净）。
        if subject_ids is not None:
            X_train, y_train, X_val, y_val, k = self._split_val_subjects(
                X, y, np.asarray(subject_ids)
            )
            self.last_val_subjects = int(k)
        else:
            X_train, y_train, X_val, y_val = X, y, X[:0], y[:0]
            self.last_val_subjects = None

        model_kwargs = dict(self.model_kwargs)
        model_kwargs.setdefault("n_domains", 2)
        model = N2P3Net(**model_kwargs)
        cfg = TrainerConfig(**self.trainer_kwargs)
        cfg.lambda4 = self.lambda4
        cfg.mmd_bandwidth = self.mmd_bandwidth
        trainer = Trainer(
            model,
            cfg,
            E_chn=self.E_chn,
            channel_mask=self.channel_mask,
            device=self.device,
        )

        Xt = torch.from_numpy(X_train)
        yt = torch.from_numpy(y_train)
        aux_t = torch.from_numpy(self.aux_X)
        aux_yt = torch.zeros(len(aux_t), dtype=torch.int64)
        domain = torch.cat(
            [torch.zeros(len(Xt), dtype=torch.int64), torch.ones(len(aux_t), dtype=torch.int64)]
        )
        loader = PreloadedDataLoader(
            torch.cat([Xt, aux_t]),
            torch.cat([yt, aux_yt]),
            domain_id=domain,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed,
            device=self.device,
        )
        val_loader = None
        if len(X_val) > 0:
            val_loader = PreloadedDataLoader(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val),
                batch_size=cfg.batch_size,
                shuffle=False,
                seed=cfg.seed,
                device=self.device,
            )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        history = trainer.fit(loader, val_loader=val_loader)
        self.fit_durations.append(time.perf_counter() - t0)
        if self.device.type == "cuda":
            self.fit_peak_memory_mb.append(torch.cuda.max_memory_allocated() / 1e6)
        else:
            self.fit_peak_memory_mb.append(float("nan"))
        self.last_history = history

        self.model_ = model
        self._fitted = True
        return self
