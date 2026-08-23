"""GTN → N2P3Net 闭环训练/评估入口。

用法：
    # microbenchmark：只训一个 fold，测量秒/epoch 与显存
    .venv/Scripts/python.exe experiments/run_n2p3net_gtn.py --benchmark --subjects 3 --epochs 1 --batch-size 256

    # 小规模 smoke
    .venv/Scripts/python.exe experiments/run_n2p3net_gtn.py --subjects 5 --epochs 3 --batch-size 256

    # 全量 LOSO（正式 Phase 2 前必须先用 benchmark 外推时间）
    .venv/Scripts/python.exe experiments/run_n2p3net_gtn.py --epochs 10 --batch-size 512 --save-scores-dir experiments/results

GTN 为 3 导。v5.1 起默认用原生 3 导（--n-channels 3）；--n-channels 8 显式恢复
旧版零填充 8 导。E_chn 使用对应蒙太奇坐标身份。

GLM v2（2026-08-23，门控参考层）默认配置：
    epochs=30 + 被试级验证早停（val_subject_frac=0.08、patience=6）——
    失败诊断 §2.4 显示 held-out 指标在 epoch 10–11 见顶后崩塌，固定 10ep 是
    「欠拟合赌博」；早停锁定每 fold 的 val 峰值而非全局赌一个 epoch 数。
    **rereference 默认开启（门控参考层）**：out = X − g⊙(1·wᵀX)，w=softmax、
    g 自由线性门 init=0 → 恒等起步。旧版强制 CAR 已废弃（3 导销毁 P3b 4.36×，
    CAR 本身在 <32 导无效——Junghöfer 2001/Luck 2014）。60 被实测：开启
    hit .9000（全系最高）vs 关闭 .8833，AUC 等价；网络自学 gate≈[0.20,0.13,0.15]。
    --no-rereference 关闭。
    **encoder-norm 默认 bn**（2026-08-23）：三组实测 BN 一致优于 LN +0.5~0.9pt
    AUC；--encoder-norm ln 回退。
    60 被实测：门控开启 hit 0.9000 / AUC 0.7536；关闭 hit 0.8833 / AUC 0.7575
    （均超 EEGNet-60 的 hit 0.8333，AUC 差距 0.054→~0.015）。
    P3b τ0=460ms/界[350,600]、σ 上界 150ms（儿童 P3b 宽达 300–650ms，ERP 实测）、
    dtau_readout=attention_softargmax、lambda_jit=0、bypass_mode=separable_pool、
    head Dropout+Linear、spatial max-norm=1。
    --model-size mini/mini_a 容量预设（~7.3k/~2.5k 参数；实测容量砍 15 倍 AUC 不降，
    容量非瓶颈；去参考后小模型也大涨但 60 被上 default 略优）。
回退 2026-08-22 v5.1 行为：--n-channels 8 --bypass-mode mean_pool --head-mlp
    --no-spatial-max-norm --encoder-dropout 0.1 --no-val-early-stop --epochs 10
    --p3b-sigma-hi 80 --no-rereference --encoder-norm ln。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.evaluate import evaluate, loso_folds  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline, N2P3NetDomainBaseline  # noqa: E402
from data.auxiliary import load_auxiliary  # noqa: E402
from data.channel import build_channel_identity  # noqa: E402
from data.preprocess import STANDARD_CHANNELS  # noqa: E402
from experiments.run_gtn_baseline import (  # noqa: E402
    GTN_STANDARD,
    _gtn_cache_filename,
    _load_gtn_cache,
    save_subject_scores,
)

GTN_MASK_8 = (True, True, True, False, False, False, False, False)

# T3 辅助域通道别名 → 标准 8 导蒙太奇（bi2014a 无 Fz，用 AFz 顶替 Fz 位置）。
_AUX_TO_STANDARD = {"AFz": "Fz"}


def _build_gtn_x_and_identity(
    X3: np.ndarray, n_channels: int = 3
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """GTN 3 导 → 原生 3 导（默认）或 8 导零填充（回退口径）。"""
    if n_channels == 3:
        X = np.asarray(X3, dtype=np.float32)
        mask = torch.tensor((True, True, True), dtype=torch.bool)
        ch_names = list(GTN_STANDARD)
    elif n_channels == 8:
        X = np.zeros((X3.shape[0], 8, X3.shape[2]), dtype=np.float32)
        X[:, :3, :] = X3
        mask = torch.tensor(GTN_MASK_8, dtype=torch.bool)
        ch_names = list(STANDARD_CHANNELS)
    else:
        raise ValueError(f"n_channels 仅支持 3（原生 GTN）或 8（旧零填充），得到 {n_channels}。")
    identity = build_channel_identity(ch_names=ch_names, channel_mask=tuple(mask.tolist()))
    E_chn = torch.from_numpy(identity.embedding)
    return X, E_chn, mask


def main() -> None:
    ap = argparse.ArgumentParser(description="GTN → N2P3Net 闭环")
    ap.add_argument("--subjects", type=int, default=None, help="限缩前 N 名被试")
    ap.add_argument("--epochs", type=int, default=30,
                    help="GLM：epoch 上限（被试级验证早停保证不冲进过拟合区；旧固定 10ep 用 --no-val-early-stop --epochs 10 复现）")
    ap.add_argument("--early-stop-patience", type=int, default=6,
                    help="验证损失连续 N epoch 不改善即停（GLM 协议）")
    ap.add_argument("--no-val-early-stop", action="store_true",
                    help="关闭被试级验证早停（回退旧固定 epoch 行为）")
    ap.add_argument("--val-subject-frac", type=float, default=0.08,
                    help="每 fold 训练被试中留作验证的比例（GLM 协议）")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda2", type=float, default=0.3)
    ap.add_argument("--lambda3", type=float, default=1e-2)
    ap.add_argument("--lambda-amp", type=float, default=1e-2)
    ap.add_argument("--lambda-jit", type=float, default=0.0,
                    help="自监督 jitter 一致性默认关闭（方案 B）")
    ap.add_argument("--jit-prob", type=float, default=0.0)
    ap.add_argument("--jit-max-ms", type=float, default=40.0)
    ap.add_argument("--encoder-depth", type=int, default=3)
    ap.add_argument("--encoder-type", default="tcn", choices=("tcn", "conformer"))
    ap.add_argument("--encoder-norm", default="bn", choices=("ln", "bn"),
                    help="TCN block 归一化（GLM 消融轴）。默认 bn：三组实测（12/60 被试，"
                         "含/不含再参考）BN 一致优于 LN +0.5~0.9pt AUC；ln=旧默认回退")
    ap.add_argument("--tokenizer-init", default="random", choices=("random", "bandpass"),
                    help="GLM v3：时间卷积初始化。random=kaiming（旧默认）；bandpass=Gabor "
                         "带通（诊断证据：随机 init 的 FIR 频谱中心 ~60Hz 且训练后几乎不动，"
                         "从未学出 ERP 形状；文献：FBCNet 滤波器组/Sinc-ShallowNet 带通）。"
                         "核长分层分配频带，k=129 占据 P3b δ-θ 带 [1.5,7]Hz")
    ap.add_argument("--tokenizer-post-norm", default="none", choices=("none", "bn"),
                    help="GLM v3：每尺度时间卷积后 BatchNorm1d（EEG-Inception/ATCNet 标准 "
                         "结构；修 4× 尺度幅值失衡 + 提供非线性位点，防多尺度线性塌缩）")
    ap.add_argument("--tokenizer-post-act", default="none", choices=("none", "elu", "gelu"),
                    help="GLM v3：时间卷积后激活（ELU 保负电位，EEG 文献论点）。"
                         "默认 none=旧行为")
    ap.add_argument("--model-size", default="default", choices=("default", "mini", "mini_a"),
                    help="GLM：mini=d_model32/4滤波器×2尺度/depth1（~7.3k 参数），"
                         "mini_a=d_model16/2滤波器×1尺度/depth0（~2.5k 参数）；"
                         "default=旧 38.5k。实测容量砍 15 倍 AUC 不降（容量非瓶颈），"
                         "且去参考后 mini 泛化更好")
    ap.add_argument("--rereference", dest="use_rereference",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="GLM v2：门控参考层（gate init=0 → 恒等起步，训练自证开合），"
                         "默认开启。60 被实测：开启 hit .9000（全系最高）vs 关闭 .8833，"
                         "AUC 0.7536 vs 0.7575（噪声带内等价）；60 被数据量下网络自学到"
                         " gate≈[0.20,0.13,0.15]（轻度再参考去共模）。Phase 3 跨数据集"
                         "时按域学参考变换（鼻参考 GTN ↔ 平均参考 ERP CORE ↔ A1 耳参考"
                         "自采 8 导）。--no-rereference 关闭。旧版「强制 CAR」已废弃"
                         "（<32 导无效，Junghöfer 2001/Luck 2014；GTN 实测销毁 P3b 4.36×）")
    ap.add_argument("--p3b-sigma-hi", type=float, default=150.0,
                    help="P3b σ 上界 ms（GLM：儿童 P3b 宽达 300–650ms，旧默认 80 过窄；"
                         "成人数据建议传 80 恢复）")
    ap.add_argument("--dtau-readout", default="attention_softargmax",
                    help="Δτ 读出路径；GTN 默认 softargmax（失败诊断：direct 的 τ0 梯度近乎为零）")
    ap.add_argument("--p3b-tau0-ms", type=float, default=460.0,
                    help="P3b 先验中心（GTN 儿童数据实测峰值 460–490ms，成人仍可用 350）")
    ap.add_argument("--p3b-tau0-lo", type=float, default=350.0, help="P3b τ0 生理界下界（ms）")
    ap.add_argument("--p3b-tau0-hi", type=float, default=600.0, help="P3b τ0 生理界上界（ms）")
    ap.add_argument("--n-channels", type=int, default=3, choices=(3, 8),
                    help="GTN 原生 3 导（默认，EEGNet 借鉴）；8=旧版零填充回退")
    ap.add_argument("--bypass-mode", default="separable_pool",
                    choices=("separable_pool", "mean_pool", "none"),
                    help="Head-A 判别旁路；mean_pool=旧方案 B，none=无旁路")
    ap.add_argument("--no-global-bypass", action="store_true",
                    help="兼容旧参数：等价 --bypass-mode none")
    ap.add_argument("--head-mlp", action="store_true",
                    help="恢复旧版 Head-A/Head-B 的 D//2 隐藏层 MLP（回退）")
    ap.add_argument("--head-dropout", type=float, default=0.25)
    ap.add_argument("--encoder-dropout", type=float, default=0.25)
    ap.add_argument("--no-spatial-max-norm", action="store_true",
                    help="关闭 tokenizer 空间权重 max-norm=1（回退）")
    ap.add_argument("--aux-dataset", default=None, choices=(None, "erpcore", "bnci008", "bi2014a"),
                    help="transfer_policy 方式 B（T3）：辅助域只进域条件仿射 + L_MMD")
    ap.add_argument("--aux-channels", default="Fz,Cz,Pz",
                    help="辅助域裁剪通道（逗号分隔；bi2014a 无 Fz，可用 AFz,Cz,Pz）")
    ap.add_argument("--aux-max-trials", type=int, default=12000,
                    help="辅助域最多采样试次数（默认 12000，避免训练时长被大辅助集支配）")
    ap.add_argument("--aux-seed", type=int, default=0)
    ap.add_argument("--lambda4", type=float, default=0.1, help="L_MMD 权重（aux-dataset 启用时生效）")
    ap.add_argument("--mmd-bandwidth", type=float, default=5.0,
                    help="RBF-MMD 固定带宽；固定值避免每 batch median heuristic 的 GPU 同步")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="只训第一个 fold 并报告时间/显存，不跑全量评估")
    ap.add_argument("--cache-dir", default="experiments/cache")
    ap.add_argument("--save-scores-dir", default=None,
                    help="逐被试 scores JSON 目录；缺省使用 run-dir")
    ap.add_argument("--run-dir", default="experiments/runs")
    ap.add_argument("--run-name", default=None, help="实验名；缺省 n2p3net_gtn_<UTC时间戳>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # v5（2026-08-22）：--subjects 限缩也从全量缓存派生（旧的 n<N> 子集缓存由旧口径生成，
    # 与全量 242 口径的被试/试次不一致，会悄悄改变分母；统一用 nall 后再 np.unique 截取）。
    cache_dir = Path(args.cache_dir)
    cache_path = cache_dir / _gtn_cache_filename(256.0, 0.1, -0.2, 0.8, "all")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"未找到 GTN 全量缓存 {cache_path}；请先运行 experiments/run_gtn_baseline.py 生成缓存。"
        )
    X3, y, digits, subject_ids, true_digits, skipped = _load_gtn_cache(cache_path)
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    if args.subjects:
        keep_subj = np.unique(subject_ids)[: args.subjects]
        keep = np.isin(subject_ids, keep_subj)
        X3, y, digits, subject_ids = X3[keep], y[keep], digits[keep], subject_ids[keep]
        true_digits = {k: v for k, v in true_digits.items() if k in set(keep_subj.tolist())}

    X8, E_chn, channel_mask = _build_gtn_x_and_identity(X3, n_channels=args.n_channels)
    print(f"[n2p3net] X={X8.shape} y={y.shape} subjects={len(np.unique(subject_ids))}")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("n2p3net_gtn_%Y%m%d_%H%M%SZ")
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = Path(args.save_scores_dir) if args.save_scores_dir else run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "schema": "n2p3net_gtn_run/1",
        "run_name": run_name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "benchmark" if args.benchmark else "evaluate",
        "args": vars(args),
        "data": {
            "cache_path": str(cache_path),
            "cache_sha256": cache_sha256,
            "X3_shape": list(X3.shape),
            "X8_shape": list(X8.shape),
            "n_trials": int(X3.shape[0]),
            "n_subjects": int(len(np.unique(subject_ids))),
            "target_rate": float(y.mean()),
            "trials_per_subject": float(X3.shape[0] / max(len(np.unique(subject_ids)), 1)),
            "skipped": [str(s) for s in skipped],
        },
    }

    bypass_mode = "none" if args.no_global_bypass else args.bypass_mode
    model_kwargs = dict(
        n_channels=args.n_channels,
        channel_names=tuple(GTN_STANDARD) if args.n_channels == 3 else None,
        spatial_max_norm=None if args.no_spatial_max_norm else 1.0,
        encoder_dropout=args.encoder_dropout,
        head_mlp=args.head_mlp,
        head_dropout=args.head_dropout,
        dtau_readout=args.dtau_readout,
        encoder_depth=args.encoder_depth,
        encoder_type=args.encoder_type,
        encoder_norm=args.encoder_norm,
        tokenizer_init=args.tokenizer_init,
        tokenizer_post_norm=args.tokenizer_post_norm,
        tokenizer_post_act=args.tokenizer_post_act,
        tau0_ms=(220.0, 300.0, args.p3b_tau0_ms),
        tau0_bounds=((180.0, 280.0), (250.0, 380.0), (args.p3b_tau0_lo, args.p3b_tau0_hi)),
        sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, args.p3b_sigma_hi)),
        bypass_mode=bypass_mode,
        use_rereference=args.use_rereference,
    )
    # GLM：容量预设（mini 系列）。实测 38.5k→7.3k/2.5k 参数 AUC 不降（容量非瓶颈）。
    if args.model_size == "mini":
        model_kwargs.update(d_model=32, filters_per_scale=4,
                            temporal_kernels=(33, 65), encoder_depth=1)
    elif args.model_size == "mini_a":
        model_kwargs.update(d_model=16, filters_per_scale=2,
                            temporal_kernels=(65,), encoder_depth=0)
    trainer_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda_amp=args.lambda_amp,
        lambda_jit=args.lambda_jit,
        jit_prob=args.jit_prob,
        jit_max_ms=args.jit_max_ms,
        augment=args.augment,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
    )
    val_subject_frac = None if args.no_val_early_stop else args.val_subject_frac

    aux_record = None
    if args.aux_dataset:
        aux_channels = tuple(c.strip() for c in args.aux_channels.split(",") if c.strip())
        aux = load_auxiliary(
            args.aux_dataset,
            str(cache_dir),
            target_channels=aux_channels,
            strict_channels=True,
            n_times=256,
        )
        model_channels = tuple(GTN_STANDARD) if args.n_channels == 3 else tuple(STANDARD_CHANNELS)
        X_aux8 = np.zeros((aux.X.shape[0], args.n_channels, aux.X.shape[2]), dtype=np.float32)
        present = {name: i for i, name in enumerate(aux.channel_names)}
        for j, name in enumerate(model_channels):
            src = present.get(name)
            if src is None:
                alias = _AUX_TO_STANDARD.get(name)
                src = present.get(alias) if alias is not None else None
            if src is not None:
                X_aux8[:, j, :] = aux.X[:, src, :]
        aux_record = {
            "dataset": args.aux_dataset,
            "channels": list(aux.channel_names),
            "n_trials": int(aux.X.shape[0]),
            "target_rate": float(aux.y.mean()),
            "lambda4": args.lambda4,
            "mmd_bandwidth": args.mmd_bandwidth,
            "aux_max_trials": args.aux_max_trials,
            "aux_seed": args.aux_seed,
        }
        model_kwargs["n_domains"] = 2

    record["model_kwargs"] = model_kwargs
    record["trainer_kwargs"] = trainer_kwargs
    record["environment"] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }
    if aux_record is not None:
        record["auxiliary"] = aux_record

    if args.aux_dataset:
        adapter = N2P3NetDomainBaseline(
            model_kwargs=model_kwargs,
            trainer_kwargs=trainer_kwargs,
            E_chn=E_chn,
            channel_mask=channel_mask,
            val_subject_frac=val_subject_frac,
            aux_X=X_aux8,
            aux_y=aux.y,
            lambda4=args.lambda4,
            mmd_bandwidth=args.mmd_bandwidth,
            aux_subsample=args.aux_max_trials,
            aux_seed=args.aux_seed,
        )
    else:
        adapter = N2P3NetBaseline(
            model_kwargs=model_kwargs,
            trainer_kwargs=trainer_kwargs,
            E_chn=E_chn,
            channel_mask=channel_mask,
            val_subject_frac=val_subject_frac,
        )

    folds = loso_folds(subject_ids)
    wall_t0 = time.perf_counter()

    # GLM v3：逐 fold 实时进度（progress.jsonl，供仪表盘消费；见 experiments/dashboard.html）
    progress_path = run_dir / "progress.jsonl"
    progress_f = progress_path.open("w", encoding="utf-8")

    def _write_progress(fold_idx: int, fold_result, records) -> None:
        # records: list[(predicted, true, subject)]（见 evaluate._evaluate_one_fold）
        hits = [1 if p == t else 0 for p, t, _ in records] if records else []
        hist = getattr(adapter, "last_history", None) or {}
        line = {
            "type": "fold",
            "fold": fold_idx,
            "n_folds_done": fold_idx + 1,
            "subject": str(records[0][2]) if records else None,
            "hit": hits[0] if hits else None,
            "fold_bacc": float(fold_result.balanced_acc),
            "fold_auc": float(fold_result.auc),
            "fit_sec": float(adapter.fit_durations[-1]) if getattr(adapter, "fit_durations", None) else None,
            "epochs_ran": len(hist.get("train_losses", [])),
            "val_losses": [round(float(v), 4) for v in hist.get("val_losses", [])][-12:],
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        progress_f.write(json.dumps(line, ensure_ascii=False) + "\n")
        progress_f.flush()

    progress_f.write(json.dumps({
        "type": "manifest",
        "run_name": run_name,
        "total_folds": len(folds),
        "n_trials": int(len(X8)),
        "model_kwargs": {k: str(v) for k, v in model_kwargs.items()},
        "trainer_kwargs": trainer_kwargs,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False) + "\n")
    progress_f.flush()

    if args.benchmark:
        train_mask, test_mask = folds[0]
        Xtr, ytr = X8[train_mask], y[train_mask]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if getattr(adapter, "fit_accepts_subject_ids", False):
            adapter.fit(Xtr, ytr, subject_ids=subject_ids[train_mask])
        else:
            adapter.fit(Xtr, ytr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"[benchmark] fit {elapsed:.2f}s for {len(Xtr)} trials × {args.epochs} epochs")
        print(f"[benchmark] seconds/epoch ≈ {elapsed / max(args.epochs, 1):.3f}")
        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"[benchmark] GPU peak memory ≈ {peak_mb:.1f} MB")
        record["benchmark"] = {
            "n_train_trials": int(len(Xtr)),
            "epochs": args.epochs,
            "wall_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(args.epochs, 1),
            "gpu_peak_memory_mb": peak_mb,
            "fit_durations_sec": adapter.fit_durations,
            "fit_peak_memory_mb": adapter.fit_peak_memory_mb,
        }
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record_path = run_dir / "record.json"
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[record] {record_path}")
        progress_f.close()
        return

    summary = evaluate(
        adapter,
        X8,
        y,
        digits,
        subject_ids,
        true_digits,
        folds,
        n_jobs=1,
        on_fold_end=_write_progress,
    )
    progress_f.write(json.dumps({"type": "done", "ts": datetime.now(timezone.utc).isoformat()},
                                ensure_ascii=False) + "\n")
    progress_f.close()
    wall_seconds = time.perf_counter() - wall_t0
    print(f"[result] N2P3Net: 命中率={summary.hit_rate_mean:.4f} "
          f"(±{summary.hit_rate_std:.4f}) bacc={summary.balanced_acc_mean:.4f} "
          f"AUC={summary.auc_mean:.4f} (chance=0.111)")

    scores_path = save_subject_scores(summary, "n2p3net", scores_dir / "n2p3net.json")
    print(f"[scores] {scores_path}")

    record["results"] = {
        "hit_rate_mean": float(summary.hit_rate_mean),
        "hit_rate_std": float(summary.hit_rate_std),
        "balanced_acc_mean": float(summary.balanced_acc_mean),
        "auc_mean": None if summary.auc_mean != summary.auc_mean else float(summary.auc_mean),
        "per_fold": [
            {
                "hit_rate": f.hit_rate,
                "balanced_acc": f.balanced_acc,
                "auc": None if f.auc != f.auc else f.auc,
                "n_subjects": f.n_subjects,
                "n_test_trials": f.n_test_trials,
            }
            for f in summary.per_fold
        ],
        "scores_path": str(scores_path),
    }
    record["timing"] = {
        "total_wall_seconds": wall_seconds,
        "fit_durations_sec": adapter.fit_durations,
        "fit_peak_memory_mb": adapter.fit_peak_memory_mb,
    }
    record["training_history_last_fold"] = adapter.last_history
    # GLM：记录早停协议是否生效（最后 fold 的验证被试数与 val 曲线长度）。
    record["protocol"] = {
        "val_subject_frac": val_subject_frac,
        "early_stop_patience": args.early_stop_patience,
        "last_fold_val_subjects": adapter.last_val_subjects,
        "last_fold_n_val_epochs": len(adapter.last_history.get("val_losses", []))
        if adapter.last_history else None,
    }

    # 最后 fold 的成分记录：τ/σ 与 target/non-target P3b τ 分布。
    last_train_mask, last_test_mask = folds[-1]
    logits_last, tau_last, sigma_last = adapter.predict_full(X8[last_test_mask])
    y_last = y[last_test_mask]
    components = {
        "tau0_bounded_ms": [float(v) for v in adapter.model_.component_window.tau0_bounded.detach().cpu().tolist()],
        "sigma_ms": sigma_last.tolist(),
        "n_test_trials": int(len(y_last)),
        "target_n": int((y_last == 1).sum()),
        "nontarget_n": int((y_last == 0).sum()),
        "tau_target_mean_ms": [float(v) for v in tau_last[y_last == 1].mean(axis=0).tolist()],
        "tau_target_std_ms": [float(v) for v in tau_last[y_last == 1].std(axis=0).tolist()],
        "tau_nontarget_mean_ms": [float(v) for v in tau_last[y_last == 0].mean(axis=0).tolist()],
        "tau_nontarget_std_ms": [float(v) for v in tau_last[y_last == 0].std(axis=0).tolist()],
    }
    record["components_last_fold"] = components
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()

    record_path = run_dir / "record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[record] {record_path}")


if __name__ == "__main__":
    main()
