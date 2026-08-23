"""BNCI2014_008 8 导 LOSO 二分类对比：N2P3Net vs EEGNet（可切 3 导参考）。

辅助域无 thought number，只用 evaluate_binary 的 bacc/AUC 口径（transfer_policy）。
BNCI2014_008：8 被 × 4200 试次（1:5 oddball P300），8 导 = 本项目原生蒙太奇
（Fz,Cz,P3,Pz,P4,PO7,PO8,Oz，含枕区——N200 通路首次结构可用）。

GLM v3（2026-08-23）：n2p3net 默认配置升级为 GTN 定案版（带通 tokenizer init +
TCN BN + 门控参考 + 被试级验证早停），并逐 fold 写 progress.jsonl 供
experiments/dashboard.html 实时可视化。

用法：
    .venv/Scripts/python.exe -u experiments/run_bnci008_loso.py --models n2p3net8 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from baselines.deep import DeepBaseline, DeepConfig  # noqa: E402
from baselines.evaluate import evaluate_binary, loso_folds  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from data.auxiliary import load_auxiliary  # noqa: E402
from data.channel import build_channel_identity  # noqa: E402

CH_8 = ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")
CH_3 = ("Fz", "Cz", "Pz")


def build_adapter(model_name: str, n_channels: int, ch_names: tuple[str, ...],
                  epochs: int, batch_size: int, early_stop_patience: int):
    if model_name == "n2p3net":
        identity = build_channel_identity(ch_names=list(ch_names), channel_mask=(True,) * n_channels)
        return N2P3NetBaseline(
            model_kwargs=dict(
                n_channels=n_channels,
                channel_names=ch_names,
                encoder_depth=3,
                encoder_type="tcn",
                dtau_readout="attention_softargmax",
                # GLM v3 定案配置（GTN 242 被：hit 与 EEGNet 完全打平 p=1.0）
                encoder_norm="bn",
                tokenizer_init="bandpass",
                # 门控参考层：恒等起步；BNCI008 记录参考类型未知，交给门自学
                use_rereference=True,
                # BNCI008 成人 P3b：τ0 先验用成人值（380ms），界放宽到儿童档以防被试差异
                tau0_ms=(200.0, 280.0, 380.0),
                tau0_bounds=((160.0, 260.0), (230.0, 360.0), (300.0, 560.0)),
                sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 120.0)),
                tmin=-0.001,
                tmax=0.8,
                sfreq=256.0,
                n_time=256,
                baseline_n=4,  # AUX epoch 从 0ms 起，用前 4 点(~16ms)作每试次基线，避免 1 点 std NaN
            ),
            trainer_kwargs=dict(
                epochs=epochs,
                batch_size=batch_size,
                lr=1e-3,
                lambda2=0.3,
                lambda3=0.01,
                lambda_amp=0.01,
                lambda_jit=0.0,
                jit_prob=0.0,
                augment=False,
                early_stop_patience=early_stop_patience,
                seed=0,
            ),
            E_chn=torch.from_numpy(identity.embedding),
            channel_mask=torch.ones(n_channels, dtype=torch.bool),
            val_subject_frac=0.12,  # 8 被试 LOSO：7 训练被试中留 1 名做验证（clamp min 2 → 2）
        )
    if model_name == "eegnet":
        return DeepBaseline(
            "eegnet",
            n_chans=n_channels,
            n_times=256,
            sfreq=256.0,
            config=DeepConfig(epochs=epochs, batch_size=batch_size, seed=0),
            device=None,
        )
    raise ValueError(model_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--models", default="n2p3net8",
                    help="逗号分隔：n2p3net8/eegnet8/n2p3net3/eegnet3（默认 n2p3net8=我们的模型原生 8 导）")
    ap.add_argument("--subjects", type=int, default=None, help="前 N 被试（benchmark 用）")
    ap.add_argument("--dataset", default="bnci008", choices=("bnci008", "erpcore"))
    ap.add_argument("--cache-dir", default="experiments/cache")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    model_specs = {
        "n2p3net8": ("n2p3net", CH_8),
        "eegnet8": ("eegnet", CH_8),
        "n2p3net3": ("n2p3net", CH_3),
        "eegnet3": ("eegnet", CH_3),
    }
    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    for spec in selected:
        model_name, ch_names = model_specs[spec]
        n_channels = len(ch_names)
        aux = load_auxiliary(
            args.dataset, args.cache_dir,
            target_channels=None if n_channels == 8 else ch_names,
            strict_channels=n_channels != 8,
            n_times=256,
        )
        X, y, subj = aux.X, aux.y, aux.subject_ids
        if args.subjects:
            keep_subj = np.unique(subj)[: args.subjects]
            keep = np.isin(subj, keep_subj)
            X, y, subj = X[keep], y[keep], subj[keep]
        folds = loso_folds(subj)

        run_name = args.run_name or f"{args.dataset}_glm_v3_{spec}"
        run_dir = ROOT / "experiments" / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        adapter = build_adapter(model_name, n_channels, ch_names,
                                args.epochs, args.batch_size, args.early_stop_patience)

        # 逐 fold 实时进度（dashboard.html 消费）
        progress_f = (run_dir / "progress.jsonl").open("w", encoding="utf-8")
        progress_f.write(json.dumps({
            "type": "manifest",
            "run_name": run_name,
            "total_folds": len(folds),
            "n_trials": int(len(y)),
            "model_spec": spec,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False) + "\n")
        progress_f.flush()

        def _on_fold(fold_idx: int, fold_result) -> None:
            hist = getattr(adapter, "last_history", None) or {}
            fit_sec = adapter.fit_durations[-1] if getattr(adapter, "fit_durations", None) else None
            line = {
                "type": "fold",
                "fold": fold_idx,
                "n_folds_done": fold_idx + 1,
                "fold_bacc": float(fold_result.balanced_acc),
                "fold_auc": float(fold_result.auc),
                "fit_sec": float(fit_sec) if fit_sec else None,
                "epochs_ran": len(hist.get("train_losses", [])),
                "val_losses": [round(float(v), 4) for v in hist.get("val_losses", [])][-12:],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            progress_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            progress_f.flush()

        t0 = time.perf_counter()
        summary = evaluate_binary(adapter, X, y, subj, folds, n_jobs=1, on_fold_end=_on_fold)
        wall = time.perf_counter() - t0
        progress_f.write(json.dumps({"type": "done", "ts": datetime.now(timezone.utc).isoformat()},
                                    ensure_ascii=False) + "\n")
        progress_f.close()

        fit_mean = None
        if isinstance(adapter, N2P3NetBaseline) and adapter.fit_durations:
            fit_mean = float(np.mean(adapter.fit_durations))
        print(f"[{model_name}-{n_channels}ch] subjects={len(np.unique(subj))} "
              f"bacc={summary.balanced_acc_mean:.4f}±{summary.balanced_acc_std:.4f} "
              f"auc={summary.auc_mean:.4f} wall={wall:.1f}s fit_mean={fit_mean}", flush=True)

        payload = {
            "model": f"{model_name}_{n_channels}ch",
            "run_name": run_name,
            "n_subjects": int(len(np.unique(subj))),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "config": "glm_v3",
            "balanced_acc_mean": summary.balanced_acc_mean,
            "balanced_acc_std": summary.balanced_acc_std,
            "auc_mean": summary.auc_mean,
            "wall_seconds": wall,
            "fit_mean_sec": fit_mean,
            "per_fold": [f.__dict__ for f in summary.per_fold],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "record.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # 兼容旧的汇总目录
        out = ROOT / "experiments" / "runs" / f"{args.dataset}_loso_compare"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        (out / f"{model_name}_{n_channels}ch_{stamp}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
