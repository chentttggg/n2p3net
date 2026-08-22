"""辅助 P300 数据监督预训练入口（P9 / transfer_policy 方式 A）。

用法：
    .venv/Scripts/python.exe experiments/run_aux_pretrain.py \
        --dataset erpcore --model eegnet \
        --channels Fz,Cz,Pz --epochs 30 --batch-size 512

约束：
    - 只用辅助域 target/non-target 标签，不接触 GTN thought number；
    - 90%/10% 按被试分层留出辅助验证集，报告辅助域 AUC/bacc；
    - 固定 epochs，不做依赖 GTN 的 early stopping；
    - checkpoint 保存到 experiments/cache/pretrain/<aux>_<backbone>_<sfreq>.pt；
    - 最终 GTN 微调/冻结对照由 run_gtn_baseline.py 的
      --pretrained-checkpoint / --freeze-prefixes 完成。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from baselines.deep import DeepBaseline, DeepConfig  # noqa: E402
from data.auxiliary import load_auxiliary  # noqa: E402

MODEL_CHOICES = ("eegnet", "inception", "conformer")


def _parse_channels(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [c.strip() for c in text.split(",") if c.strip()]


def _auto_pos_weight(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos


def main() -> None:
    ap = argparse.ArgumentParser(description="辅助 P300 数据 target/non-target 预训练（P9）")
    ap.add_argument("--dataset", required=True, choices=("erpcore", "bnci008", "bi2014a"))
    ap.add_argument("--model", required=True, choices=MODEL_CHOICES)
    ap.add_argument("--cache-dir", default="experiments/cache", help="download_datasets 输出目录")
    ap.add_argument("--channels", default=None,
                    help="逗号分隔的目标通道；缺省保留数据集原始通道。例：Fz,Cz,Pz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="0=按训练集 target 基率自动计算 neg/pos")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "xpu", "cpu"))
    ap.add_argument("--out-dir", default="experiments/cache/pretrain")
    args = ap.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        raise ValueError(f"--val-frac 须在 (0,1) 内，得到 {args.val_frac}。")

    target_channels = _parse_channels(args.channels)
    aux = load_auxiliary(
        args.dataset,
        args.cache_dir,
        target_channels=target_channels,
        strict_channels=target_channels is not None,
        n_times=256,
    )
    print(
        f"[aux] {args.dataset}: X={aux.X.shape} y={aux.y.shape} "
        f"channels={aux.channel_names} target_rate={aux.y.mean():.4f}",
        flush=True,
    )

    # 按被试分层留出辅助验证集；只用辅助域自身评估，不作为主任务验收。
    subj_ids = aux.subject_ids
    unique_subj = np.unique(subj_ids)
    if len(unique_subj) >= 2 and args.val_frac > 0.0:
        train_subj, val_subj = train_test_split(
            unique_subj, test_size=args.val_frac, random_state=args.seed, shuffle=True
        )
        train_idx = np.isin(subj_ids, train_subj)
        val_idx = np.isin(subj_ids, val_subj)
    else:
        # 没有 subject 元数据时退化为样本级分层切分
        train_idx = np.zeros(len(aux.y), dtype=bool)
        val_idx = np.zeros(len(aux.y), dtype=bool)
        tr, va = train_test_split(
            np.arange(len(aux.y)),
            test_size=max(args.val_frac, 1e-6),
            random_state=args.seed,
            stratify=aux.y,
        )
        train_idx[tr] = True
        val_idx[va] = True

    pos_weight = args.pos_weight if args.pos_weight > 0 else _auto_pos_weight(aux.y[train_idx])

    if args.device == "auto":
        device = None
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda 但当前环境不可用 CUDA。")
        if device.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("--device xpu 但当前环境不可用 XPU。")
    clf = DeepBaseline(
        args.model,
        n_chans=aux.n_channels,
        n_times=aux.n_times,
        sfreq=256.0,
        config=DeepConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=pos_weight,
            seed=args.seed,
            standardize_input=True,
        ),
        device=device,
    )
    print(
        f"[pretrain] {args.model} on {args.dataset}: train={int(train_idx.sum())} "
        f"val={int(val_idx.sum())} pos_weight={pos_weight:.2f} epochs={args.epochs}",
        flush=True,
    )
    clf.fit(aux.X[train_idx], aux.y[train_idx])

    val_logits = clf.predict_logit(aux.X[val_idx])
    val_y = aux.y[val_idx]
    auc = float(roc_auc_score(val_y, val_logits)) if len(np.unique(val_y)) == 2 else float("nan")
    val_logits_centered = val_logits - val_logits.mean()
    bacc = float(balanced_accuracy_score(val_y, (val_logits_centered > 0).astype(int)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = clf.save_checkpoint(
        out_dir / f"{args.dataset}_{args.model}_sf256.pt"
    )
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "channels": list(aux.channel_names),
        "n_train": int(train_idx.sum()),
        "n_val": int(val_idx.sum()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": pos_weight,
        "seed": args.seed,
        "val_auc": auc,
        "val_bacc": bacc,
        "checkpoint": str(ckpt_path),
    }
    report_path = out_dir / f"{args.dataset}_{args.model}_sf256.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[aux] val AUC={auc:.4f} bacc={bacc:.4f}", flush=True)
    print(f"[aux] checkpoint={ckpt_path}", flush=True)
    print(f"[aux] report={report_path}", flush=True)
    print("[aux] 注意：辅助域指标只作域漂移观察，不能替代 GTN 主任务验收（P9）。", flush=True)


if __name__ == "__main__":
    main()
