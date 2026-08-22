"""诊断：deep 基线 logit 校准 vs 决策层命中率（Phase 1 三思用，非正式评估）。

背景（2026-08-21 实测）：
    EEGNet 在 GTN 上出现「单试次 AUC ~0.69，但 bacc 0.50、命中率 ~chance」的矛盾；
    同时 SWLDA（AUC 0.712）命中率 0.467 远低于 template（AUC 0.698）命中率 0.767。
    本脚本在前 N 名被试上做一次全量 fit/predict（非 LOSO，诊断专用），输出：
      1) target vs non-target 的 logit 分布（均值/中位数/分位）；
      2) decision 层每个数字的累积 score 分布（看 target 数字是否显著突出）；
      3) 对比「V 单位」vs「z-score 输入」的 AUC/bacc。

用法（项目根目录）：
    .venv/Scripts/python.exe experiments/diagnose_logit.py --model eegnet --subjects 30 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.gtn import read_gtn_experiment  # noqa: E402
from data.preprocess import preprocess  # noqa: E402
from models.decision import decide  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

GTN_STANDARD = ("Fz", "Cz", "Pz")
GTN_ROOT = Path(__file__).resolve().parent.parent / "mne_data" / "MNE-P3-data"


def load(max_subjects: int):
    exps = sorted([d for d in GTN_ROOT.iterdir() if d.is_dir() and d.name.startswith("Experiment")])[:max_subjects]
    Xl, yl, dl, sl = [], [], [], []
    true_digits = {}
    for exp in exps:
        g = read_gtn_experiment(exp)
        r = preprocess(g.raw, g.events, standard=GTN_STANDARD, sfreq=256.0, l_freq=0.1, tmin=-0.2, tmax=0.8)
        d = g.events[r.event_indices, 2].astype(int)
        y = (d == g.thought_number).astype(int)
        Xl.append(r.data); yl.append(y); dl.append(d)
        sl.extend([g.subject_id] * len(y)); true_digits[g.subject_id] = g.thought_number
    return (np.concatenate(Xl).astype(np.float32), np.concatenate(yl).astype(np.int64),
            np.concatenate(dl).astype(np.int64), np.array(sl), true_digits)


def diagnose_logit(model, X, y, digits, sids, true_digits, tag):
    """在全体数据上 fit 一次（非严格 LOSO，仅供诊断），分析 logit 与 decision score。"""
    model.fit(X, y)
    logits = model.predict_logit(X)
    auc = roc_auc_score(y, logits) if len(np.unique(y)) == 2 else np.nan
    logits_c = logits - logits.mean()
    bacc = ((logits_c[y == 1] > 0).mean() + (logits_c[y == 0] <= 0).mean()) / 2

    print(f"\n=== {tag} ===")
    print(f"AUC={auc:.4f}  bacc(阈值0)={bacc:.4f}")
    print(f"logit 分布: target mean={logits[y==1].mean():+.3f} med={np.median(logits[y==1]):+.3f} | "
          f"non-target mean={logits[y==0].mean():+.3f} med={np.median(logits[y==0]):+.3f}")
    print(f"logit 分位(全体): p10={np.percentile(logits,10):+.3f} p50={np.percentile(logits,50):+.3f} "
          f"p90={np.percentile(logits,90):+.3f}")

    # decision 层：每个被试 target 数字的 score 排名（raw 与中心化对照）
    def _ranks(res):
        ranks = []
        for i, subj in enumerate(res.subject_ids):
            true_d = true_digits.get(subj)
            if true_d is None:
                continue
            # target 数字在 9 个 score 中的排名（1=最高）
            raw = res.raw_scores[i]
            valid = np.isfinite(raw)
            if not valid.any():
                continue
            order = np.argsort(-np.where(valid, raw, -np.inf))  # 降序
            hit_idx = np.where(res.digit_vocab[order] == true_d)[0]
            if hit_idx.size == 0:
                continue
            rank = int(hit_idx[0]) + 1
            ranks.append(rank)
        return np.array(ranks)

    res_raw = decide(logits, digits, sids, digit_vocab=list(range(1, 10)), center_logits=False)
    res_c = decide(logits, digits, sids, digit_vocab=list(range(1, 10)), center_logits=True)
    ranks_raw, ranks_c = _ranks(res_raw), _ranks(res_c)
    print(f"decision 层 target 排名(未中心化): top1={(ranks_raw == 1).mean():.3f}  "
          f"top3={(ranks_raw <= 3).mean():.3f}")
    print(f"decision 层 target 排名(中心化):   top1={(ranks_c == 1).mean():.3f}  "
          f"top3={(ranks_c <= 3).mean():.3f}  (chance: top1=0.111, top3=0.333)")
    return auc, bacc, ranks_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eegnet")
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    X, y, digits, sids, true_digits = load(args.subjects)
    print(f"X={X.shape} 被试={len(np.unique(sids))} target基率={y.mean():.4f}")

    from baselines.deep import DeepBaseline, DeepConfig

    n_times = X.shape[2]
    # V 单位
    m = DeepBaseline(args.model, n_chans=3, n_times=n_times, sfreq=256.0, config=DeepConfig(epochs=args.epochs))
    diagnose_logit(m, X, y, digits, sids, true_digits, f"{args.model} [V单位]")

    # z-score 输入
    Xz = (X - X.mean()) / (X.std() + 1e-8)
    m2 = DeepBaseline(args.model, n_chans=3, n_times=n_times, sfreq=256.0, config=DeepConfig(epochs=args.epochs))
    diagnose_logit(m2, Xz.astype(np.float32), y, digits, sids, true_digits, f"{args.model} [z-score]")

    # 对照：template 的 decision 排名（免费地板基准）
    from baselines.classic import TemplateMatching
    tm = TemplateMatching(sfreq=256.0, tmin=-0.2, window_ms=(250.0, 500.0))
    diagnose_logit(tm, X, y, digits, sids, true_digits, "template [V单位, 对照]")


if __name__ == "__main__":
    main()
