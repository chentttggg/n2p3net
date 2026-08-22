"""配对置换检验入口：比较两个已保存的逐被试结果文件。

用法（项目根目录）：
    .venv/Scripts/python.exe experiments/run_paired_test.py \
        --a experiments/results/eegnet.json \
        --b experiments/results/swlda.json \
        --n-perm 10000 --seed 0

输入文件由 run_gtn_baseline.py --save-scores-dir <dir> 生成，格式为：
    {model, n_subjects, hit_rate_mean, records: [{subject, predicted, true, hit}, ...]}
比较时按 subject 对齐并取交集；subject 数不一致会显式告警，不会静默改变分母。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.evaluate import paired_permutation_test  # noqa: E402


def _load_hits(path: Path) -> tuple[str, dict[str, int], float]:
    """加载逐被试命中记录，返回 (model_name, {subject: hit}, hit_rate_mean)。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = payload.get("model", path.stem)
    hit_rate_mean = float(payload.get("hit_rate_mean", float("nan")))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"{path} 中没有 records，无法做配对检验。")

    hits: dict[str, int] = {}
    duplicates = []
    for rec in records:
        subject = str(rec["subject"])
        if subject in hits:
            duplicates.append(subject)
        hits[subject] = int(rec["hit"])

    if duplicates:
        print(
            f"[warn] {path} 存在重复 subject 记录（{len(duplicates)} 个），已保留最后一条："
            f"{duplicates[:5]}{' ...' if len(duplicates) > 5 else ''}",
            file=sys.stderr,
        )
    n_subjects = payload.get("n_subjects")
    if n_subjects is not None and int(n_subjects) != len(records):
        print(
            f"[warn] {path} n_subjects={n_subjects} 与 records 长度 {len(records)} 不一致。",
            file=sys.stderr,
        )
    recomputed = sum(hits.values()) / len(hits) if hits else float("nan")
    if not abs(recomputed - hit_rate_mean) < 1e-6:
        print(
            f"[warn] {path} 的 hit_rate_mean={hit_rate_mean:.6f} 与 records 重算值 "
            f"{recomputed:.6f} 不一致，后续使用 records 重算口径。",
            file=sys.stderr,
        )
        hit_rate_mean = recomputed
    return model_name, hits, hit_rate_mean


def main() -> None:
    ap = argparse.ArgumentParser(description="两模型逐被试命中率的配对置换检验")
    ap.add_argument("--a", required=True, help="模型 A 的 scores JSON 路径")
    ap.add_argument("--b", required=True, help="模型 B 的 scores JSON 路径")
    ap.add_argument("--label-a", default=None, help="模型 A 显示名（默认读 JSON 内 model 字段）")
    ap.add_argument("--label-b", default=None, help="模型 B 显示名（默认读 JSON 内 model 字段）")
    ap.add_argument("--n-perm", type=int, default=10000, help="置换次数（默认 10000）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    args = ap.parse_args()

    name_a, hits_a, hit_a = _load_hits(Path(args.a))
    name_b, hits_b, hit_b = _load_hits(Path(args.b))
    label_a = args.label_a or name_a
    label_b = args.label_b or name_b

    common = sorted(set(hits_a) & set(hits_b))
    only_a = sorted(set(hits_a) - set(hits_b))
    only_b = sorted(set(hits_b) - set(hits_a))
    if not common:
        raise ValueError("两个 scores 文件没有共同 subject，无法配对。")

    scores_a = [hits_a[s] for s in common]
    scores_b = [hits_b[s] for s in common]
    obs_diff, p_value = paired_permutation_test(
        scores_a, scores_b, n_perm=args.n_perm, seed=args.seed
    )

    print(f"[paired] A={label_a}（{args.a}）")
    print(f"[paired] B={label_b}（{args.b}）")
    print(f"[paired] 共同被试数={len(common)}")
    if only_a:
        print(f"[paired] 仅在 A 中的被试数={len(only_a)}")
    if only_b:
        print(f"[paired] 仅在 B 中的被试数={len(only_b)}")
    print(f"[paired] A 命中率={hit_a:.4f}，B 命中率={hit_b:.4f}")
    print(
        f"[paired] 原始文件口径命中率：{label_a}={hit_a:.4f}，{label_b}={hit_b:.4f}；"
        f"交集口径：{sum(scores_a) / len(scores_a):.4f} vs {sum(scores_b) / len(scores_b):.4f}"
    )
    print(f"[paired] mean(A-B)={obs_diff:+.4f}")
    print(f"[paired] p_value={p_value:.6f}（n_perm={args.n_perm}，seed={args.seed}）")
    print(
        "[paired] 结论："
        + (
            f"{label_a} 显著优于 {label_b}（p<0.05）"
            if p_value < 0.05 and obs_diff > 0
            else (
                f"{label_b} 显著优于 {label_a}（p<0.05）"
                if p_value < 0.05 and obs_diff < 0
                else "差异不显著（p>=0.05）"
            )
        )
    )


if __name__ == "__main__":
    main()
