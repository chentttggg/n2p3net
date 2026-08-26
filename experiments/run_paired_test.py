"""配对置换检验入口：比较两个已保存的逐被试结果文件。

用法（项目根目录）：
    .venv/Scripts/python.exe experiments/run_paired_test.py \
        --a experiments/results/eegnet.json \
        --b experiments/results/swlda.json \
        --n-perm 10000 --seed 0

输入文件由 run_gtn_baseline.py --save-scores-dir <dir> 生成，格式为：
    {model, n_subjects, hit_rate_mean, records: [{subject, predicted, true, hit}, ...]}
正式比较要求 cohort/dataset/metric 指纹一致且被试全集完全相同；禁止取交集改变分母。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baselines.evaluate import paired_permutation_test  # noqa: E402
from baselines.experiment_protocol import canonical_sha256  # noqa: E402


def _load_score_contract(path: Path, metric: str = "primary") -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if metric == "primary":
        if payload.get("schema") != "n2p3net_subject_scores/2":
            raise ValueError(
                f"{path} is not a schema-v2 score file; use --metric legacy only for reproduction."
            )
        stored_hash = payload.get("score_sha256")
        unsigned = {key: value for key, value in payload.items() if key != "score_sha256"}
        if stored_hash != canonical_sha256(unsigned):
            raise ValueError(f"{path} score_sha256 does not match its contents.")
        metric_name = payload.get("primary_decision_metric")
        records = payload.get("primary_records", [])
        units = tuple(str(unit) for unit in payload.get("evaluation_units", []))
        if (
            not metric_name
            or not records
            or tuple(str(row.get("subject")) for row in records) != units
        ):
            raise ValueError(f"{path} lacks one ordered primary record per frozen evaluation unit.")
        if len(units) != len(set(units)) or int(payload.get("primary_n_subjects", -1)) != len(
            units
        ):
            raise ValueError(f"{path} has an invalid frozen denominator.")
        rate = float(payload.get("primary_hit_rate", float("nan")))
        identity = {
            "seed": payload.get("seed"),
            "evaluation_mode": payload.get("evaluation_mode"),
            "protocol_sha256": payload.get("protocol_sha256"),
            "cohort_sha256": payload.get("cohort_sha256"),
            "dataset_sha256": payload.get("dataset_sha256"),
        }
        if (
            isinstance(identity["seed"], bool)
            or not isinstance(identity["seed"], int)
            or identity["evaluation_mode"] not in {"development", "confirmatory"}
            or any(
                not isinstance(identity[key], str) or not identity[key]
                for key in ("protocol_sha256", "cohort_sha256", "dataset_sha256")
            )
        ):
            raise ValueError(f"{path} lacks a complete frozen run identity.")
        if identity["evaluation_mode"] == "confirmatory" and any(
            not isinstance(payload.get(key), str) or not payload.get(key)
            for key in (
                "confirmatory_id",
                "confirmatory_lock_sha256",
                "source_sha256",
                "runtime_sha256",
                "external_assets_sha256",
            )
        ):
            raise ValueError(f"{path} lacks confirmatory lock/source/runtime identity.")
    else:
        metric_name = "legacy_sum@all"
        records = payload.get("records", [])
        units = tuple(str(row.get("subject")) for row in records)
        rate = float(payload.get("hit_rate_mean", float("nan")))
        if not records:
            raise ValueError(f"{path} contains no legacy records.")

    hits: dict[str, int] = {}
    availability: dict[str, bool] = {}
    for record in records:
        subject = str(record["subject"])
        if subject in hits:
            raise ValueError(f"{path} has duplicate subject record {subject!r}.")
        hit = int(record["hit"])
        available_raw = record.get("available", True)
        if not isinstance(available_raw, bool):
            raise ValueError(f"{path} has non-boolean availability for {subject!r}.")
        available = available_raw
        predicted = record.get("predicted")
        true = record.get("true")
        if (
            hit not in (0, 1)
            or (not available and (hit != 0 or predicted is not None))
            or (metric == "primary" and available and predicted is None)
            or (predicted is not None and true is not None and hit != int(predicted == true))
        ):
            raise ValueError(f"{path} has invalid ITT outcome for {subject!r}.")
        hits[subject] = hit
        availability[subject] = available
    recomputed = sum(hits.values()) / len(units)
    if not abs(recomputed - rate) < 1e-12:
        raise ValueError(f"{path} reported rate differs from its frozen ITT records.")
    return {
        "model": payload.get("model", path.stem),
        "hits": hits,
        "availability": availability,
        "rate": rate,
        "metric": str(metric_name),
        "units": units,
        "cohort_sha256": payload.get("cohort_sha256"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "seed": payload.get("seed"),
        "evaluation_mode": payload.get("evaluation_mode"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "source_sha256": payload.get("source_sha256"),
        "runtime_sha256": payload.get("runtime_sha256"),
        "primary_metric_gate": dict(payload.get("primary_metric_gate") or {}),
    }


def _load_hits(path: Path, metric: str = "primary") -> tuple[str, dict[str, int], float, str]:
    """加载逐被试命中记录，返回 (model_name, {subject: hit}, hit_rate_mean)。"""
    contract = _load_score_contract(path, metric)
    return contract["model"], contract["hits"], contract["rate"], contract["metric"]


def _exact_mcnemar(hits_a: list[int], hits_b: list[int]) -> tuple[int, int, float]:
    from scipy.stats import binomtest

    a = np.asarray(hits_a, dtype=np.int64)
    b = np.asarray(hits_b, dtype=np.int64)
    a_only = int(np.count_nonzero((a == 1) & (b == 0)))
    b_only = int(np.count_nonzero((a == 0) & (b == 1)))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5).pvalue)
    return a_only, b_only, p_value


def _paired_bootstrap_ci(
    hits_a: list[int], hits_b: list[int], *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    difference = np.asarray(hits_a, dtype=float) - np.asarray(hits_b, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(n_bootstrap, len(difference)))
    values = difference[indices].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="两模型逐被试命中率的配对置换检验")
    ap.add_argument("--a", required=True, help="模型 A 的 scores JSON 路径")
    ap.add_argument("--b", required=True, help="模型 B 的 scores JSON 路径")
    ap.add_argument("--label-a", default=None, help="模型 A 显示名（默认读 JSON 内 model 字段）")
    ap.add_argument("--label-b", default=None, help="模型 B 显示名（默认读 JSON 内 model 字段）")
    ap.add_argument("--n-perm", type=int, default=10000, help="置换次数（默认 10000）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument(
        "--metric",
        choices=("primary", "legacy"),
        default="primary",
        help="默认比较预注册 primary hit@K；legacy 仅用于复现旧 all-trial sum",
    )
    args = ap.parse_args()

    contract_a = _load_score_contract(Path(args.a), args.metric)
    contract_b = _load_score_contract(Path(args.b), args.metric)
    name_a, hits_a, hit_a, metric_a = (
        contract_a["model"],
        contract_a["hits"],
        contract_a["rate"],
        contract_a["metric"],
    )
    name_b, hits_b, hit_b, metric_b = (
        contract_b["model"],
        contract_b["hits"],
        contract_b["rate"],
        contract_b["metric"],
    )
    if metric_a != metric_b:
        raise ValueError(f"primary metric 不一致：A={metric_a}, B={metric_b}，禁止直接配对。")
    label_a = args.label_a or name_a
    label_b = args.label_b or name_b

    if args.metric == "primary" and (
        contract_a["cohort_sha256"] != contract_b["cohort_sha256"]
        or contract_a["dataset_sha256"] != contract_b["dataset_sha256"]
        or contract_a["seed"] != contract_b["seed"]
        or contract_a["evaluation_mode"] != contract_b["evaluation_mode"]
        or contract_a["source_sha256"] != contract_b["source_sha256"]
        or contract_a["runtime_sha256"] != contract_b["runtime_sha256"]
    ):
        raise ValueError(
            "Cohort, dataset, seed, mode, source, or runtime differs; paired comparison is invalid."
        )
    if contract_a["units"] != contract_b["units"]:
        raise ValueError("Score files must contain the same frozen subject universe and order.")
    units = contract_a["units"]
    scores_a = [hits_a[subject] for subject in units]
    scores_b = [hits_b[subject] for subject in units]
    obs_diff, p_value = paired_permutation_test(
        scores_a, scores_b, n_perm=args.n_perm, seed=args.seed
    )
    a_only, b_only, mcnemar_p = _exact_mcnemar(scores_a, scores_b)
    ci_low, ci_high = _paired_bootstrap_ci(
        scores_a, scores_b, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    print(f"[paired] A={label_a}（{args.a}）")
    print(f"[paired] B={label_b}（{args.b}）")
    print(f"[paired] 冻结 ITT 被试数={len(units)}")
    print(f"[paired] metric={metric_a}")
    if args.metric == "primary":
        claim_a = contract_a["primary_metric_gate"].get("claim_eligible")
        claim_b = contract_b["primary_metric_gate"].get("claim_eligible")
        print(f"[paired] primary claim_eligible: A={claim_a}, B={claim_b}")
        if "chain_llr" in str(metric_a) and (claim_a is False or claim_b is False):
            print(
                "[paired] 注意：至少一个模型未通过 repetition primary claim gate；"
                "当前比较是描述性结果，不得作为正式主结论。",
                flush=True,
            )
    print(f"[paired] A 命中率={hit_a:.4f}，B 命中率={hit_b:.4f}")
    print(
        f"[paired] 原始文件口径命中率：{label_a}={hit_a:.4f}，{label_b}={hit_b:.4f}；"
        f"冻结 ITT 口径：{sum(scores_a) / len(scores_a):.4f} vs {sum(scores_b) / len(scores_b):.4f}"
    )
    print(
        f"[paired] mean(A-B)={obs_diff:+.4f}, paired bootstrap 95% CI=[{ci_low:+.4f},{ci_high:+.4f}]"
    )
    print(f"[paired] exact McNemar: A-only={a_only}, B-only={b_only}, p={mcnemar_p:.6f}")
    print(f"[paired] secondary sign-flip p={p_value:.6f}（n_perm={args.n_perm}，seed={args.seed}）")


if __name__ == "__main__":
    main()
