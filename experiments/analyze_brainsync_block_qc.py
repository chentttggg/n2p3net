"""Read-only decomposition of R5 hit differences into support and prediction changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402

DOSES = (1, 3, 5, 10)
METHODS = ("nn_source", "nn_scratch", "xdawn_erp", "nn_lda_xdawn")
LABELS = {
    "nn_source": "NN source",
    "nn_scratch": "NN scratch",
    "xdawn_erp": "xDAWN",
    "nn_lda_xdawn": "三混合：source＋LDA＋xDAWN",
}


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


def analyze(run, output):
    plan = read(run / "plan.json")
    summary = read(run / "analysis/summary.json")
    full = read(run / "analysis/decision_records.json")
    recovery = read(run / "recovery_map.json")
    binding = read(run / "analysis/result_bindings.json")
    output.mkdir(parents=True, exist_ok=False)
    cells = {}
    for dose in DOSES:
        cells[str(dose)] = {}
        for method in METHODS:
            seeds = [plan["target_seeds"][0]] if method == "xdawn_erp" else plan["target_seeds"]
            paired = []
            for seed in seeds:
                b = full[f"current_basic_K{dose}_{method}_s{seed}"]["future"]["blocks"]
                q = full[f"current_reviewed_K{dose}_{method}_s{seed}"]["future"]["blocks"]
                assert len(b) == len(q) == 20
                assert [(x["session_id"], x["block"], x["target"]) for x in b] == [
                    (x["session_id"], x["block"], x["target"]) for x in q
                ]
                basic_correct = np.array([x["prediction"] == x["target"] for x in b])
                reviewed_correct = np.array([x["prediction"] == x["target"] for x in q])
                usable = np.array([x["prediction"] is not None for x in q])
                rejected = [x for x in q if x["prediction"] is None]
                assert all(
                    not x["probability_available"] and min(x["observed_counts"]) < 5
                    for x in rejected
                )
                lost = int(np.sum(basic_correct & ~usable))
                harmed = int(np.sum(basic_correct & ~reviewed_correct & usable))
                rescued = int(np.sum(~basic_correct & reviewed_correct & usable))
                bh, qh = int(basic_correct.sum()), int(reviewed_correct.sum())
                assert bh - qh == lost + harmed - rescued
                paired.append(
                    {
                        "seed": seed,
                        "basic_hits": bh,
                        "reviewed_hits": qh,
                        "reviewed_predicted_blocks": int(usable.sum()),
                        "reviewed_requested_blocks": 20,
                        "basic_correct_lost_to_reviewed_rejection": lost,
                        "basic_hits_on_reviewed_support": int(np.sum(basic_correct & usable)),
                        "harmed_on_shared_support": harmed,
                        "rescued_on_shared_support": rescued,
                        "rejected_blocks": [
                            {
                                "session_id": x["session_id"],
                                "block": x["block"],
                                "observed_counts": x["observed_counts"],
                            }
                            for x in rejected
                        ],
                    }
                )
            keys = (
                "basic_hits",
                "reviewed_hits",
                "reviewed_predicted_blocks",
                "basic_correct_lost_to_reviewed_rejection",
                "basic_hits_on_reviewed_support",
                "harmed_on_shared_support",
                "rescued_on_shared_support",
            )
            avg = {k: float(np.mean([x[k] for x in paired])) for k in keys}
            avg.update(
                requested_blocks=20,
                basic_rate=avg["basic_hits"] / 20,
                reviewed_rate=avg["reviewed_hits"] / 20,
                reviewed_conditional_rate=avg["reviewed_hits"] / avg["reviewed_predicted_blocks"],
                coverage_loss_pp=5 * avg["basic_correct_lost_to_reviewed_rejection"],
                shared_prediction_net_loss_pp=5
                * (avg["harmed_on_shared_support"] - avg["rescued_on_shared_support"]),
                per_seed=paired,
            )
            assert (
                abs(
                    avg["basic_rate"]
                    - summary["panels"]["current_basic"][f"{dose}:{method}"]["fixed_future"][
                        "hit_rate"
                    ]
                )
                < 1e-12
            )
            assert (
                abs(
                    avg["reviewed_rate"]
                    - summary["panels"]["current_reviewed"][f"{dose}:{method}"]["fixed_future"][
                        "hit_rate"
                    ]
                )
                < 1e-12
            )
            cells[str(dose)][method] = avg
    training = {}
    for family in ("nn_source", "nn_scratch", "xdawn_erp", "waveform_lda"):
        training[family] = {}
        for pipeline in ("basic", "reviewed"):
            key = f"current_{pipeline}_K10_{family}_s{plan['target_seeds'][0]}"
            path = run / "results" / key / "result.json"
            if key in recovery:
                path = run / recovery[key]["local_result"]
            assert sha(path) == binding[key]
            fitted = read(path)
            history = fitted["selection"]["refit_history"]
            training[family][pipeline] = {
                "requested_calibration_flashes": fitted["requested_calibration_trials"],
                "retained_projected_epochs": fitted["retained_calibration_epochs"],
                "actual_training_rows": history["fit_scope"]["training_rows"],
                "waveform_channel_indices": history.get("waveform_channel_indices"),
                "waveform_training_policy": history.get("waveform_training_policy"),
                "source_result_sha256": sha(path),
            }
    test_inputs = {}
    for session in plan["test_sessions"]:
        row = next(x for x in plan["rows"] if x["session_id"] == session)
        b = load_epoch_dataset(
            run / row["paths"]["basic"]["path"], require_labels=True, validation="attested"
        )
        q = load_epoch_dataset(
            run / row["paths"]["reviewed"]["path"], require_labels=True, validation="attested"
        )
        ids_b, ids_q = b.event_timeline.evidence_indices, q.event_timeline.evidence_indices
        common = (ids_b >= 0) & (ids_q >= 0)
        mask = q.trial_channel_mask[ids_q[common]]
        difference = (q.X[ids_q[common]] - b.X[ids_b[common]]) * 1e6
        errors = np.sqrt(np.mean(difference**2, axis=-1))[mask]
        histogram = {str(n): int(np.sum(q.trial_channel_mask.sum(axis=1) == n)) for n in [4, 5, 6]}
        valid = {}
        for family in ("nn_source", "xdawn_erp"):
            pred = (
                run
                / "results"
                / f"current_reviewed_K10_{family}_s{plan['target_seeds'][0]}"
                / f"{session}_predictions.npz"
            )
            with np.load(pred, allow_pickle=False) as a:
                valid[family] = int(np.isfinite(a["logits"]).sum())
        test_inputs[session] = {
            "scheduled_flashes": 450,
            "basic_epochs": b.n_epochs,
            "reviewed_epochs": q.n_epochs,
            "reviewed_mask_histogram": histogram,
            "finite_trial_scores": valid,
            "shared_observed_channel_epoch_RMSE_uV": dict(
                zip(
                    ["median", "p95", "maximum"],
                    np.quantile(errors, [0.5, 0.95, 1.0]).tolist(),
                    strict=True,
                )
            ),
        }
    result = {
        "schema": "n2p3_brainsync_block_QC_decomposition/1",
        "unit": "R5: all 45 scheduled flashes per block",
        "test_sessions": plan["test_sessions"],
        "methods": list(METHODS),
        "calibration_record_counts": list(DOSES),
        "new_model_fits": 0,
        "requested_blocks": 20,
        "cells": cells,
        "C10_training": training,
        "test_inputs": test_inputs,
        "decomposition": "basic correct count minus reviewed correct count = basic correct on reviewed-rejected blocks + harmed minus rescued on shared support",
        "interpretation_limit": "mechanical support decomposition; not a separate causal ablation of training QC and test QC",
        "parent_summary_sha256": sha(run / "analysis/summary.json"),
        "parent_predictions_sha256": sha(run / "analysis/decision_records.json"),
        "script_sha256": sha(__file__),
    }
    write(output / "results.json", result)
    render(output, result)
    write(
        output / "verification.json",
        {
            "status": "passed",
            "new_fits": 0,
            "paired_keys_equal": True,
            "all_rejected_cases_have_incomplete_physical_evidence": True,
            "paired_seed_method_dose_decomposition_checks": sum(
                len(cell["per_seed"]) for dose in cells.values() for cell in dose.values()
            ),
            "all_decomposition_identities_passed": True,
            "original_summary_rates_match": True,
            "results_sha256": sha(output / "results.json"),
        },
    )
    return result


def render(output, result):
    table = []
    for dose in DOSES:
        values = [result["cells"][str(dose)][m] for m in METHODS]
        table.append(
            f"<tr><th>{dose}份</th>"
            + "".join(
                f"<td>{100 * v['basic_rate']:.2f}% → {100 * v['reviewed_rate']:.2f}%</td>"
                for v in values
            )
            + "</tr>"
        )
    decomposition = []
    for method in METHODS:
        v = result["cells"]["10"][method]
        decomposition.append(
            f"<tr><td>{LABELS[method]}</td><td>{v['basic_hits']:.2f}/20</td><td>{v['reviewed_hits']:.2f}/20</td>"
            f"<td>{v['reviewed_predicted_blocks']:.0f}/20</td><td>{100 * v['reviewed_conditional_rate']:.2f}%</td>"
            f"<td>{v['coverage_loss_pp']:.2f}个百分点</td><td>{v['shared_prediction_net_loss_pp']:.2f}个百分点</td></tr>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Block命中率与reviewed损失分解</title><style>body{font:15px/1.75 system-ui;background:#f5f8fb;color:#18334a;margin:32px}main{max-width:1350px;margin:auto}table{border-collapse:collapse;width:100%;background:white;margin:18px 0 30px}th,td{border:1px solid #d4dfeb;padding:10px;text-align:center}th{background:#eaf1f8}.note{background:white;border-left:4px solid #337bb1;padding:18px}a{color:#145a91}</style><main><h1>45闪烁一个block：命中率与QC分解</h1><div class="note">沿用两份固定测试录制、20个block、目标7和9。一个block累计5轮、45次闪烁，输出一个目标数字。神经及混合按原三个目标seed平均，xDAWN为确定性单拟合；拒判仍在20请求分母。以下每格为basic → reviewed，未重新训练任何模型。</div><h2>当前四方法结果</h2><table><tr><th>校准量</th>"""
    page += (
        "".join(f"<th>{LABELS[m]}</th>" for m in METHODS) + "</tr>" + "".join(table) + "</table>"
    )
    page += (
        """<h2>10份校准：差距来自哪里</h2><table><tr><th>方法</th><th>basic正确</th><th>reviewed正确</th><th>reviewed可判</th><th>可判子集命中率</th><th>原本正确但现被拒判的损失</th><th>共同可判block预测净损失</th></tr>"""
        + "".join(decomposition)
        + "</table>"
    )
    page += """<div class="note"><b>主要是覆盖损失，不是所有可判block都识别失败。</b>当前规则要求45个闪烁都能评分；缺一次也可能使整个R5拒判。reviewed让NN只有15/20个block完整可用；xDAWN和LDA要求固定所选六导同时观测，只有11/20完整可用。三混合要求三位专家齐备，继承11/20的覆盖。<br>对10份NN source，basic平均19.67个正确中有5个落在reviewed拒判集合；仅这部分就损失25个百分点。共同可判集合再净少1.67个正确，最终13/20。三混合则是19−9−1＝9/20，45个百分点损失来自拒判集合。<br><b>校准数据也变了。</b>NN实际拟合行由4455减到4097；xDAWN/LDA由4455减到3133，其中缺导必须完整观测的约束影响更大。reviewed也按干净段重置滤波并携带通道掩码，模型、归一化和可用输入均随之改变。本轮不能把共同可判集合的残余差异单独归因于滤波或删样本。<br>这并非每个剂量都下降：1份NN在reviewed保留的15个block上，basic原有9个正确，reviewed为11个正确；全20请求的命中仍从60%变55%，因为另3个原本正确的block被拒判。<br>保留且共同观测到的通道上，两份测试的channel-epoch波形RMSE中位数均为0，95%分位均小于0.003μV；少数局部变化另见JSON。这里的主要差异是数据支持、缺导接入和拟合范围，不能把整体差距称为普遍波形翻转或清洗损坏。<br>所有结论限单人两份已探索的录制；整段处理后回顾性计分，不代表在线设备或新录制的保证。</div><p><a href="results.json">逐block配对、掩码与训练范围</a> · <a href="verification.json">核验</a> · <a href="../index.html">原block全矩阵</a> · <a href="../single_round/index.html">单轮R1结果</a></p></main></html>"""
    (output / "index.html").write_text(page, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run.resolve(), args.output.resolve())
    for method in METHODS:
        print(
            method,
            result["cells"]["10"][method]["basic_rate"],
            result["cells"]["10"][method]["reviewed_rate"],
        )


if __name__ == "__main__":
    main()
