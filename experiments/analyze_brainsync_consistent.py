"""Recompute scored R1/R5 recognition from frozen, event-aligned model outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

from data.candidate_task import MembershipDecision  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from models.candidate_evidence import membership_decision_scores  # noqa: E402
from models.candidate_ties import candidate_top_tie_mask  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from research.target_recognition import (  # noqa: E402
    compare_common_scored_decisions,
    summarize_target_decisions,
)

VOCAB = tuple(map(str, range(1, 10)))
DOSES = (1, 3, 5, 10)


def decision_records(dataset, logits, *, session_id, revealed_target, repetitions):
    """Truth is used after scoring; missing flashes preserve their scheduled slot."""
    timeline = dataset.event_timeline
    values = np.asarray(logits, dtype=float)
    if values.shape != (dataset.n_epochs,) or timeline.n_events != 450:
        raise ValueError("BrainSync recognition expects the original 450-event ledger and aligned epoch scores")
    full = np.full(450, np.nan)
    mapping = np.asarray(timeline.evidence_indices)
    available = mapping >= 0
    full[available] = values[mapping[available]]
    width = 9 * repetitions
    if repetitions not in (1, 5):
        raise ValueError("this comparison declares R1 and R5")
    records = []
    oracles = 0
    for start in range(0, 450, width):
        stop = start + width
        codes = np.asarray(timeline.candidate_ids)[start:stop].astype(str)
        if any(set(codes[i:i+9]) != set(VOCAB) for i in range(0, width, 9)):
            raise ValueError("original physical cycle does not contain all nine candidates once")
        decision = MembershipDecision.direct(codes, VOCAB,
            event_ids=np.asarray(timeline.event_ids)[start:stop].astype(str),
            repetition_indices=np.repeat(np.arange(repetitions), 9), repetitions=repetitions,
            evidence_indices=np.arange(width))
        raw = full[start:stop]
        scores = membership_decision_scores(raw, decision, available=np.isfinite(raw))
        if np.isfinite(raw).all():
            direct = np.asarray([raw[codes == digit].sum() for digit in VOCAB])
            if not np.allclose(scores.posterior, softmax(direct), rtol=0, atol=1e-12):
                raise RuntimeError("production candidate probabilities differ from the direct sum-logit oracle")
            winners = np.flatnonzero(candidate_top_tie_mask(direct))
            predicted = VOCAB[winners[0]] if len(winners) == 1 else None
            if scores.predicted != predicted:
                raise RuntimeError("production unique candidate differs from direct argmax")
            oracles += 1
        block, cycle = start // 45 + 1, start % 45 // 9 + 1
        identifier = f"{session_id}:R{repetitions}:block:{block}" + (f":cycle:{cycle}" if repetitions == 1 else "")
        records.append({"decision_id": identifier, "session_id": session_id, "block": block,
            "cycle": cycle if repetitions == 1 else None, "repetitions": repetitions,
            "prediction": scores.predicted, "target": str(revealed_target),
            "reason": scores.reason, "status": "scored" if scores.predicted is not None else "unscored",
            "counts": scores.counts.tolist(), "posterior": scores.posterior.tolist()})
    return records, oracles


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared_path = run / "target_nominal_v1/prepared.json"
    control_path = run / "analytic_controls_nominal_v1/checkpoint.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    controls = json.loads(control_path.read_text(encoding="utf-8"))
    if prepared["status"] != "prepared" or controls["status"] != "completed":
        raise ValueError("analysis requires completed target inputs and completed analytic controls")
    decisions, cells = {}, {}
    oracle_count = 0
    evidence = []
    for name, cell in prepared["cells"].items():
        for family in ("xdawn_erp", "waveform_lda"):
            receipt = Path(controls["fits"][name + "_" + family]["result"])
            fit = json.loads(receipt.read_text(encoding="utf-8"))
            if fit["status"] != "completed" or not fit["reload_logits_bitwise_equal"]:
                raise ValueError("analytic model lacks completed fit and exact replay evidence")
            model = receipt.parent / "model.pt"
            if file_sha256(model) != fit["model_sha256"]:
                raise ValueError("saved analytic model changed after its original fit")
            by_level = defaultdict(list)
            for row in cell["rows"]:
                if row["role"] != "fixed_future_test":
                    continue
                path = Path(row["path"])
                if file_sha256(path) != row["sha256"]:
                    raise ValueError("target cache changed after preprocessing")
                dataset = load_epoch_dataset(path, validation="attested")
                prediction = receipt.parent / (row["session_id"] + "_predictions.npz")
                with np.load(prediction, allow_pickle=False) as saved:
                    if (not np.array_equal(saved["event_ids"], dataset.event_timeline.event_ids)
                            or not np.array_equal(saved["evidence_indices"], dataset.event_timeline.evidence_indices)):
                        raise ValueError("prediction vectors do not align with the original event ledger")
                    logits = saved["logits"]
                for repetitions in (1, 5):
                    records, count = decision_records(dataset, logits, session_id=row["session_id"],
                        revealed_target=row["target"], repetitions=repetitions)
                    by_level[f"R{repetitions}"].extend(records)
                    oracle_count += count
                evidence.append({"cell": name, "family": family, "session_id": row["session_id"],
                    "cache_sha256": row["sha256"], "prediction_sha256": file_sha256(prediction),
                    "model_sha256": fit["model_sha256"], "fit_receipt_sha256": file_sha256(receipt)})
            key = name + "_" + family
            decisions[key] = dict(by_level)
            cells[key] = {level: summarize_target_decisions(records) for level, records in by_level.items()}
    paired = {}
    for dose in DOSES:
        for family in ("xdawn_erp", "waveform_lda"):
            a, b = (decisions[f"{pipeline}_K{dose}_{family}"] for pipeline in ("basic", "QC"))
            paired[f"K{dose}_{family}"] = {level: compare_common_scored_decisions(a[level], b[level]) for level in ("R1", "R5")}
    result = {"schema": "n2p3_consistent_BrainSync_progress_report/1", "status": "partial_awaiting_fresh_NN_pretraining",
        "displayed_methods": ["NN source", "NN scratch", "xDAWN", "source+LDA+xDAWN"],
        "complete_methods": ["xDAWN"], "internal_pool_component_completed": "waveform LDA",
        "unexecuted_methods": ["NN source", "NN scratch", "source+LDA+xDAWN"],
        "accuracy": "correct / scored unique predictions", "coverage": "scored / all requested decisions",
        "chance": 1/9, "cells": cells, "paired_common_scored": paired,
        "independent_sum_logit_oracles": oracle_count, "evidence": evidence,
        "prepared_sha256": file_sha256(prepared_path), "controls_sha256": file_sha256(control_path),
        "role": "retrospective development; same person; two fixed recording clusters; not independent confirmation"}
    atomic_write_json(output / "results.json", result)
    atomic_write_json(output / "decisions.json", decisions)
    render(output / "index.html", result)
    return result


def render(path, result):
    def rate(value):
        return "—" if value is None else f"{100*value:.1f}%"

    sections = []
    for level in ("R1", "R5"):
        rows = []
        for dose in DOSES:
            for pipeline in ("basic", "QC"):
                m = result["cells"][f"{pipeline}_K{dose}_xdawn_erp"][level]
                rows.append(f'<tr><td>{dose} 份</td><td>{pipeline} → {pipeline}</td>'
                    '<td class="pending">待预训练</td><td class="pending">待执行</td>'
                    f'<td><b>{rate(m["accuracy"])}</b><small>{m["correct_decisions"]}/{m["scored_decisions"]}</small></td>'
                    '<td class="pending">待 NN source</td>'
                    f'<td>{rate(m["coverage"])}<small>{m["scored_decisions"]}/{m["requested_decisions"]}</small></td>'
                    f'<td>{m["unscored_decisions"]}</td></tr>')
        label = "R1 · 单轮九选一" if level == "R1" else "R5 · 五轮合并九选一"
        sections.append(f'<section id="{level}"><h2>{label}</h2><div class="scroll"><table>'
            '<thead><tr><th>校准量</th><th>处理链</th><th>NN source</th><th>NN scratch</th><th>xDAWN 识别率</th>'
            '<th>三混合</th><th>xDAWN 覆盖率</th><th>未评分</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')
    pairs = []
    for dose in DOSES:
        for level in ("R1", "R5"):
            m = result["paired_common_scored"][f"K{dose}_xdawn_erp"][level]
            pairs.append(f'<tr><td>{dose} 份</td><td>{level}</td><td>{m["common_scored_decisions"]}/{m["requested_decisions"]}</td>'
                f'<td>{rate(m["left"]["accuracy"])}</td><td>{rate(m["right"]["accuracy"])}</td></tr>')
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>BrainSync · 一致 QC 重做进度</title><style>'
    document += 'body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#1d2939;margin:0}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:30px;margin:8px 0}h2{font-size:21px}.note{background:#e9f0fb;padding:18px 22px;border-radius:10px}section{background:white;padding:22px;margin-top:24px;border-radius:12px}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:left;padding:12px;border-bottom:1px solid #e8ebf0}th{font-size:14px;color:#475467}small{display:block;color:#667085}b{font-size:19px}.pending{color:#8994a3}.scroll{overflow:auto}a{color:#215da8}.muted{color:#667085;font-size:14px}</style><main>'
    document += '<p class="muted">BrainSync / 同一个体 / 开发性比较</p><h1>一致 QC 重做：已评分目标识别率</h1>'
    document += '<p class="note">xDAWN 的两条处理链已重做并从原始 EDF 重放。新 NN 源预训练尚未执行，因此 NN source、scratch 和三混合仍待完成。下表不能用于判断完整四方法排名。</p>'
    document += '<p>识别率 = 猜对数 ÷ 已评分决策数；未评分不计入该分母。覆盖率 = 已评分 ÷ 全部请求。随机基线为 1/9（11.11%）。每个数值下面显示实际分子/分母。</p>'
    document += ''.join(sections)
    document += '<section><h2>共同已评分决策 · xDAWN 配对比较</h2><p>在两臂都能评分的同一批请求上再比较，完整覆盖率仍见上表。</p><div class="scroll"><table><thead><tr><th>校准量</th><th>证据轮数</th><th>共同已评分/请求</th><th>basic</th><th>QC</th></tr></thead><tbody>' + ''.join(pairs) + '</tbody></table></div></section>'
    if "all_remaining_cells" in result:
        rows = []
        for dose in DOSES:
            for level in ("R1", "R5"):
                for pipeline in ("basic", "QC"):
                    metric = result["all_remaining_cells"][f"{pipeline}_K{dose}_xdawn_erp"]["all_remaining"][level]
                    rows.append(f'<tr><td>{dose} 份</td><td>{level}</td><td>{pipeline}</td>'
                        f'<td>{rate(metric["accuracy"])}<small>{metric["correct_decisions"]}/{metric["scored_decisions"]}</small></td>'
                        f'<td>{rate(metric["coverage"])}<small>{metric["scored_decisions"]}/{metric["requested_decisions"]}</small></td></tr>')
        document += '<section><h2>全部剩余测试录制 · xDAWN</h2><p>包含两份固定未来测试及未用于当前校准的其他录制；较早录制标作回顾性留出。各校准量的测试名单不同，跨剂量比较请同时参考上方共同固定测试。</p><div class="scroll"><table><thead><tr><th>校准量</th><th>轮数</th><th>预处理</th><th>已评分识别率</th><th>覆盖率</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>'
    document += '<p class="muted">仅一个人、两份固定测试录制（目标 7 和 9）。这些数据此前已做质量检查和开发分析，结果属于回顾性开发证据。三混合固定为 NN source＋波形 LDA＋xDAWN。</p>'
    document += '<p><a href="results.json">完整计分与来源</a> · <a href="decisions.json">逐决策结果</a> · <a href="../checkpoint.json">运行状态</a></p></main></html>'
    path.write_text(document, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.run, args.output)
    print(json.dumps({"status": result["status"], "oracles": result["independent_sum_logit_oracles"]}))


if __name__ == "__main__":
    main()
