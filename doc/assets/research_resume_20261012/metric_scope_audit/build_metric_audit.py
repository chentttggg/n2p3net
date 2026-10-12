from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp" / "METRIC_SCOPE_AUDIT_20261012"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source(path: str, **fields):
    p = ROOT / path
    result = {"path": path, "sha256": sha(p), "bytes": p.stat().st_size}
    result.update(fields)
    return result


p1 = load("tmp/P1_NUMERIC_PANEL/analysis/final/analysis.json")
p1_rows = p1["rows"]
p1_groups = defaultdict(list)
for row in p1_rows:
    p1_groups[row["arm"]].append(row)
p1_summary = {}
for arm, rows in sorted(p1_groups.items()):
    p1_summary[arm] = {
        "requested_participants": len({r["subject"] for r in rows}),
        "requested_cells": len(rows),
        "hits": sum(bool(r["correct"]) for r in rows),
        "operational_hit": sum(bool(r["correct"]) for r in rows) / len(rows),
        "prediction_coverage": sum(bool(r["operational_available"]) for r in rows) / len(rows),
        "probability_coverage": sum(bool(r["probability_available"]) for r in rows) / len(rows),
        "nll_macro_one_decision_per_participant": sum(float(r["nll"]) for r in rows) / len(rows),
        "brier_macro_one_decision_per_participant": sum(float(r["brier"]) for r in rows) / len(rows),
        "candidate_count": sorted({len(r["probabilities"]) for r in rows}),
        "truths": sorted({r["truth"] for r in rows}),
        "fit_calls": sum(int(r["fit_calls"]) for r in rows),
        "scorer_calls": sum(int(r["scorer_calls"]) for r in rows),
    }
paired = sorted(p1["paired"], key=lambda row: row["subject"])
paired_recomputed = {
    "participants": len(paired),
    "delta_hit_macro": sum(float(r["delta_hit_private_minus_shared"]) for r in paired) / len(paired),
    "delta_nll_macro": sum(float(r["delta_nll_private_minus_shared"]) for r in paired) / len(paired),
    "delta_nll_sum_row_values": sum(float(r["delta_nll_private_minus_shared"]) for r in paired),
    "delta_brier_macro": sum(float(r["delta_brier_private_minus_shared"]) for r in paired) / len(paired),
    "rescued": sum(bool(r["rescued"]) for r in paired),
    "harmed": sum(bool(r["harmed"]) for r in paired),
    "probability_pairs": sum(bool(r["paired_probability_available"]) for r in paired),
}
paired_recomputed["A01_share_of_row_sum_nll_improvement"] = (
    float(next(r["delta_nll_private_minus_shared"] for r in paired if r["subject"] == "A01"))
    / paired_recomputed["delta_nll_sum_row_values"]
)

p1_binding = load("doc/assets/research_resume_20261011/p1_numeric_panel_acceptance_binding.json")
cloud_binding = load("doc/assets/research_resume_20261011/cloud_candidate_analysis_binding.json")
bi_binding = load("doc/assets/research_resume_20261011/bi_q1_queue_acceptance_binding.json")
bnci_binding = load("doc/assets/research_resume_20261011/bnci_q1_queue_acceptance_binding.json")

sources = [
    source("tmp/P1_NUMERIC_PANEL/analysis/final/analysis.json", schema=p1.get("schema"), role="P1 row-level readout", scope=p1.get("scope")),
    source("doc/assets/research_resume_20261011/p1_numeric_panel_acceptance_binding.json", schema=p1_binding.get("schema"), role="P1 producer/consumer acceptance", scope=p1_binding.get("scope", "original_C600_R10_one_held_decision_per_person")),
    source("doc/evidence/public_evaluation.zh.md", schema="markdown-evidence", role="BNCI public candidate panel", scope="BNCI2014-008 eight people, seed 20260918, 152 requested / 144 probability units"),
    source("doc/assets/research_resume_20261011/cloud_candidate_analysis_binding.json", schema=cloud_binding.get("schema"), role="cloud candidate analysis binding", scope="eight people, 152 requested / 144 common probability decisions"),
    source("doc/assets/research_resume_20261011/bi_q1_queue_acceptance_binding.json", schema=bi_binding.get("schema"), role="BI2014a Q1 queue binding", scope=bi_binding.get("scope"), participants=bi_binding.get("participants"), requested_decisions=bi_binding.get("main_requested_keys"), probability_decisions=bi_binding.get("main_available_keys")),
    source("doc/assets/research_resume_20261011/bnci_q1_queue_acceptance_binding.json", schema=bnci_binding.get("schema"), role="BNCI C600/R10 binding", scope=bnci_binding.get("scope"), participants=bnci_binding.get("participants"), requested_per_arm=bnci_binding.get("requested_per_arm"), arms=3, probability_coverage=bnci_binding.get("coverage_per_arm")),
    source("src/models/candidate_evidence.py", schema="n2p3_candidate_probability_evidence/1", role="generic evidence schema source"),
    source("src/research/candidate_metrics.py", schema="n2p3_candidate_probability_metrics/1 and /2", role="generic metrics and paired comparison source"),
]

crosswalk = {
    "schema": "n2p3_metric_scope_crosswalk/1",
    "status": "passed_read_only_recompute",
    "generated_from_main_head": "b26c099",
    "source_files": sources,
    "denominator_crosswalk": [
        {"name": "P1_shared8/private8", "requested_participants": 8, "requested_cells_per_arm": 8, "probability_cells_per_arm": 8, "candidate_count": 36, "truth_support": ["row:5|column:0"], "aggregation": "one held decision per participant; equal participant mean", "fit_calls": 0, "score_calls": 0, "claim_scope": "C-held development diagnostic only"},
        {"name": "BNCI_public_candidate_panel", "requested_participants": 8, "requested_decisions": 152, "probability_decisions": 144, "candidate_count": 36, "aggregation": "participant-first NLL over available common decisions; requested operational hit", "fit_calls": 0, "score_calls": 0, "claim_scope": "public calibration development, one seed"},
        {"name": "BI2014a_Q1", "requested_participants": 61, "requested_decisions": 1287, "probability_decisions": 1243, "candidate_count": 36, "aggregation": "participant macro plus micro coverage separately; 44 incomplete retained", "fit_calls": 0, "score_calls": 0, "claim_scope": "public calibration development, not adult/product"},
        {"name": "BNCI_C600_R10", "requested_participants": 8, "requested_decisions_per_arm": 240, "probability_decisions_per_arm": 240, "arms": 3, "candidate_count": 36, "aggregation": "participant macro equals micro because each participant has 30 R10 episodes", "fit_calls_in_original_run": 16, "new_fit_score_model_load_calls_in_review": 0, "claim_scope": "public calibration development; separate from P1 and label-K selector"},
    ],
    "p1_recomputed": p1_summary,
    "p1_paired_recomputed": paired_recomputed,
    "generic_schema_rejection_rules": [
        {"custom_field": "P1 status=scored", "generic_target": "CandidateEvidence.status", "decision": "reject_direct_import", "reason": "generic statuses are available/tie/fit_failure/not_available/incomplete/abstain and require schema validation"},
        {"custom_field": "P1 hits/operational_hit", "generic_target": "requested_participant_operational_hit_rate", "decision": "conditional_mapping_only", "reason": "valid only with one held decision per participant and explicit P1 scope"},
        {"custom_field": "P1 nll/brier", "generic_target": "available_probability_nll/available_probability_brier", "decision": "conditional_mapping_only", "reason": "requires same 36-way distribution, target identity, participant-first aggregation and probability denominator"},
        {"custom_field": "P1 probability_available", "generic_target": "CandidateEvidence.probability_available", "decision": "conditional_mapping_only", "reason": "boolean availability does not certify calibration or public scope"},
        {"custom_field": "P1 all_input_certificate", "generic_target": "failure/status", "decision": "reject_direct_import", "reason": "false means local normwise certificate is not an all-input interval proof; it is not a failed score"},
        {"custom_field": "P1 probability_mass_of_true_row/column", "generic_target": "log_probability", "decision": "reject_direct_import", "reason": "diagnostic marginals cannot replace the 36-way joint candidate vector"},
        {"custom_field": "P1 paired delta_nll row values", "generic_target": "CandidateProbabilityComparison.delta_paired_nll", "decision": "aggregate_before_mapping", "reason": "row sum is -14.259715; declared participant macro delta is -1.782464"},
    ],
    "claims": {
        "current_allowed": [
            "P1 private8 has 6/8 versus shared8 5/8, delta hit +0.125 and participant-macro delta NLL -1.782464 on the fixed eight held decisions",
            "P1 is fully available numerically but is not public efficacy or adult confirmation",
            "central B07 numerical owner is accepted separately; history consumer/replay and public B07 score remain pending",
        ],
        "current_forbidden": [
            "P1 5/8 or 6/8 as BNCI public 152-request hit@all",
            "P1 custom scored rows as generic CandidateEvidence v1 records",
            "P1 fixed truth row/column panel as balanced 36-candidate risk",
            "P1 local density error estimate as an all-input numerical interval certificate",
        ],
    },
}

ledger_fields = [
    "subject", "arm", "status", "truth", "winner", "correct", "operational_available", "probability_available",
    "candidate_count", "truth_rank", "p_true", "nll", "brier", "winner_confidence", "top_two_gap_nats",
    "max_density_error_estimate_nats", "density_error_budget_nats", "density_error_within_budget", "all_input_certificate",
    "fit_calls", "scorer_calls", "read_support", "model_producer_sha256", "noise_state_sha256", "consumer_source_sha256",
]
ledger = []
for row in sorted(p1_rows, key=lambda r: (r["subject"], r["arm"])):
    item = {key: row.get(key) for key in ledger_fields}
    item["candidate_count"] = len(row["probabilities"])
    item["density_error_within_budget"] = float(row["max_density_error_estimate_nats"]) <= float(row["density_error_budget_nats"])
    ledger.append(item)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "metric_crosswalk.json").write_text(json.dumps(crosswalk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUT / "p1_error_ledger.json").write_text(json.dumps({"schema": "n2p3_p1_row_error_ledger/1", "source_sha256": sha(ROOT / "tmp/P1_NUMERIC_PANEL/analysis/final/analysis.json"), "rows": ledger, "paired": paired, "recomputed": paired_recomputed}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (OUT / "p1_error_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=ledger_fields)
    writer.writeheader()
    writer.writerows(ledger)

report = f"""# 指标分母与字段审阅（只读复算）

生成于主树 `b26c099`。本交付只读取已封存 JSON、CSV、binding、证据文档和指标 schema 源码；没有模型加载、fit、score 或 EEG 读取。

## 已复算

P1 16 cell 底表有 8 个参与者、shared8/private8 各 8 个 held decision、36 个候选，两个方案概率和预测覆盖均为 100%。shared8 命中 5/8、NLL 7.3364765833、Brier 0.6790937536；private8 命中 6/8、NLL 5.5540121911、Brier 0.4776785010。配对 delta hit 的参与者宏平均为 0.125，配对 delta NLL 的参与者宏平均为 -1.7824643922；8 行 delta NLL 直接求和为 -14.2597151377，不能把行和当作声明的宏平均。A01 行占该行和改善约 71.8%。

P1 每人只有一个 held decision，因此 participant macro 与 decision micro 恰好相等。这个偶合关系不能推广到公共 BNCI 的 152 requested decisions、BI2014a 的 1287 requested episodes 或 BNCI C600/R10 的 240 requested decisions per arm。

## 字段边界

P1 使用 `independent_p1_numeric_panel_readout/1` 自定义 schema。`status=scored`、`hits`、`nll`、`brier`、`probability_available`、行列边缘概率和局部 density error 字段不能直接当作 `CandidateEvidence` 或 `CandidateProbabilityMetrics` v1 字段。映射前必须保留源 schema、候选数、真值、分母和聚合顺序。

`all_input_certificate=false` 表示误差估计不是全输入 interval certificate，不表示评分失败。P1 的 row/column probability 是 36-way joint 分布的诊断边缘，不得替代 joint `p_true`。

## 当前结论与缺口

P1 private8 在固定八人、固定 held、固定 36-way truth support 上改善了命中和 participant-macro NLL；它是 C-held development diagnostic。公共 BNCI、BI2014a、BNCI C600/R10 各自有独立 requested/probability 分母和 source/producer；它们不能合并成一个总体指标。B07 central numerical owner 已单独绑定，history consumer、capsule replay 和 public B07 score 仍是缺口。

交付文件：`metric_crosswalk.json`、`p1_error_ledger.json`、`p1_error_ledger.csv`。重新运行同目录的 `build_metric_audit.py` 可复算。
"""
(OUT / "REPORT.zh.md").write_text(report, encoding="utf-8")
