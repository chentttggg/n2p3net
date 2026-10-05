"""Recover wishlist coverage and explicit research dependencies from the source text."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def task(identifier, title, priority, dependencies, owner, question, evidence, falsifier):
    return {
        "id": identifier, "title": title, "priority": priority,
        "depends_on": dependencies, "owner_files": owner,
        "research_question": question, "exit_evidence": evidence,
        "falsification": falsifier, "design_status": "complete",
        "implementation_status": "not_started", "execution_status": "not_started",
    }


TASKS = [
    task("B00", "比较身份、评价与训练内选择", "P0", [],
         ["src/research/comparison.py", "src/research/contracts.py", "src/stats/transfer.py"],
         "如何让 hit 优先且失败、概率风险和多重选择都可解释？",
         ["固定 requested participant/decision roster", "同批配对与 C 内 nested OOF", "hit/NLL/coverage 门槛与 cluster CI"],
         "删掉失败项即可提高主指标、或选择器读取外层 held 标签，即合同失败。"),
    task("B01", "信号履历与观测支持", "P0", [],
         ["src/data/contract.py", "src/data/preprocess.py", "src/data/raw_artifacts.py", "src/transfer/raw_continuous.py"],
         "模型实际观察哪些物理样本、滤波层级和事件历史？",
         ["MAT/CSV/ADC 层级与 upstream unknown 账本", "左右边界/时钟/可用时间", "物理支持和专家各自数值视图"],
         "读取下一未授权 flash 的尾响应、或把 MAT 称 ADC raw，均不能通过。"),
    task("B02", "专家 Interface 与神经候选证据", "P0", ["B00", "B01"],
         ["src/models/candidate_evidence.py", "src/data/candidate_task.py", "src/transfer/calibrated.py", "src/research/evaluation.py"],
         "如何让不同专家对同一完整候选任务输出可比较概率？",
         ["membership log-odds 推导", "专家独立数值身份/共同物理任务", "冻结 candidate vocabulary 与无标签 score", "保存重载"],
         "把 binary AUC 当 posterior、候选缺失后缩词表、失败专家静默删掉，均失败。"),
    task("B03", "C 内概率校准、等权锚点及 gating", "P0", ["B02"],
         ["src/models/candidate_evidence.py", "src/transfer/calibration_split.py", "src/transfer/quality.py"],
         "概率平均是否提供独立互补性，温度与 gating 是否能在 C 内识别？",
         ["固定三/四专家算术概率平均", "nested grouped OOF 后 refit", "fixed weight 与 calibration-only gating 分开", "单专家失败行为"],
         "等权增加专家也可降低 hit；不能据 convex NLL 上界承诺不退步。"),
    task("B04", "神经贡献与同结构 scratch 配对", "P0", ["B03"],
         ["experiments/run_erp_comparator.py", "src/transfer/target_experiments.py", "src/transfer/recipes.py", "src/transfer/checkpoint.py"],
         "神经专家带来源迁移、训练剂量还是独立互补信息？",
         ["NN/full-fine/scratch/成熟锚点同物理支持", "六个预声明混合对照", "K0 可达性与 source 排除", "逐参与者失败/负迁移"],
         "没有 matched scratch、source 包含目标参与者/库、或将跨输入差值称迁移收益，即归因失败。"),
    task("B05", "公平 AR 阶数及逐通道谱", "P1", ["B00", "B01"],
         ["src/transfer/temporal_noise.py", "src/transfer/exact_covariance.py", "src/transfer/continuous_decision.py"],
         "白化真实噪声的收益是否超过对白化判别方向的损伤？",
         ["AR0/1/4/8/16/32 的 hold_back=32", "stationary 与 conditional 两独立面板", "Y/D 同步白化", "train/held ACF PSD 创新与判别信息"],
         "只缩短 Y 不变 D、各阶评分行不同、或 ACF 改善却当成 hit 改善，即失败。"),
    task("B06", "边界条件均值、协方差与完整保真", "P1", ["B00", "B01"],
         ["src/data/finite_filter_operator.py", "src/transfer/noise_model.py", "experiments/boundary_state_research.py"],
         "有限观察下的外部状态可识别到什么程度？修正是否保留实际 ERP 增量？",
         ["确定 padding 与外部 latent/估计误差分离", "候选相关条件均值及 Schur covariance", "共同输出网格", "完整 T(x+h) 与白噪声 gain", "仅 C 内选距离 schedule"],
         "同一 short 可由不同外部延拓产生不同 delta，不能宣称唯一恢复；输出 ERP overlap 不是伤害证明。"),
    task("B07", "ERP 字典、延迟和完整事件历史", "P1", ["B00", "B01"],
         ["src/transfer/continuous_decision.py", "src/baselines/temporal_transfer.py", "src/data/events.py"],
         "低维基、潜伏期与邻近事件是否解释当前设计失配？",
         ["Gaussian/B-spline/cosine/导数等自由度对照", "common/target/selection 识别约束", "完整卷积事件与删失", "raw/filtered basis 同步传播"],
         "用真实未知邻近目标作 oracle、变基同时增加有效自由度、或 selection effect 吸收 q，均失败。"),
    task("B08", "尺度一致 g-prior、层级先验和模型平均", "P1", ["B00", "B01"],
         ["src/transfer/continuous_decision.py", "src/transfer/noise_model.py", "src/models/parameter_groups.py"],
         "按白化设计信息定义先验能否区分尺度 gauge 与真实模型改变？",
         ["g-prior 换基/电压/MN gauge 反例", "可识别预测子空间与 rank 账本", "common/selection 分层收缩", "fixed lambda posterior mixture 与 Bayes mixture 分开"],
         "机械 pinv 给未识别系数确定 prior、用 held 选 lambda、或把改 AR 阶数当 gauge，即失败。"),
    task("B09", "空间 ERP 均值与实测布局", "P1", ["B00", "B01"],
         ["src/baselines/structured_erp.py", "src/baselines/riemann.py", "src/baselines/temporal_transfer.py", "src/data/channel.py"],
         "Pz 信息应进入均值/空间判别而非人为压低噪声吗？",
         ["xDAWN/空间 LDA/ERP geometry 锚点", "收缩 d-prime 与 graph 均值 prior", "native/ref/channel mask 等秩对照", "participant 稳定性"],
         "目标 d-prime 使用 held 标签、补零成实测导、reference 奇异秩未处理，均失败。"),
    task("B10", "输入算子和污染分机制对照", "P1", ["B00", "B01"],
         ["src/data/harmonic.py", "src/data/ocular.py", "src/data/recovery.py", "src/data/preprocess.py"],
         "改滤波/工频/EOG 是否改善任务，并满足部署可用时间？",
         ["因果/双向/selection 独立协议", "高通低通 notch 与相位分轴", "漂移/幅相调制/glitch/慢波独立挑战", "改变 raw 截止与只改读窗分开"],
         "用未来长记录修正短测试、把自适应参数视作固定 F、或仅峰下降就晋升，均失败。"),
    task("B11", "动态、重尾和联合状态模型", "P2", ["B05", "B06", "B07", "B08", "B09"],
         ["src/transfer/temporal_noise.py", "src/transfer/continuous_decision.py", "src/transfer/noise_model.py"],
         "separable MN 明确失配后，动态因子/VAR/GP/重尾层级哪个可识别？",
         ["可观测性与噪声/ERP 辨识", "channel 私有谱/共享因子", "Student-t 层级与单位", "filtering/smoothing 延迟合同"],
         "算力不能弥补有限 participant 和决策数；不识别/负迁移必须保留。"),
    task("B14", "工程收敛、源绑定、吞吐和干净交付", "P0", ["B00", "B01"],
         ["experiments/freeze_source_snapshot.py", "src/research/source_execution.py", "src/research/execution.py", "src/transfer/result_reuse.py", "src/train/runtime.py"],
         "如何一次验证外部输入、精简内部防御，并将实际计算资源送到 worker？",
         ["已验证不可变内部状态/无静默兼容兜底", "实际研究 helper 与 entrypoint 的源摘要", "CPU>=25 GPU workers>=4", "共享只读 cache 与滚动 ready DAG", "失败账本与净 Git"],
         "防御删减造成身份/fit/时间守卫丢失、计划写4而实际1、或只量 kernel 不量整链，均失败。"),
    task("B12", "四库与三 seed 完整流程复核", "P0", ["B04", "B14"],
         ["experiments/run_source_pool_matrix.py", "experiments/run_public_pool_targets.py", "experiments/verify_pool_matrix.py", "experiments/run_erp_comparator.py"],
         "预声明混合在新配对开发支持中是否复现，并降低校准成本？",
         ["四库 binary 面板 + 合法 membership candidate 面板", "三个完整 source→target seed", "K0/200/800/2000 请求与实际 ledger", "公共候选与成人产品边界"],
         "三 seed 算三倍参与者、四库伪造 candidate posterior、或只做有利 pilot 就停，均失败。"),
    task("B13", "独立成人确认与产品成本", "P2", ["B12"],
         ["src/transfer/calibrated.py", "experiments/brainsync_model_task.py", "doc/workstation.zh.md"],
         "冻结选择/概率校准/程序后，新成人九选一 hit@all 和成本是否达标？",
         ["新采集/光学时钟/未知目标/跨日协议", "事前估计量与样本量", "一份包只读推理与真实停止/延迟", "预声明效能和风险门槛"],
         "旧 P001、ALS BNCI 36选一或合同 fixture 均不能充当独立成人确认。"),
]

# One assignment per original list item. Secondary tasks capture real interactions.
ASSIGNMENTS = {
    1: [("B01",), ("B01",), ("B01",), ("B10", "B01"), ("B10",), ("B10",),
        ("B06", "B10"), ("B06",), ("B06",), ("B07", "B06"), ("B10", "B01"),
        ("B06", "B10"), ("B09", "B10")],
    2: [("B06",)] * 3 + [("B11", "B06")] + [("B06",)] * 5
       + [("B07", "B06"), ("B06",), ("B06",)],
    3: [("B07",)] * 11,
    4: [("B05",)] * 6 + [("B11", "B05")] * 5 + [("B05",)] * 3,
    5: [("B09",)] * 9 + [("B03", "B09")],
    6: [("B08",)] * 11,
    7: [("B02", "B04")] * 8 + [("B03",)] * 5 + [("B04",)] * 6 + [("B04",)] * 4,
    8: [("B00",)] * 7,
    9: [("B12",)] * 5 + [("B00",)] * 3 + [("B07", "B12"), ("B09", "B12"),
        ("B00", "B03", "B13"), ("B12", "B13")],
    10: [("B02", "B04"), ("B02", "B09"), ("B02", "B05"), ("B03",),
         ("B03",), ("B00",), ("B00", "B06"), ("B06",), ("B09",), ("B12", "B13")],
}


def source_items():
    rows = []
    group = 0
    number = 0
    section = ""
    for line_number, line in enumerate((HERE / "wishlist_source.zh.md").read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("这份愿望清单已经保存"):
            break
        if re.match(r"^\*\*[一二三四五六七八九十]+、", line):
            group += 1
            number = 0
            section = line.strip("*")
        if group and (line.startswith("- ") or re.match(r"^\d+\. ", line)):
            number += 1
            rows.append({"id": f"W{group:02d}.{number:02d}", "section": section,
                         "source_line": line_number, "text": re.sub(r"^(?:- |\d+\. )", "", line),
                         "tasks": list(ASSIGNMENTS[group][number - 1])})
    actual = Counter(int(row["id"][1:3]) for row in rows)
    if actual != Counter({key: len(value) for key, value in ASSIGNMENTS.items()}):
        raise ValueError("wishlist source and explicit task assignments differ")
    return rows


def build():
    source_config = json.loads((ROOT / "configs/source_pool_expansion.example.json").read_text(encoding="utf-8"))
    cloud = json.loads((ROOT / "configs/cloud_training_resources.json").read_text(encoding="utf-8"))
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ("wishlist_source.zh.md", "four_week_source.zh.md")}
    return {
        "schema": "n2p3_deep_research_design/1", "evidence_scope": "research_design_and_math",
        "design_date": "2026-10-05", "production_default_changed": False,
        "source_thread": "01a0eb70-bcfb-7a92-b22b-1bfd6b541ddc", "source_text_sha256": sources,
        "source_snapshot_commit": "dc506ca0c90f9e6904376bfdb5ae5f2f8ebc7493",
        "wishlist": source_items(), "tasks": TASKS,
        "experiment_contract": {
            "seeds": source_config["seeds"], "requested_K": [0, 200, 800, 2000],
            "binary_targets": list(source_config["targets"]),
            "candidate_adapters": {"BNCI2014-008": "source MAT, continuous and epoch adapters",
                                   "BI2014a": "raw CSV membership recovery; continuous adapter to be validated"},
            "other_targets_candidate_status": "require public membership and complete decision attestation; no inference from AUC",
            "historical_bnci_design": {"participants": 8, "calibration_authorization_K": 800,
                                       "group_count": 6, "flashes_per_group": 120,
                                       "fit_groups_per_outer_fold": 5, "actual_fit_labels": 600,
                                       "held_groups_per_outer_fold": 1, "outer_fold_count": 48,
                                       "unused_authorized_labels": 80,
                                       "role": "already_observed_calibration_development"},
            "selection": "nested grouped calibration OOF; outer-held labels never used for selection",
            "primary_candidate_metric": "requested-participant macro operational hit@all",
            "binary_estimand": "four-dataset equal macro paired delta AUC; candidate hit is separate",
            "development_gates": {"paired_delta_hit_min": 0.0, "paired_delta_nll_max": 0.0,
                                  "coverage_delta_min": 0.0, "failure_count_increase_max": 0,
                                  "require_no_new_missing_baseline_success": True,
                                  "nll_pairing": "same predeclared roster, paired-complete NLL plus explicit missing ledger"},
            "small_K_selection": "insufficient independent groups: use a source/predeclared frozen calibration program or mark learned calibration unavailable; no random-trial split",
            "formal_confirmation": "not declared; adult population, sample size and risk margins require new protocol",
            "baseline_experts": ["raw_ar8_lambda_1", "toeplitz", "xdawn"],
            "added_expert": "n2p3net_full_fine", "mixing": "fixed arithmetic probability average",
            "core_mixture_arms": ["neural_only", "classic_three", "neural_toeplitz", "neural_xdawn",
                                  "neural_classic_three", "scratch_classic_three"],
            "ar_orders": [0, 1, 4, 8, 16, 32], "ar_hold_back_samples": 32,
            "ar_startup_panels": ["stationary marginal common output tail", "conditional actual prefix common output tail"],
            "source_pools": source_config["source_pools"],
            "primary_source_pool": "B7_plus_three", "secondary_source_pool": "B7",
            "source_exclusion": "whole held target dataset including its participants; family sensitivity separately",
            "window_and_filter_protocols": "declared contrast axes; no universal EA/window serial prerequisite",
        },
        "resource_contract": {"cloud_config": "configs/cloud_training_resources.json", "cloud": cloud,
                              "capacity_vs_tail": ">=4 capacity; record ready fits in dependency tail",
                              "cpu_partition_at_4_workers": [6, 6, 6, 6, 1],
                              "launch_status": "not_launched_by_design_task"},
        "variant_dependencies": {"boundary_corrected": ["B06"], "alternative_ar": ["B05"],
                                 "alternative_basis": ["B07"], "gprior": ["B08"],
                                 "spatial_prior": ["B09"], "adaptive_filter": ["B10"],
                                 "dynamic_or_robust_noise": ["B11"]},
        "week_deliverables": {
            "week1": ["B00", "B01", "B02", "B03", "B04", "B14"],
            "week2_parallel": ["B05", "B06", "B07", "B08", "B09", "B10"],
            "week3": ["B04", "B12"], "week4": ["B12", "B14"],
            "research_extensions": ["B11", "B13"],
            "calendar_rule": "acceptance milestones, not evidence of execution; do not wait for all optional research variants",
        },
    }


if __name__ == "__main__":
    output = HERE / "research_plan.json"
    plan = build()
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tasks = {entry["id"]: entry for entry in plan["tasks"]}
    lines = ["# 原愿望清单覆盖表", "", "从原文逐条恢复，共 123 项；含原优先主线的重复项，保持来源可追溯。",
             "任务为方案层完成，实施与新性能矩阵均未启动。每项的实验、数学和失败条件见对应分块及独立研究报告。",
             "", "| 原项 | 原文 | 负责块及交互 | 优先级 |", "|---|---|---|---|"]
    for row in plan["wishlist"]:
        text = row["text"].replace("|", "\\|")
        assigned = ", ".join(row["tasks"])
        priority = tasks[row["tasks"][0]]["priority"]
        lines.append(f"| {row['id']} | {text} | {assigned} | {priority} |")
    (HERE / "coverage.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"plan": output.name, "wishlist_items": len(source_items()), "tasks": len(TASKS)}))
