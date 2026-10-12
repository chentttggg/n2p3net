# 指标分母与字段审阅交接

## 交付范围

本交付针对主树 `b26c099` 的指标层和数据分析层，只读取已封存数据分析 JSON/CSV、接受绑定、公共证据文档和 schema 源码。没有加载模型，没有调用 fit、score、EEG reader 或训练。

输出目录：`tmp/METRIC_SCOPE_AUDIT_20261012/`

- `metric_crosswalk.json`：数据源、schema、scope、requested/participant/probability 分母、聚合顺序、拟合/评分调用数、允许声明和字段拒绝规则。
- `p1_error_ledger.json` / `p1_error_ledger.csv`：P1 16 cell 行级字段、producer/consumer/noise SHA、winner、p_true、NLL、Brier、误差预算和覆盖状态。
- `REPORT.zh.md`：中文结果和缺口。
- `build_metric_audit.py`：无第三方数据依赖的复算脚本。

## 实际来源

P1 使用 `tmp/P1_NUMERIC_PANEL/analysis/final/analysis.json` 与 `doc/assets/research_resume_20261011/p1_numeric_panel_acceptance_binding.json`。它是 8 人、shared8/private8 各 8 个 held decision、36-way candidate、每人一个 held decision 的 C-held development diagnostic；真值全部为 `row:5|column:0`。

公共 BNCI 使用 `doc/evidence/public_evaluation.zh.md` 和 `cloud_candidate_analysis_binding.json`：8 人、152 requested、144 probability units、一个 seed。BI2014a 使用 `bi_q1_queue_acceptance_binding.json`：61 人、1287 requested、1243 probability units、3 arms。BNCI C600/R10 使用 `bnci_q1_queue_acceptance_binding.json`：8 人、240 requested per arm、3 arms、100% probability coverage；它与 P1 和 label-K prefix selector 分开。

通用字段语义来自 `src/models/candidate_evidence.py` 的 `n2p3_candidate_probability_evidence/1` 和 `src/research/candidate_metrics.py` 的 `n2p3_candidate_probability_metrics/1,/2`。

## 复算结论

P1 shared8 为 5/8、NLL 7.3364765833、Brier 0.6790937536；private8 为 6/8、NLL 5.5540121911、Brier 0.4776785010。participant-macro Δhit 为 0.125，ΔNLL 为 -1.7824643922，ΔBrier 为 -0.2014152525，救回 1 人、损害 0 人。

P1 paired 表的 8 个逐参与者 ΔNLL 直接求和为 -14.2597151377。声明的 -1.7824643922 是先在参与者内平均再跨参与者平均，不能用行和替代。A01 贡献行和改善的 71.8229%。

## 字段安全边界

P1 自定义 `status=scored` 不能直接映射到 generic `CandidateEvidence.status`；P1 `hits/operational_hit` 只有在“一人一个 held decision”条件下才可映射到 participant operational hit。P1 `nll/brier` 需要保留 36-way joint vector、truth 和 participant-first aggregation；`probability_available` 不代表 calibration；`all_input_certificate=false` 不是 scoring failure；row/column marginal 不能替代 joint candidate probability。

## 当前缺口

P1 仍不是公共效能或成人确认。B07 central numerical owner 已接受，但 history consumer、capsule replay 和 public B07 score 仍未完成。下一步先审阅并保留本 crosswalk/ledger，待 history consumer 接线后重新生成其 source/consumer binding；不要把 P1 custom 字段写回 generic v1 artifact。

## 复算命令与状态

```text
uv run --offline python tmp/METRIC_SCOPE_AUDIT_20261012/build_metric_audit.py
uv run --offline python -m py_compile tmp/METRIC_SCOPE_AUDIT_20261012/build_metric_audit.py
git diff --check
```

复算脚本 SHA-256：`2a52176cc68d2a960c2b538f04143738a2d7b1982bfd1df747bf60d588a33381`。

本交付未启动训练、fit、score、模型加载或原始 EEG 读取。
