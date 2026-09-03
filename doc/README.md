# N2P3-Net 文档控制索引

_版本：2026-09-03 · 活动源提交：`ef7a40e` · 用途：统一研究、工程与证据文档的权威层级_

---

## 📋 唯一活动入口

从本文件进入文档集合后，唯一可以改变新研究设计、代码任务或结论边界的规范是
[`research_program.zh.md`](research_program.zh.md)。
[`input_contract_math.zh.md`](input_contract_math.zh.md) 是其输入/时间轴的绑定补充合同。
其余文档提供决策理由、数学推导、实施验收或历史证据，不能单独授权新的实验或覆盖活动规范。

新工作按以下顺序阅读：

1. 本索引，确认文档角色与活动基线
2. 活动规范和输入合同，确认 estimand、数据边界与不可违反的约束
3. 与任务相关的决策、实施、数学或验收记录
4. 当前源码、测试和配置，核对文档陈述是否仍与实现一致

## 🔗 文档权威关系

```mermaid
flowchart TB
    accTitle: N2P3 Documentation Authority
    accDescr: The active research program and input contract govern new work, while decision, implementation, audit, and historical documents supply traceable supporting evidence.

    doc_index([📚 Read document index]) --> research_program[📋 Read active research program]
    research_program --> input_contract[⚙️ Apply input contract]
    research_program --> decision_records[📝 Consult decision records]
    decision_records --> implementation_records[🔧 Consult implementation and math]
    implementation_records --> source_tests[🧪 Inspect source and tests]
    source_tests --> acceptance_record[✅ Check framework acceptance]
    acceptance_record --> experiment_plan[📦 Create new attested plan]
    historical_records[🗂️ Historical records] -.->|evidence only| research_program

    classDef authority fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef record fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef evidence fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef action fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    class doc_index,research_program,input_contract authority
    class decision_records,implementation_records,historical_records record
    class acceptance_record evidence
    class source_tests,experiment_plan action
```

## 📚 文档角色

| 类别 | 文件 | 权威范围 | 使用限制 |
|---|---|---|---|
| 活动规范 | [`research_program.zh.md`](research_program.zh.md) | 研究问题、估计量、门禁、反例和下一步 | 唯一活动指导文件 |
| 绑定合同 | [`input_contract_math.zh.md`](input_contract_math.zh.md) | EEG 时间轴、单位、预处理和分组 | 不得以实验便利绕过 |
| 工程流程 | [`CODING_WORKFLOW.md`](CODING_WORKFLOW.md) | 本地开发、测试与提交习惯 | 与活动规范冲突时以后者为准 |
| 决策与推导 | [`architecture_decision_20260902.zh.md`](architecture_decision_20260902.zh.md)、[`implementation_plan_20260902.zh.md`](implementation_plan_20260902.zh.md)、[`math_verification_20260902.zh.md`](math_verification_20260902.zh.md) | 方案为何成立、如何落实、反例为何必须保留 | 记录，不是新的派工单 |
| 框架验收 | [`framework_acceptance_20260903.zh.md`](framework_acceptance_20260903.zh.md)、[`supervisor_full_audit_20260903.zh.md`](supervisor_full_audit_20260903.zh.md) | `ef7a40e` 的代码链、测试、独立复跑和开放风险 | 仅 framework/contract evidence |
| 研究依据 | [`feature_domain_transfer_research_20260902.zh.md`](feature_domain_transfer_research_20260902.zh.md)、[`literature/`](literature) | 文献与机制假设 | 不替代当前性能证据 |
| 历史文档 | [`../frozen/document_history_through_ef7a40e_20260903.manifest.json`](../frozen/document_history_through_ef7a40e_20260903.manifest.json) | 已完成派工、旧 cloud staging 计划和集成前审计 | 仅证据，不得直接执行 |
| 历史结果 | [`research_status_report_20260902.zh.md`](research_status_report_20260902.zh.md)、[`research_status_report_20260831.zh.md`](research_status_report_20260831.zh.md) | 既有 development/evidence 边界 | 不得升级为 confirmation |

## 🎯 当前活动基线

代码框架已提交为 `ef7a40e feat: add split-bn residual and decision-loss contracts`。
该提交闭合的是配置到 checkpoint 的软件/合同链，不是跨域性能或产品效果证据。

| 机制 | 当前软件状态 | 仍需的科研证据 |
|---|---|---|
| P0 Split-BN | source-domain buffer、目标 buffer selector 和 checkpoint 绑定已实现 | 成对 fold/seed、cluster CI 和 buffer-swap 机制证据 |
| P1 drop-to-consensus | source-only residual、counterfactual ledger 和 invalid-run gate 已实现 | 真实 source/target 迁移与 held-out operational hit@R |
| P2 direct-decision loss | direct-membership metadata、完整 decision batch 与训练合同已实现 | 真实 direct timeline 训练；row/column protocol dispatch 仍未实现 |

以下结论仍不得作出：feature transfer 有效、decision learning 有效、成人 BrainSync 达到 90%，或 target-calibration 已可运行。

高负载测试异常已由受控实验定性为 WorkBuddy 会话环境故障，不是 EEG/契约源码缺陷：
bulk-delete、trash、symlink shim，以及被负载放大的 15ms 并发测试窗口共同造成异常。
因此 WorkBuddy 会话内的高负载 pytest 结果不作为验收信号；独立 shell/云端或 unset 守卫变量
后的复跑才可采信。详见 [`supervisor_full_audit_20260903.zh.md`](supervisor_full_audit_20260903.zh.md)。

## 🔄 下一步门禁

1. 为实际执行树生成新的 source-only archive 与 manifest；Git commit 不是物理冻结
2. 人工决定是否清理 `%TEMP%\pytest-of-chent`；WorkBuddy 会话内不采信高负载测试，清理前后的验收均使用空闲、单任务或独立 shell
3. 准备经验证的 active source/target cache，并按 direct 与 row/column protocol 分流
4. 先完成 plan、environment、cache 与 checkpoint 的 attestation/dry-run，再启动单 worker CUDA smoke
5. 只在预声明的 matched fold/seed 后报告 paired operational `hit@R` 与 participant-cluster CI
6. 在模型级 target-calibration collector、buffer provenance 与结果 ledger 闭合前，保持该 arm fail-closed

## 🗂️ 文档维护规则

- 重大架构或合同变动同时更新活动规范、实现/验收记录和本索引
- 新实验计划必须绑定当前 source archive、环境、cache、checkpoint 与计划 digest
- 过往派工、审计和结果保留为可复核证据，但不得以其旧状态覆盖当前规范
- 需要冻结的代码或结果使用 `frozen/` 下带 manifest 的压缩包；仅有文字标签不构成冻结
