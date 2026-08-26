# N2P3-Net 路线图

> 本路线图只描述当前可执行的阶段、门槛和未完成工作。历史材料见
> [`../archives/README.md`](../archives/README.md) 与
> [`../archives/legacy_v11_docs_2026-08-25/`](../archives/legacy_v11_docs_2026-08-25/)。
> 版本：v4（2026-08-25，v12 四对象架构）。

## 阶段总览

| 阶段 | 内容 | 进入下一阶段的硬门槛 |
|---|---|---|
| S0 | 反例 harness：soft-label 语义、count 抵消、rho-chain 排序、冗余 L 融合、latency 规范、prior-shift、stopping replay | S0 七项全过 |
| P1 | LatencyMeasurement + PCW detached 消费 | S0-5、合成与已有 2-fold latency audit 全过 |
| P2 | RepetitionEvidence 可加 LLR 主干 + state residual | S0-2/3/4 全过，locked 开发折 |
| P3 | Reliability 双 estimand（fidelity / clean_probability） | S0-6 全过 |
| P4 | InnovationAudit 嵌套 M0/M1 + DynamicStopping replay/e-process | S0-7 全过 |
| P5 | 预注册 8-fold locked 开发；全部通过后按 E4a 删除旧语义与旧入口 | blueprint.md 第 7 节门槛 |

任何阶段失败都回 S0，禁止在开发折上调超参。现有 GTN 242 开发暴露队列只做 locked
development/replication；confirmatory 仍要求未暴露 cohort + 一次性锁 + >=5 seeds。

## Phase 0 数据与格式（保持）

数据契约、缓存 schema、mask、metadata 与之前一致；不新增数据集专用训练脚本。

## Phase 1 基线与锁定评估（保持）

主终点仍为 validation-calibrated `exact_llr@3` ITT hit；显式报告
exact/prefix/flash/all 与 coverage/N。次指标增加四个对象的分项报告：
`measured_tau_posterior`、repetition backbone/state 分项、`fidelity`/
`clean_probability` 分项、嵌套 M0/M1 审计与 stopping replay。

## Phase 2 四对象架构（替换旧 strict-past 研究）

正式路线仍 `final=PCW`。研究路线按 [`blueprint.md`](blueprint.md) 实施
L/R/Q/S 四对象，全部 fail-closed：

- L：fold-local 白化 P3b 模板 + amplitude profile likelihood + 显式规范锚；
- R：所有正负 flash 的 additive LLR 主干；state residual 必须自证 held-out log score
  增量；
- Q：`fidelity` 常开；`clean_probability` 仅在硬标签生成模型与 prior-shift gate 后启用；
- S：嵌套 M0/M1 cluster bootstrap；two-hypothesis conformal 异常；first-crossing replay，
  e-process 后置。

旧 strict-past fusion（单参数非负 alpha）已归档；旧 rho 0.9/0.1 概率语义已归档；
PCW attention tau 只作 routing。

## Phase 3 跨域与辅助数据

保持 T0/T1/T2/T3 规则；四对象不进入辅助域主损失，L 的模板/白化必须 fold-local。

## Phase 4 可解释性与采集

- 生理潜伏期只接受 LatencyMeasurement gate 过的 `measured_tau_posterior`；
- PCW attention / P300 PEC audit 只证明表示层贡献，不冒充生理定位；
- fidelity/clean_probability 分字段报告，不混称。

## Phase 5 消融与报告

按一次只改一个主要因素的嵌套顺序运行：四对象逐个开启，其余保持零输出；最终报告必须
包含四对象 gate 结果、cluster bootstrap CI、prior-shift、coverage、expected flashes、
risk-coverage 曲线、paired statistics、参数量/成本与删除清单执行状态。

## 里程碑

- M1：S0 harness 全过，数据/基线协议保持锁定。
- M2：L 对象通过合成 latency gate，或明确 `measured_tau=None`。
- M3：R 主干/残差在 locked 开发折上给出合法前缀后验。
- M4：Q 双 estimand 通过 unseen corruption / prior-shift / chain-NLL。
- M5：S 嵌套审计与 replay 完成；激活或退役 innovation fusion。
- M6：独立 confirmatory cohort 一次性锁运行，旧 v11 语义与入口删除。