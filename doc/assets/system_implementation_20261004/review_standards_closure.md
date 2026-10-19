# Standards 最终闭合复核（2026-10-04）

**结论：原 Standards 四项和失败 refit 建议均已关闭；本次额外发现的旧局部 identity schema 绕过问题也已修复并独立复验。当前没有本审查范围内尚未关闭的 P1。** 这是实现/证据关系验收，不是算法性能优越性结论。

修前记录 [review_standards.md](review_standards.md)、其 RED probe、前一份 [E0审查](e0_review.md) 全部保留。固定基线仍为 `frozen/system_implementation_20261004_before/source.manifest.json`，没有将当前 worktree 或 git commit冒充旧执行源码。本次只读核查，不修改 production/guides。

| 原发现 | 独立闭合证据 | 状态 |
|---|---|---|
| noise/continuous 的 scope、训练record ledger可变 | 原公开写dict/嵌套record现报TypeError；强行替换对象字段后，实际LDA fit与continuous score报integrity错误。load也拒重签空训练ledger | 关闭 |
| operator CLI在消费后重开路径计算hash | 精确定向两条真实CLI反例通过：fit中替换输入/plan、score中替换model/input/metadata，记录均仍绑定实际消费原字节；失败执行有终态 | 关闭 |
| ERP全NaN mask当成实测 | 原 `fit_calibration_candidate→predict_logit` 路径现拒绝，错误明确要求aligned boolean (N,C) | 关闭 |
| fixed_analytic任意正交基仍冒充cosine | 原重新计算digest的非cosine正交基现被公式校验拒绝 | 关闭 |
| xDAWN失败refit后继续预测旧模型（冻结基线旧问题） | 单类refit拒绝后，原test X再predict报“请先fit”，不再返回旧score | 关闭 |

公开独立 probe：[probe_closure.py](review_standards_closure/probe_closure.py)；结果 [closure_results.json](review_standards_closure/closure_results.json)，8个拒绝检查全部通过，执行期间12个被核对production文件没有变化。该快照的comparison SHA早于本次随后发现的必需字段修复，后者另附独立核验，未改写初次证据。

## E0 交叉复验与新增关闭项

对照之前R1–R4检查当前actual inner X/y/mask输入key、完整candidate语义、每source leaf raw覆盖、独立implementation变更轴。精确8项回归 `8 passed in 6.45s`：CLI三项、actual inner变化强制重fit且相同输入复用、sidecar-only拒绝、混合未绑定raw父拒绝、baseline不连带许可源码变化、decision语义变化拒绝。

本次发现旧初始E0实现产出的 `/1` comparison identity缺少decision_semantics/fit_context，仍会被新版validator认作完整当前合同。用保存的初始builder实际产生移除候选词表前后两份record，缺字段且同digest，旧gate接受；修前证据为 [old_partial_identity.json](review_standards_closure/old_partial_identity.json)。

root修复后改为 `/2`，强制所有顶层与语义/context子字段，旧局部record不升级。我独立再跑：旧record拒绝、旧pair拒绝、新record中不同候选监督拒绝，见 [identity_schema_recheck.json](review_standards_closure/identity_schema_recheck.json)。这条新P1已关闭。

## Raw Adapter 与真实账本交叉核查

新BNCI Adapter代码核查确认：MAT从同一已验bytes解码；K个labels全部保留；raw calibration截到下一未授权flash前并显式right_censored；测试范围按全矩阵Kmax固定；prediction-facing view不含target_mask；测试真值在独立scoring_truth；不完整decision保留requested分母；离线双向滤波的实际可用时刻另行记账，未把大selection gap当成滤波后统计独立证明。

`tests/transfer/test_raw_continuous.py` 独立运行 `5 passed in 9.62s`，覆盖真实parser/生产滤波、未来样本与标签不影响C、完整候选membership、gap限制、实际save/reload、requested计数及失败终态。此前跨测试文件导入fixture的问题已由owner移到 `tests/support/bnci.py`，共享factory遵守仓库标准。

另外只读重算已完成 `bnci_continuous_real_v2` 的16份sealed cell与所有companion SHA：464个requested、448个scored、16个unscorable仍保留，231次correct；从保存log evidence独立重算softmax、NLL、hit与offline latency关系通过。机器记录 [real_ledger_recheck.json](review_standards_closure/real_ledger_recheck.json)。独立softmax与原exp(logit-logsumexp)最大绝对差4.83e-12，属于不同稳定归一化写法的舍入差，未影响winner或NLL账本。没有重训、筛人或更改原结果。

新连续路线保留其已声明局限：原生八导/256Hz/无baseline/整段双向filter与旧epoch路线不同，不能将hit差解释为单一算法收益；K200的概率过置信也没有被诊断命中率掩盖。

## 快照与边界

最终实际读取的production及关键support摘要见 [final_source_sha256.json](review_standards_closure/final_source_sha256.json)。EA主路径之前已完整验收，本轮只核对其actualparent+expectedplan、不可物理cache入口合同仍在，未重复跑全部EA正常路径。

本次未运行全套pytest；全套验收由root统一执行。没有新增以命名、行数或“抽象偏好”为依据的硬性smell问题；只关闭有具体行为反例的合同问题。解析基、mask、失败状态、字节重放均收敛在原owner，未要求runner逐点补同一套校验。
