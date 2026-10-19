# 全链合同修复与 EA 类型化接线（2026-09-30）

> 2026-10-04复审更正：本记录的测试执行事实保留，但“全链/授权重放闭合”的解释不成立。
> 新反例确认外部复用可错贴输入、EA授权与实际fit未绑定、缺域输出未赋值及普通旧EA入口仍接受。
> 下文为09-30实施时记录，当前状态以[全链科研重审](system_research_20261004.zh.md)为准。
> 更正前文本及六指南已物理保存在`frozen/system_research_20261004_guidance_before/guidance.zip`并逐成员SHA核验。

本轮把 2026-09-29 深审中确认的身份、预算和 EA 缺口落实到活动源码。工作区原有并行脏改动、历史 cache、预测和冻结归档均保留；本记录只说明本轮新增的活动接口和验证，不把旧成绩重标。

## 已实现

- `transfer.binary.matched_input_digests` 对实际评分的 `X[test]`、标签、试次通道可见性、测试行和 preprocessing declaration 计算指纹，并单独计算 `test_labels_sha256`。
- `target_experiments`、跨库 binary、单被试 binary、source-chain 评分记录生成两个指纹；`verify_pool_matrix` 重放时重新计算。`stats.transfer` 对正 K 记录强制要求四类 digest，并要求对比臂完全一致。旧记录没有新字段时 fail-closed。
- `TestBudget` 将 requested minimum、planned/actual test 行、policy、独立测试录制和 relaxation reason 写入 source-prefix manifest。`limited_data_80_percent` 是显式有限数据策略；`strict_minimum` 不满足即拒绝。
- 新增 `data.ea_artifact` 深模块。`EAFeatureArtifact` 只保存 `ea_component_###` 无量纲坐标，不再伪装为带电极位置和电压 QC 的 `EpochDataset`。`EAFitAuthorization` 绑定父 cache bytes、decoded dataset、物理 subject/event 行身份、拟合行和 authority participant keys；静态或动态缺导直接拒绝；save/load 通过 NPZ bytes attestation 和 semantic digest 重放。
- EA state 使用 `relative_eigenvalue_floor × trace(covariance)/C` 的相对正则，旧绝对 V² 正则记录不进入活动 schema。`prepare_euclidean_alignment_cache.py` 输出类型化 feature artifact；目标拟合/无标签 buffer 必须显式提供 `--fit-rows`。
- candidate 缺候选时统一 fail-closed abstain；统计新增 `valid_replicate` 次级读数，固定所有声明 standard K 和 seed pair 都完整的参与者 cohort，并同时输出排除/support matrix。requested participant operational metric 仍是主读数。
- promotion active ingress 拒绝带 `provenance.data_alignment` 的旧物理 `EpochDataset`，并在 EA consumer 尚未接通时拒绝启用 EA arm，避免只换 cache 就声称完成双侧 EA。

## 验证

项目环境中通过了预算、source-prefix、unified chain、candidate/scoring、EA 数值与 artifact、promotion preflight、transfer statistics、输入 digest 等定向测试；本轮新增 EA 与 digest 测试覆盖父缓存错配、静态/动态 mask 错配、状态篡改、标签/特征/预处理变化和固定 cohort。Ruff、全套 pytest 和 `git diff --check` 均通过。

这些测试证明合同和重放行为，尚未证明 A1/Q1 获得性能收益，也未启动新云端训练。EA 当前仍是可审计的 feature artifact 旁线；要进入统一模型必须另接 artifact-aware model ingress、source/target 同空间 checkpoint、目标 calibration-only fit 以及个人只读 replay，并重新跑预声明矩阵。

## 设计判断

EA 的外部 seam 只有 `save/load/assert_parent_replay` 三个动作，复杂的 state 校验、物理行身份、mask 和 attestation 留在深模块内部。这样调用方不再自行拼 provenance 或把组件坐标当 electrode voltage；历史物理 cache 可以只读审计，但不能穿过活动 promotion seam。

## 证据边界

本轮没有把双向边界 operator、EA、A1/Q1 或 Q1 decision 适配的任何点估计变成产品默认。有限支持下的边界 artifact 仍须通过真实 BI/Huebner、纯工频、慢趋势和 ERP 保真门槛后才可另行准入。
