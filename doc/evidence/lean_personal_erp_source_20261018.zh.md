# ERP 标量、拟合与实际消费者的增量验证

本批验收六个数学与程序控制组件，原 V01–V10 全项目目标继续执行。
[验收收据](../assets/lean_personal_erp_20261018/source_components_acceptance_01.json)
绑定当前源码、逐项审查和实际检查记录；当前状态只维护在
[活动 checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)。

| 组件 | 新定理 | 严格负控 | 本批范围 |
|---|---:|---:|---|
| PersonalERPScalar | 36 | 18 | 单个 NumPy/math/比较操作、异常顺序及 JSON set 成员检查 |
| PersonalERPFit | 36 | 7 | 原始字段导出拟合控制流、候选字段和数值操作顺序 |
| ERPTemporalProducer | 56 | 15 | 已解码样本、事件逆映射、局部通道、Gram、谱选择及收据 |
| PersonalERPCurrentConsumer | 19 | 2 | 有限数组的旧解码辅助层 |
| PersonalERPScalarFields | 11 | 0 | 原始结构字段与 18 个单操作事实导出配方构造成功 |
| PersonalERPActualConsumer | 14 | 5 | 生产、复制、selector、当前 reader、实际 Scalar 和绑定的连接 |
| 合计 | 172 | 47 | 所列组件，非完整算法实现认证 |

累计历史数学组件为 1765 条定理、232 个严格负控。这些数量不是源码覆盖率，
也不表示历史组件全部重新运行或对应当前源码。原 19/2 辅助层保持其有限解码范围；
新增实际 Scalar 路线保留巨大整数、list/dict 异常，而不是用辅助层替代实际调用。

本轮新执行组合构建、六份空诊断源码重查和 172 条声明的统一公理检查，共八次。
公理仅出现 `propext`、`Classical.choice`、`Quot.sound`；其中一条声明无公理。
当前 47 个负控复用五份源码字节匹配的作者成功记录，并独立检查原始日志：
必须恰为相应数量的 `unsolved goals` 和 `False`，没有额外错误或警告。
未声称重新运行这些负控，也未重跑继承的 185 个历史负控。

[证明清单](../../frozen/lean_personal_erp_source_20261018/manifest.json)
逐成员冻结 153 份证明、配置与工具文件；
[源码与证据清单](../../frozen/lean_personal_erp_source_20261018/source_probe_manifest.json)
冻结 370 个枚举成员，包含当前 199 个源码/测试输入和本批证据。
其中出现的 P06 候选与退回审查记录不属于已验收证明，证明成员以证明清单为准。

## 真实数值反例与源码对应修订

[固定数值边界案例](../assets/lean_personal_erp_20261018/erp_fit_output_boundary_execution_review_01.json)
仅执行一次：N=4、C=T=1，输入为 `[-a,a,m,m]`，其中 a=2^-530、m=2^-20，
标签为 `[0,0,1,1]`。输入全部有限且每类两行，实际 `fit` 仍返回 `inf` 系数、
`-inf` 截距；当前 reader 随后以非有限状态拒绝，预测也出现非有限值。
因此“有限输入和类别数足够就保证产物有限且可重载”的无条件命题登记为
`false_claim_with_evidence`。理想实数截距本身在 binary64 范围内；实际异常来自先溢出的
系数继续参与运算，不能把它写成理想截距自身超范围。

该案例的墙钟为 3.256161 秒。外层资源采样只覆盖 Windows venv launcher，
其 CPU/峰值内存不代表实际数值子进程；未为补指标重跑案例。
这是合成软件反例，不是成人、公开数据或物理采集结果。

[P06 九个固定案例](../assets/lean_personal_erp_20261018/p06_numeric_source_probe_01.json)
进一步确认：三个阈值的 `10**100` 在真实 `np.isfinite` 处抛 `TypeError`，
但仅做普通比较的 `cv_splits=10**100` 仍可通过。list 的 set 成员检查和非法字符串
分别产生 `TypeError`、`ValueError`，所以不能统一增加数值有限性或类型前置门。

[P06 修订审查](../assets/lean_personal_erp_20261018/p06_reader_independent_review_02.json)
保留审查自身的勘误：EOG policy 的允许容器是 tuple；只有实际 set 站点需要
hashability 步骤，不能把 tuple 机械改成 set。Data 的 rank/dtype 顺序及位置头部、
mask、观测位置数值检查的分阶段顺序也仍待修订与连接。

## 清理与未完成范围

[实际清理收据](../assets/lean_personal_erp_20261018/source_component_cleanup_01.json)
记录替代检查通过后，先逐成员验证旧输出归档，再用 cmd 逐文件删除 56 个过时
Lean 编译报告/日志，移除 28 个空目录，共 160762 字节。科学原始证据、冻结结果、
暂停 checkpoint、活动 `.lake` 及当前成功检查日志均不在删除范围。

完整 P01 的其余配方与一般标量语义、P06 实际 raw/执行/构造入口、非有限估计与
打包结果、时基 CLI/多 scope/构造/重放/缓存关系及身份权限仍未完成。
单个数值操作的返回条件不是 NumPy/SciPy/Torch 实现或 IEEE 运算的证明；
本批没有完成这些细化义务，也没有完成原 V01–V10 全项目验证。
