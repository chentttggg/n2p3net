# 参数归属、观测统计与冻结预处理

[验收收据](../assets/lean_network_preprocessing_20261018/acceptance_01.json)记录本片
72 个定理、24 项严格负对照及与前基线的联合构建。当前整体状态只维护在
[活动 checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)。

| 切片 | 定理 / 严格负控 | 已审范围 |
|---|---:|---|
| ParameterOwnership | 15 / 6 | 实际组解析顺序、重复/交叠/缺失/外来参数拒绝；由独立名称、列表和覆盖字段推出成功 |
| ObservedStatistics | 15 / 4 | 观测均值、总体方差、尺度、逐次 cast 的不干涉，以及缺导最终写零 |
| 冻结 EOG/EA 与 fit/overlap 合同 | 42 / 14 | 冻结点变换与裁剪、参考依赖、EA 电压单位补偿、合法差分支持及给定覆盖下的 OLA |

三份独立复审分别核对
[参数归属](../assets/lean_personal_erp_20261018/v07_parameter_ownership_independent_review_01.json)、
[观测统计](../assets/lean_personal_erp_20261018/v07_observed_statistics_independent_review_01.json)和
[冻结预处理](../assets/lean_v08_fit_20261018/independent_review_01.json)。
成功方向由独立字段或样本关系推导；反向 postcondition 可假设执行成功，二者没有互相
循环充当前提。逐项公理仅为允许的三项，24 个错误替代均恰归结为 `False`。

作者共 14 项有效检查包含构建、独立直接重查、公理与负控；root 再做一次联合构建。
V07 原始执行保护 192 个基线成员，V08 原始执行保护 183 个基线成员及三项核心文件，
并各自保存本地证明依赖闭包的前后身份。后来的 198 成员联合检查不会回填旧执行身份。
V08 保留 11 项实际源码变换/guard 与两项纯数学控制的原执行，本次未重跑源码或拟合。

观测统计的全缺导方差零值只是数学总化助手；NumPy 的空观测方差仍可能是 NaN，实际
返回尺度通过另一个分支选一。全缺导单通道控制可以嵌入另一导仍有效的合法 trial，
不能据此声称 Deep 输入允许全缺导 trial 或缺导位置的 NaN。
最终写零只描述先前 subtract/divide 成功后的值，不声称缺导跳过那些操作或异常。

EA 单位结论条件于同一精确归一化谱分解；未认证 native eigensolver。
OLA 结论条件于给定非负权重和覆盖，尚未从实际 Hann 窗和排程推出全部正分母。
模型各分支的参数组清单、卷积/池化/展平/BN 状态、中心化 OLS 拟合、更广 QC 阈值/
区间和 harmonic 自适应拟合仍需继续，不以本片替代完整 V07.a/V08.a 或全项目验收。

[证明清单](../../frozen/lean_network_preprocessing_20261018/manifest.json)封存 211 个
继承与新增成员；[源码证据清单](../../frozen/lean_network_preprocessing_20261018/source_and_evidence_manifest.json)
绑定源码、三份独立复审和逐次执行证据。
[清理收据](../assets/lean_network_preprocessing_20261018/cleanup_01.json)记录归档校验后，
用 cmd 逐文件删除 23 个旧诊断文件并移除 10 个空目录。
活动构建、科学证据/结果及暂停 checkpoint 保留；未安装依赖。

## 中心化 OLS、QC 区间与 harmonic 支持增量

[联合验收](../assets/lean_gaussian_20261016/gls_preprocessing_acceptance_01.json)中预处理
部分为中心化 OLS 22/3、QC/IRLS/Hann 29/12（定理/严格负控）。
[OLS 复核](../assets/lean_v08_fit_20261018/independent_review_ols_slice_02.json)从独立设计
满列秩推导 Gram 正定、显式正规方程、唯一系数与残差正交。设计与响应都只在所选
校准行上中心化；参考不变性条件于保留相同 EOG 设计，共享参考改变设计时另有反例。
这些结论不将 sklearn 返回的 rank 整数视作精确矩阵满秩证明。

[QC/harmonic 复核](../assets/lean_v08_fit_20261018/independent_review_qc_harmonic_slice_03.json)
区分半开异常区间、检测与确认时刻，按每个非零 native 混合依赖传播观测限制。
IRLS 当前系数只满足生成它的旧权重正规方程；返回的新权重一般不能替代旧权重。
实际规则网格加末尾窗口保证覆盖，内部 Hann 权重严格为正，由此推出精确实数
overlap-add 分母为正。此前固定权重 OLA 的缺失桥接由此补齐；自适应频率、全部迭代
与实际 caller 组合仍需各自证据。

作者 11 项有效检查保留其历史 198 保护成员、三核心文件及本地闭包身份；相关 Python
源码由原合同、快照与探针绑定。root 在当前 v4/201 源码身份上完成联合构建，未把旧执行
改写为新基线。八项实际 QC、一个独立 Hann 原语和两项纯数学控制保持原类别；没有
重新拟合或训练。[原范围审查](../assets/lean_v08_fit_20261018/closure_audit_05.json)的
22 算法族缺口继续保留，局部增量不代表 V08 完成。

本片与 GLS 共用[证明清单](../../frozen/lean_gls_preprocessing_20261018/manifest.json)、
[源码证据清单](../../frozen/lean_gls_preprocessing_20261018/source_and_evidence_manifest.json)
和[清理收据](../assets/lean_gaussian_20261016/gls_preprocessing_cleanup_01.json)。

## 观测前向、SplitBN 与按折校准组件

[下一批联合验收](../assets/lean_network_preprocessing_20261018/network_risk_fold_acceptance_01.json)
中网络部分为 ObservedTemporal 25/4、SplitBNStatistics 20/8、SplitBNMomentum 16/4。
[root 独立复核](../assets/lean_personal_erp_20261018/v07_continuation_independent_root_review_01.json)
核对真实 cross-correlation、实测 slot/time 统计、BFCT scatter 与 grouped spatial 公式。
单个已选域的 BN 与全域混合分开；默认 legacy 图和 unified observed 图分开。

加权 SplitBN 先将配置权重 cast 成有效 tensor 权重，再检查正权重域的计数非零。
配置总质量容差不能当作有效权重精确归一化；方差保留 `(1-s)*mean²` 质量误差后再 clamp。
计数为有符号观察值，字面 `!=0` 不是可达训练历史的证明。Momentum 先增计数再计算
倒数，零 momentum 不冻结计数，None 的等批次平均也不是等观测平均。

同批 FoldLocal 为 32/14；[独立复核](../assets/lean_fold_artifact_20261018/independent_review_01.json)
对应实际校准支持、inner/outer 选择与 heldout 账本，保留五个已声明的非 fit 控制。
outer heldout 不进入 policy.fit；inner validation 合法参与目标与超参选择。
selected quantiles 来自整个 outer-training CV，不能据此声称 kappa 完全 nested。
完整 wrapper/loader/sidecar 组合与原 V08 家族仍按各自要求继续。

作者 24 次有效检查与 root 一次联合构建均保留真实执行版本；root 当前联合构建为
v5、202 源文件、230 个当时保护成员，墙钟 35.66 秒。
[证明归档](../../frozen/lean_network_risk_fold_20261018/manifest.json)和
[源码证据归档](../../frozen/lean_network_risk_fold_20261018/source_and_evidence_manifest.json)
固定本批身份。[清理收据](../assets/lean_network_preprocessing_20261018/network_risk_fold_cleanup_01.json)
记录 38 个旧诊断文件、19 个空目录经归档核验后用 cmd 逐项删除。

审查另由[真实非训练反例](../assets/lean_personal_erp_20261018/v07_calibration_mask_probe_red_01.json)
确认 target-calibration helper/caller 遗漏 channel mask；该安装路径未由上述局部统计
定理宣称正确，独立缺陷修复与回归继续跟踪于活动 checkpoint。

## 捕获总体、选择器与有效空间范数

[本批验收](../assets/lean_network_preprocessing_20261018/membership_bn_acceptance_01.json)
的网络部分为 CalibrationPopulation 21/4、SplitBNLifecycle 38/8 和 SpatialMaxNorm 17/4。
三份独立复核分别核对
[分块统计](../assets/lean_personal_erp_20261018/v07_calibration_population_independent_review_01.json)、
[选择与安装](../assets/lean_personal_erp_20261018/v07_splitbn_lifecycle_independent_review_01.json)及
[空间约束](../assets/lean_network_preprocessing_20261018/spatial_maxnorm_independent_review_01.json)。

实际 `delta*delta_after*block_count` 合并保持 count、sum 与 squared-sum 不变量；任意
非空块流的最终 `M2/total` 是捕获总体方差。它验证实际捕获的数据，不证明遗漏的 mask
已传入，也不从最终 total 正推出每个早期块非空或每个 hook 一定刷新。

source-domain 可独立于 target policy 选择；实际 guard 为计数非零，target 未初始化
不会 fallback 到 pooled。初始化期间不覆盖，clear/reset 后可重新安装。全已初始化的
helper 仅在 forward 之后进入诊断分支，不调用 installer；没有据此假设 callback 纯净、
跨模块安装原子性或混合子模块 training mode 的恢复已证。

空间权重按每个输出行计算 `min(limit/max(norm,epsilon),1)*row`。证明其范数同时不超过
原范数和 limit；恒等与饱和结论保留实际 floor 条件。若 limit 小于 floor，已在球内的
极小向量仍可缩小，该数学映射也未必幂等。源码每次从原参数计算临时权重，未把它
冒充无条件的标准投影或原地参数更新；这不是新报告的实现错误。

本批共 18 项作者检查与 root 一次联合构建，保留原 251/278 保护成员的执行身份。
root 联合构建绑定 v5 的 202 份源码和当时 278 个保护成员，墙钟 30.95 秒。
[证明清单](../../frozen/lean_membership_bn_20261018/manifest.json)和
[源码证据清单](../../frozen/lean_membership_bn_20261018/source_and_evidence_manifest.json)
封存本批网络及 membership 组件；
[清理回执](../assets/lean_network_preprocessing_20261018/membership_bn_cleanup_01.json)
记录归档核验后用 cmd 逐文件删除 22 个旧诊断文件并移除 11 个空目录。
完整 V07、V04 与实际 caller 组合仍继续；源码修复按主规范由科研 agent 负责。

## 读出几何与眼电调用组合

[本批验收](../assets/lean_personal_erp_20261018/neural_ocular_readout_acceptance_01.json)
中的网络增量为 29/8，眼电增量为 69/12。
[网络独立复核](../assets/lean_network_preprocessing_20261018/v07_network_readout_independent_review_01.json)
核对 ST/MST 的实际长度、六类读出宽度、unfold 的通道/时间顺序、MS 平均碰撞，
以及 affine、quadratic、MLP 的方程和数组计数。MST 本身没有额外时间池化；
默认 MLP 比 quadratic 多 `4R` 个参数单元，不能称为精确等参数。
这些结论不关闭完整 BN/ELU/dropout/wrapper 前向或 latency 读出。

[眼电独立复核](../assets/lean_ocular_chain_20261018/independent_review_01.json)
建议原 EO1/EO2 在既定数学规格和源码审计范围内闭合：实际授权 mask、选行中心化、
OLS 转置状态、冻结应用、participant/session/QC 预算、三种时间策略及参考角色链已绑定。
强度为零仍执行参考与角色守卫；新录音使用已存校准中心；CAR 仅平均选定 EEG 坐标，
但可继续依赖回归使用的 EOG。数值 rank 观察不推出精确满秩，固定 QC mask 的局部
非干扰不等于重新运行 QC 后不变；物理布线和统计效能仍需要各自证据。

17 次本批有效检查（含 root 联合构建）保留各自 293/302 保护成员的历史身份。
[本批证明归档](../../frozen/lean_neural_ocular_readout_20261018/manifest.json)
与[源码证据归档](../../frozen/lean_neural_ocular_readout_20261018/source_and_evidence_manifest.json)
固定验收身份。[cmd 清理回执](../assets/lean_personal_erp_20261018/neural_ocular_readout_cleanup_01.json)
记录 38 个旧诊断文件、19 个空目录在逐成员归档核验后删除。当前总体状态仅维护在活动 checkpoint。

## 原始 manifest 与快照清单

[25/7 组件验收](../assets/lean_personal_erp_20261018/saved_control_manifest_acceptance_01.json)
与[独立复核](../assets/lean_v08_fit_20261018/raw_manifest_independent_review_01.json)
覆盖规范解码后的 manifest、每文件描述符/字节观察、冻结快照和 attestation 清单。
实际预期数据集、casefold 路径唯一性、local/remote 摘要、verified 标记及可选 MD5
来自各自门禁；每个源描述符复制后，存储对象仍须重新核验。
逐文件成功再推出完整顺序、路径覆盖、数目和含重复内容的字节总和，
不把整个 verifier 成功或最终快照相等当作输入。

不同源路径可共享同一内容对象；相同文件数也不保证源路径覆盖。
摘要匹配不代替独立的 size 检查，更不推出任意摘要函数的字节级单射性或官方来源授权。
严格 metadata/路径/Unicode/原生描述符及摘要操作保持各自接口；
任意直接构造 attestation 不会自动补上 manifest 构造器的 casefold 唯一性门禁。
RA2 的实际 direct/ZIP-member loader 映射、报告路径与 parser 读集合仍继续，
没有凭这个清单组件关闭整个 V08。
