# ERP 非有限输出、字段构造与原始入口的增量验证

本批沿用原 V01–V10 范围，补齐六个数学与程序控制组件，并执行固定的真实软件入口案例。
[验收收据](../assets/lean_personal_erp_20261018/continuation_component_acceptance_01.json)
绑定证明、源码、独立审查、日志和清理结果；当前进度只维护在
[活动 checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)。

| 组件 | 新定理 | 严格负控 | 已审范围 |
|---|---:|---:|---|
| PersonalERPFitOutcome | 23 | 9 | 保留 Inf/NaN 输出；当前内层 reader、配方标量、外层有限检查的实际顺序 |
| PersonalERPOutputFields | 15 | 5 | 值级估计输出、字面字段构造、逐数组打包接入 fitStages；推出返回字段及 prefix/selector/clone 键唯一 |
| ERPTemporalCLI | 12 | 11 | 已准备输入之后的显式调用、解码接入、门禁及错误传播 |
| ERPTemporalMultiScope | 13 | 4 | 可变多 scope 累积、参与者权限、真实操作顺序及逐次拟合环境 |
| ERPTemporalIO | 10 | 6 | 写入、PermissionError 重试、关闭/清理顺序及发布后重放失败 |
| ERPTemporalCache | 5 | 2 | 已解码缓存摘要选择与更新顺序 |
| 合计 | 78 | 37 | 所列组件，不是完整 P01/P06/P12 或数值实现认证 |

组合构建、六份独立进程重查与 78 条声明公理检查共八次新执行，均无错误或警告。
其中三条声明不依赖公理，其余仅使用允许集 `propext`、`Classical.choice`、`Quot.sound`。
首次汇总脚本漏识别“无公理”输出，已保留该收集失败；修复仅涉及日志解析，未改变命题，
未重跑已成功的 Lean 进程。37 个严格负控复用六份字节匹配且已独立审过的原始运行，
每项恰为预期数量的 `unsolved goals` / `False`，没有辅助错误或警告。

字段证明从真实字段构造推出唯一键，不假定任意列表本来就是合法字典，也不要求估计器返回
整个“合法拟合产物”。打包操作失败时没有部分返回值，并不表示 Python 原对象 `_state` 回滚。
本批允许估计器及 cast/copy 输出非有限值；此前有限输入仍产生 Inf 的数值反例原样保留。
证明清单中的 Data/Execution/Input 三份文件是固定的支持依赖，不因此宣称完整 P06 已通过。

## 实际原始入口与配置案例

[固定八项 raw 入口执行复核](../assets/lean_personal_erp_20261018/p06_actual_raw_execution_independent_review_01.json)
记录实际 EDF/BIDS 读取、18 个校准 epoch、一次 public `fixed_erp` pooled-LDA 拟合、
目标标签已移除的未来录制及九个有限分数。另一个明确未拟合的手工单位系数模型，
逐点符合预先固定的坐标投影；复核还用独立标准库 EDF16 解码验证其九个输入数值。
实际拟合参数只检查链路和有限性，没有用拟合答案充当估计器数值正确性的预期。

八个案例全部符合执行前固定的预期。身份错误、复用校准录制、时间过早三项均在实际 EDF
读取前拒绝；ERP 内部 Inf 优先于巨大标量，巨大标量优先于额外非 ERP Inf 的外层门禁。
只读调用跟踪确认恰一次 public/StructuredERP 拟合；内部 StandardScaler 是其统计步骤。
源码 199 项、helper 三项、历史证明 153 项、活动 runtime 189 项、环境 13579 项与 native
537 项前后绑定，未安装新包、未初始化 GPU。约 60.168 秒墙钟、58.078 秒观测 CPU，
进程树采样峰值 RSS 403312640 字节；这些指标包含哈希、导入、IO、跟踪和验证，不是纯拟合吞吐。

[35 项固定配置案例的执行复核](../assets/lean_personal_erp_20261018/full_recipe_source_execution_review_01.json)
保留直接 API 与严格 JSON 的不同边界，以及类型、转换、set/hashability、默认配置和异常顺序。
35 项全部符合预期，未拟合模型；这些有限行为观察不计为 Lean 严格负控。
已有事件合同与反例测试另有 18 项通过，Ruff 和 `git diff --check` 通过。

## 归档、清理与剩余义务

[证明清单](../../frozen/lean_personal_erp_continuation_20261018/manifest.json)
冻结 171 个证明、配置、工具及支持成员；
[源码与证据清单](../../frozen/lean_personal_erp_continuation_20261018/source_probe_manifest.json)
保存本批枚举输入和原始日志。
[清理收据](../assets/lean_personal_erp_20261018/continuation_cleanup_01.json)
记录先归档并核验，再用 cmd 逐文件删除 36 个过时构建文件和 32 个本批合成 EDF/BIDS 夹具文件，
移除 35 个空目录，共 391423 字节。科学原始证据、冻结结果、暂停 checkpoint、活动依赖
及本批当前成功检查均保留；本批没有安装临时依赖包。

完整配方与 raw ndarray 入口、构造后视图、源时基快照/NPZ/权限及完整 CLI 组合仍未完成。
任意错长数组、可变 timeline 与 JSON 非有限/身份比较不能由较窄的解码类型提前排除。
NumPy/SciPy/Torch 数值实现、IEEE/存储/OS 细化及物理与成人效能各需单独证据；
本批不完成这些义务，也不完成原 V01–V10 全项目验证。

## 神经保存状态的实际前缀投影

[14/4 组件验收](../assets/lean_personal_erp_20261018/neural_ocular_readout_acceptance_01.json)
及[独立复核](../assets/lean_personal_erp_20261018/neural_state_projection_independent_review_01.json)
绑定真实 `trunk.` 开头测试与一次剥离，证明投影顺序、精确原键 lookup、张量值和键唯一性。
完整 packet 的复制/codec 结论进一步传至选定的原始 metadata 字段和投影状态；
`input_mean/input_std` 仍来自原完整状态，不能只保存 trunk 投影。

该组件只到选定的 preconstructor reader：未证明全部构造门禁、配置归一化、
普通 BN 兼容补 counter、dtype cast、strict load 或完整神经 predictor 等价。
个人 writer 明确拒绝 SplitBN；源 checkpoint 的 SplitBN 分支仍属于原 V04 待办。
组合构建、归档和 38 文件/19 目录的 cmd 清理均见本批验收，不重复统计历史证明。

## 神经构造、注册表与加载控制的有限切片

[源前范围勘误](../assets/lean_personal_erp_20261018/neural_control_scope_requirements_01.json)
保留原神经保存要求，纠正 personal reader 与 source-checkpoint SplitBN 范围的混用。
它还按本机 Torch 2.13.0 源码固定两项实际兼容语义：投影普通 dict 不带版本元数据，
普通 BN 因而可为缺失的 `num_batches_tracked` 补入新构造 counter 的 0；
标量目标也允许先将来源的单元素向量变成标量，再比较形状。
不能把这里的 `strict=True` 简化成原始键集合必须完全相等，或要求来源 dtype 等于目标 dtype。

[作者回执](../assets/lean_personal_erp_20261018/neural_control_author_checks_01.json)
记录 48 条一般关系、8 条预先固定见证及 8 项严格归约到 `False` 的错误替代。
新聚合、五个独立文件重查、公理审计和负控共 8 个检查通过，无 Lean warning 或额外公理；
各检查保留实际 v5 的 202 source/191 runtime 与当时 protected302 或 protected317，
没有为了较大的保护基线重跑已成功检查。

[源码对应](../assets/lean_personal_erp_20261018/neural_control_source_correspondence_01.json)
将有限归一化字段连接到十五个 architecture kwargs、物理 C/T/rate/start、pooling 和
完整适配器配置，复用已验完整 TargetRecipe 构造与网络宽度计划。
注册表逐项保留 ST/MST、普通 BN affine/运行矩/counter、所有 head 分支、latency query、
core residual 及 unindexed wrapper 的参数和缓冲区；residual gate 的显式 float32 与
ambient 浮点分配 dtype 分开。非张量 residual 上限、默认 latency 窗口和物理时钟仍保留，
没有因 `persistent=False` 或参数形状相同而丢弃。

加载从每个真实命名 entry 的 lookup、局部形状兼容和逐 scalar cast 回复推导，
再推出完整注册表的名字、顺序及条目数。控制层依次保留 architecture guard、模型投影
strict load、逐模块 eval、完整保存包中的输入统计检查及全保存张量有限性检查。
eval 保留张量和 target-enabled 标志，也保留 autograd 状态；调用者的 inference mode
是另一层。有限保存值不推出任意 dtype cast 后仍有限，NC05 是这个接口命题的逻辑反模型，
没有执行 native overflow、fit、训练、优化器或项目/Torch 数值探针。

[物理清单](../../frozen/lean_neural_saved_control_20261018/manifest_candidate_01.json)
冻结 8 个新文件；[清理回执](../assets/lean_personal_erp_20261018/neural_control_freeze_cleanup_01.json)
核验归档后，用 cmd 逐目录删除 9 个被替代的 build 目录及其中 18 份 log/report，
并确认目录消失，8 组最终检查和全部源码/driver/extra-pin 快照保留。没有安装临时包或修改生产代码。

本批仍是作者有限切片，等待独立复核；不自行完成 NN01/NN03/NN04 或原 V04.a/V04.b。
任意 JSON 架构字段到实际 dataclass 的完整归一化、真实 `architecture_record` 的逐字段
读回和 active latency offsets、root 的前缀/packet 组合、V07 完整 value-forward，
以及 source-checkpoint SplitBN 全 reader 仍分别保留原义务。这里不把构造成功、最终形状
正确或整个 predictor 相等作为输入前提，也不补设底层数值库/IEEE 已认证。

上述 56/8 作者切片后经
[独立复核](../assets/lean_personal_erp_20261018/neural_control_independent_review_01.json)
并入[本批验收](../assets/lean_personal_erp_20261018/saved_control_manifest_acceptance_01.json)。
复核保持原先列出的剩余义务，没有用库兼容细节替代完整的保存预测器等价证明。
13 次本批检查包含 8 次神经组件检查、4 次原始快照检查及 root 的一次联合构建；
各自保留执行时的 302/317 保护成员身份。
[证明清单](../../frozen/lean_saved_control_manifest_20261018/manifest.json)和
[源码证据清单](../../frozen/lean_saved_control_manifest_20261018/source_and_evidence_manifest.json)
固定本批交付。[清理回执](../assets/lean_personal_erp_20261018/saved_control_manifest_cleanup_01.json)
独立核验神经组件 18 文件/9 目录的删除，并记录 root 另用 cmd 删除 6 个原始快照旧诊断文件、
移除 3 个空目录；均先归档核验，未清理活动依赖或科学数据。
