# 域迁移首批实现：P1时空适配与P2源目标函数

> 导航副本：历史失效链接现指向[持久来源索引](../assets/repository_audit_20261019/docs_historical_index.json)。原文以原SHA保存在本地归档；本次只改定位入口，原结果、执行producer与未找到凭证的边界保持。


2026-09-16。用户要求结合`codebase-design`逐步实现，并取消蒸馏。
本次完成P1/P2的软件能力、调用链和反例；未启动公共/本人真实数据训练，没有新迁移效能结论。
后续次序由[路线图](domain_transfer_major_20260920.zh.md)统一维护。

修改前的323份源码、测试与指导已物理压缩并逐成员验证，
[清单](../assets/repository_audit_20261019/docs_historical_index.json)保留SHA-256；payload留在本地。
工作树原有大量并发修改及测试目录迁移全部保留，没有恢复旧平铺测试、重置源码或更改真实数据。

## 1. Module与interface选择

保留已有`SubjectAdapter`、`TargetRecipe`、源固定更新循环、任务图和产物保存路径。
新增变化集中在两个真实seam：模型参数归属和源辅助目标，不增加另一套训练runner。

- 模型owner通过`adaptation_parameter_groups()`声明实际module对象；共享检查要求每个参数恰好归属一次。
  `SubjectAdapter`一次应用选择，BN affine与buffer政策仍独立。新增未知参数会直接失败，不按名字猜归属。
- `SourceObjectiveConfig/SourceObjective`一次验证来源轴，随后用真实批次索引计算辅助损失及诊断。
  源训练循环继续负责抽样、CE、优化器、固定更新预算和快照；目标函数不拥有测试数据或额外抽样器。

两个module均主要依赖进程内PyTorch/NumPy；文件/模型持久化沿用生产interface。
测试从拟合、预测、CLI计划、checkpoint及个人模型重载进入，避免仅测试重复实现的内部细节。

## 2. P1完整处理

`p1_target_recipes()`声明以下八个NN处理和一个波形LDA，正K NN默认200更新、batch上限512：

| 处理 | 更新范围 | BN affine / buffer | dropout |
|---|---|---|---|
| `p1_full_fine` | 完整非BN权重与读出 | 可学习 / 在线 | checkpoint |
| `p1_scratch` | 与full配对的随机初始化 | 可学习 / 在线 | checkpoint |
| `p1_s0t0` | 读出 | 冻结 / 在线 | 关闭 |
| `p1_s1t0` | 空间投影与读出 | 冻结 / 在线 | 关闭 |
| `p1_s0t1` | 时间/MST与读出 | 冻结 / 在线 | 关闭 |
| `p1_s1t1` | 空间、时间/MST与读出 | 冻结 / 在线 | 关闭 |
| `p1_affine_head` | 读出与BN affine | 可学习 / 在线 | 关闭 |
| `p1_residual_head` | 既有目标残差adapter与读出 | 冻结 / 在线 | 关闭 |
| `p1_waveform_lda` | 校准波形收缩LDA | 不适用 | 不适用 |

空间组为`st_spatial`；时间组包含`st_temporal`与所有MST depthwise/pointwise。
MST pointwise仍会混合特征，不能把时间组效应解释成纯生理时间变化。
四个因素格共享BN/dropout政策，额外full/scratch保留强配方，避免把它们混成单因素效果。

`FeatureResidualAdapter`被包入显式`FeatureDomainTransfer`，其source扩展参数冻结，target门控从identity开始。
完整target配置写入recipe，实际wrapper写入架构；个人模型重载根据保存recipe重建，并要求架构完全一致。
源初始化和scratch的input statistics、拟合范围、训练预算仍由原接口约束。
特征缓存只用于真正固定的特征变换；空间/时间/adapter更新、在线BN或dropout都会触发实时特征计算。

公共入口：既有`run_cross_dataset_transfer.py --stage P1`，binary入口通过同一stage转发。
分析器`--stage P1`提供空间、时间、交互，以及各候选相对full/scratch的预声明配对对比。
K0仍单独执行source-only推理，不运行正K候选。

## 3. P2源目标函数与完整调用链

`DeepConfig.source_objective`仅用于固定更新源轨迹。支持：

- `ce`：原CE计算，新增配置显式为CE时与未启用辅助项逐参数一致。
- `cross_domain_supcon`：post-MST特征展开后无参数L2归一化；正对需同标签、不同真实域、不同参与者。
  同类但不满足正对身份的行不当作假负对；负对仅为不同标签。
- `conditional_rbf_mmd`：同一归一化特征上调用既有条件RBF-MMD；保留无偏估计可为负的定义及诊断。

辅助项不改变CE抽样；重复采样的同一输入行在辅助项中仅保留第一次。
身份/类别全局支持不足直接拒绝；批内无合法项记inactive，不伪造正对。
保存唯一辅助行数、有效anchor/正对或MMD项数、inactive批次及平均辅助损失。
首批没有加入学习型投影头，以减少与额外参数容量的混杂；分母和投影约定已同步修订原方案并进入版本化合同。

`run_pretrain_supervised.py`新增`--source-objective`、`--source-objective-weight`、
`--source-objective-temperature`和`--source-objective-min-cell-count`。
辅助项要求`--fixed-updates`及真实多域输入；使用现有身份账本和完整source训练行。
最终checkpoint和中途更新点各自保存当时诊断，加载时检查目标函数配置、实际更新数和有限值。
修改合同摘要仍不能绕过目标函数配置与诊断的交叉验证。

`run_source_training_chain.py --stages P2_source`默认生成三个同剂量source臂：CE、SupCon(0.05)、MMD(0.05)。
计划可用`source_objectives`声明其他预先固定的强度，但必须包含CE控制：

```json
{
  "source_objectives": {
    "ce": {"kind": "ce"},
    "supcon005": {"kind": "cross_domain_supcon", "weight": 0.05, "temperature": 0.1},
    "supcon020": {"kind": "cross_domain_supcon", "weight": 0.2, "temperature": 0.1},
    "mmd005": {"kind": "conditional_rbf_mmd", "weight": 0.05, "min_cell_count": 2}
  }
}
```

这只是已有完整source-chain计划中的配置片段，不是可直接启动的完整输入计划。
计划仍须提供attested源/目标cache、实际排他库、协议、source/evaluation快照和batch规则。
P2_source单独执行时默认六导布局，默认source/target seed逐一配对；不把3×3交叉seed误报成三个完整重复。
现有source worker调度、registry、resume、数据准入与CPU/GPU资源限制继续生效。

外层`--stage P2`固定full/scratch/LDA、NN各200更新，源臂写入每个cell。
静态处理复用原`TargetResultReuse`：源初始化键绑定权重/统计，scratch/LDA跨不同source臂可核验复用。
缓存重新读取并验证模型/预测/record摘要，然后写出当前消费记录与原结果身份。
已复用拟合单列，不能作为新的科学重复。嵌套配置在source任务和resume中统一用`asdict`序列化。

## 4. 软件验证与解释边界

定向反例覆盖：

- 空间/时间/读出/BN的实际更新范围；冻结参数不变，未知参数拒绝，推理不改状态。
- 完整校准refit后，替换校准范围以外的信号/标签不改变拟合状态。
- 九臂P1真实执行软件小预算；三种新个人模型实际fit、保存、加载、预测逐值一致；篡改残差架构被拒绝。
- SupCon与显式逐anchor公式相同；重复输入行、同人跨域、缺失身份、单域inactive和有限梯度。
- CE/SupCon/MMD保持同批次顺序和相同draw/update；source更新点与同seed独立短轨迹逐参数一致。
- 各更新点checkpoint携带对应辅助项范围，真实源轨迹保存后可加载；伪造配置/诊断即使重新算摘要也被拒绝。
- P2_source dry-run不拟合；跨不同source权重复用scratch/LDA，并重新拟合源迁移臂。

全量首轮为1840通过、1失败、4跳过（退出码1）。唯一失败是既有Windows缓存并发发布：
hard link的只读属性由目标与临时文件共享，并发复用者可能在发布者删除临时链接前将目标设为只读。
一次孤立复测通过，随后用确定性交错反例覆盖该窗口；未以复测通过掩盖首轮失败。
三处发布统一使用`_publish_without_replace`：Windows采用不会覆盖已有目标的原子rename，
POSIX保留link；原始文件只读取，现存目标仍核验摘要，正常发布后仍设为只读。
该修复相关测试26通过、1个符号链接测试跳过。

最终全量：**1843通过、4跳过，0失败/0错误，退出码0，217.498秒**。
两项跳过因为未设置独立采集端`BRAIN_SYNC_WORKSTATION_ROOT`，两项因为本机符号链接权限不可用。
活动源码、实验入口与测试全量Ruff通过；`git diff --check`通过。
保存真实source CLI两更新软件探针及其中间更新点的合同反例；不把这些合成拟合当真实开发矩阵。
日志保存在本地`experiments/runs/domain_transfer_implementation_20260916/`；
[结构化验收记录](../assets/repository_audit_20261019/docs_historical_index.json)绑定退出码、XML/log摘要与源码快照。

尚待后续的范围明确保留：P1-G的source/target EA拟合与评估联动、真实开发矩阵、
依据结果决定的episodic训练/电极集合编码/基础模型、新成人确认。没有实现蒸馏，也没有生成真实新模型成绩。
现有EA/xDAWN底层函数继续保留；不能据其存在声称P1-G端到端已经完成。
