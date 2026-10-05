# BrainSync冻结输入重放：实现与软件验证

日期：2026-09-09。范围为后端已保存个体包到无标签完整记录的接入；
不包含公共矩阵训练、已有模型重训、滤波默认裁决或成人效能评估。

## 已复现的问题

同一个无EOG/连续QC的Cz重参考个体包，在推理CLI可以输出结果，质量报告却报
`the measured reference must be observed in every projected trial`。
报告先按最终通道列表移除了Cz，再尝试重参考。已先把此反例加入测试，确认原生参考通过、Cz失败。

另一条反向路径是连续QC已执行Cz参考且模型只保存Pz/Fz子集：旧推理传入`channels=None`，
接入产生全部六导后再与保存子集核对，导致拒绝。两条路径需要统一处理顺序与参考归属。

连续QC全拒会在返回EpochDataset前抛出`BrainSyncNoUsableEvidenceError`。
旧推理CLI在异常分支补查人员/时间，报告只在成功返回数据后查模型合同，异常路径不一致。
EOG人员核验原本也位于数据生成后，可能被全拒异常跳过。
独立复核进一步用合成BIDS和真实flatline QC复现：改变电极坐标，非空记录按输入合同拒绝，
全拒记录却提前成为无证据结果；最终补丁将可预先确定的最终通道几何也纳入准入。

## 责任与不变量

`CalibratedPredictor.load_inference_session(session_dir)`集中冻结输入重放：
读取包内原PreprocessingSpec、EOG及连续QC；核对request与保存输入合同；
安排参考与通道选择，返回已核验的无标签EpochDataset。该interface不拟合、不预测，
不接受临时处理覆盖或独立科研时间豁免。

`data.brainsync`持有参考协调与应用。EOG或local QC实际声明并执行请求的参考时，
raw接入可直接选择最终子集；只在epoch后执行参考时，先保留实测参考通道。
已有线性基线及实测通道/坐标门禁继续生效，不改变算子或缺失通道语义。

录制准入先满足：人员等于个体包人员、session不属于校准集、
`session_start_timestamp_s > calibration_available_timestamp_s`、模拟声明一致，
以及存在时的保存采集参考声明匹配。时间恰好等于就绪时刻仍拒绝。
现有无标签loader在同一次BIDS核验后调用recording_validator，再执行信号处理；
EOG人员归属也提前到实际应用之前。普通准入错误保留错误语义，不能变成QC拒判。
native及raw已参考路径使用已有坐标解析和float32结果比较保存几何；后置参考先核验标准输出几何。
后置参考对原始坐标的检查依赖实际观测mask，仍由原投影执行，未扩大对始终被QC屏蔽通道的拒绝。

合法记录全拒时，原45/450闪烁分母保留，推理拒判，报告给出零可用epoch。
部分拒绝继续使用原排程、evidence_indices及合法NaN，不缩小词表或替换失败证据。
产品接入保持前瞻规则；已有独立科研recording_scope仍由原显式入口处理。

推理和个体包报告共同调用上述interface；基础质量报告复用data参考规则，
不再导入另一个CLI的私有函数。旧helper删除，调用者与相关测试转向所属模块。

## 验证

- 原Cz报告回归先失败，再通过；现有报告、个体模型和参考测试共58项通过。
- 新增28项interface及双CLI测试通过：原生、epoch后Cz子集、EOG Cz子集、QC Cz子集、
  native QC加epoch后Cz、EOG与QC共同参考，以及旧无陷波包。
- 新测试同时覆盖保存通道顺序、实际张量/事件映射、模型字节和metadata不变；报告禁止分类器执行。
- 异人、校准session复用、过早或等时、模拟声明不符在信号处理前拒绝；
  合法全拒仍保留45个事件；自身request与模型通道、参考、EOG/QC记录不一致先拒绝。
- 几何修正后的85项相关测试通过，另补错EOG人员与真实QC全拒的反例1项通过。
  无证据几何反例经独立复核，在正常/全拒两种输入均为普通合同错误。
- 相关Ruff通过，两个CLI从仓库外目录运行`--help`通过；最终全套结果见下方收尾记录。

所有新记录为临时合成BIDS。合成LDA/EOG/QC拟合只验证软件行为，
不作为真实校准收益、九选一准确率或成人验证证据。

### 最终验收

最终工作树全套：**1650通过、4跳过，235.03秒**。4项跳过分别为两个未配置独立采集端根目录的
集成测试，以及两个当前Windows权限不支持的符号链接测试；不将跳过算为通过。
本次8份源码/测试文件在最终全套前后SHA-256一致；相关Ruff、`git diff --check`通过，
两个CLI从仓库外目录加载通过。独立复核无剩余阻塞项。

全仓活动文件Ruff额外发现并发新增`experiments/audit_bidirectional_drift.py:12`有一项`I001`；
该文件属于另一项正在进行的修改，本次没有改写，故不声称整个并发工作树Ruff全绿。
最终源码摘要、跳过原因及压缩验证日志见[验收清单](../../frozen/brainsync_input_replay_20260909_validation/manifest.json)。

## 改动前源码

改动前11份相关源码、测试及活动指南已经物理压缩，逐成员字节和归档SHA-256均核验。
见[归档清单](../../frozen/brainsync_input_replay_20260909_before/manifest.json)。
该归档是所列文件的局部历史，不是可执行训练快照；同时保留当时dirty工作树的实际字节。
