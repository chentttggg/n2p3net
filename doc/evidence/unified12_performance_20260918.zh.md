# 十二库云训练：停训优化、验证与恢复（2026-09-18）

用户要求暂停当前训练，调用性能技能优化后再启动，并固定今后云端CPU总线程预算至少25、GPU workers至少4。
已应用performance-optimization、get-available-resources及codebase-design；保留首折已完成的参考源和A0/A1共3份模型。
验证完成后恢复完整矩阵：4个普通整库留出、1个BI相关联合排除、随后单独做完整12库部署源，seed=20260918，Lee不纳入。
本记录是运行与性能证据；完整训练、保存模型重放和归档已完成，准确率与机制结论见[最终结果分析](unified12_results_20260918.zh.md)。

## 输入和执行身份

12库共1964条原始record声明；BI2012的3号online记录因P4全程恒值被已有门禁拒绝，保留失败凭证，
其余1963条准入。12库仍全部参与source，共1430742个epoch，499个库内参与者身份。
四个公共目标的独立C_K/test视图已完成。GTN和EPFLP300不满足冻结的单记录独立C200/query支持，
只从源内调参域排除，不从source训练池删除；其余10库的合法调参支持已冻结。

原始准备绑定实际执行快照bac9654edfe63610f4775aa34569efc524c782eac81a0496b35a9e0ac79a54f7。
首折已完成模型绑定runtime_v2快照5701d51f04dc1111e30f1bb9058f5e8992383abae3860250a1c8b87e2c3bcbce。
优化后执行快照为c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24，
见[frozen清单](../../frozen/unified12_20260918_performance_v4/source.manifest.json)。旧模型通过resume清单复用，不改写其执行身份。

## 修复和资源接口

- 源链补齐显式调参支持域及排除原因；部署source组有独立role，禁止生成外层评价任务。
- 有真实membership时增加与Q0一致的CE-only scratch，保留decision scratch；无membership时等价处理只保留一份。
- 大源metadata超过NumPy单个Unicode标量大小上限，改为UTF-8 uint8缓存，保留旧格式读取，解码摘要不变。
- CPU预算由资源解析器传入实际source、源内λ及target worker，移除这些调用处的固定单线程限制。
- 同序通道在source路径显式共享只读X；全行refit不重复复制X；统计量分块；GPU成功预加载后释放归一化CPU副本。
- GPU预加载使用父调度器批准的常驻字节预算，避免后启动worker因“剩余显存30%”而意外退回CPU传输。
- 大cache读取及输入暂存共用峰值门禁；GPU拟合容量为4，嵌套λ worker受全局容量约束。

集中配置见[cloud_training_resources.json](../../configs/cloud_training_resources.json)：CPU预算25，源/目标worker容量4，
每worker数值线程6，另有调度线程；CPU/显存/主存不满足声明下限时显式失败，不静默串行。
源worker预算按常驻18 GiB加共享暂存10 GiB解释，不能将单worker瞬时峰值与四份常驻简单相加。
依赖末端不足四个就绪任务时记录实际任务数，不通过重复拟合填满worker。

## 验证

| 验证 | 结果 | 边界 |
|---|---|---|
| 全套软件测试 | 2104通过、4环境跳过；Ruff及diff检查通过 | 软件正确性 |
| 默认BF16，60000真实行上相同8次A1短拟合 | 44.27秒→32.45秒，1.36倍吞吐 | 含加载及进程启动；不是完整矩阵加速比 |
| 固定FP32及确定性算子，对照相同8次短拟合 | 41.05秒→23.48秒；统计、采样序列、预测完全一致 | 用于分离算法等价性与混合精度自动调优影响 |
| 1367058行完整输入的单worker路径 | 峰值RSS约19.47 GiB | 包含完整核验与10次更新；不是长期训练内存保证 |
| 四worker同时运行大源路径 | 每worker拟合1102996行、200次更新；4/4 GPU预加载成功，无传输回退、OOM重试为0 | 以同步起跑屏障验证同时驻留；548秒包含数据准备和等待，不能作为正式fit速度 |

四路验证的单worker历史峰值最高23.06 GiB；峰值并不同时发生。暂存门禁限制大块分配同时发生，
实测无cgroup OOM或OOM kill，但出现memory.high回收压力，后续仍须按遥测核查实际内存和吞吐。
默认BF16自动调优下新旧训练轨迹不满足1e-4的逐预测比较；受控FP32比较为零差异。
不将默认混合精度训练声称为逐位可复现，也不通过放宽重放容差掩盖保存模型问题。

## 最终状态

正式执行器完成5个评价source组、全部目标对照和全12库部署源；保存模型重放48/48通过，归档19779个成员并完成逐成员校验。实际资源记录为CPU预算25、GPU slots4、target workers4、原生线程6。
[结构化摘要](../assets/unified12_performance_20260918/summary.json)保存执行身份、测试摘要与性能范围。
完整日志、计划、遥测、模型和运行产物留在experiments/runs/unified12_20260918及对应云端运行目录。
