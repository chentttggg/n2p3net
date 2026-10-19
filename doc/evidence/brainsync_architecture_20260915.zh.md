# BrainSync工作站架构修复与验收

日期：2026-09-15。采集端为`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`；后端为本仓库。
本轮使用codebase-design、systematic-debugging、diagnosing-bugs及test-driven-development，
依次处理结果接入、确认事务、采集会话module、环境与退出。没有真人采集、云训练或效果晋升。

## 来源与保留

改动前物理快照包含363份前后端源文件、测试及活动指南，逐成员SHA-256已验证，见
`frozen/brainsync_architecture_20260915_before/manifest.json`。采集端不在Git中；后端已有并行工作，
本轮没有重置、清理或提交其他改动。原始数据、个体模型和历史结果保留。

旧Qt测试存在副作用：构造MainWindow时读取操作员配置，自动启动inspect并保存设置。
诊断中产生的inspect目录作为诊断记录保留；它不是训练。现在单元测试和模拟验收均注入临时app_root，
不再读写真实操作员配置。显式指定已有session的只读查看仍可使用。

## 修复与反例

1. `backend.ModelTask`承担命令构造和结果准入。旧前端只校验task，另一会话的数字可被显示；
   新interface同时核对schema、状态、request_id、session_dirs/session_id、subject_id、模拟声明、
   模型路径与SHA-256。inspect/infer模型摘要取自实际加载的同一份字节；消费者再核对请求前后模型字节。
   零epoch拒判也有完整身份和摘要，不因无证据绕过门禁。原任务记录不重写。
2. `SessionStore`先保存确认意图，再持久化审计事件与标签，最后移除pending标记。
   两端接入拒绝pending；分析页可恢复。恢复校验原事件前缀及会话语义摘要，只修复本次事务的未完整尾行，
   外部冲突拒绝自动恢复。活动/结束后确认走同一实现，长期事件句柄使用append，避免外部追加后旧文件位置
   覆盖确认日志。反例覆盖日志写失败、manifest发布失败、commit标记清除失败、部分尾行、冲突及幂等重试。
3. `RecordingSession`持有准备、真实样本支持、暂停/恢复、标记、EDF/BIDS导出和终态提交。
   主窗口只发命令、显示进度和处理事后确认。并发finish等待同一次收尾；标记收尾失败仍停止并导出原记录。
   晚到的采集完整性故障覆盖completed；EDF成功但SDK清理警告保留可解释的完成状态。
   原依赖MainWindow私有字段的收尾测试已迁至此interface，保留实际文件与故障结果断言。
   废弃操作绑定本次实际创建会话的目录，显式另选保存目录时也保留正确tombstone且不越界。
   SDK启动步骤保留原生Future；取消后先等申请结束再清理，已有样本先落盘，空writer关闭句柄。
   SDK自行取消Future时也执行反向清理；若native请求一直未结束，则保持recovery并禁止并发退订/关闭。
4. 前端依赖已声明并安装到自己的Python3.10虚拟环境，`uv.lock`固定完整版本。
   默认不扫描Temp/其他工程；显式SDK目录覆盖需主动配置且写日志。SDK发行版本0.3.0、运行时自报0.2.0
   分别保留；安装前后入口及原生扩展字节一致。
5. `ModelJobRunner.shutdown()`关闭任务接入并回收进程树；`MainWindow.shutdown()`停止定时器并等待各owner，
   应用main在直接quit后也完成清理再关闭qasync。旧挂起由未回收的自动inspect子进程触发：Windows IOCP
   取消回调在Qt循环关闭后无法完成；空循环和已回收进程对照正常。真实qasync的app.quit回归已覆盖运行中的
   检查任务，排队的启动回调在shutdown后也不再创建连接。

## 验收口径

最终检查结果：

| 检查 | 结果 |
|---|---|
| 前端完整`tests/run_headless.py` | 221项通过，72.101秒，退出码0；包含原来挂起的Qt故障/退出测试 |
| 后端完整`pytest -q -p no:cacheprovider` | 1816项通过、4项跳过，退出码0；进度字符账本与原始日志一并封存 |
| 前后端Ruff、Git whitespace检查 | 全部通过 |
| 真实前端Python→后端CLI→ModelTask结果接收 | 合成个体包inspect及无标签infer通过，同一模型SHA-256保持不变 |
| 模拟暂停后继续、终止保存、废弃 | 分别得到completed、aborted、仅保留tombstone；恢复/无效试次标记与分段时间轴保留 |
| 45闪烁模拟采集→后台质量报告→原生查看 | 45个epoch、4张图，未确认目标、未训练分类器、原始EDF摘要不变 |
| 原生窗口布局 | 1440×900、1160×720及大字号共11项布局检查通过；报告700宽度可查看四图，已检查截图 |
| SDK环境迁移 | 依赖从采集端自己的.venv加载，uv pip check通过；两份SDK入口/二进制文件与原环境逐字节一致 |

后端套件保留既有GTN数据对应警告、Braindecode弃用提示及Torch padding提示；
前端故障测试中的SDK异常输出为显式注入，最终测试结果为通过。

定向反例、全套退出码、模拟采集和图形结果与最终源码一同封存到
[`frozen/brainsync_architecture_20260915_validation/manifest.json`](../../frozen/brainsync_architecture_20260915_validation/manifest.json)。验收只证明软件行为；
真实设备取消语义、光学时间及成人九选一准确率仍需独立实机实验，不以模拟检查替代。
