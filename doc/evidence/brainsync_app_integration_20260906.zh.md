# BrainSync主工作站整合

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。用户在整体审计和版本比较之后授权整合为功能完整的前端app。
活动采集端选择`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`；旧目录只作来源对照，
不在两个目录同时维护新功能。算法后端仍为`D:/BCI/n2p3-net`。

## 产品流程

1. 连接设备、检查七EEG加独立EOG和接触状态。
2. 为同一人采集多个已确认目标session；九数字各10block×45闪烁为4050trial预算。
3. 在前端选择本人校准记录，选择源模型微调或明确标注的本人LDA基线；在后台拟合并保存个体模型。
4. 新测试只采1block×45闪烁，录制完成后自动推理，真实目标仍未确认。
5. 显示九候选证据、R1-R5结果与拒判原因；随后确认真实目标，保存实际命中结果。

本轮实现block完成后的自动判定，不把它称为已实现每个SDK chunk的实时神经网络输出。
没有真实采集验证前，不承诺某个人的AUC、猜数字准确率或光学触发精度。

## 环境与接口

- 主工作站新建自己的`.venv`，Python3.10.20；复用已发现且minor匹配的SDK/Qt依赖。
- 模型后端使用n2p3现有`.venv`；不把Python3.10原生SDK路径注入算法环境。
- 前端复用`ModelJobRunner.start_backend(program, arguments, task=..., output_dir=...)`，
  使用结构化argv，不拼shell命令；状态、进度、取消、日志与现有设备互斥规则联动。
- 后端入口`experiments/brainsync_model_task.py --task calibrate|infer`，输入多份`--session-dir`，
  可选`--source-checkpoint`、显式参考/EOG模型；输出个体artifact和`summary.json`。
- 已标注科研接入与无标签推理接入分开，复用真实raw/时间/通道校验；不得伪造target标签。
- 个体artifact绑定participant、校准session/可用时间、模型状态、输入统计、预处理、通道、
  参考及EOG状态。推理不能消费校准前或重复校准记录，后续补写真值不能改变预测。

## 进度

| 工作 | 当前状态 |
|---|---|
| 整体n2p3审计、AS-MMD/refit、入口与双snapshot修复 | 完成；原15文件定向集成412通过 |
| 晚确认时刻与EOG session预算调用修复 | 完成；新增9项，关联172项通过 |
| 两版只读比较与风险反例 | 完成；发现录制增强及坐标/名称回退、finalize持锁等待 |
| 主工作站物理备份及项目`.venv` | 完成；SDK/Qt/NumPy加载验证通过 |
| 录制并发、恢复与结构化后端任务接口 | 完成；并发finalize、失败重试、订阅互斥、进程树取消及解释器环境隔离均有反例 |
| 七EEG/EOG与正确head坐标合并、无标签接入 | 完成；新旧producer往返、目标补填不变性、模拟声明和错误记录门禁通过 |
| 个体校准选择、模型持久化、完整候选推理CLI | 完成；calibrate/inspect/infer，23项后端反例通过 |
| 前端校准记录管理、模型操作、新测试自动判定 | 完成；校准/推理/高级任务分层、结果后确认、收尾互斥和可取消任务 |
| headless GUI、真实双解释器调用与故障/重载测试 | 完成；实际Qt录制到源模型加EOG校准、保存重载、新pending目标R5及反馈全链通过 |
| 实机、光学时钟与真人准确率 | 未验证 |

最终回归：工作站118项headless测试通过，后端16份文件440项定向测试通过（75.44秒），
均无失败/跳过。后端5条warning来自合成EDF末块和极小LDA fixture。
一次并发调度测试原先使用后台线程抢读原子替换中的journal，Windows临时访问拒绝使
观察线程退出；已改成真实持久化完成后捕获状态，保留派发/drain/resume原断言，未改生产重试。
初次失败日志保留，最终回归已通过。
记录：[后端JUnit](../assets/technical_audit_20260906/final_backend_tests.xml)、
[前端日志](../assets/technical_audit_20260906/app/final_frontend_tests.log)。

## 冻结

修改前主工作站73份源码/配置/资源已逐成员校验压缩，原始数据与native二进制保持原位置：
[manifest](../../frozen/brainsync_newest_before_integration_20260906.manifest.json)，SHA256
`e04096d3cd4bb4fbb3a0f40a6e43f3d3f0bd8f4a8e6fcf74f7aea5e82777b7d9`。

后端新功能前的220份源码/指导另行冻结：
[manifest](../../frozen/technical_audit_before_app_20260906.manifest.json)，SHA256
`25fc6326e425e4a525fe1fae319c8fd157daf2235846c9aa4165c65a618df6e6`。
不修改既有raw或历史成绩，不用新源码hash重标旧实验。

## 启动与交付

以下为独立采集端`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`中的本地文件，
不随本仓库GitHub分发；当前操作见[工作站指南](../workstation.zh.md)。

- 设备模式：`Launch_BrainSync.cmd`。
- 模拟模式：`Launch_Mock.cmd`。
- 运行指南：`RUN_GTN_WORKSTATION.md`。

默认源模型为已验证可加载的BNCI Cz6研究checkpoint，复制后SHA256为
`0105bb90dc4a5d9a2788a55ff0e183a230e8e5cf8df20dbd601848c67b2221be`。
模型选择和Cz参考在界面显式呈现。它是公开ERP初始化，不是已验证真人性能的个体模型。
LDA单session不产生内部验证指标；神经网络必须有至少两份独立校准session。

## 整机证据

[实际Qt端到端记录](../assets/technical_audit_20260906/app/gui_e2e.json)使用真正的MockDeviceAdapter、
LocalEdfRecorder和生产BIDS导出函数，先采两份声明模拟的校准session，再由前端启动n2p3
源模型微调与眼电回归，保存并通过inspect重载。随后新1block的45个epoch全部进入
无标签推理，R5每个候选计数都为5；summary先完成，才调用操作员目标确认和反馈保存。

该模拟信号没有注入P300，最终预测2、事后确认5、反馈为未命中。这不是性能失败判定或
真人准确率估计；反而防止用设计好的模拟目标伪造“猜中”的产品演示。
神经网络自动EOG仅用内部训练session拟合，其状态冻结贯穿验证/refit/推理。

界面在1440×900和1280×800下抓取实际Qt图像；修复了小窗口结果行重叠，完整九候选
采用固定最小表高，空间不足时整页滚动。参考[结果界面](../assets/technical_audit_20260906/app/r5_result_1440x900.png)
与[小窗口布局](../assets/technical_audit_20260906/app/result_layout_1280x800.png)。

旧4份completed记录保留原raw/旧通道声明，不因新默认通过就标成已经迁移。
有基线窗/分母问题的4个独立诊断脚本及旧修复报告已移出主目录；15份旧诊断产物已压缩
至[归档](../../frozen/brainsync_retired_diagnostics_20260906.manifest.json)，不混入应用当前结果。
