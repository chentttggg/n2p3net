# BrainSync工作站指南

更新：2026-09-06。活动采集端为`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`，
后端为`D:/BCI/n2p3-net`。两个版本的增强已整合到newest，不在旧目录同步开发。
本指南汇总当前使用方式，历史测试次数和修改细节见[整合证据](evidence/brainsync_app_integration_20260906.zh.md)。

## 启动与环境

采集端是单独交付的本地项目，不包含在本仓库的GitHub clone中。
以下路径均相对采集端根目录`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`：

- 设备模式：`Launch_BrainSync.cmd`。
- 模拟模式：`Launch_Mock.cmd`。
- 详细运行指南：`RUN_GTN_WORKSTATION.md`。

采集端使用自己的Python3.10/SDK/Qt环境，模型任务通过结构化argv调用后端项目`.venv`。
模型任务支持进度、取消和日志，并与设备/录制生命周期互斥；运行问题先查实际解释器与日志。
模式必须明确，模拟数据只验证软件流程。

## 校准与新测试

1. 连接设备，核对电极角色、参考、真实波形、阻抗来源及接触变化。
2. 同一人采集已知目标校准session，结束后确认目标；预算与排程以[研究规范](research_program.zh.md)及[采集计划](../configs/brainsync_collection_plan.json)为准。
3. 在前端选择本人校准记录，选择源神经网络微调或明确标注的本人LDA基线。
4. 后台拟合并保存个体模型；可检查、重新加载模型，再开始独立新测试。
5. 新测试只采一个45闪烁block，录制完成后先无标签推理，显示九候选证据、R1–R5和拒判原因。
6. 预测保存后才确认真实目标，写入实际命中反馈；后补目标不得改变既有预测。

trial是一次闪烁。一个目标的10个block不是10次独立目标选择。

实验屏按ESC进入暂停菜单，并暂停向EDF缓冲追加样本；设备连接保留。
菜单可继续测试、终止并保存部分记录，或废弃本次测试。
继续时先恢复写入并显示3秒提示倒计时，再重做被暂停打断的试次；候选顺序不重新随机。
被打断的attempt有独立编号和无效标签，只有完整重做的attempt进入有效试次。
保存中止结果标记为`aborted`，不自动用于校准或推理；废弃会删除本次会话文件，
只在`discarded/`留下会话编号与废弃时间。ESC不会直接执行这两种结束操作。

手动暂停不保存休息样本，EDF使用拼接后的样本轴，session与BIDS声明
`segmented_recording`、暂停墙钟区间、断点sample和`RecordingType=discontinuous`。
EDF注释与BIDS事件使用该样本轴；后端在断点重置滤波并拒绝跨断点epoch。
计划内block休息仍保存并标记，不能和手动暂停混淆。

所有数字可参与本人校准，随后测试可重复这些数字；独立性在于新的EEG与标签没有参加拟合或选方。
当前是block完成后的自动判定，逐SDK chunk流式模型推理仍未实现。

## 模型与信号合同

采集槽顺序与七EEG/EOG角色由[研究规范](research_program.zh.md)和
[输入合同](input_contract_math.zh.md)定义；原PO7槽采PO8，原PO8槽采独立EOG。
旧raw保持采集时声明，不能因新默认生效就静默更名或认定完成迁移。

源NN要求至少两份独立校准session供内部选择；固定LDA至少一份含两类的session，
单session不产生内部验证分数。NN只在内部训练session拟合自动EOG，冻结后贯穿验证、refit和推理。
默认BNCI Cz6研究checkpoint只是公开ERP初始化，参考适配须显式核对，不代表真人性能已验证。

个体包绑定participant、校准session及标签可用时间、源checkpoint、模型参数、输入统计、
通道/参考/预处理与EOG状态。推理拒绝重复校准记录、早于校准就绪的记录及不一致输入合同。
模拟来源声明不能在导出或重载时丢失。科研有标签接入与无标签推理各走其生产入口。
后端任务入口为`experiments/brainsync_model_task.py`，具体参数以`--help`为准。

## 阻抗与实机边界

模拟阻抗约1.9–4.65 kΩ，与真实接触无关，不能据此认证接触合格。
设备SDK阻抗按kΩ原值显示，当前软件回调测试未发现额外缩放或2–4区间截断。
用户此前低值现象发生时的模式尚未确认，因此不能认为真实电极或固件问题已排除。
若在设备模式仍出现固定低值，下一步同步记录原始SDK回调、通道与实际接触/断开变化。
详见[阻抗核查](evidence/impedance_display_review_20260906.zh.md)。

已完成的是声明模拟的Qt录制、BIDS导出、真实源checkpoint校准、EOG、保存重载、
新45闪烁无标签推理及事后反馈的软件链验证。真实设备、光学刺激时间、丢包影响及成人
准确率仍按[执行计划](roadmap.zh.md)验收；旧EXE或历史压缩快照不自动包含后续源码修复。
