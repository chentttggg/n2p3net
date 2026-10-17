# 科研续接入口

当前状态、进程身份和恢复顺序统一读取[checkpoint](pause_checkpoint_current.json)。
本页维护续接方法；稳定规则见[文档索引](../../README.md)。

1. 读取checkpoint中的工作树、源码冻结和各任务原始暂停记录，核实际Git、PID及创建时间。
2. 对保留内存的任务，使用其原暂停记录指定的恢复方法，继续同一进程；尚未启动的任务按checkpoint安排。
3. 活动数值进程释放源码冻结后，再审阅、集成外部草案；按已有实际证据补齐新出口。
4. 科学方向依[研究规范](../../research_program.zh.md)，工作依赖依[执行计划](../../roadmap.zh.md)。

## 按需检索

| 内容 | 原始记录 |
|---|---|
| 接受范围、提交和来源 | [整合绑定](research_integration_acceptance.json) |
| 模型与数值结论 | [响应与噪声](../../evidence/response_and_noise.zh.md) |
| 校准与选择结论 | [校准与选择](../../evidence/calibration_and_selection.zh.md) |
| 公共数据评价 | [公共评估](../../evidence/public_evaluation.zh.md) |
| 执行、恢复和清理证据 | [执行与集成](../../evidence/execution_and_integration.zh.md) |
| 整理前交接与指南 | [原文恢复清单](../../../frozen/guidance_consolidation.manifest.json) |

各任务的模型、事件、原始数据和暂停记录保持原位置；精确路径由checkpoint或对应接受绑定提供。
