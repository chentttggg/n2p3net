# 项目文档

本仓库维护公开ERP源训练、目标校准和只读推理后端。BrainSync采集应用是独立项目。

## 规范入口

六指南各自维护稳定规则，其余文档通过链接引用。

| 指南 | 维护职责 |
|---|---|
| [研究规范](research_program.zh.md) | 九选一目标、通用数学、数据角色与科学比较 |
| [技术结构](technical_structure.zh.md) | 模块所有权、接口和实际调用链 |
| [输入合同](input_contract_math.zh.md) | 通道、参考、单位、时间支持与数学关系 |
| [执行计划](roadmap.zh.md) | B00–B14的范围、依赖和验收出口 |
| [工程指南](engineering.zh.md) | 环境、验证、资源、交付和旧构建清理 |
| [工作站](workstation.zh.md) | 采集、校准、个人包和只读推理 |

## 数据与结论

[训练数据与精简结果](training_data.zh.md)提供保留资产和关键结果入口。[四个证据主题](evidence/README.md)维护结论、实验读数和适用范围；结构化来源保留在原`doc/assets/`绑定中。

## 当前状态与续接

先读[科研checkpoint](assets/research_resume_20261012/pause_checkpoint_current.json)和[交接入口](assets/research_resume_20261012/HANDOFF.zh.md)，再核实际源码、Git、进程和输入。当前运行、暂停位置和下一动作只在原checkpoint维护。

[负责人交接报告（2026-11-07）](project_handover_20261107.zh.md)解释项目演进、术语、结果边界和接手顺序，是交接当日的快照。它记录的部分运行载荷原位已缺失；恢复定位见[历史索引](evidence/history.zh.md#运行载荷缺口)，不能据旧状态自动恢复任务。

[构建清理维护](build_cleanup_maintenance.zh.md)记录项目级清理事项与实际回执，细分运行只引用该入口。

## 历史与恢复

[历史索引](evidence/history.zh.md)按主题检索旧报告，并提供原路径、原始字节SHA和压缩包成员目录。[大版恢复清单](../frozen/major_domain_transfer_20260920/README.zh.md)维护版本恢复关系；载荷分发遵循[归档规则](../frozen/README.md)。历史页、旧源码与模型保留原执行身份。
