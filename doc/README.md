# 项目文档

本仓库是公开ERP源训练、目标校准与只读推理后端；BrainSync采集应用是独立项目。

## 规范入口

稳定规则由六指南按职责维护，其他文件通过链接引用。

| 指南 | 唯一维护职责 |
|---|---|
| [研究规范](research_program.zh.md) | 九选一目标、通用数学、数据角色、科学比较与研究纪律 |
| [技术结构](technical_structure.zh.md) | 模块所有权、接口与完整调用链 |
| [输入合同](input_contract_math.zh.md) | 通道、参考、单位、时间支持与数学关系 |
| [执行计划](roadmap.zh.md) | B00–B14的范围、依赖与验收出口 |
| [工程指南](engineering.zh.md) | 环境、验证、资源、交付与旧构建清理 |
| [工作站](workstation.zh.md) | 采集、校准、个人包与只读推理 |

## 当前状态与续接

[当前checkpoint](assets/research_resume_20261012/pause_checkpoint_current.json)保存运行状态、进程身份、在途工作、资源限制和恢复顺序。
[交接入口](assets/research_resume_20261012/HANDOFF.zh.md)说明如何使用该记录。
先读checkpoint，再核实际源码、Git和进程；当前暂停或执行状态只在checkpoint维护。

## 证据与历史

结论、实验读数和接受范围维护于四个[主题页](evidence/README.md#当前主题结论)：
[校准与选择](evidence/calibration_and_selection.zh.md)、[响应与噪声](evidence/response_and_noise.zh.md)、
[公共评估](evidence/public_evaluation.zh.md)、[执行与集成](evidence/execution_and_integration.zh.md)。
[整合绑定](assets/research_resume_20261012/research_integration_acceptance.json)保存接受来源及产物指针。

[证据索引](evidence/README.md)提供历史检索；[大版本恢复清单](../frozen/major_domain_transfer_20260920/README.zh.md)
保存版本恢复关系。载荷分发见[归档规则](../frozen/README.md)，本次规范整理的原文与哈希见
[整理清单](../frozen/guidance_consolidation.manifest.json)。
