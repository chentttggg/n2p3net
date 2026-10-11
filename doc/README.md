# 项目文档

本仓库是公开 ERP 源训练、目标校准与只读推理的后端；BrainSync 采集应用是独立项目。当前规范由下列六指南维护，源码和实际执行产物优先于 dated 状态。

| 指南 | 内容 |
|---|---|
| [研究规范](research_program.zh.md) | 产品估计量、数据角色、科研约束与结论边界 |
| [技术结构](technical_structure.zh.md) | 实现、接口、所有权与调用链 |
| [输入合同](input_contract_math.zh.md) | 通道、参考、采样、时间支持与数学关系 |
| [执行计划](roadmap.zh.md) | 当前研究块、依赖、配方和验收 |
| [工程指南](engineering.zh.md) | 环境、资源、源码身份、验证与维护 |
| [工作站](workstation.zh.md) | 采集、校准、个人包与只读推理 |

## 当前研究

2026-10-11继续按[愿望清单当前进展](evidence/wishlist_progress_20261010.zh.md)推进123项，恢复科研树已完成 B03 严格 nested/source-study 与 B07 `/4` native family 的 bounded 软件验收；B09 source-group 现保存 exact structural null 与派生语义关系，并通过本地 source/B05 组合验证。新证据见[科研恢复与集成](evidence/research_resume_20261011.zh.md)。这些出口不代表 inner 持久重放、公共效能、跨host整包或成人确认。单个数值门不作为所有方向的串行前置。

科研窗口负责科学合同与代码最终集成，云端窗口负责唯一执行调度。科研用一个完整 source→target seed，大版本校验才三个；两窗口合计最多10并发，科研子代理没有我方时间或额度上限。职责内自主推进，只集中同步重要交付、接口变化和阻塞。

[B00–B14 分块方案](evidence/deep_research_plan_20261005.zh.md)覆盖123项愿望；当前 B12 基础公共流程不等待所有 P1 机制。B14 执行底座、单seed完整源与两个公共面板按实际冻结源码、producer和requested账本验收；不能把准备、软件测试或已有子集成功称完整效能。

[B07字典、B08可识别先验及B05/B06/B09机制集成](evidence/research_integration_20261008.zh.md)分别追踪真实代码、独立反例、保存重放和未完成项。集成前仔细审阅相关模块逻辑；局部通过不替代完整调用链。完整结果、来源与理论见[证据索引](evidence/README.md)。

## 历史大版本

[域迁移与BrainSync稳定大版](evidence/domain_transfer_major_20260920.zh.md)保留09-18最后完整公共训练的执行来源，以及实际对应09-20发布EXE的工作站源码、默认源和恢复接口；二者来源分别说明，不混作同一个部署模型或成人效能。模型、原EEG及最终结果保持原位置。

[大版恢复清单](../frozen/major_domain_transfer_20260920/README.zh.md)保存必要代码、依赖、模型身份与最终验收；不长期保每次修复、失败、复审和重复冻结。结果落地后清理无用过程材料，当前问题只保仍必要的骨干和证据。

## 维护与分发

状态与使用方式更新原六指南，独立科学结论进入证据索引。源码、当前指导和紧凑manifest进Git；原数据、模型、最终结果及大版载荷本地保存，见[版本与分发规则](../frozen/README.md)。一次性工具完成用途后删除，历史指导不作为当前执行入口。
