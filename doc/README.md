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

2026-10-16科研继续，当前入口见[交接](assets/research_resume_20261012/HANDOFF.zh.md)。本窗口临时并发上限为6（含总监）。云owner已观察到完整candidate零fit恢复退出0：6080请求、8份原fit复用、0新fit，CPU峰值24.89核，观测副本已落地。用户当前关闭云端，304任务及8个parent的最终逐项绑定在重新开云后先落地，再继续已接受S5的后五阶段。

科研集成树位于`D:/BCI/worktrees/research-director`，实际接受范围见[整合绑定](assets/research_resume_20261012/research_integration_acceptance.json)。已合入source共享闭包、并发产物预算、显式shared AR启动因子及原保存root快速严格准入；Linux同FUSE四写者三组配对的进程耗时中位数改善5.55倍。BI原MAT01的108格与MAT57的90格均完成完整parent正验、概率篡改负例及真实capsule/内存归还；两个一epoch开发样例接下来的效能比较采用完整配方。紧凑图两切片已接受，原K950图2.06GB变47.16MB，原逻辑与90节点保持。B11 typed/generic链、真实H C68 source准入和消费者组件缓存已集成，统计future读取12→0；实际31导C500及7/7原OOF通过原政策，独立single-record连续λ完整7行已接受；C内因果Beta校准将平均NLL319.37降至3.76，命中0/7，后续响应/排序机制已进入实施，详见[响应与噪声](evidence/response_and_noise.zh.md)。快速积分核、generic原fit/capture复用及独立future继续。H13人metadata、最后session819请求及record-aware输入接口已集成。

当前按[愿望清单进展](evidence/wishlist_progress_20261010.zh.md)推进123项。B03真正prefix整程序选择及保存重放/public parent准入已集成；修后entry005完成八人152请求，选择收益随K变化。P1新stationary law两臂16/16完成、0新fit，winner与旧law一致。source cache/shared score、conditional公共白化及逐decision horizon已接入。BI61人与BNCI8人的episode三臂队列已完成并独立复算；BNCI微调的NLL改善集中A04，scratch命中/Brier较好。当前科研独立树从保留的`f1d0ec0`恢复，源码和Git证据已恢复；原BI/BNCI报告、entry005/P1结果及P1原score/receipt已按原SHA恢复到D盘持久目录，逐预测复算和恢复清理来源见[原载荷绑定](assets/research_resume_20261011/cloud_payload_recovery_binding.json)。结论集中维护在[校准与选择](evidence/calibration_and_selection.zh.md)、[响应与噪声](evidence/response_and_noise.zh.md)、[公共评估](evidence/public_evaluation.zh.md)和[执行与集成](evidence/execution_and_integration.zh.md)；六指南持有活动规范，各产物保持自己的来源与分母。

科研窗口负责科学合同与代码最终集成，云端窗口负责唯一执行调度。科研用一个完整 source→target seed，大版本校验才三个；两窗口合计最多10并发，科研子代理没有我方时间或额度上限。职责内自主推进，只集中同步重要交付、接口变化和阻塞。

[B00–B14 分块方案](evidence/deep_research_plan_20261005.zh.md)覆盖123项愿望；当前 B12 基础公共流程不等待所有 P1 机制。B14 执行底座、单seed完整源与两个公共面板按实际冻结源码、producer和requested账本验收；不能把准备、软件测试或已有子集成功称完整效能。

[B07字典、B08可识别先验及B05/B06/B09机制集成](evidence/research_integration_20261008.zh.md)分别追踪真实代码、独立反例、保存重放和未完成项。集成前仔细审阅相关模块逻辑；局部通过不替代完整调用链。完整结果、来源与理论见[证据索引](evidence/README.md)。

## 历史大版本

[域迁移与BrainSync稳定大版](evidence/domain_transfer_major_20260920.zh.md)保留09-18最后完整公共训练的执行来源，以及实际对应09-20发布EXE的工作站源码、默认源和恢复接口；二者来源分别说明，不混作同一个部署模型或成人效能。模型、原EEG及最终结果保持原位置。

[大版恢复清单](../frozen/major_domain_transfer_20260920/README.zh.md)保存必要代码、依赖、模型身份与最终验收；不长期保每次修复、失败、复审和重复冻结。结果落地后清理无用过程材料，当前问题只保仍必要的骨干和证据。

## 维护与分发

状态与使用方式更新原六指南，独立科学结论进入证据索引。源码、当前指导和紧凑manifest进Git；原数据、模型、最终结果及大版载荷本地保存，见[版本与分发规则](../frozen/README.md)。一次性工具完成用途后删除，历史指导不作为当前执行入口。
