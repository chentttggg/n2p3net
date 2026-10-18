# 执行计划与依赖

科研主目标及比较规则由[研究规范](research_program.zh.md)维护。本页定义B00–B14范围和依赖；当前进度、在途任务及下一执行动作从[checkpoint入口](README.md#当前状态与续接)读取。结果与接受来源分别查[主题证据](evidence/README.md#当前主题结论)及[整合绑定](assets/research_resume_20261012/research_integration_acceptance.json)。

## 研究块

| 块 | 工作范围与验收出口 |
|---|---|
| B00 比较与选择 | 完整请求风险、配对、覆盖、动作与概率指标 |
| B01 信号与支持 | 原始输入身份、时钟、事件及合法读取支持 |
| B02 专家接口 | 一致的fit/score、冻结状态和完整决策输出 |
| B03 校准与混合 | C内完整学习程序、因果nested选择与独立future评价 |
| B04 神经归因 | 匹配source/scratch、结构/配方与可迁移收益 |
| B05 AR与噪声 | 同物理单位/支持的噪声law、估计与数值证书 |
| B06 边界条件 | 状态、观测算子及可识别条件的推导与实际调用 |
| B07 响应与历史 | 通用J/membership、响应形状与完整历史推断 |
| B08 先验 | 原坐标、弱方向、收缩/容量及同条件对照 |
| B09 空间 | 实测通道/参考、共同效应及跨布局输运 |
| B10 输入污染 | 污染机制、可用时间、鲁棒性和实际影响 |
| B11 动态与稳健 | 共同/逐record尺度、潜变量及可计算推断 |
| B12 公共流程 | 完整source→C→future矩阵、保存重放与实际输入准入 |
| B13 成人确认 | 冻结个人程序、新记录及最终九选一评价 |
| B14 工程与吞吐 | 共享资源/缓存、生命周期、恢复及端到端吞吐 |

## 依赖与交付顺序

1. B01/B06/B09确定身份、观测与可用支持，模型和评价消费同一合同。
2. B02连接各机制的真实source、C状态、评分、保存加载与重放。
3. B00/B03在合法C内学习完整程序，B12在独立future上按完整请求比较。
4. B07/B11优先推进可迁移到九选一的通用数学；B12基础流程按实际依赖独立推进。
5. B13消费冻结后的个人程序及新成人记录；B14贯穿各阶段的资源和恢复。

每个切片交付数学/范围、区分性反例、完整调用链、实际产物与来源。构建交付同时完成[旧构建清理](engineering.zh.md#交付与旧构建清理)。

## 响应机制比较与已有程序复用

B07/B11的首项机制比较采用G0共享完整响应与G1 selection独立零均值functional变化，二者使用各自具名联合C/noise marginal定标。原非线性delay/width研究继续保nominal、mean、full-moment Gaussian、exact W四臂，G1另列为functional比较。两类law共用原权限、完整候选/历史、同record观测并集、一次startup/drift及whole/conditional同拟合评分。研究选择和数学反例见[响应与噪声](evidence/response_and_noise.zh.md#共享函数与字符变化的机制比较)。

B04/B12优先复用已有source权重及成熟target微调/早停/保存重放，保其原输入和训练来源；新的同源同输入LOSO作为具名受控实验。可独立执行的完整评价无需等待新的source重新训练，来源剂量与配对scratch各自清楚记录。

原123项清单与设计检索见[分块方案](evidence/deep_research_plan_20261005.zh.md)和[愿望清单](evidence/wishlist_progress_20261010.zh.md)；整理前的路线图由[原文清单](../frozen/guidance_consolidation.manifest.json)保留。
