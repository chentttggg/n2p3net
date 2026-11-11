# 研究与工程证据

## 当前主题结论

| 主题 | 维护内容 |
|---|---|
| [校准与选择](calibration_and_selection.zh.md) | 校准剂量、温度、完整学习程序、选择行为与保存重放 |
| [响应与噪声](response_and_noise.zh.md) | 输入质量、响应与空间模型、噪声、数值关系和实测诊断 |
| [公共评估](public_evaluation.zh.md) | 公共数据的完整请求、配对比较、迁移读数和指标口径 |
| [执行与集成](execution_and_integration.zh.md) | 实际源码、执行账本、缓存共享、集成接受和恢复来源 |

四页维护结论及其适用范围，稳定规则回到[六指南](../README.md#规范入口)。当前进度只读[原checkpoint](../README.md#当前状态与续接)，来源与接受范围由原紧凑binding记录。

## 数据与结果入口

[训练数据与精简结果](../training_data.zh.md)汇总重要资产和关键读数。BrainSync的旧R1/R5矩阵见[校准主题](calibration_and_selection.zh.md#brainsync实机的冻结大版与校准剂量矩阵)，启动裁剪和少量校准下的EOG放大见[质量诊断](response_and_noise.zh.md#裁剪后的预处理幅值与eog放大复核)。这些读数各自绑定原始输入，新的处理链需要独立评价。

## 历史与来源

旧设计、审计、运行报告和早期结果统一从[历史索引](history.zh.md)检索。原文件压缩保存，机器索引保留每份原路径、标题、SHA和成员定位，不再维护重复的日期长目录。早期文字资料的收录范围见[原收录说明](early_exploration_20260826/README.zh.md)。

历史主题页恢复关系仍由[原manifest](../../frozen/research_topic_documents_history.manifest.json)维护；2026-11-10整理前原文见[本次成员目录](../assets/build_cleanup/repository_20261110/docs_history_index.json)。原审计、分数、producer和历史manifest保持各自身份，后续纠正写在当前主题页。
