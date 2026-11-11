# 早期探索参考资料（截至 2026-08-26）

本目录收录负责人交接报告所需的早期文字证据：12 份说明、方案和工程复盘，以及 8 份小运行的文字摘要。全部参考关系在当前项目中闭合。

本目录只保存历史参考，不设第二套活动规范。文中“当前”“正式”“默认”指早期版本；现行规则从[项目文档索引](../../README.md)读取。

## 建议阅读顺序

1. 工程复盘：理解单位、参考、过拟合、时间滤波和评价分母问题。
2. 路线与蓝图：理解 PCW、strict-past 和 L/R/Q/S 的原问题。
3. 评估、数据和配方：理解当时的实验边界。
4. 小运行汇编：区分效能、覆盖与性能 benchmark。

## 文字材料

| 编号 | 材料 |
|---|---|
| H01 | [工程演进与踩坑复盘](documents/PROJECT_ENGINEERING_JOURNEY.zh.md) |
| H02 | [早期文档总览](documents/README.md) |
| H03 | [正式路线与研究路线](documents/routes.md) |
| H04 | [原始任务与目标](documents/mission.md) |
| H05 | [v12 四对象架构](documents/blueprint.md) |
| A01 | [早期科学与工程原则](documents/constitution.md) |
| A02 | [早期评估协议](documents/evaluation_protocol.md) |
| A03 | [早期数据合同](documents/datasets.md) |
| A04 | [早期训练配方](documents/neural_ride_recipe.md) |
| A05 | [性能工程教训](documents/performance_lessons.md) |
| A06 | [成分窗与时间平移研究](documents/pcw_shift_equivariance_research.zh.md) |
| A07 | [早期研究路线图](documents/roadmap.md) |
| H06—H13 | [8 份小运行文字摘要](run_records.md) |

## 来源与整理方式

[manifest.json](manifest.json)逐份记录原文字节数、原 SHA-256、阅读副本摘要和导航改动。阅读副本保科学正文和数字，只增加历史提示并整理链接；旧源码、数据或图像的外链改为历史引用文字。

12 份 Markdown 原文另外保存在[文字原文归档](../../../frozen/early_exploration_reference_20260826/source_texts.zip)，成员字节经 SHA-256 核验。运行 JSON 仅摘录汇总字段，未复制原载荷。

收录内容均为文字说明或记录；未复制原始 EEG、训练/测试数据、缓存、模型权重、逐试次数组或设备程序。单文件及整组资料均低于 100 MB。归档为本地随交接移交的材料，其位置和摘要由清单给出。
