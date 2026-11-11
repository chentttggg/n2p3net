# 历史文档检索

旧报告保留当时的设计、命令、结果和判断。现行规则从[六指南](../README.md#规范入口)读取，结论从[四主题](README.md#当前主题结论)读取，运行状态只查原checkpoint。

## 原文与成员目录

本次共收存130份历史Markdown。原文位于[文档压缩包](../../frozen/repository_consolidation_20261110/documents.zip)，逐文件的原路径、标题、主题、字节数和SHA-256见[机器索引](../assets/build_cleanup/repository_20261110/docs_history_index.json)。压缩包同时保存整理前的活动指南与交接文本；这些副本用于核对旧引用与验收身份。

查旧报告时，在机器索引搜索原文件名或关键词，再按`archive_member`读取压缩包成员，并核对`bytes`与`sha256`。原路径完整保留，不需要恢复整棵旧文档树。原有历史manifest继续指向自己的原归档，本次压缩包不替换那些身份。

恢复时，先把压缩包解到新的空目录，再按索引选取所需成员。包内包含整理前的活动指南，不能整包覆盖当前仓库；核验原文或旧验收身份时直接读取解压副本。

| 主题 | 历史页数 | 当前结论入口 |
|---|---:|---|
| [校准与选择](#校准与选择) | 11 | [校准主题](calibration_and_selection.zh.md) |
| [响应与噪声](#响应与噪声) | 24 | [响应主题](response_and_noise.zh.md) |
| [公共评估](#公共评估) | 32 | [公共主题](public_evaluation.zh.md) |
| [执行与集成](#执行与集成) | 49 | [执行主题](execution_and_integration.zh.md) |
| [形式化验证](#形式化验证) | 14 | [Lean说明](../../formal/lean/README.md) |

## 校准与选择

包括先验校准、LDA顺序、早停、完整程序选择、前缀剂量与多专家实现。按机器索引的`topic=calibration`筛选；历史实现不自动成为当前默认配方。

## 响应与噪声

包括滤波与输入审计、BrainSync质量、EOG、边界状态、ERP字典、先验和噪声推导。按`topic=response`筛选；原输入、拟合支持和数值验收范围随报告保留。

## 公共评估

包括跨库设计、候选协议、公共矩阵、统一训练结果和数据准入。按`topic=public`筛选；decision、模型、逻辑评分和物理拟合使用各自原分母。

## 执行与集成

包括全仓审计、代码实施、吞吐、云端交付、清理和旧进展报告。按`topic=execution`筛选；旧测试数与“通过”结论仍绑定原命令、日期和执行源码。

## 形式化验证

包括Lean试验、各数学切片、性能与来源审查。按`topic=formal`筛选；原证明、验收manifest和暂停checkpoint保留原位，整理文档不会恢复构建或续跑证明。

## 运行载荷缺口

本次整理前有17处活动链接指向两个已缺失的运行目录：`experiments/runs/brainsync_matrix_20261031/`和`experiments/runs/brainsync_eog_consistent_20261031/`。旧地址逐项保存在机器索引的`unavailable_runtime_references`中；文档导航已改为现存绑定或交接核查入口。

旧矩阵的[矩阵接受绑定](../assets/brainsync_matrix/acceptance.json)、[R1绑定](../assets/brainsync_matrix/single_round_acceptance.json)、[R5分解绑定](../assets/brainsync_matrix/block_reviewed_acceptance.json)以及[启动裁剪绑定](../assets/brainsync_quality/startup_cut_acceptance.json)、[幅值诊断绑定](../assets/brainsync_quality/post_cut_amplitude_acceptance.json)仍在仓库。绑定保存的SHA与原路径可用于恢复后核验；绑定本身不等于完整模型、报告或预测载荷。

其余阶段来源和交接时发现的缺口见[交接核查](../project_handover_20261107.zh.md#十一截至交接日的现场状态和材料缺口)及[原现场清单](../assets/handover_20261107/source_inventory.json)。没有取得的旧载荷继续记为未恢复，不用新的文档链接代替恢复验收。

早期探索另有[文字资料索引](early_exploration_20260826/README.zh.md)和原收录manifest，保留其原文SHA与阅读副本SHA的区别。
