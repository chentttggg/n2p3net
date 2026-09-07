# 工作树临时材料清理与数据保留

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。用户要求落实目录清理，同时保留数据和结果。本次只整理本地历史
`tmp/`，不运行历史脚本，不改训练算法，不重新解释历史分数，也不提交或回退工作树。
完成状态及校验值以[最终清单](../../frozen/tmp_cleanup_20260906.manifest.json)为准；
[清理前盘点](../assets/workspace_cleanup_20260906/inspection.json)保留原始盘点状态。

## 保留位置

原目录共16,389个文件，240个顶层文件，合计3,025,297,843字节。分层如下：

| 内容 | 文件数 | 原始字节数 | 现在的位置 |
|---|---:|---:|---|
| 四份派生数据与六份配套记录 | 10 | 1,455,372,193 | `experiments/cache/history/tmp_20260906/` |
| 研究结果、日志和来源记录 | 368 | 192,882,571 | `experiments/results_history/tmp_20260906/retained/` |
| 旧脚本、方案、文献、传输包及测试历史 | 16,011 | 1,377,043,079 | `frozen/local_history/tmp_20260906.zip` |

全文件迁移索引为
`experiments/results_history/tmp_20260906/inventory.jsonl`。每行记载原路径、分类、目标路径、
压缩包成员、大小、修改时间、原SHA-256及保留后SHA-256。压缩成员路径相对于原`tmp/`，
例如旧`tmp/l1_cloud/montage_ablation.py`对应包内`l1_cloud/montage_ablation.py`。
原`tmp/`清空后继续作为临时目录；今后的新临时文件不属于本次归档。

`mne_data/`、原有`experiments/runs/`、`cloud_inputs/`以及`tmp/`之外的数据、checkpoint和
研究结果均不在本次迁移范围。此前独立完成的去重及源码归档仍按各自manifest追溯。

数据、结果和大归档保留在本机的Git忽略目录，**Git提交或克隆不会携带这些文件**。
迁移机器时须同时复制这些目录、索引与manifest，并核对SHA-256；本地归档不等于异地备份。

## 四份数据是什么

这些都是BI2014a历史派生cache，不能仅凭文件名直接认定符合当前训练准入合同。
本次以NPZ/NPY结构化头及小体积元数据核对，不加载pickle，不改写内容。

| 文件 | X形状（epoch, channel, sample） | 研究用途与限制 |
|---|---|---|
| `l0_bi2014a_causal_v3_64subj_car16.npz` | `(61011, 16, 179)` | 历史CAR16参考对照，可用于追溯参考变换的影响 |
| `l0_bi2014a_causal_v3_64subj_car16_embedded_v1.npz` | `(61011, 16, 179)` | 带嵌入式数据摘要的变体；形状相同不代表文件身份相同，分别保留 |
| `l0_cache_bi2014a_v3_s12.npz` | `(10884, 16, 179)` | 12人构建与蒙太奇诊断，可追溯早期解析及空间消融 |
| `l0_cache_probe_bi2014a_v3_s2.npz` | `(1680, 16, 179)` | 两人小规模构建探针，支持定位构建问题，不是完整队列性能 |

共同元数据为128 Hz、伏特、0.1–30 Hz前向IIR、mean-only baseline、
`[-200,1200)` ms、179个采样点；完整处理声明见盘点中的`preprocessing_json.npy`。
通道为`FP1,FP2,F3,AFZ,F4,T7,CZ,T8,P7,P3,PZ,P4,P8,O1,OZ,O2`，
并非BrainSync七EEG加EOG产品蒙太奇。使用前仍须验证attestation、参考、身份、
时间和划分合同；这次文件保全不构成数据准入或实时因果性的重新认证。

两份CAR16各保留`.record.json`，s12与s2各保留`.record.json`和
`.candidate_audit.json`，共六份配套记录。

## 结果如何解释与查找

直接保留`analysis_dt_v2_dev64.json`、`analysis_relaxed_native.json`、
`analysis_relaxed_car16.json`、`new_run_analysis_summary.json`等原始分析输出。
它们是历史开发读数，继续按原队列、原分母、原配方选择方式解释，不能提升为独立确认。

保留的运行目录包括`l0_dryrun`、`l1_cloud`、`e1_car_e32`、`e1_reference_pair_20260904`、
`perf_audit`、`n2p3-training-295942e-formal`、`generated_manifests_final`、
`generated_manifests_stable`、`generated_partial`及`remote_manifests`。
其中的结果与记录保持嵌套布局，脚本和补丁移入压缩包。
`n2p3-training-295942e-formal`实际包含105份结果/subject-scope文件，不能按名称误删为旧源码。

`prof_step_ms256.json`含多个拼接输出，不是单个合法JSON文档，保留原字节；
分析时应先识别其实际输出格式，不能为通过JSON解析而覆盖原文件。
合成pytest历史亦完整压缩保存，其测试数据和模型不构成真人准确率证据。

旧报告中的`tmp/...`路径是执行时的来源。查询某个旧路径可使用：

```powershell
$oldPath = 'tmp/analysis_relaxed_native.json'
Get-Content experiments/results_history/tmp_20260906/inventory.jsonl |
    ConvertFrom-Json | Where-Object original -EQ $oldPath |
    Select-Object original, storage, destination, archive_member, preserved_sha256
```

历史结果JSON中的路径、哈希、分数不回写。旧脚本只作审计证据；如需复现实验，
应在独立目录中核对源快照、依赖及数据合同，不将脚本重新解压成活动CLI。

## 保全和验证

执行顺序为：完整盘点并哈希，复查原文件，复制数据/结果并压缩其余文件，
逐目标及压缩成员校验，写入删除前manifest，再次复查原目录成员和哈希，
逐文件删除已保全的原件，最后核验目标内容与空`tmp/`。
若源文件变化、出现新文件或校验失败，清理程序在删除阶段前停止。

仅两份旧连接脚本`cloud_ssh.py`与`cloud_upload.py`的明文密码赋值在归档时脱敏，
索引分别记录原哈希和脱敏后哈希；不影响数据、结果或实验数字。其余所有文件
原始与保留哈希一致。脱敏归档不能用于恢复原密码。

一次性执行工具、完整索引和盘点另封装在
`frozen/local_history/tmp_20260906_cleanup_evidence.zip`，校验信息见
[执行证据清单](../../frozen/tmp_cleanup_20260906.evidence.json)。工具原临时源码在封装核验后移除。
本次验证以逐文件保全为主，辅以当前Markdown本地链接和`git diff --check`；
没有通过重跑训练或全量旧测试来证明文件迁移正确。
