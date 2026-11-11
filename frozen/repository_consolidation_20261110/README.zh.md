# 2026-11-10 文件整合归档

本目录保存本次整理的原文、旧训练输入和历史研究记录。三个 ZIP 都保留成员的原始字节；逐成员 SHA-256 与实际清理结果见[清理证据](../../doc/assets/build_cleanup/repository_20261110/)。压缩包仅在本地保存，Git 跟踪说明和清单。

| 压缩包 | 内容 | 成员清单 |
|---|---|---|
| `documents.zip` | 130 份历史文档，以及整理前的活动文档、交接等原文，共 146 份 | [文档清单](../../doc/assets/build_cleanup/repository_20261110/documents.manifest.json) |
| `training_inputs.zip` | 三组旧 cache，NPZ、分片与 sidecar 成套保存，共 55 文件 | [输入清单](../../doc/assets/build_cleanup/repository_20261110/training_inputs.manifest.json) |
| `research_records.zip` | 早期结果、预测、producer、旧审计与工作树退休记录，共 2,753 文件 | [记录清单](../../doc/assets/build_cleanup/repository_20261110/research_records.manifest.json) |

历史文档从[检索页](../../doc/evidence/history.zh.md)查找。`documents.zip` 含旧活动指南，读取时先解压到新的空目录，再选取需要的成员，不能整包覆盖当前仓库。

旧训练输入已经封为冷存包。需要使用时，在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m zipfile -e frozen/repository_consolidation_20261110/training_inputs.zip .
```

解压前核整个 ZIP 的 SHA，解压后按清单核全部成员 SHA。恢复保留原相对路径；Huebner 合并输入须与其全部原分片和记录一起恢复。消费者仍须检查原输入合同、父数据和 producer，解压成功不代表科学准入。

历史结果同样按原相对路径恢复，建议先解压到新的空目录，仅取需要的成员。`tmp/audit/scratch/g8_inventory.json` 和 `g8_decl.json` 是现有 lineage 索引入口的必要输入，清理后仍保留原位。

本包没有补齐此前已缺失的 `experiments/runs/`，也不替换旧大版、模型、评分或证明的原接受身份。重要读数与恢复限制见[训练数据与结果](../../doc/training_data.zh.md)。
