# 工作树与 BCI 目录整理

日期：2026-09-08。范围为本机文件整理，完成记录见
[校验清单](../../frozen/workspace_cleanup_20260908.manifest.json)。

## 实际去向

| 材料 | 文件数 | 原始字节数 | 保留位置 |
|---|---:|---:|---|
| 历史 tmp 脚本、展开快照、发布副本、测试产物和资料 | 4,751 | 1,456,977,918 | `frozen/local_history/workspace_cleanup_20260908/tmp_history.zip` |
| 根目录 Python、pytest、Ruff 工具缓存 | 65 | 213,610 | 同目录 `tool_caches.zip` |
| BCI 根目录旧部署包、Git bundle、配套文件列表 | 16 | 366,835,164 | `D:/BCI/n2p3-audit-archive/transfers_20260831_20260901/` |

合计 4,832 个文件逐项保全。两个 ZIP 共 1,407,382,505 字节，
重点是目录归整和来源可追溯；大量 PDF、已有压缩包的再次压缩收益有限。
BCI 根目录现保留六个项目/归档目录，并新增本机 `D:/BCI/README.md` 导航。

## 保留的活动内容

本次盘点时 `tmp/brainsync_s1_*20260908.py` 正在更新。因此保留所有当天修改的
顶层文件或目录，以及 `cloud_ssh/`、`data_export/`、`ssh_p001.py` 支持入口。
完成时 `tmp/` 留有 14 个顶层入口；后续任务可继续写入，不能把该数字作为长期不变量。

三个 Git 工作树全部保留原路径。辅助树 `n2p3-net-perf` 和 `n2p3-net-result`
检查时均无未提交修改，但各有主分支之外的独立提交，因此没有合并、移动或删除。
主树已有大量未提交源码、测试及证据；没有 stage、commit、reset 或改写 Git 历史。
BrainSync、NRH-01、原始数据、正式 cache/results/runs、两个虚拟环境及已有冻结归档
保留原位，历史文件内嵌路径、数据、分数、哈希和凭据未改写。

## 查询和恢复

完整本地索引为 `frozen/local_history/workspace_cleanup_20260908/inventory.json`，
每项记录原绝对路径、保留位置、ZIP 成员、大小、修改时间、SHA-256 和完成状态。
归档成员保留相对于仓库的路径，例如 `tmp/training_chain_snapshot_20260907a/...`。
旧证据中的历史路径继续表示当时执行位置，按索引查找，不回写历史来源身份。

```powershell
$oldPath = 'D:\BCI\n2p3-net\tmp\automatic_resource_final_tests.xml'
Get-Content frozen/local_history/workspace_cleanup_20260908/inventory.json |
    ConvertFrom-Json | Where-Object original -EQ $oldPath |
    Select-Object original, destination, archive_member, sha256
```

需要恢复时先核对总清单中的压缩包 SHA-256，再在新的独立目录提取所需成员，
核对索引中的逐文件摘要。不要覆盖正在使用的源码或自动运行旧脚本。
传输包目录另有 `manifest.json`，记录原 BCI 根路径到当前文件的映射。
大归档、完整索引和一次性执行脚本均留在 Git 忽略目录，迁机时须单独携带。

## 验证

执行工具先盘点并哈希，生成压缩包后逐个读取成员核对 SHA-256 和长度；
移除原件前再次核对源哈希。旧传输文件复制到目标并核对两端摘要后才移除根目录原件。
路径检查拒绝越界和重解析点，只移除已保全的文件及空目录，新增并行文件不会被递归删除。

文件整理前后检查了 750 个受 Git 管理或非忽略的仓库文件，哈希全部一致；
Git index 字节未变。这一检查发生在新增本记录与维护索引之前。
项目 `.venv` 的 `ruff check --no-cache src experiments tests` 通过。
最终复核通过归档/索引摘要、16份迁移文件哈希、80个本地文档链接及 `git diff --check`；
除两份计划内索引/工程文档更新外，原有750份文件无意外变化。
并行工具重新生成的3份Ruff缓存保留；复核结果存于本地归档目录的 `verification.json`。
本次以成员校验、文件映射、源码保全及文档链接检查验收；没有运行训练或全量算法测试。
这些结果仅证明本地文件整理，不构成科学结论或异地备份。
