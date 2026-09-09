# 云端数据盘清理与保全

2026-09-08，按用户要求清理云端数据盘。没有启动训练、修改模型、重写数据合同或历史结果。
执行清单见[完整摘要](../../frozen/cloud_cleanup_20260908/manifest.json)。

## 空间变化

| 数据盘 `/root/autodl-tmp` | 清理前 | 清理后 |
|---|---:|---:|
| 已用 | 48.623 GiB | 31.780 GiB |
| 可用 | 1.377 GiB | 18.220 GiB |
| `df` 占用率 | 98% | 64% |

实际释放18,085,195,776字节，即16.843 GiB。该数来自文件系统可用空间差，
不是将文件逻辑大小简单相加。需要追溯的历史材料压缩后共11.342 GiB，保存在另一文件系统
`/root/autodl-fs/n2p3-cleanup-20260908/`；因此数据盘释放量不等于云端全部存储的净减少量。

## 清理范围

以下62,256个历史文件先压缩、逐成员核验SHA-256或符号链接目标，再逐文件删除原件，
最后仅移除已经为空的目录。完整原路径、大小、时间、哈希和归档成员均在清单中。

| 类别 | 原始大小 GiB | 文件数 | 归档文件 |
|---|---:|---:|---|
| 旧BI64/GTN epoch缓存及QC记录 | 4.989 | 45 | `legacy_epoch_caches.tar.gz` |
| 旧R8派生缓存，保留raw snapshots | 4.405 | 41 | `legacy_r8_derived_caches.tar.gz` |
| 早期dev64、domain-transfer、BI64和CUDA探针结果 | 4.743 | 14,512 | `legacy_development_runs.tar.gz` |
| 旧commit部署目录及历史training目录 | 1.242 | 41,338 | `old_deployments.tar.gz` |
| 旧Git传输bundle | 0.743 | 10 | `old_transport_bundles.tar.gz` |
| pytest及smoke临时输出 | 0.045 | 6,310 | `test_temporary_outputs.tar.gz` |

另删除`/root/autodl-tmp/uv-cache`的15,693个可重建安装下载缓存文件，共534,085,675字节。
执行前未发现uv/pip安装进程，现有环境也没有指向该目录的符号链接。训练输入缓存和
编译缓存不属于这项删除。旧算法源码和实验结果保留原字节及原身份，没有重新标记为新结果。

## 当前材料保留

保留`n2p3-cloud-inputs/`、`four_dataset_data_20260907/`、原始MNE数据、R8 raw snapshots、
四库a/b/c输入与结果、compact512、full512及metadata续跑依赖，以及`/root/n2p3-runs/`
中的完整训练链、早停与90轮实验、源模型和执行源码。主部署与现有运行环境也保持原位。

特别保留`four_study_development_20260907a/`约9.6 GiB的输入：该次启动虽然失败，
后续b/c及完整实验仍复用这些缓存，不能按失败目录名直接删除。

对最新实验目录中的JSON、Python和shell文件扫描候选删除路径，未发现引用冲突。
在清理过程中为保留目录建立169,243个文件的大小、mtime、inode及device清单，结束时全部一致；
其中106,801个NPZ、PT、Python源码和压缩快照另做SHA-256前后复核，全部通过。
三份最新训练链预测包、早停全量包和90轮结果包的摘要另与本地冻结清单一致。
现有`/root/miniconda3/bin/python`可导入Torch2.8.0+cu128及NumPy2.3.2，CUDA可用；
操作前后无训练进程。这些是文件保全和环境检查，不是新的训练或科学验证。

## 历史路径与恢复

云端每个归档旁保存同名`*.manifest.json`，其中`files[].path`记录原绝对路径。
归档成员路径相对于`/root/autodl-tmp`。历史结果中内嵌的旧路径不改写；需审计旧材料时，
先查清单并核对归档整体SHA-256，再解压到独立审计目录，不直接覆盖当前活动目录。

全部逐文件记录已回传到本地Git忽略目录
`frozen/local_history/cloud_cleanup_20260908.reports.tar.gz`，与云端包SHA-256一致。
一次性执行、复核和回传整理脚本存于同目录的`cloud_cleanup_20260908.tools.zip`，
压缩成员与执行工具原文件逐字节一致后移除临时脚本。大历史归档仍在云端文件存储，
没有声称已下载到本机；Git仅分发摘要清单，遵循[归档说明](../../frozen/README.md)。

相关工具Ruff通过，文档链接与`git diff --check`通过。未重跑训练或完整算法测试套件。
