# BI2014a 作者拒收被试：依据与删除记录

日期：2026-09-06。用户要求先核查作者是否已标记问题，确认后删除 BI62-64 并在以后实验中禁用。

## 作者原文

从 [Zenodo 官方数据页](https://zenodo.org/records/3266223) 提供的链接实时核查
[HAL 作者报告](https://hal.science/hal-02171575/file/bi2014a_report_publication.pdf)。
完整 PDF 为 811,706 字节，11 页，已保存为
`Paper/BI2014a_Author_Report_HAL_02171575.pdf`；SHA-256：
`984c4fcca62f47efe65638579c3ae4e424a8c56c4aa0491c4641c11568e40e0a`。

PDF 第 4 页 Participants 段落明确列出：

- “Participant 62 and 64: vision not perfectly corrected.”
- “Participant 63: dry electrodes had poor contact with the scalp due to thick hair.”
- 三人的数据虽公开提供，但 “are very noisy and should not be used for P300 data analysis.”

第 9 页 Table 1 也将 62、63、64 标记为 Rejected。65-71 因 tagging 问题或实验中止，
其数据未提供；这不是本次新增删除对象。36/46/52/57 的 Inconsistant ERP 注记与 Rejected
不同，本次不据此扩大排除范围。62/64 的具体原因是视力矫正不全，不能笼统改称电极接触故障。

作者示例代码和 MOABB 加载列表允许 1-64，并不推翻报告的分析排除说明。
本次排除由作者预先记录的质量/实验条件和用户明确要求决定，不按本轮模型分数筛人。

## 已完成的删除与缓存迁移

- 本地 raw：删除 62-64 各一份 CSV、MAT、ZIP，共 9 个文件及三个解压目录。
  删除前记录完整路径、字节数和 SHA-256，原始数据未另行备份到活动目录。
- 本地 7 份含拒收被试的现代 cache 已生成保留行逐点不变的新版本，再删除旧 NPZ。
- 云端 `n2p3-cloud-inputs` 的 13 份相应 cache 同样完成迁移并删除旧 NPZ；包含混合库命名空间。
- 删除本地 6 份含拒收被试的 pre-v3 旧 cache，以及本地 4 / 云端 5 份依赖旧 cache 或
  checkpoint 的过时冻结特征、标签、group 派生文件。旧 sidecar 和结果保留原哈希与历史身份。
- 公开主 cache 从 61,011 行减至 57,437 行，删除 3,574 行；01-61 的 EEG、标签、
  行顺序逐点相同，timeline/QC/mask/identity 同步切片，写入新 lineage 和 attestation。
- 活动原始目录、`cloud_inputs`、`experiments/cache` 为本次数据清理边界；冻结历史证据压缩包
  和历史预测没有被改写，不能作为新实验输入。旧 source 权重保留作结果溯源，当前加载门禁拒绝复用。

## 当前可用输入

| 位置 | 文件 | SHA-256 |
|---|---|---|
| 本地16导 | `cloud_inputs/l0_bi2014a_causal_v3_61subj_eligible61.npz` | `e50d11764a85cc77ae4c6d49c8617f49686cd4af46c3a6f3a34fefa9534e4b60` |
| 云端五导 | `/root/autodl-tmp/n2p3-cloud-inputs/bi2014a_binary_common5_20260906_eligible61.npz` | `78c3f7ab77381cf8dc96c5b0a8fdc99a1b75dbf470d0f7a1cea49be2cf5a6e0e` |
| 云端BNCI五导，未改变 | `/root/autodl-tmp/n2p3-cloud-inputs/bnci008_binary_common5_20260906.npz` | `2efd14b447c8cc426d350c4030c4774a4aecf3be9187858294179997f9f65297` |

迁移不恢复原 parser 之前丢弃的边界/尾部 trial，不修改滤波、窗口、通道或 normalization 配方。
本地/云端其他同名派生 cache 的字节哈希可能因 provenance 路径不同而不同，不能互相替代哈希。
带学习型预处理的历史 EA 视图不自动成为新活动配方；使用前必须重新核验其 fit 范围。

## 后续准入与 checkpoint

`data.exclusions` 是唯一排除规则。按原始 source identity 匹配 BI2014a / BrainInvaders2014a
及 BI candidate 来源，不把其他数据集的本地 62-64 误排除；重命名本地 namespace 不能绕过。

- BI candidate CLI 默认只选01-61，显式请求拒收被试报错，直接 raw builder 也拒绝。
- 通用 MOABB 准备入口默认过滤；显式请求62-64报错。
- EpochDataset 验证、full/attested cache 加载均受限。
- transfer checkpoint 的源训练 identity ledger 受限；含62-64的旧源模型不能继续微调。
- 个体模型重载也核对保存的源训练参与者。

不能只给旧 checkpoint 改名称或删除其身份账本，必须重新估计源统计并训练权重。
新的 BI61 源训练与后续 LDA 对照由协调任务负责，本次没有宣称删除后 AUC 已改善。

## 验证与账本

`tests/test_participant_exclusions.py` 覆盖原始 builder、改名身份、相邻61人及其他数据集反例、
full/attested 旧缓存和真实 TrainingRunContract 构造的旧 checkpoint。
与 epochs/candidate/BI/calibrated 相关测试共118项通过；Ruff 检查通过。
预训练、多域源准备与 MOABB 入口相关测试另通过，1 项跳过。
真实本地新61人 cache 经当前门禁重载通过；真实旧 BI checkpoint 被拒绝，原 BNCI checkpoint
正常通过。本地扫描9份现代活动/历史cache均不含排除身份。

逐文件账本位于 `experiments/runs/bi61_retirement_20260906/`：
`raw_deletion.json`、`local_cache_retirement.json`、`cloud_cache_retirement.json`、
`local_legacy_retirement.json`、`cloud_legacy_retirement.json`。
缓存迁移程序只有在新 NPZ 重载并逐点对比通过后才删除旧 NPZ。
