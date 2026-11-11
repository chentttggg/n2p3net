# 构建清理长期维护

本文件集中维护本项目的待清理构建、阻断原因、替代品和后续核验，是活动清单的统一维护入口。位置固定在项目的 `doc/` 主目录，不放在细分 `experiments/runs/` 下。按用途合并更新，原始回执作为不可变历史证据集中保存在 `doc/assets/build_cleanup/`。

## 维护规则

- 每次交付先确认替代品可用，登记绝对目标、范围、依赖、证据和实际结果；只在对应工作发生时更新，不自动重试删除。
- 保留原始数据、源代码、冻结科研结果、producer快照、暂停checkpoint及恢复依赖。
- 删除前核对绝对路径、内容及重解析点；不得跟随符号链接或junction出界。
- 使用一个明确绝对目标执行一次CMD删除。普通执行错误按原因处理；策略或审批拒绝后立即停止，不能换工具、拆命令或等价绕过。
- 只有重新确认目标不存在、替代品与所需证据仍完好，才能结项。文件逻辑字节不等同于实际释放空间。

## 项目文件整合（2026-11-10，已结项）

按用户要求整理当前仓库，保留源码、整合文档、重要训练输入与结果，以及运行和恢复所需文件。humanizer 用于改写文档入口；六指南、四主题和原检查点继续各自维护规则、证据与运行状态。

- 130 份历史文档退出活动目录，连同整理前活动文档共 146 份原文压入 `documents.zip`。历史引用统一从[检索页](evidence/history.zh.md)定位；17 处指向原已缺失运行目录的链接改为现存绑定，并明确载荷仍未恢复。
- 三组旧训练 cache 的 55 文件由 4,070,366,421 字节无损压至 2,583,081,719 字节；NPZ、分片与 sidecar 原字节成套保留。历史结果和审计的 2,753 文件由 245,211,088 字节压至 35,672,738 字节。全部 2,954 个归档成员重新读出并核 SHA-256，通过后才删除散件。
- 删除范围、原文件清单和哈希见[删除前清单](assets/build_cleanup/repository_20261110/targets_before.json)。168 次 CMD 调用各处理一个明确绝对目标，全部退出 0；包括已归档散件、Python 缓存、临时依赖副本、Lean 本工程及依赖的编译缓存，以及最后的空 cache 父目录。没有策略阻断、审批拒绝或剩余目标。
- `tmp/audit` 中两份现有 lineage 索引所需 JSON 已按归档 SHA 恢复原位，其余旧审计文件冷存。原始 EEG、当前 basic 权重、cloud 输入、原冻结包、Lean 证明与验收记录、暂停检查点、两个用途不同的 Python 环境和文献保留。
- 清理后 1,310 个受保护源码、配置、模型和检查点文件 SHA 不变；另一个受保护对象 `formal/lean/README.md` 只改导航，其原字节在文档归档中。74,536 个原始/保留资产的文件大小和修改时间与盘点一致；该项不是对所有原始数据重算 SHA。
- 验证时仓库逻辑大小由约 77.54 GB 降至 73.28 GB，净减约 4.26 GB；未测量实际释放的物理空间。环境与旧 basic 入口相关测试在清理前后各 16 项通过；Lean `N2P3Pilot` 在缓存退役前构建通过，证明源码及验收证据保留，之后需要按固定依赖重新取得编译缓存。21 个文档入口的链接核验通过。

原始执行日志、逐命令结果和核验记录见[本轮回执](assets/build_cleanup/repository_20261110/receipt.json)。数据与结果如何恢复见[训练数据说明](training_data.zh.md)及[整合归档说明](../frozen/repository_consolidation_20261110/README.zh.md)。本次没有重训、重算预测或恢复暂停研究。

## 旧探索临时材料退役（2026-11-07，已结项）

用户先授权删除无用临时文件，随后明确这些都是很早的材料，保留纯代码冻结即可。因此最终范围改为：先冻结旧工作区及历史补丁包中的纯代码，再删除 `D:/Temp` 中本次盘点的全部旧项目材料；该目录根保留。此授权只作用于本批旧材料，当前后端、原始数据和既有科研产物不在删除范围。

- [纯代码冻结](../frozen/early_source_retirement_20261107/README.zh.md)保存1,643个源码、测试、启动器、依赖与配置成员，含未提交源码和各历史传输包版本；压缩包6,308,680字节，全部成员删除前、删除后均重新读取核SHA-256。
- 明确删除的目录为 `D:/Temp/n2p3-net`、`D:/Temp/n2p3-archive-check`、`D:/Temp/n2p3net-patch-staging-20260827`、`D:/Temp/Paper` 和 `D:/Temp/tmp`；另有20个根目录旧文件，逐项绝对路径、内容计量和范围见[最终删除前清单](assets/build_cleanup/temp_retirement_20261107/before_expanded.json)。所有目标及其祖先均已核验，无重解析点。
- 每次CMD仅处理一个明确、带引号的绝对目标，共25次，全部退出0。实际删除31,725文件、3,426,238,928逻辑字节；未测量物理释放空间。删除后25个目标全部不存在，`D:/Temp`为空目录。
- 交接报告、已收录文字参考、原文字归档和新源码冻结等20个保留文件逐字节不变。原保守盘点中的“保留旧缓存/结果”方案由用户后续明确指令替代；初始记录保留，实际范围以 `before_expanded.json` 为准。
- CIM进程查询受限后采用非提升psutil接口检查，未发现以目标内程序或目标目录运行的进程。清理未启动训练、采集或暂停中的研究任务；本批没有策略或审批拒绝，没有剩余目标或阻断。

实际状态为 `completed`。[执行记录](assets/build_cleanup/temp_retirement_20261107/operations.json)与[最终核验回执](assets/build_cleanup/temp_retirement_20261107/receipt.json)保存实际命令、退出结果、源码冻结和保护对象验证。

## 旧构建清理事项（BCI-001至005，已结项）

最近执行与核验：2026-11-03T00:26:59-07:00（America/Denver）。用户明确授权重试 BCI-001 至 BCI-005；五项目标均已删除并确认不存在，状态为 `completed`。下表内容为删除前盘点，完整清单及执行结果见[本次重试回执](assets/build_cleanup/retry_20261103/receipt.json)。

| 编号 | 项目 | 绝对目标 | 实际状态 | 删除前内容 |
|---|---|---|---|---|
| BCI-001 | n2p3-net / BrainSync质量处理 | `D:/BCI/n2p3-net/build` | completed；CMD退出码0，目标不存在 | 285 文件，6,584,163 逻辑字节，0 重解析点 |
| BCI-002 | n2p3-net / BrainSync质量处理 | `D:/BCI/n2p3-net/experiments/runs/brainsync_quality_20261030/implementation/final` | completed；CMD退出码0，目标不存在 | 3 文件，57,936 逻辑字节，0 重解析点 |
| BCI-003 | BrainSync旧basic入口发布 | `D:/BCI/artifacts/apps/BrainSync_Release_20260920/_internal` | completed；CMD退出码0，目标不存在 | 486 文件，308,676,124 逻辑字节，0 重解析点 |
| BCI-004 | 同批旧单文件发布副本 | `D:/BCI/artifacts/apps/BrainSync_Release_20260920/BrainSync_Standalone.exe` | completed；CMD退出码0，目标不存在 | 1 文件，120,236,664 逻辑字节，0 重解析点 |
| BCI-005 | 本次打包中间产物 | `D:/BCI/tmp/brainsync_basic_entry_20261102` | completed；CMD退出码0，目标不存在 | 1,136 文件，717,280,797 逻辑字节，0 重解析点 |

### 2026-11-03重试核验

- 删除前重新核对物理绝对路径、所有祖先及目标内容；五项目标均无重解析点，文件数与大小均与原清单一致。以非提升的CIM查询核对实际进程，未发现使用目标的应用或打包进程；当前暂停checkpoint及质量交付入口未引用这些构建作为恢复依赖。
- 使用CMD，每次仅删除一个明确、带引号的绝对目标。BCI-001首次直接指定CMD shell时，引号传递产生路径语法错误（退出码123），目标仍存在；通过只读目录查询复现并确认可用调用方式后，以显式`cmd.exe /d /c`重试成功。其余四项首次执行即成功。本次没有策略或审批拒绝。
- [删除前清单](assets/build_cleanup/retry_20261103/before.json)与[删除后核验](assets/build_cleanup/retry_20261103/after.json)保存逐文件SHA256。删除合计1,911文件、1,152,835,684逻辑字节（约1.07 GiB）；未测量实际释放空间。
- 11组保留对象共799文件逐字节不变，包括已接受质量配置、七份缓存及producer、初次失败证据、wheel及交付入口、editable元数据、暂停checkpoint、basic交付证据、新发布全目录和旧发布其余文件。原始EDF/BIDS/session/clock及冻结科研结果不在删除范围。
- 配置与wheel符合原接受SHA256，wheel ZIP完整；新发布可执行文件符合原打包SHA256，发布清单的555个文件全部通过大小与SHA256核验。新发布共557文件，与删除前完全一致；复用既有启动验收，本次未重新启动GUI或科研任务。
- 暂存发布与正式发布仅有已核对的解码数据目录/设备设置及正式发布新增手册、说明、清单的差异，见[差异清单](assets/build_cleanup/retry_20261103/stage_comparison.json)；[暂存配置](assets/build_cleanup/retry_20261103/staged_decoding.json)已保存。打包spec和日志原已在交付证据中保留。
- 核验脚本初次按Windows默认cp1252解码UTF-8清单，导致中文手册文件名比较误报；明确UTF-8后完整复核通过。旧发布的junction别名及其余12文件保持，原两份清理回执和交付接受记录作为历史证据未改写；当前清理状态以本页为准。

## BrainSync：旧 basic 日常入口（2026-11-02）

本批BCI-003至005已结项，见上表及本次重试核验。独立新发布为
`D:/BCI/artifacts/apps/BrainSync_Release_20261102/BrainSync.exe`。
新发布完成固定源身份、真实历史C只读预检、实际前端任务/回执→90轮合成C→保存重载→无标签R5，
以及打包代码与两次模拟启动核验；证据见[本次接受记录](assets/brainsync_basic_entry_20261102/acceptance.json)。

原交付时已核实际物理目录及无重解析点。`D:/BCI/BrainSync_Release_20260920`是指向上述旧发布根的既有junction；
删除仅请求物理根内明确的`_internal`，并未对junction或用户目录操作。
CMD调用在进程启动前被自动策略拒绝，原文为`rejected: blocked by policy`，没有shell退出码。
拒绝后停止清理，未拆分、换工具或用覆盖替代删除。BCI-004/005均未尝试；新发布复制到全新目录，
未改旧发布任何文件，原数据、模型及配置保持。原始回执见
[本批清理回执](assets/build_cleanup/brainsync_basic_entry_20261102.receipt.json)。
上述为原交付批次历史，当时逻辑删除量为0；本次授权重试的实际结果见上表。

<a id="n2p3-net-brainsync-quality"></a>
## n2p3-net：BrainSync质量处理

项目：`D:/BCI/n2p3-net`。任务的数据处理已完成，BCI-001及002构建清理现已结项。

2026-11-02设备启动裁剪的清理状态为`not_applicable`：本次产生的是
[未预处理数据派生与核验](../mne_data/BrainSync_startup_cut_4s/manifest.json)，没有替换应用构建。
原始EDF、裁剪版EDF、旧冻结结果及执行来源保留；下列既有构建清理阻断不因数据裁剪恢复执行。

### 目标与处理依据

- **BCI-001** 是已验证wheel产生的中间 `build/`。此前CMD删除请求在进程启动前被自动审核拒绝，返回 `blocked by policy`，没有shell退出码；本次用户明确授权后重试完成。
- **BCI-002** 是初次profile构建的重复目录。其profile与已接受替代品逐字节一致；初次失败回执和producer已保存在项目的 `implementation/attempts/`。原交付因同批次策略拒绝而未尝试，本次已删除并核验保留证据。

### 替代品与保留证据

- [本批数据处理交付](/D:/BCI/n2p3-net/experiments/runs/brainsync_quality_20261030/implementation/index.html)。
- [冻结配置](/D:/BCI/n2p3-net/experiments/runs/brainsync_quality_20261030/implementation/accepted/continuous_reviewed_native7.json)，SHA256：`c52a6acaf0f18b090960056a38fd54f81b4c438876bab00192b1668535cba665`。
- [已验证wheel](/D:/BCI/n2p3-net/experiments/runs/brainsync_quality_20261030/implementation/distribution/n2p3_net-0.1.0-py3-none-any.whl)，SHA256：`17beceb5e64412575bd6c3d1381ce7aae9d83bff3481b38f0451631b7e6a84ac`。
- 七份缓存、原始EDF/BIDS/session/clock、旧科研结果、`accepted/producer/`与初次失败证据均保留。已有 `src/n2p3_net.egg-info` 是当前editable环境依赖，不在本次删除范围。
- [原始清理回执](assets/build_cleanup/brainsync_quality_20261030.receipt.json)，SHA256：`faeb4cd2151835cec5ecbc018be4e97c54ee58c8f084ddfeb115eb5a88cf5844`。该文件从细分runs移至项目级证据目录，原字节保持；它记录此前尝试，不代替本页活动状态。

### 后续维护

本批已完成；后续新清理事项仍在本页登记并按维护规则核查。本次删除不构成恢复暂停科研工作的授权。

### 维护记录

- 2026-10-30T19:34:25+08:00：按用户澄清，将本批清单合并到项目doc主目录的长期文档；细分runs只保留引用，原始回执移至doc/assets/build_cleanup。策略阻断和未尝试状态保持。
- 2026-11-03T00:26:59-07:00（America/Denver）：按用户明确请求重试BCI-001至005，五项均已删除并通过保留证据核验；本页更新为completed，原始回执保持。
