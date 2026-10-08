# public source pipeline：独立 Standards 源码评审

本轮审查只读作者树 `C:/Users/chent/.codex/worktrees/calibration-g-prior/n2p3-net`，基线为 `3f38fe12e6a000d219967dba52faf0815cf6e2e6` 的 dirty WIP。没有修改原 `D:/BCI/n2p3-net` 或作者生产代码，没有启动云准备/训练，也没有运行 Torch/pytest。最终结论绑定包含新增 source_config 的 accepted capsule：archive `fd62c48b37de5a569ad2be629e61c508b931dc9b2e80bd2d3147cee2e25e94ea`、manifest `5d2ad8d5a23367994557e50ea6e29affc7f63631874ef451e007268a353dc1f3`。

**Standards 结论：本片源码审查通过，0 项剩余关键发现。** 五项具体缺口已关闭，新增 source_config owner / producer / consumer 关系成立。该结论是软件质量出口，不宣称公共全矩阵、云吞吐或成人效能已完成。

## 范围与方法

已读 `doc/README.md` 与六份活动指南相关段；已加载 astra-research-workflow 的 engineering / integration-map，以及 code-review 的 Standards 轴。评审先通读入口到消费者，再检查状态、缓存、资源与失败关系；没有按风格凑发现数量。较早方案不作为本片验收前提，历史 source identity 保留各自原义。

| 完整路径 | 当前阅读结论 |
|---|---|
| source program / preprocessing / source catalog | 新独立程序从已有 P2_source 进入；plain CE、六导、显式 finite-zero 输入、参与者 authority、整库排除与完整声明成员由既有 owner 核验。 |
| selection / fresh full refit / checkpoint | `source_selection` 保留按域分层的完整参与者 inner 划分；inner normalization/BN 只见 inner training，DeepBaseline 每次重新以同 seed 初始化；full refit 重新 fit 统计与全部合法源行；实际 update/draw/batch sequence 随 payload 保存。 |
| SharedTrajectoryRegistry / SourceChainResume | semantic signature 事务锁包住 fit / checkpoint / registry 发表；selected reuse 绑定 cache SHA 与完整 ledger，复用原 checkpoint bytes，不改为无 selection 的 consumer contract；resume 检查旧配置、输入、架构与冻结产物。 |
| prepare_source_records / prepare_multidomain_source / PreparedSourceInventory | raw/shard 方法与实际输入绑定，完整 cache/record 字节准入；声明 manifest SHA 不相容时拒绝 ready；就绪判定包含整折全部成员，失败不缩池；publication 在完整 artifact 生成后。 |
| HostMemoryAdmission / ready DAG / release | 25 CPU 分为 4×6 加 1 coordinator；source、preparation 预留共同预算，record worker 消费控制器扣除后的 budget；merge/load/staging 共用 semaphore；失败与 finally 释放预留。capacity、ready、active 和组内顺序 seed 拓扑分列。 |
| SourceSeedCatalog → binary / candidate consumers | producer 与 checkpoint 的 seed/cache/snapshot/partition/early stopping/refit 关系先核验；comparator 和 candidate 使用 immutable source bytes，并经过 PreparedTransferCheckpoint 的输入 profile、通道、实际 preprocessing execution、target participant / whole study 排除；K0 保持 source-only，matched scratch 保留架构来源。 |

## 确认关闭的发现

1. **完成状态早于 producer/catalog 准入。** 当前顺序为载入实际 payload、核 producer、写实际 selection/refit 成本与 catalog SHA，最后才置 task completed。准备 CLI 也在 merge 与 NPZ/record SHA 完成后发表 completed。
2. **改变预算导致重 merge 与 cache SHA 漂移。** 新 `SourceChainResume.prepared_source_view` 核冻结 whole-fold cache/record 与 admitted source bindings，再直接使用旧 path/bytes；资源改变不倒改 source producer。
3. **ready cache 未绑定所声明 manifest。** 调度 ingress 对 manifest 原字节取 SHA，并传给 PreparedSourceInventory；已有 completed ledger 必须匹配同一声明。
4. **CPU budget=2 的准备节点无限 pending。** phase 最小两线程与扣除 coordinator 后的 worker budget 比较；不可负担时保存 preparation failure，再让全部依赖 source requests 明确失败退出。
5. **失败 task 上界混入 actual exposure。** 本审查独立确认 RuntimeError 前无任何 fit 的触发路径：原 task 上界仍会进入总 optimizer_updates / trial_draws。作者已将 requested upper bound 独立保存；实际 running / failed 值为 unknown，run actual totals 仅汇总完成程序，另列 unmeasured task keys。旧 executed source 的 RED 收据已读：三个 before-fit failure 却报 12 updates / 6144 draws，fitting_performed=false；作者定向 GREEN 收据 exit0 与新源码绑定，未将旧源重签成修后证据。

保留已经返回 fit 的真实物理 compute 是合理的，即使后续 producer 准入失败；该成本与 producer-admitted procedure 数量须在 scope 上区分，不能为了使 completed 字段整齐而删除尝试成本。

## 当前范式 source_config 新范围

root 后续授权 source 配方注入：pos_weight=1、val_groups_min=2、val_groups_max 从实际已准入源参与者规模确定（不继承旧 12 截断），source stopping 显式声明 32 / patience6 / min_delta1e-6 / fraction0.2。

新增源码已逐调用点阅读。`source_config` ingress 只允许 pos_weight / val_groups_min / val_groups_max，禁止借它覆盖 seed、更新数、sampling 或 objective；None 上限由 identity authority 逐 study 计数后取最大值，EpochDataset 原合同要求 identity 恰覆盖实际 subject rows。resolved 参数经 `replace` 注入每 seed 的 DeepConfig，inner / fresh refit 继续传同一参数；完整 config 与 `source_program_config` 都进入 registry signature，resume 仍比较 actual asdict(config)。program 配置还进入 checkpoint objective，runner 的 completed gate 与两个消费者的 `expected_source_config` 调用同一个 producer validator：实际 class weight、full identities / resolved cap 与逐库 held participants 数量核验一致，round / min / max 规则与现有 splitter 相同。

generic Catalog 继续允许合法旧 β8 / valmax12 producer；新 consumer plan 必须显式传 expected_source_config，才能按新程序拒绝不相容的旧配置。已有 program_config 的载入也自动核 metadata 与 actual config 的关系。验证数量直接调用既有 group_disjoint_validation_split，不复制其舍入 / min / max 规则。数据准备与 target 数值 policy 没有因该 seam 联动。

作者必要验证收据确认新增真实 BDF one-seed 短程序与三 seed dry-run config mapping exit0；实际程序 pos_weight1、resolved valmax4、inner + refit 共 3 updates / 1536 draws，registry program_config 与 checkpoint/Catalog 校验相连。245 个 actual authority 的分数划分反例得到 49 held / 196 training，未被旧 12 cap 截断。该短程序使用其自己的 2-epoch stopping fixture，不冒充正式 32-epoch上限程序的全量执行。

## 独立证据与限度

独立执行 `git diff --check` 退出 0；`git ls-files --eol` 核得关键生产 / 测试工作文件均 LF。作者测试证据仍须单独记录其命令、退出码和源文件身份。

审查者使用项目 venv 的纯 stdlib 只读脚本独立核验 final archive 全 436 成员与当前源码字节、作者 14 份 owned 文件 SHA、旧 producer 保留的 1124 份文件、新 β1 probe 保留的 31 份文件；全部无改写或 byte mismatch。脚本与原始 JSON 见本目录 `inspect_capsules.py`、`final_byte_verification.json`（退出 0）。作者 `quality_receipt.json` SHA=`8a880d194424093d0d1ae7894b29b84ea85bfdab7175137a4f6f8a4321cda8cc`；收据记录：3 项恢复/准入/预算反例、4 项失败曝光/资源/ready、6 项显式解析、1 项真实 β1 程序、14 项最终受影响合同、13 Python 文件 Ruff、diff-check 均 exit0。审查者没有把读取作者测试收据冒称独立重跑。

β1 actual probe 的 source archive 为 `95c93303a2e23e0faaf84ad8ab2e663f53a49691011a7d9568872acddd6ce86b`。它与 final fd62 的差异只有 `source_catalog.py` CRLF→LF，以及测试 import 换行；已直接从 tar 解码对照，生产代码语义一致，原 producer 保持 95c933 身份。正式 root plan SHA=`c163eb499fc6394e638a2dbd6d743d4afa3346697c8d304c3a6a7378bc931fbe`，现有纯 parser 收据核得四个整库排除折、每折完整十一库、三个 paired seed、pos1 / min2 / capNone、stopping32/6/1e-6/.2 与 4×6+1=25 / GPU capacity4。该解析不装载 cache / model，不作为云真实并发或训练完成证据。

只读复核 pytest-15 两份 native run JSON：均为三个独立 seed、六次 physical / logical fit、失败 0；每 seed 为 inner 2 updates + fresh refit 1 update。直接 cache 程序绑定 archive `6a2be24e039092bd5cf9a0afcfd2fcdf28d98bd69d74cfafa5cfda18d77f7fe8`；shared preparation DAG 绑定 `c826b6464e51ba2f4fb905d56cd6e68ab64e8f5a2a4b61ff4c9dbe6561d03557`，CPU 3 为 coordinator 1 + 准备控制器/record 2，再转 source native 2。它们只证明各自旧 executed source，不签成收尾的新源码证据，也不证明十二库/十二 checkpoint 或云吞吐已经完成。

性能上，当前每 seed 有四次 full-source `source_fit_ledger` 的 X 哈希遍历，另有 inner/held ledger；准备 merge 也可能长时间占有准备 phase 预留而只有控制器继续工作。这些只列为云实际分段计时、CPU/RAM/GPU利用率核查候选，不从静态阅读承诺提速或要求现在进行额外重构。

模块出口为 `engineering/Standards: complete`。源码、实际新/旧 producer、必要反例与独立字节核验已绑定。剩余实际执行事项属于后续 execution：public 全十二 source checkpoints / 完整两面板、真实资源利用率、ready 与末端拓扑、实际 failure denominators / source input bindings。当前报告不通过静态质量审查替代这些执行出口。
