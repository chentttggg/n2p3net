# 独立公共 source 程序 Spec 评审

本评审只覆盖 `independent_selection_refit` 源程序，以及它通向独立 calibration/K0、binary comparator 和真实 membership candidate 的合同。**Spec 轴通过，零未闭合阻塞发现**：原五项工程关系、新 β=1 / 实际 participant 验证上限、配置到真实拟合和消费者门禁已经独立源码复核并绑定稳定 capsule `fd62c48b37de5a569ad2be629e61c508b931dc9b2e80bd2d3147cee2e25e94ea`。本文件不证明十二库真实完整训练、两个公共面板效能、成人确认或 B00–B14/123 项整体完成。

## 当前科研范围更新

新 source 程序声明 `pos_weight=1`、`val_groups_min=2`、`val_groups_max=None`（无额外固定 cap，执行时由该 source group 各实际库 participant 数的最大值解析为 DeepConfig 整数），选轮声明最多 32 次检查、patience 6、min_delta `1e-6`、fraction `.2`。这是明确有限预算，达到 cap 是右删失停止，不能称最优轮数已识别。它不改变目标 policy 或准备 profile。

独立源码推导及 Fraction 反例见 `beta_counterexample.py/json`，exit 0，不导入 torch 或拟合模型。实际风险行质量为

`r_i = α_d/N_d · β^(y_i) / [n_du · (1+(β−1)π_du)]`。

所以旧 `zβ−lnβ` 仍对应 participant 权重 `α_d/N_d/Z_du(β)` 的倾斜 mixture；π 异质时不是原等库等人 mixture。两人 π=.8/.2 的反例，原等人 posterior 为 `19/24`，旧去 β posterior 为 `83/131`；β=1 精确恢复原风险测度。此结论是在 unrestricted CE population optimum 下的明确公式，不证明有限神经网络已经达到该最优或新源泛化更好。

旧 source 的冻结 class LLR 与二分类 posterior 也必须分开。当前 binary comparator 只报 AUC，固定平移不改变其排名；若新程序给 binary NLL/Brier，不能直接把减掉 source prior 的 LLR 做 sigmoid 当 proper posterior。π=.2、无信息 X 时，合法 LLR=0，而 proper posterior=.2；误用 .5 使 NLL 从 .5004 增到 .6931、Brier 从 .16 增到 .25。candidate 的 LLR 程序仍按其物理生成模型解释。

新 `source_config` 须穿过实际 DeepConfig、inner/refit、semantic signature、resume config 和 producer/consumer declaration；不能只在 plan 写 β=1。generic catalog 可继续重放合法旧 β=8，但新程序的 expected actual-config 关系必须阻止将旧包签成新 β=1。原稳定工程 capsule `cbd1e3038ac479b3002bf4a22dd57990b399c515637b45b208464bf00de3dd54` 保留为 scope change 前的独立身份，不自动覆盖新科研程序。

这条关系已读到真实实现：严格三个 source-only 参数由共享 resolver 对实际 identity/study membership 计数；runner 把 resolved 值注入 DeepConfig，selection/refit 和 signature/objective 保存同一 record。validator 对 actual optimizer config、source_calibration β、declared→resolved/counts 和 held participant 数联核，held 规则直接调用原 splitter，因此 `<4` 等边界保持同一 fail-closed owner。generic 有新 record 时也自动核自一致，无新 record 的原 β=8 可按原程序重放；binary/candidate 的新 plan 须传 `expected_source_config`，它不由 generic 合法旧包反推当前程序。source stopping 明示声明/真实选择曲线独立核验，不从三个配置字段猜出 32/6。

作者新 β=1 actual probe 的实际 archive 为 `95c93303a2e23e0faaf84ad8ab2e663f53a49691011a7d9568872acddd6ce86b`，只执行一份软件 seed 的真实 inner/refit，并验证 β=1、valmax4、冻结校准、Catalog 当前期望与错误 β 拒绝；三 seed 映射是 dry-run。该探针使用 2 次检查/1 patience/大 min_delta 的短软件规则，不冒充公共 32/6/`1e-6` 完整流程。31 个 probe 原文件全 SHA 通过，original JSON 未改签。

## 身份与动作

- 评审对象：`C:/Users/chent/.codex/worktrees/calibration-g-prior/n2p3-net`，基线 `3f38fe12e6a000d219967dba52faf0815cf6e2e6` 的未提交工作。没有改生产代码、六份活动指南或 commit。
- 消费了目标树的文档索引、六份指南相关合同、main 中现有 `spec/DESIGN.zh.md`、astra research 的 science/engineering/integration 模块及 EEG P300 规范。历史记忆仅作选轮/源池入口导航，全部结论以当前源码和原执行证据复核。
- 实际读取入口、selection/refit、registry、resume、prepared inventory、资源 DAG、catalog、checkpoint gates 和两个消费者。没有启动训练、重复准备、导入 torch、运行 pytest 或加载 tensor。
- 仅写本目录的评审和纯标准库 JSON/字节检查脚本。`inspect_evidence.py` 使用项目 venv 执行，exit 0；20 个当前相关文件的摘要在一次检查前后相同。`finish_binding.py` exit 0，核稳定 capsule 全 436 成员及当前 20 阅读文件、保存旧证据 1124 文件、新 probe 31 文件。作者树 `git diff --check` exit 0。

## 要求到完整调用链

| 当前要求 | 实际 owner / caller | 评审结论 |
|---|---|---|
| 显式独立源程序 | `resolve_source_program` → `execute_source_chain` | 限定 P2_source、plain CE 无辅助项、six、三组 paired seed、独立全局 patience；拒绝 shared reference/budget 等不同程序。当前所需有限 zero、baseline none、128 Hz/179、实测 Cz6 来自声明完整 profile 和实际执行准入；不能用历史 causal 178 重签。 |
| 四 outer、完整十一源库 | source DAG → `_prepare_source_group_inputs` → `source_identity_axes` / `source_fit_ledger` → `SourceCatalog.validate_fold` | 只有完整依赖集 ready 才派发；实际 participant/study/mask、完整成员集合和整目标研究排除都验证。prepare failed/pending 不会静默删库。实际十二库库存的物理准入仍由新云产物验收。 |
| 每个 source seed 各自选轮 | `fit_source_trajectory` → `_fit_source_with_early_stopping` → `domain_stratified_group_validation_split` → DeepBaseline / PatienceStopper | 每库内留整 participant；内层统计/BN/训练仅来自训练参与者，held 参与者只给验证损失。规则全局相同、计数器各 fit 独立，外层目标不参与。epoch 明确定义为 `ceil(inner_rows/batch_size)` 个 replacement-sampling source updates，不冒充无放回全遍历。 |
| 丢弃 inner 后从头 full refit | 同一 owner 再调用固定-update `fit_source_trajectory` | 新 model、新同 seed 初始化、新 full-source 输入统计，全部合法源行；真实 Adam/更新/draws/batch digest 入 contract。生产 DeepBaseline 明确重新 seed/init；旧 native 测试保留了与独立从头 full producer 位级比较的执行来源。 |
| 一次 whole-fold 加载、三独立流程 | source_execution_groups → `_source_group_worker` → `execute_source_chain` | outer/layout 一个持有源视图的 worker，三 seed 顺序执行以共享内存。它们在科学上独立；worker progress 明示 sequential cache-group 拓扑，不能把未派发 seed 解释成科学依赖未 ready。12 checkpoint 与首次 24 个逻辑 NN fit 分开。 |
| selected trajectory 与 resume | exact signature + SharedTrajectoryRegistry / SourceChainResume | 选轮签名绑定本次 source cache SHA 与完整 ledger digest；selected reuse 保留原 checkpoint bytes/strategy/selection，receipt 才记本次 physical=0。resume 校验原 inventory/source/cache/recipe/ledger，旧模型维持原 source snapshot。 |
| B12 source catalog / consumers | `validate_source_procedure` → `SourceSeedCatalog` → comparator / independent candidate | catalog 读取冻结字节，验证真实 per-seed producer、partition、曲线及预算，不消费独立自签“成功” flag。消费者按 study/source seed 索引，并通过 PreparedTransferCheckpoint 验实际架构/预处理/identity；不存在目标 test 选源或全路由。 |
| K0 / 同批 controls / 分母 | FrozenTrialPredictor.from_source / prefix programs / comparator requests / candidate metrics | K0 冻结 source 参数、BN 和输入统计；不拟合目标统计、温度或经典/连续专家。binary 保留 requested→actual K、不可用项并去重实际拟合；candidate 使用真实物理 membership 与共同未来 decision，失败/弃权保留 requested 分母。同 K/架构的 scratch 匹配由共同 owner 核验。 |
| CPU/RAM/GPU 共享排程 | resolve_source_resources → HostMemoryAdmission → ready DAG | coordinator 与实际 native worker 线程共用总预算，prep controller/record workers 与 source fit 共用准入；云 floor 不允许静默串行。25/4 的配置 capacity 不当作云实际吞吐证明。 |

## 已关闭的实际关系问题

1. **prepared 与请求 manifest 不一致。** PreparedSourceInventory 现在在 ready 验证前比较本次 manifest SHA；源 runner 从已验证声明生成 expected 映射。不能把同库旧 completed ledger 作为新请求输入。
2. **producer 准入前计 completed。** 独立 source runner 先载入 checkpoint 并完成 `validate_source_procedure`、选择账本和 catalog entry，最后才写 `status=completed`。校验抛异常时 parent group 明确失败，原请求和部分产物保留。
3. **改线程预算恢复重造合并 cache。** `prepared_source_view` 从原冻结 task/input bindings 找同一 whole-fold view，验证 cache 和 attestation 原 bytes；新资源预算不会改旧 view/source SHA，也不会迫使完成 seed 重训。
4. **CPU budget 2 导致无可派发 preparation 永久 pending。** 新比较使用 `worker_cpu_budget`；不足时记录 failed prep 节点，在下一轮由完整依赖判定使 source group failed，保存全部请求并终止，不创建 raw/prep/cache。
5. **失败 source task 的计划上界进入实际 optimizer exposure。** 与独立 Standards 交叉复核后，新源码把 planned upper bound 单列，running/failed 的 actual updates/draws 保持未知；父、子及终态汇总只把 completed procedure 算入 actual，另列 `unmeasured_source_fit_tasks`。实际返回 fit 的 compute 仍保留；`physical_optimizer_exposure_scope=returned_fit_attempts` 与 `optimizer_exposure_scope=completed_source_procedures` 分开解释。

没有另发现阻塞当前切片的科研 Spec 问题。第五项旧源 RED receipt 明确记录 before-fit 三失败却报告 12 updates / 6144 draws / fitting false；作者补了 red-capable 修复回归。最终 `quality_receipt.json` 列 3 项 resume/admission/无法派发、4 项 failure/resource/ready、6 项 parser、1 个实际新配置 probe 与最终 14 项相关合同，均 exit 0，Ruff 和 diff check exit 0；本评审消费这些具体测试入口/结果和稳定文件关系，没有另跑 torch/pytest。没有重复全集。

## 已有执行证据的独立复核

检查 `pytest-15/test_native_cli_saves_actual_t0/run` 与 `pytest-15/test_native_shared_preparation0/run`：两条 run **各自** 3 个请求、3 个 catalog entries、6 个 physical 和 6 个 logical fits；这是两次小规模软件运行，不是六个公共 source seed，也不是十二库 full matrix。

纯标准库脚本独立重推 patience 的 best epoch、stop 范围和 full-refit update 数；检查 inner/held 行不交叉、union 等于 full 行、participant authority 不交叉、cache SHA、179/Cz6/profile、实际 checkpoint SHA 和 source archive/checkout 文件摘要。exit 0。没有加载 checkpoint tensor、重放模型或复算性能，因此这次动作只确认 JSON 关系与 bytes 完整性。

三 seed native 原执行 archive 为 `6a2be24e039092bd5cf9a0afcfd2fcdf28d98bd69d74cfafa5cfda18d77f7fe8`；shared-preparation DAG 原执行 archive 为 `c826b6464e51ba2f4fb905d56cd6e68ab64e8f5a2a4b61ff4c9dbe6561d03557`。它们各自证明原 β=8 source 实施曾走过实际 BDF reader、filter、cache、CLI 和选择/refit，不证明后来修复或 β=1 新程序在这些快照执行。pytest 后续自动保留策略清掉原临时目录前作者已逐字节保存；现位于作者树 `tmp/public_source_delivery_20261008/previous_producers/`，仅文件定位映射变更，original manifest/report/checkpoint 都未修改。当前文件与各原 source 的差异列在 `readonly_evidence.json`。

## 科研和运行出口的范围

- 当前首批范围是四个整库 outer × 剩余十一完整 source 库 × 三组 source→target paired seed。旧 B7 九库/full+matched 48 模板、固定 60/90、SupCon/MMD 等历史实验不构成本轮必须兼容或必跑的要求。
- 新云 prepared 输入、完整 source 12 checkpoint、四库 binary 和合法 candidate 面板仍待实际任务 ledger / saved replay / complete coverage 验收。测试通过或本文源码通过不晋升模型效能。
- GTN 当前 exact stems/global person map 与论文 250 人关系、BI2012 已知 subject3 online/P4 记录排除的实际收据及所有完整库成员身份，以新 preparation/manifest 为准。不能依据论文 nominal 人数或旧 record 总量修改成功分母，也不能静默减少 source 池。
- 云实际并发、利用率、峰值、依赖尾部与墙钟由 root 的真实实例采样确认；本评审没有重复启动 root 已运行的 preparation owner，也没有干扰本地 B08 1344 运行。
