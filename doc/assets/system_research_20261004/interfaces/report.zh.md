# 2026-10-04 接口、拟合证据与统计支持集独立复审

本报告针对 2026-09-30“全链合同修复”之后的当前源码。只读审查生产代码；新文件和合成产物仅保存在本目录与 `experiments/runs/system_research_20261004/interfaces/`。没有改变历史数据、源码、模型、旧结果或活动指导，也没有运行新性能矩阵。以下反例说明当前实现允许什么，不能反推历史十二库结果已经遭遇这些缺陷。

使用 `codebase-design` 与 `eeg-p300-research` 的 Interface/Module、实际调用链及独立反例方法。先读文档索引、研究规范、输入合同、技术结构、工程说明和 09-30 实施记录，再追当前源码。旧记录只用于定位，不作为当前实现证据。

结论：前轮修复有真实价值，但“全链已闭合”的强描述不成立。问题既有具体实现遗漏，也有更深层的概念混同：**相同物理证据、相同数值输入、同一拟合授权、同一算子执行不是同一种身份。** 现在一处把它们当作几个互不关联的自校验记录，另一处又要求它们永远完全相等。这会同时放过错误复用，并阻止合法的新滤波/新几何算子实验。

## 可复现材料

- 独立行为探针：`doc/assets/system_research_20261004/interfaces/probe_interfaces.py`。
- 结果及本次检查的九个生产文件 SHA-256：`experiments/runs/system_research_20261004/interfaces/probe_results.json`。
- 运行：`.venv/Scripts/python.exe doc/assets/system_research_20261004/interfaces/probe_interfaces.py`；最终运行退出码 0。
- 使用仓库合成 fixture 构造具有合法 channel、preprocessing、event timeline 和 participant identity 的输入；不是手造一个绕过全部模型门禁的裸数组。
- EA 未写数组探针仅把 `np.empty_like` 的初值固定为 123，便于判定哪些元素没被写入；拟合、CLI 分支、保存和加载校验均使用当前生产实现。这个探针证明**未赋值**，不声称真实运行一定出现数值 123。

## I1：跨记录结果复用可以给旧预测重新贴上新评分输入指纹

严重性：高。已执行完整的共享目标执行 Interface 反例。

源码位置：

- `src/transfer/target_experiments.py:439–461`：外部 `reuse_target_records` 接纳匹配的 source checkpoint、**父** target cache、运行环境及 compact stage，复用 key 只包括 participant、recipe、calibration/test 行号。
- 同文件 `508–517`：本次实际 scoped dataset 的 identity 与评分输入指纹已重新计算。
- 同文件 `551–596`：导入复用仅再核对预测的 train/test 行号、y、模型状态；没有比较旧 cell 的 input 指纹、本次 scope identity、执行 snapshot 或实际校准输入。
- 同文件 `686–703`：使用当前 scope identity 与当前 snapshot 为复用 cell 重新构建 evaluation contract。

反例：首先正常生成一个 compact waveform LDA cell。第二次保留同一父 dataset/cache，给 `scope_datasets` 提供明确不同的 materialized 视图（测试 X 取反），并声明新的 scope `input_sha256` 与执行 snapshot，同时导入前一个 compact record。三次都是通过 `execute_target_tasks` 的实际逻辑执行。

| 运行 | 声明的评分输入 | 实际 AUC |
|---|---|---:|
| 首次拟合/评分 | 原 test X | 0.49 |
| 新视图 + 外部复用 | **新** test X 指纹 | **0.49，旧预测** |
| 相同新视图直接执行 | 新 test X 指纹 | 0.51 |

第二次正常 completed、`fit_executed=false`，旧预测被新 input digest 覆盖；原/新指纹不同，新视图复用和直接执行的声明指纹相同。保存位置见 JSON 的 `result_reuse.saved`。这是 Interface 容许错误复用的证据；本次没有证明历史常规命令曾传入这种不同 scope。

成熟修法：外部导入与 worker-local reuse 共同使用一个 Module 的同一 `ExecutionIdentity`。至少绑定真实 fit view、score view、初始化状态、recipe、运行语义、实现身份；导入时先对照原始记录，成功后仍保留执行者身份。不能“先复用 scores，再用当前声明生成正确 hash”。代码 snapshot 允许不同必须有明确、受检的算子兼容声明，不能由路径或同名 recipe 默示。

## I2：EA 授权、拟合状态和输出数组未绑定，三重校验仍接受错误拟合

严重性：高；目前 EA promotion 被关闭，因此这是旁线产物与下一轮接线风险，不是已证实的十二库 A1/Q1 病因。

源码位置：

- `src/data/ea_artifact.py:139–170`：`authorization_from_dataset` 接受调用者提供的 parent hashes 与 fit_rows，检查行有效性、identity 和 mask，**并不从研究计划裁定这些行是否允许拟合**。
- 同文件 `209–229`：artifact 只检查授权行是 output row IDs 的子集；分别解码 state，但不检查 state scope 与 authorization scope 一致、sample_count 与授权行数一致、buffer_digest 与获准 X 一致、state 通道维度与 artifact 一致，也不验证 X 是 state(parent X)。
- 同文件 `353–373`：`assert_parent_replay` 只核对调用者传入的 parent hashes、授权行是否属于 parent，以及 participant key 集合；没有消费 parent 数值或外部 expected fit authorization。

反例：合法 parent 四行；authorization 声明只允许 row 0、1，scope 为 `target_calibration`；实际 state 则用 row 2、3 的 X 放大 7 倍、scope 为 `source_rows` 拟合。构造 artifact、保存、attested load、`assert_parent_replay` 全部通过。相对于真正获准拟合的输出，RMS 相对误差 **0.861412**。进一步将输出 X 全改成 123，保留同一 state/authorization，保存和加载仍通过。

这不是要求 hash 防御恶意调用者；是普通调用者把“邻近变量里的 state”接错时，Module 未维护自身声称的科学不变量。独立记录都能自证未变，不等于它们之间关系成立。文档称“科学 fit authorization 由 ea_artifact 负责”，当前 Implementation 尚未履行这个职责。

成熟修法：把 `fit authorized view → numerical state → transformed output → sealed artifact` 收进一个深 Module。外部 Interface 接受由冻结 split/plan 生成的 `AuthorizedFitView` 和待变换 view，由内部计算实际 fit buffer digest、row/domain→state 映射、state dimension/scope/count，再发布 artifact。重载用实际 parent 与 **expected scope** 对照；不能让 artifact 自己同时提供“声明”和“证明”。校验 metadata 的快速模式与数值重放认证模式要明确分工；在源/目标消费者正式接入前至少完成一次独立重放认证。

附带可复现事实：attested load 返回的 `X.flags.writeable` 为 true，且 frozen dataclass 中的 ndarray/dict 仍可原位修改。`load_ea_feature_artifact` 的“immutable”描述只对读取瞬间字节成立。应决定对象是否真正不可变；若后续处理必须可写，则生成新派生产物并使旧 identity 失效。

## I3：per-domain EA 对未出现在 fit_rows 中的域发布未初始化内容

严重性：高。已执行 CLI 保存/重载反例。

源码：`experiments/prepare_euclidean_alignment_cache.py:148–152` 用 `domains[fit_rows]` 拟合 states，随后为**全部** parent X 创建 `np.empty_like`，仅写入 states 中存在的域；没有检查 `set(domains) <= set(states)`，也没有定义未拟合域策略。

反例：四行 domain A、A、B、B；`--fit-scope per_domain --fit-rows [0,1]`。只生成 A state，B 两行保持 allocation sentinel 123，仍成功保存和 attested load。artifact 没有 per-row domain ID/state assignment 字段，加载者也无法仅凭 artifact 检查每行究竟由哪个 state 生成。

成熟修法：输出 view 必须明确路由到一个已经授权拟合的 state。缺域时立即拒绝，或由协议明确输出子集/独立状态；不能用零填充、identity transform 或邻域 state 静默补洞。执行前保证完整路由，再分配数组。artifact 保存 domain/state assignment 的身份；这也为以后新增目标域、校准与只读预测解耦提供真正的 Interface。

## I4：旧 EA 物理 cache 只在单个 promotion 入口关闭，普通模型入口仍接受

严重性：高；历史影响尚未查明。

源码：

- `experiments/run_candidate_promotion_matrix.py:2405–2411` 正确拒绝旧 `provenance.data_alignment`；`2482` 处正确关闭未接通的 EA route。
- `src/data/epochs.py:323` 的通用 validate、`1300` 的 loader 没有这种数值坐标类别检查。
- `src/transfer/checkpoint.py:586–620` 对比物理 channel/preprocessing/reference 字段，未拒绝 `data_alignment`；`913–918` 准备模型时直接消费该入口。

反例：把合法 fixture 的 X 真正变为 EA 的无量纲值，保留 FZ/CZ/PZ、电压 preprocessing，并标记旧 `data_alignment`，重新产生一致 QC 后，`save_epoch_dataset → load_epoch_dataset(validation='attested') → PreparedTransferCheckpoint → trunk_for_target('p1')` 全部通过。新类型产物没有替代旧类型的全部活动消费者。

成熟修法：坐标类别是共享数据/模型 Interface 的属性，由该 owner 裁定。物理 cache 的 active ingress 拒绝任何已经施加非物理线性坐标变换的 payload；只读历史审计走显式离线路径。把门禁放在一个 promotion CLI 里不具有 Locality。正式 EA 模型必须绑定 feature-space state/geometry，源与目标同空间，同时保存个人状态并只读 replay。

## I5：valid 参与者点估计已固定，但缺失原因矩阵被提前 intersection 损坏

严重性：中。点估计并未被本反例推翻。

源码：`experiments/analyze_candidate_cross_decision.py:1085–1112` 先计算所有 arm/replicate 的完整参与者交集，再只为该交集生成 `values`；`1246–1263` 又根据过滤后的 values 反推出原始缺失。

反例：s2 只有 left/r1 缺失；left/r0、right/r0、right/r1 都有效。固定交集排除 s2 是预期行为。但 published support 把 s2 记成 left 的 r0/r1 **和** right 的 r0/r1 全部缺失。此时无法区分数据缺失、方法失败和“因共同 cohort 被排除”。

成熟修法：先保存完整 `Participant × Arm × Replicate × K` 支持与失败原因账本，再计算支持掩码和 point estimate。原始 observed/missing 与 analysis included/excluded 是不同字段，后者不能覆盖前者。对比新增一个 arm 是否改变原有 valid cohort，应由 ContrastPlan 明确规定是全矩阵交集还是每个预声明对比交集。

## I6：旧活动配对 helper 没有同步新指纹，预算账本也没有到达所有入口

严重性：中；已执行配对 helper 反例，预算传播为源码追踪结论。

- `src/transfer/binary.py:174–180` 的 `paired_transfer_effects` 仍仅比较 calibration/test 行号 hash。
- `experiments/run_cross_dataset_transfer.py:695` 和 `experiments/run_binary_subject_transfer.py:425` 仍调用它。
- 反例给 source 与 scratch 不同 `input_sha256` 与 `test_labels_sha256`，仅行号相同，该 helper 正常产生 `delta_auc=0.4`。新 `stats.transfer` 会拒绝这种对比，但活动结果内嵌的 `paired_transfer_effects` 不会。
- `run_binary_subject_transfer.py:197–208` 调 resolver 后，`216–227` 只保存 budget adjustments，`303` 起的 cell 也未保存 `split.test_budget`；该 CLI 只有 minimum 参数，没有 test-size-policy。少于 minimum、却未改变 calibration K 时，adjustments 为空，新的 relaxation ledger 在这个入口仍会丢失。
- 标准 source-prefix 路径并非如此：`source_records.py:55–64` 核验 budget/protocol 一致，`398–400` 与 `643–645` 发布 actual count，`724/911` 加载检查，unified scoped tasks 在 `target_experiments.py:331–334` 接回预算。

成熟修法：paired identity 和 TestBudget publication 使用同一个 owner；旧 helper 不应维护第二套较弱的“配对”定义。每个已执行评分单元直接携带预算账本引用，并在摘要中分别列 requested allocation、authorized labels、retained training rows、scheduled test、scorable test；不依赖 adjustments 非空来暗示放宽。

## I7：更重要的设计限制——数值输入相等不能替代物理证据相等

性质：整体研究设计/Interface 缺口，而非旧 AUC 的算术错误。

`matched_input_digests`（`binary.py:108–155`）正确覆盖 test X、y、mask、preprocessing，但没有物理 raw/event/participant/channel/reference identity，且按其定义不覆盖 calibration X/y。`stats.transfer.py:287–290` 又把这一数值指纹当成所有对比臂必须相同的条件。

两个方向的反例/限制：

1. 对同一 dataset 的 calibration X/y 作巨大改变，test X/y 不动，`matched_input_digests` 保持相同。这本身符合“scored test input”定义；但它不能支持“同一拟合证据”的更强声明。行号 hash 同样不能补足。
2. 若研究旧/新滤波算子，同一参与者、同一原始事件、同一真值、同一已分配标签预算的 materialized test X 必然可能变化。当前 `analyze_transfer_contrasts` 必然报 `contrast arms must have matched row and scored-input identity digests`。即使补一个 physical events hash，也没有可声明“这是允许变化轴”的 Interface。本次 probe 验证了拒绝行为。

因此继续机械增加“全部 digest 必須相等”会把研究锁在原算子里，正是用户指出的局部最优风险。建议在既有数据/执行/统计 owner 中采用三个清晰概念，而不是再建一堆透传包装：

| 概念 | 描述 | 用在哪里 |
|---|---|---|
| EvidenceCohortIdentity | authority participant、raw record digest、真实 event ID/clock、true label、原始候选 schedule、已授权时间/标签范围 | 建立可比较的物理证据与固定 denominator |
| MaterializedInputIdentity | 上述 evidence 的某一合法 operator 实现、配置、fit state、mask、实际 X、列语义、时间网格 | 数值复算、保存模型、缓存、结果复用 |
| ContrastPlan | 比较中固定哪些 evidence/fit-budget/recipe 轴，允许改变哪些 operator/algorithm 轴；如何记缺失 | A/B、2×2、算法替换、滤波/窗口实验 |

同算子模型对比要求 physical evidence 和 materialized input 都匹配。滤波对比要求 physical event/label 与授权范围匹配，但 materialized operator 明确为 treatment。窗口长度导致 eligibility 变化时，预声明固定原始事件 denominator，并另给共有支持敏感性分析；不能为了获得相同 X 而阻止实验，也不能因形状不同就让统计程序悄悄换人。

## 已经成立的改进与未发现的问题

- 相对 EA floor 的数值 state 模块确实按 `epsilon × trace(R)/C` 构造，并重算协方差矩阵关系；不能把 I2 错说成数值正则仍是旧的绝对 1e-6 V²。
- `result_reuse.read_verified_bytes` 读取、验证并消费同一份 bytes；model/prediction 从 BytesIO 解码。这条通用路径正确处理了常见 TOCTOU，不应因为外部 treatment identity 缺失而整体否定它。
- 新 stats 对模型臂的 scored-input mismatch 会失败，source-prefix budget 主要路径也有严格检查。这些修复真实存在。
- candidate valid 点估计的固定交集在本反例中正确；损坏的是原始支持原因。
- promotion 禁用未接通 EA 的行为正确，应保持，直至上述 fit/space/replay 链闭合。

## 验证与下一步顺序

本次执行了六个现有测试文件：`test_ea_artifact.py`、`test_prepare_euclidean_alignment_cache.py`、`test_calibration_budgets.py`、`test_input_digests.py`、`test_transfer_statistics.py`、`test_candidate_cross_decision_analysis.py`；全部退出码 0。它们通过的同时，上述独立行为反例也成立。说明应该补上当前缺失的“组合关系”验收，而不是用更多同构单元测试代替设计。

建议顺序：

1. 先冻结 EvidenceCohort/MaterializedInput/ContrastPlan 的语义，将 prepared raw views、model origin、result reuse 与 stats 接成同一条链；不另造平行实验框架。
2. 先修外部结果复用、未初始化域、旧 EA physical ingress 和支持账本。这些是成熟工程问题，可直接完成。
3. EA 的拟合与输出生成收进 owner，消费显式 plan authorization；补 source/target 共同空间、目标校准限定、个人只读 replay。完整 observed subspace 可用性仍是科学准入前提。
4. 为边界算子/几何算法研究提供明确的允许变化轴，随后才执行模型性能矩阵。每个算子先过物理范围与数值失真门槛，再谈优化收益。

最低组合验收：同样 parent 但不同 test X 绝不可复用旧 scores；fit view 变但 test view 不变必须产生新 fit identity；authorization row 换成 test row 的 state 必须被拒；遗漏域不得发布；新滤波与旧滤波可以在相同物理事件/标签的 ContrastPlan 下比较；单 arm/replicate 缺失不得被改写成所有臂缺失；正常 sealed artifact 的只读重放仍要通过。
