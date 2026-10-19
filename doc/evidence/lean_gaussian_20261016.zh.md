# Lean 连续候选：低秩、尺度、投影与高斯核

日期：2026-10-16。接续[首批验收](lean_first_slice_20261016.zh.md)，
推进 V03.a、V03.b 及 V04.a 的数学子项。本文件保存各分期回执；
当前累计数与状态统一见[活动checkpoint](../assets/lean_gaussian_20261016/active_checkpoint.json)。
后续时间白化与算子LDA固定证据见[对应专题](lean_temporal_operator_20261018.zh.md)。
固定参数的声明 Gaussian 模型已接通联合分布、条件核、真实概率矩和完整预测积分；
实际跨段噪声前提、超参数不确定性与数值实现对应继续独立验收。

最新[候选层验收](../assets/lean_candidates_20261016/acceptance.json)还完成有限候选概率、
NLL、实际容差平局及物理覆盖门禁，并从真实系数后验积分推出候选归一化密度。
后续[联合条件核与状态语义验收](../assets/lean_joint_state_20261016/acceptance.json)已完成
同一候选联合测度的条件核组合；事件设计/公开事件史和已解码账本另增88条验收，完整模型族/推理合同继续验证。

## 已构造的数学规格

| 模块 | 定理数 | 主要关系 |
|---|---:|---|
| [GaussianLowRank](../../formal/lean/N2P3/GaussianLowRank.lean) | 29 | Woodbury、det/二次型、低秩与完整评分、通道数及最终 aggregate clamp |
| [ScaleGauge](../../formal/lean/N2P3/ScaleGauge.lean) | 36 | 联动先验尺度的 gauge、物理单位变换、Kronecker 次序、实际 augmented ridge QR 与 Cholesky 正规方程 |
| [NuisanceProjection](../../formal/lean/N2P3/NuisanceProjection.lean) | 48 | QR 投影、真实保留坐标与维数、stationary AR1 白化、常数/趋势字典秩与投影空间评分 |
| [GaussianConditioning](../../formal/lean/N2P3/GaussianConditioning.lean) | 25 | 多通道完成平方、加权 trace、likelihood×prior 核分解、正规方程解唯一及核模式 |

四模块共 138 个定理已各自编译通过，且经过[独立语义审查](../assets/lean_gaussian_20261016/review.json)。
总入口为 `N2P3Continuous`；[聚合验收](../assets/lean_gaussian_20261016/acceptance_v2.json)已通过，
逐模块重查无诊断，138 个定理仅依赖标准允许集，6 条错误证明逐条被拒。
与首批合计已验收 274 个定理，全部 189 份生产/实验源码摘要一致。
当前证明、检查器和配置共 31 个文件已压缩并逐成员核对，见
[物理归档清单](../../frozen/lean_gaussian_20261016_v2/manifest.json)。

[verify_continuous.py](../../formal/lean/scripts/verify_continuous.py)首轮正确拒绝将一条
未化简到 False 的负例当作成功。等价的有限矩阵定义修复后，
[增量验收器](../../formal/lean/scripts/accept_repaired_continuous.py)核对所有成功检查的
源码/配置/运行器身份不变，仅重新检查这两条投影负例。成功检查各自保留原生产回执及日志摘要，
原失败轮回执留在本地，未改写历史身份。

下一层[概率切片验收](../assets/lean_probability_20261016/acceptance.json)新增 53 个定理：

| 模块 | 定理数 | 已闭合的问题 |
|---|---:|---|
| [GaussianMeasure](../../formal/lean/N2P3/GaussianMeasure.lean) | 20 | 从真实标量 Gaussian 积分、Fubini 和 Lebesgue 矩阵换元推出非负、可积、质量 1 与概率测度 |
| [RetainedBasis](../../formal/lean/N2P3/RetainedBasis.lean) | 9 | 从 `QᵀQ=I` 导出保留原 Q 列的完整正交基与 `n−q` 维非冗余 B，不把 B 存在作为前提 |
| [MatrixNormalCoordinates](../../formal/lean/N2P3/MatrixNormalCoordinates.lean) | 24 | 实际行优先坐标、`R⊗L⁻¹`、完整 trace/Gram/绝对 Jacobian，源码中心化核对应的归一化概率密度 |

三个新模块另进程重查全部无诊断，53 个定理仅使用标准允许公理；5 条新错误证明
逐条化简到 False 后被拒。错误转置在非对称实例中把正确能量 2 改成 4 或 1。
[独立语义审查](../assets/lean_probability_20261016/review.json)与
[39 文件逐成员物理归档](../../frozen/lean_probability_20261016/manifest.json)保留本层身份。
组合入口为 `N2P3.ProbabilitySlice`，不改写先前已冻结的 `N2P3Continuous`。

后续 [Bayes/支持测度层验收](../assets/lean_posterior_20261016/acceptance.json)再新增 83 个定理：

| 模块 | 定理数 | 已验收范围 |
|---|---:|---|
| [BayesDensity](../../formal/lean/N2P3/BayesDensity.lean) | 31 | 实际 kernel 可积、正有限证据积分 `c/J`，正规 prior/likelihood 的各自质量 1，保留双 Jacobian 的 Bayes 密度商 |
| [RetainedMeasure](../../formal/lean/N2P3/RetainedMeasure.lean) | 12 | 从真实 Lebesgue 换元证明基无关、非零且集中于 `Px=x` 的保留空间体积，包括零维 |
| [RetainedMatrixMeasure](../../formal/lean/N2P3/RetainedMatrixMeasure.lean) | 12 | `B⊗I_channels` 的实际行优先坐标、Gram/投影、基无关多通道体积及 `PE=E` 集中性 |
| [GaussianPredictive](../../formal/lean/N2P3/GaussianPredictive.lean) | 28 | 相关校准/held 噪声的创新、实际二阶矩协方差与零交叉项、Schur PSD/PD、残差/均值/行协方差代数及 C=0 退化 |

四模块重查均无诊断；83 项公理审查通过，5 条新错误证明逐条严格拒绝。
8 项聚合检查墙钟共 471.03 秒（包含独占构建和命令排程），4 次文件重查为 31.95–47.51 秒。
[独立审查](../assets/lean_posterior_20261016/review.json)分别绑定最终源码；
[48 文件物理快照](../../frozen/lean_posterior_20261016/manifest.json)已逐成员校验。
组合入口为 `N2P3.PosteriorSlice`，复用前 327 个定理的原身份。

下一层[联合条件分布与预测验收](../assets/lean_conditional_20261016/acceptance.json)
再新增 149 个定理：

| 模块 | 定理数 | 已验收范围 |
|---|---:|---|
| [GaussianJointLaw](../../formal/lean/N2P3/GaussianJointLaw.lean) | 26 | 真实 joint 质量 1、观测边缘证据、可测 Markov 后验核、同一 joint 的 compProd 与非负函数 disintegration |
| [GaussianConvolution](../../formal/lean/N2P3/GaussianConvolution.lean) | 35 | 一般 SPD 与非零均值、完整归一化常数、先证可积再积分系数，得到实际预测概率密度 |
| [RetainedGaussianLaw](../../formal/lean/N2P3/RetainedGaussianLaw.lean) | 27 | 真正保留空间上的候选均值 law、withDensity/质量/支持/基无关性、源码 small-Cholesky 与 clamp 评分；Q/SPD 导出 B/H/L 存在性 |
| [GaussianAffineLaw](../../formal/lean/N2P3/GaussianAffineLaw.lean) | 5 | 仿射密度测度等于 inverse-whitening 推送测度，Jacobian 来自实际 Lebesgue 换元 |
| [GaussianMoments](../../formal/lean/N2P3/GaussianMoments.lean) | 15 | 标量与有限乘积 Gaussian 的实际可积性、L2、均值 0 和协方差 I |
| [MatrixNormalMoments](../../formal/lean/N2P3/MatrixNormalMoments.lean) | 14 | 运输真实概率矩，识别实际源码后验均值及行优先 `rowCov⊗Sigma` 协方差 |
| [ProjectedRegression](../../formal/lean/N2P3/ProjectedRegression.lean) | 22 | 实际投影校准的 Gram/cross-Gram、均值/rowCov 与正规坐标后验；held 残差、小 Gram 及逐 record 堆叠 |
| [ContinuousPosteriorModel](../../formal/lean/N2P3/ContinuousPosteriorModel.lean) | 5 | 对实际校准后验核积分 held likelihood，使用源码 `C Cᵀ=rowCov` 因子，最终得到源码候选评分加共同常数 |

15 项检查全部通过：组合构建、8 个独立空日志重查、149 项标准公理审查，
以及 5 文件中固定 11 条错误证明的严格拒绝。负例必须以 exit 1 退出，逐条只有
`unsolved goals` 与 `⊢ False`；警告、占位、额外信任构造或导入/类型错误均不能算通过。
本层聚合墙钟 574.84 秒，其中构建 200.46 秒、公理审查 243.37 秒；
8 次独立重查为 24.65–42.05 秒。该时间不包含此前构造证明、远端推理和依赖准备。

[独立审查](../assets/lean_conditional_20261016/review.json)绑定最终源码；
[64 文件物理快照](../../frozen/lean_conditional_20261016/manifest.json)逐成员核对通过。
旧 410 个定理、48 个冻结成员、工具链与全部 189 份生产/实验源码的身份保持不变。
新组合入口为 `N2P3.ConditionalSlice`，复现入口为 `scripts/verify_conditional.py`。
新构建确认后，按用户要求用 cmd 逐文件删除 16 个已替代的尝试目录，另核实代理已用 cmd
清理的 5 个 Convolution 旧目录不存在；诊断日志已压缩并逐成员校验保存。
[清理回执](../assets/lean_conditional_20261016/cleanup.json)记录结果和旧安装压缩包此前被
自动执行审查拒绝删除的例外；可复用依赖缓存继续保留，未发现残留 `.part` 下载。

最终的 `projected_source_posterior_predictive_log_score` 先证明真实校准坐标的
`Bcᵀ Pc Bc=I`，再把源码 `sourceMean/sourceRowCov` 接到同一 disintegrating posterior。
随后在该实际后验核测度上积分，并使用源码 Cholesky 因子 C 构造 `U=D C`。
它没有用“均值和协方差相同”替代分布相等，也没有把奇异环境空间当作 Lebesgue 密度支持。

### 有限候选、容差与实际覆盖

[本层验收](../assets/lean_candidates_20261016/acceptance.json)新增 94 个定理：

| 模块 | 定理数 | 源码合同与已证关系 |
|---|---:|---|
| [CandidateProbability](../../formal/lean/N2P3/CandidateProbability.lean) | 23 | 从 `exp(e−logsumexp(e))` 的有限正权重导出正性/和为1、NLL、共同常数消去；显式合法候选先验及均匀先验特例 |
| [CandidateTie](../../formal/lean/N2P3/CandidateTie.lean) | 20 | 冻结索引顺序中的首个最大值、全部近峰候选计数、严格分差条件、共同常数不变；严格唯一最大值仍可弃权 |
| [CandidateCoverage](../../formal/lean/N2P3/CandidateCoverage.lean) | 38 | 精确 shape/binary/proper-row/positive-column 门禁、逐物理 flash 缺失和超额、四种拒绝优先级及完整 test 滤波可用时间 |
| [GaussianCandidates](../../formal/lean/N2P3/GaussianCandidates.lean) | 13 | 真实投影校准后验积分等于实际 Gaussian 密度，先导出正性，再接共同 log 常数及候选概率/NLL/log-density 容差判定 |

10 项命令全部通过：组合构建、4 次空日志独立重查、94 条标准公理审查、
4 文件共 10 条固定错误证明严格拒绝。聚合墙钟 226.01 秒，构建 121.87 秒，
公理审查 64.32 秒，独立重查为 15.69–22.00 秒；不包含此前证明构造和远端推理时间。
[独立语义审查](../assets/lean_candidates_20261016/review.json)逐项绑定最终源码，
[75 文件物理快照](../../frozen/lean_candidates_20261016/manifest.json)逐成员校验通过。
继承的 559 个定理、64 个冻结成员及全部 189 份生产源码身份不变。

通用命题来自实际源码和调用者的表达式；[证明合同](../assets/lean_candidates_20261016/science_contract.json)
记录独立验收义务及能区分错误版本的反例。Gaussian 模型的输入字段只有矩阵、支持、
正定/ridge 和精确因子方程，没有预测积分、密度正性、log 等式、概率和为1或 winner 结论。
证明按“实际后验测度→独立卷积积分等式→Gaussian 密度正性→共同 log 常数→有限候选结果”
推进；合法候选先验的和为1是输入合同，输出概率的和为1另行推导。
构造的 `zeroDesignModel` 排除了模型前提互相矛盾的空洞证明；它是无 nuisance 的抽象见证，
实际两列 nuisance 和事件输入对应由独立生产调用探针检验。

[真实源码探针](../assets/lean_candidates_20261016/source_probe_02.json)在三组双通道
固定合法模型上直接调用 `score`，用另写卷积、显式 stationary 白化矩阵和独立补空间
构造完整 Kronecker Gaussian。候选相关 det 项在这些例子中不相等；最大 logPDF 误差
为 `1.954e−14`，Decimal 70 位独立归一化的最大概率误差为 `2.665e−15`，NLL 表达式对照一致。
这未调用完整 BNCI 训练 runner，也未认证真实 EEG 的概率校准或 IEEE 误差界。

实际评分还复现：分差约 `5.00001e−11` 时 b 是严格最大者，但输出 `tied_evidence`；
约 `2.00000e−10` 时输出 b。Lean 的 `1/10000000000` 是绝对容差的实数规格，
浮点常量及边界舍入仍属数值对应义务。共同常数等价比较的是 evidence 和 **log density**，
不能把相同绝对容差直接用于 density/probability；首 argmax 顺序对应词表的冻结索引。

只使用 6 次行闪烁，或使用各行 9 次/各列 11 次的 120 事件，都可令全部 36 候选有
membership，后者甚至令各候选恰有 20 次 evidence；实际 score 可以接受，而物理覆盖门禁
会拒绝，因为计划要求 12 种 flash 各 10 次。成功 raw view 的 `stop≤available=test_scope.stop`
已证明；实际探针为 `19451<33000`。`score` 没有当前时钟输入，不能由可用时间元数据推断
已经实现在线时钟门禁。31 份探针/实际导入源码及 NLL 调用者字节在运行前后相同。

候选层首先证明正规化条件似然与显式候选先验的权重关系；下节进一步构造同一候选—观察
联合测度及 Markov 后验核，包括合法零先验。每个候选密度的质量1从实际 law 推出。
整项目、完整 V03.a/V03.b 与 Python/IEEE 对应仍保持未完成。
构建验收后按用户规定用 cmd 逐文件清理 17 个旧尝试目录及 1 份被替代的探针输出；
35 份诊断先压缩并逐成员核验。[清理回执](../assets/lean_candidates_20261016/cleanup.json)
记录已核实不存在的路径。本轮没有下载新临时包，未发现 `.part` 残留；
旧安装包此前的自动审批拒绝删除例外继续原样记录。

### 真实候选条件核、分类风险与状态语义

[本层8项验收](../assets/lean_joint_state_20261016/acceptance.json)新增 88 个定理：

| 模块 | 定理数 | 已证内容与范围 |
|---|---:|---|
| [CandidateJointLaw](../../formal/lean/N2P3/CandidateJointLaw.lean) | 33 | 从实际保留空间 Gaussian law 构造候选—观察联合概率，推导两个边缘、可测 Markov 后验、同一 joint 的 compProd/非负函数 disintegration，源码均匀先验概率等于后验单点质量 |
| [FrozenState](../../formal/lean/N2P3/FrozenState.lean) | 41 | 连续模型10字段及联合噪声4字段的独立语义编解码、递归值复制、摘要/核心检查顺序及正常形往返；返回已解码构造输入，不声明完整源构造器已经通过 |
| [CandidateDecisionRisk](../../formal/lean/N2P3/CandidateDecisionRisk.lean) | 14 | 实际后验测度下0–1损失积分、返回候选的最小条件分类错误率、真实后验 NLL，以及近峰候选的错误率差界；没有为弃权动作指定代价 |

候选密度的质量1先由 `candidateLaw_probability/withDensity` 导出，再构造共同保留体积
乘有限计数测度上的 joint。显式候选先验只要求非负且总和1，允许零先验；正有限边缘密度、
Markov 性和同一 joint 的分解等式都是输出结论。源码概率与均匀先验核单点质量的等式最后
实例化，未仅靠“归一化权重”这个名字推断已构造条件分布。支持外的公式是一个可测版本，
该 joint 仅在观察边缘几乎处处确定后验。

分类错误率由真实 `∫ 1[truth≠prediction]` 定义并推出 `1−posterior{prediction}`。
接受的 source winner 在该声明模型、均匀候选先验和0–1损失下最优。对 log 分差不超过 tol
的近峰候选，其相对最佳候选的错误率差不超过 `1−exp(−tol)`；这不为独立弃权结果提供损失保证。

8 项命令全部通过：组合构建、3 次空日志独立重查、88 条标准公理审查、3 文件固定9条
错误证明严格拒绝；聚合墙钟 312.56 秒，构建 151.52 秒、公理审查 130.92 秒，
独立重查为 20.74–28.00 秒。这不包含此前证明构造及独立模块依赖构建时间。
[独立审查](../assets/lean_joint_state_20261016/review.json)绑定最终字节；
[84文件证明快照](../../frozen/lean_joint_state_20261016/manifest.json)逐成员核验通过。
递归状态 ADT 使用 mutual 块，新验收器明确记录其作用域并覆盖全部三个递归复制证明；
此前已接受的检查器和75个成员保持冻结，189份生产源码摘要不变。

状态层没有把 `decode(encode s)=s` 放入前提：字段编码、独立字段读取和递归值树复制先展开
证明，再推出按 schema→metadata摘要→数组摘要→核心字段的协议结果。摘要函数是任意函数，
不假设 SHA256 无条件单射或带有外部授权意义。当前核心只覆盖明确列出的维度、布局、
phi/sfreq、身份列表和 provenance 对象等检查；JSON/NPZ/字节类型、物理写保护、完整账本、
PSD/Cholesky/浮点容差以及构造后的全链预测不变性仍需各自的证明。

[真实状态探针](../assets/lean_state_20261016/source_probe_01/report.json)确认连续模型正常
构造/保存/重载/评分精确重放、输入和嵌套元数据复制、不可写数组、错误摘要与语义状态拒绝、
`xb` 不覆盖。显式写入后抛异常的 writer 故障注入会留下 partial 文件，因此没有原子发布保证；
该文件已用 cmd 清理。个人模型正常流程也通过；只有显式修改公开
`predictor.state['trunk.classifier.bias'][1] += 1` 后，live trunk 保持原值，保存重载后的 logit
增加约1。此反例说明任意外部状态修改下的双表示保持不能无条件成立，未判为未修改官方流程失败。

进一步的[账本边界实测](../assets/lean_state_20261016/ledger_probe_01/report.json)发现当前
analytic 分支按 Python 真值检查，而非按 records 的 list 类型检查：`None/False/0` 都可通过
构造、保存和加载，随后相同合法评分输入会在构造对象和重载对象上分别抛出原始 `TypeError`。
`records={}` 则可迭代且评分通过；`reference_id=True` 或空白字符串也通过。
training 构造器接受 `ridge=True` 作为正数，拒绝 `ridge=False`；record 的 exact-int 计数则排除 bool。
这是“构造器接受必然推出评分就绪”的明确反例；完整 StateLedger 必须忠于这些不同分支，
不能悄悄加强检查后宣称验证了当前源码。该探针没有运行 fit、Torch 或训练。

两组真实探针分别绑定193和191个文件，均完整包含原189份生产源码，额外文件属于探针和fixture，
不增加 Lean 已证覆盖。实际源字节、两份原回执和10个 NPZ/PT 产物共206成员已
[逐成员封存](../../frozen/lean_joint_state_20261016/state_probe_manifest.json)。
本轮没有修改生产算法；后续继续完整账本与评分就绪合同、各模型族状态一致性，以及实际事件设计和完整历史组合。
[续接清单](../assets/lean_joint_state_20261016/next_slice_review.json)逐项记录实际 `+=` 事件设计、
完整设计对标签仿射的边界、重叠/漏事件/删失反例，以及两个构造器的真实账本类型规则。
验收后已用cmd逐文件清理6个旧尝试目录；12份诊断先压缩并校验保留，详见
[清理回执](../assets/lean_joint_state_20261016/cleanup.json)。本轮无新临时包或 `.part` 残留；
旧安装包仍保留此前自动审批拒绝删除的记录。

### 事件设计、完整事件史与构造器账本

新增[聚合验收](../assets/lean_event_ledger_20261016/acceptance.json)通过8项检查，
包括构建、三个独立空诊断重查、88条定理的允许公理审查及11条严格错误证明拒绝。
本切片新增88条，累计829条；新负控为3/3/5，累计66条。
[独立语义审查](../assets/lean_event_ledger_20261016/review.json)绑定最终证明、负控和运行器字节，
确认结论没有作为前提，实例及负控前提均非空洞。

| 模块 | 新定理 | 源码对应及推导 |
|---|---:|---|
| [EventDesign](../../formal/lean/N2P3/EventDesign.lean) | 28 | 从逐事件 `+=` fold 推导有限和、重叠累加、两块列顺序、系数响应、观察裁行与完整矩阵对标签的仿射关系 |
| [EventHistory](../../formal/lean/N2P3/EventHistory.lean) | 21 | decoder每轮12闪烁排列、真实事件子集与每码10次覆盖共同推出身份集合完整；全局顺序与相邻selection门禁排除外部响应，再连接实际局部设计循环 |
| [StateLedger](../../formal/lean/N2P3/StateLedger.lean) | 39 | 已解码有限值域中的Python真值、29码点空白、原始ID唯一性、exact int/bool、SHA格式、真实guard顺序及规范生产者账本通过 |

EventHistory先证明局部onset位于实际切片范围，再作自然数减法与kernel平移，
没有把非法future/负onset送入Python helper后假设其为零。重复无关的源事件顺序由
Nodup和实际ID集合推出枚举等价；浮点累加的置换不变性未包含在精确实数定理内。
非空三selection实例同时触发前后邻域检查，并在半开端点等号处放行。
负控分别删除身份子集、全局顺序，或错误包含响应末端，均严格化简到False。

[本轮真实源码探针](../assets/lean_event_ledger_20261016/source_probe_01.json)直接调用实际helper和公开API，
三组重叠设计与独立逐采样求和及卷积精确一致，覆盖仿射/非完整线性和裁行。
通用记录与评分校验仍接受漏事件输入：从 `[1,3,5,7]` 删除onset3后，设计差范数为2.5且
候选evidence改变。因此，通用NPZ调用者的完整刺激史仍需外部账本保证；该校验不提供历史完整性。
公开BNCI raw caller的上述结论依赖已审decoder、subset、物理coverage与邻域几何的组合。

StateLedger保留当前源码的弱检查：analytic `records=None/False/0`可接受但不能遍历；
空字符串/空容器迭代安全不代表完整score就绪。`RecordIdTraversalReady`仅表示取record ID的
迭代/下标阶段，未包含后续set的hashability或数值评分。Continuous接受`ridge=True`，
exact-int事件计数排除bool，Joint shrinkage则明确排除bool。先收集所有record ID再逐项检查keys的
实际错误顺序也已证明。一般生产者定理的前提是显式计数界、ID、digest和ridge/shrinkage约束，
不预设构造成功、roundtrip或评分相等。

新增18个Continuous和18个Joint手工构造器fixture按真实接受/异常行为核验，未运行fit、训练或Torch。
既有正常state重放、analytic异常及raw校准删失的三份回执只核验摘要后复用，未称为本轮重新执行。
其中原校准budget8保留删失事件5/6/7及其授权标签；本轮实际record允许right-censor，held score仍拒绝不完整尾部。
有限ERP响应支持没有证明整段零相位IIR滤波局部性，也不提供噪声独立或物理授权。

[93文件证明快照](../../frozen/lean_event_ledger_20261016/manifest.json)和
[196成员源码/回执快照](../../frozen/lean_event_ledger_20261016/source_probe_manifest.json)均逐成员解压校验。
189份生产源码未变。聚合墙钟293.62秒，其中构建217.82秒、CPU56.05秒；
本地检查计时不包括agent构造与审查，不能换算整个证明工作中的agent/CPU时间占比。
验收后用cmd逐文件清理16个旧尝试目录、32个文件，原诊断先压缩验证，见
[清理回执](../assets/lean_event_ledger_20261016/cleanup.json)；最终模块检查和活动依赖仍用于追溯。
旧leantar安装包此前被自动执行审批以`blocked by policy`拒绝删除，未重试或绕过。

[下一家族义务](../assets/lean_event_ledger_20261016/next_family_obligations.json)核对11组家族、5种个人模型和30份源码，
全部标为未证明。下一步先验证官方未修改生命周期的显式字段、复制和表示转换，再组合真实推理入口；
V04.a整体及原V01–V10全项目目标继续进行。

### 个人封装、LDA恢复与正式录制入口

新增[个人层聚合验收](../assets/lean_personal_20261016/acceptance.json)通过9项检查，
四模块逐一独立重查为空日志，125条新定理逐项仅使用允许的标准公理，三组错误证明各6条严格化简到False。
本层新增125条、18条负控，累计954条定理、84条负控；原V01–V10全项目目标继续进行。
[独立语义审查](../assets/lean_personal_20261016/review.json)绑定9份最终文件并逐项记录前提、非空洞见证与边界。

| 模块 | 新定理 | 已连接的具体内容 |
|---|---:|---|
| [PersonalState](../../formal/lean/N2P3/PersonalState.lean) | 31 | 完整递归元数据、命名张量dtype/shape/逐标量内容、五种kind、/1与/3与/4读取、摘要与legacy QC顺序，恢复到复制后的构造器输入 |
| [PersonalLDA](../../formal/lean/N2P3/PersonalLDA.lean) | 38 | 四个独立字段、正scale、缺省全通道、逐行/整批缺导策略、channel-major展平，以及实际标准化/点积/截距公式 |
| [PersonalAdmission](../../formal/lean/N2P3/PersonalAdmission.lean) | 39 | 原始录制callback、输入/会话字段和错误顺序、Python有限JSON比较、分组时间检查与其弱路径、保存元数据到scope字段、无证据弃权及请求数守恒 |
| [PersonalRestore](../../formal/lean/N2P3/PersonalRestore.lean) | 17 | 从真实named Tensor逐值和shape读取四字段及int64通道，再接封印加载、构造子层与整个波形LDA分支 |

恢复评分从命名字段和独立解码推导，未把整状态roundtrip或预测相等当作前提。
具体非空包含float64向量、形状 `[1,1]` 的单元素截距和额外未使用张量，恢复后实际有限和评分为45。
PersonalState保留全部五种kind的输入表示；其结论止于构造器输入，不代表五种完整数值构造器和前向已证明。
PersonalRestore只连接LDA四字段子构造；源constructor并未验证特征宽度等于真实feature map宽度，
该宽度绑定是typed evaluator的显式额外域，不能由“构造接受”推断评分必完成。

[真实公开API探针](../assets/lean_personal_20261016/source_probe_01.json)使用4个合法合成epoch，
运行14个构造、8个scope、8个codec实例，正常LDA的实际feature仿射式、保存/加载及复制行为一致。
截距 `[1,1]`、额外张量等按实际规则处理；0或59维状态可以构造，而实际60维feature评分会失败。
未拟合、未训练，未新跑神经或ERP前向。Torch单线程运行10.7561秒，进程峰值约343MiB；
临时PT通过cmd单文件删除并确认不存在。先前193输入身份的scratch神经正常重放与显式public-state修改反例仅核摘要复用。

新探针还直接确认：`metadata.session`全None/NaN、或带必需列的0行元数据表，可以在非空epoch上
通过dataset/input/scope检查并评分；mixed表只检查present-ID组。命名组中的NaN时间会拒绝，
但私有session比较函数单独接收NaN时可通过。formal分组保留这些弱路径，present key限定为源官方builder产生的字符串；
没有把任意pandas对象的先分组再str转换冒称为已精化。

正式CLI先同一份artifact字节加载，再调用原始录制入口。其callback先核参与者、独立会话及严格晚于
校准标签可用时间，再核simulation、采集参考和几何；数值处理前另有preprocessing/EOG/QC/reference准备步。
返回dataset后再次检查输入与scope；`inference_record`先scope再评分，而直接`predict_logit`只检查输入。
原始QC抛无可用证据异常时，CLI直接弃权且保留scheduled分母，不能宣称每个推理请求都会产生预测。
数学route对解析、身份解析、几何和QC/EOG数值保留显式外部步骤，不由此推断物理时钟真实性。

[10项有限JSON比较原语探针](../assets/lean_personal_20261016/source_probe_equality_01.json)单独验证
Python的bool/int/float数值别名、嵌套比较、对象字段重排、missing与present-null、列表顺序和大整数/float边界。
这只是语言比较原语证据，未重跑公开admission。Formal PyEq递归规范化有限数值并把对象表示为带Option的key lookup，
保留missing/null区别；`is None`/`is False`检查仍使用独立精确tag规则。

[103文件证明快照](../../frozen/lean_personal_20261016/manifest.json)与
[198成员源码/回执快照](../../frozen/lean_personal_20261016/source_probe_manifest.json)均按原生产摘要逐成员核对。
189份生产源码未变。聚合耗时209.44秒；构建123.08秒、CPU51.25秒，本地计时不包含agent构造和审查。
[清理回执](../assets/lean_personal_20261016/cleanup.json)确认已用cmd逐文件删除14个废弃目录中的28个文件，
28份旧诊断先压缩逐成员核验；11份最终模块检查和活动依赖继续保留。旧leantar包的既有自动审批拒绝未绕过。

完整神经/adapter/BN和ERP数值构造与前向、全部QC/EOG、TemporalNoise/OperatorLDA/EA/ACW及真实I/O/IEEE精化
仍独立推进；同表示LDA恢复不等同于sklearn float32两次舍入路径的bitwise等价。
[下一切片预备](../assets/lean_personal_20261016/next_slice_review.json)已实读TemporalNoise各mode与OperatorLDA调用者；
26项小型数值检查及10条实际拒绝观察只作source-bound预备，尚未Lean形式化。

### 真实观察支持

源码 `_whiten_and_project` 移除白化后的常数与趋势方向。令 `QᵀQ=I`，
`P=I−QQᵀ`，则投影后的真实时间协方差为 `P+UUᵀ`，在整个环境空间上奇异。
`I+UUᵀ` 作为恒等扩展可以得到相同的保留坐标 det/二次型，但概率密度应在
真实 `N−2` 维观察支持上建立，不能从环境矩阵正定直接推断投影数据的环境密度。

独立审查发现仅 `BBᵀ=P` 还允许冗余 Parseval frame。最终真实协方差定理已同时
要求并使用 `BBᵀ=P` 与 `BᵀB=I`，识别 `Bᵀ(P+UUᵀ)B`，并给出具体冗余反例。
新 `RetainedBasis` 已证明这样的 B 确实存在，并证明 `q≤n`，没有自然数差截断漏洞。
`RetainedMeasure` 与 `RetainedMatrixMeasure` 进一步闭合了基无关嵌入体积及实际多通道支持。
该体积上的完整预测 Gaussian 密度与条件化构造已在上述联合条件层闭合；
数值 QR 实现对应仍独立待验收。

### AR1 与尺度

从实际有限白化矩阵条目证明 `W K_AR1 Wᵀ=I`，包括首样本
`W₀₀=sqrt(1−phi²)`；由此推出 W 可逆、`|phi|<1` 下 K 正定。
`[ones, linspace(-1,1)]` 的两列独立及其白化后秩不变也已证明。
这些是固定参数下的精确矩阵性质，随机过程假设和拟合参数的统计有效性分别验收。

协方差规范变换 `K→aK, Sigma→Sigma/a` 必须同时令 `lambda→lambda/a`，
才能保持先验与预测协方差。电压单位变化 `Y→sY, Sigma→s²Sigma`
则保持 lambda；相同维数候选的评分差不变。没有把重新估计噪声后的完整 refit
等同于冻结协方差参考模型的代数不变性。

### 高斯核与概率

已从实际正规方程推导完成平方及非归一化核的分解，没有假设待证的后验均值。
后续 `BayesDensity` 已独立证明 W 无关因子与实际积分的关系：若核分解为 c 倍中心化核，
Jacobian normalizer 为 J，则实际积分为 `c/J>0`；先证可积后完成归一化。
正规 prior 使用 `sqrt(lambda) I`，likelihood 使用 `H⁻¹`（`K=HHᵀ`），
两者各自质量 1，再推出规范化 prior×likelihood 与同一矩阵密度相等。
固定 Mathlib v4.24.0 具有标量高斯与 Banach 空间 Gaussian 接口，未包含后续版本
的 multivariate density 文件。本版本的有限维积分及仿射 Jacobian 现已在 `GaussianMeasure`
中闭合，`MatrixNormalCoordinates` 将其绑定到实际行优先矩阵能量。
上述联合条件层已将 Bayes 密度恒等式提升为同一联合测度的条件分布/disintegration，
并完成实际预测边缘积分；实际噪声前提绑定与数值精化仍使完整 V03.a 保持进行中。

预测链还需明确校准与 held 噪声的跨段协方差。非重叠 raw 读取和不同 record ID
只能约束输入复用，不能推出随机噪声独立。源码的逐段独立模型可按显式前提验证；
同一 MAT 的实际 scopes 是否满足或近似满足该前提，需结合白化/投影及跨段相关另行验收，
不能由本层核归一化推断真实完整后验已经成立。

### 跨段噪声与预测模型边界

[当前源码合成探针](../assets/lean_gaussian_20261016/predictive_cross_block_probe.json)绑定 11 份
实际已导入项目/producer 源码，运行前后摘要不变；直接调用实际 event design、白化、QR 投影和
`ContinuousDecisionModel.score`，使用明确声明的固定参数数值参考，没有执行新公开数据评分。

在共享空间协方差的联合 Gaussian 模型中，记校准/held 噪声协方差为 Kc、Kh，
跨协方差为 C，拟合后 kernel 均值/行协方差为 m、V。令 `J=C Kc⁻¹`，正确的条件表达式为

\[
m_h=D_hm+J(Y_c-D_cm),\qquad
T_h=K_h-JC^\top+(D_h-JD_c)V(D_h-JD_c)^\top.
\]

源码的 `K_h+D_h V D_hᵀ` 与均值 `D_h m` 对应 C=0 的分段独立模型。
对相邻 `[0,16)`、`[16,32)`、phi=0.5 的合法 stationary AR1 与非对角双通道空间协方差，
实际白化/趋势投影后的跨协方差范数仍为 0.241796。源码评分与独立段 dense 公式相符，
与相关联合模型的候选 log Bayes factor 相差约 −0.00488105。
[独立复核](../assets/lean_gaussian_20261016/predictive_cross_block_review.json)从实际投影协方差的
特征基出发，构造完整行优先 Kronecker 联合观测协方差后直接 Schur 条件化，复现同一差异。
两种计算都核实联合矩阵正定，差异不来自非法协方差或冗余坐标。

本轮只读调用链复核确认：plan 预先提供 basis 和 lambda；源码先以 phi=0 拟合，
仅从校准记录内相邻残差估计并截断 phi，再重拟合均值/rowCov，随后用残差估计 Sigma。
状态冻结、保存/重载后，held score 只读使用这些参数，候选仅改变公开 membership 生成的设计。
这闭合了声明模型的参数使用顺序审查；数据估计的参数代入没有积分超参数不确定性。

`raw_continuous` 对整个授权 test suffix 一次零相位滤波后才裁出各 decision。
decision 索引区间互斥不等于其滤波依赖范围互斥；源码也明确不声明这些 decision
统计独立。C/T 分别滤波和 reset 可以保持授权 raw 范围，不能推出 same-record 噪声独立。

该反例推翻的是“原始支持不重叠即可推出噪声独立”的过强候选命题；
[前提待绑定记录](../assets/lean_gaussian_20261016/predictive_model_obligation.json)将其归为
独立段生成模型与同记录数据解释之间的待验证假设，没有判定源码独立模型的代数有误。
合成短间隔的差异也不作为真实公开数据效应量。

只读的 16 个既有 C/T 单元及 8 人 E2 scope 回执显示同一 MAT 的分段使用；
最小 calibration→完整 filtered test 输入间隔为 244 样本，第一实际评分 view 的间隔为 4557 样本。
该 view 来自整段 test 的双向滤波，4557 不能替代完整处理输入的支持间隔。
E2 相邻组间隔为 1754–2786 样本，held fold 的训练还可能包含其两侧组。
这些元数据约束几何，不直接给出 C=0 或忽略 C 的误差界。

正确的未滤波 AR1 `conditional_prefix` 创新可以消除这个合成例子的直接跨噪声协方差；
新增已观察 prefix 所含信息、kernel posterior 的更新及实际有限滤波后的条件对象仍需独立推导。
源码、两个独立探针及 16 份实际消费的几何回执已逐成员封存，见
[证据归档](../../frozen/lean_predictive_boundary_20261016/manifest.json)。

## 源码、行为及资源证据

独立审查绑定最终证明与生产源码 SHA-256。三组双通道源码评分与独立密集参考
比较通过，最大误差约 `4.27e−14`；18 项相关已有测试通过。
这些是合成软件证据，未读取原始 EEG 或训练，不替代 Python/IEEE refinement。

[性能回执](../assets/lean_gaussian_20261016/performance.json)与
[启动/并发测量](lean_execution_performance_20261016.zh.md)记录相同冻结证明的交替对照。
安全路径重排约快 1.90 倍；两个独立检查由 58.29 秒降至 31.09 秒。
这些收益分别测量，不能相乘作为整个形式化工作的加速比。

候选概率、容差和覆盖的初始[只读预备记录](../assets/lean_conditional_20261016/next_slice_review.json)
已由上述 94 条候选层验收落实，候选联合条件核也已在后续88条切片中闭合。
各模型状态封装仍分别验证；个人模型与连续模型的冻结/保存实现不能互相代替。
[下一切片预备](../assets/lean_candidates_20261016/next_slice_review.json)记录有限候选联合测度的
实际 pinned API 和 V04.a 全 family 义务。摘要匹配不是签名，不能假设 SHA256 无条件单射；
连续模型的 `xb` 防覆盖也不代表原子发布。个人模型在显式外部公开state修改后的双表示反例
已由上面的真实探针复现；未修改的正常保存/预测重放通过，完整family合同仍待验证。

## 实际 ridge 诊断增量

[本片验收](../assets/lean_gaussian_20261016/regularized_diagnostic_acceptance_01.json)
新增 14 个定理和两个严格错误替代。源码保存原始 `F K Fᵀ`，实际白化因子对应
`F K Fᵀ + rho I`，精确实数规格为 `rho=max_eigenvalue/10^12`。
证明由 PSD 和正标量推出正定/因子存在性，再推导实际求解平方能量、带通道数的
对角 logdet、全空间密度质量一及同一传入残差坐标下的两密度差。
未把密度质量或整个评分等价作为前提。

[独立 root 复审](../assets/lean_gaussian_20261016/regularized_diagnostic_independent_review_01.json)
核对作者四项逐次固定身份检查、14 项标准公理和两个恰为 `False` 的错误替代；
root 另完成与前基线的联合构建。原 198 成员执行与后续 211 成员联合执行分别保留身份。
参数名 `whitened_residuals` 不意味着已经恢复原始时间坐标；若输入确已做时间白化，
不能把本片结果称作原始残差似然修正。空间白化是另一项条件，不替代该时间限制。
噪声改变后重新拟合/积分响应 posterior、实际 startup/prior-scale caller 组合仍分别待证。

[证明清单](../../frozen/lean_regularized_diagnostic_20261018/manifest.json)封存 215 个成员；
[源码证据清单](../../frozen/lean_regularized_diagnostic_20261018/source_and_evidence_manifest.json)
保留本片源码与执行。
[清理收据](../assets/lean_gaussian_20261016/regularized_diagnostic_cleanup_01.json)
记录归档核验后用 cmd 删除八个旧诊断文件、移除四个空目录，保留科学证据和活动构建。

## 实际 GLS startup、先验尺度与调用者组合

[GLS/预处理联合验收](../assets/lean_gaussian_20261016/gls_preprocessing_acceptance_01.json)
中 GLS 部分新增 40 个定理和六项严格错误替代。
[独立复核](../assets/lean_gaussian_20261016/operator_gls_caller_independent_review_01.json)
把三种 startup 与两种 prior scale 对应到实际有限白化循环、原记录 nuisance 网格、
逐记录投影、堆叠和增广 QR。由独立 inverse solve 方程推出实际 mean 与对称 rowCov，
再接上 retained-coordinate posterior kernel 与低秩 score 的积分等价；未假设整个 fit
或评分成功，也未把需要证明的积分等式作为前提。

条件 prefix 保留真实已观察历史、输出 n-p 行；first-onset 与排序只推出已建模事件在
prefix 的设计为零，不能推出未知物理历史完整。record ID 和不重叠支持也不能推出噪声独立。
每个数值操作的精确 QR/Cholesky/solve 接口与已拟合固定参数仍为明确条件；
原 V03.a 的全算法族审查、V03.b 的候选与可用时间以及保存/授权生命周期分别验收。

作者五次检查记录 v4 的 201 源码、190 runtime、215 个当时保护成员；root 与此前
基线和 OLS/QC 的联合构建用同一身份通过，墙钟 73.82 秒。原执行版本不回填。
[清理收据](../assets/lean_gaussian_20261016/gls_preprocessing_cleanup_01.json)
记录联合验收后，将 GLS 与 OLS/QC 的 33 个旧诊断文件归档、逐成员核验，再用 cmd
逐文件删除并移除 13 个空目录。最终检查、源码快照、活动 .lake 与科学证据保留；未安装包。

## 原 V03.a 范围验收

[独立全范围审查](../assets/lean_gaussian_20261016/v03a_original_scope_independent_review_02.json)
逐项核对 M01–M04、原七个源文件及实际 BNCI/operator/E2 调用者，确认固定参数的条件
Gaussian、矩阵正态与 row-major Kronecker、保留空间、低秩 quadratic/logdet 和 posterior
积分等价均有对应证明。[原子项验收](../assets/lean_gaussian_20261016/v03a_original_scope_acceptance_01.json)
据此关闭原 V03.a 的数学规格和源码审计维度；本次没有增加定理计数。

[源码 v5 交付](../assets/lean_gaussian_20261016/source_v5_delivery_01.json)记录两项审查收尾：
把 E2 实际导入的 SelectionERP 纳入活动清单，并将 exact covariance 诊断说明限定为
带 ridge 和通道数 C 的同传入残差密度差。仅该 docstring 改变，非文档 AST 相同；
模块对自身字节取 hash，因此新对象的 provenance 身份会改变，旧产物不重标。
新合同 202 源文件、191 runtime 与 230 个当时保护证明经独立空日志 direct 重查，
墙钟 14.13 秒；旧 v4 审查及更早执行保持原身份。前后源码与证据均已物理封存。

完成范围条件于合法冻结参数、各精确因子/求解方程和明确的工作噪声模型。
Python/native refinement、浮点误差认证与物理 Gaussian/独立性不是本次已证结论。
原 V03.b 的单位/gauge、事件/覆盖/真正可用时间，V04 保存状态及 V01 拟合授权仍是
原项目必须继续完成的独立子项；本验收不关闭整个 V03 或全项目。

## V03.b 一般 membership 与操作拒判组件

[B01 独立复核](../assets/lean_v03b_20261018/membership_independent_review_01.json)确认
45 条定理和 11 个严格错误替代，并纳入
[membership/BN 联合验收](../assets/lean_network_preprocessing_20261018/membership_bn_acceptance_01.json)。
由实际 slot 注入、有限基数与 prefix 保留关系推出完整物理 schedule；未观测证据不能
通过改变 mask 恢复。可用列模式的 unique count 对应候选可辨识性，计数及完整九选一
R5 构造均由原字段得出。前段 dtype/string/shape/finite 校验保持其显式接口边界。

先验总质量允许源码的 `1e-8` 容差，故无证据时 posterior 为 `prior/sum(prior)`；只有
额外精确质量一条件才等于 prior 本身。三项拒判的先后与实际相对平局规则分别建模。
[非训练反例](../assets/lean_v03b_20261018/nonfit_probe_02.json)确认，相对容差 API 的
操作判定可能因共同 score 常数而从预测变成 tie，尽管 strict argmax 与 posterior 保持。
原共同常数不变性要求作为 `false_claim_with_evidence` 保留；未改命题来迁就这个行为。

该错误与校准 mask 遗漏已[合并移交科研侧](../assets/lean_gaussian_20261016/research_error_handoff_delivery_01.json)，
验证侧未合并修复。绝对容差 Gaussian scorer 的既有 gauge 结果不受此例影响。
[原范围审查](../assets/lean_v03b_20261018/coverage_audit_01.json)的 B02–B05 仍需实际
SelectionERP seam/LLR/先验、各连续端点、可用时间与其余聚合分支的组合，完整 V03.b
保持未完成。新的数学组件不替代这些源算法要求。
