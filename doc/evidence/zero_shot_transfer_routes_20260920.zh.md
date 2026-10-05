# Zero-shot 正迁移路线与下一轮对照任务（2026-09-20）

更新：2026-09-22。窗口判定先行；唯一开发效应主门槛为四库等权的 `mean ΔAUC >= +0.01`。
原有冲突门槛已物理归档，见[归档清单](../../frozen/input_window_audit_20260922/manifest.json)。

2026-09-29证据勘误：标准双侧EA必须配套源端对齐后的模型/normalizer/BN，不能固定旧plain
checkpoint仅白化target后称复现EA；原始完整P300 RPA使用目标校准标签；GALIA的
`2.12±1.88%`是classification accuracy而非AUC增益。相关路线身份和表述在本页更正，
旧实验原始产物不变，核查依据与实施规格见[优化研究](unified_optimization_20260929.zh.md)。
十二库存在不同实测缺导mask；完整EA必须先核逐折共同观测子空间与秩，不得把补零当实测白化或静默删库。
BrainForm/BI2012探针的共同空间仅Pz−Cz，使完整十二库及四个普通留库折的全源共同空间最多一维；
原六导标准EA当前不准入。下述EA为通过空间准入后的条件候选，不是可直接启动的原十二库矩阵。

## 结论先行

先按[执行计划](../roadmap.zh.md)完成窗口判定、冻结输入身份，再比较以下路线。下一轮不再继续按 SupCon、MMD、prior、decision、adapter 逐个摸索。应固定当前已知较强的 source full-fine 迁移基线、scratch、LDA 和 plain source zero-shot，然后比较有公开 EEG/BCI 迁移先例的完整路线：

1. **Euclidean Alignment（EA）**；
2. **Riemannian Procrustes Analysis（RPA）/ xDAWN + Riemannian tangent-space**；
3. **P300 joint feature alignment（GALIA 类路线）**；
4. **source selection / active sampling** 作为后续有标签或无标签目标筛选旁线；
5. 条件 MMD/AS-MMD 只作为已有深度基线的一个完整对照，不再单独堆叠为默认路线。

历史“15%”有相对chance比例与逐人正迁移率两种读法，均不再作为路线晋升门槛。
论文平均提升不能替代本项目固定cohort的配对增量。

## 先把术语分开

| 名称 | 是否读取目标数据 | 是否用目标标签 | 本项目口径 |
|---|---|---|---|
| strict zero-shot | 否 | 否 | source checkpoint 直接作用于固定 target test |
| zero-label alignment | 读取声明的目标无标签buffer | 否 | 无标签EA；读取评分test EEG时另记transductive；RPA子路线/GALIA须先核标签需求 |
| few-shot adaptation | 读取目标校准标签 | 是 | 当前K=200/800 full-fine、scratch、LDA及原始完整P300 RPA |

如果让 EA/RPA 用目标无标签样本估计协方差，就必须把它单列为 zero-label transductive，不能写成 strict zero-shot。目标 test 是否可用于估计变换也必须预先声明；更稳妥的是只用无标签 calibration stream，不读取 test 标签，并另设“目标全无数据”的 strict zero-shot 臂。

## 有明确迁移先例的路线

| 路线 | 公开先例 | 已核实的有效信息 | 对本项目的判断 |
|---|---|---|---|
| EA | He & Wu, 2020, DOI `10.1109/TBME.2019.2913914`; Wu, 2025, DOI `10.1088/1741-2552/addd49` | EEG/BCI 的跨被试/跨域二阶统计对齐，使用目标无标签协方差；公式和适用边界清楚 | 第一优先级，成本低，易做成完整 zero-label route；不保证标签条件分布对齐 |
| RPA | Rodrigues et al., 2019, DOI `10.1109/TBME.2018.2889705` | 跨被试、跨数据集 Riemannian 对齐，包含多范式 EEG；使用域几何对齐而非单纯通道缩放 | 第二优先级；需要严格固定协方差、切空间原点和 fit 范围 |
| xDAWN + Riemannian | Li et al., 2020, DOI `10.3390/app10051804` | 直接针对 P300 的 xDAWN 空间滤波和 Riemannian 分类迁移 | 强经典 baseline；可作为神经 source zero-shot 的独立竞争者，不应只当辅助诊断 |
| P300 joint feature alignment / GALIA | Altındış et al., 2023, DOI `10.1109/JBHI.2023.3299837`, HAL `hal-04255249` | P300 跨域 joint alignment；本地全文审计记录作者报告平均提升 `2.12 ± 1.88%` | 直接相关路线；但论文平均提升不能换成本项目逐被试正迁移率，且本地旧 `GaliaAdapter` 公式并不等价于论文 GALIA，必须重命名或重实现 |
| source selection / active sampling | Kilani et al., 2024, DOI `10.3389/fnins.2024.1360709`; Flores et al., 2024, arXiv `2412.17833` | P300/BCI source sample selection、跨健康状态和多中心快速校准 | 后续旁线；若筛选使用目标无标签特征，属于 zero-label transductive；若使用目标标签，则不属于 zero-shot |
| AS-MMD / conditional MMD | Chen & Delorme, arXiv `2510.21969`, DOI `10.1109/BIBM66473.2025.11356119` | 近期 P300 cross-dataset 的自适应 MMD 方向，有明确算法和训练策略 | 作为完整深度 route 对照；本地当前 conditional feature MMD 不是论文 AS-MMD，不能直接声称复现论文 |
| latent alignment | Bakas et al., 2025, DOI `10.1088/1741-2552/adb336` | EEG decoding，含 P300，讨论后层对齐和类比例敏感性 | 可作为第二阶段候选；当前没有足够逐行复核的本项目可执行数字，不先作为主路线 |

文献来源和公式边界已记录在[域迁移一手来源审计](domain_transfer_chain_primary_source_audit_20260904.zh.md)和[域迁移方案](domain_transfer_plan_20260915.zh.md)。其中已有明确的论文平均提升记录，但没有一篇被核实为“与本项目相同 target cohort、同一 scratch 基线、逐被试正迁移率至少 15%”。

## 本项目唯一开发效应主门槛

每个目标参与者u使用同一test行，配对route与plain source zero-shot；源训练数据、预算和seed匹配，checkpoint按各自输入坐标绑定。只有保持源坐标不变的state对照才复用同一source checkpoint：
`Δ_u = AUC_u(route) - AUC_u(plain source zero-shot)`。
先在每库内对参与者等权平均，再对预声明的四库等权：
`Δ_macro = (1/4) Σ_d [(1/N_d) Σ_u Δ_du]`。
唯一主门槛为 **`Δ_macro >= +0.01`（绝对AUC增加1个百分点）**。
不得将BI的样本量优势变成库权重，也不能缺一库后重新归一化宣称通过四库门槛。

`p(Δ>0)`、`p(Δ>0.01)`、median、p10、最差值、逐库估计、BI家族敏感性和区间均为风险诊断。
合法fit范围、完整配对与失败分母是效应门槛之前的准入条件；失败单元不能删去后晋升。
按参与者配对bootstrap报告区间；共享source和单seed相关性仍须通过后续paired-seed复核处理。
Wilson区间仅描述正负计数，不能作为平均Δ的区间。
点估计通过只解锁3–5 paired seeds的开发复核，不等于确认性显著性或成人产品准入。

few-shot另对同K、同监督的plain full-fine计算增量，full-fine相对scratch是监督迁移锚点；
不得用需要目标标签的scratch定义strict zero-shot效果。

## 下一轮固定矩阵

### A. 零目标标签路线（区分strict与zero-label）

- `Z0`: 当前 plain CE source checkpoint，source statistics，strict zero-shot。
- `Z1`: EA route；若用目标无标签协方差，标为 `zero_label_transductive_EA`。
- `Z2`: 仅限已证明无目标标签的RPA子路线；covariance模板也须无标签。原始完整P300 RPA另列few-shot，目标fit支持、标签、协方差和切空间原点全部写入state。
- `Z3`: xDAWN + Riemannian route；xDAWN、原点和分类器只在合法 source fit 范围拟合。
- `Z4`: 实际论文一致的P300 joint feature alignment；zero-label标签需求未核定前不准入。不能复用当前名称错误的共享Q/对角白化实现。

合法zero-label route和`Z0`做主要正迁移比较，四库整库留出保持不变；共享原始target test行、通道布局和窗口/滤波合同。标准双侧EA改变源输入坐标，须拟合route专属source模型；架构、源成员、预算与seed配对固定，不能要求与plain权重字节相同。额外无标签buffer成本同时报告。

### B. 已知强基线锚点

- `F`: 当前 plain CE source + full-fine，K=200/800。
- `S`: 同结构 target-only CE scratch，监督范围与 `F` 完全匹配。
- `L`: 原波形收缩 LDA。

`F−S` 是“有标签迁移是否值得”的强锚点；`Z1/Z2/Z3/Z4−Z0` 是“无标签域迁移是否值得”的独立问题。不能把 `Z−S` 当 zero-shot 效果。

### C. 不再作为首轮主问题

- SupCon 权重网格；
- MMD 权重网格；
- prior 强度网格；
- decision CE 与多个 adapter 自由度；
- 按目标库结果挑 route。

这些机制只有在某个完整 zero-label route 达到预注册门槛后，才进入第二阶段，并且每次只加一个完整机制。

## 必须保留的反例和失败条件

1. 目标域二阶矩相同但标签条件关系旋转：EA/RPA 可能不改善；不能把协方差对齐写成充分条件。
2. 缺导或真实 mask：不能把补零矩阵直接做完整协方差白化；先限制到共同实测布局。
3. test 统计污染：如果 route 读取 test 无标签统计，必须单列 transductive 结果；strict zero-shot 不读取 test。
4. 源模型零校准偏置：AUC 排序和 hit/threshold 质量分开报告；不能用温度/截距改变后冒称排序迁移。
5. route 失败或 route≤0 增益：保留失败分母和完整模型，不按被试结果退回另一方法。
6. 总体平均可能掩盖某一库退化：必须报告逐库Δ和负迁移尾部；总体开发门槛通过不自动晋升产品默认。

## 推荐执行顺序

1. 先做BI2014a 61人窗口判定；cache裁剪与raw重建分别标识，冻结合法输入后继续。
2. 用4个整库留出比较`Z0/Z1/Z2/Z3`，保留同一输入身份及配对source；新窗不兼容旧head时须重训并绑定新checkpoint。
3. 先EA和现有Riemannian baseline，通过合同及`Δ_macro >= +0.01`后再实施论文一致的P300 joint alignment。
4. 通过的路线做3–5 paired training seeds复核；随后才做source-set归因。未通过则归档完整失败分母，不盲调组件。
5. 成人BrainSync候选另经独立冻结验收；开发AUC门槛不代替九选一`hit@all`。

## 只替换域迁移模块时的接口约束

本轮应把“路线”限制在**输入变换/state 模块**，避免同时改变编码器、loss、target recipe、训练轮数、BN 策略和 classifier：

- `src/data/euclidean_alignment.py` 已提供最接近的深接口：`fit_*_alignment(...) -> EuclideanAlignmentState`，`transform_euclidean_alignment(X, state) -> (N,C,T)`；EA 可以直接替换当前 alignment state，N2P3 trunk 和目标 fit 继续不变。
- `src/data/domain_alignment.py` 当前是 `(N,P) -> (N,P)` feature-state 接口；它可作为 RPA/GALIA 类路线的适配位置，但现有 `GaliaAdapter` 的数学实现不能继续冒称论文 GALIA。若保留接口，至少把 state schema、fit_scope、domain_id、buffer digest 和 transform shape 固定。
- `src/baselines/riemann.py` 的 xDAWN+Riemannian 是完整 classical predictor，不是 N2P3 的域迁移模块。它应作为独立强 baseline，不要为了“接口一致”把它伪装成 N2P3 输入变换。
- AS-MMD、DANN、SupCon 和 source selection 是训练目标或数据采样路线，不能混入本轮“只替换域迁移模块”的主比较；否则消融变量会再次扩张。

窗口合同冻结后的路线顺序是`plain source zero-shot → 完整EA路线 → 标签需求明确的RPA/GALIA`。
EA主比较固定后端网络结构、target recipe、batch、epoch、原始test行和评分器，source checkpoint绑定各route；
RPA的SPD特征构造及classifier若改变，另列完整路线对照，不称raw输入state的等价替换。

## 类似条件下的预期增益

下面的范围是把本项目历史强基线与已审计文献放在同一量纲后的**工作先验**，不是承诺值；外部论文通常报告平均 AUC/accuracy，未报告本项目定义的逐被试 `p_+`。

| 条件/路线 | 对照 | 本项目或文献锚点 | 合理预期 |
|---|---|---|---|
| strict zero-shot plain source | 无域迁移 | 当前 plain source zero-shot 是必须保留的锚点；旧四库源 zero-shot 等权 AUC 约0.576 | 增益定义为0；任何路线必须相对它计算 `ΔAUC` |
| EA / RPA zero-label | plain source zero-shot | He & Wu 2020、Wu 2025、Rodrigues 2019 有跨域/跨被试对齐先例；本项目旧 EA-target-cal 分支未显示稳定增益 | 先验约 `0–+0.02 AUC`，方向不确定；须在固定cohort上验证平均增量与负迁移风险 |
| P300 joint feature alignment | 标签预算匹配的完整路线 | Altındış 2023摘要报告classification accuracy平均`+2.12±1.88%`，对照为subject-specific pipeline | 无法从该accuracy数字推定本项目ΔAUC；标签需求及原实现核定后实测，不承诺数值收益 |
| xDAWN + Riemannian | plain source / LDA | Li 2020 为 P300 经典路线，当前 `XdawnRiemann` 已有独立接口 | 预期是强 classical anchor，约 `0–+0.03 AUC` 相对 plain source；它不是 N2P3 域模块替换 |
| few-shot full-fine | same-K scratch | 当前十二库 K200 A0Q0−CE scratch 约 `+0.041 AUC`；旧四库 K48/K96/K192 为 `+0.048/+0.047/+0.028 AUC` | K200 约 `+0.03–+0.05`，K800 约 `+0.01–+0.03`，高 K 逐渐接近0；这是最强已知 target adaptation 锚点 |
| few-shot EA/RPA + full-fine | same-K plain full-fine | 当前统一矩阵 Q1/源辅助没有稳定附加收益；历史 EA 分支也未显示稳定优势 | 额外增益应先验为 `-0.01–+0.01 AUC`；只有达到 route gate 才继续 |
| source selection / PDS | same source budget random/ uniform | Kilani 2024、Flores 2024 有 source selection 先例；本地 PDS-inspired 目标与论文不等价 | 预期 `0–+0.02 AUC` 或降低负迁移尾部；不能称 strict zero-shot |

历史 full-fine 的逐被试收益率可以作为“few-shot 迁移已知可行”的参考：旧四库报告 K48/K96/K192 分别 95/123、99/123、93/123 人 AUC 高于 scratch，约 77.2%、80.5%、75.6%；这些是**有标签校准**结果，不能转成 zero-shot 证据。

## 对旧四库 K0=0.576 的重新解释

旧四库 LODO 表中的“源模型零校准”四库等权 participant-macro AUC 为 `0.5760`，K48/K96/K192 三行复用同一个 zero-shot 预测；它没有读取目标标签。相对随机排序基线 `AUC=0.5`：

- 绝对增益是 `0.5760 - 0.5 = +0.0760`，即 **+7.6 个百分点**；
- 若把随机基线的 0.5 当分母，形式上的相对比例是 `0.076/0.5=15.2%`；
- 这两个数字都不是逐被试正迁移率 `p(ΔAUC>0)`，也不是某个域迁移模块带来的增量。

所以“15%”如果指相对 chance 的比例，确实与旧 K0 结果接近；如果指 route 的逐被试正迁移率，它是另一种指标，不能从 0.576 推出。

## 能力读数与路线增量分别报告

plain source的绝对AUC及相对chance区间是能力诊断；0.57不作为第二套路线门槛。
路线只按上面的`Δ_macro >= +0.01`进行开发筛选，逐人正负计数保留为风险诊断。
历史few-shot full-fine对scratch的增益不改变这个对照定义。

## 从现有结果直接计算的正迁移/有效率

这里把不同概念分别计算，避免把 AUC 高于 chance、route 相对基线变好、平均 AUC 增益混成一个数：

| 口径 | 完成配对 | 正向数量 | 正向率 | 95% Wilson 区间 |
|---|---:|---:|---:|---|
| 旧四库 K0 source zero-shot：AUC > 0.5 | 123 | 95 | **77.24%** | [69.07%, 83.75%] |
| 旧四库 K200 full-fine − scratch | 123 | 89 | **72.36%** | [63.87%, 79.49%] |
| 旧四库 K800 full-fine − scratch | 59 | 42 | **71.19%** | [58.62%, 81.16%] |
| 十二库 K200 A0Q0 − CE scratch | 123 | 100 | **81.30%** | [73.50%, 87.20%] |
| 十二库 K800 A0Q0 − CE scratch | 72 | 61 | **84.72%** | [74.68%, 91.25%] |
| 十二库 K200 A1Q1 − A0Q0 | 123 | 64 | **52.03%** | [43.28%, 60.67%] |
| 十二库 K800 A1Q1 − A0Q0 | 72 | 37 | **51.39%** | [40.07%, 62.57%] |
| BI family K200 A1Q1 − A0Q0 | 102 | 39 | **38.24%** | [29.39%, 47.93%] |

旧四库 K0 的 77.24% 是“source zero-shot 在随机排序之上”的被试比例，不是某个域迁移路线相对 plain zero-shot 的正迁移率。十二库 A0Q0 的 81.30%/84.72% 是有标签 full-fine 相对 CE scratch 的正迁移率；A1Q1 相对 A0Q0 只有约 51%，说明完整联合包不是稳定改进路线。

正迁移率的计算脚本/数据来源分别是旧四库 `analysis_c/participant_metrics.csv`、`analysis_c/paired_participants.csv`，以及十二库 `results/ordinary_contrasts.json` 和 `results/family_contrasts.json` 的 `participant_rows`。Wilson 区间只描述二元正负计数，未处理训练 seed 不确定性和同一 source checkpoint 造成的相关性。
