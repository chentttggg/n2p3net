# 源学习—本人校准—决策链：重新建模与优化候选

日期：2026-10-04。性质：只读机制研究与独立算子反例；没有训练公共数据矩阵，没有改生产源码。

本分支使用 `codebase-design` 的 Module/Interface/Seam/Adapter 约束与 `eeg-p300-research` 的源码、公式、反例、接线顺序。实际读取当前代码，不能用旧技能中的历史模型排名覆盖活动指南。复现脚本见 [probe_learning.py](probe_learning.py)，输出见 [probe_learning.json](../../../../experiments/runs/system_research_20261004/learning/probe_learning.json)。源码SHA256随输出保存。

## 1. 先重建问题，而不是假设A1某个损失有罪

当前总体问题是：源样本来自不同电极观测、参考、刺激时序与类别先验；怎样用合法的本人少量标注，把可靠的生理区分信息转成新decision的正确排序。A1/Q1是实现这个目标的一组处理包，不是问题的定义。

已有冻结十二源结果只有四个常规目标库、一个配对训练seed。三库Q1只含parameter prior，BNCI才含prior+decision。BNCI K200/K800只有1/6个不同物理selection，不能把120/720 flashes或90/180次抽组当作独立decision证据。旧分层分析显示BNCI贡献了负交互点估计的81.05%，这是加权分解，不是因果责任比例。输入边缘、共同参考、相同test cohort若在A/Q四臂共享，可能影响绝对性能和模型敏感性，但不能单凭共同存在证明它们造成A×Q交互。

应并行问三个问题：

1. 输入/估计量是否相同，现有分数能支持哪些比较？
2. 在相同输入上，成熟的结构化判别方法能否比现有小网络更有效利用K个trial？
3. decision与跨库学习分别获得了多少可识别信息，当前目标是否约束了模型实际使用的自由度？

`EA可行→再查prior/decision`不是必要科研依赖。成熟ERP基线、decision信息量和mask/BN风险可以独立前进。EA在完整空间有价值，在当前十二源交集仅一维时不能成为全局必经路线。

## 2. 实际纵向链及新增发现

当前源码定位（与JSON里的文件SHA256一起使用）：

- `src/baselines/deep.py:1169`：source固定更新训练；`:1220` risk mass/importance；`:1305` 同一forward的CE/aux求和。
- `src/models/n2p3net.py:1130`：实测trace打包BN；`:1166` 卷积到post-MST/residual；`:1230` 附带features的readout。
- `src/transfer/source_objective.py:281`：实际feature loss；`:308` L2 normalization；`:353` SupCon；`:380` conditional MMD。
- `src/transfer/subject_adapter.py:306`：target masked statistics；`:437` calibration BN；`:915` target/source input norm；`:1697` trial/完整decision联合workset。
- `src/models/parameter_groups.py:37`：source参数坐标prior；`src/transfer/decision_loss.py:692`：sum-membership CE。
- `src/transfer/recipes.py:197`：当前Q0/Q1 recipe；`src/baselines/riemann.py:126`、`:133`、`:140`：xDAWN、裸covariance、tangent；`src/transfer/binary.py:464`、`:469`：20Hz波形与现SLDA。

### 2.1 当前实现本来已很小，不能套用“大模型过拟合”解释

按当前 `N2P3ArchitectureConfig.unified()`、六导179点实例化：1279参数，head输入176维；spatial96、temporal696、readout354、normalization56、feature_residual77。这个数字是当前构造器探针，不代替历史checkpoint结构核验。低容量新方案必须靠正确的噪声模型、结构先验与稳定估计取得优势，不能只说“现有网络参数太多”。

### 2.2 源风险、观测BN、角度对齐是三个不同的加权对象

`baselines.deep._fit_fixed_updates` 对CE使用显式目标risk mass和采样importance correction；input normalization使用 `observed_source_statistics`。当前记录已声明importance correction不等于BN上下文等权。

更具体的未闭环性质来自 `N2P3Net._observed_temporal_features`：它把每条实测electrode trace打包后进入普通BN。若每域抽相同trial数，六导域的BN样本质量是单导域的六倍；源CE即便域等权也不会改变这一点。

独立实际代码反例：两域各两个trial，一域六导全为+1，一域只一导为−1，identity temporal kernel。域等权均值为0，实际BN running_mean为0.714285731，即5/7；BN质量为6/7与1/7。不是计算错误，且更多实测导也确实包含更多观测；它证明“已域平衡”不能同时覆盖CE、input norm和所有BN统计。是否有害必须用实际域×观测支持分布和源内留出验证。

`SourceObjective.loss` 又把每个实际post-residual feature `h` 投影为 `z=h/||h||`，SupCon与MMD只约束z，head仍使用未归一化h。实际反例把A域h乘.25、B域h乘4：辅助损失变化严格0，固定head logit最大变化12.89。径向梯度内积最大1.46e−8。因此辅助loss小不保证实际decision function域不变。

公式上 `∂z/∂h=(I−zzᵀ)/||h||`，径向自由度被留出，角度梯度也受feature norm缩放。这不是证明A1失败；它否证“共同forward tensor就已经约束同一个决策表征”的过强解释。更不能直接加一项norm penalty，ERP幅度可能是真实信号，统一半径可能消灭它。

源端可研究的三个竞争解释必须分开：

| 假设 | 目前已知 | 可推翻条件 |
|---|---|---|
| 辅助角度对齐留下readout可用的域半径捷径 | 实际loss反例成立；尚未检查实际训练h | 源内留出h半径不携带域/观测信息，或其信息不影响label risk |
| 类内真实ERP时延/形态不同，二元标签对齐过强 | SOA/参考/支持不同，目标只按y对齐 | 在保留类别充分统计的变换下，强对齐仍改善新域；或真实类内形态差异不影响risk |
| mask与BN使“域”等同硬件支持 | BN计权反例；空间卷积仍固定槽位 | 匹配mask/context后源内留出风险及A1交互不变 |

成熟修正优先是明确源内评估与风险单位、充分的强锚点，不是堆第三个辅助loss。源域等权并不保证所有域正迁移；可用source-only留域评估识别source negative transfer，但不能用最终target test选择库/权重。

### 2.3 当前名为xDAWN Riemann的路径缺少成熟ERP协方差的关键信息

`src/baselines/riemann.py:120–145` 是 `Xdawn.fit_transform→Covariances(Xd)→TangentSpace→LDA`。它不是pyRiemann的 `XdawnCovariances`：后者把训练范围内的类别ERP原型与trial共同构造协方差，保留原型—trial交叉块。

对任何冻结线性空间滤波W，裸协方差满足 `Cov(WX)=Cov(−WX)`，并对epoch时间样本的共同置换不变。ERP符号和与刺激锁定的时间位置可以因此丢失。当前类的独立运行确认：paired ±ERP数据上，当前路径AUC=.5、正负score完全相等、任意time permutation后score完全不变；官方 `XdawnCovariances` 原型增广与waveform shrinkage LDA均为1.0。这是能力反例；不是公共数据收益，不证明裸covariance对所有真实P300都失败。

`Cov([P; X])`包含 `PXᵀ`，极性翻转会改变交叉项，能保留模板对齐证据。P、xDAWN、tangent reference、分类器均必须在同一合法fit范围拟合。小K时原型噪声与类别稀少仍可能使它差于LDA；nfilter也不能默认超出实测秩。

官方文档已HTTP200核实，原文与请求记录见 [pyriemann_docs.json](pyriemann_docs.json)：[XdawnCovariances](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.estimation.XdawnCovariances.html)。文档明确它是ERP专用协方差、prototypes与signals的联合路线；`applyfilters=False`有不同跨人泛化/维度取舍，不能当作无需验证的更好设置。

### 2.4 Q1必须由独立decision信息量裁决容量

当前 `SubjectAdapter._joint_training_batch` 为每个trial batch再抽一个合法完整decision，拼接workset，冻结calibration BN，联合回传CE、source prior和sum-membership decision CE。G=1时所有decision更新一直重复同一目标状态；改抽样遍历不能解决它。

对6×6 row/column单decision的120×36 membership矩阵M，rank(M)=11；候选softmax Fisher推回evidence域 `M(diag(p)−ppᵀ)Mᵀ` 在零evidence时rank=10。120次flash不是120个独立候选标签，也不是35个无约束候选对比。这个rank不包含trial CE监督，不是说模型总信息只有10维，也不是可以直接用rank=10设置样本量门槛。

因此不推荐用“一律γ缩小66倍”作为主线。应把是否允许decision改变backbone，交给预声明的信息支持规则：不同物理selection数、目标覆盖、不同录制状态、独立验证单位、拟合自由度。G=1时不具备target内组级验证或稳定估计decision噪声的能力；推荐默认trial-only适配，并将源内冻结的decision规则用于推理。若有足够组证据，可比较固定backbone的低自由度判别适配与full joint，不能因一个任意G阈值就宣称充分。

只在推理端调整正temperature，对完整均衡曝光的BNCI候选ranking不产生变化；共同logit intercept同样加到所有候选相同常数。温度可以改善概率校准，但不能解释为修复hit。训练temperature改变梯度，是另外一种处理。

### 2.5 相关噪声建模有潜力，但“相关”本身不足以改善decision ranking

在一个明确候选生成模型 `e|c ~ N(a m_c, Σ)` 下，

`s_c=a m_cᵀΣ⁻¹e − (a²/2)m_cᵀΣ⁻¹m_c + log π_c`。

当Σ=σ²I、各候选曝光相等时，退化为现有sum evidence（加共同常数/缩放）。若重复间只是同一compound-symmetry相关，`Σ⁻¹1`与1共线，同样不能提高ranking。探针20,000序列验证ranking相同率=1。

只有可估计的不等可靠度或候选相关结构，GLS才可能增加信息。构造最后一遍噪声方差为36、前四遍为1时，oracle covariance的hit从.25435到.56625。这仅是信息价值反例；Σ已知，不能说已获得新算法。BNCI G=1绝不能拟合120×120 Σ。合法候选应是源内冻结/低维强收缩噪声模型，参数只能在授权calibration内更新；重复、时间/质量状态作为协变量，不允许使用未知target身份或test拟合权重。

## 3. 不限于现架构的优化候选

| 路线 | 成熟基础与当前缺口 | source拟合 | 本人拟合 | 可读test状态 | 首要否证 |
|---|---|---|---|---|---|
| 原CE/full-fine、scratch、现waveform SLDA | 已有强锚点；固定输入/剂量 | CE source | 固定授权C；同K | 冻结preprocess/norm/weights | 新方案不能稳定超它就不升级 |
| ERP prototype covariance + shrinkage head | pyRiemann成熟实现；当前类未增广原型 | 可无source；迁移版另有source原型合同 | xDAWN、P、tangent、LDA只fit C | 全部冻结 | 极性、时序、秩、少正例、真实后续风险 |
| 标准ToeplitzLDA | 成熟算法与代码；官方要求baseline disabled、等宽bins、channel-prime | target-only可先无source | 在新声明无baseline raw视图C fit | 同视图冻结 | baseline/边缘非平稳；不能给旧baseline cache硬套 |
| 传播实际线性算子的结构协方差LDA | 标准协方差传播+Toeplitz的项目组合研究，非已有成熟ERP产品 | 合法source/无baseline中间量估结构；独立留域选方 | C估残差/均值/收缩 | 冻结协方差与输入算子 | 结构误差、未观测协方差不可识别、PSD与rank |
| 保留电极身份的共享时间基+本人判别 | 低秩线性ERP/模板/收缩成熟；跨缺导组合为研究方案 | 等域/人提取binary ERP时间基B，不对白化六槽补零 | 用实测导`X B` fit shrinkage判别；必要时只fit少量延迟/增益 | B和本人头冻结 | 类形态不在B空间、参考投影/时延、source基不如固定通用基 |
| 独立decision支持充分时的低维联合判别 | 将decision监督限制在可估自由度 | 所需噪声模型与规则源内冻结 | 校准有独立组才允许拟合；G=1 trial-only | 完整membership+冻结规则 | 同一selection多前缀不能伪增样本；hit/NLL均无收益则放弃 |

### 3.1 对已成熟方法踏实补齐，不让EA挡住强基线

首先把现waveform SLDA、ERP prototype covariance和ToeplitzLDA置于同等地位，不能只给神经模型充分工程支持。比较前记录每方法实际支持的导、feature维度、原型正例数、协方差rank/condition、独立拟合自由度、相同校准/test身份。动态缺导不能先补零再当完整covariance；先在每个目标自身的合法实测布局做target-only，完全不要求十二源交集为六维。

Toeplitz官方前提见文献分支 [toeplitz_code_full.md](../literature/toeplitz_code_full.md)，论文DOI [10.1088/1741-2552/ac9c98](https://doi.org/10.1088/1741-2552/ac9c98)。这不是“替换sklearn一行”即可完成：当前固定baseline改变噪声协方差、当前特征flatten顺序还需与channel-prime核验，新无baseline是新输入身份，源/目标都要匹配。若只为了保持旧输入，可把其作为研究候选而非伪称标准复现。

### 3.2 创新候选A：把输入算子纳入判别噪声模型

有限段滤波/重采样/baseline会改变噪声结构，盲目追求在输出域“统一stationary covariance”不合理。若X_out=A X_raw，噪声协方差应是 `C_out=A C_raw Aᵀ`。对baseline `P=I−1bᵀ`，

`PΣPᵀ=Σ−1bᵀΣ−Σb1ᵀ+(bᵀΣb)11ᵀ`。

相对Σ的修正位于span(1,Σb)，rank≤2；baseline不是无害的均值预处理。协方差要在实际可识别输出子空间内正定化，不能把被投影掉的时间常数维硬解释成生理信息。

这是科研组合，不声称已有论文直接实现本仓库的算子链。它必须从合法raw或无baseline中间量估C_raw；仅保存PX时原Σ不能唯一恢复。滤波边缘、真实瞬态和脑活动非平稳仍可能违背stationary模型。先用文献分支的baseline/PSD反例、已知ERP与噪声注入否证，再做预声明真实开发比较；不为了让理论通过而按test挑净段。

### 3.3 创新候选B：跨库迁移时间结构，空间结构留给真实观测

避免十二源六维公共白化的瓶颈：不强迫所有来源进入相同完整空间，先学习低维时间基 `B∈R^(T×r)`。每个源/人只使用实际导上的类均值差与残差，按声明domain/participant权重估B；缺失导不生成ERP、不填零进入covariance。目标特征为每个实测导 `Z=X B`，由本人calibration拟合收缩LDA/小线性头，r由source-only留域选择。

时间基迁移不等于参考自动等价；参考是空间线性算子，只有保留了足够时间span且无角色混淆时才有希望共享B。若进一步用源先验收缩target均值，只允许在有明确参考投影/同电极角色的坐标上使用；未知空间部分保留不确定性。极少校准时可冻结B与噪声结构，仅fit均值/小头；应与无source固定平滑基、现source_feature_lda、full-fine同时比较，防止把“低维”当新颖性。

候选时延不能在unknown test上按真值对齐。可用source-only给出的有限shift集合，对target calibration选全人一个delay或在明确生成模型中边缘化；禁止逐trial自由warping把噪声最大峰变ERP。真实延迟增益仅在全体配对后续风险成立后认可。

更激进的latent electrode observation model可写作 `X_d=H_d Z+ε`，在各自观测子空间估likelihood；但H_d、参考、噪声、latent basis同时未知存在旋转/scale不可识别。当前不应越过简单时间基路线直接开发大latent模型。它需要真实坐标和外部可识别约束，不可用网络学习填补“未知通道”的事实。

## 4. 接线必须贯穿source/calibration/save/infer/evaluate

建议沿现有 `Baseline.fit/predict_logit` 和 `CalibratedPredictor` 深化，而非在每个CLI复制算法条件。这个真实Seam已有神经、waveform两个Adapter，补充ERPcov/结构协方差Adapter才有Leverage。Interface除形状还必须明确物理/feature单位、observed支持、fit权限、持久化、错误与缺decision输出。

| Module拥有的事实 | 源码落点/消费者 | 必须一并实现 |
|---|---|---|
| 真实输入算子与允许观测 | data.preprocess/epochs/source_records | 新无baseline或operator-aware身份；旧cache不改名冒充；raw支持与test ledger |
| 标准化、mask、covariance、feature map状态 | 新/现baselines Adapter；transfer.calibration_candidates | 只有fit(scope)可变；test只transform；ERP模板/时间基/协方差矩阵及顺序保存 |
| Source recipe与source-only模型选择 | train.factory、research.source_matrix、source execution | target-study完全排除；B/λ/结构rank/噪声超参只源内选；每family清晰checkpoint身份 |
| 本人拟合容量和风险单位 | TargetRecipe、SubjectAdapter、calibration_decision_groups | G/target coverage/physical unit账本；是否允许decision更新参数预声明；不以AUC选择gate |
| 持久化个人推理 | transfer.calibrated、checkpoint | family dispatch不伪装N2P3 state；reload逐score相等；test不refitBN/P/TS/Σ |
| 决策估计量与统计 | membership_decision_scores、outcomes、stats | operational requested cohort主读数；LLR/score单位、prior和temperature身份；支持不全不删分母 |

Input/fit/model state应由各自owner统一封装；不能新建一个什么都知道的大ExperimentConfig，同时让调用者还要自己记忆模板、BN、reference与loss先后。也不能把“可序列化EA artifact”误当完整模型路线，成熟classifier先提供端到端Adapter，再宣称用户可用。

## 5. 推荐顺序与否证门槛

1. **当天可完成的无新训练取证**：核对实际执行selected prior λ、source h的norm/angle与mask关系、CE/aux分项梯度及固定optimizer状态的一步变化；从合法保存state取证，不用重跑几十个loss系数组合。lambda实际为0的cell不能归因prior。若state缺失，就明确无法确认，不用默认值补。
2. **先补齐成熟端到端对照**：现SLDA、ERP增广covariance、标准无baseline Toeplitz；每个都有save/reload及真正后续predict路径。输入差异是独立因素：相同input比较模型，不同input比较整条route，不能把二者合成单模块因果贡献。
3. **并行做两个小研究原型**：operator传播的结构协方差；observed电极上的共享时间基+本人收缩判别。先否证单位、时延、rank/PSD、mask、reference、baseline/finite support，再按source-only冻结；不得只在污染理想高斯合成数据上成功就晋升。
4. **Q1重建成容量研究**：trial-only vs 冻结trunk低维decision vs full joint；G/目标覆盖/更新剂量匹配。G=1独立decision信息不够时不要自动开full joint。γ规范化最多是一个次级变量，不能替代这个实验问题。
5. **公开开发验收**：同真实raw、same K/requested trials、test identity、相同超参选择机会；四库等权主估计、逐人差、negative-transfer率、失败/覆盖、低K与高K分开。沿用活动路线mean ΔAUC≥.01门槛时须预声明并按共同合法目标计算；BNCI另给operational hit@all及decision稳定性。一个K或库受益不能靠后验拼winner部署。
6. **停止规则**：若成熟低容量路线与新prototype都不比full-fine/SLDA稳定，保留简单强基线，继续采集可靠新成人calibration/test以定位上限；不继续叠对齐损失“消化”负结果。若仅oracle GLS有益但合法C无法估可靠度，路线不晋升；若prototype结构误差主导，即使速度/condition很好也不晋升。

四库旧test已经参与大量开发，本轮新的真实比较仍是public development。成人BrainSync产品hit@all需独立新采集，不能由这里的合成AUC或公共row/column hit替代。

## 6. 本分支实际完成与未完成

- 完成：当前source objective、source CE/normalizer、masked temporal BN、parameter prior、target joint loop、decision loss、经典baseline和recipe链只读核验。
- 完成：5组独立能力/反例脚本、当前模型容量测量、官方pyRiemann文档联网核验；使用项目 `.venv`、2个CPU线程，数秒规模，没有公共数据training。
- 验证：研究脚本Ruff通过；定向`git diff --check`通过。只验证该研究产物，不把它写成生产全套测试或新模型收益。
- 反例中的当前裸covarianceLDA触发一次 `explained_variance_ratio_` 0/0 RuntimeWarning，因为构造数据类间covariance完全相同；保存score有限且AUC=.5，这正是反例，未隐藏warning。
- 未完成：实际历史每折selected prior λ与分项梯度（本分支未取得模型轨迹）；新的公共数据效能矩阵；任何默认算法升级。
- 学术支持：ToeplitzLDA为既有方法；prototype ERPcovariance为成熟库能力；operator传播/低维共享时间基/GLS在本仓库中的完整路线均是待验证科研组合，不能写成已经解决问题。

## 附录A：重新查找十二库实际λ，确认缺的是哪份产物

本轮没有直接沿用09-29的“λ缺失”说法。重新检查了 `doc/assets/unified12_results_20260918`、`experiments/runs/unified12_20260918/results` 全部9份JSON，逐字段递归查找prior/lambda/selected_weight；**0个字段命中**。`experiments/runs/unified12_20260918/analysis` 当前不存在。检索记录、9文件字节数与SHA256见 [prior_provenance_search.json](../../../../experiments/runs/system_research_20261004/learning/prior_provenance_search.json)。

追到本次执行分析器 `experiments/runs/unified12_20260918/analyze_unified_results.py:37`：它从每个实际cell的`recipe`读取`source_prior_weight`，与source_arm、outer、family_exclusion、原record路径和cell index一起保存，写出`analysis/cells.csv`、`analysis/normalized_cells.json`。因此这项事实**不是没有记录**，也不需要重新训练或载入大权重；是这两份逐cell分析产物未出现在本地目录。当前`integrity_inventory.json`只有48份云端record的路径、SHA和cell数量，不能反解λ。

含隐藏/忽略文件的文件名检索覆盖`experiments/runs`、`frozen`、`cloud_inputs`、`assets`、`tmp`，未找到`verified_execution`或十二库的逐cell文件；只发现09-16四源run的`normalized_cells.json/cells.csv`。09-16中确有λ=.1记录，**不能移植为09-18十二源执行事实**。文本中`packed_prior=.001`来自性能计时，不是选中的λ。

十二库收据仍指向云端`/root/autodl-fs/n2p3-runs/unified12_20260918/verified_execution.tar.gz`，11,990,531,333字节，SHA256 `db3456fada03f6cc7177c225b822c4884aef4e2204c646218c029ef0bd99c790`。本分支没有连接云端，也没有检查其他盘、任意重命名压缩包的全部内容；因此结论是**当前已检查本地输出无法恢复逐fold A0/A1真实λ**，不是声明云端已丢失。

最小补证应优先只取云端`analysis/normalized_cells.json`与`matrix/target_jobs.json`，按outer×source_arm×Q1检查λ唯一性，再与job绑定的`source_prior_selection/selection.json`核对selected_weight及selection digest。须核对每fold实际A0/A1是否共享选择，不从源码默认推定它们的数值。拿到这些小文件后即可判断哪些负交互cell中prior实际上被关闭；不能在此之前将全部Q1差异归因prior。
