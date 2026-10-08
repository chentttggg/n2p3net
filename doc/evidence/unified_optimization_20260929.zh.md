# 十二库 A1/Q1 负交互：分层复算、机制反例与优化规格

日期：2026-09-29。交付类型：科研方案与可复现分析；未实施新训练算法、未启动训练、未消费成人确认集。

同日后续[全链深审](deep_chain_audit_20260929.zh.md)进一步发现当前有限raw边缘保真问题及EA的
默认V²尺度、静态mask漏检、变换单位/fit范围问题。后续执行先闭环这些输入与比较合同，
再解释窗口或模型收益；不能沿用“EA CLI已完整拒绝所有缺导”的笼统判断。本文方案未启动训练。

## 1. 科研裁决

**下一轮优先采用 CE 源训练与普通 full-fine 作为锚点，先解决真实时间窗与实测导联空间的可比性。现有十二库不能直接实施标准六导EA；几何路线须先通过空间准入。Q1 的后续改造要把 prior 与 decision 分开，不能把它们同时判为无效，也不应直接增加新的辅助损失。**

这延续[执行计划](../roadmap.zh.md)的窗口优先与域变换路线；不自动恢复09-20已停止的 SupCon/MMD/prior/decision 大网格。完整 Q1 仍不进入产品默认，但 prior-only 的小幅正效应值得保留为后续待复核假设。

本轮联合使用 astra-research-workflow、eeg-p300-research、scientific-critical-thinking、experimental-design、research、research-architecture-decision；并行分支使用 statistical-analysis 和 eeg-bci-transfer-learning。旧技能中的固定模型排名、R5主指标和历史目录均以当前六份指南为准。

## 2. 先修正问题的定义

十二库是**源池**，常规外部目标只有 BI2014a、BI2015a、BNCI2014-008、Huebner2017 四库。每个普通折排除目标库；另有 BI 联合排除敏感性折。单配对训练 seed=20260918；这不是十二个独立测试域或成人 BrainSync 确认。

- A1=CE+0.05 SupCon+0.05 class-conditional MMD，是联合源包。
- 三个 binary-only 目标的 Q1=trial CE+source parameter prior。
- BNCI 的 Q1 才是 trial CE+prior+membership decision CE。
- 交互定义为 `I=(A1Q1−A1Q0)−(A0Q1−A0Q0)`。它是冻结处理的差中之差，既不是训练梯度冲突的实测量，也不能识别 SupCon、MMD、prior、decision 的单项因果贡献。

本轮从已有 participant-level AUC 记录重算，按库内参与者配对重采样20,000次、库等权；analysis seed=20260929。以下新增区间均为探索性，条件于固定模型、固定源选择和固定库，没有训练随机性或跨新库泛化保证。

| K=200范围 | 人数 | A0Q0 | A0Q1 | A1Q0 | A1Q1 | A×Q交互 |
|---|---:|---:|---:|---:|---:|---:|
| 常规四库 | 123 | .69949 | .69707 | .70033 | .68948 | −.00844 |
| binary-only三库 | 115 | .70941 | .71214 | .70872 | .70932 | −.00213 |
| BNCI | 8 | .66972 | .65187 | .67516 | .62995 | −.02735 |

关键解读：

1. 四库 A1@Q0 的 ΔAUC 仅 +0.00084，95%区间[−0.00582,+0.00763]；没有稳定源增强收益。
2. **三库 prior-only Q1@A0 是 +0.002728**，区间[+0.002041,+0.003448]；K800也是 +0.002753。效应很小，且只是一组冻结训练的结果，但它直接反驳“prior本身已被证实失效”。
3. K200四库负交互区间[−0.01229,−0.00467]。按 `I四库=¾I三库+¼IBNCI`，BNCI的加权项占负交互点估计81.05%。这只是算术分解，**不是decision loss解释了81%的因果责任**。
4. BNCI K800 operational `hit@all`：A0Q0=.71552、A1Q1=.50862，差−.20690，参与者配对区间[−.28879,−.11638]。其唯一完整Q1包明显值得优先诊断；仍无法仅凭该对照分离 prior 与 decision。
5. K800 interaction原10,000次bootstrap区间上端+0.000064，本次20,000次为−0.000024，不能利用Monte Carlo过零变化宣称稳健显著。共同72人的整包效应从K200到K800变化−.00590，区间[−.01892,+.00776]，也不能确认“校准越多越有害”。

BNCI K200/K800每人只有1/6个不同物理完整selection；它们甚至不保证统计独立。90/180次抽组或切出多个R前缀不会创造独立目标状态。测试每人29个物理selection、28个完整，coverage=224/232=.96552；所有10次重复进入 `hit@all`。

原[十二库结果](unified12_results_20260918.zh.md)的BNCI两类scratch行标签颠倒，已按原JSON更正：CE scratch=.38793/.58621，decision scratch=.34052/.50000。原`source_jobs=48`字段实际来自`len(target_jobs)`，是48个目标分片job，不能理解为48次源拟合。本轮没有修改历史JSON或原预测。

完整结果、输入SHA256、共同cohort和统计方法见[分层复算](../assets/unified_optimization_20260929/evidence_audit.md)及[JSON](../assets/unified_optimization_20260929/evidence_audit.json)。
复现脚本为[recompute_evidence.py](../assets/unified_optimization_20260929/recompute_evidence.py)；用项目venv加`--check`已逐项复现全部数值，未改写结果文件。

## 3. 机制排序：事实、解释与可推翻条件

| 机制 | 当前证据 | 可证伪预测 | 对方案的影响 |
|---|---|---|---|
| 决策监督只有少量物理selection | 直接来自授权组账本；K200=1、K800=6 | 增加不同selection并固定其余条件后，decision泛化仍不改善，则“监督量不足”不是充分解释 | 不把flash数或R前缀当独立decision；产品采集报告G及目标覆盖 |
| trial与decision风险尺度不一致 | 精确梯度可算；当前γ=1 | 规范化后实际参数更新/后续风险不变，或更差，则尺度方案无价值 | 只作后续一个独立变量，不承诺收益 |
| A1与prior坐标共同变化 | prior的尺度使用实际源权重RMS；同λ并不同惩罚矩阵 | λ实际为0则该fit不能归因prior；固定metric后交互不变则该解释减弱 | 先找回每折实际λ和各role度量，不能从默认值猜 |
| 强制跨域同类形状接近 | SupCon/MMD在L2归一化时序特征上施加；CE仍保留原特征 | 类条件差异若只是nuisance，对齐也可能有益；没有当前坍缩证据 | 不把源包失败解释成“所有对齐无效” |
| 辅助项有效域权重不均 | active-anchor/active-cell均值不等于请求域等权；CE采样权重不自动传给辅助项 | 若各域类的实际风险质量和梯度贡献接近声明目标，则解释减弱 | 先测真实贡献，不再盲目扩源或重新加域平衡模块 |
| 参数先验不能固定输入/BN变化后的函数 | 参数未动时prior=0；target-prefix/BN已换统计 | 不保证函数保持不等于无正则价值；三库小正效应是反证 | 保留prior研究资格，不新增蒸馏来替代未完成归因 |

八个核心机制/runner文件已与执行v4源码逐字节核对；v4压缩包SHA256为`c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24`。首折复用v2保留其独立执行身份。当前dirty HEAD `f1a303c42c68173e8b1452ceb26bd84923b8a6aa` 不能代替全部执行源码身份。

额外设计限制：source exposure和λ来自A1参考源的共享选择策略；比较估计的是该冻结策略下的包效应，不是每个方法各自最优配置。λ候选为{0,.001,.01,.1}，但本地拉回摘要缺实际每折selected_weight，本轮没有确认其具体值。

逐项代码证据、行号与竞争解释见[机制审查](../assets/unified_optimization_20260929/mechanism_audit.md)。其中组件诊断建议只作为后续候选，当前主线仍按下一节执行。

## 4. 优化主线：先让同一生理信息进入同一坐标

### 4.1 P0：时间窗与输入支持先冻结

继续600/800/1200 ms真实raw重建，主比较600 vs 1200，800为敏感性。频带、基线、参考、实测通道、合法fit/test范围和预算固定；按最长实际算子支持purge。裁剪已FFT重采样cache仅为特征截取诊断，不能替代真正的短窗。

LDA/target-only先提供输入敏感性对照。`full_unfold`窗口改变head维度时，必须记录自由度并重训相应head；不能把参数量变化隐去。沿用[输入审计](input_contract_review_20260922.zh.md)的标签和3×3边界，不改写BI flash membership。

### 4.2 P1：校准统计冻结的双侧EA，保持CE目标

**先做空间可行性准入，EA目前只是条件候选。** 十二库现有输入使用实测mask，并不都有完整六导。
[源接入证据](domain_transfer_major_20260920.zh.md)记录BrainForm、Cattan、BI2012不同缺导；GTN原生仅Fz/Cz/Pz，
在Cz差分后也不具备全六导。当前EA cache入口拒绝非完整观测，不能直接套在这些六槽零填充输入上。
六槽顺序为Fz/P3/Pz/P4/PO8/Oz（各自减Cz）：BrainForm探针mask=`101011`，BI2012=`011100`，
二者交集已经仅剩Pz−Cz。完整十二库和四个普通留库折均包含二者，因此全源共同实测空间
**最多一维**；这是已核实探针给出的上界，是否还有逐记录Pz缺测需全量prepared核验。
进一步检查交付manifest：BrainForm的P02/20241021_120406声明同时排除Fz、Pz，
按当前loader合同将只余PO8/Oz；GTN仅Fz/Pz，两者共同空间为空。BrainForm167条准备成功的摘要已核，
但云端最终逐行prepared mask未拉回，故“空交集”标为manifest与源码共同推导，
不能冒称本轮已遍历全部最终mask。相关普通/BI家族/部署折均保留GTN与BrainForm。
故“维持原六导/原网络语义直接跑标准EA”目前不准入。拟合前仍须输出逐fold/source/target的实测集合、共同交集和秩。
若共同子空间过小，保留全部请求源并报告route不准入，不偷偷删库；若明确另建共同子空间方案，
plain/EA/scratch/LDA必须同子空间、同自由度重训，保留原六导强锚点。按每个mask分别白化再回填
不是标准共同坐标EA，其新增语义需要独立研究，不能拿来绕过该门槛。

令合法fit单位d的EEG为 `X_i∈R^(C×T)`，先限定共同实测导联与同一物理参考：

\[
R_d=\frac1{n_d}\sum_{i\in\mathcal C_d}\frac{X_iX_i^\top}{T},
\qquad R_{d,\rho}=(1-\rho)R_d+\rho\frac{\operatorname{tr}R_d}{C}I,
\qquad \widetilde X_i=R_{d,\rho}^{-1/2}X_i.
\]

这里是平均trial二阶矩，不能静默改成每trial单位trace协方差或另加时间去均值。现有EA模块已有shrinkage和正定门禁；首臂沿用冻结配置，不同时优化ρ、BN、学习率和source pool。若后续选ρ，只能在源内、匹配目标K及时间支持的伪校准/后续query上选择；每个伪query同样禁止参与协方差拟合。绝对正则化floor须与输入单位核对，不能假设V与µV自动等价。

完整路线：

1. 源端按预声明fit单位（例如participant×recording，不能临时按结果合并）估计EA，仅消费其合法源训练行；用对齐后的源输入重新拟合CE编码器及normalizer/BN。
2. 目标K>0时只用 `C_K` EEG拟合 `W_T`，标签只供原来的full-fine；训练和未来推理共用已冻结 `W_T`。不同K各自建state，禁止借用K800协方差处理K200。
3. K0若需要目标无标签buffer，必须显式预算、范围、时长和状态身份，记作zero-label；若buffer就是被评分test，则另记transductive。strict zero-shot不读取目标EEG，也不能假装获得了 `W_T`。
4. 完整程序只提供一个预声明route；不根据某目标库/人的test成绩切回full-fine或LDA。失败保留分母。

**关键修正：不能同时要求“标准双侧EA”与“沿用未对齐源模型的同一checkpoint”。** 标准EA可固定架构、原始行、seed、CE目标与训练预算，但源权重、normalizer、BN和cache必须绑定该route。仅将目标送入白化矩阵会改变旧模型输入坐标，即使shape完全相同也不等价。

如果研究问题必须保持旧source function，可另定义 `X_T'=R_S^(1/2)R_T^(-1/2)X_T` 的二阶矩transport；它需要冻结多源参考的聚合权重、统计坐标与算子顺序，且不是当前EA函数，暂不并入首臂。

**EA不是充分解。** 若A同维满秩、R_S正定、`R_T=A R_S Aᵀ`，且ρ=0、正则化floor未激活，理想白化后仍留

\[
O=R_T^{-1/2}AR_S^{1/2},\qquad OO^\top=I,
\]

即无法仅从二阶矩确定的正交变换。源/目标协方差同为I但标签方向旋转时，EA可完全无效。对齐残差变小只说明几何操作生效，不能代替ΔAUC或hit验收。

### 4.3 P2：有证据再升级类条件几何

仅在合法空间中的EA通过既定开发门槛后，才考虑P300 RPA或论文一致的joint feature alignment。
若当前十二库的空间门槛失败，记录该路线不准入；不为继续升级而把一维Pz试验冒称原六导方案。
原始P300 RPA的ERP covariance模板及classwise旋转使用目标校准标签；完整路线应作为few-shot对照。无标签recenter/stretch子集需另命名，并核验covariance构造本身是否也依赖目标标签。

GALIA本轮核实摘要与历史全文审计，尚不能将其所有版本叫zero-label。官方摘要的 `2.12±1.88%` 是classification accuracy改善，**不能换算为本项目+0.01–0.03 ROC-AUC预测**。当前`domain_alignment.py`的共享Q/对角白化也不能冒称该论文算法。

波形收缩LDA继续作为必备同输入/同K对照；xDAWN+Riemannian是独立完整预测器，比较时明确它同时改变特征和分类器，不能称为只替换N2P3输入state。

## 5. Q1后续重设计：监督匹配、固定尺度、单因素归因

这部分是几何路线过门槛后的备选，不是本轮新增训练任务。

### 5.1 先固定Q0，保留prior-only的研究位置

主比较仍用Q0，防止输入路线与新的正则/损失同时改变。要研究Q1时，先固定**一个source和一个输入route**，拆成 `CE / CE+prior / CE+decision / CE+prior+decision`。三库只有前两者有物理含义；BNCI/BrainSync才可定义后两者。每个有decision的处理配CE-only与decision-matched scratch，不能继续混为一个“scratch”。

prior度量若将来调整，先记录各role的中心、尺度、梯度与实际λ，固定fan-in度量或只约束部分参数均只是单独待检验假设。没有证据支持直接增加LoRA、模型容量或教师蒸馏；当前共同网络只有1279参数，已知强full-fine也不能被默认的受限适配替换。

### 5.2 把decision风险变成有明确单位的加项

设 `e_i` 为trial log-odds，`M`为真实物理membership，V为冻结候选数：

\[
L_T=\frac{\sum_iw_i\operatorname{BCE}(e_i,y_i)}{\sum_iw_i},\quad
L_D=-\log\operatorname{softmax}(M^\top e/T)_{c^*},
\]
\[
\nabla_eL_T=\frac{w\odot(\sigma(e)-y)}{\sum_iw_i},\qquad
\nabla_eL_D=\frac{M\operatorname{softmax}(M^\top e/T)-M_{:c^*}}T.
\]

后续最小候选复用09-16已有提案：

\[
a_D=\frac{\|\nabla_eL_{T,D}(0)\|_2}{\|\nabla_eL_D(0)\|_2},\qquad
L=L_T+\bar\gamma\,\mathbb E_g[a_gL_{D,g}]+\lambda\Omega.
\]

T=1、完整对称排程、类平衡trial风险下，BNCI六行六列R10得到 `a_D=.015`，九选一direct R5为 `.05625`。从物理group计算，不能按数据集名硬编码；不完整或不可辨识组仍按合同拒绝。该系数只规范化**同一组、零logit处**的梯度尺度，不是最优权重，也不保证泛化改善。

同时记录 `cos(g_trial,g_decision)`、各role参数梯度、clone同一Adam状态的一步更新及独立后续风险。不能把66.67倍logit梯度说成66.67倍参数步长；不能用除loss替代score除R；不能用温度降低NLL后声称hit提高。

PCGrad/CAGrad留作有实际冲突证据后的单独候选。统计A×Q负交互发生于先源后目标的阶段组合，不代表两梯度在同一步相遇。即使同一步原始梯度内积非负，Adam各坐标预条件也可使primary risk的一阶变化变坏，本轮已构造反例。

### 5.3 完整物理decision是预算与统计单位

报告 `B=(K,N_+,G,R,目标覆盖,录制时长,phase)`。一个selection的所有前缀总风险质量固定为1，统计按原selection/participant聚类；不为凑G借用校准范围外flash。当前BNCI的1/6组不足以支撑稳定、丰富的group内选方，不通过拆R前缀制造验证样本。

若未来比较固定G的校准，应另建manifest与预算轴，与原same-K分开。BNCI原protocol的1800-flash校准/2400-flash测试也须另建输入身份；不能重标当前单录制前缀混合阶段的旧结果。产品主指标始终是本任务定义的 `hit@all`。

## 6. 下一轮最小实验与停止条件

| 阶段 | 唯一主要变化 | 对照及指标 | 前置/通过条件 |
|---|---|---|---|
| W | 真实时间窗600/800/1200 | 同输入预算LDA与target-only；600 vs 1200为主 | raw支持、标签、隔离及head自由度入账；选择后冻结新合同 |
| G1 | plain vs完整双侧EA路线 | strict plain Z0；显式zero-label EA；K200/800 Q0；同K CE scratch/LDA | 逐折共同实测空间/秩先准入；fit_scope和保存重放通过；不混淆strict/zero-label/few-shot |
| G2 | 通过G1后的几何升级 | EA vs合法RPA/JFAL；few-shot和zero-label分栏 | 标签需求和论文实现核对；不将完整classwise方法冒称zero-label |
| Q（后续） | 固定source/route拆prior×decision | 同G/K/工作集、更新和监督匹配scratch | 明确实际λ、完整组预算；先单系数规范化，再考虑其他机制 |

G1沿用活动路线的四库等权、库内participant-macro `mean ΔAUC≥+.01`，对照为plain source zero-shot；它是允许消费额外无标签EEG的route收益，必须同时报该buffer成本。此门槛是开发决策标准，不是文献保证。K200/800 EA+full-fine相对plain full-fine是另一估计量，不能用G1门槛冒充其已获晋升。

route通过后做3–5组预声明配对training seeds，不能反复看同seed直到区间过零。固定研究集可采用保留seed与participant交叉结构的配对bootstrap或层级模型；不能把每个seed×participant当独立人。四库太少，不声称已估计所有未来study分布。主对比和多重性家族拟合前锁定，其余标探索性。decision的failure/tie/abstention保留在operational请求分母；AUC不存在天然的“失败分数”，须在冻结完整cohort上可评分，缺少预测则本轮G1不通过，不静默删除失败参与者后算均值。逐库、p10和最差被试为风险诊断。BI相关联合排除继续保留，K200/K800同时报告exact-K与共同cohort。

收益不达门槛就结束该route，不用目标test挑ρ、源库、阈值或后备算法。成人确认只能用独立后续采集，公共AUC与BNCI36选一不能晋升成人九选一默认。

未来若进入云训练，执行完整已声明矩阵；每次启动/恢复加载performance-optimization与get-available-resources，沿用CPU总线程≥25、GPU worker容量≥4，实测有效并发和利用率，源缓存只读共享并保留已完成产物。本轮没有训练吞吐或新性能成绩。

## 7. 实施切片与联动清单

1. **输入合同切片**：复用`data.contract/preprocess/epochs`和raw scope builder；完成窗口身份、掩码、滤波支持、匹配行与source head兼容。验收为独立raw视图和保存重放。
2. **几何state切片**：复用`data.euclidean_alignment`，由一个owner统一fit/transform/record/reload；补fit row digest、通道顺序/参考、raw支持、算子顺序、K或无标签预算以及source route digest。源与目标state值不同是合法的，核验的是route语义与各自fit授权，不要求state hash相同。
3. **源/个人调用链**：`prepare_euclidean_alignment_cache.py`、source runner、`target_experiments`、`SubjectAdapter.fit/_refit_full_prefix`、`CalibratedPredictor`、checkpoint gate同时接通。当前EA算子存在，但unified目标拟合、个人包与只读推理尚未形成完整EA路线，不能只改cache CLI后宣称完成。
4. **分析切片**：复用固定participant/decision主键与失败账本；分列prior-only和prior+decision、strict/zero-label/few-shot，记录各route配套source模型及预算；同一baseline逻辑别名复用但不增加独立样本数。
5. **后续Q切片**：系数归`membership_decision_loss`或其唯一合同owner；recipe、fit/refit、selection、manifest、checkpoint和分析器同版本迁移。共同工作集与BN冻结策略保持匹配，不在不同CLI各算一次a_D。

EA新增验收须覆盖：共同实测mask；源与目标本来相同的坐标反例；矩阵不交换的算子顺序；相同二阶矩但标签方向旋转；test值/标签变化不改变calibration state；不同K不复用越界state；重载一致；route不兼容checkpoint拒绝。形状不变不等于语义兼容。

## 8. 本轮验证与文献界限

独立脚本[tmp/counterexample_verification_20260929.py](../../tmp/counterexample_verification_20260929.py)通过11组断言，结果见[math_probes.json](../assets/unified_optimization_20260929/math_probes.json)：负交互无需decision或梯度冲突；零参数prior不保证函数保持；两种物理排程的autograd/系数；loss缩放不等于温度；正温度和联合候选置换；Adam预条件反例；双侧EA正增益不变性与旧checkpoint错配；相同协方差的旋转反例；state只读与重载；相同AUC可有不同hit。

项目venv运行`tests/data/test_euclidean_alignment.py`、`tests/train/test_source_objectives.py`、`tests/train/test_unified_chain.py`，75项通过。反例只证明公式/边界，合同测试只证明所测软件行为；本轮未重放全部历史模型、未复算原trial预测、未跑新训练，也不声称全部dirty tree回归通过。
`Ruff check src experiments tests`及两份研究脚本通过，`git diff --check`通过；本报告本地链接均存在。
命令、退出码、限制与交付哈希见[验证清单](../assets/unified_optimization_20260929/validation_manifest.json)。

一手文献检索及访问层级见[文献审查](../assets/unified_optimization_20260929/literature_audit.md)。主要证据使用方式：

- He & Wu，EA，[原文](https://arxiv.org/abs/1808.05464)：已读全文；逐域完整逆平方根和源/目标双侧处理有依据，离线使用全部目标EEG的效果不能移作仅calibration统计的效果。
- Rodrigues等，RPA，[作者实现](https://github.com/plcrodrigues/RPA)：已核实P300代码；ERP模板与classwise旋转的目标标签需求直接限制zero-label声明。
- Altındış等，GALIA/JFAL，2023：本轮核官方摘要，HAL全文访问受阻；具体公式需沿用并再核已有全文证据，不能拿accuracy增益推算AUC。
- Zhao等，2019，[PMLR原文](https://proceedings.mlr.press/v97/zhao19a.html)：不变表示并非域泛化充分条件；其无监督适配设定不直接证明当前class-conditional MMD有害。
- SupCon、PCGrad/CAGrad及L2-SP：提供目标、梯度协调、源中心正则的先例；不构成本项目已获性能收益的证据。用途和访问限制在文献审查逐篇注明。

本轮是围绕已观察失败机制的定向研究，不是穷尽系统综述。没有把文献benchmark、软件PASS或几何对齐质量转成产品准确率承诺。

## 9. 交付合同与续接

| 模块 | 入口/输入 | 本轮动作与出口 | 状态 |
|---|---|---|---|
| delivery | 优化方案请求、六指南、dirty tree | 冻结方案范围；保留并行改动与历史数据 | complete |
| research | A1/Q1竞争解释、既有文献 | 一手来源核查与访问界限；literature_audit | complete |
| science | 冻结参与者指标、真实源码 | 分层复算、公式、11组反例、实验与停止合同 | complete |
| engineering review | 当前与v4机制文件 | 调用链/EA缺口及切片清单、75项测试 | complete |
| execution | 用户本轮只要求方案 | 未启动新训练；无新模型效能出口 | not-applicable |
| knowledge | 上述证据 | 更新既有roadmap/研究规范入口、证据索引及明确勘误 | complete |

交接：`research/science → engineering`；输入为本报告、三份分支审查、数学结果与输入SHA256；尚未解决的是实际每折λ、几何收益与独立成人效能。下一可执行切片是既定真实窗口验证，其后才是完整EA fit/state/replay接线。实现与训练状态仍为未执行，不因本设计交付而标成已完成。
