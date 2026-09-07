# 跨数据集 ERP 迁移训练方案

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。状态：设计完成，待数据准入与训练实施。
目标：利用当前本地公开数据，学习降低成人 BrainSync 九选一校准成本的 ERP 表征。
本文不新增性能结果。活动规范仍为 [research_program](../research_program.zh.md)；
配套 [实施清单](implementation_plan_cross_dataset_20260906.zh.md)、
[数学验证](math_verification_cross_dataset_20260906.zh.md)。

## 1. 决策与已有证据

推荐路线：**多源监督 ERP 预训练 → 整库留出开发 → 冻结配方的外部公开库评估 →
成人 BrainSync 少量校准与独立目标切换测试**。

首轮模型保留 N2P3Net。先证明完整训练方法的迁移收益，再检验域对齐或复杂结构。
数据集名称相异、目标被试留出、目标数据集整体未参加 source，是三种不同事实。
跨库主实验必须满足第三种，并继续核查跨分发/跨库真实身份。

当前源码、缓存记录和已有结果文档提供以下依据：

| 已有开发结果 | 对本方案的约束 |
|---|---|
| BI2014a 全64人，63→1预训练后 K192 微调：五导 AUC 0.7297、16导 0.7681 | 不把所有源库永久裁到最小交集；高导信息保留值得另测 |
| BI→BNCI 五导原配方 zero-shot 约0.5075，微调约0.5276；合理 scratch 0.6357 | 同为P300并不保证正迁移，必须保留强 target-only 与经典方法 |
| BI→BNCI 改用前缀统计及更新BN的强微调，单seed K192约0.6688 | 统计与优化配方是可检验原因，不能归因成幅值已完全解释 |
| BNCI 7→1 LOSO 可达较好读数 | 该 source 已包含目标库其他人，不能用作整库未见结果 |

数字来自当前 [BI64报告](bi64_protocol_factorial_results_20260906.zh.md) 与
[既有执行报告](p300_transfer_results_20260906.zh.md)，本轮未重新训练或逐预测重算。

## 2. 现存数据与准入层级

本轮实际遍历 `mne_data`，读取全部两套 NEMAR 的通道/事件 sidecar，读取代表性 EEG
文件头及2048采样点，并检查 ZIP 中央目录。证据见
[design_audit.json](../assets/cross_dataset_20260906/design_audit.json)。
ZIP 可读不等于 CRC、官方checksum或全部信号有效；事件行数不是最终可用epoch数。

| 数据源 | 本地事实 | 七导产品布局中的实测覆盖 | 本方案角色 |
|---|---|---|---|
| BI2014a | 64 ZIP及已解压MAT/CSV；当前64人causal-v3 cache记录存在 | Cz/P3/Pz/P4/Oz，5导 | 已观察开发源；干电极域；全64人与作者61人敏感性分开 |
| BNCI2014-008 | 8 MAT；已有builder、缓存与迁移结果 | 全7导 | ALS公共开发源/伪目标，不能代表健康成人 |
| BI2015a | 43 ZIP目录可读；MOABB为43人、32EEG | 除Fz外6导 | 优先新增源；刺激时长条件保留为metadata |
| Huebner2017 NEMAR nm000199 v1.0.2 | 13人，342事件文件，162757事件行，统一31EEG；nose参考 | Fz/Cz/P3/Pz/P4，5导 | 优先新增源；跨实验室、视觉LLP拼写范式 |
| Mainsah2025 S2 NEMAR nm000313 v1.0.3 | 24人，288事件文件，207360事件行，统一32EEG | 全7导 | 候选整库封存公开目标；准入后可另训练部署source版 |
| BI2012 | 25 ZIP目录可读；加载器实际17EEG，不能把第17导误当stim | Fz/Cz/P3/Pz/P4，5导 | 后续源多样性/缺导分布对照 |
| BI2013a | 73 ZIP目录可读；加载器24人，多session/run | Cz/P3/Pz/P4/Oz，5导 | 后续跨session与适应模式对照 |
| BI2014b / BI2015b | 19/22个双人组ZIP；加载器38/44人、每人32EEG | 除Fz外6导 | 第二批源，先处理dyad依赖及各条件的本人target语义 |
| `MNE-llp-data` | 8 ZIP均不能打开中央目录 | 未准入 | Huebner同一研究的旧分发，不计第二个域或新增被试 |
| `MNE-P3-data` / GTN | 248 ZIP目录可读 | 既有三导儿童路径 | 独立开发压力测试，不并入七导成人主验证 |
| BrainSync | 当前指导明确尚无正式成人数据 | 7EEG + 独立EOG | 最终真实目标；本轮没有新增采集证据 |

BI通道覆盖来自已安装 MOABB 1.6.1 加载器，与原始通道争议分开记录；F3/F5不确定列
不参与产品七导映射。BI2013a原始Target码为33285，而NEMAR Huebner事件表的Target值为2；
必须按每个分发的真实label map解释，不能复用任意全局整数标签规则。

明确准入问题：

1. Mainsah的288份sidecar参考均为 `n/a`；不得改字符串假装与其他库同参考。
   channels.tsv将58/62Hz记在low/high cutoff，可能是转换的notch描述，须核对上游。
   代表性原始EDF/sidecar约256.00008Hz，而转换BDF头为256Hz，样本数42760→42752；
   应逐分发核对采样率、事件sample/onset及末端截断，不把两者当逐字节相同raw。
2. 两套NEMAR都标为 `DatasetType=derivative`、由MOABB转换。转换分发不是新的研究域；
   须证明信号单位、事件、滤波历史和边界，再重建当前causal recipe，不能声称回到不可逆处理前。
3. Huebner本地事件表有38297个Target、124460个NonTarget；Mainsah有17280/190080。
   前者约23.5%、后者约8.33%，不能固定认为所有公开P300都是1:5或BrainSync的1:8。
4. Huebner原研究含辅助EOG，但当前转换通道表只有31EEG、EOG计数为0；不宣称这些文件
   已提供独立眼电验证。raw里的刺激状态/target辅助列也不得混入分类EEG。
5. 跨年份不证明被试不同。BI2014a/b可能来自相关招募；BI家族存在身份不明时，
   保留“跨采集域、身份未完全核实”的开发层级，并用整个BI家族留出检验。
   dyad依赖作为split grouping记录，不新增替代 `participant_key` 的身份字段。
6. 原始Huebner全文确认同一句63字符句子重复三遍、每遍开始重置在线分类器；
   导入后的三个session不证明跨日。S2原始资料是24人、每人一次session，House/Tool
   是刺激类别，不能分别标成Target/NonTarget。其Train/Test精确分配仍待原始参数核实。

缺candidate身份只阻止特定决策任务，不阻止可靠binary ERP预训练。
不采用旧普查中“BI2015a不可作为source”“BI家族都是固定row/column”的裁决。

## 3. 训练与验证的两层结构

### 3.1 开发池及一次性外部评估

首批开发池在准入后固定为：

\[
\mathcal D_{dev}=\{BI2014a,BI2015a,BNCI2014\text{-}008,Huebner2017\}.
\]

开发期四折：每次完整留出一库作为伪目标，其余三库source训练；该库所有被试、
所有session、统计、EA、teacher、预训练head及早停信息都不进入source。
伪目标只在声明的early calibration内适配，later test产生比较结果。
这些四折读数允许选择方法，因此是开发结果，不再作为被选方法的无偏确认成绩。

额外做整族压力测试：BI2014a与BI2015a同时留出，source为BNCI+Huebner；
两库分开报告再族内宏平均。这是跨家族证据，不把年份数量当独立实验室数量。

Mainsah S2先保留为独立公开评估库：只做结构/来源/预处理核查，不看模型预测选配方。
首轮最终配方由开发池决定，然后source在整个开发池refit，才评估Mainsah。
本轮只查过Mainsah结构和标签计数；此前是否用于其他任务调参仍需execution记录确认。
因此目前称“候选封存外部开发验证”，不宣称已完成前瞻确认。
若曾被观察，仍可使用，但如实降为development，不能靠换seed恢复独立性。

Mainsah参考/信号准入若未解决，先完成开发四折，保留外部评估未完成状态，
不以已经观察过的BNCI顶替“全新确认库”。后续BI2012/13a/14b/15b按固定增量扩源计划加入。

### 3.2 严格区分三个评估问题

| 名称 | source包含目标库其他人 | 本人标签 | 本人未标记数据fit |
|---|---|---|---|
| 跨库Z0 | 否 | 无 | 无，统计来自source |
| 跨库U(K) | 否 | 无 | 仅预声明early prefix，单列无标签校准成本 |
| 跨库S(K) | 否 | 仅early calibration | 同一calibration内允许 |
| 库内LOSO+校准参考线 | 是 | 仅early calibration | 同一calibration内允许 |

U(K)不能称零校准Z0；整段target白化或BN更新属于transductive实验，不纳入本轮主线。
源库标签映射正确不代表可迁移部分纯属P300，视觉成分与伪迹捷径仍需单独诊断。

## 4. 空间接口：保留源信息但不伪造测量

### 4.1 首轮产品布局

复用 `data.contract` 七导顺序及 `trial_channel_mask`，每库只将真实电极映射到同一布局。
缺导用未观测mask及标准化后的零占位，不当作测量或插值。EOG与状态列不进入张量。
训练行的观测并集必须覆盖所有输入导；检查发生在outer/inner exclusions之后。
在上述四折，BNCI提供全7导；留出BNCI时，BI2015a的PO8/Oz与Huebner的Fz可共同覆盖7导。
这只是可训练性条件，不能证明缺导模式没有被网络当作域捷径。

\[
 Z_{ict}=m_{ic}\frac{X_{ict}-\mu_c^{fit}}{\sigma_c^{fit}},\quad
 \mu_c^{fit}=\frac{\sum_{i\in fit,t}m_{ic}X_{ict}}{T\sum_{i\in fit}m_{ic}}.
\]

不同native参考不能直接混成一个物理合同。跨库主桥接优先验证共同实测Cz参考：
输出Fz/P3/Pz/P4/PO8/Oz六个差分，保留缺导mask；只有确认各EEG共享采集参考、Cz实测
且未被不兼容的上游变换改变，才能应用。Mainsah的 `n/a` 不能自动满足该前提。
七导原生参考与六差分必须分表报告；本方案不宣称后者一定更准确。

\[
 (s_c-r)-(s_{Cz}-r)=s_c-s_{Cz}.
\]

BNCI既有native七导优于Cz六差分，说明参考可比性有信息成本。native五导BI↔BNCI
已核实同参考的桥接继续作为复现对照。Huebner没有Oz，不能为了纳入它悄悄把该五导
复现改成四导；同样不能把所有库各自CAR的结果称为同一个参考空间。

### 4.2 全通道teacher是第二阶段假设

对同一source trial训练16/31/32导teacher，产品布局student复用N2P3Net：

\[
 L_{KD}=L_{CE}+\lambda\tau^2\operatorname{KL}
 [\operatorname{stopgrad}(p^{\tau}_{teacher}(X_{native}))\Vert
 p^{\tau}_{student}(X_{product},m)].
\]

teacher也只用合法source；先比较\(\lambda=0\)与一个开发期冻结的非零值，
保持student结构、source trial与更新数一致，teacher计算成本另报。
该方案是源特权信息蒸馏，当前未实施；缺失导中独有信息不可能被student凭空恢复。
BIOT只提供异构输入结构的先例，不是此蒸馏方法或本项目性能的证据。
不直接引入大Transformer或重写已有特征适配wrapper。

## 5. 源训练目标与公平预算

先保留可复核的加权CE路径。设训练域d、被试u、trial i，类别权重为\(a_y>0\)：

\[
 L_S=\sum_d\alpha_d\frac1{|U_d|}\sum_{u\in U_d}
 \frac{\sum_i a_{y_{dui}}\,CE(f_\theta(Z_{dui}),y_{dui})}
      {\sum_i a_{y_{dui}}},\quad \alpha_d\ge0,\quad\sum_d\alpha_d=1.
\]

每个训练折保留的域等权，域内participant等权；这是主要多源风险设计。
另做BI家族合计与BNCI/Huebner各1/3的族权重敏感性，避免复制同家族库就增加其总权重。
权重由现有 `resolve_source_risk_weights` 计算，不另写低级sampler。
同一participant多run不获得额外participant权重；session等权是另一个估计量，不能默认替换。

当前 `DeepConfig` 的CE正类权重默认8，source CLI尚未显式暴露它，不能把旧训练称为
逐域1:1平衡风险。实施时显式记录/开放该参数：先复现8，再在source开发中比较1；
不同时使用过采样、class weight和按类逆频率，除非公式明确包含这些因素。
域内正负类精确各半是可选新目标，不是上式的自动性质。

首轮训练起点：batch512、Adam lr1e-3、weight_decay1e-4、最多30epoch、
source内participant分组早停patience6；seed20260906/07/08，沿用当前模型配置。
这些是可执行起点，不是跨域最优定论；硬件采用已有BF16/compile/fused及实际fallback记录。

必须分开两类实验：

- **方法效用**：各方法有同等开发调参机会，在相同目标预算/测试集上比较最终方法；
  源数据量、实际updates、epochs与耗时全部报告。
- **源多样性或机制归因**：source单域/多域匹配总draws、每update batch、总updates，
  配对源被试数与正例数或分开报告其变化。相同epoch不代表相同训练预算。

当前训练器是epoch驱动，精确update预算与跨域宏平均epoch选择需要小范围联动实施。
加权CE的batch分母也随类别构成改变；不能把全数据权重精确性等同SGD每步严格域等权。
该边界已有反例，见数学文档；初始方法比较可保留现有训练语义，机制归因前要绑定预算/目标。

## 6. 目标校准与最小实验矩阵

### 6.1 校准预算与时间隔离

公共binary桥接继续前48/96/192个trial，所有K共享最大预算后的同一purged test。
同时报告正例数、原始elapsed秒数、run/session数；不跨库声称同K等同时间成本。
以128Hz、[-200,1200)ms、连续forward IIR/steady-state及mean-only保持当前输入。

多run数据先恢复原研究Train/Test及采集顺序；不能按字符串run-10/run-2排序，
也不能将多个重置到0的时间轴直接拼接。优先用完整早期calibration runs，
later独立test runs保持固定；house/tool等condition单列，不把条件混入时间收益。
run不足的库退回已声明的单记录binary协议，并按participant先汇总run再跨人平均。

原始epoch窗口不重叠，内层train/validation/refit皆属于授权calibration；
固定IIR允许保留过去状态，FFT重采样的可用时点以完整epoch计算。
缺正类或负类不从test补标签、不事后扩大预算：标记fit失败/回退到预声明source预测，
保留请求队列、覆盖和失败账本，报告不同方法共同可评分的配对队列。

### 6.2 核心对照

| 臂 | 初始化与fit | 回答什么 |
|---|---|---|
| target-only shrinkage LDA | calibration内scaler/协方差 | 强低样本经典基线 |
| target-only N2P3Net | 全部随机参数和BN；calibration统计 | 预训练是否降低目标标签成本 |
| source Z0 | source参数/统计，无目标fit | 真正整库未见泛化 |
| source classifier_fine | 保留源head，冻结trunk | 特征可迁移性与小预算收益 |
| source full_fine | 保留源模型，calibration内全网更新 | 目标域偏移是否需要改特征 |
| source feature + shrinkage LDA | source特征固定，head仅calibration拟合 | 复用现有低复杂度特征对照 |

full_fine预先比较轻配方15epoch/lr1e-4与强配方60epoch/lr1e-3；
源统计/冻结BN和前缀统计/更新BN分为明确的配方因素，不能全部混成一个“微调”名称。
scratch保留合理60epoch/lr1e-3/前缀统计/训练BN及其同等开发调优机会。
先一seed诊断，再三seed复核；诊断数据继续标development。
可在calibration内部选择配方；不足以形成合法内层时使用开发期冻结fallback，禁止test选人选臂。

EA、Split-BN、条件特征对齐、蒸馏各自单独对plain增量比较；有稳定的附加收益后才组合。
EA的target拟合记U/S预算；Split-BN未见域buffer由source固定组合或calibration生成，
没有一个“已训练的目标BN域”。目标原生名称不能索引source域的私有残差。

## 7. 指标、晋升和最终使用

公共主读数为各库participant-macro trial AUC；再对库等权汇总，避免大库淹没小库。
报告每人\(\Delta(K)=AUC_{transfer}(K)-AUC_{scratch}(K)\)、相对LDA差、负迁移人数、
低分位、BACC、AP及目标比例、概率质量、calibration时间与失败率。
确定性LDA不因复制3个seed而增加样本数；先在人内平均seed，不能把trial当独立bootstrap单位。
四个开发库和少量seed不支持广泛人群泛化结论；固定模型重采样CI与训练程序不确定性分开。

核心比较预先固定K96迁移减scratch，并辅报减LDA；K48/192为完整预算曲线。
此K96是公共开发检查点，不等于当前产品4050trial校准预算。开发晋升看跨库一致性、效应大小、
负迁移和计算成本，不要求每库都正增益，也不按不显著宣判等效。
首轮将AUC 0.01仅作为开发资源优先级的工作阈值，不当文献给定或确认性SESOI；
产品SESOI/样本量须结合真实成人决策成本和方差另行冻结。

通过Mainsah评估后，可用包括Mainsah的全部合格公开源再训练**部署候选**。
它是新的checkpoint，先前Mainsah结果不能用于声称该新checkpoint对Mainsah未见。
所有raw/转换/cache/训练行/源码archive/环境/模型绑定到各自版本。

成人阶段复用更新后的采集计划：7EEG+独立EOG，九数字各10block×45刺激校准，
共4050trial/450target；独立新test每次1block×45刺激，主R5。测试数字可以曾作为校准目标，
但测试EEG/标签不得参与拟合，`unseen_calibration_codes`保留为额外对照。
固定R与请求决策分母，正式指标为九选一operational hit@R、
correct selections/min及成本；BNCI36选一、Mainsah72选一不能直接合并成九选一准确率。
没有成人数据时，到公开开发与软件闭环为止。

## 8. 外部依据及边界

以下来源在本轮在线核实；摘要范围的证据不冒充全文协议复现。

1. Song et al. (2024), *Signal alignment for cross-datasets in P300 brain-computer interfaces*.
   [DOI 10.1088/1741-2552/ad430d](https://doi.org/10.1088/1741-2552/ad430d)。
   摘要：跨拼写器/人脸/听觉P300数据对齐幅值、极性与潜伏期；支持问题可研究，
   目标fit范围和其成绩尚未据全文核实，不据此预设具体对齐必然有效。
2. He & Wu, *A Euclidean Space Data Alignment Approach*.
   [DOI 10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914)。
   摘要包含MI与ERP，无目标标签；无标签不等于无目标数据，也不保证类条件一致。
3. He & Wu, *A Label Alignment Approach*.
   [DOI 10.1109/TNSRE.2020.2980299](https://doi.org/10.1109/TNSRE.2020.2980299)。
   摘要明确目标每类需要标签，实验为MI；不作为无标签P300结果引用。
4. Gonsalvez & Polich (2002), *P300 amplitude is determined by target-to-target interval*.
   [DOI 10.1017/S0048577201393137](https://doi.org/10.1017/S0048577201393137)。
   原实验摘要显示TTI/前置non-target/ISI影响P300；类别权重不能自动纠正条件分布。
5. Yang et al., BIOT, NeurIPS 2023.
   [会议摘要](https://papers.nips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)、
   [作者代码](https://github.com/ycq091044/BIOT)。异构通道建模先例，非本项目蒸馏或成人P300优效证据。
6. Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*.
   [作者DomainBed选择规则](https://github.com/facebookresearch/DomainBed#available-model-selection-criteria)。
   源内、额外留域与target oracle规则明确分开；此处用于设计模型选择权限，不引用视觉结果当EEG性能。
7. 数据原始入口：[BI2015a](https://doi.org/10.5281/zenodo.3266930)、
   [Huebner研究](https://doi.org/10.1371/journal.pone.0175856)、
   [bigP3BCI](https://physionet.org/content/bigp3bci/1.0.0/)。
   本地分发身份及更具体准入问题见第2节和盘点JSON。
8. Jones & Sellers (2019), *Faces, locations, and tools: a proposed two-stimulus P300 brain computer interface*.
   [DOI 10.1088/1741-2552/aaff22](https://doi.org/10.1088/1741-2552/aaff22)。
   已核实原始摘要的location/tool条件及刺激特异分类对照，尚未核实S2完整参数。
   [PhysioNet原页](https://physionet.org/content/bigp3bci/1.0.0/)确认S2单session与
   采集端滤波的一般说明，未给出S2参考和具体截止频率。
   [Huebner原论文全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0175856)
   确认31EEG/nose/1000Hz及三遍句子；不据NEMAR命名推断跨日。
