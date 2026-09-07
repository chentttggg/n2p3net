# P300 研究改进与实验目录

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06（检索跨 09-05 至 09-06，Asia/Shanghai）。核查源码：`c8ee573`。
性质：研究建议与实验设计目录；不是已完成实验、已批准的生产合同或云端执行配置。
范围：现有 BI2014a、其他有适用条件的公开 EEG，以及成人 BrainSync 九选一后续独立决策。
全文证据、查询记录与事实冲突见 [配套核查记录](research_evidence_audit_20260906.zh.md)。

执行优先级与最新结果以[全局执行计划](p300_transfer_execution_plan_20260906.zh.md)为准。
用户随后明确公开数据仅承担binary ERP预训练角色，因此BI物理candidate冲突不阻塞该主线。
当前产品硬件是八个ADC槽、七路EEG加一路EOG：原PO7槽采PO8，原PO8槽采EOG；
下文早期“八导”候选设计按此最新输入角色解释，不要求八路都进入分类器。

## 1. 研究方向与新增优先级

下一阶段最有信息量的工作是区分四种竞争解释：数据和任务定义有误、信号表示不足、
小样本目标估计不稳、真实跨域差异。不能用同一个 AUC 平台把四者合并为模型容量问题。

本次复核发现，原来建议直接执行的 E0/E1 之前，还需要一个 **E-1 协议与数据锚定**：

1. BI2014a 原始技术报告 p8 明说物理符号的 row/column 分组每个 repetition 随机重建；
   当前代码按固定 group ID 累加，并以 target pair 变化切 selection。原始事件又存在
   连续多轮相同 target pair，尚不能判定论文描述与实际发布实现谁更准确。应先查版本、
   group-to-symbol membership 和实际 level 边界。冲突未解决前，现有跨 R 结果只能作
   当前解析器生成分组上的读数，不能作为物理字符选择能力。trial 二分类可独立推进。
2. 原报告明确建议不将 62/63/64 用于 P300 分析；当前 dev64 含这三人。先建立原作者
   建议的 61 人研究口径及全 64 人口径；忠实复现MOABB时必须使用其01–64名单，
   61人结果另列，不能直接与64人的85.77比较。不按新模型成绩挑人。
3. 原报告电极名是 F5/F6，当前代码和 MOABB 执行加载器是 F3/F4。Header 文件与采集
   provenance 是下一步裁决依据；尚不能仅据论文改名或计算依赖这两导的几何对齐。
4. 当前 cache 的 metadata 有 selection/repetition，但统一 event timeline 标为
   `complete=False`、candidate 为空、repetition=-1。读两条路径的消费者会得到不同信息。
5. 现有 MOABB 论文和官方结果页能核实 BI2014a 的经典基线，但没有原批判文档中的
   EEGITNet 86.66 / EEGNet 82.09 P300 行，不能把这些数字当本项目验收门槛。

因此新的顺序是：**E-1 → E0 双轨基线和 E1 信号诊断 → 简单校准及合法证据聚合 →
容量/窗口/训练扫描 → 有门槛的迁移机制比较 → 成人前瞻确认**。
停止新增机制不等于把未检验机制判成无效；CI 含 0 也不等价于实用等效。

## 2. 统一设计与裁决约定

### 2.1 四种问题分别回答

| 轨道 | 数据与切分 | 主要用途 | 不支持的外推 |
|---|---|---|---|
| E0-A 文献复现 | 锁定论文/MOABB 版本、被试名单、预处理、方法和 session 内随机分层五折 | 验证外部参照与本地实现差异 | 后续独立决策、在线时间、跨被试上限 |
| E0-B 同被试诊断 | 完整可信 acquisition block/selection 分组五折；另设固定 early→later 测试 | 区分记忆相邻数据、时间漂移与表示不足 | 少量校准的绝对上限 |
| E0-C 校准学习曲线 | source 排除目标；各 K 使用共同后续测试段，并同时报告各预算可用全队列 | source prior 是否减少目标标签需求 | 未见成人/设备上的产品能力 |
| E0-D 产品确认 | 成人、真实七EEG+一路EOG、目标切换、预声明 calibration、后续独立 decisions | 九选一与真实时间/失败成本 | 认知或医疗诊断 |

若物理 selection 尚不可信，E0-B 使用可独立验证的 raw session/block 或连续时间块，
边界及 temporal embargo 必须显式记录。不要改用随机 flash 来绕过分组问题。
相邻 epoch 的共享 raw 样本、低频状态和未来信息使用必须分别核查：因果滤波携带过去状态
可符合在线部署，随机切分共享样本则是不同的评估依赖问题。

### 2.2 比较的共同条件

- 配对单位固定为 participant；session、selection、训练 replicate 是不同层级。
  同一次训练的 64 个 LOSO fold 不是 64 次独立训练程序复现。
- 训练/验证/test 的 trial ID、源数据组成、normalization、QC、超参数搜索预算、seed、
  epoch 选择规则、实际 optimizer updates、dropout/BN 状态与精度模式均写入 manifest。
  被比较方法可使用各自合理训练配方，但要区分“完整方法比较”和“单因素机制比较”。
- 所有拟合步骤在训练或声明的 calibration 范围内完成，包括 xDAWN、ICA、EA、
  whitening、特征选择、尺度统计、早停、temperature、动态停止阈值和超参数选择。
- 测试不能为缺失的刺激创造 EEG；原系统提前停止后的反事实观测无法靠补零、复制、
  跨 selection 拼接或只保留长尾来恢复。IPW 也需可识别停止机制及 positivity，不能默认使用。
- BI 开发可把 subject-macro AUC 作为主要诊断指标；真实候选任务另报告固定预算准确率。
  AUC 高不能替代决策正确率，缺失候选信息的数据也可用作 binary source，不能一概废弃。
- 对有效 candidate 数据报告每个 R 的 coverage、hit、NLL/Brier、失败原因和同 cohort
  配对曲线。R1/R2 优先级来自当前解析器覆盖读数，不是所有数据集的通用常量。
- 给出 observed/eligible/attempted 三种适用分母。采集未继续导致缺失与模型拒判/失败
  分开记账；产品 operational 分母保留实际请求的失败，不能用条件准确率替代。
- 调参和候选筛选只产生 development 证据。dev64 已被多轮观察，不可重新命名为未触碰
  confirmation；确认使用新成人数据、未用于筛选的 session/数据域或事先隔离的真实样本。

### 2.3 统一决策符号

下表是候选实验，不是要求把全部组合跑完。72 项涵盖当前证据下有明确问题的空间，
不声称穷尽所有可能神经网络。P0 为影响解释或低成本高信息量；P1 为诊断后跟进；
P2 为条件触发/新数据。成本 V0=审计或小数学检查，V1=冻结模型分析，V2=单被试训练，
V3=源训练/LOSO，V4=raw 重建或复杂信号分析，V5=新采集；这些不是未经实测的耗时承诺。

共用裁决：

- **G0 语义门**：标签/时间/身份/membership 证据不足时，相关端点不作科研性能结论；
  不影响独立成立的 binary 或软件检查。
- **G1 定位门**：只回答某种解释是否更可信；差异不能直接归因为唯一机制。
- **G2 采纳门**：预声明的主要端点成对改善，同时可靠性/覆盖/成本未超过预声明非劣界；
  至少在独立 seed 或 held-out session 复现。没有预设效应量前不能计算一个武断的样本数。
- **G3 简化门**：结合预声明 SESOI、等效/非劣检验及资源成本裁定；CI 很宽只是未决。
- **G4 产品门**：固定配方和 calibration 预算，在新成人真实蒙太奇/目标切换数据确认。

## 3. A：数据身份与任务协议，8 项

复用：`src/data/bi2014a_schedule.py`、`bi2014a_candidate.py`、`candidate_task.py`、
`events.py`、`epochs.py`、`raw_artifacts.py`；先修真实 builder，再更新消费者和测试。

| ID / 级别 / 成本 | 问题或竞争假设 | 最小实验/核查 | 读数与裁决 |
|---|---|---|---|
| A01 / P0 / V0 | row/column group 是否跨 repetition 对应相同物理符号 | 对照 HAL 原文、原游戏版本/配置/随机种子、发布数据 Header/事件；尝试恢复每闪物理 membership，不用 EEG 分类结果反推 | 提供 event→symbol 映射或明确不可恢复；G0，决定跨 R 聚合是否有意义 |
| A02 / P0 / V0 | target pair 变化是否就是新的目标决策 | 对照独立 level/block markers 与现有 pair-change；专查同物理目标但重排分组、目标改变但 pair 相同、104/105/106 含义 | boundary confusion、各来源计数；标签驱动边界只允许声明为 retrospective proxy |
| A03 / P0 / V0 | 二分类标签、单位、时间与 raw 是否忠实 | 逐行 sample/flash code/raw label/cache label 对齐；核对 V/uV、极性、原始时间跳变与丢弃尾部 | 错配必须定位到文件/事件，不以 ERP 外观或测试通过替代；G0 |
| A04 / P0 / V0+V2 | 作者已知问题样本是否主导平台 | 固定61人名单(01–61)与全64人双轨；36/46/52/57只做预声明敏感性，不因当前AUC删除 | paired AUC、分布尾部、缺失原因；不是通过排除提高宣传分数 |
| A05 / P0 / V0 | F3/F4 与 F5/F6 命名冲突 | 读取原 Header/采集配置并与 loader 对账；其余确定通道继续分析，歧义导不参与精确几何推断 | canonical 名称/坐标/provenance；不把 AFz 或接地 Fz 当实测 Fz |
| A06 / P0 / V0 | metadata 与 timeline 信息分裂 | 经真实 builder 重建抽样，对比 candidate/repetition/selection/evidence status；核查观察到的事件与完整计划区别 | 字段一致性、消费者同输入同输出；不能仅把 complete 改成 True |
| A07 / P0 / V0 | session、同一人、双人同步记录被拆散后泄漏 | 检查当前 session/run 全0是否丢原层级；公共数据身份、同日同人、BI双人共享记录联合核对 | 同一人重复保留同一participant；双人保留不同participant_key，按dyad/session联合分割或聚类，避免共同记录的依赖泄漏 |
| A08 / P0 / V0+V1 | 短R由旧系统成功停止而产生选择偏差 | 按独立确认的停止/重启状态、原算法反馈及R分布分层；比较同批完整case曲线；无法恢复则标为不可识别 | censoring机制、coverage、可观察预算范围；不声称由长尾推断全队列R8 |

## 4. B：基线与评估锚定，6 项

复用 `baselines/evaluate.py`、`classic.py`、`riemann.py`、`deep.py`、
`experiments/run_eeg_loso.py`。其中 `within_subject_folds` 当前是完整组随机holdout，
不是完整五折；本地 `SWLDA` 是单变量筛选近似，不是文献原始多变量逐步算法；
本地 `XdawnRiemann` 也不能仅凭名称当成 MOABB 的 `XDAWNCov+TS+SVM`。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| B01 / P0 / V2 | 外部基准可在锁定版本下复现 | 原版 MOABB `XDAWNCov+TS+SVM`、`XDAWNCov+MDM`；01–64、相同epoch/单位/五折/内部搜索；01–61单列 | AUC均值/分布及版本差异；不以0.82/0.75二分；G1 |
| B02 / P0 / V2 | 主干比强参照弱还是整个管线弱 | 当前管线同fold比较 N2P3Net、EEGNet-8,2、窗口LR或shrinkage LDA、原版Riemann参照 | 同条件AUC差、train/validation差距、失败率；优先定位唯一落后层 |
| B03 / P0 / V2 | 随机flash五折高分依赖相邻数据 | 固定模型/预处理比较随机trial、完整block/可信selection五折、early→later；严格记录raw overlap与边界 | 泛化落差而非只报最高分；随机trial结果只归E0-A |
| B04 / P0 / V2 | 验证/早停或实现缺陷造成训练不足 | 小型真实train subset overfit诊断、固定真实test；从随机初始化重训的合法组级标签置换、恒定/噪声输入；另做metadata-only对照 | 真标签train可学、null不过度超随机；去掉raw code中直接编码target的部分；带真实标签预训练的模型不能当完全null |
| B05 / P1 / V2+V3 | 限制在源迁移导致错失目标独立训练基线 | 同目标标签量比较 target-only、source zero-shot、source head/full-fine；再加较大目标训练量参考 | learning curve、标签/时间成本；within-session大训练量不是few-shot上限保证 |
| B06 / P0 / V0+V2 | 名称相同掩盖模型与训练预算不同 | 核对官方/本地经典及深度模型算子、超参、参数/可训练参数、训练updates；分别报告原配方与统一预算 | baseline实现差异清单；先纠正命名/配方再归因“新网络更强” |

## 5. C：时间轴与预处理，8 项

复用 `preprocess.py`、`EEGDataContract`、`PreprocessingSpec`、物理感受野配置。
改变可用截止时间须从 raw 重建或验证算子支持，不能只裁现有FFT重采样cache。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| C01 / P0 / V1+V4 | 分类主要依赖早期VEP/晚期ERP/非事件慢变 | 单独重训0–250、250–600、600–1000ms及全窗；冻结遮挡另列，训练量固定 | AUC、ERP拓扑、G1；早期有效不自动是伪迹，晚期有效不自动是P3 |
| C02 / P0 / V4+V2 | 1200ms证据是否值得等待 | raw重建截止600/800/1000/1200ms，固定tmin、滤波/采样；先宽度固定，再与E01做交互 | AUC/合法hit与最早可用时刻、latency frontier；G2 |
| C03 / P0 / V4 | pre-onset信息来自未来混入还是刺激历史 | 源采样轴pre-onset-only、cache pre-onset-only、仅历史时序诊断模型；保留完整时间分组 | 非零pre-onset AUC不是自动泄漏；隔离FFT、前序target重叠和顺序先验 |
| C04 / P1 / V4+V2 | baseline噪声大于漂移校正收益 | raw重建：当前[-200,0)均值、无扣除、预声明短窗/中位数；固定训练尺度统计 | baseline方差、AUC、共享残差；不用逐trial标准差归一化抹去幅值 |
| C05 / P1 / V4+V2 | 因果滤波与文献离线配方有代价 | forward对zero-phase仅作离线参照；另扫高通0.1/0.5、低通15/20/30Hz，分开归因 | 频响/相位/ERP与AUC；双向滤波幅频也改变，不能都归因延迟 |
| C06 / P0 / V0+V4 | chunk状态、重采样和可用时刻不一致 | whole raw与真实chunk state-carry比较；截止后脉冲攻击；丢包/重连时间戳回放 | tensor/logit数值与evidence-time一致性；在线宣称须所有算子支持闭合 |
| C07 / P1 / V4+V2 | 采样率与感受野错配影响表达 | 128/256Hz对照，保持卷积毫秒跨度、epoch与参数比较规则；检查anti-aliasing | AUC、RF、耗时/内存；改善不能归为单纯“更高采样率” |
| C08 / P1 / V4 | 重叠/不应期/潜伏期解释相关噪声 | 实际前后刺激间隔分层；regularized rERP/Unfold诊断，先验检验设计矩阵秩；与label-blind延迟估计区分 | 事件响应/历史交互、残差协方差；不使用未知测试target标签清理测试输入 |

## 6. D：伪迹、参考与空间采样，8 项

复用 `artifact.py`、`qc_features.py`、`domain.py`、`select_epoch_channels`；
MNE ICA/EOG回归作为成熟算子，仅在证据支持时接入既有raw链路。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| D01 / P0 / V2 | 前额导/后部导决定解码 | 原16、去Fp1/Fp2的14、真实交集Cz/P3/Pz/P4/Oz五导；各自重训；空间名歧义先走A05 | 成对AUC、参数/预算、合法低R；五导不是产品八导，掉点不是眼电证明 |
| D02 / P0 / V1+V2 | 模型分数与伪迹负担相关 | train-fit幅度/flatline/前额尖峰分层；Fp-only诊断、完整模型、同质量匹配对照 | 分层AUC、class-dependent QC、score关联；Fp-only不等于EOG-only |
| D03 / P1 / V4+V2 | ICA清理是否改善并保留任务信息 | 无ICA与训练/calibration raw拟合ICA；高通副本拟合、原任务带宽应用、最后baseline；成分选择盲于test标签 | IC/rank/provenance、残余伪迹、ERP/AUC；不要从已baseline cache直接开始标准ICA |
| D04 / P1 / V4或V5 | EOG回归与ICA收益是否相同 | 有独立VEOG/HEOG时比较未清理、训练fit回归、ICA；没有独立EOG仅作代理敏感性 | 伪迹残差与任务损失；缺EOG时不可声称完成眼电因果归因 |
| D05 / P1 / V2 | CAR删除了共模信号还是改善参考迁移 | native/CAR16；五导后CAR与CAR16后取五导分开；诊断共模和差分成分 | AUC、rank、ERP幅度、跨session；CAR16负效不推广至全部参考 |
| D06 / P1 / V2 | QC改善conditional性能但损失coverage | 当前QC关闭/现有fold-local QC；匹配拒绝率；train剔除、channel mask和test拒判分开 | risk–coverage、类差异拒绝、operational hit/耗时；不能删test换高分 |
| D07 / P1 / V1+V2 | 坏导和接触偏移比域机制更关键 | 真实QC层/接触记录；预声明单导缺失、尺度偏差和offset模拟；重训稳健臂与冻结故障诊断分开 | 降级曲线、拒判/恢复时间、最差被试；合成鲁棒性不是实机证据 |
| D08 / P2 / V5 | 产品八导与替代电极的实际收益 | 成人同时扩展记录中比较真实八导/候选布局；开发内选电极，另日摘戴确认 | 校准曲线、坏导率、安装时间；不据BI五导结果决定硬件删P3/P4 |

## 7. E：主干、容量和优化，8 项

复用 `N2P3ArchitectureConfig`、`models/n2p3net.py`、`baselines/n2p3net.py`、
`baselines/deep.py`、`run_eeg_loso.py`。源预训练CLI目前只显式构造temporal kernel配置，
宽度扫描必须把有效architecture贯穿预训练、checkpoint、adapter和manifest，不能只改一个入口。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| E01 / P1 / V2 | MST输出宽度是瓶颈 | 每scale特征2/4/8/16，先固定尺度数、ST和时间窗；记录全部参数与可训练参数 | train/val AUC、rank/effective dimension、成本；四通道本身不证明欠拟合 |
| E02 / P1 / V2 | ST而非MST压缩丢信息 | temporal filters与spatial depth multiplier分别加宽；在近似参数匹配下对比E01 | 哪一阶段宽度有稳定收益；避免只比较总参数 |
| E03 / P1 / V2 | 时间尺度与P300变化不匹配 | 复用K35/K65和MST物理跨度；保持输出宽度与readout，先单轴再2×2交互 | AUC、潜伏期分层、长短分支贡献；RF不等于独立生理频段 |
| E04 / P1 / V2 | 展开头方差大于时间结构收益 | 当前full_unfold、global_average、已有ms_flatten/LMBC、正则线性头；先匹配主干 | AUC、泛化gap、参数/校准样本数；不重新实现已有pooling |
| E05 / P0–P1 / V2+V3 | 训练不足或学习率不适配 | 当前16源epoch与更长预算，如32/64；记录optimizer updates与学习曲线；固定lr和cosine在同更新预算比较 | 收敛、train/val差距、耗时；无scheduler不是缺陷证据，冷启动耗时独立报告 |
| E06 / P1 / V2 | 正则与小数据不匹配 | dropout 0/当前0.25、weight decay两档；source训练与target微调分开设计 | 学习曲线及seed方差；先小型2×2而非全超参笛卡尔积 |
| E07 / P1 / V2 | 标签不平衡/采样目标改变学到的分数 | weighted CE、无权CE、平衡采样；固定epoch含义或明确更新预算；不同时双重补偿 | AUC/AP、BACC、NLL/校准、candidate排序；加权CE logit不自动是真LLR |
| E08 / P2 / V2+V3 | 结构增大或预训练数据增强有独立价值 | 仅在E01–07定位后考虑EEGInception/轻量TCN；幅值/小时间抖动/channel dropout按生理允许范围单测；大型FM/Transformer待独立收益证据 | matched预算、跨session稳定性、真实增益与复杂度；禁止用更多调参预算制造优势 |

## 8. F：重复证据与决策，8 项

**共同前置：A01/A02闭合，知道每次闪烁对应什么物理候选集合。** 不满足时在具有明确
scheduled membership的其他数据或前瞻BrainSync执行。复用 `candidate_evidence.py`、
`candidate_decision.py`、`candidate_set_loss.py`、`decision_loss.py` 与 `outcomes.py`。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| F01 / P0 / V1 | R曲线的提升/平台被样本组成改变混淆 | 在同批完整decision比较R1…Rk，另保留每R自然coverage；按原停止机制分层 | paired hit/row/column、样本名单；只适用于所选可观察子集 |
| F02 / P1 / V1 | ERP输入平均优于score累加 | 同一合法selection和物理刺激集合先平均再冻结前向，对比单flash sum-logit；R1为同一性检查，BN/dropout冻结 | paired hit、NLL、score尺度与分布变化；非线性不保证ERP平均更优 |
| F03 / P1 / V1 | 相关噪声降低有效重复数 | 训练/calibration估计score残差协方差，区分被试/selection/候选/重复/历史；held-out检验预测方差曲线 | empirical covariance、R_eff适用性；不由两点hit反推单一rho |
| F04 / P1 / V1+V2 | 少数坏trial或异方差主导累加 | 等权、train-fit QC权重/截尾或Huber聚合；对照同有效证据数；检查各候选权重不平衡 | hit/NLL/coverage与偏差；权重拟合不看test标签，不把权重总量当次数 |
| F05 / P1 / V1 | 共同漂移需要中心化 | score层每轮共同中心化作数值零效应对照；输入层中心化单独检验并记录可用时刻 | 等覆盖时score中心化应不改argmax；若改需先查实现/覆盖；输入中心化可能删ERP |
| F06 / P1 / V1+V2 | 潜伏期抖动抵消输入平均收益 | 使用训练模板的label-blind延迟边缘化/已有LMBC；和未对齐平均、score累加比较 | 双峰/低SNR反例、hit、延迟；不能按真实test target挑峰 |
| F07 / P2 / V2+V3 | trial代理损失与decision目标不一致 | CE vs CE+已有candidate损失，lambda含0和小档；完整decision batch、固定prefix预算和coverage | 梯度检查、trial AUC与合法hit共同读；decision数少时不单独训练复杂set encoder |
| F08 / P2 / V1+V5 | 动态停止可节约时间且控制错误 | 独立校准拟合LLR/阈值；固定R对照；在观测支持内replay，完整前瞻记录确认 | 正确selection/min、错误/拒判率、actual latency与NLL；旧系统提前停止后缺数据不能反事实补齐 |

## 9. G：少量目标校准，8 项

复用 `SubjectAdapter` 中 `classifier_fine`、`linear`、`full_fine`、`adapter`；
复用 `baselines/calibration.py` 与真实calibration split。少量目标数据不能同时支撑大量
超参数、early-stop、BN估计与温度选择，优先把选择规则在source pseudo-target上冻结。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| G01 / P0 / V2 | 现有linear劣效主要来自初始化/配方混杂 | zero-shot、source `classifier_fine`、随机linear、full-fine；固定normalization、selection预算、epoch选择规则并记录dropout | paired AUC/合法hit、参数变化；随机probe有表征诊断价值但不是唯一迁移基线 |
| G02 / P1 / V2 | 闭式/收缩估计优于小样本随机头 | frozen feature上shrinkage LDA、ridge、regularized logistic、source-centered ridge；超参只在内层 | 泛化与条件数、样本/维度比；复用sklearn，ridge不冒称概率LLR |
| G03 / P0–P1 / V2 | 简单校准能以较少标签逼近目标独立训练 | K=0/1/3/5/10/20合法decisions或固定target-flash数；共同suffix对比，并报告各K可用队列 | AUC/hit对真实标签数/分钟、最差分位数；K不同不允许偷偷换test难度 |
| G04 / P1 / V2 | target统计替换制造输入域偏移 | source、target-prefix、shrinkage(alpha 0/0.25/0.5/1)；每种配同一冻结/可训练头 | AUC、尺度漂移、校准量；当前linear和full配方差不能都归于头 |
| G05 / P1 / V2 | BN适应有益但小样本估计不稳 | frozen source BN、calibration-only重估、affine-only调整；batch大小固定，dropout对照；GN仅重训对照 | buffer来源/方差、AUC、batch/permutation敏感性；不以解冻BN必优作默认 |
| G06 / P1 / V2 | 全量微调遗忘source或优化噪声大 | head-only→最后块→full逐步解冻；source参数锚定与普通weight decay分开；lr/步数source内层选择 | 目标学习曲线、source-retention诊断、参数移动；不靠同test选择最好层 |
| G07 / P1 / V1+V2 | 概率错误而非排序错误导致拒判失当 | 当前正温度/正斜率Platt、无校准；独立calibration验证，NLL/Brier和risk–coverage为主要读数 | 全候选共用的正仿射log-odds保持AUC与等覆盖sum-logit argmax；sum probability不具有后一不变性 |
| G08 / P1 / V2+V5 | 校准样本多样性比数量重要 | 同trial/时长预算：早期集中、时间分散、覆盖不同目标/位置；置信选择仅用可获得信息且保留随机对照 | 后续switch性能、覆盖目标种类、时间漂移；自训练伪标签另作P2，不默认启用 |

## 10. H：数据来源与迁移机制，6 项

共同前置：二分类基线稳定、source/target同意标签语义及通道输入，身份和数据使用范围
闭合。机制不能凭名字进入默认配方，先证明它实际激活并与简单校准有公平对照。

| ID / 级别 / 成本 | 假设 | 最小对照 | 读数与裁决 |
|---|---|---|---|
| H01 / P1 / V3 | 数据量/被试多样性比复杂对齐更关键 | source人数如4/16/32/全部可用；固定总trial数与固定每人trial数两种设计；按身份排除target | 目标AUC/校准曲线、label成本；分开估计量与多样性效应 |
| H02 / P1 / V3 | 负迁移由source选择/参考不同而非head造成 | target-only、全部source、预声明相近source；统一normalization/参考、相同trial预算 | paired差、目标异质性；相近定义不由test label选择 |
| H03 / P1 / V3 | EA收益依赖协方差估计范围及正则 | none、合法per-fold/source/per-target-cal EA；样本量和shrinkage对照，限制condition number | AUC、rank/condition、reference digest；test参与无监督fit也改变estimand，不以无标签自动合法 |
| H04 / P2 / V3 | TSD/Dense只是在不同训练量下比较 | uniform等N、TSD、Dense；相同initial checkpoint、updates、source范围与标签配额 | AUC、选择稳定性/覆盖、计算量；native全量对selected从零不是选择机制对照 |
| H05 / P2 / V3 | AS-MMD/Split-BN实际优于普通联合训练 | target-only fine-tune、source+target joint CE、joint CE+SplitBN、再加logit MMD；真实active计数/梯度/buffer验证 | paired主端点、各类间距/塌缩、buffer-swap；marginal距离下降不证明判别保持 |
| H06 / P2 / V3 | 域残差/分解/adapter有独立贡献 | 每次加入一个既有机制，保持架构/预算；真实声明域；source-residual置零、private任务probe、adapter范数反例 | 任务不劣+独立机制证据后才G2；CI宽维持未决，证实实用等效后G3物理归档 |

## 11. I：统计、可靠性与研究效率，6 项

复用 `stats/hierarchical.py`、`research/contracts.py`、分析manifest与现有accounting。
当前统计接口分别支持固定模型的participant bootstrap与训练replicate bootstrap，
不是已经自动完成participant×seed双维总体推断。

| ID / 级别 / 成本 | 问题 | 最小设计 | 读数与裁决 |
|---|---|---|---|
| I01 / P0 / V0+V1 | 指标分母/尺度混用 | 逐arm同时列subject-macro AUC、micro仅诊断、eligible/attempted/observed、真实候选K、各R计数 | +0.0541 valid-hit与-0.0255 AUC分开；不能将二者相同单位排名 |
| I02 / P0 / V0 | 模型选择已过拟合同一dev64 | 保存全部尝试和配置，开发内层选超参，外层固定；确认另留真实未用数据 | 搜索预算/失败/负结果均公开；不把挑中模型的原dev CI当confirmatory |
| I03 / P1 / V0+V3 | 单seed或少participant决定方向 | matched seed复现与participant配对分开；训练重复先做3seed诊断，确认数量由方差和SESOI计划决定 | seed/participant/session变异，非只报均值；更多bootstrap次数不能替代独立重复 |
| I04 / P1 / V0 | 无显著差异被误判为等效 | 在看新结果前声明SESOI，使用等效/非劣设计；用已有方差范围作MDE/样本数敏感性或模拟 | CI、实用收益与计算成本；不做观察到的effect反算post-hoc power |
| I05 / P1 / V1 | 高AUC掩盖不可靠概率或低覆盖 | decision NLL/Brier、校准曲线、risk–coverage、错误/拒判原因；混淆感知信息率与正确selection/min | 不把ECE低或nominal ITR高单独晋升；动态候选/非均匀错误要声明假设 |
| I06 / P1 / V0+V2 | 消融数量多但交互不可识别 | 基线后选择少数因子做2×2（宽度×截止时间、head初始化×normalization、BN×dropout）；大筛选采用声明alias的设计 | 主效应/交互、独立确认；不一次跑72项再挑最显著者 |

## 12. J：成人产品与前瞻范式，6 项

这些项目需要真实采集，不能以合成信号或现有 BI/GTN 代替。遵循已批准的人体研究方案，
仅在新增操作超出既有方案时补充相应研究流程；本目录不添加无关审批负担。

| ID / 级别 / 成本 | 假设 | 最小前瞻对照 | 读数与裁决 |
|---|---|---|---|
| J01 / P0–P1 / V5 | 稳定物理membership与显示时序能否真实落盘 | 真实八导+可追溯全部candidate集合、计划/显示onset、photodiode延迟、丢帧/丢包和结束状态 | 可执行raw→模型replay一致性；G4的采集前置，不是另造事件系统 |
| J02 / P1 / V5 | 同session校准收益可跨时间/跨日维持 | 已知early校准→未知target-switch later；另日/摘戴重复，zero-shot/head/full-fine配对 | subject-macro九选一、每人CI、随时间性能和再校准成本 |
| J03 / P1 / V5 | 模型主要读凝视/低级视觉而非任务注意 | 独立EOG/eye tracking；自然与中央注视、主动与被动条件；数字位置重映射，亮度/字体配平 | EEG/EOG-only诊断、早晚ERP、行为依从性、跨布局；凝视依赖不等同眼电污染 |
| J04 / P1 / V5 | SOA/刺激时长比网络机制更影响效率 | 当前SOA为锚做较快/当前/较慢，被试内顺序平衡；时长与SOA分别改变 | actual correct selections/min、疲劳、错误/拒判、ERP；不能从BI速度外推最优值 |
| J05 / P2 / V5 | 候选间隔和不应期造成有效重复数不足 | 当前完整随机排列与预声明最小候选间隔，对齐次数与总时长；固定/变R分开 | 历史条件效应、hit、置信度、耗时；不允许每个目标隐含不同先验 |
| J06 / P1 / V5 | 离线收益在设备/参与者变化下消失 | 完整shadow/replay到前瞻推断；真实chunk、dropout故障、坏导、跨日；锁定算法与停止规则 | 非空/非重叠UI状态、p50/p95延迟、失败账本、G4；版本更新后重新绑定source/data |

## 13. 数学依据与反例

### 13.1 生成模型与可区分的来源

\[
X_c(t)=\sum_i a_i h_{y_i,c}(t-t_i-\delta_i)+B_c e(t)+d_c(t)+\epsilon_c(t).
\]

`h`可含P1/N1/N2/P3等ERP；`e`是眼电/运动，`d`是慢漂移，`a,delta`由注意/历史/疲劳改变。
同一准确率下降可来自信号删除、噪声减少后分布改变、训练不足或空间信息减少。
所以“去Fp掉点”与“短窗掉点”都需要多对照才能归因。

### 13.2 平均与似然累积

对于固定类别下条件独立的观测，最优Bayes更新为：

\[
\log\frac{P(Y=1\mid x_{1:R})}{P(Y=0\mid x_{1:R})}
=\log\frac{\pi_1}{\pi_0}+\sum_{r=1}^{R}\log\frac{p(x_r\mid1)}{p(x_r\mid0)}.
\]

若二类是共享协方差的Gaussian，单trial LLR是仿射函数 `f(x)=w^T x+b`，于是
`sum f(x_r)=R f(mean x_r)`。score累积并不天然浪费独立噪声下的平方根收益。
神经网络分数不是自动校准的LLR；CE权重/类先验、score相关及错设密度影响概率解释。

对非线性 `f(x)=x^2`，target两次输入`[-1,+1]`、non-target`[0,0]`，
score平均可以分开，输入平均则完全丢失信号。这是否符合真实P300不是反例的用途；
它只否定“非线性保证先平均更优”的一般推论，F02负责真实裁决。

### 13.3 相关噪声与共同偏置

在同方差、等相关、稳定均值差的简化模型中：

\[
\operatorname{Var}(\bar e)=\frac{\sigma^2}{R}[1+(R-1)\rho],
\qquad R_{eff}=\frac{R}{1+(R-1)\rho}.
\]

假定`R=8,R_eff=4.2`才有`rho=0.12925`；不能反过来拿组成不同的R1/R8 hit证明这些前提。
若 `e_{jkr}=mu_{jk}+b_{jr}+epsilon_{jkr}`，且所有候选同覆盖，共同偏置`b_jr`在候选
分数差中抵消。真正有害的可能是候选相关漂移、时延、缺失不平衡，而非任意共享方差。

### 13.4 源头先验与小样本估计

frozen feature矩阵F上，source-centered ridge诊断为：

\[
\hat w=\arg\min_w \|y-Fw\|^2+\lambda\|w-w_s\|^2
=(F^TF+\lambda I)^{-1}(F^Ty+\lambda w_s).
\]

实作用线性求解，不显式求逆；intercept是否惩罚必须声明。它检验保留源先验的价值，
不是自动的最优分类器或LLR。GLM/logistic对应惩罚一般没有这个闭式解。

### 13.5 概率校准与排序的不变性

正仿射 `e'=a e+b,a>0` 保持AUC。每候选恰好R次时：

\[
S'_k=aS_k+Rb \Rightarrow \arg\max_k S'_k=\arg\max_k S_k.
\]

这里Platt概率需转回log-odds，且斜率/偏置由所有候选共用，才对应上述正仿射变换。
任意单调score变换、候选各自不同校准或直接累加概率不保证这个聚合不变性。
因此现有temperature/正斜率Platt的主要检验是NLL/Brier/停止，而不是固定预算argmax。
覆盖不等时偏置项变为`n_k b`，可改变winner；需要完整coverage或显式缺失模型。

### 13.6 物理候选置换反例

固定真实目标A，第二轮把A从抽象group pair(0,0)重映射到(1,1)，EEG仍响应同一个A。
合法系统依赖每轮membership回到A；只按pair变化切selection会把一个目标拆成两个。
反之同一pair也可能在下一轮指向不同物理符号。标签码与binary label相符仅证明二分类，
不证明物理候选身份。应通过真实builder/源日志测试，不能手造同旧假设一致的contract。

本轮在项目`.venv`中实际执行并通过八项小检查：仿射平均恒等式、共同候选漂移抵消、
正仿射校准排序不变、非线性平均丢信号反例、source-centered ridge一阶条件、
等相关R_eff关系、覆盖不等时intercept翻转winner，以及实际MNE epoch FFT重采样的
截止后脉冲影响早期输出。它们是数学/实现反例，不是性能证据。

## 14. 第一批最小可执行安排

1. **审计批次**：A01–A07、I01；补齐原始Header/游戏版本/事件解释的证据。可并行做
   binary标签/时间核对、61/64队列统计与官方经典基线复现。候选身份无证据时不运行F组。
2. **基线批次**：B01–B04、G01的匹配head对照；E0-A和E0-B分别保存配置/结果。
   在小型预声明development子集上验证fit/predict/分母，再扩至完整适用队列；抽样不挑高AUC。
3. **信号批次**：C01–C03、D01–D02；输入短窗从raw重建。ICA依D02及raw可得性进入，
   不把raw/ICA采集成本混入“零训练改动”。
4. **机制定位批次**：G02–G04与合法F01–F03。若target-only/shrinkage已解决主要问题，
   优先量化标签成本；若表征不足才进入E01–E07。
5. **条件拓展**：针对已定位因素安排2×2交互及独立复现；有matched基础后运行H组。
   新成人J组从完整事件记录与真实八导开始，不能等算法完成后才检查采集可用性。

训练量按 `sum(各轨道被试×外层fold×模型×seed×内层候选fit数)` 估计，另加raw构建与
冻结推断；early→later不是自动又乘五折。先实测代表性fit/推断时间与GPU资源，再选择
批量大小，不承诺“单卡几十分钟”。没有必要为这次目录先运行72项或启动云端成本。

## 15. 代码联动与针对性测试

| 变更面 | 必须同步的链路 | 新范式需要的反例 |
|---|---|---|
| 任务/身份 | 原始parser→timeline/metadata→cache/contract→runner→outcome/analyzer→docs | 同物理target跨轮group重排；同pair不同target；未知marker；统一timeline缺字段 |
| split/预处理 | raw时钟→filter/resample→epoch可用时间→split→训练统计→manifest | 截止后脉冲影响早输出；相邻epoch跨fold共享raw；训练fit读取test；chunk状态重置 |
| 主干 | architecture config→全部CLI→checkpoint→load/adapter→统计record | 宽度参数真正生效；n_times/RF一致；head形状/参数count；同输入初始化对照 |
| 校准/QC | prefix split→统计/BN/ICA/校准→predict→失败账本 | 偷看test标签/统计；极小prefix；class不全；QC拒绝抬高conditional值但损坏coverage |
| 聚合 | flash membership→完整batch→loss→score→physical decision→停止 | R1 identity；candidate/真值联合置换；不完整覆盖；unequal-count offset；训练/推理聚合不一致 |
| 统计 | per-trial prediction→participant/replicate层级→paired contrasts→ledger | fold当replicate；missing按0或静默删除；组级bootstrap；SESOI/多重比较/停止规则漂移 |

相关现有测试：`test_bi2014a_candidate.py`、`test_preprocess.py`、
`test_evaluation_contract_builder.py`、`test_candidate_promotion_matrix.py`、
`test_candidate_cross_decision_analysis.py`、`test_candidate_set_loss.py`、
`test_n2p3net.py`、`test_artifact.py`、`test_splitbn_subject_adapter.py`。
实施时删除被证伪合同专属的旧断言，保留仍成立的数值/时间/身份不变量；
测试必须经真实builder产生contract，不能只让新代码和同一假设手造fixture互相印证。

## 16. 暂不采纳与重访条件

- 不采纳“0.73是天花板”“4通道等价xDAWN+LDA”“59%纯噪声”“去额区掉点即眼电”
  “所有机制CI含0即可删除”等绝对归因。相应实验在上表有明确反证路径。
- 不在基线前加set attention、大型EEG foundation model、横向嵌套、多级adapter堆叠。
  只有简单方法留下稳定、可定位、与成本相关的差距，才评估额外结构。
- 不把无candidate metadata的数据集整体废除：可作为binary pretrain/source或分类基线；
  只有candidate loss、physical selection、跨R/停止要求完整相关语义。
- 不把现有BI family普查中的少量样本或解析失败推广成整个家族没有长R；每个候选target
  先查实际可恢复语义与coverage，区别not observed、not available和not identifiable。
- 旧错误合同在证据裁决与替代链闭合后删除，不保留猜测兼容；需要保留的旧源码/结果
  物理压缩归档、校验manifest、标明用途。本次研究目录没有执行源码退役或改写旧结果。

本次产出是可复核的研究地图及72项条件化实验目录。尚无E0/E1新性能结果、成人确认结果，
也没有证明任何候选改进优于当前模型。
