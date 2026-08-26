# N2P3-Net 项目宪法（Constitution）

> 本文档是项目的最高准则。所有代码、架构决策、实验设计都必须服从本文档。
> 当 mission / techstack / roadmap 与本文档冲突时，以本文档为准。
> 修改本文档必须显式记录理由（见第六节）。
> 版本：v12-四对象架构（2026-08-25）。现行唯一架构规范是 `blueprint.md`：
> LatencyMeasurement / RepetitionEvidence / Reliability 双 estimand /
> InnovationAudit+Stopping 四个独立可证伪对象，全部 fail-closed。
> v11 strict-past 架构文档已归档到 `archives/legacy_v11_docs_2026-08-25/`，
> 不再作为可执行规范。本文顶部只描述当前正式口径；历史版本记录不构成可执行规范。

## 一、任务定义

项目同时覆盖两个物理角色：成人 8 导干电极数据是部署目标域；GTN 是确认性主域，使用原生
Fz/Cz/Pz 三导。GTN 在 oddball「猜数字」范式下判定被试心选的数字（1–9，chance ≈ 11.1%）。
成人目标域可记录年龄、性别等元数据（用于条件化与协变量分析）；主任务验收由 GTN 决定。
详见 mission.md。

## 二、不可违背的核心原则

P1 小样本优先。数据 ≤ 数千试次。任何设计决策的第一问是「会不会过拟合」。
   禁止以「模型更大」换取单点精度；复杂度必须服务于泛化、可解释或多成分建模。

P2 准端到端，止步于重采样/高通/切分/剔除。网络之外只允许保留「重采样、连续域高通（默认
   0.1 Hz）、epoch 切分、阈值法伪迹剔除」这些物理/工程对齐与数据质量步骤（高通须在 epoching
   之前做）。参考无关化、幅值标准化必须以可学习/可微形式（加权再参考、InstanceNorm/LayerNorm）
   吸收进网络，不得写成前置手工步骤。

P3 成分感知优先于黑盒。N2/P3a/P3b 等 ERP 成分以「参数化成分窗」显式注入。分类器内部的 `pcw_tau` 只是 routing 参数；生理潜伏期只能由独立的
    LatencyMeasurement 对象（fold-local 白化模板 + amplitude profile likelihood +
    显式规范锚）输出 `measured_tau_posterior`，PCW 消费该后验必须 stop-gradient。
    在 latency S0 合成 gate 通过前，任何 tau 都只能称为「fold-calibrated structured ERP window」。禁止：
    (a) 硬编码时间窗/峰值检测作为特征；(b) 完全无先验的黑盒自注意力作为唯一解码器；
    (c) 事后从注意力统计潜伏期（soft-argmax 副产品，无监督）；
    (d) 把分类 attention 解释为生理潜伏期。

P4 可解释性不可牺牲。模型必须输出成分级候选证据（每个试次的 N2/P3 窗中心、幅值、地形）。
   这些输出必须与已注册且实际生成的组级统计分析相互印证；当前可执行门槛是 PCW claim gate、
   split-half reliability 和完整 audit。命中率是必要指标，但不是充分指标。

P5 格式无关。通道身份用真实三维坐标编码，禁止用通道名字符串或列号作为物理身份。支持任意
   真实通道子集；跨记录布局默认取交集或 strict 校验，不通过补零/替代电极伪造统一布局。
   显式坏导 mask 必须贯穿模型；输入对鼻梁/乳突/平均/Cz 等参考方式的鲁棒性必须实测。

P6 跨域鲁棒必须可审计。域条件仿射（per-domain 可学习 scale/shift）和特征级 RBF-MMD
   只在独立跨域实验中启用；不得把未经验证的 MMD 放进主确认性 Loss。TCN 的归一化是预注册
   消融轴；当前 GTN runner 默认 BatchNorm1d（训练折统计，推理冻结），LayerNorm 为回退。
   任何 BN 统计不得读取测试折。

P7 时域多任务输出。模型必须暴露主分类（target/non-target，决策层累加为 9 选 1）、早期
    证据头（N2，由 τ0_N2+σ_N2 自动限制在早期窗）和 PCW 的 τ/σ/拓扑读数。辅助输出不自动
    进入 Loss：逐试次潜伏期只能由 LatencyMeasurement 对象输出，且在 latency S0 合成 gate
    通过前不训练；PCW 的 `pcw_tau` 只保留为分类 routing 参数。禁止退化为不输出机制读数的
    黑盒单头模型，也禁止把 routing 参数重新包装成生理测量。

P8 先基线，后创新。任何新模块必须先与 SWLDA、xDAWN+RG、EEGNet、EEG-Inception、EEG Conformer
   基线对比，并证明复杂度收益来自跨域泛化/可解释/多成分，而非单数据集精度。基线须含两个
   「免费地板」：手工窗特征逻辑回归、grand-average 模板匹配相关。

P9 辅助数据只预训练/域对齐，主监督与验收只属于主域。其他 P300 数据集（Brain Invaders /
   BNCI008 / ERP CORE 等）只允许以两种方式参与：(a) 预训练特征提取器后，每个 GTN fold
   从该初始化开始、用 GTN 微调；(b) 共享编码器 + 域对齐，分类头仅由 GTN 监督。禁止把辅助
   试次与 GTN 拼接后联合优化主分类 BCE；禁止辅助域梯度更新主分类头/决策层。最终实验情景
   （猜数字）的分类头、fold 协议与验收指标一律由 GTN 决定。细则见 doc/transfer_policy.md。

## 三、工程经验条例（E 系列 —— 从 prior art 的坑中提炼）

E1 干电极漂移是硬伤，但去漂移方式有讲究。Clements 2016 实测干电极 δ/θ 频段功率持续高于湿电极。
   默认用 0.1 Hz 连续域高通 + 基线校正 + InstanceNorm 组合去漂移；**禁止默认 0.5 Hz 高通**——
   Tanner 2015 实测 ≥0.3 Hz 高通会在慢成分（N2/P300/N400）前制造反极性伪峰并衰减幅值，
   0.5 Hz 只作消融对照（Phase 4 负面验证）。

E2 参数化成分窗禁用集合匹配。DETRtime 的 Hungarian loss 依赖「事件边界框」ground truth，而 ERP
   成分无边界标注且会重叠。参数化成分窗（τ 生成 A 的位置寻址）天然不涉及集合匹配，
   禁止引入 Hungarian / bipartite 匹配。

E3 潜伏期标签不可标注，但可以用显式规范锚测量。Depuydt 2023 的教训：真实 EEG 无
    逐试次潜伏期 ground truth。分类器内部的 `pcw_tau` 不足以证明物理可识别；潜伏期
    测量必须由 LatencyMeasurement 对象完成：fold-local 白化模板 + amplitude profile
    likelihood + 显式规范锚（模板固定或生理先验）。S0 合成平移/振幅混淆 gate 全过前，
    任何 tau 都只能称为 fold-calibrated structured ERP window；禁止把窗口内峰值当作
    真实标签监督。

E4 性能优先、容量受控，硬上限 ≤80k。80k 是容量 ceiling，不是要求用满的目标。
   只在预注册验证与消融证明新增模块带来有意义的性能增益或互补信息时扩容；收益不足、不可复现
   或改进困难时保留更小实现。Stage 2 序列编码保留 depth∈{0,1,2,3,4} 消融轴；默认 TCN
   深度以 `src/train/recipe.py` 为准（2026-08-26 定案为 4 层膨胀 TCN，dilation 1/4/16/32，
   depth=3 为轻量对照）。2026-08-20 在旧 50k 上限下，实测 58k 的默认轻量 Conformer 曾被否决；
   该历史结果不再单独构成否决理由，必须在同等训练预算下用性能、校准和跨域验证重新裁决。

E4a 新方法通过替代性验证后，删除旧方法。若新方法在预注册、被试级和锁定评估中证明
正确，并且明确替代旧路径，就应删除旧实现、旧入口、旧测试和旧文档；仅有历史复现价值的
材料移入 archives/。不要为了保留而保留，也不要把已否决方法伪装成默认兼容选项。

E5 时间分辨率在正式 PCW 路径不可被池化抹掉。CNN 窗口化 + 池化会丢失潜伏期信息并造成
   latency-amplitude confound。正式 `attention_softargmax` 读出必须吃全 T 的 Z'；PCW 主分类
   只使用三个成分表示 H，不存在 Head-A 全局/残差/池化旁路。`global_pool`、`maxmean` 和
   `attention` 仅可作为显式 claim-gate 研究对照，不得由正式默认入口启用。
   full-Z2 辅助分类头（`--z2-aux-head add|replace`，`head_z2(Z2)` 与 PCW 相加或替换主 logit）
   同属显式 claim-gate 研究对照：只存在于命名研究 recipe、默认关闭、保留 `logit_pcw` 与
   全部 PCW 读数；进入正式路线必须先通过预注册嵌套 M0/M1 cluster-bootstrap 门槛并修订本条目。

E6 跨域对齐是边际增益，非雪中送炭。AS-MMD 实测跨数据集单试次仅 0.61→0.66，跨域对齐无法抵消
   根本域差（参考/设备/年龄）。跨受试命中率须按物理天花板（~80%）诚实预期；跨数据集单试次
   接近随机，故「猜数字」依赖「每数字多次试次平均」而非单试次跨域。

E7 成分窗须防坍缩（根治，v4 修订）。free attention 的多查询共享 1-bit 监督会坍缩（DETR 系失败模式）。
   参数化成分窗以「位置寻址」替代「内容寻址」——不同成分先验中心 τ0 不同，位置天然区分，
   无需 JSD 散度；只保留进头/进损失的成分（N2/P3a/P3b）。P3b Δτ 界放宽为 [−50,150]ms，
    覆盖真实 P3b 300–600ms；P3a 仍只前移 [−30,0]ms，保留二者下界不重叠。
      v5：GTN（儿童）实测 P3b 峰值 460–490ms，故 GTN τ0_P3b 先验 460ms、τ0 界 [350,600]；
      成人先验 350ms/[280,500] 不变。

E8 元数据是资产，不是噪声。年龄/性别作为 subject metadata 嵌入输入网络，并作为协变量进入
   Phase 4 回归；P300 潜伏期随年龄递增（成人亦然）。GTN（儿童）与成人目标存在年龄域差，
   GTN 定位为「跨年龄迁移源域」而非同源主数据。

E9 辅助预训练权重只是初始化或对齐信号，不是成品。即使使用辅助 P300 预训练，也必须保存
   checkpoint、显式记录层加载/冻结映射，并在每个 GTN fold 完成微调后才能进入主评估；
   辅助域单独精度不得作为主任务验收指标（P9 / transfer_policy.md）。

E10 分类饱和不等于潜伏期已标定。合成诊断实测分类 AUC 可接近 1.0，但 `pcw_tau` 仍可能
      不随真实 latency jitter 协变。v5 的 L_jit 负结果不可继续引用为方法失败的证据：
      ±40ms 三成分共同跟踪在 dtau 界下不可达，且 known_time_shift 零填充存在边界捷径。
      该实验既不能证明“方法不行”，也不能证明“方法行”。后续 latency 验证一律走
      blueprint.md 对象 L 的 S0 合成 gate 与两层真实数据门槛；`lambda_jit=0` 保持默认。

E11 评估必须归纳且锁定。LOSO 中 ERP 校准、早停、二分类阈值、LLR/repetition 温度校准和
      clean/artifact 密度拟合只能使用训练侧被试；测试 logit 中心化 bacc 只能标为
      transductive。GTN 主指标为预注册的 subject-level hit@K（Neural-RIDE 默认
      validation-calibrated `prefix_minK_chain_llr@3`；共享主终点为 `exact_llr@3`；
      报告 K=1/3/5/10/15/all 的 coverage/N）；未就绪 repetition fold 必须保留在 total
      分母，all-trial hit 为次要指标，同时报告训练折数字先验和 count-only 基线。
      confirmatory 至少 5 seeds、统计上未暴露的完整队列并使用一次性锁；架构选择只能在
      development 模式进行。动态停止只允许先做 first-crossing replay（错误率、未决率、
      expected flashes、risk-coverage 曲线）；普通 posterior 阈值不得宣称固定错误率控制，
      anytime-valid 声明必须用通过 `E[e]<=1` 测试的 e-process。

E12 Loss 必须与可识别信息匹配。经过参考和标准化后缺少绝对尺度的表示不能承担单试次微伏
    幅值回归；移动 attention target、参考不一致或固定核宽都不能作为默认物理监督。`L_amp`
    默认关闭，`L_jit` 默认关闭，`L_MMD` 只能作为独立跨域假设验证。

E13 集合级目标必须防泄漏。GTN 的 subject×digit fixed-K 证据集合只能由当前训练/验证分割内部
    构造；外层测试被试不得参与集合损失、校准或超参选择。集合缺失必须记录 coverage，而不是
    用补齐、重采样或隐式改变分母掩盖。条件 repetition 模型必须保留真实 flash 采集顺序；禁止
    独立打乱各数字历史，禁止以质量权重乘 logit 冒充合法 LLR。

E14 异方差必须 faithful 且可校准。均值路径使用独立的复频谱 + Huber 目标；方差 NLL 对残差、
    均值和共享表示停梯度，先均值 warmup 再逐步开启。bootstrap target variance 与 split-half
    reliability 只描述训练折 averaged ERP 目标，禁止称为单试次 latency 监督。部署方差校准只能
    在 subject-disjoint validation 拟合，测试折必须报告覆盖率、Gaussian NLL、sharpness 和
    标准化残差；跨试次聚合使用总预测方差的逆方差权重并记录有效样本数。

E15 四对象分离且 fail-closed。LatencyMeasurement / RepetitionEvidence / Reliability
     双 estimand / InnovationAudit+Stopping 是四个独立可证伪对象；每个对象有独立数据
     契约、gate 和删除规则。跨对象进入主路径必须 stop-gradient；未过 gate 的对象保持
     默认零输出，`final=PCW`。

E16 增量证据必须成对嵌套。判断 L 是否有分类增量，只接受同族模型 `M0:a+bS` 与
     `M1:a+bS+cL`（c 无约束拟合）加 subject-cluster bootstrap CI；推断阶段允许负 c，
     激活阶段必须符号预注册。禁止单参数非负拟合、普通 block permutation 和
     `corr(S,L)` 符号翻转。

E17 概率语义必须可溯源。只有显式二元污染生成模型 + 硬标签 + 部署 prior 存在时才允许
     输出 `clean_probability`；连续 corruption 严重度只能输出 `fidelity`。1-a 插值标签
     不是 P(clean)；soft-BCE 0.9/0.1 不产生 clean 概率。group calibration patch 不得
     直接部署；prior-shift 必须做 odds 换算。

E18 停止规则不得伪装错误率控制。动态停止先报经验 replay；固定错误率声明必须由
     e-process/anytime-valid 方法支撑，且单测验证非负性与 `E[e]<=1`。raw NLL 不得
     直接作可靠性 gate。


## 四、明令禁止（Do Not）

D1 禁止将 LaBraM/BENDR 等大基础模型作为冻结骨干。
D2 禁止用图神经网络（GNN）建模 8 电极图（信息量/参数量比过低）。
D3 禁止在 9 选 1 命中率任务上报告原始试次准确率；一律报告 hit@K / inductive balanced acc / AUC。
D4 禁止仅在单一数据集上做 self-play 评估；必须包含 within-subject、LOSO、跨数据集三层协议。
D5 禁止在成分定位未经生理学检验时宣称「可解释」。
D6 禁止引入 Hungarian / bipartite 匹配作为参数化成分窗的损失（见 E2）。
D7 禁止把窗口内峰值/均值当作逐试次潜伏期真实标签（见 E3）。
D8 禁止默认 0.5 Hz 高通（见 E1，Tanner 2015 失真风险）。
D9 禁止辅助 P300 试次与 GTN 试次混合后直接训练主分类；禁止把辅助试次放入 GTN 测试 fold；
   禁止辅助域标签进入主分类/早期证据损失；禁止以辅助域精度替代主任务验收
   （P9 / transfer_policy.md）。
D10 禁止把未通过可识别性和端点消融的 `L_amp`、`L_tau`、`L_jit` 或 `L_MMD` 当作主确认性
      训练目标；禁止把校准初始化产生的 tau/sigma 宣称为 learned single-trial localization；
      禁止把 PCW attention 输出解释为生理潜伏期；禁止把 `fidelity` 改名为 clean 概率。

## 五、决策优先级（冲突时自上而下裁决）

1. P1 小样本（防过拟合） > 一切「更大更强」的冲动
2. P4 可解释性 > 纯精度
3. P2 / P5 格式无关 > 实现便利
4. P8 基线对比 > 创新叙事
5. E 系列经验条例 > 实现便利（E 与 P 同级）
6. 其余按 P3 → P7 → P6 → P9

## 六、修订规则

- 本文档仅在以下情况修订：实测数据推翻某条原则/经验条例的假设；出现新的硬约束。
- 修订必须记录：修订理由、旧条目、新条目、影响范围，并同步检查 mission / techstack / roadmap 的一致性。

### 修订记录

- v12.2-z2-aux-gate（2026-08-26）：把 E5 claim-gate 研究对照的适用范围显式扩展到默认关闭的
  full-Z2 辅助分类头（`--z2-aux-head add|replace`），用于受控证伪「PCW-only 丢弃窗外判别信息」；
  生产默认仍 PCW-only，进入正式路线必须先过预注册嵌套 M0/M1 cluster bootstrap 并再次修订 E5。
  同时把 τ0/σ/dτ 默认值收敛为 `component_window.py` 的单一 canonical 常量族（成人）与
  `GTN_CHILD_*` 命名覆盖（儿童）。影响：E5、recipe.py、component_window.py、heads.py、
  n2p3net.py、erp_calibration.py、run_n2p3net_gtn.py、run_eeg_loso.py、routes/recipe 文档与测试。

- v12.1-depth4（2026-08-26）：按 depth3/depth4 消融探针结论把 Stage 2 推理深度默认从 3 上调到 4，
  dilation 轴扩展为 1/4/16/32；depth=3 保留为轻量对照。full-cohort 两次运行主决策指标不一致，
  只作方向参考，不构成定案证据。影响：E4、recipe.py、n2p3net.py、encoder.py、
  neural_ride_recipe.md 与对应测试。

- v12-四对象（2026-08-25）：按独立核验后的判决重构指导口径。科学测量（LatencyMeasurement）、
  分类增量（嵌套 M0/M1）、异常检测（fidelity/clean_probability 双 estimand）、序贯停止
  （replay + e-process）拆为四个独立可证伪对象；PCW `tau` 降级为 routing 参数，测量后验
  进入 PCW 必须 stop-gradient；删除单参数非负 fusion、soft-BCE 0.9/0.1 概率语义、L_jit
  旧负结果的证据效力；新增 E15–E18。影响：blueprint、routes、recipe、evaluation、roadmap。

- v11-strict-past（2026-08-25）：删除 residual classifier、ERP subtraction、alpha/ramp 与旧域对齐
  训练入口。PCW 保留唯一神经判别职责；互补证据改为两类别假设下的 strict-past 条件密度，
  结构由独立 audit 被试 fail-closed 选择，融合系数只用独立 validation 被试。learned ERP waveform
  decoder 从正式 recipe 关闭，fold-local 解析类别均值直接进入 likelihood。60 被开发/锁定折未证明
  互补增益后，likelihood 也降为显式 research recipe，正式 v11 为 PCW fail-closed。影响：
  blueprint、recipe、innovation/prequential、baseline、evaluation 和 CLI。
