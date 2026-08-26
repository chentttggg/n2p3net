# N2P3-Net 路线图

> 本路线图只描述当前可执行的阶段、门槛和未完成工作。历史材料见 archives/README.md。
> 版本：v3（2026-08-24）。
> 路线选择不在本表重复定义，统一见 [routes.md](routes.md)。

## 阶段总览

| 阶段 | 当前状态 | 进入下一阶段的硬门槛 |
|---|---|---|
| Phase 0 数据与格式适配 | 基础链路已通；质量登记和跨数据集基线仍需收口 | 所有数据集统一 tensor、通道 mask、元数据和数据质量表 |
| Phase 1 基线与锁定评估 | GTN seed-0 开发口径已建立，当前完整队列均已开发暴露 | GTN locked replication；为新增或未触碰外部 cohort 建立一次性 confirmatory lock |
| Phase 2 N2P3-Net 主干 | GTN seed-0 已追平 EEGNet；PCW claim gate 未通过 | 端点、loss、RIDE 训练日程完成开发验证并冻结 |
| Phase 3 跨域与辅助数据 | BNCI/ERP CORE/Brain Invaders 已有方向性结果 | 按 T0/T1/T2/T3 的 GTN 主指标显著性判定 |
| Phase 4 可解释性与采集 | 审计接口已具备；生理 claim 未闭环 | PCW gate、ERP 统计、标记时序验证全部通过 |
| Phase 5 消融与报告 | 未开始确认性矩阵 | 每项复杂度有端点归因，报告可复现 |

## Phase 0：数据与格式适配

### 目标

把 GTN、ERP CORE、BNCI008、Brain Invaders 和自有 EEG 统一为 (N,C,T) 试次、标签、
subject/session/run、通道名/坐标、reference、年龄/性别和质量字段。C 由每个 run 的固定真实布局决定。

### 当前规范

- 网络外只做连续域重采样、0.1 Hz 高通、epoch 切分和阈值伪迹剔除；高通在 epoching 之前执行。
- GTN 是 NIX/HDF5，不按 BrainVision 读取；当前缓存口径为 248 个目录中 242 名可评估被试。
- 读 GTN 元数据时用 NIX 内部 subject id 精确匹配 `.txt`，并登记无 thought、重复 subject、全试次剔除等原因。
- 参考、baseline、标准化、通道选择、坐标和 mask 都必须写入缓存元数据。
- 多记录帽型用真实交集或 strict 策略；显式通道列表缺失必须失败。坏导 mask 若启用，必须贯穿
  augmentation、Trainer、reference 和 forward，不能在缺失位置添加噪声。

### 验收

- 三类异构数据可加载，并能复现当前缓存形状和质量分母。
- `sfreq/tmin/T` 能推导 baseline 范围；无刺激前 baseline 的数据集有显式替代策略。
- 坐标通道身份、mask、metadata、NIX 多 `.txt` 匹配和零试次登记有集成测试。

## Phase 1：基线与锁定评估

### 目标

在 GTN 建立 SWLDA、xDAWN+RG、EEGNet、EEG-Inception、EEG Conformer、窗口特征逻辑回归和模板匹配的统一对照，冻结 LOSO 和 subject-level 终点。

### 当前口径

- 主终点：validation-calibrated `exact_llr@3` ITT hit；显式报告 exact/prefix/flash/all 与 coverage/N。
- 次指标：trial AUC、inductive balanced accuracy、Brier/ECE/LLR reliability、count-only 和 digit-prior 基线。
- 训练侧拟合 `pos_weight`、二分类阈值、ERP 校准和 Platt LLR；外层 test 只使用一次。
- 被试级验证、早停、best-weight restoration 和所有模型必须共用同一 folds/seeds。

### 当前结果

开发口径的 GTN 242 被试结果：GLM v3 hit `0.8388`、AUC `0.7543`；EEGNet hit `0.8395`、AUC `0.7620`；Inception/Conformer hit `0.8512`。这些是 seed-0 复核结果，不能作为最终确认性声明。

### 锁定复现与确认性运行门槛

- 当前 GTN 242 个 model-ready 单位只做 locked development/replication；完整 ITT 分母仍为 245，
  其中 3 个无可用 epoch 的单位计 miss。至少 5 个 unique seeds 用于复现和方差估计，但不恢复
  confirmatory 身份。
- 真正确认性运行必须使用新增受试者或从未查看过的外部 cohort，并在首次 test 前创建一次性
  `confirmatory_id` 锁；当前 GTN manifest 的 confirmatory 入口必须 fail-closed。
- 每次运行记录 cohort hash、缓存 hash、配置、训练/验证/test 分割、primary metric 和 compute budget。
- 不能使用日期评审中的旧 raw accuracy、test-side calibration 或固定试次数补齐规则。

## Phase 2：N2P3-Net 与 Neural-RIDE

### 目标

保留细时间分辨率和 ERP 先验，在 80k 参数硬上限内完成正式 PCW 路线，并把 strict-past
条件密度作为隔离的研究候选对照。两条路线都直接对齐 GTN 的重复证据排序；性能优先，
新增容量必须通过预注册消融证明有效，否则保留更小模型。

### 当前结构

- Stage 0：恒等初始化的门控再参考、数据集感知 baseline/normalization、坐标和 metadata embedding。
- Stage 1：核长 `{13,33,65,129}` 的 stride-1 时间卷积、空间深度卷积、带通初始化和缺失 mask。
- Stage 2：TCN depth `{0,1,2,3}` 消融，默认 BN；参数化 N2/P3a/P3b 窗保留全时间轴。
- Stage 3：PCW head 与 N2 early head；PCW 是唯一神经判别 logit，不存在 global/residual bypass。
- 正式路线固定为 PCW-only；strict-past likelihood 只在 `--lambda-innovation > 0` 的登记研究路线中出现。其结构为两条类别假设历史、fold-local VAR(32)、在线 AR(1) 和 5 层因果 depthwise-separable TCN（k=9，dilation 1/2/4/8/16，R=249）。
- 研究分支协方差为 `D+UU^T`、rank 1--2，并用 Woodbury/Cholesky NLL；缺失通道必须在假设历史、VAR/TCN、Gaussian marginal 和 audit 中使用同一显式 mask。训练/audit 使用 observed-scalar normalized NLL，序贯证据使用未中心化的 time-summed LLR。六个候选构成固定 DAG，动态低秩末端必须同时优于动态对角和静态低秩两个 parent。零输出不能继承 parent 增益，正负 ERP 泄漏也不能在 pooled mean 中抵消。
- 研究融合使用 validation-subject LOSO cross-fit；60 被开发 5 折仅 1/5 开启，随后锁定 5 折 0/5 开启；加入 direct-parent gate 后的 folds 16--20 是 density eligible 0/5、融合 0/5，加入最终逐类复均值 gate 后的 folds 21--25 仍为 0/5、0/5。因此 outer claim gate 失败，正式路线的 `logit_final = logit_PCW`，研究路线也在失败 fold fail closed。
- learned ERP waveform decoder 已从正式 recipe 关闭。strict-past 研究 recipe 使用 fold-local target/non-target 解析均值构造两个 likelihood 假设，避免标签未知时的单试次 subtraction 循环。
- GTN 集合损失按 subject 构造 ragged online sequence，并在真实采集 checkpoint
  K={1,3,5,10,15} 上监督；各 K 独立 coverage，缺高 K 不得移除低 K 被试。
- repetition 模型专属指标为 validation-calibrated `prefix_minK_chain_llr@3`，最低 availability 0.90；未就绪折
  保留在分母。K15 因全量缓存结构 coverage 约 29%，仅作次要分析。

### Loss 默认与候选

- 当前可辩护对照是 fold-weighted `L_target`，`L_PCW` 可作为结构支路约束。
- `L_early`、`L_digit`、`L_recon` 是候选，必须由 subject-disjoint validation 选择。
- `L_amp` 和 `L_jit` 默认 0；`L_tau` 在 PCW 冻结/校准口径下关闭；`L_MMD` 只在独立跨域实验使用。
- 成分重建如启用，均值使用复数分频误差 + 归一化 Huber；方差使用 faithful NLL，先 5 epoch
  均值 warmup 再 10 epoch ramp。profile 记录类别分层 bootstrap target variance 与 split-half
  reliability；禁止回到 raw waveform MSE，禁止把 averaged target 称为单试次 jitter 监督。

### Phase 2 门槛

- 主端点不劣于最强基线，并以 paired subject-level 统计和实际效应量报告。
- PCW claim gate 的 synthetic shift、gradient health、real split-half、fixed-window/mean-pool 四项全部通过；否则保留结构化窗口但降低表述。
- 所有 RIDE 新增项完成独立消融，不能因为组件写进 blueprint 就视为已验证。
- strict-past 若要晋升正式路线，必须单独满足 [routes.md](routes.md) 的 locked outer gate；通过后若新方法已替代旧方法，应删除旧实现和旧入口，不为兼容而永久保留。

## Phase 3：跨域与辅助数据

### 允许方式

- T0：GTN from scratch。
- T1：辅助域监督预训练后，逐 GTN fold 微调。
- T2：辅助预训练后冻结骨干，只在 GTN 训练 head。
- T3：共享编码器 + 域条件参考/对齐，GTN 独占主监督。

辅助域不得进入 GTN 主分类头、GTN test fold 或测试侧调参。T1/T2/T3 只有在 GTN 主终点相对 T0 显著改善时才可采用，否则回退 T0。

### 已有方向性证据

- BNCI008 修正标准化后，N2P3-Net 与 EEGNet 在 bacc/AUC 基本打平。
- Brain Invaders 16 导干电极的 60 被试结果中，N2P3-Net 在 AUC 上显著超过 EEGNet。
- 早期 60 被试辅助预训练/冻结/MMD 试点未达显著，不能据此宣布辅助数据有普适增益。

### 下一步

- 先冻结 GTN-only 配置，再以同 folds/seeds/早停规则重跑 T0/T1/T2/T3。
- 注册坐标上的 chordal Matérn GP canonical projection、faithful posterior covariance、rank-8 adapter、
  shared/private heads 和单优化器交替 trainer 已实现；下一步不是继续改结构，而是锁定配置后跑
  leave-one-dataset-out、zero-shot、30/60 秒无标签校准和随机 channel masking。
- 比较 per-domain gated reference、REST 与 GP canonical 的跨参考/跨帽型方案；固定 montage、
  坐标映射和 arbitrary-layout 必须分开报告。
- 按年龄、reference、通道数和设备报告 transfer 结果，避免把单一辅助域的收益外推到成人自有数据。

## Phase 4：可解释性与采集有效性

- 使用 [`evaluation_protocol.md`](evaluation_protocol.md) 的 PCW claim gate，不以 attention 图或校准后的 tau 数值直接宣称生理定位。
- 输出 tau/sigma、成分拓扑、有效参考权重和 P300 audit；统计扩展必须先注册并产出独立分析
  artifact，当前正式对照为 PCW claim gate 与 split-half reliability。
- 在内层 subject-disjoint validation 拟合 ERP 方差缩放，外层 test 报 Gaussian NLL、sharpness、
  标准化残差 RMS 与 50/80/90/95% coverage；跨试次 ERP 用总方差逆权重聚合。
- 对自有 BrainSync 数据做样本级 marker 时钟校正，并用光电二极管或音频 onset 验证 EDF annotation 延迟分布。
- 检查 1/f、subject identity、频带干预和 sham control；只有干预同时影响主端点时才考虑修改频谱 Loss。

## Phase 5：消融与报告

按一次只改一个主要因素的嵌套顺序运行：

1. `L_amp=0`、PCW freeze、`L_early=0`、fold-local `pos_weight`。
2. fixed-K/listwise CE 与 trial BCE 的主次关系。
3. reference gate、bandpass init、BN/LN、TCN depth、VAR/神经均值和 diagonal/low-rank likelihood 候选。
4. 通道数/mask、baseline 策略、0.1/0.5 Hz、高通和参考抖动。
5. 在 GTN-only 最优配置冻结后单独加入 MMD 或辅助预训练。

最终报告必须同时包含：主次指标、NLL/ECE、`exact_llr@3`/`exact_llr@5`/`exact_llr@10`/`exact_llr@15`、coverage、bits/selection、在有实测
采集耗时时的 bits/min ITR、达到固定错误率所需 repetition、配对统计、seed 变异、参数量/运行
成本、成分审计、失败被试分析、消融归因和与 constitution 的一致性说明。
“SOTA”只能在预注册多数据集、多 seed 外层测试完成后按主指标声明；通过 prequential audit 只证明概率结构合格，不等价于性能领先。

## 里程碑

- M1：数据分母、缓存和基线协议锁定。
- M2：Neural-RIDE 开发配置通过端点与 PCW gate，或明确降级为 calibrated structured window。
- M3：跨域 T0/T1/T2/T3 得到可重复的 GTN 主任务结论。
- M4：自有成人 8 导采集时序验证和最终统计报告完成。
