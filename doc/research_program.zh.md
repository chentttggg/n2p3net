# N2P3-Net 科研总纲：从可证伪理念到 GTN 终局验证

日期：2026-08-28
状态：living research guide
审计基线：`research/n2p3-transfer-ssl`，`6547976`；启动正式实验前必须重新记录
`HEAD`、dirty diff、cache SHA-256 和环境锁定信息。

## 0. 本文的权威边界

本文取代 `roadmap.md` 作为未来研究的唯一总入口，但不改写历史实验。
日期化结果文档和原始 `record.json` 保持不可变。信息冲突时按以下顺序裁决：

详细的 2026-08-28 代码/文献证据展开保存在
`deep_model_research_directions_20260828.zh.md`；它是日期化附录，不是第二套契约。

1. 带 SHA-256、完整 folds 和完成标记的原始实验制品；
2. 当前可执行代码与测试；
3. 本文记录的研究状态和下一步门禁；
4. `constitution.md`、`blueprint.md` 等原则文档；
5. 日期化计划、旧配置和历史结果只作审计材料。

模型状态必须使用以下四类词，不再混用“默认”“冠军”“已证明”：

| 状态 | 含义 |
|---|---|
| 软件默认 | 代码当前默认，主要代表兼容性和可运行性，不代表性能最好 |
| 探索领先 | 在已查看的数据上均值领先，可生成新假设，不能作确认性结论 |
| 确认冠军 | 在预注册、未被用于调参的外层数据上通过统计与鲁棒门禁 |
| 部署冠军 | 确认冠军再通过延迟、内存、校准成本和故障行为门禁 |

当前没有“确认冠军”或“部署冠军”。

## 1. 当前证据到底支持什么

### 1.1 已完成事实

- 离线主输入合同为 128 Hz、2--30 Hz、`[-200,800) ms`、V、逐 trial/通道
  `[-200,0) ms` 减均值。原始采集率和参考电极属于 provenance，不因模型重采样消失。
- BI2014a 的 128 Hz head ablation 是完整 64-subject LOSO。该轮 EEGNet
  AUC/BACC 为 `0.73955/0.67541`，MS-EEGNet 为 `0.73484/0.66740`，LMBC 为
  `0.72421/0.65904`。结论仅是 LMBC 在该数据和该合同下未晋升。
- 后续 prior-free ablation 同样完成 64 subjects、61,015 test trials。linear
  `full_unfold` 为 `0.745109/0.677407`，EEGNet 为 `0.739513/0.675382`，
  MS-EEGNet 为 `0.734211/0.667776`。审计重算显示 full-unfold 相对 MS 的
  AUC 差约 `+0.01090`，相对 EEGNet 约 `+0.00560`；后者 BACC 区间跨零。
- quadratic 和同预算 MLP 没有超过 linear full-unfold。现有证据更支持
  “保留时间坐标”而不是“增加非线性或容量”。

### 1.2 仍然只是探索

- BI2014a 的同一 outer test 已被用于 head、epoch、patience、采样率等追加观察，
  因而已经被“花掉”。它可继续做机制诊断，不能再充当独立确认集。
- `full_unfold_k35` 是内层 sensitivity 后注册的候选。仓库没有可独立审计的
  本地 screening records，也没有未查看外层上的确认结果。
- GTN 只有 legacy 256 Hz cache；当前没有 128 Hz causal attested cache、没有
  GTN `hit@R` record、没有预训练 checkpoint、没有 transfer result。
- 现有 SSL、subject adapter、prefix/suffix runner 是原型代码证据，不是性能证据。

### 1.3 不允许的推论

- “LMBC 在 BI2014a 失败”不等于“时延边缘化被否定”。GTN 候选链、重复聚合和
  latency-stratified 反例才有权裁决该机制。
- “full-unfold 在 BI 均值领先”不等于它是 GTN 冠军。绝对时点模板可能在被试间
  latency shift 下失效。
- “AUC 约 0.78”不能推出 `hit@8` 必然超过 0.85；iid 高斯推导只是模型外推，
  不是上界。
- 外部论文的 character accuracy、trial accuracy、AUC、BACC、CRR 和 ITR 不可
  脱离候选数、重复数、目标人训练量和 split 直接排序。

## 2. GTN 最终问题的严格定义

GTN 最终验证包含两个不可混报的 estimand：

1. **GTN cross-subject confirmation**：完整留出目标被试，回答零目标数据泛化；
2. **GTN causal adaptation**：同一被试早期 prefix 到严格后期 suffix，回答少校准适配。

第二项的正式定义为：

```text
train/calibrate = target subject chronological prefix
test            = untouched chronological suffix after a raw-sample embargo
primary         = subject-macro hit@5 under M=5,R=5
secondary       = hit@8 on a predeclared high-repetition eligible subset
curve           = hit@R with the denominator/coverage reported at every R
```

每个目标人都必须计入 requested cohort。数据不足、候选链不全、checkpoint 泄漏、
QC 失败和无法判定都要进入 exclusion/failure ledger；不能从分母删除后只报告成功者。

GTN 是 3 导、学龄人群数据。本地 249 个 TXT 年龄字段范围为 7--17 岁，而项目
mission 面向成人、主要 8 导。因此 GTN 可裁决 GTN benchmark/model mechanism，
不能单独证明 BrainSync 成人 8 导部署有效；后者需要独立目标域确认。

旧 event ledger 在新 `[-200,800) ms` 时间隔离规则下的 schedule-only 可用性为：

| Prefix M | Suffix R | 严格可用 selection | 用途 |
|---:|---:|---:|---|
| 3 | 5 | 130 | 低校准敏感性 |
| 5 | 5 | 101 | 建议 causal-adaptation 主 estimand |
| 5 | 8 | 69 | 高 repetition 次要分析 |
| 8 | 8 | 29 | 选择偏倚大，只作 secondary |

所以 `M=8,R=8` 不能作为全体主分析。若业务必须稳定报告 `hit@8`，应补采更多
repetitions/selection，而不是放宽 raw-sample embargo。

同时报告：

- trial AUC、BACC、group NLL、ECE；
- `hit@1..8`、最差 decile、abstain rate、平均停止 repetition 和 ITR；
- prefix 的绝对 repetitions、trials、字符/决策数和分钟数，而不只报百分比；
- 参数量、batch-1/部署 batch 延迟、峰值内存；
- 3 seeds 的 subject-paired 差值和区间。

术语固定：0 个目标决策为 zero-calibration；1--5 个目标决策为 short adaptation；
使用目标数据 60% 只能称 target-heavy adaptation，不能称 few-shot。

## 3. 当前模型的数学诊断

### 3.1 总感受野，而不是单层核宽

当前主干为 ST temporal conv、pool `P`、两个 MST temporal branch。若共享核为
`K0`，分支核为 `Ks`，输入域总感受野为

```text
R_s = K0 + (P - 1) + (Ks - 1) P.
D_s = 1000 (R_s - 1) / f_s  ms.
```

默认 `K0=65, P=4, Ks={5,17}, f_s=128`，两支路并不是文档曾写的
125/500 ms，而是约 648/1023 ms。长支路已覆盖整个 1 s epoch。K35 候选约为
414/789 ms。K35 的潜在收益首先应解释为恢复局部性，而不是简单减参。

反例：在短支路名义窗口外、但其总感受野内放置脉冲，短支路仍会响应。测试应以
冲激和输入 Jacobian 测真实支持域，不能只打印局部 kernel span。

### 3.2 信息保留不等于泛化

第二级平均池化 `P: R^T -> R^M, M<T` 必有非零核空间，因此不同时间信号可产生
相同池化输出。full-unfold 去掉了这一次碰撞：

```text
h = vec(H),                 z = W h + b.
```

这解释其表达能力，但也引入绝对潜伏期假设。反例：将同一 P300 平移一个 32 Hz
feature bin，所有被试生理形态相同，full-unfold 却读到完全不同坐标。

### 3.3 LMBC 的可辨识性边界

当前 LMBC 用候选窗与 pre-stimulus reference 的均值差：

```text
c_delta = mean(H[t in W_delta]) - mean(H[t in R])
p(delta|H) = softmax(q^T c_delta / (tau sqrt(K)))
z = sum_delta p(delta|H) c_delta.
```

共同常数偏移会抵消，这是优点。问题是宽箱形窗高度重叠：宽平台在每个候选窗中
均值相同，`p(delta)=uniform`，无法识别 latency。窗内伪迹也可以被当成 P300。

因此 LMBC 不是删除项，而是待改写假设：保留生理约束，但把箱形均值改成
形态归一化相关，并让不可辨识 trial 输出高 entropy。

### 3.4 trial 目标与 9 选目标不一致

trial CE 优化

```text
L_trial = -w_y log softmax(z)_y
```

而最终决策使用候选证据

```text
l_i = z_i,1 - z_i,0
LLR_i = calibrated(l_i) - log(pi_train / (1-pi_train))
S_g,d = sum_(i in group g, candidate d) LLR_i
d_hat_g = argmax_d S_g,d.
```

一个错误候选的单次极端伪迹可让高 trial AUC 对应零 group hit。后验 log-odds 只有
减去训练先验后才是可加 LLR；候选 trial 数不等或 dynamic stopping 时常数不会抵消。

若同候选重复相关系数为 `rho`，理想高斯分离度不再是 `sqrt(R)d'`，而是

```text
d'_R = sqrt(R / (1 + (R-1)rho)) d'.
```

`rho=1` 时重复不增加信息。相同边际 AUC 也可因 9 个候选的联合相关结构不同而有
完全不同的 hit rate。因此 AUC->hit 表只能作 sensitivity model。

## 4. 可证伪研究方向

### H1. 按总感受野重建多尺度

精确有限输入支持、LMBC 反例、ERF 协议和 matched 机制矩阵见
[`total_receptive_field_research_20260828.zh.md`](total_receptive_field_research_20260828.zh.md)。

理念：尺度应代表不同的输入域支持，而不是不同的局部 kernel 名称。

数学：先给定目标 `R_short < R_long < T`，再由

```text
Ks = 1 + (R_s - K0 - (P-1)) / P
```

选择可实现的奇数核。共享 `K0` 必须足够短，尺度差异主要放在分支。

反例：默认短/长总感受野几乎都覆盖 P300 全窗，双分支只剩参数复制而非尺度分离。

代码链：`N2P3ArchitectureConfig -> N2P3Net ST/MST -> architecture_record ->
run_n2p3_sensitivity -> GTN runner`。

实验：保留 full-unfold readout，仅比较参数匹配的 K65、K35 和按目标总感受野反解的
两尺度版本。先做 inner-only 3-seed selection，再进入 GTN untouched suffix。

晋级：两个冻结协议各自的 primary `hit@R` paired CI 正向，BACC 不劣，且 jitter
曲线更稳；否则保留最简单者。

### H2. 让训练目标直接看见候选决策

理念：trial detector 只是中间件，最终风险发生在候选组。

数学：在 trial CE 外加入组级 listwise loss：

```text
L_group = -S_g,d* + logsumexp_d S_g,d
L = L_trial + lambda_group L_group.
```

重复相关时可用训练 prefix 估计 compound-symmetry covariance，或对同一 repetition
先聚合再进入 `S`，避免把复制 trial 当独立证据。

反例：给错误候选一个 +20 logit、其余 trial 全部正确。trial AUC 仍高，group loss 会
明确惩罚最终错误。

代码链：`calibration -> candidate/repetition batch sampler -> group loss -> decide -> hit@R`。

实验：同一 trunk 上 `trial CE` 对 `trial CE + group CE`；固定 calibration、candidate
chain 和训练 epoch。先不引入 attention 或更大 backbone。

晋级：group NLL 和 `hit@R` 改善，trial AUC 不显著恶化，且极端 logit 反例通过。

### H3. 多被试有监督预训练优先，联合 SSL 只检验净增益

理念：与 GTN 最接近的强证据来自大规模 P300 有监督跨被试训练和候选聚合，而不是
纯重建。Lee 2020 有 55 人 LOSO session test 加 12 名真实在线新被试；Gao 2021 用
150 人训练、独立 50 人测试。两者都显示零/少校准可行，但依赖多 repetition。

数学主线：

```text
L = L_cls + lambda_t L_mask_time + lambda_s L_mask_space.
```

其中 `lambda_t=lambda_s=0` 的 source-supervised 同骨干是必要对照。纯重建另设一臂，
不能预设为主路线。

反例：重建网络只复原 1/f 背景即可得到低 loss，却不改善 P300 target logit。相反，
只做 supervised 也可能记住 source subject。两者都必须经过 subject probe 和时序打乱
负对照。

代码链：`source cohort -> cross-fitted checkpoint -> SubjectAdapter -> calibrated GTN suffix`。

实验顺序：source-supervised -> supervised+MTCN-style auxiliary -> masked reconstruction ->
组合。每个 target 必须由 checkpoint 明确排除；同一 checkpoint 只有在排除了所有目标
subjects 时才能复用。

晋级：相同绝对 prefix 预算下超过 scratch neural 和 xDAWN-RG；若只有 full fine-tune
改善，结论只能是 target training 有效，不能称 transfer 有效。

### H4. 形态保持的 latency 边缘化

理念：允许 P300 平移，但不让模型在全时轴任意寻找伪迹。

数学候选：

```text
u_delta = H - reference(H)
e_delta = <u_delta, psi(t-delta)> /
          (||u_delta|| ||psi|| + eps)
p(delta|H) proportional to prior(delta) exp(e_delta/tau)
z = sum_delta p(delta|H) e_delta.
```

`psi` 可由训练 fold 的 target/non-target contrast 学得，并限制在合理窗。报告 posterior
entropy、shift 和 jitter，不只报告增强后 amplitude。

反例：宽平台应产生高 entropy；窗外强峰应被拒绝；窗内伪迹若产生低 entropy，则
由 artifact stress test 判负。真实 P300 平移应保持分类而让 posterior shift 同向变化。

代码链：新 readout 与 `full_unfold`、LMBC 并列，不改 trunk；先做合成 identifiability，
再做 GTN latency-stratified paired test。

晋级：GTN `hit@R` 和 jitter robustness 同时改善，且 entropy 能识别不可辨识样本。

### H5. 坐标条件化空间投影与多通道源

理念：3/8/16 导数据不能靠重参考后直接 concat；当前空间卷积绑定固定通道数和顺序。

数学候选：把 EEG 看作头皮连续场，在固定物理锚点 `a_r` 上投影：

```text
V_r(t) = sum_c m_c kappa(a_r,p_c) U_c(t) /
         (sum_c m_c kappa(a_r,p_c) + eps).
```

反例：同时置换信号和坐标后输出应不变；缺一个通道时权重应重归一化。当前固定
Conv2d 不具备这两个性质。

代码链：`EEGDataContract channel_positions -> spatial stem -> shared temporal trunk`。

实验：先比较共同真实通道子集与 dataset-specific stem；坐标 stem 只有在跨 montage
方向一致且 3 导 GTN 不退化时晋级。

### H6. 条件域适配，而不是盲目边际 MMD

理念：GTN 自然 target:non-target 为 1:8；AS-MMD 论文把两域都平衡为 1:1，并且不是
LOSO。直接匹配边际 `P_s(z)` 与 `P_t(z)` 会把纯 label shift 当成域差异。

数学：若使用 MMD，至少比较

```text
MMD(P_s(z), P_t(z))
vs
sum_y omega_y MMD(P_s(z|y), P_t(z|y)).
```

反例：设两域 `P(x|y)` 完全相同，仅类先验不同。正确分类器已经可迁移，边际 MMD
仍非零并可能把表示拉坏。

实验：MMD 仅在 source-supervised baseline 成立后进入；必须保留 no-adaptation、
prior-weighted 和 class-conditional 三臂。AS-MMD 数字不作为本项目期望值。

## 5. 文献路线的批判结论

### 主线证据

- Lee 2020：真实在线零校准 P300，证明“大规模有监督跨被试 + 多 repetition”值得先做。
- Gao 2021：200 人独立训练/测试及短校准回退，支持 invariant supervised pretraining。
- MTCN 2024：支持 supervised 主任务与 SSL auxiliary 的联合消融，不支持纯重建优先。
- EEG2ERP 2025：55 人 P300 数据上证明 few-trial ERP estimation 和 predictive variance
  有价值；它优化 ERP `R2`，不是 9 选分类证据。
- 2026 latency realignment：支持受限时窗、shift/jitter/entropy 联合审计；真实 EEG 仅
  10 人，不能直接证明分类增益。

### 必须保留的对照

- xDAWN-Riemann、Riemannian mean-field、Bayesian signal matching：它们能输出可校准
  score，同样可以做 9 选候选聚合，不能因“非深度”而排除。
- SpellerSSL：提供 time mask + FFT + G=2 calibration aggregation 的机制线索，但只有
  II-A/II-B 两个源被试，94% CRR@7 不能外推 GTN。
- AS-MMD：目标域实际汇集 40 人共 400 标签并随机 trial CV，不是新被试 10-trial。
- dynamic stopping：只在固定模型和 prefix 内选阈值后评估，不能用 suffix 调停机点。

### 暂不采用

- Active Sampling 2024：只报 raw accuracy，六刺激全负基线约 83.3%，并用 test accuracy
  选采样量；不进入主线。
- Adaptive EM-GMM 2026：实验仍需大量目标数据，结果不完整，公式存在可执行性疑点。
- 重型 CST/Transformer：目标人使用 60% fine-tune，且可能随机拆重叠 window；只作
  容量上界，不作 few-shot 证据。
- 通用 EEG FM：在 specialist、身份轴和低频偏置审计前不进入最终模型。

## 6. GTN 分阶段实验

### Gate 0：协议闭环，未过则禁止训练

1. 重建 GTN 128 Hz offline/causal matched caches，记录 source reference、event
   ledger、cache SHA；LOSO 与 causal adaptation 分开报告。
2. prefix 的最大 evidence-available time 必须早于 suffix 最早 epoch start；中间 trial
   自动 embargo，不允许共享原始样本。
3. 内层训练必须“更早 train -> 稍后 validation”，不能随机抽 repetition block。
4. fold-local QC、标准化、校准只拟合 prefix train；suffix 保留在分母。
5. source checkpoint 写出训练 subject keys；任何 target 在 source train 中则整轮作废。
6. 请求 cohort、排除 group、原因、fold->subject、trial predictions 和完整环境写入 artifact。

当前工作树已修正全局 LOSO 覆盖、sensitivity fold 标签、K35 采样率缩放、GTN
timing provenance、LMBC manifest、`None` 分母、无方差 precision 和 checkpoint target
重叠等确定性错误。仍待闭环的是 chronological inner validation、transfer fold-local QC/
校准、独立 subject probe、完整预训练 artifact 和真实 GTN cache。

### Gate 1：GTN floor

在 cross-subject LOSO 和 causal `M=5,R=5` 两个冻结协议上，使用同一 seeds、
同一可比预算运行：

```text
window_lr, xDAWN-RG, EEGNet, MS-EEGNet/full-unfold scratch
```

先建立 `hit@R`、AUC/BACC、校准和最差 decile。深度模型未超过 classical floor 时，
不得进入复杂 transfer。

### Gate 2：架构机制

固定训练器，只比较：

```text
EEGNet
MS-EEGNet K65
full-unfold K65
full-unfold K35 / total-RF design
LMBC
shape-latency head（仅在合成反例通过后）
```

预注册少量 contrasts 并做 Holm 校正。BI 结果只用于确定这些候选，不参与 GTN 判定。

### Gate 3：迁移与校准预算

```text
S0 scratch target-only
S1 source-supervised zero-calibration
S2 S1 + short target adaptation
J1 S1 + temporal/spatial auxiliary SSL
R1 pure masked reconstruction control
```

每个 target 使用 cross-fitted checkpoint；报告 0/1/3/5/8/12 prefix repetitions 以及
对应分钟数。若 S1/S2 已胜，R1/J1 必须证明额外净增益。

### Gate 4：决策、鲁棒性与部署

- sum、mean、trimmed mean；precision 只有存在逐 trial predictive variance 才能运行；
- group CE、correlation-aware aggregation、dynamic stopping；
- +/-10/20/40 ms jitter、channel dropout、幅值缩放、参考扰动、眼动/尖峰注入；
- batch-1 与完整 decision batch 的 CPU/CUDA 延迟、内存、abstain/failure 行为。

## 7. 统计与停止规则

- outer unit 是 subject，不是 trial。bootstrap 和 permutation 都在 subject 级配对进行。
- 至少 3 seeds；先对每个 subject 聚合 seed，再做 paired inference，避免把 seed 当被试。
- 主比较预注册，双侧 paired sign-flip 使用 plus-one 修正；多比较用 Holm。
- 超参数只由 chronological prefix validation 选择。GTN suffix 每个冻结方案只读一次。
- 候选晋升要求：cross-subject 的预注册 `hit@R` 与 causal `M=5,R=5` 的 `hit@5`
  paired CI 正向，BACC 不低于预设非劣界，worst decile 不恶化，且成本/鲁棒门禁通过。
- “达到 85%”可表示 point estimate `>=0.85`，必须同时给 CI；只有 CI 下界也 `>=0.85`
  时才能写“可靠达到 85%”。
- 任一 hash、cohort、fold、target holdout 或 suffix 使用不匹配，整组比较作废，不补跑挑好者。

## 8. 立即执行顺序

1. 完成 Gate 0 剩余项并用短 SOA、missing candidate、错 holdout、重复 fold 反例测试。
2. 生成 GTN causal cache 与完整 cohort ledger。
3. 跑 Gate 1，分别建立 GTN-LOSO zero-target floor 与 causal adaptation floor。
4. 冻结 Gate 2 候选；第一优先是 total receptive field 与 group objective，不再加深网络。
5. 先跑 source-supervised transfer，再检验联合 SSL；纯重建不预设为主线。
6. latency、坐标 stem、MMD 按前置条件逐项进入，不同时改 trunk、loss、head 和聚合。

最终原则：创新先从“要消除哪一种不可辨识性或风险错配”开始；数学公式必须带反例，
反例必须落成测试，测试通过后才允许使用真实 GTN suffix。任何漂亮均值都不能越过这条链。
