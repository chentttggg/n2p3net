# N2P3-Net v12 blueprint: 四个独立可证伪对象

> 版本：v12（2026-08-25）。本文档是现行唯一架构规范；v11 的 strict-past
> blueprint、decision record、routes 与 recipe 已归档到
> [`../archives/legacy_v11_docs_2026-08-25/`](../archives/legacy_v11_docs_2026-08-25/)，
> 只作历史复现，不再作为实现或评估依据。
>
> 代码迁移已完成默认切换：生产 recipe 为 `neural_ride_v12_pcw_fail_closed`，
> R/Q/S 默认 fail-closed；`NEURAL_RIDE_V11_LEGACY` 只保留为显式命名的
> 历史对照，任何新代码不得再扩展 v11 语义。

## 0. 总原则

把科学测量、分类增量信息、异常检测、序贯停止拆成四个独立对象。每个对象
有独立数据契约、独立 gate、独立删除规则；谁过不了谁 fail-closed。禁止
`tau/rho/alpha` 一个量承担多种科学含义。

| 对象 | 科学问题 | 输出 | fail-closed 默认 |
|---|---|---|---|
| L: LatencyMeasurement | 单试次 P3b 潜伏期是否可测量 | `q_i(tau)`、均值、90% 区间、entropy | `measured_tau=None`，PCW `tau` 只算 routing 参数 |
| R: RepetitionEvidence | 前缀证据如何合法累加 | `score(c|prefix)` | 纯可加 LLR 主干，state residual 收缩到零 |
| Q: Reliability | clean/artifact 判别与信息保真度 | `fidelity(q)`、可选 `clean_probability(q; prior)` | 无硬标签时禁止输出概率 |
| S: InnovationAudit + DynamicStopping | LLR 有无增量；何时停止 | 嵌套 M0/M1 审计、replay 指标 | `final=PCW`，停止只作经验 replay |

四条铁律：

1. **stop-gradient**：L 的后验进入 PCW 必须 detach；分类 BCE 不得反传进测量分支。
2. **语义分离**：`pcw_tau`（分类参数）与 `measured_tau_posterior`（测量）分字段；
   `fidelity` 与 `clean_probability` 分字段。
3. **增量证据要成对嵌套**：判断某量是否带来分类增量，必须用同一族模型
   `M0` 与 `M1` 做 subject-cluster bootstrap；禁止用单参数非负拟合当增量证明。
4. **先 S0 反例 harness，后开发集**：S0 任一测试不过，禁止进入 8-fold 开发折。

## 1. 数据分区

每个 outer LOSO fold 内按被试分四类角色：

```text
outer train
  +-- optimization: 梯度、模板、白化、质量归一化
  +-- audit: 结构选择、density family 选择、held-out log score
  +-- validation: 校准、阈值、嵌套 M0/M1 交叉拟合
outer test: 只读指标
```

- L 的白化/模板只在 optimization 上估计，validation 只做覆盖/方差校准，test 只读；
- Q 的 `clean_probability` 校准必须做 prior-shift 与未见 corruption-type 双重审计；
- S 的 M0/M1 只用 validation subjects 做 leave-one-subject-out，最终激活条件
  使用 cluster bootstrap，禁止普通 block permutation 当零分布。

## 2. 对象 L：LatencyMeasurement

### 2.1 定义

对 fold-local 训练侧 target trials 建立白化 P3b 模板 `g`，对每个 trial 计算
amplitude profile likelihood：

```math
ell_i(tau) = max_a [-0.5 (x_i - a g_tau)^T Sigma^{-1} (x_i - a g_tau)],
quad
a*(tau) = (g_tau^T Sigma^{-1} x_i)/(g_tau^T Sigma^{-1} g_tau).
```

后验：

```math
q_i(tau) propto pi(tau) exp(ell_i(tau)),
quad tau_i = tau0 + delta_i,  E[delta_i]=0.
```

### 2.2 规范锚（必选其一，不能只用 E(delta)=0）

- **模板固定**：fold-local ReSync/Woody 式受限迭代先固定模板，再以 `E(delta)=0`
  识别个体偏移；或
- **显式生理先验**：GTN 儿童 P3b `460 ± 30 ms`、成人 `350 ± 30 ms`。
- 报告“锚点先验敏感度”；仅报告“初始化敏感度 < 2 ms”不充分。

### 2.3 与 PCW 的边界

PCW 只消费 detached 期望窗：

```text
A_c(t) = sum_tau q_i(tau) * Gaussian(t; tau0_c + tau, sigma_c),
q_i 来自 L 且 detach。
```

分类可以因该输入获益，但不得反传进入 L；L 的训练目标只来自模板匹配、已知
shift 合成校准和覆盖校准。PCW 的 `attention_softargmax` 永远只称 routing，
不得称生理潜伏期。

### 2.4 门槛（两层）

- S0 合成：bias < 5 ms、RMSE < 10 ms、paired slope 0.9–1.1、90% 区间覆盖
  85–95%、初始化敏感度 < 2 ms；
- 真实数据：只有 S0 全过后，才用 split-half 稳定性、区间覆盖、与
  mass-univariate/峰值法的相关性申请；256 Hz 下 RMSE<10 ms 不得直接当真实
  数据硬门槛。
- 通过后输出 `measured_tau_posterior`；不通过保持 `measured_tau=None`。

## 3. 对象 R：RepetitionEvidence

### 3.1 主干：可加 LLR，消费全部正负 flash

```math
S_c(n)=log pi_c + sum_{t=1}^{n} [
  1(d_t=c) LLR_1(e_t,q_t) + 1(d_t != c) LLR_0(e_t,q_t) ].
```

- `LLR_1/LLR_0` 来自每个 flash 的条件密度比，上下文只能由 `(e,q,t)` 决定，
  不含 candidate-specific hidden；
- 候选必须消费**全部** flash：候选自己的 flash 是正证据，其他 8 个数字的
  flash 是负证据；只聚合候选自己的集合是 contract 错误。

### 3.2 状态残差

```math
S_c(n)=S_c^{add}(n)+sum_t delta_t^{(c)},
quad delta^{(c)}=small GRU(e,q,1(d=c)),
quad L_shrink = lambda * sum (delta_t^{(c)})^2.
```

- 残差初始化为零，强 L2 收缩到零；
- 只有它在 untouched audit subjects 的 held-out prequential log score 上形成
  strict-majority 且 cluster CI 下界 > 0 的增量，才允许非零；否则置零。

### 3.3 count 先验的正确形式

- `exact@K`：所有候选计数相同，`beta_0 + beta_1 log K` 在 softmax 中严格抵消，
  禁止进入；
- `prefix_minK`：候选计数 `n_c` 可能不同，合法项是 `gamma f(n_c)`；任何全局
  `log K` 项都非法。
- 计数信息主要进入累计方差/有效样本量/停止策略，而不是 ad hoc 候选先验。

### 3.4 多 K 训练

- 同一累积轨迹在 K=1/3/5/10/15 处取 CE；**禁止 per-K 独立头**，前后缀必须
  由同一条件模型递推一致；
- 主开发终点只使用共同支撑 K=1/3/5；K=10/15 仅作带 coverage 的次要分析；
- 高 K 权重不得再进入主 estimand。

## 4. 对象 Q：Reliability 双 estimand

### 4.1 `fidelity(q)`：严重度/保真度，不叫概率

- 训练目标从 soft-BCE(0.9/0.1) 换成 margin/rank 目标；
- gate 必须覆盖**未见被试 + 未见 corruption type**；
- 输出进入 mixture 时作为收缩权重：

```math
p(e|q,y)=w(q) p_clean(e|y,q,sigma_fid) + (1-w(q)) p_artifact(e|q),
quad
sigma_fid(q)=exp(gamma_0+gamma_y^T q).
```

- `w(q)` 是保真度权重；未建立硬标签概率模型时禁止写 `P(clean|q)`。

### 4.2 `clean_probability(q; prior)`：仅在显式二元污染生成模型存在时启用

- 生成器：`c ~ Bernoulli(pi0)`；`c=1` 为 clean，`c=0` 用预注册 artifact 模型；
  标签为硬 `c`；
- 训练目标可用 log loss 或 Brier（二者均为严格真分数；只允许以有限样本
  稳健性为理由选择，不允许声称“更严格真”）；
- 部署前必须做 prior-shift odds 换算：

```math
odds_deploy(rho) =
odds_cal(rho) * [pi_deploy/(1-pi_deploy)] * [(1-pi_cal)/pi_cal].
```

- 校准只用全局单调映射；按 subject/target/corruption 打的 group patch 不得
  直接部署；
- 单调校准不保证 chain 排序不变：`log(rho A+(1-rho)B)` 对 rho 的单调方向
  逐 step、逐 candidate 可变。因此验收必须包含完整 digit-chain NLL 与决策
  复验，不能只看 rho 的 Brier/ECE/AUC。

### 4.3 删除

删除 `reliability_identification_loss` 的 0.9/0.1 概率锚；删除
“rho gate 通过 ⇒ chain 概率可信”的隐含链条。

## 5. 对象 S：InnovationAudit + DynamicStopping

### 5.1 分类增量审计（嵌套比较）

对每个 validation subject 做 LOSO：

```math
M0: a + b S,      M1: a + b S + c L,   c in R.
```

- S 为 PCW logit（或 Platt 校准后的 S）；
- 报告 c 的 subject-cluster bootstrap 95% CI、strict-majority 改善、AUC
  非劣界；
- **激活条件**：c 符号符合预注册（理论默认 c>0；允许负号必须显式预注册）、
  CI 不跨 0、subject-macro NLL 改善 ≥ 0.5%、AUC 非劣界 −0.005；
- 推断阶段允许 c<0 是为了避免边界选择偏差；跨 fold 显著为负是符号异常警报，
  不是增量证据；
- 禁止 `corr(S,L)` 强制翻转符号；删除 `_fit_nonnegative_fusion_coefficient`
  的非负单参数融合契约。

### 5.2 异常价值审计

- 在 y=0 与 y=1 两个假设下分别计算 predictive typicality：
  `T_i^(y) = -log p_y(x_i)`；
- 用 calibration subjects 构造 per-class conformal p-value；**只有两个假设的
  p-value 都拒绝**才标记 out-of-model 并进入“继续采样”，该标记只影响停止
  策略与 descriptive 权重，不冒充错误率控制；
- raw NLL 不得直接作可靠性 gate。

### 5.3 动态停止

1. 第一步只做 replay：在完整未见序列上回放
   `max_c p_c(n) >= 1-epsilon` 的 first crossing；
   报告已决集错误率、未决率、expected flashes、risk-coverage 曲线；
   阈值只在 validation 拟合，禁止在 test 上挑 K/阈值。
2. 需要 anytime-valid 保证时，再实现可预测 quality gate 的 e-process；
   单元测试必须验证非负性与 `E[e] <= 1`。普通 posterior 阈值不得宣称
   固定错误率控制。

## 6. S0 反例 harness（全部通过后才准接触开发折）

1. `s0_soft_label_semantics`：Brier 与 soft-BCE 的总体最优点都是 `E[y|z]`；
   1-a 只是质量分数，不可能是 clean 状态概率。
2. `s0_count_prior_cancellation`：`exact@K` 下 `beta log K` 对 softmax 响应为
   0；`prefix_minK` 下 ragged counts 会改变排序但该先验无合法生成模型来源。
3. `s0_monotone_rho_chain_rank`：构造单调 rho 校准使候选 chain 排序翻转，
   证伪“rho AUC 不变 ⇒ chain 不变”。
4. `s0_redundant_likelihood_fusion`：DAG `S -> Y <- ?` 反例中，旧非负 alpha
   给出表观 BCE 增益，而嵌套 M0/M1 的 c CI 覆盖 0。
5. `s0_latency_gauge`：全体 tau 平移 +c、模板平移 -c 时似然不变；加锚后
   profile likelihood 通过 bias<5/RMSE<10/slope 0.9–1.1/覆盖 85–95%。
6. `s0_reliability_prevalence_shift`：prevalence=0.2 上校准的检测器在部署
   prior 下 ECE 超限；显式硬标签生成模型 + odds 换算才通过。
7. `s0_stopping_replay`：错设 posterior 阈值违反固定错误率；e-process 非负且
   `E[e]<=1`；replay 指标可计算且不用 test 标签调参。

## 7. 预注册门槛汇总

- Latency：S0 门槛 + 真实数据 split-half/覆盖/相关性；锚点先验敏感度必须报告。
- Fusion：subject-macro NLL 改善 ≥ 0.5%；subject cluster 95% CI 下界 > 0；
  AUC 非劣界 −0.005；c 符号预注册。
- Reliability：未见被试、未见 corruption type、prior-shift、完整 digit-chain
  NLL 四项全过；不允许只报 rho ECE。
- Low-K：主开发终点只用共同支持 K=1/3/5；K=10/15 只做带 coverage 的次要分析。
- Stopping：固定 clean reject 5% 后 risk-coverage 曲线改善，且固定错误率下
  expected flashes 至少下降 2%。

## 8. 与现有代码的落点

| 对象 | 落点 | 改动 |
|---|---|---|
| L | `erp_calibration.py`、`erp_uncertainty.py`、`component_window.py` | 新增 `latency_measurement.py`；PCW 增加 detached 期望窗；`attention_softargmax` 降级为 routing |
| R | `repetition.py`、`losses.py`、`trainer.py` | `RepetitionEvidenceModel` 拆 `AdditiveLLRBackbone` + `StateResidual`；删除 per-K 头 |
| Q | `repetition.py`、`n2p3net.py` audit | 双输出 `fidelity`/`clean_probability`；gate 拆两层 |
| S | `n2p3net.py` fusion、evidence protocol | 替换非负 alpha；新增 `stopping_replay.py`；e-process 后置 |

默认全部 fail-closed：新模块默认权重 0、`final=PCW`；只有对应 outer gate 通过
才启用。

## 9. 删除/废弃清单（S0 与对应 Phase 通过后执行）

- `_fit_nonnegative_fusion_coefficient` 与非负 alpha 融合契约；
- `reliability_identification_loss` 的 0.9/0.1 概率锚；
- `known_time_shift` 零填充在 latency audit 中的使用；
- “attention_softargmax 输出 = 生理潜伏期”的文档/报告表述；
- per-K 独立 scorer、`beta log K` 候选先验；
- 普通 posterior 阈值被宣称为固定错误率控制的任何报告口径。

## 10. 实施阶段与算力

| Phase | 内容 | 通过标准 | 算力 |
|---|---|---|---|
| 0 | 只写 S0 harness，冻结真实开发折 | S0 1–7 全过 | 单 GPU/CPU，小时级 |
| 1 | L 测量 + PCW detached 消费 | S0-5、合成与已有 2-fold latency audit 全过 | 同上 |
| 2 | R 主干/残差拆分 | S0-2/3/4 全过，locked 开发折 | 8 折 `fold_jobs=4` |
| 3 | Q 双 estimand 拆分 | S0-6 全过 | 同上 |
| 4 | S 嵌套审计 + stopping replay | S0-7 全过 | 同上 |
| 5 | 预注册 8-fold 确认 | 第 7 节门槛 | 全部剩余算力 |

任何阶段失败都回 S0，禁止在开发折上调超参。

## 11. 科研叙事

主论文聚焦“可靠性条件化的序贯证据机（reliability-conditioned sequential
evidence machine）”；可辨识潜伏期是独立测量工作或第二篇工作；innovation
要么通过嵌套 M0/M1，要么只作两假设典型性/补采信号。不再把四个对象硬塞进
一个“万能网络”。