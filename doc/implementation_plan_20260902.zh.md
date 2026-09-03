# N2P3-Net 架构重构实施方案：P0 / P1 / P2

_版本：2026-09-02（r2，含反例修正）· 绑定活动规范 `research_program.zh.md`（2026-09-02b）· 决策依据 [`architecture_decision_20260902.zh.md`](architecture_decision_20260902.zh.md) · 数学验证 [`math_verification_20260902.zh.md`](math_verification_20260902.zh.md)_

> r3 修订（2026-09-03 数学闭合复核）：修正 hit@R 的 top-1 repetition 语义；将缺失候选的 `-inf` 与 incomplete 状态门禁分离；补全 `alpha_r` repetition loss、temperature 梯度、residual epsilon/max/source `m=1` 规则；降级横向集成的表达力/塌缩断言；补全 Split-BN buffer 来源与训练期作用边界。r2 的 A-D 修正保留。

本文档给出三项机制（P0 Split-BN、P1 drop-to-consensus、P2 可微决策损失）的完整数学公式、实现代码清单、反例与验收标准。按 P0→P1→P2 顺序实施，每项独立可回退。**实施者：你按此清单推进；总监角色负责契约一致性审查。**

---

# P0：Split-BN 跨域归一化诊断

## 0.1 目标

验证负迁移中 BatchNorm 混合统计失配的贡献，并给出是否采用 per-domain 归一化的裁决。这是最低成本、最可能快速解释 -21.94pp 的一步。

## 0.2 数学

N2P3Net core 的 `st_temporal_norm` 与 `st_spatial_norm` 现在是标准 `nn.BatchNorm1d`，用整个 batch 的统计量归一化：

$$
\hat{x}_i = \gamma \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} + \beta
$$

混合多域 batch 时，$\mu_{\text{batch}}, \sigma^2_{\text{batch}}$ 是跨域混合统计：

$$
\mu_{\text{batch}} = \frac{1}{N}\sum_{d} n_d \mu_d, \qquad
\sigma^2_{\text{batch}} = \frac{1}{N}\sum_d n_d \left(\sigma_d^2 + \mu_d^2\right) - \mu^2_{\text{batch}}
$$

每个域 d 的 trial 实际被错误的均值方差归一化（偏差正比于 $\|\mu_d - \mu_{\text{batch}}\|$ 与 $\|\sigma_d^2 - \sigma^2_{\text{batch}}\|$）。这个失配发生在输入标准化之后，输入层 mean/std 更换无法修正它。这正是输入 mean/std 只恢复 +0.07pp 的原因。

**Split-BN** 保留共享 affine 参数 $\gamma, \beta$，但按域使用独立 running statistics：

$$
\hat{x}_{i,d} = \gamma \frac{x_{i,d} - \mu_d}{\sqrt{\sigma^2_d + \epsilon}} + \beta
$$

其中 $\mu_d, \sigma^2_d$ 只在 domain d 的样本上累积。

## 0.3 实现清单

### 0.3.1 新增 `SplitBN1d` 模块 `src/models/adapters.py`（或新文件 `src/models/splitbn.py`）

（注：N2P3Net 含三处 BN——st_temporal_norm、st_spatial_norm、MST branch 的 BatchNorm1d（n2p3net.py:285）。Split-BN 臂必须三处同换，见修正 C。）

```python
class SplitBN1d(nn.Module):
    """Per-domain BatchNorm with shared affine and isolated running stats.

    domain_indices must be provided at forward time. During training,
    running stats are updated only on rows belonging to the active domain
    index, so mixing multiple domains in one batch never contaminates a
    domain's normalization statistics. At eval time the caller selects the
    domain whose buffers should be used.
    """
    def __init__(self, num_features, *, eps=1e-5, momentum=0.1, domain_count=None):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.domain_count = domain_count  # None => adapt to observed identity list
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)
        self._reset_running_stats()

    def _reset_running_stats(self):
        # per-domain shape (D, C) when domain_count is known
        if self.domain_count is not None:
            self.register_buffer("running_mean",
                torch.zeros(self.domain_count, self.num_features))
            self.register_buffer("running_var",
                torch.ones(self.domain_count, self.num_features))
            self.register_buffer("num_batches_tracked",
                torch.zeros(self.domain_count, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
```

### 0.3.2 注入路径

选项 A（推荐）：`N2P3Net` 增加一个可选 `splitbn_domain_axis` 构造参数，当非 None 时把两个 norm 换成 `SplitBN1d`。默认 None 保持完全向后兼容。

```python
class N2P3Net(nn.Module):
    def __init__(self, ..., splitbn_domain_axis=None):
        if splitbn_domain_axis is not None:
            self._norm_mode = "splitbn"
        else:
            self._norm_mode = "batchbn"
```

### 0.3.3 训练侧 domain 传参

`N2P3Net.forward_features` 需在 splitbn 模式下接收 domain_id。因为 `FeatureDomainTransfer` 是核心入口，让 wrapper 在 splitbn 模式下把 `domain_indices` 传入 core 的 forward。

### 0.3.4 基准对照

| 臂 | 归一化策略 | 目的 |
|---|---|---|
| `B0` | pooled BN（现有） | 基线负迁移 |
| `SBN` | Split-BN per-domain | 检验 BN 失配假说 |
| `SBN+pooled` | 先 Split-BN 再标准 BN | 分离两层贡献 |

### 0.3.5 验收标准

1. 比较必须成对：同 fold、同 seed、同初始化，仅归一化策略不同；报告 subject-macro hit@8 成对差值及 cluster bootstrap 95% CI；
2. `SBN` 成对提升 ≥ +5pp 且 CI 不跨 0，进入"BN 失配是重要因素"的**工程采纳**——这是采纳门槛，不是因果结论：BN 改动同时改变优化轨迹、有效正则化、batch 顺序敏感性与收敛速度，因果归因需要 buffer-swap、顺序敏感性等机制证据链共同支持；
3. 提升 < +5pp 或 CI 跨 0：回退 pooled-BN，方向转向 P1；
4. 反例 17 只能检验推理时 running buffer 的选择：buffer-swap 后结果不变可否定“推理 buffer 选择是主要机制”，但不能否定 Split-BN 在训练期改变 batch 归一化梯度、优化轨迹或正则化的作用；swap 后结果变化也不能单独确认因果。
5. unseen target 的 buffer 来源必须预注册并写入 ledger：`pooled_source`（source 合并统计，estimand 是 source-mixture）、`source_weighted`（按声明权重组合 source buffer，estimand 是该 source composition），或 `target_calibration`（只用 calibration 数据构建并在 test 前冻结，estimand 是 calibration-defined target）。没有 target calibration 时禁止临时创建 target buffer；不同来源必须作为不同 arm。

### 0.4 实施状态（2026-09-03）

P0 代码框架已落地，但尚无真实 EEG 成效结论：

- `src/models/splitbn.py` 实现 `SplitBN1d/2d`；affine 参数共享，source running
  statistics 按整数 domain index 隔离，另维护真实 pooled-source buffer。
- `N2P3Net` 默认仍使用原 `BatchNorm2d/1d`；启用 SplitBN 时同时替换 temporal
  2d、spatial 2d 和全部 MST branch 1d normalization。训练缺 `domain_indices`
  直接失败；eval 缺显式 buffer selector 直接失败。
- unseen-target 来源只接受 `pooled_source`、`source_weighted`、
  `target_calibration`。`source_weighted` 使用
  `sum w_d(var_d + mean_d^2) - mean^2`，权重必须是按冻结 domain vocabulary
  命名的有限、非负、和为 1 的 mapping；`target_calibration` 未显式安装并冻结
  buffer 时不可评估，普通 target forward 不更新任何 running state。
- core 只接收 integer index/count，不持有 domain 名称。通用
  `n2p3_model_source_domain_axis/1` 在 baseline/checkpoint 层绑定 ordered vocabulary、
  `domain_to_index` 和 keyed weights；任一 indexed P0/P1/private mechanism 的 count
  必须与该轴一致。
- checkpoint schema 升为 `n2p3_transfer_checkpoint/2`；normalization mode、domain
  count、buffer source、weights、estimand、model-domain-axis digest 及 P1 wrapper v2
  record 任一缺失或不一致均 fail-closed。旧 `/1` checkpoint 必须重建。
- 当前验证属于 synthetic/contract evidence，只证明隔离、透传和拒绝规则按实现执行；
  SplitBN 是否改善 held-out operational `hit@R`，仍须按 0.3.5 的 paired
  fold/seed 与 participant-cluster CI 实验裁决。

---

# P1：drop-to-consensus 源域残差

## 1.1 目标

修正旧 `DomainAdapterBank` 的 off-manifold 缺陷，让共享 trunk 在训练期被显式训练成跨域共识模型，同时保留每域可吸收采集差异的容量。这是对负迁移机制（共享滤波器被迫取折中面）的直接回应。

## 1.2 数学

设源域 d 有低秩残差算子 $\Delta_d$（与 target adapter 同形状：depthwise + down + up + tanh gate）：

$$
\Delta_d(Z) = W_{up,d} \operatorname{ELU}\left(W_{down,d} \operatorname{DWConv}_d(Z)\right)
$$

域适配特征：

$$
H_d = Z_{inv} + m_d \cdot \rho_d \tanh(a_d) \Delta_d(Z_{inv})
$$

其中：

$$
m_d \sim \operatorname{Bernoulli}(1-p), \qquad p \in [0.2, 0.5]
$$

**关键性质**（drop-to-consensus）：

- 训练时对每个样本以概率 p 丢弃域残差（$m_d = 0$），trunk 单独工作成为显式训练过的合法操作点；
- 目标推理路径取 $\Delta=0$（不 route 任何 $\Delta_d$），与训练期的 $m_d=0$ 路径数值一致；
- 语义澄清：p 只控制 trunk 共识路径被训练的频率，不控制残差幅度；幅度由 $\rho_d\tanh(a_d)\,\|\Delta_d(Z)\|$ 决定，二者不可混谈；
- 训练目标是混合 $(1-p)\,\mathbb E[L\mid m{=}1]+p\,\mathbb E[L\mid m{=}0]$，推理只使用 $m=0$ 分量。drop 消除的是"推理操作点从未被训练"这一缺陷，不等于训练/推理分布一致，也不保证 trunk 收敛到最优 consensus——每个 run 必须分别报告 $L(m{=}0)$、$L(m{=}1)$ 及其差值，泛化结论只能由 held-out hit@R 判定。

**反向推断**：$m_d = 0$ 是 Bernoulli 采样，不能与 $\Delta_d$ 的梯度图分离——$m_d$ 是离散 mask 乘以残差路径，梯度只流经残差路径，$m_d$ 本身不可微。

## 1.3 实现清单

### 1.3.1 `FeatureDomainTransfer` 增加 `source_domain_residuals`

```python
class FeatureDomainTransfer(nn.Module):
    def __init__(self, core, *, config):
        ...
        if config.source_domain_count > 0:
            self.source_domain_residuals = nn.ModuleList([
                FeatureResidualAdapter(
                    self.feature_channels,
                    bottleneck_channels=config.target_bottleneck_channels,
                    kernel_size=config.target_kernel_size,
                    max_residual=config.max_target_residual,
                ) for _ in range(config.source_domain_count)
            ])
        else:
            self.source_domain_residuals = None
        self._source_drop_p = config.source_drop_p  # e.g. 0.3
```

### 1.3.2 `FeatureTransferConfig` 增加字段

```python
@dataclass(frozen=True)
class FeatureTransferConfig:
    ...
    source_domain_count: int = 0      # >=2 时启用 source residuals
    source_drop_p: float = 0.3        # Bernoulli(1-p) 中 p
    source_drop_mode: str = "sample"  # "sample" | "batch"
```

### 1.3.3 训练时采样（修正 B：改用 residual() + 显式 scale）

```python
def source_domain_view(self, features, domain_idx):
    """训练期：按 source domain 采样域残差，随机丢弃。"""
    if self.source_domain_residuals is None:
        return features  # 无域残差，直接返回
    adapter = self.source_domain_residuals[domain_idx]
    # 训练模式：per-sample Bernoulli mask
    if self.training:
        B = features.shape[0]
        m = (torch.rand(B, 1, 1, device=features.device) > self.config.source_drop_p).float()
        # 显式残差路径（不用 adapter.forward，它已包含 features +）
        delta = adapter.scale_tensor(features.dtype) * adapter.residual(features)
        return features + m * delta
    else:
        # 推理：不 route 任何域残差
        return features
```

### 1.3.4 evaluation 与 target inner loop

推理与 inner loop 永远不 route source residual，`source_domain_residuals` 参数冻结。`_target_adapter_enabled` 设置为 False。

### 1.3.5 基准对照

| 臂 | 源域残差 | drop | 目的 |
|---|---|---|---|
| `B0` | 无 | — | no-transfer 基线 |
| `D0` | 有 | p=0.0（永不丢弃） | off-manifold 缺陷复现 |
| `D1` | 有 | p=0.3 | drop-to-consensus |
| `D2` | 有 | p=0.5 | 强置零灵敏度 |

### 1.3.6 反例与验收

1. **反例 15**：trunk 单独路径（全部 Δ_d 置零）必须与推理路径逐位一致。测试：`drop_mask=all zeros` 时 `model.eval()` 下 `forward_features` 与 `forward_with_features` 输出相同；
2. **反例 8**：推理路径必须验证不含 source Δ_d；
3. **反例 10（修正语义）**：`max_residual` 只界定标量门 $|\rho\tanh(a)|\le\rho$，不界定 $\|\Delta(Z)\|$。按样本、在 feature 的非 batch 维度上固定 $\epsilon_{res}=10^{-6}$，计算
   $r_j=\|m_j\rho\tanh(a)\Delta(Z_j)\|_2/\max(\|Z_j\|_2,\epsilon_{res})$，报告 mean/q50/q95/q99/max，并以 max 与预声明 `r_max=0.5` 判定。source residual 检查固定 $m_j=1$ 的 counterfactual，不得让 dropout 零 mask 稀释比例。任一 $r_j>r_{max}$ 时整次 run 记为 `evaluation_failed`/invalid，禁止删除样本后继续汇报；零分母且零分子定义为 0，零分母而分子非零必超限。对 Δ 输出显式 norm-clamp 作为独立 arm。zero-init 位级等价 identity 的要求保留；
4. **验收**：目标 hit@8 不低于 B0，且 `D1` 大于 `D0` — 不能只报告域 probe 下降。

---

# P2：可微决策级损失

## 2.1 目标

把决策层从训练后聚合（后处理）改造成可微模块，让梯度回流到编码器，优化 candidate top-1 ranking surrogate（decision 级 CE）。`hit@R` 是 prefix `r=R` 后一次 9 选 1 top-1 selection accuracy，不是 top-R ranking；candidate NLL 与 0-1 命中可因概率校准与决策边界不同而反向。candidate NLL 下降只能称 surrogate 改善；主指标仍由评估协议报告 operational hit@R，方向不一致时按反例 14 处理。

## 2.2 数学

对每个 training decision，设第 i 个 epoch 归属候选 $c_i$（$c_i \in \{0,\dots,8\}$），其证据由编码器给出：

$$
e_i = f_\theta(x_i)
$$

候选聚合（count-neutral mean，与 `candidate_evidence` 冻结语义一致）：

$$
s_d = \frac{1}{n_d} \sum_{i: c_i = d} e_i
$$

9-choice softmax（candidate posterior，维度由冻结 vocabulary 决定；BrainSync 示例为 9）：

$$
P(d \mid E) = \frac{\exp(s_d/\tau)}{\sum_{d'\in V} \exp(s_{d'}/\tau)},\qquad \tau>0
$$

决策损失（对真候选 $d^*$）：

$$
L_{candidate} = -\log P(d^* \mid E)
$$

给定 membership 与目标候选，$y_i=\mathbb 1[c_i=d^*]$，因此 decision label 与
membership 一起会确定全部 trial binary labels；保留 `L_trial` 是优化稳定性、局部
监督、有限样本正则化和 trial-level 校准的预注册设计选择，不是标签数量导致的数学必然。
若支持完整 repetition 曲线，定义
\[
L_{prefix}=\sum_{r=1}^{R_{max}}\alpha_rL_{candidate,r},
\quad \alpha_r\ge0,\quad \sum_r\alpha_r=1,
\]
并预注册固定预算 arm 的 `alpha_8=1` 与曲线 arm 的 `alpha_r=1/R_max`。

**梯度**（关键推导）：

$$
\frac{\partial L_{candidate}}{\partial e_i} = \frac{P(c_i \mid E) - \mathbb{1}[c_i = d^*]}{\tau\,n_{c_i}}
$$

- 真候选的 epoch：梯度幅度正比于 $(1 - P(d^*))$，判对就几乎不再推
- 错候选的 epoch（如排第二的混淆候选）：梯度幅度正比于 $P(c_i)$，拿最强负梯度

这不同于 per-trial CE：

$$
\frac{\partial L_{trial}}{\partial e_i} = P(\text{target} \mid x_i) - y_i
$$

每个 epoch 的梯度只取决于自身 logit，与"正在和真答案抢第一"无关。

## 2.3 约束

- `λ_candidate ∈ [0.1, 0.5]`：这是**预注册搜索范围**，不是信息论推导的上界。实际梯度比例取决于 temperature、binary pos_weight、candidate count、logit 尺度、decision 间方差与 batch 内 decision 数；"每 decision 1 个标签 vs trial n 个"只说明 L_candidate 必须是加项、不能单独训练（反例 16），不决定 λ 的数值区间
- 总损失：`L = L_trial + λ_candidate * L_candidate`
- 聚合算子第一版固定 count-neutral mean，禁止任意 MLP（小数据下过拟合）
- decision 分组由 batch sampler 提供：batch = K 个完整 decision（K·N ≥ 128，修正 D），decision 边界以 index 张量随 batch 传递
- 若只优化最终预算，预注册 `alpha_8=1`；若训练整条曲线，预注册
  `L_prefix = sum_r alpha_r L_candidate,r` 且 `alpha_r=1/R_max`。count-neutral mean
  不会自动表达 evidence 数量，evidence-aware arm 需预注册 `n_eff=B^2/Q`、
  `s_r=(A/B)*n_eff^beta`（默认 beta=0.5；mean 对照 beta=0）或等价的
  `tau_r=tau_0/r^beta`，不能在 test 上调节。

## 2.4 实现清单

### 2.4.1 新增 `DecisionLevelLoss` 模块（修正 A：绑定冻结 vocabulary，覆盖不完整 fail-closed）

```python
class DecisionLevelLoss(nn.Module):
    """Differentiable candidate-decision loss over within-decision trial groups."""
    def __init__(self, *, num_candidates=9, temperature=1.0):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.temperature = float(temperature)

    def forward(self, logits, candidate_membership, target_candidate):
        """
        logits: (N_trials,) per-trial evidence scores
        candidate_membership: (N_trials,) candidate index per trial (must be in [0, num_candidates))
        target_candidate: () scalar truth candidate
        Returns: (scalar,) decision-level CE, or None if coverage incomplete
        """
        observed = torch.unique(candidate_membership)
        expected = torch.arange(self.num_candidates, device=candidate_membership.device)
        if not torch.equal(torch.sort(observed).values, expected):
            # 覆盖不完整：decision 状态为 incomplete；不产生 posterior/CE
            return None
        # 按候选聚合（强制完整 vocabulary [0..8]）
        scores = torch.stack([
            logits[candidate_membership == d].mean()
            for d in range(self.num_candidates)
        ])
        log_scores = scores / self.temperature
        log_probs = F.log_softmax(log_scores, dim=0)
        return -log_probs[target_candidate]
```

### 2.4.2 batch sampler：group trials by decision（修正 D：K 个完整 decision，K·N ≥ 128）

训练时按 decision 分组采样，batch 由 K 个完整 decision 拼成（K·N ≥ 128），decision 边界以 index 张量传入损失函数。单 decision batch（K=1, N~32）会让 BN 统计高方差漂移（数学验证 PASS #14），禁止。

调用方收到 `None` 时必须写入训练 ledger（decision id、prefix、缺失候选、状态
`incomplete`），不得把该项当作零损失、从 batch 分母删除，或在剩余候选上继续生成
posterior；最终状态只能转为后续 prefix 的 `complete`，或 `wait/abstain`。

### 2.4.3 编码器接入

在 `N2P3NetBaseline`（或 `deep.py` 的 `fit` 循环）中，完成一次前向后：

```python
if self._candidate_loss_weight > 0.0:
    decision_loss = decision_module(logits, candidate_membership, target_candidate)
    total_loss = trial_loss + self._candidate_loss_weight * decision_loss
```

### 2.4.4 基准对照

| 臂 | 损失 | 目的 |
|---|---|---|
| `B0` | 仅 trial CE | no-transfer 基线 |
| `C0` | trial CE + λ·L_decision | P2 单轴 |
| `C1` | 仅 L_decision（λ_trial=0） | 反例 16 拒绝 |

### 2.4.5 反例与验收

1. **反例 14**：trial AUC 提升不能替代 candidate hit@R 提升 — 必须直接报告 hit@R 变化；
2. **反例 16**：`L_candidate` 单独训练不得作为可晋升 arm；
3. **反例 11**：inner loop 不能访问 outer later decision；
4. **验收**：`C0` 的目标 hit@8 不低于 `B0`，且 trial AUC 与 hit@R 方向一致。

---

# 三、实施顺序与依赖

## 依赖图

```
P0 (Split-BN)    →   独立，可先行
P1 (drop-to-consensus) → 依赖 FeatureDomainTransfer 已有模块
P2 (可微决策)     →   依赖 decision 分组 batch sampler，配置到训练循环
```

## 各阶段日志记录要求

### P0
- domain axis 怎么传的
- per-domain 运行统计的 shape
- splitbn 训练 / eval 模式

### P1
- p 值、drop mode、每域残差 scale
- 每域残差 gradient cosine
- 目标 hit@8

### P2
- λ_candidate
- 聚合算子（第一次固定 mean）
- 温度参数
- decision 分组策略

---

# 四、完整测试清单

### P0 测试
- `test_splitbn_forward`: SplitBN1d 正常前向
- `test_splitbn_domain_iso`: 混合域 batch 中各域统计彼此隔离
- `test_splitbn_pooled_equivalence`: 当所有域相同 → 结果与 pooled BN 相同

### P1 测试
- `test_drop_to_consensus_identity`: trunk 单独路径 = 推理路径
- `test_drop_to_consensus_bounded`: 残差 bound 检查
- `test_drop_to_consensus_zero_init`: zero-init 等价于 identity
- `test_source_residual_no_route_eval`: eval 不 route source Δ_d
- `test_source_residual_drop_mask`: 训练时正确应用 mask 采样

### P2 测试
- `test_decision_level_forward`: 决策级损失前向
- `test_decision_level_gradient`: 梯度正确回流到编码器
- `test_decision_level_membership`: candidate_membership 映射正确
- `test_decision_level_bias_invariant`: constant bias 不影响候选排序
- `test_decision_level_temperature`: 温度参数行为正确

---

**实施者**：按 P0→P1→P2 顺序推进，每项独立验证并记录。**总监**：契约一致性审查，确保不偏离活动规范。
