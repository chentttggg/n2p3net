# N2P3-Net 锁定评估协议

> 本协议在任何 backbone、Loss 或 Neural-RIDE 结构确认性改动前冻结。开发实验可以使用子集和多个候选配置；确认性实验必须使用统计上未暴露的完整队列、subject-disjoint folds 和一次性锁。
> 版本：v4（2026-08-25，v12 四对象架构评估门；四个对象 gate 以
> [`blueprint.md`](blueprint.md) 为准，预算语义仍冻结）。

## 1. 任务和数据分母

- GTN 是主域：248 个源目录按独立 eligibility manifest 冻结为 245 个 ITT 单位，其中当前预处理
  cache 有 242 个单位具备模型证据、3 个全 epoch 不可用单位仍作为 ITT miss。重复 NIX subject
  与缺 thought-number truth 的源记录在 manifest 中显式排除，不能由某次 cache 是否加载成功决定。
- 完整 242 个 model-ready 被试的 headline/fold 结果已经参与结构与 gate 决策，因此当前 245-unit
  GTN 队列整体属于 development-exposed；不能通过事后切一个内部子集恢复 confirmatory 身份。
- trial 输入必须带 `subject_id`、`run/session`、digit、target/non-target、通道名、reference、年龄/性别和可用通道 mask。
- cache 必须逐 scheduled stimulus 保存事件账本；可用 epoch 以 `evidence_index>=0` 双射到模型张量，
  `artifact_rejected`、`boundary_dropped`、`missing`、`acquisition_rejected` 事件保留在账本并令
  `evidence_index=-1`。旧 cache 或只含 observed epochs 的伪时间轴不得用于正式评测。
- 9 选 1 的 chance 仅作理论参考；GTN 非均匀 thought-number 分布还要报告 fold-local digit-prior 与 count-only baseline。
- 其他 P300 数据只能按 [`transfer_policy.md`](transfer_policy.md) 做预训练、冻结对照或域对齐，不能进入 GTN 测试 fold 或 GTN 主分类头。

## 2. 泄漏边界

- LOSO 外层 test subject 不参与 ERP 校准、baseline/normalization 统计、pos_weight、阈值、Platt LLR、
  repetition 温度、reliability/fidelity 参数、早停、集合损失构造或超参选择。
- LatencyMeasurement 的模板/白化只由 optimization subjects 估计；PCW 消费测量后验必须
  stop-gradient；分类 BCE 不得反传进测量分支。
- 所有 calibration/statistics 只由当前外层训练 subjects 的内层 subject-disjoint validation 拟合；保存的 full-cache calibration JSON 不能直接进入 LOSO。
- 二分类 balanced accuracy 使用训练侧学习到的阈值；测试侧中心化结果只能标为 `transductive_balanced_acc`。
- 模型时间边界使用显式毫秒，并满足 `1000 * n_time / sfreq` 在一个采样点内一致。
- N2P3-Net 与 EEGNet/Inception/Conformer 共享 subject-disjoint validation split、patience 规则和 best-weight restoration。
- 集合级 `subject x digit x K` 证据只能在对应训练/验证分割内部构造；缺失数字或不足 K 必须记录 coverage，不能从 test subject 补齐。

## 3. 主终点和报告表

跨模型确认性主终点默认为 `exact_llr@3`。Neural-RIDE 的模型专属生产终点可以另行预注册为
`prefix_minK_chain_llr@3`，但不得与 `exact_llr@3` 混称或直接配对。每个模型都必须报告：

| 类别 | 指标 |
|---|---|
| 共享主终点 | `exact_llr@3` 的 ITT hit、conditional hit、coverage/N |
| 模型专属终点 | `prefix_minK_chain_llr@3`、coverage/N（最低 availability 0.90） |
| 证据累积 | 显式 `exact_*`、`prefix_minK_*`、`flash_*`、`time_*` 与 `all_*`，不得使用 `llr@K` 等旧名 |
| 单试次 | trial AUC、inductive balanced accuracy |
| 校准 | Brier、ECE、LLR reliability；校准只用训练侧验证被试 |
| 效率 | 固定 K 错误率、达到预注册错误率所需 K、Wolpaw bits/selection；有实测 repetition duration 时报告 bits/min ITR |
| ERP 不确定性 | Gaussian NLL、sharpness、标准化残差 RMS、50/80/90/95% coverage；方差缩放只用训练侧验证被试 |
| 基线 | digit-prior、count-only、SWLDA、xDAWN+RG、EEGNet、Inception、Conformer、window LR、template |
| 机制 | 正式路线报告 PCW；strict-past 研究路线另报四个对象：`measured_tau_posterior`、repetition backbone/state 分项、`fidelity`/`clean_probability` 分项、嵌套 M0/M1 审计与 stopping replay |

预算语义冻结如下：`exact@K` 对每个候选恰取最早 K 条可用证据；`prefix_minK@K` 取截至所有候选
均获得 K 条可用证据之时的全部可用前缀；`flash@N` 取前 N 个 scheduled flashes，因此被拒事件
仍消耗 flash 预算；`time@Ts` 只允许在线因果预处理，并按 evidence availability timestamp 截止；
`all` 使用全部可用证据。每个预算独立执行 coverage 处理再 argmax。chain score 不做
测试侧中心化：先去除 `log(pos_weight)+logit(pi_train)`，应用内层验证温度，再按真实全 flash
顺序对每个候选执行概率链式法则。score 后 z-score 只能用于展示。
达到固定错误率所需 K 必须同时满足预注册 minimum coverage；它是固定预算曲线的描述统计，
不能在 test 曲线上挑 K 后称为在线自适应 stopping。ragged/rejected 数据禁止以
`K * repetition_duration` 推导 ITR；bits/min 只能由在线因果 `time@T` 和事件级时间戳得到。

冻结全集记为 `U`，availability 与命中分别为 `A_u`、`H_u`。所有 score 文件必须对每个
`u in U` 恰有一行；不可用时 `predicted=null, available=false, hit=0`。正式主结果为
`ITT accuracy = sum(A_u H_u)/|U|`，并另报 `coverage=sum(A_u)/|U|` 与仅用于诊断的 conditional
accuracy。任何缺 ID、重复 ID、未知 ID、fold 重叠、被试泄漏或 LOSO 非唯一完整覆盖均立即失败。

### 研究依据与边界

- [ASAP](https://arxiv.org/abs/2203.07807) 证明了字符级 Bayesian accumulation 的价值：每次
  flash 后更新全部候选，联合使用 target 与 non-target 证据。Neural-RIDE 保留这一候选级更新
  原则，并进一步以条件概率链处理跨 flash 依赖。
- [SpellerSSL](https://arxiv.org/abs/2509.19401) 的 `G=2` 是按 stimulus code 对齐后，对相邻
  repetition 做重叠滑窗平均；它只用于 calibration/training，在线推理不做聚合。其结果支持适度
  平均可改善 SNR/ITR，但不构成 trial 可靠度、伪迹污染或跨 repetition 依赖模型。因此 `G=2`
  只能作为独立数据增强消融，不能替代 `rho`、条件密度或 chain LLR。
- 截至本协议冻结，问题2 strict-past 条件密度分支没有通过 outer claim gate，问题3 repetition
  分支也没有完成相对共享 `exact_llr@3` 基线的未暴露配对优势验证。二者的代码完备性、NLL
  改善或模型专属 chain 指标都不能表述为总体性能提升，更不能表述为 SOTA。

## 4. 统计比较

- 每个 subject 的 hit/miss 是配对二元结果，模型比较使用 exact McNemar。
- hit 差、AUC 差和 calibration 差使用 subject-level paired bootstrap 95% CI。
- 多 seed 时 subject 是抽样单元，seed 是重复测量；不能把 `subject x seed` 当成独立样本。
- 预先登记最小实际效应量和劣效界；不能只凭单次均值差或单次 p 值改默认架构。
- 所有结果保留完整逐被试记录，包含无法评估的 subject 及原因，避免只输出可用被试汇总。

## 5. 开发与确认性模式

### Development

允许使用 12/30/60 被试、单 seed 和候选超参。用途是发现数据链 bug、估计方向和选择下一轮消融；结果不得写成最终性能声明。

### Confirmatory

必须满足：

1. 使用全新、从未查看过的外部队列或新增受试者；当前 GTN manifest 将
   `confirmatory_status` 标为 unavailable，正式入口必须直接失败；
2. 至少 5 个 unique seeds；
3. 唯一且一次性使用的 `confirmatory_id`；
4. 配置、cohort hash、cache hash、primary metric 和预算在运行前锁定；
5. 外层 test 只执行一次，不根据 test 结果改结构或阈值。
6. 神经网络确认性运行显式使用 CUDA；禁止因设备不可用静默回退 CPU。

每个 seed 的 score 必须绑定 model、seed、evaluation mode、dataset hash、protocol hash、metric、
cohort hash 和有序全集。跨 seed 汇总要求全集及顺序完全一致，禁止取交集。单 seed 模型比较使用
exact McNemar 和 subject-paired bootstrap；比较文件还必须具有相同 seed、dataset、cohort、mode
与 metric。

确认性入口只接受尚未暴露的 eligibility manifest。下面的 GTN 命令因此应当 fail-closed，
它是验证锁和 cohort 身份保护是否生效的负向 dry-run，而不是可启动的正式实验：

```powershell
.\.venv\Scripts\python.exe experiments/run_locked_multiseed.py `
  --runner n2p3net --mode confirmatory --confirmatory-id gtn-final-v1 `
  --output-dir experiments/runs/gtn-final-v1 --dry-run -- `
  --epochs 30 --batch-size 256
```

不得对当前 GTN manifest 移除 `--dry-run` 或绕过失败。GTN 的 242 个 model-ready 单位可以按相同
配置完成 >=5 seeds 的 locked development/replication，用于估计方差和复现工程结果，但不能改称
confirmatory。只有新增受试者或未触碰外部 cohort 的独立 manifest 才能创建正式一次性锁；运行前还
必须人工确认配置、cohort/cache hash、primary metric 和计算预算。

## 6. LatencyMeasurement gate（替代旧 PCW claim gate 的潜伏期语义）

只有全部通过 S0 合成门槛，才允许输出 `measured_tau_posterior` 并使用
`measured single-trial ERP localization`：

1. Synthetic known-shift recovery：bias `<5 ms`，RMSE `<10 ms`；
2. paired slope `0.9–1.1`；
3. 90% 区间覆盖 `85–95%`；
4. 初始化敏感度 `<2 ms`，并报告锚点先验敏感度；
5. 振幅混淆测试：P3b 平移 +40 ms 且振幅×0.9 时，τ bias `<5 ms`、振幅 bias `<10%`。

真实数据另用 split-half 稳定性、区间覆盖、与 mass-univariate/峰值法的相关性作第二层
门槛；不得把 256 Hz 下 RMSE<10 ms 直接设为真实数据硬门槛。

PCW `attention_softargmax` 输出只称 `pcw_tau`（routing 参数），不进入生理潜伏期报告。

## 7. Loss 和模型比较的附加规则

- 先建立正式 PCW 路线的 fold-weighted trial BCE 对照，再单独登记加入 `L_early`、`L_digit`、条件 NLL 或 strict-past likelihood；可选 `L_recon` 不得进入正式 logit。strict-past 研究结果不得回填为正式路线结果。
- `pos_weight` 必须逐外层训练折精确取 `n_neg/n_pos`。若不是精确比值，raw weighted-BCE logit
  不得称为 LLR；必须显式去除权重和训练先验偏置，温度只能在内层验证被试拟合。
- repetition set 必须保存真实采集顺序；只允许 shuffle 完整 set，禁止独立打乱各数字的历史。
  每个 K 独立计算 ragged coverage；未通过 reliability gate 的折的 chain 预测只进入
  descriptive diagnostics，不得进入正式 `prefix_minK_chain_llr@K` 的 covered 记录，但该折
  必须保留在 ITT 分母中，禁止以删除 unavailable fold 的方式把 coverage 伪装为 100%。
- Reliability 双 estimand：`fidelity(q)` 必须通过未见被试 + 未见 corruption type 的排序 gate；
  `clean_probability(q; prior)` 只在显式二元污染生成模型 + 硬标签时启用，必须通过 prior-shift
  odds 换算与完整 digit-chain NLL。单调校准不保证 chain 排序不变，因此校准验收不能只看 rho 的
  Brier/ECE/AUC；group calibration patch 不得直接部署。
- Fusion 只接受嵌套 `M0:a+bS` vs `M1:a+bS+cL`（c 无约束）+ subject-cluster bootstrap CI；
  激活条件：c 符号预注册、CI 不跨 0、subject-macro NLL 改善 ≥0.5%、AUC 非劣界 −0.005。
  禁止单参数非负 alpha、普通 block permutation、`corr(S,L)` 符号翻转。
- 动态停止先报 first-crossing replay：已决集错误率、未决率、expected flashes、risk-coverage
  曲线；普通 posterior 阈值不得宣称固定错误率控制，anytime-valid 声明必须通过 e-process 的
  `E[e]<=1` 单测。
- 汇总级 primary metric gate 不再中止实验或删除汇总：chain primary 未达到最低 availability
  时，结果继续保存并标记 `primary_metric_gate.claim_eligible=false`、`effect=
  descriptive_only_no_result_suppression`；下游配对/多种子报告必须原样携带该标记，不得把
  描述性结果改写成正式主结论。正式主指标最低 coverage 由独立的
  `primary_min_coverage` 控制，不得复用效率曲线的 `efficiency_min_coverage`。gate 失败折的
  逐被试预测、候选 scores 与 reliability 必须另存为 descriptive records，不能覆盖 formal
  unavailable 记录。
- repetition efficiency 必须显式记录并匹配 primary metric 的 `aggregation` 与
  `budget_semantics`；禁止在 chain/prefix 主指标下静默回退并报告 trial/exact 曲线。
- `L_amp`、`L_jit` 默认关闭；`L_tau` 不能被解释成真实 latency 监督；`L_MMD` 只能在 GTN-only 配置冻结后单独加入。
- ERP bootstrap target variance 和 split-half reliability 只由外层训练折估计且只描述 averaged
  target；部署方差缩放只在内层验证拟合，禁止 test-side calibration。跨试次 ERP 聚合须使用
  aleatoric+epistemic 总方差的逆方差权重，并报告聚合方差与有效样本数。
- 任何可解释性辅助项的早停必须以验证侧 Head-A/LLR 端点为依据，不能用 total loss 掩盖主终点下降。
- 频谱干预先报告 trial AUC 和 `exact_llr@3`/`exact_llr@15`，并用 sham、target/non-target、subject 和年龄分层控制；频谱更均匀不是目标本身。
