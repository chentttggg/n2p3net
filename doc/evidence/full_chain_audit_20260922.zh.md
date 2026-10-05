# 全链条只读审查（2026-09-22）

2026-09-29复核说明：下文保留原审查记录；[后续原始产物与真实入口深审](deep_chain_audit_20260929.zh.md)
纠正了部分泛化结论。V1的within-subject调用者已另报requested分母的operational hit，
不能仅凭helper的条件率断言全部hit高估；实际不一致另见新报告F7。V5的Holm/sign-flip
有本地独立分析脚本来源，不能因公共函数未被调用而称无产生路径。LMBC非默认pool参数
保存/重建探针逐位一致，原疑点被推翻。V4真实可复现但不适用于unified固定更新源验证的另一条可加路径。
新报告另外核实了有限段波形、EA静态mask与单位、minimum放宽和复用scope/snapshot等问题。

日期：2026-09-22。性质：**只读审查**。未运行训练、未修改任何代码、契约、默认值或缓存。
与[输入合同与路线审查](input_contract_review_20260922.zh.md)配套；那份处理输入合同与路线，
本份逐层覆盖数据 → 模型 → 迁移 → 决策 → 评估 → 产品 → 工程。

## 0. 方法与可信度声明

四条独立只读通道分别审查数据/预处理、模型/源训练、迁移/决策、评估/统计/产品/工程；
随后对**每一条高严重度结论做独立代码验证**（读取原始行，不采信通道自述）。

标注约定：

- `[已验证]` 我亲自读到代码行并确认机制。
- `[报告-未验证]` 通道给出 `文件:行号`，我未逐条复核，按线索对待。
- `[已推翻]` 通道报告与我实测相反。

**本轮推翻了一条通道标为"高"的结论**（见 §1.2）。这条纠错本身是审查方法有效性的证据，
也说明单靠默认值推断行为不可靠。

## 1. 验证结论汇总

### 1.1 已独立验证成立

| # | 结论 | 证据 | 影响 |
|---|---|---|---|
| V1 | `hit@R` 把 incomplete 决策组**移出分母**，与同文件另一函数口径相反 | `transfer/evaluation.py:351-354`（`continue`，不计 eligible）vs 同文件 `:94-99`（`total += 1; continue`） | hit@R 被高估，且与 `research_program.zh.md` 「失败/tie/incomplete/abstain 保留分母」冲突 |
| V2 | 共同 test 行由 `max(budgets)` 决定，**K=0 的测试队列随同批最大 K 漂移** | `transfer/binary.py:112` `maximum = max(budgets)`；`:143-146` `builder(budget, maximum)` + 断言共用 test | 同一 K 跨批次不可比；zero-shot 锚点同样是漂移量 |
| V3 | 跨运行复用键**不含 `evidence_scope`**，进程内复用键含 | `transfer/target_experiments.py:456-457`（跨运行 key）vs `:533`（进程内 `cache_key`） | development 旧结果可被 software_correctness 运行复用，cell 却按当前 scope 重打标 |
| V4 | 验证损失不是可加统计量（`forward()` 被用，可加的 `components()` 被弃） | `baselines/deep.py:782-788` 用 `loss_fn(...)` 后 `× len(xb)` 再平均；`:254-274` 提供可加分子/分母；`:910` 正确用了可加形式 | 源选轮/早停判据依赖 batch 组成；OOM 减半 batch 后 epoch 间不可比 |
| V5 | `paired_permutation_test` 在活动树中是**死代码**，`src/stats/` 无任何 sign-flip 实现 | 定义 `baselines/evaluate.py:1529`；唯一引用 `tests/research/test_evaluate.py:16,55`；`stats/hierarchical.py` 仅 `paired_interval`，`stats/transfer.py::_summary` 仅 bootstrap | 文档中的 Holm p 值缺少受测试保护的产生路径 |
| V6 | BrainSync EOG 拟合范围的守卫是**可选参数（fail-open）**，非必填 | `experiments/fit_brainsync_eog.py:72` `mask = np.ones(raw.n_times)`，仅显式传 `fit_intervals_by_session` 才收窄（`:74-82`） | **但见 §1.2 的严重度修正** |

### 1.2 已推翻

| 通道结论 | 通道给的严重度 | 实测 | 推翻依据 |
|---|---|---|---|
| 「默认 `observed_channel_norm=False` 使缺导零填充进入 temporal BN，affine 把缺失电极复活」 | 高 | **主链恒为 True，该结论不成立** | `research/source_matrix.py:31` 显式 `N2P3ArchitectureConfig(observed_channel_norm=True, …)`；`transfer/checkpoint.py:204` 加载时强制 `values["observed_channel_norm"] = True`；`transfer/calibrated.py:233` 要求 `source.observed_channel_norm` 为真；`tests/brainsync/test_unified_predictor.py:42` 断言 |

补充事实：缺导处理正是项目自己要求的资格项。
`doc/assets/unified_model_research_20260916/eligibility_inventory.md:65` 写明
「BI2014a/BI2015a/Huebner 历史六槽掩码均存在缺导，固定 stem 首个 temporal BN 须明确实测矩统计
和 affine 后 remask」，而 `n2p3net.py:1130-1164::_observed_temporal_features` 正是该实现
（BN 只消费 pack 后的实测槽，之后 `new_zeros` + `restored[channel_mask] = encoded` 完成 remask），
`architecture_audit.md:131` 的 β 复活反例亦指向同一修法。**该资格已落实，不作缺陷报告。**

同时修正 V6 的严重度：`fit_brainsync_eog_sessions` 在产品链
（`experiments/brainsync_model_task.py:367`）不传 `fit_intervals_by_session`，但产品协议是
**跨 session**（多处校准 session 拟合 → 独立后续 session 推理），整段校准 session 拟合不构成泄漏。
问题降级为 **API 默认 fail-open**：将来任何 within-session 实验复用此函数且忘记传区间，
会静默把测试段拟合进 EOG，而 `data/ocular.py:333-338` 的校验只比对 session 白名单、不比对样本区间。

## 2. 贯穿性判断

项目的**输入合同层**极其严格：`PreprocessingSpec` 单一源、执行凭证、digest 门禁、fail-closed 处处可见。
问题集中在另一类东西——**同一语义在代码库里存在 ≥2 套实现或口径，彼此不一致**：

| 语义 | 口径 A | 口径 B | 位置 |
|---|---|---|---|
| incomplete 决策组的分母 | 保留（规范口径） | 剔除 | `evaluation.py:94-99` vs `:351-354` |
| 结果复用键 | 含 `evidence_scope` | 不含 | `target_experiments.py:533` vs `:456` |
| 验证损失归约 | 可加分子/分母 | batch 平均再平均 | `deep.py:254-274`/`:910` vs `:782-788` |
| `n_times` 几何 | floor/半开 | round/闭端 | `contract.py:88` vs `preprocess.py:715`（`[报告-未验证]` 另有两份） |
| 主 hit 口径 | operational（门禁） | valid（默认分析） | `tests/research/promotion/test_plan.py:62` vs `experiments/analyze_candidate_cross_decision.py:60`（`[报告-未验证]`） |

这解释了 `unified12_analysis_critique_20260918.zh.md` 的结论「执行完整性高、统计推断强度中等偏低」：
**执行层（身份、digest、重放）已被强约束，而指标与归约层（软契约）没有同等强度的单一源约束。**

## 3. 逐层发现

### 3.1 数据接入与预处理

- `[已验证]` V6/E06：EOG fit 范围守卫 fail-open（见 §1.2）。
- `[已验证]` V2：共同 test 随 `max(budgets)` 漂移。**对下一轮 zero-shot 路线的直接影响**：
  `zero_shot_transfer_routes_20260920.zh.md` 把「旧四库 K0 = 0.576」当作可引用的锚点，
  但那批预测绑定的是 K48/96/192 批次的 test 边界；新路线若按 K0/200/800 批次运行，
  其 zero-shot 落在不同 cohort 上，**两者不能直接比较，0.576 不能作为先验**。
- `[报告-未验证]` `data/moabb.py:1110-1112` 因果 raw 入口把 `event_id` 中所有非 `target_label`
  的注释映射为 0，未要求恰为 2 类；若某库带第三类事件键，会被当作 non-target 生成 epoch。
  该路径未调用同模块 `encode_binary_labels`（`:1017`）的二类守卫。
- `[报告-未验证]` `data/domain.py:481-521`：`adapt_observed_montage` 不校验 anchor 是否落在
  `target_channels` 内；若 anchor 在目标通道中，该列恒零却被标记 observed。现有调用方均避开，
  属潜在陷阱而非现行缺陷。
- `[报告-未验证]` `data/source_records.py:397-398`：内层划分失败被 `except ValueError`
  转为 `status="unavailable"`，`verify_prepared_prefixes` 只要求有 reason 即放行；
  真实契约错误会被翻译成「本 K 无早停」并静默退回整段预算。

### 3.2 模型与源训练

- `[已验证]` V4：验证损失归约。
- `[报告-未验证]` LMBC 的窗/温度超参不在 `N2P3ArchitectureConfig` 内
  （`n2p3net.py:51-53` 自称完整可序列化契约，但 `model_kwargs()` 不含），
  `default_architecture_record` 把 `[250,600]/[-200,0]/±100 ms/0.5` 写死（`:952-960`）；
  非默认 LMBC 配置在 checkpoint 重载时会静默回落默认值。
- `[报告-未验证]` SupCon/条件 MMD 在 autocast(bf16) 区内计算
  （`deep.py:1345-1351`，`transfer/source_objective.py:293,365,399-411`），
  `features.float()` 无法阻止 matmul/cdist 被降精度。
- `[报告-未验证]` 源热循环逐 batch 存在 host 同步（`deep.py:2206,2216-2218,2224-2237`），
  与声明的 `host_sync_policy="epoch_boundary"`（`:1159`）矛盾，并会破坏
  `compile_mode="reduce-overhead"` 的 cudagraph 捕获。
- `[报告-未验证]` `deep.py:299,381-382` 的 `max_update_batch_size` 上限只在 `_fit_attempt`
  生效，固定更新主路径 `_fit_fixed_updates` 直接用 `cfg.batch_size`（`:1375,1378,1424`）。
  当前调用方恰好传 512，未触发。
- `[报告-未验证]` `scale_architecture_preserving_spans` 只换算卷积核、不换算池化尺寸，
  跨采样率臂并非"纯采样率对照"。

### 3.3 迁移与决策

- `[已验证]` V1：hit@R 分母。
- `[已验证]` V3：复用键不对称。
- `[报告-未验证]` AS-MMD 在主链未接线（`subject_adapter.py:82` `as_mmd_lambda0=0.0`，
  `transfer/recipes.py` 各工厂无赋值）；GALIA 仅在 `experiments/fit_galia_alignment.py`；
  EA 仅在离线 cache 脚本与测试；`XdawnRiemann` 只作源基线。**这些"能力"当前无迁移消费路径。**
- `[报告-未验证]` `data/domain_alignment.py` 的 `GaliaAdapter` 数学上不等价于论文 GALIA（命名不实），
  与 `zero_shot_transfer_routes_20260920.zh.md` 的判断一致。
- 未发现问题：checkpoint 门禁（`checkpoint.py:894-904`）、scratch 不继承源状态、
  decision loss 未按模型当前分数挑选可用组、80% 回退未见 off-by-one。

### 3.4 评估与统计

- `[已验证]` V5：sign-flip 是死代码。
- `[报告-未验证]` 默认分析指标取 valid 口径
  （`experiments/analyze_candidate_cross_decision.py:60`），而晋升门禁断言 operational 口径
  （`tests/research/promotion/test_plan.py:62`）；valid 口径对无分母者静默剔除。
- `[报告-未验证]` valid 指标对被试的缺失 replicate 做静默平均（`:1139-1145`），缺失数不记录，
  稳定性可能虚高。
- `[报告-未验证]` bootstrap 区间与 sign-flip p 的零假设不同却并列报告，易被读作同一推断的两面。

### 3.5 产品与工程

- `[报告-未验证]` `experiments/brainsync_model_task.py:135-137,198-210`：无标签推理在没有
  有限时间戳时仍返回 `status="predicted"` 且 `evidence_available_time_s=None`，时间门禁记录留空。
- `[已验证]` `configs/*.json` 与 `src/` dataclass 默认值逐项核对一致，未发现配置漂移；
  `conftest.py` 只做 `setdefault` 环境变量与 `.mne_home` 创建。
- `[已验证]` 5 个模块 docstring 仍引用已删除的 `constitution P8`/`techstack`/`roadmap Phase 1`
  （`baselines/{classic,deep,features,riemann}.py`、`data/gtn.py`）；`deep.py:9` 另引用不存在的
  `train/trainer.py`。
- `[报告-未验证]` 测试盲区：`stats/domain_folds.py`、`data/slow_trend.py`、
  `data/domain_alignment.py`、`transfer/candidate_set_loss.py` 无专属用例。

## 4. 优先级建议

| 优先 | 动作 | 理由 |
|---|---|---|
| P0 | 统一 `hit@R` 分母口径，删除剔除分支，同时输出 `eligible` 与 `requested` 两列 | V1 直接改变所有 hit 读数的解释；改动小、影响面明确 |
| P0 | 把 `evidence_scope` 补进跨运行复用键，并让已复用 cell 记录**来源** scope 而非当前 scope | V3 是审计完整性问题，与项目「不重标旧结果」原则冲突 |
| P0 | 冻结 zero-shot 锚点的 test cohort：为新路线单独生成一份固定 cohort 的 zero-shot 预测，不引用 0.576 | V2 使 09-20 路线的门槛基准不可靠 |
| P1 | 验证损失改用 `components()` 的累计分子/分母 | V4；同时使 OOM 回退后 epoch 间保持可比 |
| P1 | 为 `paired_permutation_test` 接回生产路径或删除；补 `+1` 修正的断言 | V5；否则 p 值永远不可复现 |
| P1 | 在 `fit_brainsync_eog_sessions` 上把 fit 区间改为必填（或对 within-session 入口显式拒绝） | §1.2；消除 fail-open 默认 |
| P2 | 清理失效 docstring 引用 | 低风险一致性 |
| P2 | 核对并按需补齐 `[报告-未验证]` 各项，每项单独判定 | 这些均有行号线索，但需各自验证后决定是否成立 |

## 5. 边界

- 本报告不含任何训练结果；P0/P1 项均未修改代码，等待裁决。
- `[报告-未验证]` 各项来自四路只读通道的行号线索，**未经我逐条复核**，
  不应直接作为缺陷定论使用；其中「缺导 BN」一项已证明为误报，提示其余各项同样需要验证。
- 未读取 `experiments/runs/` 内结果 JSON，结论不依赖历史成绩反推。
- 旧执行身份、缓存、结果、冻结归档全部保持原状。
