# Phase 0/1 复核与 Phase 2 风险评审（Review v6 · follow-up）

> 评审日期：2026-08-21
> 评审对象：review v5 之后的当前工作树（data/baselines/models/train/experiments/docs 全部代码与文档）
> 评审方式：逐文件研读 + 192 项测试全绿 + GTN 248 目录逐被试实测 + 30 被试 LOSO 复测 +
>   合成数据反例 + 缺失通道/AMP/MMD 数值实证。
> 结论效力：本报告为建议修订依据，不直接改动代码；是否采纳按 constitution 第六节记录。
> 本报告与 review v5 的关系：确认 v5 的多数结论，但**推翻了 v5 对 SWLDA 低命中率的归因**，
> 并发现 v5 未覆盖的三条 P0 级集成断链。

---

## 0. 总体判定

**代码骨架与数据层整体成立，但当前最危险的 bug 不在模型结构，而在决策层与训练集成层：**

1. **决策层存在 P0 校准 bug**：`decide()` 把含常数偏置的分类 logit 直接按 `score(d)=Σlogit` 累加。
   被试内 z-score 是作用在 9 个 score 上的同一仿射变换，**不能消除** `c·n_d`（常数偏置 × 每数字
   试次数）项。实测把 SWLDA 的 LOSO 命中率从 0.467 压到 0.467，而只需每被试先中心化 logit 再累加，
   同一模型命中率升到 0.667（按每数字均值聚合为 0.700）。
2. **N2P3-Net 训练链路没有把 channel_mask 传到 Trainer / 增强层**。GTN 5 个缺失通道填 0 后，
   `reference_jitter` 与 `gaussian_noise` 会把缺失通道重新变成非零，再经 Stage 0 基线段标准化放大成
   std≈1 的「幻象通道」。这是宪法 P5 的直接失守，review v3/v4 只修了 Stage 0 本身，没修训练入口。
3. **Phase 2 的 τ 可识别性存在真实风险**。初步合成探针显示：全模型可以在 `τ_P3b≈350ms` 不动的情况下，
   对真实潜伏期 250–500ms 的合成 P3b 都分类到接近 0 损失——分类饱和不保证 τ 学到真值。这动摇了
   blueprint D8「分类直接监督 τ」的强论断，Phase 2 的 `MAE(τ,τ_true)<40ms` 诊断必须重做且做对。

---

## 1. P0-1【决策层】常数校准偏置破坏 logit 累加（推翻 review v3 §3.6 与 review v5 §3.2 的归因）

### 1.1 数学错误

设分类器输出 `logit_i = s_i + c`，其中 c 是模型截距/类先验/pos_weight 引入的常数偏置。
决策层当前实现（`models/decision.py`）：

```python
raw_scores[d] = Σ_{i∈d} logit_i = Σ s_i + c·n_d
z_scores[d]  = (raw_scores[d] − mean(raw_scores)) / std(raw_scores)
```

- `z_scores` 是作用在 9 个数字 score 上的**同一单调仿射变换**，不改变 argmax。
- 因此 `c·n_d` 项进入 argmax。GTN 每数字试次数 n_d≈14–23 且随机波动 ±20%，c 只要量级与单试次
  判别差相当，就会主导/严重扰动排名。
- review v3 §3.6 与 blueprint §6 认为「被试内 z-score 去校准偏差」——**该命题在试次数不等时数学上
  不成立**。z-score 只能去掉 score 向量整体的均值/尺度，去不掉 `c·(n_d−n̄)` 这种逐数字差分。

### 1.2 实测证据（本评审独立复现）

(a) 合成反例：2000 次仿真，每数字试次数随机（均值 22），target logit=+0.2、non-target=−0.2，
    附加 c=−5.0：
- 当前 `decide()`：命中率 **0.184**；
- 先按被试中心化 logit 再 `decide()`：命中率 **1.000**。

(b) GTN 30 被试 LOSO（同一 SWLDA，同一 fold，仅决策层不同）：

| 聚合方式 | SWLDA | WindowLR | Template |
|---|---|---|---|
| 当前 Σlogit（未中心化） | 0.467 | 0.433 | 0.767 |
| 先被试内中心化 + Σ | 0.667 | 0.400 | 0.767 |
| 先被试内中心化 + 每数字均值 | 0.700 | 0.433 | 0.800 |

(c) `experiments/diagnose_logit.py` 复测 10 被试（fit=test 诊断模式）：EEGNet V 单位输入下
    target logit 均值 −0.188、non-target −0.200，AUC 0.744 但 top1 比例 0.100（≈chance）；
    输入全局 z-score 后 AUC 0.797、top1 0.900。常数 −0.2 的偏置与仅 0.012 的单试次差正是 1.1 的
    现实版本。

### 1.3 对 review v5 结论的修正

review v5 §3.2 把「SWLDA AUC 0.712 但命中率 0.467」归因为「logit 尺度/校准被试间漂移 + 累加后
区分度不足」，方向对但不完整：**主因是决策层没有先去除逐被试常数偏置**，而不是模型本身没有判别
信息。中心化后 SWLDA 命中率 0.667–0.700，逼近 template。此项修正对 Phase 1 复现 77%±3 锚点有
直接意义。

### 1.4 建议修复

1. `decide()` 增加 `center_logits: bool = True`（或 `"subject"`），在累加前对每个 subject 的有限
   logit 减去该 subject 均值；NaN/inf 守卫不变。
2. 增加 `aggregation: {"sum","mean"}` 消融轴。GTN 30 被试上「中心化 + mean」略优于「中心化 + sum」
   （SWLDA 0.700 vs 0.667，template 0.800 vs 0.767）；全量 247 被试上实测裁决。
3. `evaluate()` 的 balanced acc 阈值不能固定 0（deep 基线实测 AUC 0.744 时 bacc@0=0.500）。
   建议：logit 按 fold 中心化后再用 0 阈值，或同时报告 AUC 与 threshold-free 指标；主口径仍按
   constitution D3 用命中率/balanced acc/AUC。
4. 补两条语义测试：(i) 常数偏置 + 不等试次数的校准不变性；(ii) `aggregation=mean` 与空集处理。

---

## 2. P0-2【训练集成】channel_mask 未贯穿 Trainer 与增强层（GTN/缺失通道场景必踩）

### 2.1 现状

- `models/n2p3net.py::forward` 已正确接受 `channel_mask`，`models/reference.py` 也实现了
  mask 重归一化与「缺失通道恒 0」（review v3/v4 修好）。
- 但 `train/trainer.py` **不接受、也不传递 channel_mask**：`self.model(X, self.E_chn, self.E_sub)`
  无 mask 参数。
- 更隐蔽的是增强层：`train/augment.py::reference_jitter` 与 `gaussian_noise` 对**全部 8 通道**操作，
  会把缺失通道（零填充）变成非零；之后即使 Stage 0 有 mask，缺失通道也保持非零并被基线段标准化
  放大成 std≈1 的幻象。

### 2.2 数值实证（B=4，3 导存在 + 5 导缺失填 0）

| 路径 | 缺失通道经 Stage 0 后的逐试次 std |
|---|---|
| 无 mask，`reference(X, None)` | **0.943**（幻象通道） |
| 有 mask，`reference(X, mask)` | 0（正确） |
| 先 `reference_jitter(X, p=1)`，再有 mask | **0.998**（增强先破坏了不变量） |
| 先 `gaussian_noise(X, σ=0.1)`，再有 mask | **0.986**（增强先破坏了不变量） |

结论：**只修 Trainer 传 mask 不够，增强层也必须 mask-aware**。`time_warp` / `amplitude_jitter`
/ `channel_dropout` 对零填充通道是安全的（0 保持 0 或 ×0），但 reference_jitter 与 gaussian_noise
必须只在存在通道上施加（或施加后把缺失通道强制归零）。

### 2.3 建议修复

1. `Trainer.__init__` 增加 `channel_mask: Optional[torch.Tensor]`，`_train_step`/`_evaluate`
   的 forward 统一传 `channel_mask=...`。
2. `apply_augmentations` / `reference_jitter` / `gaussian_noise` 增加可选 `channel_mask`；
   reference 凸组合只用存在通道计算，噪声只加在存在通道；增强后缺失通道强制归 0。
3. GTN 接 N2P3-Net 时，用 `preprocess(..., standard=8)` 产出 (N,8,256) 零填充 + mask，而不是
   baselines 用的 3 导子集；两类路径在实验脚本里显式区分。
4. 补集成测试：3 导 + 5 缺失，跑 `apply_augmentations` → `N2P3Net.forward`，断言缺失通道在
   Stage 0 出口恒 0。

---

## 3. P0-3【GTN 数据链】三个数据事实修正与两个加载器缺口

### 3.1 确认 review v5 的正确修正

- **推翻 review v4 §1.4 成立**：248 个目录、245 个可扫描 NIX 中，刺激 label 只有
  `Stimulus/S 1..9`（共 50,569 次）+ `New Segment/`（245 次）+ `Stimulus/S 13`(×1) /
  `Stimulus/S 15`(×12)；**没有任何 `0`/port_code=10 事件**。target 基率按 1/9 计，
  `pos_weight≈8` 正确，review v4 的「8→9」错误。
- `.sce` 确实定义了 `tx0`，但它未出现在 NIX 记录中；不得用 `.sce` 推断实际事件流。
- 每被试原始数字事件 mean≈205（range 58–372，scan 口径）；默认 ±150μV 伪迹剔除后
  **全量实际 mean≈156 试次/被试，K≈17.3/数字**，比 review v5 写的 K≈14–16 略高。

### 3.2 修正 review v5 对 Experiment_611 的描述

review v5 写「Experiment_611 损坏，HDF5 无法同步打开」。实测**该 .nix 可正常打开**，内部
data array 完整（P3Numbers_20150107_f_14_002），只是 `Data/` 目录下**没有 .txt**，因此无法得到
`the number thought`，不能用于 9 选 1 评估。描述应改为「元数据缺失」，不是「HDF5 损坏」。

### 3.3 加载器缺口

1. **多 .txt 匹配不可靠**（`data/gtn.py::read_gtn_experiment`）：`Experiment_515/531` 的
   `Data/` 下各有两个 .txt，函数 `glob("*.txt")` 后取第一个。当前文件系统顺序恰好匹配 NIX 内部
   被试名，但这是未定义行为。应从 NIX 的 `data/EEG Data/data_arrays` 读内部被试名，再按 stem
   精确匹配 .txt；多出的 txt 记为孤儿并告警。
2. **零试次被试静默消失**：实测 3 名被试（Experiment_456/627/660）在 ±150μV 下全部 epoch 被剔除
   （raw 幅值异常：epoch 峰值 13–96 mV），`load_gtn_subjects()` 仍把它们写入 `true_digits` 但
   `subject_ids` 不含它们，LOSO 命中率分母静默从 247 缩到 244。须把这类被试计入显式的
   `excluded_by_quality`，并让 `evaluate()` 在 true_digits 存在但 subject_ids 无对应记录时告警。
3. 因此 Phase 1 全量口径应写成：**248 目录 − 1 无 thought 元数据 − 3 全 epoch 剔除 = 244 名可评估**。

---

## 4. P1 级问题（进 Phase 1 收尾/Phase 2 前修）

### 4.1 `run_gtn_baseline.py --model all` 名不副实

第 167 行：`models = ["eegnet","inception","conformer"] if args.model=="all" else [args.model]`。
`--model all` 只跑 3 个 deep 基线，不跑 SWLDA/xDAWN/windowlr/template。要么补齐 7 个基线，
要么把参数改名为 `--model all-deep`。

### 4.2 deep 基线缺输入标准化，logit 可坍缩

`DeepBaseline` 直接吃 V 单位输入（GTN 幅值 ~1e-5–1e-4 V）。10 被试诊断实测 EEGNet 的 logit 被
压到 −0.2±0.02 的窄带（AUC 0.744 但阈值 0 下 bacc=0.500、命中率≈chance）；对 X 做全局 z-score 后
AUC 0.797、命中率 0.900。建议 deep 基线 fit 前按通道/试次做稳健标准化（或至少实验脚本对 deep 分支
标准化），并在 LOSO 下复测 logit 尺度。

### 4.3 `build_subject()` 不支持 `standard` 长度 <8

`data/dataset.py::build_subject` 把 `preprocess` 的 `standard` 透传下去，但随后
`build_channel_identity(channel_mask=result.channel_mask)` 固定用 8 导坐标。`standard=("Fz","Cz","Pz")`
时实测报 `ValueError: operands could not be broadcast together with shapes (8,) (3,)`。
要么在入口拒绝非 8 导 standard（GTN baselines 已绕开本层，可接受），要么让
`build_channel_identity` 接受与 mask 对应的 ch_names；当前契约与实现不一致。

### 4.4 `N2P3Net.baseline_n=51` 与 T 解耦

基线标准化固定取前 51 点（只对 T=256、tmin=−200ms 正确）。缓存 MOABB 数据有 T=257/tmin=0，
tests 用 T=128 也会静默用错基线窗。应由 `sfreq/tmin/T` 推导，或显式校验。

### 4.5 已知未修项确认（review v4/v5 清单仍在）

- `train/losses.py::rbf_mmd2` 固定 `bandwidth=1.0`：D=64 时实测 `rbf_mmd2(x, x+1)` ≈ −1.3e-16，
  梯度全零。Phase 3 一开 MMD 就是空操作。改 median heuristic 或多带宽。
- `pyproject.toml` 仍无 `[build-system]`，不可 `pip install -e .`；experiments 靠 sys.path 补丁。
- `models/reference.py` 未做 `w.to(X.dtype)`；当前 torch 在 autocast 下输出仍为 bf16（本机实测），
  但直接以 bf16 输入、不开 autocast 会 `dtype mismatch`。补显式对齐，成本为零。
- `Stage2Encoder` TCN depth>len(dilations) 时静默少建层；`encoder_type` 在 depth=0 时不校验。
- `Head-D amplitude`（`models/heads.py`）在 `compute_losses` 中完全不被消费，head 参数梯度为零、
  仅在 AdamW 权重衰减下漂移；constitution P7 的「幅值回归」尚未真正实现（roadmap 已记 TODO）。
- `tau0` 不受 L_tau 监督且无生理界约束，AdamW weight decay 会缓慢把它向 0 拉；需数据驱动初始化
  （grand-average 峰位）并至少做范围约束/排除 decay。
- `data/datasets.md` 仍写 GTN 是 BrainVision（应为 NIX），与 gtn.py 矛盾。

### 4.6 文档/风格

- Stage 1 tokenizer 全线性（Conv1d → 空间线性 → pointwise，无激活/归一化），等价于可学习时空
  FIR 滤波器组。数学上可行，但 blueprint 未明确；要么补文档说明这是有意设计，要么加非线性并
  重测参数账。
- ruff 检查有 400+ 项，多数是 N8 命名风格，少数是真清理项（未用 import 等），不影响正确性。

---

## 5. Phase 2 风险评审：PCW 的 τ 可识别性（三思级，可能推翻 D8 的强论断）

### 5.1 初步探针结果

合成数据：target 试次在 Pz 叠加高斯 P3b（幅值 4 a.u.，σ≈31ms），non-target 纯噪声；真实峰潜伏期
分别取 250/300/400/450/500ms；训练 N2P3-Net 默认配置 40 epochs，BCE 到 ~1e-4（分类已完美）。

结果：`component_window.tau0_P3b` 在所有条件下都停在 **349±0.05ms**，`tau_mean_P3b≈352ms`，
与真实潜伏期几乎无关。分类靠 Stage 1/2 的可学习时间映射与宽窗尾巴即可成功，PCW 的 τ 没有被
「分类直接、单调监督」到真值。

### 5.2 解读（谨慎，不据此直接推翻 D8）

- 这是**高 SNR、恒定潜伏期**的快速探针，不能作为最终判决；真实多试次 latency jitter 与低 SNR
  下 τ 可能获得更强梯度。
- 但它证明 roadmap Phase 2 的验收诊断「模拟数据 MAE(τ,τ_true)<40ms」**不能用当前全模型直接跑**
  ——Stage 1/2 可以把时间信息平移/抹掉后让固定窗也能分类，τ 读数的生理含义会失效。
- 正确诊断至少应分两层：(a) 冻结 tokenizer/encoder，只让 PCW+heads 学习，测 τ 对时间局部化
  特征的恢复能力；(b) 全模型 + 试次间随机 latency jitter，检查 τ 是否随 jitter 协变。
- 若 (a)/(b) 不达标，必须修订 D8/E5 的论证或给 τ 增加可识别约束（如 Head-D 物理幅值监督、
  t<300ms 证据门控、τ 的先验中心软约束），而不是把 MAE 诊断做成事后补救。

---

## 6. 自有数据采集侧（BrainSync SDK 示例）的标记时序风险

`brainsync-sdk-example-main/python/edf_recording_with_lsl_markers.py` 收到 LSL marker 后用
`time.time() − start_time` 写 EDF annotation，丢弃了 LSL 时间戳；`eeg_recording_with_triggerbox.py`
用 50ms 轮询状态变化打标。两条路径的刺激 onset 抖动都是「毫秒级不可忽略」（轮询路径最坏 ±25ms+
asyncio 调度），而本项目 Phase 4 要拿 τ 与 mass-univariate 对表，锁时误差会直接污染 P3 潜伏期。
建议下一阶段：
1. 用 TriggerBox 的连续 ADC 流（`triggerhub_adc_stream`）或 LSL 双时钟 offset 校正做**样本级**标记；
2. 写一个光电二极管/音频 onset 验证实验，量化 EDF annotation 与实际刺激的时延分布；
3. 确认 BrainSync 8 通道物理接线与 `channel_config.json` 的 10-20 标签一致，并记录 REF/GND 位置
   （项目 P5 需要）。

---

## 7. 下一步任务优先级（建议）

### P0（立即，Phase 1 实跑之前）

1. 修 `decide()` 校准中心化 + sum/mean 聚合轴，补校准不变性语义测试。
2. 修 `evaluate()` 的 bacc 阈值口径（中心化后阈值 0，或只报 AUC/threshold-free）。
3. Trainer + augment 全链路 channel_mask；补 3 导缺失通道端到端集成测试。
4. `read_gtn_experiment` 按 NIX 内部 ID 精确匹配 txt；611/多 txt/全剔除被试显式登记，
   命中率分母固定 244 并输出数据质量表。
5. 修 `--model all`；deep 基线输入标准化；重跑 30 被试 LOSO 复测 v5 表格（重点看 SWLDA/xDAWN
   在中心化决策后的变化）。

### P1（Phase 1 收尾）

6. 全量 244 被试 LOSO 七基线 + 配对置换检验；确认 77%±3 锚点与 free floor。
7. MOABB 缓存接入脚本：编码 y、裁剪 T=257→256、保存通道名，跑 bi2014a/bnci008/erpcore 的
   AUC/bacc（二分类协议）。
8. 补 build-system；SWLDA 特征选择向量化；修正 datasets.md。

### P2（Phase 2 开工门槛）

9. 先做 §5 的两层 τ 可识别性诊断，结果回填 blueprint D8/E5 是否成立；τ0 数据驱动初始化
   （grand-average target−non-target 峰位）与 τ0 范围约束。
10. 实现 Head-D 物理幅值（A + 归一化原始 X_Pz）或从损失/宪法 P7 明确降级。
11. MMD 带宽 median heuristic；baseline_n 参数化；Stage 1 线性化设计决策落文档。

---

## 附：本评审推翻了什么、保留了什么

- 推翻 review v4 §1.4（pos_weight 8→9）：保留 review v5 的「保持 8」。
- 推翻 review v3 §3.6 / blueprint §6 的「score 后 z-score 去校准偏差」：必须**累加前**逐被试
  中心化，否则 c·n_d 项进入 argmax（数学 + 合成 + 真实 LOSO 三证据）。
- 修正 review v5：「SWLDA 低命中率」主因不是单试次噪声平均，而是决策层校准偏置；
  「Experiment_611 HDF5 损坏」应为「缺 .txt thought 元数据」。
- 新增 review v5 未覆盖的三条 P0：Trainer/augment 的 mask 断链、`--model all` 缺经典基线、
  零试次被试静默缩小命中率分母。
- 新增 Phase 2 高风险预警：PCW τ 在全模型下可能不被分类监督到真值，验收诊断须分两层设计。
