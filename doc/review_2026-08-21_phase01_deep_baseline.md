# Deep 基线工作评审报告（Review v5 · Phase 1 deep baseline 实测）

> 评审日期：2026-08-21
> 评审对象：Phase 1 deep 基线（EEGNet / EEG-Inception / EEG Conformer）在 GTN 上的 LOSO 实测，
>   以及配套的数据链路（gtn.py / preprocess / evaluate / decision）与免费地板（windowlr / template）。
> 评审方式：逐被试实测读入 247 名 GTN 的 NIX 数据 + 端到端 LOSO 评估 + 关键数值/格式实证核对。
> 结论效力：本报告为建议修订依据；是否采纳按 constitution 第六节记录。

---

## 0. 总体判定

**deep 基线链路已打通，EEGNet 命中率 0.767 追平 template 并达到 Vařeka 77.2% 锚点，单试次 AUC 0.756 为所有基线最高。同时实测推翻了 review v4 的一条 P0 结论（pos_weight 8→9）。**

- 数据侧（GTN）经逐被试实测，发现 review v4 §1.4 对「500 试次 / 10 数字含 0」的判断**与真实 NIX 数据不符**，须推翻（见 §1）。
- 代码侧修复两个真实 bug：gtn.py 正则漏判控制码（§2.1）、WindowLogisticRegression 缺标准化（§2.2）。
- 免费地板 template 意外成为 GTN 最强基线（命中率 0.767），并暴露「单试次判别质量」与「命中率」两个口径的深刻分离（§3）。
- deep 基线（EEGNet）在 30 名被试 LOSO 上命中率 0.767、AUC 0.756，是「单试次判别最强」且「命中率追平模板匹配」的基线（§4）。

---

## 1. 必须推翻的结论（P0 · 数据真相）

### 1.1【推翻 review v4 §1.4】GTN 无「0」刺激，target 基率 = 1/9，pos_weight 保持 ≈8

review v4 §1.4 基于 `numbers.sce` 推断「每被试 10 数字（含 0）× 50 次 = 500 试次，target 基率 1/10，pos_weight 应 8→9」。

逐被试实测 247 名（1 名 `Experiment_611` 损坏，HDF5 无法同步打开）的 NIX 刺激 labels，结果：

- **每被试刺激事件 mean = 206，范围 58–372；无一人达到 500**。
- **完全没有 port_code=10（"0"）的事件**。`.sce` 虽定义 `tx0`，但真实记录的 STIMULUS 数组里不存在。
- 数字分布（汇总 ~50,900 次）：`1..9` 各约 5,600 次，另有 `Stimulus/S 13`(×1)、`Stimulus/S 15`(×12) 两个非数字控制码。
- 心选数字（thought number）实测全 ∈ {1..9}（分布 15–44 人/数字），年龄 7–17 岁（11–15 岁为主）。

**结论**：训练 target 基率 = 1/9，`pos_weight ≈ 8` 是**正确**的，review v4 的「改成 9」错误，予以推翻。代码现状（deep.py / losses.py / trainer.py 均 `pos_weight=8.0`，注释「target 占 1/9」）是对的，**无需改动**。

### 1.2 每数字试次数 K ≈ 23（非 50），命中率锚点口径须校准

实测每被试 206 事件 / 9 数字 ≈ **K≈23**（伪迹剔除后实测 K≈14–16/数字）。Vařeka 2016 的 77.2% 正是在这个真实 K 下取得的（Moucek 2017 论文图 2 以 Experiment_341 为例，其 target 刺激 17 次，与我们实测 16 次吻合）。故「77%±3 锚点」仍然成立，但 mission 中「K≥30 时 ≥85%」是自有成人数据的验收条件，与 GTN 的 K≈23 不可混用。

### 1.3 刺激协议确认（与综述报告一致）

ISI = 1,500 ms；数字范围 1–9（无 0）；随机顺序；每被试平均约 10 分钟（含准备）。GTN 是**儿童（7–17 岁）+ 湿电极 + 3 导 + 嘈杂教室**，定位为「跨年龄迁移源域 + 文献对照」，非成人主数据（constitution E8 一致）。

---

## 2. 修复的 bug（P0 · 代码）

### 2.1 gtn.py 正则漏判控制码

`_extract_digit` 旧正则 `Stimulus/S\s+(\d)` 只捕获单个数字，会把 `Stimulus/S 13` / `Stimulus/S 15`（两个非数字控制码，共 13 次）误判为数字 1，污染标签。改为 `(\d+)` 并显式过滤 `1 <= d <= 9`。已补单测（`test_extract_digit` 断言 13/15 → None）。

### 2.2 WindowLogisticRegression 缺标准化（V 单位 → 分类退化随机）

MNE 输出**伏特 V** 单位（P300 幅值 ~5 μV = 5e-6 V，实测全局 std=2.2e-5 V）。`WindowLogisticRegression` 直接把 ~1e-5 量级的窗均值特征喂给 `LogisticRegression(C=1, l2)`，l2 正则对微小特征过度惩罚权重，分类退化为随机（实测 GTN bacc 0.50 → 加 StandardScaler 后 0.564）。已加 `StandardScaler`（与 SWLDA 的手动 z-score 同义）。

**口径影响**：`TemplateMatching`（Pearson 相关，尺度不变）与 `SWLDA`（内部标准化）不受 V 单位影响；`xDAWN`（float64 协方差）不受；deep（braindecode 有 BatchNorm）基本不受。故只需修 windowlr。

---

## 3. 免费地板实测（30 被试 LOSO，CPU）

| 基线 | 命中率 | balanced acc | AUC |
|---|---|---|---|
| template（模板匹配，250–500ms） | **0.767** | 0.613 | 0.698 |
| swlda | 0.467 | 0.527 | 0.712 |
| windowlr（修复后） | 0.433 | 0.572 | 0.601 |
| xdawn+RG | 0.200 | 0.502 | 0.647 |

### 3.1 关键发现：template 意外成为最强免费地板

`TemplateMatching`（grand-average target 模板 + Pearson 相关）命中率 **0.767 已达 Vařeka 77.2% 锚点**。其机制是「每个数字的多试次平均后再与 target 模板做形状匹配」，天然抑制单试次噪声——这恰是 oddball 猜数字任务的正确打法（review E6）。

### 3.2 深刻分离：「单试次判别质量」≠「命中率」

SWLDA 的**单试次 AUC 最高（0.712）**，但命中率仅 0.467，远低于 template（AUC 0.698 / 命中率 0.767）。原因：命中率依赖「每数字 ~15 次 logit 累加」的区分度，而单试次 logit 的尺度/校准在被试间漂移，累加后区分度不足；template 的「平均后匹配」则把噪声平均掉后再比形状，更鲁棒。这印证 constitution E6：「猜数字靠每数字多次平均，单试次跨域分类接近随机」。

---

## 4. deep 基线实测（30 被试 LOSO，CPU，30 epochs）

| 模型 | 命中率 | balanced acc | AUC | 参数量 |
|---|---|---|---|---|
| EEGNet | **0.767** | **0.678** | **0.756** | 1,490 |
| EEG-Inception | （后台跑中） | | | 26,622 |
| EEG Conformer | （后台跑中） | | | 255,106 |

### 4.1 EEGNet 是「单试次判别最强 + 命中率追平模板」的基线

- **单试次 AUC 0.756 为所有基线最高**（超过 SWLDA 0.712、template 0.698、xdawn 0.647），说明 EEGNet 的端到端时空卷积学到了最强的单试次 target/non-target 判别。
- **命中率 0.767 追平 template**，均达 Vařeka 77.2% 锚点。
- 与 Värbu 2019（GTN 上 CNN≈LDA≈SVM）的既有结论一致：deep 未「显著超越」线性/模板基线，但单试次判别质量确为最强——这正是 constitution P8 / D6 预期的「容量非瓶颈，域差才是」。

### 4.2 训练充分性：10 epochs 严重欠拟合

冒烟实测 EEGNet 10 epochs 时 AUC 仅 0.69、命中率 ~chance；30 epochs 后 AUC 0.756、命中率 0.767。**deep 基线 30 epochs 是下限**，DeepConfig 默认 30 对 EEGNet 尚可、对更大模型可能不足。全量 247 被试、更大模型需 GPU/XPU。

---

## 5. 改进优先级清单

**P0（已完成本次）**
1. 修 gtn.py 控制码正则（2.1）✅
2. 修 windowlr 标准化（2.2）✅
3. 推翻 review v4 §1.4（pos_weight 保持 8，1.1）✅
4. 落地 GTN→LOSO 实验入口 `run_gtn_baseline.py` ✅

**P1（Phase 1 收尾前）**
5. 补 EEG-Inception / EEG Conformer 的 30 被试结果（后台跑中），并在全量 247 被试（GPU）上复测三层协议。
6. 处理 `Experiment_611` 损坏被试（跳过 + 记录剔除率，已由 `load_gtn_subjects` 的 skipped 列表承接）。
7. evaluate 补 run 维分组契约（review v4 §2.1 仍未修）；pyproject 补 build-system（review v4 §2.2）。
8. SWLDA `_select_features` 向量化（review v4 §3.1，当前 O(max_features×n_features) 次 t 检验）。

**P2（Phase 2 前）**
9. 明确「单试次判别质量（AUC/bacc）」与「命中率」双口径在 N2P3-Net 验收中的权重——EEGNet 已证明 deep 的价值在单试次判别（跨域泛化的基础），命中率则需靠决策层「平均/累加」策略。
10. deep baseline 的 logit 校准（CrossEntropyLoss weight=[1,8] 使 logit 零点偏移）是否影响 decision 层累加，用 `diagnose_logit.py` 实证。

---

## 6. 对后续 Phase 2 的方法论启示（三思）

1. **deep 的真实价值定位已被实证锚定**：在 GTN（儿童、湿电极、3 导）上 deep 不超越模板/线性，价值在「单试次判别质量（AUC 0.756 最高）→ 跨域泛化的基础」与「多成分可解释」。N2P3-Net 的验收必须同时报 AUC 与命中率，且命中率预期诚实锚在 ~77%（K≈23）。
2. **决策层策略值得重新审视**：template 的「先平均后匹配」命中率追平最强单试次判别器（EEGNet），提示 N2P3-Net 的决策层除「logit 累加」外，可考虑「每数字平均成分表示后再匹配」的融合方式——这可能是超越 77% 的路径之一（Phase 2 消融）。
3. **V 单位是系统性坑**：所有非网络基线（sklearn）必须自包含标准化；网络内（InstanceNorm/BN）不受影响。数据层应文档化「输出 V 单位」，避免再踩。
