# Phase 0 / Phase 1 代码评审报告（Review v4）

> 评审日期：2026-08-21
> 评审对象：Phase 0（data 层：preprocess/channel/metadata/dataset）+ Phase 1（baselines：classic/riemann/features/deep/evaluate）+ 关联共享契约（decision/reference/losses/augment）
> 评审方式：逐文件研读 + 186 项测试实测全绿 + 关键数值/坐标/格式实证核对（MNE standard_1020、参数预算、GTN 实际数据格式）
> 结论效力：本报告为建议修订依据，不直接改动代码；是否采纳按 constitution 第六节记录。

---

## 0. 总体判定

**方向正确、实现质量高、但 Phase 1 存在一条数据侧的「硬断链」——核心验收数据 GTN 当前根本无法读入。**

- 代码分层、契约、三思记录与 blueprint/constitution/roadmap 高度对齐；186 项测试全绿（含 6 个关键语义测试）；参数预算精确命中文档（TCN depth=3 = 38,359 ≈ 文档 38k/≤50k；Conformer depth=1 = 58,199 ≈ 文档 58k 超预算）；通道坐标逐点核对与 MNE standard_1020 完全一致；时间轴 arange 约定与 MNE 无 off-by-one。
- 但 Phase 1 的验收锚点「GTN 复现 77%±3」所依赖的 GTN 数据，实际是 **NIX（.nix/HDF5）格式**，而 data 层只支持 BrainVision/EDF/FIF/EEGLAB 等，MNE 亦无 NIX reader、nixio/pynix 未安装。**文档 datasets.md 关于 GTN 是 BrainVision 的描述是错误的**。这是当前最优先、也是唯一真正阻断性的问题。

---

## 1. 必须修复的问题（P0）

### 1.1【阻断】GTN 数据加载断链（NIX 格式 vs 文档声称 BrainVision）

实测证据：
- `mne_data/MNE-P3-data/`（GTN，248 名被试）每个被试目录为 `Experiment_XXX_P3_Numbers.nix`（文件头 `89 48 44 46 0d 0a 1a 0a` = HDF5 魔数，NIX 即 HDF5-based）+ `Data/P3Numbers_*.txt` + `Scenario/numbers/*`。
- `datasets.md` 第 115 行写「BrainVision（GTN）→ MNE `read_raw_brainvision`」，**与事实不符**。
- `data/dataset.py::_EXTENSION_READERS` 无 `.nix`/`.txt` 分支；`mne.io` 无 `read_raw_nix`；环境无 `nixio`/`pynix`。

影响：Phase 1 验收数据读不进来，`load_dataset()` 会直接 `ValueError`。

修复（三选一，推荐 ①）：
1. 引入 `nixio` 依赖 + 官方 loader（github.com/moucek/guess_the_number_py）的思路，或用 `h5py` 直读 NIX 结构（`data/EEG Data/data_arrays` 已确认存在），写一个 `data/gtn.py` 专用加载器，接入 dataset.py。
2. 离线把 .nix 批量转成 BrainVision/FIF 缓存（一次性脚本），再走现有 preprocess。
3. 单独维护一个「GTN 转换层」，不污染通用 data 层。

### 1.2【阻断】GTN 元数据未解析（age/sex/心选数字/在线猜测）

`Data/P3Numbers_*.txt` 每被试 8 行元数据（实测字段）：
```
sex: female
age: 10 years
the number thought: 1      ← 心选数字（决策层 true_digits）
the first guessed: 1       ← 在线首猜（77.2% 锚点来源）
the second guessed: -
the third guessed: -
handedness: right
other: -
```
- `the number thought` ∈ {1..9}（249 名被试实测无 0）→ 决策层 digit_vocab=1–9 正确。
- 但 data 层无任何解析代码，`age`/`sex`/`true_digits` 无法进入 SubjectData 与评估。

修复：新增 `.txt` 元数据解析器，产出 age/sex/subject_id/true_digit/online_guess 字段。

### 1.3【阻断】评估协议与已下载数据语义不匹配

`experiments/download_datasets.py` 产出的是**二分类 target/non-target**（`X,y` + metadata），无 `digits`/`true_digits`；而 `evaluate.py` 需要 `digits` + `true_digits` 才能算 9 选 1 命中率。且 bnci008 / bi2014a / erpcore 均为 **P300 speller**（36 符号 / 5 字母 / 行列范式），**并非「猜数字 9 选 1」**，9 选 1 命中率指标对它们不直接适用。

影响：Phase 1 若只跑已下载的 MOABB 数据，命中率指标无数据支撑；真正能跑 9 选 1 的只有 GTN（而 GTN 又受 1.1 阻断）。

修复：明确三层分工——
- GTN：唯一跑「9 选 1 命中率」的数据（须先修 1.1/1.2）。
- speller 数据（bi2014a/bnci008/erpcore）：只报单试次 target/non-target 的 AUC / balanced acc（协议 D-hit-vs-bacc 已有此分层）。
- 若要在 speller 上做「N 选 1」，需从 metadata 反推每试次刺激符号 + 每 block 目标（额外映射层，Phase 1 可后置）。

### 1.4【三思级修正】GTN 目标基率是 1/10，不是 1/9

实测 `Scenario/numbers/numbers.sce`：定义了 `tx0..tx9`（数字 0–9，共 10 个刺激），trial `t0`（显示 "0"，`port_code=10`）到 `t9`（`port_code=9`），末尾 `LOOP $i 50`。即**每被试 10 数字 × 50 次 = 500 试次**，其中 "0" 是**始终非目标**的闪现刺激。

推论：
- 「猜数字」仍是 9 选 1（心选数字恒 1–9，chance 11.1% ✓，与文档一致）。
- **但分类器训练的 target 基率 = 50/500 = 1/10，非 1/9**。故 `pos_weight` 应 ≈ **9**，而非 blueprint/constitution/losses 一致写的 ≈8。二者差别虽小（决策层被试内 z-score 对校准不敏感），但属于「三思」应纠正的口径。
- 决策层 `decide()` 对 digit=0 的试次不累加（0 ∉ vocab 1–9），行为正确，但需确认 0 试次确实应被排除在决策累加之外（语义上「0 永非答案」成立）。

修复：① 与官方论文（Moucek 2017 Sci Data）核对刺激集；② 若确认 10 刺激，把 pos_weight 默认从 8 改为 9，并在文档 D-pos-weight 记录此修订。

---

## 2. 正确性/契约问题（P1）

### 2.1 evaluate 与 within_subject_folds 分组契约不一致

`within_subject_folds(subject_ids, run_ids)` 需分开两个数组；`evaluate(...)` 只接受单一 `subject_ids`（group key）。Phase 2「单受试按 run 留一」验收需要 `f"{subj}_{run}"` 组合键，但两函数分组约定不统一、run 维无法传递。当前只能「外部拼好组合键再喂 evaluate」，属隐式契约、易漏。

修复：evaluate 增加可选 `run_ids` 入参（或统一约定 group key = 组合串），并补一条跨模块集成测试。

### 2.2 打包/导入结构不完整

模块用顶层绝对导入（`from data.preprocess import ...`），但 `pyproject.toml` 无 `[build-system]`、无法 `pip install -e .`；`src` 仅靠 pytest 的 `pythonpath=["src"]` 生效。experiments/ 里除 `download_datasets.py`（只 import moabb）外，一旦写训练/评估脚本 `from baselines import ...` 即 ImportError。

修复：补 build-system（setuptools/hatchling）+ 包发现，或至少约定 `PYTHONPATH=src` 并写进 CODING_WORKFLOW。

### 2.3 reference.py 缺 AMP dtype 对齐（与 tokenizer/component_window 不一致）

tokenizer（D-time-pe）与 component_window（D-dtype-align）都做了「参数/buffer 显式对齐 Z.dtype」；但 `WeightedRereference` 的 `self.w = softmax(w_logits)`（float32）与 autocast(bf16) 下的 X 在 `einsum("c,bct->bt", ...)` 混算，会 promote 到 float32，破坏 autocast 语义（非崩溃，性能/数值一致性问题）。

修复：forward 内 `w = self.w.to(X.dtype)`。

### 2.4 已知未修项（roadmap 已记录，此处确认仍存在）

- `train/losses.py::rbf_mmd2` 固定 `bandwidth=1.0`：D=64 特征距离量级 ≫1，`exp(-d²/2)` 梯度全零（Phase 3 待办）。
- `train/augment.py::time_warp` `.cpu().numpy()` 往返 + B×C 双循环（Phase 3 待办，建议挪进 collate_fn）。

---

## 3. 次要问题（P2）

1. `SWLDA._select_features` 单变量 p 值在每个外层迭代重复计算（O(max_features×n_features) 次 t 检验），可向量化：一次算全 p 值排序取 top-k。
2. `preprocess.reject_epochs` 的 NaN 分支在实际调用路径是死代码（调用时 `data=epochs.get_data()` 无 NaN），无害，仅防御。
3. `preprocess` 的 `n_times=256` 裁剪丢弃 +800ms 末点（窗口实际 [-200, +796.09]ms），已文档化，P3b 在 300–500ms 无影响。
4. ERP CORE 尚未下载（cache 仅有 bi2014a / bnci008），Phase 3 跨数据集验证需补下。
5. `src/stats/__init__.py` 为空——Phase 4 的 mass-univariate / cluster-permutation / LIMO 未实现（符合 roadmap，预期）。
6. bi2014a 是 16 导干电极、缺 Fz/PO7/PO8，download 用全量 16 导，与 8 导蒙太奇不匹配；需 subset/零填充路径（datasets.md 已提、脚本未落地）。

---

## 4. 可行性评估

| 维度 | 判定 | 说明 |
|---|---|---|
| 架构可行性 | **高** | 设计论证充分（因果反转 D8 / 不对称窗 D9 / 参考无关重参数化），参数克制（38k），测试全绿，文档自洽 |
| Phase 0 数据层 | **高** | 网络外预处理 + 坐标通道 + 元数据嵌入 + 标签对齐全部正确且测试覆盖到位 |
| Phase 1 基线 | **高（代码）** | 三个免费地板 + xDAWN/黎曼 + 三个深度基线齐全，接口统一，评估协议分层清晰 |
| Phase 1 验收 | **当前受阻（P0）** | 唯一跑 9 选 1 的 GTN 读不进来（NIX）+ 元数据未解析 + speller 数据无命中率语义 |
| 修复成本 | **低** | NIX→试次+标签是纯工程（官方 loader 现成 / h5py 直读），非算法风险 |

**结论**：架构与代码本身没有不可行之处；卡点集中在「GTN 数据接入」这一条数据侧断链。修 1.1–1.3 后，Phase 1 可正常推进。77%±3 锚点来自 3 导 MLP（Vařeka 2016），本项目 8 导蒙太奇是 Fz/Cz/Pz 的超集，复现路径清晰。

---

## 5. 改进优先级清单

**P0（进 Phase 1 实跑前必须）**
1. GTN NIX loader（nixio 或 h5py 直读）+ 接入 dataset.py（1.1）
2. .txt 元数据解析（age/sex/thought/guessed）（1.2）
3. 修正 datasets.md 的 GTN 格式错误（BrainVision → NIX）
4. 明确 speller 数据只报 AUC/bacc、命中率仅 GTN 的协议（1.3）
5. 核实刺激集，pos_weight 8 → 9 的修订（1.4）

**P1（Phase 2 前）**
6. evaluate 补 run 维分组契约（2.1）
7. 打包/build-system + PYTHONPATH 约定（2.2）
8. reference dtype 对齐（2.3）

**P2（Phase 3/4）**
9. SWLDA 特征选择向量化；10. MMD 带宽 median heuristic；11. time_warp 挪 collate_fn；12. 下 ERP CORE；13. stats 模块实现。
