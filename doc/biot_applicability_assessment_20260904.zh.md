# BIOT 在本项目情景下的应用能力评估

日期：2026-09-04 | 触发：T 判定「当前域迁移这套不太行得通」，问 BIOT 能否顶上
方法：只读评估（本地代码读参 + 外部文献核对）

---

## 1. 结论摘要

| 角色 | 判定 | 理由 |
|---|---|---|
| **BIOT 作为现成预训练模型** | ❌ **否决** | 三重不兼容（时窗 / 通道 / 语义），且有第三方文献实测为 `Incompatible` |
| **BIOT 作为架构思想** | 🟡 **仅一个场景有价值** | 通道无关 tokenization 是 BI×GTN 2 导交集困局的**唯一**技术解 |
| **BIOT 作为域适应机制** | ❌ **不解决 T 的问题** | BIOT 做的是「跨格式联合预训练」，T 要的是「源域 → 极少 target 数据适配」 |
| **替代方案** | 🟡 SAMBA 是唯一对 ERP 兼容的 | 但增益未证明，且其 P300 精度不高于 T 的专用小模型 |

一句话：**BIOT 救不了当前的架构困境，它和 T 的痛点不在同一个维度上。**

---

## 2. BIOT 是什么（能力边界）

NeurIPS 2023，Yang / Westover / Sun（`github.com/ycq091044/BIOT`）。

- **核心设计**：每通道切成固定长度 segment → 每个 segment 做 STFT → 所有通道所有 segment flatten 成一条长「句子」→ 加 channel embedding（空间）+ relative position embedding（时间）→ Linear Transformer
- **解决什么**：通道数不匹配、长度可变、缺失值 —— 让不同格式的数据集能**联合预训练**
- **预训练语料**：SHHS（sleep EEG）、PREST（resting EEG）、Cardiology（ECG）
- **论文增益**：CHB-MIT 癫痫检测上，vanilla 超基线 3% BACC，预训练再 +4%

**注意它的任务谱**：sleep staging、癫痫检测、异常检测、ECG 分类。**没有 ERP / oddball / P300**。

---

## 3. 三重不兼容（硬证据）

### 3.1 时窗：epoch 比 STFT 窗口还短 —— 架构级致命伤

| 项 | 值 | 来源 |
|---|---|---|
| T 的 epoch | `sample_rate_hz=128`、`tmin_ms=-200`、`tmax_ms=1200` ⇒ **n_times = 179** | `src/data/contract.py:18-22, 36-37` |
| BIOT 分词 | STFT **window = 200 点**，hop = 100 点 @ 200 Hz（即 1.0 s / 0.5 s） | SAMBA 论文 §4.3 |

按 SAMBA 论文的做法把 T 的数据重采样到 200 Hz：179 点 → **280 点**。
在 280 点上以 window=200、hop=100 切分，能切出的完整窗口数：

```
floor((280 - 200) / 100) + 1 = 1
```

⇒ **每通道只得到 1 个 token。**

后果不是「效果差一点」，而是 BIOT 的核心能力**物理上无法启用**：
- 跨 segment 的时序建模退化为单点，relative position embedding 无意义
- 整个 1.4 s epoch 被压成一个频谱块，**P300 的 ~300 ms 时域锁相峰在频谱里被彻底抹平**

SAMBA 论文对这一点的原话：

> "In time-locked ERP paradigms, the P300 component (~300 ms long)... Such components are
> **highly time-sensitive and often span sub-second durations, making them vulnerable to
> distortion or loss under fixed-length segmentation**."

### 3.2 通道：checkpoint 绑死 18 通道 TCP 双极蒙太奇

- BIOT 最大 checkpoint = `PRESET+SHHS, 18 channels`
- NeuralBench 明确警告：`The pretrained positional embeddings are tied to the 18 TCP channels...
  there is **no guarantee that the resulting channels align with the original TCP bipolar semantics**`
- T 的通道谱：BI2014a 16 导（干）、BI2015a/2015b **32 导**（超 18 上限）、GTN **3 导**、目标硬件 **8 导干电极**

即便用 Conv1d 强行映射（NeuralBench 的做法），空间嵌入的语义已经错位。

### 3.3 语义：频谱化摧毁 ERP 的时域锁相信息

SAMBA 论文原话：

> "BIOT adopts a unified model across subjects but **relies on spectral-domain transformations
> (e.g., FFT), which may compromise temporal information**."

P300 的信息在**精确的时间锁定与波形形态**，不在稳态频谱能量的相对分布上。这个归纳偏置是反的。

---

## 4. 第三方实测：不是我推断，是被测过

SAMBA 论文 Table 5（`arXiv:2511.18571`），三个 P300 ERP 数据集
（P300-A/B/C：**2 类、64 通道、240 Hz、0.8 s**），linear probing：

| 模型 | P300-A | P300-B | P300-C |
|---|---|---|---|
| SAMBA-E | 56.26 | 57.76 | **61.68** |
| SAMBA-T | **58.49** | **59.22** | 61.28 |
| EEGPT | 55.79 | 54.00 | 57.36 |
| LaBraM | **Incompatible** | **Incompatible** | **Incompatible** |
| **BIOT** | **Incompatible** | **Incompatible** | **Incompatible** |

论文给出的原因，同时命中上面三条：

> "LaBraM uses a patch size of 200 samples (i.e., 1 s at 200 Hz), and BIOT performs STFT with
> 200-point windows and a hop length of 100... these two models remain **incompatible with
> short-duration datasets (e.g., 800 ms P300 datasets)** whose duration is shorter than the
> required patch or window size. Furthermore, the EEGPT and BIOT checkpoints are only compatible
> with input channel numbers **less than 58 and 18**, respectively."

⚠ 注意：这三组 P300 是 BCI Competition 的 64 导数据，与 T 的 BI（16 导）/ GTN（3 导）不同，
数字**不可直接对比** T 的 N2P3Net 结果。但 `Incompatible` 是二值判定，不受数据集差异影响。

---

## 5. BIOT 真正能救的唯一场景

BIOT 的通道无关 tokenization 在 T 的场景里只有一个不可替代的用处：

> **BI × GTN 的通道交集只有 2 导**（`cross_dataset_bi_gtn_ruling_20260904.zh.md` §4）。
> 2 导 CAR 只保留 `Cz-Pz` 差分方向，并删除选定集合的共模；本 cache 的朴素 AUC 消融下降，
> 但这不是“所有 P300 都被摧毁”的定理。传统「通道交集 + 统一 CAR」方案在此构成科学阻塞。
> BIOT 式 tokenization **可以避免固定通道交集这一维度约束**，但不会恢复被删除的参考/共模信息，
> 也不能自动解决单位、参考、空间语义、时间轴和范式差异；16 导、32 导、3 导数据不能据此直接
> 混合预训练，仍需独立的跨域合同与 held-out 验证。

而 BI 家族**内部**（12–32 导交集）已具备较低的空间秩亏风险，`data/domain.py` 的 common-channel CAR
可以提供统一参考口径，但任务信息是否保留仍需消融验证 ——
所以 BIOT 的卖点在家族内迁移里价值是被稀释的，只有在跨到 GTN 时才真正稀缺。

**但代价要说清楚**：要用它就意味着
1. STFT 窗口必须调到 ERP 尺度（比如 30–50 ms），
2. 调完就**用不了官方预训练权重**，等于自研一条新路线，
3. BIOT 论文那 +4% 的预训练增益**不复存在**。

---

## 6. 它不解决 T 的核心问题

| T 的问题 | BIOT 提供什么 | 匹配？ |
|---|---|---|
| 源域 → target 被试，target 只有几十~几百 trial | 联合预训练 + 微调 | ❌ 不同问题 |
| 目标硬件 8 导干电极，与训练数据蒙太奇不同 | 通道无关 tokenization | 🟡 但被 18 通道上限卡住 |
| 9 选 1 direct 语义无数据层实现 | 无关（BIOT 不建模候选结构） | ❌ |
| hit@R 的候选证据聚合 | 无关（BIOT 是 trial 级编码器） | ❌ |
| P0/P1/P2 三项域适应强制机制 | 无对应物 | ❌ |

T 现有的 P0 Split-BN / P1 drop-to-consensus / P2 决策级损失，针对的正是**少样本 target 适配**。
BIOT 类方法优化的是「怎么把更多异构数据塞进预训练」—— 两者互补而非替代。

**注意互补关系**：BIOT 让「更多源数据」成为可能，P0/P1 让「源数据迁移到 target」更有效。
理论上可以 BIOT 做源、P0/P1 做适配。但前提是先把 §3.1 的时窗问题解决掉。

---

## 7. 如果一定要引入外部基础模型

| 模型 | P300 兼容性 | 任意通道 | 任意时长 | 备注 |
|---|---|---|---|---|
| **BIOT** | ❌ 实测 Incompatible | ❌ 上限 18 导 | ❌ 需 ≥ 1 s | — |
| **LaBraM** | ❌ 实测 Incompatible | ❌ | ❌ patch 1 s | — |
| **EEGPT** | 🟡 可跑（55.79 / 54.00 / 57.36） | 🟡 上限 58 导 | 🟡 | P300 上精度偏低 |
| **SAMBA** | ✅ 可跑且最优 | ✅ 电极 xyz 坐标嵌入 | ✅ 长度无关 | 需自行验证 |

SAMBA 的 SAIE（基于电极三维坐标的空间嵌入）**与电极名称、蒙太奇无关**，
这点对 T 的「多导数据集预训练 → 8 导干电极硬件部署」是唯一天然契合的。

⚠ 但 SAMBA 在 P300 二分类上只有 **58–61% Acc（chance = 50%）**，
而 T 的 N2P3Net 在 BI2014a matched LOSO 上是 **AUC 0.7451 / BACC 0.6774**。
数据集不同不能严格对比，但至少说明：**ERP 基础模型目前并没有跑赢专用小模型**。

---

## 8. 建议

1. **不要把 BIOT 当作当前架构困境的解药** —— 它解决的是另一个维度的问题，且对 ERP 有三重不兼容。
2. **当前主线仍是 P0/P1/P2 + BI 家族域迁移**（D-10 待裁决：R_dev 降级 / 换数据 / AUC-only）。
3. **如果将来要做 BI × GTN 跨数据集联合预训练**（突破 2 导交集困局），
   借鉴 BIOT 的通道无关 tokenization —— 但**按 ERP 尺度重新设计 segment**，当作自研模块，
   不要指望官方权重。
4. **若要引入外部基础模型做对照臂**，优先 SAMBA（唯一对 ERP 兼容），而不是 BIOT。
   且必须做 matched 消融，不能只报 linear probing 数字。

---

## 9. 本次评估的未覆盖项

- BIOT 的 STFT 窗口是否为可配置超参数（论文按 1 s 设计，用于 sleep staging 的 30 s epoch）；
  若可调至 ERP 尺度，§3.1 的判定需重算 —— **但改后预训练权重作废，见 §5**。
- SAMBA 的 checkpoint 是否公开、许可是否允许商用（未核实）。
- 未实测 BIOT / SAMBA 在 T 的 BI2014a cache 上的真实数字（本轮为静态评估）。
- 库内 `EEG_FM_Compass_arXiv_2601.17883.pdf`、`Identity_Trap_in_EEG_Foundation_Models_arXiv_2606.06647.pdf`
  两篇未精读，可能在基础模型选型与身份泄漏上有更细的结论。
