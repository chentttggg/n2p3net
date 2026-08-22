# N2P3-Net 公开数据集链接集（下载与使用指南）

> 面向「oddball 范式 + P300 检测 + 认知判定」场景，按对口程度排序。
> 核实时间：2026-08-20。所有链接均已联网核实真实存在。
>
> ⭐ **首选方案：MOABB 自动下载**（见文末「零、MOABB 一键下载」）——它覆盖 ERP CORE P3、
> Brain Invaders、BNCI2014_008，且统一成 numpy 格式。只有 GTN 需手动 EEGBase。

---

## 零、MOABB 一键下载（推荐，替代手动下载 ①③ 及补充）

---

## 一、三个核心数据集（强烈建议全下）

### ① Brain Invaders bi2014a —— 最对口「硬件」（干电极 P300）

**这是与你硬件场景最匹配的公开数据：干电极 + P300 oddball。**

| 项 | 内容 |
|---|---|
| 场景 | 干电极 P300 oddball（Brain Invaders 游戏，36 符号网格，1 target + 35 non-target 伪随机闪烁） |
| 被试 | MOABB BI2014a().subject_list 为 **64 名**成人；文献口径 71 名/多 session，以实际缓存为准 |
| 电极 | **16 个 active dry electrodes（干电极）** |
| 格式 | .mat + .csv |
| 许可 | Zenodo 开放（CC-BY 类） |
| 下载 | https://doi.org/10.5281/zenodo.3266223 |
| 论文 | https://hal.archives-ouvertes.fr/hal-02171575 |
| 代码 | https://github.com/plcrodrigues/py.BI.EEG.2014a-GIPSA |

**用途**：这是唯一「干电极 + P300 oddball」的公开数据集，可直接验证 N2P3-Net 在干电极低 SNR 下的
P300 检测能力。注意任务不是「9 选 1 猜数字」而是「36 选 1 单目标检测」，可当作**干电极 P300 检测的
单试次二分类基准**（target/non-target），直接套我们的 evaluate 协议。

---

### ② GTN（Guess the Number）—— 最对口「任务」（猜数字 9 选 1）

**这是唯一「猜数字 9 选 1」的数据集，也是 mission 里 77.2% 锚点（Vařeka 2016）的来源，必须下。**

| 项 | 内容 |
|---|---|
| 场景 | 猜数字：被试心选 1–9 一个数字，数字随机闪现，实验者凭 P300 波形在线猜中 |
| 被试 | **250 名** 7–17 岁学龄儿童 |
| 电极 | **3 导：Fz / Cz / Pz**（鼻梁参考 + 耳接地） |
| 格式 | **NIX**（.nix，HDF5 容器；EEGBase 下载实际格式，另有 Data/*.txt 元数据） |
| 许可 | **CC BY-NC**（非商业）⚠️ |
| 论文 | https://www.nature.com/articles/sdata2016121/ （DOI 10.1038/sdata.2016.121） |
| 主下载入口 | https://eegdatabase.kiv.zcu.cz —— **注册后**下载包 `PROJECT DAYS P3 NUMBERS` |
| 备用（分被试） | Harvard Dataverse（250 个 DOI 分散，如 https://doi.org/10.7910/DVN/2K3C5W ） |
| 官方加载脚本 | https://github.com/moucek/guess_the_number_py |
| 读取库 | 本项目 `data/gtn.py`（h5py 直读 NIX + .txt 元数据，review v5/v6 已接入） |

**用途**：复现 77% 基线（roadmap Phase 1 验收）、做 9 选 1 命中率评估。注意：
- **下载要注册 EEGBase**，且许可 CC BY-NC（若未来商用需换许可）。
- 3 导是 8 导蒙太奇的子集（Fz/Cz/Pz 前 3 位），正好验证我们 review 里修的「缺失通道子集」路径。

---

### ③ ERP CORE（P3 任务）—— 最对口「人群」（成人 oddball）

**成人 oddball P300 的黄金标准，用于跨数据集迁移测试。**

| 项 | 内容 |
|---|---|
| 场景 | Active visual oddball：字母 A–E 随机呈现（各 p=.2），每 block 指定一个字母为 target |
| 被试 | **40 名**成人（neurotypical） |
| 电极 | 标准 30 导（10-20 扩展，含 Fz/Cz/Pz/P3/P4/Oz） |
| 格式 | .set（EEGLAB）/ BIDS |
| 许可 | **CC-BY-4.0 / CC BY-SA 4.0** ✅ |
| 官网 | https://erpinfo.org/erp-core |
| OSF 总入口 | https://osf.io/thsqg/ |
| P3 任务 Wiki | https://osf.io/etdkz/wiki/ |
| GitHub | https://github.com/lucklab/ERP_CORE |
| MNE 示例 | https://mne.tools/mne-bids-pipeline/stable/examples/ERP_CORE.html |

**用途**：成人 oddball P300（target 概率 0.2，比 GTN 的 1/9 更接近常规 oddball），30 导可截取
我们的 8 导子集（Fz/Cz/P3/Pz/P4/Oz 都有，PO7/PO8 需确认）。用于验证 N2P3-Net 在成人数据上的
跨数据集泛化。文件较大（P3 原始数据 ~GB 级），MNE 可直接读 .set。

---

## 二、两个补充数据集（P300 speller 经典基准，可选）

| 数据集 | 场景 | 被试 | 电极 | 下载 |
|---|---|---|---|---|
| BCI Competition III Dataset II | P300 speller（行列范式） | 2 | 64 导 | http://www.bbci.de/competition/iii/desc_II.pdf |
| BNCI Horizon 2020（008-2014 等） | P300 speller（含 ALS 患者） | 多个 | 8–16 导 | https://bnci-horizon-2020.eu/database/data-sets |

**用途**：P300 speller 是「行列强化」范式（含空间信息），与我们「猜数字」不同，但可作为单试次
P300 检测的经典对照（尤其 BNCI2014-008 是 8 导，与我们的 8 导蒙太奇规模一致）。

---

## 三、推荐使用顺序与数据角色（对应 roadmap / transfer_policy P9）

| 阶段 | 数据 | 角色 |
|---|---|---|
| Phase 1 基线复现 | ② GTN | **主域**：242-fold LOSO 训练/测试、分类头监督与最终验收 |
| Phase 2 干电极验证 | ① Brain Invaders | **辅助预训练域**：只做 target/non-target 二分类预训练或对照 |
| Phase 3 辅助迁移 | ③ ERP CORE P3 + ① + BNCI008 | **辅助域**：只做预训练或域对齐；每个 GTN fold 仍微调 |
| Phase 4 部署目标 | 自有 8 导成人数据 | 最终 zero/few-shot 验收，不混入 GTN |

辅助数据使用红线：不得与 GTN 试次拼接后联合训练主分类；不得进入 GTN 测试 fold；
辅助域精度不作为主验收。三臂/四臂实验协议见 `doc/transfer_policy.md`。

## 四、许可与合规提醒

- **GTN（②）是 CC BY-NC**：非商业可用，商用需与 EEGBase 联系换许可。这是「学术研究」没问题，
  但如果你后续要产品化，注意这一点。
- **ERP CORE（③）CC BY-SA**：可自由用，但衍生作品须同许可分发。
- **Brain Invaders（①）Zenodo 开放**：学术自由使用，引用原文即可。

## 五、格式对接（与现有代码的关系）

| 格式 | 我们的处理 |
|---|---|
| NIX（GTN） | `data/gtn.py` 读成 Raw + events + thought 元数据 → 走 `data/preprocess.py` |
| .mat/.csv（Brain Invaders） | 需写一个轻量加载器（`data/` 下新增），把 mat → (N, C, T) 试次 + 标签 |
| .set（ERP CORE） | MNE `read_raw_eeglab` → 走 `data/preprocess.py` |

> 三个数据集的通道都不是我们完整 8 导蒙太奇（Fz/Cz/P3/Pz/P4/PO7/PO8/Oz），下载后接入时：
> classic/riemann 用 `subset_channels`（存在通道子集），deep 零填充到 8 导——这正是 review P0
> 修复好的路径。

---

## 零、MOABB 一键下载（⭐ 推荐，替代手动下载）

**`pip install moabb`**（依赖 MNE，数据缓存在 `~/mne_data/`）。MOABB 把数据统一成
`X (n_epochs, n_channels, n_times) + y 标签 + metadata(DataFrame)`，直接对齐 baselines 契约。

### 覆盖情况

| 数据集 | MOABB 类名 | 关键参数 |
|---|---|---|
| ERP CORE P3（③） | `moabb.datasets.ErpCore2021_P3` | 40 成人、30 导、1024Hz、2 类、CC BY 4.0 |
| Brain Invaders（①） | `moabb.datasets.BI2014a` | 71 被试、16 干电极、512Hz、2 类 |
| **BNCI2014_008**（补充） | `moabb.datasets.BNCI2014_008` | 8 被试（ALS）、**8 导、256Hz** |
| GTN（②） | ❌ 无 | 仍需手动 EEGBase |

> ⭐ `BNCI2014_008` 电极 = `Fz, Cz, Pz, Oz, P3, P4, PO7, PO8`，**集合与我们的 8 导蒙太奇完全一致**，
> 是最对口电极配置的现成数据。

### 用法（选通道 + 重采样到 256Hz）

```python
from moabb.datasets import ErpCore2021_P3, BI2014a, BNCI2014_008
from moabb.paradigms import P300

paradigm = P300(resample=256, channels=["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"])
X, y, metadata = paradigm.get_data(dataset=ErpCore2021_P3(), subjects=[1, 2, 3])
# X: (n_epochs, 8, T) float；y: Target/NonTarget；metadata 含 subject/session/run
```

### 与手动下载的关系

- 用 MOABB 时，**ERP CORE P3、Brain Invaders、BNCI2014_008 都不用再手动下**。
- 唯一手动项 = **GTN**（EEGBase 注册下载，见上文 ②）。
- MOABB 数据已切好 epoch（1s trial），可**跳过 `data/preprocess.py` 的 epoch 切分**，直接进 baseline；
  但仍需按需重采样/去漂移（MOABB 的 paradigm `resample` 参数已可做重采样）。
