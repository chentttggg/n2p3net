# N2P3-Net 技术栈（Techstack）

> 深度设计（张量流/模块/损失/关键决策）见 blueprint.md。
> 版本：v2（吸收 review v1 + 成人为主）。

## 选型总表

| 层 | 选择 | 理由 | 否决的替代 |
|---|---|---|---|
| 语言 | Python 3.11 / 3.12 | 科学计算生态最稳 | 3.13（部分库 wheel 滞后） |
| 深度学习 | PyTorch（≥2.5） | EEG 生态 + braindecode 兼容；2.5+ 原生支持 CUDA/XPU 双后端 | TensorFlow（生态弱）、JAX（门槛高） |
| EEG 预处理 | MNE-Python | 重采样/连续域高通/epoch/伪迹剔除的事实标准 | 自研（重造轮子） |
| EEG 深度学习库 | braindecode | 现成 EEGNet/DeepConvNet/ATCNet 基线 | — |
| 传统基线 | scikit-learn + pyriemann | SWLDA/LDA + xDAWN/Riemannian | — |
| 统计验证 | MNE(cluster permutation) + LIMO | 成分可解释性对照 + 年龄/性别协变量 | — |
| 数值 | numpy / scipy / pandas | 标准 | — |
| 实验管理 | hydra（配置）+ wandb 或 MLflow | 可复现实验 | 硬编码参数 |
| 代码质量 | ruff + black + pytest | 统一风格与测试 | — |

## 关键选型说明

1. **PyTorch + braindecode**：braindecode 已实现并维护 EEGNet、DeepConvNet、ShallowConvNet、
   ATCNet、EEG Conformer 等基线，直接服务于宪法 P8「先基线后创新」。
2. **MNE-Python**：负责「重采样 + 0.1 Hz 连续域高通 + epoch 切分 + 阈值伪迹剔除」这些网络外
   允许项（P2）。注意：高通必须在 epoching 之前的连续数据上做；0.5 Hz 高通有 P3b 失真风险
   （Tanner 2015），仅作消融对照。参考无关化与幅值标准化在模型内以可微层（加权再参考 + 归一化）
   再做一遍，MNE 不做（或仅作对照预处理）。
3. **pyriemann**：xDAWN + 黎曼几何是 P300 的强基线（Kaggle BCI 竞赛冠军方案），必须纳入对照。
4. **hydra**：三层评估协议 + 多任务头 + 跨域组件 + 消融轴（depth/高通/参考抖动）参数矩阵大，
   必须配置化、可一键复现。
5. **归一化体系**：全程 InstanceNorm / LayerNorm / GroupNorm，**不使用 BatchNorm**；跨域对齐用
   「域条件仿射」（per-domain scale/shift），不用 Split-BN（BN 不存在时其为空操作）。
6. **环境隔离**：沿用项目 venv，禁止全局 pip install。
7. **设备可移植性**：代码须在 Intel Arc 130T（XPU）/ NVIDIA 5070（CUDA）/ CPU 三端无缝运行：
   动态检测 `DEVICE`（CUDA→XPU→CPU）、统一 `.to(DEVICE)`、AMP 用 bf16、batch_size 参数化。
   完整硬规则见 device-portability.md（DP1–DP6）。注意 torch 须按设备选源安装（§7），
   PyPI 官方 wheel 为 CPU 版、不带 CUDA/XPU。

## prior art 参考实现映射（每个组件 → 可借鉴的工作）

| 组件 | 参考实现 | 借鉴什么 | 规避什么 |
|---|---|---|---|
| 坐标式通道身份 | BrainOmni (arXiv:2505.18185) | 传感器坐标替代通道名 | 大规模预训练（数据量不够） |
| 参数化成分窗（位置寻址） | DETRtime (arXiv:2206.08672) | query 思想（但改位置寻址，弃 free cross-attention） | Hungarian 匹配 / query 坍缩（E2/E7） |
| 单试次潜伏期估计 | Depuydt 2023 (Brain Topogr 36) | NN 估计潜伏期、数据驱动先验 | 依赖模拟数据标注（E3/D7） |
| 多尺度时间卷积 | EEG-Inception (IEEE TNSRE 2020) | 多尺度核适配不同潜伏期成分 | — |
| 跨域对齐配方 | AS-MMD (arXiv:2510.21969) | 域条件仿射 + RBF-MMD + 目标加权 | 高估效果（E6）；勿用 Split-BN |
| 掩码自监督 | LaBraM / EEG2Rep (2024) | 掩码重建 + 轻量 token | 通道名匹配 / 大语料 |
| 去漂移高通依据 | Tanner, Morgan-Short & Luck (2015, Psychophysiology) | 0.1 Hz 连续域高通 | 0.5 Hz（P3b 失真，E1/D8） |
| N2 早期证据 | Kaufmann 2011 / Hong 2009 | 多成分协同优于单 P300 | 当作独立新任务（实为同标签集成） |
| 去漂移证据 | Clements 2016 (J Neural Eng) | δ/θ 频段漂移量化 | — |
| 基线实现 | braindecode + pyriemann | EEGNet/ATCNet + xDAWN | — |

## 依赖清单（示意）
mne, braindecode, torch, numpy, scipy, pandas, scikit-learn, pyriemann,
hydra-core, wandb(或 mlflow), ruff, black, pytest

## 数据源
- **GTN**（Moucek 2017, Sci Data 4:160121）——儿童（7–17 岁）3 导 P300，**跨年龄迁移源域** + 文献对照。
- **ERP CORE**（Kappenman 2021）视觉 oddball P3 + 听觉 MMN——成人 30 导 / 1024 Hz，异构格式迁移测试集。
- **自有 8 导干电极数据**——成人为主，主训练/评估集（含年龄/性别元数据）。

## 目录约定（建议）
```
n2p3-net/
├── constitution.md / mission.md / techstack.md / blueprint.md / roadmap.md
├── src/
│   ├── data/          # 格式无关适配器（重采样/连续域高通/坐标通道/掩码/元数据加载器）
│   ├── models/        # N2P3-Net（Stage0–Stage4，参数化成分窗 ×3）
│   ├── baselines/     # SWLDA/xDAWN/EEGNet/EEG-Inception/EEG Conformer + 两个免费地板
│   ├── stats/         # mass-univariate / cluster-permutation / LIMO（含年龄/性别协变量）
│   └── train/         # 训练配方（自监督/域条件仿射+MMD/多任务损失）
├── configs/           # hydra 配置（含消融轴 depth/高通/参考抖动）
├── experiments/       # 三层评估协议 + 消融 + 配对置换检验
└── tests/
```
