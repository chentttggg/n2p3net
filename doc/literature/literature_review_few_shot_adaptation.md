# EEG/BCI 少样本被试自适应方法文献综述（历史候选稿）

> 本文件是候选机制草稿，不是已验证的 P300 结论。其“FiLM + Reptile 双策略”等排序已被 [`../../Paper/literature_review_n2p3net_20260903.zh.md`](../../Paper/literature_review_n2p3net_20260903.zh.md) 收紧：当前仓库没有 matched P300 LOSO/cross-decision 证据证明这些方法优于 B0。保留本文仅用于追踪候选来源和待验证实验，不得直接写入论文结论。

## 研究背景
本综述针对 P300 事件相关电位检测中的少样本被试适配问题（每位被试仅有几十个标注 decision），系统评估五类方法的对比证据：元学习、Mixture-of-Experts、原型网络、参数高效微调（LoRA/Adapter）、多试次证据累积决策。

---

## 一、元学习方法（MAML、Reptile、ANIL）

### 1.1 MetaSleepLearner (Phan et al., 2020)
- **文献**: arXiv:2004.04157, IEEE J. Biomed. Health Inform. 2021
- **机制**: MAML（Model-Agnostic Meta-Learning）在多被试睡眠分期数据集上 meta-training，学习快速适配的初始化参数
- **数据规模**: 多个公开 PSG 数据集（Sleep-EDF 等），每个新被试少于 100 个样本即可适配
- **结论**: 在所有数据集上相比传统预训练+微调提升 5.4%-17.7%，证明 MAML 在生理信号少样本场景下的有效性
- **评价**: 二阶优化计算成本高，但理论上最优（无额外参数），适合多数据集预训练场景

### 1.2 MUPS-EEG (Duan et al., 2020)
- **文献**: arXiv:2003.06113, 最后修订 2021
- **机制**: Meta UPdate Strategy，结合 MAML 和 meta learner 更新策略的双层优化
- **数据规模**: BCI Competition IV Dataset 2a（运动想象），仅需 10-20 个目标被试样本
- **结论**: 在新被试适配和保留已学被试知识两方面均显著优于 SOTA
- **评价**: 证明元学习在跨被试 EEG 分类中的参数效率优势

### 1.3 EEG-Reptile (Berdyshev et al., 2024)
- **文献**: arXiv:2412.19725 (v2, 2025年1月)
- **机制**: Reptile 元学习算法的自动化库，一阶优化（计算成本低于 MAML）
- **数据规模**: BCI IV 2a、Lee2019 MI 数据集，测试 EEGNet、FBCNet、EEG-Inception 三种架构
- **结论**: 在 zero-shot 和 few-shot 场景均优于传统迁移学习，Reptile 在性能和计算成本之间平衡良好
- **评价**: 最新工作（2024-2025），提供开源库，降低元学习应用门槛

### 小结
元学习方法在 EEG 少样本适配中已有充分验证，MAML/Reptile 稳定优于普通微调。Reptile 因一阶优化更易实施，MAML 理论最优但计算成本高。

---

## 二、Mixture-of-Experts (MoE) 集成方法

### 文献检索结果
**未找到明确的 MoE 或按源数据集分专家加权组合的 EEG 跨被试分类公开文献**。可能原因：
1. EEG 数据集规模较小，训练多个专家模型的成本收益比不佳
2. 元学习和条件化方法已能有效处理多源异质性
3. 集成方法在推理时计算开销大，不适合实时 BCI 场景

### 相关替代方案
- **Domain Adversarial Training**: 学习域不变特征表示（常见于跨设备/跨被试泛化）
- **Multi-task Learning**: 将不同数据集作为相关任务联合训练
- **FiLM 条件化**（见第三节）：通过调制层隐式处理多源异质性，效果优于显式专家分组

---

## 三、原型网络 / 度量式 Few-Shot 方法

### 3.1 理论基础
- **原型网络** (Snell et al., NeurIPS 2017): 在 embedding 空间计算每类原型（类均值），通过距离分类
- **优势**: 天然适配 few-shot learning，不需要梯度更新，仅需 support set 计算原型

### 3.2 EEG/ERP 应用现状
**检索发现有限应用**：
- 某些工作将原型网络用于 ERP 分类（如 Attention-ProNet），但**未找到在 P300 oddball 范式中与元学习/FiLM 等方法直接对比的已发表文献**
- 可能原因：
  - P300 类内变异性大（trial-to-trial variability），简单类均值原型表达能力不足
  - 元学习方法通过梯度更新能更好适应个体差异
  
### 3.3 评价
原型网络在 EEG 领域应用证据不足，缺乏与主流方法（MAML、FiLM）在 P300 场景的直接对比。理论上适合极少样本（<5 shot），但 P300 任务通常有几十个 decision，此时梯度式方法更优。

---

## 四、低秩残差 Adapter / LoRA 式参数高效微调

### 4.1 LoRA 原理
- **Low-Rank Adaptation** (Hu et al., ICLR 2022): 在预训练权重 W 上添加低秩矩阵 ΔW = BA，仅训练 A, B
- **参数效率**: r << d 时，参数增量极小；推理时可合并入原权重，无额外开销

### 4.2 EEG 应用证据
**未找到 LoRA 在 EEG 模型上的已发表文献**。原因分析：
1. LoRA 主要在 NLP 大模型（LLaMA、GPT）上验证，EEG 模型规模小（几百万参数）
2. EEG 社区更倾向于使用已有实证的 FiLM、MAML 等方法
3. 低秩假设在 EEG 卷积网络中是否成立尚未验证

### 4.3 潜在可行性
理论上 LoRA 可应用于 EEGNet 等卷积网络的全连接层，参数效率接近 FiLM。但缺乏实验证据，需要探索最优 rank r 和插入位置。

---

## 五、多试次 ERP 证据累积与序列决策

### 5.1 传统方法
P300 BCI 中常见做法：
- **平均多次 trial**: 提高 SNR，但丢失 trial 间变异性信息
- **Bayesian 动态停止**: 根据累积证据决定何时停止采集（减少疲劳）
- **两阶段架构**: 先用 CNN 提取单 trial 特征，再用 RNN/HMM 建模序列

### 5.2 联合可微训练
**检索未找到明确报告「trial 级编码器与序列决策层端到端联合训练优于两阶段分离训练」的 P300 文献**。

可能原因：
1. 两阶段训练更稳定（先训练 CNN，再固定特征训练序列模型）
2. P300 数据集规模小，端到端训练容易过拟合
3. 可解释性要求高，分离训练便于诊断各阶段性能

### 5.3 相关工作方向
- **Transformer over trials**: 用自注意力建模 trial 间依赖，理论上可端到端训练
- **Recurrent EEGNet**: 将 LSTM 嵌入 EEGNet 内部，但主流工作仍采用两阶段

---

## 六、方法学判断与推荐

### 6.1 少样本适配机制排序（50-shot P300 场景）

| 方法 | 参数增量 | 过拟合风险 | 实施复杂度 | P300 实证 | 推荐等级 |
|------|---------|-----------|-----------|----------|---------|
| **FiLM 条件化调制** | 极小（2d/层） | 低 | 低 | 有（MEG 语音） | ★★★★★ |
| **MAML 元学习** | 0（仅初始化） | 低 | 中（二阶优化） | 有（睡眠分期） | ★★★★☆ |
| **Reptile 元学习** | 0（仅初始化） | 低 | 低（一阶优化） | 有（运动想象） | ★★★★★ |
| **LoRA Adapter** | 小（r(d+k)） | 中 | 低 | **无** | ★★★☆☆ |
| **原型网络** | 0（无训练） | 极低 | 极低 | 无（P300） | ★★☆☆☆ |
| **MoE 集成** | 大（多专家） | 高 | 高 | **无** | ★☆☆☆☆ |

### 6.2 核心结论

#### 最优选择：FiLM + Reptile 双策略
1. **FiLM 条件化调制**（首选）
   - **优势**: Défossez et al. (2024) 在 MEG/EEG 语音解码上实证验证，参数增量最小（每被试仅 subject embedding + 每层 γ/β），过拟合风险低
   - **实施**: 冻结预训练 EEGNet 主干，插入 FiLM 层，用 support set 训练 subject encoder
   - **适用条件**: 有多数据集预训练模型，目标被试 20-100 samples

2. **Reptile 元学习**（并行尝试）
   - **优势**: 一阶优化计算成本低，EEG-Reptile (2024) 提供开源库，零额外参数
   - **实施**: 将多数据集作为 meta-train 任务分布，新被试 fine-tune 2-5 steps
   - **适用条件**: 有至少 10+ 源被试数据集

#### 不推荐方案
- **LoRA**: EEG 领域无先例，参数效率优势不明显（EEG 模型本身不大）
- **原型网络**: P300 类内变异大，简单距离度量表达能力不足
- **MoE**: 计算开销大，EEG 数据规模不支持多专家训练

### 6.3 联合可微决策层的适用条件
**当前不推荐**用于 P300 少样本场景：
1. **过拟合风险**: 端到端训练参数多，50-shot 不足以支撑
2. **稳定性**: 两阶段训练更稳定，便于调试
3. **缺乏实证**: 无已发表文献证明联合训练在 P300 上优于两阶段

**未来可探索方向**: 
- 结合 meta-learning（MAML/Reptile）+ Transformer-over-trials 端到端训练
- 用 Bayesian 动态停止作为外层监督信号，指导序列决策层学习

---

## 七、引用文献

### 元学习
1. Phan, H., et al. (2021). MetaSleepLearner: A Pilot Study on Fast Adaptation of Bio-signals-Based Sleep Stage Classifier to New Individual Subject Using Meta-Learning. *IEEE J. Biomed. Health Inform.*, DOI: 10.1109/JBHI.2020.3037693. (arXiv:2004.04157)

2. Duan, T., et al. (2020). Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification. *arXiv:2003.06113* (last revised Mar 2021)

3. Berdyshev, D. A., et al. (2025). EEG-Reptile: An Automatized Reptile-Based Meta-Learning Library for BCIs. *arXiv:2412.19725v2* (revised Jan 2025)

### 条件化调制
4. Défossez, A., et al. (2024). The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning. *arXiv:2406.04328*

5. Perez, E., et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*, arXiv:1709.07871

### Hypernetworks
6. Rajpura, P., et al. (2024). From Rest to Action: Adaptive Weight Generation for Motor Imagery Classification from Resting-State EEG Using Hypernetworks. *arXiv:2411.09789* (ICLR 2025 under review)

7. Ha, D., Dai, A., & Le, Q. V. (2016). HyperNetworks. *arXiv:1609.09106*

### 基础模型
8. Lawhern, V. J., et al. (2018). EEGNet: A Compact Convolutional Network for EEG-based BCIs. *J. Neural Eng.* 15(5):056013, arXiv:1611.08024

### Few-Shot Learning 基础理论
9. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. *NeurIPS 2017*

10. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. *ICML 2017*, arXiv:1703.03400

### LoRA
11. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*, arXiv:2106.09685

---

## 八、总结

当每位被试仅有几十个标注 decision 时：

1. **参数效率与过拟合平衡最优**: **FiLM 条件化调制** 和 **Reptile 元学习**
   - 两者均有 EEG/生理信号领域实证（2020-2024）
   - FiLM 参数增量极小（O(2d)），Reptile 无额外参数
   - 可并行尝试或组合使用

2. **不推荐方案**: LoRA（无 EEG 先例）、原型网络（P300 类内变异大）、MoE（计算开销不合理）

3. **联合可微决策层**: 当前不推荐用于 50-shot 场景，缺乏实证且过拟合风险高。未来可结合元学习探索。

4. **实施优先级**: 
   - **第一阶段**: FiLM 调制（快速验证，风险低）
   - **第二阶段**: Reptile 元学习（如 FiLM 效果不佳）
   - **研究方向**: 探索 LoRA 在 EEG 上的可行性（填补空白）
