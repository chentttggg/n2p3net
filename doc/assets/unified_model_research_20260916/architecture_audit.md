# 统一编码器、个人适配与训练阶段：架构独立审查

日期：2026-09-16。范围为设计与局部数学验证；未改生产模型、未执行真实训练或启动云端。已阅读当前六指南相关条款及 N2P3Net、FeatureDomainTransfer、SourceObjective、SubjectAdapter、checkpoint 重建链。前轮算法修复与云端冻结矩阵各自保留其执行身份。

## 1. 独立裁决

建议把**同一紧凑编码器的源中心 MAP 全参数适配**作为统一图的首个主候选，同时保留**现有残差模块**与**中间非线性前的 affine 适配**两个可比较限制。三者共用输入、编码器结构、head、阶段接口、拟合范围和保存格式；个人自由度由声明的参数组与先验决定。不要先建立一个“模型选择器套三个产品模型”的新常态。

理由是当前六导网络仅 1202 个参数，其中 head 已有 354 个。新增 FiLM 并不能把个人拟合变成几十个参数的任务；FiLM+可训练 head 仍有 394 个参数。现成残差路径需训练 431 个。全参数 MAP 可连续控制偏离源模型的程度，也允许在有证据时适配早期空间/时间特征。其收益和正则强度仍需源内开发验证，不能因模型小而认定不会过拟合。

本轮加载的迁移 skill 把 FiLM 列为既定首选，并引用 MEG 语音证据。本审查不接受这个排名作为本项目结论：下文给出 FiLM 的表达冗余反例、实际参数量以及适用边界。现有 P1/P2 结果在完整返回前也不能提前决定统一架构。

## 2. 真实参数量和三个候选

实测配置为当前默认 `full_unfold`、六导、128 Hz、179 点。post-MST 张量为 `(B,4,44)`。参数按实际 module 枚举，而非配置字符串估算：

| 当前组成 | 参数量 |
|---|---:|
| 空间投影 | 96 |
| 时间卷积与 MST depthwise/pointwise | 696 |
| BN affine | 56 |
| 读出 | 354 |
| 六导 core 总计 | 1202 |
| 七导 core 总计 | 1218 |
| 当前 FeatureDomainTransfer wrapper 总计 | 1344 |
| wrapper 的 target residual | 77 |
| wrapper 的 source extensions | 65 |

这些数值不适用于未来电极集合编码器或不同窗口；六导与七导空间权重维度不同，不能直接给同一 checkpoint 改通道数。

| 候选 | 图与个人自由度 | 六导总参数 | 个人可训练参数，BN affine 固定时 | 主要可证伪问题 |
|---|---|---:|---:|---|
| A：源中心 MAP | 相同 core；个人参数从源参数开始，按组惩罚偏移 | 1202 | 非 BN 全网 1146；含 BN affine 则 1202 | 限制自由度是否比有原则的连续收缩更好？ |
| B：结构残差 | 复用现有 FeatureResidualAdapter，post-MST 后接原 head | 最小图 1279；现有 wrapper 为 1344 | residual 77 + head 354 = 431 | 能否以更少个体自由度修正时序/特征混合，而不损失 full 的收益？ |
| C：中间 affine | st_spatial BN 后/ELU 前及两条 MST BN 后/ELU 前，共 20 个 feature 通道 | 新增 FiLM 时 1242 | affine 40 + head 354 = 394 | 相同位置 BN affine 的适配是否已经给出全部可用效果？ |

C 可直接以已有 BN affine 的相对源偏移实现，模型参数仍为 1202，无须再添加一层 FiLM。两种表示应先做函数等价验证，再比较正则化/初始化造成的优化差异。C 不能同时释放同位置原 BN affine 与独立 FiLM 后把两套参数当成不同生理机制。

P1 实际可训练量供对照：head 354、spatial+head 450、temporal+head 1050、非 BN 全网 1146、BN affine+head 410、residual+head 431、full 1202。在线 BN buffer 不计入可训练参数，但属于改变的拟合状态，必须单列。

## 3. 表达范围和初始化反例

### 3.1 末端 FiLM 可折叠到完全可训练的线性读出

对展开特征 z，令同一 feature 通道共享的 FiLM 展开为 Dγ 和 bβ：

\[
W(D_\gamma z+b_\beta)+b=(WD_\gamma)z+(b+Wb_\beta).
\]

只要 W,b 自由，末端 FiLM 没有增加该映射的函数集合。实测 float64 折叠最大误差 `2.842170943040401e-14`。它可能改变优化或先验，但不能把这种收益归因于新的表征能力。

将 affine 放到 ELU 之前则不同：一般 `ELU(γh+β)` 不能通过对 `ELU(h)` 仅加线性 head 得到。25 个一维点的独立最小二乘反例中，对 `ELU(x+1)` 拟合 `a*ELU(x)+b` 的最小二乘解仍有最大误差 `0.37582746776547965`。这证明某种表达差异，不证明完整 EEG 网络上的性能优势。

### 3.2 BN 的位置决定 affine 是否可辨识

训练态、同一通道的 batch mean/variance 下：

\[
BN(\gamma h+\beta)=
\operatorname{sgn}(\gamma)\frac{h-\mu}{\sqrt{v+\epsilon/\gamma^2}}.
\]

平移被消去；正比例缩放在 epsilon 相对方差很小时近似消去。实测正 scale/shift 经训练态 BN 后最大差为 `1.82e-11`（epsilon=1e-12）和 `1.82e-4`（epsilon=1e-5）。这不适用于固定 running statistics 的同一结论，也不适用于跨通道混合的所有归一化形式。

BN 后 FiLM 又满足：`γF*(γBN*h+βBN)+βF` 可合并为新的 BN affine。所以 C 的诚实对照是相同位置的 BN affine 适配；若未来 affine 由校准上下文生成器预测，生成器的容量、输入与 fit 范围才是新增机制。

### 3.3 现有残差初始化有可学习入口，但双零会停住

当前模块为

\[
R_{g,\psi}(h)=h+\rho\tanh(g)r_\psi(h).
\]

在 g=0 且随机 rψ 非零时，输出严格等于 h；`∂R/∂ψ=0`，但 `∂R/∂g=ρ*rψ(h)` 一般非零。实测初始 identity 误差为 0，gate 梯度范数 `0.8624349236488342`，其余内部参数首步梯度为 0。若再把 `up.weight` 与 `up.bias` 同时置零，gate 和全部内部参数梯度均为 0。不能把 gate 零初始化和输出层零初始化叠加，除非明确证明另有训练入口。

还需改正“bounded residual”的解释：`|ρ tanh(g)|≤ρ` 仅限制 gate，不能限制 `||R(h)-h||/||h||`。只将 up.bias 设为 100，现有 ρ=.25 的实测比值即约 `19.02`。这个反例用于拒绝错误数学保证；不把大 bias 当成当前真实模型常见故障。最小实现应改为准确命名并记录幅度，是否增加真正的函数幅度约束另作科研决策。

源预训练后 identity-at-init 应解释为“个人开始前与保存的源/元模型预测一致”。若源阶段已经学出非零 gate，再在每名新参与者开始时强行将 gate 清零，会改变预训练函数；统一图中不能这样重置。只有新模块首次接入时才使用结构 identity 初始化。

## 4. 同一计算图的三阶段数学

统一预测图写为

\[
s=f_{\theta,\phi}(\mathcal P_{\eta}(X,M)),
\]

其中 θ 为共享编码器/读出参数，φ 为声明的个人自由度，η 为 QC、normalizer、BN 等拟合状态；M 保留真实可用性。阶段改变谁能更新、从何处初始化以及损失，而不在目标路径临时插入未经源训练的另一套 encoder。

**阶段 S：普通源预训练。** 从合法源学习 θ₀,φ₀，首先保留 CE 的同剂量对照。当前 SupCon/MMD 可作为预声明源附加项，其参数/域/参与者支持及损失质量维持现有合同。不要把跨多个源的 domain alignment 直接搬到只有一名参与者的校准批次，制造无意义的单域辅助项。残差候选从首次 identity 初始化开始源训练；个人阶段从保存的完整源函数初始化。

**阶段 E：源内 episodic 适配训练。** 一个 episode 是同一源参与者的合法早期支持 Sₑ 和随后 query Qₑ；训练/查询按记录、session、block 和完整 raw 支持隔离。任务不是随机挑两类图像式 N-way relabeling。目标被留出的整个研究库、参与者及统计不进入 episode。

对 A 的组别 g，定义实际 inner 目标

\[
J_e(\vartheta;\theta_0)=L_{S_e}(\vartheta)+
\frac12\sum_g\lambda_g\|D_g(\vartheta_g-\theta_{0,g})\|_2^2.
\]

Dg 是源阶段预声明且冻结的尺度，具体归一化与分组见第 9 节；identity 仅作为未归一化 raw-L2 对照，不在目标数据估计精度矩阵。λ=0 恢复无源中心惩罚的完整微调。若同时使用 AdamW，须把其对零的 weight decay 与上述对源的惩罚分别记账，不能把 weight decay 冒称源中心 MAP。

\[
\vartheta_e^{(k+1)}=U(\vartheta_e^{(k)},\nabla J_e),
\quad\vartheta_e^{(0)}=\theta_0,
\quad\min_{\theta_0}\mathbb E_e L_{Q_e}(\vartheta_e^{(K)}).
\]

B/C 仅把 inner 更新限制到对应 residual/affine/head；outer 仍可学习 encoder，形成与其实际校准过程匹配的目标。ANIL 是 head-only inner 的源内控制，不能据图像结果预先判定 EEG trunk 无须适配。

使用实际 N2P3Net、固定 BN buffers 与 `torch.func.functional_call`，已验证一步 exact MAML 的 query loss 与全部二阶 meta-gradient 有限，16 个参数 tensor 的 meta-gradient 非零，BN buffers 不变。因此本网络没有必须先退化为一阶近似的已知软件障碍；多步成本仍待测。首轮比较少步 exact 与一阶近似，记录真实 inner/outer forward-backward 数；200 次普通目标更新不能被命名为 200 次等价 meta 更新。

另外已攻击空间 max-norm 激活区：将空间原始权重放大 100 倍，最小原始滤波器范数为 `39.18896`，有效范数最大为 `1.000000119`（float32 舍入），连续执行两个 functional inner 更新。全部 meta-gradient 有限；原始参数 mapping、model 本体参数与 BN buffers 均逐值未变。当前 `_MaxNormSpatialConv.effective_weight()` 返回 `weight * clamp(max_norm / norm)`，没有 `.data` 赋值或 in-place 投影，故此处无须为避免状态污染退化成 FO。

该验证位于严格出界的光滑分支；恰好 `norm=max_norm` 的分段边界不保证唯一经典二阶导数。源中心 MAP 当前约束原始参数，而函数只读取有效投影，径向冗余会影响先验解释，应明示参数化。若未来改成 optimizer 步后投影，exact meta 必须把该投影作为返回新 fast tensors 的更新映射并保留其 Jacobian；不能在共享参数上 `no_grad/.data` 改写后仍称 exact。FOMAML 也不能豁免 episode 间的参数隔离。

若 inner 用 SGD 一步，meta Jacobian 为 `I-αH_S`。FOMAML 省略 Hessian 项是近似；将 fast tensor 全部 detach 而不建立正确 outer 更新连接会使元梯度消失。现有 SubjectAdapter 的 in-place optimizer 训练对象不能直接当可微 inner-loop；应把已有 loss、参数组和批次合同用于一个显式 functional episode 接口，普通部署仍复用既有拟合器。

iMAML 只作后续候选。其隐式形式需要 inner 解满足相应驻点条件，Hessian+先验精度可解；非凸模型加任意小 λ 不能保证正定，也不能把有限步未收敛结果直接称为精确隐式解。当前网络小、实际需要的校准步数尚未验证，暂缺采用该复杂度的收益依据。

**阶段 C：本人校准与部署。** 只消费授权的 C，候选选择仅在合法 inner T/V 上进行，最终从同一 θ₀,φ₀ 在完整 C refit。需要 η 时各 episode/inner 只在其支持集拟合；完整 refit 重新拟合的状态按既有 policy 合同保存。已冻结 raw QC/EOG 则保持来源与实际输入一致。

后续未知目标 E 只执行保存的 f 与 η，不使用 source EEG、source logits、支持样本队列、原始 calibration 行或源参与者索引。部署包应自足包含最终权重、必要参考/布局/normalizer/BN/QC 状态和完整结构/角色声明。来源摘要继续保存，训练矩阵不进入推理依赖。λ、原 θ₀ 只有继续校准时才需要，不能让普通推理读取训练数据或重新拟合。

## 5. 归一化、mask 与校准状态的反例

1. **query BN 污染。** query 即使没有标签，也会在 train-mode BN 更新 buffers 并使一个 query 的预测依赖其他 query。主方案令 query/inference 固定使用 support 拟合后冻结的状态；query 内容/顺序/分批不能改变拟合状态。若以无 running stats 的替代归一化研究另一种机制，须独立声明，不能在 episode 里静默切换。
2. **输入统计与 BN 不得跨 episode 混用。** 支持集统计仅影响本 episode；下一 episode 从约定源状态重置。meta-gradient 不应穿过一个来自其他参与者/未来 query 的共享 mutable buffer。
3. **BN/FiLM bias 会复活缺失电极。** 仅在 raw/标准化后令缺失导为零，不保证逐电极 temporal block 之后仍为零。独立两导反例中，缺失导经过 affine 的 β=.7 后进入 spatial sum，输出从合法的 1.2 变成 1.9。若统一编码器保留逐电极中间表征，应在该层完成 affine/归一化后、空间混合前再次应用物理 mask。post-MST 已混成四个 latent feature，不能将六个电极 mask 生硬广播到它们上。
4. **后验屏蔽不能撤回先前污染。** 已被 spatial mixing、BN population statistics 消费的坏导数值，不能靠最终 logits 后的 mask 修复。永久坏导先剔除、局部支持从前端传播、保留波形再映射零槽的现有输入修复继续有效。是否把合法缺导模式作为明确模型输入，应由统一输入 owner 裁决并测试，不能从零值猜测缺失。
5. **可缓存特征的条件。** full/MAP、空间/时间更新、在线 BN 或 dropout 变化时，不能继续复用“冻结特征”。B/C 只在 adapter 之前的 encoder 与状态固定时允许共享前级缓存；meta outer 改 θ 后缓存须失效。

## 6. 最小比较与判决方式

先固定 encoder/input/operator/目标预算，对 A/B/C 做同实际 K、同 source 初始化来源、同支持/query 行与合理同等开发机会的比较。现有全网微调与 scratch、波形 LDA 保留为外部读数。不要把“参数少”作为自动通过条件。

首先比较 S→C，再比较 S→E→C；episode 方法相对普通源训练的增量必须包含 source 训练计算成本。主要问题是 K200 的同人适配收益、K800 的不劣化与负迁移尾部；精确 K 和回退 K 分开。若 B/C 不能稳定超过合理收缩的 A，则统一图保留 A 的参数组/先验接口，而不让无收益模块继续堆积。若预声明的较少更新成本分支胜出，也须在独立支持/query 上验证，不能用同一开发结果重新定义门槛。

需要的数学/软件验收包括：source/person 初始化预测一致；FiLM-head 折叠和 BN-affine 等价；单零/双零梯度；episode 参数/optimizer/BN 独立性；替换 query 或未来测试不改支持轨迹；有限差分核对少步 meta-gradient；mask 缺导反例；K0 无目标 fit；最终模型保存重载与推理 batch/order 不变。现有本轮局部 probe 只覆盖其中的参数量、函数等价、初始化梯度及一步实际 encoder 二阶可微性。

## 7. 代码联动清单

| 现有 owner | 必须对齐的职责 |
|---|---|
| `models/n2p3net.py`、`models/parameter_groups.py` | 一个结构 owner，声明 encoder/readout/adaptation 实际 module；完整覆盖且不重叠；禁止 stage 创建隐式新参数 |
| `models/adapters.py` | 复用 FeatureResidualAdapter；统一图中移除不消费的 source private bank 后另立准确新结构身份；原 1344 参数 wrapper 不可重标为 1279 |
| `transfer/recipes.py`、`subject_adapter.py` | fast 参数组、源中心先验、BN/dropout、初始化、更新预算显式化；保存实际状态；失败后拒绝推理 |
| `baselines/deep.py`、`transfer/source_objective.py` | 普通源目标继续用同一 encoder；当前 fixed-update plain-only 门禁须随统一结构精确迁移，不能绕门禁传 wrapper |
| `research/source_selection.py`、`source_execution.py`、`source_resume.py` | episode 的合法参与者/记录支持及 query 来源、预算与 seed 进入任务/轨迹签名；学习过程变化不得复用旧 source 轨迹 |
| `transfer/calibration_candidates.py`、`target_experiments.py` | 复用既有 inner/refit 划分与身份检查；参数组/先验和所有拟合状态进入缓存/结果复用键 |
| `transfer/checkpoint.py`、`calibrated.py` | source 和 personal 重建相同结构；完整 source/person 状态、adaptive groups、normalizer/BN 与代码身份进入包；不需训练矩阵即可推理 |
| `experiments/run_pretrain_supervised.py`、`run_source_training_chain.py`、公共目标 runner、`brainsync_model_task.py` | 阶段成为训练程序声明，source/meta/calibration/inference 权限与参数传播闭合；不新增一套独立产品 runner |

当前源码 SHA-256：N2P3Net `48a1d171f779c92c5b4126c0bd19a7291e5ea4d58fd12bcf60b16dbee1c9ea86`；adapters `1a5555e6f04c492a096acaf3c3f15f6cde1af9378b0612435d8151ff0a071b70`；source_objective `99eb8a2e6650104d600ef9febe51e3a8233800a085091fff3a4f992021225f09`；subject_adapter `0bc643cbfa54dc62b3421739479e88a49242cc2fda0e9daacd7dcfcc6a5c6ebc`；checkpoint `7cc5efeabc61d10c2499fe7cc98928d370780040bb77091b3af4cbbb35f19e4b`。

## 8. 本次在线核查的原始资料

共 8 次有针对性的官方 arXiv 页面请求，均成功；核查层级均为**官方标题与摘要**，没有将未读全文的实验设置、数学定理条件或百分比外推到本项目。以下机制公式由本次独立推导/数值 probe 支撑；原论文的 EEG/P300 效能按其原范围解释。

| 原始资料 | 本次可支持的范围 | 对本项目的限制 |
|---|---|---|
| Finn et al., [MAML, arXiv:1703.03400](https://arxiv.org/abs/1703.03400) | 直接训练可少步适配的初始化；分类/回归/RL框架 | 摘要证据为图像等任务，不是跨库 P300 验证 |
| Raghu et al., [ANIL, arXiv:1909.09157](https://arxiv.org/abs/1909.09157) | head-only inner 可用于区分特征复用与快速 trunk 适配 | 不能据图像/RL结果认定 ERP 也只需 head |
| Rajeswaran et al., [iMAML, arXiv:1909.04630](https://arxiv.org/abs/1909.04630) | 以隐式梯度解耦 inner 优化路径，处理多步计算/内存问题 | 本次未逐条核查全文定理条件，不承诺任意非凸 EEG inner 都适用 |
| Perez et al., [FiLM, arXiv:1709.07871](https://arxiv.org/abs/1709.07871) | 条件化逐 feature affine 的通用机制 | 原文是视觉推理；不支持 P300 的预设排名 |
| Lawhern et al., [EEGNet, arXiv:1611.08024](https://arxiv.org/abs/1611.08024) | 紧凑 depthwise/separable EEG CNN，摘要明确包含 P300 | 不认证本项目窗口、mask、跨库和成人九选一结果 |
| Jayalath et al., [The Brain's Bitter Lesson, arXiv:2406.04328](https://arxiv.org/abs/2406.04328) | 跨参与者/数据集 MEG 语音解码和自监督规模研究 | 摘要不能核实具体 FiLM 排名；MEG 语音不等于 EEG P300 |
| [MetaSleepLearner, arXiv:2004.04157](https://arxiv.org/abs/2004.04157) | MAML 的个体生理信号适配先导，睡眠分期 | 不是 oddball/P300，且少数睡眠 epoch 预算不能换算成 flash 数 |
| [MUPS-EEG, arXiv:2003.06113](https://arxiv.org/abs/2003.06113) | 跨人 EEG meta update、适配与旧任务保持的研究 | 摘要没有足够细节确认与本项目相同范式/划分；旧任务保持也不是只读推理所必需 |

本目录 `architecture_math.py` 为独立验证脚本，从文件位置解析项目根目录，使用项目 `.venv` 执行；默认输出同目录 `architecture_math.json`，也可传 `--output`。首次临时探针因 PyTorch 不允许训练 BN 的 epsilon=0 而退出；改用正 epsilon 并明确理论极限后最终退出码为 0。脚本已函数化并通过 Ruff format/check；该局部脚本没有修改生产源文件。

## 9. core 全参数 MAP 与非线性残差共存时的参数规范

若总方案采用一个同时包含 core 与现有非线性 residual 的模型，本节补足其先验定义。前述 A/B 变成同一图的可训练参数组消融，C 用于检验 affine 是否已足够；共同图本身不构成两种机制都必要的证据。

### 9.1 源中心 L2 不是函数空间中唯一的距离

原始参数先验 `||θ-θ₀||²` 依赖坐标。当前网络已有可构造的相同函数、不同参数距离：

- MST 的 depthwise D 与其后的 pointwise P 之间没有非线性。逐输入 feature 作 `Dₖ→cDₖ`、`P[:,k]→P[:,k]/c`，卷积组合不变，但两块参数的 L2 偏移分别乘 c²、c⁻²。
- temporal BN 的 affine 输出后立即接空间线性投影。在空间 max-norm 两个表示均未激活的范围内，`(γₖ,βₖ)→c(γₖ,βₖ)`、对应空间权重 `Wₖ→Wₖ/c` 保持该局部函数不变。训练态后续 BN 还可能近似抵消正 scale；有限 epsilon 和状态冻结条件按第 3 节处理。
- 当前有效空间权重为 `Pρ(W)=W*min(1,ρ/||W||)`。若两个同方向权重范数均超过 ρ，继续径向放大不会改变有效权重，raw L2 却继续改变。

因此有限 L2-SP/MAP 是**在固定模型参数规范下的先验选择**。不能把它描述为重参数化不变的“距离源函数多远”。分组归一化减少任意尺度的影响，仍不能消除所有跨层、逐通道规范自由度。

### 9.2 推荐首版归一化先验

不从个人 C 或 query 重新估计先验尺度。源 checkpoint 固定后，对每个普通 weight tensor l 保存

\[
s_l^2=\operatorname{mean}(W_{l,0}^2)+\sigma_{init,l}^2,
\quad
R_l=\frac{\|W_l-W_{l,0}\|_F^2}{2n_l s_l^2}.
\]

`σinit,l²` 来自该层**实际声明的初始化分布**；当前 PyTorch 默认 Conv/Linear 的均匀初始化对应 `1/(3*fan_in)`。自定义初始化须保存自己的数值，不能从层名猜。这个正尺度对源零权重也有定义；尺度是固定常数，不参与元梯度优化。若进入 E 阶段，首版在 S 结束时计算一次尺度并冻结，不能随每个 outer 更新重算却省略相关依赖；最终保存它与其 S 阶段来源。保留每 tensor 的 mean，避免大矩阵仅因参数个数更多而自动主导先验。

角色组 g 中再取 `R_g=mean_{l∈g} R_l`，总目标为

\[
J_C=L_C+\lambda\sum_g a_gR_g.
\]

首版固定正的角色比例 a_g=1，并分别记录各组损失与梯度；仅调一个有限 λ。`L_C` 的类权重及归约必须固定，不能因 CE 从总和改成均值而沿用原 λ。组的分割本身影响这个先验，故分组、tensor 名/shape、尺度和 a_g 都进入版本化配方。

角色至少区分 encoder ordinary weights、readout weights、普通 bias、BN γ、BN β、residual 内部 weights/bias、residual gate。BN γ/β 已作用于归一化特征，首版尺度固定为 1，且分成独立组；不能用接近零的 β 源 RMS 放大先验。普通 bias 使用源 RMS² 加其实际初始化方差；没有随机 bias 初始化的层用预声明的单位尺度。gate 的原参数 g 使用固定尺度 1，源中心为保存的 g₀；不除以初始为零的 gate RMS。

**空间 max-norm 组单独声明。** 普通 raw 标准化 L2 可作为最小实现基线；若首版要让空间主惩罚对应实际 forward 权重，采用有效权重偏移并同时锚定源半径：

\[
R_{sp}=\frac{\|P_\rho(W)-P_\rho(W_0)\|_F^2}{2n s_{eff}^2}
+\frac{\nu}{2m}\sum_{j=1}^m
\left(\frac{\|W_j\|_2-\|W_{0,j}\|_2}{r_j}\right)^2,
\]

其中 `s_eff²=mean(Pρ(W₀)²)+σinit²`、`r_j²=||W₀,j||²+k_j σinit²`；m 为滤波器数，k_j 为其参数数。首版 ν 固定为 1，作为与有效权重项同等显式的源半径约束；不另开未声明的 ν 调参。源点处两项均为零，第二项避免仅惩罚有效权重后 raw 半径任意漂移。它仍是一种声明的参数规范选择，不能再声称完全函数不变。raw-L2 与这项有效权重/半径先验应是预声明的替代比较，不能看目标测试选择。

两 logit head 也有共同偏移的不可辨识方向；当前全部推理依据 logit 差时，共同权重/偏置变化不改变概率。首版保留 source 参数规范并由源先验固定共同方向，不把 raw head 位移直接解释成 EEG 证据变化。若日后改成单 log-odds head，应单独核验 loss、概率及持久化等价。

### 9.3 residual 的门和末层具有乘积冗余

令残差分支为 `a[U vψ(h)+bU]`，其中 `a=ρ tanh(g)`。在 gate 范围允许时，`a→ca`、`U→U/c`、`bU→bU/c` 完全保留该分支输出。单独惩罚 a 或单独惩罚 U 都允许另一个量补偿；a 有界仍不限制分支输出幅度。

首版保持现有参数化，**同时**对 g 与 residual 内部权重/bias 施加上述有限、源中心、分组尺度固定的先验；记录 gate、内部权重范数、实际 residual/feature 比值。这里不宣称已得到唯一可辨识的神经生理分解。若以后要消除乘积规范自由度，可单独研究合并末层有效参数 `(aU,a bU)`，但其初始化与优化轨迹不同，不能当作无行为变化的整理。

g=0、内部随机输出非零时，普通源 CE 首步只给 gate 有效梯度；gate 更新后内部参数通常可获梯度。这是可唤醒的启动延迟。若源阶段永久冻结 g=0，则普通源 CE 根本没有训练 residual 内部表示，目标时打开 gate 会暴露随机分支；须据实报告，不能称已预训练个性化模块。若门与末层输出同时零初始化，梯度启动会完全失败。继续沿用当前“零 gate＋非零随机内部输出”，源预训练允许 gate 更新；个人阶段从完整保存的 `(g₀,ψ₀)` 开始。参数梯度为零不等于 AdamW 不会对该参数执行 weight decay，账本应分开。

### 9.4 full core 与 residual 部分重叠，并非普遍完全重复

若残差退化为固定线性 `Ah+c`，自由线性 head 可完全吸收它。局部一阶扰动中，core 的 `Jθ δθ` 与 residual 的变化也可能落在重叠子空间，二者能互相补偿，增加个体估计方差。

现有 residual 包含时间 depthwise 卷积、通道混合和额外 ELU，增加了非线性组合和局部时序作用；不能证明它总能由原固定深度 core 的参数更新精确吸收。也不能仅凭多一层非线性断定有益。必须比较同图下 full core-only、residual-only、两者同时更新，匹配支持行、source 身份、更新预算及合法 source 开发机会。两者共训只有在独立 query 上带来增量，才保留其额外自由度。

### 9.5 λ 的合法选择及一步 inner 的不可学习性

首版建议在源内预声明有限集合 `λ∈{0,1e-4,1e-3,1e-2,1e-1,1}`，按合法源内参与者/整库 episode 的 query 评价选择；λ=0 是显式无源中心惩罚控制。所有候选保留相同 inner 预算，source 选方后冻结到目标任务。新成人/外部测试不能参与这些选择；目标 C 的内部 T/V 仅能选择事先声明且预算匹配的有限配方。

不联合最小化单个 inner 的 `L_C+λR` 来学习 λ：R≥0 时这会鼓励 λ→0，不能解释为学到了可靠先验。若后续研究 meta-learned precision，应从源 query 的 outer 目标学习，并将 `λ=λ_min+(λ_max-λ_min)sigmoid(η)` 限于预声明有限正区间；无惩罚控制另列。精度参数、尺度与可训练性分别记账，不能让网络同时自由缩放参数和 prior scale 绕过正则。

尤其对标准梯度 inner，初始化 `θ^(0)=θ₀` 时源中心二次罚梯度为零，第一步

\[
\theta^{(1)}=\theta_0-\alpha\nabla L_{S}(\theta_0)
\]

与 λ 无关。因此**一步普通梯度 MAML 不能通过该路径学习源中心二次罚的 λ**；本文件的一步 exact-meta probe 只验证模型可微性，未验证可学习强度。至少两步后先验才通常影响轨迹，或者明确采用不同的 proximal inner 更新。例如固定对角精度 Λ 的 SGD metric 下

\[
\operatorname{prox}_{\alpha R}(\theta_0-\alpha g)
=\theta_0-\alpha(I+\alpha\Lambda)^{-1}g
\]

首步即依赖 Λ。proximal 更新不是把原 AdamW 换个名称，必须与实际校准算法一致并独立验证。若 λ 还控制初始化、学习率或其他项，则其可学习性是另一条显式路径，不能冒充源中心罚本身的一步作用。
