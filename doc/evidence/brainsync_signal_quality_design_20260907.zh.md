# BrainSync 连续质量与滤波状态研究方案

> 导航副本：历史失效链接现指向[持久来源索引](../assets/repository_audit_20261019/docs_historical_index.json)。原文以原SHA保存在本地归档；本次只改定位入口，原结果、执行producer与未找到凭证的边界保持。


日期：2026-09-07。交付范围为文献核查、数学与反例、实施规格和验证顺序。
本轮没有修改生产算法、替换预处理默认、生成新训练缓存或训练模型。
前置实测见[瞬态诊断](brainsync_transient_quality_20260907.zh.md)，时钟问题见
[时轴核查](brainsync_epoch_timing_audit_20260907.zh.md)。

## 1. 推荐裁决

首选路线是 **连续原始质量识别 -> 显式污染区间 -> 清洁段滤波状态管理 -> 完整 epoch
准入 -> 校准内 EOG/局部通道 QC -> 保留全部请求的决策评价**。
先补采集时间和原始污染语义，再检验清理是否保留真实 ERP。现阶段保持
`[-200,1200) ms / 128 Hz / 0.1-30 Hz / full_unfold+K35` 作为对照，参考按源 checkpoint
的显式合同处理；没有证据要求把模型改成 100-200 ms 或可学习 epoch。

第三、第五份保留为开发和恢复研究材料，不能未经处理重新纳入当前有效模型。
第一份维持已拒绝/隔离状态，不因本方案推演而重新准入。
第五份及其余 pilot 已被查看，后续新算法需要新的独立采集验证。

| 候选路线 | 裁决 | 原因及进入条件 |
|---|---|---|
| 大阶跃/缺口识别、污染区间隔离、清洁段重建状态 | 第一优先 | 防止局部异常通过滤波状态污染后续正常 EEG；保留恢复边界与拒判账本 |
| 校准内 EOG 回归和可学习局部 QC | 第二优先 | 复用现有模块，先补物理接法、有效拟合样本与独立分组合同 |
| 原始永久阶跃减法 | 条件候选 | 第三份反例支持其诊断价值，但须证明分段常数假设、估计误差及 ERP 保真；不能套到第五份多种瞬态 |
| 高通改 0.5/1 Hz、低通改 15 Hz、零相位替换因果滤波 | 对照项 | 会改变形态、时序或信息范围，不能以图更平滑为选择依据 |
| ICA、ASR、自动空间插值 | 后续候选 | 七导及重参考后的低秩、清洁基准、重建失真均未验证，不作为首轮默认堆叠 |
| 整份删除、固定丢前 N 秒/某个 block | 不作自动规则 | 当前证据定位的是局部污染；应以可用信息和完整决策覆盖判定，而非均值回零程度 |

## 2. 外部证据与适用范围

本轮是针对问题的机制研究，不宣称系统综述或穷尽检索。已通过公开索引、开放全文和官方
API 文档核验来源；有全文/只有摘要的边界逐项列在分支材料和本页末尾。

- Widmann 2015 [1] 支持用脉冲/频率响应、阶数与相位评估滤波失真。
- Tanner 2015 [2] 在24人 N400/P600 中显示不当高通可制造相反极性成分。其双向
  Butterworth 和 -6 dB 截止定义不同于本项目单向四阶 -3 dB，不能移植数值阈值。
- de Cheveigne 2018 [3] 的全文直接展示阶跃的高通长尾和零相位的向前传播。
  其去阶跃实证主要是高密度 MEG/SQUID，不是对 BrainSync EEG 电极瞬态修复的验证。
- Jas 2017 [4] 支持交叉验证的局部阈值；其空间修复不等于七导缺失信息可恢复。
- Gratton 1983、Croft 2000 [5,6] 指出回归传播估计、刺激相关活动与眼动类别的重要性。
  当前整段中心化 OLS 不能称作完整 Gratton EMCP。
- PREP 2015 [7] 支持参考与坏通道检测的相互依赖。共同成分被参考投影删除，不保证被删的
  只有噪声。ASR 经验参数也不能从驾驶 EEG 直接移植为本项目默认 [8]。
- MNE/SciPy 的分段滤波和稳态初始化接口可复用 [9]；LSL、PsychoPy、BIDS 明确不同时间
  证据的语义 [10-12]，不会自动消除 BLE 传输偏移或证明像素亮屏时刻。

## 3. 采集与排程先行

### 3.1 样本时轴的最小账本

前端保留原始电压，并以可对账的 sidecar/journal 保存：

| 对象 | 必需证据 |
|---|---|
| sample | 原始 seq、delta_time、转换状态、接收行号、存储 sample 或拒绝原因、segment_id |
| batch | 回调入口 monotonic_ns、首末 seq、接收/写入数、写入前后 sample、队列/落盘时刻 |
| clock segment | SDK/固件与实际源码 hash、counter 语义、映射拟合范围、漂移/残差、未知固定延迟 |
| stimulus | 原始 scheduled/request/submit 时间、event/trial/attempt、映射源 sample、光电时点若有、对齐等级 |

回调时刻只能叫 delivery time，追加完成只能叫 liveness time，不能命名为 ADC acquisition。
本地 SDK 实查有 `seq_num` 和单位10 us的 `delta_time`，没有绝对采样 timestamp；
delta 在丢包/重连时的语义待确认。其 batch 必须250的整数倍，不能直接改成10/25点回调。
安装 metadata=0.3.0、运行 `__version__`=0.2.0，需同时记录并绑定原生文件 hash。

段内模型为 `t_acq=a+b*t_device`，但观测的是
`t_receive=t_acq(last)+d_transport`。只有 callback 时不能辨识固定 `d_transport`；
回归残差小也不能证明绝对偏移小。应分别标为 `delivery_anchored`、经验证的
`device_mapped`、同一采样时钟实测的 `optically_measured`。
软件账本和相对一致性可以先完成；真实亮屏潜伏期主张需光电或等效实测，不阻塞基础软件开发。

序号回绕、重复、乱序、已知缺样、未知重置必须区分。失败时保留原始行索引，不将后续数据
压缩成假连续信号；EDF表达不了缺失时用显式分段/断点与原始counter映射。暂停跨批次的
边界也必须以实际接受的样本定位，不能只用按下恢复按钮的时刻。

### 3.2 更正旧 pilot 的覆盖原因

本轮四份保留原始记录逐一重放，加上已隔离五会话缓存复核，一致得到：
block 1 有45个 epoch，block 2-10 各44个。丢失的是 `B002-T001..B010-T001`，
**首刺激的 -200 ms 基线跨入前一休息**，不是旧文档描述的前九 block 最后刺激缺尾部。
例如第三份 B002 onset=91.031 s，前休息结束91.015 s，基线从90.831 s开始。
10%的严格R5覆盖数不变，但完整的是第一个 block，不是最后一个。

采集必须同时保证：

```text
first_onset + t_min >= max(rest_end, usable_segment_start, recovery_end)
next_rest_start >= last_onset + t_max
recording_stop >= last_onset + t_max
```

当前至少需要休息后200 ms有效基线、末次刺激后1200 ms尾部，另加时钟不确定性边界。
完成尾段要核验实际样本收到并落盘，不只等待主机时钟。大批回调的交付等待计入决策时间。
正常休息不删除样本、不随意重置滤波；用户手动暂停仍保持既有分段合同，恢复后重建状态。

## 4. 连续质量处理的数学合同

### 4.1 先区分原始异常与滤波影响

设 `x=s+a`，滤波器为H，基线算子为P，epoch切片为E，重采样为R，则污染为

```text
Delta X_i = P R E_i H a
```

仅检查 `E_i a` 无法知道 epoch 之前污染了多少滤波状态。
对永久阶跃 `a[n]=Delta*1[n>=j]`，若 `g=H{unit_step}`，源采样轴上基线后的污染为

```text
e_i[m] = Delta * (g[e_i_sample+m-j] - mean_(b in baseline) g[e_i_sample+b-j])
```

生产验证应将同一 R、P 纳入；多个污染的上界应累加或使用已证明的联合状态界，不能只取
最大单次事件，不能依靠未知符号的偶然抵消。阶跃响应只适用于阶跃假设，不能代替脉冲、
持续饱和、缺口或指数电极恢复的模型。

### 4.2 检测与隔离的候选接口

拟议 `ContinuousQCSpec/FittedContinuousQC` 先在连续原始域计算标签无关特征：逐样本差分、
两侧稳健水平变化、持续平台/尖峰、可证明的硬件饱和、设备缺口和参考异常。
例如 `z_c[n]=|diff(x_c)[n]-median_fit(diff)| / scale_fit_c`；scale来自合法拟合范围的
MAD及已声明量化/噪声下界，不用全测试会话重新拟合。单通道异常和多通道同步异常均保留。
EDF动态导出量程的极值不等于ADC饱和；历史没有ADC状态时只能记录波形疑似平台。

水平变化若用未来w个点确认，检测延迟为至少w/fs。输出必须保存检测/确认时间、
原始污染区间 `[start_sample,end_sample)`、通道范围、reason、uncertainty、fit范围和版本。
判据不得使用 thought_digit、target标签或平均ERP。标签无关仍可能优先剔除真实大P300，
必须另验证干净ERP注入误报与冻结规则后的类保留率。

三类处置分开：硬缺样/饱和/无法追踪参考的片段不可修复为真实观测；可确认的瞬态片段
进入隔离与恢复判断；满足分段常数模型的阶跃才进入独立“修复候选”。
无依据时不把全部高幅值视为硬故障，也不设固定整会话删除百分比。

### 4.3 状态管理不是只加 BAD 注释

当前 `preprocess.py` 的因果分支只取 `edge*` onset，不消费 duration；
新增 `BAD_transient` 不会阻止坏样本更新 IIR，仅在异常起点重置也不能隔离持续污染。
需要让因果与零相位路径消费同一组完整原始区间，在清洁连续段分别运行，禁止跨异常卷积。
epoch涉及原始异常、状态恢复不确定区间或基线不完整时，保留事件账本并标为不可用。

重置后也不能立即宣称干净。状态误差满足

```text
z[n+1] = A z[n] + B x[n]
Delta y[n+k] = C A^k Delta z[n]
```

若A可对角化，`A=V Lambda V^-1`，则可按模态绝对值和构造整个后续尾部的上界；
不可对角化/数值病态时用可靠矩阵幂界或保守数值与剩余尾部界，不直接套特征值半径。
单纯知道 `rho(A)<1` 只证明渐近稳定，不给定有限时间误差保证。

对孤立阶跃幅度上界Amax，可定义

```text
L(epsilon,Amax) = min L: Amax * sup_(k>=L) |g[k]| <= epsilon
```

本轮当前SOS的数值模态分解与直接滤波吻合；以已见14.2 mV阶跃作例，epsilon=1 uV时，
9.288 s首次低于门限之后仍可反弹875.35 uV，保守模态边界39.928 s。
epsilon=.5/2 uV对应42.824/37.032 s。这些是机制反例，不是生产默认阈值或固定预热秒数。
基线差分可能将误差界放大；必须对实际 `P R E` 输出验证容差。

首选在异常后可信清洁段重建状态，阻断历史异常；再以初态不确定性及稳定窗口验证恢复。
旧异常已经进入状态时，才使用其剩余响应界评估必要隔离范围。两者的误差来源不同，
不能把去掉历史状态后的段也机械隔离40秒。
首次过零、固定等3秒、只看一帧平稳均不是充分恢复证据。无法界定误差时保持拒绝/重采。

在线决策使用截至已收到的样本；前缀处理结果不得受后来测试后缀修改影响。
需未来确认的输入不能提前发布；`evidence_available_times_s` 必须计入检测、确认和交付
依赖，不再无条件等于onset+tmax。完整recording批处理仍不称逐chunk在线推理。

## 5. 参考、EOG 与个人 QC 的次序

连续硬污染先处理，再选择与checkpoint一致的最终参考，再在清洁合法校准片段拟合EOG。
现有 `FittedEOGRegression.apply` 冻结系数/中心逐点相减可复用，但拟合需新增sample mask。
先核实EOG是共享参考的单端输入还是双极差分；当前同时对EEG/EOG减Cz隐含共享参考，
不能在接法未证实时自动启用。

`X=S+ZB^T` 时普通OLS包含偏差
`Bhat^T=B^T+(Zc^T Zc)^-1 Zc^T Sc`。若任务脑信号与EOG相关，就可能扣掉真实ERP。
优先比较无EOG与冻结EOG；专门的眨眼校准可作为候选，其额外时间必须计入预算。
不采用“相关下降最多”或“波形最平滑”作为选方标准。

现有 `FoldLocalArtifactPolicy` 的阈值CV按subject分组，单P001时退回quantile .99及
坏通道比例 .5；应用入口还要求至少两个训练subject。应增加独立的真实session/block
`fit_groups` 合同，participant身份不变，不能把block伪装成不同人。
个人阈值在inner-training拟合，再应用inner-validation；选定后全calibration refit并冻结。
分组不足要报告无内部选择证据，不把默认值包装为已优化结果。

局部坏EEG用已训练支持的mask，不以插值伪装观测；Cz参考坏则其所有差分通道受影响，
不能只mask一个Cz解决。CAR或Cz都能消掉真实共同ERP；原始七路、最终模型参考和EOG
应在诊断中并列显示，但只使用明确定义的模型输入作推理。

## 6. 从科研验证到产品的最小实验

### 阶段A：软件与已知真值

先修复账本及双侧guard，并在模拟完整排程验证10/10 block各45个可用epoch。
合成/半合成覆盖干净宽P300、慢正波、脉冲、平台、永久阶跃、指数恢复、多个近邻异常、
共同参考故障、丢样、跨批暂停、初始化及持续正弦。先用已知异常边界验证算子，再单独
测自动检测器；已知边界成功不等于检测器成功。

本轮10组数学/软件反例已通过：模态上界、过零反弹、基线污染传播、只排BAD不足、
重置仍有初态误差、chunk不变、参考丢真ERP、相关EOG误删、标签无关门限选择偏倚、
R5缺一不可与休息基线边界。未实现自动检测器，因此不报告其敏感度、误报率或恢复成功率。

### 阶段B：现有记录只作开发

第三份先比较已知启动阶跃的隔离/重置与条件去阶跃；也审查第八block局部污染。
第五份比较完整局部污染账本及原始/Cz差分，不以“删最后block”当固定清理规则。
可用量按有效原始支持、每通道mask、每候选/重复覆盖计算，不由ERP外观确定。

### 阶段C：预先冻结的配对矩阵

| 臂 | 相对同一固定参考/模型/采集预算的增量 |
|---|---|
| Q0 | 当前处理作为历史开发基线；已知无效段不进入新产品推理 |
| Q1 | 连续质量区间与滤波状态/恢复期合同 |
| Q2 | Q1 + 合法校准拟合并冻结的EOG |
| Q3 | Q1 + 合法校准拟合并冻结的epoch局部QC |
| Q4 | Q1 + EOG + 局部QC，前两项分别通过后再验证组合 |

固定公共源checkpoint、网络、参考、seed、学习率与预算规则，不同时改窗口、滤波频带、
源域组成和解码器。包含目标自身baseline/匹配scratch，以排除仅是新处理更适合某个源。
采集请求预算相同不等于有效K相同：记录清理后actualK、targets、updates、draws。
主对照评价完整处理链的固定采集预算效益；机制对照另用共同有效行及匹配优化剂量，
不能用额外训练遍历解释成单纯去伪迹收益。具体新拟合预算在实施前冻结，本轮未排程。

### 阶段D：新独立采集

所有新配方选定后再采新session。当前看过的第五份不再作新配方的独立确认。
主要结果为全部请求block的operational hit@R5，同时报告coverage、拒判率、条件有效准确率、
实际flash/time及波形保真。清理后只在幸存trial上AUC更高不是产品成功。
严格45闪烁R5要求每候选五次有效证据；发生污染时可拒判。若另做补采协议，保存原失败
attempt及额外flash/time，不能只保留补采成功结果后仍标45闪烁成本。

验收门限分三类：合同不变量须严格通过；注入ERP的幅值/面积/潜伏期失真与误报上限需
在开发范围预声明；真实决策改善要给配对不确定性与代价。当前四会话不足以给成人总体
结论或稳健样本量假设，先用新先导方差制定确认规模。

## 7. 实施清单与兼容性

| 顺序 | 现有所有者/拟议接口 | 必须联动的验收 |
|---|---|---|
| P0 | 前端runtime/recording_clock/storage/bids/ui | 逐包/批次账本、异常attempt、实际写入计数、双侧guard；模拟丢50点不能造成后续假提前200 ms |
| P1 | data.contract/epochs + 拟议data.continuous_qc | 检测拟合范围、连续异常区间、状态恢复误差、通道角色与source hash；无污染路径保持原结果 |
| P1 | data.preprocess/brainsync/bids_eeg/events | 因果/离线共同消费完整区间；有效epoch与时间账本一一对应，未来确认依赖进入available time |
| P2 | data.ocular | 物理参考前置、fit_sample_mask、校准中心/系数/digest、清理与非清理对照 |
| P2 | data.artifact/qc_features + transfer.calibrated/calibration_candidates | 真实fit_groups、inner范围、final参考下重算QC、个人队列有效性与保留量 |
| P3 | transfer.checkpoint/subject_adapter、cache保存/runner | 新QC/时钟/拟合语义升版并绑定digest；旧个体包不重标，公共源是否可复用经显式门禁确认 |
| P3 | candidate_decision/candidate_evidence/research.evaluation/outcomes | 冻结词表、scheduled/valid/positive-weight counts、全部block分母、失败与补采成本 |
| 收尾 | 三个前后端入口、相关tests、六份活动指导 | 构建/校准/保存/重载/推理一致；完整相关反例与端到端模拟后再实机 |

仅新增epoch删除mask未必改变有效段算子；分段重置、去阶跃或参考变更则改变输入生成，
两者必须在兼容性门禁中分别声明。不得改旧cache元数据冒充新策略，也不应未经检查就
要求重训所有公共源。新个体拟合及其QC状态必须重新生成并绑定所执行的策略。

## 8. 来源与本地证据

[1] Widmann et al. 2015. [Digital filter design for electrophysiological data](https://doi.org/10.1016/j.jneumeth.2014.08.002). 官方元数据/摘要，未核验全文。

[2] Tanner et al. 2015. [How inappropriate high-pass filters can produce artifactual effects](https://doi.org/10.1111/psyp.12437). [PMC4506207全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4506207/)，已核验方法/结果/讨论。

[3] de Cheveigne & Arzounian. 2018. [Robust detrending, rereferencing, outlier detection, and inpainting](https://doi.org/10.1016/j.neuroimage.2018.01.035). [PMC5915520全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5915520/)，已核验。

[4] Jas et al. 2017. [Autoreject](https://doi.org/10.1016/j.neuroimage.2017.06.030). 官方元数据/摘要及[官方实现示例](https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)。

[5] Gratton et al. 1983. [A new method for off-line removal of ocular artifact](https://doi.org/10.1016/0013-4694(83)90135-9). 官方元数据/摘要。

[6] Croft & Barry. 2000. [Removal of ocular artifact from the EEG: a review](https://doi.org/10.1016/S0987-7053(00)00055-1). 官方元数据/摘要。

[7] Bigdely-Shamlo et al. 2015. [The PREP pipeline](https://doi.org/10.3389/fninf.2015.00016). [PMC4471356全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4471356/)，已核验相关段落。

[8] Chang et al. 2020. [Evaluation of Artifact Subspace Reconstruction](https://doi.org/10.1109/TBME.2019.2930186). 官方元数据/摘要；驾驶EEG域，非本任务验证。

[9] [MNE Raw.filter](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter)、[MNE EOGRegression](https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html)、[SciPy sosfilt_zi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html)，官方API已核验。

[10] LSL [Stream Outlets](https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/outlet.html)和[Time Synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)，官方正文已核验。

[11] PsychoPy [Window API](https://www.psychopy.org/api/visual/window.html)和[Timing precision](https://www.psychopy.org/general/timing/millisecondPrecision.html)，官方正文已核验。

[12] BIDS [Events](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html)和[EEG](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html)，官方正文已核验。

分支研究材料与核验边界在 `experiments/runs/brainsync_quality_research_20260907/`：
[滤波](../assets/repository_audit_20261019/docs_historical_index.json)、
[时钟](../assets/repository_audit_20261019/docs_historical_index.json)、
[EOG/QC/参考](../assets/repository_audit_20261019/docs_historical_index.json)。
[10组反例与五会话覆盖复核](../assets/repository_audit_20261019/docs_historical_index.json)，
[反例图](../assets/repository_audit_20261019/docs_historical_index.json)。
脚本为`tmp/verify_continuous_qc_design_20260907.py`，使用项目venv；这是机制验证，不是新自动
检测器验证。更正前指导已[物理归档并校验](../../frozen/brainsync_quality_design_20260907/manifest.json)。
