# P300 模型输入合同：离散时间、单位与分组

## 0. BrainSync 采集角色（2026-09-06）

唯一产品布局由 `data.contract` 的 `BRAIN_SYNC_ACQUISITION_CHANNELS` 与
`BRAIN_SYNC_ACQUISITION_TYPES` 定义：Fz/Cz/P3/Pz/P4/PO8/EOG/Oz，共8采集槽。
原PO7槽改采PO8，原PO8槽用于独立EOG；模型只用七路Fz/Cz/P3/Pz/P4/PO8/Oz。
EOG保持原raw/BIDS通道类型，不进入EEG卷积、head输入或头皮CAR，不伪造头皮坐标。
用户已确认当前单路EOG为左眼眼角单HEOG；2026-09-08明确REF是**左耳耳垂耳夹**，
规范值为`left earlobe`，不是T7/T3、乳突、右耳垂或双耳平均。
`data.contract.brainsync_reference_record()`及其独立前端分发JSON绑定位置、左右侧和`ear_clip`。
`EOGDerivation`保存位置、单端/双极及参考；历史未声明记录不自动补写。单HEOG不等于完整眼动观测。
旧活动session/BIDS参考声明已按用户证言显式更正，原EDF字节和历史执行结果保留。
原始采集参考与软件重参考分别记账；旧`T7(T3)`不作为新参考的兼容别名。
未启用经校准拟合的清理器时，
provenance必须写 `ocular_artifact_processing.status=not_applied`，不能称已去伪迹。
公共数据保留原始电极名/类型；该槽位调整不是全局PO7/PO8别名或历史数据重标规则。

## 1. 证据边界

2026-09-16分段记录新增左右原因集合：record、acquisition、artifact及未分类实际reset。
同坐标和合并gap的原因取并集；包含artifact或未知reset时不享受自然边界豁免，任一端需要认证均保留
完整双端状态项。混合gap的保守处置可新增拒绝，原因和数量保留；持续输入污染仍不在状态界保证内。
历史MNE全局skip与逐导后端各按实际支持执行，不因共享解释器改变旧滤波坐标/顺序。
永久坏非参考导先从数值输入剔除，坏/缺/错误角色参考导在投影前拒绝；只在最终固定槽表示缺失为0。
空间输出可用性为其全部非零系数依赖导可用性的逻辑与，Cz失效传播到所有Cz差分输出；不以`NaN*0`代替掩码。
相应来源、mask、QC特征、lineage及凭证同步，见[修复与反例](evidence/algorithm_chain_repairs_20260916.zh.md)。

2026-09-16的P1空间/时间参数组和P2源特征辅助损失均不改变物理输入合同。
P2中的L2归一化只作用于source训练的post-MST特征；它不是raw重参考、EA白化或通道补全。
现有参考、实测mask、epoch时间支持和校准/test隔离仍独立验证；后续EA接入必须另存合法fit范围，
不能把全目标cache预处理当成校准范围内拟合。

设源记录的采样率为 `f_src`，第 `c` 个物理 EEG 通道为
`x_c[n] in V`，第 `i` 个刺激在源记录中的 sample index 为 `e_i`。
`x_c[n]`、`f_src` 和 `e_i` 属于采集证据，不因模型输入采样率改变。

当前统一软件默认张量为：

```text
f_model = 128 Hz
passband = 0.1..30 Hz
epoch = [-200, 1200) ms
T = floor((1200 - (-200)) * 128 / 1000) = 179
unit = V
baseline = per-trial, per-channel mean over [-200, 0) ms
```

该默认来自已完成 GTN steady-state 2x2 的 achieved-best recipe；成人 BrainSync
仍需 prospective confirmation，但在出现更强证据前不保留较差降级接口。
online causal profile 另声明 `filter_phase=forward` 和
`causal_iir_initial_state=steady_state_first_sample`。

## 2. 可执行算子

### 2.0 BIDS raw 时间轴与休息段

BrainSync session v4 输入只接受 `completed` 的连续或显式分段 BIDS raw，并要求每个 block
由完整 9 候选随机排列循环组成。设刺激时刻为 `t_i`，模型 epoch 为
半开区间 `[t_i+t_min, t_i+t_max)`，第 `k` 个休息段为 `[r_k0,r_k1)`。
缓存保留该刺激当且仅当对所有休息段都有：

```text
t_i + t_max <= r_k0  or  t_i + t_min >= r_k1
```

等价地，排除条件是：

```text
(t_i + t_min < r_k1) and (t_i + t_max > r_k0)
```

连续滤波在完整 raw 上先执行，休息段通过 `BAD_brainsync_rest` 只参与 epoch
选择。不能先把休息段拼接删除再滤波，因为拼接点一般满足
`x(r_k0-) != x(r_k1+)`，会制造非生理阶跃及 IIR 瞬态。反例测试同时验证：
跨休息事件仍保持原始秒数、相交 epoch 的 `evidence_index=-1`、未相交 epoch
继续生成缓存。

2026-09-06新增手动暂停的显式变体：`time_base=segmented_recording`。
暂停时停止追加样本，恢复时从`boundary_sample`继续；每次暂停同时声明
`start_wall_seconds/end_wall_seconds`，因此文件内时间不能直接当成连续墙钟时间。
导出按各段起始sample与恢复时刻映射软件marker，段末以已写样本边界截断。
这是软件/回调时间映射，仍不是硬件触发或光学校准证据。

BIDS `acquisition_boundary`与session断点sample必须一致。读取后生成`BAD boundary`
和`EDGE boundary`：跨边界epoch不可用，因果IIR在每段首样本重新初始化，
零相位滤波也逐段执行。`interrupted_stimulus`保留被打断attempt的记录但不作为有效刺激。
正常block休息仍遵循上面的连续raw规则。无断点声明的拼接仍被禁止。

P001真实先导的旧五会话缓存复核为block 1有45个epoch，block 2-10各44个；四份保留
记录逐事件确认缺的是休息后首trial，其刺激前基线跨入前一休息，450个scheduled仅保留441个。
严格R5只有第一个block完整；此前“末次flash缺尾部/最后block完整”的文字解释错误，已更正，
见[覆盖复核与方案](evidence/brainsync_signal_quality_design_20260907.zh.md)。缺口保留在分母。
排程同时要求首onset加tmin不早于休息结束/恢复可用边界，下一休息或录制结束不早于
末次onset加tmax；不能只等待主机时间，还须确认所需样本收到并落盘。
首刺激、休息后及暂停恢复后均先在当前segment累积真实baseline样本，再checkpoint确认后发刺激。
每block末刺激的响应尾窗也先确认，再进入休息；等待可暂停/取消，超时终止而不静默继续。
有finalized sample clock时，`clock.segments`的内部边界必须逐项等于timeline pause及BIDS边界，
防止样本账本有断点而epoch loader遗漏EDGE。无clock历史记录保持原合同，不补造历史样本。

### 2.1 连续域滤波

2026-09-09用户要求加入工频陷波。实施合同为：BrainSync新输入默认50Hz、Q=30的二阶IIR陷波，
在原始采样率上先陷波再带通，二者合为同一SOS级联，沿用因果稳态初值和逐采集段/异常重置。
`notch_freqs_hz`与`notch_quality_factor`进入预处理签名；频率及Q可显式配置，空频率序列表示关闭。
公共数据合同保持各自已声明处理，通用raw入口支持显式陷波，不从参与者或库名猜测50/60Hz。
Nyquist按原始采样率核验，不静默跳过不可实现频率；不自动添加谐波。
恢复证书必须由包含陷波的同一SOS计算，旧QC和模型不重标为已含陷波；旧模型推理读取原合同。
依据为[SciPy iirnotch定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html)：
`Q=f0/bw`，本次50Hz的−3dB宽度约1.67Hz。该选择须经通带保真、因果性、断点重置、
保存重载和真实数据反例核验，不能将数学零点当作任意漂移工频的零残留保证。

令 `B4` 为所选通带的四阶 Butterworth IIR。offline `phase=zero`
执行前向/反向滤波，因此

```text
x_bar_c = filtfilt(B4, x_c)
|H_zero(f)| = |H_B4(f)|^2
```

它是离线、非因果算子。不能把 `e_i + 800 ms` 解释为在线系统真正可用
证据的时刻。

2026-09-09新增显式`zero_phase_edge_policy=mne_reflect_limited`：完整SOS的脉冲响应决定
MNE延拓长度，逐采集段限制为段长减一；缺字段为`legacy`，保留历史有/无陷波各自的处理。
新BrainSync默认个人合同为`brainsync_bidirectional_v1`，使用完整记录双向处理。
双向有效时通用账本按完整raw结束样本计可用时间，不能写成`onset+tmax`；
整记录双向后取前R轮只代表整记录结束后的回顾性分数。严格在线R需按实际可用前缀重滤。
校准/测试先划分独立raw再滤波，不能拼接后双向；短epoch不能单独代替连续输入。
N=4带通实际单遍8阶，含二阶陷波后10阶，双遍等效20阶；幅频响应平方是本次比较的一部分。
padding不提供恢复或真实样本认证，新旧边界策略都存在位置相关误差，见
[首轮数学与边界证据](evidence/filter_phase_20260909.zh.md)。
双向状态误差为`J F_N J O_N delta_zL + J O_N delta_zR`，后向状态包含第一遍误差耦合。
`data.bidirectional`在真实段长和epoch位置计算两端模态包络的条件界，不假设单调恢复秒数。
自然记录/采集段端点属于声明的有限算子，敏感度保留为诊断；只要片段任一端由伪迹切段引入，
就使用完整双端界作门禁，不能删掉另一端的耦合项。默认QC新拟合使用`range_action=diagnostic`、
`boundary_action=artifact_edges`，旧缺字段保留原含义，详见[推导与验证](evidence/bidirectional_qc_20260909.zh.md)。

online `phase=forward` 使用单向 SOS：

```text
z0 = sosfilt_zi(B4) * x_c[0]
x_bar_c, z_end = sosfilt(B4, x_c, zi=z0)
```

不能使用全零初态。真实 GTN 通道含 mV 级 offset，零初态会把前期 trial 制造成
mV 级 startup transient，而 later suffix 已稳定，形成假的时序域移。若处理真实
stream chunk，后续 chunk 必须携带上一 chunk 的 `z_end`；首文件才使用首样本稳态。

首样本稳态不消除后续真实电位阶跃。2026-09-07在BrainSync `0903_173947`发现0.328秒的
14.2 mV Pz阶跃，其因果IIR恢复响应污染了3.064秒开始的刺激epoch；未识别的原始阶跃及
滤波恢复期仍是当前质量处理缺口。基线只令刺激前均值为零，不保证epoch末端回零，
也不能以提高高通或加强平滑后图形变好作为选方依据，见[瞬态诊断](evidence/brainsync_transient_quality_20260907.zh.md)。

`data.continuous_qc`已实现冻结差分/滚动范围检测、完整异常区间及状态恢复合同。全局区间由因果和
零相位路径共同消费；forward使用单向模态恢复，zero使用显式双端有效状态模型。局部异常注释仅重置受影响导，
恢复注释只屏蔽相交通道，不反复重置；普通休息仍连续滤波。最终空间参考的依赖决定传播范围。
重置不保证立即可用，未知恢复上界仍拒绝该段后续证据；不能用首次过零或固定预热秒数放行。

经验恢复令`A=V diag(lambda) V^-1`，`L`为实际epoch重采样、截取及均值基线算子，校准模态包络为
`gamma[c,k]`，使用`E_c(d)=max_j sum_k abs((L O V)[j,k])*gamma[c,k]*abs(lambda[k])**d`。
非负项保证后续起点的界单调不增；参考/EOG混合W使包络变为`abs(W) @ gamma`。
必须对最终Cz空间满足容差，不能把两导各自1uV的差仍称1uV。模式身份固定排序并随记录校验。
序列化后保留原模态身份及摘要；跨平台重算仅容许特征值1e-12绝对舍入差，不容许模态错序。
该容差作用于身份数值核验，不放宽1uV物理恢复界、SOS、经验包络或质量门限。

既有开发profile为`continuous_qc/3`、`epoch_state_recovery/2`，额外异常前沿guard为350源样本；
guard从已检测区间向前扩展并截在当前采集段，物理检测/确认时间不后移。该预防范围不认证
异常真正起点；小幅持续斜坡仍可漏检。最低有效3/6导同时约束loader、epoch QC和推理。
被屏蔽数值为零占位并携带mask，LDA仅使用完整可观测特征；不能学习缺导占位数值。
完整recording的确认和交付依赖进入availability，EDF填充值不计入真实样本支持。
推导、实际恢复26.672-30.216秒及限制见[局部恢复证据](evidence/brainsync_recovery_20260908.zh.md)。

2026-09-08新个人QC要求`n2p3_continuous_qc/4`已实际应用，且为逐导模式、显式启用
`flatline_min_duration_s`。本次取5秒：连续相等的原始数字样本须满足`(n-1)/fs >= 5`，
250Hz时为1251点。判定在减Cz/其他参考之前，不跨缺样或采集断点拼接；恒值参考的影响仍按
依赖矩阵传播。未启用此字段时保留旧 `/3` 的序列化和摘要，不能给旧JSON直接重标版本。
精确恒值只覆盖平直线的保守子集，不检测所有带噪断线、饱和或慢漂移。

个人epoch方法为`personal_epoch_signal_evidence`，记录schema为`n2p3_personal_epoch_quality/1`。
每导独立计算`a=PTP(x)`、`s=STD(x)`和`q=RMS(diff(x))/STD(x)`，正值取log。
每个合法校准session/block组冻结`median +/- 6*1.4826*MAD`，跨组取包络；组内至少4个可用值。
无足够校准支持标记为未覆盖，不制造阈值；6倍MAD是开发诊断参数，没有宣称固定误拒率。
这些分布偏移只记诊断，幅度高、STD低或粗糙度高均不单独硬mask。
epoch `PTP==0`的实测导标记`no_dynamic_signal`；恒值导超过原观测导一半时拒该epoch，
并继续应用既有最低有效导数和全导不可用门禁。屏蔽值只作mask占位，不修饰幸存波形。
`PTP_c/median_observed(PTP)`不再进入新个人硬门：本导不变而他导/mask变化也能改变其判定。
旧方法及公共FoldLocalArtifactPolicy保持原义，数学反例见[QC裁决](evidence/brainsync_qc_redesign_20260908.zh.md)。

`data.slow_trend`在原始参考/滤波前按1秒完整bin取中位数，5/15/30秒窗取所有bin对斜率的中位数
`median_{i<j}((b_j-b_i)/(t_j-t_i))`，单位V/s；另记去趋势残差MAD、bin范围以及前一不重叠同尺度
窗口的斜率差。完整窗口和比较窗口均不得跨采集断点、缺样、局部不支持段或非有限值。
这些量不拟合阈值、不改X/mask；`continuing_input_quality_certified=false`明确没有认证持续输入清洁。
现有6个斜坡漏检中存在趋势抵消，不能靠绝对斜率阈值解决，详见[科研反例](evidence/brainsync_qc_remaining_20260908.zh.md)。

`EpochStateRecovery.recovery_certificate()`给每导最小放行样本、前一采样误差界、刺激onset及完整
epoch支持的最早位置。age从reset后首源采样到epoch起点，不是到onset。当前Cz六导26.672-30.216秒
在现模态不确定集内不能缩短；换频带须重新计算SOS/经验包络及最终投影，不复用旧等待数字。

### 2.2 源事件切片与重采样

2026-09-15执行器按MNE实际闭端源网格记账。绝对源索引
`p_i=e_i+round(tmin*f_src)`、`q_i=e_i+round(tmax*f_src)`；数组切片须减`raw.first_samp`。
通用可用时间为`max(onset+tmax, q_i/f_src)`，实际双向滤波再取完整recording结束；
adapter继续合并更晚的QC、clock和交付时刻。每个排程事件都有源起止sample，未产出证据的事件
保留原拒绝原因和分母。250Hz、1199ms窗口的1ms早报已修复；FFT会把闭端样本传到最终保留点。
数值顺序保持源epoch→FFT→前T点→半开基线；接近目标频率时沿用生产np.isclose的跳过规则，
恢复证书共用该实际数值映射。median仅可作数值基线，不能获得线性矩阵/恢复界能力。
这些是时间账本和证书语义修正；新派生cache使用新执行凭证，既有cache/成绩保持原义。
见[数学裁决](evidence/preprocessing_contract_research_20260915.zh.md)与
[实施反例](evidence/preprocessing_execution_20260915.zh.md)。

先在源 sample 轴上按 `e_i` 切片，再对每个 epoch 重采样：

```text
E_i[c,m] = x_bar_c[e_i + round(-0.2 * f_src) + m]
E_hat_i = R_fft(E_i; f_src -> f_model,
                npad=auto, window=auto, pad=edge)
```

MNE epoch 的右端点是闭合的；缓存随后只保留合同推导的前 `T` 点，从而得到
统一的半开区间。滤波必须在连续信号上执行，因为一般有

```text
Crop(Filter(x)) != Filter(Crop(x)).
```

反例：在 `-200 ms` 左侧放置一个脉冲。先裁剪会把它完全删除；先连续滤波
会使其滤波响应进入 epoch。测试覆盖这一不交换性。

### 2.3 基线均值扣除

模型时间点为 `t_j = tmin_ms + 1000*j/f_model` ms，基线索引集为
`B={j | -200 <= t_j < 0}`。最终输入为

```text
mu_i,c = (1 / |B|) * sum_(j in B) E_hat_i[c,j]
X_i[c,j] = E_hat_i[c,j] - mu_i,c
```

因此对任意逐试次、逐通道常数 `b_i,c`，有

```text
Baseline(E_hat_i + b_i,c) = Baseline(E_hat_i).
```

只允许减均值，不允许除以逐试次基线标准差。若 `E2=2*E1`，逐试次
z-score 会令 `z(E2)=z(E1)`，从而抹去可能有判别意义的 ERP 幅值，同时把
单位从 V 改为无量纲，破坏 `channel_std_v` 和 `epoch_scale_v` 的物理语义。

## 3. 卷积核的物理尺度

奇数长度 `K` 的中心卷积核在采样率 `f` 上，首末采样点的物理跨度为

```text
D_ms(K,f) = 1000 * (K - 1) / f.
```

当前尺度为：

```text
ST default K35: D_ms(35, 128) = 265.625 ms
ST broad K65:   D_ms(65, 128) = 500 ms
MST-short after /4 pooling: D_ms(5, 32) = 125 ms
MST-long  after /4 pooling: D_ms(17,32) = 500 ms
```

这些是单层局部跨度。若共享 ST 核为 `K0`、平均池化宽度/stride 为 `P`、
MST 分支核为 `Ks`，输入域端到端感受野为

```text
R_s = K0 + (P - 1) + (Ks - 1)P
D_total_ms = 1000 * (R_s - 1) / f.
```

临时默认 `K0=35,P=4,Ks={5,17}` 得 `R={54,102}`，即约
`414.06/789.06 ms`；K65 broad reference 得 `648.44/1023.44 ms`。因此“短/长分支”必须按
总感受野解释，不能把 125/500 ms 的局部 MST span 当成完整模型支持域。

输入 epoch 边界与卷积核长度是固定配置，学习的是卷积和分类权重。
当前默认 `full_unfold` 使用整段卷积特征；`250-600 ms` 只属于可选 LMBC 读出的
基准证据窗，其固定偏移集合上的注意力可学习，不是任意可移动的 epoch 或卷积扫描范围。
五份 BrainSync 会话的 2205 epoch 已独立复算，未发现分段整体提前；采集端录制零点
尚未绑定首个样本，12 秒断流告警也不能证明毫秒级对齐，详见
[窗口与时轴核查](evidence/brainsync_epoch_timing_audit_20260907.zh.md)。

把采样率从 `f0` 改为 `f1` 时，保持跨度的奇数核满足

```text
K1 = 1 + round_even((K0 - 1) * f1 / f0).
```

所以严格的 256 Hz 对照是 `129/9/33`。`127/9/33` 的 ST 跨度为
`492.1875 ms`，不是纯采样率对照；若保留，必须标成额外的感受野扰动。

## 4. 参考电极与跨数据集

2026-09-16统一模型设计要求新源与个人输入从raw执行共同声明的产品profile；旧因果缓存不重标成双向。
双向处理须在互斥raw fit/validation/test范围内分别执行，先整记录处理再切epoch不足以保证隔离；
同profile的长短记录仍有不同有限时间算子与真实支持。计划比较`[六个Cz差分;实测Cz]`的可逆空间表示，
第七维须真实观测并保留原参考，缺Cz继续使差分无效；旧六导不可恢复该信息。
首版已实现：`RawSampleScope=[start,stop)`先裁原始连续输入再滤波；事件授权独立列出，响应尾窗内后续刺激不自动获得标签。
每个K分别生成C、inner T/V，与固定later test独立处理，原始事件ID、真实session/run和剔除台账保留。
当前精简排程的个人校准固定90轮，准备CLI默认仅生成实际消费的C_K和固定test，省去个人inner T/V。
若研究另启个人早停，仍须使用独立raw inner视图；停止模式不改变事件授权和时间隔离要求。
模型用CPU布尔`[N,C]`实测mask逐batch传递；temporal Conv/BN只消费pack后的实测轨迹，再scatter回物理槽。
缺导不计入BN均值/方差，亦不让NaN占位进入卷积梯度；全实测时与原生Conv/BN语义一致。
source准备入口当前只执行有限值、恒值及通道准入，learned QC/EOG启用仍须另行绑定合法fit范围。
原生七导的可逆参考扩展仍待研究。公式、mask及源/目标联动见
[统一模型输入方案](evidence/unified_model_plan_20260916.zh.md#5-输入统一与信息保留)。

若真实头皮电势为 `s_c(t)`，参考电极为 `r(t)`，记录值是

```text
x_c(t) = s_c(t) - r(t).
```

两个数据集使用不同 `r(t)` 时，即使 `s_c(t)` 相同，输入也不同。任意空间
权重和不为零的分类器都会响应这一公共偏移。因此 source reference 必须记录；
不同或未知参考的数据不能直接拼接。共同重参考必须是独立、显式、可消融的
变换，不能只修改 provenance 字符串。

BrainSync本人模型的默认基准保留左耳垂原生七导。已有源checkpoint需要Cz输入时，
显式生成`Fz/P3/Pz/P4/PO8/Oz - Cz`六导，按同一Cz合同比较迁移与scratch。
差分矩阵`D`为6×7且`rank(D)=6`、`D*1=0`，仅靠六导不能还原七导；保留原生Cz才能逆转。
Cz任务响应和噪声都参与差分，故共同参考不等于普遍去噪或性能最优。
七导与六导分别绑定QC、统计和模型；`--reference-mode native`拒绝隐式继承Cz QC/EOG。
数学反例、文献和更正记录见[左耳垂合同更正](evidence/brainsync_reference_20260908.zh.md)。

眼电共享参考旧模式仍先同时减Cz再回归，要求显式接法。新显式`preserve_acquired_eog`仅EEG减Cz、
EOG保持采集值，拟合`EEGc-Cz ~ EOG`，应用为`EEGc-Cz-beta*EOG`，Cz依赖系数始终-1。
同一校准设计矩阵下beta等于原生OLS的`beta_c-beta_Cz`；不意味着真实脑电与EOG统计独立。
新ocular `/4`固定原采集参考和EOG接法记录，未知保持未知；旧 `/3`及摘要不变，不能跨模式解释系数。

## 5. 分组证据

设 `g(i)` 为 run/session/decision group。所有外层和内层掩码必须满足

```text
{g(i) | i in train} intersect {g(i) | i in test} = empty.
```

随机 epoch 划分通常违反该式，并让相邻刺激或同一重复序列同时进入训练和
测试。没有至少两个真实 group 的被试不能生成单被试结果。

## 6. 代码逻辑链

```text
data.contract.EEGDataContract
  -> epochs.PreprocessingSpec.validate
  -> BIDS EEG validator | MOABB | BrainSync | GTN | raw manifest adapter
  -> BrainSync session v4 + completed/balanced-cycle + events/channels/electrodes/coordsystem gates
  -> preprocess: continuous IIR + declared phase/state -> source epoch -> FFT resample -> mean baseline
  -> EpochDataset.validate + QC features + SHA-256 attestation
  -> run_eeg_loso: physical contract + source provenance fail-closed
  -> train.factory -> N2P3NetBaseline
  -> N2P3Net architecture record (samples + physical milliseconds)
  -> source/calibration-only QC -> true-time split/embargo or fixed budget -> held-out logits
  -> positive-slope LLR -> tie-aware candidate aggregation + coverage/cost ledger
```

## 7. 尚未解决的边界

### 训练统计与预算的明确约定（2026-09-07）

当前compact512固定60遍历，batch上限512、实际`min(512,n_fit)`，保留真实尾批；
同人同K的NN使用同一局部随机生成器行序列，不受dropout消耗全局RNG影响。
同K匹配实际updates/draws；跨K更新数随校准量变化，不声称纯样本量效应。
不运行batch搜索、固定updates敏感性、BN一次重估或新增source统计/采样实验。
冻结BN buffer臂只冻结运行统计，gamma/beta仍可训练；冻结学习型trunk臂同时冻结非head参数
和BN affine，但保留online buffer及相同dropout。各臂拟合来源、参数和buffer分别记账。

以下是保留研究接口的数学约定，不是本轮排程。一次BN重估在该臂输入统计和初始权重安装后执行；依前向顺序逐层收集BN前激活，
上游buffer先安装并冻结。每通道激活总数N包含batch外的空间/时间维，
`mean=sum/N`、`variance=sumsq/N-mean^2`使用总体方差；后续优化/预测均不再更新此buffer。
这不是train-mode EMA遍历，也不能用推理batch划分改变估计。gamma/beta与其他参数的
可训练状态、dropout开关、BN模式分别记录。

训练对象与checkpoint共用实际输入变换`z=(x-mu)/sigma`；关闭标准化时该变换为
`mu=0, sigma=1`，而非把经验统计重新应用到原始输入。经验统计可单独保存作诊断，
缺导mask仍在变换后执行。保存/重建不得改变这个算子。

S1观测域/被试统计在通道c的真实观测集合内逐级等权：域、域内被试、被试内行、时间点。
令归一化质量为w，使用`mu=sum(w*x)`及`var=sum(w*x^2)-mu^2`，包含域间均值差；
不存在的观测不分配质量，输入统计不乘正类权重。普通有放回抽样记录p_i与完整目标风险r_i，
损失为`mean((r_i/p_i)*CE_i)`，r_i只出现一次。重要性修正不消除训练BN的batch上下文差异。

旧B0固定updates接口使用`U_ref=60*ceil(n_fit/128)`，本轮不调用；跨batch不能同时声称
draws及updates匹配。缺导、尾批、失败或OOM不能静默重定义标签池、风险质量或更新上限。

- BI2014a raw CSV candidate path 可恢复多 character decisions。causal-v2 与其
  cross-decision 结果已压缩隔离；当前须从 64 人 raw source 重建 causal-v3，不能
  改 cache 名称或 metadata 冒充新合同。
- GTN 的参考电极并非逐文件完整记录，跨数据集前需要独立重参考方案。
- GTN candidate occurrence 异步；不能把相同 occurrence index 当同步 block。
- 128 Hz 与 256 Hz 的比较必须共享源事件、带宽、基线、QC、fold、seed 和
  校准；只允许采样率及其跨度匹配核发生变化。
