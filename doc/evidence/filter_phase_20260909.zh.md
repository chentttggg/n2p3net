# 单向与双向滤波：科研裁决及实施

> 导航副本：历史失效链接现指向[持久来源索引](../assets/repository_audit_20261019/docs_historical_index.json)。原文以原SHA保存在本地归档；本次只改定位入口，原结果、执行producer与未找到凭证的边界保持。


本文件保存首轮无QC比较。后续按用户确定的双向方向完成个人默认与QC迁移，当前实现见
[双向滤波与QC联合优化](bidirectional_qc_20260909.zh.md)，以下首轮裁决不覆盖后续默认。

2026-09-09。检查实际滤波、BrainSync raw入口、时间账本、恢复证书、缓存和源模型门禁，
完成文献核查、已知信号反例及四会话三滤波条件×两参考的24次LDA拟合。

## 裁决

双向零相位对本次已知ERP的形态保真明显更好，但真实分类未见一致改善。
保留BrainSync单向稳态默认；新增显式双向科研合同及统一边界策略，修正双向证据可用时间。
双向路径用于独立完整记录的离线比较，不晋升为已经验证的个人猜数字配方。

已有完整block收齐后再推理的产品流程在原理上允许block内双向处理；当前个人恢复证书仍只
支持forward，且本轮真实数据是多block整记录。尚未实现或验证双向block范围的个人QC恢复
证书与同上下文训练，所以个人模型入口继续拒绝zero。该范围限制来自当前实现和本轮证据。

## 实际算子与数学

当前BrainSync在原始采样率上级联50Hz、Q30陷波及0.1–30Hz Butterworth带通，随后
切[-200,1200)ms epoch、FFT重采样至128Hz、保留179点、减[-200,0)ms均值。
连续域滤波先于epoch，真实采集断点独立处理，EOG不进入EEG滤波选择。

令实际级联为H，单向首段以`sosfilt_zi(H)*x[0]`初始化；流式后续段携带状态。
远离有限记录边缘时，实系数前后向算子满足：

```text
H_fb(z) = H(z) H(z^-1)
H_fb(exp(j*w)) = |H(exp(j*w))|^2
phase_fb(w) = 0
```

同一SOS双遍会改变幅频响应。带通自身原−3.0103dB截止变为−6.0206dB；比较涵盖相位及
幅度变化，不单独归因于相位。SciPy带通设计阶数N=4对应单遍8阶，加陷波后10阶，双遍
等效20阶。零相位也不保证滤后波峰或起始位置与真实成分完全相同。

已有zero存在两种边界行为：无陷波交给MNE；含陷波直接SciPy默认odd延拓。
当前250Hz级联的SciPy padlen=33点（0.132s），MNE从同一SOS脉冲响应估计2201点（8.804s）。
padding是数值延拓，不能解释成观察到的EEG，也不是伪迹恢复认证。

新增`zero_phase_edge_policy=mne_reflect_limited`统一经MNE公开接口对实际完整SOS估计padding，
逐采集段执行，长度截为`min(estimated_padlen, segment_length-1)`。
旧字段缺省为`legacy`并从旧格式record中省略，保持原波形和签名。
新策略名称明确处理方式；没有宣称较长padding普遍更优。

## 文献核查

以下于本次联网核查。摘要证据与全文证据分别标明。

| 来源 | 已核实证据及适用边界 |
|---|---|
| Widmann等，2015，[10.1016/j.jneumeth.2014.08.002](https://doi.org/10.1016/j.jneumeth.2014.08.002)，摘要；另核实MNE引用 | 应同时检查幅频、脉冲响应、阶数、过渡带、延迟与因果性；FIR/IIR和单/双向是不同设计轴。 |
| Acunzo等，2012，[10.1016/j.jneumeth.2012.06.011](https://doi.org/10.1016/j.jneumeth.2012.06.011)，摘要 | 非因果高通可系统性改变早期ERP/ERF成分。时间解释风险不能直接转成BCI准确率裁决。 |
| Tanner等，2015，[全文PMC4506207](https://pmc.ncbi.nlm.nih.gov/articles/PMC4506207/)，[10.1111/psyp.12437](https://doi.org/10.1111/psyp.12437)，NCBI BioC全文 | 双向Butterworth在连续EEG上处理；其0.1Hz较高截止保守。短平均ERP上使用同一0.1Hz仍可减P600、增伪N400。论文最终四阶/−6dB定义不同于本库设计N=4，不能直接照抄。 |
| Zhang、Garrett、Luck，2024，[I](https://doi.org/10.1111/psyp.14531)、[II](https://doi.org/10.1111/psyp.14530)，摘要 | 应结合真实ERP信噪比/测量误差与无噪模拟失真；最优设置依赖成分、计分、群体和噪声。 |
| Farquhar、Hill，2013，[10.1007/s12021-012-9171-0](https://doi.org/10.1007/s12021-012-9171-0)，摘要 | 超50个ERP BCI数据集的预处理与分类器交互，不支持滤波方案脱离后续分类器独立最优。 |
| Memmott等，2025，[全文PMC11914135](https://pmc.ncbi.nlm.nih.gov/articles/PMC11914135/)，[10.3389/fnhum.2025.1551214](https://doi.org/10.3389/fnhum.2025.1551214)，全文 | 30名成人RSVP P300，按inquiry加两端1s缓冲相对整校准集滤波，LDA MCC 0.343对0.334，在线模拟差异不显著。其带通是单向，只有notch双向；支持训练/推理上下文一致性，不能引用成双向带通胜出。 |

官方算子依据：[SciPy filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)、
[sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)、
[butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)、
[MNE filtering background](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html)。
MNE本地实际源码另核实了`reflect_limited`及按段限制padlen；软件版本随执行产物保存。

## 已知信号反例

8μV、350ms/80ms宽正高斯减3μV、150ms/50ms宽负高斯，完整生产epoch/重采样/均值基线：

| 条件 | 对无滤波生产输入的RMSE/μV | P300峰/ms | 未来1mV阶跃前1s最大污染/μV |
|---|---:|---:|---:|
| forward | 1.040144 | 354.6875 | 0 |
| zero legacy | 0.003801 | 346.8750 | 566.603 |
| zero reflect | 0.003801 | 346.8750 | 566.603 |

上述ERP位于120秒已知信号中央；不能外推为噪声抑制或真实准确率。
把同一ERP放在20秒记录的1/5/10/18秒处，以同算子长记录结果为参照并扣除基线差：
legacy边缘RMSE为0.000017/0.000140/0.001003/0.020215μV，reflect为
0.049812/0.004801/0.000015/0.051131μV。新策略在这一组位置并非普遍更好，不能按padding长短选优。
单向因已知信号的前史为零，本组边缘差为零，不是任意真实信号的无边缘误差声明。

[图像历史定位：幅频、ERP及未来阶跃反例](../assets/repository_audit_20261019/docs_historical_index.json)

附加反例覆盖非零first_samp下的断点隔离、1/2/32/250/5000点常量短段、EOG独立性、旧
SciPy波形逐位重放、无效策略拒绝、未来依赖、不可用epoch的NaN账本及缓存重载。
单向恢复、旧源权重及个人模型门禁拒绝zero，未把旧证书重新标记为双向支持。

## 四会话配对分类

使用已确认同一人的四份保留记录：0903_172410、0903_173947、0904_171926、0907_150258。
每次留出一个完整会话，另外三个训练；校准与测试raw分别滤波，既不拼接，也不随机拆epoch。
这是允许时间顺序相反的回顾性整记录留出开发实验。四折共享训练记录，只能作为一名参与者
的描述性结果，不能用四折充当四名独立被试。

每份450个scheduled保留441个原始支持epoch，训练1323、测试441；四份总1764/1800。
三臂逐事件身份、支持行及标签完全相同。每臂分别按原生七导和线性减Cz六导计算100ms窗均值，
使用相同LDA `lsqr/shrinkage=auto/priors=(0.5,0.5)`，24次拟合全部完成。
本比较未应用EOG或连续/epoch QC，包含已知污染，不能直接代表已部署QC后的分类器。

| 参考 | forward平均AUC | zero legacy | zero reflect | reflect−forward |
|---|---:|---:|---:|---:|
| 原生七导 | 0.477805 | 0.464002 | 0.473739 | −0.004066 |
| Cz六导 | 0.509121 | 0.507305 | 0.513560 | +0.004439 |

按上述会话顺序，reflect相对forward的原生七导ΔAUC为
−0.058934/+0.018875/+0.049514/−0.025718，Cz六导为
−0.027436/−0.004808/+0.077749/−0.027749。Cz平均改善由一个会话贡献，另外三个下降。

所有三臂两参考的首五轮九选一均为0/4，4/4会话首五轮完整；固定thought digit的每份session
只算一个声明的decision，十个presentation block不制造十份独立目标。
zero得分只在整记录结束后可用：本表的R5是回顾性首五轮得分，不能报告成第五轮结束可决策。
全记录处理范围也不能代替新测试只有一个block的产品效果。

当前数据没有提供一致的分类优势，且平均AUC接近随机。没有据此修改带通截止、恢复容差、
已有源权重或个人默认；也没有声称双向普遍较差。

## 实施与验证记录

`data.contract.BRAIN_SYNC_OFFLINE_DATA_CONTRACT`及
`configs/brainsync_offline_preprocessing.json`提供显式新合同。
`PreprocessingSpec`、BrainSync/GTN/BI/BNCI/raw manifest/source-record入口完整传递边界策略；
MOABB高层P300不具备该执行能力时明确拒绝。所有缓存/模型签名消费其record，旧缺字段仍为legacy。

通用`preprocess`原先把zero也记成`onset+tmax`，现将实际启用zero滤波的可用时间保守记为
`(raw.first_samp+raw.n_times)/f_src`，不可用事件保持NaN。`online_causal=false`保持显式。
历史缓存没有原始记录长度，不能事后修改其时间账本；新研究须从raw重建。

可复算入口为`experiments/run_filter_phase_study.py --sessions <完整会话目录...> --output <新目录>`。
运行前冻结的执行源码见`frozen/filter_phase_20260909_final_execution/source.manifest.json`；
最初dirty工作树也已压缩核验于`frozen/filter_phase_20260909_before/`，并发工作保留。
最终产物在`experiments/runs/filter_phase_20260909_final/`：预先保存的protocol、全部模型线性参数、
训练session身份、测试特征、标签、分数、决策账本、逐文件哈希、原始EDF前后摘要及图像。
保存分数由线性参数独立复算；其AUC和决策分母另作最终重载审计。

`reload_audit.json`已完成24模型、4份原始EDF核验；从原始会话重建特征/事件支持，并用保存
线性参数复算分数、秩统计复算AUC，验证每个测试session未进入其训练身份集。
本轮环境Python3.14.3、NumPy2.5.2、SciPy1.18.1、MNE1.12.1、sklearn1.9.0；
拟合与重放均限制BLAS为1线程。执行脚本、14个相关代码/配置/测试文件与冻结版本逐字节一致。

相关入口回归239通过、3跳过；新增14项反例通过。最终全套**1610通过、4跳过**（281.01秒），
Ruff及`git diff --check`通过。此前全套1606通过、4跳过、1失败，失败为并行源训练子进程
OpenBLAS内存分配重试耗尽；将OPENBLAS/OMP/MKL线程上限设为1后全套通过，原失败日志保留。
两次全套之间并发维护重整测试、净增加三个案例，因此总数不同；本轮滤波生产代码与执行快照保持一致。
最终检查日志及XML随结果保存；最终源码和证据归档清单分别位于
`frozen/filter_phase_20260909_delivery/source.manifest.json`与
`frozen/filter_phase_20260909_evidence/manifest.json`。
