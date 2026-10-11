# 输入、预处理与真实信号审查（2026-09-29）

范围：只读生产代码与原始数据，执行独立数值/真实数据探针，不训练、不删除被试、不修改旧缓存、预测或结果。此文件是全链审计的输入分支证据，不是新活动指南。实施状态：未实施。

## 1. 最重要结论

**当前最值得优先处理的是独立有限 raw 段的双向滤波边界与真实污染叠加，不是未证实的标签错位或缺导填零污染 normalizer。** BI2014a 01 的最后一个 C200 epoch 出现原生 Cz 高频干扰经有限段滤波边界传播形成的毫伏级坡尾；49/26 的大量异常却在完整记录处理时仍保留，因此两类问题必须分开。所有问题都不能单独解释 A1×Q1 性能交互。

另外，最小测试样本数合同有已复现的静默降级；EA 有默认正则化单位尺度、静态缺导准入、白化后单位声明及旧拟合范围保留等独立问题。EA 不是当前 plain unified12 的路径，不应把这些旁路问题当作其已执行成绩的病因。

## 2. 当前源码与执行身份

`source_identity_comparison.json` 将当前以下十文件与 `frozen/unified12_20260918_performance_v4/source.source-only.tar.gz` 逐字节核对，全部相同：

- `data/source_parsers.py`、`source_records.py`、`preprocess.py`、`epochs.py`、`domain.py`、`bnci2014_008_candidate.py`、`dataset.py`；
- `transfer/binary.py`、`subject_adapter.py`、`train/source_sampling.py`。

v4 archive=`c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24`。实际探针运行于当前项目 `.venv`；环境版本在 `input_probes.json`。源码等同不等于宣称历史全矩阵已逐位重放，本轮仅重建列明的真实切片。

## 3. F0：BI01 存在强有限段末端效应，Cz 参考将它传播到所有差分槽

### 可定位事实

原 MAT：`mne_data/MNE-braininvaders2014a-data/zenodo/3266223/subject_01/subject_01.mat`，SHA-256=`eecb29c236f21b5d8d1b8b3cc07e4aa96b107eeb29033201fcab407d94fdd759`。

原始第 199 个事件（零基序号）在 sample 64734，即 126.43359375 s，标签为 non-target。其源 epoch 为 `[64632,65349)`，恰好终止于 C200 独立 raw 的右边界。当前合同是 0.1–30 Hz、50 Hz/Q30 notch、128 Hz、`[-200,1200)`、双向 `mne_reflect_limited`、逐 trial baseline、Cz 差分。

同一原始 200 个事件仅改变连续滤波支持，P3-Cz 得到：

| 处理 | 全 C200 的最大 epoch RMS | 全 C200 的最大绝对值 |
|---|---:|---:|
| 独立 C200、zero | 1517.55 µV | 3545.04 µV |
| 整个原 record、zero，再取相同事件 | 249.17 µV | 998.82 µV |
| 独立 C200、forward | 372.82 µV | 1404.88 µV |

四个实测差分槽上，独立 zero 的 PTP>500 µV 涉及 22/200 个 epoch，整 record zero 为 14/200，forward 为 19/200。这里 500 µV 仅是固定描述性诊断值，不是新拒绝规则，也没有删除任何 epoch。

独立 zero 与 full-record zero 的相同 C200 张量最大差 3565.29 µV，RMS 差 165.21 µV。最后一 epoch 的大坡尾在 full-record/forward 中不出现；这不表示 forward 的整个 C200 都干净。

### 机制的独立线性分解

`probe_bi_boundary_linearity.py` 直接调用生产 SOS、`filter_bidirectional_segment` 和 epoch FFT/baseline 核，不构造模型。P3-Cz 最后 10 s 主谱为 50 Hz；描述性正弦投影为 50.012187 Hz、振幅 6687.58 µV。分解的是 `ΔL=(L_C200−L_full)`，并始终保留原信号与残差。

- 最后一 epoch 的 `ΔL(raw)` RMS=1522.88 µV。
- `ΔL(50 Hz 投影)` RMS=1629.51 µV；`ΔL(残差)` RMS=106.63 µV。
- 残差范数约为总边缘差的 7.00%；分量可相消，所以这不是“93%因果解释率”。
- 原生通道的边缘差 RMS：P3=127.22 µV，Cz=1650.10 µV；`ΔL(P3−Cz)=ΔL(P3)−ΔL(Cz)` 的最大残差仅 `5.6e-15 V`。此样例以 Cz 的边界状态贡献为主。
- 当前自动 padding 为 4472 sample，约 8.73 s。`preprocess.py:435-438` 的 odd reflection 使用 `2*x_end-reversed(past)`；截断强周期信号的相位与边界外推共同改变有限滤波结果。

同 200 个事件 P3-Cz 的**裸总体 std**为 207.655 µV（独立 zero）对 61.094 µV（full-record zero）。`SubjectAdapter` 实际加 1 µV epsilon 后分别为 208.655/62.094 µV。后者不是 forward 对照，也不是整个多源训练池的 normalizer。这个变化说明相同 normalizer 公式会消费非常不同的实际输入分布。

原始、差分、连续滤波和 epoch 对照图：`bi01_raw_extreme_anchor.png`。频谱/线性分解图：`bi01_boundary_linearity.png`。原始 sample、SHA、各导峰值及所有大幅 epoch 的事件位置/标签见对应 JSON。完整小型数值数组保存在忽略目录 `experiments/runs/deep_chain_audit_20260929/input/bi01_boundary_linear_components.npz`。

### 影响范围与边界

这直接影响采用同一独立有限 C200 和该 raw 的重建数值；核心代码与 v4 一致。尚未重放全部历史 source/target 张量或证明它导致多少 AUC/hit 下降。Q0、Q1 和同输入 scratch 共同消费该输入，因此不能只归因给 decision loss。实际外层矩阵的完整数值身份仍以对应 run/checkpoint 为准。

不能把 full-record 对照晋升成合法校准方法：它在这里仅作为只读算子诊断，可能消费当前校准边界外 EEG。修复必须明确合法上下文/端点不确定性、实际保留 K、质量账本与新 cache 身份；不能通过读测试段来消除边缘。

## 4. F0 的反例：BI49/BI26 的异常不能统称为末端效应

按父审计独立普查的高幅样本选择两例，使用同样固定公式比较相同 C200 事件，不按分数筛选。

| 人 | 原始最坏事件与位置 | 独立 zero 中任一导 PTP>500 µV | full-record zero | forward | 结论边界 |
|---|---|---:|---:|---:|---|
| 49 | event 46，sample 13430，26.230 s；距 C200 尾端106.97 s | 200/200 | 200/200 | 200/200 | 原生 Pz 大幅尖峰/慢波保留，最坏 epoch full/prefix 逐点相同 |
| 26 | event 2，sample 3086，6.027 s；距 C200 尾端121.14 s | 179/200 | 173/200 | 173/200 | 原生 Oz 强周期干扰及大幅低频变化保留，同时另有末端附加效应 |

49 的最坏 Pz-Cz epoch 峰值约3001.86 µV，原生 Pz 在该源 epoch 的 PTP=5633.63 µV；相同 epoch full/prefix 差为0。26 的最坏 Oz-Cz epoch 峰值3577.97 µV，原生 Oz 对应 PTP=30824.85 µV；相同 epoch full/prefix 差同样为0。26 的其他末端 epoch 仍出现最大2924.71 µV 的 full/prefix 差，因此“真实污染”与“有限段附加效应”可以并存。

**full/prefix差为0只排除新增C200右端贡献，不能排除两种处理共有的左端初始化/延拓效应。** 两者均从原record起点开始；26最坏事件仅在开始后约6 s，必须保留共有左端瞬态这一竞争解释。原raw的高幅干扰事实已直接观察，但其滤波后低频形态中有多少来自共有左端状态，本分支尚未通过另一独立控制分解。49也不因距右端较远而自动获得“完全无边界影响”的认证。

图片与全部 sample 锚点：`bi49_raw_extreme_anchor.*`、`bi26_raw_extreme_anchor.*`。未从已滤波信号宣称 ADC 饱和，也未把原生大幅波形自动分类成唯一生理/设备原因。

## 5. F1：minimum_test_trials 在 fallback 中静默失效

源代码：`src/transfer/binary.py:118` 先检查 `len(test)<minimum`，捕获后 `:130` 的 feasibility 只要求 `len(test)>0`，`:138-141` 又只记录 `requested budget > effective maximum` 的调整。最终返回不重查 minimum。

独立纯函数反例：N=300、请求 K=(0,200)、minimum=200，K 已小于80%上限，返回 test=100 而 `adjustments={}`。独立 recording-disjoint 测试池只有100行时，同样在 minimum=200 下被放行且没有原因记录。

真实端到端反例：Huebner2017 subject1/run0 原 raw 有476事件，调用 `prepare_source_record_prefixes(K=0/200/800,min_test=200)` 得到 K=0/200/380、test=91；`verify_prepared_prefixes` 和 `load_target_prefixes` 均接受。准备 manifest 仍声明 minimum=200。见 `minimum_test_probe.json`；原prepared manifest已逐字节移至 `experiments/runs/deep_chain_audit_20260929/input/minimum_test_real_prefix/target_prefixes.json`。其内部绝对路径保留执行时原值，按 `relocation_manifest.json` 解析当前位置，不把旧路径假称为仍存在。

用户已允许的“有限数据按约80%回退”本身不是漏洞。问题是字段名看似底线、实际只作首次触发条件，并且某些路径完全不记录放宽原因。应显式区分 requested minimum、是否允许有限数据放宽、effective minimum、实际 test 数与原因；或严格执行底线。不能事后无说明重标旧评分。

父考古分支另已找到 09-16 unified 真 artifact：计划 minimum=200，但 BI2014a 人6/K200 `test_rows=85`；见 `experiments/runs/deep_chain_audit_20260929/comparability/unified_0916_cells.csv`。本分支不猜测其准备阶段使用了16还是200，因为当前代码在所有层都写200时也能生成此类低于200的结果。

## 6. F2–F5：EA 旁路已复现的问题

### F2：默认 covariance floor 与 V² 尺度不相称

`data/euclidean_alignment.py:59,169-174` 与 CLI `:95` 默认 `regularization=1e-6`，直接施加到单位 V² 的平均 trial 二阶矩。数值求解可执行，但对普通微伏 EEG 大幅压平几何。

- 真实 Huebner C64：floor 是平均特征值的4579.5倍；W 与 `999.904 I` 的相对 Frobenius 差只有0.0123%；输出二阶矩白化残差0.99978。
- 旧 BI EA target/source 两份真实状态：白化残差0.96179/0.92976；报告的 condition=1.447/2.426 是加 floor 后的矩阵，实际变换输出 condition 仍为76.98/116.17。
- 使用同一默认数值处理 V 与 µV 表示，输出相对差98.74%；实际单位转换不能只改字符串。一个按当前原始能量定标的小 floor 的算子诊断可达接近单位矩，但没有宣称该参数已获模型性能收益。

证据：`ea_units_probe.json`。现有旧 EA 弱收益不能作为正确尺度、匹配源/目标坐标 EA 的否证。本问题不作用于 plain unified12。

### F3：EA CLI 只检查动态 mask，静态缺导可被重标为实测

`prepare_euclidean_alignment_cache.py:144-147` 只检查存在的 `trial_channel_mask`；`:156,166,169` 无条件把输出静态 mask/QC observed 置全真。

用真实 Huebner两导加一个明确未测 Oz 零槽，形成合法 `EpochDataset(channel_mask=[1,1,0], trial_channel_mask=None)`；save、attested reload、实际 CLI 都成功，EA输出 `[1,1,1]`，Oz 数值仍全0但 QC 声称 observed=True。`ea_static_mask_probe.json` 及对应前后小缓存保存证据。

前后cache/sidecar现位于 `experiments/runs/deep_chain_audit_20260929/input/ea_static_mask/`；原JSON的producer路径未改写，迁移前后SHA见 `relocation_manifest.json`。

此前“EA CLI拒绝所有缺导”的概括过强，应更正为“它会拒绝含缺失的动态 trial mask，遗漏静态 mask”。本地 `cloud_inputs` 十份现存有效 record 的有限库存未发现此表示，尚不能指认哪个历史正式 run 触发它。

### F4：EA 输出仍声明 V，并生成以伏为单位的 QC

EA 的 `W X` 是二阶矩白化坐标；CLI `:157,164,169` 保持原 `PreprocessingSpec.signal_unit='V'`，同时对变换后数值计算 `channel_std_v/epoch_scale_v`。当前 attested loader接受真实 `bi2014a_ea_target_calibration_relaxed_eligible61.npz`：median `channel_std_v=0.030435`、最大“V”=9.588746。这些不是可解释成物理伏特的头皮电压。

`checkpoint.py:525-538,587-619` 比较输入预处理/参考/通道，但没有 data_alignment 路线坐标兼容门禁；新的完整 EA 不能依赖这些门禁自动保证单位一致。应把物理电压质量状态与模型坐标变换状态分开，并冻结与训练权重/normalizer 相匹配的 route 合同。

### F5：裁掉被试行没有撤回已拟合 EA 的贡献

现存旧 `*_eligible61` lineage 明确先执行 `euclidean_alignment`，之后 `exclude_author_rejected_participants`，并标记 `retained_rows_unchanged=True`：

| 缓存 | 当前保留行 | EA state sample_count | 后删行 |
|---|---:|---:|---:|
| target_calibration_relaxed_eligible61 | 12252 | 12900 | 648 |
| source_rows_relaxed_eligible61 | 57437 | 61011 | 3574 |

当前 loader仍接受该 target cache。状态仍包含后来排除的62/63/64三人贡献。这是现存派生缓存的实际资格问题。

影响必须分开：考古分支已证实09-05的64人旧 EA run使用裁人前缓存，它当时并未声称61人，不能把后来的裁人错误倒扣给它；目前没有证据显示 eligible61 EA缓存被后续新训练实际消费。旧64人实验的默认 floor/坐标问题另论。详情由 `../comparability/` 的执行考古材料承接。

## 7. 真实标签、时间与单位核验：当前没有发现错位

`public_raw_probe.json` 与 `input_probes.json` 覆盖13份真实记录，直接从原始字节/原标注交叉核对：

- BNCI A01–A08：共33600 flashes；sample-wise `y_stim` 非零上升沿、MATLAB `trial−1`、raw二类标签、每120flash的行/列目标归属全部吻合。A01新建 K200/K800得到200/800epoch、34/133正例、1/6完整selection；经过 FFT、baseline、reference 后标签和原始 sample 仍一致。
- BI2014a subject01：CSV flash code/membership 与末列 binary label、MAT、MOABB事件1188个/198target全部吻合。MAT与CSV EEG最大差约0.49707 µV，事件/标签零差；没有把两份 raw 的不同字节身份冒充同一输入。MOABB 输出与 MAT EEG×1e-6逐值相同。
- Huebner subject1/run0：476事件/112target与原 events.tsv逐项相同；BDF头µV由MNE转换为V，源/目标切片标签映射正确。
- Cattan subject01 PC/VR：各720事件/120target；原22列MAT中的两类指示列与源 parser输出完全吻合，EEG×1e-6逐值相同；VR仅按已有明示合同去掉一个无事件坏尾采样。
- BrainForm P01：原CSV600事件/60target；startup trim=1245仍保留绝对 first_samp，第一事件sample2254；原列标注、事件位置与×1e-6单位变换一致。

这些验证只覆盖所列记录，不认证所有源、全部人或原设备物理标定，也不把 dataset 内部自洽当作真实人体目标状态的独立认证。

## 8. mask/normalizer/身份：排除的疑点与保留的机制风险

### 已排除的常见误报

真实 Huebner缺导切片上，把未测槽写入 `1e12` 仅作为独立数组攻击，三套 masked统计（普通 DeepBaseline、fixed-source统计、SubjectAdapter）对实测导得到相同均值/std；缺导攻击不改变结果。生产 EpochDataset本身还要求缺导值严格0。标准化后各路径重新把缺导槽置0。

统一 `observed_temporal_bn/1` 确实先pack实测导，在 affine之后scatter回零槽；不能再报告“缺导零填充进入首个BN、beta复活缺导”的已推翻结论。

五份真实 C200视图（Cattan PC/VR保留同一个原始participant及不同session，经显式命名空间）按生产 binary projection → namespacing → concatenate → channel reverse → event subset → save/attested reload，X、y、mask与event_ids逐值/逐字节一致。共1000行，subset保留667行。未显式 binary projection 的 mixed candidate/binary合并按要求拒绝。

### 已存在的域信息，不等于已确认标签泄漏

五个小切片的 mask 固定于记录，实际模式如下（Fz,P3,Pz,P4,PO8,Oz）：

| 来源 | mask | C200 target比例 |
|---|---|---:|
| BI2014a 01 | 011101 | .165 |
| BNCI A01 | 111111 | .170 |
| Huebner 1/run0 | 111100 | .235 |
| BrainForm P01 | 101011 | .100 |
| Cattan 01 PC+VR | 111101 | .1675 |

mask没有在这些记录内部随标签改变，但它暴露域身份；域的类别比例也不同。`probe_mask_mechanism.py` 构造一个明确参数状态：observed-temporal BN affine bias=1而缺导仍0，随后合法空间卷积核 `[1,2,4,8,16,32]` 可给五种 mask 不同数值。这是表示能力的存在性反例，不是已训练网络使用该捷径的证据。

`common_channel_intersection` (`domain.py:30-40`) 只做槽名交集；固定六槽被补零后它仍返回六名字，不能把返回值当成每record/epoch实测交集。现有完整CAR/selection owner随后还有mask门禁，所以该工具本身不等于现行错误重参考。

BrainForm manifest有167记录，其中3条声明坏导：P03一录制排除Oz、P02一录制排除Fz/Pz、P14一录制排除Pz。如果最终source准入保留这些记录，则即便跨库“最多Pz-Cz”的共同候选维度也会被消除；最终prepared inventory尚须核验，不可假设十二库存在完整公共多导子空间。

### 一个已复现但主链未触发的API陷阱

`adapt_observed_montage(dataset, ['Cz','Pz'], reference_channel='Cz')` 接受Cz作为输出槽并将恒零 `Cz−Cz` 标observed。当前统一 `SIX` 明确不含Cz，因此不是本轮性能原因。应在接口入口拒绝anchor进入target槽，或明确该槽不应算作实测分类特征。

## 9. 时间支持攻击与缓存身份

- 真实 Huebner C64后注入10mV未来扰动，独立 raw view 的全部64epoch逐位不变，forward/zero均通过；独立raw隔离生效。
- 相同zero profile先处理完整record再截C64，与独立C64最大相差30.66µV/RMS3.21µV；forward该样例为0差。相同profile名不是相同实际支持。
- nonzero `first_samp=12345` 的独立scope测试保留绝对onset及原始标签，不存在该探针触发的样本减重/标签off-by-one。
- `source_preparation_method`绑定data源码与包版本，prepared cache绑定NPZ与record摘要；当前未发现所查source缓存复用忽略预处理配置。minimum等语义字段的执行缺口仍可在这些字节身份门禁全部通过时存在。

## 10. 质量图与结论限度

`signal_quality_diagnostics.png` 使用完整C200（Cattan PC/VR各200），固定公共ERP纵轴和公共log-RMS纵轴，没有删除高幅点；底层表为 `signal_channel_summary.csv`、`signal_erp_curves.csv`。图已经实际打开检查，最大点未裁掉。原始幅度阈值只是诊断，不作分类器调优。

BI01类均值含明显高幅/斜坡影响，不能当作可信ERP形状或准确率。其余几个小切片峰值<70µV、没有恒平epoch，也不意味着其全部库/全部record合格。滤波后的epoch不能单独确证ADC clipping或上游硬件故障。

## 11. 实施建议与验证出口

1. 先把独立 raw 两端质量/边界不确定性列入正式输入准入与结果账本，严格保留允许的时间/标签范围。先解释同事件算子差，再解释模型差；不使用后续测试EEG作为隐藏边界上下文。
2. 对minimum-test回退补完整语义记录与准入行为；新旧合同分别绑定，不重标历史结果。
3. EA在上述问题解决后另建route：有效mask、物理单位/模型坐标、relative数值稳定策略、fit行/人/时间资格、保存重放和checkpoint兼容同时集中实现。
4. 新的raw/窗长/参考试验必须报告边界敏感度，否则改变窗长同时改变截止相位可能被误解为生理窗收益。
5. 对源模型是否利用mask域信息，只在源内冻结测量或未来匹配实验中诊断；不凭mask可区分来源就声称它造成负迁移。

全部owned脚本已跑到真实入口并保留JSON/日志；独立原始切片与代数反例不是训练结果。Ruff检查本目录脚本通过。生产修改、全库回归、完整历史张量重放、新训练及成人效能确认未执行。

## 12. 本地证据整理

本分支生成的15份cache NPZ、record和prepared manifest共8,144,855字节已移至忽略目录 `experiments/runs/deep_chain_audit_20260929/input/`，每份移动前后SHA-256一致。与最终身份修正版本同名的早期 `real_mask_*` 产物单独保留在 `prior_probe_output/`，没有覆盖原已在忽略目录的最终版本。`doc/assets`仅保留小图、摘要、诊断表和脚本。

原执行JSON、日志、sidecar和manifest字节保留，故其中部分原producer路径现在是历史路径；`relocation_manifest.json`提供逐文件新旧位置及核验摘要，`artifact_locations.json`提供当前入口。修改前的九份审计脚本快照保存在忽略目录 `audit_script_snapshots/before_output_relocation/`；old/new审计脚本哈希及path-only变化另记入迁移清单。这不是生产源码改动，也没有重新计算raw、跑数值探针或训练。后续EA/前缀重建脚本默认写入新的本地 `rebuilds/` 子目录，以保留本轮执行证据。
