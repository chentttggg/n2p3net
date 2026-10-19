# 07 — 输入合同 / 数学契约审计（doc/input_contract_math.zh.md ↔ src/data, src/models）

状态：完成（增量落盘后整理）
审计人：subagent 07
基线：当前工作树（非 HEAD）；基准提交 f1a303c
范围：doc/input_contract_math.zh.md（443 行全文）、README 合同段、doc/technical_structure.zh.md 相关行；src/data/{epochs,preprocess,events,brainsync_contract,brainsync,finite_filter_operator,harmonic,euclidean_alignment,bidirectional,ocular}.py；src/models/n2p3net.py
方法：grep 定位 + 精读关键函数 + python 小信号实验（脚本在 tmp/audit/scratch/check1..4.py）；未运行训练、未改源文件

## 结论摘要

1. **P1** 零相位滤波有两条后端：新 `mne_reflect_limited` 的 padding 由 MNE 按完整 SOS 脉冲响应估计，250 Hz 下 **2201 样本（8.80 s）**；`legacy` 分支用 scipy 启发式，仅 **27 样本（108 ms，250 Hz）/ 211 ms（128 Hz）**，两者相差 67–82 倍，边缘保真度不可比（`preprocess.py:521-525`）。
2. **P1** 原生 128 Hz 源的 epoch 首样本实际在 **−203.125 ms**，不是合同写的 −200 ms（MNE 按 `round(tmin·f_src)/f_src` 取网格）；截前 179 点后保留窗是 **[−203.125, 1187.5) ms**，而 250 Hz 源是 **[−200, 1190.625) ms**。文档 §2.3 的 `t_j = tmin + 1000j/f_model` 只对 250 Hz（及 tmin·f_src 为整数的率）成立（实测 128/256/512：−203.125/−199.219/−199.219 ms）。
3. **P2** EOG 在两条后端下预处理不同：新路径只对 `_eeg_channel_names` 滤波（`preprocess.py:509`），legacy 路径 `raw.filter(...)` 不带 picks，连 EOG 一起滤（`preprocess.py:532-540`）；receipt 的 `filter_segments` 只记录 EEG（`preprocess.py:775`）。EOG 回归的拟合/应用因此随后端变化且不可追溯。
4. **P2** 250→128 Hz 的 FFT 重采样在目标 Nyquist 之上不是砖墙：实测 70 Hz 仅 **−22.2 dB**、100 Hz −35.1 dB；60 Hz（<64）正常通过。前级 0.1–30 Hz 带通已压制该段，风险低，但文档未给抗混叠指标。
5. **P2** “半开窗口”只在输出索引上成立：被丢弃的 1200 ms 端点样本会经重采样核漏回基线窗，1 V 端点扰动 → 首 26 点最大 **9.5e−4 V**（−60 dB）；真实 EEG 下约 0.1 µV，可忽略，但文档只写了单向传播。
6. **P2** 校准/测试 raw 独立切分后各自双向滤波，heldout 起点处只有段内 odd 延拓（padlen 被 cap 到段长−1），8.8 s 的 0.1 Hz 瞬态仍进入首批 test trial；`prefix_raw_scopes` 未加 guard 或预热丢弃，仅文档层面声明风险（`preprocess.py:116-148`）。
7. **P2** README 第 91-92 行把 “default causal input contract … forward IIR” 写成默认，而 `DEFAULT_P300_DATA_CONTRACT.filter_phase="zero"`（`contract.py:77`），`P300_PERFORMANCE_PREPROCESSING` 也是零相位；文档 §2.1 亦区分 offline zero / online forward。
8. **正面** 50 Hz/Q30 陷波实测 −3 dB 宽 **1.6666 Hz**（250 与 128 Hz 相同），50 Hz 处 −280.3/−292.9 dB；0.1–30 Hz 四阶 Butterworth 在 0.1 与 30 Hz 均为 −3.01 dB、0.2 Hz −0.02 dB；T=179、|B|=26、250→128 的 351→180→179 几何全部与文档一致；`mne_reflect_limited` 输出与 MNE 1.12.1 的 IIR zero-phase 逐样本相同（差 0.0）。

## 逐条文档—实现对照

| 合同条款 | 文档位置 | 实现证据 | 判定 |
|---|---|---|---|
| f_model=128 Hz, T=floor(1400·128/1000)=179 | §1 L62-71 | contract.py:68,90；epochs.py:117-123 | ✅ |
| f_src=250 Hz 只作为采集证据 | §1 L58-60 | brainsync_contract.py:410-423 与 BIDS 实际率比对，未硬编码 250 | ✅ |
| 单位 V | §1 L69 | contract.py:74；epochs.py:126-127 拒绝非 V | ✅ |
| 7 EEG + 独立 EOG，EOG 不进卷积/head/CAR | §0 L6-8 | contract.py:10-20；preprocess.py:235-239（只 pick EEG） | ✅（新路径）|
| 参考 left earlobe，非 T7/乳突 | §0 L9-11 | contract.py:12,23-31；brainsync_contract.py:156-172 | ✅ |
| 0.1–30 Hz 四阶 Butterworth | §1 L66 | contract.py:69-70；preprocess.py:413-415；实测 −3.01 dB@0.1/30 | ✅ |
| 50 Hz、Q=30 二阶 IIR 陷波，先陷波后带通，同一 SOS | §2.1 L132-140 | contract.py:122；preprocess.py:404-405,416；实测 1 节、宽 1.6666 Hz | ✅ |
| epoch [-200,1200) ms | §1 L67 | contract.py:71-72；preprocess.py:111-112 | ⚠️ 索引正确，物理起点见 F2 |
| per-trial/per-channel mean-only baseline，禁 scale | §2.3 L277-295 | preprocess.py:578-601；575-576 明确拒绝 std/mad | ✅ |
| epoch 域 FFT 重采样，npad/window=auto，pad=edge | §2.2 L259-267 | epochs.py:149-160；preprocess.py:622-625 | ✅ |
| MNE 闭端 epoch → 前 T 点 → 半开基线 | §2.2 L248-257,267 | preprocess.py:926-946；实测 250 Hz：351→180→179 | ✅ |
| 缺导/坏导：永久坏非参考导剔除；参考坏拒绝；缺失置 0 并带 mask | §1 L43-45 | preprocess.py:902-907（非有限即拒绝）；map_channels/mask | ✅（浅查）|
| EOG 共享参考旧模式 vs preserve_acquired_eog | §4 L376-379 | ocular.py:161-197, 200-266 | ✅ |
| 恢复映射由含陷波的同一 SOS 计算 | §2.1 L137 | recovery.py:23-30 调 design_continuous_sos | ✅ |
| 双向状态误差/条件界 | §2.1 L179-183 | bidirectional.py（只读接口，未逐式复算） | 未验证 |

## 发现

### F1（P1，已证实）零相位两条后端的 padding 相差 67–82 倍，legacy 边缘失真远超其延拓长度
证据：`preprocess.py:507-525`。新策略 `mne_reflect_limited` 用 `mne.filter.construct_iir_filter(dict(sos=sos))["padlen"]`，实测 0.1–30+notch（5 节）在 250 Hz 为 **2201 样本 / 8.804 s**，128 Hz 为 **1166 / 9.109 s**；legacy 分支（`preprocess.py:524`）写死 `3*(2*len(sos)+1-min(...))`，0.1–30 纯带通（4 节）为 **27 样本**（250 Hz 108 ms、128 Hz 211 ms），带陷波（5 节）为 33 样本。两条路径都执行 odd 延拓：`filter_bidirectional_segment` 与 `scipy.signal.sosfiltfilt(padlen=2201, padtype='odd')` 最大差 **0.0**，与 `mne.filter.filter_data(..., phase='zero')` 最大差 **0.0**（check2.py）。
影响：0.1 Hz 高通时间常数 1/(2π·0.1)=1.59 s，MNE 估计的 8.8 s 延拓≈5.5τ；legacy 只延拓 108–211 ms，段两端约 1 个 τ 内仍有明显瞬态。任何仍标 legacy 的缓存与 `brainsync_bidirectional_v1` 缓存不能用于边界/首末 trial 的对比；`assert_p300_input_contract` 虽含 `zero_phase_edge_policy`，但没有区分两种情况下的保真度。
建议：在文档与 receipt 中显式写出两种后端的 `padding_limit_samples` 物理值（现在只有样本数）；对 legacy 缓存禁用“首/末 trial”类结论，或在读取端按 padding 给可用性降级。
验证：check2.py 输出；`construct_iir_filter` 的 padlen 与 `sosfiltfilt` 对比。

### F2（P1，已证实）128 Hz 原生源的 epoch 物理起点偏移 3.125 ms，截断后窗口与 250 Hz 不同
证据：`preprocess.py:111-112` 用 `round(tmin·f_src)`、`round(tmax·f_src)`；MNE `Epochs` 的网格也是 round：实测 128 Hz 得 n=181、`times[0]=-0.203125`、`times[-1]=1.203125`；250 Hz n=351、t0=−0.200；256/512 Hz t0=−0.199219（n=359/717）。`preprocess.py:938-939` 由 `epochs.times` 取 `start_offset/end_offset`（−26/154 @128），`_transform_epoch_samples` 从第 0 点截 179 点 → 保留窗 **[-203.125, 1187.5) ms**；250 Hz 源为 **[-200, 1190.625)**。基线 `start=ceil((-0.2 - (-0.2))·128 - 1e-9)=0`、`stop=ceil(25.6-1e-9)=26`，于是 128 Hz 下把 **−203.125 ms（窗外）** 的样本计入均值，而严格 [-200,0) 只应有 25 个网格点（`preprocess.py:589-590`）。
影响：文档 §2.3 的 `t_j = tmin_ms + 1000·j/f_model` 与实际采样时刻差 3.125 ms（0.4 样本）；ERP 潜伏期/基线均值有系统偏移，且 128 Hz 与 250 Hz 的“同一窗口”实际末端相差 3.125 ms，违反 §7 末条“128 与 256 Hz 比较共享基线/带宽”的意图。`moabb.py:1875` 的检查用 `atol=0.5/sfreq`（128 Hz 时 3.9 ms），恰好放行该偏移。
建议：`PreprocessingSpec` 或 `EpochDataset` 记录 `actual_t0_s = round(tmin·f_src)/f_src`，基线索引引用实际 t0 计算；或在截断前按 `round((tmin_actual - tmin)·f_model)` 丢弃首样本，使 128/250 网格严格同相。
验证：check3/check4.py 的 MNE epoch 几何；用 128 Hz raw 重建 179 点索引算得首末时刻。

### F3（P2，已证实）EOG 是否被滤波取决于后端路径，且未进入 receipt
证据：`preprocess.py:500-528` 新路径 `for name in _eeg_channel_names(raw)`，只滤 EEG；`preprocess.py:529-540` legacy 路径 `raw.filter(l_freq, h_freq, ..., skip_by_annotation=FILTER_SKIP_ANNOTATIONS)` 没有 `picks`，会滤所有数据导（含 EOG/ECG）。`preprocess.py:775` 的 `filter_segments` 只遍历 `result.channel_names`（EEG）。`ocular.py:229-236` 用 EOG 拟合线性回归。
影响：同一 `PreprocessingSpec`（仅 `zero_phase_edge_policy`/notch 不同）会产生“滤波 EOG”与“原始 EOG”两种输入；EOG 回归系数、残余眼电与 EEG 的相位关系随之不同，且缓存里查不到 EOG 的滤波支持。
建议：显式规定 EOG 是否随 EEG 滤波（当前 `PRESERVE_ACQUIRED_EOG` 语义倾向“不滤”），在 receipt 增加 EOG 通道的 filter_segments；若统一为不滤，legacy 路径需记录差异。
验证：读代码 + 对 `_eeg_channel_names` 的 pick 类型确认（eog=False）。

### F4（P2，已证实）重采样的抗混叠在目标 Nyquist 之上只有 −22…−35 dB
证据：`preprocess.py:623-625` 用 `mne.filter.resample(..., method='fft', npad='auto', window='auto', pad='edge')`。实测 250→128 Hz、351 点 epoch：70 Hz 残余 RMS 0.05482（输入 0.7071，**−22.2 dB**）、100 Hz 0.01243（−35.1 dB）、60 Hz 0.70254（正常通过）。`npad='auto'` 对短 epoch 的频域过渡带很宽，不是砖墙。
影响：若某数据集未做 0.1–30 Hz 连续滤波（或 h_freq=None），64–100 Hz 的肌电/工频谐波会以百分之几的量级折叠进 0–64 Hz。默认合同有 30 Hz 低通，实际风险低；但“重采样抗混叠”在文档中没有量化指标，容易被误读为已消除。
建议：文档补“抗混叠由前级 30 Hz 低通承担，epoch FFT 仅作数值重采样”；若允许 h_freq=None，应在 `PreprocessingSpec.validate` 警告或要求显式声明。
验证：check3.py 的 60/70/100 Hz 注入。

### F5（P2，已证实）FFT 重采样使被丢弃的 1200 ms 端点样本漏进基线窗
证据：`preprocess.py:622-627` 先对整段闭端 epoch 重采样再 `data[..., :n_times]` 截断。把 250 Hz、351 点 epoch 的最后 1 个源样本 +1.0 V，前 26 个输出样本（基线窗）最大改变 **9.516e−4 V**，全局最大 0.717 V。
影响：端点若出现伪迹/下一刺激瞬态，会以 −60 dB 量级偏置该 trial 基线（1 mV 伪迹 → 约 1 µV）；对真实 ERP 可忽略，但文档 §2.2 L252 只写了“FFT 会把闭端样本传到最终保留点”，未写反向传播。
建议：文档改为“重采样核的支撑覆盖整个闭端 epoch，半开仅指输出索引”；极端端点伪迹可在 QC 里单独计权，而不是声称窗口严格半开。
验证：check2.py 的末样本扰动实验。

### F6（P2，已证实）heldout raw 起点没有滤波预热/guard，首批 test trial 仍吃 8.8 s 瞬态
证据：`preprocess.py:140-148` `heldout_start = max(stops[:maximum])`，heldout scope 从“第 Kmax 个 epoch 的末端”开始；随后 `execute_preprocessing` 在裁剪后的 raw 上独立做双向滤波（`preprocess.py:708`），起点 odd 延拓只覆盖段内样本，0.1 Hz 滤波器的 8.8 s 瞬态真实存在。文档 §2.1 L174-177 已量化 BI2014a 的该风险并说“时间授权正确不能替代信号保真检查”，但代码没有丢弃、预热或标注受影响的 trial。
影响：若 Kmax 校准块只有几十秒，前 ~6 个 test trial 处于边缘瞬态区；这些 trial 会计入固定 test 分母并影响指标，且无法从缓存中识别（receipt 不记录 epoch 距边界的时间）。
建议：在 `prefix_raw_scopes` 里把 `heldout_start` 前移 `ceil(padlen)` 样本（或记录 `edge_distance_s` 并让评估端排除）；至少把距边界 < padlen 的 epoch 标成 `edge_support` 诊断。
验证：读 `prefix_raw_scopes` + padlen=2201 实测；未跑真实数据（见“未能验证”）。

### F7（P2，已证实）README 的“默认因果合同”与代码默认零相位冲突
证据：README.md:91-92 “The default causal input contract is 128 Hz … forward IIR filtering”；`contract.py:77` `filter_phase="zero"`，`contract.py:96` `DEFAULT_P300_DATA_CONTRACT` 用于 `epochs.py:285` 的 `P300_PERFORMANCE_PREPROCESSING`；§2.1 也把 offline zero 与 online forward 分开。
影响：新读者/新脚本会按 README 认为默认是因果，从而把 `SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT` 与离线 LOSO 合同混用；这是可复现性风险而非数值错误。
建议：README 改为“默认离线合同是零相位；单被试时序协议使用 causal 合同”，或直接链接 §2.1 的两行合同名。
验证：读 README 与 contract.py。

### F8（P2，存疑）EA 的总体协方差包含 ERP 外积，文档未定义估计量
证据：`euclidean_alignment.py:199-205`：`covariance += einsum("nct,nut->cu", block, block)`，再除以 `n·T`；没有逐 trial 去均值，也没有减跨 trial 均值。`EuclideanAlignmentState.record` 写 `output_unit="dimensionless"`、`regularization_policy="eigenvalue_floor_times_empirical_trace_per_channel"`。
影响：若目标是“噪声协方差”，该估计混入 ERP 信号功率（P300 的 outer product），ERP 越强参考越偏；若目标是“含信号的二阶矩”（EA 原文常用 per-trial covariance），则当前实现是合理变体。文档 §1/§4 只提 EA 的 artifact/坐标语义，未给出公式，无法判定意图。
建议：在 `EuclideanAlignmentState` 文档或 doc 中明确“二阶矩 = E[x xᵀ]，含 ERP”，并给出与 per-trial 去均值版本的差异实验；否则跨数据集比较 EA 时口径不明。
验证：读代码；未做数值对比实验。

### F9（P2，已证实）恢复证书/状态界依赖的 modal 系统与 SOS 次序一致，但未覆盖 notch 后的 10 阶数值条件
证据：`recovery.py:23-30` 用同一 `design_continuous_sos`（含 notch）；`recovery.py:96-98` 用 `np.allclose(rtol=1e-12)` 校验记录中的 SOS；`finite_filter_operator.py:30` 只要求极点模 <1。
影响：0.1 Hz + 50 Hz notch 的 10 阶系统在 250 Hz 下最大极点模 0.999043（时间常数 4.18 s），modal 分解的条件数没有在证书里量化；跨平台重算只容许 1e-12 的特征值差（文档 L210），但 5 节 SOS 的极点聚集可能让该容差在非 x86/BLAS 差异下过紧或过松。未发现实际失败。
建议：在证书里记录 modal 矩阵条件数与极点最小间隔；对 10 阶系统做一次跨平台复算。
验证：check1.py 的极点统计；未做跨平台实验。

## 建议优先级（给 Lead）

1. **先修 F2**：在 `PreprocessingSpec`/receipt 里落盘 `actual_t0_s`，并让基线按实际 t0 取窗；否则 128 Hz 与 250 Hz 的“同一合同”在时间轴上差 3.125 ms，属于可复现性口径问题。
2. **再定 F1**：把 `padding_limit_samples` 写进文档与契约断言；legacy 缓存的边缘结论需要显式降级，避免新旧缓存混比。
3. **F6 作为评估守卫**：即使不重跑数据，也应在评估端输出每个 trial 的 `edge_distance_s`，让“首批 test trial 吃瞬态”成为可检查的诊断而不是文字声明。
4. **F3/F4/F5 是文档与 receipt 的小修**：EOG 滤波语义、抗混叠指标、重采样核支撑，三者都不改数值结果，但决定别人能否复现解释。
5. F7/F8 是表述与口径：README 默认相位、EA 估计量定义，建议在下一轮 doc 修订一起处理。

## 未能验证与存疑

- 未运行 end-to-end 缓存构建（禁止训练/长任务），因此 F6 的“首批 test trial 实际受污染数量”只有几何推断，没有真实 BrainSync 记录上的 epoch 级证据。
- 未在 512/256 Hz 真实数据上验证 `_transform_epoch_samples` 的 `round(N·128/f_src)` 与 `mne.filter.resample` 实际输出长度是否始终一致（只在 250 Hz 验证 351→180）。
- `bidirectional.py` 的双端状态界公式（文档 §2.1 L179）只读了接口，未逐项复算 `J F_N J O_N δzL + J O_N δzR`。
- `harmonic.py` 的“局部 0–30 Hz 保护 + 分段离线自适应”只在算子层验证了分段/窗长约束（`harmonic.py:159-179`），未追踪调用方是否把它用于冻结模型推理；文档 L36-37 已声明它属于离线自适应。
- 缺导/坏导在 `data.channel`/`qc` 的传播只做了浅查，未覆盖 `channel.py` 的坐标配准与 `domain.py` 的 mask 组合。
- 未审计 experiments/、formal/、doc 其余五份正文；未跑 ruff/pytest（时间预算）。

## 已检查范围

- doc/input_contract_math.zh.md 1-443（全文）
- doc/technical_structure.zh.md 的 15/34/94/96/163/193/253 行附近；README.md 84-101
- src/data/preprocess.py 1-160, 230-255, 380-550, 540-800, 800-1030
- src/data/epochs.py 55-290, 540-630
- src/data/contract.py 1-208
- src/data/ocular.py 157-266；euclidean_alignment.py 140-219；harmonic.py 1-194；finite_filter_operator.py 1-91；recovery.py 20-40；bids_eeg.py 199-225；n2p3net.py 1-280
- 实验：tmp/audit/scratch/check1.py（SOS/padlen/极点/baseline）、check2.py（sosfiltfilt/MNE 等价、notch、重采样泄漏）、check3.py（notch 宽度、通带、MNE epoch 几何、混叠）、check4.py（多采样率 epoch 网格）
