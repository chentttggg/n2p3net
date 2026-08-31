# P300 模型输入合同：离散时间、单位与分组

## 1. 证据边界

设源记录的采样率为 `f_src`，第 `c` 个物理 EEG 通道为
`x_c[n] in V`，第 `i` 个刺激在源记录中的 sample index 为 `e_i`。
`x_c[n]`、`f_src` 和 `e_i` 属于采集证据，不因模型输入采样率改变。

成人 offline 软件默认张量为：

```text
f_model = 128 Hz
passband = 2..30 Hz
epoch = [-200, 800) ms
T = floor((800 - (-200)) * 128 / 1000) = 128
unit = V
baseline = per-trial, per-channel mean over [-200, 0) ms
```

这不是所有 cohort 的准确率最优硬合同。GTN steady-state 2x2 已比较
`l_freq in {0.1,0.5}` 与 `tmax in {800,1200}`，development winner 冻结为
0.1 Hz/1200 ms；对应 `T` 由同一公式推导。
online causal profile 另声明 `filter_phase=forward` 和
`causal_iir_initial_state=steady_state_first_sample`。

## 2. 可执行算子

### 2.1 连续域滤波

令 `B4` 为所选通带的四阶 Butterworth IIR。offline `phase=zero`
执行前向/反向滤波，因此

```text
x_bar_c = filtfilt(B4, x_c)
|H_zero(f)| = |H_B4(f)|^2
```

它是离线、非因果算子。不能把 `e_i + 800 ms` 解释为在线系统真正可用
证据的时刻。

online `phase=forward` 使用单向 SOS：

```text
z0 = sosfilt_zi(B4) * x_c[0]
x_bar_c, z_end = sosfilt(B4, x_c, zi=z0)
```

不能使用全零初态。真实 GTN 通道含 mV 级 offset，零初态会把前期 trial 制造成
mV 级 startup transient，而 later suffix 已稳定，形成假的时序域移。若处理真实
stream chunk，后续 chunk 必须携带上一 chunk 的 `z_end`；首文件才使用首样本稳态。

### 2.2 源事件切片与重采样

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

把采样率从 `f0` 改为 `f1` 时，保持跨度的奇数核满足

```text
K1 = 1 + round_even((K0 - 1) * f1 / f0).
```

所以严格的 256 Hz 对照是 `129/9/33`。`127/9/33` 的 ST 跨度为
`492.1875 ms`，不是纯采样率对照；若保留，必须标成额外的感受野扰动。

## 4. 参考电极与跨数据集

若真实头皮电势为 `s_c(t)`，参考电极为 `r(t)`，记录值是

```text
x_c(t) = s_c(t) - r(t).
```

两个数据集使用不同 `r(t)` 时，即使 `s_c(t)` 相同，输入也不同。任意空间
权重和不为零的分类器都会响应这一公共偏移。因此 source reference 必须记录；
不同或未知参考的数据不能直接拼接。共同重参考必须是独立、显式、可消融的
变换，不能只修改 provenance 字符串。

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
  -> MOABB | BrainSync | GTN | raw manifest adapter
  -> preprocess: continuous IIR + declared phase/state -> source epoch -> FFT resample -> mean baseline
  -> EpochDataset.validate + QC features + SHA-256 attestation
  -> run_eeg_loso: physical contract + source provenance fail-closed
  -> train.factory -> N2P3NetBaseline
  -> N2P3Net architecture record (samples + physical milliseconds)
  -> source/calibration-only QC -> true-time split/embargo or fixed budget -> held-out logits
  -> positive-slope LLR -> tie-aware candidate aggregation + coverage/cost ledger
```

## 7. 尚未解决的边界

- BI2014a raw CSV candidate path 可恢复多 character decisions；旧 cache 的 repetition
  index 写错。当前 raw source 缺失，待恢复后才能构建 causal-v2；合法指标按 early
  known decisions -> later unknown decisions，当前没有 cross-decision 性能结果。
- GTN 的参考电极并非逐文件完整记录，跨数据集前需要独立重参考方案。
- GTN candidate occurrence 异步；不能把相同 occurrence index 当同步 block。
- 128 Hz 与 256 Hz 的比较必须共享源事件、带宽、基线、QC、fold、seed 和
  校准；只允许采样率及其跨度匹配核发生变化。
