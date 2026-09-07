# BI 与 BNCI 跨库输入对齐审计

日期：2026-09-06。针对 cross_dataset_start_20260906 首轮的真实输入、源 checkpoint 和预测。
本次为只读诊断，没有训练权重、改写旧缓存/预测或删除被试。新增统计对照发生于观察结果后，
属于 development，不参与选择生产配方。

## 结论

未发现可以解释 chance AUC 的全局标签反转、epoch/标签行错位、通道排列错误或漏做单位换算。
但不能把输入链称为已经充分解决：BI 源统计被少数超大幅记录主导，本轮幅值 QC 关闭；
BI 的 timeline 采样点字段使用事件序号；实际采样栅格与声明 tmin 有亚毫秒偏差。
后两项没有证据能解释本轮整体迁移失败。物理显示到触发的跨库延迟尚未验证。

## 证据身份与范围

- 执行源码：`frozen/cross_dataset_start_20260906b.source-only.tar.gz`，SHA-256
  `d902c1663f261a6c4530a6661c72bc8c78697a3340001629e029e934107c1304`。
  当前 preprocess、BI/BNCI parser、domain 和 checkpoint 文件与该 manifest 逐字节相同。
- 云端 BI cache SHA-256：`3f81f9cf361035474a6a721d72b297a64b510fbc0bd90bdd26a164cf0b67d4e2`。
- 云端 BNCI cache SHA-256：`2efd14b447c8cc426d350c4030c4774a4aecf3be9187858294179997f9f65297`。
- 全量读取、attested 校验、标签/事件映射/时间递增和源统计复算：BI 61,011 + BNCI 33,600 行。
- 原始数据逐点重建：BI01 1,188 行、BNCI A01 4,200 行；不是全部原始记录重建。
- 原路径 CPU zero-shot 逐值复现：BI 01-08、BNCI A01-A08，共 16 人，最大 score 差 0。
- 固定 K192 前缀统计对照：全部 72 人，保留原测试行，无权重/BN 更新。

## 基础对齐检查

| 检查 | 实测或代码依据 | 结果与限制 |
|---|---|---|
| 二分类语义 | 两库 raw label 1=non-target、2=target；cache y=(raw label==2) | 全 94,611 行相符；BI 还与 60-65/80-85 flash code 交叉核对 |
| EEG 与标签行 | event evidence 索引排序后严格等于 0..N-1；parser 同步选择标签和 metadata | 全量双射成立；原始记录抽检进一步检验真实采样点 |
| 事件时间 | BI metadata.flash_sample/512；BNCI (MATLAB index-1)/256 | 与 cache onset_times_s 全量一致，每被试严格递增 |
| 通道 | CZ/P3/PZ/P4/OZ，以名称选取 | 两库一致；原始重放按原 EEG 通道顺序选取后吻合 |
| 参考 | native:right earlobe，未做 CAR | 声明一致，属于原始数据参考来源记录，不是物理参考等效性测量 |
| 单位 | 原始 microvolt 乘 1e-6 后进入 MNE/cache | 两库重放一致，无漏乘/重复乘的证据 |
| 滤波与窗口 | 原采样率连续四阶 Butterworth 0.1-30 Hz forward，steady-state 初态；再 epoch/FFT resample | 同一合同；最终 128 Hz、179 点、mean-only 基线 |
| 源统计 | 用实际全部 source 行重算 mean、std+1e-6 | 与两个 checkpoint 相符；来源正确不等于统计对离群值稳健 |
| 测试划分 | 使用 onset_times_s 与 evidence_available_times_s，而非 onset_samples | 抽检预测文件 calibration/test 的 epoch 支持不重叠；IIR 保留过去状态 |
| 推理符号 | target logit 减 non-target logit | 保存分数逐值复现，不存在读取错误输出列或 AUC 标签不一致 |

原始重放：BI01 最大误差 0 V，BNCI A01 最大误差 1.8189894e-12 V。
BI CSV 与原 MAT 的 EEG 差异符合 CSV 约五位有效数字的精度，raw 标签完全一致；
CSV 时间列被量化，不能用相邻相同时间值当丢采样证据。当前取行采样点/512 的方式避免了该量化。

## 发现的问题

### 1. 源尺度受到极端幅值主导

两个 checkpoint 均为 `qc_ptp_uv=0`、`qc_dropped_source_epochs=0`。缓存保留原始幅值，
源训练使用 pooled mean/std，未在这次配方中启用幅值 QC 或稳健尺度估计。

checkpoint 输入分母为逐通道标准差加 1 microvolt：

| 通道 | BI 分母 (microvolt) | BNCI 分母 (microvolt) |
|---|---:|---:|
| CZ | 684.71 | 10.73 |
| P3 | 150.93 | 10.30 |
| PZ | 629.72 | 10.24 |
| P4 | 250.24 | 9.95 |
| OZ | 465.07 | 11.21 |

BI63 仅 420 行，占 BI source 约 0.69%，却贡献了 CZ 86.21%、PZ 87.54%、OZ 67.00%
的缓存平方能量。其 CZ/PZ 标准差为 7,651/7,090 microvolt；epoch 全通道 RMS 最大值
47,693 microvolt。这里的贡献是平方能量比例，不冒称严格的中心化方差贡献率。
BI44 另贡献 P4 45.09% 的平方能量。

这不是已经证实的单位换算 bug，也没有证明全部大幅信号的伪迹成因；它明确显示当前
source 输入尺度高度受少量大幅记录影响。跨库输入相对源尺度可能被强烈压缩或放大，
冻结源 BN 的统计适用性也需要核验。不能仅扩大模型并把全部失败归因于容量。
不能按新模型成绩删 62-64；需独立信号诊断，预声明 source-only QC/尺度对照并保留全请求队列。

### 2. BI onset_samples 字段语义不一致

`data.bi2014a_candidate.build_bi2014a_subject_dataset` 使用 `observed_only_timeline`，
后者将 `onset_samples` 设为 `arange(n_events)`。BI 61,011 行该字段均不等于真实采样点；
真实值仍在 `metadata.flash_sample`，而 BNCI 的 `onset_samples` 是真实采样点。

这是元数据接口风险，消费者不能对 BI 的该字段直接除以 512 来恢复秒。
本轮 raw epoch 发生在 timeline 构建之前，使用真实 flash sample；binary_time_split 使用
真实 onset_times_s，因此该问题没有造成已检查 epoch 或本轮切分错位。正式修复应统一
接口并更新缓存身份，不能改写旧实验 cache 冒充旧结果的原始输入。

### 3. 声明时间栅格有小偏差，物理触发延迟仍未验证

MNE 在 512/256 Hz 都把 -200 ms 的 epoch 起点取到 -199.21875 ms；代码保留声明
tmin=-200 ms，相差 0.78125 ms。两个源率下 resample 前包含 717/359 点，resample 后
为 179/180 点，后者再裁为 179。共同点数合同存在，但这不是连续时间端点严格相同的证明。
需要明确采样栅格语义；目前没有证据表明这种量级的偏差足以导致 chance AUC。

代码没有通过 photodiode 等测量证明两库 event marker 到实际显示 onset 的延迟一致，
也未建立跨库硬件延迟补偿。标签/事件索引对齐不等于 P300 生理潜伏期或真实视觉 onset 对齐。
不能用目标 test 标签搜索最佳时移后将分数当作无偏迁移结果。

## 无训练的尺度诊断

保持同一源模型、原测试行和源 BN，只把输入 mean/std 改为本人前 192 条 EEG 的统计。
不读取这些行的标签来估计尺度，不训练参数；由于使用目标数据统计，它不是严格 zero-shot。

| 方向 | 原 zero-shot | 仅前缀统计 |
|---|---:|---:|
| BI -> BNCI | 0.511738 | 0.520733 |
| BNCI -> BI | 0.510757 | 0.515343 |

简单替换输入尺度没有恢复迁移；不能据此否定 source 学习阶段的大幅信号影响，
也不能声称标准化是唯一根因。应先完成 source 信号质量/统计与事件物理延迟的核验，
再将已冻结处理配方与容量/训练预算进行匹配比较。

## 复核产物

全部文件位于 `experiments/runs/input_alignment_audit_20260906/`：
`audit_cache.py`/`cache_audit.json`、`replay_raw.py`/`raw_replay.json`、
`audit_scale.py`/`scale_audit.json`，以及两份 cache 切片和 72 份诊断预测。
脚本使用运行快照与同一云端环境；本地 raw replay 使用项目 .venv。

现有相关测试：`python -m pytest tests/test_preprocess.py tests/test_bi2014a_candidate.py
tests/test_bnci2014_008_candidate.py -q`，46 项通过。测试包含原始 A01 解析探查；
测试通过不替代本次真实缓存和原始数据重放。
