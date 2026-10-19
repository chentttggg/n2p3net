# BrainSync 时钟与连续质量实施

日期：2026-09-07。实施依据为[已批准方案](brainsync_signal_quality_design_20260907.zh.md)，
前置诊断和文献仍以该页及其来源为准。本页区分软件合同、旧数据开发和待验证的工作条件。

## 已实施

| 边界 | 当前行为 |
|---|---|
| 样本账本 | 保存原始 seq/delta、接收时间、拒绝原因、连续段与存储索引；回绕、重复、缺口、未知计数分开记录 |
| 落盘 | 电压与时钟日志双流 flush/fsync 后提交共同边界；失败回滚，恢复只读取共同提交的数据 |
| 结束/暂停 | 异步收尾后再次检查完整性；末刺激响应期暂停会记录失败 attempt 并重试；尾段必须属于同一采集段 |
| EDF 尾部 | `acquired_samples` 与 `edf_samples` 分开，`padding_samples` 不算真实 EEG；后端排除格式填充支持 |
| 排程 | 共享 `epoch_support.json`，休息后保证基线、休息前和结束前保证响应，并等待实际记录样本 |
| 原始 QC | 显式校准 mask 拟合逐通道差分 median/MAD；冻结阈值、噪声下界、确认长度、人员、参考、样本和文件摘要 |
| 滤波 | 因果与零相位路径消费同一组完整排除区间；清洁段分别滤波，普通休息和恢复注释不再次重置状态 |
| 恢复 | 未知初态误差保持不可用；可选 SOS 状态误差盒证书检验整个后续模态上界，并核对实际 epoch 算子增益、参考和 EOG 摘要 |
| HEOG | 用户明确为左眼眼角单端 HEOG；REF现已确认为左耳耳垂耳夹。原执行配置误记`T7(T3)`，2026-09-08按证言更正；原快照保留。EOG禁止进入分类 EEG/CAR，不假定完整眼动可观测 |
| EOG 拟合 | 显式有效样本 mask；改变测量参考前必须声明共享参考；保存系数、中心、接法及上游 QC；递归核查校准预算 |
| 个人 epoch QC | 真正 session/block `fit_groups` 与 participant 分开；inner train 拟合、冻结用于验证，最终完整 calibration 重新拟合 |
| 保存/推理 | 个体模型 schema `/3`、EOG `/3`；QC 状态与摘要随模型保存，重载验证；新记录只应用冻结状态 |
| 决策 | 不可用行保留计划事件。局部缺导与整 epoch 失败分开；R5 任意候选不足五次均拒判，整份不可用仍保留请求分母 |

没有改动模型窗口、频带或网络：仍为 `[-200,1200) ms`、128 Hz、179 点、
0.1-30 Hz 四阶前向 Butterworth、`full_unfold+K35`。250-600 ms 是可选 LMBC 读出窗口，
不是当前卷积扫描范围。历史分段一致性及早峰解释见[时轴审计](brainsync_epoch_timing_audit_20260907.zh.md)。

## 时间与因果边界

新时钟等级是 `delivery_anchored`，未知固定传输偏移保持 `null`，不能声称真实亮屏已对齐。
接收回调减去批内时长给出首样本估计；不是 ADC 时间实测，也不提供历史固定偏移修正。
后端核验 manifest、最终 clock sidecar、journal SHA、批次总数、连续段和实际 EDF 样本数。
路径提示不是数据身份；只打开会话内 manifest 指定的文件，允许迁移后的旧路径提示失效。
新会话另记录源码文件 SHA 与 SDK 运行版本、安装版本和本地二进制 SHA；不是设备固件测量。

带连续 QC 或新时钟的入口采用完整 recording 批处理的 evidence availability。
这不冒称逐 SDK chunk 在线推理。原始异常确认、收包和会话就绪依赖不能省略为 onset+tmax。
正常休息保留原始数据和滤波状态；用户暂停、缺口和污染隔离属于另一类边界。

## 恢复证书与限制

当前检测器识别突变差分，不宣称覆盖缓慢漂移、所有平台、眨眼或设备饱和。
EDF 自动量程极值不能作为 ADC 饱和证据。异常采用全 epoch 保守排除，但仍保存实际触发通道。
首段保持原合同的首样本稳态初值；异常后或显式分段后的未知状态不得自动宣称恢复。

有限恢复依赖明确的初态误差盒，而不是从测试记录振幅猜一个上界。
模态分解病态或不稳定时不生成有限恢复点。epoch 增益通过实际 MNE 重采样、截取和均值基线
的线性基计算；证书小于实际增益、SOS/参考/EOG 不一致时拒绝。
这些是给定假设下的软件保证，真实误差盒、ERP 失真容限与误报上限尚待独立开发验证。

Q4 若同时使用 EOG 与有限恢复证书，先冻结 EOG，再提供绑定该 EOG 摘要的证书；
不得让待拟合的 EOG 与待生成的恢复证书相互定义。
自动 EOG 仍使用合法 inner-training 会话拟合后保持冻结，不把它重标为完整 calibration 重拟合。
局部 epoch QC 则在最终 calibration 重新拟合，两个生命周期分别写入记录。
分组不足时记录无内部阈值选择证据，不能把 fallback 当作已优化结果。

## 验证证据

- 实际前端控制器的 10 block × 45 刺激排程，经后端连续滤波、基线与分段，得到 **450 个完整 epoch，每 block 45 个**。
- 真实前端 EDF+BIDS 导出覆盖 250/251/500/501 个采集样本，验证半秒尾部、精确填充数量、路径迁移和错误元数据拒绝。
- 故障注入覆盖双流部分写入、共同提交失败、恢复日志截断、收尾期间 acquisition fault、最后响应期暂停重试。
- 回归覆盖校准以外电压改写不改变拟合、错误人员/参考/未来上游 QC 拒绝、静态缺导 JSON 往返、全局污染拒判、整份 R5 失败分母与模型重载。
- 最终后端全量 **1,254 通过、4 跳过、0 失败**；起止源码 SHA 完全一致。前端全量 **159 通过**。
- 相关后端 Ruff、前端新模块 Ruff/关键错误检查、`git diff --check`和两个独立复核均通过；Qt 新控件截图已检查。

验证原件位于 `experiments/runs/brainsync_quality_implementation_20260907/`：
`backend_full_final.xml`、`backend_verification.json`、`development_replay.json`。
开始实施前的源码为 `frozen/brainsync_quality_implementation_before_20260907/`。
本次最终代码和验证清单见 `frozen/brainsync_quality_implementation_20260907/`。
这些文件与软件测试不构成实机时钟、成人效能或独立样本外确认。

## 第三、第五份开发重放

使用第二份保留记录的全段稳健差分尺度作 **未认证清洁的开发探针**，报告全部预声明
z=8/12/20，不按标签、ERP 外观或成绩选择最优门限；确认长度 25 个原始点，噪声下界
0.1 uV，均不是生产默认。原始文件未改写，未拟合新的分类器。

| 记录 | z=8/12/20 原始异常区间数 | 首个检测样本 | 解释 |
|---|---|---|---|
| 第三份 0903_173947 | 128/100/59 | 82，即 0.328 s | 再次定位到已知启动阶跃 |
| 第五份 0907_150258 | 122/66/29 | 713，即 2.852 s | 多个局部瞬态；不能归结为删除最后 block |

未提供真实恢复误差证书时，两份在该保守探针下都没有获得可认证的完整刺激支持。
这表明当前不能自动重新准入，不证明整份不可恢复，也不把所有触发都认证为伪迹。
第三、第五份仍保留作开发；独立确认须使用冻结配方后的新采集。

第一份 `0903_170721` 及包含它的旧缓存/个体模型维持已拒绝、物理归档和活动入口撤回。
此前永久删除被自动审批策略拒绝，未换方法重试，也不称已永久删除。

## 使用入口

工作站校准页提供“校准内局部 QC”与“连续 QC 模型”选项，默认不激活未经确认的新阈值。
命令行：

```powershell
.venv/Scripts/python.exe experiments/fit_brainsync_continuous_qc.py --calibration-manifest <explicit-clean-intervals.json> --output <new-qc.json>
.venv/Scripts/python.exe experiments/prepare_brainsync_cache.py --session-dir <session> --continuous-qc-model <new-qc.json> --output <new-cache.npz>
.venv/Scripts/python.exe experiments/brainsync_model_task.py --task calibrate --session-dir <calibration-session> --continuous-qc-model <new-qc.json> --epoch-qc --model-artifact <new-personal.pt> --output-dir <new-run>
```

校准 manifest schema 为 `brainsync_continuous_qc_calibration/1`，包含 `spec`
（difference_z、difference_floor_v、settle_samples）和 `sessions`；每项明确 session_dir、
clean_intervals_samples（原生半开样本区间）及 clean_interval_evidence。可选 recovery 为
SOS、state_error_bound_v、tolerance_v、epoch_operator_gain、evidence、input_reference、ocular_model_digest。
跨会话校准须用授权 inner-training 范围内的 raw QC；未来/验证会话拟合的 QC 会被拒绝。
新推理只加载保存包内的处理状态，不从测试会话重新拟合。旧个体模型与缓存不重标新版。
