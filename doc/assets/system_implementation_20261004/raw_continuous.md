# BNCI 真实连续输入纵向切片与完整诊断（2026-10-04）

新增 `src/transfer/raw_continuous.py` 和 `experiments/run_bnci_continuous_study.py`，接通原始 MAT → 同字节身份验证 → 明确 C/T raw 授权 → 连续算子 → 全部 K 标签拟合 → 冻结模型 → 无真标签的 36 候选评分 → requested outcome/概率/覆盖/延迟账本。没有把连续数据伪装成 `EpochDataset`，也没有切换产品默认。

## 固定设计与实际完成

- 执行前计划：`doc/assets/system_implementation_20261004/bnci_continuous_plan.json`。8 名 BNCI2014-008 参与者 A01–A08，全 K200/K800，未因先导成绩停跑或选人。
- 原件：`mne_data/MNE-bnci-data/~bci/database/008-2014/A0*.mat`。计划逐文件 SHA256，reader 从同一被核验 bytes 解码；不先验证文件再另开路径读取。
- 新输入路线固定：原生八导、原右耳参考、256 Hz、V；共享 0.1–30 Hz Butterworth4 + 50 Hz/Q30、有限区间双向反射滤波；不逐 epoch baseline、不 resample。模型 basis8、响应1.2 s、ridge1、preroll0.2 s；所有参数在评分前固定。
- 每人串行、本地 native/PyTorch 2 线程；当前 host 可用内存有限，不累计八人的原始数组，不构造 N×N 协方差。
- 最终运行：`experiments/runs/system_implementation_20261004/bnci_continuous_real_v2/result.json`。
- 16/16 cell 完成，0 拟合失败；464 个 requested decision、448 个可评分、16 个不完整全部留在分母。每人每 K 的实际分割产生29个请求（28完整+1部分），由真实 schedule 推导，不是硬编码数字。
- 最终执行35个实际已加载源码文件保存在该运行的 `executed_source/`；`execution_source.json`摘要 **a1b9d41e086d8d2c2a8dcb9de80c75aaab9a3b26b0d9ef6dda6c0d6c7475cfa6**。开始/各参与者/结束验证这些 Implementation 未变，不把修改前全仓快照冒充执行源码。

| K | 8人平均 requested hit@all | 8人平均 coverage | 条件于可评分决策的平均 NLL |
|---:|---:|---:|---:|
| 200 | 0.4051724 | 0.9655172 | 6.9018254 |
| 800 | 0.5905172 | 0.9655172 | 3.2426752 |

这是新连续输入路线的公开开发诊断，不是相对旧epoch模型的配对算法收益。输入参考/通道、采样、无baseline及模型对象同时变化，不能把差值归给单一新算子。没有独立成人或设备性能证据。K200的NLL比均匀36候选 `log(36)=3.5835` 更差；A02 K200 NLL=33.1674，显示小校准下概率严重过置信，不能因hit高于随机就宣传概率可靠。

## 全部 K 标签与时间合同

复用了 `prefix_raw_scopes`/`source_epoch_support` 与 `resolve_calibration_splits`：原epoch授权仍为[-200,1200]ms的原采样闭区间，固定测试起点只由全矩阵实际 Kmax 决定，K200和K800共享测试范围与purge。测试不用当前 K 单独移动；原始事件和上下游预算账本均保存。

本次落地时发现，K200/K800都截在一次selection中间。原授权末个flash+1.2s会包含后续未知flash；若忽略它们就误设连续响应历史，若读取它们的目标标签又超K。因此实际连续C截于 `min(original_authorized_stop, next_unauthorized_flash_onset)`，保存全部K labels、全部已知onsets，对越右界的响应核显式删失，而非只取完整决策并丢掉多余80个标签。`CalibrationRecord.right_censored=True` 由连续模型owner实现，预测仍要求完整响应尾部。

A01 K200具体账本：original authorized stop=16892，next unauthorized onset=16648，实际 C=[0,16648)，保留200labels；原始事件196–199的核右删失。对其后所有raw值加巨大干扰、翻转所有后续label，C实际滤波数组、labels、拟合kernel_mean、innovation covariance逐值不变。该反例确认不读取下一flash的样本/标签；不是仅检查名为“calibration”的字符串。

响应模型长度采用 `round(1.2×256)=307` 个样本，stop为exclusive；原epoch授权的闭端采用 `round(1.2×256)+1=308`，两者在scope ledger明确保存。模型末尾比授权上界保守1sample，没有off-by-one越权。

## 决策角色与滤波限制

每个 hypothesis q 的membership仅由物理 row/column flash code构造，36个候选词表固定；prediction view没有target truth。完整targets只在scoring_truth中交给独立评分者；score方法的参数不包含真目标。每个可评分decision仅使用自己的disjoint连续样本区间，每个样本在该decision似然中计一次。

所有参与者实际检查相邻selection gap（当前最小8.25 s）；如果小于声明响应支持+preroll则保留decision并拒绝评分，绝不借邻居真label做context。测试还人为缩短实际gap，确认返回明确incomplete原因。

**大gap不证明双向IIR后的统计独立。** 共享滤波在整个独立T作用，长尾与acausal response仍可能跨selection；连续模型使用有限因果核、AR1×空间协方差，是可检验的近似。scope ledger明确 `scope_filtering_makes_decisions_statistically_independent=False`、`filter_state_uncertainty_integrated=False`。这条路线没有解决R1的真实Γ/Ω可辨识性，不能据此声称已实现全部算子不确定性传播。

还分别保存两个时延：有限核等待约30.95 s，而实际双向整段test filtering必须等到test scope结束，平均实际availability latency **576.7306 s**。该结果是离线诊断，不能当成实时BCI。未来要减少延迟必须另声明逐decision/因果算子并重新拟合，不能只改报告的latency数字。

## 结果完整性与验收

每个模型保存重载；448个可评分decision的全部36 log_evidence逐值相同，winner一致。最终记录保留 raw log evidence、probability、scored-input SHA、NLL、requested outcome、coverage以及可用时刻；model和prediction companion bytes被seal绑定。

未评分decision的probability/log_evidence/NLL保持null；主要NLL单列可评分数量与missing数量。另有明确命名的“requested NLL with uniform abstention”作为报告策略，为弃权分配log36损失；它不是伪造的模型posterior，也没有用0掩盖失败。

独立复算脚本 `doc/assets/system_implementation_20261004/verify_bnci_continuous.py` 验证所有record封印与companion bytes、35源码快照、scope digest和C/T隔离，从原始log_evidence重新计算softmax/NLL/argmax/status和hit/coverage。结果：`bnci_continuous_verification.json`，全部通过。

v1原件保留。v2仅增加完整log_evidence/scored-input字段与cell evidence_scope后按相同计划全跑，v1/v2汇总值完全相同；不是数据驱动的参数搜索。v1 cell执行总137.14s。

软件测试 `tests/transfer/test_raw_continuous.py` 五项通过，覆盖原MAT结构→真实parser/生产滤波、全部K labels和固定test、未来样本/labels干扰隔离、membership全词表、实际gap反例、partial分母、无label评分、模型重载与真实offline latency，以及raw准入失败后终态和请求账本。共享MAT造数位于 `tests/support/bnci.py`，parser测试与连续测试共用，不互导测试文件。Ruff、git diff --check通过。测试容器里的信号为合成，只有上述8人矩阵属于真实公开数据开发证据。

v2结束后只补充runner的异常收尾：捕获运行失败将status从running关闭为failed，冻结请求cell网格并为尚未执行项记录原因；已有目录被拒时不覆盖旧结果。该更改经过专门失败反例验收，不影响fit/score数学；v2继续绑定它当时的实际执行快照，不谎称跑过此后源码。
