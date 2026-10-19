# 第三阶段修复臂独立复审

第三阶段原运行的`raw_ar32_lattice_marginal3`在旧快照下有16/48个`Matrix is not positive definite`失败；这组结果已保留在[旧分析](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/analysis_stage3/report.zh.md)中，不能被覆盖。

随后使用QR/来源门禁修订后的源码建立独立修复运行`calibration_groups_stage3_repair`。修复运行的`executed_source`摘要为当前修订版本，768旧请求之外只重新执行该48个失败臂；8人×6折共48/48完成，sealed prediction/model/source均独立重放一致。

本报告严格绑定repair运行的`executed_source`，不自动代表之后继续编辑的工作树字节；最终源快照若再变化，需重新核对身份或重放。

修复臂结果：hit=26/48（0.5417），NLL=4.2172，coverage=1.0，held ACF1均值=0.4268，held/train方差比中位数=0.9887，预测二次型/投影维度中位数=0.9834。最大候选概率均值0.7671，中位数0.9221；最坏NLL=28.653，仍是近似后验，不能称概率校准完成。

旧快照该臂的32个成功请求为hit=21/32（0.4375）、NLL=5.2915，另16个失败；修复后新增16个请求才使完整48请求可报告。由于源码已改变，修复结果不能与旧32个成功值作为同一代码的逐值配对收益；旧失败分母与新完整结果必须同时保留。

修复运行的48个模型factor condition约`2.51e7–7.18e7`，机器精度放大指标`5.57e−9–1.59e−8`，低于预声明`1e−6`警戒线，因此precision warning计数为0。高condition仍记录为数值敏感性事实，不能把“未过警戒线”写成稳健性证明。

独立重放校验了每一折五个训练selection、600训练标签、120留出事件、候选词表、truth scorer-only分离、模型加载和sealed prediction SHA；没有使用测试标签回拟合。旧第三阶段的752成功+16失败属于旧执行快照，修复运行属于新执行快照，二者不合并成一次768成功实验。

当前临时红队回归：`tests/transfer/test_temporal_noise.py`与`tests/transfer/test_continuous_decision.py`共27项通过；唯一警告来自故意构造病态平稳协方差的拒绝测试。覆盖了重签空ledger、篡改startup因子、错误PARCOR/完整性和连续模型重签输入等门禁。
