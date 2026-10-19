# 深度全链审计工作账本

请求：核对全部历史无迁移LOSO、同人scratch/LDA、旧四库迁移、P1/P2及统一矩阵之间的数据和成绩差异，寻找超出已有测试的隐藏问题。

本轮生产代码和原始数据只读。入口时417份源码、配置、活动指南已归档并逐成员核验，身份见entry_manifest.json；截至首轮检查均未改变。新增内容仅为独立审计脚本、JSON和报告。

| 环节 | 独立证据与当前状态 |
|---|---|
| 原始数据/预处理 | Huebner真实BDF+events逐项匹配；校准后raw扰动隔离；整记录双向vs独立前缀出现真实算子差。BI/BNCI和新source适配继续核验。 |
| 缺导/标准化 | 三套masked统计对实测子集一致；污染缺导槽不改变统计。没有复现“统一主链未pack实测BN”的旧猜测。 |
| EA旁路 | 真实历史state的1e-6 V² floor远大于信号协方差，多数白化被压制；先EA后人员排除的state保留原fit人数。消费者与当前矩阵影响范围继续追踪。 |
| 模型/训练 | 独立source验证损失归约可随batch改变并反转候选排序；unified固定更新源验证另走正确可加路径。scratch1279架构/seed符合声明。非默认LMBC checkpoint正确重建，推翻旧疑点。 |
| 适配/预算 | P1实际固定200 updates，非plan中60 epochs；unified90 epochs、calibration冻结BN/dropout off，与旧online BN不同。minimum_test放宽及无调整日志已出现真路径反例。 |
| 原预测/指标 | 旧四库3862、P1/P2 32238、09-16统一12150 NPZ独立复算通过；同run匹配行/标签未见错误。09-18完整原产物本地缺失，不冒称本轮重放。 |
| 历史可比性 | 69/123 zero-shot测试队列漂移；相同source、共同测试行预测几乎相同。历史LOSO的数据人数、通道、频带、窗、QC/预算不同。 |
| 决策/统计 | 缺失candidate两口径、未来重复中心化影响过去前缀、valid样本跨replicate伪配对、domain_lodo未初始化计数等反例已触发；历史实际消费范围继续核验。 |
| 复用/身份 | 生产execute_target_tasks无拟合反例证实compact跨运行import可把旧software_correctness与旧snapshot重标成development与新snapshot；未证明实际十二库发生。 |

约束：不把API可触发漏洞等同真实矩阵污染；不把配置设计差异等同代码错误；不把来源不可得等同结果错误；不把已有报告的“已验证”当本轮无需复核的结论。

最终状态：调查完成，生产修复与新训练未执行。BI/BNCI及其他真实切片已核；BI2014a全61人C200 zero/forward和实际test/full-context对照完成，失败0；11个个人包/重放实例完成。85,818份归档预测条目复算/比对，其中48,250份由独立评分公式核验；分支交叉复核不重复累计。

最终结论、适用范围和下一步实施次序统一见`doc/evidence/deep_chain_audit_20260929.zh.md`。已有53项合同测试通过，但新增反例揭示其覆盖不足；全部审计脚本Ruff与diff检查通过。后续必须先处理有限raw输入保真与比较身份，再解释窗口或A1/Q1效果。

大型逐cell、逐epoch、缓存与roundtrip包只留在Git忽略目录`experiments/runs/deep_chain_audit_20260929/`；原路径通过移动清单追溯，原始研究产物未修改。
