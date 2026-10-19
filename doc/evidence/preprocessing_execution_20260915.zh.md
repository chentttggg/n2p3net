# 预处理合同执行：实施与验收

2026-09-15。承接[科研裁决](preprocessing_contract_research_20260915.zh.md)，用户已授权全部实施。
实现、正式反例与六份指南同步完成；最终测试结果、退出码和固定源码摘要集中记录在
[validation.json](../assets/preprocessing_execution_20260915/validation.json)。
本记录仅覆盖软件与合成反例，没有启动云训练、重建真实缓存或修改既有模型成绩。

## 实施结果

`data.preprocess.execute_preprocessing(raw, events, preprocessing, **layout_options)`
先校验完整Spec、源Nyquist、源闭端/目标点数和实际基线网格，再执行数值流程。
layout参数不能覆盖数值合同。显式JSON通过`parse_preprocessing_spec/load_preprocessing_spec`
拒绝空配置和缺字段；完整规范record保留历史无notch/legacy边界的省略约定。
低层`preprocess(n_times=None)`保留研究能力，但不自动产生严格执行凭证。

七处数值转写全部迁移：BrainSync、MOABB raw、GTN、BI2014a candidate、BNCI2014-008
candidate、manifest及新增source_records。`build_subject/load_dataset`提供显式Spec参数，
GTN cohort bridge复用共享contract转换。来源身份、单位转换、候选解释、参考、拟合QC及
采集时钟仍由各入口持有。MOABB高层P300保持原独立实现与不支持notch/新edge的门禁，
不会获得raw执行器凭证。

数值核保持连续滤波→源闭端epoch→FFT→前T点→半开基线→float32。
`transform_epoch_samples`同时服务生产和条件恢复证明；`recovery.epoch_linear_projection`
集中线性矩阵构造，LRU至多4项，basis分块至多8 MiB，矩阵只读。
双端状态耦合界保留，none/mean_only属于线性；median trial_reference可计算但不准入线性证明。

## 实测缺口与修正

| 边界 | 原行为 | 现在的行为及验证 |
|---|---|---|
| 250Hz、tmax=1199ms | 事件4s的FFT输入消费5.200s样本，availability却5.199s | 使用`max(onset+tmax, source_end_sample/source_rate)`；注入末样本确实改变全部179点，正反例保留 |
| 非零first_samp | 绝对事件索引必须与数组偏移区分 | 所有排程记录绝对起止源sample；物理相同录制输出逐位等价 |
| source=127.999Hz、model=128Hz | 生产np.isclose跳过重采样，旧恢复按精确不等执行FFT | 两者共用实际数值核；常见128/250/512Hz路径继续独立MNE逐位对照 |
| legacy zero且无notch | MNE按全局annotation分段，不能照搬逐导实现的分段记录 | 凭证记录实际全局支持；两导相同波形、仅单导标注的反例验证 |
| 模块加载后磁盘变化 | 若执行结束才读源码摘要，会误记新磁盘文件 | 两个模块加载时固定源码摘要；临时副本动态导入后改磁盘的回归验证 |
| 自相矛盾凭证 | 只检查摘要不足以核对物理含义 | 验证schema、完整声明、源网格、输出几何、分段、版本/源码摘要、SOS存在性、后端和padding必要条件 |

完整记录双向availability继续取更保守的recording结束，BrainSync额外QC/交付只能提高该界。
新时间账本可能影响非网格对齐配置的purge/划分，这不是可以回写旧结果的纯命名变化。
本轮没有修改任何旧cache/结果；重建时必须保存新派生身份。

## 执行证据、缓存与模型门禁

声明`PreprocessingSpec.record()`保持历史哈希含义；新的
`provenance.preprocessing_execution`采用`n2p3_preprocessing_execution/1`，包含实际SOS、
padding上界、每导分段、源率/first_samp/长度、epoch网格、输出/事件映射摘要、后端版本与
模块加载时源码摘要。该摘要绑定加载时文件，不代替外部完整运行快照或原始数据字节证明。
凭证摘要用于一致性核验，不是签名，也不能仅凭版本字符串证明任意版本数值等价。

cache保存/加载与Dataset.validate核对凭证；concat按独立记录聚合并保留父谱系。
混合无凭证历史输入以`unverified_inputs`记账，不能声称全部来自已验证执行器。
通道选择、参考投影、行选择和具名合同绑定保留父执行证据，派生操作通过谱系另外解释。
声明比较忽略纯名称绑定，不将父执行输出通道数误当成后续派生张量的通道数。

新源预训练checkpoint和个人模型分别绑定执行语义与执行provenance摘要。
若模型含完整执行身份，目标输入须有同语义证据；若旧模型无此字段，仍按历史声明门禁解释，
不补造旧执行凭证。source/target数据摘要、录制长度、源采样率及fit范围不要求相同。
个人校准和测试的录制上下文不放入`_input_contract`相等判断；模型文件摘要和训练合同
同时约束保存的执行证据。完成产物准入继续使用已有完整artifact/源身份核验。

## 验收和归档

正式测试包含独立MNE数值对照、七入口真实合成造数、历史record、源端点脉冲、低层None、
median反线性、近等率恢复、basis内存与缓存、凭证篡改、cache/concat/派生和新旧模型门禁。
实现过程中发现checkpoint局部导入遮蔽了类型名，已修正；相关44项模型/重放回归复跑通过。
入口定向累计165通过/3跳过，独立核心反例27通过，恢复/连续QC定向74通过。
这些集合可能重叠，不能相加充当全套测试总数；全套以validation.json为准。

工作树含既有大范围未提交与测试目录迁移。测试旧路径删除属于此前工作，本轮没有恢复旧平铺文件，
也没有提交或重置并发修改。测试期间保持活动源码与六指南固定；验证记录含开始/结束摘要。
Ruff只扫描活动src/tests/experiments直属入口，避开历史运行快照。

实施前324文件归档在
`frozen/preprocessing_execution_20260915_before/source_before.tar.gz`，对应manifest逐成员验证。
实施后固定源码、测试日志与结构化验收归档见
`frozen/preprocessing_execution_20260915_acceptance/manifest.json`。
归档payload留本地，manifest入版本管理；源码HEAD不能代替实际dirty工作树字节。
此前科研报告与14项探针保留原研究时点解释，不把旧基线测试改写成实施验收。
