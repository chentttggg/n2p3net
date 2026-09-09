# BrainSync 左耳垂 REF 更正与算法路径裁决

2026-09-08。用户明确：“我们的REF是左耳耳夹，在耳垂采集”，并要求通过科研判断决定Cz路径。
本页记录事实更正、参考选择的推导与实际联动；不把输入更正当作新的分类效能证据。

## 事实与裁决

采集REF为**左耳耳垂耳夹**。规范值为`left earlobe`，结构化字段为
`side=left/site=earlobe/electrode=ear_clip`。旧T7/T3为声明错误，不是兼容别名。
单端左眼角HEOG共享该REF；GND、左右乳突、右耳垂与双耳平均均不是这个参考的同义词。

保留两条显式路径：

| 使用目的 | 默认/要求 | 理由与限制 |
|---|---|---|
| 本人target-only信息保留基准 | 左耳垂原生七导；当前本人LDA入口默认如此 | 没有跨源兼容需要时保留全部已测信息；未宣称BrainSync准确率已胜出 |
| 复用现有Cz源checkpoint | 显式Cz六导，与同Cz的scratch配对 | 满足已训练源模型的参考/通道合同，不将左耳垂改名为右耳垂 |
| 原生七导对Cz六导研究 | 分别拟合QC、统计和模型，报告共同支持及原请求分母 | 这是完整输入流程对照；参考、维数及支持变化不能混作单一初始化效应 |

更正物理REF并不使已有Cz权重变成七导模型。原生参考与QC/EOG参考冲突时明确拒绝，
不会为了兼容而静默转成Cz，也不会插入一个假零Cz通道。

## 推导、反例与文献

令七路原生电压为`x_c=v_c-r_L`，共同Cz输入为`z_c=x_c-x_Cz=v_c-v_Cz`。
差分矩阵D为6×7、秩6、`D*1=0`。任何`x+a(t)*1`产生同一个z，因此每时刻丢掉一维共同分量。
若原生`x_Cz`另行保存，可用`x_c=z_c+x_Cz`还原；六导自身不能还原左耳垂七导。
本轮正式反例验证秩、共同分量抵消、Cz噪声传播和保留原生Cz后的精确重建。

目标差异波满足`delta_z_c=delta_x_c-delta_x_Cz`。Cz携带的任务响应与噪声都会进入差分。
原生左右耳垂电压之差为未知时变的`r_R-r_L`，不能当成常数或元数据别名。
共同Cz抵消原REF需要共同参考、相同时轴、兼容线性增益，不能补救饱和或事件错位。

- Bigdely-Shamlo et al., 2015，*The PREP pipeline: standardized preprocessing for large-scale EEG analysis*，
  [DOI 10.3389/fninf.2015.00016](https://doi.org/10.3389/fninf.2015.00016)，
  [全文下游分析](https://pmc.ncbi.nlm.nih.gov/articles/PMC4471356/#sec16)。本轮直接核验：
  visual oddball中被减掉的参考波形自身含任务信息，作者保存参考以便加回。
  不能将其64导结果当成BrainSync七导/Cz六导的预期AUC。
- de Cheveigne & Arzounian, 2018，*Robust detrending, rereferencing, outlier detection, and inpainting for multichannel data*，
  [DOI 10.1016/j.neuroimage.2018.01.035](https://doi.org/10.1016/j.neuroimage.2018.01.035)，
  [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5915520/)。支持坏参考/坏通道的空间污染传播；
  不证明Cz或左耳垂是所有任务的最优参考。
- Riccio et al., 2013，*Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis*，
  [DOI 10.3389/fnhum.2013.00732](https://doi.org/10.3389/fnhum.2013.00732)，
  [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3825256/)。原文为右耳垂参考，并在Cz测量P300差异波，
  直接说明Cz不是没有任务响应的“零电位”。

本地BNCI既有8人配对记录，K192原生七导轻微调减Cz六导轻微调为+0.0261184，
8/8人方向为正，描述性区间[0.0147709,0.0378164]。
本轮核对`experiments/runs/p300_mainline_20260906/analysis/verified_summary.json`，
它同时改变参考/维数/完整训练过程，不是BrainSync性能证明。
最新[BrainSync S1实验](brainsync_s1_calibration_20260908.zh.md)全部使用Cz六导，
没有原生七导对照，不能把该实验负迁移直接归因于Cz。

## 实际修改

后端`data.contract.brainsync_reference_record()`为权威，通过
`configs/brainsync_acquisition_reference.json`分发至独立采集端
`models/acquisition_reference.json`。前端默认montage、通道配置、EOG和BIDS均消费这一声明。
后端检查session REF、BIDS `EEGReference/EEGReferenceDetails`和EEG通道参考的一致性；
不会把左耳垂与公共源右耳垂合并。

本人LDA和源微调分别保存参考选择。原生选择传`--reference-mode native`，
后端拒绝隐式Cz QC/EOG；保存模型记录该选择并在重载时继续核对。
模型结果分别显示采集参考与最终输入参考。已有源checkpoint的通道/参考门禁保持有效。

七导重载验证另发现大小写问题：模型保存的`OZ`遇到原EDF的`Oz`，局部QC注释不匹配。
现在用共享通道标识解析到原始实测名称后再投影注释，补充了真实局部异常的回归测试。
没有添加任意别名或改变通道物理身份。

## 记录、缓存与历史

更正前源码、文档、配置和活动会话共474文件已压缩并逐成员核验，见
[原状态清单](../../frozen/brainsync_reference_20260908/before.manifest.json)。
随后更正9个活动会话目录的24份session/BIDS元数据，包括原采集目录及四份规范准备副本；
18份EDF完整文件SHA256保持相同。见[逐文件更正清单](../../frozen/brainsync_reference_20260908/metadata/manifest.json)。

原始二进制头与追加式事件日志保持采集时字节；其中旧参考文字由显式`reference_correction`
记录覆盖，BIDS/session为算法读取的更正声明。原R5采集目录缺少部分sidecar，P002记录也未因本次
更正获得准入；没有补造文件或改变实验完成状态。四份规范准备记录均通过严格读取。

历史cache、QC、模型、预测及压缩快照不改写，不把旧digest替换成新digest。
重放历史运行须使用更正前归档中的元数据与原执行源码；当前训练使用新更正输入。

更正后的派生物在`experiments/runs/brainsync_reference_20260908/`：

| 会话 | Cz连续/epoch QC后 | 原生七导连续/epoch QC后 |
|---|---:|---:|
| R2 | 342 / 327 | 370 / 367 |
| R3 | 289 / 247 | 327 / 294 |
| R4 | 368 / 348 | 377 / 375 |
| R5 | 273 / 229 | 307 / 296 |

连续检测阈值和原生状态包络仍只来自R2。原EDF摘要逐份匹配后，为Cz质量模型生成元数据更正副本，
其数值参数不重拟合；三个已有Cz缓存的X/y/mask/准入逐项与旧缓存相同。
七导使用原生投影，epoch QC在七导R2内独立拟合并冻结，不继承Cz六导阈值。
这两组保留率包含不同空间依赖、最少有效导数与独立QC，不可据此选择准确率赢家。

`quality/continuous_native7.json`与`epoch_native7.json`用于原生七导；
`quality/continuous_cz6.json`与`epoch_cz6.json`用于Cz六导。
`correct_brainsync_reference.py`提供带显式旧标签/证言的更正入口；
其连续QC声明更正函数要求原拟合会话和全部原始EDF摘要一致，不能用于真正换接参考后的数据。

## 验证与边界

原生七导已用真实R2通过CLI本人LDA校准、保存及R5无标签重载推理：LDA实际使用209个完整七导
校准epoch，R5为307个输入epoch/100个可评分完整七导事件；其余保留拒判账本，未发布准确率。
这是链路验证模型，不替换生产默认模型。Cz源模型在R2+R4上的真实校准预检通过，未重训源NN。

参考一致性、秩/反例、显式native冲突、元数据归档与EDF保留、通道注释重载均有对应测试。
全量后端1351项中1347通过、4跳过、0失败，见`backend_final.xml/log`；前端全量173项通过。
上述后端全量运行开始于本地10:20:26，耗时271.54秒；并行任务随后进入的个人质量模块改造
不自动继承这一完整验证记录。本轮参考合同的最终专项检查另保留。
并行修改合入后，42项参考/采集/局部恢复/模型校准专项测试通过，见`post_merge_tests.xml`；
同一个原生七导LDA模型再次通过真实R5无标签推理，结果仍为原覆盖不足拒判，模型文件摘要未变。
当前工作区`ruff check src experiments tests`与`git diff --check`通过。
1024×768及1440×1000的
实际Qt参考设置已检查，文字与控件尺寸匹配。Ruff及`git diff --check`作为最终收尾检查。

本轮没有云端训练、没有新NN矩阵或前瞻成人确认。左耳垂物理接法已按用户证言确认；
七导相对Cz的BrainSync准确率仍需配对实验，不能由元数据更正或保留率推出。

更正前快照、逐文件更正记录与57成员交付归档均已逐成员校验，见
[总清单](../../frozen/brainsync_reference_20260908/manifest.json)。后续合并验证的receipt单独保存，
不改写较早全量测试、模型或原始数据的身份。
