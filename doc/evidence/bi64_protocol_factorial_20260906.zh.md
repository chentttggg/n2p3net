# BI64：源LOSO、校准单位与通道数的配对复核

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

预声明日期：2026-09-06，首轮seed=20260906。全64人均已被历史研究观察，
本轮是固定factorial的development复核，不是独立确认。源与target数据保持原始身份。

## 1. 问题与设计

不能将旧64人/16导/5selection/轻微调0.7308直接和新8人/5导/192trial/强微调0.8153相减。
本轮固定每个目标u，源训练仅使用其余63人；源内部epoch选择也不得使用u。
另报告不使用u标签的zero-shot，与用u的早期标签校准后的supervised adaptation。
后者不是完全不接触target标签的zero-shot LOSO，也不是在校准行上报测试性能。

\[
\theta_{-u}=\operatorname{Train}(\cup_{v\ne u}D_v),\qquad
\theta_{u,K}=\operatorname{Adapt}(\theta_{-u},D_u^{cal}(K)),\qquad
M_u=\operatorname{AUC}(f_{\theta_{u,K}}(D_u^{test}),y_u^{test}).
\]

| 因素 | 固定候选 |
|---|---|
| source外层 | 全64人逐一留出，63→1；不把target校准放入source |
| 通道 | 原生16导；共同Cz/P3/Pz/P4/Oz五导；不新增CAR |
| 校准 | 旧parser前5selection；前192trial |
| 优化 | 保留source参数/统计、冻结BN的15epoch/lr1e-4与60epoch/lr1e-3 |
| 对照 | zero-shot；合理target-only随机初始化60epoch/lr1e-3/target统计/BN更新；raw收缩LDA |
| test | 各协议原有test；另在两者真实row交集上配对重算AUC |
| source训练 | 当前既有16epoch上限、source内层选epoch后全source refit、batch512、BF16/compile/fused |
| 主统计单位 | participant；单seed首次定位，不把64折当64次独立训练程序 |

比较顺序：同source/通道/test下低高优化预算；同通道/优化/共同test下校准规则；
同协议/优化/test下5导与16导；同一已训练fold集合内分别汇总01–08、09–64、全64人。
01–61可另列目标子队列敏感性，但其source仍含其他原64人，不称作者推荐61人source实验。
最初8人共同留出的53人source checkpoint只作为不同source组成的参考，不叫LOSO。
校准单位、人数、通道组合造成的完整方法差不能全部归因于一个机制。

## 2. 已完成的split预检

原cache61011trial；5导/16导视图均保留同一X行、y与身份，不恢复任何原parser已丢弃的raw尾部。
64人旧5selection均可用，校准trial数min/median/max=60/204/420。
**64/64人的selection校准行与其前N个保留trial完全相同。**
因此只把相同N行改称trial校准，输入、标签、顺序、seed、优化器相同时，结果应相同；
不应重复训练这种数学上相同的命名对照。
192trial分支64人均保留两类test；不因新AUC删被试。
旧原始parser在64人中只丢弃2个不足一轮的尾部flash；新视图也继承这一处理，
因此本次改称trial校准没有恢复这些尾部波形。

旧parser的实际问题是依赖target行列组合/100/104标记推断边界、物理membership来源冲突，
以及这些边界参与校准预算和测试选择。旧算法并未把selection内的EEG波形先平均。
旧test还按selection边界排除与校准时间重叠的整个组，并仅保留每组前8次repetition；
必须量化这些行的分布，不能将其统称为“丢失P300信息”。

## 3. 运行与验证

冻结执行源码F：`frozen/binary_transfer_20260906f.manifest.json`，SHA256
`d7eb716caa4d1e23f688e5ad0a9c8d4007fce8ad85753e9356982e91c730af8d`。
新研究harness/计划另外保存文件hash；不改写原冻结结果。
工作目录：`experiments/runs/bi64_protocol_factorial_20260906`；云端同名目录位于
`/root/autodl-tmp/n2p3-runs/`。依次保存preflight、source/evaluation计划、journal、GPU资源与逐trial预测。
source每进程连续8个fold复用cache，初始4workers×4线程；按实际显存/时间调整并发，
不改变seed、batch、epoch、source名单或测试行。

完成前逐项验证：每份checkpoint恰有63个训练身份；target不在source；校准与test行、
原始时间窗不重叠；两通道方案同一protocol的test身份与标签一致；共同test不含任一calibration；
AUC从保存预测重算；失败不静默删去。只在完整结果后给出主效应与交互解释，不边跑边挑最优。

## 4. 对照入口一致性

使用同一53人源权重与subject01核对旧新入口：X、y、test row及zero-shot/收缩LDA分数逐值相同。
原模型与SubjectAdapter源码A/B/F校验值相同。15epoch微调最大score差1.55e-6；
60epoch微调/合理scratch分别约0.00116/0.00504，AUC差0/0.000029。
随后重复未经改动的原binary入口，相同seed也出现0.00121/0.00329的score差，
AUC差-0.000007/0.000022。该证据指向GPU长训练的数值重复性限制，不能宣称位级复现。
没有修改科学配置或扩大成算法收益；保留`conformance.json`与`repeat_drift.json`，
结果解释精度不超出这次观测的数值稳定性。原严格score容差未通过的记录也保留。

## 5. 固定八人的源组成补充对照

完整factorial后，固定前8目标、五导/seed/共同test/强微调，53人source均值0.8246，
63人source0.7843，相差-0.0403；这不是增加测试队列造成的平均值变化。
只追加8份source60模型：01–61中排除目标，被排除的62–64是作者预先标注的采集问题资料，
不根据本轮AUC挑source。目标与校准/test完全不变。对照解释为源组成和source fit敏感性，
不能宣称仅改变了人数（source内层epoch选择及updates仍可能随来源变化）。
该对照在查看source60结果前确定，不替换全64人的主结果，不作新的最佳配方搜索。
