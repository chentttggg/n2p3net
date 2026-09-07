# AUC增益解释与泄漏复核

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

针对用户追问补做的审计；不改算法、原结果或已冻结交付包。解释对象是
[公开ERP研究结果](p300_transfer_results_20260906.zh.md)，不是成人BrainSync确认。

## 1. 不能直接比较旧0.73与新0.8153

旧值为dev64 `native__full_fine_no_mmd` AUC=0.7308，本身已经使用目标校准；
旧zero-shot是0.7091。因此不能把差异解释为“以前未校准，现在有校准”。
旧计划为64人、16导、每次留出1人、单seed、5个解析器selection校准、15epoch/lr1e-4。
新BI pilot为01–08八人共同留出、09–61源训练、共同五导、三seed、192trial校准和固定purged后缀。
队列、source集合、通道、校准与test范围同时变化，未做完整同口径旧新重跑前，
无法将0.8153-0.7308归为算法版本的独立提升。

同一新BI cache、source checkpoint、8人/3seed/K192/test下：

| 方法 | AUC |
|---|---:|
| source zero-shot | 0.6565 |
| source full-fine，15epoch/lr1e-4 | 0.7075 |
| source full-fine，60epoch/lr1e-3 | 0.8153 |
| 冻结source feature+收缩LDA | 0.7854 |
| target-only旧弱配方，15epoch/lr1e-4/冻结BN | 0.5246 |
| target-only合理配方，60epoch/lr1e-3/目标统计/更新BN | 0.7322 |

这支持“适配配方与小样本估计是重要瓶颈”，不支持把增益归因于新结构、加宽或EOG清理。
源feature保持不变的收缩头已经达到0.7854，说明源表示中已有可用信息；并不能据此量化
全部增益中多少是P300神经特征。原生BI主结果没有新增EOG清理。
BNCI七导zero-shot0.7760→轻微调0.7989只增加0.0229；相对scratch0.6786的0.1203是
同预算完整迁移方法对目标独立训练的比较，不是相对历史算法版本的提升。
source参数/统计/BN与scratch的配方有区别，不能再把完整方法差解释为单一权重初始化效应。

## 2. 本次实际核查

审计脚本：`experiments/runs/p300_mainline_20260906/audit_auc_leakage_20260906.py`。
输出：`experiments/runs/p300_mainline_20260906/analysis/auc_leakage_audit_20260906.json`。
在既有云端项目.venv中读取真实cache与checkpoint；没有启动重训或据此选择新配方。

- BI五导与BNCI七导cache中，跨被试完全相同的epoch数均为0。此检查不排除近重复、
  未知重复身份或共同采集依赖，但未发现明显复制文件/样本问题。
- 以`training_identity_ledger`的实际训练身份重建source行，数量与checkpoint一致；
  重新计算源mean/std（包含既有1e-6尺度稳定项），与checkpoint相符，未包含target被试。
- 复核K192的224个BI与120个BNCI评估cell：source行与目标校准/test行不交叉；
  校准与test行不交叉；test的原始epoch起点不早于校准epoch最晚可用时刻；
  保存的test标签与cache原标签一致，重算AUC一致。
- 代码调用链确认adapter.fit只接收校准X/y；目标mean/std只在fit训练行估计；
  predict进入eval/inference模式，复用已拟合统计。source epoch选择发生在排除target后的
  source内部，随后只在source行refit。未发现将target后缀交给optimizer或BN拟合的路径。
- checkpoint字段`source_subjects`实际保存cache全体被试，名称有歧义。不能以它代替
  `training_identity_ledger`判断训练成员；后者、本次实际重算及训练X/y切片均符合排除要求。

## 3. 边界敏感性

固定原预测，不重新训练。删除共享calibration selection的test行，或将test在原purge基础上
额外后移5/30秒后重算。所有值先participant内平均三seed，再平均participant。

| K192场景 | 原后缀 | 删除同组test行 | 再间隔5秒 | 再间隔30秒 |
|---|---:|---:|---:|---:|
| BI强微调 | 0.8153 | 0.8129 | 0.8135 | 0.8186 |
| BNCI七导轻微调 | 0.7989 | 0.7993 | 0.7993 | 0.7996 |
| BNCI七导zero-shot | 0.7760 | 0.7764 | 0.7763 | 0.7766 |
| BNCI七导scratch | 0.6786 | 0.6786 | 0.6783 | 0.6779 |

BNCI每人原4003个test trial中有43个与校准同selection；BI八人对应59/0/59/23/11/0/35/12行。
BI组仍是解析器surrogate，物理membership未恢复。删除这些行后高分没有消失，
不支持“主要由边界同组泄漏制造提升”的解释。但这些子集不同，微小升降不作新方法收益；
30秒也不构成严格IIR有限支持证明。两者都没有解决同session长期状态、刺激/凝视相关性。

## 4. 已存在的乐观偏差与结论边界

**后缀成绩参与了配方比较和选择。** LR/epoch/统计/BN等实验的后缀结果被研究者观察，
BI强配方与BNCI轻配方由开发证据支持后作为代表报告。这是开发集模型选择偏差；
若把同一后缀再称为独立test，它就是测试集信息泄漏。标明development是在限定结论，
不能消除该偏差。3seed复现同一批被试也不是新确认数据。

现有bootstrap只条件于已训练/已挑选方法；未覆盖多轮方法选择的不确定性。
LOSO训练集合还互相重叠，8人不能被扩成24个独立样本。因此CI排除0不能证明没有泄漏，
也不能当预先声明检验的确认性显著性。
这一风险与[Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)讨论的
有限数据模型选择偏差一致；该外部来源的核查等级与查询记录见已有文献核查文档R20。

另需区分：同session迁移比跨日、重新佩戴及未知目标决策容易；由眼动/视觉诱发成分形成的
可预测捷径不一定是训练数据泄漏，却会削弱P300归因与产品外推。本轮尚无独立EOG效果或
真实BrainSync确认。随机flash五折的raw重叠问题只属于该诊断分支，不能拿其分数作主结论；
连续块五折已单列，但也不是新的独立成人证据。

结论：目前没有发现上述样本/身份/源统计/优化器输入层面的直接泄漏证据；
已经存在开发后缀参与选方案的偏差，且真实独立会话泛化未验证。
下一步应冻结候选与配方，把调参放入开发或嵌套内层，并用新session/新被试尤其真实BrainSync
进行确认。不能把0.8153写成“已确认从0.73提升到0.82”，也不能提前承诺成人准确率。
