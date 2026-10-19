# 评分、结果归约与统计：首轮独立审查

日期：2026-09-29。范围：当前评分源码、历史公开开发原始预测和局部行为反例。生产代码、原始结果、缓存和模型均未修改，未训练或重新拟合模型。脚本、小型摘要、反例与哈希清单保留本目录；本轮完整逐cell复算结果位于Git忽略的`experiments/runs/deep_chain_audit_20260929/scoring/`。

## 当前能证明的结论

**没有找到已检查公开矩阵的AUC/AP/BACC算错、正类反转或保存预测与指标不一致。** 从原始压缩包直接读取48,250份预测，独立使用Mann–Whitney秩统计计算AUC、同分阈值归并计算AP，不调用sklearn指标函数。其中09-16统一矩阵和P1/P2矩阵还逐份检查预测文件SHA256、校准/test行摘要、实际K，以及同人同K跨臂和seed的test标签一致性。

**确实找到了同一模型因测试队列变化而得到很不同AUC的真实实例。** 旧四库b/c矩阵的123人中69人测试队列改变，同checkpoint在共同测试行的AUC完全一致；BI2014a-13的总体AUC从0.574939变为0.761719，仅因为test从299行变为96行。因此旧zero-shot、旧无迁移和新迁移分数必须先对齐测试队列，不能只对齐模型名与K。

**另有当前候选评分/分析API的行为缺陷，但未取得它们污染十二库成绩的证据。** 缺候选时的总体/分重复hit不一致、开启中心化后后续score影响较早R、valid分析跨不同有效replicate集合相减均已复现；这些路径不参与已检查统一矩阵的binary AUC归约。当前十二库09-18的原始预测完整归档本地未找到，本轮不把09-16重算冒充09-18原预测重算。

## 入口及数据字段图

| 路径 | 预测与评分 | 归约及科学单位 |
|---|---|---|
| `experiments/run_eeg_loso.py` → `baselines.evaluate` | `predict_logit`的一维正类margin；AUC用原margin；BACC用校准阈值 | binary按fold等权`nanmean`；标准完整LOSO一fold一人时等于participant macro；candidate总体按decision池化 |
| `run_binary_subject_transfer.py` / `run_cross_dataset_transfer.py` | source checkpoint、SubjectAdapter或波形LDA返回margin；`y=1`为正类；AUC用margin，后者BACC以`score>0` | 每人、actual K、arm记录；跨run不可只凭相同K比较 |
| `transfer.target_experiments` | AUC/AP用margin，Brier用expit(margin)，NLL用稳定`logaddexp(0,s)-y*s` | 每个真实person/K/seed/arm单元保存NPZ与sealed JSON；共享基线有复用身份 |
| `stats.transfer` / `analyze_training_chain.py` | 消费cell AUC和行摘要；不从模型重新预测 | 同人先配arm，再按声明seed平均；固定库内participant macro，库等权；exact K与fallback分开 |
| `models.candidate_evidence.membership_decision_scores` | 固定物理membership的sum-margin；softmax沿单个decision的candidate维度；完整性、可识别性、tie门禁 | posterior不自动宣称已校准；BNCI hit另以物理decision请求分母计算 |
| `transfer.candidate_decision` → `run_candidate_cross_decision.py` | row/column前缀累加；完整覆盖和tie输出DecisionOutcome | requested、data eligible、evaluated分别记账；同一被试多decision不是独立参与者 |
| `transfer.evaluation` → `run_within_subject_transfer.py` | 普通candidate曲线、raw-all及balanced-all | helper可给条件eligible hit；runner另用requested分母重建operational字段 |
| `transfer.outcomes` → `analyze_candidate_cross_decision.py` | 状态账本完整cross-product核验 | requested与valid两种estimand；默认valid路径存在缺replicate配对问题，见后文 |

机器可读入口索引见`entry_map.json`：46个当前评分/归约函数和43个本地独立分析/验证脚本入口。初版索引按prediction文件名搜索；后续原始结果审查也覆盖了以数字命名的NPZ，不能拿该索引的12条显式命名NPZ当全部结果数。

## 原始预测独立复算

| 归档 | 目标预测数 | 最大AUC误差 | 最大AP误差 | 最大BACC误差 | 额外检查 |
|---|---:|---:|---:|---:|---|
| `four_dataset_20260907/four_study_development_20260907b.results.tar.gz` | 2,214 | 2.98e-8 | 原cell未统一保存AP | 0 | 原始y/score重算，跨K zero-shot预测核对 |
| `four_dataset_20260907/four_study_development_20260907c.analysis.tar.gz` | 1,648 | 2.98e-8 | 原cell未统一保存AP | 0 | 同上；真实actual K保持原值 |
| `unified_cloud_20260916/export/verified_results.tar.gz` | 12,150 | 2.22e-16 | 3.33e-16 | 0 | 逐预测SHA、行摘要、K及同人K配对标签；0异常 |
| `domain_transfer_cloud_20260916/export/results.tar.gz` | 32,238 | 3.33e-16 | 4.44e-16 | 0 | 同上；0异常 |

后两包的Brier/NLL最大误差分别不超过1.11e-16/8.89e-16。这里只验证保存预测的评分、行和标签身份，未重载模型重新预测，未认证所有训练/预处理行为。前两包误差在3e-8内，属于历史指标数值精度范围，没有改变实质结论；后两包达到双精度舍入级一致。

复算脚本：[audit_public_predictions.py](audit_public_predictions.py)、[audit_unified_archive.py](audit_unified_archive.py)，默认将完整结果写入上述本地忽略目录。可分发的小型总表见[summary.json](summary.json)，移动前后逐文件SHA256和本地路径见[local_results_manifest.json](local_results_manifest.json)。

本地完整结果：[public_prediction_audit.json](../../../../experiments/runs/deep_chain_audit_20260929/scoring/public_prediction_audit.json)、[unified_20260916_prediction_audit.json](../../../../experiments/runs/deep_chain_audit_20260929/scoring/unified_20260916_prediction_audit.json)、[domain_20260916_prediction_audit.json](../../../../experiments/runs/deep_chain_audit_20260929/scoring/domain_20260916_prediction_audit.json)。后两份JSON包含每个受检member及其输入/预测摘要，不依赖原报告文字判断。完整结果不随Git分发；原始训练归档和旧结果未移动或删除。

主AUC链的方向证据：`src/baselines/deep.py:2817`与`src/transfer/subject_adapter.py:1804`使用`logits[:,1]-logits[:,0]`；LDA使用二分类`decision_function`；`src/baselines/evaluate.py:443-453`要求一维有限score；`src/transfer/target_experiments.py:645-650`区分AUC/AP margin与sigmoid概率质量。未发现主链把threshold结果或softmax其他轴送进AUC。

## 有真实run证据的比较偏移

### E1：相同checkpoint，测试队列改变，zero-shot AUC大幅变化

生产机制在`src/transfer/binary.py:89-146`：同一批K共用按最大预算确定的后续test。它保证批内公平，但改变该批最高预算就会改变跨run test；这是比较身份问题，不能仅凭本机制判模型训练有bug。

| 参与者 | b矩阵test条数 / AUC | c矩阵test条数 / AUC | 共同test上两边AUC |
|---|---|---|---|
| BI2014a-13 | 299 / 0.574939 | 96 / 0.761719 | 0.761719 / 0.761719 |
| BI2014a-02 | 300 / 0.570080 | 96 / 0.699219 | 0.699219 / 0.699219 |
| BI2014a-25 | 408 / 0.521972 | 120 / 0.402000 | 0.402000 / 0.402000 |

123人中69人队列改变；所有共同测试标签一致，共同score最大变化4.77e-7，所有共同test AUC完全一致。四库等权zero-shot从0.5760259变为0.5820929；participant直接混池均值则从0.5669536变为0.5768285。具体归档member保存在`zero_shot_cohort_comparisons`。

### E2：同一组预测，换归约单位即可让方法差值变号

旧c矩阵K2000只有21人、BNCI/Huebner两库；不是四库结论。对这些完全相同的保存预测：

| 归约 | full-fine AUC | scratch AUC | full−scratch |
|---|---:|---:|---:|
| 库等权、库内participant宏均值 | 0.8347467 | 0.8307857 | +0.0039610 |
| 全部trial混池后计算单个AUC | 0.8415019 | 0.8420254 | −0.0005235 |

跨被试混池AUC会比较不同个人模型之间的绝对score，包含模型偏置/尺度差异，不能替代participant宏AUC。该例是估计量变化的真实反例，不证明原报告曾错误混用；原四库分析脚本明确按库等权。

### E3：BACC阈值定义不同，不可直接对比

`baselines.evaluate.py:590`使用在训练/验证范围得到的校准阈值，并取`>=`；迁移`target_experiments.py:647`使用原score零阈值并取`>`。两边字段语义不同，AUC不受这些阈值影响。已审查真实矩阵的原零阈值BACC均精确复现；未发现阈值被错误用于其AUC。

## 当前API行为缺陷：已经复现，但历史触发范围有限

所有反例由`counterexamples.py`生成`counterexamples.json`。除对真实归约器注入已固定fold预测外，不替换评分代码；没有拟合模型。此节不能用于声称十二库分数已被污染。

### S1：LOSO候选总体hit与分重复hit对缺候选处理相反

- 入口：`baselines.evaluate.evaluate_candidate_selection`，由`experiments/run_eeg_loso.py:680-691`实际调用。
- 总体路径`evaluate.py:1443-1455`调用`decide_by_group`，后者调用`candidate_evidence_scores`时保留默认`missing_candidate_policy="exclude"`；缺候选仍可能给出正确预测。
- 同一报告的分R路径`evaluate.py:1468`调用`hit_correct_total_by_repetition`，其`evaluate.py:243-250`显式`missing_candidate_policy="abstain"`。
- 两个group，一个完整并正确、另一个仅剩真实目标candidate：真实summary reducer返回`primary_hit_rate=1.0`，同一R1返回0.5；严格outcome账本也为0.5。
- 默认不要求开启center flag。前提是实际候选缺失；全候选完整时这个反例不触发。CLI的测试epoch排除默认False（`run_eeg_loso.py:347-349`），显式QC删除或原输入缺候选时才需重点检查。
- 历史证据限制：本地松散LOSO record仅找到旧8人binary结果，candidate指标为null、AUC无缺失，不能证明某次真实candidate run已触发S1。统一AUC矩阵及BNCI membership hit不走此路径。

### S2：显式中心化会使后续score改变较早R的预测

- 入口同上；`hit_correct_total_by_repetition`在`evaluate.py:216-222`先用所有`max_repetitions`行算group均值，再在`:233-249`截取每个R。
- 反例scores `[1,0,1,0]`、candidate `[0,1,0,1]`、repetition `[0,0,1,2]`，只把最后R3 score改为100，R2命中由1变0。R2输入score没有变化。
- 触发要求：`center_logits=True`且较早prefix候选计数不等；等覆盖时公共中心项会相消。应在每个prefix内部中心化，或用明确完整覆盖门禁使不完整R保持incomplete。
- **默认关闭**：`run_eeg_loso.py:357-359`和`evaluate_candidate_selection(...center_decision_logits=False)`。本轮搜索松散run JSON/Python/log未找到真实运行启用True的记录；匹配到的仅是源码副本。因此是条件API缺陷，未证明历史成绩受影响。

### S3：默认valid分析可把不同有效训练replicate集合称为paired

- `analyze_candidate_cross_decision.py:60`及builder CLI `build_candidate_cross_decision_manifest.py:489-493`默认`valid_subject_operational_hit_rate`。
- `analyze_candidate_cross_decision.py:1137-1145`各arm各participant只平均仍有valid分母的replicate；`:1178-1190`再将这些均值直接相减。
- 反例p1只有A/r1=0.9和B/r2=0.1有效，根本没有共同有效replicate；p2两replicate配对差都为0。该路径仍报告p1配对差+0.8、总体+0.4。
- 触发要求：valid指标、至少两个replicate、有效分母随arm/replicate变化。requested指标保留全部失败分母，单seed也没有这个跨replicate集合问题。
- 本地松散历史JSON/分析输出搜索未找到使用该valid指标的真实运行记录，不能声称某次已完成矩阵已触发。十二库主分析使用`stats.transfer`及单seed，二者不受本反例直接影响。

### S4：通用统计配对只验证行号摘要，不绑定输入与标签

- `stats.transfer.py:287-289`只配对`calibration_rows_sha256`与`test_rows_sha256`。索引相同而不同raw view、预处理或标签也可能有相同行号摘要。
- 反例显式给A/B不同`input_sha256`和`test_labels_sha256`，该分析器仍接受并计算+0.8。
- 这不是证明实际矩阵混用输入：本轮后两归档实际每份预测标签、行摘要已独立核对，未发现混用。建议统计入口消费完整输入/标签身份，或至少明确要求并验证producer合同保证。

### S5：domain基线fold验证入口直接抛UnboundLocalError

- `baselines.evaluate._validate_folds`仅在LOSO分支`:388`初始化`test_counts`，domain分支`:436`直接读取。
- 两域合法互补留出折，`fold_protocol="domain_lodo"`，无模型拟合即抛`UnboundLocalError`。`domain_ceiling`共享该未初始化变量。
- 当前统一/跨库迁移矩阵不调用这项baseline验证入口，当前experiment脚本未搜到传这两个字符串的调用；这是可单独修复的接口缺陷，不解释已完成统一矩阵的准确率。

### S6：trim/precision接受可靠性权重，却让零权重观测参与评分

- `candidate_evidence_scores`文档`:291`将`trial_weights`声明为可选非负、无标签可靠性权重，没有声明trim/precision会忽略它们。`:326-331`统一验证非负权重；`:345-349`仅检查候选至少有一个正权重。
- `candidate_evidence.py:365-380`的trim分支使用全部`selected_values`、precision分支仅使用inverse variance，均不使用`selected_weights`；因此权重为零的行仍可污染分数。
- 反例scores `[1,100,2]`、candidate `[0,0,1]`、weights `[1,0,1]`：trim给候选0分数101、候选1分数2，precision给50.5和2，两者都预测0。删除零权重行后两者都预测1。第七组行为反例已保存。
- 当前调用图限制：全部当前生产直接调用中，只有`decide_by_group`向该核转发`trial_weights`，而它只允许sum/mean/tempered三种模式，拒绝trim/precision。其他调用未传外部trial_weights。故这是直接API合同缺陷，**没有找到当前生产路径可以同时传非平凡权重与这两种扩展聚合**。
- 松散run JSON精确字段搜索未找到`aggregation=trim0.2/precision`或`trial_weights`运行声明；不把文献JSON中普通precision词汇当使用记录。未证明历史或十二库矩阵受影响。修复可以实现并声明权重语义，或对不支持的权重组合明确拒绝，不能继续默默忽略。

## 对09-22旧审计的复核与纠偏

1. **V2得到真实产物支持**：最大K改变test队列，已有同checkpoint/共同score的数值证据；应定位为跨run比较身份变化，而不是批内配对出错。
2. **V1不能按旧表述泛化**：`candidate_evidence_endpoints`确实给correct/eligible，但其生产调用`run_within_subject_transfer.py:708-715`累计计数，`:873-890`同时明确输出eligible和完整requested分母的operational字段。helper条件率本身不证明runner的operational率被高估。S1是另一个确切不一致的旧LOSO候选入口。
3. **V5的“Holm p没有产生路径/不可复现”不成立**：`experiments/runs/unified12_20260918/analyze_unified_results.py:61-65`有50,000次配对participant sign-flip、+1修正和3项Holm。公共stats模块未消费独立`paired_permutation_test`意味着分析实现分散，不等于p值没有来源。单seed、固定study和条件模型推断限制仍成立。
4. 验证loss归约由主审查另行处理；本报告不把普通LOSO `_validation_pass`问题泛化到统一矩阵fixed-update源训练路径。
5. 跨运行结果重标scope/snapshot由主审查另行复现，本分支未重复操作。

## 当前尚未证明及建议的对齐顺序

未证明：09-18十二库每份原始预测重算、S1/S2/S3在历史真实candidate run被触发、任何新成人产品收益。当前较高旧无迁移分数不自动说明新迁移代码出错；首先应逐项对齐输入身份、source/target人群、实际校准行、共同test标签、训练剂量和统计单位。

建议把下一步“旧无迁移vs当前迁移”比较固定成一份可机器核验的对照账本：同一participant、同一raw view及标签摘要、同一实际K/校准/test行、同一score定义和宏平均。先重放评分；只有差异仍存在，才解释为模型/训练机制变化。本轮48,250份预测重算及E1/E2证明，评分数字本身可精确正确，而不同实验身份仍能产生完全不同的科学解释。
