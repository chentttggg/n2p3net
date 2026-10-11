# 公共数据评估与配对分析

数据范围：BNCI2014-008 八人、一个完整 source/target seed（20260918）、每人19个共同请求，共152个36选1决策。读取原80配对对照和20指定只读导出，核对原文件SHA并从逐人数据复算全部100行配对差值。本次新增拟合0。

## 原云候选面板：完整分母与配对读数

云窗口独立零拟合恢复已完成8人、152共同requested decisions、13376预测、448 fit records、32物理包、88指标与原80配对对照。科研总监实际读取本地terminal/actual exit/delivery/independent statistics，逐字核其SHA与delivery吻合，再读完整80原对照及20指定导出；其中8行已有原对照、12行为原保存预测的新只读派生。原executor exit1、scientific source `75a61532…`及其499冻结manifest均保留，恢复exit0、新source/target fit均0。完整失败账本留本地，不以科研新prefix协议改签。

## 校准剂量与温度程序

hit@all以全部152个请求为分母；fixed正K各档与grouped K800/2000的概率覆盖144/152（94.74%），grouped K200则全部不可用。前者8个缺失均为每人selection16在固定Kmax后及原epoch支持purge处的不完整决策，不能删除或填补。NLL越低越好，只在144个共同有概率决策上计算，先在participant内平均再跨八人平均。

| 授权K | fixed命中 | grouped命中 | fixed NLL | grouped NLL | grouped−fixed NLL |
|---|---:|---:|---:|---:|---:|
| 200 | 83/152（54.61%） | 0/152（全部不可用） | 1.744632 | 不可用 | 不可用 |
| 800 | 113/152（74.34%） | 114/152（75.00%） | 0.851218 | 0.999665 | +0.148447 |
| 2000 | 132/152（86.84%） | 136/152（89.47%） | 0.367592 | 0.485101 | +0.117508 |

fixed从K200到K800增加30个命中（+19.74个百分点），从K800到K2000增加19个（+12.50个百分点）；相应NLL也下降。当前数据支持增加目标校准剂量的收益；后续把校准时间、曝光成本和新数据结果一起纳入预算选择。

grouped K800只增加1个命中，NLL增加0.148447；K2000增加4个，NLL增加0.117508（相对fixed上升31.97%）。两档都未通过原hit/NLL联合开发点估计门。K200全部不可用是完整C组不足，不能解释成合法输出上的零识别率。

## 逐人风险：K2000主pool

| 人 | fixed正确/19 | grouped正确/19 | fixed NLL | grouped NLL | ΔNLL |
|---|---:|---:|---:|---:|---:|
| A01 | 16 | 16 | 0.4482 | 0.6268 | +0.1786 |
| A02 | 17 | 17 | 0.3473 | 0.4413 | +0.0940 |
| A03 | 17 | 17 | 0.4550 | 0.5079 | +0.0529 |
| A04 | 12 | 15 | 0.6359 | 1.1053 | +0.4695 |
| A05 | 18 | 18 | 0.3964 | 0.5073 | +0.1109 |
| A06 | 16 | 17 | 0.5453 | 0.4793 | -0.0660 |
| A07 | 18 | 18 | 0.0894 | 0.1517 | +0.0623 |
| A08 | 18 | 18 | 0.0232 | 0.0612 | +0.0380 |

K2000的4个净新增命中集中于A04（+3）和A06（+1），其余六人命中数未变；总体6个救回、2个损害。7/8人的NLL变差，只有A06改善。A04命中12→15，但NLL0.6359→1.1053，说明命中次数与完整概率质量关注不同结果；后续按错误决策的置信度与模型失配进一步定位风险。

## K2000 fixed专家与组合：原预声明配置

| 配置 | 正确/152 | hit@all | NLL（共同144） |
|---|---:|---:|---:|
| nn_xdawn | 133 | 87.50% | 0.219186 |
| scratch | 135 | 88.82% | 0.255905 |
| nn | 128 | 84.21% | 0.292781 |
| nn_toeplitz | 131 | 86.18% | 0.295845 |
| matched_scratch_pool | 133 | 87.50% | 0.332658 |
| xdawn | 129 | 84.87% | 0.359888 |
| waveform_lda | 131 | 86.18% | 0.359925 |
| primary_transfer_pool | 132 | 86.84% | 0.367592 |
| classical_three | 127 | 83.55% | 0.427169 |
| toeplitz | 125 | 82.24% | 0.760679 |
| raw_cz6 | 95 | 62.50% | 2.818903 |
| raw_native8 | 103 | 67.76% | 4.769221 |

原预声明NN+xDAWN组合相对单NN：128→133个命中，NLL0.292781→0.219186，救回7个、损害2个，原开发点估计门true；主四专家pool则为132个/NLL0.367592，增加命中同时提高NLL。这支持继续研究有限组合与整程序选择；预声明的C内选择算法和新later数据将提供后续确认。

## 源预训练与匹配scratch

方向统一为scratch−source；正Δ命中/负ΔNLL表示scratch改善。

| K | 对照 | Δ正确/152 | ΔNLL | 救回/损害 |
|---|---|---:|---:|---:|
| 200 | 单NN | +1 | +0.052668 | 18/17 |
| 200 | 匹配pool | -1 | +0.039382 | 8/9 |
| 800 | 单NN | +3 | +0.180063 | 14/11 |
| 800 | 匹配pool | +0 | +0.056788 | 4/4 |
| 2000 | 单NN | +7 | -0.036876 | 11/4 |
| 2000 | 匹配pool | +1 | -0.034934 | 4/3 |

K2000单NN的scratch多7个命中且NLL低0.036876，matched scratch pool多1个且NLL低0.034934；该方向表示scratch改善source。K200/K800 fixed pool的source在NLL上较好，显示迁移作用随预算和参与者而变化。K0源NN仅7/152命中，主pool因依赖目标拟合专家而全部不可用，给后续source-only排序与低剂量适配提供了明确基线。

## 估计量与下一步

本面板以八名参与者为独立单位，152次决策为其内部重复测量；一个seed的读数用于公开校准开发。原开发gate是hit/NLL的点估计规则；大版本的完整seed与新数据评价将补充训练随机性和泛化证据。

后续落实C内整程序选择及独立later评价。P1固定response的16格噪声面板已经完成，数值稳定化正在[响应与噪声](response_and_noise.zh.md)推进。缓存和白化优化以相同概率输出及实际耗时评价。

图和完整CSV保存在本地科研树 `tmp/DATA_ANALYSIS_20261011/`：overview.png、configuration_metrics.csv、all_paired_contrasts.csv、participant_paired_metrics.csv。来源、分析程序与产物SHA见[紧凑绑定](../assets/research_resume_20261011/cloud_candidate_analysis_binding.json)；原始云结果保持原位置及producer身份。

## 跨数据集队列

下一跨数据集队列先做 **BI2014a**，每库完成并封存产物后再进入下一库。已有`finite_zero_none_targets`的61份记录、976文件属于binary prefix preparation，raw512Hz→model128Hz/179点；candidate准备复用`data.bi2014a_candidate`/`prepare_bi2014a_candidate`恢复CSV中的真实membership、episode和物理支持。

61人的零拟合CSV清点得到1,592个由完整12-flash repetition构成的decoder episode，R1–R8数量依次为412/400/298/161/109/78/48/86。按“首五段校准”的候选分配，后续共同1,287段的R1–R8为351/329/242/123/79/62/36/65，只有65段（5.05%）达到R8。首五段各人使用60–372标签。2026-10-12已明确将本轮Q1冻结为首五个实际完整raw episodes的episode预算程序，各人actual labels另记；与B12的label-K0/200/800/2000矩阵分列。新队列主端点采用每段自身全部完整repetitions的终点，并保留每段R和requested分母；subject57 selection23主R=1，末尾R2 partial单列。原metric入口要求单一R，需接逐decision horizon或正确加权的分层汇总。

完整记录的末段另核completion/censor：subject57原有1,454 flashes，decoder保1,452。selection23的完整R1仍是1,287主请求中的一个，主horizon为R1并进入主指标；末尾额外2 flashes按R2 terminal_partial单独记censor/operational账本，不进入主指标、不新增第1,288个episode，不能靠floor-trim认作完整R2 episode。首五段输入清点的42个短gap涉及60个相邻段；按现strict mapper的pre-roll规则，实际拒42个右侧未来段，60是支持重叠并集诊断。共同未来从每人固定Kmax raw cutoff一次派生，所有K和三臂保同requested episode及具体不可用原因。

cloud source07已报告原leave-BI2014a producer完成load/replay及排除审计：seed20260918，checkpoint SHA `fc8d8cbdef405263ff9c233e592755ceeeb2264bc8287800b8b2ffacfb199c02`，438个source participant key与61个target identity无交集。该source的producer身份保持。当前接合finite-zero-none入口、实际观测Cz6 projection与逐decision horizon；新caller复用原fit_calibration_candidate/FrozenTrialPredictor，完成三臂fresh-load LLR及36-choice概率重放后自主执行61人。target grouped temperature不在该队列学习。每库保存plan、输入/source binding、participant receipts、模型重放、完整失败账本、指标和资源终态；进展见[BI2014a队列计划](../assets/research_resume_20261011/bi2014a_queue_plan.json)。

### BI2014a Q1 完整 episode 队列终态

Q1 已按首五个完整 raw episode 的 episode-budget 程序封库：61 人、三臂（`source_zero_shot`、`source_full_fine`、`matched_scratch`），183 个模型与183个 arm receipts，queue/science真实 exit 0，实际墙钟 257.6227 s，0 worker failure，owned process tree 终态为0。实际执行源码摘要为`8ddd3af9d3506448f543cf668f625e5952a3a2d048539f9133ef6b021c774226`，消费原leave-BI checkpoint `fc8d8cbd...`；原`4b075a2d...` source archive及producer保持自己的来源。

主1287个同key的逐episode census保留原始R分母：1243个可用，44个incomplete（subject34一个、subject52共43个）。宏概率覆盖为98.851%，micro覆盖为1243/1287=96.581%，两种权重分列。实际保留的C拟合标签为57–372，原raw首五episode安排为60–372。subject57/selection23完整R1进入主三臂指标，三臂均错；额外R2保持独立secondary null-probability账本。

| arm | participant-macro hit（全部1287请求） | participant-macro NLL（1243有概率） |
|---|---:|---:|
| source_zero_shot | 0.0985773716 | 3.4449344201 |
| source_full_fine | 0.2407763003 | 3.0339166286 |
| matched_scratch | 0.1318186686 | 3.6243966579 |

`source_full_fine−matched_scratch` 的 paired hit 差为 `+0.1089576317`，以每人的全部requested key为分母后跨61人平均；paired NLL差为`−0.5904800293`，在共同1243个有概率key上先人内再人际平均。全微调救回205段、损害70段，概率覆盖没有损失。这是本轮BI Q1公开校准开发读数，episode-budget与前述B12 label-K面板分别记录。报告位于云窗口主树`tmp/public_dataset_queue01/BI2014a/report.json`，queue/actual exit分别为其上级目录`queue_terminal.BI.json`和`queue.actual_exit.BI.json`；本地完整模型压缩包及云报告保持原位置，见[终态绑定](../assets/research_resume_20261011/bi_q1_queue_acceptance_binding.json)。

独立审阅在本地流式核验完整676个归档成员的SHA与字节数、61×3模型/receipt、请求/实际fit身份及全部保存概率，未加载模型或重算EEG评分；三臂宏hit/NLL/Brier及配对效应与原报告一致。审阅另核实际529成员执行源码8ddd与原checkpoint/source角色，见`tmp/BI_Q1_ACCEPTANCE/verification.json`。

新consumer审阅复现BI外部descriptor与actual timeline错映缺口；修后builder/load共用label-free schedule compiler，逐字段核episode、global evidence rows、R/censor和完整horizon prefixes。五个旧门反例先RED，再由新门拒绝；相关52项调用链验证及最终8项BI入口验证通过。原封库模型、缓存与报告保持原字节，原61cohort的实际映射另作零拟合只读核验。

四个旧zero-shot分片在fit前因16-channel/causal缓存与source合同不符而拒绝，新增fit为0。新builder正接显式profile/channel owner；实际source的同名双向profile为`baseline=none`，个人默认常量的`mean_only`另保原用途。校准raw先隔离再零相位处理，缺FZ/PO8保slot mask且CZ必须实测；三臂调用继续核同architecture、固定mass风险及fresh-load重放。已完成22文件固定来源的独立审阅和零fit反例，完整本地handoff在`tmp/BI2014a_Q1_AUDIT/independent_audit.zh.md`，摘要及SHA并入上述队列计划。

## 逐任务horizon接口

指标owner已支持明确的完整episode模式：传入`requested_evidence_levels: Mapping[DecisionKey, int]`，省略scalar `evidence_level`。计划必须包含全部原requested keys，包括typed失败、samples缺失和两臂都没有输出的任务；每项保其真实R。指标采用`n2p3_candidate_probability_metrics/2`及`endpoint=full_episode`，保存完整不可变horizon计划和绑定原DecisionPlan的digest。原fixed-R入口保持v1结构。

outcome owner按每key一个R生成cell，各R组分母分别为其任务数，整体用原任务数；participant/seed层级继续保原权重。例如R1/R1/R8三段分别正确、平局、缺失，整体命中为1/3，各R均值再平均会错误得到1/4。完整平局保概率进入NLL/Brier，失败和缺失保operational分母；没有概率的整人/seed保原风险可用性。paired compare核完整计划，包括共同缺失key的R。

作者107项直接相关验证通过。root逐读两个owner和调用路径，核四个固定body、patch与验证日志SHA后接受该软件接口，见[接受绑定](../assets/research_resume_20261011/variable_horizon_acceptance_binding.json)。BI reader/caller继续负责原raw scope、预算、requested horizon及保存重放关系；本接口不从模型结果选择这些字段。
