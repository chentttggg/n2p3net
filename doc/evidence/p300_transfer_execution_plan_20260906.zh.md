# 公开ERP预训练辅助成人猜数字：全局执行计划

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

日期：2026-09-06。授权：用户明确公开BI数据主要用于学习P300特征，要求拟定全局计划并
自行依次执行；后续授权至少六小时自主迭代，无需逐项询问。
产品主线是成人BrainSync七路头皮EEG+独立一路EOG、本人已知校准、后续未知目标九选一。
最新预算为九数字各450闪烁，共4050 trial；测试新1 block=45 trial，主hit@R5。
当前真实接口与遗漏已在[技术结构](../technical_structure.zh.md)和[结构审计](technical_audit_20260906.zh.md)
重新核对；以下时间化实验记录仍按执行时口径保留，不自动触发重复云训练。
72项目录是候选库，不是本计划的阻塞清单或必须全跑的矩阵。

## 1. 已决定的路线

1. BI/其他公开ERP作为binary source，学习target/non-target的时空表示及源分类头。
   不要求源数据具有数字身份、固定row/column或完整R8。源candidate语义从活动训练视图
   中移除，保留原始父cache和转换provenance以复核。
2. source participant-disjoint验证及公共pseudo-target校准实验只回答表示/迁移可用性。
   最终收益必须是同等目标标签/时间预算下，预训练优于target-only，并改善真实九选一。
3. 当前简洁基线为原N2P3Net + supervised source CE + source-head保留/微调/full-fine。
   EA、AS-MMD、Split-BN、TSD是后续相同数据/预算下的增量对照，不再预设为默认必要链。
4. montage先走真实共同五导(Cz,P3,Pz,P4,Oz)的可执行研究桥接；它不是完整产品蒙太奇。
   最新硬件槽顺序Fz/Cz/P3/Pz/P4/PO8/EOG/Oz：原PO7槽接PO8，原PO8槽接EOG。
   分类器只消费Fz/Cz/P3/Pz/P4/PO8/Oz七路，EOG保留独立通道用于伪迹处理与QC；
   不擅自把单路EOG命名为VEOG/HEOG。七导上线前需真实source覆盖或已验证的空间接口。
   不用插值/补零冒充缺失电极，不把F3/F5歧义带进五导几何。
5. BI的物理candidate冲突不再阻塞binary pretraining。epoch时间、标签、单位、训练/测试
   身份隔离仍必须成立；源训练按participant分组而非按不确定的selection做监督。

## 2. 目标函数与最小对照

\[
L_S=\sum_d\alpha_d\frac{1}{|U_d|}\sum_{u\in U_d}
\frac{1}{N_u}\sum_i CE(f_\theta(X_{ui}),y_{ui}),
\qquad \alpha_d\ge0,\ \sum_d\alpha_d=1.
\]

域/participant等权是多源对照的显式选择，不应与weighted CE的class mass混同。
首轮单源沿用已验证训练器，实际权重和updates记入checkpoint。
目标阶段仅在calibration拟合，比较：scratch全网、source zero-shot、source classifier_fine、
source full_fine。每个K使用同一测试suffix；K按真实trial/正例/秒数记账，避免源selection假设。

\[
\Delta(K)=M_T(\operatorname{Adapt}(\theta_S,D_T^{cal}(K)))
-M_T(\operatorname{Train}(\theta_0,D_T^{cal}(K))).
\]

公共pseudo-target主M为participant-macro AUC；成人主M为requested-participant operational
九选一准确率及真实correct selections/min，同时报告AUC、coverage和校准曲线。
不能用BI AUC提升替代产品收益，也不能因BI没有九选一标签阻止源学习。

## 3. 顺序与验收

| 阶段 | 执行动作 | 完成条件 | 当前状态 |
|---|---|---|---|
| S0 | 活动指导转向binary source→target adaptation；原指导物理归档；资源/目标数据普查 | 主线、数据分层、资源与阻塞有记录 | 完成 |
| S1 | 修复/复用binary source projection；单源入口；真实五导派生cache与身份保留 | 针对性测试、attested cache、元数据不会触发candidate训练 | 完成；32个相关测试通过 |
| S2 | 真实source监督预训练及公共被试留出校准对照 | source checkpoint、固定suffix的四臂/标签剂量结果、失败账本 | 三seed、强scratch/收缩LDA/源feature对照和同被试五折诊断全部完成 |
| S3 | 加入第二公开数据域；固定source数量/updates比较联合训练/源选择/EA等 | 外部域的target-only/zero-shot/微调配对证据 | BNCI已完成；两人多源pilot、七导与Cz六差分的8人三seed LOSO完成；未声称所有机制已检验 |
| S4 | 七EEG+一EOG输入接口；montage；BrainSync校准与推理接口 | 离线科研输入和无标签输入分别验收 | 新主工作站已完成校准/持久化/inspect/block完成推理，Qt模拟源模型+EOG端到端通过；逐chunk推理未实现 |
| S5 | 独立成人确认、设备replay、性能与概率校准、冻结部署包 | 固定预算/配方的真实九选一与成本；source/data/checkpoint全部绑定 | 软件已可采集与判定，等待真实设备/信号时钟和成人验证，不以模拟或公共开发结果替代 |

S2首轮是本地可执行的development pilot：BI2014a公共被试01–08共同留出，源训练
09–61；62–64按原作者说明不进入该pilot。固定seed=20260906、共同五导、128Hz、
当前causal-v3输入。源先跑小预算检查，再达到预声明16epoch上限（现有source内层早停及
全source refit），记录实际epoch。pseudo-target用前48/96/192个trial作calibration，
共同test从192trial之后且无raw epoch重叠的时刻开始；不是48/96/192个独立样本或决策。
该pilot早已属于被观察过的BI development资料，不能当confirmation。
GPU逐阶段执行，source训练采用既有BF16/compile/fused训练器，先测吞吐再调并发；
target小样本fit避免重复cache解压和无效编译。配置/seed/更新预算不因性能调度静默改变。

S3先普查BI2013a/BI2015a等binary数据及真实通道/参考；没有candidate identity不排除
binary source资格。构建仍复用MOABB、raw artifact manifest和`prepare_multidomain_source`。
若参考不同，native不伪装相同参考；CAR/显式参考变换必须作为对照而非无条件默认。

## 4. 已核实的执行条件

- 本地`.venv`为Torch2.13.0+cpu，CUDA不可用；32逻辑CPU、约16GB物理内存，启动时可用约3GB。
- 原`n2p3cloud`地址westb:10646连接被拒；用户提供的新westc:41560已用现有SSH公钥连通。
  RTX5090 32GB、Torch2.8.0+cu128、cgroup实际25CPU，数据盘约28GB可用；复用项目.venv。
- 首轮检查的旧BrainSync data根可见3个v4 session均为error，target标签pending。
  后续用户提供newest目录，本次结构审计发现4份completed/confirmed候选记录和3份error；
  通道/坐标/时钟与成人准入尚未闭合，见[版本比较](brainsync_versions_20260906.zh.md)。
  不把记录存在等同已经取得有效产品性能。
- 原始BI2014a和attested causal-v3 64人cache在本地；现有binary projection、source
  pretrainer、SubjectAdapter、BrainSync cross-decision runner可复用。

## 5. 代码与反例链

- `data.domain.project_binary_evidence_source_view`同时处理timeline和metadata/provenance；
  反例是timeline无candidate但metadata仍有row/column，不能no-op；X/y/identity不可变。
- `adapt_eeg_domain.py --event-contract binary_evidence`复用上述函数，不另造BI专用投影。
- 公共pseudo-target评估不调用BI candidate parser；通过真实checkpoint loader验证target
  排除、geometry、reference、preprocessing，使用现有SubjectAdapter更新权重。
- chronological calibration/test按真实时间purge；各预算同test；fit不得读取test统计；
  source与scratch采用匹配结构，分别保存origin，不能把随机模型叫预训练checkpoint。
- 源代码物理archive含本轮工作树内容和逐文件hash；source_commit仅标父提交，working-tree
  状态单独声明。训练与评估绑定各自真实执行archive，旧artifact不覆盖。

## 6. 执行记录

本节随完成阶段更新。存在外部阻塞时继续执行不依赖它的阶段，不把未执行部分标完成。
自主窗口始于2026-09-06 01:23:37+08:00，首轮按至少六小时执行到07:30附近。
持续复核GPU/CPU利用率、显存、每阶段耗时、训练日志与科学边界，重复任务只在前一阶段
结果支持时启动；窗口结束保留完整结果、未决问题与可恢复计划，不让计费GPU空转等待。

### 首轮进度（约01:50，development）

云端工作目录 `/root/autodl-tmp/n2p3-runs/p300_mainline_20260906`。
源码A：`frozen/binary_transfer_20260906a.*`，archive SHA256
`25a2ab1b63bbcd3dcfd42bf798bf06d1c5a20af2100333e18cb765e7d1420445`。
源码B：`frozen/binary_transfer_20260906b.*`，archive SHA256
`4325d50f8e025006c78364060a318fb1c21065160445edeaec05bb63a1f686a8`。
B修正EOG软件链并增加强target-only对照、避免重复checkpoint验证；评估archive与源训练
archive分别记录，源模型A无需因评估代码改变重训。

source使用53人、50189 trial；三个seed20260906/07/08，fit耗时21.56/19.49/12.96s，
最佳epoch15/15/12；BF16、fused Adam、train_step compile、preload生效，未OOM。
三seed×8人×3预算×4臂共288cell完成，均为公共binary开发结果。

| source seed | zero-shot AUC | K192 classifier-fine | K192 full-fine | K192 scratch旧小预算 |
|---|---:|---:|---:|---:|
| 20260906 | 0.6689 | 0.6862 | 0.7136 | 0.5194 |
| 20260907 | 0.6629 | 0.6830 | 0.7186 | 0.5488 |
| 20260908 | 0.6376 | 0.6625 | 0.6902 | 0.5056 |

旧scratch采用lr1e-4、15epoch、冻结BN，可能明显欠训练，不能以它为唯一基线。
下一轮显式加入scratch lr1e-3/60epoch/BN训练，以及20Hz特征+shrinkage LDA。
这属于development配方诊断，改变预算要明示，不能把同一dev挑中的最优当确认。

### 适配配方与跨数据域进展

强对照seed20260906：K48/96/192的target-only tuned AUC为0.5885/0.6243/0.7294，
raw20Hz shrinkage LDA为0.6344/0.6800/0.7366。现有source模型保持不变，
full-fine改lr1e-3、60epoch、source统计/冻结BN后达到0.7187/0.7637/0.8255。
因此不能把原弱scratch对照的差全归因于预训练，目标优化配方需要单独诊断。

BNCI2014-008的8份MAT约172MB完成本地/云端SHA256核验（无官方MD5，记录这一限制）；
真实builder生成33600trial、当前causal-v3缓存，再显式投影为native共同五导及完整七导binary。
BI source→BNCI八人原配方zero-shot=0.5075、K192 full-fine=0.5276；target-only tuned
=0.6357、raw shrinkage LDA=0.6486。两域channel_std中位数：BI约29–53uV，BNCI约7–8uV。
这只支持先查简单幅值/统计/优化失配，不证明EA/MMD必有效或某域P300不存在。
BNCI七导source(A03–A08)已训练，A01/A02留出，内部validation AUC约0.7873。

下一组预先列明的development对照：保持epoch60/lr1e-3、相同suffix与calibration预算，
比较source统计、target-prefix统计及target-prefix+BN更新；另加冻结源feature上的
shrinkage LDA。先一个seed定位，再复核其余seed。随后以BNCI-only、BI-only和联合source
比较真实跨域收益；联合训练同时记录样本量、updates、domain风险权重和数据顺序。
宽度扫描复用MST参数2/4/8。执行后源码/权重核查纠正了shuffle开关的解释：原名
`shuffle_each_epoch`仅选择整轮GPU重排或逐batch gather，两者训练样本顺序相同；
width2两个checkpoint权重逐位相同。已将活动代码字段/CLI改名为
`materialize_shuffled_epoch`/`--materialize-shuffled-epoch`，不保留误导别名。
这些带shuffle文件名的历史单元只是执行策略对照，不能作为“是否随机训练”的科研消融。

EOG软件链已加入`fit_brainsync_eog.py`（只读明确calibration sessions）、
`prepare_brainsync_cache.py --ocular-model`（同participant、相同reference、无未来calibration），
以及eval对EOG拟合session不超过head calibration预算的检查。
采用sklearn线性回归估计系数并以冻结calibration均值逐样本减除；MNE现有EOGRegression.apply
会重新中心化整段待处理EOG，不直接用于这里的因果应用。没有强迫native数据改CAR。
这只是软件/合成验证；没有真实EOG清理收益或成人准确率结果。

采集端同期修正了实际工作站的默认槽位、角色、坐标、BIDS 7EEG/1EOG计数与当前配置；
保留REF/GND设置。旧文件位于`frozen/brainsync_producer_before_eog_20260906.zip`，
SHA256 `557375c01445daadf54e0fb593baff2bfc162b2c8aad51ce921af6483689fc92`。
21个producer单元用例运行，16通过、5个需要SDK/pyedflib的集成用例跳过；
实际producer sidecar→MNE EDF→n2p3 consumer的独立跨项目测试已通过。
旧exe未重打包，也未进行真实设备采集；后续采集需使用当前源码。

### 完整产品布局与参考对照

已将观测mask贯穿source预训练：缺失通道不参与统计，标准化后仍保持占位零；
合并相同物理网格时，全局可用mask取并集，逐trial mask保留每个来源的实际观测。
每个模型输入通道必须在最终source训练行中有真实观测，不能靠留出的target数据声称已学习。
BI缺Fz/PO8时仅这两列为明确未观测；BNCI提供其真实观测。这是待验证的联合七导实验，
不是插值或把缺失信号当测量，也不是已经证明有效。

\[
\mu_c=\frac{\sum_{i,t}m_{ic}X_{ict}}{T\sum_i m_{ic}},\qquad
Z_{ict}=m_{ic}(X_{ict}-\mu_c)/\sigma_c.
\]

采集端当前配置REF=T7(T3)，公开BI/BNCI资料声明右耳垂；两者不能重命名为同一参考。
增加共同实测Cz参考的独立对照，输出Fz/P3/Pz/P4/PO8/Oz六个差分通道；
原始采集仍是七EEG+一EOG。每次都要求Cz被真实观测，未观测的目标通道保持mask与零占位。

\[
(s_c-r_s)-(s_{Cz}-r_s)=s_c-s_{Cz}=(s_c-r_t)-(s_{Cz}-r_t).
\]

该不变性只在共同参考和共同实测anchor假设下成立；它可能删除有用共模，必须和native七导比较。
CLI为`--reference-mode measured_channel --reference-channel Cz`，缺导额外显式声明
`--missing-channel-policy mask`。不同观测集合不能各自CAR后假称一个共同参考。
时变原参考抵消、缺失列不得变成负参考波形、不同session坏导mask合并、source mask传入
真实预训练builder等针对性测试已加入。

后续采集参数已按最新用户需求更新`configs/brainsync_collection_plan.json`：每个session
仍只有一个真实目标，九数字各10 block×45刺激，共4050trial/450target；在1.5秒SOA下
刺激预算6075秒。新test每次1 block×45刺激，主R5、R1/2/3/4作诊断，全部九个目标都可用。
27个test decisions是每目标3次的开发起点，不是确认样本量结论。实际时间另计休息、确认、
准备和失败成本。450刺激的长session不得误计成多个不同目标decision。
旧135trial/未见校准数字方案已随清理前快照冻结；严格BNCI结果保持原口径。

参考必须先于EOG回归确定。单路EOG与EEG共享采集REF时，Cz对照对EEG和EOG都先减Cz，
再在calibration上拟合/应用回归；否则先做native EOG回归再换参考，不能声称仍有上述不变性。
`fit_brainsync_eog.py --reference-channel Cz`将该顺序绑定在模型中，缓存自动输出六个
差分EEG通道；原始7EEG+1EOG不变。已清理缓存不能再偷偷更换参考。无参考变化时仍保持
native七导，是否启用清理/参考分支需真实数据验证。

### 结果复核与补充诊断（06:30以后）

首个完整结果包两端SHA256一致：
`271e79be7bc2f78ba1b28bdd4514edaf25f87f919e949eeba1d51a668fd37ea2`。
对83份binary record的1968个预测NPZ逐个重算AUC，全部与记录一致；逐cell核对评估合同digest、
source checkpoint训练participant排除、模型/源码digest、训练测试row不交叉，以及同cache/participant
所有K共享相同测试row与标签。该数字是评估单元数，包含相同zero-shot在不同K的重复，
不是1968个独立实验或被试。

三seed先在每个participant内平均，再作participant级差值统计。BI K192 full-fine强配方
0.8153、tuned scratch 0.7322、raw收缩LDA 0.7366；相对scratch的平均差0.0832，
conditional frozen-source participant bootstrap 95%区间[0.0502,0.1266]。
BNCI七导LOSO K192轻微调0.7989、zero-shot0.7760、强微调0.7364、scratch0.6786。
LOSO source集合彼此重叠，其participant重采样区间仅作描述性稳定性摘要，不能当独立训练程序CI。

共同Cz参考由两人pilot扩到8人×3seed，完全沿用七导LOSO的被试、优化配方和测试suffix。
变化是共同参考变换及降为六个差分通道，不能拆称仅reference的单独效应。

在BNCI原始candidate cache与七导binary cache逐row标签、身份、时间和EEG数值对齐后，
用冻结七导LOSO模型执行ERP先平均探针。只使用共同K192测试suffix中完整的33个selection/人，
原始12个row/column group每R覆盖完整；不是九选一。R=1两路径数值一致。
R3 sum-logit准确率0.4331，ERP先平均0.4167；R10分别0.7412/0.7146。
不因非线性或重复相关就把ERP平均设为默认；训练与测试均值分布不同也是竞争解释。

最后补做BI开发01–08、共同五导、当前causal-v3的同被试随机分层五折及连续时间块五折。
后者按raw epoch支持purge，不宣称消除了无限IIR状态依赖；两者都是离线同session诊断。
N2P3Net/EEGNet/20Hz收缩LDA用相同fold，NN固定60epoch/lr1e-3、外层train统计与class weight，
无source初始化、无test早停。它不是64人16导原版MOABB论文复现，也不当少样本迁移上限。

收尾结果与未决项已整合到[执行结果](p300_transfer_results_20260906.zh.md)。
最终为2400个binary评估单元、240个同session单元、240个ERP聚合汇总单元，72份源权重文件。
6个BNCI全8人source候选完成refit；没有按test选择最佳seed或默认发布产品模型。
BrainSync真实决策runner同步加入target-only tuned/raw收缩LDA，复用binary模块，源参数/统计
污染反例通过；训练与评估快照分开。实际权重→合成BIDS探针还发现默认模板MRI/head框架
错标，已修正采集端默认与当前配置，并以真实producer几何一致性测试覆盖。
当前代码、外部producer与结果包的交付身份见`frozen/p300_execution_delivery_20260906.manifest.json`。
