# 现有公开数据能否扩入域迁移：文件、合同与调用链审计

日期：2026-09-08。范围：本地数据、当前源码、官方分发元数据和无训练接口探针。
结论：现有四库不是数据能力上限，也不以用完全部本地库为目标。GTN和BI2013a保留候选价值，
双人合作/竞争库不进入默认主矩阵，优先补单人视觉P300的独立来源。
当前四库实验程序不能只追加路径就扩库。文件完整、接口可运行、科学准入、
迁移收益分别判断；本次没有训练、预测、选模型或改变现有实验配方。

## 1. 现状和逐库判断

当前四库为BI2014a、BI2015a、BNCI2014-008、Huebner2017，合计123名已准入被试。
常规外折每次留出一库，实际是三库source训练、一库target评价，并非每折四库全作source。
四库历史结果及checkpoint保持各自执行身份。

下表的通道是六槽`Fz/P3/Pz/P4/PO8/Oz`中的实测Cz差分，不是原始导数。
候选库人数来自分发及加载器，不代表已完成全队列质量和作者准入审查。

| 候选 | 本次本地核验 | 六槽中的有效差分 | 判断与未完成项 |
|---|---|---|---|
| GTN | 248份NIX；247份有匹配元数据，去掉2个重复身份后245份；年龄7-17岁 | Fz、Pz | 可作为binary ERP辅助源及自身九选一开发目标；补共同参考依据、全量新合同cache与独立增量比较 |
| BI2012 | 25个ZIP，对应25人；代表性原始MAT可读 | P3、Pz、P4 | 可扩源；Fz是ground，必须排除，不能把它当实测EEG；保留training/online角色与单位/QC审计 |
| BI2013a | 73个ZIP，对应24人；代表性原始MAT可读 | P3、Pz、P4、Oz | 可扩源；按人保留多日session及adaptive/nonadaptive、training/online条件，不能把73包当73人 |
| BI2014b | 19个双人组ZIP，对应38人；代表性Solo MAT可读 | P3、Pz、P4、PO8、Oz | 仅各自Solo条件作为后备；现有MOABB未加载Collaboration。双人协作不进入默认扩源 |
| BI2015b | 删除前核到22个双人组ZIP、44人；代表性原始MAT可读 | P3、Pz、P4、PO8、Oz | 本轮随后按用户明确要求删除本地分发，不进入扩源池；删除前的核验仍是历史事实 |
| Mainsah S2 | 本地24人、288组原始/转换文件；现有完整准入证据仍有效 | 全六槽 | 保持外部评估角色。方法冻结并完成外部评价后，可另训包含它的部署模型，不能继承未见Mainsah的评价身份 |
| MNE-llp-data | 本次8/8 ZIP中央目录读取均为BadZipFile | 不适用 | 不使用损坏分发；Huebner NEMAR是同一研究的替代分发，不能计为额外库 |

另四个BI库全部139个ZIP，本次完成CRC逐成员检查、SHA256计算，
并通过官方Zenodo的文件大小和MD5比对：**139/139**。
ZIP内部完整和官方字节一致仍不证明每条EEG有效，也不替代作者剔除、条件和标签审查。
四个BI原始MAT探针各检查一份文件；没有将代表性读取写成全库波形认证。
BI2012代表文件存在很大的全记录峰峰值，后续需定位异常时间支持，不能凭能读取就跳过QC。

官方描述与实际解析也不能互相代替：已安装MOABB的BI2015b静态元数据含50人及BI2015a
描述残留，而可执行subject_list与当前官方分发均为44人。准入不能照抄这段静态metadata。

## 2. GTN为何可用，又为何不能直接拼入旧配方

公开source只需要可靠的target/non-target标签。GTN用实际刺激数字是否等于thought digit
生成二分类标签，儿童人群、数字刺激和较少通道属于需要测量的域差异，不构成永久拒绝理由。
它也不能代替成人BrainSync的独立target-switch确认。一个selection前缀标注、后缀测试，
只能解释为同selection的binary开发或oracle proxy，不能改称产品校准后猜新数字。

全量NIX/元数据扫描得到：

- 245个可读非重复身份均为Fz/Cz/Pz/EOG、单位V、1000Hz；实际刺激词表均为1-9。
  总计50566个排程数字刺激、5581个target。这是排程数，不是最终训练epoch数。
- 205份硬件低通40Hz，40份45Hz，记录的硬件高通均0Hz；仍保留各自上游采集身份。
- 19份明确写nose参考，226份未写；不能给后者补造nose、耳垂或Cz元数据。
- Experiment_611缺少与NIX内部身份匹配的TXT；531重复530的内部身份，644重复643。
  当前builder默认会拒绝这些问题；显式排除和账本必须随全量构建保留。

若各EEG与实测Cz共享采集参考r，且上游各导采用相同线性处理L，则

`L(v_c - r) - L(v_Cz - r) = L(v_c - v_Cz)`。

因此，原参考具体位置未写全并非数学上的永久障碍；但必须补共同参考及上游处理的证据，
不能把适配器接受一个非空reference字符串当成这项物理事实已经获证。
现有GTN导入保留Fz/Cz/Pz，EOG仍在原始NIX中；没有执行或声称已完成EOG清理。

本次用Experiment_341和344走当前`build_gtn_epoch_dataset`，预处理取`gtn_preprocessing_for_cohort("causal")`，再经
`adapt_observed_montage(reference_channel=Cz)`、binary projection及study membership：
得到289个epoch，原生`(289,3,179)`，适配后`(289,6,179)`。
标签251/38保持不变，Fz-Cz和Pz-Cz逐值精确一致，缺失四槽全零且mask=false。
study membership恢复为GTN。它证明接口可行，没有证明模型有效或全库已准入。

保留六槽时GTN的mask为`[1,0,1,0,0,0]`；其他合法源继续贡献其实际导数。
不能为了纳入GTN，把所有库裁到严格交集：扩展池在当前Cz空间只共同保留Pz-Cz，
会丢失其他源的空间信息。单独GTN训练六槽source也被生产覆盖门禁正确拒绝：
`source fit has a channel with no real observations after exclusion`。
GTN单源控制应声明自己的可观测布局或另行验证可变输入架构，不能解除门禁伪造四导已训练。

## 3. 当前代码需要泛化的边界

1. `prepare_multidomain_source.py`、`data.domain`已有任意数量源的binary投影、缺导mask、
   Cz差分与provenance路径；无需为GTN复制另一套基础预处理。
2. `run_source_training_chain.py`目前明确要求普通外折恰好3个source域，整族折恰好2个。
   新库进入该runner前要由实际计划解析源集合、排除集合和合法布局，不能只改为允许更多整数。
3. `source_selection.SOURCE_CHAIN_LAYOUTS.common_three`是P3/Pz/P4，runner要求每行全观测。
   GTN缺P3/P4，因此现有E2/S2三导分支直接不满足合同；需选择适合缺导池的声明实验路径，
   在每个outer/inner折检查剩余source的通道覆盖。
4. S2固定剂量现在是多源每库每人128、单源每人384，且每库8人，依赖三源假设。
   扩库必须重新声明总唯一trial剂量和optimizer更新预算，不能把增加数据/更新误当纯域多样性收益。
5. 双人库的participant仍要分别标识，但source内层划分及不确定性估计还需要显式dyad依赖组。
   当前source内层按participant authority分组；不自动保证同dyad不会跨train/validation。
   若跨年份参与者重叠未知，dataset namespace也不能证明生物学身份独立，需补证据并保留BI整族留出。
6. 增加BI家族库后，“每库等权”会增加该家族总风险权重；域权重、类别权重与采样预算
   须分别记录。GTN与不同BI条件的target比例不同，不能假设统一1:5或仅靠目录大小配权。

这些是扩源实施需求，本次没有修改训练源码、放宽身份/空间门禁或重标历史模型。

## 4. 建议的下一轮比较

优先完成Chailloux2020、EPFLP300与本轮补入的Speier2017的科学准入，再评估Cattan2019和BrainForm。
本地BI2013a保留扩源优先级，GTN以单独增量进入比较；BI2012先补信号质量审查。
在共同的原四库伪目标、相同actual K=0/200/800/2000、校准/test行及训练预算下，
分别比较加入一个新源前后的变化，并保留新池加入/不加入GTN的配对。
每次都从source中完全排除当前目标库，不默认加入两个双人库。
固定更新比较与固定总唯一trial剂量比较单列，target-only scratch和原波形收缩LDA继续保留。
这不是新增训练排程；全库准入、合同泛化及同配方对照通过后才具备执行条件。

数据集数不等于独立实验室数。Mainsah继续保留外部角色。
扩展开发证据应同时报告逐库/逐人差值、负迁移、失败与覆盖，
新库泛化和BI整族外推分别解释。已有观察的开发数据不重新称作冻结确认集。

## 5. 新的单人视觉数据

新增分发和网络恢复证据保存在本次运行目录，完成状态以
[逐文件下载账本](../../experiments/runs/dataset_expansion_audit_20260908/additional_downloads.json)及
[EPFL账本](../../experiments/runs/dataset_expansion_audit_20260908/epfl_download.json)为准。
没有把下载当成训练准入，也没有调用模型给新库评分。

最终四套所选分发均已完整下载：2428个文件，4536021765字节，约4.22 GiB。
EPFL 8个ZIP、Chailloux 2364个raw/BIDS对象、Cattan 48个MAT/说明文件、BrainForm 8个发布文件。
全部保存SHA256；Cattan/BrainForm同时匹配官方MD5，Chailloux匹配清单ETag并使用If-Match，
EPFL及BrainForm ZIP完成CRC。可分发的紧凑证据见
[获取清单](../assets/dataset_expansion_20260908/acquisition_manifest.json)；原始信号留在`mne_data/`。
Zenodo内容API曾限流，失败尝试账本保留；最终使用官方公开文件链接，按已观察到的133次/分钟
限额控制请求速率并尊重Retry-After，保持TLS验证，复用已验证分段后2420/2420成功、0剩余失败。

| 数据集 | 公开分发与空间覆盖 | 选用判断 |
|---|---|---|
| EPFLP300 | 8人，4名健康成人及4名残障参与者；32 EEG，Cz六槽实测5槽，缺PO8 | 优先准入；单人六选一图片、4次session/2天，保留健康状态与实际会话 |
| Chailloux2020 / ds003190 | 19名健康成人、8 EEG，覆盖产品七导及Cz六槽全部位置；三次session、4-9选项 | 优先准入；空间和单选任务较接近产品，需要修正有证据的metadata缺失并声明刺激条件 |
| Cattan2019_VR | 21人、16 EEG，完整PC/VR两条件MAT版；Cz六槽实测5槽，缺PO8 | 补充源；两条件属于同一人、同一研究，且仍与BI家族共享实验室/范式背景 |
| RomaniBF2025ERP / BrainForm | 发布编号P01-P24，默认工具排除未完成全流程的P15/P18；8 EEG，六槽实测Fz/Pz/PO8/Oz | 补充消费级设备域；只用有可靠真值的校准记录，保留失败校准供同一质量规则审查 |

EPFL的8个原始ZIP共2208626585字节，均完成CRC和本地SHA256。
全部192份原始MAT已实际通过已安装parser：数值有限、两类标签齐全、排程标签数量保持一致。
这不是完整预处理后epoch准入数。其parser使用实测MA1/MA2平均参考，并去掉记录尾部补零；
generic MOABB静态`reference=None`不能代表实际输出参考，需要明确的生产来源声明。

Chailloux官方对象清单包含384个BrainVision记录，选取raw及BIDS资料共2364个对象，
采用对象ETag匹配、条件请求和SHA256绑定下载身份，不假称统一冻结OpenNeuro版本。
实读结果：382份有明确电极名、数值有限、两类标签齐全；2份只有`chan001..chan010`，
现有parser会把全部通道丢弃并报错，尚未准入。这2份分别为sub-019的ses-01/ses-03、
task-ctos_run-5H。原README声明382份记录，与有明确电极名的文件数一致。

该库全部384份侧车把EEGChannelCount写为0，通道角色与单位均为unknown；这些不能照抄为
真实物理角色。原README/BIDS写右乳突M2，原论文写右耳垂，两条证据同时保留，
不能任选一条覆盖另一条。共同参考加实测Cz可在已核共同采集/线性处理前提下消去参考位置。
原论文还明确采集端0.5-60 Hz及notch；额外0.1-30 Hz软件滤波不能恢复已失去的低频信息。
CNOS的4/6/7/8/9选项记录采用此前5选项比较中成绩更高的刺激类型；这是原采集设计的一部分，
不能把这些记录当作未经任何个体条件选择的固定刺激实验。

Cattan选择完整42份实验MAT及6份header/questionnaire资料，不重复下载等价的逐采样CSV版。
42份均有两类刺激标记。41份全矩阵数值有限；`subject_01_VR.mat`的最后一个采样行
（0-based 289814）有部分EEG非有限值，其他采样有限。最后刺激距有效记录尾17.65625秒，
超过当前1.2秒响应支持，显式舍弃这一不完整尾帧可以保留原本完整支持的因果epoch；
当前只记录该事实，原MAT未修改，通用cache导入仍需实现并记录相应有效采样边界。
PC/VR不能按名字假定采集先后。现有MOABB将原连续MAT切成block/repetition小Raw，
新因果cache需审查这种切分的滤波状态与原始时轴，不能默认重置状态仍等价于连续处理。

BrainForm保留作者发布的BIDS、失败校准、原始CSV及pilot原件。无真值推理段不能因为
事件码也出现1/2就冒充target/non-target监督。也不能按原分类器校准成败直接删掉训练记录。
生产身份必须绑定原始Pxx，不能把排除部分人后重新enumerate的0..21当作不变的来源身份。
BIDS.zip实际有23个原始编号、61份events，失败包有22个编号、104份events，两包并集为24人。
事件表用T1-T10表示刺激身份；现成加载器按前600事件切校准并重新映射标签。
这一步需要结合原始CSV的校准/推理标记和真实时间支持独立核验，不能把整张events表当二分类真值。
上述标签与转换问题已在本轮后续补核，详见第6节；全队列质量准入仍未完成，保持辅助源角色。

进一步候选：Lee2019 ERP的官方对象清单核到108个MAT、99909771701字节，即93.05 GiB，
含54名健康成人的两次会话；当前D盘初始可用64.00 GiB，本次没有启动其大规模下载。
Speier2017原文确认10人、32导、256 Hz、左耳参考。最初Dataverse目录/API返回403，
本轮后续已通过NEMAR镜像完成有标签原始DAT下载，并核对原站单文件，详见第6节。
其32导也并非覆盖所有产品电极，原配置缺P3/P4。
Sosulski2019/Kojima2024B为听觉范式，不进入当前视觉主线；Simoes2020是预切epoch分发，
不能直接满足当前连续因果滤波合同；ERP CORE P3的按键任务暂列次要辅助研究。

新来源：

- EPFL：[作者分发](https://documents.epfl.ch/groups/m/mm/mmspg/www/BCI/p300/subject6.zip)，
  [论文DOI](https://doi.org/10.1016/j.jneumeth.2007.03.005)。
- Chailloux：[OpenNeuro](https://openneuro.org/datasets/ds003190)，
  [已核原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7765532/)，DOI `10.3390/s20247198`。
- Cattan：[官方分发](https://doi.org/10.5281/zenodo.2605205)。
- BrainForm：[官方分发](https://zenodo.org/records/17225966)，论文 `arXiv:2510.10169`。
- Lee2019：[官方数据DOI](https://doi.org/10.5524/100542)，论文 `10.1093/gigascience/giz002`。
- Speier：[已核原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5391014/)，
  [数据DOI](https://doi.org/10.7910/DVN/PHHHB6)。

## 6. BrainForm补齐、通道排除、Speier镜像与Lee体积

### BI2015b删除

按用户明确要求删除`mne_data/MNE-braininvaders2015b-data`，实际删除17376个文件、
9092161086字节，约8.47 GiB。删除前校验目录边界、无reparse point并保存22个ZIP的摘要，
删除后确认目录不存在，见[删除清单](../assets/dataset_expansion_20260908/bi2015b_removal.json)。
其他原始分发未删除。

### BrainForm原始证据与输入视图

本轮完整实读168份主实验原始CSV，覆盖P01-P24，不采用MOABB重排后的整数身份。
166次完整校准各600刺激，另一次P23部分校准有373刺激，一份P01记录没有刺激。
有标签的167份中，61份在正式目录、106份在失败校准目录；不按原分类器成功与否筛选。
7份`PILOT_Txx`也完成数值与标记盘点，但与主队列的身份重叠及pilot协议未获证，暂不计为额外7人。

作者公开仓库固定到`8e7ffefd90f1e7b1fe7d289165237069ec6c3649`。只做静态读取，未执行SDK DLL。
`ERPAutotrainingPipeline.SetTrigger`确认第9-14列依次为触发、刺激编号、target、nontarget、
轮次和轮末；`FlashController`仅在训练模式给训练目标置target。全部非空主记录的
校准前缀都通过候选1-10顺序、轮次、目标T1及轮末一致性检查，推理段不能将nontarget列当真值。
SDK接线同时确认raw logger位于notch和分类器预处理之前。

共恢复99973个校准标签（9998 target、89975 non-target），并生成167份逐事件TSV。
165份有事件的BIDS EDF与对应原始CSV逐样本比较通过：CSV按微伏乘`1e-6`成为V，
误差不超过EDF半个量化步长，原始到BIDS的采样偏移均为1245，即4.98秒。
其中一份P01失败校准的BIDS只剩594事件；原CSV保有600事件，剩余594事件与波形也已对回。
另两份P02完整校准未进入BIDS，已从CSV补入索引。

直接从开机首样本进行0.1 Hz因果滤波会把启动伪迹传播到早期epoch。
输入视图因此显式沿用作者分发的1245采样起始裁切，保留原始采样编号，再调用当前
`data.preprocess.preprocess`的因果IIR、179点FFT重采样和均值基线。
该裁切使上述P01前8个刺激缺完整基线，P23部分校准的末12个刺激缺响应尾窗；
最终99953个时间窗通过实际预处理（89957 non-target、9996 target），20个损失仍在事件账本中。
这不是伪迹QC全通过，也没有将恢复出的6个早期标签当成裁切后仍有完整波形的证据。

按用户允许排除坏通道的决定，已在输入视图写入四个记录/通道排除项：

| 原始记录 | 排除导联 | 波形依据 |
|---|---|---|
| P02，20241021_120406，cb失败校准 | Fz、Pz | 原生相邻样本跳变分别约313 mV、173 mV |
| P14，20241105_125757，cb失败校准 | Pz | 裁切后Cz差分epoch峰峰值P95约813 uV，局部严重异常，保守排除 |
| P03，20241021_130354，grain失败校准 | Oz | 裁切后Cz差分epoch峰峰值P95约510 uV，局部严重异常，保守排除 |

其余记录不据此统一删Fz/Pz/Oz。上述处置基于波形，不依据标签预测成绩；排除清单也不认证
未列通道无伪迹。原生八EEG仍在原始CSV中；Cz六槽通常为`[1,0,1,0,1,1]`，
三份异常记录再按其排除项置false，至少保留两个实际差分导联。
参考/ground具体左右侧在原始BIDS未记录，论文仅明确乳突位置，MOABB声称右乳突参考。
保留证据层级，后续仅按共同参考前提构造实测Cz差分，不把右侧声明改写成已经独立确认的事实。

事件视图位于`mne_data/MNE-RomaniBF2025ERP-data/derivatives/calibration_index_20260908/`。
[结构化补齐证据](../assets/dataset_expansion_20260908/brainform_admission.json)绑定原件、事件表、
波形对齐与源码摘要；[排除清单](../assets/dataset_expansion_20260908/channel_exclusions.json)同时记录
Chailloux两份匿名通道记录不进入输入，以及Cattan只应舍弃不完整尾采样而非整导。
当前只完成输入视图和无训练探针，通用runner尚未消费这批视图，仍需逐折artifact QC及源数量泛化。
BrainForm固定T1、固定1-10顺序，不能作为目标切换确认；只保留辅助源与独立增量比较资格。

### Speier2017镜像

已从[NEMAR nm000259 v1.0.4](https://github.com/nemarDatasets/nm000259/tree/v1.0.4/sourcedata)
下载全部60份有标签原始DAT、共506863680字节（483.4 MiB），覆盖10人、FF/Inv两条件、每条件3次训练。
逐文件大小、SHA256全部与镜像provenance一致；`Subject_1_FF1.dat`另从Harvard原站下载，
SHA256与镜像逐字节相同。20份无真值Online记录不在这个监督子集中。

全部60份实际通过已安装BCI2000 parser；头部32导名字、256 Hz、gain=1、offset=0确认，
数值有限，无整导恒值。按`StimulusBegin`上升沿取`StimulusType`，共72240刺激，
60200 non-target、12040 target；parser未改变标签、触发位置或单位。
不要用延迟的`StimulusCodeRes/StimulusTypeRes`作为原始刺激时刻。
该库也缺P3/P4，Cz六槽实测Fz/Pz/PO8/Oz；后续仍需连续预处理与逐折QC。
见[获取及实读证据](../assets/dataset_expansion_20260908/speier_acquisition.json)。

### Lee2019有用数据的体积

最初只查清单和说明；随后用户要求下载，已启动固定v1.0.4的全部连续ERP BDF及配套元数据，
216份BDF、1845个对象合计41478547576字节，约38.63 GiB，支持断点续传和校验后完成标记。
当前状态以本地`experiments/runs/dataset_expansion_audit_20260908/lee_download_progress.json`为准，
完整SHA256/Git blob复核结束后生成`lee_download_result.json`和`doc/assets/dataset_expansion_20260908/lee_acquisition.json`。
93.05 GiB本来就是ERP专属MAT，
不含MI和SSVEP；训练和测试都有监督标签，均有用。

| 分发/子集 | 文件数 | 已核字节数 | GiB |
|---|---:|---:|---:|
| 原始ERP MAT，session1，54人 | 54 | 49998882139 | 46.57 |
| 原始ERP MAT，session2，54人 | 54 | 49910889562 | 46.48 |
| 原始ERP MAT，两会话 | 108 | 99909771701 | 93.05 |
| NEMAR连续ERP BDF，两会话、训练与测试 | 216 | 41462816664 | 38.62 |

[NEMAR nm000323 v1.0.4](https://github.com/nemarDatasets/nm000323/tree/v1.0.4)的BDF分发更省下载量，
另需少量事件/通道元数据；其转换波形、单位、时间轴尚未与MAT实读对照，不能当作已完成准入。
若依当前六槽、179采样、float32形成epoch信号，名义估计为
`54*2*(1980+2160)*6*179*4 = 1920827520`字节，约1.79 GiB。
这是未压缩信号张量估计，不是实测完整cache，未计元数据、掩码、边界损失与QC；
也不能据此声称只需下载1.79 GiB。见[体积与逐文件清单](../assets/dataset_expansion_20260908/lee_storage_options.json)。

## 7. 验证与证据定位

本地探针及原始检查账本：

- [audit.py](../../experiments/runs/dataset_expansion_audit_20260908/audit.py)
- [audit.json](../../experiments/runs/dataset_expansion_audit_20260908/audit.json)，含NIX/ZIP摘要、
  逐文件官方校验、GTN元数据、代表性parser读数和9份被审计源码SHA256。

当前dirty工作树的base commit为`f54e45c48014d05a71d9644b0cecd7aa041f873f`，
实际读取源码以JSON中的逐文件摘要为准，不能把base commit当全部执行代码。
项目venv运行GTN、domain adapter、multidomain、study membership及MOABB相关测试：
103项中102通过、1跳过，Windows符号链接创建不可用；未训练模型。
本次审计脚本及读取的数据模块Ruff通过，`git diff --check`通过。
后续补齐使用当前venv另跑预处理、manifest、domain adapter和MOABB合同测试，
119通过、1因Windows符号链接创建不可用而跳过。167份校准事件表逐份回读校验通过，
Speier全部60文件实读通过；这些仍不是模型训练或真实迁移收益证据。
Zenodo旧API随后超时，改用官方`/records/{id}/export/json`完成139份MD5/大小核验。

公开来源：

- BI2012：[DOI 10.5281/zenodo.2649069](https://doi.org/10.5281/zenodo.2649069)。
- BI2013a：[DOI 10.5281/zenodo.2669187](https://doi.org/10.5281/zenodo.2669187)。
- BI2014b：[DOI 10.5281/zenodo.3267302](https://doi.org/10.5281/zenodo.3267302)，
  [作者解析示例](https://github.com/plcrodrigues/py.BI.EEG.2014b-GIPSA)。
- BI2015b：[DOI 10.5281/zenodo.3268762](https://doi.org/10.5281/zenodo.3268762)，
  [作者解析示例](https://github.com/plcrodrigues/py.BI.EEG.2015b-GIPSA)。
- [当前四库证据](four_study_launch_20260907.zh.md)、[Mainsah完整准入](mainsah_s2_admission_20260907.md)。
