# W0 现时资格盘点（限定只读）

2026-09-16。只读已有JSON/TSV元数据、索引及adapter；原始EEG、NPZ、ZIP/NIX/BDF/EDF内容均未读取，未重算原始文件hash，未重建cache或训练。每库只核验一个代表性原始路径与列明sidecar的存在性。已有SHA是历史索引声明，本轮未重新验证字节；unknown不被补写成认证。

**所有新共同profile的actual-K与完整group数均为pending，当前没有可据此确认启动的统一模型矩阵。** 五折定义取自`experiments/runs/domain_transfer_cloud_20260916/matrix_plan.json`；其远端旧cache和执行任务未修改，也未连云验证。

| 数据 | 当前读到的索引/adapter | 实测公共导证据与参考 | 新profile/decision资格 |
|---|---|---|---|
|BI2014a|experiments\runs\source_eog_20260909\local_prepared\BI2014a\preparation.json|P3,PZ,P4,OZ,CZ；right earlobe|pending；候选adapter已存在；binary索引不能认证新profile下完整候选组|
|BI2015a|experiments\runs\source_eog_20260909\local_prepared\BI2015a\preparation.json|P3,PZ,P4,PO8,OZ,CZ；right earlobe|pending；已见二分类标签；本轮未核实可执行候选membership恢复|
|BNCI2014-008|src/data/bnci2014_008_candidate.py|FZ,P3,PZ,P4,PO8,OZ,CZ；right earlobe|pending；已有物理行列membership adapter；新raw范围及完整组数待生成|
|Huebner2017|experiments\runs\source_eog_20260909\local_prepared\Huebner2017\preparation.json|FZ,P3,PZ,P4,CZ；nose|pending；已有Target/NonTarget事件parser；本轮未核实候选membership恢复|
|GTN|experiments\runs\source_eog_20260909\local_prepared\GTN\preparation.json|FZ,PZ,CZ；nose|pending；已有九候选adapter；逐记录thought/stimulus及完整组须随新scope重建|
|BrainSync|experiments\runs\source_eog_20260909\inputs\P001_20260904_171926_01\session.json|Fz,Cz,P3,Pz,P4,PO8,Oz；left earlobe|pending；session v4九候选；本轮未重验thought及逐事件有效性|

## 五折与名义成本

- exclude_BI2014a：target=BI2014a；排除=['BI2014a']；family_sensitivity=False。
- exclude_BI2015a：target=BI2015a；排除=['BI2015a']；family_sensitivity=False。
- exclude_BNCI2014-008：target=BNCI2014-008；排除=['BNCI2014-008']；family_sensitivity=False。
- exclude_Huebner2017：target=Huebner2017；排除=['Huebner2017']；family_sensitivity=False。
- exclude_BI2014a_BI2015a：targets=BI2014a、BI2015a（同一源折的两目标）；排除=['BI2014a', 'BI2015a']；family_sensitivity=True。

8种源因素状态×5折×3 seed=120，仅是条件规划名义数，可作这张有限排程的预估上界；不是可启动资格证明，也不是包含源内调参/refit/失败重试的总成本上界。BI家族敏感性折不作为第五独立库进入等权均值。

## 已核验的有限样本路径及历史摘要

### BI2014a

- 路径：`D:\BCI\n2p3-net\mne_data\MNE-braininvaders2014a-data\zenodo\3266223\subject_01\subject_01.mat`；exists=True；已有SHA256=`eecb29c236f21b5d8d1b8b3cc07e4aa96b107eeb29033201fcab407d94fdd759`；member=None。
- 旧preparation索引列出61条记录、61个subject标识；仅描述历史索引，不是新准入人数。
- 阻塞：单记录prefix须先分raw再filtfilt并执行支持purge；masked stem/BN共同基线和新候选组数尚未落实。

### BI2015a

- 路径：`D:\BCI\n2p3-net\mne_data\MNE-braininvaders2015a-data\zenodo\3266930\subject_02_mat.zip`；exists=True；已有SHA256=`2be09de7dbf7351e9d66a020a6ec8780d4ddbaee78797348d2719390cbf5126e`；member=subject_02_session_03.mat。
- 旧preparation索引列出123条记录、41个subject标识；仅描述历史索引，不是新准入人数。
- 阻塞：保留archive-member摘要声明，本轮未复核压缩包/成员字节；末会话隔离及新滤波/QC实际K待落实；membership未证实时D不可用。

### BNCI2014-008

- 路径：`D:\BCI\n2p3-net\mne_data\MNE-bnci-data\~bci\database\008-2014\A01.mat`；exists=True；已有SHA256=`unknown`；member=None。
- 阻塞：本轮所读元数据无raw SHA，重建前须完成字节准入；单记录prefix须先切raw；完整行列组及purge计数待生成。

### Huebner2017

- 路径：`D:\BCI\n2p3-net\mne_data\huebner2017_nemar_nm000199_v1.0.2\sub-1\ses-0\eeg\sub-1_ses-0_task-p300_run-0_eeg.bdf`；exists=True；已有SHA256=`c4e72cb9d5693215b602fb9864129742121ca2ffd07ccaec61728d785877afb5`；member=None。
- 旧preparation索引列出342条记录、13个subject标识；仅描述历史索引，不是新准入人数。
- 阻塞：历史索引未提供sidecar摘要，新准备须认证channels/events；旧六槽mask缺PO8/Oz，masked stem/BN为必需资格；末会话raw隔离及新profile实际K/group数待生成。

### GTN

- 路径：`D:\BCI\n2p3-net\mne_data\MNE-P3-data\Experiment_341_P3_Numbers\Experiment_341_P3_Numbers.nix`；exists=True；已有SHA256=`9d7967c8c2e636e60ea449ab8b68ab2f2961a9cf8929cc5f312c4125188afec3`；member=None。
- 旧preparation索引列出245条记录、245个subject标识；仅描述历史索引，不是新准入人数。
- 阻塞：仅实测Fz/Cz/Pz，只有两差分，不能独立充当完整Cz6源；新profile raw范围及QC后membership/K尚未生成。

### BrainSync

- 路径：`D:\BCI\n2p3-net\experiments\runs\source_eog_20260909\inputs\P001_20260904_171926_01\sub-P001\ses-20260904T17192601\eeg\sub-P001_ses-20260904T17192601_task-gtn_run-01_eeg.edf`；exists=True；已有SHA256=`unknown`；member=None。
- 阻塞：历史P001属于开发资料，不是新成人确认；本轮session字段没有raw/sidecar摘要，须新增字节认证；短记录上下文、raw隔离、QC/EOG拟合范围及操作组覆盖待落实。

## 当前不能省略的落地资格

1. 公共源/目标从raw先分scope，再分别filtfilt；single_recording_prefix两库必须实际枚举purge与完整group，不能沿用旧因果cache的K计数。
2. BI2014a/BI2015a/Huebner历史六槽掩码均存在缺导，固定stem首个temporal BN须明确实测矩统计和affine后remask；不能把这项资格推到集合encoder扩展。
3. GTN不是四库五折矩阵的自动附加源；它只有两路Cz差分，不能独立满足完整六导source覆盖。
4. 新profile下每库每K的源率/滤波能力、参考依赖、raw/sidecar字节身份、QC/EOG拟合来源、真实类计数、membership、完整decision及失败原因逐项补齐。
5. D不可用的binary队列使用R/S/M八格；真实decision队列才有十六格。设计表代数满秩不等于每库实际对比可估计。
6. BrainSync样例来自历史输入快照，当前live采集根/新成人队列未知，不能把该样例当新确认数据。

详细字段和已有source-cache摘要见同目录eligibility_inventory.json。上述pending意味着后续执行须完成相应明确步骤，并不等于删除数据或宣称数据不可用。
