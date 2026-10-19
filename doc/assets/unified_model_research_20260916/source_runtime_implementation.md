# 统一源目标、批次与任务身份实施

日期：2026-09-16。软件核验记录，未启动真实公开数据或云端训练。

## 联合目标

`SourceObjectiveConfig`使用显式`supcon_weight`、`mmd_weight`、`feature_location`。
`SourceObjective.loss`在一次物理行去重与无参数归一化后返回
`alpha * SupCon + beta * conditional_MMD`；训练器直接加至CE，仅执行同一optimizer step。
SupCon和MMD复用原数学核；两个原始分量、加权总量、有效计数和各自不活跃批次分别记录。
MMD无偏估计可能为负，因此checkpoint加权一致性误差容限按两个分量的绝对量确定，允许抵消。

新训练记录为`n2p3_source_objective/2`；旧`kind/weight`训练入口已移除。
冻结`/1`checkpoint仅在原合同校验路径只读解释，保留原来的未加权mean及post-MST语义；
不生成新训练配方，也不将它重命名为`/2`。默认P2仍为CE/SupCon/MMD三臂。

## 实测mask和实际结构

`MatrixBatchSource`集中持有CPU bool `[N,C]` mask，按真实抽样行对齐stream/preload/shuffle/index/repeated rows。
内部批次接口为`(X,y,weights,domains,channel_mask)`；旧两项`batches`接口不变。
`DeepClassifier`固定更新、epoch训练、两类验证及预测将mask传至N2P3Net/已有wrapper。
普通Braindecode路径不接收额外参数。实测channel packing启用时，原compile策略禁用CUDA graphs并记录原因；
本次CUDA探针未启用compile，不能据此宣称编译吞吐通过。

源辅助位置由实际模型判定：无core残差为post-MST，有残差为post-residual，声明不符在训练前失败。
监督CLI、source selection、registry、resume、outer plan传递完整architecture configuration。
改变observed normalization或残差结构会改变registry键，resume还核对保存checkpoint实际架构。
source checkpoint保存原始预处理执行凭证并将其绑定到training contract；源优化器仍为现有Adam。

## 矩阵与目标输入

显式源阶段`unified_source`生成R/S/M八个状态，全部启用实测channel normalization；
R控制core残差，S/M分别控制0/0.05两级权重。架构、辅助位置、配置和checkpoint实际值相互核验。
`run_domain_transfer_matrix.py --program unified`接同一源执行器与目标worker；P1/P2旧程序保持独立入口。
新source入口要求完整显式PreprocessingSpec和实际执行凭证。

目标`unified`从`load_target_prefixes`消费每K独立raw处理的校准视图与共同固定test视图，
只合并已执行数组，不从Cmax epoch截片制造Ck。每scope记录原组成cache摘要、通道投影和组合摘要，
目标复用键和evaluation contract消费该组合身份。C/T的参考和真实candidate合同须相同；
membership资格由授权校准行上的`calibration_decision_groups`确定，缺少真实membership只运行binary路径。
source因素由真实checkpoint核对，D0仍保留group批次，D1启用decision损失，scratch不承接源中心罚项。

用户随后明确统一改用全局早停。现统一程序复用`GlobalEarlyStoppingPolicy`：默认patience=6、min_delta=1e-6，
source上限32个曝光/验证周期，target上限200 epoch，source验证参与者比例0.2。
源每周期为`ceil(inner_rows/batch_size)`次原分布抽样更新，并非每行恰好遍历一次；
源先按参与者隔离验证，再从同seed独立全源refit选中周期数；
目标先在各K校准raw内部独立重建T/V，选择epoch后完整C refit。source与target保存同一全局policy，互不共享patience计数。
历史固定updates数学探针保留原含义，新矩阵不再将4000/200更新写成实际预算。

`IndependentEpochSelectionData`使用实际receipt哈希和`raw_identity`校验单记录或多记录T/V：
每个原记录的raw区间及原event分配精确覆盖其授权C；同几何的多个raw可以复用一份operator receipt，
原cache组合身份仍保留全部物理输入。所有缺失合法inner的情况留失败记录，禁止自动回退到C中epoch切分。

source内λ选择复用同一参与者split、源trajectory registry及target早停/refit函数。
参考程序为R1/S0/M0；membership可用时用D0同工作集，缺失时D不适用。
参考架构消费主计划同一残差配置，包含显式非默认参数；先从参考source训练及自身早停验证中排除源内留出人，
再由这些人的C200及独立query选择一个公共λ。C200内部T/V只负责选轮，独立query负责比较λ。
候选默认为1e-4、1e-3、1e-2、0.1、1，seed平均后参与者等权AUC，1e-6内并列选小值，任一请求失败令该候选无资格。
K200不足则pending，外层目标不补入开发集。`source_prior_selection.prefix_manifests`显式提供原raw配对缓存；
开发结果绑定完整fold、原cache/manifest/checkpoint/prediction摘要及额外训练成本，再传给全部R/S/M状态。
可在各`outer_folds[*].source_prior_selection`提供该外折的源内manifest列表；只准入当前合法源内的participant authority key。
参考source复用既有registry，λ目标拟合按person/seed在spawn进程间并行，接原GPU信号量和内存准入；
每进程按校验后的manifest只载入本人的cache，避免序列化只读attestation或重复传输全部源矩阵。
同一person/seed中的λ共用只读source初始化，并逐次核验未改变其state。

## 已执行的定向验证

- 联合目标/runtime/source sampling：121 passed、1 deselected，exit 0。
- 原deep模型路径：21 passed，exit 0。
- 本地CUDA联合SupCon+MMD+observed mask+core残差：1 passed，exit 0；不含compile吞吐。
- source training chain/resume/resources/supervised CLI/objectives：120 passed，exit 0。
- 原target/cross runner：20 passed，exit 0。
- 增强架构恢复、CLI双目标/残差及实际raw作用域后：90 passed，exit 0。
- 全局早停源checkpoint→matrix plan→目标worker，以及源内λ开发/真实2worker执行：5 passed，exit 0，33.70秒。
- 复核发现λ参考架构未读取非默认残差声明，已原处改为消费主计划同一配置；最小调用链反例1 passed、5 deselected，exit 0，7.00秒。
- 源生命周期、source objectives、global停止、目标选轮及原目标执行相关套件：152 passed、1 failed；
  失败来自测试硬要求2个worker，但当时可用主存618930176字节，资源准入正确限制为1个worker，全部源任务仍成功。
  对应测试已改为核验实际准入、子进程PID及全部完成/更新数；独立复测2 passed、8 deselected，exit 0，52.82秒。
最终本模块Ruff与git diff --check通过。

初始RED包含3项源接口和9项mask接口缺失。早期固定预算合成链曾验证K0/K4/K8七个binary单元；
预算修订后新增全局早停源checkpoint→矩阵计划→共享目标worker的实际合成链验证，K0/K16四个单元成功。
Ruff和git diff --check按最终模块状态复核。上述数字来自对应时点，不代表并发工作树已完成全量验收。

## 待整体验收的边界

未测真实EEG泛化、成人设备表现及编译吞吐；源内lambda选择器仍需要实际公开数据的独立合法开发记录。
全局早停和λ探针使用合成raw及短epoch上限；实际公开数据矩阵仍需准备完整raw范围、独立源内开发与执行快照。

## 本模块主要文件

生产接口：`src/transfer/source_objective.py`、`src/train/runtime.py`、`src/baselines/deep.py`、
`src/research/source_selection.py`、`src/research/source_resume.py`、`src/research/source_matrix.py`、
`src/transfer/target_experiments.py`及`src/transfer/epoch_selection.py`的独立raw视图接入。
入口：`run_pretrain_supervised.py`、`run_source_training_chain.py`、`run_domain_transfer_matrix.py`、
`run_cross_dataset_transfer.py`、`run_binary_subject_transfer.py`。
反例/集成：`tests/train/test_source_objectives.py`、`test_runtime.py`、`test_source_resume.py`、
`test_source_training_chain.py`、`test_source_execution_resources.py`、`test_unified_chain.py`。
global stopping、raw prefix生产器、真实membership、参数中心及个人包由并行模块共同完成，其独立记录另列。
