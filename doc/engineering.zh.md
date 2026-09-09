# 工程指南

更新：2026-09-07。合并原工程流程与环境指导，设备策略集中于本文件。
先读[研究规范](research_program.zh.md)与[技术结构](technical_structure.zh.md)，再看实际调用链。

## 环境与入口

后端统一使用项目`.venv/Scripts/python.exe`。依赖权威为
[pyproject.toml](../pyproject.toml)，解析版本见[uv.lock](../uv.lock)；
`requirements.txt`只是便捷安装入口，不另立环境真相。
Python至少3.11；Linux x86_64且Python<3.14的Torch/Torchaudio固定2.8.0，
其他平台/版本按项目声明的Torch>=2.4开发分支安装，不能当作RTX5090训练验证。
历史GPU环境为Torch2.8.0/CUDA12.8，当前硬件与可用资源须在运行前重新核对。

BrainSync源模型加载保留`weights_only=True`，对NumPy产生的Unicode身份标量提供受限allowlist，
不回退到任意对象反序列化。冻结恢复模型跨Windows/Linux重载时保留原序列化模态身份，
重算特征值仅容许1e-12绝对舍入差；原QC摘要不改写，错序/明显变化仍拒绝。
复现实测差1.51e-13且恢复样本数一致，见[新源校准记录](evidence/brainsync_s1_calibration_20260908.zh.md)。

NumPy/SciPy/MNE承担信号与格式处理，sklearn/pyRiemann承担经典基线，PyTorch承担神经网络。
CLI使用argparse与显式JSON合同；Hydra不在依赖中。配置优先来自数据类和已有接口，
不得凭旧说明重新引入已删除的默认YAML。

```powershell
uv sync --locked --extra signals --extra baselines --extra dev
.\.venv\Scripts\python.exe experiments\brainsync_model_task.py --help
.\.venv\Scripts\python.exe -m ruff check src experiments tests
.\.venv\Scripts\python.exe -m pytest tests/test_calibrated_predictor.py tests/test_brainsync.py
git diff --check
```

测试文件按改动范围选择，上例不是每轮必跑全集。入口/import变更还要从项目外目录验证CLI。
BrainSync原生SDK/Qt使用采集端自己的Python3.10环境，通过结构化进程参数调用后端；
不向后端注入另一minor版本的原生扩展，实际启动方式见[工作站指南](workstation.zh.md)。

工作站计算设备通过结构化`--device`传入，`auto`复用`train.device.get_device`，
显式CUDA先检查后端可用性和设备编号；LDA计算仍在CPU，不能按GPU选项声称LDA加速。
修改个体入口时覆盖QC开关下的源失败、固定LDA全范围EOG/冻结QC、dry-run与真实执行一致性，
并验证模型重载及无标签推理不改写个体包。采集端设备设置也须测试持久化与忙碌状态互斥。
2026-09-08本轮入口修复在当前后端环境全套1539通过、4跳过；前端35项通过。
CUDA实际拟合、保存重载及CLI推理验证见[执行计划](roadmap.zh.md#已完成与未完成)，
该合成验证不构成CPU/GPU吞吐比较或成人效能结果。

采集后质量报告需要`signals`依赖组，Matplotlib已显式声明并由`uv.lock`锁定；绘图使用Agg后端。
报告后台固定一个计算线程，避免占满采集工作站。Qt集成验证入口位于独立前端的
`tests/verify_quality_report.py`，使用模拟采集和独立输出目录，不能据其波形或报告认定真实信号合格。

## 开发与验证

工频陷波的回归入口为`tests/test_notch_filter.py`，覆盖50/60Hz抑制、通带保真、因果性、
断点及chunk状态、EEG/EOG角色、原始Nyquist、恢复误差、缓存/模型重载及旧源不兼容。
新增陷波频率或Q须同步实际SOS及恢复证书，不能只向滤波器追加调用而保留旧恢复时间。
历史缓存签名由`PreprocessingSpec.record()`保持无陷波原义；元数据不得重标成新处理。

每轮依次完成理念、数学概念、公式、反例、调用链联动和文档/产物整理。
模块明确shape、dtype、metadata、fitted状态、失败语义；复用已实现的合同和算法。
数据合同、划分和评价工具先能服务同一比较，再添加模型；不能因追求速度省去真实builder。

测试重点是训练范围、真实身份与group隔离、时间截止、mask/通道角色、完整候选词表、
失败分母和保存/重载。模型变化还检查CPU forward/backward与参数量；
梯度或目标函数变化用数值等价与反例，旧全量测试通过不证明新科学判断正确。
合同改变同步builder、runner、checkpoint、缓存/结果门禁与相关tests；
删除只为退役接口服务的旧断言，保留仍成立的数学不变量。

source、内部校准train/validation、完整calibration refit、final test的状态来源分开。
QC、EOG、EA、BN及阈值选择服从同一范围，不能只查模型优化器。
连续质量改动还须运行`test_continuous_qc`、`test_recording_clock`、`test_acquisition_contract`、
`test_personal_quality`、`test_ocular`和个体保存/推理测试；大范围改动运行全量。
逐导恢复另覆盖`test_epoch_state_recovery`、`test_local_recovery_pipeline`、`test_recovery_contamination`、
`test_recovery_lda`、`test_calibration_quality_ingress`、`test_frozen_epoch_quality`；数值注入必须使用真实
loader/参考/重采样/基线和后续QC，不只检验简化滤波脚本。总污染与重置状态误差分开报告。
前端使用其独立`.venv`并先调用`gtn_workstation.app.configure_runtime_paths()`再运行unittest。
真实EDF导出的填充、失败写入与暂停后响应支持必须作反例，不能只检查JSON字段存在。
本次验证和源码归档入口见[实施记录](evidence/brainsync_quality_implementation_20260907.zh.md)。
个人信号证据QC另覆盖`test_personal_signal_quality`、`test_personal_quality_dispatch`和
`test_raw_constant_quality`：他导/mask不变性、低幅动态信号、宽ERP、原始恒值减参考、持续时间边界、
新旧schema分派以及新方法必须依赖已应用连续QC `/4`。公共旧QC测试保留原语义。
新个人模型包写 `/4`；旧 `/3`、`/1`只按原合同读取。未启用原始恒值的连续QC仍写 `/3`，
启用时写 `/4`，明确5秒等实际参数，不修改旧profile摘要。
2026-09-08最终后端全套1404项中1400通过、4跳过、0失败/错误，前端29项通过。
云端57模型/114预测在最终门禁下CPU重放，有限值mask完全一致，最大logit差1.431e-6；
不是CPU/GPU逐位相等声明。详细环境、反例和归档见[QC验证](evidence/brainsync_qc_redesign_20260908.zh.md)。
结果复用`research.execution.atomic_write_json`，序列化失败保留旧结果。
完成目标结果由`transfer.result_reuse`封存`n2p3_result_integrity/1`：record/cell语义摘要绑定
预测和完整模型文件摘要。读取后直接解码同一份已验证字节；产物损坏或缺少原始声明时明确
中止复用，不计作被试拟合失败，也不在消费时补算摘要来重标旧结果。
源模型保存实际执行的输入变换；声明关闭标准化而携带非恒等推理统计的checkpoint被拒绝。
固定更新源训练在设备端累计数值健康状态，snapshot/进度点/结束时检查，避免逐步CPU同步；
适配器任何阶段失败后均拒绝推理和统计导出。回归包括NaN loss、非有限梯度/参数和refit中断。
监控PID存活使用已声明依赖`psutil`的只读查询，Windows禁止以`os.kill(pid, 0)`探测。
仪表盘的监控认领在已有锁中完成；缺失数值保持缺失，真实零值仍保留，见
[全库核查与修复](evidence/code_quality_audit_20260908.zh.md#修复落实)。
Dense PDS的GPU距离使用直接差值路径，避免矩阵平方展开的相消；坐标指数超出安全平方范围
时复用既有CPU `hypot`计算，不放宽半径。该边界由float64精度和特征维数推导，backend照实记录。
每个实验保存实际配置、训练/评估各自源码snapshot、训练行、预测、指标、包版本及资源记录。
dirty工作树父commit不等于执行源码；不得用新hash重标旧实验。
本轮QC的GPU执行快照`brainsync_qc_redesign_20260908/source/`与最终门禁完善后的
`final_source/`分别冻结。新增入口门禁不改变本轮已满足 `/4` 的输入及预测数学，最终重载审计单独绑定
实际评估源码。研究/前端/输入与结果压缩逐成员校验，Git只保存`frozen/`清单。

其余四项质量联动增加`test_slow_trend`、`test_recovery_certificate`、`test_independent_raw_scope`、
`test_brainsync_clock_segments`、`test_ocular_native_regressor`及跨仓库采集/个体CLI回归。
新频带必须显式声明完整预处理配置，恢复SOS及包络对应同一实际算子；源模型仍核验原合同。
新EOG策略保存 `/4`，旧 `/3`字节语义保持；完整raw重建不能把冻结EOG记录写成None。
12条件72拟合/144预测从`brainsync_qc_remaining_20260908/source_preparation_v2/checkout/`执行，
并在同一冻结源码下完成CPU重放。启动前先更正继承的旧问题文字，初始未执行计划单独归档，
不得把模板残留描述当作实际fit范围。后续测试夹具及GUI改动另绑定最终实现，不重标训练源码。
详细验证、吞吐和独立证据见[四项收尾](evidence/brainsync_qc_remaining_20260908.zh.md)。

目标早停消融使用`--stage ES`，plan可声明`epoch_selection={"patience":6,"min_delta":1e-6}`。
固定轮数扩展可在`epoch_selection`中声明`epochs:90`、`treatments:["full","scratch"]`、
`modes:["fixed"]`、`include_baselines:false`；后者省略已复用的K0/LDA任务。
轮数必须为正整数，模型/策略列表必须非空、无重复且使用受支持名称。
ES轮数读取`epoch_selection.epochs`，缺省60；普通非stage入口的`target_epochs`不覆盖ES。
`--dry-run`同时列出inner和完整refit的成本上界；`--stage ES`分析记录实际与缓存后的物理成本。
先跑满轨迹选轮、再单独比较patience停止，不能用训练loss下降或外层test确定停止时点。
固定source、学习率和batch512；非法校准内划分按同一固定recipe回退并保留无选模证据标记。

## 设备与性能

### 云端训练性能优先

用户明确偏好：**优先缩短实际训练墙钟，不以保持代码简单为由保留性能瓶颈**。
必要的任务图、进程调度、缓存、共享登记表和资源准入复杂度可以增加；以端到端吞吐和正确性验收。
已有独立任务时，默认充分使用当前机器实际可用的CPU配额和GPU容量，而不是固定单进程/单核。

每次云端启动或续跑前执行以下步骤：

1. 调用`performance-optimization`及`get-available-resources`技能。检查CPU affinity/cgroup配额、
   可用内存/显存、CUDA运行时及其他活跃任务；沿用同环境已核验的短时证据，避免重复冗长预检。
2. 分别检查数据加载/解压/合同校验、source训练、E2校准、外层评估及结果写盘的真实调用链。
   对有独立工作的阶段启用资源自适应并发；重复模型和固定元数据只处理一次，跨进程复用须有锁和来源摘要。
3. 若实现仍串行或GPU在等待CPU工作，修改算法、任务图和代码后再按实测吞吐确定执行拓扑。
   仅调大线程参数不算完成优化。worker/native/compiler线程总量和GPU准入受真实配额约束，batch服从用户明确上限。
4. 冻结实际执行源码，保存CPU/GPU预算、requested/effective并发、复用、OOM及编译回退。
   已完成产物直接核验复用；不为更换调度器从头重复训练，不把新快照身份倒填旧模型。
5. 启动后核对真实进程树、各阶段进度、GPU/CPU采样和已完成单元速率；空闲时定位剩余依赖或瓶颈并处理，
   不能只报告“配置了并发”。以墙钟改善为目标，不承诺所有依赖阶段的CPU/GPU百分比始终100%。

每次修改须保留身份/时间/fit范围、随机配方、预测和失败分母的可核验性；性能探针与科学结果分别记账。
默认完整落实已授权实现和云训练，不反复请求已有授权。准入、测试和必要的短测应支持快速执行。
用户再次明确：**说训练就是执行已约定的全量矩阵**，除非用户明确限制为小试、只读或dry-run。
必要的软件测试和小规模验证由执行方自行完成，通过后自动进入全量，不再次请求启动许可；
不能停在试跑、执行清单或因小试成绩不理想而自行终止后续训练。真实失败/合同冲突仍须处理，
但已经通过的相同环境和快照检查应直接复用，不重复冗长预检。

`operator_fixed_target_batch_rule`允许记录有明确理由的执行方固定batch配置；
其provenance为`run_configuration`，没有选优证据，不冒充用户逐字指定或完成B0搜索。
P001先导采用上限512，在校准数据上比较1/2个spawn进程后选择双进程、每进程1线程；
最新时耗及云端源码/输入验证见[P001运行](evidence/brainsync_p001_pilot_20260907.zh.md)。

复用`train.runtime.GpuPerformanceScheduler`与`MatrixBatchSource`，设备顺序CUDA、XPU、CPU，
张量经所选`torch.device`迁移，不硬编码`.cuda()`。CPU是正确性参照，
加速器比较允许声明精度误差，不能以CPU合成测试宣称CUDA或实机验证。

监督`DeepConfig`默认请求fused Adam和`reduce-overhead`整train-step编译；CUDA按支持情况
执行，CPU/XPU回退非fused eager，记录requested/effective/fallback。
`--no-fused-adam --compile-mode none`提供显式eager对照。SSL和SubjectAdapter有独立优化循环，
不能将监督训练测得的默认性能直接套用。BF16按设备能力启用，否则FP32；
batch和更新预算显式记录，不通过隐藏梯度累积或OOM重试改变科学比较。

预载矩阵须留显存余量；连续pinned视图可异步拷贝，随机gather走现有同步安全路径。
不能复用仍被DMA读取的单一pinned staging buffer；未来预取须有多缓冲和完成事件。
`--shuffle-each-epoch`会额外占一份矩阵，历史配对测量更慢，保持显式可选。
训练loss在设备累积、每epoch读取，预测按整pass回CPU，避免逐batch同步。

GPU folds用spawned进程隔离RNG和CUDA上下文，不能共享训练线程。
自动GPU worker数依设备显存及总预算决定；显式并发覆盖须有当前设备吞吐/显存证据。
`--cpu-threads`是总预算，经`resolve_cpu_worker_plan`与`cpu_thread_budget`按worker分配，
同时约束Torch/BLAS/OpenMP并读取cgroup配额。源数组共享只读，fold-local拟合状态各自私有。
同一外层划分的公共QC可跨候选复用；校准内部选择不能复用只满足outer-fit范围的QC。

优化先测预处理、训练、目标适配和写结果各段耗时；记录冷/热编译、峰值显存及延迟中位/p95。
大账本checkpoint应在任务组进入时完成整体校验、固定摘要，组内只做逐人身份排除与状态独立化。
不能以GPU空闲直接提高worker数；先检查反复反序列化/合同递归处理。2026-09-07实测该问题
令4个E2 worker各占一CPU核而GPU长期0%，一次合同校验约1.57秒。
优化后的4单元真实5090校验为27.31到4.24秒，确定性GPU条件下预测完全相同；
默认CUDA自身重复存在浮点波动，不将此性能检查声称为独立科学样本。
历史5090耗时、并发策略和归约细节保留在[设备测量记录](evidence/device-portability.md)，
其数值不是当前机器的实时能力。方法效用比较与等draws/等updates的机制归因分别记账。

本次[目标适配计时](evidence/cross_dataset_start_20260906.zh.md)中CUDA单任务约快6.4%，
不支持“小模型必选CPU”的规则，也不替代并发吞吐测试。CPU/CUDA随机训练预测可能明显
不同，运行后端须冻结入配方；性能探针不冒充新增独立科学样本。

## 数据、工作树与文档

BrainSync采集REF更正为左耳耳垂耳夹时，活动元数据通过`correct_brainsync_reference.py`
按显式session列表、旧标签和用户证言修订；该工具预检全部记录、压缩校验原声明后再写入。
原EDF和追加式事件日志不改写，旧二进制头中的声明由BIDS/session更正记录明确覆盖。
历史cache/QC/模型/结果保持执行身份；新派生物使用新目录，禁止直接改旧digest假装旧模型已经重训。
见[更正清单](../frozen/brainsync_reference_20260908/metadata/manifest.json)及
[科研/验证说明](evidence/brainsync_reference_20260908.zh.md)。

训练链条入口的`--dry-run`生成任务、真实身份/行范围、recipe、候选失败与预算清单，
不得拟合normalizer、BN、LDA或网络。已完成compact运行使用`--stage compact`，plan声明
`target_stage=compact`、`target_batch_size=512`及用户固定batch规则，先核对123人、305正K、
1953个评分。实际batch为`min(512,n_fit)`，60遍历；无需B0搜索或冻结结果，不据此声称最优性。
旧控制器及后续B0/B0_score/E1/S1/E2/S2/E3队列已终止，不得自动恢复；新compact队列单独冻结和记录。
减少重复拟合只可复用完整数据/状态/配置digest一致的结果，不改变本轮recipe和独立初始化语义。

保留的B0接口先只评分inner validation，
`--declare-batch-cohort-from ... --batch-cohort-design ...`将全部预检与预声明participant清单绑定，
实际B0通过`--batch-cohort ...`使用该范围；
`run_cross_dataset_transfer.py --freeze-batch-from ... --batch-cohort ... --output batch_rule.json`冻结规则；
`--stage B0_score --batch-rule ...`才执行其final test评分。这些是未排程的研究能力，
其缺省batch32不能覆盖本轮用户固定512策略。两类binary入口从项目外目录使用绝对路径调用。
新产品选择器同样使用`--batch-rule`；缺省明确回退32，旧产品独立固定64。
本轮compact512不更改产品默认batch，也不运行E3选择器benchmark。
用户随后授权的扩展全量通过`include_historical_controls=false`仅运行batch512的E1/E3处理；
source plan默认自动并发，各worker使用一个原生计算线程和一个编译线程，按真实CPU配额限制任务组。
GPU采用带余量的全局任务准入预算；任务结束释放临时内存，进程退出释放上下文。
共享文件registry保留跨进程任务复用，JSON发表和checkpoint哈希核验在语义签名锁内完成。
内存预算取主机可用量与容器`memory.max-current`的较小值，并保留15%余量；不能使用宿主机
总内存直接排程。source组暂按6 GiB、E2/外层worker按4 GiB估算，可在plan显式调整；这是
基于已有进程RSS的准入估计，不是内存硬上限。嵌套worker同时预留全局CPU和内存，余量不足时
由已加载数据的父进程评分；source组结束后退出进程，使后续评分可扩大并发。
单任务组也经过相同准入。中断恢复读取并校验各子组记录及外部引用文件，保留原快照身份。
临时attested副本和结果目录分别检查磁盘容量；`N2P3_VERIFIED_CACHE_SNAPSHOT_DIR`可指定
空间充足的私有目录。2026-09-07旧数据盘仅余1.4 GiB，触发校验副本的空间拒绝后，改用另一个
有25 GiB余量的文件系统续跑，旧数据和结果保留原位。

E3共享候选缓存仅驻留当前人/实际K；切换scope清空，避免GPU状态和数据随人数累积。
复用账本区分请求次数、真正fit和cache hit，不能将命中时间记成独立训练速度。
产品dry-run加载确定性输入并报告来源兼容性与inner范围，自动EOG仅列出拟合计划。
分析器默认拒绝software/synthetic或未声明证据；显式`--allow-software-correctness`仍将所有
区间标为软件验证，不进入公共开发或总体推断。

`freeze_source_snapshot.py --output 新目录 --materialize`保存当前文件字节、父commit、dirty状态
和物理归档，生成`checkout/`隔离副本。实验从该副本调用生产CLI；source和evaluation可绑定
不同快照。执行核验也逐项比对归档成员，不能通过重写manifest摘要给旧归档换身份。

2026-09-07本次本地环境为Python3.14.3、Torch2.13.0+cpu；管理工具可见5070 Laptop，
但项目Torch无CUDA运行时。本轮软件/CPU计时不作为目标5090的吞吐、显存或batch选择证据；
当前设备结果及正式执行前的资源缺口见[实施证据](evidence/training_chain_implementation_20260907.zh.md)。

2026-09-08经用户授权新增本地GPU专用`.venv-gpu`，原CPU`.venv`保留。GPU环境先按项目锁定
依赖安装，再显式覆盖为`torch==2.13.0+cu130`；`uv pip check`及真实CUDA运算通过。
本机实测RTX5070 Laptop 8GB、驱动610.62；使用单GPU worker、固定batch32的独立P001开发对照。
该环境不是公共云5090配方。启动前核对实际包版本、显存与源码快照，完整记录依赖覆盖与执行路径。
结果与脚本见[恢复实验](evidence/brainsync_recovery_20260908.zh.md)；不修改默认CPU环境或静默回退CPU训练。

2026-09-08代码质量修复期间，另一安装进程将主`.venv`的Torch改为`2.13.0+cu130`，
最终1530通过/4跳过基于该实际环境，GPU故障注入也已执行。小型CUDA/BF16/fused Adam及
显式`aot_eager`检查通过；本机没有Triton，未验证Inductor/CUDA Graph或云端5090吞吐。
Windows编译验证使用`python -X utf8`，避免当前Torch模板读取采用GBK；这不修改项目默认环境。

校准扫描默认统一使用`transfer.binary.STANDARD_CALIBRATION_TRIALS=(0,200,800,2000)`。
`run_binary_subject_transfer.py`的CLI默认和`run_cross_dataset_transfer.py`缺省plan字段
共用该定义。有限数据按`LIMITED_CALIBRATION_FRACTION=0.8`回退，
结果`budget_adjustments`记录请求与实际预算；同一有效K去重，K0只跑zero-shot。
历史冻结运行脚本保留原显式预算以便复现，不作为新实验配置模板。
正K迁移必须同时包含target-only scratch；两类binary runner共享实际行摘要核对，
并将`paired_transfer_effects`写入结果。默认保留原波形LDA同K对照，
不能仅因迁移臂得分高就宣称迁移有价值。

原始数据、派生cache、运行结果、当前源码和冻结快照按[目录层级](technical_structure.zh.md#目录与数据层级)
分别保存。删旧文件前物理压缩、逐文件校验；结果内嵌路径和分数保持原字节。
冻结材料不得重新作为默认配置/可导入源码；有效唯一数据不当临时垃圾删除。
保留无关dirty修改；stage或commit须属于用户明确要求的提交整理范围。

历史`tmp/`的数据和结果现存cache/results history，完整查找方法见
[数据保留记录](evidence/workspace_cleanup_20260906.zh.md)。Git忽略的大材料迁机时单独携带，
不能把Git clone当数据备份。代码归档与研究成绩各自保留原身份。

2026-09-08再次整理本地临时材料，4,816份历史文件及工具缓存压缩保存在
`frozen/local_history/workspace_cleanup_20260908/`；BCI根目录16份旧传输文件集中到
`D:/BCI/n2p3-audit-archive/transfers_20260831_20260901/`。当天脚本、SSH/EDF支持目录和
三个Git工作树保留原位。旧路径查询、校验与恢复见
[本次整理记录](evidence/workspace_cleanup_20260908.zh.md)。

2026-09-08云端数据盘已清理，旧BI64/GTN/R8派生缓存、早期开发结果和旧部署展开目录
经压缩逐文件校验后退役，归档位于`/root/autodl-fs/n2p3-cleanup-20260908/`。
当前四库输入、完整训练链、源模型和早停/90轮实验仍在原位；历史路径查询、清理清单与
本地证据包见[云端清理记录](evidence/cloud_disk_cleanup_20260908.zh.md)。

GitHub不分发历史压缩归档；`frozen/`只跟踪清单、哈希与交付记录，详见
[归档说明](../frozen/README.md)。移除已进入未推送历史的归档时，先在本地备份原Git历史，
再仅改写未推送提交；保留旧新commit映射，不回写历史manifest的`source_commit`。
推送前检查远端同步、全部待推送对象大小、暂存内容、文档链接、Ruff、相关测试及lock。

当前指导固定在[六份正文](README.md)，实施状态统一更新执行计划。
独立实验/来源/推导放`evidence/`，原JSON、日志、图片放原结果目录或`assets/`。
修改文档核对本地链接；替换的指导先归档再删除，合并关系写入manifest。
