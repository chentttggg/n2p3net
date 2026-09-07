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

## 开发与验证

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
结果复用`research.execution.atomic_write_json`，序列化失败保留旧结果。
每个实验保存实际配置、训练/评估各自源码snapshot、训练行、预测、指标、包版本及资源记录。
dirty工作树父commit不等于执行源码；不得用新hash重标旧实验。

## 设备与性能

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
历史5090耗时、并发策略和归约细节保留在[设备测量记录](evidence/device-portability.md)，
其数值不是当前机器的实时能力。方法效用比较与等draws/等updates的机制归因分别记账。

本次[目标适配计时](evidence/cross_dataset_start_20260906.zh.md)中CUDA单任务约快6.4%，
不支持“小模型必选CPU”的规则，也不替代并发吞吐测试。CPU/CUDA随机训练预测可能明显
不同，运行后端须冻结入配方；性能探针不冒充新增独立科学样本。

## 数据、工作树与文档

原始数据、派生cache、运行结果、当前源码和冻结快照按[目录层级](technical_structure.zh.md#目录与数据层级)
分别保存。删旧文件前物理压缩、逐文件校验；结果内嵌路径和分数保持原字节。
冻结材料不得重新作为默认配置/可导入源码；有效唯一数据不当临时垃圾删除。
保留无关dirty修改；stage或commit须属于用户明确要求的提交整理范围。

历史`tmp/`的数据和结果现存cache/results history，完整查找方法见
[数据保留记录](evidence/workspace_cleanup_20260906.zh.md)。Git忽略的大材料迁机时单独携带，
不能把Git clone当数据备份。代码归档与研究成绩各自保留原身份。

GitHub不分发历史压缩归档；`frozen/`只跟踪清单、哈希与交付记录，详见
[归档说明](../frozen/README.md)。移除已进入未推送历史的归档时，先在本地备份原Git历史，
再仅改写未推送提交；保留旧新commit映射，不回写历史manifest的`source_commit`。
推送前检查远端同步、全部待推送对象大小、暂存内容、文档链接、Ruff、相关测试及lock。

当前指导固定在[六份正文](README.md)，实施状态统一更新执行计划。
独立实验/来源/推导放`evidence/`，原JSON、日志、图片放原结果目录或`assets/`。
修改文档核对本地链接；替换的指导先归档再删除，合并关系写入manifest。
