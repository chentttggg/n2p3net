# 全库代码质量核查

日期：2026-09-08。父提交：`f54e45c48014d05a71d9644b0cecd7aa041f873f`，分支 `main`。
核查对象包含当时未提交、未跟踪的活动实现，父提交不代表本次执行源码。
使用 `codebase-design` 的 Module、Interface、Depth、Locality 原则检查职责及调用关系。

结论：静态检查与既有测试通过，但反例确认 **1 项 P1、6 项 P2**。
主要问题在进程探测、模型状态生命周期、checkpoint 输入变换和结果复用完整性。
本次未修改业务实现、既有测试、原始数据、缓存或训练结果，也没有启动云训练。

## 范围与验证

| 范围 | 文件数 | Python 行数 | 核查方式 |
|---|---:|---:|---|
| `src/data` | 36 | 17886 | 静态扫描、全部相关测试、输入/身份/参考/连续及 epoch QC 关键调用链 |
| `src/models` | 5 | 3175 | 静态扫描、模型/候选证据/损失/梯度/重建测试 |
| `src/baselines` | 10 | 5420 | 静态扫描、全测试、训练和失败路径反例 |
| `src/train` | 5 | 2284 | 静态扫描、资源/采样/批次合同测试、训练调用检查 |
| `src/transfer` | 28 | 11489 | 静态扫描、全测试、校准/保存重建/复用/失败状态反例 |
| `src/research` | 9 | 2090 | 静态扫描、执行/来源/统计范围测试、checkpoint 发布与恢复调用检查 |
| `src/stats` | 3 | 544 | 静态扫描、配对/分母/统计作用域测试和实现检查 |
| `experiments/*.py` | 50 | 20502 | 静态扫描、48 个可执行入口外目录 `--help`、主要 runner 调用链 |
| `tests/*.py` | 109 | 31905 | 全套测试执行、覆盖情形核对 |

另有空的 `src/__init__.py`。共 147 个实现文件、109 个测试文件，Python 合计 95295 行。
根 `conftest.py`、依赖配置、启动脚本、dashboard HTML/JavaScript 另行检查。
历史归档、`experiments/runs` 中冻结 checkout/单次实验脚本、第三方压缩 Chart.js、虚拟环境和
独立 BrainSync 采集工程不作为当前后端活动代码混合计数。

这是全库静态扫描、全套测试和风险关键路径的深入审查，不是逐行人工证明，也没有测量行覆盖率。

| 检查 | 结果 |
|---|---|
| `.venv/Scripts/python.exe -m ruff check .` | 通过 |
| `.venv/Scripts/python.exe -m pytest --tb=short` | 1489 passed、4 skipped、20 warnings，344.13 秒 |
| `uv lock --check --offline` | 通过，151 个解析包 |
| `uv pip check --python .venv/Scripts/python.exe` | 通过，114 个安装包 |
| 48 个 `__main__` CLI 从系统临时目录执行 `--help` | 全部通过 |
| dashboard 内联 JavaScript 语法解析 | 通过 |
| `git diff --check` | 通过；Git 提示部分现存文件 LF/CRLF 转换 |

环境为 Windows、项目 `.venv` 的 Python 3.14.3、Torch 2.13.0+cpu，CUDA 不可用。
4 个跳过项已分别复核：两个独立前端 producer 测试未设置 `BRAIN_SYNC_WORKSTATION_ROOT`，
两个符号链接反例受 Windows 创建符号链接权限限制。未将这些项算作通过。
警告主要来自合成 EDF 尾部补齐、单样本协方差、Braindecode 弃用提示及 GTN 多 TXT 提示。
没有在本轮验证 CUDA 编译/并发、云吞吐、真实采集设备、光学时钟或成人效能。

可重放反例：[probes.py](../assets/code_quality_audit_20260908/probes.py)。
结构化结果及逐文件 SHA-256：[results.json](../assets/code_quality_audit_20260908/results.json)。
反例只拟合小型合成数据、改写临时目录中的自建预测文件，并使用自建的隔离进程。
NaN 和训练中断是显式故障注入，不代表既有实验已经遭遇这些故障。

```powershell
.\.venv\Scripts\python.exe doc\assets\code_quality_audit_20260908\probes.py
```

## 已确认问题

### Q1 / P1：Windows 存活探测会发送中断信号

位置：`experiments/watch_resources.py:29`，`_pid_alive()` 内 `os.kill(pid, 0)`；
同文件 `main()` 的资源记录循环直接调用它。

Windows 的信号值 0 与 `CTRL_C_EVENT` 重合，不能沿用 POSIX 的无副作用存活检查语义。
监控脚本在普通控制台运行时可能中断训练及共享控制台的其他进程。
在新建、隐藏、独立控制台中，先确认自建目标进程打印 ready，再执行正式探测函数，
隔离控制台退出为 `0xC000013A / STATUS_CONTROL_C_EXIT`。
另外，新进程组中屏蔽 Ctrl+C 的子进程可以存活，说明具体后果依赖控制台与信号处理配置；
这不能使该函数成为可靠的只读存活判断。

建议：使用不会发信号的进程查询 Interface，例如已可用的 psutil，或独立实现 Windows
进程句柄查询；明确 Windows 不支持的进程资源字段，不能靠发信号探测。
验收：目标和同控制台的其他自建进程在探测后均存活，已退出 PID 返回 false。
Windows 集成测试必须放在隔离控制台内，避免测试本身中断 pytest。

### Q2 / P2：关闭输入标准化的源 checkpoint 重建改变预测

位置：`src/research/source_selection.py:292`，源 payload 总是保存经验 `input_mean/input_std`；
`src/baselines/deep.py:666` 按 `standardize_input` 决定是否应用变换；
`src/transfer/checkpoint.py:683` 的预测输入处理总是应用 checkpoint 统计。

`DeepConfig(standardize_input=False, fixed_updates=1, ...)` 是合法配置，源训练使用原始输入，
但相同权重通过正式 transfer loader 重建后使用标准化输入。
合成反例的最大 logit 差为 **0.03542153164744377**；只把 `standardize_input` 改为 true，
同样流程的差为 **0**。因此 artifact 对当前训练对象的预测行为不再等价。

建议：由一个 Interface 导出实际执行的输入变换；关闭标准化时导出恒等变换，
或在训练开始前明确拒绝该 checkpoint 类型不支持的配置。统计诊断与推理变换分别保存。
验收：true/false、实测通道 mask、源模型重建及保存到文件后加载均保持预测一致。
本次未据此推断既有标准化开启的实验受到影响。

### Q3 / P2：固定更新源训练把 NaN 状态作为拟合成功发布

位置：`src/baselines/deep.py:1238` 的 `train_step`、`:1268` 的更新循环及 `:1299` 的成功状态。

该分支没有检查 loss、梯度或发布前模型状态的有限性。NaN 不会自动使 PyTorch 抛异常，
因此现有异常清理不会执行。注入非有限 CE 后，`fit()` 正常返回、`_fitted=True`，
两个候选 snapshot 均被保留，模型和记录的损失包含非有限值。
`research.source_selection.fit_source_trajectory` 随后可把这些状态构造成源 payload。
即使后续评价或严格 JSON 写入最终报错，也已失去训练阶段明确失败与及时停止的语义。

建议：复用明确的数值失败类型，在不破坏吞吐的检查点验证损失/梯度/待发布状态，
失败时清空无效 snapshot 并使拟合状态失效；不要每一步都引入不必要的 CPU/GPU 同步。
验收：NaN loss、有限 loss 配非有限梯度、非有限最终状态均不能产生可用模型或成功产物。

### Q4 / P2：跨运行复用未校验预测内容

位置：`src/transfer/target_experiments.py:332` 至预测读取处 `:337`。

`reuse_target_records` 会核对旧合同摘要、cache/source 字节声明、行和标签，
也会检查模型 state 摘要，但没有绑定 `.npz` 预测分数的摘要或复算其与模型的一致性。
反例只将临时旧结果的 scores 取反，行、标签、checkpoint、模型 state 和旧 record 全部保持不变，
新的 E1 仍标记 completed 且 `fit_executed=False`，AUC 从 **0.49** 变为 **0.51**。
这让被误覆盖或混入另一处理臂的预测文件被接受为原拟合的结果。

建议：复用 Interface 绑定完整结果文件摘要，至少包含预测、模型及消费合同；
在原结果生成时登记，消费时验证。发现完整性错误应按产物错误处理，不能当作普通被试拟合失败。
验收：scores、行、标签、模型、record 任意单项变化都被拒绝；合法复用仍保留原来源并生成当前消费合同。

### Q5 / P2：适配器再次拟合失败后仍允许使用部分更新状态

位置：`src/transfer/subject_adapter.py:865` 的拟合状态重置段与 `:1763` 的预测门禁。

`SubjectAdapter` 仅在构造时把 `_fitted` 设为 false，再次 `fit()` 没有使既有成功状态失效，
也没有在异常退出时回滚或清理。先成功拟合，再在第二次拟合完成一次更新后注入中断，
正式预测 Interface 仍正常返回；`last_history` 保留上一次成功摘要，实际状态摘要却已不同。
调用者捕获失败后无法判断当前对象是完整旧模型还是部分新模型。

建议：将 fit 的状态提交集中在一个 Module：开始变更前失效，成功后提交；
或者明确保证完整回滚到旧状态。模型、输入统计、BN、历史及 fitted 标识必须一起处理。
验收：在输入变换、优化器更新、验证及 full-prefix refit 阶段注入失败后，
只能得到明确拒绝推理的对象或可证明等价于旧 artifact 的完整回滚状态。

### Q6 / P2：仪表盘永远匹配不到既有资源监控进程

位置：`experiments/dashboard_server.py:90`。

`_command_line()` 返回参数列表，但 `_matching_pids()` 先把它 join 成字符串，
后续 `for part in command` 逐字符查找完整的 `watch_resources.py`，条件始终失败。
用模拟 `/proc` 记录传入完整合法命令行，结果仍为空列表。
`_claim_monitor()` 因而可能重复启动监控并覆盖 `monitor_workers[run]` 中的进程引用，
多个监控竞争同一结果文件；既有监控也可能被显示为离线。

建议：从读取到脚本/参数匹配全程保留 argv 结构，查询与监控创建作为一次受锁管理的操作。
验收：已有进程、带空格路径、`--run=value`、多客户端重复 claim、释放最后一个 lease 均覆盖。

### Q7 / P2：缺失指标被显示为数值零

位置：`experiments/dashboard.html:228` 的 `finiteNumber()`；GPU 和最终 AUC 的 KPI 消费该结果。

`Number(null)`、`Number("")`、`Number(false)` 均为 0，随后通过有限数判断。
真实资源采样在没有 GPU 或查询失败时使用 null，因此会显示成 0%/0 MiB；
显式 null 的 AUC/验证记录也会变成数值点，混淆“未取得数据”和“测得零”。
通过 Node 执行文件中的原函数，三个输入均返回 0。

建议：数值转换前保留缺失值，并只接受明确支持的数值及非空数值字符串。
验收：null/undefined/空白/布尔值均保留缺失，真实 0 仍显示为 0，图表对缺失值留空。

## Module 设计评价

可保留的组织：`data.contract`、identity/lineage、checkpoint 验证、candidate evidence、
训练资源调度和声明式 TargetRecipe 都已有明确用途和真实调用者。
大量反例测试使用实际数据 builder，已有合同门禁能服务多个 runner。
不能因为文件数量多、`data` 使用语义哈希，或单个函数较长，就判定需要整体重写。

| 改进方向 | 当前证据 | 应集中承担的职责 |
|---|---|---|
| 拟合状态和失败处理 | Q3、Q5；`DeepBaseline._fit_attempt` 1319 行，`SubjectAdapter.fit` 635 行 | 可共享状态提交及数值验证，保留不同训练目标/优化器/协议的 implementation |
| 输入变换到 artifact 的一致性 | Q2 | checkpoint 导出应封装执行变换和验证，caller 不再自行拼接推理统计 |
| 结果完整性 | Q4；已有 `TargetResultReuse`、`SourceChainResume`、原子 writer | 扩展既有 artifact 验证能力，集中不同复用途径必须共同知道的不变量 |
| 候选拟合与产品包装依赖 | `calibrated.py:34` 重导出候选拟合接口，`target_experiments.py:257` 经产品模块导入它们 | 公共 runner 直接消费候选 Module，batch 规则放回明确的共享所有者 |
| 预处理调用知识 | `preprocess` 有 35 个参数，多个 loader 负责拼装 | 优先消费现有 `PreprocessingSpec`，集中参数映射；保留原始格式和协议差异 |

静态依赖图发现三个包含延迟导入的环：`continuous_qc <-> ocular`、`study <-> checkpoint`、
`calibrated <-> target_experiments`。当前测试及所有 CLI import 通过，不能把这些环直接
报告为已发生的循环导入崩溃；它们是后续评估 Locality 的候选位置。
`run_candidate_cross_decision.main` 1140 行、`run_pretrain_supervised.main` 1129 行是审查热点，
不是按行数判定的缺陷。拆分时应让 Interface 封装必要知识，而非增加只转发参数的浅层 Module。

## 修复顺序与证据边界

先修 Q1 的进程副作用；再处理 Q2/Q3/Q5 的模型一致性和失败生命周期，随后补齐 Q4 的复用门禁，
最后修 Q6/Q7 的监控识别与缺失值显示。让本次反例转化为正式 Interface 回归测试，
通过后再评估是否需要调整内部组织；不先铺设统一大 Runner 或通用插件框架。

本报告没有证明已完成的公开/本人实验存在数值污染，也没有重新训练或重算历史总体成绩。
全测试通过仅说明已有用例通过，不能覆盖本报告新发现的反例，也不能替代独立前端和实机验收。

## 修复落实

以下是同日经用户授权后的实现与验收，前文保留原核查时点。修复沿用
`codebase-design`、`systematic-debugging`、`test-driven-development`，并在本地CUDA验证前
使用`performance-optimization`与`get-available-resources`检查资源及验证边界。

| 问题 | 实际修复 | 验收 |
|---|---|---|
| Q1 | 原监控入口改用`psutil.pid_exists`，直接依赖同步`pyproject.toml/uv.lock` | 临时进程存活/退出及不发送信号的回归通过 |
| Q2 | `DeepBaseline`保存实际仿射变换，两个源导出入口复用`fitted_input_statistics`；禁用标准化导出恒等变换 | 标准化开/关、缺导开/关及物理checkpoint保存/重建通过；错误声明被加载门禁拒绝 |
| Q3 | 每次更新在设备端累计loss/梯度有限性，已有进度点、snapshot和最终发布前检查state；异常清空snapshot | CPU/CUDA分别注入异常loss、梯度、state；不能发布成功模型，后续合法fit可恢复 |
| Q4 | 现有`result_reuse`持有完成record/cell与伴随文件摘要，验证并解码同一份bytes；独立完整性异常 | 分数/标签/行/模型参数及历史/cell/run记录变化、文件缺失、旧记录无声明、读取期间替换、畸形形状均拒绝；合法复用通过 |
| Q5 | 原`SubjectAdapter.fit`从入口到full-prefix refit采用统一失败清理与成功提交 | 输入、优化、验证、refit、中断五类反例通过；失败拒绝推理/统计导出，重新拟合与独立新对象一致 |
| Q6 | argv结构保持到匹配结束，已有锁覆盖查询/创建/租约登记，优先复用已登记worker | 带空格路径、equals参数、四客户端共享一个worker及最后租约释放通过 |
| Q7 | 统一严格数值转换；运行KPI、累计AUC、验证曲线及进程内存聚合保留缺失 | 真实零值保持零；1440×900及390×844浏览器检查缺失KPI/曲线/内存，文字适配且无页面异常 |

没有增加新的运行时模块或统一大Runner。候选实验直接调用现有`calibration_candidates`，
消除了经`calibrated`重导出导致的相互依赖。新完整性能力集中在原`result_reuse`，
序列化仍复用原子writer及语义hash；进程内cache登记不重复反序列化刚保存的模型。
缺少原始完整性声明的旧结果只拒绝自动复用，不改写历史分析数据，也不补造旧摘要。

环境重装后另发现Dense PDS自动CUDA路径的两个已存在问题：极小距离平方下溢，
以及`cdist`矩阵平方展开对大而相近坐标发生相消。CPU已有稳定`hypot`实现，修复在原处
根据float64精度、坐标量级和特征维数选择该既有路径；正常GPU路径使用直接差值距离。
原`1e-200`反例恢复为2次尝试且无fallback；大坐标28行选择与CPU一致，18项采样测试通过。
这不修改采样半径、类别混合或科学比较条件，也没有重算旧实验。

最终全套：**1530 passed、4 skipped、20 warnings，361.46秒**。
随后最终页面改动又通过7项仪表盘测试和上述双视口浏览器检查。
收尾期间检测到其他任务继续修改个人校准/QC调用链；保留后又对当前个体包、质量分派、
候选缓存、目标适配、并行训练链和BrainSync先导入口补跑，**97项通过**。
张量引用清理后的源数值故障及输入变换路径另有10项通过。完整套件、后续定向验证和
最终工作区不是同一个静止时点，验证摘要分别记录这些证据，不把并行改动冒充本轮实现。
Ruff、`git diff --check`、离线lock检查及安装依赖兼容检查均通过。
4个跳过仍是两项未配置的独立前端producer集成和两项Windows符号链接权限测试。
验证摘要和当前逐文件摘要见[validation.json](../assets/code_quality_repair_20260908/validation.json)。
最终后端源码物理快照使用既有冻结入口生成并逐成员验证，见
[source.manifest.json](../../frozen/code_quality_repair_final_20260908/source.manifest.json)；
dashboard HTML的当前摘要另包含于验证摘要，后端冻结入口不包含HTML。

执行中检测到另一个安装进程更新了主`.venv`，因此前文CPU-only审计环境与最终环境不同。
最终实际为Python3.14.3、Torch2.13.0+cu130；源码修复没有安装或修改显卡驱动。
CUDA/BF16/fused Adam与显式`aot_eager`各执行2次更新并通过，见
[CUDA结果](../assets/code_quality_repair_20260908/cuda-results.json)及
[检查脚本](../assets/code_quality_repair_20260908/cuda_check.py)。脚本用`python -X utf8`运行，
以处理当前Torch编译器在Windows默认GBK下读取UTF-8模板的第三方兼容性问题。
本机未安装Triton，**Inductor/CUDA Graph和云端5090吞吐未验证**；这些冷启动耗时不能用于加速结论。

浏览器证据：[桌面](../assets/code_quality_repair_20260908/dashboard-1440.png)、
[手机](../assets/code_quality_repair_20260908/dashboard-390.png)、
[结构化结果](../assets/code_quality_repair_20260908/dashboard-results.json)。
使用真实页面和Chart.js、合成HTTP响应，不连接或终止现有训练进程。
本轮没有执行新的真实数据训练矩阵，也没有更改数据准入、临床/成人确认边界或历史成绩。
