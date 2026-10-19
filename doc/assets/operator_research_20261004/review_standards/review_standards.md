# Standards 独立复审：算子与 Stage3 研究链

复审日期：2026-10-05。固定基准为 `frozen/system_implementation_20261004_after`，archive SHA256：`043893ac95e4ff8e9b5f3db718e216ed6c33c4178af5a01e826f2237c4b3973a`。该物理快照早于本轮算子新文件，因此“scope 文件不在基准 archive”是预期的新增范围事实，不是把它们误当作旧实现的一部分。

复审只读，未修改生产代码、runner、wrapper、Stage3结果或测试。最终复审字节摘要保存在 `results.json`；探针源为 `probe.py`，执行范围为 `software_correctness_independent_review`。并发 `continuous_operator_mechanism` 在复审期间曾修改这些文件，已停止后才以稳定 SHA 重跑；本报告不采用并发中间态的旧结果。

## Findings

### [P1] 运行源码身份仍可绑定磁盘新字节，而不是实际已加载实现

位置：[continuous_model.py:100](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/continuous_model.py:100)、[continuous_model.py:142](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/continuous_model.py:142)、[continuous_model.py:249](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/continuous_model.py:249)。`fit` 在调用时重新对 `Path(__file__).read_bytes()` 求摘要；`__post_init__` 再用同一磁盘摘要校验。若模块已经被 Python 导入，随后文件被替换，拟合仍执行内存中的旧函数，却把新磁盘字节写入 `implementation_sha256`。独立探针先由旧字节导入模块，再替换 review 副本文件为必然抛错版本；旧函数拟合成功，artifact 声称的摘要却等于新文件（旧：`b3425072…056c8`，新：`60b78dab…d2d3`）。这违反工程指南“实际模块加载时源码摘要随凭证保存”和 AGENTS“Bind each experiment to the source that actually executed it”。应在模块加载时固定 implementation digest，或由 runner 传入已冻结执行源 digest，并拒绝调用时重新读磁盘作为执行身份。

### [P1] executor 创建异常会留下永久 `running` 结果

位置：[run_calibration_study.py:197](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/run_calibration_study.py:197)–[run_calibration_study.py:203](/D:/BCI/n2p3-net/doc/assets/operator_research_20261004/run_calibration_study.py:203)。runner 先写入 `result.json` 的 `status: running`，再进入 `ProcessPoolExecutor`；executor 初始化异常发生在 `with` 之前，外层没有 `try/finally` 或失败落盘逻辑。review-only 替换 executor 为抛出 `OSError`，结果保留 `running`、请求768、完成0，且没有 failed 状态或完整 not-executed cell 账本。工程指南要求任务异常保存 failed，不能留下伪 running 完成记录；应在资源/进程池启动阶段捕获异常，写入 `failed`、错误和全部请求的 `not_executed` 条目，再重新抛出非零退出。

## 已关闭或不构成 Finding 的项目

- 连续模型空训练账本：当前 `__post_init__` 要求非空 target calibration ledger、每条记录字段和 noise/model ledger 一致；重签空 `records` 被拒绝。
- 非对称通道协方差：当前加载时显式检查对称性；人为把上三角改大10后被拒绝。
- finite operator 语义：当前 `FiniteFilterOperator.from_record` 拒绝错误 `edge_rule/state_role`，`OperatorGLS.load` 使用该解析器；错误声明不再静默恢复为 observed odd reflection。
- `TemporalNoise` 保存/加载：当前训练账本、implementation SHA、schema、PARCOR/启动状态和数值摘要均做重放检查；本轮未发现独立可复现的 standards 缺口。
- SelectionERP 家族分派：当前独立分析器按 `family` 选择 `OperatorGLS` 或 `SelectionERP`；真实 Stage3 waveform artifact 已由 `SelectionERP.load` 成功重载，见 `analysis_dispatch_closure.json`。直接用连续 loader 读取 waveform artifact 的失败只是预期的负对照，不是当前分析器 bug。
- requested 失败分母：Stage3 最终 `result.json` 保存 768/768 请求，752 completed、16 failed；未把失败删出分母，分析器对缺失 cohort 采取显式不可估计路径。
- 资源行为：本地 Stage3 使用计划内 workers/CPU 预算；没有把它冒称云端训练，也未发现本轮要求的云端25 CPU/4 GPU被静默降级问题。

## 复审边界

本报告不重新解释 Stage3 的科学效能，也不把失败16个 cell归因于单一算法机制。`state_identifiability` 的独立 Spec 报告另指出 artifact 重签和 Stage3 16个 AR32 lattice 失败的 spec 问题；本文件只保留 Standards 轴结论。当前生产代码和活动指南未因本报告直接修改。
