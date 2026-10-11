# 小选择组与单完整组协议评估

本评估回答两个方案：把 BNCI2014-008 的完整选择拆成更小的组，或把 grouped temperature 的最低组数降到一个。它只使用 A01 原始 MAT 的物理 owner 解码和解析温度反例，没有新增 EEG fit、future score 或公共效能读数。原 MAT SHA 为 `cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd`；只读探针见 `tmp/PREFIX_SMALL_GROUP_EVAL_20261011/probe.json`。

当前一个 `CalibrationSelection` 是 120 个 flash：12 个候选刺激 × 10 次 repetition。每次 repetition 只有 12 个事件、2 个目标标签，而且同一 selection 的 10 次 repetition 使用相同目标位置。当前物理 owner 的响应宽度为 154 samples、pre-roll 为 26 samples，flash 间隔为 64 samples。

因此，直接把一个 selection 切成 1、2 或 5 个 repetition 不能作为当前的独立完整组：

| repetition 数 | 事件数 | 目标标签数 | 下一未纳入 flash 是否落入响应尾 |
|---:|---:|---:|---|
| 1 | 12 | 2 | 是 |
| 2 | 24 | 4 | 是 |
| 5 | 60 | 10 | 是 |
| 10 | 120 | 20 | 否 |

小块仍属于同一 target decision，且同一个连续记录的响应窗口受相邻未知 flash 影响。若缩短响应尾、忽略邻居或把相同 target pair 的 repetitions 当作独立 OOF 组，就改变了输入支持、AR/nuisance 模型和重复测量单位。现有 `prepare_bnci_calibration_groups` 因此要求完整 120 flash selection，并核对前后未知事件、raw scope 不重叠和完整 membership。

一个完整组足以做 fixed baseline 的目标拟合：以 K200 为例，可以使用最早完整 120-label selection，保留后面的 80 个请求标签为 unused/partial，固定温度 `T=1`。它不能支持当前 grouped temperature，因为 grouped 的 leave-one-group-out 在唯一组被留出后没有任何训练组；把门槛从 3 改成 1 只会把“无 OOF 训练数据”改写成可运行的错误协议。

一个组内直接同时拟合专家和温度也不能充当 OOF。温度损失在数学上可计算，但它使用了同一组的拟合输出和 truth，属于 in-sample calibration。只用一个真实观测，温度会被单个 truth 推到边界，并且 truth 翻转会把温度推向相反边界；A01 物理 probe 还显示所有 repetition 的 target pair 相同。解析反例中，真实概率已经校准时，单个观测学习的温度仍可令总体 NLL 从 1.049 增至 5.009。

可以考虑三个有明确边界的替代方案：

1. **K200 fixed baseline。** 使用一组完整 selection 拟合专家，温度固定为 1；grouped arm 返回结构性 `insufficient_calibration_groups`，保留完整 requested denominator。这是当前最小改动且信息边界清楚的方案。
2. **预冻结 source-global 温度。** 在独立 source/C 开发程序中先固定 `T` 或其先验，再在目标 K200 拟合专家。目标组不学习温度，因此这不是 target grouped temperature，source/global 与 target 的估计量必须分开记录。
3. **带预冻结先验的单组温度诊断。** 例如固定 (u_0=1)、固定 λ 后最小化 (L(u)+\lambda(u-u_0)^2/2)。这在数值上可算，但 (u_0,lambda) 必须在看到目标 truth 前冻结；它仍没有独立目标风险估计，不能自动成为 selector 或默认程序。

把 K200 的后 80 个标签拿来训练、再评价先前完整 120 个 selection，是一个可能的 offline backcast 诊断，但它使用未来目标数据评价过去决策，不能称 causal chronological prefix，也不能复用当前 complete-group caller。反过来用前 120 个训练、后 80 个评价时，后者是不完整 selection，不能提供当前 36-candidate 完整概率和操作性 winner。要采用 partial-label/refit 协议，必须新增 partial membership、支持边界、连续 AR/nuisance、保存重放和失败分母的完整 owner；不能只改 `group_count`。

若未来要把小组变成正式协议，最小实现不是降低一个常数，而是新增一种估计量：预声明 chunk 的物理切分、每个 chunk 的未知事件处理、同一 target decision 的重复测量关系、局部 AR/过滤支持、完整/部分 membership 及 outer held 规则。它还需要独立反例证明 chunk truth 不会进入自身 fit/temperature/selector，并把所有结构不足和数值失败保留在 requested ledger。

当前裁决：**不把完整组拆小，也不把 grouped temperature 的正式下限降为一个。** K200 先保 fixed baseline；若研究需要低剂量温度，优先实现 source-global frozen temperature 或显式 regularized diagnostic，并单独标记其估计量和不具备的独立风险证据。真正 target 内 grouped temperature 至少需要多个互不重叠的完整 selection；数量下限仍应由冻结的学习程序和资格规则决定，不能根据某次结果临时降低。

这项评估不阻塞已集成的 source cache/shared scoring，也不改变 P1 16 格 native 面板的失败账本。后者的数值门诊断另见 `tmp/P1_CHART_DIAGNOSIS_20261011/HANDOFF.zh.md`。
