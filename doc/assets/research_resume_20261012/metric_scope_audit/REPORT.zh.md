# 指标分母与字段审阅（只读复算）

生成于主树 `b26c099`。本交付只读取已封存 JSON、CSV、binding、证据文档和指标 schema 源码；没有模型加载、fit、score 或 EEG 读取。

## 已复算

P1 16 cell 底表有 8 个参与者、shared8/private8 各 8 个 held decision、36 个候选，两个方案概率和预测覆盖均为 100%。shared8 命中 5/8、NLL 7.3364765833、Brier 0.6790937536；private8 命中 6/8、NLL 5.5540121911、Brier 0.4776785010。配对 delta hit 的参与者宏平均为 0.125，配对 delta NLL 的参与者宏平均为 -1.7824643922；8 行 delta NLL 直接求和为 -14.2597151377，不能把行和当作声明的宏平均。A01 行占该行和改善约 71.8%。

P1 每人只有一个 held decision，因此 participant macro 与 decision micro 恰好相等。这个偶合关系不能推广到公共 BNCI 的 152 requested decisions、BI2014a 的 1287 requested episodes 或 BNCI C600/R10 的 240 requested decisions per arm。

## 字段边界

P1 使用 `independent_p1_numeric_panel_readout/1` 自定义 schema。`status=scored`、`hits`、`nll`、`brier`、`probability_available`、行列边缘概率和局部 density error 字段不能直接当作 `CandidateEvidence` 或 `CandidateProbabilityMetrics` v1 字段。映射前必须保留源 schema、候选数、真值、分母和聚合顺序。

`all_input_certificate=false` 表示误差估计不是全输入 interval certificate，不表示评分失败。P1 的 row/column probability 是 36-way joint 分布的诊断边缘，不得替代 joint `p_true`。

## 当前结论与缺口

P1 private8 在固定八人、固定 held、固定 36-way truth support 上改善了命中和 participant-macro NLL；它是 C-held development diagnostic。公共 BNCI、BI2014a、BNCI C600/R10 各自有独立 requested/probability 分母和 source/producer；它们不能合并成一个总体指标。B07 central numerical owner 已单独绑定，history consumer、capsule replay 和 public B07 score 仍是缺口。

交付文件：`metric_crosswalk.json`、`p1_error_ledger.json`、`p1_error_ledger.csv`。重新运行同目录的 `build_metric_audit.py` 可复算。
