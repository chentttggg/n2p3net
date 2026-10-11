# Standards 独立审查摘要

固定基线为 `frozen/system_implementation_20261004_before/source.manifest.json`；本轮 changed hunks 为 `implementation.diff`，执行快照见 `review_standards/runtime_snapshot.json`。未改 production 或活动指南，新增探针已实跑。

**四项需修复的文档标准违背：**

1. **P1，冻结授权可被改写。** `noise_model.py:80`、`continuous_decision.py:150` 仅复制 dict，嵌套 provenance 仍可变。真实反例中错 scope 本被拒，修改 `scope_id` 后 LDA fit/save/load 全过；清空 continuous `records` 后保存重载，原 calibration 可当 held-out 预测。违反 AGENTS 拟合范围合同及工程指南只读推理要求。
2. **P1，CLI 记录的字节不是实际执行字节。** `run_operator_decision_study.py:122` 读取输入完成拟合后重新打开路径计算 SHA。真实 noise-fit 回调替换路径后，模型使用原数组，但报告绑定替换文件。违反 AGENTS “Bind each experiment to the source that actually executed it”及字节重放原则；plan/model/metadata 类似路径也需统一修复。
3. **P1，ERP mask 非布尔也可成为实测。** `calibration_candidates.py:354` 新分支对 mask 直接 `.all()`。全 NaN mask 产生全有限 score，违反静态/动态观测 mask 合同。应在公共预测入口验证 bool dtype、shape 与行覆盖。
4. **P2，固定解析基的角色未验证公式。** `temporal_transfer.py:94` 只核空 receipts 和正交性。任意正交矩阵重新摘要即可保留 fixed_analytic 角色；真实 StructuredERP 保存结构重载后 score 最大变化 4.0873。既然 recipe 声明 fixed_cosine，应独立核对解析构造，不能仅凭元数据角色。

另发现 xDAWN 失败 refit 后仍预测旧模型；冻结基线已有同缺口，单列建议而不计入新改动四项。未依据行数或命名强报 smells。共同问题是 Interface 宣称的“冻结/真实输入/解析角色”尚未由 owner 一次性保障；应在原 Module 收口，避免 caller 逐处补字段。

详细反例及排除证据见 [附件](review_standards/details.md)。
