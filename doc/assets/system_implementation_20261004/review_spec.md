# Spec 轴独立评审

固定点：`frozen/system_implementation_20261004_before/source.manifest.json`（archive SHA256 `e17eef169319ac18ce8f97a02d1088a566337e5a99280c2ed38c7fb94fad2250`）；范围为 `implementation.diff` / `changed_files.json`。按 `code-review` 的 Spec 轴只读审查，未用旧 HEAD 脏差异归责。

1. **[P1] xDAWN 尚未执行实测秩门禁。** 规格 M1：“nfilter不得超实测秩”（研究报告第90行）。`src/baselines/riemann.py:129` 直接传递 nfilter；OAS 数值满秩不等于观测信号满秩。独立探针中三条完全重复通道、实测 rank=1，`nfilter=3` 仍成功保存六个 filters。应在合法拟合范围计算实测秩，限制或明确拒绝超过支持的请求，并保存有效维数。

2. **[P1] 短记录让公共比较整批退出且丢失请求账本。** E4：“requested全部保留”；S1：“报告失败而非删人”。`experiments/run_erp_comparator.py:66` 遇缺少请求 K 即抛错，直到113行才创建结果。实际 BDF 读取/准备探针得到 effective K=[0,8,16]；请求[8,200]后 K8也未执行，未发布任何结果。应保留 K200 unavailable 原因，继续合法cell；K16只可另列fallback，不可冒充精确K200。

3. **[P2] 成熟锚点和统计出口仍未组成完整 S1。** M2要求“当前输入＋当前波形SLDA”“无baseline输入＋相同SLDA”；S1要求共同A0Q0+scratch。`run_erp_comparator.py:56` 仅接受新ERP families，`pooled_lda`不是现有 waveform SLDA 路径，也不能加入NN锚点。输出未携带现有统计所需study/seed声明，尚无分析适配。需增加原锚点或与既有executor联动，固定选择机会并接库等权配对汇总，不能把六臂接口测试当完整成熟对照。

4. **[P2，部分实现] E3/E5 的真实证据接线尚缺。** 规格要求“preprocess→raw scope/views→noise model→baseline/decision”和“raw样本/decision不重复计权”。当前 `noise_model.py:83`、`operator_lda.py:69`、`continuous_decision.py:154` 接受调用者的数组/角色声明；研究CLI未提供既有attested raw scopes、真实A/B及requested decision ledger的Adapter。数值核已有价值，但正式使用前须补这一接线与outcomes/coverage/latency评分。此项不要求假称已解决Γ可识别性。

证据：`review_spec/probe.py`、`review_spec/results.json`；探针退出0，Ruff通过。已认可新EA父输入数值重放、原物理cache门禁、时间基源库排除及个人未知标签重载；未发现需推翻现有连续高斯Woodbury/logdet公式的证据。未跑公共性能矩阵、Γ未识别、harmonic未改默认均为真实限制，不能宣布E0–E5/S0–S5全部完成。
