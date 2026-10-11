# Spec 闭合复核（保留原 RED）

基准仍为实施前实际脏树 `frozen/system_implementation_20261004_before/source.manifest.json`。本次只读复核原 `review_spec.md` 四项、E0关系和活动指南；未用旧HEAD差异，也未改生产文件。原RED证据保留。

1. **xDAWN 实测秩：已闭合。** 新 `riemann.py` 从合法训练数据实际rank计算 `effective_nfilter=min(requested,rank)`；`StructuredERP`保存并复核两者。重放原三条重复导反例：rank1、请求3、有效1、总filters2；导出重载分数逐值相等。不是把可用低秩输入一律拒掉。

2. **缺 K 请求账本：已闭合。** 重放原BDF前缀，K8正常completed，K200记录protocol_unavailable/actual_K=null，不将K16伪装为K200；请求覆盖表完整。见 `review_spec_closure/results.json`。

3. **成熟锚点与统计：Implementation 已闭合，公共完整性能待执行。** 原waveform LDA有/无baseline、NN full-fine/scratch、源码冻结、真实checkpoint seed、数值重载、库等权配对统计已接。末轮实际runner曾接受仅full-fine、预算不等、K35/K17不同架构三种违例，原RED保存在 `review_spec_closure/anchor_admission.json`。随后新增preflight逐可用迁移cell绑定严格同输入/实际K/seed/真实architecture/recipe的scratch及原waveform，并记录映射与逐人迁移差值。最终独立GREEN在每个攻击中保留其他合法控制，分别确认缺scratch、错预算、异架构、缺waveform都被拒；合法三臂和scratch-only仍completed，见 `review_spec_closure/anchor_green/results.json`。另核明确对比不能借别处合法scratch绕过不匹配control门禁。没有把different-budget探索算成迁移主效应。

4. **真实连续Adapter：E5及R2纵向能力已闭合；E3整体仍部分实现。** 已核真实BNCI八人×双K全16cell，原始字节→全部K标签右删失→冻结模型→448评分重载→464requested含16incomplete→NLL/coverage/真实offline时延；独立算术/封印验证可追溯。真实Γ/Ω与逐事件A/B自动识别、R1/R2完整联合仍未解决，不以这次Adapter掩盖。

E0源码已把实际inner输入、完整候选schedule、fit/score与执行snapshot纳入复用；旧四hash不足时不冒充新完整身份，source/target contrast计划分开。未确认新的E0 P1。

六指南和实施报告准确保留新连续route非匹配旧算法收益、概率过置信、无实时/成人结论、完整四库成熟性能和S5尚未执行的限制。原四项中的成熟工程与真实R2/E5 Interface 已按上述范围关闭，没有新确认P1；不把软件闭合等同S1/S3/S5真实性能完成。最终全套验收由主任务统一记录，本复核没有重复跑全suite。
