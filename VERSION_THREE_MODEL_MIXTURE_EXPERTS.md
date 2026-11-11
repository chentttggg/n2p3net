# 当前 main 工作树：三模型混合专家版

这是当前 `main` 未提交工作树整理后的三专家项目版本。三专家路线由以下实现组成：

- `NN source`：`src/transfer/ensemble_fit.py` 的 trial expert
- `waveform LDA`：同一专家合同下的经典波形基线
- `xDAWN ERP`：独立 ERP 空间基线

主要入口是 `experiments/run_candidate_ensemble_study.py`，核心合同在 `src/transfer/ensemble.py`、`src/transfer/ensemble_fit.py` 和 `src/transfer/ensemble_views.py`。专家名单、权重、失败策略、校准和保存重放必须由候选计划显式声明；缺失叶节点时整组拒绝，不能悄悄退化成少数专家。

本版本只发布源码、测试、配置和整合文档。数据、cloud_inputs、训练缓存、运行结果、模型载荷、环境、Git 历史和旧 frozen 归档不在分支内。
