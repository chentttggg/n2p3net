# 当前 main 工作树：单 NN source 版

这是当前 `main` 未提交工作树整理后的单源模型项目版本。工作站入口是 `experiments/brainsync_model_task.py`，固定配置是 `configs/brainsync_old_basic_nn_source.json`，配置声明 `old_basic_nn_source` 的 source 初始化、目标前缀统计、校准 BN buffer 和 90 轮配方。

该版本只保留单个 NN source 适配路径。个人模型和源权重需要由使用者按输入合同另行提供；当前分支不包含 EEG、cloud_inputs、缓存、预测结果或模型文件。

本版本与三模型混合专家版共享数据合同、校准安全检查、保存重放和测试基础设施。它不把 LDA、xDAWN 或其他专家隐式加入单模型入口。
