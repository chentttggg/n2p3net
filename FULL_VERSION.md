# N2P3-Net 信息完整版（无大体积数据）

这个版本保留当前 `main` 工作树的源码、测试、配置、现行文档、历史说明和小型验收元数据，方便接手者读懂项目并重建环境。

保留内容包括：

- `src/`、`experiments/`、`scripts/`、`tests/` 和 `configs/`
- 六份现行指南、四个证据主题、交接说明和早期文字参考
- 小型 JSON、CSV、YAML、TOML、Lean 源码和验收 manifest
- 三模型混合专家与单 NN source 两条实现入口

排除内容包括原始 EEG、BrainSync 录制、cloud_inputs、训练缓存、运行结果、预测载荷、模型权重、PDF/图片、压缩包、虚拟环境、Git 历史和旧 frozen 归档。每个被排除文件都记录在 `OMITTED_LARGE_DATA.json`，按路径、类别和大小汇总。

这个版本可以读代码、读合同、跑不依赖数据的单元测试，也可以按 `pyproject.toml` 和 `uv.lock` 重建环境。它不能直接重跑历史训练或推理；那些任务需要由数据保管方另外提供输入和模型。
