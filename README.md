# N2P3-Net 与 BrainSync 稳定大版（2026-09-20）

此分支从已核验的历史物理快照恢复，保留后端真实执行源码和工作站发布源码；不含 2026 年 10 月的科研开发改动。

- 仓库根目录：09-18 最后完成的十二库公共训练后端，346 文件逐字节匹配原 source archive `c71a4d2109eaed23bea734c76845940bcf05d51a3af4478ac3c1db54fe75fd24`。历史 `f1a303c` 仅是当时 dirty tree 的 base commit，不是实际执行源码。
- `frontend/`：对应 09-20 BrainSync EXE 的工作站源码、配置和界面资源。19 个模块与原 EXE code object 匹配的历史证据保存在 `publication/final_evidence/stable_release_source_binding.json`。它使用独立 Python 3.10 环境。
- `publication/source_identity.json`：恢复、排除项与每个已发布原始源码文件的 SHA-256。四份实际 source archive 均逐成员重新核验；只有最终主后端在根目录物化，早期复用与准备来源保留原 manifest 和身份。

后端使用本目录的 `pyproject.toml` 与 `uv.lock`；参考 `doc/README.md` 的六份当期指南。工作站使用 `frontend/pyproject.toml`、`frontend/uv.lock` 与 `frontend/RUN_GTN_WORKSTATION.md`，并在其配置中指定实际后端路径。原模型、USB DLL、字体与发布 runtime 留在本地，需要从原发布目录恢复；本分支不拼装或替换运行时。

原始 EEG、准备缓存、模型、个人数据、完整结果及历史压缩包不分发。`frozen/major_domain_transfer_20260920/manifest.json` 保留原归档身份与外部资产位置；其中历史本地路径不表示 GitHub 包含对应资产。

公共开发结果原记录为 48/48 保存模型重放、4,082 逻辑单元（3,166 去重独立单元）。工作站默认 BNCI 源仍是原 `f3267cd4…` 来源，不是十二库部署源。本次发布没有新训练、未知 EEG 预测、设备连接或成人效能验证。
