# 全仓冗余审查与活动工具退役（2026-10-08）

本轮按用户要求审查整个仓库的重复逻辑、过度防御和旧临时工具，先追实际 caller、动态入口、保存格式与 Git 来源，再做可证明的删合。科研/云端并发作者持有的文件仅提出精确改动，不直接覆盖。

## 已落实

- PDS 选择记录的缓存/行身份准入集中到 `load_pds_selection_record`；两个 runner 删除完全重复的 private 包装，并将每次读取的两次完整构造/摘要校验收敛为一次，原 raw record 仍完整保存。
- `data.digests` 集中七个 runner 的文件 SHA-256，以及两组各自相同的数组摘要实现。普通 JSON header 与 compact JSON header 是两个不同的历史编码，分别保留；不将不同摘要程序强行合并。
- 删除 artifact 的单通道旧兼容包装及旧 PTP helper、dashboard 未使用的路径 helper；其他 function AST 与原版逐项相同。删除没有独立含义的 `SUPPORTED_METRIC` 与 `COUNT_AGGREGATIONS` 别名并联动真正 caller。
- `calibration_data_from_model` 只读取模型已有独立验证输出；删除两个从未使用的训练参数，两个生产入口不再为该函数布尔切片复制训练矩阵/标签。缺乏验证时仍拒绝 resubstitution。
- 删除 `_fit_attempt` 的三个未读取 private 参数及 `sample_source_rows` 的未读取 probability 参数；保原 feature-alignment 状态、预计算 CDF、每次 draws 与 RNG 状态。
- 将只校验、不改变 split 的 `selection_repetition_floor` 从五个源/测试/CLI/cache/lineage 位置完整退休；三种真正不同的浅层决策分母案例保留。真正的 `min_evidence_level` 声明门禁代码保持原行为，旧 cache/sidecar/freeze 原样保存。
- 半开 epoch 长度集中到 `data.contract.epoch_sample_count`，合同属性与两个 prepare CLI 使用同一原公式，删除两 private helper 并联动原测试；cloud 持有的 `epochs` 验证调用另以精确 patch 交总集成，不抢写。
- 先物理压缩并逐成员 SHA-256 验证，后退役 `requirements-minimal.txt` 与两个已完成用途的 BI2014a 一次性迁移/删除工具。当前依赖继续由 `pyproject.toml`/`uv.lock` 负责；当前 candidate 环境指纹不再列已退役文件，旧运行环境指纹不改写。

退役字节与恢复入口见[归档清单](../../frozen/redundancy_cleanup_20261008/manifest.json)。原始数据、cache、模型、结果、旧冻结载荷及其他工作树均保留。

## 覆盖与证据范围

开始时 inventory 为 929 个 tracked 或本轮科研新增文件、452 个 Python 文件。逐文件读取字节/身份并对全部 Python AST 做定义、import、call、重复函数、连续语句、未用参数和 private 零引用扫描；按 data/preprocess/model/train/transfer/research/experiment/test/config/doc/history 分模块追关键调用与保存链。此为全仓范围的结构及冗余审查，不声称逐行完成每个科学机制的语义证明或整个仓库行为验收。

九份 config 可解析；活动 README/AGENTS/index/六指南/frozen README 的 387 个本地链接均有目标。历史图片与空 package marker 有字节重复，但有各自证据或包语义，保留。RBF-MMD、causal/double-pass recovery、旧个人包 schema 仍具有不同数学或实际重放用途，未因外观相似删除。

独占变更的测试均在本地 Windows 仓库虚拟环境运行：PDS 初片 33 项；加入摘要共用后 35 项；独立校准/训练路由 2 项；CDF draws/RNG、真实 consumer、hash tamper 与 resume 9 项；compact 摘要入口 2 项；floor 退役和时间轴原有用例 8 项。不同阶段有重叠，不能把合计当独立测试数。另对两种数组 wire 编码逐版本比较 45 组 dtype/shape/stride/Unicode/empty 输入摘要，72 组时间轴与原 math/numpy 公式对照，四种大小文件与完整 SHA-256 对照；缓存/行身份和无 resubstitution 行为探针通过。Ruff 与 `git diff --check` 通过。

没有执行本地全套测试、公共矩阵或云端训练；测试中的合成小模型 fit 仅验证软件路径。方法收益、真实公共输入和成人设备确认仍由其原科研/运行任务分别验收。

## 尚待单作者收口

时间轴 canonical helper 已接合同和两个 CLI；cloud writer 持有的 `epochs` 验证调用仍需总集成吸收云 capsule 时应用已交付的 import/expected patch，保实际 declared n_times 匹配门禁。cloud ERP 的未调用 `_saved_replay`、root ensemble 的纯转发 `_predictor_digest` 与 source runner 未读取 seed_pairs 已交当前作者；这些 owner 项尚未记 complete。

`.gitattributes` 可由逐文件清单收敛为 active Python 路径模式，但原 index 有 96 CRLF、75 mixed 文件，另 167 working CRLF/index LF 文件；模式迁移后的未来 checkout 会改变相应执行字节。本轮没有全库 renormalize 或将旧 freeze 重标为新字节。

当前覆盖、作者清单、环境与验证记录见[机器验收记录](../assets/redundancy_cleanup_20261008/review.json)。详细 inventory、diff 与独立源绑定保存在本次工作树的 `tmp/research_line_20261008/REDUNDANCY/`，最终合入由科研总集成协调。
