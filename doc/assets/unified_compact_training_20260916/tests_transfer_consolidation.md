# 统一模型：个人适配与产品测试合并

2026-09-16。范围为 `tests/transfer/`、`tests/brainsync/` 的统一模型相关测试；本分支未修改生产实现。
依据 `doc/engineering.zh.md` 的“同一次拟合覆盖多个不变量”“测试文件不互相导入”和
`codebase-design` 的 interface 测试原则执行。既有稳定产品链没有整体搬迁。

修改前源码及测试已物理冻结于
[source.manifest.json](../../../frozen/unified_compact_training_20260916_before/source.manifest.json)。
本次确实删除重复文件及假拟合层，未通过跳过测试、缩小参数化收集范围或隐藏用例来压低数字。

## 文件归属

| 原文件 | 当前归属 | 处理 |
|---|---|---|
| `transfer/test_source_parameter_prior.py` | `transfer/test_subject_adapter.py` | 源中心、尺度、梯度、零权重、非法权重及实际拟合归入适配 owner；删除旧文件 |
| `transfer/test_joint_decision_adaptation.py` | `transfer/test_subject_adapter.py` | 单次更新、D0/D1 工作集、预算隔离和缺失组风险归入同一 owner；删除旧文件 |
| `transfer/test_global_early_stopping.py` | `transfer/test_epoch_selection.py` | 全局策略传播、严格拒绝非法内层划分、保存与复用账本归入选轮 owner；删除旧文件 |
| `transfer/test_calibration_fit_reuse.py` | `transfer/test_calibration_candidates.py` | 保留原四项拟合复用测试，按实际被测 owner 更名 |
| `transfer/test_calibration_decision_groups.py` | `transfer/test_calibration_candidates.py`、`transfer/test_target_recipes.py` | membership/质量/拟合留在 candidate owner；配方声明及单配方选择留在 recipes owner；删除旧文件 |
| `brainsync/test_unified_predictor.py` | 原文件 | 保留统一产品流程的独立测试入口；合并重复 CLI 校准 |

## 删除与合并理由

1. 原先单独执行一次单步计数拟合，再用两次拟合比较 D0/D1，再用两次拟合比较源先验。
   现只拟合三条轨迹 `(D, prior)=(0,0),(1,0),(1,0.2)`，固定输入、初始化与随机种子。
   每条轨迹均检查真实 AdamW 更新次数、microbatch 上限及抽样工作集；两条成对比较分别识别 D 和先验作用。
   保留原 source 不变、先验中心摘要和非零惩罚检查。拟合数从 5 降到 3。
2. 源先验的中心为零、恢复梯度和引用不别名断言复用同一个模型；全零源参数的有限梯度仍为独立反例。
   非法权重仍按三个独立输入参数化。没有删掉 prior 的独立数学检查。
3. 全局早停复用原先在八个神经网络臂重复相同生命周期。现在通过独立 manifest 测试检查全部八臂的政策传递，
   实际执行复用保留普通 NN、残差 NN、LDA 三种保存状态；同时检查修改 patience 必须重新拟合。
   此项实际 NN 拟合从 18 降到 6。全臂实际运行仍由现有 `test_structural_adaptation.py` 承担，未移动或删除该测试。
4. “严格全局早停禁止静默退回固定预算”并入既有非法内层划分测试。同一数据同时检查严格拒绝与仍受支持的显式历史 fallback，
   不将仍可调用的旧 interface 误删为死语义。
5. 删除只拦截 kwargs 的 `CaptureAdapter` 及其两个参数化用例，连同纯 remap 重复断言，换成真实 D0/D1 各一次拟合。
   仅在已有质量变换 seam 强制移除首个完整 R5，实际执行候选构建、QC scope remap、SubjectAdapter 与损失。
   原请求 3 组、首组全 `-1`、风险仍为各 `1/3`、后两组映射及 D0/D1 相同工作集均在返回账本中验证。
   原独立 direct/row-column 未来标签攻击和 observed-only 禁止伪造 membership 保留。
6. 删除单独重复两会话校准的 CLI 成功测试；短验证会话的 CE 反例现在同时运行默认 CLI dry-run 与实际 CLI 校准，
   检查未写模型的预检、真实选轮、最终保存及重载。普通与开启 QC 的模型重放两条参数化路径保持。
   BrainSync 的实际 NN 拟合从 8 降到 6。

## 实测工作量

计数在同一项目环境、相同 CPU 线程设置下完成，仅运行表中相关文件；不包含全套或其他目录。
包装 `SubjectAdapter.fit` 记录调用和正常返回，包装 `torch.optim.AdamW.step` 记录真实更新。
因此“NN 拟合”不计 LDA；失败准入调用分别留在 attempts 中。每个测试的明细保存在
[合并前计数](tests_transfer_before.json)与[合并后计数](tests_transfer_after.json)。

| 指标 | 合并前 | 合并后 | 减少 |
|---|---:|---:|---:|
| 相关文件 | 9 | 5 | 4 |
| pytest 用例 | 67 | 58 | 9 |
| `SubjectAdapter.fit` 调用 | 112 | 96 | 16 |
| 正常完成的 NN 拟合 | 105 | 89 | 16（15.2%） |
| AdamW 更新 | 294 | 254 | 40（13.6%） |
| 测试与本地辅助代码行数 | 1165 | 1047 | 118 |

单次实测墙钟为 12.10 秒与 10.29 秒，仅用于本次软件验证记录，不能推断稳定吞吐提升。
首次计数脚本漏注册 pytest 插件，导致计数器自身的 `KeyError(None)`；修正插件注册后重跑的有效基线退出码为 0，
上述 JSON 和统计均来自通过的两次运行。

## 定向验收

- 合并前 67 项通过，合并后 58 项通过；两次退出码均为 0。
- 最终补齐先验中心全参数不别名断言后，该单项再次通过；没有增加拟合或优化更新。
- 当前五文件 Ruff 和 `git diff --check` 通过。
- AST 检查未发现测试互导；共享造数继续来自既有 `tests/support/`。
- 七份活动指南、活动测试及直接实验入口未找到被删除测试文件的路径引用；历史证据中的旧命令保留原时点含义。

保留标签/时间/预算隔离、D0/D1 同工作集、完整 membership、缺组风险分母、源先验、单次 optimizer step、
早停完整范围重拟合和个人模型保存重放。此结果只证明测试整理后的软件行为；未训练公开数据矩阵。
