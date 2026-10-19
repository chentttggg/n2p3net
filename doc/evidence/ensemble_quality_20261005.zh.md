# 候选多专家主线：独立代码质量评估

日期：2026-10-05。对象为[本轮实际代码实施](ensemble_implementation_20261005.zh.md)，固定比较`git diff a6a3da5...d3e6027`；受审HEAD为`d3e602701716748d0ca561400699204268d4f8d6`。两名独立reviewer分别评估实现标准和规格符合性；主线根据实际RED反例修改，再由原reviewer独立复验。结论限于本轮软件范围。

## Standards

**passed，未发现未解决的已验证标准缺陷；Fowler判断性smell为0。** 依据AGENTS、六活动指南及用户“减少防御性编程、保留并发工作、保持树干净”的要求，不将格式工具已覆盖的问题重复计为发现，也不按行数评价Module深度。

最终独立复验确认：复数概率、log-evidence及prior入口拒绝；numeric state无公开可变tensor，深层record/温度与scorer/participant属性只读；已验证的同一份模型bytes直接解码；配方、AR阶数、实际模型和拟合参与者错配均拒绝；异常同时保存父/子failed终态并保留已完成cell；缺少scratch/waveform匹配对照在任何fit前拒绝。

source checkpoint一次path读取被full与三个OOF拟合共用，四次decoder输入摘要完全相同；冻结后替换文件不影响执行。Burg合法数据退化保留具名fit_failure，非法AR参数仍传播合同错误。

最终HEAD相关八个测试文件117项通过，失败/错误/跳过均0。Ruff及diff check通过。144文件执行清单覆盖实际源码、入口和用到的fixture/探针，探针到测试收尾逐字节保持一致；核对时树clean。原始报告与证据在质量压缩包的`tmp/ensemble_implementation_20261005/standards_review/`，最终报告为`final_standards_d3e6027.zh.md`，最终探针为`final_8knyb1oz/final_candidate_results.json`，关系反例为`relations_lojat2tz/artifact_relation_results.json`。

架构收敛采用已验证入口和只读数值状态；`FrozenTrialPredictor`成为ERP/NN保存与重放的共享owner，旧comparator的重复decode被删除。物理view、授权label、概率证据与requested评价分别有单一职责；不能以减少检查为由删掉外部身份合同。异常只在研究边界将已识别数值退化归为fit_failure，其他ValueError不被笼统吞掉。

## Spec

**PASS，当前实施范围没有剩余已验证Spec缺陷。** 六个独立行为探针均退出0，13项行为判定全部通过。无标签score、完整概率/operational覆盖、requested participant/seed风险、固定producer/权重、source整目标库排除、同结构scratch、实际分组OOF及模型重放已接通。

| 独立探针 | 检测到的缺口与最终行为 |
|---|---|
| `probe_contracts.py` | raw不完整时保留missing；经典缺测不执行classifier；错参考/词表优先拒绝；raw拟合及推理绑定布局/参考；owner/scorer/temperature/state只读；waveform float32/64重放精确一致 |
| `probe_fixed_source.py` | source原可逐fold重读漂移；修后两真实outer fold在原文件被替换后仍消费启动时同一SHA，且实际拟合同输入scratch/waveform |
| `probe_expected_failure.py` | 合法零EEG原中断全run；修后两臂两fold均保留失败与requested分母，hit/概率coverage为0 |
| `probe_runner.py` | 项目外cwd真实MAT CLI执行，4个预测单元及逐专家保存重载完全一致，实际入口前后SHA一致 |
| `probe_artifact_binding.py` | 对外壳重签不能替换profile布局、participant、raw顺序、source来源、LLR offset或数值state；正常包精确重放 |
| `probe_source_oof.py` | source整目标库/目标参与者准入拒绝；source+scratch共8次真实fit、三个OOF组不含外部held，源文件中途改变后仍用同一冻结SHA |

129个src/实际candidate与ERP入口/环境文件的前后SHA完全一致，受审HEAD未变且树clean；fixture及探针另有SHA。原始报告在质量压缩包`tmp/ensemble_implementation_20261005/spec_review/final_d3e6027/spec_review.zh.md`，同目录`review_manifest.json`保留完整argv、cwd、返回码和输出。此前0f8074a复验途中HEAD改变的过程记录保留为历史，不作为最终单一冻结验收。

## 测试身份与原始证据

[验证收据](../assets/ensemble_implementation_20261005/validation_receipt.json)从原始JUnit解析：完整测试在`0f8074a`上2425通过/19跳过，58项质量回归随后形成该提交；只读字段调整后55项通过形成`d3e6027`，最终独立Standards再执行117项。各范围有重叠，不相加；本轮未在d3e6027重复执行整套。主线完整测试没有独立逐字节全集快照，提交身份与原始调用记录的边界在收据明确标注；最终两路受审实现另有独立字节清单。

原始pytest日志、JUnit、已记录的主线命令/退出码、reviewer脚本、RED→GREEN及synthetic模型通过[质量证据清单](../../frozen/ensemble_implementation_20261005/quality_manifest.json)绑定本地物理压缩包；JUnit中主机字段仅留在本地原件。19项跳过涉及独立前端、可选数据、Windows symlink权限和CUDA，不能据全套其余通过声称这些运行边界已验证。

[实际源码归档](../../frozen/ensemble_implementation_20261005/manifest.json)包含src/tests/configs、所有实际研究入口、环境、六指南、科研规格及本验收；其实际工作字节逐成员核验，并调用现有`verified_executed_snapshot`核对正在执行的文件与物理归档。提交作为版本上下文，工作字节SHA是归档身份；文档收尾晚于受审实现，不被伪称reviewer对新文档重新跑过测试。压缩包与镜像留本地，Git只分发清单。

## 未关闭的研究范围

完整四库/十二库公共配对矩阵、三个完整source→target流程seed、其他库的candidate/raw Adapter、复杂gating与非等权选择、fixed-holdback32/联合noise MLE/超参数积分、source-only K0、工作站接入、云端资源DAG/吞吐和成人独立确认尚未完成。两路最终探针只拟合synthetic fixtures；本轮原A01只读解析不属于公开训练或效能比较。

软件测试支持接口、数值、生命周期与信息边界结论，不证明候选算法提升AUC/hit或成人可靠性。本轮软件状态为complete；123项/15块完整科研优化目标继续active。
