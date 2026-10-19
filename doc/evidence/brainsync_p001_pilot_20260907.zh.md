# P001 本人数据与云端迁移先导实验

日期：2026-09-07。用户确认活动目录中所有 P001 开头文件夹属于同一人。
本次完成真实采集记录的派生输入整理、云端训练、后续 session 评分和全部有效记录 refit。
独立 participant 为 1，年龄未声明；这不是成人总体或可靠九选一的确认结果。

**后续撤回：** 用户已拒绝`P001_20260903_170721_01`，本页旧五会话缓存、个体模型、
结果与交付包已从活动路径撤下并压缩校验，见[质量处置](brainsync_transient_quality_20260907.zh.md)
和[归档清单](../../frozen/brainsync_quality_20260907/manifest.json)。
以下分数仅记录当时包含该会话的实验，不是排除后的结果；没有训练替代模型。

## 数据整理

原目录为 `D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/data`。
8 份 P001 记录均有可读 EEG；其中 3 份 `recording_error` 仅保存 121/126/132 秒，
无法覆盖完整刺激，已从活动目录移入本次结果的 `excluded_recordings/`。
递归删除被工具策略拒绝，因此原失败文件并未永久删除。原始 8 份记录先压缩并逐文件 SHA-256 校验。
P002 开头目录不属于本次授权的 P001 分组，没有纳入训练。

| 规范 session | 原始内部 subject | 目标 | 可用 trial / target |
|---|---|---|---|
| P001_20260903_170721_01 | P001 | 1 | 441 / 49 |
| P001_20260903_172410_01 | P002 | 2 | 441 / 49 |
| P001_20260903_173947_01 | P003 | 3 | 441 / 50 |
| P001_20260904_171926_01 | P004—1 | 4 | 441 / 50 |
| P001_20260907_150258_01 | P005 | 5 | 441 / 49 |

派生副本显式记录原编号到 P001 的用户确认映射。前四份原记录第 6 槽已经声明 `eog`，
仅将其旧标签 PO7 规范为 EOG，并移除错误头皮坐标，未修改其物理 EEG/EOG 角色。
旧 EEG 坐标逐一匹配 MNE `standard_1005` 原生模板后，按模板 fiducial 变换注册到 head；
最新记录已使用注册模板。变换矩阵、原/新坐标和导出器源码摘要均保留。
EDF 导出前后逐通道数字样本位级一致，物理量程、采样率也一致。

最新记录缺少 EEG JSON 和 events TSV，使用原始追加式 `recording_marker` 日志及原 session
元数据，通过采集端现有 BIDS 导出器重建。全部派生记录再通过后端严格读取。
输入为七 EEG、独立 EOG 保留在 raw；采用实测 Cz 差分得到六导 FZ/P3/PZ/P4/PO8/OZ。
128 Hz、V、0.1–30 Hz 因果四阶 IIR、稳态初值、[-200,1200) ms、mean-only baseline，179 点。
本轮没有拟合 EOG 回归或学习型伪迹 QC，也没有依据测试分数删除 session。

每 session 原排程450 flash，严格入口各排除9 trial，总计2250 scheduled / 2205 usable /
247 positive。2026-09-07后续逐事件复核更正：缺失的是B002-T001至B010-T001，即休息后
首刺激的-200 ms基线跨入前一休息，非此前误写的前九block末次刺激缺尾部。
旧五会话缓存均为block 1有45个epoch、block 2-10各44个，只有第一个block满足严格R5，
覆盖上限仍为1/10，指标和预测不改写。证据及原文归档见[研究方案](brainsync_signal_quality_design_20260907.zh.md)。

## 方案与隔离

前四个 session 为校准池（1764 trial / 198 target）；前三个用于内部拟合（1323），
第四个用于验证（441）。9 月 7 日的第五个 session 始终留到选方完成后评分（441 / 49）。
完整 session 隔离，同时验证校准标签可用时间早于测试开始。

三个源均为已经完成的公开 ERP 监督 checkpoint：BNCI-only、BI2014a+BI2015a+BNCI，
以及 BI2015a+BNCI+Huebner。后两个来自对应整库留出预训练，名称表示其实际源组成。
复用源权重，不重新运行公共 source 训练；三个源均通过目标身份、Cz 参考、通道和输入门禁。

每个源比较：轻微调（15 epoch / 1e-4）、完整微调（60 / 1e-3）、冻结 BN buffer 强微调，
以及同结构、同数据、同学习率/遍历预算的 scratch 低/高两臂。另有原波形收缩 LDA，共 16 臂。
NN 使用校准统计、AdamW、wd=1e-4、seed=20260907、预声明 batch 上限 512，尾批真实保留。
512 是本次执行配置，不是用户逐字指定值或准确率选优结果，使用 `operator_fixed` 规则记录。
不改变其他实验的 `user_fixed` 或 B0 选择合同，不做本轮之外的 K 档扫描。

各臂仅在校准内部评分，再从初始化完整 refit 校准池。跨臂优先内部 operational R5，
其次内部 AUC，平分优先基线/较小预算/固定名称。`selection_before_test.json` 在任何 test 评分前保存。
全数据待 refit 的配置也在此时固定，随后不根据 test 改赢家。

## 实测结果

| 方法 | 内部验证 AUC | 后续测试 AUC |
|---|---:|---:|
| 预先选中：BI2015a+BNCI+Huebner 轻微调 | 0.6029 | 0.4926 |
| BNCI-only 轻微调 | 0.6002 | 0.4802 |
| BI2014a+BI2015a+BNCI 轻微调 | 0.5226 | 0.5475 |
| 同预算 scratch 轻训练 | 0.5635 | 0.4533 |
| 原波形 LDA | 0.4473 | 0.5200 |

16 臂的严格 R5 均为 0/10：9 次覆盖不足拒判，1 次完整 block 预测错误。
10 个 block 是一个固定目标 session 内重复观测，不是 10 次独立 target-switch。
共有 9 组同预算迁移/scratch 配对，其中 2 组 delta AUC 为负；这不是 9 个独立人的负迁移率。
预先选中方案相对同预算 scratch 为 +0.0393，但自身 AUC 接近随机，不能宣称已获得实用收益。
测试中 0.5475 的方案未被追认成选方赢家。当前数据仅支持这一次后续 session 的描述性读数。

## 性能与产物

目标服务器为 RTX 5090 32 GB、Torch 2.8.0+cu128，CPU cgroup 配额 25 核。
同两次校准工作，1 个进程端到端 11.507 秒，2 个进程 8.548 秒，减少 25.7%；
正式使用两个 spawn 进程、每进程一个 Torch/BLAS/OpenMP 线程，不共享 RNG/CUDA 训练上下文。
16 臂校准与 refit 共 23.952 秒，4 个全部数据个体包拟合共 7.526 秒；计时不含上传和探针。
本路径复用 SubjectAdapter 的矩阵预载和 GPU 调度；没有套用 source 训练的编译性能结论。
完成后服务器 GPU 为 0% / 2 MiB，没有遗留本次 trainer。

三个源各自选中的轻微调，以及 LDA，当时用全部 5 session / 2205 trial refit。
以下包现已撤下，历史字节保存在上述校验归档的`historical_pilot/run/deployment/`中：

- 校准预选多源模型：`bi15_bnci_huebner_full_low.pt`
- BI 与 BNCI 多源模型：`bi_bnci_full_low.pt`
- BNCI 模型：`bnci_full_low.pt`
- 本人 LDA：`waveform_lda.pt`

这些旧全数据包还包含原留出 session，因此没有独立测试成绩；现已撤回，不能继续部署。
本轮没有将模型设为采集应用的生产默认。新采集需为休息后首flash保留完整刺激前基线，
同时为末次flash保留完整epoch尾部，
并继续补足目标范围及独立新 session，不能只增加优化器遍历弥补这些缺口。

本地完整证据在 `experiments/runs/brainsync_p001_20260907/`，远端为
`/root/autodl-tmp/brainsync_p001_20260907/`。源执行快照见
[源码清单](../../frozen/brainsync_p001_20260907/source.manifest.json)。
结果包 SHA-256：`e785461e4ad14f7562a400733edbbfeff1aefccff76ae370b731b6de97d5156a`。
`verification.json`逐一核验 16 份预测的 y/trial 对齐、AUC、全部 R5 分母及配对行摘要，
4 个个体包的 SHA/participant/完整拟合量与重载均通过。CPU 重放 GPU 预测最大 logit 误差
为 1.91e-6；全数据模型对旧校准记录的独立测试误用门禁也通过。

全库检查期间有其他任务并发修改 source-chain/checkpoint 代码，首次出现 7 个导入失败；
这些文件不属于本次修改，训练使用独立冻结副本。软件检查的最终记录见本次 `validation.json`。
冻结执行副本相关检查为109通过/2跳过；全库1177项首次仅缺一个Git忽略的公共实验计划fixture，
补入原文件后该项复测通过，合并最终结果为1168通过/9跳过。冻结源码全量Ruff与
工作树`git diff --check`通过；没有为通过测试改写已执行源码。
