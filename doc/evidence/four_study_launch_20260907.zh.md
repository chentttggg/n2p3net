# 四库整库留出开发启动

最终状态：单seed=20260906的四折source及123人六臂评估已完成。
核验2214个唯一participant/K/arm单元，全部completed，无protocol排除。
性能重复任务未计入该数量。CUDA非严格确定性的重复预测差异见文末。

本轮运行目录：云端`/root/autodl-tmp/n2p3-runs/four_study_development_20260907b`。
源码隔离目录：`/root/autodl-tmp/n2p3-four-study-20260907b`；物理快照为
`frozen/four_study_development_20260907b.source-only.tar.gz`，SHA256
`d58db080b70b65bec8858f82522ed004db63de571dee5480d3c25822d4c7c26e`。
这是含未提交修改的冻结工作树，base commit不能替代源码归档。

| 开发库 | 作者准入被试 | epoch | Cz参考后的六槽观测mask |
|---|---:|---:|---|
| BI2014a | 61 | 57437 | 0,1,1,1,0,1 |
| BI2015a | 41 | 63684 | 0,1,1,1,1,1 |
| BNCI2014-008 | 8 | 33600 | 1,1,1,1,1,1 |
| Huebner2017 | 13 | 162757 | 1,1,1,1,0,0 |

槽位为Fz/P3/Pz/P4/PO8/Oz，来自七导产品布局实测Cz差分，不能称七路独立EEG。
未观测槽保持零和mask=false。每折整库排除后核对source观测覆盖和canonical身份，
每个source库内部按participant划分验证，训练/验证均为库等权、库内被试等权。
BI2014a排除62-64，BI2015a排除1/27。未对外部Mainsah执行模型评分。

用户随后缩减为单seed=20260906，K=48/96/192。六臂：zero-shot、源head微调、全网微调、
随机初始化target-only、原波形收缩LDA、冻结源特征收缩LDA，共72个汇总组合。
实际请求123人，预计2214个subject/K/arm单元，失败保留记录。
原三seed计划保留为历史argv；旧调度器已停止，20260907/08不执行。
source最多30epoch、batch512、pos_weight8，编译+BF16+fused Adam；
target采用已声明强配方60epoch/lr1e-3，三种神经网络fit共享预算与校准统计规则。
本轮固定配方不看目标test选参，不构成等source更新数的机制因果比较。

BI2014a/BNCI使用单recording校准prefix及最大K之后共同purged test；
BI2015a/Huebner使用原始顺序前两个session校准、session2共同test。
Huebner不是跨天，BI2015a session间闪烁时长变化不是纯时间效应。
source模型先按源participant验证选epoch，然后完整source refit。

性能：云端RTX5090 32GB、25核配额、90GB内存限额。评估每study/seed仅载入一次缓存，
共用各K的zero-shot和冻结特征；特征预测分批256，避免整个大test一次进入GPU。
native/Torch线程使用统一预算。缓存构建一次、各轮复用；新prepare入口按实际CPU配额
并行独立库加载。本轮已完成的旧串行prepare未为测加速而重新构建，不能声称实测提速比。

第一折留出BI2014a：62名source被试、260041epoch；fit日志38.49秒，最佳epoch14，
训练与验证各库达成权重1/3。日志确认BF16/compile reduce-overhead/fused Adam实际生效，
峰值分配约1.2GB，零OOM重试。随后单seed矩阵已完成，状态以本文顶部最终核验为准。

启动a尝试因manifest缺少byte_size被训练前校验拒绝，没有产出模型；错误日志保留。
b清单经生产验证器通过，输入复用a已完整生成的缓存和哈希，不重复构建。
本地相关源验证/多域/划分测试39项、runner反例2项、Mainsah审计反例3项通过。
外部准入证据见[Mainsah审计](mainsah_s2_admission_20260907.md)，证据已物理压缩存档。

当前入口为`continue_single_seed.py`，`single_seed_execution.json`保存每项argv、状态、耗时及错误；
各目标`results_*/progress.jsonl`、逐cell JSON、预测NPZ和模型PT持续落盘。
已运行的BI2014a目标评估保留且完成1098单元；其他source采用两个worker并发。
目标阶段用同四名被试、同配方比较2/4worker耗时，选较快并发数后按被试分组执行，
性能探针单独标记，不重复计入科学结果。增量机制不混入plain主比较。

并发实测：同四名BI2014a被试、同三K六臂、单进程一条native线程，
2worker为34.472秒、4worker为18.749秒，端到端吞吐提高1.84倍，选择4worker。
运行中GPU利用率采样94%-99%，显存约3.2GB。CPU/缓存阶段与GPU阶段分开观察。
72份重复预测中校准/test行与标签完全相同；zero-shot、classifier_fine和两种LDA
预测逐值一致，全网微调与scratch并不数值等价（最大logit差0.929）。
当前CUDA训练未启用严格确定性，因此不能将单seed解释为逐bit可重放，
也不能将这两种神经训练的变化作为额外机制效应。原始差异保留在
`parallel_performance_verified.json`的`numerically_equivalent=false`记录中。
额外同seed重复再次复现全网微调和scratch差异，而其他四臂仍逐值相同；
因此不将数值差异直接归因于worker数量或作为性能优化保持逐bit结果的证据。
完整运行归档为`experiments/runs/four_dataset_20260907/four_study_development_20260907b.results.tar.gz`，
云端SHA256为`4b311349f18c9f487d8ea9ec80bdc1834a6dfd4fcc335212b02dd61587bb93c8`。
