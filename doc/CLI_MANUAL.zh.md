# N2P3-Net 现行命令行手册

> 工作目录：D:\Temp\n2p3-net。本文只列当前代码中存在的入口。

## 1. 环境

    cd D:\Temp\n2p3-net
    .venv\Scripts\python.exe -m pytest -q

所有脚本可用 --help 查看完整参数。数据缓存、运行记录和 checkpoint 默认不进入 Git。
云端上传、依赖核验、SSH 转义和后台任务启动规则见
[`cloud_operations.zh.md`](cloud_operations.zh.md)。

## 2. 准备通用 EEG 缓存

### MOABB

    .venv\Scripts\python.exe experiments\prepare_eeg_dataset.py moabb \
      --dataset-class BNCI2014_008 \
      --subjects 1-8 \
      --channels Fz,Cz,P3,Pz,P4,PO7,PO8,Oz \
      --montage standard_1005 \
      --tmin-ms 0 --tmax-ms 1000 --n-times 256 \
      --baseline-mode trial_reference \
      --trial-reference-window-ms 0,50 \
      --trial-reference-center mean \
      --trial-reference-scale none \
      --output experiments\cache\bnci008_neural_ride_v8.npz

--dataset-class 是已安装 MOABB 的公开类名；--subjects 支持 1-8,12；
--channels native 保留原生 EEG 布局，也可给逗号分隔的精确子集。
`trial_reference` 使用 epoch 内显式物理时间窗做逐试次参考，适用于没有刺激前基线的
0 ms 起始数据；它不改变 `trial` 的严格刺激前基线语义。参考窗、中心统计量和尺度统计量
都会写入缓存的 preprocessing record。

### 原始文件 manifest

    .venv\Scripts\python.exe experiments\prepare_eeg_dataset.py manifest \
      --manifest data\adult_oddball.json \
      --output experiments\cache\adult_oddball_neural_ride_v8.npz

manifest schema、事件格式、montage 和布局策略见 [datasets.md](datasets.md)。
成功后同时生成同 stem 的 .record.json。

## 3. 通用 binary P300 LOSO

    .venv\Scripts\python.exe experiments\run_eeg_loso.py \
      --dataset-cache experiments\cache\bnci008_neural_ride_v8.npz \
      --models n2p3net,eegnet \
      --epochs 30 \
      --batch-size 256

常用参数：

| 参数 | 含义 |
|---|---|
| --dataset-cache | 必填，n2p3net_epoch_dataset/2 NPZ |
| --models | n2p3net,eegnet 的任意子集 |
| --subjects | 只取前 N 名被试做 smoke |
| --fold-jobs | 同时训练的 fold 数；5090 可设为 `4`，串行为 `1` |
| --validation-subject-fraction | 训练折内部被试级验证比例 |
| --erp-calibration fold/fixed | fold-local prior；fixed 仅允许 development 内置 prior 或独立冻结 prior |
| --frozen-erp-prior | 独立开发集冻结 prior JSON；confirmatory fixed 模式必填 |
| --lambda-morphology-l0 | 可选形态原子的 Hard-Concrete/L0 稀疏权重 |
| --variance-warmup-epochs | 仅训练均值、关闭方差 NLL 的 epoch 数 |
| --variance-ramp-epochs | 方差 NLL 线性升权 epoch 数 |
| --recon-bootstrap-samples | 折内类别分层 ERP target bootstrap 次数 |
| --recon-split-half-repeats | averaged ERP split-half 可靠性重复次数 |
| --run-dir/--run-name | 版本化运行记录 |

每个模型写 manifest、逐 fold `progress.jsonl`、训练中的
`epochs/fold_<fold>.jsonl` 和最终 `record.json`。BNCI2014、BI2014 及其他
符合 `EpochDataset` schema 的二分类 P300 缓存共用这一入口；dashboard 会在 epoch
开始写入后自动显示训练/验证 loss、早停状态和资源监测。分段运行可用
`--fold-offset N --max-folds K`，显示编号仍保持原 LOSO fold 编号。

## 4. 辅助域预训练

    .venv\Scripts\python.exe experiments\run_eeg_pretrain.py \
      --dataset-cache experiments\cache\bi2014a_neural_ride_v8.npz \
      --model eegnet \
      --channels Fz,Cz,Pz \
      --epochs 30 \
      --device cuda

--channels 是可选精确子集，缺通道会失败。验证集按 subject 切分，不允许试次随机泄漏。
输出 checkpoint 和 JSON 报告；辅助域指标不能替代 GTN 验收。

## 5. GTN 与两条路线

默认命令走正式 PCW 路线：

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
      --lambda-innovation 0 --run-dir tmp\production

strict-past 是显式研究路线，不是默认增强项：

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
      --lambda-innovation 1 --audit-subjects 4 --run-dir tmp\strict-past-research

full-Z2 auxiliary head 是 E5 claim-gate 研究对照，生产默认 `--z2-aux-head off`：

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
      --z2-aux-head add --z2-aux-pool attention --run-dir tmp\z2-aux-add-research

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py `
      --z2-aux-head replace --z2-aux-pool attention --run-dir tmp\z2-aux-replace-research

`add`/`replace` 只生成命名研究 recipe；未过预注册嵌套门槛前不得进入生产默认，
且 `head_z2` 永远不称为成分级证据。

`--lambda-innovation 0` 对应 `neural_ride_v12_pcw_fail_closed`，正数对应
`neural_ride_v12_strict_past_research`。运行后以 `record.json` 中的 resolved recipe、
`use_innovation_likelihood` 和嵌套 `M0:a+bS` vs `M1:a+bS+cL` 审计结果为准；不要只
依据 run 名称判断。strict-past 未通过嵌套 cluster bootstrap 门槛时必须保留
`final == PCW`，不能写成正式性能提升。路线边界和旧方法删除规则见
[routes.md](routes.md) 与 [blueprint.md](blueprint.md)。

生产默认已启用 v12 additive-LLR 主干与 fidelity 审计；`--repetition-state-residual` 只启用
初始为零的 state residual（须过 audit gate 才可非零），`--repetition-state-residual-l2-weight`
是 blueprint 3.2 的 delta² shrink 权重。legacy v11 仅作为显式历史/负对照 recipe 保留。

仅在开发态比较验证端点与实际泛化轨迹时，可追加 `--epoch-trajectory-audit`。它会把每个
raw epoch checkpoint 写入 `epochs/checkpoints/fold_<fold>/epoch_<epoch>.pt`，并在 fold 训练
结束后生成同目录的 `trajectory.json`，记录 task/objective validation loss、innovation NLL、
outer-test trial AUC、all-trial digit hit 和 `prefix_minK_sum@3` hit。outer-test 轨迹严禁参与
checkpoint 选择；confirmatory 模式会直接拒绝该参数。该审计不逐 epoch 重做 repetition
temperature/density refit，因此不包含 chain hit。

先生成当前 [-200,+1200) 缓存：

    .venv\Scripts\python.exe experiments\run_gtn_baseline.py \
      --epoch-tmax 1.2 --prepare-cache-only

Neural-RIDE benchmark：

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py \
      --subjects 5 --epochs 16 --batch-size 256 --benchmark

AutoDL Linux + NVIDIA GPU 的开发全量/分段运行使用独立多进程 fold worker；每个 worker
拥有独立 Python/CUDA context。5090 首次验证建议 4 个 worker：

    ./.venv/bin/python -u experiments/run_n2p3net_gtn.py \
      --device cuda --seed 0 --evaluation-mode development \
      --batch-size 1024 --fold-jobs 4 --fold-backend process \
      > experiments/gtn_seed0_b1024_process_f4.log 2>&1 &

确认 `nvidia-smi` 中出现多个 python PID 后，再视显存和吞吐调整 `--fold-jobs`。

实时查看最近一次 GTN 运行：

    .venv\Scripts\python.exe experiments\watch_gtn_progress.py
    ./.venv/bin/python experiments/watch_gtn_progress.py

指定 run 或只输出一次：

    .venv\Scripts\python.exe experiments\watch_gtn_progress.py --run <run_name>
    .venv\Scripts\python.exe experiments\watch_gtn_progress.py --once

strict-past 研究运行的 epochs 必须覆盖 variance warmup/ramp 和至少一个 joint epoch；
正式 PCW 路线的 variance objective 已禁用，不得把研究分支的日程要求误当成生产门槛。
正式运行：

    .venv\Scripts\python.exe experiments\run_n2p3net_gtn.py \
      --epochs 30 --batch-size 1024 --primary-decision exact_llr@3

跨数据集训练只走独立的多 montage 验收入口：

    .venv\Scripts\python.exe experiments\run_multidataset_transfer.py --help

## 6. 锁定多种子复现与确认性评估

    .venv\Scripts\python.exe experiments\run_locked_multiseed.py --help
    .venv\Scripts\python.exe experiments\run_pcw_claim_gate.py --help
    .venv\Scripts\python.exe experiments\run_paired_test.py --help
    .venv\Scripts\python.exe experiments\run_with_dashboard.py --help

主任务结论必须遵守 [evaluation_protocol.md](evaluation_protocol.md) 的完整 ITT cohort、多 seed
和被试级配对。当前 GTN 的全部 245 个 eligibility 单位均已开发暴露，只能做 locked
development/replication；它的 confirmatory 入口会 fail-closed。一次性 confirmatory lock 仅可
用于新增受试者或从未查看过的外部 cohort，神经网络正式运行显式要求 CUDA。

## 7. 已归档入口

数据集专用下载脚本、BNCI/BI/ERP CORE 专用 LOSO、旧辅助预训练脚本、日期化诊断脚本和旧缓存
均已从现行代码删除。完整快照、逐文件校验清单与 SHA-256 见
[归档索引](../archives/README.md)。归档只用于隔离复现，不加入 PYTHONPATH。
