# 云端训练操作规程

> 版本：v1（2026-08-26）。适用于 PowerShell 本地端通过 SSH 操作 AutoDL Linux 训练机。
> 本文记录实际踩过的操作问题；每次上传或启动云端任务前按清单执行。

## 1. 固定事实

- 远端仓库：`/root/autodl-tmp/n2p3-net`
- 远端 Python：`/root/autodl-tmp/n2p3-net/.venv/bin/python`
- 远端数据缓存：`experiments/cache/`
- 远端运行结果：`experiments/runs/`
- 源码包不包含 `.git`、`.venv`、`mne_data`、`experiments/cache`、`experiments/runs`、`tmp`。
- 数据缓存单独上传到远端 `experiments/cache/`，不能混入代码 patch。

## 2. PowerShell/SSH 转义

PowerShell 中反斜杠 `\` 不是引号转义符。此前把 `\"` 嵌入双引号命令后，反斜杠原样到达远端，导致 Python 收到非法源码并报 `SyntaxError`。

规则：

1. SSH 远端命令尽量用本地 PowerShell 单引号包住，远端 shell 参数用双引号；不要把 Bash 的 `\"` 习惯带入 PowerShell。
2. 远端 Python 代码超过一个简单 import/print 时，不要嵌套在 SSH 命令中；把校验脚本作为代码包文件上传后执行，或使用明确的远端 heredoc。
3. 复杂命令先拆成“上传/解包”“依赖检查”“缓存检查”“启动”四次 SSH 操作，出错时定位清楚。
4. 密码只通过交互式 SSH/SCP 输入，不写入脚本、命令文件、日志或运行记录。

推荐先执行简单校验：

```text
ssh -p 35832 root@connect.weste.seetacloud.com 'cd /root/autodl-tmp/n2p3-net && .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
```

## 3. 上传与路径

源码包与数据缓存分开处理：

1. 本地打包时显式排除 `cache/runs/tmp`，打包后用 `tar -tzf` 检查归档列表。
2. SCP 目标目录必须明确。上传到仓库根目录的代码归档，解包后删除归档；上传数据时目标必须是 `.../experiments/cache/`。
3. 多文件 SCP 前确认远端目标目录存在；不要假设 `host:/path/` 会自动替文件归类到子目录。
4. 解包只覆盖代码路径，不删除远端历史运行结果。清理失败任务时只能删除本次明确创建的完整运行目录。
5. 上传后必须远端检查：缓存 shape、被试数、preprocessing record、代码导入、GPU 型号和 CUDA 可用性。

## 4. 依赖与设备

依赖安装后不能只看 pip 成功，必须实际 import。Torch 的 `major.minor` 和 torchaudio 必须一致，CUDA 后缀也必须一致。例如远端已有 `torch 2.8.0+cu128` 时，不能直接安装最新 `torchaudio 2.11.0`；这会在 import 时寻找不存在的 `libcudart.so.13`。

启动前最少检查：

```text
.venv/bin/python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import pandas, braindecode; print('runtime imports ok')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

若需要补包，先读取现有 Torch 版本，再按对应 CUDA wheel 安装；不要让通用 PyPI 依赖解析器随意升级 Torch 生态包。

## 5. 训练入口契约

- 通用 binary LOSO 入口必须从 `ScheduledEventTimeline` 生成 `row_acquisition_indices`，并传入 `evaluate_binary`；启用 repetition head 的 N2P3Net 没有 acquisition ordinal 会 fail-closed。
- fold 并行数必须是显式 CLI 参数 `--fold-jobs`，不能在 runner 内写死 `n_jobs=1`。5090 的并行实验使用 `--fold-jobs 4`，并在日志中确认 4 个 fold 同时出现。
- fold-local ERP 校准的 tau bounds 必须通过 `N2<P3a<P3b` 的绝对上界约束后才能构造 PCW；不能只验证峰值中心，不验证 bounds 可行性。
- BNCI 0 ms 起始数据必须确认 `baseline_mode=trial_reference`、参考窗 `0,50 ms`，不能误回退到需要负时间基线的 `trial`。

## 6. 启动与核验

默认不额外跑 smoke；先完成上面的导入、缓存和模型构造核验。正式任务使用独立 run name，并将 stdout/stderr 重定向到该 run 的日志。

启动后必须通过新的 SSH 连接检查三件事：

1. `ps` 中存在真正的 `python ... run_eeg_loso.py`；不要只相信 `echo $!`，SSH 的 shell wrapper 可能才是返回的 PID。
2. 日志已经进入 epoch 或 fold setup，且没有 traceback。
3. `nvidia-smi` 显示 GPU 利用率/显存变化，并与日志中的 fold 并行数一致。

后台启动建议使用 `nohup` 加独立会话，并保存日志；若 SSH 会话没有退出，不要因此杀掉训练，先用独立连接确认实际 Python 子进程，再只关闭本地 SSH 客户端。

## 7. 本次问题记录

| 问题 | 现象 | 固化规则 |
|---|---|---|
| PowerShell 使用 `\"` | 远端 Python `SyntaxError` | PowerShell 不用反斜杠转义；复杂 Python 改为脚本上传 |
| SCP 目标层级错误 | 缓存落在仓库根目录 | 代码归档和数据缓存使用不同的显式目标目录 |
| torchaudio 版本漂移 | import 找不到 `libcudart.so.13` | Torch/torchaudio 的版本与 CUDA 后缀必须配套 |
| 二分类入口漏传 acquisition | N2P3Net 在训练前报无 scheduled acquisition indices | runner 从 event timeline 生成 ordinal 并传给 evaluate |
| fold ERP bounds 不可行 | PCW 报有序 latency bounds 不满足 | 校准器输出前修复后续成分上界并留严格不等式裕量 |
| fold 并行参数写死 | 传环境变量仍只有串行 fold | runner 提供 `--fold-jobs`，启动命令显式传 `4` |
| 后台 PID/SSH 会话误判 | `$!` 与真实 Python PID 不同，SSH 客户端持续 | 通过独立 SSH 的 `ps + log + nvidia-smi` 核验，不盲杀训练 |

## 8. 结果与清理

- `progress.jsonl` 是运行中的事实来源；`record.json` 只在模型完成后生成。
- 训练中禁止覆盖已有 run name；失败重启前只清理本次失败的临时 run 目录，并保留历史正式结果。
- 本地上传用的 tar 包在远端解包成功后删除，避免根目录堆积压缩包；远端 cache、runs 和 checkpoint 按实验记录保留。
