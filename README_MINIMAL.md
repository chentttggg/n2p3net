# N2P3-Net strict-past minimal cloud package

这个包只保留当前 GTN strict-past 训练所需代码，不包含 cache、数据、测试、历史归档或虚拟环境。

## 首次配置

```bash
cd /root/autodl-tmp
tar -xzf n2p3net55k-min.tar.gz
cd n2p3-net
bash run.sh setup
```

`setup` 只执行一次：创建 `.venv`，保留已有 CUDA Torch；如果没有可用 CUDA Torch，默认从
`cu128` 源安装，然后安装最小依赖。也可以显式指定：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash run.sh setup
```

配置完成后，`run.sh`、`patch.sh`、训练、监控和 dashboard 默认严格使用项目
`.venv/bin/python`；不会静默回退到系统 Python。只有 `.venv` 尚未创建时，`setup` 才使用
`python3` 作为创建 virtualenv 的 bootstrap，可用 `PYTHON_BOOTSTRAP` 覆盖。

Windows 本地直接使用项目根目录的固定入口，避免等待系统 Python/py launcher 探测：

```powershell
.\python_project.cmd -c "import sys; print(sys.executable)"
.\python_project.cmd experiments\run_n2p3net_gtn.py --help
```

上传新版 cache 到：

```text
experiments/cache/gtn_events_v2_3ch_sf256_lf0.1_tm-0.2_tx1.2_nall.npz
```

检查和 benchmark：

```bash
bash run.sh check
BATCH_SIZE=4096 FOLD_JOBS=1 bash run.sh benchmark
```

## 训练

单张大显存 Blackwell 卡可以使用 4 个独立 fold process：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=4096 FOLD_JOBS=4 FOLD_CPU_THREADS=2 MAX_FOLDS=4 \
  bash run.sh train --fold-backend process --epochs 24 \
  --run-name pro6000_f4_b4096 > pro6000_f4_b4096.log 2>&1 &
```

`FOLD_CPU_THREADS=2` 表示每个并行 fold worker 使用 2 个 CPU/BLAS 线程；默认就是 2。GPU 利用率已经较高时不要盲目继续增加。

训练结果写入 `experiments/runs/<run-name>/`。cache 和运行结果不会被 patch 覆盖。

同一个逻辑 run 可以排队追加多个模型批次。第二次使用同一个 `--run-name` 时，系统会自动
追加 progress、连续编号 epoch/fold，并合并最终指标；不需要手动改 run 名：

```bash
nohup env FOLD_JOBS=2 bash run.sh train --run-name depth_sweep_20260826 \
  --encoder-depth 3 --max-folds 5 > depth3.log 2>&1 &
# 等第一条完成后再启动：
nohup env FOLD_JOBS=2 bash run.sh train --run-name depth_sweep_20260826 \
  --encoder-depth 4 --max-folds 5 > depth4.log 2>&1 &
```

dashboard 会显示累计的 `10/10` folds；`--fold-offset` 仍只用于选择原始数据的 LOSO 起始折，
不会用于区分这类模型批次。

## 图形化监控

后台训练启动后，记录资源曲线（PID 换成训练主进程 PID）：

```bash
bash run.sh monitor pro6000_f4_b4096 3306
```

监测器会同时写入 `resources.jsonl`（历史曲线）和 `resources.latest.json`（原子发布的最新一条样本）；dashboard 会直接读取这两个文件，不需要额外 API 或数据库。

页面上的“终止当前训练”按钮会先确认，再向当前 run 的训练主进程发送 `SIGTERM`；按钮不会终止 dashboard 自身，也不会直接发送 `SIGKILL`。dashboard 默认只监听 `127.0.0.1`，通过 SSH 隧道访问更安全。

打开 dashboard 后会自动 claim 资源监测器并每 5 秒发送心跳；页面会区分 dashboard/SSH 失联、资源监测器失联和训练数据未更新。关闭 dashboard 页面会释放由 dashboard 自动启动的资源监测器，租约超时也会自动回收；训练主进程不会因此停止。手工启动的 `run.sh monitor` 不由 dashboard 代管。

另开终端启动 dashboard：

```bash
bash run.sh dashboard
```

本地通过 SSH 隧道访问云端 dashboard：

```bash
ssh -N -L 18812:127.0.0.1:8812 -p 35832 root@connect.weste.seetacloud.com
```

浏览器打开 `http://127.0.0.1:18812/dashboard.html?run=pro6000_f4_b4096`。页面会实时读取
`epochs/fold_<fold>.jsonl` 的 epoch loss、phase、early-stop patience，以及
`resources.jsonl` 的 GPU 显存、利用率、worker RSS/PSS/private memory；PSS/private 曲线用于
区分 fork 共享页和真实私有内存增长。页面会分别显示 progress/epoch/resource 三个数据源状态；旧 run 若没有 epoch 文件，会明确显示“epoch 无记录”，但 fold 和资源曲线仍可用。

Windows 本地推荐直接运行仓库根目录的 `open_dashboard.ps1`：

```powershell
.\open_dashboard.ps1
```

如果系统禁止直接运行 `.ps1`，改用同目录的启动器：

```powershell
.\open_dashboard.cmd
```

它会自动建立或复用 `18812 -> 云端 8812` 的 SSH 隧道、检查 dashboard API、必要时拉起云端
dashboard 服务、打开不带 `?run=` 的页面并自动跟踪最新 run。首次连接时按提示输入 SSH
密码；密码不会写入脚本。云端 dashboard 服务异常时可能需要再次输入一次密码，但不会重启训练。
脚本创建新隧道后会保持当前窗口运行；保持该窗口打开即可持续使用本地端口，按 `Ctrl+C`
或关闭窗口会结束本地隧道，不会停止云端训练。

常用选项：

```powershell
.\open_dashboard.ps1 -NoBrowser
.\open_dashboard.ps1 -Run strict_past_full4_v3_dualckpt_20260825
.\open_dashboard.ps1 -Stop
```

## 后续快速 patch

云端环境配置好后，只上传新的代码 patch 包：

```bash
bash patch.sh /root/autodl-tmp/n2p3net55k-patch.tar.gz
```

patch 包应包含 `n2p3-net/` 根目录，内部只放需要更新的 `src/`、`experiments/` 或脚本文件。
脚本会先做路径安全检查和 Python 语法检查，再备份旧文件并原子替换；依赖没变时不再重新安装。

```bash
bash run.sh check
bash run.sh benchmark
```
