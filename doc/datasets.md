# EEG 数据与通用入口规范

> 现行 schema：n2p3net_raw_manifest/1 -> n2p3net_epoch_dataset/2。

## 1. 模块边界

数据准备与训练严格分离：

    原始 EEG / MOABB
            |
            v
    prepare_eeg_dataset.py
            |
            v
    EpochDataset NPZ  --只含物理数据契约，不含模型配方
            |
            +--> run_eeg_loso.py
            +--> run_eeg_pretrain.py
            +--> run_n2p3net_gtn.py（GTN 专用 scheduled-event cache 由 run_gtn_baseline.py 生成）

训练入口只读取 (N,C,T)、二分类标签、subject id、通道名/坐标和预处理 profile。
数据集名称不会改变 Neural-RIDE 结构或 Loss。

## 2. 支持的来源

### 连续 EEG

src/data/dataset.py 使用 MNE 的格式分派器。可接入当前 MNE 能读取的连续格式，包括
EDF/EDF+、BDF、BrainVision (.vhdr)、FIF、EEGLAB SET、CNT、GDF、CTF、EGI/MFF 等。
某些格式需要配套 header/data 文件或可选依赖；MNE 无法读取时入口会保留原始异常并失败。

事件可来自：

- MNE annotations；
- TSV/CSV（sample 或秒单位 onset，以及可选 label/target/trial_type）；
- JSON 事件表；
- NPY (n,3) MNE event array；
- NPZ 的 events 和可选 labels。

GTN 的 NIX/HDF5 语义特殊，仍由 src/data/gtn.py 读取，再生成版本化 GTN 缓存。

### MOABB

prepare_eeg_dataset.py moabb 接受当前安装环境中任意公开 MOABB dataset class。
P300 paradigm 统一输出 epoch、标签和 subject/session/run metadata。BNCI2014-008、
Brain Invaders 2014a 和 ERP CORE 不再有专用下载或训练脚本。

## 3. 通道布局

一个模型 run 只有一个固定的物理布局，但布局不固定为 3/8/16 导：

- channels: null：单记录使用全部真实 EEG 通道；
- manifest layout_policy: intersection：多记录按规范化电极名取真实交集，顺序沿第一条记录；
- layout_policy: strict：所有记录必须具有相同电极集合；
- 显式 channels：每条记录必须逐一提供，缺任一电极立即报错。

标准入口禁止：

- 为缺失电极补零或 NaN 后伪装成观测；
- 用 AFz 替代 Fz，或把两个不同电极映射到同一位置；
- 只凭数组列号猜测通道；
- 对无坐标的厂商通道名生成虚构坐标。

通道坐标的来源按物理可信度排序：个人 Raw 内嵌 digitization（LPA/RPA/Nasion 注册，可选刚体
ICP）、设备/manifest 自定义 montage、MNE `standard_1005` 平均头。单位球只能作为显式记录的
最后 fallback，不能冒充真实头皮坐标。厂商别名只有在确认是同一物理传感器时才能通过
channel_aliases 显式登记。

标准预处理入口在未显式指定设备 montage 时自动优先使用 Raw 内嵌 digitization；默认
`standard_1005` 只在记录不携带 montage 时生效。显式设备/custom montage 视为人工覆盖并写入
provenance。

模型仍保留 channel_mask 契约，用于明确登记的单试次坏导/传感器失效实验。标准离线准备输出
的活动通道全部为真实观测；mask 不能用于跨记录补齐不同帽型。

`models/canonical.py` 提供注册后米制三维点上的 chordal Matérn-3/2 GP 投影和后验不确定性。
这不是头皮 geodesic；后者必须提供 mesh 与 Laplace--Beltrami 谱后另行实现和预注册。
它不是电极替代：未观测位置必须携带更高后验方差，且当前标准 LOSO 默认使用原生/交集布局。

## 4. Manifest 示例

    {
      "schema": "n2p3net_raw_manifest/1",
      "name": "adult_dry_oddball",
      "montage": "standard_1005",
      "layout_policy": "intersection",
      "label_map": {"NonTarget": 0, "Target": 1},
      "preprocessing": {
        "name": "neural_ride_v8",
        "sfreq": 256.0,
        "l_freq": 0.1,
        "h_freq": null,
        "tmin_ms": -200.0,
        "tmax_ms": 1200.0,
        "n_times": 358,
        "baseline_mode": "trial",
        "reject_threshold_v": 0.00015
      },
      "records": [
        {
          "path": "sub-01/eeg/sub-01_task-oddball_eeg.edf",
          "event_file": "sub-01/eeg/sub-01_task-oddball_events.tsv",
          "subject_id": "sub-01",
          "session": "ses-01",
          "run": "run-01",
          "reference": "linked-mastoid",
          "age": 34,
          "sex": "F"
        }
      ]
    }

路径相对 manifest 所在目录解析。未知字段会失败，避免拼写错误被静默忽略。
自定义帽型可把 montage 写为相对路径；Raw 已携带可靠坐标时写 embedded。

## 5. EpochDataset 契约

NPZ 必须包含：

- schema/name；
- X: float32 (N,C,T) 与可选 y: int64 (N,)；
- subject_ids；
- 唯一物理 channel_names、米制 channel_positions_m (C,3)、channel_mask (C,)；
- 完整 PreprocessingSpec；
- table-oriented metadata JSON 与 provenance JSON。

当 `baseline_mode` 为 `trial_reference` 时，PreprocessingSpec 还必须声明
`trial_reference_window_ms`、`trial_reference_center`（`mean`/`median`）和
`trial_reference_scale`（`none`/`std`/`mad`）。参考窗使用 epoch 的物理毫秒轴，右端点为
exclusive；它适用于没有刺激前段的 stimulus-locked 数据。`trial` 仍严格要求
`tmin_ms < 0`，不会自动退化为 epoch 前几个采样点。

读取固定使用 allow_pickle=False。物理时间轴、样本数、非有限值、零坐标、非零缺失通道、
schema 缺失或预处理不一致都 fail-closed。旧 NPZ 不能通过裁剪或补零“升级”。

## 6. 数据角色

| 数据 | 角色 |
|---|---|
| GTN 儿童 3 导 | 猜数字主域与最终 GTN 验收锚点 |
| BNCI2014-008 | 8 导 P300 辅助域/独立二分类 LOSO |
| Brain Invaders 2014a | 干电极 P300 辅助域/独立二分类 LOSO |
| ERP CORE P3 | 成人 P3 辅助域/年龄域对照 |
| 自有成人干电极 | 最终部署目标域 |

辅助域不得进入 GTN 主分类监督或测试折，细则见
[transfer_policy.md](transfer_policy.md)。
