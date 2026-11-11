# N2P3-Net

N2P3-Net是BrainSync九选一P300解码的研究后端：用公开ERP数据训练源模型，再用参与者自己的早期录制校准，在独立后续录制上评价。采集应用是另一个项目；本仓库不包含设备SDK。

从[文档入口](doc/README.md)开始。研究目标、数学与比较规则由[研究规范](doc/research_program.zh.md)维护；输入处理和实际调用链分别见[输入合同](doc/input_contract_math.zh.md)与[技术结构](doc/technical_structure.zh.md)。

## 使用与验证

Python至少3.11，依赖以`pyproject.toml`和`uv.lock`为准。在仓库根目录执行：

```powershell
uv sync --locked --extra signals --extra baselines --extra dev
.\.venv\Scripts\python.exe experiments\brainsync_model_task.py --help
.\.venv\Scripts\python.exe experiments\check_docs.py
.\.venv\Scripts\python.exe -m ruff check src experiments tests
git diff --check
```

按改动范围运行相关测试；环境分支、完整验证、云端资源和交付要求见[工程指南](doc/engineering.zh.md)。Linux使用`.venv/bin/python`。测试通过证明相应软件行为，真实设备的九选一效能需要独立记录验收。

## 从数据到预测

链条依次是原始数据准入、输入cache、监督源训练、个人校准、独立后续评分。输入、参与者身份、拟合范围与实际执行源码都随产物绑定。

| 任务 | `experiments/`入口 |
|---|---|
| 公共EEG和合法candidate输入 | `prepare_eeg_dataset.py`、`prepare_bnci2014_008_candidate.py`、`prepare_bi2014a_candidate.py` |
| 源通道、参考和binary视图 | `adapt_eeg_domain.py`、`prepare_multidomain_source.py` |
| 监督源预训练 | `run_pretrain_supervised.py` |
| 公共目标校准与跨库比较 | `run_binary_subject_transfer.py`、`run_cross_dataset_transfer.py` |
| 完成的BrainSync录制转cache | `prepare_brainsync_cache.py` |
| 个人校准、包检查和无标签预测 | `brainsync_model_task.py` |
| 独立后续decision评价 | `run_brainsync_cross_decision.py` |

参数以各入口的`--help`为准。默认研究路径使用监督源训练和保留源分类头的适配；EA、AS-MMD、Split-BN、源选择等机制各有显式合同和对照。

BrainSync采集为七路EEG（`Fz,Cz,P3,Pz,P4,PO8,Oz`）和一路独立EOG，原生250Hz。产品预算是9个校准目标×10个block×45次闪烁，随后用新的45次闪烁作一次R5判定；R1和trial AUC分别报告。不同历史配方的滤波、参考、QC和EOG处理以各自输入合同为准。

工作站已用声明的模拟数据验收保存完成后自动预测的调用链。采集、个人模型和新block操作见[工作站指南](doc/workstation.zh.md)；逐SDK chunk的流式解码另属后续范围。

## 数据、结果与历史

[训练数据与精简结果](doc/training_data.zh.md)汇总保留资产、结果口径与恢复入口。结论与读数由[四个证据主题](doc/evidence/README.md)维护，运行状态由[原检查点](doc/README.md#当前状态与续接)维护。

原始数据、cache、模型、冻结结果和文献有不同用途，见[目录层级](doc/technical_structure.zh.md#目录与数据层级)。Git主要分发源码、测试、指南和紧凑清单；本地历史载荷按[归档规则](frozen/README.md)保管。旧文档的原路径、成员SHA和检索方法集中在[历史索引](doc/evidence/history.zh.md)。

需要进度面板时运行：

```powershell
.\.venv\Scripts\python.exe experiments\dashboard_server.py --port 8812 --bind 127.0.0.1 --directory experiments
```

打开[本地面板](http://127.0.0.1:8812/dashboard.html)。只有输出其`progress.jsonl`/`record.json`协议的运行会出现在面板；SSH接入配置见工程指南。
