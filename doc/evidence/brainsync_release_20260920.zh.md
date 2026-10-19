# BrainSync发布版整理与EXE验收

日期：2026-09-20。源采集端为`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`，
发布目录为`D:/BCI/BrainSync_Release_20260920`。本次创建独立发布副本，保留源工作树、
原始数据、开发环境、模型运行历史和测试文件；开发入口只从发布副本移除。

## 源目录盘点

源采集端约1.23 GB、8,323个文件。主要内容如下：

| 内容 | 处理 |
|---|---|
| `gtn_workstation/`、配置、资源 | 进入EXE；资源按PyInstaller数据文件嵌入 |
| `data/`、`logs/`、`models/runs/` | 不复制历史内容；发布目录创建空目录 |
| `.venv/`、`_internal/`、开发缓存 | 不复制；发布EXE内部只保留运行所需二进制 |
| `tests/`、`*.py`入口、`Launch_*.cmd` | 不进入发布目录 |
| `models/source/` | 只复制默认源模型和说明 |
| `config.toml`、`config.default.toml`、`channel_assignments.json` | 复制为用户配置初始值 |
| `libusb-1.0.dll` | 作为设备支持依赖嵌入EXE构建，不单独暴露开发入口 |

源目录内最大的无关发布内容是PySide6的Qt WebEngine核心库约203 MB；应用未导入WebEngine，
构建显式排除该模块及其他未使用的Qt3D/Quick/Multimedia等模块。

## 发布目录

发布目录只包含：

```text
BrainSync_Release_20260920/
├─ BrainSync.exe
├─ _internal/             # PySide6/shiboken6/NumPy/SciPy等运行时
├─ RELEASE.md
├─ release-manifest.json
├─ channel_assignments.json
├─ config.toml
├─ config.default.toml
├─ config/                 # 用户运行时配置
├─ data/                   # 用户采集数据
├─ logs/                   # 运行日志
└─ models/
   ├─ README.md
   ├─ model_manifest.example.json
   └─ source/              # 默认源模型
```

`_internal/`是发布运行时，不含开发Python环境、SDK工程目录、源码、测试、模拟入口或历史数据。
PyInstaller onedir将Qt/PySide6/shiboken6及科学计算二进制放在该目录，避免onefile临时解包和
QtCore DLL搜索问题；“不包括runtime”按发布载荷排除`_internal/`核对。

## 验收

| 检查 | 结果 |
|---|---|
| PyInstaller onedir构建 | 通过，Python 3.10.20，PyInstaller 6.22.3；主入口与`_internal/`同目录，另保留单文件备用构建 |
| `BrainSync.exe --mock`离线启动 | 正常 Windows 环境和清空开发PATH后均保持运行8秒，无早退崩溃 |
| Qt WebEngine | 未进入EXE；构建内容无`Qt6WebEngineCore.dll` |
| Qt/PySide6依赖 | 显式收集shiboken6运行库、QtCore依赖的`icuuc.dll`，注册`_internal/`根目录，并清除外部Python/Qt环境变量；污染环境后仍能启动8秒 |
| 发布载荷体积（不含`_internal/`运行时） | 134,552,186 bytes / 128.319 MiB，小于200 MiB |
| `_internal/`运行时 | 294.375 MiB；按用户要求不计入200 MiB载荷限制 |
| 发布哈希与文件清单 | [release-manifest.json](../../../../BCI/BrainSync_Release_20260920/release-manifest.json) |
| 源前端完整套件 | 291项通过，退出码0 |
| 设备模式边界 | 未扫描、连接或执行真实串口/BLE录制 |

发布EXE仍需要真实设备、有效SDK硬件链路和用户选择的后端算法环境才能执行对应的真人采集或模型任务。
软件启动与模拟模式通过不代表实机吞吐、固件恢复或成人识别效能已经验收。
