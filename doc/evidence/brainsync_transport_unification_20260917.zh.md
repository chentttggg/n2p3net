# BrainSync串口/BLE统一与固定250 Hz验收

日期：2026-09-17。活动前端为`D:/BCI/BrainSync_Tool_newest/BrainSync_Tool`，
后端为`D:/BCI/n2p3-net`。本轮实现用户要求的串口/BLE共用模块，随后按明确指示把两者都固定为250 Hz。
前端不在Git中，修改前已保存70文件并逐成员核验，见
[修改前清单](../../frozen/brainsync_transport_unification_20260917_before/manifest.json)。
后端已有并发修改保持原状，本轮没有改模型、启动训练或接触真实采集数据。

## 最终实现

| 所属module | 统一职责 |
|---|---|
| `transport.py` | 串口/BLE opener、地址提示、BLE连接回调/等待与读回超时能力；采样率验证及独立连接偏好 |
| `DeviceAdapter` | 同一个连接/配置/预览/阻抗/录制/恢复/断开流程，保留原生操作实际所有权 |
| `LocalEdfRecorder` | 同一个电压缓冲、样本时钟、暂停段、联合检查点及EDF原子发布 |
| `RecordingSession` | 原有准备、标记、暂停恢复、收尾、BIDS导出及失败状态，串口不再走另一种保存方式 |
| `MainWindow` | 保存串口端口和BLE名称、尊重当前模式自动连接、固定250 Hz显示、操作互斥及退出等待 |

删除串口的SDK原生EDF启动/注释/停止分支及`pausable`选择参数，两个真实传输都从回调保存数据。
录制就绪和写入停滞只依据实际回调样本，不能用SDK计数替代已写入样本。
两传输共用暂停、恢复重订阅、异常保存及EDF/BIDS样本定位；EDF设备字段不再把串口误标为BLE。
读取历史SDK EDF的修复逻辑仍用于原始历史文件，不是新的录制分支。

采样率引用现有数据合同`DEFAULT_SAMPLE_RATE_HZ=250`，真实及模拟configure均拒绝其他值。
UI只显示“250 Hz（固定）”，不保存可变采样率；串口与BLE分别保存地址及增益。
SDK callback batch是独立传送单位，固定起流/恢复请求为250包，合法批量为其正整数倍。
通用EDF/时钟/滤波读取模块仍可处理既有不同采样率资料，不重写历史数据。

启动沿用`Launch_BrainSync.cmd`，默认恢复上次连接选择；
`--transport serial --address COM7`直接选择串口，`--transport ble --address BrainSync`选择BLE。
省略地址使用该模式的保存值；没有保存值时串口自动发现、BLE使用默认名称。
无需复制启动器、窗口或录制控制器。

## 复现与修复

| 缺陷或重复路径 | 最终行为与反例 |
|---|---|
| 启动强制BLE，即使界面选择串口 | 启动与手动连接读取同一选择；实际FakeSdk opener及界面状态均核对 |
| COM口和BLE名称共用一个未隔离文本值 | 两种profile各自持久化，来回切换及重建窗口后保留 |
| 串口native EDF和BLE callback录制维护两套所有权 | native EDF fake接口成为失败哨兵；两传输实际读取EDF波形与注释并比较 |
| 共用writer含BLE专属设备标记 | 中性设备字段，实际EDF header断言通过 |
| 串口不进入回调恢复，恢复批量与采样率耦合 | 同一恢复流程及250包订阅；失败不重复注册，不清除历史gap |
| 阻抗SDK会把采样率/增益改为默认值 | enable前标记配置失效，预览及录制前重新应用250 Hz/所选增益；读回失败不能起流 |
| 取消native opener的Future会丢失迟到句柄 | shield原生receipt；取消后仍有owner，迟到句柄释放后才允许新连接 |
| close失败就清掉handle | 保留handle及recovery状态，后续重试关闭成功才断开 |
| 旧BLE断开回调清掉新的串口/BLE连接 | callback按连接generation核验；BLE→serial和BLE→BLE均有反例 |
| 等待firmware期间掉线，旧应答仍发布READY | 应答后再次核验同一有效handle；拒绝READY/None组合 |
| shutdown在pending opener未结束时关闭qasync | 关闭流程等待连接owner清理，普通取消仍有界；Qt循环保持响应直到资源结束 |

串口/BLE并非取消全部差异：SDK连接状态回调只用于BLE；串口采样率必须读回。
BLE只有读回超时且后续固件请求证明链路仍活着时，才记录
`write_acknowledged_readback_unavailable`。`readback_confirmed`确认采样率，未宣称增益读回。
增益使用已接受写命令的设置值；不猜测SDK未声明的gain字段。

## 验收与可复查证据

软件测试使用前端自己的Python3.10虚拟环境；后端相关合同使用后端虚拟环境。
`test_transport_sample_contract.py`覆盖两传输250 Hz的序号65536回绕、10微秒单位、实际样本计数、
暂停断点、恢复订阅、EDF及RecordingSession→BIDS；非250请求在SDK参数写入前被拒绝。
测试用固定合成采样时钟，显式声明基线支持，避免快速造数依赖测试机负载。

`test_transport_connection.py`、`test_transport_ui.py`覆盖连接清理和独立偏好；
原启动取消、local finalize失败重试、worker未结束不能重试等合同保持。
`tests/verify_transport_ui.py`在真实qasync/Qt下分别运行串口和BLE的启动、250 Hz预览、停止与断开，
1160×720无水平溢出；截图已人工检查。全部测试根目录位于TEMP，没有读取操作员设置。

最终完整测试数量、退出码、命令、日志、截图、差异与源码哈希记录在
[最终验收清单](../../frozen/brainsync_transport_unification_20260917_validation/manifest.json)。
软件故障注入、模拟EDF和SDK接口核对不等同于真实串口/BLE吞吐、掉线恢复或成人识别效能验证。
本轮未进行设备扫描、连接或真实录制。若native opener永久不返回，关闭继续等待，不假报已释放。
