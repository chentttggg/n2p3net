# 阻抗低值现象核查

> 历史证据：正文保留写作时的实验口径、状态和路径，不作为当前执行指令。
> 当前任务以[执行计划](../roadmap.zh.md)为准；原文压缩快照和迁移说明见[证据索引](README.md)。

用户报告改变接触时数值始终约2–4 kΩ。当前无法确认报告发生时的运行模式；
本轮进程检查未发现正在运行的BrainSync窗口，已向用户询问模拟/设备模式。
此前助手实际启动过`--mock`窗口，因此先验证这一可复现路径。

## 已验证

`MockDeviceAdapter._impedance_loop`使用固定公式：

`2.6 + 0.7*sin(step*0.4 + channel*0.6) + (channel % 4)*0.45`。

其全时域范围约1.9–4.65 kΩ，不读取任何电极。实际回调循环测试复现了这一区间。
此前UI会将这些模拟值显示为“良好/合格”，并将`impedance_passed`设为True，
这是已复现的显示与质量语义缺陷，不应据此证明真实设备阻抗正常。

已修正模拟窗口标题、阻抗弹窗来源提示、逐通道模拟标记及停止后的状态，
模拟值不再产生接触合格结论，session质量记录增加`measurement_source=simulated/device_sdk`。
没有改变模拟数值去伪装电极响应，也没有擅自修改设备读数比例。

安装的SDK参考`D:/Temp/brainsync-sdk-example-main/python/README.zh.md`明确
`ImpedanceData.impedances`单位为kΩ；BLE/serial示例也直接按kΩ显示。
当前DeviceAdapter将其原值转float后送到UI，未发现额外缩放或2–4区间截断。
用2.5、30、100、180、201、500、1200、3.5 kΩ回调验证，显示链原值保留。

## 验证与边界

新增`tests/test_impedance_display.py`：模拟质量结论反例先失败，修复后三项通过；
覆盖真实Mock循环、模拟不合格认证、设备kΩ原值显示及阈值变化。
连同既有录制/设备生命周期相关回归，本轮60项通过（21.874秒），未连接真实硬件。
这不是实际电极/固件测量验证。如果用户确认当时为设备模式，应继续检查原始SDK回调
及接触/断开时变化，不能以模拟路径吻合就排除设备或SDK问题。

修改前UI与现有物理快照`frozen/brainsync_integrated_app_20260906.zip`的文件hash一致，
没有覆盖原快照。真实入口为主工作站的`Launch_BrainSync.cmd`。
