# 首批 Lean 规格的源码与反例独立审查

本次审查对应 `V01.a / V02.a / V02.b`，使用真实活动源码与纯内存合成输入。源码身份、环境、逐项结果和误差记录见 [source_review.json](../assets/lean_first_slice_20261016/source_review.json)，可重跑入口为 [check_first_slice_bridge.py](../../formal/lean/scripts/check_first_slice_bridge.py)。没有读取原始 EEG、启动训练、修改算法或消费正式评测。

## 结论与覆盖

9 组行为探针全部通过，其中有限滤波包含 3 类 SOS × 5 类样本数/填充几何共 15 个实例。格式化后的最终独立重跑主体执行 1.52 秒；该计时从依赖导入后开始，原始回执另存为 [source_review_final_probe.json](../assets/lean_first_slice_20261016/source_review_final_probe.json)。7 份相关现有测试文件共 84 项通过，pytest 报告 47.95 秒。新脚本 Ruff 检查与格式检查通过。逐项数值用于复现实例，不构成统一浮点误差界。

| 主题 | 实际调用与独立验收 | 结果 |
|---|---|---|
| 获准 raw 与标签 | `build_subject → execute_preprocessing → crop_raw_scope`；分别把校准右侧、测试左侧未授权波形改成 NaN，并把未授权标签改成 999 | 数据和已选标签逐位相等；原事件身份保留 |
| K0、共同测试与嵌套范围 | 仅有 `first_samp/n_times/sfreq` 的几何对象调用 `prefix_raw_scopes`；内训练范围之外加固定扰动 | K0 无拟合范围；固定 Kmax 下测试范围相同；内训练结果不变，外层结果变化 |
| 精确网格与闭端点 | 非零 MNE `first_samp`；128、127.999、512 Hz 精确裁剪；250 Hz 闭端点加一 | 裁剪等于原数组精确切片；少一个末样本导致明确 `boundary_dropped` |
| 连续路线右删失 | `_calibration_view → _filter_scope → filter_bidirectional_segment`；把首个未授权 onset 及其后数据改成 NaN，翻转后续标签 | 获准 K 个标签保留；样本严格止于下一 onset 前；不完整尾部必须显式授权右删失 |
| onset 边界 | `_validate_record` 和 `_event_design` | onset 等于独占 stop 被拒；只使用实际可见 basis 行，不把缺失响应补成观测零 |
| 校准内划分 | `calibration_inner_split`；mask 以外标签改 999、时间改 NaN/Inf、录制身份替换 | 合法划分及三个掩码完全相同 |
| 有限 SOS | 独立标量 DFII-T 递推与常值平衡初态，对比生产实现和 `scipy.sosfiltfilt` | 包含一采样、零 pad、截短 pad、多节非平凡系数；最大独立重放残差约 3.52e-13 |
| 双端分解与确定性状态 | `FiniteFilterOperator.decompose/apply/zero_state/state_basis`；初态也单独用非零任意状态测试 | 最大重建残差约 1.83e-11；`Q=HᵀH` 有限实例通过；4S 列不代表秩 4S |
| 联合协方差 | 400 组真实确定性 padding 状态；`JointNoiseModel.fit/project` 对比 `F Cov(x) Fᵀ`；堆叠重叠行 | 最大差约 1.27e-14；删除 raw/state 交叉项产生约 1.96 的范数误差 |
| 近似 PSD 准入 | 真实 `validate_covariance → JointNoiseModel → project` | 可准入的微小负特征值经算子放大后仍可形成显著负二次型，见下节 |
| 精确传播数值桥接 | `ExactPropagatedCovariance.compute` 对比显式 dense `F K Fᵀ` 及 Cholesky 重建目标 | 最大差约 1.55e-15；因子对应加 ridge 后矩阵；数值秩 37/40 |

## 必须保留的科学与实现前提

`RawSampleScope` 的样本区间为绝对 MNE 半开区间，事件和标签则由显式原始行号授权。`prefix_raw_scopes` 本身只规划范围，不读 EEG 和标签。`execute_preprocessing` 先 `crop_raw_scope` 再滤波、epoch；`build_subject` 使用恢复后的原事件索引取标签。`source_records.prepare_source_prefixes` 逐 K 重建 raw 视图，并在父校准范围内另建 inner train/validation 视图；后续 load 准入还检查相同原始输入、范围与执行回执。

这些是数组和调用链性质。完整原文件解析、文件哈希、通道准入、注释、几何、时钟、配置以及外部已有拟合状态不由本次抽象数组非干扰定理隔离；它们必须固定或另行授权。不能把相同裁剪数组推广成所有元数据和产物身份也相同。

**通用 epoch 前缀与连续候选前缀的样本授权不同。** 通用路线包含获准事件的完整响应尾部，尾部可能覆盖下一个未授权事件的波形；后者的标签依然不被授权。连续路线在 `_calibration_view` 中额外取 `min(authorized_stop, next_onset)`，在下一个 onset 前终止采样，并保留 K 个标签，按实际观测样本右删失。二者各自有明确实例，不能用前者宣称已经排除未来刺激波形。

`FiniteFilterOperator` 校验有限、归一化且稳定的 SOS；底层 `filter_bidirectional_segment` 依赖调用者提供有效 SOS。实数有限递推的线性与分解不需要稳定性；稳定性控制长期响应与库初态存在性，属于额外准入条件。Lean 从精确 DC gain 推导单节与累积级联的单位阶跃平衡初态，条件为每节 `1+a1+a2≠0`；固定初态表的线性缩放也已证明。SciPy 求解器的计算结果与该精确表的形式化对应、稳定根检查到非零 DC 分母的对应以及底层浮点统一舍入界仍未证明。

## 会被错误规格遗漏的反例

1. **数值 PSD 准入不等于精确 PSD。** `noise_model.validate_covariance` 使用 `1e-10 × max(||Σ||₂,tiny)` 容差。`diag(1,-5e-11)` 通过真实准入。以 `M=[0,1e6]` 投影得到 `MΣMᵀ=-50`。输入负误差的放大量为 `ε||M||₂²=50`；ridge=1 与 ridge=50 都不能使该实例正定，ridge=51 可以。另以 `M=diag(1,3)`、人为指定的对称残差 `E=-0.25I` 得到达到 `-(9ε+0.25)` 的实例，严格超过该合成误差界的 ridge 成功。这里的 E 是已知的合成输入，不是对真实矩阵乘法舍入残差的估计；尚未证明 NumPy 的特征值、范数或乘法能给出已认证的 ε/η。这是现有数值准入行为的边界，不构成修改准入策略的要求。
2. **反射滤波 F 通常不对称，且裁剪与滤波不交换。** 对确定的 101 样本网格，`||F-Fᵀ||≈17.78`；固定 301 样本输入先滤后裁与先裁后滤的差范数约 17.86。双向零相位不能替代有限算子的边界证明。
3. **边界状态由观测波形确定。** 对同一 raw 残差做 padding 得到的状态存在强交叉协方差；独立加两份边缘协方差会改变结果。上述实际状态探针保留所有交叉项与重叠输出行。
4. **4S 仅为状态坐标数。** 五节 SOS、一采样实例的 basis 有 20 列，秩最多为 1；不能把 4S 写成满秩保证。
5. **加任意正 ridge 不足以修复经放大的近似 PSD 输入。** Lean 应要求 ridge 严格超过已传播的负下界及残差项，而非仅假设 ridge>0。

## Lean 命题独立审查

审查对象为 [RawScope.lean](../../formal/lean/N2P3/RawScope.lean)、[FiniteSOS.lean](../../formal/lean/N2P3/FiniteSOS.lean)、[FiniteAdjoint.lean](../../formal/lean/N2P3/FiniteAdjoint.lean)、[Covariance.lean](../../formal/lean/N2P3/Covariance.lean) 与 [FirstSlice.lean](../../formal/lean/N2P3/FirstSlice.lean)，并复核其复用的 Pilot 命题与总验收脚本。最终审查快照共计 136 个 theorem/lemma 声明，其中 Pilot 14 个。摘要写入结构化回执；编译和公理检查由本轮总验收工具记录。

- `RawScope` 先证明获准坐标上的相等推出真实提取输入相等，再推出固定元数据下确定性处理相等；没有以待证的处理输出相等作前提。父子范围、事件身份、共同 heldout、闭端点转换和连续右删失分别建模。首次审查指出的 MNE 非零 origin 与严格 onset<stop 的合法性已在后续源码补入：范围含 `origin≤first`、`stop≤origin+nSamples`，并证明局部数组与绝对 MNE 坐标的关系；连续保留 onset 严格位于删失 stop 前。请求 epoch 端点使用有符号整数，保留负起点的待 purge 请求；已保留证据的行号继续组合进获准原始行号，而非重新生成身份。
- `FiniteSOS` 从归一化 DFII-T 单步公式构造 section/cascade/time 的线性映射，逐段构造实际 left/right padding 状态；线性并非先验假设。实数 `F(x)=Qx+B a(x)` 使用两端实际状态，单位阶跃平衡初态从精确系数推导。显式 `flatCoordinate` 对应左端 `2×section+coord`、右端 `2S+2×section+coord`，与 Python 转置/reshape 及 basis 激励一致；15 个实例用独立逐列激励复核。
- `FiniteAdjoint` 由真实 `run/cascade` 递推推导自主状态轨迹、冲激系数与有限下三角 Toeplitz 表达，再证明 `RHR=Hᵀ`、`Q=HᵀH` 与 Q 的半正定性；没有先假设 Toeplitz 或伴随恒等式。早期独立审查发现的 Q 数学证明缺口已关闭。结论针对零初态 Q，不能扩展为整个反射算子 F 对称或正定。
- `Covariance` 从有二阶矩随机变量的协方差定义证明线性推送、联合块、PSD 与确定性状态传播，未添加 raw/state 或 epoch 独立性假设。近似 PSD 使用精确实数下界，明确 L2 算子范数而非元素范数；再传播下界与对称残差，并要求严格超过误差界的 ridge。其结论依赖已认证的 ε/η，尚未将运行时容差提升为该认证。
- `FirstSlice` 把真实 scope 样本索引写成逐行精确选择矩阵，与 `paddedFilter` 的矩阵组合，证明先抽取后滤波的非干扰及真实随机变量协方差传播。scope、事件几何、滤波参数、初态表和已有拟合状态在该概率命题中必须固定；数据依赖的随机拟合算子需另行建立条件分布或联合分析，不能直接使用固定矩阵推送公式。
- `verify_stage.py` 的清单解析包含同一行 `@[simp] theorem`、命名空间和 section；每个清单命题都要求有匹配且仅一次的 `#print axioms` 结果，并按标准公理白名单检查。独立审查发现的“多个错误例子只需其中一个拒绝”缺口已修复：负例文件要求每个预期 example 都有对应的 `unsolved goals` 错误且总 `⊢ False` 数相同，额外错误、导入失败或仅一个例子失败不计通过。源码和证明文件在总验收前后分别比较摘要。

本切片的验收含义是“所列实数规格的内核证明 + 所列真实 Python 路径的合成行为核验 + 有来源的未闭合义务”。它没有完成全链信息流 `V01.b`、Python/MNE/SciPy 形式化 refinement、统一 IEEE 浮点界、完整滤波/QC/EOG 组合或公共数据/真人效能验收。

## 复现

使用仓库 `.venv`：

```powershell
.venv/Scripts/python.exe formal/lean/scripts/check_first_slice_bridge.py
.venv/Scripts/python.exe -m ruff check formal/lean/scripts/check_first_slice_bridge.py
.venv/Scripts/python.exe -m pytest tests/data/test_raw_preprocessing_scope.py tests/data/test_finite_filter_operator.py tests/data/test_bidirectional_recovery.py tests/transfer/test_raw_continuous.py tests/transfer/test_noise_model.py tests/transfer/test_exact_covariance_and_bounds.py tests/transfer/test_calibration_inner_split.py
```

JSON 中的 `probe_script_sha256` 与逐文件摘要绑定本次实际执行的脚本/源码；源码在探针前后保持一致。测试计数来自上述实际测试运行，不以脚本中的断言个数代替覆盖。
