# N2P3-Net 编码工作流（Coding Workflow）

> 本文档规定「如何写代码」，与 roadmap（做什么）正交。优先级：constitution > routes > blueprint > 本文档。
> 版本：v2。

## 0. 方法论判断（为什么不用增量式 / 整体式）

- **增量式（逐步加功能）**：跨会话丢失上下文与 CoT，接口约定漂移，反复返工。
- **整体式（一次写完再改）**：错误集中爆发，无法局部验证，调试成本随规模指数上升。
- **本项目采用：模块原子化 + 契约先行 + 测试驱动**。每个模块「一次做到位」（接口 + 类型 +
  docstring + 两级测试），模块间靠「契约」连接；后续会话读模块 docstring + 测试即可恢复上下文，
  无需回忆此前的 CoT。后期只做「小接口微调」即可串通全项目——这正是本工作流的目标。

## 1. 三条铁律

F1 契约先行。写实现前先钉死接口：输入/输出张量形状、dtype、边界条件、异常行为。
   契约 = 模块 docstring + blueprint v12 四对象契约的细化。

F2 模块原子化。每个模块自包含（一个 .py + 一个 test），一次写到 techstack 要求；
   模块内不追求过度防御（P1 克制），只写必要的健壮性。

F3 测试驱动。先写测试（或同时），跑通才算完成。测试分两级：
   - 冒烟测试（合成数据）：验证形状、dtype、不报错；
   - 语义测试（已知答案合成样例）：关键模块验证「语义正确」，如参数化窗 claim-gate 诊断、
      决策层先中心化再累加时的 argmax 正确。

## 2. 编码流程

```
Phase A  契约冻结     —— 细化每个模块的接口契约，写入 docstring 骨架
Phase B  逐模块实现   —— 按依赖顺序，每个模块一次做到位，跑通两级测试
Phase C  集成联调     —— 合成数据跑端到端（前向 → 损失 → 反向），只做接口微调
Phase D  真实数据验证 —— 换 GTN / 自有数据，跑 Phase 1 基线 + Phase 2 模型
```

- **Phase A 产出**：每个模块的 docstring 骨架（含接口契约 + 决策理由 + 引用 blueprint v12 对象编号 L/R/Q/S）。
  这一步冻结契约，后续会话据此实现，无需重读全部讨论。
- **Phase B 顺序**：严格按依赖顺序（见 §3），上游模块的测试通过后再写下游。
- **Phase C 标准**：一个最小合成样本（B=4, C 通道, T 时间点，含假 target 标签）跑通完整
  forward + backward，无 NaN、无形状错误。
- **Phase D**：进入 roadmap Phase 1/2，本工作流的「模块开发」阶段结束。

### 2.1 先冻结路线

每次涉及模型、Loss 或训练入口的改动，先在任务记录中写明正式路线或
strict-past 研究路线，并引用 [`routes.md`](routes.md)。strict-past 的四个对象
（L/R/Q/S）与 fail-closed 行为必须作为独立路径实现和测试，不能通过默认配置、
隐式 flag 或共享状态混入正式 PCW 路线。

## 3. 模块清单（依赖顺序）

| # | 模块 | 职责 | 关键契约（输入 → 输出） | 测试 |
|---|---|---|---|---|
| 1 | data/preprocess.py | 重采样/连续域高通/epoch/伪迹剔除 | Raw+events → (N,C,T) | 冒烟+语义 |
| 2 | data/channel.py | 坐标式通道身份 + 缺失掩码 | 坐标 → E_chn (C,6·n_freqs) | 冒烟 |
| 3 | data/metadata.py | 年龄/性别嵌入 | 元数据 → E_sub (2·n_freqs+3) | 冒烟 |
| 4 | data/dataset.py | MNE 格式/事件统一加载器 | EEGRecord → (N,C,T)+元数据 | 冒烟+语义 |
| 5 | models/reference.py | 加权再参考（9参数） | X → X_ref | 冒烟+语义(参考无关) |
| 6 | models/tokenizer.py | Stage1 多尺度卷积+空间卷积 | (B,C,T) → Z (B,T,D) | 冒烟+语义(无池化保T) |
| 7 | models/encoder.py | Stage2 序列编码（depth 消融） | Z → Z' | 冒烟 |
| 8 | models/component_window.py | 参数化成分窗 ×3（τ/σ 不对称高斯） | Z' → H(3,D), τ(3), σ(3) | 冒烟+**claim-gate 诊断** |
| 9 | models/heads.py | Stage3 多头（A/B/C/D） | H,τ,σ → p_target,p_early,Â | 冒烟 |
| 10 | models/decision.py | 决策层先中心化再聚合与 LLR 校准 | p_target(试次) → score(d) → d̂* | 冒烟+**语义(argmax正确)** |
| 11 | models/n2p3net.py | 组装完整模型 | (B,C,T) → 全部输出 | 冒烟 |
| 12 | train/losses.py | 总损失（4 项，D8/D9） | 模型输出 → L | 冒烟 |
| 13 | train/augment.py | 数据增强（时间扭曲/加噪/通道dropout/参考抖动） | X → X' | 冒烟 |
| 14 | train/trainer.py | 训练循环（early stop / 多任务 λ 网格；**须遵守 device-portability.md**） | 数据 → 训练模型 | 冒烟 |

> 带 **语义** 的模块（4/5/6/8/10）是「关键路径」，其测试必须用已知答案的合成样例验证正确性，
> 因为它们决定了下游正确性；其余模块冒烟即可。

## 4. 每个模块的完成定义（Definition of Done）

1. docstring 写清：接口契约（形状/dtype/边界）、「为什么这么设计」（1–2 句，引用 blueprint D 编号）；
2. 类型注解完整（mypy/ruff 可过）；
3. 冒烟测试跑通（合成数据，无形状/dtype 错误）；
4. 语义测试跑通（若在关键路径）；
5. 无 TODO/FIXME 遗留（除非显式标注「Phase 3 实现」）。

## 5. 契约变更机制

- 接口契约变更必须：① 在 docstring 更新；② 同步更新测试；③ 在本文档 §3 模块表注明变更理由。
- 变更集中在「集成联调（Phase C）」阶段做，开发（Phase B）阶段尽量不改上游契约。
- 原则：宁可 Phase A 多花时间想清楚接口，也不要 Phase C 大面积返工。
- 新方法经完整协议证明正确且已经替代旧方法后，主动删除旧实现、旧入口、旧测试和旧文档；
  只把确有历史复现价值的内容移入 `archives/`，不要为了保留而保留。

## 6. 上下文抗丢失机制（回应多 agent 协作痛点）

- **全局上下文** → blueprint.md（架构决策 + 张量约定 + 三思记录），一次读全。
- **局部上下文** → 每个模块 docstring 顶部的「决策理由」+ 底部「依赖的决策」（引用 D 编号）。
- **行为上下文** → 每个模块的测试（读测试即知模块该做什么、边界在哪）。
- 后续会话流程：读 blueprint（全局）→ 读目标模块 docstring + 测试（局部）→ 写代码，无需回忆此前 CoT。

## 7. 云端操作规程

涉及 SSH 上传、远端依赖或 GPU 训练时，必须先阅读
[`cloud_operations.zh.md`](cloud_operations.zh.md)。该文档记录 PowerShell/SSH 转义、
SCP 目标路径、Torch/torchaudio 版本配套、远端 PID 核验，以及 acquisition/fold-jobs 等
训练入口契约；这些步骤属于工程工作流的一部分，不依赖临时会话记忆。
