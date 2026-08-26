# N2P3-Net 设备可移植性规范（Device Portability）

> 规定代码如何在两台异构设备上无缝运行，并回退 CPU。优先级：constitution > blueprint > 本文档 > CODING_WORKFLOW。
> 版本：v1（2026-08，已核实 IPEX EOL 与原生 XPU 支持现状）。
>
> 目标设备：
> - 设备 A：Intel Arc 130T（Core Ultra Series 2 / Arrow Lake-H 集成显卡，XPU，共享系统内存）
> - 设备 B：NVIDIA RTX 5070 Laptop（Blackwell，8 GB 显存，CUDA）
> - 兜底：CPU

## 0. 三思修正（先读，这三条推翻/修正了初版规范）

| # | 初版规范的写法 | 三思后的修正 | 依据 |
|---|---|---|---|
| C1 | `try-except import intel_extension_for_pytorch as ipex` | **弃用 IPEX，改用原生 `torch.xpu`** | IPEX 已于 2026-03 底 EOL；Intel 官方建议直接用原生 PyTorch。PyTorch 2.5+ 原生支持 XPU（`torch.xpu.is_available()`），Arc 130T 属 Arrow Lake-H，在官方支持列表内 |
| C2 | "强制 AMP + 配合 GradScaler" | **AMP 用 bf16，默认不需要 GradScaler** | GradScaler 是 fp16 专用的（防 loss under/overflow）。bf16 指数位与 fp32 相同，天然无此问题；5070(Blackwell) 与 Arc(Xe2) 均支持 bf16。用 bf16 既省显存又省去 GradScaler 复杂度 |
| C3 | "8GB 显存规避"当作硬需求 | **本项目 8GB 绰绰有余，此套为防御性通用规范** | N2P3-Net ≤80k 参数、输入 (B,8,256)，即使 batch=256 显存占用也仅数百 MB，8GB 不会 OOM。AMP/梯度累积/OOM 保护是为「通用大模型训练」保留的防御能力，非本项目必需 |

> 其余条目（禁止硬编码 `.cuda()`、`.to(device)` 统一、batch_size 参数化、pin_memory 动态、
> OOM 提示）判断正确，予以保留并细化。

## 1. 硬规则（DP 系列，全部强制）

- **DP1 禁止硬编码 `.cuda()`**。任何 `.cuda()` / `.xpu()` / `.cpu()` 硬编码均禁止，统一 `.to(DEVICE)`。
- **DP2 设备动态检测**。入口处按 `CUDA → XPU → CPU` 优先级检测，全局单例 `DEVICE`。
- **DP3 模型与张量统一 `.to(DEVICE)`**。model、inputs、labels、以及所有新造张量（含合成数据的
  buffer、可学习参数初始化）一律 `.to(DEVICE)`，禁止漏掉任何一处。
- **DP4 AMP 按设备动态启用**。CUDA/XPU 启用（默认 bf16），CPU 禁用。禁止写死 `device_type="cuda"`。
- **DP5 batch_size 外部传入**。经 argparse 传入，严禁写死常量；配套 `accum_steps`
  梯度累积参数，物理 batch 过小时用累积模拟大 batch。
- **DP6 显存与异常设备感知**。`empty_cache`、显存打印、OOM 捕获均须按 `DEVICE.type` 分支，禁止只写 CUDA 分支。

## 2. 设备检测（标准实现）

```python
import torch


def get_device() -> torch.device:
    """动态检测设备：CUDA → XPU → CPU。禁止硬编码 .cuda()。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # PyTorch 2.5+ 原生，无需 IPEX（已 EOL）
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = get_device()
```

要点：
- `torch.xpu` 是**原生 API**（PyTorch ≥2.5），**不要**再 `import intel_extension_for_pytorch`。
  该扩展已 EOL，装了反而不利于长期维护（C1）。
- 检测顺序 CUDA 优先：若某台机器意外同时可用 CUDA 与 XPU，稳定选 CUDA。

## 3. AMP 混合精度（bf16 优先）

```python
use_amp = DEVICE.type in ("cuda", "xpu")
AMP_DTYPE = torch.bfloat16  # 5070 与 Arc 均支持；bf16 无需 GradScaler

# 前向
with torch.amp.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=use_amp):
    output = model(inputs.to(DEVICE))

# 反向（bf16 下不需要 scaler；若改用 fp16，则加下面两行）
# scaler = torch.amp.GradScaler(device_type=DEVICE.type, enabled=use_fp16)
# scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
```

决策（C2）：
- **默认 bf16**：数值范围 = fp32，无需 loss scaling；8 GB 显存下 bf16 激活/梯度约省一半。
- **fp16 仅作可选**：仅当遇到 bf16 精度不足（罕见，本项目损失含 KL/MMD，建议实测）或算子不支持时
  才回退 fp16，此时必须启用 `GradScaler`。
- **CPU 上 `enabled=False`**，等价纯 fp32，无 AMP 开销。

## 4. 显存管理（8GB RTX 5070）

```python
def _print_device_memory() -> None:
    """启动时打印总显存，供人工核对环境是否正确。"""
    try:
        if DEVICE.type == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        elif DEVICE.type == "xpu":
            total = torch.xpu.get_device_properties(0).total_memory / 1024**3
        else:
            return
        print(f"[device] {DEVICE} 总显存 ≈ {total:.1f} GiB")
    except Exception as e:
        print(f"[device] 显存查询失败（不影响运行）：{e}")


def _empty_cache() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "xpu":
        torch.xpu.empty_cache()
```

- batch_size / accum_steps 经 config 传入（DP5）；训练脚本启动时打印二者，便于 OOM 时快速定位。
- 梯度累积范式：
  ```python
  loss = loss / accum_steps
  loss.backward()
  if (step + 1) % accum_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
  ```

## 5. 数据加载器

```python
pin_memory = DEVICE.type == "cuda"  # 仅 CUDA 开 pinned 内存加速 H2D 拷贝
loader = DataLoader(
    ds, batch_size=cfg.batch_size, pin_memory=pin_memory, num_workers=cfg.num_workers
)

for inputs, labels in loader:
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # DP3：显式 .to(DEVICE)
```

## 6. OOM 异常保护

```python
try:
    ...  # 训练一步
except torch.OutOfMemoryError:  # torch 2.x 统一 OOM 异常（CUDA/XPU 共用）
    raise RuntimeError(
        "显存溢出（OOM）：请减小 batch_size（当前 "
        f"{cfg.batch_size}）、调大 accum_steps、或关闭其他占用显存的程序后重试。"
    ) from None
except RuntimeError as e:  # 旧版兜底
    if "out of memory" in str(e).lower():
        raise RuntimeError("显存溢出（OOM）：请减小 batch_size 后重试。") from None
    raise
```

## 7. 环境安装（torch 须按设备选源，不能一条命令装全）

| 设备 | torch 安装命令 | 说明 |
|---|---|---|
| NVIDIA 5070 | `pip install torch --index-url https://download.pytorch.org/whl/cu132` | 5070 是 Blackwell（sm_120），需 CUDA ≥12.8；cu128 索引只到 torch 2.11，而 Python 3.14 需 torch 2.13，故用 cu132（2026-08-21 实测该索引有 2.13.0+cu132 cp314 win wheel） |
| Intel Arc 130T | `pip install torch --index-url https://download.pytorch.org/whl/xpu` | **XPU 专用源**；PyPI 官方 wheel 是 CPU 版、不带 XPU |
| CPU 兜底 | `pip install torch` | PyPI 官方 CPU 版 |

- 其余依赖（mne/numpy/...）用 `pip install -r requirements.txt` 统一装，torch 因源不同单独先装。
- Arc 130T 需先装 Intel GPU 驱动；若 `torch.xpu.is_available()` 为 False，先查驱动，不要怀疑代码。
- 装完自检：`python -c "import torch; print(torch.__version__, torch.cuda.is_available(), hasattr(torch, 'xpu') and torch.xpu.is_available())"`。

## 8. 补充注意（三思新增）

1. **Windows DataLoader**：`num_workers > 0` 时多进程用 spawn，训练脚本必须 `if __name__ == "__main__":` 包裹入口，否则递归启动崩溃。
2. **跨设备浮点差异**：同模型在 CUDA/XPU/CPU 上结果存在轻微差异（GPU 浮点非确定性），属正常；不做
   deterministic 强制（会显著拖慢）。评估结论（命中率）不受影响，但别拿不同设备的结果做逐位对比。
3. **自定义损失 dtype**：`L_tau`（KL）、`L_MMD` 等在 bf16 autocast 下可能精度不足，关键损失项
   可在 autocast 外以 fp32 显式计算，或在 Phase 2 实测后决定是否对损失段单独 `enable=False`。
4. **iGPU 共享内存**：Arc 130T 无独立显存，`total_memory` 返回系统分配额度；其 XPU 训练定位是
   「可跑通验证」，主力训练用 5070，性能预期勿等同。

## 9. 规范落点

- 本规范的**主要消费方**是 `train/trainer.py`（模块 #14）与 `models/`（所有 `.to(DEVICE)` 处）。
- CODING_WORKFLOW §3 模块 #14 已标注「须遵守本文档」；techstack 已加引用。
- 后续新增任何 `.cuda()` 字样即违反 DP1，属 code review 红线。
