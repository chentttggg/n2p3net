"""模块：设备检测与显存管理（Device）。

职责（device-portability.md DP1–DP6）：
    提供 CUDA→XPU→CPU 动态设备检测（DP2）、显存打印（DP6）、缓存清理（DP6）。
    被 train/trainer.py 与 baselines/deep.py 共用，作为设备规则的单一事实来源，
    避免「设备检测逻辑」在多处重复漂移。

明确「不做」：
    - 不 import 已 EOL 的 IPEX（C1）；PyTorch ≥2.5 原生 torch.xpu。
    - 不含任何模型/训练逻辑（纯环境函数）。

依赖的决策：device-portability.md（DP1–DP6）、constitution（无）。
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """动态检测设备：CUDA → XPU → CPU（DP2，禁止硬编码 .cuda()）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # PyTorch ≥2.5 原生 XPU
        return torch.device("xpu")
    return torch.device("cpu")


def optimize_device_for_training(device: torch.device) -> None:
    """按设备开启 PyTorch 训练期加速选项（幂等，D-device-tune）。

    - CUDA：cudnn.benchmark + TF32（Conv/MatMul）。本项目输入形状固定、batch 大小变化
      但卷积核形状固定，开启 benchmark 让 cuDNN 自动选最快算法；TF32 在不牺牲收敛
      稳定性的前提下显著加速卷积与线性层。
    - XPU/CPU：当前 PyTorch 版本无等价的全局 kernel 自动调优开关，保持默认。
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        # The model uses BF16 autocast for the bulk of the forward path.  This
        # setting also permits TF32 for the remaining float32 matmuls.
        torch.set_float32_matmul_precision("high")


def print_device_memory(device: torch.device) -> None:
    """启动时打印总显存，供人工核对环境（DP6）。"""
    try:
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            total = torch.xpu.get_device_properties(0).total_memory / 1024**3
        else:
            return
        print(f"[device] {device} 总显存 ≈ {total:.1f} GiB")
    except Exception as e:  # noqa: BLE001
        print(f"[device] 显存查询失败（不影响运行）：{e}")


def empty_cache(device: torch.device) -> None:
    """按设备类型清缓存（DP6）。"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
