"""Low-level device adapter used exclusively by ``train.runtime``.

Scheduling, memory telemetry, batch admission, and worker concurrency belong
to :class:`train.runtime.GpuPerformanceScheduler`. This module only provides
portable device selection plus backend tuning and lifecycle cleanup.
"""

from __future__ import annotations

import gc

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
        conv_precision = getattr(torch.backends.cudnn, "conv", None)
        if conv_precision is not None and hasattr(conv_precision, "fp32_precision"):
            # PyTorch 2.9+ replaced the boolean allow_tf32 switch with a
            # per-domain precision selector. Keep both paths explicit.
            conv_precision.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        # The model uses BF16 autocast for the bulk of the forward path.  This
        # setting also permits TF32 for the remaining float32 matmuls.
        torch.set_float32_matmul_precision("high")


def release_device_memory(device: torch.device) -> None:
    """Release temporary host/device references after a failed or finished workload.

    This is deliberately a lifecycle operation rather than a per-batch action.
    Calling it in the hot path would introduce synchronization and hurt throughput.
    """

    gc.collect()
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
    except (AttributeError, RuntimeError):
        # Cleanup is best-effort. Never obscure the training error that caused it.
        pass
