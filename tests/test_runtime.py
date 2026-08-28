from __future__ import annotations

import pickle

import torch

import train.runtime as runtime_module
from train.runtime import (
    GpuPerformanceScheduler,
    MatrixBatchSource,
    cpu_thread_budget,
    is_oom_error,
    resolve_cpu_threads,
    resolve_optimizer_execution,
    resolve_precision,
)


def test_cpu_runtime_uses_fp32_and_respects_update_batch_cap() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    model = torch.nn.Linear(8, 2)

    assert resolve_precision(torch.device("cpu")).name == "fp32"
    assert runtime.choose_batch_size(
        64,
        (2, 4),
        model=model,
        max_update_batch_size=16,
    ) == 16


def test_optimizer_execution_defaults_to_cuda_and_audits_cpu_fallback() -> None:
    cuda_policy = resolve_optimizer_execution(torch.device("cuda:0"))
    assert cuda_policy.fused_adam is True
    assert cuda_policy.compile_mode == "reduce-overhead"
    assert cuda_policy.uses_cuda_graphs is True

    cpu_record = resolve_optimizer_execution(torch.device("cpu")).record()
    assert cpu_record["fused_adam_requested"] is True
    assert cpu_record["compile_mode_requested"] == "reduce-overhead"
    assert cpu_record["fused_adam"] is False
    assert cpu_record["compile_mode"] is None
    assert cpu_record["optimizer_fallback_reason"] == "compile_and_fused_adam_require_cuda"

    compile_only = resolve_optimizer_execution(
        torch.device("cpu"), fused_adam=False
    ).record()
    assert compile_only["optimizer_fallback_reason"] == "compile_requires_cuda"

    fused_only = resolve_optimizer_execution(
        torch.device("cpu"), compile_mode=None
    ).record()
    assert fused_only["optimizer_fallback_reason"] == "fused_adam_requires_cuda"


def test_matrix_batch_source_preserves_row_label_alignment() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    X = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    y = torch.arange(5, dtype=torch.int64)
    source = MatrixBatchSource(X, y, runtime, preload=True)
    permutation = torch.tensor([4, 1, 3, 0, 2])

    rows = list(source.batches(2, indices=permutation))
    observed_X = torch.cat([xb for xb, _ in rows])
    observed_y = torch.cat([yb for _, yb in rows if yb is not None])

    assert not source.preloaded
    assert torch.equal(observed_X, X[permutation])
    assert torch.equal(observed_y, y[permutation])


def test_shared_worker_budget_is_explicit_and_oom_detection_is_narrow() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    runtime.configure_shared_worker_budget(2)
    restored = pickle.loads(pickle.dumps(runtime))

    assert runtime.shared_worker_count == 2
    assert restored.shared_worker_count == 2
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


def test_gpu_worker_recommendation_requires_large_total_memory(monkeypatch) -> None:
    runtime = GpuPerformanceScheduler(torch.device("cuda:0"))
    monkeypatch.setattr(
        runtime_module,
        "_memory_info",
        lambda _device: (8 * 1024**3, 16 * 1024**3),
    )
    assert runtime.recommended_concurrent_workers(4) == 1

    monkeypatch.setattr(
        runtime_module,
        "_memory_info",
        lambda _device: (28 * 1024**3, 32 * 1024**3),
    )
    assert runtime.recommended_concurrent_workers(4) == 4

    xpu_runtime = GpuPerformanceScheduler(torch.device("xpu:0"))
    assert xpu_runtime.recommended_concurrent_workers(4) == 1


def test_cpu_budget_is_divided_per_worker_and_restored() -> None:
    assert resolve_cpu_threads(2, total_threads=16, available_threads=16) == 8
    assert resolve_cpu_threads(3, total_threads=16, available_threads=16) == 5

    previous = torch.get_num_threads()
    with cpu_thread_budget(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == previous


def test_available_cpu_threads_respects_cgroup_quota(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module.os, "process_cpu_count", lambda: 128, raising=False)
    monkeypatch.setattr(
        runtime_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(128)),
        raising=False,
    )
    monkeypatch.setattr(runtime_module, "_linux_cgroup_cpu_quota_threads", lambda: 16)

    assert runtime_module.available_cpu_threads() == 16

def test_matrix_batch_source_one_shot_shuffle_preserves_permutation() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    X = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    y = torch.arange(5, dtype=torch.int64)
    source = MatrixBatchSource(X, y, runtime, preload=False)
    # Exercise the one-shot branch without an accelerator by hand-wiring the
    # already-validated preload state on the CPU tensors.
    source.preloaded = True
    source.device_X = X
    source.device_y = y
    source.shuffle_each_epoch = True
    generator = torch.Generator().manual_seed(7)
    rows = list(source.shuffled_batches(2, generator))
    observed_X = torch.cat([xb for xb, _ in rows])
    observed_y = torch.cat([yb for _, yb in rows if yb is not None])

    expected_generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(5, generator=expected_generator)
    assert torch.equal(observed_X, X[permutation])
    assert torch.equal(observed_y, y[permutation])
