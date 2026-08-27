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
    assert runtime.recommended_concurrent_workers(4) == 2


def test_cpu_budget_is_divided_per_worker_and_restored() -> None:
    assert resolve_cpu_threads(2, total_threads=16, available_threads=16) == 8
    assert resolve_cpu_threads(3, total_threads=16, available_threads=16) == 5

    previous = torch.get_num_threads()
    with cpu_thread_budget(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == previous
