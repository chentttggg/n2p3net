from __future__ import annotations

import pickle

import pytest
import torch

import train.runtime as runtime_module
from train.runtime import (
    GpuPerformanceScheduler,
    MatrixBatchSource,
    cpu_thread_budget,
    is_oom_error,
    resolve_cpu_threads,
    resolve_cpu_worker_plan,
    resolve_optimizer_execution,
    resolve_precision,
)


def test_cpu_runtime_uses_fp32_and_respects_update_batch_cap() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    model = torch.nn.Linear(8, 2)

    assert resolve_precision(torch.device("cpu")).name == "fp32"
    assert (
        runtime.choose_batch_size(
            64,
            (2, 4),
            model=model,
            max_update_batch_size=16,
        )
        == 16
    )


def test_stateful_graph_fallback_keeps_requested_policy_and_non_graph_compile() -> None:
    original = resolve_optimizer_execution(torch.device("cuda"), compile_mode="reduce-overhead")
    effective = original.without_cuda_graphs(reason="tested_stateful_buffer_failure")
    assert effective.requested_compile_mode == "reduce-overhead"
    assert effective.compile_mode == "default"
    assert effective.fused_adam is True
    assert effective.fallback_reason == "tested_stateful_buffer_failure"
    assert original.compile_mode == "reduce-overhead"
    default = resolve_optimizer_execution(torch.device("cuda"), compile_mode="default")
    assert default.without_cuda_graphs(reason="not_applicable") is default
    cpu = resolve_optimizer_execution(torch.device("cpu"))
    assert cpu.without_cuda_graphs(reason="not_applicable") is cpu


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

    compile_only = resolve_optimizer_execution(torch.device("cpu"), fused_adam=False).record()
    assert compile_only["optimizer_fallback_reason"] == "compile_requires_cuda"

    fused_only = resolve_optimizer_execution(torch.device("cpu"), compile_mode=None).record()
    assert fused_only["optimizer_fallback_reason"] == "fused_adam_requires_cuda"


def test_matrix_batch_source_preserves_row_label_alignment() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    X = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    y = torch.arange(5, dtype=torch.int64)
    source = MatrixBatchSource(X, y, runtime, preload=True)
    permutation = torch.tensor([4, 1, 3, 0, 2])

    rows = list(source.batches(2, indices=permutation))
    observed_X = torch.cat([xb for xb, _ in rows])
    observed_y = torch.cat([yb for _, yb in rows])

    assert not source.preloaded
    assert torch.equal(observed_X, X[permutation])
    assert torch.equal(observed_y, y[permutation])


def test_matrix_batch_source_preserves_row_weight_alignment() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    X = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    y = torch.arange(5, dtype=torch.int64)
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
    source = MatrixBatchSource(X, y, runtime, preload=False, row_weights=weights)
    generator = torch.Generator().manual_seed(11)
    rows = list(source.shuffled_batches_with_weights(2, generator))
    observed_X = torch.cat([xb for xb, _, _ in rows])
    observed_y = torch.cat([yb for _, yb, _ in rows])
    observed_weights = torch.cat([wb for _, _, wb in rows])
    permutation = torch.randperm(5, generator=torch.Generator().manual_seed(11))

    assert torch.equal(observed_X, X[permutation])
    assert torch.equal(observed_y, y[permutation])
    assert torch.equal(observed_weights, weights[permutation])


def test_shared_worker_budget_is_explicit_and_oom_detection_is_narrow() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    runtime.configure_shared_worker_budget(2)
    restored = pickle.loads(pickle.dumps(runtime))

    assert runtime.shared_worker_count == 2
    assert restored.shared_worker_count == 2
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


@pytest.mark.parametrize(
    "device,free_gib,total_gib,expected",
    [
        ("cuda:0", 8, 16, 1),
        ("cuda:0", 28, 32, 4),
        ("xpu:0", 28, 32, 1),
    ],
)
def test_gpu_worker_recommendation_requires_large_total_memory(
    monkeypatch, device, free_gib, total_gib, expected
):
    monkeypatch.setattr(
        runtime_module, "_memory_info", lambda _device: (free_gib * 1024**3, total_gib * 1024**3)
    )
    runtime = GpuPerformanceScheduler(torch.device(device))
    assert runtime.recommended_concurrent_workers(4) == expected


def test_cpu_budget_is_divided_per_worker_and_restored() -> None:
    assert resolve_cpu_threads(2, total_threads=16, available_threads=16) == 8
    assert resolve_cpu_threads(3, total_threads=16, available_threads=16) == 5

    previous = torch.get_num_threads()
    with cpu_thread_budget(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == previous


def test_cpu_worker_plan_respects_tasks_quota_and_explicit_budget() -> None:
    automatic = resolve_cpu_worker_plan(6, available_threads=25)
    assert automatic.record() == {
        "task_count": 6,
        "requested_workers": "auto",
        "effective_workers": 6,
        "available_threads": 25,
        "total_thread_budget": 25,
        "threads_per_worker": 4,
    }

    bounded = resolve_cpu_worker_plan(
        20,
        requested_workers=8,
        total_threads=12,
        max_workers=4,
        available_threads=25,
    )
    assert bounded.effective_workers == 4
    assert bounded.total_thread_budget == 12
    assert bounded.threads_per_worker == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task_count": 0},
        {"task_count": 1, "requested_workers": 0},
        {"task_count": 1, "total_threads": 0},
        {"task_count": 1, "max_workers": 0},
    ],
)
def test_cpu_worker_plan_rejects_invalid_limits(kwargs) -> None:
    with pytest.raises(ValueError):
        resolve_cpu_worker_plan(**kwargs)


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
    source.materialize_shuffled_epoch = True
    generator = torch.Generator().manual_seed(7)
    rows = list(source.shuffled_batches(2, generator))
    observed_X = torch.cat([xb for xb, _ in rows])
    observed_y = torch.cat([yb for _, yb in rows])

    expected_generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(5, generator=expected_generator)
    assert torch.equal(observed_X, X[permutation])
    assert torch.equal(observed_y, y[permutation])


@pytest.mark.parametrize("preloaded,materialize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_matrix_source_preserves_channel_masks_across_all_batch_paths(
    preloaded, materialize, shuffle
):
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    values = torch.arange(120, dtype=torch.float32).reshape(8, 3, 5)
    labels = torch.arange(8)
    masks = (torch.arange(24).reshape(8, 3) % 4) != 0
    source = MatrixBatchSource(values, labels, runtime, preload=False, row_channel_mask=masks)
    if preloaded:
        source.preloaded = True
        source.device_X, source.device_y = values, labels
        source.device_row_channel_mask = masks
    source.materialize_shuffled_epoch = materialize
    generator = torch.Generator().manual_seed(19) if shuffle else None
    expected = (
        torch.randperm(8, generator=torch.Generator().manual_seed(19))
        if shuffle
        else torch.arange(8)
    )
    batches = list(source.iter_batches(3, generator=generator))
    assert torch.equal(torch.cat([batch[0] for batch in batches]), values[expected])
    assert torch.equal(torch.cat([batch[4] for batch in batches]), masks[expected])
    assert (
        source.nbytes
        == values.numel() * values.element_size()
        + labels.numel() * labels.element_size()
        + masks.numel()
    )
    rows = torch.tensor([5, 1, 5])
    indexed = list(
        source.iter_index_batches((rows,), require_full_coverage=False, allow_repeated_rows=True)
    )
    assert torch.equal(indexed[0][4], masks[rows])


@pytest.mark.parametrize(
    "mask",
    [torch.ones(3, 2), torch.ones(3, 3, dtype=torch.bool), torch.zeros(3, 2, dtype=torch.bool)],
)
def test_matrix_source_refuses_invalid_physical_channel_masks(mask):
    with pytest.raises(ValueError, match="channel"):
        MatrixBatchSource(
            torch.ones(3, 2, 4),
            None,
            GpuPerformanceScheduler(torch.device("cpu")),
            preload=False,
            row_channel_mask=mask,
        )


def test_device_preload_can_release_only_the_redundant_host_matrix():
    class Runtime:
        device = torch.device("cuda")
        def can_preload(self, size):
            return True
        def to_device(self, value):
            return value.clone()

    values = torch.arange(120, dtype=torch.float32).reshape(8, 3, 5)
    labels = torch.arange(8)
    source = MatrixBatchSource(values, labels, Runtime(), preload=True, release_cpu_after_preload=True)
    assert source.preloaded and source.cpu_X is None and source.n_rows == 8
    assert source.nbytes == values.numel() * 4 + labels.numel() * 8
    batches = list(source.batches(3))
    assert torch.equal(torch.cat([b[0] for b in batches]), values)
    assert torch.equal(torch.cat([b[1] for b in batches]), labels)


def test_admitted_resident_budget_uses_remaining_process_quota(monkeypatch):
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    runtime.device = torch.device("cuda")
    monkeypatch.setattr("train.runtime._memory_info", lambda device: (9 * 2**30, 32 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 5 * 2**30)
    runtime.configure_resident_memory_budget(7 * 2**30)
    assert runtime.can_preload(1 * 2**30)
    assert not runtime.can_preload(3 * 2**30)
    runtime.configure_resident_memory_budget(None)
    assert runtime.can_preload(2 * 2**30)
    with pytest.raises(ValueError):
        runtime.configure_resident_memory_budget(0)
