"""Behavioral controls for resource limits and compilation lifetime; no training."""
from pathlib import Path

import pytest

from train import runtime as actual_runtime


@pytest.fixture(scope="module")
def runtime():
    return actual_runtime


def test_windows_affinity_restricts_runtime_workers(runtime, monkeypatch):
    import psutil

    monkeypatch.setattr(runtime.os, "process_cpu_count", lambda: 32, raising=False)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", None, raising=False)
    monkeypatch.setattr(runtime, "_linux_cgroup_cpu_quota_threads", lambda: None)
    monkeypatch.setattr(runtime, "_windows_job_cpu_limit_threads", lambda count: None, raising=False)
    monkeypatch.setattr(psutil.Process, "cpu_affinity", lambda self: [0, 1, 2, 3])
    assert runtime.available_cpu_threads() == 4


def test_runtime_includes_windows_job_hard_cap(runtime, monkeypatch):
    monkeypatch.setattr(runtime.os, "process_cpu_count", lambda: 32, raising=False)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda process: set(range(32)), raising=False)
    monkeypatch.setattr(runtime, "_linux_cgroup_cpu_quota_threads", lambda: None)
    monkeypatch.setattr(runtime, "_windows_job_cpu_limit_threads", lambda count: 6, raising=False)
    assert runtime.available_cpu_threads() == 6


def test_fractional_linux_quota_does_not_round_worker_capacity_up(runtime, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    original = Path.read_text
    def cpu_max(path, *args, **kwargs):
        if str(path).replace("\\", "/") == "/sys/fs/cgroup/cpu.max":
            return "150000 100000"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", cpu_max)
    assert runtime._linux_cgroup_cpu_quota_threads() == 1


def test_task_cleanup_keeps_a_valid_compiled_graph(monkeypatch):
    import torch

    from research import source_execution as owner
    from train import device
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(device, "release_device_memory", lambda dev: None)
    compiled_graphs = []
    def compile_backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward
    function = torch.compile(lambda value: value.sin()+value.cos(), backend=compile_backend, fullgraph=True)
    tensor = torch.arange(8, dtype=torch.float32)
    expected = tensor.sin()+tensor.cos()
    assert torch.equal(function(tensor), expected)
    owner.release_source_worker("cuda")
    assert torch.equal(function(tensor), expected)
    assert len(compiled_graphs) == 1


def test_random_streaming_does_not_allocate_a_whole_matrix_pinned_copy(runtime):
    import torch

    class StreamingRuntime:
        device = torch.device("cuda")
        def can_preload(self, nbytes):
            return False
        def pin_memory(self, tensor):
            raise AssertionError("random streaming created an unused whole-matrix pinned copy")
        def to_device(self, tensor):
            return tensor.clone()
    values = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3)
    labels = torch.arange(8)
    source = runtime.MatrixBatchSource(values, labels, StreamingRuntime(), preload=False)
    rows = torch.tensor([7, 0, 3])
    actual = source.gather_indices(rows)
    assert source.cpu_X.data_ptr() == values.data_ptr()
    assert torch.equal(actual[0], values[rows])
    assert torch.equal(actual[1], labels[rows])
