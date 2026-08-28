"""Device-aware training runtime for matrix-shaped EEG workloads.

The runtime owns accelerator-specific policy only: precision, memory telemetry,
batch admission, non-blocking transfers, and in-process device leases. Model
and scientific protocol code remain outside this module.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import ceil, prod
from pathlib import Path
from threading import Lock, RLock
from typing import Literal

import torch

from train.device import optimize_device_for_training, release_device_memory

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - sklearn installations normally provide it.
    threadpool_limits = None

PrecisionPreference = Literal["auto", "bf16", "fp32"]
_ACCELERATOR_TYPES = frozenset({"cuda", "xpu"})


def is_accelerator(device: torch.device) -> bool:
    """Return whether ``device`` exposes an accelerator memory allocator."""

    return device.type in _ACCELERATOR_TYPES


def is_oom_error(error: BaseException) -> bool:
    """Recognize PyTorch OOM variants without masking unrelated runtime errors."""

    oom_type = getattr(torch, "OutOfMemoryError", ())
    if oom_type and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "out of memory" in message or "cuda oom" in message or "显存溢出" in message


def _linux_cgroup_cpu_quota_threads() -> int | None:
    """Return a finite Linux cgroup CPU quota, when one is enforced."""

    if sys.platform != "linux":
        return None
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota_text != "max":
            quota, period = int(quota_text), int(period_text)
            if quota > 0 and period > 0:
                return max(1, ceil(quota / period))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def available_cpu_threads() -> int:
    """Return CPUs usable by this process, including a Linux cgroup quota."""

    process_count = getattr(os, "process_cpu_count", None)
    detected = process_count() if callable(process_count) else os.cpu_count()
    limits = [max(1, int(detected or 1))]
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            limits.append(max(1, len(affinity(0))))
        except OSError:
            pass
    quota = _linux_cgroup_cpu_quota_threads()
    if quota is not None:
        limits.append(quota)
    return min(limits)


def resolve_cpu_threads(
    worker_count: int,
    *,
    total_threads: int | None = None,
    available_threads: int | None = None,
) -> int:
    """Divide a host CPU budget across independent fold workers."""

    if worker_count < 1:
        raise ValueError("worker_count must be positive.")
    available = available_cpu_threads() if available_threads is None else int(available_threads)
    if available < 1:
        raise ValueError("available_threads must be positive.")
    if total_threads is not None and total_threads < 1:
        raise ValueError("total_threads must be positive or None.")
    budget = available if total_threads is None else min(int(total_threads), available)
    return max(1, budget // worker_count)


@contextmanager
def cpu_thread_budget(threads: int) -> Iterator[None]:
    """Apply one CPU budget to PyTorch and native numeric thread pools.

    Call this around a whole executor for thread workers, or inside one spawned
    process worker. It intentionally does not configure PyTorch inter-op
    threads, because that setting is process-global and immutable after use.
    """

    if threads < 1:
        raise ValueError("threads must be positive.")
    previous_threads = torch.get_num_threads()
    changed_torch_threads = False
    try:
        torch.set_num_threads(int(threads))
        changed_torch_threads = True
    except RuntimeError:
        # A pre-existing parallel region can reject a late change. BLAS limits
        # still protect the workload when threadpoolctl is available.
        pass
    try:
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=int(threads)):
                yield
    finally:
        if changed_torch_threads:
            try:
                torch.set_num_threads(previous_threads)
            except RuntimeError:
                pass


def configure_spawned_worker_threads(threads: int) -> None:
    """Set the one-time PyTorch inter-op budget in a fresh spawned worker."""

    if threads < 1:
        raise ValueError("threads must be positive.")
    try:
        torch.set_num_interop_threads(min(2, int(threads)))
    except RuntimeError:
        # PyTorch permits this exactly once before inter-op execution. A worker
        # that has already used it keeps its existing safe configuration.
        pass


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved autocast policy for one selected device."""

    preference: PrecisionPreference
    amp_enabled: bool
    amp_dtype: torch.dtype | None

    @property
    def name(self) -> str:
        if not self.amp_enabled or self.amp_dtype is None:
            return "fp32"
        return "bf16" if self.amp_dtype is torch.bfloat16 else str(self.amp_dtype).removeprefix("torch.")


@dataclass(frozen=True)
class MemorySnapshot:
    """Allocator state in MiB; unavailable backend fields remain ``None``."""

    device: str
    allocated_mb: float | None = None
    reserved_mb: float | None = None
    peak_allocated_mb: float | None = None
    peak_reserved_mb: float | None = None
    free_mb: float | None = None
    total_mb: float | None = None

    def record(self) -> dict[str, str | float | None]:
        return asdict(self)


def _device_index(device: torch.device) -> int:
    return 0 if device.index is None else int(device.index)


def _is_pinned_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a CPU tensor is in accelerator-pinned host memory."""

    try:
        return bool(tensor.is_pinned())
    except (AttributeError, RuntimeError):
        return False


def _backend(device: torch.device):
    return getattr(torch, device.type, None)


def _backend_call(device: torch.device, name: str) -> float | None:
    backend = _backend(device)
    function = getattr(backend, name, None)
    if not callable(function):
        return None
    try:
        return float(function(device))
    except (AttributeError, RuntimeError, TypeError):
        return None


def _memory_info(device: torch.device) -> tuple[float | None, float | None]:
    backend = _backend(device)
    function = getattr(backend, "mem_get_info", None)
    if not callable(function):
        return None, None
    try:
        free_bytes, total_bytes = function(device)
    except (AttributeError, RuntimeError, TypeError):
        return None, None
    return float(free_bytes), float(total_bytes)


def _supports_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        function = getattr(torch.cuda, "is_bf16_supported", None)
        if not callable(function):
            return False
        try:
            return bool(function(including_emulation=False))
        except TypeError:
            return bool(function())
    if device.type == "xpu":
        function = getattr(getattr(torch, "xpu", None), "is_bf16_supported", None)
        if callable(function):
            try:
                return bool(function())
            except RuntimeError:
                return False
        # XPU autocast supports bf16 in supported PyTorch releases.
        return True
    return False


def resolve_precision(
    device: torch.device,
    preference: PrecisionPreference = "auto",
) -> PrecisionPolicy:
    """Resolve a conservative BF16 policy without silently using an unsupported dtype."""

    if preference not in {"auto", "bf16", "fp32"}:
        raise ValueError("precision preference must be 'auto', 'bf16', or 'fp32'.")
    if preference == "fp32" or not is_accelerator(device):
        return PrecisionPolicy(preference=preference, amp_enabled=False, amp_dtype=None)
    if _supports_bf16(device):
        return PrecisionPolicy(preference=preference, amp_enabled=True, amp_dtype=torch.bfloat16)
    if preference == "bf16":
        raise RuntimeError(f"BF16 was requested but is unavailable on {device}.")
    return PrecisionPolicy(preference=preference, amp_enabled=False, amp_dtype=None)


class MatrixBatchSource:
    """A contiguous matrix source that preloads when safe and otherwise streams batches.

    ``X`` is always two- or higher-dimensional with rows on axis zero. The
    class never iterates individual samples; the only loop is the required
    optimizer-batch loop.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None,
        runtime: GpuPerformanceScheduler,
        *,
        preload: bool,
    ) -> None:
        if X.device.type != "cpu" or X.ndim < 2:
            raise ValueError("MatrixBatchSource requires a CPU tensor with a row axis.")
        if y is not None and (y.device.type != "cpu" or y.shape != (len(X),)):
            raise ValueError("Labels must be a CPU vector aligned with X.")
        self.runtime = runtime
        self.cpu_X = X.contiguous()
        self.cpu_y = None if y is None else y.contiguous()
        self.device_X: torch.Tensor | None = None
        self.device_y: torch.Tensor | None = None
        self.preloaded = False
        self.transfer_fallback = False
        self.shuffle_each_epoch = False
        self._shuffle_x: torch.Tensor | None = None
        self._shuffle_y: torch.Tensor | None = None

        if preload and runtime.can_preload(self.nbytes):
            try:
                self.device_X = runtime.to_device(self.cpu_X)
                self.device_y = None if self.cpu_y is None else runtime.to_device(self.cpu_y)
                self.preloaded = True
            except RuntimeError as error:
                if not is_oom_error(error):
                    raise
                self.device_X = None
                self.device_y = None
                self.transfer_fallback = True
                runtime.release_temporary_memory()

        if not self.preloaded and is_accelerator(runtime.device):
            self.cpu_X = runtime.pin_memory(self.cpu_X)
            if self.cpu_y is not None:
                self.cpu_y = runtime.pin_memory(self.cpu_y)

    @property
    def n_rows(self) -> int:
        return int(self.cpu_X.shape[0])

    @property
    def nbytes(self) -> int:
        label_bytes = 0 if self.cpu_y is None else self.cpu_y.numel() * self.cpu_y.element_size()
        return self.cpu_X.numel() * self.cpu_X.element_size() + label_bytes

    def make_generator(self, seed: int) -> torch.Generator:
        if self.preloaded:
            try:
                generator = torch.Generator(device=self.runtime.device)
            except RuntimeError:
                generator = torch.Generator()
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def random_permutation(self, generator: torch.Generator) -> torch.Tensor:
        if self.preloaded:
            generator_device = getattr(generator, "device", None)
            if generator_device is not None and generator_device.type == self.runtime.device.type:
                return torch.randperm(
                    self.n_rows, generator=generator, device=self.runtime.device
                )
            # A backend without accelerator Generator support received a CPU
            # fallback generator. Generate on CPU and move the permutation once;
            # row order and label alignment are unchanged.
            return torch.randperm(self.n_rows, generator=generator).to(self.runtime.device)
        return torch.randperm(self.n_rows, generator=generator)

    def shuffled_batches(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        """Yield row-permuted batches using one device-side shuffle per epoch.

        When the full matrix is already resident and extra device headroom was
        admitted, one ``index_select`` of the complete permutation replaces
        one ``index_select`` per optimizer batch. The resulting rows are
        identical to the per-batch path; only the number and size of gather
        kernels change. If shuffle buffers are unavailable, this falls back to
        the original per-batch path with the same permutation.
        """

        permutation = self.random_permutation(generator)
        if not self.preloaded or not self.shuffle_each_epoch:
            yield from self.batches(batch_size, indices=permutation)
            return
        assert self.device_X is not None
        try:
            if self._shuffle_x is None or self._shuffle_x.shape != self.device_X.shape:
                self._shuffle_x = torch.empty_like(self.device_X)
            torch.index_select(self.device_X, 0, permutation, out=self._shuffle_x)
            shuffled_y: torch.Tensor | None = None
            if self.device_y is not None:
                if self._shuffle_y is None or self._shuffle_y.shape != self.device_y.shape:
                    self._shuffle_y = torch.empty_like(self.device_y)
                torch.index_select(self.device_y, 0, permutation, out=self._shuffle_y)
                shuffled_y = self._shuffle_y
        except RuntimeError as error:
            if not is_oom_error(error):
                raise
            # Keep the exact row order; only lose the one-shot shuffle
            # optimization. Releasing buffers at this boundary is safe because
            # no batch from the failed shuffle has been yielded yet.
            self.shuffle_each_epoch = False
            self._shuffle_x = None
            self._shuffle_y = None
            self.runtime.release_temporary_memory()
            yield from self.batches(batch_size, indices=permutation)
            return

        for start in range(0, self.n_rows, batch_size):
            stop = min(start + batch_size, self.n_rows)
            xb = self._shuffle_x[start:stop]
            yb = None if shuffled_y is None else shuffled_y[start:stop]
            yield xb, yb

    def batches(
        self,
        batch_size: int,
        *,
        indices: torch.Tensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if indices is not None and (indices.ndim != 1 or len(indices) != self.n_rows):
            raise ValueError("indices must be a complete one-dimensional row permutation.")

        for start in range(0, self.n_rows, batch_size):
            stop = min(start + batch_size, self.n_rows)
            if self.preloaded:
                assert self.device_X is not None
                if indices is None:
                    xb = self.device_X[start:stop]
                    yb = None if self.device_y is None else self.device_y[start:stop]
                else:
                    rows = indices[start:stop]
                    xb = self.device_X.index_select(0, rows)
                    yb = None if self.device_y is None else self.device_y.index_select(0, rows)
            else:
                if indices is None:
                    selected_x = self.cpu_X[start:stop]
                    selected_y = None if self.cpu_y is None else self.cpu_y[start:stop]
                else:
                    cpu_rows = indices[start:stop].cpu()
                    selected_x = self.cpu_X.index_select(0, cpu_rows)
                    selected_y = (
                        None if self.cpu_y is None else self.cpu_y.index_select(0, cpu_rows)
                    )
                xb = self.runtime.to_device(selected_x)
                if self.cpu_y is None:
                    yb = None
                else:
                    assert selected_y is not None
                    yb = self.runtime.to_device(selected_y)
            yield xb, yb


class GpuPerformanceScheduler:
    """Own accelerator policy and serialize conflicting in-process workloads.

    One process may host CPU preprocessing workers, but model construction and
    execution on the same accelerator receive an exclusive lease. This avoids
    shared CUDA RNG and allocator races. Parallel folds use isolated spawned
    processes; when they share one large accelerator, their batch and preload
    budgets are divided explicitly before model construction.
    """

    _registry_lock = Lock()
    _device_locks: dict[tuple[str, int], RLock] = {}

    def __init__(
        self,
        device: torch.device,
        *,
        precision: PrecisionPreference = "auto",
        batch_memory_fraction: float = 0.55,
        preload_memory_fraction: float = 0.30,
    ) -> None:
        if not 0.05 <= batch_memory_fraction < 1.0:
            raise ValueError("batch_memory_fraction must be in [0.05, 1).")
        if not 0.05 <= preload_memory_fraction < 1.0:
            raise ValueError("preload_memory_fraction must be in [0.05, 1).")
        self.device = torch.device(device)
        self.precision = resolve_precision(self.device, precision)
        self.batch_memory_fraction = float(batch_memory_fraction)
        self.preload_memory_fraction = float(preload_memory_fraction)
        self.shared_worker_count = 1
        self.last_memory: MemorySnapshot | None = None

    @property
    def device_key(self) -> tuple[str, int]:
        return self.device.type, _device_index(self.device)

    @classmethod
    def _lock_for(cls, key: tuple[str, int]) -> RLock:
        with cls._registry_lock:
            return cls._device_locks.setdefault(key, RLock())

    def configure(self) -> None:
        """Enable stable backend choices once a lease has been acquired."""

        optimize_device_for_training(self.device)

    def autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.precision.amp_dtype or torch.float32,
            enabled=self.precision.amp_enabled,
        )

    @contextmanager
    def lease(self) -> Iterator[GpuPerformanceScheduler]:
        """Serialize one in-process workload for this logical device."""

        lock = self._lock_for(self.device_key)
        with lock:
            self.configure()
            self.reset_peak_memory()
            try:
                yield self
            finally:
                self.synchronize()
                self.last_memory = self.memory_snapshot()

    def synchronize(self) -> None:
        backend = _backend(self.device)
        function = getattr(backend, "synchronize", None)
        if callable(function) and is_accelerator(self.device):
            try:
                function(self.device)
            except (AttributeError, RuntimeError, TypeError):
                pass

    def reset_peak_memory(self) -> None:
        backend = _backend(self.device)
        function = getattr(backend, "reset_peak_memory_stats", None)
        if callable(function) and is_accelerator(self.device):
            try:
                function(self.device)
            except (AttributeError, RuntimeError, TypeError):
                pass

    def memory_snapshot(self) -> MemorySnapshot:
        if not is_accelerator(self.device):
            return MemorySnapshot(device=str(self.device))
        free_bytes, total_bytes = _memory_info(self.device)
        mib = 1024.0**2
        return MemorySnapshot(
            device=str(self.device),
            allocated_mb=_divide_mib(_backend_call(self.device, "memory_allocated"), mib),
            reserved_mb=_divide_mib(_backend_call(self.device, "memory_reserved"), mib),
            peak_allocated_mb=_divide_mib(_backend_call(self.device, "max_memory_allocated"), mib),
            peak_reserved_mb=_divide_mib(_backend_call(self.device, "max_memory_reserved"), mib),
            free_mb=_divide_mib(free_bytes, mib),
            total_mb=_divide_mib(total_bytes, mib),
        )

    def memory_record(self) -> dict[str, str | float | None]:
        return (self.last_memory or self.memory_snapshot()).record()

    def configure_shared_worker_budget(self, worker_count: int) -> None:
        """Split conservative allocator headroom across isolated GPU workers.

        This is called before a spawned fold worker begins model construction.
        It does not attempt to synchronize allocator state across processes;
        instead, every worker reserves only its share of the parent policy.
        """

        if worker_count < 1:
            raise ValueError("worker_count must be positive.")
        self.shared_worker_count = int(worker_count)

    def recommended_concurrent_workers(self, requested: int, *, cap: int = 4) -> int:
        """Return a hardware-aware default for one physical accelerator.

        A compact model can fit many copies, but allocator fragmentation and
        simultaneous CPU-to-device staging still make that a poor default on
        mobile 8--16 GiB cards. Explicit CLI configuration may override this
        recommendation after a CUDA smoke benchmark.
        """

        if requested < 1 or cap < 1:
            raise ValueError("requested and cap must be positive.")
        if self.device.type != "cuda":
            return 1
        _, total_bytes = _memory_info(self.device)
        if total_bytes is None or total_bytes < 24 * 1024**3:
            return 1
        return min(int(requested), int(cap))

    def can_preload(self, nbytes: int) -> bool:
        """Admit a whole matrix only when it leaves headroom for activations."""

        if nbytes < 0:
            raise ValueError("nbytes must be non-negative.")
        if not is_accelerator(self.device):
            return False
        free_bytes, _ = _memory_info(self.device)
        per_worker_fraction = self.preload_memory_fraction / self.shared_worker_count
        return free_bytes is not None and nbytes <= int(free_bytes * per_worker_fraction)

    def choose_batch_size(
        self,
        requested: int,
        sample_shape: Sequence[int],
        *,
        model: torch.nn.Module,
        max_update_batch_size: int | None = None,
    ) -> int:
        """Return a conservative batch cap using live allocator headroom.

        The configured batch is an upper bound, not a target to maximize. This
        preserves more optimizer updates when memory permits and never uses
        gradient accumulation.
        """

        if requested < 1:
            raise ValueError("requested batch size must be positive.")
        if not sample_shape or any(int(value) < 1 for value in sample_shape):
            raise ValueError("sample_shape must contain positive dimensions.")
        cap = requested if max_update_batch_size is None else min(requested, max_update_batch_size)
        if cap < 1:
            raise ValueError("max_update_batch_size must be positive when set.")
        if not is_accelerator(self.device):
            return cap

        free_bytes, _ = _memory_info(self.device)
        if free_bytes is None:
            return cap
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        # Parameters, gradients, and Adam moments are persistent. Activation
        # memory is model dependent; use a deliberately conservative multiplier.
        persistent_bytes = parameter_bytes * 5
        activation_bytes_per_row = max(1, prod(int(value) for value in sample_shape) * 4 * 16)
        per_worker_fraction = self.batch_memory_fraction / self.shared_worker_count
        available = int(free_bytes * per_worker_fraction) - persistent_bytes
        if available <= 0:
            return 1
        admitted = max(1, available // activation_bytes_per_row)
        return max(1, min(cap, admitted))

    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or not is_accelerator(self.device):
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        non_blocking = is_accelerator(self.device) and _is_pinned_tensor(tensor)
        return tensor.to(self.device, non_blocking=non_blocking)

    def release_temporary_memory(self) -> None:
        """Run expensive allocator cleanup only at workload boundaries."""

        release_device_memory(self.device)


def _divide_mib(value: float | None, mib: float) -> float | None:
    return None if value is None else value / mib
