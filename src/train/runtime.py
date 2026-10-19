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
from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path
from threading import Lock, RLock
from typing import Literal

import torch
from threadpoolctl import threadpool_limits

from train.device import optimize_device_for_training, release_device_memory

PrecisionPreference = Literal["auto", "bf16", "fp32"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]
DEFAULT_FUSED_ADAM = True
DEFAULT_COMPILE_MODE: CompileMode = "reduce-overhead"
_ACCELERATOR_TYPES = frozenset({"cuda", "xpu"})
MATRIX_ROW_ORDER_PROGRAM = "torch_CPU_generator_randperm_then_device_gather/1"


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
                return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def _windows_job_cpu_limit_threads(host_threads: int) -> int | None:
    """Read the current Job's hard CPU-rate cap without changing that Job.

    CPU-rate limits are a fraction of the host's scheduling capacity. Weight
    modes are relative priorities and therefore cannot be treated as a cap.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class CpuRateControl(ctypes.Structure):
        _fields_ = (("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD))

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.QueryInformationJobObject
    query.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                      ctypes.POINTER(wintypes.DWORD))
    query.restype = wintypes.BOOL
    information, returned = CpuRateControl(), wintypes.DWORD()
    if not query(None, 15, ctypes.byref(information), ctypes.sizeof(information), ctypes.byref(returned)):
        return None  # No enclosing Job, or an unavailable query, is unknown.
    flags = information.ControlFlags
    if flags & 1 and flags & 4 and not flags & 2 and 0 < information.CpuRate <= 10000:
        return max(1, host_threads*information.CpuRate//10000)
    return None


def available_cpu_threads() -> int:
    """Bound worker capacity by process affinity, cgroup quota and Windows Job."""

    process_count = getattr(os, "process_cpu_count", None)
    detected = process_count() if callable(process_count) else os.cpu_count()
    limits = [max(1, int(detected or 1))]
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            limits.append(max(1, len(affinity(0))))
        except OSError:
            pass
    else:
        import psutil

        try:
            limits.append(max(1, len(psutil.Process().cpu_affinity())))
        except (psutil.Error, AttributeError, NotImplementedError):
            pass
    quota = _linux_cgroup_cpu_quota_threads()
    if quota is not None:
        limits.append(quota)
    job_limit = _windows_job_cpu_limit_threads(max(1, os.cpu_count() or 1))
    if job_limit is not None:
        limits.append(job_limit)
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


@dataclass(frozen=True)
class CpuWorkerPlan:
    """Resolved process-wide CPU budget for independent parallel tasks."""

    task_count: int
    requested_workers: int | None
    effective_workers: int
    available_threads: int
    total_thread_budget: int
    threads_per_worker: int

    def record(self) -> dict[str, int | str]:
        return {
            "task_count": self.task_count,
            "requested_workers": (
                "auto" if self.requested_workers is None else self.requested_workers
            ),
            "effective_workers": self.effective_workers,
            "available_threads": self.available_threads,
            "total_thread_budget": self.total_thread_budget,
            "threads_per_worker": self.threads_per_worker,
        }


def resolve_cpu_worker_plan(
    task_count: int,
    *,
    requested_workers: int | None = None,
    total_threads: int | None = None,
    max_workers: int | None = None,
    available_threads: int | None = None,
) -> CpuWorkerPlan:
    """Resolve bounded task workers and native threads from one CPU budget."""

    if task_count < 1:
        raise ValueError("task_count must be positive.")
    available = available_cpu_threads() if available_threads is None else int(available_threads)
    if available < 1:
        raise ValueError("available_threads must be positive.")
    if requested_workers is not None and requested_workers < 1:
        raise ValueError("requested_workers must be positive or None.")
    if total_threads is not None and total_threads < 1:
        raise ValueError("total_threads must be positive or None.")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive or None.")
    budget = available if total_threads is None else min(int(total_threads), available)
    requested = task_count if requested_workers is None else int(requested_workers)
    effective = min(task_count, requested, budget)
    if max_workers is not None:
        effective = min(effective, int(max_workers))
    threads_per_worker = resolve_cpu_threads(
        effective,
        total_threads=budget,
        available_threads=available,
    )
    return CpuWorkerPlan(
        task_count=int(task_count),
        requested_workers=requested_workers,
        effective_workers=int(effective),
        available_threads=int(available),
        total_thread_budget=int(budget),
        threads_per_worker=int(threads_per_worker),
    )


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
        # still protect the workload.
        pass
    try:
        with threadpool_limits(limits=int(threads)):
            yield
    finally:
        if changed_torch_threads:
            try:
                torch.set_num_threads(previous_threads)
            except RuntimeError:
                pass


def configure_spawned_worker_threads(threads: int, *, interop_threads: int | None = None) -> None:
    """Set the one-time PyTorch inter-op budget in a fresh spawned worker."""

    if threads < 1:
        raise ValueError("threads must be positive.")
    if interop_threads is not None and (type(interop_threads) is not int or interop_threads < 1):
        raise ValueError("explicit inter-op threads must be a positive integer.")
    requested = min(2, int(threads)) if interop_threads is None else interop_threads
    try:
        torch.set_num_interop_threads(requested)
    except RuntimeError:
        # PyTorch permits this exactly once before inter-op execution. A worker
        # that has already used it keeps its existing safe configuration.
        pass
    if interop_threads is not None and torch.get_num_interop_threads() != requested:
        raise RuntimeError("worker cannot establish its declared inter-op thread budget")


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
        return (
            "bf16"
            if self.amp_dtype is torch.bfloat16
            else str(self.amp_dtype).removeprefix("torch.")
        )


@dataclass(frozen=True)
class OptimizerExecutionPolicy:
    """Requested and device-effective optimizer/compiler settings."""

    requested_fused_adam: bool
    requested_compile_mode: CompileMode | None
    fused_adam: bool
    compile_mode: CompileMode | None
    fallback_reason: str | None = None

    @property
    def uses_cuda_graphs(self) -> bool:
        return self.compile_mode in {"reduce-overhead", "max-autotune"}

    def without_cuda_graphs(self, *, reason: str) -> OptimizerExecutionPolicy:
        """Keep requested controls while avoiding unsupported graph state replay."""
        if not reason.strip():
            raise ValueError("a CUDA Graph fallback requires a concrete reason.")
        if not self.uses_cuda_graphs:
            return self
        return replace(self, compile_mode="default", fallback_reason=reason)

    def record(self) -> dict[str, bool | str | None]:
        return {
            "fused_adam_requested": self.requested_fused_adam,
            "compile_mode_requested": self.requested_compile_mode,
            "fused_adam": self.fused_adam,
            "compile_mode": self.compile_mode,
            "compile_scope": "train_step" if self.compile_mode is not None else None,
            "optimizer_fallback_reason": self.fallback_reason,
        }


def resolve_optimizer_execution(
    device: torch.device,
    *,
    fused_adam: bool = DEFAULT_FUSED_ADAM,
    compile_mode: CompileMode | None = DEFAULT_COMPILE_MODE,
) -> OptimizerExecutionPolicy:
    """Enable compile and fused Adam on CUDA and audit portable fallback."""

    if not isinstance(fused_adam, bool):
        raise ValueError("fused_adam must be boolean.")
    if compile_mode not in {None, "default", "reduce-overhead", "max-autotune"}:
        raise ValueError("compile_mode is invalid.")
    requested_device = torch.device(device)
    if requested_device.type == "cuda":
        return OptimizerExecutionPolicy(
            requested_fused_adam=fused_adam,
            requested_compile_mode=compile_mode,
            fused_adam=fused_adam,
            compile_mode=compile_mode,
        )
    if fused_adam and compile_mode is not None:
        fallback = "compile_and_fused_adam_require_cuda"
    elif fused_adam:
        fallback = "fused_adam_requires_cuda"
    elif compile_mode is not None:
        fallback = "compile_requires_cuda"
    else:
        fallback = None
    return OptimizerExecutionPolicy(
        requested_fused_adam=fused_adam,
        requested_compile_mode=compile_mode,
        fused_adam=False,
        compile_mode=None,
        fallback_reason=fallback,
    )


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
    optimizer-batch loop. Streaming retains the caller's CPU storage. Random
    gathers produce ordinary CPU tensors and use synchronous copies; explicit
    pinned caller storage still permits asynchronous contiguous slice copies.
    No hidden whole-matrix pinned duplicate is allocated for random batches.
    """

    row_order_program = MATRIX_ROW_ORDER_PROGRAM

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None,
        runtime: GpuPerformanceScheduler,
        *,
        preload: bool,
        row_weights: torch.Tensor | None = None,
        row_domains: torch.Tensor | None = None,
        row_channel_mask: torch.Tensor | None = None,
        release_cpu_after_preload: bool = False,
        row_order_observer=None,
    ) -> None:
        if X.device.type != "cpu" or X.ndim < 2:
            raise ValueError("MatrixBatchSource requires a CPU tensor with a row axis.")
        if y is not None and (y.device.type != "cpu" or y.shape != (len(X),)):
            raise ValueError("Labels must be a CPU vector aligned with X.")
        if row_weights is not None and (
            row_weights.device.type != "cpu"
            or row_weights.shape != (len(X),)
            or not row_weights.is_floating_point()
        ):
            raise ValueError("Row weights must be a floating CPU vector aligned with X.")
        if row_weights is not None and (
            not bool(torch.isfinite(row_weights).all()) or not bool((row_weights > 0).all())
        ):
            raise ValueError("Row weights must be finite and positive.")
        if row_domains is not None and (
            row_domains.device.type != "cpu"
            or row_domains.dtype != torch.long
            or row_domains.shape != (len(X),)
        ):
            raise ValueError("Row domains must be an integer CPU vector aligned with X.")
        if row_channel_mask is not None and (
            row_channel_mask.device.type != "cpu"
            or row_channel_mask.dtype != torch.bool
            or row_channel_mask.shape != X.shape[:2]
            or not bool(row_channel_mask.any(dim=1).all())
        ):
            raise ValueError("Row channel masks must be boolean CPU (N,C), retaining an observed channel per row.")
        self.runtime = runtime
        self.cpu_X = X.contiguous()
        self._n_rows = int(X.shape[0])
        self._matrix_nbytes = X.numel() * X.element_size()
        self.cpu_y = None if y is None else y.contiguous()
        self.cpu_row_weights = None if row_weights is None else row_weights.contiguous()
        self.cpu_row_domains = None if row_domains is None else row_domains.contiguous()
        self.cpu_row_channel_mask = None if row_channel_mask is None else row_channel_mask.contiguous()
        self.device_X: torch.Tensor | None = None
        self.device_y: torch.Tensor | None = None
        self.device_row_weights: torch.Tensor | None = None
        self.device_row_domains: torch.Tensor | None = None
        self.device_row_channel_mask: torch.Tensor | None = None
        self.row_order_observer = row_order_observer
        self.preloaded = False
        self.transfer_fallback = False
        self.materialize_shuffled_epoch = False
        self._shuffle_x: torch.Tensor | None = None
        self._shuffle_y: torch.Tensor | None = None
        self._shuffle_row_weights: torch.Tensor | None = None
        self._shuffle_row_domains: torch.Tensor | None = None

        if preload and runtime.can_preload(self.nbytes):
            try:
                self.device_X = runtime.to_device(self.cpu_X)
                self.device_y = None if self.cpu_y is None else runtime.to_device(self.cpu_y)
                self.device_row_weights = (
                    None
                    if self.cpu_row_weights is None
                    else runtime.to_device(self.cpu_row_weights)
                )
                self.device_row_domains = (
                    None
                    if self.cpu_row_domains is None
                    else runtime.to_device(self.cpu_row_domains)
                )
                self.device_row_channel_mask = (
                    None if self.cpu_row_channel_mask is None else runtime.to_device(self.cpu_row_channel_mask)
                )
                self.preloaded = True
            except RuntimeError as error:
                if not is_oom_error(error):
                    raise
                self.device_X = None
                self.device_y = None
                self.device_row_weights = None
                self.device_row_domains = None
                self.device_row_channel_mask = None
                self.transfer_fallback = True
                runtime.release_temporary_memory()

        if self.preloaded and release_cpu_after_preload and is_accelerator(runtime.device):
            # Device storage owns the complete matrix; streaming fallback is
            # already resolved, so the normalized host duplicate can be freed.
            self.cpu_X = None

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def nbytes(self) -> int:
        label_bytes = 0 if self.cpu_y is None else self.cpu_y.numel() * self.cpu_y.element_size()
        weight_bytes = (
            0
            if self.cpu_row_weights is None
            else self.cpu_row_weights.numel() * self.cpu_row_weights.element_size()
        )
        domain_bytes = (
            0
            if self.cpu_row_domains is None
            else self.cpu_row_domains.numel() * self.cpu_row_domains.element_size()
        )
        return (
            self._matrix_nbytes
            + label_bytes
            + weight_bytes
            + domain_bytes
            + (0 if self.cpu_row_channel_mask is None else self.cpu_row_channel_mask.numel())
        )

    def _channel_mask_batch(self, rows: slice | torch.Tensor) -> torch.Tensor | None:
        """Select the same small physical channel axis in every batching mode."""
        if self.cpu_row_channel_mask is None:
            return None
        masks = self.device_row_channel_mask if self.preloaded else self.cpu_row_channel_mask
        assert masks is not None
        selected = (masks[rows] if isinstance(rows, slice)
                    else masks.index_select(0, rows.to(device=masks.device)))
        return selected if self.preloaded else self.runtime.to_device(selected)

    @property
    def has_row_weights(self) -> bool:
        return self.cpu_row_weights is not None

    @property
    def has_row_domains(self) -> bool:
        return self.cpu_row_domains is not None

    def make_generator(self, seed: int) -> torch.Generator:
        # Preload admission depends on live memory, so a device RNG would make
        # a fixed seed depend on whether this source happened to be resident.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator

    def random_permutation(self, generator: torch.Generator) -> torch.Tensor:
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("MatrixBatchSource row permutations require an explicit CPU generator.")
        permutation = torch.randperm(self.n_rows, generator=generator, device="cpu")
        if self.row_order_observer is not None:
            self.row_order_observer(permutation)
        return permutation.to(self.runtime.device) if self.preloaded else permutation

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
        if not self.preloaded or not self.materialize_shuffled_epoch:
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
            self.materialize_shuffled_epoch = False
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

    def shuffled_batches_with_weights(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
        """Yield the same shuffled rows with one aligned positive weight per row."""

        if self.cpu_row_weights is None:
            raise ValueError("MatrixBatchSource has no row weights.")
        permutation = self.random_permutation(generator)
        if not self.preloaded or not self.materialize_shuffled_epoch:
            yield from self.batches_with_weights(batch_size, indices=permutation)
            return
        assert self.device_X is not None and self.device_row_weights is not None
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
            if (
                self._shuffle_row_weights is None
                or self._shuffle_row_weights.shape != self.device_row_weights.shape
            ):
                self._shuffle_row_weights = torch.empty_like(self.device_row_weights)
            torch.index_select(
                self.device_row_weights,
                0,
                permutation,
                out=self._shuffle_row_weights,
            )
        except RuntimeError as error:
            if not is_oom_error(error):
                raise
            self.materialize_shuffled_epoch = False
            self._shuffle_x = None
            self._shuffle_y = None
            self._shuffle_row_weights = None
            self.runtime.release_temporary_memory()
            yield from self.batches_with_weights(batch_size, indices=permutation)
            return

        for start in range(0, self.n_rows, batch_size):
            stop = min(start + batch_size, self.n_rows)
            yield (
                self._shuffle_x[start:stop],
                None if shuffled_y is None else shuffled_y[start:stop],
                self._shuffle_row_weights[start:stop],
            )

    def batches(
        self,
        batch_size: int,
        *,
        indices: torch.Tensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        """Gather an explicit n_rows-long row plan; repeats remain caller-owned.

        Use random_permutation for an epoch permutation, or a declared sampling
        plan when rows may repeat. This gather interface proves alignment, not
        the statistical sampling measure.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if indices is not None and (indices.ndim != 1 or len(indices) != self.n_rows):
            raise ValueError("indices must be a one-dimensional row plan with n_rows entries.")

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

    def batches_with_weights(
        self,
        batch_size: int,
        *,
        indices: torch.Tensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]]:
        if self.cpu_row_weights is None:
            raise ValueError("MatrixBatchSource has no row weights.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if indices is not None and (indices.ndim != 1 or len(indices) != self.n_rows):
            raise ValueError("indices must be a complete one-dimensional row permutation.")

        for start in range(0, self.n_rows, batch_size):
            stop = min(start + batch_size, self.n_rows)
            if self.preloaded:
                assert self.device_X is not None and self.device_row_weights is not None
                if indices is None:
                    xb = self.device_X[start:stop]
                    yb = None if self.device_y is None else self.device_y[start:stop]
                    wb = self.device_row_weights[start:stop]
                else:
                    rows = indices[start:stop]
                    xb = self.device_X.index_select(0, rows)
                    yb = None if self.device_y is None else self.device_y.index_select(0, rows)
                    wb = self.device_row_weights.index_select(0, rows)
            else:
                if indices is None:
                    selected_x = self.cpu_X[start:stop]
                    selected_y = None if self.cpu_y is None else self.cpu_y[start:stop]
                    selected_w = self.cpu_row_weights[start:stop]
                else:
                    cpu_rows = indices[start:stop].cpu()
                    selected_x = self.cpu_X.index_select(0, cpu_rows)
                    selected_y = (
                        None if self.cpu_y is None else self.cpu_y.index_select(0, cpu_rows)
                    )
                    selected_w = self.cpu_row_weights.index_select(0, cpu_rows)
                xb = self.runtime.to_device(selected_x)
                yb = None if selected_y is None else self.runtime.to_device(selected_y)
                wb = self.runtime.to_device(selected_w)
            yield xb, yb, wb

    def iter_batches(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ]:
        """Yield normalized ``(x, y, weights, domains, channel_mask)`` batches.

        ``weights`` and ``domains`` are ``None`` when the source carries no
        such axis, so training loops that route rows through per-domain
        feature adapters can consume one fixed-arity interface. With
        ``generator`` the rows follow its permutation (one-shot device
        shuffle when the fast path is eligible, per-batch gather otherwise);
        without it rows arrive in stored order. Row order and alignment are
        identical to the specialized ``shuffled_batches*`` paths.
        Fast batches are views of a reused epoch buffer. Consume each batch
        before requesting the next epoch; retain an explicit clone for a
        snapshot that must outlive that buffer's current contents.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        permutation = None if generator is None else self.random_permutation(generator)
        fast_path = self.preloaded and self.materialize_shuffled_epoch and permutation is not None
        if fast_path:
            assert self.device_X is not None and permutation is not None
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
                shuffled_w: torch.Tensor | None = None
                if self.device_row_weights is not None:
                    if (
                        self._shuffle_row_weights is None
                        or self._shuffle_row_weights.shape != self.device_row_weights.shape
                    ):
                        self._shuffle_row_weights = torch.empty_like(self.device_row_weights)
                    torch.index_select(
                        self.device_row_weights,
                        0,
                        permutation,
                        out=self._shuffle_row_weights,
                    )
                    shuffled_w = self._shuffle_row_weights
                shuffled_d: torch.Tensor | None = None
                if self.device_row_domains is not None:
                    if (
                        self._shuffle_row_domains is None
                        or self._shuffle_row_domains.shape != self.device_row_domains.shape
                    ):
                        self._shuffle_row_domains = torch.empty_like(self.device_row_domains)
                    torch.index_select(
                        self.device_row_domains,
                        0,
                        permutation,
                        out=self._shuffle_row_domains,
                    )
                    shuffled_d = self._shuffle_row_domains
            except RuntimeError as error:
                if not is_oom_error(error):
                    raise
                self.materialize_shuffled_epoch = False
                self._shuffle_x = None
                self._shuffle_y = None
                self._shuffle_row_weights = None
                self._shuffle_row_domains = None
                self.runtime.release_temporary_memory()
                fast_path = False
            else:
                for start in range(0, self.n_rows, batch_size):
                    stop = min(start + batch_size, self.n_rows)
                    yield (
                        self._shuffle_x[start:stop],
                        None if shuffled_y is None else shuffled_y[start:stop],
                        None if shuffled_w is None else shuffled_w[start:stop],
                        None if shuffled_d is None else shuffled_d[start:stop],
                        self._channel_mask_batch(permutation[start:stop]),
                    )
                return

        for start in range(0, self.n_rows, batch_size):
            stop = min(start + batch_size, self.n_rows)
            if self.preloaded:
                assert self.device_X is not None
                if permutation is None:
                    xb = self.device_X[start:stop]
                    yb = None if self.device_y is None else self.device_y[start:stop]
                    wb = (
                        None
                        if self.device_row_weights is None
                        else self.device_row_weights[start:stop]
                    )
                    db = (
                        None
                        if self.device_row_domains is None
                        else self.device_row_domains[start:stop]
                    )
                else:
                    rows = permutation[start:stop]
                    xb = self.device_X.index_select(0, rows)
                    yb = None if self.device_y is None else self.device_y.index_select(0, rows)
                    wb = (
                        None
                        if self.device_row_weights is None
                        else self.device_row_weights.index_select(0, rows)
                    )
                    db = (
                        None
                        if self.device_row_domains is None
                        else self.device_row_domains.index_select(0, rows)
                    )
            else:
                if permutation is None:
                    selected_x = self.cpu_X[start:stop]
                    selected_y = None if self.cpu_y is None else self.cpu_y[start:stop]
                    selected_w = (
                        None if self.cpu_row_weights is None else self.cpu_row_weights[start:stop]
                    )
                    selected_d = (
                        None if self.cpu_row_domains is None else self.cpu_row_domains[start:stop]
                    )
                else:
                    cpu_rows = permutation[start:stop].cpu()
                    selected_x = self.cpu_X.index_select(0, cpu_rows)
                    selected_y = (
                        None if self.cpu_y is None else self.cpu_y.index_select(0, cpu_rows)
                    )
                    selected_w = (
                        None
                        if self.cpu_row_weights is None
                        else self.cpu_row_weights.index_select(0, cpu_rows)
                    )
                    selected_d = (
                        None
                        if self.cpu_row_domains is None
                        else self.cpu_row_domains.index_select(0, cpu_rows)
                    )
                xb = self.runtime.to_device(selected_x)
                yb = None if selected_y is None else self.runtime.to_device(selected_y)
                wb = None if selected_w is None else self.runtime.to_device(selected_w)
                db = None if selected_d is None else self.runtime.to_device(selected_d)
            yield xb, yb, wb, db, self._channel_mask_batch(
                slice(start, stop) if permutation is None else permutation[start:stop]
            )

    def gather_indices(
        self,
        indices: torch.Tensor,
        *,
        allow_repeated_rows: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Gather one validated row-index batch across every carried row axis."""

        if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional torch tensor.")
        if indices.dtype != torch.long:
            raise ValueError("indices must have dtype torch.long.")
        if indices.device.type != "cpu":
            raise ValueError("indices must be a CPU tensor before device-aware gathering.")
        if len(indices) == 0:
            raise ValueError("indices cannot be empty.")
        if not allow_repeated_rows and len(torch.unique(indices)) != len(indices):
            raise ValueError("indices within one batch must be unique.")
        if bool(((indices < 0) | (indices >= self.n_rows)).any()):
            raise ValueError("indices contain an out-of-range row.")

        if self.preloaded:
            assert self.device_X is not None
            device_indices = indices.to(self.runtime.device)
            xb = self.device_X.index_select(0, device_indices)
            yb = None if self.device_y is None else self.device_y.index_select(0, device_indices)
            wb = (
                None
                if self.device_row_weights is None
                else self.device_row_weights.index_select(0, device_indices)
            )
            db = (
                None
                if self.device_row_domains is None
                else self.device_row_domains.index_select(0, device_indices)
            )
            return xb, yb, wb, db, self._channel_mask_batch(device_indices)

        selected_x = self.cpu_X.index_select(0, indices)
        selected_y = None if self.cpu_y is None else self.cpu_y.index_select(0, indices)
        selected_w = (
            None if self.cpu_row_weights is None else self.cpu_row_weights.index_select(0, indices)
        )
        selected_d = (
            None if self.cpu_row_domains is None else self.cpu_row_domains.index_select(0, indices)
        )
        return (
            self.runtime.to_device(selected_x),
            None if selected_y is None else self.runtime.to_device(selected_y),
            None if selected_w is None else self.runtime.to_device(selected_w),
            None if selected_d is None else self.runtime.to_device(selected_d),
            self._channel_mask_batch(indices),
        )

    def iter_index_batches(
        self,
        index_batches: Sequence[torch.Tensor],
        *,
        require_full_coverage: bool = True,
        allow_repeated_rows: bool = False,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ]:
        """Yield explicit row-index batches, optionally proving exact epoch coverage."""

        batches = tuple(index_batches)
        if not batches:
            raise ValueError("index_batches cannot be empty.")
        for indices in batches:
            if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
                raise ValueError("each index batch must be a one-dimensional torch tensor.")
            if indices.dtype != torch.long or indices.device.type != "cpu":
                raise ValueError("each index batch must be a CPU torch.long tensor.")
        if require_full_coverage:
            flattened = torch.cat(batches)
            if len(flattened) != self.n_rows or not torch.equal(
                torch.sort(flattened).values,
                torch.arange(self.n_rows, dtype=torch.long),
            ):
                raise ValueError("index_batches must cover every source row exactly once.")
        for indices in batches:
            yield self.gather_indices(indices, allow_repeated_rows=allow_repeated_rows)


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
        self.resident_memory_budget_bytes: int | None = None
        self.input_staging_gate = None
        self.host_process_resident_budget_bytes: int | None = None
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

    def configure_resident_memory_budget(self, nbytes: int | None) -> None:
        """Apply a parent-admitted total resident allocation for this process."""
        if nbytes is not None and (type(nbytes) is not int or nbytes < 1):
            raise ValueError("resident memory budget must be positive bytes")
        self.resident_memory_budget_bytes = nbytes

    def configure_host_process_resident_budget(self, nbytes: int | None) -> None:
        """Assign the whole source-group host lease for retained input residency."""
        if nbytes is not None and (type(nbytes) is not int or nbytes < 1):
            raise ValueError("host process resident budget must be positive bytes")
        self.host_process_resident_budget_bytes = nbytes

    @contextmanager
    def stage_inputs(self, *, nbytes=0):
        """Bound host staging peaks while other workers continue GPU fitting."""
        if self.host_process_resident_budget_bytes is not None and nbytes > self.host_process_resident_budget_bytes:
            raise RuntimeError("source normalized inputs exceed the reserved whole-process resident budget")
        if self.input_staging_gate is None:
            yield None
        else:
            with self.input_staging_gate.reserve("nn", nbytes) as lease:
                yield lease

    def complete_input_staging(self, lease, *, retained_bytes):
        """Transfer retained CPU inputs to the live whole-source group lease."""
        import psutil

        host_rss = psutil.Process().memory_info().rss
        budget = self.host_process_resident_budget_bytes
        if budget is not None and host_rss > budget:
            raise RuntimeError(
                f"source input staging exceeds its whole-process resident budget: "
                f"rss={host_rss} budget={budget} retained_normalized={retained_bytes}; "
                "cannot meet the declared source-worker capacity")
        if lease is not None:
            lease.release_unused(0 if budget is not None else retained_bytes)
        return {"host_process_resident_budget_bytes": budget, "host_rss_after_input_staging_bytes": host_rss,
            "normalized_host_resident_bytes": retained_bytes}

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
        if self.resident_memory_budget_bytes is not None and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            headroom = self.resident_memory_budget_bytes - allocated
            return free_bytes is not None and nbytes <= min(headroom, max(0, free_bytes - 512 * 2**20))
        per_worker_fraction = self.preload_memory_fraction / self.shared_worker_count
        return free_bytes is not None and nbytes <= int(free_bytes * per_worker_fraction)

    def choose_batch_size(
        self,
        requested: int,
        sample_shape: Sequence[int],
        *,
        model: torch.nn.Module,
        max_update_batch_size: int | None = None,
        adaptive: bool = False,
    ) -> int:
        """Keep a fixed declared batch, or explicitly request live-memory admission.

        Adaptive programs record their effective batch and update trace. Fixed
        programs keep their optimizer partition when storage/preload changes.
        """

        if requested < 1:
            raise ValueError("requested batch size must be positive.")
        if not sample_shape or any(int(value) < 1 for value in sample_shape):
            raise ValueError("sample_shape must contain positive dimensions.")
        cap = requested if max_update_batch_size is None else min(requested, max_update_batch_size)
        if cap < 1:
            raise ValueError("max_update_batch_size must be positive when set.")
        if not adaptive or not is_accelerator(self.device):
            return cap

        free_bytes, _ = _memory_info(self.device)
        if free_bytes is None:
            return cap
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
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
