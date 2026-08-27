# Device Portability

Use the repository `.venv`. Runtime selection order is CUDA, XPU, then CPU;
all tensors and modules move through a selected `torch.device`. Hard-coded
`.cuda()` and `.xpu()` calls are prohibited.

AMP is optional and device-aware. Its dtype, batch size, accumulation steps,
warm-up count, and profiling batch sizes are experiment configuration. A model
promotion report includes median and p95 inference latency plus peak allocated
and reserved memory where the backend exposes both measures. CPU is the
correctness reference; accelerator results are performance measurements, not
bitwise-reference outputs.

## Runtime Scheduling

`train.runtime.GpuPerformanceScheduler` is the shared accelerator policy. It
uses BF16 autocast when the selected accelerator supports it, otherwise FP32;
there is no gradient accumulation. Matrix-shaped EEG inputs are preloaded only
when live free memory leaves the configured headroom. Otherwise, contiguous
pinned CPU batches use non-blocking device copies.

The optimizer hot path has no per-batch host synchronization. Gradients are
set to `None` before forward, training loss is accumulated on device and read
once per epoch, and validation/inference logits return to CPU once per complete
pass. This follows the PyTorch performance guidance to avoid `.item()` and
`.cpu()` inside accelerator batch loops:
`https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html`.

LOSO fold execution accepts `--fold-jobs` and `--fold-backend`. CPU work may
use thread or process workers. GPU folds use spawned processes, never shared
training threads, so each worker owns its CUDA RNG and context. The automatic
GPU limit is one worker below 24 GiB total memory and two workers at or above
that threshold. `--gpu-fold-jobs N` explicitly raises or lowers it after a
CUDA smoke benchmark. The scheduler divides batch and preload headroom across
those workers. Process workers receive read-only EEG source arrays through
shared memory, then perform their fold-local preprocessing privately. Each
experiment record includes peak allocated/reserved memory, the effective
executor, and the input transport.

Fold-local QC is model-independent. A multi-model ablation fits every outer
fold's QC policy once in the runner and reuses the frozen thresholds for all
candidates. Recomputing QC per model is prohibited because it wastes host CPU
and can introduce accidental candidate-specific preprocessing drift.

`--cpu-threads` is a total host budget, not a per-worker number. The runtime
divides it across the effective fold workers and applies the result to PyTorch,
BLAS, and OpenMP pools. On a 16-vCPU host with two active folds this yields
eight CPU threads per worker, allowing one fold's CPU preprocessing to overlap
the other fold's GPU work without native thread oversubscription.
