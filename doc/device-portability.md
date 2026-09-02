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
views of the pinned CPU source use non-blocking copies, while randomly gathered
rows use a correctness-first blocking copy. A single reusable pinned staging
buffer is prohibited: without a CUDA event it can be overwritten while the
previous DMA still reads it. A future streaming prefetcher must use at least a
two-slot ring, a copy stream, and per-slot completion events.

`MatrixBatchSource.shuffled_batches` implements an optional one-shot epoch
shuffle: one device-side row permutation per epoch followed by contiguous
narrowed batches. Row order, label alignment, and loss accumulation are
identical to the per-batch `index_select` path. The feature reserves an extra
training-matrix copy, so it is opt-in through `--shuffle-each-epoch`. On the
matched RTX 5090 audit below it slowed a warmed compiled 30-epoch fold from
15.22 s to 17.70 s and raised peak allocation from 457 MiB to 859 MiB, so it
remains off by default. When enabled and admitted by the memory headroom check,
fold records report `shuffle_each_epoch=True`.

Precision, fused Adam, one-shot shuffle, and `torch.compile` are experiment
configuration, not hidden policy. Compile mode covers the complete training
step (forward, weighted loss, backward, and Adam update), rather than only the
model forward. The LOSO runner exposes
`--precision {auto,bf16,fp32}`, `--fused-adam`,
`--compile-mode {none,default,reduce-overhead,max-autotune}`, and
`--shuffle-each-epoch`. All supervised `DeepConfig` models default to fused
Adam plus `reduce-overhead`; `--no-fused-adam --compile-mode none` restores the
eager ablation. CUDA applies the requested settings. CPU and XPU automatically
fall back to non-fused eager execution, while runtime records retain both the
requested and effective values plus the fallback reason. Experiment manifests
retain the requested config and ordinary LOSO fold records retain the effective
config. SSL pretraining and the subject adapter use separate AdamW loops and are
outside this measured default contract.

The optimizer hot path has no per-batch host synchronization. Gradients are
set to `None` before forward, training loss is accumulated on device and read
once per epoch, and validation/inference logits return to CPU once per complete
pass. This follows the PyTorch performance guidance to avoid `.item()` and
`.cpu()` inside accelerator batch loops:
`https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html`.

LOSO fold execution accepts `--fold-jobs` and `--fold-backend`. CPU work may
use thread or process workers. GPU folds use spawned processes, never shared
training threads, so each worker owns its CUDA RNG and context. The mainline
requests four folds but leaves `--gpu-fold-jobs` unset: automatic policy uses
one GPU worker below 24 GiB and up to four at or above 24 GiB. An explicit
`--gpu-fold-jobs N` overrides it after a device-specific benchmark. The
scheduler divides batch and preload headroom across those workers. Process workers receive
read-only EEG source arrays through shared memory, then perform their
fold-local preprocessing privately. Each experiment record includes peak
allocated/reserved memory, the effective executor, the input transport, and
per-fold `preloaded` / `shuffle_each_epoch` / compile / optimizer flags.

## RTX 5090 Launch-Overhead Audit (2026-08-28)

The matched target was BI2014a fold 0, N2P3Net `ms_flatten`, BF16, physical
batch 512, Torch 2.8.0 + CUDA 12.8. The 8-epoch fit took 9.408 s eager and
9.426 s with fused Adam; a cold `reduce-overhead` fit took 21.771 s. With the
Inductor cache warm, a forced 30-epoch fit took 23.603 s eager versus 14.747 s
compiled (37.5% lower fit time).

The production-shaped four-process, four-fold cold-start comparison used the
same fold set and seed. Eager wall time was 41.343 s with per-fold fits of
34.820--35.869 s. Full-step `reduce-overhead` wall time was 38.956 s with fits
of 29.250--31.532 s; peak allocation rose from 469.0 to 524.6 MiB. This is a
5.8% end-to-end wall reduction and about a 12% fit-stage reduction.

The subsequent production-shaped 64-subject / 8-fold run used four GPU workers,
30 epochs, BF16, and batch 512. Eager execution completed 232 fold-epochs in
72.161 s (1110.879 ms per fit epoch). `reduce-overhead` plus fused Adam completed
240 fold-epochs in 46.875 s (634.655 ms per fit epoch), a 42.9% normalized
fit-epoch reduction and 35.0% lower wall time despite running eight extra
epochs. CUDA logs recorded 24 graph trees and zero graph skips. By explicit
project decision, this measured combination is now the CUDA default for all
supervised deep baselines. Full 64-fold metrics remain the scientific promotion
gate; the 8-fold result establishes the runtime default, not an accuracy claim.

Fold-local QC is model-independent. A multi-model ablation fits every outer
fold's QC policy once in the runner and reuses the frozen thresholds for all
candidates. Recomputing QC per model is prohibited because it wastes host CPU
and can introduce accidental candidate-specific preprocessing drift.

`--cpu-threads` is a total host budget, not a per-worker number. The runtime
divides it across the effective fold workers and applies the result to PyTorch,
BLAS, and OpenMP pools. On a 16-vCPU host with two active folds this yields
eight CPU threads per worker, allowing one fold's CPU preprocessing to overlap
the other fold's GPU work without native thread oversubscription.

Independent CPU task builders use the same `train.runtime.resolve_cpu_worker_plan`
policy rather than reading host `os.cpu_count()` directly. The plan intersects
task count, requested workers, cgroup-aware available threads, an optional total
thread budget, and an optional cap; `cpu_thread_budget` then bounds PyTorch,
BLAS, and OpenMP inside the executor. Every persisted builder records requested
and effective workers, total CPU budget, and threads per worker. Multi-source
cache loading follows this policy in parallel, while final concatenate/save
remains serial to bound large-array copies.
