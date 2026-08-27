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
