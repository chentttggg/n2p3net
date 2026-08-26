"""Benchmark Neural-RIDE GPU scheduling without changing model semantics.

The matrix compares the exact fused and legacy tokenizer paths across eager and
torch.compile modes. Each timed iteration executes Trainer._train_step, including
AMP, the production loss graph, backward, and fused AdamW on CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

if __name__ == "__main__" and sys.platform == "win32" and not sys.flags.utf8_mode:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=env, check=False)
    raise SystemExit(completed.returncode)

import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.n2p3net import N2P3Net  # noqa: E402
from train.contracts import TrialContext  # noqa: E402
from train.device import optimize_device_for_training  # noqa: E402
from train.recipe import NEURAL_RIDE_V12  # noqa: E402
from train.trainer import COMPILE_MODES, Trainer, TrainerConfig  # noqa: E402

TOKENIZER_PATHS = ("legacy", "fused")
POINTWISE_PATHS = ("conv1d", "linear")
CHANNEL_NAMES = {
    3: ("Fz", "Cz", "Pz"),
    8: ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"),
}


def _csv_choices(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = sorted(set(values) - set(allowed))
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated subset of {allowed}; invalid={invalid}"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--channels", type=int, choices=tuple(CHANNEL_NAMES), default=8)
    parser.add_argument("--times", type=int, default=256)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--tmin-ms", type=float, default=-200.0)
    parser.add_argument("--tmax-ms", type=float, default=800.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument(
        "--compile-modes",
        default=",".join(COMPILE_MODES),
        help=f"comma-separated subset of {COMPILE_MODES}",
    )
    parser.add_argument(
        "--tokenizer-paths",
        default=",".join(TOKENIZER_PATHS),
        help=f"comma-separated subset of {TOKENIZER_PATHS}",
    )
    parser.add_argument(
        "--pointwise-paths",
        default=",".join(POINTWISE_PATHS),
        help=f"comma-separated subset of {POINTWISE_PATHS}",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collect module scopes and top CUDA operators for eager variants",
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "tmp" / "gpu_schedule_benchmark.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _build_trainer(
    args: argparse.Namespace,
    tokenizer_path: str,
    pointwise_path: str,
    compile_mode: str,
) -> Trainer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = N2P3Net(
        n_channels=args.channels,
        channel_names=CHANNEL_NAMES[args.channels],
        d_model=NEURAL_RIDE_V12.d_model,
        tmin_ms=args.tmin_ms,
        tmax_ms=args.tmax_ms,
        sfreq=args.sfreq,
        n_time=args.times,
        temporal_kernels=(13, 33, 65, 129),
        filters_per_scale=16,
        encoder_depth=NEURAL_RIDE_V12.encoder_depth,
        encoder_type=NEURAL_RIDE_V12.encoder_type,
        encoder_norm=NEURAL_RIDE_V12.encoder_norm,
        encoder_dropout=NEURAL_RIDE_V12.encoder_dropout,
        tcn_pointwise_execution=pointwise_path,
        tokenizer_init=NEURAL_RIDE_V12.tokenizer_init,
        tokenizer_post_norm=NEURAL_RIDE_V12.tokenizer_post_norm,
        tokenizer_post_act=NEURAL_RIDE_V12.tokenizer_post_act,
        tokenizer_temporal_spatial_fusion=tokenizer_path == "fused",
        head_dropout=NEURAL_RIDE_V12.head_dropout,
        use_rereference=NEURAL_RIDE_V12.use_rereference,
        component_decoder=False,
        use_innovation_likelihood=False,
        use_repetition_evidence=False,
    )
    config = TrainerConfig(
        epochs=1,
        batch_size=args.batch_size,
        compile_mode=compile_mode,
        lr=NEURAL_RIDE_V12.lr,
        weight_decay=NEURAL_RIDE_V12.weight_decay,
        lambda2=NEURAL_RIDE_V12.lambda2,
        lambda3=NEURAL_RIDE_V12.lambda3,
        lambda_pcw=NEURAL_RIDE_V12.lambda_pcw,
        lambda_digit=0.0,
        lambda_conditional_nll=0.0,
        lambda_amp=0.0,
        lambda_recon=0.0,
        lambda_innovation=0.0,
        augment=False,
        auto_pos_weight=False,
        track_pcw_gradients=False,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    E_chn = torch.randn(args.channels, 48, generator=generator)
    channel_mask = torch.ones(args.channels, dtype=torch.bool)
    return Trainer(
        model,
        config,
        E_chn=E_chn,
        channel_mask=channel_mask,
        device=torch.device("cuda"),
    )


def _build_context(args: argparse.Namespace) -> TrialContext:
    generator = torch.Generator().manual_seed(args.seed + 2)
    X = torch.randn(args.batch_size, args.channels, args.times, generator=generator)
    y = torch.randint(0, 2, (args.batch_size,), generator=generator).float()
    return TrialContext(X=X.cuda(), y=y.cuda())


def _timed_step(trainer: Trainer, context: TrialContext, step: int) -> torch.Tensor:
    return trainer._train_step(context, step)


def _event_value(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _profile_step(
    trainer: Trainer,
    context: TrialContext,
    *,
    steps: int,
    trace_path: Path | None,
) -> dict[str, Any]:
    scopes: dict[int, Any] = {}
    handles = []

    def add_scope(name: str, module: torch.nn.Module) -> None:
        def pre_hook(current: torch.nn.Module, _inputs) -> None:
            scope = torch.profiler.record_function(f"module::{name}")
            scopes[id(current)] = scope
            scope.__enter__()

        def post_hook(current: torch.nn.Module, _inputs, _output) -> None:
            scope = scopes.pop(id(current))
            scope.__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    add_scope("tokenizer", trainer.model.tokenizer)
    add_scope("encoder", trainer.model.encoder)
    add_scope("component_window", trainer.model.component_window)
    add_scope("heads", trainer.model.heads)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            for step in range(steps):
                _timed_step(trainer, context, step)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(trace_path))

    rows = []
    modules: dict[str, dict[str, float]] = {}
    for event in profile.key_averages():
        device_total_us = _event_value(event, "device_time_total", "cuda_time_total")
        self_device_us = _event_value(event, "self_device_time_total", "self_cuda_time_total")
        row = {
            "name": event.key,
            "calls": int(event.count),
            "device_total_ms": device_total_us / 1000.0,
            "self_device_ms": self_device_us / 1000.0,
            "self_device_memory_mb": _event_value(
                event, "self_device_memory_usage", "self_cuda_memory_usage"
            )
            / 1e6,
        }
        rows.append(row)
        if event.key.startswith("module::"):
            modules[event.key.removeprefix("module::")] = row
    rows.sort(key=lambda item: item["self_device_ms"], reverse=True)
    return {"modules": modules, "top_cuda_operators": rows[:25], "trace": str(trace_path) if trace_path else None}


def _benchmark_variant(
    args: argparse.Namespace,
    tokenizer_path: str,
    pointwise_path: str,
    compile_mode: str,
) -> dict[str, Any]:
    if compile_mode != "eager" and hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    torch.cuda.empty_cache()
    trainer = _build_trainer(args, tokenizer_path, pointwise_path, compile_mode)
    context = _build_context(args)

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    loss = _timed_step(trainer, context, 0)
    torch.cuda.synchronize()
    cold_start_seconds = time.perf_counter() - cold_start

    for step in range(args.warmup_steps):
        loss = _timed_step(trainer, context, step + 1)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for step in range(args.steps):
        loss = _timed_step(trainer, context, step + args.warmup_steps + 1)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6

    profile = None
    if args.profile and compile_mode == "eager":
        trace_path = (
            args.trace_dir / f"{tokenizer_path}_{pointwise_path}_{compile_mode}.json"
            if args.trace_dir is not None
            else None
        )
        profile = _profile_step(
            trainer,
            context,
            steps=args.profile_steps,
            trace_path=trace_path,
        )

    return {
        "tokenizer_path": tokenizer_path,
        "pointwise_path": pointwise_path,
        "compile_mode": compile_mode,
        "fusion_active": bool(trainer.model.tokenizer.uses_fused_temporal_spatial),
        "cold_start_seconds": cold_start_seconds,
        "steady_total_ms": elapsed_ms,
        "steady_step_ms": elapsed_ms / args.steps,
        "steady_samples_per_second": args.batch_size * args.steps / (elapsed_ms / 1000.0),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "final_loss": float(loss),
        "profile": profile,
    }


def main() -> None:
    args = _parser().parse_args()
    args.compile_modes = _csv_choices(args.compile_modes, COMPILE_MODES, "compile modes")
    args.tokenizer_paths = _csv_choices(
        args.tokenizer_paths, TOKENIZER_PATHS, "tokenizer paths"
    )
    args.pointwise_paths = _csv_choices(
        args.pointwise_paths, POINTWISE_PATHS, "pointwise paths"
    )
    if min(args.batch_size, args.times, args.warmup_steps, args.steps, args.profile_steps) < 1:
        raise SystemExit("batch/times/warmup/steps/profile-steps must all be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this scheduling benchmark")

    optimize_device_for_training(torch.device("cuda"))
    properties = torch.cuda.get_device_properties(0)
    record: dict[str, Any] = {
        "schema": "n2p3net_gpu_schedule/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mb": properties.total_memory / 1e6,
            "bf16": torch.cuda.is_bf16_supported(),
        },
        "shape": {
            "batch": args.batch_size,
            "channels": args.channels,
            "times": args.times,
            "d_model": NEURAL_RIDE_V12.d_model,
        },
        "settings": {
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "profile_steps": args.profile_steps,
            "compile_modes": args.compile_modes,
            "tokenizer_paths": args.tokenizer_paths,
            "pointwise_paths": args.pointwise_paths,
        },
        "variants": [],
    }

    for tokenizer_path in args.tokenizer_paths:
        for pointwise_path in args.pointwise_paths:
            for compile_mode in args.compile_modes:
                label = f"{tokenizer_path}/{pointwise_path}/{compile_mode}"
                print(f"[benchmark] {label}", flush=True)
                try:
                    result = _benchmark_variant(
                        args, tokenizer_path, pointwise_path, compile_mode
                    )
                    print(
                        f"  {result['steady_step_ms']:.3f} ms/step, "
                        f"{result['steady_samples_per_second']:.1f} samples/s, "
                        f"peak={result['peak_allocated_mb']:.1f} MB, "
                        f"cold={result['cold_start_seconds']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "tokenizer_path": tokenizer_path,
                        "pointwise_path": pointwise_path,
                        "compile_mode": compile_mode,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    print(f"  failed: {result['error']}", flush=True)
                record["variants"].append(result)
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[record] {args.output.resolve()}")


if __name__ == "__main__":
    main()
