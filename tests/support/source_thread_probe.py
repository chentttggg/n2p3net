"""Tiny CUDA kernel behind the actual source-worker entry for resource probes."""

import json
import os
from pathlib import Path


def cuda_source_thread_actor(arguments):
    import psutil
    import torch

    from experiments import run_source_training_chain as runner
    from research.evaluation import verified_executed_snapshot

    plan, output, barrier, admission = arguments
    original = runner.execute_source_chain

    def tiny_source_kernel(current, *, output, **options):
        native, interop = torch.get_num_threads(), torch.get_num_interop_threads()
        assert (native, interop) == (5, 1)
        # First tensor work uses the admitted CUDA source phase, with no EEG.
        values = torch.arange(16, device=options["device"], dtype=torch.float32)
        assert values.square().sum().item() == 1240.
        barrier.wait(timeout=120)
        live = admission.record()
        assert live["reserved_cpu_threads"] == 25 and live["active_gpu_groups"] == live["source_groups"] == 4
        receipt = {"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                   "native_before_first_tensor": native, "interop_before_first_tensor": interop,
                   "child_native": current["_native_threads_per_worker"],
                   "child_interop": current["_source_interop_threads_per_worker"],
                   "tensor_device": str(values.device), "source_CPU_claim": current["_source_cpu_threads_per_worker"],
                   "live_actor_RSS_bytes": psutil.Process().memory_info().rss, "owner_live": live,
                   "public_EEG_rows": 0, "source_fitting": False}
        Path(output).joinpath("source_thread_kernel_probe.json").write_text(json.dumps(receipt), encoding="utf-8")
        snapshot = verified_executed_snapshot(current["source_snapshot_manifest"], root=runner.ROOT)
        Path(output).joinpath("source_chain_record.json").write_text(json.dumps({"source_snapshot": snapshot}), encoding="utf-8")
        barrier.wait(timeout=120)

    runner.execute_source_chain = tiny_source_kernel
    try:
        Path(output).mkdir(parents=True)
        return runner._source_group_worker((plan, output, "cuda:0", ("P2_source",)))
    finally:
        runner.execute_source_chain = original
