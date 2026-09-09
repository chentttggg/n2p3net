"""Bounded CUDA and AOT-autograd checks for the repaired source training loop."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from research.source_execution import release_source_worker  # noqa: E402


def main():
    torch.set_num_threads(1)
    X = np.random.default_rng(4).normal(size=(32, 3, 180)).astype(np.float32)
    y = np.tile([0, 1], 16)
    original_compile = torch.compile
    results = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "triton_available": importlib.util.find_spec("triton") is not None,
               "scope": "software_validation_not_throughput_or_population_evidence", "checks": []}

    def aot_compile(function, **kwargs):
        kwargs.pop("mode", None)
        return original_compile(function, backend="aot_eager", **kwargs)

    for backend in ("eager", "aot_eager"):
        config = DeepConfig(fixed_updates=2, snapshot_updates=(1, 2), batch_size=32,
                            sampling="uniform_rows", compile_mode=None if backend == "eager" else "default",
                            val_group_frac=None, precision="bf16", fused_adam=True)
        model = N2P3NetBaseline(3, 180, 128, config=config, device=torch.device("cuda"))
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with patch.object(torch, "compile", aot_compile):
                model.fit(X, y)
            scores = model.predict_logit(X)
            torch.cuda.synchronize()
            assert np.isfinite(scores).all()
            assert model.last_history["optimizer_updates"] == 2
            assert sorted(model.candidate_snapshots_) == [1, 2]
            results["checks"].append({"backend": backend, "status": "passed",
                                      "seconds_including_cold_start": time.perf_counter() - started,
                                      "precision": model.last_runtime["precision"],
                                      "fused_adam": model.optimizer_execution.fused_adam,
                                      "optimizer_updates": 2})
        finally:
            del model
            release_source_worker("cuda")
    Path(__file__).with_name("cuda-results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
