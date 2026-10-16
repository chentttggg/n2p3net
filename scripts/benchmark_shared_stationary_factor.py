"""Measure the explicit factor producer against one read-only frozen noise law.

This is a local numerical benchmark with no EEG/source/target fitting. Native
threads are pinned before importing the scientific stack. The original dense
root is read from the existing codec; this command never regenerates it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

from research import channel_noise_mechanism as owner  # noqa: E402
from research.shared_stationary_factor import compile_shared_stationary_root  # noqa: E402
from research.stationary_noise_artifact import stationary_root  # noqa: E402
from transfer.noise_model import array_digest, metadata_digest  # noqa: E402
from transfer.temporal_noise import TemporalNoise  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("benchmark output must be new")
    process = psutil.Process()
    started = time.perf_counter()
    before = {"available_bytes": psutil.virtual_memory().available,
              "rss_bytes": process.memory_info().rss, "affinity_logical": len(process.cpu_affinity())}
    raw = args.frozen_source.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    if original_sha != args.source_sha256:
        raise ValueError("frozen original source differs from the declared byte identity")
    original = json.loads(raw)["noise"]
    del raw
    if original["schema"] != "n2p3_channel_noise_mechanism/3" or original["family"] != "shared":
        raise ValueError("benchmark requires an original shared /3 source codec")
    prefix = original["conditioning_samples"]
    geometry = owner.NoiseGeometry(**original["geometry"])
    temporal = TemporalNoise.from_record(original["compiled_temporal_records"][0])
    reflections = np.asarray(original["reflections"])
    if any(not np.array_equal(row, temporal.reflection_coefficients) for row in reflections):
        raise ValueError("saved source is not the identical stored PARCOR law")
    sigma = np.asarray(original["innovation_covariance"])
    old_root = np.asarray(original["compiled_prefix_root"])
    timings = {"read_and_admit_original_temporal_seconds": time.perf_counter()-started}
    calls = {"compile": 0, "admit": 0}
    original_compile, original_admit = owner.compile_shared_stationary_root, owner.admit_shared_stationary_root

    def compile_timed(*args, **kwargs):
        tick = time.perf_counter()
        result = original_compile(*args, **kwargs)
        timings["compile_seconds"] = time.perf_counter()-tick
        calls["compile"] += 1
        return result

    def admit_timed(*args, **kwargs):
        tick = time.perf_counter()
        result = original_admit(*args, **kwargs)
        timings.setdefault("admissions_seconds", []).append(time.perf_counter()-tick)
        calls["admit"] += 1
        return result

    owner.compile_shared_stationary_root, owner.admit_shared_stationary_root = compile_timed, admit_timed
    args.output.mkdir(parents=True)
    with threadpool_limits(limits=1):
        tick = time.perf_counter()
        fitted = owner.FittedChannelNoise.from_shared_stationary_factors(
            (temporal,)*len(geometry.channels), sigma, geometry=geometry,
            conditioning_samples=prefix, provenance=original["provenance"])
        timings["factory_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        fitted.save(args.output/"new_factor_noise.json")
        timings["save_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        restored = owner.FittedChannelNoise.load(args.output/"new_factor_noise.json")
        timings["load_seconds"] = time.perf_counter()-tick
        if restored.record() != fitted.record():
            raise AssertionError("saved original factor state did not restore exactly")
        samples = np.random.default_rng(20261016).normal(size=(prefix+8, len(geometry.channels)))*1e-6
        white = fitted.whiten(samples)
        if not np.array_equal(white, restored.whiten(samples)):
            raise AssertionError("saved factor root changed actual whitening")
        old_prefix_white = solve_triangular(old_root, samples[:prefix].reshape(-1), lower=True)
        new_prefix_white = white[:prefix].reshape(-1)
        # A same-process/same-input small dense baseline complements the
        # historical large dense observation without repeating that large fit.
        small_prefix, small_channels = 8, min(4, len(geometry.channels))
        small_sigma = sigma[:small_channels, :small_channels]
        small_reflections = temporal.reflection_coefficients
        small_runs = []
        stationary_root([small_reflections]*small_channels, small_sigma, small_prefix)
        compile_shared_stationary_root(small_reflections, small_sigma, small_prefix)
        for _ in range(3):
            tick = time.perf_counter()
            dense = stationary_root([small_reflections]*small_channels, small_sigma, small_prefix)
            dense_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            small_root, _ = compile_shared_stationary_root(small_reflections, small_sigma, small_prefix)
            small_runs.append({"dense_seconds": dense_seconds, "factor_seconds": time.perf_counter()-tick})
        small_baseline = {"prefix": small_prefix, "channels": small_channels, "warmup_each": 1,
            "runs": small_runs, "root_max_abs_difference": float(np.max(np.abs(dense-small_root)))}
    if _sha(args.frozen_source) != original_sha:
        raise AssertionError("read-only original source bytes changed during benchmark")
    repository = Path(__file__).resolve().parents[1]
    source_files = ["scripts/benchmark_shared_stationary_factor.py", "src/research/channel_noise_mechanism.py",
        "src/research/shared_stationary_factor.py", "src/research/stationary_noise_artifact.py",
        "src/transfer/temporal_noise.py", "src/transfer/dictionary_artifact.py"]
    memory = process.memory_info()
    receipt = {"schema": "n2p3_shared_stationary_factor_benchmark/1", "scope": "local_numeric_no_fits",
        "original_source_sha256": original_sha, "original_source_bytes": args.frozen_source.stat().st_size,
        "original_noise_state_sha256": metadata_digest(original),
        "new_noise_state_sha256": metadata_digest(fitted.record()),
        "original_dense_root_sha256": array_digest(old_root),
        "new_factor_root_sha256": array_digest(fitted.compiled_prefix_root),
        "original_codec": original["schema"], "new_codec": fitted.record()["schema"],
        "channels": len(geometry.channels), "prefix_samples": prefix, "joint_rows": prefix*len(geometry.channels),
        "original_parameters_unchanged": array_digest(sigma, reflections) == array_digest(
            fitted.innovation_covariance, np.asarray([v.reflection_coefficients for v in fitted.temporal])),
        "fit_calls": 0, "root_calls": calls, "timings": timings,
        "same_process_small_baseline": small_baseline,
        "certificate": fitted.shared_factor_state["certificate"],
        "saved_original_state_equal": restored.record() == fitted.record(), "whitening_replay_bitwise_equal": True,
        "old_new_root_max_abs_difference": float(np.max(np.abs(old_root-fitted.compiled_prefix_root))),
        "old_new_prefix_whitening_max_abs_difference": float(np.max(np.abs(old_prefix_white-new_prefix_white))),
        "old_new_startup_logdet_difference": float(np.log(np.diag(old_root)).sum()
            -np.log(np.diag(fitted.compiled_prefix_root)).sum()),
        "resource_before": before, "resource_after": {"available_bytes": psutil.virtual_memory().available,
            "rss_bytes": memory.rss, "peak_rss_bytes": getattr(memory, "peak_wset", None),
            "cpu_seconds": process.cpu_times().user+process.cpu_times().system},
        "threadpools": threadpool_info(), "native_threads": 1, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "source_files_sha256": {name: _sha(repository/name) for name in source_files},
        "new_noise_file_sha256": _sha(args.output/"new_factor_noise.json")}
    (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
