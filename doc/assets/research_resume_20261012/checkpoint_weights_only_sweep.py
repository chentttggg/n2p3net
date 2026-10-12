"""Read-only sweep: do real torch artifacts load identically with weights_only=True?

Scratch evidence for replacing weights_only=False loaders. Never writes next to
artifacts; only writes the JSON report given on the command line.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT_SUBDIRS = ("experiments/runs", "experiments/results_history", "experiments/cache", "cloud_inputs")
MAIN = Path("D:/BCI/n2p3-net")


def _roots():
    listing = subprocess.run(["git", "-C", str(MAIN), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
    trees = [Path(line[9:]) for line in listing.splitlines() if line.startswith("worktree ")]
    roots = []
    for tree in trees:
        for sub in ROOT_SUBDIRS:
            candidate = tree / sub
            if candidate.is_dir():
                roots.append(candidate)
    return roots


def _allowlist(np):
    # Exactly the allowlist of transfer.calibrated._load_calibration_checkpoint.
    return [np.str_("").__reduce__()[0], np.dtype, type(np.dtype("U"))]


def _extended(np):
    extra = []
    for expr in ("np._core.multiarray._reconstruct", "np.ndarray", "np._core.multiarray.scalar"):
        try:
            extra.append(eval(expr, {"np": np}))  # noqa: S307 - fixed probe expressions
        except AttributeError:
            pass
    for name in dir(np.dtypes):
        value = getattr(np.dtypes, name)
        if isinstance(value, type):
            extra.append(value)
    return extra


def _compare(a, b, path, out, torch, np):
    if len(out) > 20:
        return
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            out.append(f"{path}: tensor differs")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(
            a, b, equal_nan=a.dtype.kind in "fc"
        ):
            out.append(f"{path}: ndarray differs")
    elif isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            out.append(f"{path}: keys differ")
        for key in a:
            if key in b:
                _compare(a[key], b[key], f"{path}/{key!s}", out, torch, np)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length differs")
        for index, (x, y) in enumerate(zip(a, b, strict=False)):
            _compare(x, y, f"{path}[{index}]", out, torch, np)
    elif isinstance(a, float) and a != a:
        if not (isinstance(b, float) and b != b):
            out.append(f"{path}: nan differs")
    elif a != b:
        out.append(f"{path}: value differs")


def _missing_globals(message):
    return sorted(set(re.findall(r"GLOBAL ([\w.]+)", message)))


def _check(item):
    import numpy as np
    import torch

    path, digest = item
    content = Path(path).read_bytes()
    result = {"path": path, "sha256": digest, "bytes": len(content)}
    try:
        unsafe = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - classify unreadable artifacts
        result.update(kind="unreadable", error=f"{type(error).__name__}: {error}"[:300])
        return result
    schema = unsafe.get("schema") if isinstance(unsafe, dict) else None
    if schema == "n2p3_transfer_checkpoint/2":
        kind = "transfer_checkpoint"
    elif isinstance(unsafe, dict) and {"state", "recipe"} <= set(unsafe):
        kind = "target_model"
    else:
        kind = "other"
    result.update(kind=kind, schema=schema, top_type=type(unsafe).__name__,
                  keys=sorted(map(str, unsafe))[:40] if isinstance(unsafe, dict) else None)
    for label, allow in (("baseline", _allowlist(np)), ("extended", _allowlist(np) + _extended(np))):
        try:
            with torch.serialization.safe_globals(allow):
                safe = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        except Exception as error:  # noqa: BLE001
            message = str(error)
            result[f"{label}_ok"] = False
            result[f"{label}_missing"] = _missing_globals(message) or [message[:300]]
            continue
        mismatches = []
        _compare(unsafe, safe, "", mismatches, torch, np)
        result[f"{label}_ok"] = not mismatches
        result[f"{label}_mismatch"] = mismatches
        if label == "baseline" and not mismatches:
            result["extended_ok"] = True
            break
    return result


def main():
    started = time.perf_counter()
    files = []
    for root in _roots():
        for pattern in ("*.pt", "*.pth"):
            files.extend(str(path) for path in root.rglob(pattern) if path.is_file())
    files = sorted(set(files))
    by_digest = {}
    total_bytes = 0
    for name in files:
        with open(name, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        total_bytes += os.path.getsize(name)
        by_digest.setdefault(digest, []).append(name)
    items = [(paths[0], digest) for digest, paths in by_digest.items()]
    print(json.dumps({"enumerated": len(files), "unique": len(items)}), flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_check, items, chunksize=8))
    for result in results:
        result["copies"] = len(by_digest[result["sha256"]])
    summary = {
        "files": len(files), "unique": len(items), "bytes": total_bytes,
        "seconds": round(time.perf_counter() - started, 1),
        "python": sys.version.split()[0],
    }
    Path(sys.argv[1]).write_text(json.dumps({"summary": summary, "results": results}, indent=1),
                                 encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
