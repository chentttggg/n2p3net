"""Shared byte digests; persisted callers retain their exact encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path) -> str:
    """Stream a file's bytes without loading the complete artifact."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(value: np.ndarray) -> str:
    """Retain the source-selection and alignment dtype/shape/bytes encoding."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def compact_array_digest(value: np.ndarray) -> str:
    """Retain the exact compact header used by EA and matched binary inputs."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
