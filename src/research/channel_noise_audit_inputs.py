"""Shared verified paired-input seam for CLI audits and installed consumers."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from research.contracts import semantic_sha256
from transfer.ensemble_views import prepare_bnci_calibration_groups
from transfer.noise_model import require_sha256


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BoundInput:
    """Verify the physical file once and return copied numeric arrays, no pickle."""

    def __init__(self, manifest):
        self.root = manifest.parent.resolve()
        self.files = {}
        self.cache = {}

    def file(self, descriptor):
        if not isinstance(descriptor, dict) or not {"path", "sha256"} <= descriptor.keys():
            raise ValueError("input file descriptor requires a path and actual SHA-256")
        require_sha256(descriptor["sha256"], "input file")
        path = (self.root / descriptor["path"]).resolve()
        if path not in self.files:
            self.files[path] = _sha(path)
        if self.files[path] != descriptor["sha256"]:
            raise ValueError(f"input byte identity differs: {path}")
        return path

    def arrays(self, descriptor, *names):
        path = self.file(descriptor)
        if path not in self.cache:
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != self.files[path]:
                raise ValueError("input bytes changed before numeric decoding")
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                self.cache[path] = {key: np.asarray(archive[key]).copy() for key in archive.files}
        values = []
        for name in names:
            key = descriptor[f"{name}_key"]
            value = self.cache[path][key]
            if value.dtype.kind not in "fiu" and not (name in {"labels", "membership"} and value.dtype.kind == "b"):
                raise ValueError(f"NPZ array {key} must contain actual real numeric values")
            values.append(value)
        return tuple(values)

    def verify_unchanged(self):
        for path, digest in self.files.items():
            if _sha(path) != digest:
                raise ValueError(f"study input changed during execution: {path}")


def paired_input_digests(views):
    """Bind the actual shared owner records, not a success-only result roster."""
    return (semantic_sha256([view.raw_input_digest for view in views]),
            semantic_sha256([view.physical_record() for view in views]))


def public_calibration_authority(source_records, inputs):
    """Actual source/C labels and identities, independent of admitted successes."""
    authority = {}
    if not source_records:
        raise ValueError("public paired input requires original MAT owner reconstruction")
    for source in source_records:
        path = inputs.file(source["source"])
        groups = prepare_bnci_calibration_groups(path, source_sha256=source["source"]["sha256"], **source["owner"])
        for group in groups.groups:
            if group.view.record_id in authority:
                raise ValueError("paired original source roster repeats a physical group")
            authority[group.view.record_id] = group
    return authority
