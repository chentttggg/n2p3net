"""Archive six inactive derived pools exactly, then reclaim their external bytes."""
from __future__ import annotations

import gzip
import hashlib
import json
import stat
from pathlib import Path

_CACHE_NAMES = (
    "deployment_all12", "exclude_BI2014a", "exclude_BI2015a",
    "exclude_BI_related", "exclude_BNCI2014-008", "exclude_Huebner2017",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _compact_one(admission, inputs, output, name):
    build_path = inputs / (name + ".build.json")
    build_bytes = build_path.read_bytes()
    build = json.loads(build_bytes)
    original = inputs / (name + ".npz")
    before = original.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Inactive cache cleanup requires the exact original regular file")
    if Path(build["source_cache"]).resolve(strict=True) != original:
        raise ValueError("Historical cache build does not identify the exact cleanup target")
    if before.st_size != build["bytes"]:
        raise ValueError("Historical cache size differs from its original build")
    record_sha256 = sha(inputs / (name + ".record.json"))
    destination = output / name
    with admission.artifact_directory(destination, before.st_size + 64 * 2**20):
        destination.mkdir(exist_ok=False)
        archive = destination / (name + ".npz.gz")
        encoded_hash = hashlib.sha256()
        with original.open("rb") as source, archive.open("xb") as compressed:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=1,
                              fileobj=compressed, mtime=0) as encoder:
                for block in iter(lambda: source.read(2**20), b""):
                    encoded_hash.update(block)
                    encoder.write(block)
        if encoded_hash.hexdigest() != build["source_cache_sha256"]:
            raise ValueError("Inactive input differs from its original scientific cache identity")
        recovered = hashlib.sha256()
        recovered_bytes = 0
        with gzip.open(archive, "rb") as source:
            for block in iter(lambda: source.read(2**20), b""):
                recovered.update(block)
                recovered_bytes += len(block)
        if recovered.hexdigest() != encoded_hash.hexdigest() or recovered_bytes != before.st_size:
            raise ValueError("Lossless cache archive did not restore the exact original bytes")
        entry = {"original_path": str(original), "original_sha256": encoded_hash.hexdigest(),
                 "original_bytes": before.st_size, "archive_path": str(archive),
                 "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
                 "build_sha256": hashlib.sha256(build_bytes).hexdigest(),
                 "record_sha256": record_sha256, "exact_original_byte_restoration": True,
                 "original_removed": False}
        admission.write_json(destination / "receipt.json", entry)
        # The parent ledger rechecks identity and rejects any named-output
        # overlap. Its narrow external-file credit cannot double-charge leases.
        entry["removal"] = admission.remove_external_file(original, expected_stat=before)
        entry["original_removed"] = True
        admission.write_json(destination / "receipt.json", entry)
    return entry


def run(arguments):
    from research.source_execution import current_input_admission

    inputs = Path("/root/autodl-fs/n2p3-runs/unified12_20260918/pools_v2").resolve(strict=True)
    output = Path(arguments["output"])
    admission = current_input_admission()
    output.mkdir(exist_ok=False)
    entries = [_compact_one(admission, inputs, output, name) for name in _CACHE_NAMES]
    receipt = {"schema": "n2p3_inactive_cache_lossless_compaction/2", "status": "completed",
               "scope": "six inactive 20260918 uncompressed derived pools; source/models/raw unchanged",
               "retained_build_and_record_directory": str(inputs), "files": entries,
               "logical_bytes_saved": sum(row["original_bytes"] - row["archive_bytes"] for row in entries),
               "external_capacity_credited_bytes": sum(row["removal"]["credited_bytes"] for row in entries),
               "deferred_allocation_upper_bound_bytes": sum(
                   row["removal"]["deferred_allocation_upper_bound_bytes"] for row in entries)}
    admission.write_json(output / "summary/receipt.json", receipt)
    return {"status": "completed", "receipt": str(output / "summary/receipt.json"),
            "logical_bytes_saved": receipt["logical_bytes_saved"],
            "external_capacity_credited_bytes": receipt["external_capacity_credited_bytes"],
            "deferred_allocation_upper_bound_bytes": receipt["deferred_allocation_upper_bound_bytes"],
            "cache_count": len(entries)}
