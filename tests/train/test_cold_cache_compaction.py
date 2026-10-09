"""The fixed old-cache adapter deletes only after exact archive restoration."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import compact_cold_caches
from research import source_execution
from research.source_execution import SourceInputAdmission


def _inactive_pool(tmp_path, monkeypatch, *, freed_bytes=8192):
    inputs, output = tmp_path / "old", tmp_path / "archive"
    inputs.mkdir()
    output.mkdir()
    name = "deployment_all12"
    original = inputs / (name + ".npz")
    payload = b"immutable original derived cache bytes" * 4096
    original.write_bytes(payload)
    original_lstat, original_unlink = Path.lstat, Path.unlink
    free = {"bytes": 256 * 2**20}

    def lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path != original:
            return info
        values = {key: getattr(info, key) for key in (
            "st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")}
        return SimpleNamespace(**values, st_blocks=16)

    def unlink(path, *args, **kwargs):
        original_unlink(path, *args, **kwargs)
        if path == original:
            free["bytes"] += freed_bytes

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(source_execution, "filesystem_uses_host_memory", lambda path: False)
    monkeypatch.setattr(source_execution.shutil, "disk_usage", lambda path:
        SimpleNamespace(total=2**30, used=2**30 - free["bytes"], free=free["bytes"]))
    build = {"source_cache": str(original), "bytes": len(payload),
             "source_cache_sha256": hashlib.sha256(payload).hexdigest()}
    build_path = inputs / (name + ".build.json")
    build_path.write_text(json.dumps(build), encoding="utf-8")
    record_path = inputs / (name + ".record.json")
    record_path.write_bytes(b'{"original_producer":"unchanged"}')
    owner = SourceInputAdmission(SimpleNamespace(SourceInputState=source_execution._SourceInputState),
                                 capacity_bytes=0, load_workers=1)
    return owner, inputs, output, name, original, payload, build_path, record_path


@pytest.mark.parametrize("freed_bytes", [8192, 0])
def test_exact_archive_is_retained_and_deletion_credits_the_existing_owner(tmp_path, monkeypatch, freed_bytes):
    owner, inputs, output, name, original, payload, build_path, record_path = _inactive_pool(
        tmp_path, monkeypatch, freed_bytes=freed_bytes)
    original_build, original_record = build_path.read_bytes(), record_path.read_bytes()
    entry = compact_cold_caches._compact_one(owner, inputs, output, name)
    archive = Path(entry["archive_path"])
    assert gzip.decompress(archive.read_bytes()) == payload
    assert not original.exists() and entry["original_removed"]
    assert build_path.read_bytes() == original_build and record_path.read_bytes() == original_record
    assert entry["removal"]["credited_bytes"] == freed_bytes
    assert entry["removal"]["deferred_allocation_upper_bound_bytes"] == 8192 - freed_bytes
    state = owner.record()
    assert state["external_file_reclaims"] == [entry["removal"]]
    lease = state["active_leases"][0]
    assert lease["committed"] and lease["disk_bytes"] == sum(
        path.stat().st_size for path in (archive, output / name / "receipt.json"))
    assert json.loads((output / name / "receipt.json").read_bytes()) == entry


@pytest.mark.parametrize("failure", ["original_sha", "restoration", "changed_after_verification"])
def test_failed_archival_preserves_original_and_charges_surviving_partial(tmp_path, monkeypatch, failure):
    owner, inputs, output, name, original, payload, build_path, _ = _inactive_pool(tmp_path, monkeypatch)
    if failure == "original_sha":
        build = json.loads(build_path.read_bytes())
        build["source_cache_sha256"] = "0" * 64
        build_path.write_text(json.dumps(build), encoding="utf-8")
        expected_error = "scientific cache identity"
    elif failure == "restoration":
        monkeypatch.setattr(compact_cold_caches.gzip, "open", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
        expected_error = "exact original bytes"
    else:
        write_json = owner.write_json

        def mutate_after_receipt(path, entry):
            write_json(path, entry)
            original.write_bytes(b"X" * len(payload))

        monkeypatch.setattr(owner, "write_json", mutate_after_receipt)
        expected_error = "identity changed"
    with pytest.raises(ValueError, match=expected_error):
        compact_cold_caches._compact_one(owner, inputs, output, name)
    assert original.exists() and not owner.record()["external_file_reclaims"]
    archive = output / name / (name + ".npz.gz")
    assert archive.exists()
    lease = owner.record()["active_leases"][0]
    assert lease["committed"] and lease["disk_bytes"] >= archive.stat().st_size > 0
