"""Changed report shards, one atomic state index, and legacy JSON consumption.

The index is the current checkpoint. Its immutable numeric/report shards grant
no data authority. Successful finalization also materializes the original report
schema once for existing analysis tools. Historical JSON retains all producer
fields and needs no index or migration.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from research.execution import _atomic_write_stream, atomic_write_json

INDEX_SCHEMA = "n2p3_source_chain_report_index/1"
INDEX_NAME = "source_chain_state.json"
SHARD_DIRECTORY = "source_chain_state"
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_VERIFIED_WRITER_SHARDS = {}


def _version(path):
    information = path.stat()
    return (information.st_dev, information.st_ino, information.st_size,
        information.st_mtime_ns, information.st_ctime_ns)


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode("utf8")


def source_chain_report_reference(path):
    """Select the explicit current checkpoint, or the original standalone JSON."""
    path = Path(path)
    if path.name == INDEX_NAME:
        return path
    index = path.parent/INDEX_NAME
    if index.exists():
        record = json.loads(index.read_bytes())
        if record.get("schema") == INDEX_SCHEMA and record.get("compatibility_snapshot") == path.name:
            return index
    return path


def source_chain_report_exists(path):
    return source_chain_report_reference(path).exists()


def _read(path, verify):
    expected = verify(path) if verify is not None else None
    content = path.read_bytes()
    if expected is not None and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("source report dependency differs from the externally frozen inventory")
    return content


def load_source_chain_report(path, *, verify=None):
    """Cold read one complete checkpoint closure, optionally with external pins."""
    selected = source_chain_report_reference(path)
    content = _read(selected, verify)
    record = json.loads(content)
    if selected.name != INDEX_NAME:
        return record
    if (record.get("schema") != INDEX_SCHEMA or record.get("producer_fields_preserved") is not True
            or record.get("grants_data_authority") is not False):
        raise ValueError("unsupported current source report index schema")
    cache = {}

    def shard(reference):
        digest = reference["sha256"]
        if (len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest)
                or reference["file"] != f"{SHARD_DIRECTORY}/{digest}.json"):
            raise ValueError("source report shard path/content identity changed")
        if digest not in cache:
            raw = _read(selected.parent/reference["file"], verify)
            if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("source report immutable shard changed")
            cache[digest] = json.loads(raw)
        # Repeated equal values are independent report objects for later resume merging.
        return deepcopy(cache[digest])

    restored = {}
    for key, field in record["fields"].items():
        if field["kind"] == "items":
            restored[key] = [shard(reference) for reference in field["items"]]
        elif field["kind"] == "value":
            restored[key] = field["value"]
        elif field["kind"] == "shard":
            restored[key] = shard(field["reference"])
        else:
            raise ValueError("unsupported source report field encoding")
    if restored.get("schema") != record["report_schema"]:
        raise ValueError("source report original schema differs from its current index")
    return restored


def checkpoint_source_chain_report(path, report, *, materialize=False, terminal_state=None):
    """Publish only changed immutable payloads, then atomically switch the index.

    A crash before the index switch leaves the previous complete closure valid.
    Orphan shards are harmless and retained for recovery; no older evidence is
    overwritten. Payload writes scale with changed records; the small list of
    references still grows with the requested task count.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_directory = path.parent/SHARD_DIRECTORY
    shard_directory.mkdir(exist_ok=True)
    written_bytes, written_shards = 0, 0

    def put(value):
        nonlocal written_bytes, written_shards
        content = _encode(value)
        digest = hashlib.sha256(content).hexdigest()
        destination = shard_directory/(digest+".json")
        if destination.exists():
            version = _version(destination)
            if _VERIFIED_WRITER_SHARDS.get(str(destination)) != version:
                if destination.read_bytes() != content:
                    raise ValueError("existing immutable source report shard changed")
                _VERIFIED_WRITER_SHARDS[str(destination)] = version
        else:
            _atomic_write_stream(destination, lambda: iter((content,)), len(content))
            _VERIFIED_WRITER_SHARDS[str(destination)] = _version(destination)
            written_bytes += len(content)
            written_shards += 1
        return {"file": f"{SHARD_DIRECTORY}/{digest}.json", "bytes": len(content), "sha256": digest}

    fields = {}
    for key, value in report.items():
        if isinstance(value, (list, tuple)):
            fields[key] = {"kind": "items", "items": [put(item) for item in value]}
        elif len(_encode(value)) <= 1024:
            fields[key] = {"kind": "value", "value": value}
        else:
            fields[key] = {"kind": "shard", "reference": put(value)}
    state = terminal_state or report.get("status") or ("completed" if materialize else "partial")
    index = {"schema": INDEX_SCHEMA, "report_schema": report.get("schema"), "state": state,
        "fields": fields, "compatibility_snapshot": path.name,
        "codec_producer_sha256": IMPLEMENTATION_SHA256,
        "producer_fields_preserved": True, "grants_data_authority": False}
    encoded_index = _encode(index)
    current_path = path.parent/INDEX_NAME
    changed_index = not current_path.exists() or current_path.read_bytes() != encoded_index
    if changed_index:
        _atomic_write_stream(current_path, lambda: iter((encoded_index,)), len(encoded_index))
        written_bytes += len(encoded_index)
    if materialize:
        atomic_write_json(path, report)
    return {"written_payload_shards": written_shards, "written_checkpoint_bytes": written_bytes,
        "index_changed": changed_index, "compatibility_snapshot_materialized": materialize,
        "state": state}
