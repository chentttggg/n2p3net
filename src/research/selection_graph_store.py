"""Complete logical selection graphs with content-addressed immutable scopes.

The physical store has its own byte seals. Expanded graph/selection SHA values
retain the historical scientific record exactly, including every requested row.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from research.contracts import semantic_sha256
from research.execution import _atomic_write_stream, atomic_write_json_stream
from research.frozen_metadata import FrozenMetadata, freeze_metadata

SCHEMA = "n2p3_selection_graph_store/2"
SCOPE_REFERENCE = "$n2p3_scope"


def _scope(value):
    return isinstance(value, Mapping) and set(value) == {"actual_K", "input"} and isinstance(value["input"], Mapping)


def _artifact(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def _pairs_owner():
    pool = {}

    def pairs(items):
        record = dict(items)
        if len(record) != len(items):
            raise ValueError("selection graph repeats a JSON mapping key")
        value = FrozenMetadata(record)
        if _scope(value):
            return pool.setdefault(semantic_sha256(value), value)
        return value

    return pairs


def _decode(content):
    return json.loads(content, object_pairs_hook=_pairs_owner())


def save_selection_graph(path, graph, *, input_admission=None):
    """Publish complete scopes once without expanding their repeated placements."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"selection graph already exists: {path}")
    graph = freeze_metadata(graph)
    pool, object_keys = {}, {}

    def compact(value):
        if isinstance(value, Mapping):
            if _scope(value):
                cached = object_keys.get(id(value))
                key = semantic_sha256(value) if cached is None else cached[1]
                object_keys[id(value)] = value, key
                pool.setdefault(key, value)
                return FrozenMetadata({SCOPE_REFERENCE: key})
            if SCOPE_REFERENCE in value:
                raise ValueError("logical selection graph uses a reserved storage reference")
            return FrozenMetadata({key: compact(child) for key, child in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(compact(child) for child in value)
        return value

    record = compact(graph)
    receipts = {}
    for key, scope in sorted(pool.items()):
        relative = f"graph_scopes/{key}.json"
        target = path.parent / relative
        if target.exists():
            existing = _decode(target.read_bytes())
            if (set(existing) != {"record", "record_sha256"} or existing["record_sha256"] != key
                    or semantic_sha256(existing["record"]) != key):
                raise ValueError("existing selection scope differs from its complete identity")
        else:
            atomic_write_json_stream(target, {"record": scope, "record_sha256": key}, input_admission=input_admission)
        receipts[key] = {"path": relative, **_artifact(target)}
    payload = {"schema": SCHEMA, "record": record, "record_sha256": semantic_sha256(graph),
        "selection_sha256": semantic_sha256(graph["selection"]), "scopes": receipts}
    atomic_write_json_stream(path, payload, input_admission=input_admission)
    return path


def _legacy_minified_chunks(receipt):
    """Remove only whitespace outside literals, retaining state across blocks."""
    source = Path(receipt["path"])
    if source.is_symlink():
        raise ValueError("legacy graph path is a symlink")
    digest, nbytes = hashlib.sha256(), 0
    literal, escaped = False, False
    delimiters = re.compile(rb'["\\]')
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            nbytes += len(block)
            result, offset = bytearray(), 0
            if escaped:
                result.append(block[0])
                offset, escaped = 1, False
            while offset < len(block):
                if not literal:
                    quote = block.find(b'"', offset)
                    stop = len(block) if quote < 0 else quote
                    result.extend(block[offset:stop].translate(None, b" \t\r\n"))
                    if quote < 0:
                        break
                    result.append(34)
                    offset, literal = quote + 1, True
                else:
                    match = delimiters.search(block, offset)
                    if match is None:
                        result.extend(block[offset:])
                        break
                    stop = match.start()
                    if block[stop] == 34:
                        result.extend(block[offset:stop + 1])
                        offset, literal = stop + 1, False
                    else:
                        result.extend(block[offset:min(stop + 2, len(block))])
                        offset = stop + 2
                        escaped = offset > len(block)
            if result:
                yield bytes(result)
    if literal or escaped:
        raise ValueError("legacy graph has an unterminated JSON literal")
    if nbytes != receipt["bytes"] or digest.hexdigest() != receipt["sha256"]:
        raise ValueError("legacy graph raw bytes differ from the original external seal")


def prepare_selection_graph_store(source_receipt, destination, *, expected_graph_sha256,
                                  expected_selection_sha256=None, input_admission=None):
    """Create a separate consumer store without decoding the expanded original.

    Call only after the original writer is retired. Its real retirement/source
    outcome belongs to the caller; this function never replaces those facts.
    """
    from research.source_execution import available_host_memory_bytes, current_input_admission

    if set(source_receipt) != {"path", "sha256", "bytes"}:
        raise ValueError("legacy import requires the complete original byte receipt")
    source, destination = Path(source_receipt["path"]).resolve(), Path(destination).resolve()
    if destination.parent.is_relative_to(source.parent):
        raise ValueError("legacy graph import must use a separate consumer namespace")
    if destination.exists():
        raise FileExistsError("legacy consumer graph already exists")
    cache = destination.with_name(destination.stem + ".legacy-minified.json")
    if cache.exists():
        raise FileExistsError("legacy token cache already belongs to another invocation")
    # First stream verifies the original byte seal and counts the exact cache
    # before any artifact reservation/write. The second rechecks the same seal.
    count = sum(len(chunk) for chunk in _legacy_minified_chunks(source_receipt))
    _atomic_write_stream(cache, lambda: _legacy_minified_chunks(source_receipt), count,
        input_admission=input_admission)
    admission = input_admission or current_input_admission()
    # Explicit decoder input estimate, not a proof of the whole process peak.
    # Scope interning bounds retained duplicates; bytes/string decode coexist.
    reservation = count * 4 + 16 * 2**20
    lease = None
    if admission is not None:
        lease = admission.try_resident_input(reservation)
        if lease is None:
            raise MemoryError("legacy minified graph cannot enter the actual shared memory budget")
    elif reservation > available_host_memory_bytes():
        raise MemoryError("legacy minified graph exceeds currently available host memory")
    try:
        # Sidecar locators, if the source already uses codec2, belong to its
        # original directory. The token cache is a separate physical artifact.
        owned = SelectionGraphStore.from_bytes(cache.read_bytes(), directory=source.parent,
            expected_graph_sha256=expected_graph_sha256,
            expected_selection_sha256=expected_selection_sha256)
        selection_sha256 = owned.selection_sha256
        save_selection_graph(destination, owned.graph, input_admission=input_admission)
        prepared = SelectionGraphStore.from_record(owned.graph, directory=destination.parent,
            expected_graph_sha256=expected_graph_sha256, expected_selection_sha256=selection_sha256)
        encoded = destination.read_bytes()  # Small codec envelope, never expanded scope values.
        payload = _decode(encoded)
        prepared._storage_sha256 = hashlib.sha256(encoded).hexdigest()
        prepared._scope_receipts = freeze_metadata([{"path": str(destination.parent / row["path"]),
            "sha256": row["sha256"], "bytes": row["bytes"]} for row in payload["scopes"].values()])
        prepared._resident_lease = lease
        del owned
    except BaseException:
        if lease is not None:
            lease.release_unused()
        raise
    try:
        record = {"schema": "n2p3_legacy_selection_graph_import/1", "original": dict(source_receipt),
            "consumer_store": {"path": str(destination), **_artifact(destination)},
            "token_cache": {"path": str(cache), **_artifact(cache)},
            "logical_graph_sha256": expected_graph_sha256, "selection_sha256": selection_sha256,
            "decode_input_reservation_bytes": reservation, "new_fits": 0, "EEG_PT_decodes": 0,
            "role": "derived metadata storage; original graph and numeric producers unchanged"}
        atomic_write_json_stream(destination.with_suffix(".import.json"), record, input_admission=input_admission)
    except BaseException:
        prepared.close()
        raise
    return prepared, record


def import_legacy_selection_graph(source_receipt, destination, *, expected_graph_sha256,
                                 expected_selection_sha256=None, input_admission=None):
    """Persist an independent store and return only its transportable facts."""
    owner, record = prepare_selection_graph_store(source_receipt, destination,
        expected_graph_sha256=expected_graph_sha256, expected_selection_sha256=expected_selection_sha256,
        input_admission=input_admission)
    owner.close()
    return record


class SelectionGraphStore:
    """One actual private ingress; no caller flag can stand in for file admission."""

    def __init__(self, path, *, expected_graph_sha256=None, expected_selection_sha256=None):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("nested prefix graph path is a symlink")
        self._path = path.resolve()
        self._resident_lease = None
        content = path.read_bytes()
        self._admit_content(content, expected_graph_sha256, expected_selection_sha256)

    @classmethod
    def from_bytes(cls, content, *, directory, expected_graph_sha256, expected_selection_sha256=None):
        if not isinstance(content, bytes) or expected_graph_sha256 is None:
            raise TypeError("prepared graph requires actual bytes and its external logical seal")
        owner = cls.__new__(cls)
        owner._path = Path(directory).resolve() / "prefix_selection.json"
        owner._resident_lease = None
        owner._admit_content(content, expected_graph_sha256, expected_selection_sha256)
        return owner

    @classmethod
    def from_record(cls, graph, *, directory, expected_graph_sha256, expected_selection_sha256=None):
        """Snapshot caller mappings; already owned immutable nodes remain shared."""
        if expected_graph_sha256 is None:
            raise ValueError("prepared graph requires its external original seal")
        if not isinstance(graph, Mapping):
            raise TypeError("prepared graph requires actual record fields, not an admission flag")
        owner = cls.__new__(cls)
        owner._path = Path(directory).resolve() / "prefix_selection.json"
        owner._resident_lease = None
        owner._storage_sha256, owner._scope_receipts = None, ()
        owner._admit_graph(freeze_metadata(graph), expected_graph_sha256,
            expected_graph_sha256, expected_selection_sha256)
        return owner

    def _admit_graph(self, graph, claimed, expected_graph_sha256, expected_selection_sha256):
        digest = semantic_sha256(graph)
        selection_digest = semantic_sha256(graph["selection"])
        if (graph.get("schema") != "n2p3_whole_prefix_selection_replay/1" or digest != claimed
                or expected_graph_sha256 is not None and digest != expected_graph_sha256
                or expected_selection_sha256 is not None and selection_digest != expected_selection_sha256):
            raise ValueError("nested prefix graph differs from externally sealed original selection")
        self._graph, self._graph_sha256, self._selection_sha256 = graph, digest, selection_digest

    @property
    def path(self):
        return self._path

    @property
    def graph(self):
        if self._graph is None:
            raise RuntimeError("selection graph owner is closed")
        return self._graph

    @property
    def graph_sha256(self):
        return self._graph_sha256

    @property
    def selection_sha256(self):
        return self._selection_sha256

    @property
    def storage_sha256(self):
        return self._storage_sha256

    @property
    def scope_receipts(self):
        return self._scope_receipts

    def close(self):
        self._graph = None
        if self._resident_lease is not None:
            self._resident_lease.release_unused()
            self._resident_lease = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _admit_content(self, content, expected_graph_sha256, expected_selection_sha256):
        payload = _decode(content)
        if not isinstance(payload, Mapping):
            raise ValueError("selection graph envelope must be a mapping")
        self._storage_sha256 = hashlib.sha256(content).hexdigest()
        self._scope_receipts = ()
        if payload.get("schema") == SCHEMA:
            if set(payload) != {"schema", "record", "record_sha256", "selection_sha256", "scopes"}:
                raise ValueError("selection graph store changed its complete fields")
            pool, used, receipts = {}, set(), []
            for key, receipt in payload["scopes"].items():
                if (len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                        or set(receipt) != {"path", "sha256", "bytes"}
                        or receipt["path"] != f"graph_scopes/{key}.json"):
                    raise ValueError("selection graph scope locator changed")
                target = self.path.parent / receipt["path"]
                if target.is_symlink() or target.parent.is_symlink() or target.resolve().parent != (self.path.parent / "graph_scopes").resolve():
                    raise ValueError("selection graph scope escapes its actual store")
                encoded = target.read_bytes()
                if len(encoded) != receipt["bytes"] or hashlib.sha256(encoded).hexdigest() != receipt["sha256"]:
                    raise ValueError("selection graph scope bytes changed")
                scope = _decode(encoded)
                if (set(scope) != {"record", "record_sha256"} or scope["record_sha256"] != key
                        or not _scope(scope["record"]) or semantic_sha256(scope["record"]) != key):
                    raise ValueError("selection graph scope complete identity changed")
                pool[key] = scope["record"]
                receipts.append({"path": str(target.resolve()), "sha256": receipt["sha256"], "bytes": receipt["bytes"]})

            def resolve(value):
                if isinstance(value, Mapping):
                    if SCOPE_REFERENCE in value:
                        if set(value) != {SCOPE_REFERENCE} or value[SCOPE_REFERENCE] not in pool:
                            raise ValueError("selection graph has an unknown or malformed scope reference")
                        key = value[SCOPE_REFERENCE]
                        used.add(key)
                        return pool[key]
                    return FrozenMetadata({key: resolve(child) for key, child in value.items()})
                if isinstance(value, (list, tuple)):
                    return tuple(resolve(child) for child in value)
                return value

            graph = resolve(payload["record"])
            if used != set(pool):
                raise ValueError("selection graph scope roster includes unused or missing authority")
            self._scope_receipts = freeze_metadata(receipts)
        elif payload.get("schema") == "n2p3_whole_prefix_selection_replay/1":
            if expected_graph_sha256 is None:
                raise ValueError("prepared bare graph requires its external original seal")
            if self.storage_sha256 != expected_graph_sha256:
                raise ValueError("prepared canonical graph bytes differ from their external original seal")
            self._admit_graph(payload, expected_graph_sha256, expected_graph_sha256, expected_selection_sha256)
            return
        else:
            # Legacy expanded records are read with complete-scope hash-consing.
            graph = payload["record"]
        self._admit_graph(graph, payload["record_sha256"], expected_graph_sha256, expected_selection_sha256)
        if payload.get("schema") == SCHEMA and self.selection_sha256 != payload["selection_sha256"]:
            raise ValueError("nested prefix graph differs from externally sealed original selection")
