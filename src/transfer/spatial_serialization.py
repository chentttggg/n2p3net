"""Bounded raw-JSON serialization for owned immutable spatial model operands.

Artifact hashes retain noise_model.metadata_digest's original JSON protocol.
File output independently retains atomic_write_json's UTF-8/indent/newline
protocol. Neither path normalizes numeric leaves or signed zero.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from research.execution import _atomic_write_stream


def record_owners(value, *, opaque=()):
    """Bind frozen record owners and array views without visiting numeric rows.

    A readonly buffer still permits ndarray shape/dtype changes. Keep those
    view coordinates alongside the original array owner. Frozen JSON metadata
    and pure numeric/Fraction tuples remain opaque immutable leaves.
    """
    owners, pending = [], [value]
    while pending:
        owner = pending.pop()
        for field in fields(owner):
            item = getattr(owner, field.name)
            signature = (item.shape, item.strides, item.dtype.str) if isinstance(item, np.ndarray) else None
            owners.append((item, signature))
            if any(item is leaf for leaf in opaque):
                continue
            if is_dataclass(item):
                pending.append(item)
            elif isinstance(item, tuple):
                # Factors and compressed array owners have record-scale count;
                # never descend into nested rows, event ledgers or scalar tuples.
                for child in item:
                    if isinstance(child, np.ndarray):
                        owners.append((child, (child.shape, child.strides, child.dtype.str)))
                    elif is_dataclass(child):
                        owners.append((child, None))
                        pending.append(child)
    return tuple(owners)


def same_record_owners(current, original):
    return len(current) == len(original) and all(
        value is saved and signature == saved_signature
        for (value, signature), (saved, saved_signature) in zip(current, original, strict=True))


class SpatialRecord:
    """Raw record protocol shared by spatial owners of immutable operands.

    Subclasses provide a small payload view of their already owned leaves.
    Only the public record method materializes fresh ordinary dict/list values.
    Payload and complete-record hashes have distinct cached identities.
    """

    @cached_property
    def _artifact_identity(self):
        owners = self._serialization_owners()
        return owners, record_digest(self._record_payload())

    def _serialization_owners(self):
        return record_owners(self)

    def _require_record_owners(self):
        owners, artifact = self._artifact_identity
        if not same_record_owners(self._serialization_owners(), owners):
            raise ValueError("spatial record owners or array view coordinates changed after admission")
        return artifact

    def digest(self):
        return self._require_record_owners()

    def _record_view(self):
        artifact = self.digest()
        return {**self._record_payload(), "artifact_sha256": artifact}

    def record(self):
        return plain_record(self._record_view())

    @cached_property
    def _complete_record_sha256(self):
        return record_digest(self._record_view())

    @property
    def _record_sha256(self):
        self._require_record_owners()
        return self._complete_record_sha256


class _ArrayList(list):
    """Let Python's JSON iterator visit original array rows without tolist()."""

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)


def _numeric_default(value):
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else _ArrayList(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def plain_record(value):
    """Return fresh ordinary JSON containers at the public record boundary."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_record(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


_ARRAY_BLOCK_VALUES = 16384
_ARRAY_BLOCK_ROWS = 512


def _record_tokens(value, encoder, key_encoder, pretty, depth, active):
    """Keep container formatting streaming; C-encode bounded numeric blocks.

    Wide rows, ndarray subclasses and nonstandard dtypes retain the original
    iterator. Its local pretty indentation is shifted to the record's depth.
    """
    indent = "  " * depth if pretty else ""
    if type(value) is np.ndarray:
        row_values = value.shape[1] if value.ndim == 2 else 1
        numeric = value.dtype.kind in "biuf" and value.dtype.itemsize <= 8
        if value.ndim in (1, 2) and numeric and row_values <= _ARRAY_BLOCK_VALUES:
            if not len(value):
                yield "[]"
                return
            rows = min(_ARRAY_BLOCK_ROWS, _ARRAY_BLOCK_VALUES // max(1, row_values))
            yield "["
            for start in range(0, len(value), rows):
                if start:
                    yield ","
                # Both the Python scalar list and the C encoder's result are
                # bounded by rows and scalar count, independently of T.
                block = encoder.encode(value[start:start + rows].tolist())
                if pretty:
                    yield block[1:-2].replace("\n", "\n" + indent)
                else:
                    yield block[1:-1]
            if pretty:
                yield "\n" + indent
            yield "]"
            return
    if isinstance(value, (dict, list, tuple)):
        if not value:
            yield "{}" if isinstance(value, dict) else "[]"
            return
        marker = id(value)
        if marker in active:
            raise ValueError("Circular reference detected")
        active.add(marker)
        try:
            mapping = isinstance(value, dict)
            yield "{" if mapping else "["
            items = (sorted(value.items()) if encoder.sort_keys else value.items()) if mapping else value
            first = True
            for item in items:
                if not first:
                    yield ","
                first = False
                if pretty:
                    yield "\n" + indent + "  "
                if mapping:
                    key, item = item
                    # Delegate accepted key types, float spelling and escaping
                    # to stdlib rather than inventing a second key protocol.
                    yield key_encoder.encode({key: None})[1:-6]
                    yield ": " if pretty else ":"
                yield from _record_tokens(item, encoder, key_encoder, pretty, depth + 1, active)
            if pretty:
                yield "\n" + indent
            yield "}" if mapping else "]"
        finally:
            active.remove(marker)
        return
    for chunk in encoder.iterencode(value):
        yield chunk.replace("\n", "\n" + indent) if pretty else chunk


def record_chunks(value, *, pretty=False):
    """Yield bounded UTF-8 chunks from the two existing raw JSON protocols."""
    encoder = (json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False, default=_numeric_default)
               if pretty else json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                                                allow_nan=False, default=_numeric_default))
    key_encoder = json.JSONEncoder(ensure_ascii=not pretty, separators=(",", ":"), allow_nan=False)
    pending, size = [], 0
    for chunk in _record_tokens(value, encoder, key_encoder, pretty, 0, set()):
        pending.append(chunk)
        size += len(chunk)
        if size >= 65536:
            yield "".join(pending).encode("utf-8")
            pending, size = [], 0
    yield ("".join(pending) + ("\n" if pretty else "")).encode("utf-8")


def record_digest(value):
    digest = hashlib.sha256()
    for chunk in record_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def save_record(path, value):
    """Use the existing counted atomic publication lifecycle with owned leaves."""
    def chunks():
        return record_chunks(value, pretty=True)

    nbytes = sum(map(len, chunks()))
    return _atomic_write_stream(path, chunks, nbytes)


def saved_record_identity(path, value):
    """Join actual saved bytes to one complete live record, including EOF."""
    digest, nbytes = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for expected in record_chunks(value, pretty=True):
            actual = stream.read(len(expected))
            if actual != expected:
                raise ValueError("saved source model bytes differ from the original live producer")
            digest.update(actual)
            nbytes += len(actual)
        if stream.read(1):
            raise ValueError("saved source model bytes exceed the original live producer")
    return {"sha256": digest.hexdigest(), "bytes": nbytes}
