"""Owned immutable JSON metadata with reusable complete canonical encoding.

This owner is separate from mathematical noise-model implementation identity.
External containers are snapshotted; only this owner's frozen nodes are shared.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


class FrozenMetadata(dict):
    """Deep JSON snapshot; cache only after complete canonical validation."""

    def __init__(self, record):
        items = record.items() if isinstance(record, Mapping) else record
        super().__init__((key, freeze_metadata(value)) for key, value in items)
        self._canonical_tree = None
        self._normalized = False
        self._canonical_bytes = None
        self._semantic_digest = None

    def _reject(self, *args, **kwargs):
        raise TypeError("fitted metadata is frozen")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def canonical_tree(self):
        if self._normalized:
            return self
        if self._canonical_tree is None:
            from research.contracts import _canonical_value

            def canonical(value):
                if isinstance(value, FrozenMetadata):
                    return value.canonical_tree()
                if isinstance(value, tuple):
                    return tuple(canonical(item) for item in value)
                # All leaves were snapshotted by the owner. The shared semantic
                # compiler still owns finite-float and signed-zero normalization.
                return _canonical_value(value)

            record = {}
            for key, value in self.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("semantic contract mapping keys must be non-empty strings.")
                record[key] = canonical(value)
            normalized = dict.__new__(FrozenMetadata)
            dict.__init__(normalized, record)
            normalized._canonical_tree = None
            normalized._normalized = True
            normalized._canonical_bytes = None
            normalized._semantic_digest = None
            self._canonical_tree = normalized
        return self._canonical_tree

    def canonical_bytes(self):
        if self._canonical_bytes is None:
            from research.contracts import _encode_canonical_tree

            self._canonical_bytes = _encode_canonical_tree(self.canonical_tree())
        return self._canonical_bytes

    def semantic_digest(self):
        if self._semantic_digest is None:
            digest = hashlib.sha256()
            if self._canonical_bytes is not None:
                digest.update(self._canonical_bytes)
            else:
                for chunk in canonical_chunks(self):
                    digest.update(chunk)
            self._semantic_digest = digest.hexdigest()
        return self._semantic_digest


def canonical_chunks(value):
    """Encode the same canonical JSON in bounded chunks, sharing frozen children."""
    from research.contracts import _canonical_value

    owned = freeze_metadata(value)
    canonical = owned.canonical_tree() if isinstance(owned, FrozenMetadata) else _canonical_value(owned)
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    pending, size = [], 0
    for chunk in encoder.iterencode(canonical):
        pending.append(chunk)
        size += len(chunk)
        if size >= 65536:
            yield "".join(pending).encode("ascii")
            pending, size = [], 0
    if pending:
        yield "".join(pending).encode("ascii")


def freeze_metadata(value):
    """Own JSON-like leaves and containers, preserving all semantic values."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, FrozenMetadata):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return freeze_metadata(value.item())
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"unsupported immutable metadata value {type(value).__name__}")


def plain_metadata(value):
    """Make fresh standard JSON containers for the existing safe Torch schema."""
    if isinstance(value, FrozenMetadata):
        return {key: plain_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_metadata(item) for item in value]
    return value
