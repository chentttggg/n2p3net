"""Compact PROV-style lineage for source and derived EEG entities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DATA_LINEAGE_SCHEMA = "n2p3net_data_lineage/1"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("lineage parameters cannot contain NaN or infinity.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("lineage parameter keys must be non-empty strings.")
            output[key] = _canonical(item)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        return _canonical(value.item())
    raise TypeError(f"unsupported lineage parameter {type(value).__name__}.")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return text


@dataclass(frozen=True)
class LineageEntity:
    operation: str
    parent_entity_digests: tuple[str, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        if not operation:
            raise ValueError("lineage operation must be non-empty.")
        parents = tuple(sorted({_sha256(value, "parent_entity_digest") for value in self.parent_entity_digests}))
        if operation == "source_ingress" and parents:
            raise ValueError("source_ingress cannot have parent entities.")
        if operation != "source_ingress" and not parents:
            raise ValueError("derived lineage entities require parent entities.")
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise ValueError("lineage parameters must be a non-empty mapping.")
        _canonical(self.parameters)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parent_entity_digests", parents)

    def semantic_record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "parent_entity_digests": list(self.parent_entity_digests),
            "parameters": _canonical(self.parameters),
        }

    def digest(self) -> str:
        return _digest(self.semantic_record())

    def payload(self) -> dict[str, object]:
        return {**self.semantic_record(), "entity_digest": self.digest()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LineageEntity:
        if set(payload) != {
            "operation",
            "parent_entity_digests",
            "parameters",
            "entity_digest",
        }:
            raise ValueError("lineage entity fields are invalid.")
        parents = payload["parent_entity_digests"]
        parameters = payload["parameters"]
        if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
            raise ValueError("parent_entity_digests must be a string list.")
        if not isinstance(parameters, Mapping):
            raise ValueError("lineage entity parameters must be a mapping.")
        entity = cls(
            operation=str(payload["operation"]),
            parent_entity_digests=tuple(parents),
            parameters=dict(parameters),
        )
        if payload["entity_digest"] != entity.digest():
            raise ValueError("lineage entity digest does not match its semantic record.")
        return entity


@dataclass(frozen=True)
class DataLineage:
    entities: tuple[LineageEntity, ...]
    schema: str = DATA_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DATA_LINEAGE_SCHEMA:
            raise ValueError(f"unsupported data lineage schema {self.schema!r}.")
        if not self.entities:
            raise ValueError("data lineage requires at least one entity.")
        seen: dict[str, LineageEntity] = {}
        for entity in self.entities:
            digest = entity.digest()
            if digest in seen and seen[digest] != entity:
                raise ValueError("lineage contains conflicting records for one digest.")
            missing = set(entity.parent_entity_digests) - set(seen)
            if missing:
                raise ValueError(f"lineage entity refers to unavailable parents {sorted(missing)}.")
            seen[digest] = entity
        if len(seen) != len(self.entities):
            raise ValueError("lineage entities must be unique and topologically ordered.")

    @property
    def entity_digest(self) -> str:
        return self.entities[-1].digest()

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "entities": [entity.payload() for entity in self.entities],
            "entity_digest": self.entity_digest,
        }
        payload["lineage_digest"] = _digest(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DataLineage:
        if set(payload) != {"schema", "entities", "entity_digest", "lineage_digest"}:
            raise ValueError("data lineage payload fields are invalid.")
        entities = payload["entities"]
        if not isinstance(entities, list) or not all(isinstance(item, Mapping) for item in entities):
            raise ValueError("data lineage entities must be a mapping list.")
        lineage = cls(
            entities=tuple(LineageEntity.from_payload(item) for item in entities),
            schema=str(payload["schema"]),
        )
        expected = lineage.payload()
        if payload["entity_digest"] != expected["entity_digest"]:
            raise ValueError("data lineage head digest is invalid.")
        if payload["lineage_digest"] != expected["lineage_digest"]:
            raise ValueError("data lineage digest is invalid.")
        return lineage

    @classmethod
    def source(cls, *, parameters: Mapping[str, Any]) -> DataLineage:
        return cls(
            entities=(
                LineageEntity(
                    operation="source_ingress",
                    parent_entity_digests=(),
                    parameters=parameters,
                ),
            )
        )

    @classmethod
    def derive(
        cls,
        parents: Sequence[DataLineage],
        *,
        operation: str,
        parameters: Mapping[str, Any],
    ) -> DataLineage:
        if not parents:
            raise ValueError("derived lineage requires at least one parent.")
        merged: dict[str, LineageEntity] = {}
        order: list[str] = []
        for parent in parents:
            for entity in parent.entities:
                digest = entity.digest()
                previous = merged.get(digest)
                if previous is not None and previous != entity:
                    raise ValueError("parent lineages disagree about one entity digest.")
                if previous is None:
                    merged[digest] = entity
                    order.append(digest)
        derived = LineageEntity(
            operation=operation,
            parent_entity_digests=tuple(parent.entity_digest for parent in parents),
            parameters=parameters,
        )
        if derived.digest() in merged:
            raise ValueError("derived operation duplicates an existing lineage entity.")
        return cls(entities=tuple(merged[digest] for digest in order) + (derived,))


__all__ = ["DATA_LINEAGE_SCHEMA", "DataLineage", "LineageEntity"]
