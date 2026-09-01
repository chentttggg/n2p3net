"""Versioned participant-identity contracts for leakage-safe transfer.

Local subject labels are dataset indexes, not person identities.  This module
keeps source-provenance identities and optional cross-study registry identities
separate so derived caches cannot make a participant appear novel merely by
renaming the dataset or subject.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

import numpy as np

DATASET_IDENTITY_SCHEMA = "n2p3net_dataset_identity_table/1"
IDENTITY_STATUSES = frozenset(
    {"source_verified", "global_verified", "cross_study_unknown"}
)
IdentityStatus = Literal[
    "source_verified", "global_verified", "cross_study_unknown"
]
IdentityExclusionPolicy = Literal["source", "source_or_global", "global"]


def _clean_identifier(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    if "\0" in text:
        raise ValueError(f"{field_name} cannot contain NUL.")
    return text


def origin_subject_key(source_dataset_id: object, source_subject_id: object) -> str:
    """Build an opaque source-scoped key without claiming cross-study identity."""

    source = _clean_identifier(source_dataset_id, field_name="source_dataset_id")
    subject = _clean_identifier(source_subject_id, field_name="source_subject_id")
    return (
        "n2p3-origin://"
        f"{quote(source, safe='')}/participant/{quote(subject, safe='')}"
    )


@dataclass(frozen=True)
class ParticipantIdentityRecord:
    """Identity evidence attached to one local subject index."""

    local_subject_id: str
    origin_subject_keys: tuple[str, ...] = ()
    global_person_keys: tuple[str, ...] = ()
    identity_status: IdentityStatus = "cross_study_unknown"

    def __post_init__(self) -> None:
        local = _clean_identifier(
            self.local_subject_id, field_name="local_subject_id"
        )
        origins = tuple(
            sorted(
                {
                    _clean_identifier(value, field_name="origin_subject_key")
                    for value in self.origin_subject_keys
                }
            )
        )
        globals_ = tuple(
            sorted(
                {
                    _clean_identifier(value, field_name="global_person_key")
                    for value in self.global_person_keys
                }
            )
        )
        status = str(self.identity_status)
        if status not in IDENTITY_STATUSES:
            raise ValueError(
                f"identity_status must be one of {sorted(IDENTITY_STATUSES)}."
            )
        if status == "source_verified" and not origins:
            raise ValueError("source_verified identity requires origin_subject_keys.")
        if status == "global_verified" and (not origins or not globals_):
            raise ValueError(
                "global_verified identity requires source and global identity keys."
            )
        if globals_ and status != "global_verified":
            raise ValueError(
                "global_person_keys require identity_status='global_verified'."
            )
        if status == "cross_study_unknown" and (origins or globals_):
            raise ValueError(
                "cross_study_unknown is reserved for development-only records "
                "without authoritative identity keys."
            )
        object.__setattr__(self, "local_subject_id", local)
        object.__setattr__(self, "origin_subject_keys", origins)
        object.__setattr__(self, "global_person_keys", globals_)
        object.__setattr__(self, "identity_status", status)

    def payload(self) -> dict[str, object]:
        return {
            "local_subject_id": self.local_subject_id,
            "origin_subject_keys": list(self.origin_subject_keys),
            "global_person_keys": list(self.global_person_keys),
            "identity_status": self.identity_status,
        }

    def authority_key(
        self,
        policy: IdentityExclusionPolicy = "source_or_global",
    ) -> str:
        """Return one explicit sampling key for this participant."""

        if policy not in {"source", "source_or_global", "global"}:
            raise ValueError(
                "identity authority policy must be source, source_or_global, or global."
            )
        candidates = (
            self.global_person_keys
            if policy == "global" or (
                policy == "source_or_global" and self.global_person_keys
            )
            else self.origin_subject_keys
        )
        if len(candidates) != 1:
            raise ValueError(
                f"participant {self.local_subject_id!r} requires exactly one {policy} "
                f"authority key, found {len(candidates)}."
            )
        return candidates[0]

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> ParticipantIdentityRecord:
        required = {
            "local_subject_id",
            "origin_subject_keys",
            "global_person_keys",
            "identity_status",
        }
        if set(value) != required:
            raise ValueError(
                "participant identity record fields must be exactly "
                f"{sorted(required)}."
            )
        origins = value["origin_subject_keys"]
        globals_ = value["global_person_keys"]
        if not isinstance(origins, list) or not all(
            isinstance(item, str) for item in origins
        ):
            raise ValueError("origin_subject_keys must be a list of strings.")
        if not isinstance(globals_, list) or not all(
            isinstance(item, str) for item in globals_
        ):
            raise ValueError("global_person_keys must be a list of strings.")
        return cls(
            local_subject_id=str(value["local_subject_id"]),
            origin_subject_keys=tuple(origins),
            global_person_keys=tuple(globals_),
            identity_status=str(value["identity_status"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DatasetIdentityTable:
    """Complete local-to-source/global identity mapping for one dataset."""

    records: tuple[ParticipantIdentityRecord, ...]
    schema: str = DATASET_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DATASET_IDENTITY_SCHEMA:
            raise ValueError(f"Unsupported dataset identity schema {self.schema!r}.")
        records = tuple(sorted(self.records, key=lambda item: item.local_subject_id))
        if not records:
            raise ValueError("DatasetIdentityTable requires at least one participant.")
        local_ids = [record.local_subject_id for record in records]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("DatasetIdentityTable local_subject_id values must be unique.")
        object.__setattr__(self, "records", records)

    @property
    def local_subject_ids(self) -> tuple[str, ...]:
        return tuple(record.local_subject_id for record in self.records)

    @property
    def is_development_only(self) -> bool:
        return any(
            record.identity_status == "cross_study_unknown"
            for record in self.records
        )

    def authority_keys(
        self,
        policy: IdentityExclusionPolicy = "source_or_global",
    ) -> tuple[str, ...]:
        """Return exactly one sampling key per participant."""

        keys = tuple(record.authority_key(policy) for record in self.records)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "identity table maps multiple local participants to one authority key."
            )
        return tuple(sorted(keys))

    def record_for(self, local_subject_id: object) -> ParticipantIdentityRecord:
        target = _clean_identifier(
            local_subject_id, field_name="target local_subject_id"
        )
        matches = [record for record in self.records if record.local_subject_id == target]
        if len(matches) != 1:
            raise ValueError(
                f"Identity table has no unique record for local subject {target!r}."
            )
        return matches[0]

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "records": [record.payload() for record in self.records],
        }

    def digest(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def payload(self) -> dict[str, object]:
        return {**self._canonical_payload(), "digest": self.digest()}

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> DatasetIdentityTable:
        if set(value) != {"schema", "records", "digest"}:
            raise ValueError(
                "dataset identity payload fields must be exactly schema, records, digest."
            )
        records = value["records"]
        if not isinstance(records, list) or not all(
            isinstance(item, Mapping) for item in records
        ):
            raise ValueError("dataset identity records must be a list of mappings.")
        table = cls(
            records=tuple(ParticipantIdentityRecord.from_payload(item) for item in records),
            schema=str(value["schema"]),
        )
        if value["digest"] != table.digest():
            raise ValueError("dataset identity digest does not match its records.")
        return table

    @classmethod
    def from_source_rows(
        cls,
        local_subject_ids: Sequence[object],
        source_dataset_ids: Sequence[object],
    ) -> DatasetIdentityTable:
        """Create source identities from aligned, producer-declared source rows."""

        if len(local_subject_ids) != len(source_dataset_ids) or not local_subject_ids:
            raise ValueError(
                "local_subject_ids and source_dataset_ids must be aligned and non-empty."
            )
        origins_by_local: dict[str, set[str]] = {}
        for local_value, source_value in zip(
            local_subject_ids, source_dataset_ids, strict=True
        ):
            local = _clean_identifier(local_value, field_name="local_subject_id")
            source = _clean_identifier(source_value, field_name="source_dataset_id")
            origins_by_local.setdefault(local, set()).add(
                origin_subject_key(source, local)
            )
        return cls(
            records=tuple(
                ParticipantIdentityRecord(
                    local_subject_id=local,
                    origin_subject_keys=tuple(origins),
                    identity_status="source_verified",
                )
                for local, origins in origins_by_local.items()
            )
        )

    def subset(self, local_subject_ids: Iterable[object]) -> DatasetIdentityTable:
        requested = {
            _clean_identifier(value, field_name="subset local_subject_id")
            for value in local_subject_ids
        }
        available = set(self.local_subject_ids)
        missing = requested - available
        if missing:
            raise ValueError(
                f"Identity subset contains unknown local subjects {sorted(missing)}."
            )
        if not requested:
            raise ValueError("Identity subset cannot be empty.")
        return DatasetIdentityTable(
            records=tuple(
                record for record in self.records if record.local_subject_id in requested
            )
        )

    def relabel_local_subjects(
        self, mapping: Mapping[str, str]
    ) -> DatasetIdentityTable:
        if set(mapping) != set(self.local_subject_ids):
            raise ValueError(
                "Identity relabel mapping must cover every local subject exactly."
            )
        return DatasetIdentityTable(
            records=tuple(
                ParticipantIdentityRecord(
                    local_subject_id=mapping[record.local_subject_id],
                    origin_subject_keys=record.origin_subject_keys,
                    global_person_keys=record.global_person_keys,
                    identity_status=record.identity_status,
                )
                for record in self.records
            )
        )

    @classmethod
    def concatenate(
        cls, tables: Sequence[DatasetIdentityTable]
    ) -> DatasetIdentityTable:
        if not tables:
            raise ValueError("At least one identity table is required.")
        records_by_local: dict[str, ParticipantIdentityRecord] = {}
        for table in tables:
            for record in table.records:
                previous = records_by_local.get(record.local_subject_id)
                if previous is not None and previous != record:
                    raise ValueError(
                        "Cannot concatenate identity tables with a local-subject collision: "
                        f"{record.local_subject_id!r}. Apply an explicit namespace first."
                    )
                records_by_local[record.local_subject_id] = record
        return cls(records=tuple(records_by_local.values()))


def assert_target_identity_excluded(
    training: DatasetIdentityTable,
    target: ParticipantIdentityRecord,
    *,
    policy: IdentityExclusionPolicy = "source_or_global",
) -> None:
    """Reject identity overlap under one explicit evidence policy."""

    if policy not in {"source", "source_or_global", "global"}:
        raise ValueError("identity exclusion policy must be source, source_or_global, or global.")
    if training.is_development_only or target.identity_status == "cross_study_unknown":
        raise ValueError(
            "confirmation identity exclusion cannot use cross_study_unknown records."
        )
    if policy in {"source", "source_or_global"}:
        target_origins = set(target.origin_subject_keys)
        training_origins = {
            key for record in training.records for key in record.origin_subject_keys
        }
        if not target_origins or any(not record.origin_subject_keys for record in training.records):
            raise ValueError("source identity exclusion requires complete origin_subject_keys.")
        if target_origins & training_origins:
            raise ValueError(
                f"pretraining identity ledger includes target subject {target.local_subject_id!r} "
                "through shared source identity."
            )
    if policy in {"global", "source_or_global"}:
        target_globals = set(target.global_person_keys)
        training_globals = {
            key for record in training.records for key in record.global_person_keys
        }
        if policy == "global" and (
            not target_globals
            or any(not record.global_person_keys for record in training.records)
        ):
            raise ValueError(
                "global identity exclusion requires explicit global keys for every participant."
            )
        if target_globals & training_globals:
            raise ValueError(
                f"pretraining identity ledger includes target subject {target.local_subject_id!r} "
                "through shared global identity."
            )


def training_identity_ledger_from_rows(
    identity_table: DatasetIdentityTable,
    subject_ids: Sequence[object],
    training_rows: Sequence[object],
) -> DatasetIdentityTable:
    """Derive checkpoint identity evidence from rows that actually reach fitting."""

    subjects = np.asarray(subject_ids).astype(str)
    rows = np.asarray(training_rows)
    if rows.dtype != np.dtype(bool) or rows.shape != subjects.shape:
        raise ValueError("training_rows must be a boolean mask aligned with subject_ids.")
    retained = subjects[rows]
    if len(retained) == 0:
        raise ValueError("training identity ledger cannot be built from zero rows.")
    return identity_table.subset(np.unique(retained).tolist())


__all__ = [
    "DATASET_IDENTITY_SCHEMA",
    "DatasetIdentityTable",
    "IdentityExclusionPolicy",
    "ParticipantIdentityRecord",
    "assert_target_identity_excluded",
    "origin_subject_key",
    "training_identity_ledger_from_rows",
]
