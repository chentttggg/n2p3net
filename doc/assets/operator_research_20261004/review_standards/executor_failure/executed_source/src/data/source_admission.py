"""Admit completed preparation artifacts without loading EEG into the scheduler."""
from __future__ import annotations

import json
from pathlib import Path

from data.epochs import _read_epoch_cache_record, parse_preprocessing_spec
from data.identity import DatasetIdentityTable, source_dataset_ids_from_identity
from data.source_records import SOURCE_PREPARATION_SCHEMA, file_sha256, verify_prepared_cache


def bind_source_inputs(previous, prepared):
    """Freeze arrived inputs across resumes; never overwrite an old fit identity."""
    bound = dict(previous)
    for study, record in prepared.items():
        identity = {key: record[key] for key in
                    ("manifest_sha256", "cache_sha256", "cache_record_sha256", "preparation_method")}
        if study in bound and bound[study] != identity:
            raise ValueError(f"{study}: source input changed after run binding; use a new output directory")
        bound[study] = identity
    return bound


def verify_source_preparation(record, *, study, preprocessing):
    """Check the published cache, physical study identity and requested operator.

    Download receipts and a caller-supplied `admitted` flag are not preparation
    evidence. The original full-validation receipt must accompany the NPZ.
    Raw/shard resume verification stays with prepare_source_records; consumers
    bind the published merged artifact and its recorded preparation procedure.
    """
    if (record.get("schema") != SOURCE_PREPARATION_SCHEMA or record.get("status") != "completed"
            or record.get("study") != study or record.get("errors") or record.get("target_prefixes") is not None):
        raise ValueError(f"{study}: a completed source preparation ledger is required")
    method = record.get("preparation_method", {})
    if parse_preprocessing_spec(method.get("preprocessing", {})) != preprocessing:
        raise ValueError(f"{study}: preparation preprocessing differs from the requested profile")
    rows = record.get("records", [])
    if not rows or any(row.get("record", {}).get("study") != study for row in rows):
        raise ValueError(f"{study}: preparation record identities are missing or mixed")
    if any(row.get("preparation", {}).get("method") != method for row in rows):
        raise ValueError(f"{study}: preparation records disagree on the executed method")
    cache = Path(record["cache"])
    verify_prepared_cache(cache, cache_sha256=record.get("cache_sha256"),
                          cache_record_sha256=record.get("cache_record_sha256"))
    metadata, _ = _read_epoch_cache_record(cache)
    identity = DatasetIdentityTable.from_payload(metadata["identity"])
    if set(source_dataset_ids_from_identity(identity)) != {study}:
        raise ValueError(f"{study}: prepared cache contains a different physical source study")
    if (parse_preprocessing_spec(metadata["preprocessing"]) != preprocessing or
            metadata["provenance"].get("source_reference") != "measured_channel:CZ"):
        raise ValueError(f"{study}: prepared cache input contract differs")
    if (record.get("participants") != metadata["n_subjects"] or
            record.get("n_epochs") != metadata["shape"][0] or
            record.get("scheduled") != metadata["events"]["n_scheduled"]):
        raise ValueError(f"{study}: preparation totals differ from the published cache")
    return record


class PreparedSourceInventory:
    """Inspect required studies, returning ready, pending and failed separately.

    Completed byte verification is cached while the ledger, NPZ and receipt
    retain the same filesystem identities. Training still uses an immutable
    attested cache load. Missing/failed members never shrink the requested pool.
    """

    def __init__(self, directory, *, preprocessing):
        self.directory = Path(directory)
        self.preprocessing = preprocessing
        self._verified = {}

    def inspect(self, studies):
        required = set(studies)
        ready, pending, failed, seen = {}, {}, {}, set()
        for path in sorted(self.directory.glob("*/preparation.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as error:
                # The standard preparation directory is named for its study.
                if path.parent.name in required:
                    failed[path.parent.name] = f"unreadable preparation ledger: {error}"
                continue
            study = record.get("study")
            if study not in required:
                continue
            if study in seen:
                failed[study] = "duplicate preparation ledgers for one study"
                ready.pop(study, None)
                continue
            seen.add(study)
            if record.get("status") in {"running", "pending"}:
                pending[study] = record["status"]
                continue
            try:
                if record.get("status") != "completed":
                    raise ValueError(f"preparation status={record.get('status')}: {record.get('errors', [])}")
                cache = Path(record["cache"])
                if not cache.is_absolute():
                    cache = path.parent / cache
                    record = {**record, "cache": str(cache.resolve())}
                fingerprint = (file_sha256(path), *[
                    (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                    for p in (cache, cache.with_suffix(".record.json"))])
                if self._verified.get(study) != fingerprint:
                    verify_source_preparation(record, study=study, preprocessing=self.preprocessing)
                    self._verified[study] = fingerprint
                ready[study] = record
            except (OSError, ValueError, KeyError, TypeError) as error:
                failed[study] = str(error)
        for study in sorted(required - seen - set(failed)):
            pending[study] = "missing preparation ledger"
        for study in failed:
            ready.pop(study, None)
            pending.pop(study, None)
        return {"ready": ready, "pending": pending, "failed": failed}
