"""Explicit source-stream membership for study-level exclusion and grouped folds."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from data.candidate_task import CandidateTaskContract
from data.epochs import EpochDataset, materialize_dataset_identity, materialize_dataset_lineage
from data.identity import DatasetIdentityTable, source_dataset_ids_from_identity
from data.lineage import DataLineage

STUDY_MEMBERSHIP_SCHEMA = "n2p3_source_study_membership/1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def _record(mapping: Mapping[str, str], evidence: list[dict], **extra) -> dict:
    record = {
        "schema": STUDY_MEMBERSHIP_SCHEMA,
        "stream_to_study": dict(sorted(mapping.items())),
        "evidence": evidence,
        **extra,
    }
    return {**record, "digest": _digest(record)}


def _validated(record: Mapping) -> dict:
    if not isinstance(record, Mapping) or record.get("schema") != STUDY_MEMBERSHIP_SCHEMA:
        raise ValueError("study membership requires its explicit versioned schema")
    content = {key: value for key, value in record.items() if key != "digest"}
    if record.get("digest") != _digest(content):
        raise ValueError("study membership digest disagrees with its evidence")
    mapping = record.get("stream_to_study")
    if (
        not isinstance(mapping, Mapping)
        or not mapping
        or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in mapping.items()
        )
    ):
        raise ValueError("study membership must map explicit nonempty stream and study identifiers")
    if not isinstance(record.get("evidence"), list) or not record["evidence"]:
        raise ValueError("study membership requires producer evidence")
    return dict(record)


def _from_metadata(
    identity: DatasetIdentityTable, lineage: DataLineage, provenance: Mapping
) -> dict:
    streams = set(source_dataset_ids_from_identity(identity))
    grouped: dict[str, str] = {}
    evidence = []
    ancestors: dict[str, set[str]] = {}

    def assign(source_ids, study, *, basis, entity):
        if not isinstance(study, str) or not study.strip():
            raise ValueError("producer study identifier must be explicit")
        selected = sorted(set(source_ids) & streams)
        for stream in selected:
            if stream in grouped and grouped[stream] != study:
                raise ValueError(
                    f"conflicting producer study assignment for source stream {stream!r}"
                )
            grouped[stream] = study
        if selected:
            evidence.append(
                {
                    "basis": basis,
                    "lineage_entity_digest": entity,
                    "source_dataset_ids": selected,
                    "study_id": study,
                }
            )

    for entity in lineage.entities:
        parameters = entity.parameters
        if entity.operation == "source_ingress":
            source_ids = parameters.get("source_dataset_ids")
            if not isinstance(source_ids, list) or not all(
                isinstance(item, str) for item in source_ids
            ):
                raise ValueError(
                    "source ingress must preserve its original dataset stream identifiers"
                )
            inherited = set(source_ids)
        else:
            inherited = set().union(*(ancestors[parent] for parent in entity.parent_entity_digests))
        ancestors[entity.digest()] = inherited
        candidate = parameters.get("source_candidate_task_contract")
        if candidate is not None:
            contract = CandidateTaskContract.from_record(candidate)
            assign(
                inherited,
                contract.dataset_id,
                basis="lineage_source_candidate_task_contract",
                entity=entity.digest(),
            )
        source_provenance = parameters.get("source_provenance", {})
        if isinstance(source_provenance, Mapping):
            if source_provenance.get("dataset_class") is not None:
                assign(
                    inherited,
                    source_provenance["dataset_class"],
                    basis="ingress_dataset_class",
                    entity=entity.digest(),
                )
            candidate = source_provenance.get("candidate_task_contract")
            if candidate is not None:
                assign(
                    inherited,
                    CandidateTaskContract.from_record(candidate).dataset_id,
                    basis="ingress_candidate_task_contract",
                    entity=entity.digest(),
                )
    candidate = provenance.get("candidate_task_contract")
    if candidate is not None:
        assign(
            streams,
            CandidateTaskContract.from_record(candidate).dataset_id,
            basis="dataset_candidate_task_contract",
            entity=lineage.entity_digest,
        )
    if provenance.get("dataset_class") is not None:
        assign(
            streams,
            provenance["dataset_class"],
            basis="dataset_class",
            entity=lineage.entity_digest,
        )
    # An atomic producer dataset identifier stays literal; grouping several streams
    # requires the scoped producer evidence above, never a namespace or prefix rule.
    for stream in sorted(streams - set(grouped)):
        grouped[stream] = stream
        evidence.append(
            {
                "basis": "literal_producer_source_dataset_id",
                "identity_digest": identity.digest(),
                "source_dataset_ids": [stream],
                "study_id": stream,
                "cross_stream_grouping_asserted": False,
            }
        )
    return _record(
        grouped,
        evidence,
        identity_digest=identity.digest(),
        lineage_digest=lineage.payload()["lineage_digest"],
    )


def dataset_study_membership(dataset: EpochDataset) -> dict:
    """Recover study membership from producer contracts scoped by lineage ancestry."""
    return _from_metadata(
        materialize_dataset_identity(dataset),
        materialize_dataset_lineage(dataset),
        dataset.provenance,
    )


def study_ids_for_identity(identity: DatasetIdentityTable, membership: Mapping) -> tuple[str, ...]:
    mapping = _validated(membership)["stream_to_study"]
    streams = source_dataset_ids_from_identity(identity)
    missing = set(streams) - set(mapping)
    if missing:
        raise ValueError(f"unresolved source stream study membership: {sorted(missing)}")
    return tuple(sorted({mapping[stream] for stream in streams}))


def load_study_membership_proof(cache_path: str | Path) -> dict:
    """Verify exact cache bytes and embedded producer metadata without decoding EEG."""
    path = Path(cache_path).resolve()
    sidecar = path.with_suffix(".record.json")
    sidecar_bytes = sidecar.read_bytes()
    record = json.loads(sidecar_bytes)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if record.get("cache_attestation", {}).get("sha256") != sha:
            raise ValueError("study proof cache bytes disagree with sidecar attestation")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            identity_payload = json.loads(str(archive["identity_json"].item()))
            lineage_payload = json.loads(str(archive["lineage_json"].item()))
            provenance = json.loads(str(archive["provenance_json"].item()))
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("study proof cache changed during verification")
    if any(
        _digest(record.get(name)) != _digest(value)
        for name, value in (
            ("identity", identity_payload),
            ("lineage", lineage_payload),
            ("provenance", provenance),
        )
    ):
        raise ValueError("study proof sidecar metadata differs from embedded cache metadata")
    identity = DatasetIdentityTable.from_payload(identity_payload)
    lineage = DataLineage.from_payload(lineage_payload)
    membership = _from_metadata(identity, lineage, provenance)
    content = {key: value for key, value in membership.items() if key != "digest"}
    content["proof_source"] = {
        "cache": str(path),
        "cache_sha256": sha,
        "record": str(sidecar),
        "record_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "identity_digest": identity.digest(),
        "signals_decoded": False,
    }
    return {**content, "digest": _digest(content)}


def checkpoint_study_membership(payload: Mapping, *, proofs: Sequence[Mapping] = ()) -> dict:
    """Resolve a checkpoint's actual studies without relabelling its historical state."""
    from transfer.checkpoint import (
        checkpoint_training_contract,
        checkpoint_training_identity_ledger,
    )

    contract = checkpoint_training_contract(payload)
    identity = checkpoint_training_identity_ledger(payload)
    embedded = payload.get("source_study_membership")
    declared = contract.objective.get("source_study_membership")
    if embedded is not None or declared is not None:
        if embedded is None or declared is None or _digest(embedded) != _digest(declared):
            raise ValueError(
                "checkpoint study membership disagrees with its bound training contract"
            )
        result = _validated(embedded)
        study_ids_for_identity(identity, result)
        return result
    streams = set(source_dataset_ids_from_identity(identity))
    mapping = {}
    evidence = []
    verified_proofs = []
    for proof in proofs:
        proof = _validated(proof)
        for stream in streams & set(proof["stream_to_study"]):
            study = proof["stream_to_study"][stream]
            if stream in mapping and mapping[stream] != study:
                raise ValueError("conflicting source study membership proofs")
            mapping[stream] = study
            evidence.append(
                {
                    "basis": "external_exact_stream_producer_proof",
                    "source_dataset_ids": [stream],
                    "study_id": study,
                    "proof_digest": proof["digest"],
                }
            )
        verified_proofs.append(proof.get("proof_source", {"membership_digest": proof["digest"]}))
    risk = contract.objective.get("source_risk_contract", {})
    domain_mass = risk.get("domain_mass", {}) if isinstance(risk, Mapping) else {}
    for stream in streams - set(mapping):
        if stream in domain_mass:
            mapping[stream] = stream
            evidence.append(
                {
                    "basis": "exact_source_id_in_bound_training_domain_mass",
                    "source_dataset_ids": [stream],
                    "study_id": stream,
                    "training_contract_digest": contract.digest(),
                }
            )
    if streams - set(mapping):
        raise ValueError(
            f"unresolved legacy checkpoint source streams require producer proofs: {sorted(streams - set(mapping))}"
        )
    result = _record(
        mapping,
        evidence,
        identity_digest=identity.digest(),
        proofs=verified_proofs,
        checkpoint_training_contract_digest=contract.digest(),
        original_source_cache_sha256=contract.source_cache_sha256,
        historical_checkpoint_unchanged=True,
    )
    if domain_mass and set(study_ids_for_identity(identity, result)) != set(domain_mass):
        raise ValueError("proved study membership disagrees with bound source risk domains")
    return result
