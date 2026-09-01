"""Build strict evaluation contracts from resolved datasets and model origins."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.epochs import EpochDataset
from data.identity import IdentityExclusionPolicy
from research.contracts import EvaluationRunContract, semantic_sha256
from transfer.checkpoint import checkpoint_training_contract


def checkpoint_model_origin(
    payload: Mapping[str, object], *, checkpoint_sha256: str
) -> dict[str, object]:
    contract = checkpoint_training_contract(payload)
    return {
        "kind": "checkpoint",
        "checkpoint_sha256": checkpoint_sha256,
        "training_contract_digest": contract.digest(),
    }


def scratch_model_origin(initialization: Mapping[str, Any]) -> dict[str, object]:
    if not initialization:
        raise ValueError("scratch initialization contract cannot be empty.")
    return {
        "kind": "scratch",
        "initialization_contract_digest": semantic_sha256(initialization),
        "initialization": dict(initialization),
    }


def build_evaluation_run_contract(
    *,
    arm_name: str,
    model_origin: Mapping[str, Any],
    dataset: EpochDataset,
    target_cache_sha256: str,
    source_snapshot_sha256: str,
    requested_subjects: Sequence[str],
    identity_policy: IdentityExclusionPolicy,
    target_protocol: Mapping[str, Any],
    adaptation: Mapping[str, Any],
    decision: Mapping[str, Any],
    evidence_scope: Mapping[str, Any],
) -> EvaluationRunContract:
    if dataset.identity_table is None:
        raise ValueError("evaluation dataset lacks a materialized identity table.")
    requested = tuple(str(subject) for subject in requested_subjects)
    requested_table = dataset.identity_table.subset(requested)
    requested_participants = requested_table.authority_keys(identity_policy)
    return EvaluationRunContract(
        arm_name=arm_name,
        model_origin=dict(model_origin),
        target_cache_sha256=target_cache_sha256,
        target_identity_digest=dataset.identity_table.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        target_protocol=dict(target_protocol),
        adaptation=dict(adaptation),
        decision=dict(decision),
        requested_participant_keys=requested_participants,
        evidence_scope=dict(evidence_scope),
    )


def source_snapshot_sha256_from_archive_manifest(path: str | Path) -> str:
    """Verify a physical source freeze and return its archive SHA-256."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source snapshot manifest must contain a mapping.")
    if payload.get("schema") != "n2p3_source_freeze/1":
        raise ValueError("source snapshot manifest has an unsupported schema.")
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source snapshot manifest has an invalid source_commit.")
    digest = payload.get("archive_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("source snapshot manifest has an invalid archive_sha256.")
    archive = payload.get("archive")
    if not isinstance(archive, str) or not archive:
        raise ValueError("source snapshot manifest lacks archive path.")
    archive_path = (manifest_path.parent / archive).resolve()
    try:
        archive_path.relative_to(manifest_path.parent)
    except ValueError as error:
        raise ValueError("source snapshot archive escapes its manifest directory.") from error
    if not archive_path.is_file():
        raise ValueError(f"source snapshot archive does not exist: {archive_path}")
    byte_size = payload.get("byte_size")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 1:
        raise ValueError("source snapshot manifest has an invalid byte_size.")
    if archive_path.stat().st_size != byte_size:
        raise ValueError("source snapshot archive byte_size disagrees with its manifest.")
    member_count = payload.get("member_count")
    if isinstance(member_count, bool) or not isinstance(member_count, int) or member_count < 1:
        raise ValueError("source snapshot manifest has an invalid member_count.")
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive_stream:
            actual_members = len(archive_stream.getmembers())
    except (OSError, tarfile.TarError) as error:
        raise ValueError("source snapshot archive is not a valid tar.gz file.") from error
    if actual_members != member_count:
        raise ValueError("source snapshot archive member_count disagrees with its manifest.")
    hasher = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != digest:
        raise ValueError("source snapshot archive hash disagrees with its manifest.")
    return digest


__all__ = [
    "build_evaluation_run_contract",
    "checkpoint_model_origin",
    "scratch_model_origin",
    "source_snapshot_sha256_from_archive_manifest",
]
