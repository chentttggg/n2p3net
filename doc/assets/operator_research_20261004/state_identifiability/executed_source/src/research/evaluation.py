"""Build strict evaluation contracts from resolved datasets and model origins."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research.contracts import EvaluationRunContract, semantic_sha256

if TYPE_CHECKING:
    from data.epochs import EpochDataset
    from data.identity import IdentityExclusionPolicy


def checkpoint_model_origin(
    payload: Mapping[str, object], *, checkpoint_sha256: str
) -> dict[str, object]:
    from transfer.checkpoint import checkpoint_training_contract

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


def verified_source_snapshot(path: str | Path) -> dict[str, object]:
    """Verify a declared physical source archive and return its recorded identity.

    The returned mapping binds the archive bytes (``archive_sha256``), the
    manifest document itself (``manifest_sha256``), and the source commit the
    manifest declares. It does not compare the archive with the running
    checkout; establishing which files executed requires a separate audit.
    """

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
    manifest_hasher = hashlib.sha256()
    manifest_hasher.update(manifest_path.read_bytes())
    working_tree = payload.get("working_tree_snapshot", False)
    if not isinstance(working_tree, bool):
        raise ValueError("working_tree_snapshot must be boolean when declared.")
    source_role = payload.get("source_commit_role", "exact_commit")
    if working_tree and source_role != "base_commit_only":
        raise ValueError("working-tree freezes must identify source_commit as base_commit_only.")
    return {
        "archive_sha256": digest,
        "manifest_sha256": manifest_hasher.hexdigest(),
        "source_commit": source_commit,
        "byte_size": byte_size,
        "member_count": member_count,
        **(
            {"working_tree_snapshot": True, "source_commit_role": source_role}
            if working_tree
            else {}
        ),
    }


def source_snapshot_sha256_from_archive_manifest(path: str | Path) -> str:
    """Verify a physical source freeze and return its archive SHA-256."""

    return str(verified_source_snapshot(path)["archive_sha256"])


def verified_executed_snapshot(
    path: str | Path, *, root: str | Path, required_files: Sequence[str] = (),
) -> dict[str, object]:
    """Bind the executing backend, including newly added runtime modules."""
    result = verified_source_snapshot(path)
    root = Path(root).resolve()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("executed snapshot requires a nonempty files digest mapping")
    required = set(required_files) | {"pyproject.toml", "uv.lock"}
    required.update(item.relative_to(root).as_posix() for item in (root / "src").rglob("*.py"))
    required.update(item.relative_to(root).as_posix() for item in (root / "experiments").glob("*.py"))
    missing = required - set(recorded)
    if missing:
        raise ValueError(f"executed snapshot missing runtime files: {sorted(missing)}")
    for name, digest in recorded.items():
        actual_path = (root / name).resolve()
        if not actual_path.is_relative_to(root):
            raise ValueError("snapshot file path escapes executing checkout")
        if not actual_path.is_file() or hashlib.sha256(actual_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"executed source differs from snapshot: {name}")
    with tarfile.open(Path(path).resolve().parent / payload["archive"], "r:gz") as archive:
        members = {member.name: member for member in archive if member.isfile()}
        for name, digest in recorded.items():
            member = members.get(name)
            if member is None or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != digest:
                raise ValueError(f"snapshot file digest disagrees with physical archive: {name}")
    return {**result, "executed_files_verified": len(recorded)}


__all__ = [
    "build_evaluation_run_contract",
    "checkpoint_model_origin",
    "scratch_model_origin",
    "source_snapshot_sha256_from_archive_manifest",
    "verified_source_snapshot",
    "verified_executed_snapshot",
]
