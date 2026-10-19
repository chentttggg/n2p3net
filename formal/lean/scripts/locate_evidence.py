"""Resolve a historical Lean output without restoring obsolete build trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Evidence locator leaves the supplied repository")
    return target


def reconstruct(receipt: object, record: dict) -> bytes:
    if record["kind"] == "exact_empty_log_bytes":
        if record["bytes"] != 0 or record["sha256"] != digest(b""):
            raise ValueError("Invalid empty-log reconstruction")
        return b""
    pending = [receipt]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
            if value.get("report_sha256") != record["sha256"] or not isinstance(value.get("measurement"), dict):
                continue
            for indent in (2, 4):
                for ascii_only in (True, False):
                    text = json.dumps(value["measurement"], indent=indent, ensure_ascii=ascii_only) + "\n"
                    for data in (text.encode(), text.replace("\n", "\r\n").encode()):
                        if len(data) == record["bytes"] and digest(data) == record["sha256"]:
                            return data
    raise ValueError("Historical report cannot be reconstructed to its recorded SHA-256")


def inspect(reference: str, *, repository: Path, locator: Path) -> dict:
    repository = repository.resolve()
    registry = json.loads(locator.read_bytes())
    rows = [r for r in registry["references"] if r["reference"] == reference]
    if len(rows) != 1:
        raise ValueError("Historical reference is not uniquely indexed")
    row = rows[0]
    status = row["status"]
    if status in {"archived_original", "archived_directory_members"}:
        locations = row.get("locations", [])
        for group in row.get("members", {}).values():
            locations.extend(group)
        verified = []
        archive_hashes = {}
        for location in locations:
            archive = inside(repository, location["archive"])
            if archive not in archive_hashes:
                archive_hashes[archive] = digest(archive.read_bytes())
            if archive_hashes[archive] != location["archive_sha256"]:
                raise ValueError("Historical archive SHA-256 mismatch")
            with zipfile.ZipFile(archive) as stream:
                value = stream.read(location["member"])
            if len(value) != location["bytes"] or digest(value) != location["sha256"]:
                raise ValueError("Historical archive member SHA-256 mismatch")
            verified.append(location)
        return {"reference": reference, "status": status,
                "verified_locations": verified, "reconstructed_build": False}
    if status == "exact_receipt_reconstruction":
        record = row["reconstruction"]
        receipt = inside(repository, record["source_receipt"])
        value = receipt.read_bytes()
        if digest(value) != record["source_receipt_sha256"]:
            raise ValueError("Historical producer receipt SHA-256 mismatch")
        reconstructed = reconstruct(json.loads(value), record)
        # The reconstruction remains a derived byte object, identified by its
        # original SHA; the original producer receipt stays unchanged.
        return {"reference": reference, "status": status,
                "original_sha256": record["sha256"], "bytes": record["bytes"],
                "producer_receipt_verified": True,
                "reconstructed_sha256_verified": digest(reconstructed) == record["sha256"],
                "reconstructed_build": False}
    if status.startswith("documented_"):
        source = inside(repository, row["source"])
        value = source.read_bytes()
        expected = row.get("source_sha256")
        if expected is not None and digest(value) != expected:
            raise ValueError("Historical reference classification receipt SHA-256 mismatch")
        return {"reference": reference, "status": status,
                "classification_source": row["source"], "reconstructed_build": False}
    raise ValueError("Historical reference requires explicit producer selection")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--locator", type=Path)
    args = parser.parse_args()
    locator = args.locator or args.repo / "doc/assets/repository_audit_20261019/lean_evidence_locator.json"
    print(json.dumps(inspect(args.reference, repository=args.repo, locator=locator), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
