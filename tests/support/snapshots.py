"""Physical source archives and their manifests, without model dependencies."""

import hashlib
import json
import tarfile
from pathlib import Path


def write_source_snapshot(
    directory: Path,
    *,
    source_value: int = 1,
    source_commit: str = "a" * 40,
) -> tuple[Path, str, Path]:
    source = directory / "snapshot_source.py"
    source.write_text(f"SNAPSHOT_VALUE = {source_value}\n", encoding="utf-8")
    archive = directory / "source_snapshot.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        stream.add(source, arcname="snapshot_source.py")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = directory / "source_snapshot.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "n2p3_source_freeze/1",
                "archive": archive.name,
                "archive_sha256": digest,
                "source_commit": source_commit,
                "member_count": 1,
                "byte_size": archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    return manifest, digest, archive

