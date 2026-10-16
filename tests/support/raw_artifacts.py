"""Physical raw-artifact records shared by ingestion and manifest tests."""

import hashlib
import json
from pathlib import Path

from data.raw_artifacts import RAW_ARTIFACT_MANIFEST_SCHEMA, RAW_ARTIFACT_PATH_SEMANTICS


def file_record(path: Path, relative_path: str, *, with_md5: bool = True) -> dict[str, object]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "official_md5": hashlib.md5(payload).hexdigest() if with_md5 else None,  # noqa: S324
        "local_sha256": digest,
        "remote_sha256": digest,
        "verified": True,
    }


def write_manifest(
    tmp_path: Path,
    *,
    dataset_class: str,
    files: list[dict[str, object]],
    mapping_entries: list[dict[str, object]],
) -> Path:
    root_contract: dict[str, object] = {
        "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
        "expected_mne_data_root": "mne_data",
        "mne_dataset_path_key": f"MNE_DATASETS_{dataset_class.upper()}_PATH",
        "moabb_loader_mapping": {
            "schema": "n2p3_moabb_loader_mapping/1",
            "entries": mapping_entries,
        },
    }
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "dataset_class": dataset_class,
        "official_source": {"url": "https://example.invalid/official"},
        "official_record": {"record_id": f"test:{dataset_class}"},
        "artifact_root_contract": root_contract,
        "files": files,
    }
    path = tmp_path / f"{dataset_class}-raw-artifacts.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path
