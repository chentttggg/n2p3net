from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from data.raw_artifacts import (
    RAW_ARTIFACT_MANIFEST_SCHEMA,
    RAW_ARTIFACT_PATH_SEMANTICS,
    RawArtifactManifest,
    load_raw_artifact_manifest,
    verified_snapshot,
    verify_raw_artifact_manifest,
)


def _file_record(path: Path, relative_path: str) -> dict[str, object]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "official_md5": hashlib.md5(payload).hexdigest(),  # noqa: S324
        "local_sha256": digest,
        "remote_sha256": digest,
        "verified": True,
    }


def _write_manifest(
    tmp_path: Path,
    *,
    dataset_class: str,
    files: list[dict[str, object]],
    mapping_entries: list[dict[str, object]] | None = None,
) -> Path:
    root_contract: dict[str, object] = {
        "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
        "expected_mne_data_root": "mne_data",
        "mne_dataset_path_key": f"MNE_DATASETS_{dataset_class.upper()}_PATH",
    }
    if mapping_entries is not None:
        root_contract["moabb_loader_mapping"] = {
            "schema": "n2p3_moabb_loader_mapping/1",
            "entries": mapping_entries,
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


def _attest(tmp_path: Path, root: Path, manifest_path: Path, dataset_class: str):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return verify_raw_artifact_manifest(
        manifest_path,
        root,
        snapshot_root=cache / "raw-snapshots",
        cache_workspace_root=cache,
        expected_dataset_class=dataset_class,
    )


def _direct_fixture(tmp_path: Path, dataset_class: str = "PublicP300"):
    root = tmp_path / "raw"
    source = root / "mne_data" / "nested" / "source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"small-public-source-artifact\n")
    relative = "mne_data/nested/source.bin"
    manifest = _write_manifest(
        tmp_path,
        dataset_class=dataset_class,
        files=[_file_record(source, relative)],
        mapping_entries=[
            {
                "subject": 1,
                "loader_relative_path": "nested/source.bin",
                "source": {"kind": "manifest_file", "relative_path": relative},
            }
        ],
    )
    return root, source, manifest


def _zip_fixture(
    tmp_path: Path,
    *,
    member: str = "official/subject_01/session_01.mat",
    payload: bytes = b"canonical-official-eeg-member",
):
    root = tmp_path / "raw-zip"
    archive = root / "mne_data" / "downloads" / "subject_01_mat.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(member, payload)
    archive_relative = "mne_data/downloads/subject_01_mat.zip"
    loader_relative = "subject_01/session_01.mat"
    manifest = _write_manifest(
        tmp_path,
        dataset_class="ZipP300",
        files=[_file_record(archive, archive_relative)],
        mapping_entries=[
            {
                "subject": 1,
                "loader_relative_path": loader_relative,
                "source": {
                    "kind": "zip_member",
                    "archive_relative_path": archive_relative,
                    "archive_member": member,
                },
            }
        ],
    )
    return root, archive, manifest, loader_relative, payload


def test_manifest_verification_creates_content_addressed_snapshot(tmp_path: Path) -> None:
    root, source, manifest = _direct_fixture(tmp_path)
    attestation = _attest(tmp_path, root, manifest, "PublicP300")
    snapshot = attestation.verified_snapshots[0]

    assert snapshot.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert Path(snapshot.snapshot_path).name == snapshot.sha256
    assert Path(snapshot.snapshot_path).read_bytes() == source.read_bytes()
    assert snapshot.snapshot_relative_path == f"objects/sha256/{snapshot.sha256}"
    assert attestation.manifest_sha256 == load_raw_artifact_manifest(manifest).digest()
    provenance = attestation.source_provenance_record("mne_data/nested/source.bin")
    assert str(tmp_path) not in json.dumps(provenance)
    assert provenance["raw_artifact_snapshot"]["source_relative_path"] == (
        "mne_data/nested/source.bin"
    )


def test_original_replacement_after_snapshot_cannot_change_output(tmp_path: Path) -> None:
    root, source, manifest = _direct_fixture(tmp_path)
    attestation = _attest(tmp_path, root, manifest, "PublicP300")
    expected = attestation.snapshot_for("mne_data/nested/source.bin")
    source.write_bytes(b"replacement-with-different-bytes")

    with expected.open_verified() as handle:
        assert handle.read() == b"small-public-source-artifact\n"


def test_in_place_source_mutation_during_copy_is_rejected(monkeypatch, tmp_path: Path) -> None:
    from data import raw_artifacts as module

    root, source, manifest = _direct_fixture(tmp_path)
    original = module._stream_digest
    attacked = False

    def mutate_during_copy(handle, *, sink, include_md5):
        nonlocal attacked
        if sink is not None and not attacked:
            attacked = True
            source.write_bytes(b"X" * len(b"small-public-source-artifact\n"))
        return original(handle, sink=sink, include_md5=include_md5)

    monkeypatch.setattr(module, "_stream_digest", mutate_during_copy)
    with pytest.raises((RuntimeError, ValueError), match="changed|mismatch"):
        _attest(tmp_path, root, manifest, "PublicP300")
    assert not list((tmp_path / "cache" / "raw-snapshots").rglob(".snapshot-*.tmp"))


def test_tampered_source_is_rejected_and_temp_is_removed(tmp_path: Path) -> None:
    root, source, manifest = _direct_fixture(tmp_path)
    source.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="mismatch"):
        _attest(tmp_path, root, manifest, "PublicP300")
    assert not list((tmp_path / "cache" / "raw-snapshots").rglob(".snapshot-*.tmp"))


def test_existing_snapshot_tampering_is_rejected(tmp_path: Path) -> None:
    root, _, manifest = _direct_fixture(tmp_path)
    attestation = _attest(tmp_path, root, manifest, "PublicP300")
    object_path = Path(attestation.verified_snapshots[0].snapshot_path)
    object_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    object_path.write_bytes(b"corrupt-object")

    with pytest.raises(ValueError, match="Snapshot.*mismatch"):
        _attest(tmp_path, root, manifest, "PublicP300")


def test_concurrent_snapshot_publish_is_single_and_leaves_no_temp(tmp_path: Path) -> None:
    _, source, manifest = _direct_fixture(tmp_path)
    expected = load_raw_artifact_manifest(manifest).files[0]
    snapshot_root = tmp_path / "cache" / "snapshots"
    snapshot_root.mkdir(parents=True)

    def publish() -> str:
        return verified_snapshot(
            source,
            source_relative_path=expected.relative_path,
            expected_size_bytes=expected.size_bytes,
            expected_sha256=expected.local_sha256,
            expected_md5=expected.official_md5,
            snapshot_root=snapshot_root,
        ).snapshot_path

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: publish(), range(16)))

    assert len(set(paths)) == 1
    assert not list(snapshot_root.rglob(".snapshot-*.tmp"))


def test_snapshot_temp_cleanup_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from data import raw_artifacts as module

    path = tmp_path / ".snapshot-transient.tmp"
    path.write_bytes(b"temporary")
    original_unlink = Path.unlink
    attempts = 0

    def transient_unlink(target: Path, *args, **kwargs) -> None:
        nonlocal attempts
        if target == path and attempts < 2:
            attempts += 1
            raise PermissionError("transient scanner lock")
        original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transient_unlink)

    module._unlink_with_retry(path, delay_s=0.0)

    assert attempts == 2
    assert not path.exists()


def test_stale_temp_file_is_never_treated_as_a_snapshot(tmp_path: Path) -> None:
    root, source, manifest = _direct_fixture(tmp_path)
    object_dir = tmp_path / "cache" / "raw-snapshots" / "objects" / "sha256"
    object_dir.mkdir(parents=True)
    stale = object_dir / ".snapshot-attacker.tmp"
    stale.write_bytes(b"unverified-stale-content")

    attestation = _attest(tmp_path, root, manifest, "PublicP300")
    snapshot = attestation.verified_snapshots[0]
    assert Path(snapshot.snapshot_path).read_bytes() == source.read_bytes()
    assert snapshot.snapshot_path != str(stale)


def test_snapshot_root_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    root, _, manifest = _direct_fixture(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="beneath cache_workspace_root"):
        verify_raw_artifact_manifest(
            manifest,
            root,
            snapshot_root=tmp_path / "outside",
            cache_workspace_root=cache,
            expected_dataset_class="PublicP300",
        )

    physical = cache / "physical"
    physical.mkdir()
    link = cache / "linked"
    try:
        link.symlink_to(physical, target_is_directory=True)
    except OSError:
        pytest.skip("The platform does not permit a test directory symlink.")
    with pytest.raises(ValueError, match="symlinks/reparse"):
        verify_raw_artifact_manifest(
            manifest,
            root,
            snapshot_root=link,
            cache_workspace_root=cache,
            expected_dataset_class="PublicP300",
        )


def test_zip_mapping_extracts_only_from_verified_archive_snapshot(tmp_path: Path) -> None:
    root, archive, manifest, loader_relative, payload = _zip_fixture(tmp_path)
    attestation = _attest(tmp_path, root, manifest, "ZipP300")
    archive.unlink()

    materialization = attestation.materialize_moabb_loaders([1])
    loader = materialization.paths_by_subject[1][0]
    materialization.verify_loader_paths([loader])
    assert loader.relative_to(Path(materialization.expected_mne_data_root)).as_posix() == (
        loader_relative
    )
    assert loader.read_bytes() == payload
    assert materialization.attestation.derived_loader_files[0].archive_member == (
        "official/subject_01/session_01.mat"
    )
    assert materialization.attestation.zip_archive_audits[0].crc_verified is True


@pytest.mark.parametrize(
    ("archive_name", "members", "loader_paths"),
    [
        (
            "subject01_session01.zip",
            ("subject01_session01/Session1/1.mat",),
            ("subject_01/Session1/1.mat",),
        ),
        ("subject_01.zip", ("subject_01.mat",), ("subject_01/subject_01.mat",)),
        (
            "group_01_mat.zip",
            ("group_01_sujet_01.mat",),
            ("subject_01/group_01_sujet_01.mat",),
        ),
        (
            "subject_01_mat.zip",
            tuple(f"subject_01_session_{index:02d}.mat" for index in range(1, 4)),
            tuple(f"subject_01/subject_01_session_{index:02d}.mat" for index in range(1, 4)),
        ),
        (
            "group_01_mat.zip",
            tuple(f"group_01_s{index}.mat" for index in range(1, 5)),
            tuple(f"subject_01/group_01_s{index}" for index in range(1, 5)),
        ),
    ],
)
def test_real_braininvaders_loader_layouts_are_manifest_driven(
    tmp_path: Path,
    archive_name: str,
    members: tuple[str, ...],
    loader_paths: tuple[str, ...],
) -> None:
    root = tmp_path / "bi-raw"
    archive = root / "mne_data" / "official" / archive_name
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for index, member in enumerate(members):
            handle.writestr(member, f"verified-member-{index}".encode())
        if archive_name == "subject01_session01.zip":
            handle.writestr("subject01_session01/meta.yml", b"runs: []\n")
    archive_relative = f"mne_data/official/{archive_name}"
    entries = [
        {
            "subject": 1,
            "loader_relative_path": loader_path,
            "source": {
                "kind": "zip_member",
                "archive_relative_path": archive_relative,
                "archive_member": member,
            },
        }
        for member, loader_path in zip(members, loader_paths, strict=True)
    ]
    manifest = _write_manifest(
        tmp_path,
        dataset_class="BrainInvadersLayout",
        files=[_file_record(archive, archive_relative)],
        mapping_entries=entries,
    )
    attestation = _attest(tmp_path, root, manifest, "BrainInvadersLayout")
    materialization = attestation.materialize_moabb_loaders([1])

    mne_root = Path(materialization.expected_mne_data_root)
    assert (
        tuple(path.relative_to(mne_root).as_posix() for path in materialization.paths_by_subject[1])
        == loader_paths
    )
    assert not (mne_root / "subject01_session01" / "meta.yml").exists()


def test_zip_crc_corruption_is_rejected_from_snapshot(tmp_path: Path) -> None:
    root, archive, manifest, _, _ = _zip_fixture(tmp_path)
    payload = bytearray(archive.read_bytes())
    central_header = payload.index(b"PK\x01\x02")
    payload[central_header + 16] ^= 0xFF
    archive.write_bytes(payload)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["files"] = [_file_record(archive, "mne_data/downloads/subject_01_mat.zip")]
    manifest.write_text(json.dumps(record), encoding="utf-8")
    attestation = _attest(tmp_path, root, manifest, "ZipP300")

    with pytest.raises(ValueError, match="CRC"):
        attestation.materialize_moabb_loaders([1])


def test_zip_member_traversal_is_rejected(tmp_path: Path) -> None:
    root, _, manifest, _, _ = _zip_fixture(tmp_path, member="../session_01.mat")
    attestation = _attest(tmp_path, root, manifest, "ZipP300")
    with pytest.raises(ValueError, match="Unsafe ZIP archive member path"):
        attestation.materialize_moabb_loaders([1])


def test_mapping_rejects_missing_member_and_duplicate_loader_paths(tmp_path: Path) -> None:
    root, _, manifest, _, _ = _zip_fixture(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    entry = record["artifact_root_contract"]["moabb_loader_mapping"]["entries"][0]
    entry["source"]["archive_member"] = "missing.mat"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    attestation = _attest(tmp_path, root, manifest, "ZipP300")
    with pytest.raises(ValueError, match="absent"):
        attestation.materialize_moabb_loaders([1])

    entry["source"]["archive_member"] = "official/subject_01/session_01.mat"
    record["artifact_root_contract"]["moabb_loader_mapping"]["entries"].append(
        {**entry, "subject": 2}
    )
    manifest.write_text(json.dumps(record), encoding="utf-8")
    attestation = _attest(tmp_path, root, manifest, "ZipP300")
    with pytest.raises(ValueError, match="repeats loader_relative_path"):
        attestation.materialize_moabb_loaders([1, 2])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("official_record"), "strict JSON mapping"),
        (lambda value: value.__setitem__("official_record", {}), "non-empty"),
        (
            lambda value: value["artifact_root_contract"].__setitem__(
                "expected_mne_data_root", "."
            ),
            "exactly 'mne_data'",
        ),
        (
            lambda value: value["files"][0].__setitem__("relative_path", "../bad"),
            "beneath the artifact root",
        ),
        (
            lambda value: value["files"][0].__setitem__("local_sha256", "ABC"),
            "SHA-256",
        ),
    ],
)
def test_manifest_contract_counterexamples(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    _, _, manifest = _direct_fixture(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(record)
    with pytest.raises(ValueError, match=message):
        RawArtifactManifest.from_record(record)


def test_verification_rejects_dataset_mismatch_and_missing_file(tmp_path: Path) -> None:
    root, source, manifest = _direct_fixture(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="dataset_class mismatch"):
        verify_raw_artifact_manifest(
            manifest,
            root,
            snapshot_root=cache / "snapshots",
            cache_workspace_root=cache,
            expected_dataset_class="WrongP300",
        )
    source.unlink()
    with pytest.raises(FileNotFoundError, match="Raw artifact is missing"):
        verify_raw_artifact_manifest(
            manifest,
            root,
            snapshot_root=cache / "snapshots",
            cache_workspace_root=cache,
            expected_dataset_class="PublicP300",
        )


def test_load_manifest_rejects_non_mapping_and_duplicate_keys(tmp_path: Path) -> None:
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON mapping"):
        load_raw_artifact_manifest(non_mapping)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_raw_artifact_manifest(duplicate)
