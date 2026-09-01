"""Physical source-file attestation for public EEG dataset caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

RAW_ARTIFACT_MANIFEST_SCHEMA = "n2p3_raw_artifact_manifest/1"
RAW_ARTIFACT_PATH_SEMANTICS = "posix_relative_path_beneath_explicit_artifact_root"
RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT = "mne_data"
RAW_ARTIFACT_ZIP_MAX_MEMBERS = 100_000
RAW_ARTIFACT_ZIP_MAX_MEMBER_BYTES = 64 * 1024**3
RAW_ARTIFACT_ZIP_MAX_TOTAL_BYTES = 256 * 1024**3
RAW_ARTIFACT_ZIP_MAX_EXPANSION_RATIO = 200.0
RAW_ARTIFACT_SNAPSHOT_OBJECT_ROLE = "content_addressed_verified_source"
RAW_ARTIFACT_MOABB_WORKSPACE_ROLE = "controlled_read_only_mne_materialization"
RAW_ARTIFACT_COPY_CHUNK_BYTES = 1024 * 1024


def _unlink_with_retry(path: Path, *, attempts: int = 20, delay_s: float = 0.01) -> None:
    """Remove a closed temporary file despite short Windows scanner locks."""

    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay_s)


_MANIFEST_FIELDS = {
    "schema",
    "dataset_class",
    "official_source",
    "official_record",
    "artifact_root_contract",
    "files",
}
_FILE_FIELDS = {
    "relative_path",
    "size_bytes",
    "official_md5",
    "local_sha256",
    "remote_sha256",
    "verified",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MD5_PATTERN = re.compile(r"[0-9a-f]{32}")
_WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:")
_MNE_DATASET_PATH_KEY_PATTERN = re.compile(r"MNE_DATASETS_[A-Z0-9_]+_PATH")


def _canonical_json_value(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} cannot contain NaN or infinity.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings.")
            output[key] = _canonical_json_value(item, name=name)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item, name=name) for item in value]
    raise TypeError(f"{name} contains non-JSON value {type(value).__name__}.")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_json_value(value, name="raw artifact manifest"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _validated_md5(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _MD5_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be null or one lowercase MD5 digest.")
    return value


def _validated_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("files[].relative_path must be a non-empty string.")
    if "\\" in value or "\x00" in value:
        raise ValueError("files[].relative_path must use safe POSIX separators.")
    raw_parts = value.split("/")
    if (
        value.startswith("/")
        or _WINDOWS_DRIVE_PATTERN.fullmatch(raw_parts[0]) is not None
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError("files[].relative_path must stay beneath the artifact root.")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("files[].relative_path must be a canonical POSIX relative path.")
    if any(part != part.strip() or ":" in part for part in raw_parts):
        raise ValueError("files[].relative_path contains an unsafe path component.")
    return value


def _validated_root_relative_path(value: object) -> str:
    if value != RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT:
        raise ValueError(
            "artifact_root_contract.expected_mne_data_root must be exactly 'mne_data'."
        )
    return RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT


def _validated_archive_member(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ZIP archive members must have non-empty names.")
    candidate = value[:-1] if value.endswith("/") else value
    if not candidate:
        raise ValueError("ZIP archives must not contain an unnamed root member.")
    try:
        return _validated_relative_path(candidate)
    except ValueError as exc:
        raise ValueError(f"Unsafe ZIP archive member path {value!r}.") from exc


@dataclass(frozen=True)
class RawArtifactFile:
    relative_path: str
    size_bytes: int
    official_md5: str | None
    local_sha256: str
    remote_sha256: str
    verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _validated_relative_path(self.relative_path))
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("files[].size_bytes must be a non-negative integer.")
        object.__setattr__(
            self,
            "official_md5",
            _validated_md5(self.official_md5, name="files[].official_md5"),
        )
        local = _validated_sha256(self.local_sha256, name="files[].local_sha256")
        remote = _validated_sha256(self.remote_sha256, name="files[].remote_sha256")
        if local != remote:
            raise ValueError("files[].local_sha256 must equal files[].remote_sha256.")
        if self.verified is not True:
            raise ValueError("Every raw artifact manifest file must have verified=true.")
        object.__setattr__(self, "local_sha256", local)
        object.__setattr__(self, "remote_sha256", remote)

    def record(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "official_md5": self.official_md5,
            "local_sha256": self.local_sha256,
            "remote_sha256": self.remote_sha256,
            "verified": True,
        }

    @classmethod
    def from_record(cls, value: object) -> RawArtifactFile:
        if not isinstance(value, Mapping) or set(value) != _FILE_FIELDS:
            raise ValueError("Each raw artifact file must be a strict JSON record.")
        return cls(
            relative_path=value["relative_path"],  # type: ignore[arg-type]
            size_bytes=value["size_bytes"],  # type: ignore[arg-type]
            official_md5=value["official_md5"],  # type: ignore[arg-type]
            local_sha256=value["local_sha256"],  # type: ignore[arg-type]
            remote_sha256=value["remote_sha256"],  # type: ignore[arg-type]
            verified=value["verified"],  # type: ignore[arg-type]
        )


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_link_or_reparse(path: Path) -> bool:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _assert_existing_components_are_physical(path: Path) -> None:
    absolute = _absolute_path(path)
    for candidate in (absolute, *absolute.parents):
        if not candidate.exists():
            continue
        if _is_link_or_reparse(candidate):
            raise ValueError(
                f"Snapshot paths must not contain symlinks/reparse points: {candidate}."
            )


def _prepare_snapshot_root(
    snapshot_root: str | Path,
    *,
    cache_workspace_root: str | Path,
) -> tuple[Path, Path]:
    workspace_input = _absolute_path(cache_workspace_root)
    _assert_existing_components_are_physical(workspace_input)
    workspace = workspace_input.resolve(strict=True)
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)

    snapshot_input = _absolute_path(snapshot_root)
    try:
        snapshot_input.relative_to(workspace_input)
    except ValueError as exc:
        raise ValueError("snapshot_root must stay beneath cache_workspace_root.") from exc
    if snapshot_input == workspace_input:
        raise ValueError("snapshot_root must be a dedicated child of cache_workspace_root.")
    _assert_existing_components_are_physical(snapshot_input)
    snapshot_input.mkdir(parents=True, exist_ok=True)
    _assert_existing_components_are_physical(snapshot_input)
    snapshot = snapshot_input.resolve(strict=True)
    try:
        snapshot.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("snapshot_root resolves outside cache_workspace_root.") from exc
    if not snapshot.is_dir():
        raise NotADirectoryError(snapshot)
    return snapshot, workspace


def _open_regular_nofollow(path: Path) -> int:
    _assert_existing_components_are_physical(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"Snapshot source is not a regular file: {path}.")
    return descriptor


def _stream_digest(
    handle: BinaryIO,
    *,
    sink: BinaryIO | None,
    include_md5: bool,
) -> tuple[int, str, str | None]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5() if include_md5 else None  # noqa: S324 - official checksum
    size_bytes = 0
    while block := handle.read(RAW_ARTIFACT_COPY_CHUNK_BYTES):
        if sink is not None:
            sink.write(block)
        size_bytes += len(block)
        sha256.update(block)
        if md5 is not None:
            md5.update(block)
    return size_bytes, sha256.hexdigest(), md5.hexdigest() if md5 is not None else None


def _verify_open_handle(
    handle: BinaryIO,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_md5: str | None,
    label: str,
) -> None:
    handle.seek(0)
    before = os.fstat(handle.fileno())
    size_bytes, sha256, md5 = _stream_digest(
        handle,
        sink=None,
        include_md5=expected_md5 is not None,
    )
    after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed while its open descriptor was verified.")
    if size_bytes != expected_size_bytes:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size_bytes}, got {size_bytes}."
        )
    if sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {sha256}.")
    if expected_md5 is not None and md5 != expected_md5:
        raise ValueError(f"{label} official MD5 mismatch: expected {expected_md5}, got {md5}.")
    handle.seek(0)


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class VerifiedSnapshot:
    """One immutable content-addressed object, plus non-machine identity metadata."""

    source_relative_path: str
    size_bytes: int
    sha256: str
    md5: str | None
    snapshot_relative_path: str
    role: str
    snapshot_root_path: str
    snapshot_path: str

    def __post_init__(self) -> None:
        source = _validated_relative_path(self.source_relative_path)
        digest = _validated_sha256(self.sha256, name="snapshot sha256")
        checksum = _validated_md5(self.md5, name="snapshot md5")
        relative = _validated_relative_path(self.snapshot_relative_path)
        if relative != f"objects/sha256/{digest}":
            raise ValueError("snapshot_relative_path must be the SHA-256 object path.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("snapshot size_bytes must be a non-negative integer.")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("snapshot role must be a non-empty string.")
        root = _absolute_path(self.snapshot_root_path).resolve(strict=True)
        path = _absolute_path(self.snapshot_path)
        _assert_existing_components_are_physical(path)
        resolved = path.resolve(strict=True)
        if resolved != root.joinpath(*PurePosixPath(relative).parts):
            raise ValueError("snapshot_path does not match its content-addressed role.")
        object.__setattr__(self, "source_relative_path", source)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "md5", checksum)
        object.__setattr__(self, "snapshot_relative_path", relative)
        object.__setattr__(self, "snapshot_root_path", str(root))
        object.__setattr__(self, "snapshot_path", str(resolved))

    @contextmanager
    def open_verified(self) -> Iterator[BinaryIO]:
        descriptor = _open_regular_nofollow(Path(self.snapshot_path))
        with os.fdopen(descriptor, "rb") as handle:
            _verify_open_handle(
                handle,
                expected_size_bytes=self.size_bytes,
                expected_sha256=self.sha256,
                expected_md5=self.md5,
                label=f"Snapshot {self.snapshot_relative_path!r}",
            )
            yield handle

    def verify(self) -> None:
        with self.open_verified():
            pass

    def record(self) -> dict[str, object]:
        return {
            "source_relative_path": self.source_relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "md5": self.md5,
            "snapshot_relative_path": self.snapshot_relative_path,
            "role": self.role,
        }


def _publish_verified_stream(
    handle: BinaryIO,
    *,
    source_relative_path: str,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_md5: str | None,
    snapshot_root: Path,
    role: str,
) -> VerifiedSnapshot:
    digest = _validated_sha256(expected_sha256, name="snapshot expected SHA-256")
    checksum = _validated_md5(expected_md5, name="snapshot expected MD5")
    object_dir = snapshot_root / "objects" / "sha256"
    object_dir.mkdir(parents=True, exist_ok=True)
    _assert_existing_components_are_physical(object_dir)
    target = object_dir / digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".snapshot-",
        suffix=".tmp",
        dir=object_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            handle.seek(0)
            try:
                before = os.fstat(handle.fileno())
            except (AttributeError, OSError):
                before = None
            size_bytes, sha256, md5 = _stream_digest(
                handle,
                sink=sink,
                include_md5=checksum is not None,
            )
            sink.flush()
            os.fsync(sink.fileno())
            after = os.fstat(handle.fileno()) if before is not None else None
        if before is not None and after is not None:
            before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if before_identity != after_identity:
                raise RuntimeError(
                    f"Raw artifact {source_relative_path!r} changed while its descriptor was copied."
                )
        if size_bytes != expected_size_bytes:
            raise ValueError(
                f"Raw artifact size mismatch for {source_relative_path!r}: "
                f"expected {expected_size_bytes}, got {size_bytes}."
            )
        if sha256 != digest:
            raise ValueError(
                f"Raw artifact SHA-256 mismatch for {source_relative_path!r}: "
                f"expected {digest}, got {sha256}."
            )
        if checksum is not None and md5 != checksum:
            raise ValueError(
                f"Raw artifact official MD5 mismatch for {source_relative_path!r}: "
                f"expected {checksum}, got {md5}."
            )
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        else:
            _fsync_directory(object_dir)
        _unlink_with_retry(temporary)
        _make_read_only(target)
    finally:
        if temporary.exists():
            try:
                temporary.chmod(stat.S_IWRITE | stat.S_IREAD)
                temporary.unlink()
            except OSError:
                pass
    snapshot = VerifiedSnapshot(
        source_relative_path=source_relative_path,
        size_bytes=expected_size_bytes,
        sha256=digest,
        md5=checksum,
        snapshot_relative_path=f"objects/sha256/{digest}",
        role=role,
        snapshot_root_path=str(snapshot_root),
        snapshot_path=str(target),
    )
    snapshot.verify()
    return snapshot


def verified_snapshot(
    source: str | Path,
    *,
    source_relative_path: str,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_md5: str | None,
    snapshot_root: str | Path,
    role: str = RAW_ARTIFACT_SNAPSHOT_OBJECT_ROLE,
) -> VerifiedSnapshot:
    """Copy one source descriptor into an immutable content-addressed object."""

    root = _absolute_path(snapshot_root).resolve(strict=True)
    descriptor = _open_regular_nofollow(_absolute_path(source))
    with os.fdopen(descriptor, "rb") as handle:
        return _publish_verified_stream(
            handle,
            source_relative_path=source_relative_path,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            expected_md5=expected_md5,
            snapshot_root=root,
            role=role,
        )


@dataclass(frozen=True)
class DerivedLoaderFile:
    """A loader object extracted directly from one verified ZIP snapshot."""

    loader_relative_path: str
    size_bytes: int
    sha256: str
    derived_from_archive_relative_path: str
    archive_member: str
    snapshot_relative_path: str

    def __post_init__(self) -> None:
        loader_path = _validated_relative_path(self.loader_relative_path)
        archive_path = _validated_relative_path(self.derived_from_archive_relative_path)
        archive_member = _validated_archive_member(self.archive_member)
        snapshot_path = _validated_relative_path(self.snapshot_relative_path)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("Derived loader size_bytes must be a non-negative integer.")
        digest = _validated_sha256(self.sha256, name="derived loader sha256")
        if snapshot_path != f"objects/sha256/{digest}":
            raise ValueError("Derived loader snapshot path must be content-addressed.")
        object.__setattr__(self, "loader_relative_path", loader_path)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "derived_from_archive_relative_path", archive_path)
        object.__setattr__(self, "archive_member", archive_member)
        object.__setattr__(self, "snapshot_relative_path", snapshot_path)

    def record(self) -> dict[str, object]:
        return {
            "loader_relative_path": self.loader_relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "derived_from_archive_relative_path": (self.derived_from_archive_relative_path),
            "archive_member": self.archive_member,
            "snapshot_relative_path": self.snapshot_relative_path,
            "snapshot_role": "zip_member_derived_loader",
        }


@dataclass(frozen=True)
class ZipArchiveAudit:
    archive_relative_path: str
    member_count: int
    total_uncompressed_bytes: int
    crc_verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "archive_relative_path",
            _validated_relative_path(self.archive_relative_path),
        )
        if (
            isinstance(self.member_count, bool)
            or not isinstance(self.member_count, int)
            or self.member_count <= 0
        ):
            raise ValueError("ZIP audit member_count must be a positive integer.")
        if (
            isinstance(self.total_uncompressed_bytes, bool)
            or not isinstance(self.total_uncompressed_bytes, int)
            or self.total_uncompressed_bytes < 0
        ):
            raise ValueError("ZIP audit total_uncompressed_bytes is invalid.")
        if self.crc_verified is not True:
            raise ValueError("ZIP audit requires a complete CRC verification.")

    def record(self) -> dict[str, object]:
        return {
            "archive_relative_path": self.archive_relative_path,
            "member_count": self.member_count,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "crc_verified": True,
            "safety_limits": {
                "max_members": RAW_ARTIFACT_ZIP_MAX_MEMBERS,
                "max_member_bytes": RAW_ARTIFACT_ZIP_MAX_MEMBER_BYTES,
                "max_total_uncompressed_bytes": RAW_ARTIFACT_ZIP_MAX_TOTAL_BYTES,
                "max_expansion_ratio": RAW_ARTIFACT_ZIP_MAX_EXPANSION_RATIO,
            },
        }


@dataclass(frozen=True)
class _ZipMember:
    archive_relative_path: str
    archive_snapshot: VerifiedSnapshot
    member_name: str
    size_bytes: int


@dataclass(frozen=True)
class MoabbLoaderSpec:
    subject: int
    loader_relative_path: str
    source_kind: str
    source_relative_path: str | None = None
    archive_relative_path: str | None = None
    archive_member: str | None = None

    @classmethod
    def from_record(cls, value: object) -> MoabbLoaderSpec:
        if not isinstance(value, Mapping):
            raise ValueError("MOABB loader entries must be strict JSON records.")
        common = {"subject", "loader_relative_path", "source"}
        if set(value) != common or not isinstance(value["source"], Mapping):
            raise ValueError("MOABB loader entries must contain subject, loader path, and source.")
        subject = value["subject"]
        if isinstance(subject, bool) or not isinstance(subject, int) or subject <= 0:
            raise ValueError("MOABB loader subject must be a positive integer.")
        loader_path = _validated_relative_path(value["loader_relative_path"])
        if loader_path == RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT or loader_path.startswith(
            f"{RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT}/"
        ):
            raise ValueError(
                "MOABB loader_relative_path is relative to, and must not repeat, mne_data."
            )
        source = value["source"]
        kind = source.get("kind")
        if kind == "manifest_file" and set(source) == {"kind", "relative_path"}:
            return cls(
                subject=subject,
                loader_relative_path=loader_path,
                source_kind=kind,
                source_relative_path=_validated_relative_path(source["relative_path"]),
            )
        if kind == "zip_member" and set(source) == {
            "kind",
            "archive_relative_path",
            "archive_member",
        }:
            return cls(
                subject=subject,
                loader_relative_path=loader_path,
                source_kind=kind,
                archive_relative_path=_validated_relative_path(source["archive_relative_path"]),
                archive_member=_validated_archive_member(source["archive_member"]),
            )
        raise ValueError(
            "MOABB loader source must be one strict manifest_file or zip_member record."
        )


@dataclass(frozen=True)
class MoabbLoaderMaterialization:
    paths_by_subject: Mapping[int, tuple[Path, ...]]
    loader_snapshots: Mapping[str, VerifiedSnapshot]
    attestation: RawArtifactAttestation
    workspace_root_path: str
    expected_mne_data_root: str

    def __post_init__(self) -> None:
        if not self.paths_by_subject:
            raise ValueError("MOABB materialization must contain at least one subject.")
        workspace = _absolute_path(self.workspace_root_path).resolve(strict=True)
        mne_root = _absolute_path(self.expected_mne_data_root).resolve(strict=True)
        try:
            mne_root.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Controlled MNE root escapes the materialization workspace.") from exc
        for subject, paths in self.paths_by_subject.items():
            if isinstance(subject, bool) or not isinstance(subject, int) or subject <= 0:
                raise ValueError("Materialized MOABB subject keys must be positive integers.")
            if not paths:
                raise ValueError(f"MOABB subject {subject} has no materialized loader files.")
            for path in paths:
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(mne_root)
                except ValueError as exc:
                    raise ValueError(
                        "Materialized MOABB loader escapes controlled mne_data."
                    ) from exc
        materialized_relative = {
            path.resolve(strict=True).relative_to(mne_root).as_posix()
            for paths in self.paths_by_subject.values()
            for path in paths
        }
        if materialized_relative != set(self.loader_snapshots):
            raise ValueError("Materialized loader paths disagree with their snapshots.")

    def verify_loader_paths(self, paths: Sequence[str | Path]) -> None:
        mne_root = Path(self.expected_mne_data_root)
        if not paths:
            raise ValueError("MOABB returned no materialized loader paths.")
        for value in paths:
            path = _absolute_path(value)
            _assert_existing_components_are_physical(path)
            resolved = path.resolve(strict=True)
            try:
                relative = resolved.relative_to(mne_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    "MOABB attempted to read outside its controlled loader tree."
                ) from exc
            snapshot = self.loader_snapshots.get(relative)
            if snapshot is None:
                raise ValueError(f"MOABB attempted to read unmapped loader {relative!r}.")
            descriptor = _open_regular_nofollow(resolved)
            with os.fdopen(descriptor, "rb") as handle:
                _verify_open_handle(
                    handle,
                    expected_size_bytes=snapshot.size_bytes,
                    expected_sha256=snapshot.sha256,
                    expected_md5=None,
                    label=f"Materialized MOABB loader {relative!r}",
                )


@dataclass(frozen=True)
class RawArtifactManifest:
    dataset_class: str
    official_source: Mapping[str, Any]
    official_record: Mapping[str, Any]
    artifact_root_contract: Mapping[str, Any]
    files: tuple[RawArtifactFile, ...]
    schema: str = RAW_ARTIFACT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RAW_ARTIFACT_MANIFEST_SCHEMA:
            raise ValueError(
                f"Raw artifact manifest schema must be {RAW_ARTIFACT_MANIFEST_SCHEMA!r}."
            )
        if not isinstance(self.dataset_class, str) or not self.dataset_class.strip():
            raise ValueError("dataset_class must be a non-empty string.")
        if self.dataset_class != self.dataset_class.strip():
            raise ValueError("dataset_class must not contain surrounding whitespace.")
        if not isinstance(self.official_source, Mapping) or not self.official_source:
            raise ValueError("official_source must be a non-empty JSON record.")
        if not isinstance(self.official_record, Mapping) or not self.official_record:
            raise ValueError("official_record must be a non-empty JSON record.")
        if not isinstance(self.artifact_root_contract, Mapping) or not self.artifact_root_contract:
            raise ValueError("artifact_root_contract must be a non-empty JSON record.")
        source = _canonical_json_value(self.official_source, name="official_source")
        official_record = _canonical_json_value(self.official_record, name="official_record")
        root_contract = _canonical_json_value(
            self.artifact_root_contract, name="artifact_root_contract"
        )
        if root_contract.get("path_semantics") != RAW_ARTIFACT_PATH_SEMANTICS:
            raise ValueError(
                "artifact_root_contract.path_semantics must declare the supported root contract."
            )
        expected_mne_data_root = _validated_root_relative_path(
            root_contract.get("expected_mne_data_root")
        )
        root_contract["expected_mne_data_root"] = expected_mne_data_root
        mne_dataset_path_key = root_contract.get("mne_dataset_path_key")
        if (
            not isinstance(mne_dataset_path_key, str)
            or _MNE_DATASET_PATH_KEY_PATTERN.fullmatch(mne_dataset_path_key) is None
        ):
            raise ValueError(
                "artifact_root_contract.mne_dataset_path_key must match "
                "MNE_DATASETS_[A-Z0-9_]+_PATH."
            )
        if not self.files or not all(isinstance(item, RawArtifactFile) for item in self.files):
            raise ValueError("files must contain at least one raw artifact file record.")
        paths: dict[str, str] = {}
        for item in self.files:
            collision_key = item.relative_path.casefold()
            if collision_key in paths:
                raise ValueError(
                    "Raw artifact relative paths must be unique across case-insensitive filesystems: "
                    f"{paths[collision_key]!r} and {item.relative_path!r}."
                )
            paths[collision_key] = item.relative_path
        object.__setattr__(self, "official_source", source)
        object.__setattr__(self, "official_record", official_record)
        object.__setattr__(self, "artifact_root_contract", root_contract)
        object.__setattr__(
            self, "files", tuple(sorted(self.files, key=lambda item: item.relative_path))
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_class": self.dataset_class,
            "official_source": _canonical_json_value(self.official_source, name="official_source"),
            "official_record": _canonical_json_value(self.official_record, name="official_record"),
            "artifact_root_contract": _canonical_json_value(
                self.artifact_root_contract, name="artifact_root_contract"
            ),
            "files": [item.record() for item in self.files],
        }

    def digest(self) -> str:
        """Return the canonical manifest SHA-256 without a self-referential field."""

        return hashlib.sha256(_canonical_json_bytes(self.record())).hexdigest()

    @classmethod
    def from_record(cls, value: object) -> RawArtifactManifest:
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise ValueError("Raw artifact manifest must be a strict JSON mapping.")
        files = value["files"]
        if not isinstance(files, list):
            raise ValueError("Raw artifact manifest files must be a JSON list.")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            dataset_class=value["dataset_class"],  # type: ignore[arg-type]
            official_source=value["official_source"],  # type: ignore[arg-type]
            official_record=value["official_record"],  # type: ignore[arg-type]
            artifact_root_contract=value["artifact_root_contract"],  # type: ignore[arg-type]
            files=tuple(RawArtifactFile.from_record(item) for item in files),
        )


@dataclass(frozen=True)
class RawArtifactAttestation:
    """Typed result of verifying every manifest file against physical bytes."""

    dataset_class: str
    manifest_path: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    official_source: Mapping[str, Any]
    official_record: Mapping[str, Any]
    artifact_root_contract: Mapping[str, Any]
    artifact_root_path: str
    expected_mne_data_root: str
    snapshot_root_path: str
    cache_workspace_root: str
    verified_files: tuple[RawArtifactFile, ...]
    verified_snapshots: tuple[VerifiedSnapshot, ...]
    actual_loader_files: tuple[RawArtifactFile, ...] = ()
    derived_loader_files: tuple[DerivedLoaderFile, ...] = ()
    zip_archive_audits: tuple[ZipArchiveAudit, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_class, str) or not self.dataset_class.strip():
            raise ValueError("attestation dataset_class must be non-empty.")
        if not isinstance(self.manifest_path, str) or not self.manifest_path.strip():
            raise ValueError("attestation manifest_path must be non-empty.")
        object.__setattr__(
            self,
            "manifest_sha256",
            _validated_sha256(self.manifest_sha256, name="attestation manifest_sha256"),
        )
        if (
            isinstance(self.file_count, bool)
            or not isinstance(self.file_count, int)
            or self.file_count <= 0
        ):
            raise ValueError("attestation file_count must be a positive integer.")
        if (
            isinstance(self.total_bytes, bool)
            or not isinstance(self.total_bytes, int)
            or self.total_bytes < 0
        ):
            raise ValueError("attestation total_bytes must be a non-negative integer.")
        source = _canonical_json_value(self.official_source, name="attestation official_source")
        official_record = _canonical_json_value(
            self.official_record, name="attestation official_record"
        )
        root_contract = _canonical_json_value(
            self.artifact_root_contract, name="attestation artifact_root_contract"
        )
        if not isinstance(source, dict) or not source:
            raise ValueError("attestation official_source must be a non-empty JSON record.")
        if not isinstance(official_record, dict) or not official_record:
            raise ValueError("attestation official_record must be a non-empty JSON record.")
        if (
            not isinstance(root_contract, dict)
            or root_contract.get("path_semantics") != RAW_ARTIFACT_PATH_SEMANTICS
        ):
            raise ValueError("attestation artifact_root_contract is invalid.")
        expected_relative = _validated_root_relative_path(
            root_contract.get("expected_mne_data_root")
        )
        root_contract["expected_mne_data_root"] = expected_relative
        mne_dataset_path_key = root_contract.get("mne_dataset_path_key")
        if (
            not isinstance(mne_dataset_path_key, str)
            or _MNE_DATASET_PATH_KEY_PATTERN.fullmatch(mne_dataset_path_key) is None
        ):
            raise ValueError("attestation mne_dataset_path_key is invalid.")
        artifact_root = Path(self.artifact_root_path).resolve(strict=True)
        expected_root = Path(self.expected_mne_data_root).resolve(strict=True)
        snapshot_root = Path(self.snapshot_root_path).resolve(strict=True)
        cache_workspace = Path(self.cache_workspace_root).resolve(strict=True)
        if not artifact_root.is_dir() or not expected_root.is_dir():
            raise ValueError("attestation artifact roots must be physical directories.")
        try:
            expected_root.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                "expected_mne_data_root must stay beneath artifact_root_path."
            ) from exc
        try:
            snapshot_root.relative_to(cache_workspace)
        except ValueError as exc:
            raise ValueError("snapshot_root must stay beneath cache_workspace_root.") from exc
        if not self.verified_files or not all(
            isinstance(item, RawArtifactFile) for item in self.verified_files
        ):
            raise ValueError("attestation verified_files must contain physical file records.")
        if self.file_count != len(self.verified_files):
            raise ValueError("attestation file_count disagrees with verified_files.")
        if self.total_bytes != sum(item.size_bytes for item in self.verified_files):
            raise ValueError("attestation total_bytes disagrees with verified_files.")
        if len(self.verified_snapshots) != len(self.verified_files):
            raise ValueError("attestation requires one immutable snapshot per manifest file.")
        expected_by_path = {item.relative_path: item for item in self.verified_files}
        snapshots_by_path = {item.source_relative_path: item for item in self.verified_snapshots}
        if set(snapshots_by_path) != set(expected_by_path):
            raise ValueError("attestation snapshot sources disagree with manifest files.")
        for relative_path, snapshot in snapshots_by_path.items():
            expected = expected_by_path[relative_path]
            if (
                snapshot.size_bytes != expected.size_bytes
                or snapshot.sha256 != expected.local_sha256
                or snapshot.md5 != expected.official_md5
                or Path(snapshot.snapshot_root_path) != snapshot_root
            ):
                raise ValueError(f"Snapshot disagrees with manifest file {relative_path!r}.")
        if not all(isinstance(item, RawArtifactFile) for item in self.actual_loader_files):
            raise ValueError("attestation actual_loader_files must contain file records.")
        if not all(isinstance(item, DerivedLoaderFile) for item in self.derived_loader_files):
            raise ValueError("attestation derived_loader_files must contain derived file records.")
        if not all(isinstance(item, ZipArchiveAudit) for item in self.zip_archive_audits):
            raise ValueError("attestation zip_archive_audits must contain audit records.")
        audit_paths = [item.archive_relative_path for item in self.zip_archive_audits]
        if len(set(audit_paths)) != len(audit_paths):
            raise ValueError("ZIP archive audits must be unique by manifest path.")
        verified_paths = {item.relative_path for item in self.verified_files}
        if not set(audit_paths) <= verified_paths:
            raise ValueError("ZIP archive audits must refer to verified manifest files.")
        derived_archive_paths = {
            item.derived_from_archive_relative_path for item in self.derived_loader_files
        }
        if not derived_archive_paths <= set(audit_paths):
            raise ValueError("Every derived loader ZIP source requires a CRC audit.")
        loader_paths = [item.relative_path for item in self.actual_loader_files] + [
            item.loader_relative_path for item in self.derived_loader_files
        ]
        if len({value.casefold() for value in loader_paths}) != len(loader_paths):
            raise ValueError("Attested loader file paths must be unique.")
        object.__setattr__(self, "official_source", source)
        object.__setattr__(self, "official_record", official_record)
        object.__setattr__(self, "artifact_root_contract", root_contract)
        object.__setattr__(self, "artifact_root_path", str(artifact_root))
        object.__setattr__(self, "expected_mne_data_root", str(expected_root))
        object.__setattr__(self, "snapshot_root_path", str(snapshot_root))
        object.__setattr__(self, "cache_workspace_root", str(cache_workspace))
        object.__setattr__(
            self,
            "verified_files",
            tuple(sorted(self.verified_files, key=lambda item: item.relative_path)),
        )
        object.__setattr__(
            self,
            "verified_snapshots",
            tuple(sorted(self.verified_snapshots, key=lambda item: item.source_relative_path)),
        )
        object.__setattr__(
            self,
            "actual_loader_files",
            tuple(sorted(self.actual_loader_files, key=lambda item: item.relative_path)),
        )
        object.__setattr__(
            self,
            "derived_loader_files",
            tuple(
                sorted(
                    self.derived_loader_files,
                    key=lambda item: item.loader_relative_path,
                )
            ),
        )
        object.__setattr__(
            self,
            "zip_archive_audits",
            tuple(
                sorted(
                    self.zip_archive_audits,
                    key=lambda item: item.archive_relative_path,
                )
            ),
        )

    def assert_dataset_class(self, dataset_class: str) -> None:
        if self.dataset_class != dataset_class:
            raise ValueError(
                "Raw artifact attestation dataset_class mismatch: "
                f"expected {dataset_class!r}, got {self.dataset_class!r}."
            )

    @property
    def mne_dataset_path_key(self) -> str:
        return str(self.artifact_root_contract["mne_dataset_path_key"])

    def snapshot_for(self, relative_path: str) -> VerifiedSnapshot:
        normalized = _validated_relative_path(relative_path)
        matches = [
            item for item in self.verified_snapshots if item.source_relative_path == normalized
        ]
        if len(matches) != 1:
            raise ValueError(f"No unique verified snapshot exists for {normalized!r}.")
        matches[0].verify()
        return matches[0]

    def source_provenance_record(self, relative_path: str) -> dict[str, object]:
        snapshot = self.snapshot_for(relative_path)
        return {
            "raw_artifact_manifest_schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
            "raw_artifact_manifest_role": "explicit_verified_input_manifest",
            "raw_artifact_manifest_sha256": self.manifest_sha256,
            "raw_artifact_official_source": _canonical_json_value(
                self.official_source, name="attestation official_source"
            ),
            "raw_artifact_official_record": _canonical_json_value(
                self.official_record, name="attestation official_record"
            ),
            "raw_artifact_snapshot": snapshot.record(),
        }

    def materialize_moabb_loaders(self, subjects: Sequence[int]) -> MoabbLoaderMaterialization:
        """Extract only manifest-mapped loaders from verified objects."""

        selected = tuple(subjects)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("MOABB materialization needs unique selected subjects.")
        specs = _moabb_loader_specs(self.artifact_root_contract, selected)
        workspace = Path(self.snapshot_root_path) / "workspaces" / self.manifest_sha256
        mne_root = workspace / RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT
        mne_root.mkdir(parents=True, exist_ok=True)
        _assert_existing_components_are_physical(mne_root)

        manifest_files = {item.relative_path: item for item in self.verified_files}
        direct: dict[str, RawArtifactFile] = {}
        derived: dict[str, DerivedLoaderFile] = {}
        audits: dict[str, ZipArchiveAudit] = {}
        paths_by_subject: dict[int, list[Path]] = {subject: [] for subject in selected}
        loader_snapshots: dict[str, VerifiedSnapshot] = {}
        destinations: set[str] = set()
        zip_specs: dict[str, list[MoabbLoaderSpec]] = {}
        for spec in specs:
            if spec.source_kind == "zip_member":
                assert spec.archive_relative_path is not None
                zip_specs.setdefault(spec.archive_relative_path, []).append(spec)
        for archive_relative_path, archive_specs in zip_specs.items():
            expected = manifest_files.get(archive_relative_path)
            if expected is None:
                raise ValueError(
                    f"MOABB mapping references unmanifested ZIP {archive_relative_path!r}."
                )
            archive_snapshot = self.snapshot_for(archive_relative_path)
            members, member_count, total_uncompressed = _inspect_zip_archive(
                expected, archive_snapshot
            )
            index = {item.member_name: item for item in members}
            requested: list[tuple[_ZipMember, str]] = []
            for spec in archive_specs:
                assert spec.archive_member is not None
                member = index.get(spec.archive_member)
                if member is None:
                    raise ValueError(
                        f"Mapped ZIP member {spec.archive_member!r} is absent from "
                        f"{archive_relative_path!r}."
                    )
                requested.append((member, spec.loader_relative_path))
            extracted, audit = _snapshot_zip_members(
                expected,
                archive_snapshot,
                requested=requested,
                member_count=member_count,
                total_uncompressed_bytes=total_uncompressed,
                snapshot_root=Path(self.snapshot_root_path),
            )
            loader_snapshots.update(extracted)
            audits[archive_relative_path] = audit

        for spec in specs:
            collision_key = spec.loader_relative_path.casefold()
            if collision_key in destinations:
                raise ValueError(
                    f"MOABB loader mapping repeats loader_relative_path "
                    f"{spec.loader_relative_path!r}."
                )
            destinations.add(collision_key)
            if spec.source_kind == "manifest_file":
                assert spec.source_relative_path is not None
                expected = manifest_files.get(spec.source_relative_path)
                if expected is None:
                    raise ValueError(
                        f"MOABB mapping references unmanifested file {spec.source_relative_path!r}."
                    )
                source_snapshot = self.snapshot_for(expected.relative_path)
                direct[expected.relative_path] = expected
            else:
                assert spec.archive_relative_path is not None
                assert spec.archive_member is not None
                source_snapshot = loader_snapshots[spec.loader_relative_path]
                derived[spec.loader_relative_path] = DerivedLoaderFile(
                    loader_relative_path=spec.loader_relative_path,
                    size_bytes=source_snapshot.size_bytes,
                    sha256=source_snapshot.sha256,
                    derived_from_archive_relative_path=spec.archive_relative_path,
                    archive_member=spec.archive_member,
                    snapshot_relative_path=source_snapshot.snapshot_relative_path,
                )
            destination = mne_root.joinpath(*PurePosixPath(spec.loader_relative_path).parts)
            _materialize_snapshot(
                source_snapshot,
                destination=destination,
                workspace_root=workspace,
            )
            loader_snapshots[spec.loader_relative_path] = source_snapshot
            paths_by_subject[spec.subject].append(destination.resolve(strict=True))

        bound = replace(
            self,
            actual_loader_files=tuple(direct.values()),
            derived_loader_files=tuple(derived.values()),
            zip_archive_audits=tuple(audits.values()),
        )
        return MoabbLoaderMaterialization(
            paths_by_subject={key: tuple(value) for key, value in paths_by_subject.items()},
            loader_snapshots=loader_snapshots,
            attestation=bound,
            workspace_root_path=str(workspace),
            expected_mne_data_root=str(mne_root),
        )

    def provenance_record(self) -> dict[str, object]:
        if not self.actual_loader_files and not self.derived_loader_files:
            raise ValueError(
                "Raw artifact provenance requires re-attested physical MOABB loader files."
            )
        return {
            "raw_artifact_manifest_schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
            "raw_artifact_manifest_role": "explicit_verified_input_manifest",
            "raw_artifact_manifest_sha256": self.manifest_sha256,
            "raw_artifact_file_count": self.file_count,
            "raw_artifact_total_bytes": self.total_bytes,
            "raw_artifact_mne_dataset_path_key": self.mne_dataset_path_key,
            "raw_artifact_snapshots": [item.record() for item in self.verified_snapshots],
            "raw_artifact_loader_resolver": {
                "schema": "n2p3_moabb_loader_mapping/1",
                "upstream_data_path_bypassed": True,
                "workspace_role": RAW_ARTIFACT_MOABB_WORKSPACE_ROLE,
            },
            "raw_artifact_official_source": _canonical_json_value(
                self.official_source, name="attestation official_source"
            ),
            "raw_artifact_official_record": _canonical_json_value(
                self.official_record, name="attestation official_record"
            ),
            "raw_artifact_root_contract": _canonical_json_value(
                self.artifact_root_contract, name="attestation artifact_root_contract"
            ),
            "raw_artifact_actual_loader_files": [
                {
                    "relative_path": item.relative_path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.local_sha256,
                    "derived_from": "verified_manifest_file",
                }
                for item in self.actual_loader_files
            ],
            "raw_artifact_derived_loader_files": [
                item.record() for item in self.derived_loader_files
            ],
            "raw_artifact_zip_archive_audits": [item.record() for item in self.zip_archive_audits],
        }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Raw artifact manifest contains duplicate JSON key {key!r}.")
        output[key] = value
    return output


def load_raw_artifact_manifest(path: str | Path) -> RawArtifactManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Raw artifact manifest contains invalid number {value!r}.")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Raw artifact manifest is not valid UTF-8 JSON: {manifest_path}."
        ) from exc
    return RawArtifactManifest.from_record(payload)


def _moabb_loader_specs(
    root_contract: Mapping[str, Any], subjects: Sequence[int]
) -> tuple[MoabbLoaderSpec, ...]:
    mapping = root_contract.get("moabb_loader_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != {"schema", "entries"}:
        raise ValueError(
            "artifact_root_contract.moabb_loader_mapping must contain schema and entries."
        )
    if mapping["schema"] != "n2p3_moabb_loader_mapping/1":
        raise ValueError("MOABB loader mapping schema must be n2p3_moabb_loader_mapping/1.")
    entries = mapping["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("MOABB loader mapping entries must be a non-empty list.")
    parsed = tuple(MoabbLoaderSpec.from_record(item) for item in entries)
    loader_paths = [item.loader_relative_path.casefold() for item in parsed]
    if len(loader_paths) != len(set(loader_paths)):
        raise ValueError("MOABB loader mapping repeats loader_relative_path.")
    selected = set(subjects)
    invalid = [value for value in selected if isinstance(value, bool) or not isinstance(value, int)]
    if invalid:
        raise ValueError("MOABB selected subjects must be integers.")
    available = {item.subject for item in parsed}
    missing = selected - available
    if missing:
        raise ValueError(f"MOABB loader mapping is missing subjects {sorted(missing)}.")
    return tuple(item for item in parsed if item.subject in selected)


def _materialize_snapshot(
    snapshot: VerifiedSnapshot,
    *,
    destination: Path,
    workspace_root: Path,
) -> None:
    workspace = _absolute_path(workspace_root).resolve(strict=True)
    target = _absolute_path(destination)
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Materialized loader path escapes its controlled workspace.") from exc
    _assert_existing_components_are_physical(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_existing_components_are_physical(target.parent)
    if target.exists():
        descriptor = _open_regular_nofollow(target)
        with os.fdopen(descriptor, "rb") as handle:
            _verify_open_handle(
                handle,
                expected_size_bytes=snapshot.size_bytes,
                expected_sha256=snapshot.sha256,
                expected_md5=None,
                label=f"Materialized loader {target.name!r}",
            )
        _make_read_only(target)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".materialize-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with snapshot.open_verified() as source, os.fdopen(descriptor, "wb") as sink:
            size_bytes, sha256, _ = _stream_digest(source, sink=sink, include_md5=False)
            sink.flush()
            os.fsync(sink.fileno())
        if size_bytes != snapshot.size_bytes or sha256 != snapshot.sha256:
            raise RuntimeError("Verified snapshot changed while it was materialized.")
        try:
            os.link(temporary, target)
        except FileExistsError:
            descriptor = _open_regular_nofollow(target)
            with os.fdopen(descriptor, "rb") as handle:
                _verify_open_handle(
                    handle,
                    expected_size_bytes=snapshot.size_bytes,
                    expected_sha256=snapshot.sha256,
                    expected_md5=None,
                    label=f"Concurrent materialized loader {target.name!r}",
                )
        else:
            _fsync_directory(target.parent)
        _unlink_with_retry(temporary)
        _make_read_only(target)
    finally:
        if temporary.exists():
            try:
                temporary.chmod(stat.S_IWRITE | stat.S_IREAD)
                temporary.unlink()
            except OSError:
                pass


def _inspect_zip_archive(
    expected: RawArtifactFile,
    source: VerifiedSnapshot,
) -> tuple[tuple[_ZipMember, ...], int, int]:
    try:
        with source.open_verified() as handle:
            with zipfile.ZipFile(handle) as archive:
                infos = archive.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"Manifest ZIP is unreadable: {expected.relative_path!r}.") from exc
    if not infos:
        raise ValueError(f"Manifest ZIP is empty: {expected.relative_path!r}.")
    if len(infos) > RAW_ARTIFACT_ZIP_MAX_MEMBERS:
        raise ValueError(
            f"Manifest ZIP exceeds the member-count safety limit: {expected.relative_path!r}."
        )
    seen: dict[str, str] = {}
    total_uncompressed_bytes = 0
    members: list[_ZipMember] = []
    for info in infos:
        member_name = _validated_archive_member(info.filename)
        collision_key = member_name.casefold()
        previous = seen.get(collision_key)
        if previous is not None:
            raise ValueError(
                "Manifest ZIP members must be unique across case-insensitive filesystems: "
                f"{previous!r} and {info.filename!r}."
            )
        seen[collision_key] = info.filename
        if info.flag_bits & 0x1:
            raise ValueError(
                f"Encrypted ZIP members are not allowed: {expected.relative_path!r} "
                f"member {info.filename!r}."
            )
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if stat.S_ISLNK(unix_mode) or file_type not in {
            0,
            stat.S_IFREG,
            stat.S_IFDIR,
        }:
            raise ValueError(f"ZIP member is not a regular file/directory: {info.filename!r}.")
        if info.file_size < 0 or info.file_size > RAW_ARTIFACT_ZIP_MAX_MEMBER_BYTES:
            raise ValueError(f"ZIP member exceeds the size safety limit: {info.filename!r}.")
        if info.is_dir():
            continue
        total_uncompressed_bytes += info.file_size
        if total_uncompressed_bytes > RAW_ARTIFACT_ZIP_MAX_TOTAL_BYTES:
            raise ValueError(
                f"Manifest ZIP exceeds the total extraction safety limit: "
                f"{expected.relative_path!r}."
            )
        members.append(
            _ZipMember(
                archive_relative_path=expected.relative_path,
                archive_snapshot=source,
                member_name=member_name,
                size_bytes=info.file_size,
            )
        )
    expansion_ratio = total_uncompressed_bytes / max(expected.size_bytes, 1)
    if expansion_ratio > RAW_ARTIFACT_ZIP_MAX_EXPANSION_RATIO:
        raise ValueError(
            f"Manifest ZIP exceeds the expansion-ratio safety limit: {expected.relative_path!r}."
        )
    return tuple(members), len(infos), total_uncompressed_bytes


def _publish_content_stream(
    handle: BinaryIO,
    *,
    source_relative_path: str,
    expected_size_bytes: int,
    snapshot_root: Path,
) -> VerifiedSnapshot:
    object_dir = snapshot_root / "objects" / "sha256"
    object_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".snapshot-", suffix=".tmp", dir=object_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            size_bytes, digest, _ = _stream_digest(handle, sink=sink, include_md5=False)
            sink.flush()
            os.fsync(sink.fileno())
        if size_bytes != expected_size_bytes:
            raise ValueError(
                f"ZIP member size mismatch for {source_relative_path!r}: "
                f"expected {expected_size_bytes}, got {size_bytes}."
            )
        target = object_dir / digest
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        else:
            _fsync_directory(object_dir)
        _unlink_with_retry(temporary)
        _make_read_only(target)
    finally:
        if temporary.exists():
            try:
                temporary.chmod(stat.S_IWRITE | stat.S_IREAD)
                temporary.unlink()
            except OSError:
                pass
    snapshot = VerifiedSnapshot(
        source_relative_path=source_relative_path,
        size_bytes=expected_size_bytes,
        sha256=digest,
        md5=None,
        snapshot_relative_path=f"objects/sha256/{digest}",
        role="zip_member_derived_loader",
        snapshot_root_path=str(snapshot_root),
        snapshot_path=str(target),
    )
    snapshot.verify()
    return snapshot


def _snapshot_zip_members(
    expected: RawArtifactFile,
    source: VerifiedSnapshot,
    *,
    requested: Sequence[tuple[_ZipMember, str]],
    member_count: int,
    total_uncompressed_bytes: int,
    snapshot_root: Path,
) -> tuple[dict[str, VerifiedSnapshot], ZipArchiveAudit]:
    requested_by_member: dict[str, list[str]] = {}
    member_records: dict[str, _ZipMember] = {}
    for member, loader_relative_path in requested:
        requested_by_member.setdefault(member.member_name, []).append(loader_relative_path)
        member_records[member.member_name] = member
    output: dict[str, VerifiedSnapshot] = {}
    try:
        with source.open_verified() as snapshot_handle:
            with zipfile.ZipFile(snapshot_handle) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(
                        f"Manifest ZIP has a bad CRC member {bad_member!r}: "
                        f"{expected.relative_path!r}."
                    )
                for member_name, loader_paths in requested_by_member.items():
                    member = member_records[member_name]
                    with archive.open(member_name, "r") as handle:
                        base = _publish_content_stream(
                            handle,
                            source_relative_path=loader_paths[0],
                            expected_size_bytes=member.size_bytes,
                            snapshot_root=snapshot_root,
                        )
                    for loader_path in loader_paths:
                        output[loader_path] = replace(base, source_relative_path=loader_path)
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise ValueError(
            f"Manifest ZIP failed CRC verification or extraction: {expected.relative_path!r}."
        ) from exc
    return output, ZipArchiveAudit(
        archive_relative_path=expected.relative_path,
        member_count=member_count,
        total_uncompressed_bytes=total_uncompressed_bytes,
    )


def verify_raw_artifact_manifest(
    manifest_path: str | Path,
    artifact_root: str | Path,
    *,
    snapshot_root: str | Path,
    cache_workspace_root: str | Path,
    expected_dataset_class: str,
) -> RawArtifactAttestation:
    """Snapshot every declared file from one open descriptor before it can be used."""

    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = load_raw_artifact_manifest(manifest_file)
    if manifest.dataset_class != expected_dataset_class:
        raise ValueError(
            "Raw artifact manifest dataset_class mismatch: "
            f"expected {expected_dataset_class!r}, got {manifest.dataset_class!r}."
        )
    root_input = _absolute_path(artifact_root)
    _assert_existing_components_are_physical(root_input)
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    snapshot, cache_workspace = _prepare_snapshot_root(
        snapshot_root,
        cache_workspace_root=cache_workspace_root,
    )
    expected_relative = manifest.artifact_root_contract["expected_mne_data_root"]
    expected_mne_data_root = root.joinpath(*PurePosixPath(str(expected_relative)).parts).resolve(
        strict=True
    )
    try:
        expected_mne_data_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("expected_mne_data_root escapes the artifact root.") from exc
    if not expected_mne_data_root.is_dir():
        raise NotADirectoryError(expected_mne_data_root)
    total_bytes = 0
    snapshots: list[VerifiedSnapshot] = []
    for expected in manifest.files:
        candidate = root.joinpath(*PurePosixPath(expected.relative_path).parts)
        _assert_existing_components_are_physical(candidate)
        try:
            source = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Raw artifact is missing: {expected.relative_path!r} beneath {root}."
            ) from exc
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Raw artifact path escapes the declared root: {expected.relative_path!r}."
            ) from exc
        source_snapshot = verified_snapshot(
            source,
            source_relative_path=expected.relative_path,
            expected_size_bytes=expected.size_bytes,
            expected_sha256=expected.local_sha256,
            expected_md5=expected.official_md5,
            snapshot_root=snapshot,
        )
        snapshots.append(source_snapshot)
        total_bytes += source_snapshot.size_bytes
    return RawArtifactAttestation(
        dataset_class=manifest.dataset_class,
        manifest_path=str(manifest_file),
        manifest_sha256=manifest.digest(),
        file_count=len(manifest.files),
        total_bytes=total_bytes,
        official_source=manifest.official_source,
        official_record=manifest.official_record,
        artifact_root_contract=manifest.artifact_root_contract,
        artifact_root_path=str(root),
        expected_mne_data_root=str(expected_mne_data_root),
        snapshot_root_path=str(snapshot),
        cache_workspace_root=str(cache_workspace),
        verified_files=manifest.files,
        verified_snapshots=tuple(snapshots),
    )


__all__ = [
    "RAW_ARTIFACT_EXPECTED_MNE_DATA_ROOT",
    "RAW_ARTIFACT_MANIFEST_SCHEMA",
    "RAW_ARTIFACT_PATH_SEMANTICS",
    "RAW_ARTIFACT_ZIP_MAX_EXPANSION_RATIO",
    "RAW_ARTIFACT_ZIP_MAX_MEMBER_BYTES",
    "RAW_ARTIFACT_ZIP_MAX_MEMBERS",
    "RAW_ARTIFACT_ZIP_MAX_TOTAL_BYTES",
    "DerivedLoaderFile",
    "MoabbLoaderMaterialization",
    "MoabbLoaderSpec",
    "RawArtifactAttestation",
    "RawArtifactFile",
    "RawArtifactManifest",
    "VerifiedSnapshot",
    "ZipArchiveAudit",
    "load_raw_artifact_manifest",
    "verified_snapshot",
    "verify_raw_artifact_manifest",
]
