"""Generic MOABB P300 adapter producing the universal EpochDataset contract."""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
import re
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, MethodType
from typing import Any
from uuid import uuid4

import mne
import moabb
import numpy as np
import pandas as pd

from data.channel import DEFAULT_MONTAGE, build_channel_identity
from data.dataset import annotation_events_and_labels, build_subject
from data.epochs import (
    P300_PERFORMANCE_PREPROCESSING,
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    save_epoch_dataset,
)
from data.events import observed_only_timeline, selection_group_id
from data.preprocess import apply_trial_baseline
from data.raw_artifacts import MoabbLoaderMaterialization, RawArtifactAttestation

ATTESTED_MOABB_RESOLVER_SCHEMA = "n2p3_attested_moabb_resolver/1"
ATTESTED_MOABB_OFFLINE_GUARD_SCHEMA = "python_socket_offline_guard/1"
ATTESTED_MOABB_LOADER_GUARD_SCHEMA = "python_audit_read_only_loader_guard/1"
ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA = "n2p3_private_moabb_parser_snapshot/1"
MOABB_SUBJECT_EXECUTION_SCHEMA = "n2p3_moabb_subject_execution/1"
MOABB_SUBJECT_JOURNAL_SCHEMA = "n2p3_moabb_subject_journal/1"
VALIDATED_MOABB_VERSION = "1.6.1"
MOABB_DOWNLOAD_PROVIDER_ENV = "MOABB_DOWNLOAD_PROVIDER"
_WORKER_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

CANDIDATE_METADATA_ALIASES = (
    "candidate_id",
    "stimulus_candidate_id",
    "command_id",
)
TARGET_CANDIDATE_METADATA_ALIASES = (
    "target_candidate_id",
    "selected_candidate_id",
    "intended_candidate_id",
)
REPETITION_METADATA_ALIASES = (
    "repetition_index",
    "repetition_id",
    "sequence_index",
)

_MOABB_IO_SCOPE_LOCK = threading.RLock()
_MOABB_LOADER_AUDIT_LOCK = threading.Lock()
_MOABB_LOADER_AUDIT_STATE = threading.local()
_MOABB_LOADER_AUDIT_INSTALLED = False
_WORKER_THREADPOOL_LIMITER: object | None = None


class _AttestedMoabbNetworkAccessError(RuntimeError):
    """Raised when trusted parser code reaches guarded Python socket APIs."""


class _AttestedMoabbLoaderMutationError(RuntimeError):
    """Raised when an attested parser attempts to mutate its filesystem view."""


class _ShardCleanupError(RuntimeError):
    """Raised when a controlled shard workspace cannot be removed completely."""

    def __init__(self, workspace: Path, issue: Mapping[str, str]) -> None:
        self.workspace = workspace
        self.issue = dict(issue)
        super().__init__(
            f"MOABB shard cleanup failed at {workspace.name!r}: "
            f"{self.issue['error_type']}: {self.issue['message']}"
        )


@dataclass
class _AttestedMoabbDataPathBinding:
    paths_by_subject: dict[int, tuple[Path, ...]]
    calls: list[int] = field(default_factory=list)
    parser_call_start: int | None = None

    def begin_parser_phase(self) -> None:
        self.parser_call_start = len(self.calls)

    def parser_calls(self) -> tuple[int, ...]:
        if self.parser_call_start is None:
            raise RuntimeError("Attested MOABB parser phase was not started.")
        return tuple(self.calls[self.parser_call_start :])

    def assert_parser_consumed(self, subjects: Sequence[int]) -> None:
        requested = set(subjects)
        consumed = set(self.parser_calls())
        if consumed != requested:
            raise RuntimeError(
                "MOABB parser did not consume the complete attested data_path mapping: "
                f"requested={sorted(requested)}, consumed={sorted(consumed)}."
            )


@dataclass(frozen=True)
class _MoabbRuntimeResolver:
    declared_nemar_id: str | None
    requested_download_provider: str
    strict_base_dataset_contract: bool
    subjects: tuple[int, ...]

    def record(self) -> dict[str, object]:
        return {
            "schema": ATTESTED_MOABB_RESOLVER_SCHEMA,
            "moabb_version": VALIDATED_MOABB_VERSION,
            "declared_nemar_id": self.declared_nemar_id,
            "requested_download_provider": self.requested_download_provider,
            "effective_download_provider": "upstream",
            "prefetch_bypassed": self.strict_base_dataset_contract,
            "upstream_data_path_bypassed": True,
            "parser_data_path_verified": self.strict_base_dataset_contract,
            "subjects": list(self.subjects),
            "parser_trust_boundary": {
                "dependency": f"moabb=={VALIDATED_MOABB_VERSION}",
                "role": "trusted_pinned_parser_not_process_sandbox",
            },
            "parsed_byte_authority": ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA,
            "network_guard": {
                "schema": ATTESTED_MOABB_OFFLINE_GUARD_SCHEMA,
                "role": "defense_in_depth_not_process_network_sandbox",
            },
            "parser_snapshot_schema": ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA,
            "python_audit_guard": {
                "schema": ATTESTED_MOABB_LOADER_GUARD_SCHEMA,
                "role": "defense_in_depth_not_parsed_byte_attestation",
            },
        }


@dataclass(frozen=True)
class _MoabbLoaderPathResolution:
    subject: int
    reported_path: Path
    resolved_loader_path: Path
    appendmat_resolution: bool

    def record(self, artifact_root: Path) -> dict[str, object]:
        def display(path: Path) -> str:
            absolute = path.resolve(strict=False)
            try:
                return absolute.relative_to(artifact_root).as_posix()
            except ValueError:
                return str(absolute)

        return {
            "subject": self.subject,
            "reported_path": display(self.reported_path),
            "resolved_loader_path": display(self.resolved_loader_path),
            "appendmat_resolution": self.appendmat_resolution,
        }


@dataclass(frozen=True)
class _MaterializedLoaderContract:
    """Serializable byte contract for one already-materialized loader file."""

    path: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _CausalSubjectTask:
    dataset_class: str
    subject: int
    loader_contracts: tuple[_MaterializedLoaderContract, ...]
    parser_snapshot_root: str
    shard_path: str
    persist_shard: bool
    channels: tuple[str, ...] | None
    montage: str
    preprocessing: PreprocessingSpec
    target_label: str
    source_reference: str
    source_sample_rate_hz: float
    raw_artifact_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _CausalSubjectResult:
    subject: int
    shard_path: str
    n_epochs: int
    session_runs: tuple[tuple[str, str], ...]
    loader_path_resolutions: tuple[Mapping[str, object], ...]
    runtime_resolver: Mapping[str, object]
    parser_snapshot: Mapping[str, object] = field(default_factory=dict)
    dataset: EpochDataset | None = None


def _code_references_name(code: CodeType, name: str) -> bool:
    return name in code.co_names or any(
        isinstance(constant, CodeType) and _code_references_name(constant, name)
        for constant in code.co_consts
    )


def _call_graph_references_name(
    function: Callable[..., object],
    name: str,
    *,
    visited: set[int] | None = None,
) -> bool:
    """Follow same-module Python helpers and inspect their referenced names."""

    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    globals_by_name = getattr(function, "__globals__", None)
    if code is None or not isinstance(globals_by_name, dict):
        return False
    seen = set() if visited is None else visited
    identity = id(function)
    if identity in seen:
        return False
    seen.add(identity)
    if _code_references_name(code, name):
        return True
    module_name = getattr(function, "__module__", None)
    for referenced_name in code.co_names:
        candidate = globals_by_name.get(referenced_name)
        if (
            inspect.isfunction(candidate)
            and getattr(candidate, "__module__", None) == module_name
            and _call_graph_references_name(candidate, name, visited=seen)
        ):
            return True
    return False


def _assert_validated_moabb_runtime(dataset: object) -> bool:
    """Fail closed if the inspected MOABB 1.6.1 dispatch contract has changed."""

    from moabb.datasets.base import BaseDataset

    installed_version = str(moabb.__version__)
    if installed_version != VALIDATED_MOABB_VERSION:
        raise RuntimeError(
            "Attested MOABB loading is validated only for "
            f"moabb=={VALIDATED_MOABB_VERSION}; installed {installed_version}."
        )
    required_base_names = {
        "_prefetch_nemar_sourcedata",
        "_get_selected_subject_data",
    }
    if not all(
        _code_references_name(BaseDataset.get_data.__code__, name) for name in required_base_names
    ):
        raise RuntimeError("MOABB BaseDataset.get_data dispatch contract has changed.")
    if "_get_single_subject_data_using_cache" not in (
        BaseDataset._get_selected_subject_data.__code__.co_names
    ):
        raise RuntimeError("MOABB selected-subject dispatch contract has changed.")
    cache_loader_names = set(BaseDataset._get_single_subject_data_using_cache.__code__.co_names)
    if not {"_sourcedata_store", "_get_single_subject_data"} <= cache_loader_names:
        raise RuntimeError("MOABB single-subject loader dispatch contract has changed.")

    if not isinstance(dataset, BaseDataset):
        raise RuntimeError("Attested MOABB loading requires a BaseDataset instance.")
    if getattr(type(dataset), "get_data", None) is not BaseDataset.get_data:
        raise RuntimeError("Attested MOABB loading does not accept an overridden get_data method.")
    parser = getattr(dataset, "_get_single_subject_data", None)
    if not callable(parser) or not _call_graph_references_name(parser, "data_path"):
        raise RuntimeError(
            "MOABB single-subject parser no longer dispatches through the instance data_path."
        )
    return True


def _blocked_network_io(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise _AttestedMoabbNetworkAccessError(
        "Network I/O is forbidden while parsing attested MOABB snapshots."
    )


_FILESYSTEM_MUTATION_AUDIT_EVENTS = frozenset(
    {
        "os.chmod",
        "os.chown",
        "os.link",
        "os.mkdir",
        "os.remove",
        "os.removexattr",
        "os.rename",
        "os.rmdir",
        "os.setxattr",
        "os.symlink",
        "os.truncate",
        "os.utime",
    }
)


def _attested_loader_audit_hook(event: str, args: tuple[object, ...]) -> None:
    if not getattr(_MOABB_LOADER_AUDIT_STATE, "active", False):
        return
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        write_mode = isinstance(mode, str) and any(value in mode for value in "wax+")
        write_flags = isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write_mode or write_flags:
            raise _AttestedMoabbLoaderMutationError(
                "Filesystem writes are forbidden while parsing attested MOABB loaders."
            )
    elif event in _FILESYSTEM_MUTATION_AUDIT_EVENTS:
        raise _AttestedMoabbLoaderMutationError(
            f"Filesystem mutation {event!r} is forbidden while parsing attested MOABB loaders."
        )


def _ensure_attested_loader_audit_hook() -> None:
    global _MOABB_LOADER_AUDIT_INSTALLED
    if _MOABB_LOADER_AUDIT_INSTALLED:
        return
    with _MOABB_LOADER_AUDIT_LOCK:
        if not _MOABB_LOADER_AUDIT_INSTALLED:
            sys.addaudithook(_attested_loader_audit_hook)
            _MOABB_LOADER_AUDIT_INSTALLED = True


@contextmanager
def _scoped_attested_loader_read_only() -> Iterator[None]:
    """Deny Python-level filesystem mutation for the complete third-party parse."""

    _ensure_attested_loader_audit_hook()
    if getattr(_MOABB_LOADER_AUDIT_STATE, "active", False):
        raise RuntimeError("Attested loader read-only scopes must not be nested.")
    _MOABB_LOADER_AUDIT_STATE.active = True
    try:
        yield
    finally:
        _MOABB_LOADER_AUDIT_STATE.active = False


@contextmanager
def _scoped_attested_moabb_io(
    dataset: object,
    subjects: Sequence[int],
    binding: _AttestedMoabbDataPathBinding,
) -> Iterator[_MoabbRuntimeResolver]:
    """Pin MOABB dispatch and guard trusted parser code without claiming a sandbox."""

    from moabb.utils import get_download_provider

    strict_contract = _assert_validated_moabb_runtime(dataset)
    declared_nemar_id = getattr(dataset, "nemar_id", None)
    if declared_nemar_id is not None:
        declared_nemar_id = str(declared_nemar_id)
    requested_provider = str(get_download_provider())
    resolver = _MoabbRuntimeResolver(
        declared_nemar_id=declared_nemar_id,
        requested_download_provider=requested_provider,
        strict_base_dataset_contract=strict_contract,
        subjects=tuple(subjects),
    )
    socket_targets = [
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket.socket, "send"),
        (socket.socket, "sendall"),
        (socket.socket, "sendto"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    ]
    if hasattr(socket.socket, "sendmsg"):
        socket_targets.append((socket.socket, "sendmsg"))

    with _MOABB_IO_SCOPE_LOCK:
        missing = object()
        previous_provider: object = os.environ.get(MOABB_DOWNLOAD_PROVIDER_ENV, missing)
        originals: list[tuple[object, str, object]] = []
        try:
            os.environ[MOABB_DOWNLOAD_PROVIDER_ENV] = "upstream"
            if get_download_provider() != "upstream":
                raise RuntimeError("MOABB ignored the scoped upstream provider override.")
            for owner, attribute in socket_targets:
                original = getattr(owner, attribute)
                originals.append((owner, attribute, original))
                setattr(owner, attribute, _blocked_network_io)
            binding.begin_parser_phase()
            yield resolver
            if strict_contract:
                binding.assert_parser_consumed(subjects)
        finally:
            for owner, attribute, original in reversed(originals):
                setattr(owner, attribute, original)
            if previous_provider is missing:
                os.environ.pop(MOABB_DOWNLOAD_PROVIDER_ENV, None)
            else:
                os.environ[MOABB_DOWNLOAD_PROVIDER_ENV] = str(previous_provider)


def _resolve_reported_moabb_path(value: str | Path, *, subject: int) -> _MoabbLoaderPathResolution:
    reported = Path(value)
    appended_mat = Path(f"{reported}.mat") if reported.suffix == "" else None
    if reported.exists():
        if appended_mat is not None and appended_mat.exists():
            raise ValueError(
                "MOABB reported an ambiguous extensionless loader path because both the "
                f"reported path and its .mat form exist: {reported}."
            )
        resolved = reported.resolve(strict=True)
        appendmat_resolution = False
    elif appended_mat is not None:
        if not appended_mat.is_file():
            raise FileNotFoundError(
                f"MOABB extensionless loader path and its .mat form are both missing: {reported}."
            )
        resolved = appended_mat.resolve(strict=True)
        appendmat_resolution = True
    else:
        raise FileNotFoundError(f"MOABB reported loader path is missing: {reported}.")
    return _MoabbLoaderPathResolution(
        subject=subject,
        reported_path=reported,
        resolved_loader_path=resolved,
        appendmat_resolution=appendmat_resolution,
    )


def _reported_moabb_loader_paths(
    dataset: object, subjects: Sequence[int]
) -> tuple[_MoabbLoaderPathResolution, ...]:
    data_path = getattr(dataset, "data_path", None)
    if not callable(data_path):
        raise TypeError("An attested MOABB dataset must expose the BaseDataset.data_path API.")
    reported: list[_MoabbLoaderPathResolution] = []
    for subject in subjects:
        values = data_path(
            subject,
            path=None,
            force_update=False,
            update_path=False,
            verbose=False,
        )
        if isinstance(values, (str, Path)):
            values = [values]
        if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
            raise TypeError("MOABB data_path must return a sequence of physical paths.")
        reported.extend(_resolve_reported_moabb_path(value, subject=subject) for value in values)
    if not reported:
        raise ValueError("MOABB data_path returned no physical paths for selected subjects.")
    return tuple(reported)


def _install_attested_data_path(
    dataset: object,
    paths_by_subject: dict[int, tuple[Path, ...]],
) -> _AttestedMoabbDataPathBinding:
    """Bind one dataset instance to verified local files without upstream I/O."""

    binding = _AttestedMoabbDataPathBinding(paths_by_subject=paths_by_subject)

    def data_path(
        _dataset: object,
        subject: int,
        path: str | Path | None = None,
        force_update: bool = False,
        update_path: bool | None = None,
        verbose: object = None,
    ) -> list[str]:
        del verbose
        if path is not None or force_update or update_path not in {None, False}:
            raise ValueError("Attested MOABB data_path is read-only and accepts no overrides.")
        try:
            paths = binding.paths_by_subject[subject]
        except KeyError as exc:
            raise ValueError(f"No attested loader mapping exists for subject {subject}.") from exc
        binding.calls.append(subject)
        return [str(value) for value in paths]

    dataset.data_path = MethodType(data_path, dataset)  # type: ignore[attr-defined]
    return binding


def _available_mne_source_paths(instances: Sequence[object]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for instance in instances:
        filenames = getattr(instance, "filenames", None)
        if filenames is None:
            raw = getattr(instance, "_raw", None)
            filenames = getattr(raw, "filenames", None)
        if filenames:
            paths.extend(Path(value) for value in filenames if value is not None)
    return tuple(paths)


def _available_cpu_count() -> int:
    """Return the scheduler-visible CPU count, including Linux affinity limits."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            return max(1, len(get_affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _effective_subject_workers(requested: int, n_subjects: int) -> int:
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("subject_workers must be a positive integer.")
    if isinstance(n_subjects, bool) or not isinstance(n_subjects, int) or n_subjects < 1:
        raise ValueError("n_subjects must be a positive integer.")
    cpu_budget = max(1, _available_cpu_count() - 2)
    return min(requested, n_subjects, cpu_budget)


def _initialize_moabb_subject_worker() -> None:
    """Keep each spawned parser single-threaded so subject processes do not oversubscribe."""

    global _WORKER_THREADPOOL_LIMITER
    for name in _WORKER_THREAD_ENV:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - required transitively by scikit-learn
        raise RuntimeError("threadpoolctl is required for bounded MOABB subject workers.") from exc
    _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _normalized_physical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=True)))


def _verify_materialized_loader_contracts(
    contracts: Sequence[_MaterializedLoaderContract],
    *,
    expected_mne_data_root: str | Path,
    paths: Sequence[str | Path],
) -> None:
    """Verify only immutable materialized files, never the original ZIPs, inside a worker."""

    root = Path(expected_mne_data_root).resolve(strict=True)
    expected: dict[str, _MaterializedLoaderContract] = {}
    for contract in contracts:
        resolved = Path(contract.path).resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Worker loader contract escapes controlled mne_data.") from exc
        if relative != contract.relative_path:
            raise ValueError("Worker loader contract path disagrees with its relative path.")
        key = _normalized_physical_path(resolved)
        if key in expected:
            raise ValueError("Worker loader contracts must identify unique physical files.")
        expected[key] = contract

    reported: dict[str, Path] = {}
    for value in paths:
        resolved = Path(value).resolve(strict=True)
        key = _normalized_physical_path(resolved)
        if key not in expected:
            raise ValueError(f"MOABB attempted to read unmapped loader {resolved}.")
        reported[key] = resolved
    if set(reported) != set(expected):
        missing = sorted(expected[key].relative_path for key in set(expected) - set(reported))
        raise ValueError(f"MOABB did not report every subject loader: {missing}.")

    for key, path in reported.items():
        contract = expected[key]
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"Materialized MOABB loader is not a regular file: {path}.")
            digest = hashlib.sha256()
            size_bytes = 0
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while block := stream.read(1024 * 1024):
                    size_bytes += len(block)
                    digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise RuntimeError(f"Materialized MOABB loader changed while read: {path}.")
        if size_bytes != contract.size_bytes or digest.hexdigest() != contract.sha256:
            raise ValueError(
                f"Materialized MOABB loader {contract.relative_path!r} mismatches its attestation."
            )


def _loader_contracts_for_subject(
    materialization: MoabbLoaderMaterialization,
    subject: int,
) -> tuple[_MaterializedLoaderContract, ...]:
    root = Path(materialization.expected_mne_data_root).resolve(strict=True)
    contracts: list[_MaterializedLoaderContract] = []
    for path in materialization.paths_by_subject[subject]:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        snapshot = materialization.loader_snapshots[relative]
        contracts.append(
            _MaterializedLoaderContract(
                path=str(resolved),
                relative_path=relative,
                size_bytes=snapshot.size_bytes,
                sha256=snapshot.sha256,
            )
        )
    return tuple(contracts)


def _private_parser_loader_snapshots(
    contracts: Sequence[_MaterializedLoaderContract],
    *,
    parser_snapshot_root: str | Path,
    shard_workspace: str | Path,
) -> tuple[
    tuple[Path, ...],
    tuple[_MaterializedLoaderContract, ...],
    Path,
    dict[str, object],
]:
    """Copy verified loaders from stable descriptors into one private parser tree."""

    workspace = Path(shard_workspace).resolve(strict=True)
    root = Path(parser_snapshot_root).resolve(strict=False)
    try:
        root.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Private parser snapshot root escapes the shard workspace.") from exc
    if root.exists():
        raise FileExistsError(f"Private parser snapshot root already exists: {root.name}.")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    mne_root = root / "mne_data"
    mne_root.mkdir(mode=0o700)
    root.chmod(stat.S_IRUSR | stat.S_IWRITE | stat.S_IEXEC)
    mne_root.chmod(stat.S_IRUSR | stat.S_IWRITE | stat.S_IEXEC)
    parser_paths: list[Path] = []
    parser_contracts: list[_MaterializedLoaderContract] = []
    records: list[dict[str, object]] = []
    try:
        for contract in contracts:
            source = Path(contract.path)
            destination = mne_root.joinpath(*contract.relative_path.split("/"))
            resolved_parent = destination.parent.resolve(strict=False)
            try:
                resolved_parent.relative_to(mne_root.resolve(strict=True))
            except ValueError as exc:
                raise ValueError("Private parser loader escapes its mne_data tree.") from exc
            current_parent = mne_root
            relative_parent = destination.parent.relative_to(mne_root)
            for component in relative_parent.parts:
                current_parent = current_parent / component
                try:
                    current_parent.mkdir(mode=0o700)
                except FileExistsError as exc:
                    if not current_parent.is_dir() or current_parent.is_symlink():
                        raise ValueError(
                            "Private parser snapshot parent is not a physical directory."
                        ) from exc
            current_parent = destination.parent
            while True:
                current_parent.chmod(stat.S_IRUSR | stat.S_IWRITE | stat.S_IEXEC)
                if current_parent == root:
                    break
                current_parent = current_parent.parent
            source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            source_descriptor = os.open(source, source_flags)
            temporary_descriptor: int | None = None
            temporary: Path | None = None
            try:
                temporary_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".parser-snapshot-",
                    suffix=".tmp",
                    dir=destination.parent,
                )
                temporary = Path(temporary_name)
                before = os.fstat(source_descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"MOABB loader source is not a regular file: {source}.")
                digest = hashlib.sha256()
                size_bytes = 0
                with (
                    os.fdopen(source_descriptor, "rb", closefd=False) as source_stream,
                    os.fdopen(temporary_descriptor, "wb") as destination_stream,
                ):
                    while block := source_stream.read(1024 * 1024):
                        destination_stream.write(block)
                        digest.update(block)
                        size_bytes += len(block)
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
                after = os.fstat(source_descriptor)
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    getattr(before, "st_ctime_ns", None),
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    getattr(after, "st_ctime_ns", None),
                )
                if identity_before != identity_after:
                    raise RuntimeError(
                        f"MOABB loader {contract.relative_path!r} changed during snapshot copy."
                    )
                if size_bytes != contract.size_bytes or digest.hexdigest() != contract.sha256:
                    raise ValueError(
                        f"MOABB loader {contract.relative_path!r} failed private snapshot digest."
                    )
                os.replace(temporary, destination)
                destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            finally:
                try:
                    os.close(source_descriptor)
                except OSError:
                    pass
                try:
                    if temporary_descriptor is not None:
                        os.close(temporary_descriptor)
                except OSError:
                    pass
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            resolved = destination.resolve(strict=True)
            parser_paths.append(resolved)
            parser_contracts.append(
                _MaterializedLoaderContract(
                    path=str(resolved),
                    relative_path=contract.relative_path,
                    size_bytes=contract.size_bytes,
                    sha256=contract.sha256,
                )
            )
            records.append(
                {
                    "source_loader_relative_path": contract.relative_path,
                    "parser_snapshot_relative_path": f"mne_data/{contract.relative_path}",
                    "size_bytes": contract.size_bytes,
                    "sha256": contract.sha256,
                    "copied_from_stable_descriptor": True,
                }
            )
    except BaseException as error:
        try:
            _remove_private_parser_snapshot(root)
        except BaseException as cleanup_error:
            error.add_note(
                "Private parser snapshot cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    return (
        tuple(parser_paths),
        tuple(parser_contracts),
        mne_root.resolve(strict=True),
        {
            "schema": ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA,
            "role": "ephemeral_private_parser_input",
            "byte_authority": "single_pass_hash_and_copy_from_stable_source_descriptor",
            "lifecycle": "deleted_after_parser_preload",
            "files": records,
        },
    )


def _remove_private_snapshot_entry(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_reparse = bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
    if stat.S_ISLNK(metadata.st_mode) or is_reparse:
        try:
            path.unlink()
        except (IsADirectoryError, PermissionError):
            path.rmdir()
        return
    if stat.S_ISDIR(metadata.st_mode):
        with os.scandir(path) as entries:
            child_paths = [Path(entry.path) for entry in entries]
        for child in child_paths:
            _remove_private_snapshot_entry(child)
        path.rmdir()
        return
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD, follow_symlinks=False)
        path.unlink()


def _remove_private_parser_snapshot(root: str | Path) -> None:
    snapshot_root = Path(root)
    for attempt in range(8):
        try:
            _remove_private_snapshot_entry(snapshot_root)
            return
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.025 * (attempt + 1))


def _sorted_mapping_items(
    value: object,
    *,
    label: str,
) -> list[tuple[object, object]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"MOABB returned no {label} mapping.")
    keyed = [(str(key), key, item) for key, item in value.items()]
    if len({text for text, _, _ in keyed}) != len(keyed):
        raise ValueError(f"MOABB {label} keys are ambiguous after string normalization.")

    def natural_key(text: str) -> tuple[tuple[int, object], ...]:
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r"(\d+)", text)
            if part
        )

    return [
        (key, item)
        for text, key, item in sorted(keyed, key=lambda row: (natural_key(row[0]), row[0]))
    ]


def _atomic_write_subject_journal(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _unlink_subject_shard_with_retry(path: Path, *, attempts: int = 8) -> dict[str, str] | None:
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return None
        except OSError as exc:
            if attempt + 1 == attempts:
                return {
                    "path": path.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            time.sleep(0.025 * (attempt + 1))
    raise AssertionError("unreachable")


def _clean_subject_shard_files(workspace: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for pattern in ("*.npz", "*.record.json", "*.tmp.npz"):
        try:
            paths = list(workspace.glob(pattern))
        except OSError as exc:
            issues.append(
                {
                    "path": workspace.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        for path in paths:
            issue = _unlink_subject_shard_with_retry(path)
            if issue is not None:
                issues.append(issue)
    parser_snapshots = workspace / "parser-snapshots"
    try:
        _remove_private_parser_snapshot(parser_snapshots)
    except OSError as exc:
        issues.append(
            {
                "path": parser_snapshots.name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    return issues


def _clear_successful_subject_workspace(workspace: Path) -> dict[str, object]:
    """Atomically remove the discoverable shard directory, then delete its renamed contents."""

    if not workspace.exists():
        return {"status": "completed", "atomic_detach": False, "removal_attempts": 0}
    tombstone = workspace.with_name(f".{workspace.name}.complete-{uuid4().hex}")
    try:
        workspace.replace(tombstone)
    except OSError as exc:
        raise _ShardCleanupError(
            workspace,
            {"error_type": type(exc).__name__, "message": str(exc)},
        ) from exc
    for attempt in range(8):
        try:
            shutil.rmtree(tombstone)
            return {
                "status": "completed",
                "atomic_detach": True,
                "removal_attempts": attempt + 1,
            }
        except OSError as exc:
            if attempt == 7:
                raise _ShardCleanupError(
                    tombstone,
                    {"error_type": type(exc).__name__, "message": str(exc)},
                ) from exc
            time.sleep(0.025 * (attempt + 1))
    raise AssertionError("unreachable")


def _optional_metadata_column(
    metadata: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    integer: bool = False,
) -> np.ndarray | None:
    present = [name for name in aliases if name in metadata]
    if not present:
        return None
    reference = metadata[present[0]].to_numpy()
    for name in present[1:]:
        if not np.array_equal(reference.astype(str), metadata[name].to_numpy().astype(str)):
            raise ValueError(f"Conflicting metadata aliases {present}.")
    if integer:
        if not np.issubdtype(reference.dtype, np.integer) or np.issubdtype(
            reference.dtype, np.bool_
        ):
            raise ValueError(f"Metadata column {present[0]!r} must contain integers.")
        return reference.astype(np.int64, copy=False)
    values = reference.astype(str)
    if np.any(np.char.strip(values) == ""):
        raise ValueError(f"Metadata column {present[0]!r} contains empty identifiers.")
    return values


def resolve_moabb_dataset(dataset_class: str):
    """Resolve any installed MOABB dataset class by public class name."""

    import moabb.datasets

    dataset_type = getattr(moabb.datasets, dataset_class, None)
    if dataset_type is None or not isinstance(dataset_type, type):
        available = sorted(
            name
            for name, value in vars(moabb.datasets).items()
            if isinstance(value, type) and not name.startswith("_")
        )
        raise ValueError(
            f"Unknown MOABB dataset class {dataset_class!r}. Installed public classes include "
            f"{available}."
        )
    return dataset_type()


def encode_binary_labels(values: np.ndarray, *, target_label: str = "Target") -> np.ndarray:
    labels = np.asarray(values)
    if np.issubdtype(labels.dtype, np.number):
        unique = set(np.unique(labels).tolist())
        if unique <= {0, 1}:
            return labels.astype(np.int64)
    normalized = np.char.lower(np.char.strip(labels.astype(str)))
    target = target_label.strip().lower()
    unique = set(normalized.tolist())
    if target not in unique or len(unique) != 2:
        raise ValueError(
            f"Binary P300 labels must contain target={target_label!r} and one non-target class; "
            f"got {sorted(unique)}."
        )
    return (normalized == target).astype(np.int64)


def _causal_run_dataset(
    *,
    task: _CausalSubjectTask,
    raw: object,
    session_id: object,
    run_id: object,
    label_map: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> EpochDataset:
    events, labels = annotation_events_and_labels(raw, label_map)
    record = build_subject(
        raw,
        events,
        labels,
        subject_id=str(task.subject),
        metadata={
            "dataset_id": task.dataset_class,
            "session": str(session_id),
            "run": str(run_id),
            "selection_id": f"{session_id}:{run_id}",
            "reference": task.source_reference,
        },
        sfreq=task.preprocessing.sfreq,
        l_freq=task.preprocessing.l_freq,
        h_freq=task.preprocessing.h_freq,
        tmin=task.preprocessing.tmin_ms / 1000.0,
        tmax=task.preprocessing.tmax_ms / 1000.0,
        n_times=task.preprocessing.n_times,
        reject_threshold=task.preprocessing.reject_threshold_v,
        baseline=None,
        baseline_mode=task.preprocessing.baseline_mode,
        trial_reference_window_ms=task.preprocessing.trial_reference_window_ms,
        trial_reference_center=task.preprocessing.trial_reference_center,
        trial_reference_scale=task.preprocessing.trial_reference_scale,
        filter_method=task.preprocessing.filter_method,
        filter_order=task.preprocessing.filter_order,
        filter_phase=task.preprocessing.filter_phase,
        causal_iir_initial_state=task.preprocessing.causal_iir_initial_state,
        resample_domain=task.preprocessing.resample_domain,
        resample_method=task.preprocessing.resample_method,
        resample_npad=task.preprocessing.resample_npad,
        resample_window=task.preprocessing.resample_window,
        resample_pad=task.preprocessing.resample_pad,
        channels=None if task.channels is None else list(task.channels),
        montage=task.montage,
    )
    n_epochs = record.n_epochs
    return EpochDataset(
        name=task.dataset_class,
        X=np.asarray(record.data, dtype=np.float32),
        y=np.asarray(record.labels, dtype=np.int64),
        subject_ids=np.repeat(str(task.subject), n_epochs),
        channel_names=record.channel_names,
        channel_positions_m=record.channel_positions_m,
        channel_mask=record.channel_mask,
        preprocessing=task.preprocessing,
        event_timeline=record.event_timeline,
        metadata=pd.DataFrame(
            {
                "subject": np.repeat(str(task.subject), n_epochs),
                "session": np.repeat(str(session_id), n_epochs),
                "run": np.repeat(str(run_id), n_epochs),
            }
        ),
        provenance={
            "source": "moabb_raw_causal_p300",
            "dataset_class": task.dataset_class,
            **dict(provenance),
            "source_reference": task.source_reference,
            "source_sample_rate_hz": float(raw.info["sfreq"]),
            "model_input_sample_rate_hz": task.preprocessing.sfreq,
            "signal_unit": task.preprocessing.signal_unit,
            "filter_phase": "forward",
        },
    )


def _process_causal_subject_task(task: _CausalSubjectTask) -> _CausalSubjectResult:
    """Parse, preprocess, and persist exactly one subject in an isolated process."""

    shard_path = Path(task.shard_path)
    parser_snapshot_root = Path(task.parser_snapshot_root)
    try:
        dataset = resolve_moabb_dataset(task.dataset_class)
        if task.subject not in dataset.subject_list:
            raise ValueError(
                f"MOABB worker subject {task.subject} is absent from {task.dataset_class}."
            )
        acquisition = getattr(getattr(dataset, "metadata", None), "acquisition", None)
        worker_reference = str(getattr(acquisition, "reference", None) or "unspecified")
        worker_sample_rate = getattr(acquisition, "sampling_rate", None)
        if (
            worker_reference != task.source_reference
            or worker_sample_rate is None
            or not np.isclose(float(worker_sample_rate), task.source_sample_rate_hz)
        ):
            raise RuntimeError("MOABB dataset acquisition metadata changed across worker spawn.")
        event_id = getattr(dataset, "event_id", None)
        if not isinstance(event_id, dict) or task.target_label not in event_id:
            raise ValueError(
                f"MOABB dataset {task.dataset_class} does not expose target event "
                f"{task.target_label!r}."
            )
        label_map = {
            str(description): int(str(description) == task.target_label) for description in event_id
        }
        (
            parser_paths,
            parser_contracts,
            parser_mne_root,
            parser_snapshot_record,
        ) = _private_parser_loader_snapshots(
            task.loader_contracts,
            parser_snapshot_root=parser_snapshot_root,
            shard_workspace=shard_path.parent,
        )
        binding = _install_attested_data_path(
            dataset,
            {task.subject: parser_paths},
        )
        resolutions = _reported_moabb_loader_paths(dataset, [task.subject])
        reported_paths = tuple(item.resolved_loader_path for item in resolutions)
        _verify_materialized_loader_contracts(
            parser_contracts,
            expected_mne_data_root=parser_mne_root,
            paths=reported_paths,
        )
        with _scoped_attested_moabb_io(dataset, [task.subject], binding) as resolver:
            with _scoped_attested_loader_read_only():
                all_runs = dataset.get_data(subjects=[task.subject])
                if not isinstance(all_runs, Mapping) or set(all_runs) != {task.subject}:
                    raise ValueError(
                        "MOABB worker returned an incomplete or extra subject mapping."
                    )
                subject_runs = all_runs[task.subject]
                session_items = _sorted_mapping_items(subject_runs, label="session")
                ordered_runs: list[tuple[object, object, object]] = []
                raw_instances: list[object] = []
                for session_id, session_runs in session_items:
                    for run_id, raw in _sorted_mapping_items(session_runs, label="run"):
                        if not isinstance(raw, mne.io.BaseRaw):
                            raise TypeError("MOABB run mappings must contain MNE Raw instances.")
                        raw.load_data()
                        if not raw.preload:
                            raise RuntimeError("MOABB Raw remained lazy after load_data().")
                        ordered_runs.append((session_id, run_id, raw))
                        raw_instances.append(raw)
        loader_paths = reported_paths + _available_mne_source_paths(raw_instances)
        _verify_materialized_loader_contracts(
            parser_contracts,
            expected_mne_data_root=parser_mne_root,
            paths=loader_paths,
        )
        loader_records = tuple(item.record(parser_mne_root) for item in resolutions)
        _remove_private_parser_snapshot(parser_snapshot_root)
        parser_snapshot_record["subjects"] = {
            str(task.subject): [contract.relative_path for contract in task.loader_contracts]
        }
        parser_snapshot_record["removed_after_preload"] = True
        runtime_record = resolver.record()
        subject_provenance = {
            **dict(task.raw_artifact_provenance),
            "raw_artifact_loader_path_resolutions": list(loader_records),
            "parser_loader_snapshot": parser_snapshot_record,
            "moabb_attested_resolver": runtime_record,
        }
        datasets = [
            _causal_run_dataset(
                task=task,
                raw=raw,
                session_id=session_id,
                run_id=run_id,
                label_map=label_map,
                provenance=subject_provenance,
            )
            for session_id, run_id, raw in ordered_runs
        ]
        subject_dataset = concatenate_epoch_datasets(
            datasets,
            name=task.dataset_class,
            provenance={
                "source": "moabb_raw_causal_p300",
                "dataset_class": task.dataset_class,
                **subject_provenance,
                "subjects": [task.subject],
                "source_reference": task.source_reference,
                "source_sample_rate_hz": task.source_sample_rate_hz,
                "model_input_sample_rate_hz": task.preprocessing.sfreq,
                "moabb_version": moabb.__version__,
                "mne_version": mne.__version__,
                "signal_unit": task.preprocessing.signal_unit,
            },
        )
        if task.persist_shard:
            save_epoch_dataset(shard_path, subject_dataset, compressed=False)
        return _CausalSubjectResult(
            subject=task.subject,
            shard_path=str(shard_path),
            n_epochs=subject_dataset.n_epochs,
            session_runs=tuple(
                (str(session_id), str(run_id)) for session_id, run_id, _ in ordered_runs
            ),
            loader_path_resolutions=loader_records,
            runtime_resolver=runtime_record,
            parser_snapshot=parser_snapshot_record,
            dataset=None if task.persist_shard else subject_dataset,
        )
    except BaseException as exc:
        try:
            _remove_private_parser_snapshot(parser_snapshot_root)
        except BaseException as parser_cleanup_error:
            exc.add_note(
                "Private parser snapshot cleanup failed: "
                f"{type(parser_cleanup_error).__name__}: {parser_cleanup_error}"
            )
        cleanup_issues = [
            issue
            for issue in (
                _unlink_subject_shard_with_retry(shard_path),
                _unlink_subject_shard_with_retry(shard_path.with_suffix(".record.json")),
                _unlink_subject_shard_with_retry(
                    shard_path.with_suffix(shard_path.suffix + ".tmp.npz")
                ),
            )
            if issue is not None
        ]
        if cleanup_issues:
            exc.add_note(f"Subject shard cleanup issues: {cleanup_issues}")
        raise


def _create_subject_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_moabb_subject_worker,
    )


def _subject_journal_payload(
    *,
    run_status: str,
    dataset_class: str,
    selected_subjects: Sequence[int],
    requested_workers: int,
    effective_workers: int,
    execution_mode: str,
    statuses: Mapping[int, Mapping[str, object]],
    fatal_error: Mapping[str, object] | None = None,
    shard_cleanup: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": MOABB_SUBJECT_JOURNAL_SCHEMA,
        "run_status": run_status,
        "dataset_class": dataset_class,
        "requested_subjects": list(selected_subjects),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "execution_mode": execution_mode,
        "subjects": [dict(statuses[subject]) for subject in selected_subjects],
        "fatal_error": None if fatal_error is None else dict(fatal_error),
        "shard_cleanup": None if shard_cleanup is None else dict(shard_cleanup),
    }


def _combined_runtime_resolver(
    results: Sequence[_CausalSubjectResult],
    selected_subjects: Sequence[int],
) -> dict[str, object]:
    if not results:
        raise ValueError("At least one subject resolver record is required.")
    combined = dict(results[0].runtime_resolver)
    for result in results[1:]:
        comparison = dict(result.runtime_resolver)
        comparison["subjects"] = combined.get("subjects")
        if comparison != combined:
            raise RuntimeError("MOABB runtime resolver changed between subject workers.")
    combined["subjects"] = list(selected_subjects)
    return combined


def _accept_subject_result(
    task: _CausalSubjectTask,
    result: object,
    results_by_subject: Mapping[int, _CausalSubjectResult],
) -> _CausalSubjectResult:
    if not isinstance(result, _CausalSubjectResult):
        raise TypeError("MOABB subject worker returned an unsupported result type.")
    if result.subject != task.subject:
        raise RuntimeError(
            "MOABB subject worker result identity mismatch: "
            f"task={task.subject}, result={result.subject}."
        )
    if result.subject in results_by_subject:
        raise RuntimeError(f"MOABB subject worker repeated subject {result.subject}.")
    expected_shard = Path(task.shard_path).resolve(strict=False)
    returned_shard = Path(result.shard_path).resolve(strict=False)
    if returned_shard != expected_shard:
        raise RuntimeError("MOABB subject worker returned an unexpected shard path.")
    if task.persist_shard != (result.dataset is None):
        raise RuntimeError("MOABB subject worker returned the wrong shard transport mode.")
    if result.n_epochs < 1:
        raise ValueError("MOABB subject worker returned an empty subject shard.")
    expected_snapshot_fields = {
        "schema",
        "role",
        "byte_authority",
        "lifecycle",
        "files",
        "subjects",
        "removed_after_preload",
    }
    if set(result.parser_snapshot) != expected_snapshot_fields:
        raise ValueError("MOABB parser snapshot provenance fields are incomplete or unknown.")
    if result.parser_snapshot.get("schema") != ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA:
        raise ValueError("MOABB subject worker lacks private parser snapshot provenance.")
    if result.parser_snapshot.get("role") != "ephemeral_private_parser_input":
        raise ValueError("MOABB parser snapshot provenance has the wrong role.")
    if result.parser_snapshot.get("byte_authority") != (
        "single_pass_hash_and_copy_from_stable_source_descriptor"
    ):
        raise ValueError("MOABB parser snapshot provenance has the wrong byte authority.")
    if result.parser_snapshot.get("lifecycle") != "deleted_after_parser_preload":
        raise ValueError("MOABB parser snapshot provenance has the wrong lifecycle.")
    if result.parser_snapshot.get("removed_after_preload") is not True:
        raise ValueError("MOABB subject worker did not attest parser snapshot cleanup.")
    subjects = result.parser_snapshot.get("subjects")
    expected_relative_paths = [contract.relative_path for contract in task.loader_contracts]
    if not isinstance(subjects, Mapping) or dict(subjects) != {
        str(task.subject): expected_relative_paths
    }:
        raise ValueError("MOABB parser snapshot provenance has the wrong subject mapping.")
    expected_files = [
        {
            "source_loader_relative_path": contract.relative_path,
            "parser_snapshot_relative_path": f"mne_data/{contract.relative_path}",
            "size_bytes": contract.size_bytes,
            "sha256": contract.sha256,
            "copied_from_stable_descriptor": True,
        }
        for contract in task.loader_contracts
    ]
    if result.parser_snapshot.get("files") != expected_files:
        raise ValueError("MOABB parser snapshot file mapping disagrees with task contracts.")
    if result.runtime_resolver.get("parsed_byte_authority") != (
        ATTESTED_MOABB_PARSER_SNAPSHOT_SCHEMA
    ):
        raise ValueError("MOABB runtime resolver does not bind the parser byte authority.")
    return result


def _prepare_causal_subjects(
    *,
    dataset_class: str,
    materialization: MoabbLoaderMaterialization,
    selected_subjects: Sequence[int],
    channels: Sequence[str] | None,
    montage: str,
    preprocessing: PreprocessingSpec,
    target_label: str,
    source_reference: str,
    source_sample_rate_hz: float,
    requested_workers: int,
) -> EpochDataset:
    effective_workers = _effective_subject_workers(requested_workers, len(selected_subjects))
    execution_mode = "serial_subjects" if effective_workers == 1 else "spawn_process_subjects"
    workspace_parent = Path(materialization.attestation.cache_workspace_root) / "shards"
    workspace_parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(materialization.attestation.cache_workspace_root).resolve(strict=True)
    if workspace_parent.resolve(strict=True).parent != cache_root:
        raise ValueError("MOABB shard parent must be a direct child of cache_workspace_root.")
    workspace = workspace_parent / f"run-{uuid4().hex}"
    workspace.mkdir()
    journal_path = workspace / "subject-workers.journal.json"
    raw_provenance = materialization.attestation.provenance_record()
    tasks = [
        _CausalSubjectTask(
            dataset_class=dataset_class,
            subject=subject,
            loader_contracts=_loader_contracts_for_subject(materialization, subject),
            parser_snapshot_root=str(
                workspace / "parser-snapshots" / f"subject-{subject}-{uuid4().hex}"
            ),
            shard_path=str(workspace / f"{subject}.npz"),
            persist_shard=effective_workers > 1,
            channels=None if channels is None else tuple(channels),
            montage=montage,
            preprocessing=preprocessing,
            target_label=target_label,
            source_reference=source_reference,
            source_sample_rate_hz=source_sample_rate_hz,
            raw_artifact_provenance=raw_provenance,
        )
        for subject in selected_subjects
    ]
    statuses: dict[int, dict[str, object]] = {
        subject: {"subject": subject, "status": "queued"} for subject in selected_subjects
    }

    def write_journal(
        run_status: str,
        fatal_error: Mapping[str, object] | None = None,
        *,
        shard_cleanup: Mapping[str, object] | None = None,
        target: Path = journal_path,
    ) -> None:
        _atomic_write_subject_journal(
            target,
            _subject_journal_payload(
                run_status=run_status,
                dataset_class=dataset_class,
                selected_subjects=selected_subjects,
                requested_workers=requested_workers,
                effective_workers=effective_workers,
                execution_mode=execution_mode,
                statuses=statuses,
                fatal_error=fatal_error,
                shard_cleanup=shard_cleanup,
            ),
        )

    write_journal("running")
    results_by_subject: dict[int, _CausalSubjectResult] = {}
    active_subject: int | None = None
    active_stage = "subject_parse"
    try:
        if effective_workers == 1:
            for task in tasks:
                active_subject = task.subject
                statuses[task.subject] = {"subject": task.subject, "status": "running"}
                write_journal("running")
                result = _accept_subject_result(
                    task,
                    _process_causal_subject_task(task),
                    results_by_subject,
                )
                results_by_subject[task.subject] = result
                statuses[result.subject] = {
                    "subject": result.subject,
                    "status": "completed",
                    "n_epochs": result.n_epochs,
                    "session_runs": [list(value) for value in result.session_runs],
                }
                write_journal("running")
        else:
            executor = _create_subject_executor(effective_workers)
            futures: dict[Future[_CausalSubjectResult], _CausalSubjectTask] = {}
            try:
                for task in tasks:
                    future = executor.submit(_process_causal_subject_task, task)
                    futures[future] = task
                    statuses[task.subject] = {"subject": task.subject, "status": "running"}
                write_journal("running")
                for future in as_completed(futures):
                    task = futures[future]
                    active_subject = task.subject
                    result = _accept_subject_result(
                        task,
                        future.result(),
                        results_by_subject,
                    )
                    results_by_subject[task.subject] = result
                    statuses[result.subject] = {
                        "subject": result.subject,
                        "status": "completed",
                        "n_epochs": result.n_epochs,
                        "session_runs": [list(value) for value in result.session_runs],
                    }
                    write_journal("running")
            except BaseException as exc:
                for future, task in futures.items():
                    subject = task.subject
                    if subject in results_by_subject:
                        continue
                    if subject == active_subject:
                        statuses[subject] = {
                            "subject": subject,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "stage": active_stage,
                        }
                    else:
                        statuses[subject] = {
                            "subject": subject,
                            "status": "cancelled" if future.cancel() else "cancelling",
                        }
                try:
                    write_journal(
                        "failed",
                        {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "subject": active_subject,
                            "stage": active_stage,
                        },
                        shard_cleanup={"status": "pending_worker_shutdown"},
                    )
                except BaseException as journal_exc:
                    exc.add_note(
                        "Could not persist immediate worker failure journal: "
                        f"{type(journal_exc).__name__}: {journal_exc}"
                    )
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except BaseException as shutdown_exc:
                    exc.add_note(
                        "Worker shutdown also failed: "
                        f"{type(shutdown_exc).__name__}: {shutdown_exc}"
                    )
                raise
            else:
                active_subject = None
                active_stage = "executor_shutdown"
                executor.shutdown(wait=True, cancel_futures=True)

        active_subject = None
        active_stage = "subject_result_accounting"
        if set(results_by_subject) != set(selected_subjects):
            raise RuntimeError(
                "MOABB subject worker result set mismatch: "
                f"requested={list(selected_subjects)}, returned={sorted(results_by_subject)}."
            )
        ordered_results = [results_by_subject[subject] for subject in selected_subjects]
        subject_datasets: list[EpochDataset] = []
        for result in ordered_results:
            active_subject = result.subject
            active_stage = "subject_shard_validation"
            shard = (
                result.dataset
                if result.dataset is not None
                else load_epoch_dataset(
                    result.shard_path,
                    require_labels=True,
                    validation="attested",
                )
            )
            shard.validate(require_labels=True)
            if set(shard.subject_ids.astype(str)) != {str(result.subject)}:
                raise ValueError("MOABB subject shard contains cross-subject rows.")
            if shard.n_epochs != result.n_epochs or (
                not shard.metadata.empty
                and set(shard.metadata["subject"].astype(str)) != {str(result.subject)}
            ):
                raise ValueError("MOABB subject shard accounting is inconsistent.")
            subject_datasets.append(shard)
        active_subject = None
        active_stage = "aggregate_validation"
        resolver = _combined_runtime_resolver(ordered_results, selected_subjects)
        loader_resolutions = [
            dict(record) for result in ordered_results for record in result.loader_path_resolutions
        ]
        execution_record = {
            "schema": MOABB_SUBJECT_EXECUTION_SCHEMA,
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "execution_mode": execution_mode,
            "scheduler_visible_cpus": _available_cpu_count(),
            "subjects": [
                {
                    **dict(statuses[result.subject]),
                    "loader_path_resolutions": [
                        dict(record) for record in result.loader_path_resolutions
                    ],
                    "moabb_attested_resolver": dict(result.runtime_resolver),
                    "parser_loader_snapshot": dict(result.parser_snapshot),
                }
                for result in ordered_results
            ],
        }
        merged = concatenate_epoch_datasets(
            subject_datasets,
            name=dataset_class,
            provenance={
                "source": "moabb_raw_causal_p300",
                "dataset_class": dataset_class,
                **raw_provenance,
                "raw_artifact_loader_path_resolutions": loader_resolutions,
                "moabb_attested_resolver": resolver,
                "parser_loader_snapshots": [
                    dict(result.parser_snapshot) for result in ordered_results
                ],
                "subject_execution": execution_record,
                "subjects": list(selected_subjects),
                "source_reference": source_reference,
                "source_sample_rate_hz": source_sample_rate_hz,
                "model_input_sample_rate_hz": preprocessing.sfreq,
                "moabb_version": moabb.__version__,
                "mne_version": mne.__version__,
                "signal_unit": preprocessing.signal_unit,
            },
        )
    except BaseException as exc:
        for subject in selected_subjects:
            if statuses[subject]["status"] not in {"completed", "failed"}:
                statuses[subject] = {"subject": subject, "status": "cancelled"}
        if active_subject is not None:
            statuses[active_subject] = {
                "subject": active_subject,
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stage": active_stage,
            }
        fatal_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "subject": active_subject,
            "stage": active_stage,
        }
        try:
            write_journal(
                "failed",
                fatal_error,
                shard_cleanup={"status": "pending"},
            )
        except BaseException as journal_exc:
            exc.add_note(
                "Could not persist failure journal before cleanup: "
                f"{type(journal_exc).__name__}: {journal_exc}"
            )
        cleanup_issues = _clean_subject_shard_files(workspace)
        cleanup_record: dict[str, object] = {
            "status": "incomplete" if cleanup_issues else "completed",
            "issues": cleanup_issues,
        }
        try:
            write_journal(
                "failed",
                fatal_error,
                shard_cleanup=cleanup_record,
            )
        except BaseException as journal_exc:
            exc.add_note(
                "Could not persist failure journal after cleanup: "
                f"{type(journal_exc).__name__}: {journal_exc}"
            )
        if cleanup_issues:
            exc.add_note(f"Shard cleanup issues: {cleanup_issues}")
        raise
    cleaning_journal_error: BaseException | None = None
    try:
        write_journal("cleaning", shard_cleanup={"status": "pending"})
    except BaseException as error:
        cleaning_journal_error = error
    finally:
        try:
            cleanup_record = _clear_successful_subject_workspace(workspace)
            cleanup_error: _ShardCleanupError | None = None
        except _ShardCleanupError as error:
            cleanup_error = error
            cleanup_record = {
                "status": "incomplete",
                "workspace": error.workspace.name,
                "issues": [error.issue],
            }
    if cleaning_journal_error is not None or cleanup_error is not None:
        primary_error = cleaning_journal_error or cleanup_error
        assert primary_error is not None
        if cleaning_journal_error is not None and cleanup_error is not None:
            cleaning_journal_error.add_note(
                f"Shard cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        if cleanup_error is not None:
            failure_workspace = cleanup_error.workspace
        else:
            failure_workspace = workspace_parent / (
                f".{workspace.name}.journal-failure-{uuid4().hex}"
            )
            failure_workspace.mkdir(mode=0o700)
        cleanup_journal = failure_workspace / journal_path.name
        failure_stage = (
            "cleaning_journal" if cleaning_journal_error is not None else "shard_cleanup"
        )
        cleanup_record = {
            **cleanup_record,
            "failure_audit_workspace": failure_workspace.name,
        }
        try:
            write_journal(
                "failed",
                {
                    "error_type": type(primary_error).__name__,
                    "message": str(primary_error),
                    "subject": None,
                    "stage": failure_stage,
                },
                shard_cleanup=cleanup_record,
                target=cleanup_journal,
            )
        except BaseException as journal_exc:
            primary_error.add_note(
                "Could not persist post-cleanup typed failure journal: "
                f"{type(journal_exc).__name__}: {journal_exc}"
            )
        raise primary_error
    execution_record["shard_cleanup"] = cleanup_record
    return merged


def prepare_moabb_p300(
    dataset_class: str,
    *,
    raw_artifact_attestation: RawArtifactAttestation,
    subjects: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    montage: str = DEFAULT_MONTAGE,
    preprocessing: PreprocessingSpec = P300_PERFORMANCE_PREPROCESSING,
    target_label: str = "Target",
    subject_workers: int = 1,
) -> EpochDataset:
    """Process an attested MOABB P300 dataset without dataset-specific branches."""

    from moabb.paradigms import P300

    preprocessing.validate()
    if not isinstance(raw_artifact_attestation, RawArtifactAttestation):
        raise TypeError("raw_artifact_attestation must be a verified RawArtifactAttestation.")
    raw_artifact_attestation.assert_dataset_class(dataset_class)
    if preprocessing.reject_threshold_v is not None:
        raise ValueError(
            "Fixed absolute-voltage artifact rejection is retired; set reject_threshold_v=None "
            "and use fold-local artifact QC during evaluation."
        )
    if (
        isinstance(subject_workers, bool)
        or not isinstance(subject_workers, int)
        or subject_workers < 1
    ):
        raise ValueError("subject_workers must be a positive integer.")
    if preprocessing.filter_phase != "forward" and subject_workers != 1:
        raise ValueError("subject_workers > 1 is supported only for forward causal preprocessing.")
    dataset = resolve_moabb_dataset(dataset_class)
    acquisition = getattr(getattr(dataset, "metadata", None), "acquisition", None)
    source_reference = str(getattr(acquisition, "reference", None) or "unspecified")
    source_sample_rate = getattr(acquisition, "sampling_rate", None)
    if source_reference == "unspecified" or source_sample_rate is None:
        raise ValueError(
            f"MOABB dataset {dataset_class} lacks acquisition reference/sample-rate metadata."
        )
    selected_subjects = list(subjects) if subjects is not None else list(dataset.subject_list)
    if not selected_subjects:
        raise ValueError("At least one MOABB subject must be selected.")
    if len(set(selected_subjects)) != len(selected_subjects):
        raise ValueError("MOABB subject selection must not contain duplicates.")
    unknown_subjects = set(selected_subjects) - set(dataset.subject_list)
    if unknown_subjects:
        raise ValueError(f"Unknown subjects for {dataset_class}: {sorted(unknown_subjects)}.")
    materialization = raw_artifact_attestation.materialize_moabb_loaders(selected_subjects)
    _install_attested_data_path(
        dataset,
        dict(materialization.paths_by_subject),
    )
    loader_path_resolutions = _reported_moabb_loader_paths(dataset, selected_subjects)
    reported_loader_paths = tuple(item.resolved_loader_path for item in loader_path_resolutions)
    materialization.verify_loader_paths(reported_loader_paths)
    if preprocessing.filter_phase == "forward":
        return _prepare_causal_subjects(
            dataset_class=dataset_class,
            materialization=materialization,
            selected_subjects=selected_subjects,
            channels=channels,
            montage=montage,
            preprocessing=preprocessing,
            target_label=target_label,
            source_reference=source_reference,
            source_sample_rate_hz=float(source_sample_rate),
            requested_workers=subject_workers,
        )
    paradigm = P300(
        fmin=preprocessing.l_freq,
        fmax=preprocessing.h_freq,
        tmin=preprocessing.tmin_ms / 1000.0,
        tmax=preprocessing.tmax_ms / 1000.0,
        baseline=None,
        resample=None,
        channels=None if channels is None else list(channels),
    )
    parser_workspace = (
        Path(materialization.attestation.cache_workspace_root)
        / "parser-snapshots"
        / f"run-{uuid4().hex}"
    )
    parser_workspace.mkdir(parents=True)
    parser_snapshot_root = parser_workspace / "snapshot"
    subject_contracts = {
        subject: _loader_contracts_for_subject(materialization, subject)
        for subject in selected_subjects
    }
    combined_contracts = tuple(
        contract for subject in selected_subjects for contract in subject_contracts[subject]
    )
    try:
        (
            parser_paths,
            parser_contracts,
            parser_mne_root,
            parser_snapshot_record,
        ) = _private_parser_loader_snapshots(
            combined_contracts,
            parser_snapshot_root=parser_snapshot_root,
            shard_workspace=parser_workspace,
        )
        offset = 0
        parser_paths_by_subject: dict[int, tuple[Path, ...]] = {}
        for subject in selected_subjects:
            count = len(subject_contracts[subject])
            parser_paths_by_subject[subject] = parser_paths[offset : offset + count]
            offset += count
        parser_binding = _install_attested_data_path(dataset, parser_paths_by_subject)
        parser_resolutions = _reported_moabb_loader_paths(dataset, selected_subjects)
        parser_reported_paths = tuple(item.resolved_loader_path for item in parser_resolutions)
        _verify_materialized_loader_contracts(
            parser_contracts,
            expected_mne_data_root=parser_mne_root,
            paths=parser_reported_paths,
        )
        with _scoped_attested_moabb_io(
            dataset,
            selected_subjects,
            parser_binding,
        ) as runtime_resolver:
            with _scoped_attested_loader_read_only():
                epochs, raw_labels, metadata = paradigm.get_data(
                    dataset=dataset,
                    subjects=selected_subjects,
                    return_epochs=True,
                )
                if runtime_resolver.strict_base_dataset_contract:
                    if not isinstance(epochs, mne.BaseEpochs):
                        raise TypeError("MOABB paradigm must return an MNE Epochs instance.")
                    epochs.load_data()
                    if not epochs.preload:
                        raise RuntimeError("MOABB Epochs remained lazy after load_data().")
        parser_loader_paths = parser_reported_paths + _available_mne_source_paths([epochs])
        _verify_materialized_loader_contracts(
            parser_contracts,
            expected_mne_data_root=parser_mne_root,
            paths=parser_loader_paths,
        )
        parser_loader_records = [item.record(parser_mne_root) for item in parser_resolutions]
    except BaseException as error:
        try:
            _remove_private_parser_snapshot(parser_workspace)
        except BaseException as cleanup_error:
            error.add_note(
                "Private parser workspace cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    _remove_private_parser_snapshot(parser_workspace)
    parser_snapshot_record["subjects"] = {
        str(subject): [contract.relative_path for contract in subject_contracts[subject]]
        for subject in selected_subjects
    }
    parser_snapshot_record["removed_after_preload"] = True
    raw_artifact_provenance = materialization.attestation.provenance_record()
    raw_artifact_provenance["raw_artifact_loader_path_resolutions"] = parser_loader_records
    raw_artifact_provenance["parser_loader_snapshot"] = parser_snapshot_record
    raw_artifact_provenance["moabb_attested_resolver"] = runtime_resolver.record()
    expected_tmin = preprocessing.tmin_ms / 1000.0
    if not np.isclose(float(epochs.times[0]), expected_tmin, atol=0.5 / preprocessing.sfreq):
        raise ValueError(
            "MOABB epoch origin does not match the declared stimulus-relative time contract: "
            f"{float(epochs.times[0]):g}s vs {expected_tmin:g}s."
        )
    source_epoch_sfreq = float(epochs.info["sfreq"])
    source_epoch_events = np.asarray(epochs.events, dtype=np.int64).copy()
    if not np.isclose(source_epoch_sfreq, preprocessing.sfreq):
        epochs.resample(preprocessing.sfreq, verbose=False)
    X = epochs.get_data()
    raw_labels = np.asarray(raw_labels)
    if len(X) != len(raw_labels) or len(metadata) != len(X):
        raise ValueError("MOABB epochs, labels, and metadata are not row-aligned.")
    if "subject" not in metadata:
        raise ValueError("MOABB metadata does not contain a subject column required for LOSO.")
    requested_subject_keys = {str(subject) for subject in selected_subjects}
    returned_subject_keys = set(metadata["subject"].astype(str))
    missing_before_rejection = requested_subject_keys - returned_subject_keys
    if missing_before_rejection:
        raise ValueError(
            f"MOABB returned no epochs for selected subjects {sorted(missing_before_rejection)}."
        )
    if X.shape[2] < preprocessing.n_times:
        raise ValueError(
            f"MOABB returned {X.shape[2]} time samples; {preprocessing.n_times} are required."
        )
    X = X[:, :, : preprocessing.n_times].astype(np.float32, copy=False)
    X = apply_trial_baseline(
        X,
        sfreq=preprocessing.sfreq,
        tmin=preprocessing.tmin_ms / 1000.0,
        baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale,
    )
    keep = np.ones(len(X), dtype=bool)
    metadata = metadata.reset_index(drop=True).copy()
    channel_names = tuple(str(name) for name in epochs.ch_names)
    identity = build_channel_identity(
        channel_names,
        channel_mask=np.ones(len(channel_names), dtype=bool),
        montage=montage,
        allow_missing_positions=False,
    )
    metadata["acquisition_time_s"] = source_epoch_events[keep, 0].astype(float) / source_epoch_sfreq
    subjects_array = metadata["subject"].astype(str).to_numpy()
    sessions = (
        metadata["session"].astype(str).to_numpy()
        if "session" in metadata
        else [""] * len(metadata)
    )
    runs = metadata["run"].astype(str).to_numpy() if "run" in metadata else [""] * len(metadata)
    selections = (
        metadata["selection_id"].astype(str).to_numpy()
        if "selection_id" in metadata
        else subjects_array
    )
    groups = np.asarray(
        [
            selection_group_id(dataset_class, subject, session, run, selection)
            for subject, session, run, selection in zip(
                subjects_array, sessions, runs, selections, strict=True
            )
        ]
    )
    timeline = observed_only_timeline(
        dataset_id=dataset_class,
        subject_ids=subjects_array,
        stimulus_ids=source_epoch_events[keep, 2],
        onset_times_s=metadata["acquisition_time_s"].to_numpy(dtype=float),
        group_ids=groups,
        selection_ids=np.asarray(selections, dtype=str),
        session_ids=np.asarray(sessions, dtype=str),
        run_ids=np.asarray(runs, dtype=str),
        candidate_ids=_optional_metadata_column(metadata, CANDIDATE_METADATA_ALIASES),
        target_candidate_ids=_optional_metadata_column(metadata, TARGET_CANDIDATE_METADATA_ALIASES),
        repetition_indices=_optional_metadata_column(
            metadata, REPETITION_METADATA_ALIASES, integer=True
        ),
        online_causal=False,
        timing_source=(
            "moabb_post_resample_epoch_events;observed_epochs_only;preprocessing_not_online_causal"
        ),
    )
    result = EpochDataset(
        name=dataset_class,
        X=X,
        y=encode_binary_labels(raw_labels, target_label=target_label),
        subject_ids=subjects_array,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=identity.mask,
        preprocessing=preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "moabb_p300",
            "dataset_class": dataset_class,
            **raw_artifact_provenance,
            "subjects": selected_subjects,
            "moabb_version": moabb.__version__,
            "mne_version": mne.__version__,
            "source_sample_rate_hz": source_sample_rate,
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "source_reference": source_reference,
            "source_ground": getattr(acquisition, "ground", None),
            "event_time_basis": "post_resample_epoch_event_samples",
            "grouping_fidelity": "subject_session_run_only; source blocks are not exposed",
            "signal_unit": preprocessing.signal_unit,
            "filter": {
                "method": preprocessing.filter_method,
                "order": preprocessing.filter_order,
                "phase": preprocessing.filter_phase,
            },
            "resample_domain": preprocessing.resample_domain,
            "resample": {
                "method": preprocessing.resample_method,
                "npad": preprocessing.resample_npad,
                "window": preprocessing.resample_window,
                "pad": preprocessing.resample_pad,
            },
            "montage": montage,
            "coordinate_registration": identity.registration.record(),
            "right_endpoint": "exclusive_crop",
            "epoch_baseline": {
                "mode": preprocessing.baseline_mode,
                "window_ms": (
                    list(preprocessing.trial_reference_window_ms)
                    if preprocessing.trial_reference_window_ms is not None
                    else [preprocessing.tmin_ms, 0.0]
                ),
                "center": preprocessing.trial_reference_center,
                "scale": (
                    "std"
                    if preprocessing.baseline_mode == "trial"
                    else preprocessing.trial_reference_scale
                ),
            },
            "artifact_rejection": {"method": "fold_local_ptp_cv", "n_before": int(len(keep))},
        },
    )
    result.validate(require_labels=True)
    return result
