from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from data.contract import CAUSAL_IIR_INITIAL_STATE
from data.epochs import PreprocessingSpec
from data.moabb import prepare_moabb_p300
from data.raw_artifacts import (
    RAW_ARTIFACT_MANIFEST_SCHEMA,
    RAW_ARTIFACT_PATH_SEMANTICS,
    RawArtifactAttestation,
    verify_raw_artifact_manifest,
)


def _fake_dataset(subjects=(1, 2), *, loader_path: Path | None = None):
    acquisition = SimpleNamespace(
        sampling_rate=100.0,
        reference="right earlobe",
        ground="Fz",
    )
    dataset = SimpleNamespace(
        subject_list=list(subjects),
        metadata=SimpleNamespace(acquisition=acquisition),
    )
    if loader_path is not None:
        dataset.data_path = lambda *args, **kwargs: [loader_path]
    return dataset


def _verified_artifact_attestation(tmp_path: Path, dataset_class: str) -> RawArtifactAttestation:
    artifact_root = tmp_path / f"artifacts-{dataset_class}"
    mne_data_root = artifact_root / "mne_data"
    mne_data_root.mkdir(parents=True)
    files = []
    entries = []
    for subject in (1, 2):
        artifact = mne_data_root / f"source-{subject}.bin"
        artifact.write_bytes(f"physical-source:{dataset_class}:{subject}".encode("ascii"))
        sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        relative = f"mne_data/source-{subject}.bin"
        files.append(
            {
                "relative_path": relative,
                "size_bytes": artifact.stat().st_size,
                "official_md5": None,
                "local_sha256": sha256,
                "remote_sha256": sha256,
                "verified": True,
            }
        )
        entries.append(
            {
                "subject": subject,
                "loader_relative_path": f"subject-{subject}/source.bin",
                "source": {"kind": "manifest_file", "relative_path": relative},
            }
        )
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "dataset_class": dataset_class,
        "official_source": {"kind": "unit_test_fixture"},
        "official_record": {"record_id": f"unit-test:{dataset_class}"},
        "artifact_root_contract": {
            "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
            "expected_mne_data_root": "mne_data",
            "mne_dataset_path_key": f"MNE_DATASETS_{dataset_class.upper()}_PATH",
            "moabb_loader_mapping": {
                "schema": "n2p3_moabb_loader_mapping/1",
                "entries": entries,
            },
        },
        "files": files,
    }
    manifest_path = tmp_path / f"raw-artifacts-{dataset_class}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "cache" / dataset_class).mkdir(parents=True)
    return verify_raw_artifact_manifest(
        manifest_path,
        artifact_root,
        snapshot_root=tmp_path / "cache" / dataset_class / "snapshots",
        cache_workspace_root=tmp_path / "cache" / dataset_class,
        expected_dataset_class=dataset_class,
    )


def _configure_attested_mne_root(monkeypatch, attestation: RawArtifactAttestation) -> None:
    expected = attestation.expected_mne_data_root
    allowed = {"MNE_DATA", attestation.mne_dataset_path_key}
    monkeypatch.setattr(
        mne,
        "get_config",
        lambda key: expected if key in allowed else None,
    )


def _verified_zip_artifact_attestation(
    tmp_path: Path,
    dataset_class: str,
    *,
    member_name: str = "official/group_01_s1.mat",
    payload: bytes = b"canonical-mat-member",
) -> tuple[RawArtifactAttestation, Path, Path]:
    artifact_root = tmp_path / f"zip-artifacts-{dataset_class}"
    mne_data_root = artifact_root / "mne_data"
    archive_path = mne_data_root / "downloads" / "official.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, payload)
    reported_loader = mne_data_root / "unused-original-loader"
    resolved_loader = reported_loader
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "dataset_class": dataset_class,
        "official_source": {"kind": "unit_test_zip_fixture"},
        "official_record": {"record_id": f"zip-unit-test:{dataset_class}"},
        "artifact_root_contract": {
            "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
            "expected_mne_data_root": "mne_data",
            "mne_dataset_path_key": f"MNE_DATASETS_{dataset_class.upper()}_PATH",
            "moabb_loader_mapping": {
                "schema": "n2p3_moabb_loader_mapping/1",
                "entries": [
                    {
                        "subject": 1,
                        "loader_relative_path": "extracted/group_01_s1",
                        "source": {
                            "kind": "zip_member",
                            "archive_relative_path": "mne_data/downloads/official.zip",
                            "archive_member": member_name,
                        },
                    }
                ],
            },
        },
        "files": [
            {
                "relative_path": "mne_data/downloads/official.zip",
                "size_bytes": archive_path.stat().st_size,
                "official_md5": None,
                "local_sha256": archive_sha256,
                "remote_sha256": archive_sha256,
                "verified": True,
            }
        ],
    }
    manifest_path = tmp_path / f"zip-raw-artifacts-{dataset_class}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "cache" / dataset_class).mkdir(parents=True)
    attestation = verify_raw_artifact_manifest(
        manifest_path,
        artifact_root,
        snapshot_root=tmp_path / "cache" / dataset_class / "snapshots",
        cache_workspace_root=tmp_path / "cache" / dataset_class,
        expected_dataset_class=dataset_class,
    )
    return attestation, reported_loader, resolved_loader


def _verified_braininvaders_mat_attestation(
    tmp_path: Path,
    dataset_class: str,
    loader_payloads: list[tuple[str, str, np.ndarray]],
) -> RawArtifactAttestation:
    artifact_root = tmp_path / f"mat-artifacts-{dataset_class}"
    mne_data_root = artifact_root / "mne_data"
    source_root = mne_data_root / "official"
    source_root.mkdir(parents=True)
    files = []
    entries = []
    for index, (loader_relative_path, variable_name, payload) in enumerate(loader_payloads):
        source = source_root / f"source-{index:02}.mat"
        savemat(source, {variable_name: payload})
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_relative = source.relative_to(artifact_root).as_posix()
        files.append(
            {
                "relative_path": source_relative,
                "size_bytes": source.stat().st_size,
                "official_md5": None,
                "local_sha256": digest,
                "remote_sha256": digest,
                "verified": True,
            }
        )
        entries.append(
            {
                "subject": 1,
                "loader_relative_path": loader_relative_path,
                "source": {"kind": "manifest_file", "relative_path": source_relative},
            }
        )
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "dataset_class": dataset_class,
        "official_source": {"kind": "synthetic_braininvaders_parser_fixture"},
        "official_record": {"record_id": f"parser-fixture:{dataset_class}"},
        "artifact_root_contract": {
            "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
            "expected_mne_data_root": "mne_data",
            "mne_dataset_path_key": f"MNE_DATASETS_{dataset_class.upper()}_PATH",
            "moabb_loader_mapping": {
                "schema": "n2p3_moabb_loader_mapping/1",
                "entries": entries,
            },
        },
        "files": files,
    }
    manifest_path = tmp_path / f"mat-raw-artifacts-{dataset_class}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_root = tmp_path / "cache" / dataset_class
    cache_root.mkdir(parents=True)
    return verify_raw_artifact_manifest(
        manifest_path,
        artifact_root,
        snapshot_root=cache_root / "snapshots",
        cache_workspace_root=cache_root,
        expected_dataset_class=dataset_class,
    )


def _forbid_moabb_downloads(monkeypatch) -> list[str]:
    import moabb.datasets.base as base_module
    import moabb.datasets.download as download_module

    calls: list[str] = []

    def deny(name: str):
        def blocked(*args, **kwargs):
            del args, kwargs
            calls.append(name)
            raise AssertionError(f"Forbidden MOABB download entrypoint called: {name}")

        return blocked

    monkeypatch.setattr(download_module, "data_dl", deny("data_dl"))
    monkeypatch.setattr(base_module, "nemar_dl", deny("nemar_dl"))
    monkeypatch.setattr(base_module, "nemar_sourcedata_dl", deny("nemar_sourcedata_dl"))
    monkeypatch.setattr(base_module, "nemar_store", deny("nemar_store"))
    return calls


def _allow_narrow_moabb_test_double(monkeypatch) -> None:
    monkeypatch.setattr("data.moabb._assert_validated_moabb_runtime", lambda dataset: False)


class _InlineSubjectExecutor:
    """Exercise Future scheduling deterministically without spawning patched test doubles."""

    def submit(self, function, task):
        future = Future()
        try:
            future.set_result(function(task))
        except BaseException as exc:  # noqa: BLE001 - Future transports the worker exception
            future.set_exception(exc)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is True


def _spawn_roundtrip(value):
    return value


def _causal_parallel_fixture(monkeypatch, tmp_path: Path):
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)
    info = mne.create_info(["Fz", "Cz"], 100.0, ch_types="eeg")

    def raw(value: float):
        times = np.arange(600, dtype=float) / 100.0
        signal = np.vstack(
            [
                np.sin(2.0 * np.pi * 5.0 * times) + value,
                np.cos(2.0 * np.pi * 7.0 * times) - value,
            ]
        )
        instance = mne.io.RawArray(signal, info, verbose=False)
        instance.set_montage("standard_1005")
        instance.set_annotations(
            mne.Annotations(
                onset=[1.0, 2.0],
                duration=[0.0, 0.0],
                description=["NonTarget", "Target"],
            )
        )
        return instance

    runs_by_subject = {
        subject: {
            "session-b": {"run-2": raw(subject * 10 + 2), "run-1": raw(subject * 10 + 1)},
            "session-a": {"run-2": raw(subject * 10 + 4), "run-1": raw(subject * 10 + 3)},
        }
        for subject in (1, 2)
    }

    def resolve(_name: str):
        dataset = _fake_dataset((1, 2))
        dataset.event_id = {"NonTarget": 1, "Target": 2}
        dataset.get_data = lambda subjects: {
            subject: runs_by_subject[subject] for subject in subjects
        }
        return dataset

    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", resolve)
    profile = PreprocessingSpec(
        name="causal_subject_parallel",
        sfreq=100.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=500.0,
        n_times=50,
        baseline_mode="none",
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    return (
        adapter_module,
        _verified_artifact_attestation(tmp_path, "ParallelP300"),
        profile,
    )


def _source_race_fixture(monkeypatch, tmp_path: Path, dataset_class: str):
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)
    attestation = _verified_artifact_attestation(tmp_path, dataset_class)
    materialization = attestation.materialize_moabb_loaders([1])
    source_path = materialization.paths_by_subject[1][0]
    canonical = source_path.read_bytes()
    observed: list[bytes] = []
    parser_action = {"before": lambda: None, "after": lambda: None}

    def resolve(_name: str):
        dataset = _fake_dataset((1,))
        dataset.event_id = {"NonTarget": 1, "Target": 2}

        def get_data(*, subjects):
            parser_action["before"]()
            parser_path = Path(dataset.data_path(subjects[0])[0])
            payload = parser_path.read_bytes()
            observed.append(payload)
            parser_action["after"]()
            scale = 1.0 if payload == canonical else 50.0
            info = mne.create_info(["Fz", "Cz"], 100.0, ch_types="eeg")
            times = np.arange(600, dtype=float) / 100.0
            raw = mne.io.RawArray(
                scale
                * np.vstack(
                    [
                        np.sin(2.0 * np.pi * 5.0 * times),
                        np.cos(2.0 * np.pi * 7.0 * times),
                    ]
                ),
                info,
                verbose=False,
            )
            raw.set_montage("standard_1005")
            raw.set_annotations(
                mne.Annotations(
                    onset=[1.0, 2.0],
                    duration=[0.0, 0.0],
                    description=["NonTarget", "Target"],
                )
            )
            return {subjects[0]: {"0": {"0": raw}}}

        dataset.get_data = get_data
        return dataset

    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", resolve)
    profile = PreprocessingSpec(
        name="causal_private_parser_snapshot",
        sfreq=100.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=500.0,
        n_times=50,
        baseline_mode="none",
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    return adapter_module, attestation, source_path, canonical, observed, parser_action, profile


def test_attested_scope_bypasses_nemar_prefetch_and_restores_process_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import socket

    from moabb.datasets.base import BaseDataset

    from data.moabb import _install_attested_data_path, _scoped_attested_moabb_io

    class SnapshotDataset(BaseDataset):
        nemar_id = "nm-unit-test"

        def __init__(self):
            super().__init__(
                subjects=[1],
                sessions_per_subject=1,
                events={"Target": 2, "NonTarget": 1},
                code="SnapshotDataset",
                interval=[0.0, 1.0],
                paradigm="p300",
            )

        def data_path(self, subject, **kwargs):
            del subject, kwargs
            raise AssertionError("Class-level upstream data_path must be bypassed.")

        def _get_single_subject_data(self, subject):
            paths = self.data_path(subject)
            assert len(paths) == 1
            assert Path(paths[0]).read_bytes() == b"attested"
            return {"0": {"0": object()}}

    loader = tmp_path / "attested.bin"
    loader.write_bytes(b"attested")
    dataset = SnapshotDataset()
    binding = _install_attested_data_path(dataset, {1: (loader,)})
    download_calls = _forbid_moabb_downloads(monkeypatch)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    original_connect = socket.socket.connect

    with _scoped_attested_moabb_io(dataset, [1], binding) as resolver:
        result = dataset.get_data(
            subjects=[1],
            process_pipeline=SimpleNamespace(steps=[]),
        )
        assert os.environ["MOABB_DOWNLOAD_PROVIDER"] == "upstream"

    assert result[1]["0"]["0"] is not None
    assert binding.parser_calls() == (1,)
    assert download_calls == []
    assert os.environ["MOABB_DOWNLOAD_PROVIDER"] == "nemar"
    assert socket.socket.connect is original_connect
    assert resolver.record() == {
        "schema": "n2p3_attested_moabb_resolver/1",
        "moabb_version": "1.6.1",
        "declared_nemar_id": "nm-unit-test",
        "requested_download_provider": "nemar",
        "effective_download_provider": "upstream",
        "prefetch_bypassed": True,
        "upstream_data_path_bypassed": True,
        "parser_data_path_verified": True,
        "subjects": [1],
        "parser_trust_boundary": {
            "dependency": "moabb==1.6.1",
            "role": "trusted_pinned_parser_not_process_sandbox",
        },
        "parsed_byte_authority": "n2p3_private_moabb_parser_snapshot/1",
        "network_guard": {
            "schema": "python_socket_offline_guard/1",
            "role": "defense_in_depth_not_process_network_sandbox",
        },
        "parser_snapshot_schema": "n2p3_private_moabb_parser_snapshot/1",
        "python_audit_guard": {
            "schema": "python_audit_read_only_loader_guard/1",
            "role": "defense_in_depth_not_parsed_byte_attestation",
        },
    }


def test_attested_scope_fails_closed_on_unvalidated_moabb_version(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from data import moabb as adapter_module

    loader = tmp_path / "attested.bin"
    loader.write_bytes(b"attested")
    dataset = _fake_dataset((1,))
    binding = adapter_module._install_attested_data_path(dataset, {1: (loader,)})
    monkeypatch.setattr(adapter_module.moabb, "__version__", "1.6.2")

    with pytest.raises(RuntimeError, match="validated only for moabb==1.6.1"):
        with adapter_module._scoped_attested_moabb_io(dataset, [1], binding):
            raise AssertionError("Unvalidated MOABB must fail before parsing.")


def test_attested_scope_rejects_parser_that_does_not_consume_bound_data_path(
    tmp_path: Path,
) -> None:
    from moabb.datasets.base import BaseDataset

    from data.moabb import _install_attested_data_path, _scoped_attested_moabb_io

    class NonConsumingDataset(BaseDataset):
        def __init__(self):
            super().__init__(
                subjects=[1],
                sessions_per_subject=1,
                events={"Target": 2, "NonTarget": 1},
                code="NonConsumingDataset",
                interval=[0.0, 1.0],
                paradigm="p300",
            )

        def data_path(self, subject, **kwargs):
            del subject, kwargs
            raise AssertionError("Class-level data_path must be bypassed.")

        def _get_single_subject_data(self, subject):
            if subject < 0:
                self.data_path(subject)
            return {"0": {"0": object()}}

    loader = tmp_path / "attested.bin"
    loader.write_bytes(b"attested")
    dataset = NonConsumingDataset()
    binding = _install_attested_data_path(dataset, {1: (loader,)})

    with pytest.raises(RuntimeError, match="did not consume the complete attested data_path"):
        with _scoped_attested_moabb_io(dataset, [1], binding):
            dataset.get_data(
                subjects=[1],
                process_pipeline=SimpleNamespace(steps=[]),
            )


def test_attested_scope_blocks_socket_io_and_restores_guard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import socket

    from moabb.datasets.base import BaseDataset

    from data.moabb import _install_attested_data_path, _scoped_attested_moabb_io

    class NetworkAttemptDataset(BaseDataset):
        def __init__(self):
            super().__init__(
                subjects=[1],
                sessions_per_subject=1,
                events={"Target": 2, "NonTarget": 1},
                code="NetworkAttemptDataset",
                interval=[0.0, 1.0],
                paradigm="p300",
            )

        def data_path(self, subject, **kwargs):
            del subject, kwargs
            raise AssertionError("Class-level data_path must be bypassed.")

        def _get_single_subject_data(self, subject):
            self.data_path(subject)
            socket.create_connection(("127.0.0.1", 9))
            return {"0": {"0": object()}}

    loader = tmp_path / "attested.bin"
    loader.write_bytes(b"attested")
    dataset = NetworkAttemptDataset()
    binding = _install_attested_data_path(dataset, {1: (loader,)})
    original_create_connection = socket.create_connection
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")

    with pytest.raises(RuntimeError, match="Network I/O is forbidden"):
        with _scoped_attested_moabb_io(dataset, [1], binding):
            dataset.get_data(
                subjects=[1],
                process_pipeline=SimpleNamespace(steps=[]),
            )

    assert socket.create_connection is original_create_connection
    assert os.environ["MOABB_DOWNLOAD_PROVIDER"] == "auto"


def test_bi2013a_reads_eight_sessions_only_from_attested_snapshot_mapping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from moabb.datasets import BI2013a

    from data.moabb import (
        _install_attested_data_path,
        _reported_moabb_loader_paths,
        _scoped_attested_loader_read_only,
        _scoped_attested_moabb_io,
    )

    payloads = [
        (
            f"subject_01/Session{session:02}/data_{session}.mat",
            "data",
            np.zeros((64, 17), dtype=np.float64),
        )
        for session in range(1, 9)
    ]
    attestation = _verified_braininvaders_mat_attestation(tmp_path, "BI2013a", payloads)
    materialization = attestation.materialize_moabb_loaders([1])
    dataset = BI2013a()
    binding = _install_attested_data_path(dataset, dict(materialization.paths_by_subject))
    resolutions = _reported_moabb_loader_paths(dataset, [1])
    materialization.verify_loader_paths([item.resolved_loader_path for item in resolutions])
    download_calls = _forbid_moabb_downloads(monkeypatch)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")

    with _scoped_attested_moabb_io(dataset, [1], binding) as resolver:
        with _scoped_attested_loader_read_only():
            runs = dataset.get_data(
                subjects=[1],
                process_pipeline=SimpleNamespace(steps=[]),
            )

    assert set(runs[1]) == {str(index) for index in range(8)}
    assert all(len(session_runs) == 1 for session_runs in runs[1].values())
    assert binding.parser_calls() == (1,)
    assert download_calls == []
    assert resolver.record()["declared_nemar_id"] == BI2013a.nemar_id
    assert resolver.record()["requested_download_provider"] == "auto"
    assert os.environ["MOABB_DOWNLOAD_PROVIDER"] == "auto"


def test_bi2015a_bypasses_upstream_strip_logic_with_attested_session_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from moabb.datasets import BI2015a

    from data.moabb import (
        _install_attested_data_path,
        _scoped_attested_loader_read_only,
        _scoped_attested_moabb_io,
    )

    payloads = [
        (
            f"subject_01/subject_01_session_{session:02}.mat",
            "DATA",
            np.zeros((64, 34), dtype=np.float64),
        )
        for session in range(1, 4)
    ]
    attestation = _verified_braininvaders_mat_attestation(tmp_path, "BI2015a", payloads)
    materialization = attestation.materialize_moabb_loaders([1])
    dataset = BI2015a()
    binding = _install_attested_data_path(dataset, dict(materialization.paths_by_subject))
    download_calls = _forbid_moabb_downloads(monkeypatch)

    with _scoped_attested_moabb_io(dataset, [1], binding):
        with _scoped_attested_loader_read_only():
            runs = dataset.get_data(
                subjects=[1],
                process_pipeline=SimpleNamespace(steps=[]),
            )

    assert set(runs[1]) == {"0", "1", "2"}
    assert all(set(session_runs) == {"0"} for session_runs in runs[1].values())
    assert binding.parser_calls() == (1,)
    assert download_calls == []


def test_moabb_adapter_builds_causal_cache_from_raw_runs(monkeypatch, tmp_path: Path) -> None:
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)

    info = mne.create_info(["Fz", "Cz"], 100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info, verbose=False)
    raw.set_montage("standard_1005")
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 4.0],
            duration=[0.0] * 4,
            description=["NonTarget", "Target", "NonTarget", "Target"],
        )
    )
    attestation = _verified_artifact_attestation(tmp_path, "FakeP300")
    _configure_attested_mne_root(monkeypatch, attestation)
    dataset = _fake_dataset(
        (1,), loader_path=Path(attestation.expected_mne_data_root) / "source.bin"
    )
    dataset.event_id = {"NonTarget": 1, "Target": 2}
    dataset.get_data = lambda subjects: {1: {"0": {"0": raw}}}
    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: dataset)
    profile = PreprocessingSpec(
        name="causal_mock",
        sfreq=100.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=800.0,
        n_times=100,
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )

    prepared = prepare_moabb_p300(
        "FakeP300",
        raw_artifact_attestation=attestation,
        preprocessing=profile,
    )

    assert prepared.X.shape == (4, 2, 100)
    assert prepared.preprocessing.filter_phase == "forward"
    assert prepared.event_timeline.online_causal is True
    assert prepared.provenance["raw_artifact_manifest_sha256"] == (attestation.manifest_sha256)
    assert prepared.provenance["raw_artifact_file_count"] == 2
    assert prepared.provenance["raw_artifact_actual_loader_files"] == [
        {
            "relative_path": "mne_data/source-1.bin",
            "size_bytes": len(b"physical-source:FakeP300:1"),
            "sha256": hashlib.sha256(b"physical-source:FakeP300:1").hexdigest(),
            "derived_from": "verified_manifest_file",
        }
    ]
    assert prepared.provenance["raw_artifact_loader_resolver"] == {
        "schema": "n2p3_moabb_loader_mapping/1",
        "upstream_data_path_bypassed": True,
        "workspace_role": "controlled_read_only_mne_materialization",
    }


def test_causal_subject_workers_match_serial_output_and_stabilize_order(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)
    created_workers: list[int] = []

    def inline_executor(max_workers: int):
        created_workers.append(max_workers)
        return _InlineSubjectExecutor()

    monkeypatch.setattr(adapter_module, "_create_subject_executor", inline_executor)
    serial = prepare_moabb_p300(
        "ParallelP300",
        raw_artifact_attestation=attestation,
        subjects=[2, 1],
        preprocessing=profile,
        subject_workers=1,
    )
    parallel = prepare_moabb_p300(
        "ParallelP300",
        raw_artifact_attestation=attestation,
        subjects=[2, 1],
        preprocessing=profile,
        subject_workers=4,
    )

    assert created_workers == [2]
    np.testing.assert_array_equal(parallel.X, serial.X)
    np.testing.assert_array_equal(parallel.y, serial.y)
    np.testing.assert_array_equal(parallel.subject_ids, serial.subject_ids)
    np.testing.assert_array_equal(
        parallel.event_timeline.onset_times_s,
        serial.event_timeline.onset_times_s,
    )
    pd.testing.assert_frame_equal(parallel.metadata, serial.metadata)
    assert parallel.identity_table.payload() == serial.identity_table.payload()
    assert parallel.lineage.payload() == serial.lineage.payload()
    assert parallel.metadata[["subject", "session", "run"]].drop_duplicates().values.tolist() == [
        ["2", "session-a", "run-1"],
        ["2", "session-a", "run-2"],
        ["2", "session-b", "run-1"],
        ["2", "session-b", "run-2"],
        ["1", "session-a", "run-1"],
        ["1", "session-a", "run-2"],
        ["1", "session-b", "run-1"],
        ["1", "session-b", "run-2"],
    ]
    assert parallel.provenance["subject_execution"]["requested_workers"] == 4
    assert parallel.provenance["subject_execution"]["effective_workers"] == 2
    assert parallel.provenance["subject_execution"]["execution_mode"] == ("spawn_process_subjects")
    assert parallel.provenance["subject_execution"]["shard_cleanup"]["status"] == "completed"
    assert serial.provenance["subject_execution"]["execution_mode"] == "serial_subjects"
    assert not list((Path(attestation.cache_workspace_root) / "shards").glob("run-*"))


def test_parallel_subjects_materialize_and_build_raw_provenance_only_in_parent_once(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)
    monkeypatch.setattr(
        adapter_module,
        "_create_subject_executor",
        lambda _workers: _InlineSubjectExecutor(),
    )
    calls = {"materialize": 0, "provenance": 0}
    original_materialize = RawArtifactAttestation.materialize_moabb_loaders
    original_provenance = RawArtifactAttestation.provenance_record

    def counted_materialize(self, subjects):
        calls["materialize"] += 1
        return original_materialize(self, subjects)

    def counted_provenance(self):
        calls["provenance"] += 1
        return original_provenance(self)

    monkeypatch.setattr(
        RawArtifactAttestation,
        "materialize_moabb_loaders",
        counted_materialize,
    )
    monkeypatch.setattr(RawArtifactAttestation, "provenance_record", counted_provenance)

    prepared = prepare_moabb_p300(
        "ParallelP300",
        raw_artifact_attestation=attestation,
        subjects=[1, 2],
        preprocessing=profile,
        subject_workers=4,
    )

    assert prepared.n_epochs > 0
    assert calls == {"materialize": 1, "provenance": 1}


def test_causal_subject_worker_failure_keeps_typed_journal_without_partial_shards(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)
    monkeypatch.setattr(
        adapter_module,
        "_create_subject_executor",
        lambda _workers: _InlineSubjectExecutor(),
    )
    original_worker = adapter_module._process_causal_subject_task

    def fail_second_subject(task):
        if task.subject == 2:
            raise RuntimeError("injected subject parser failure")
        return original_worker(task)

    monkeypatch.setattr(adapter_module, "_process_causal_subject_task", fail_second_subject)

    with pytest.raises(RuntimeError, match="injected subject parser failure"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1, 2],
            preprocessing=profile,
            subject_workers=4,
        )

    run_roots = list((Path(attestation.cache_workspace_root) / "shards").glob("run-*"))
    assert len(run_roots) == 1
    journal = json.loads((run_roots[0] / "subject-workers.journal.json").read_text())
    assert journal["schema"] == "n2p3_moabb_subject_journal/1"
    assert journal["run_status"] == "failed"
    assert journal["fatal_error"]["error_type"] == "RuntimeError"
    assert journal["fatal_error"]["subject"] == 2
    assert not list(run_roots[0].glob("*.npz"))
    assert not list(run_roots[0].glob("*.record.json"))


def test_parallel_failure_journal_is_failed_before_waiting_for_sibling_shutdown(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)
    observed_at_shutdown: dict[str, object] = {}

    class FailingThenPendingExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, _function, _task):
            self.calls += 1
            future = Future()
            if self.calls == 1:
                future.set_exception(RuntimeError("first subject failed"))
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True
            roots = list((Path(attestation.cache_workspace_root) / "shards").glob("run-*"))
            observed_at_shutdown.update(
                json.loads((roots[0] / "subject-workers.journal.json").read_text())
            )

    monkeypatch.setattr(
        adapter_module,
        "_create_subject_executor",
        lambda _workers: FailingThenPendingExecutor(),
    )

    with pytest.raises(RuntimeError, match="first subject failed"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1, 2],
            preprocessing=profile,
            subject_workers=4,
        )

    assert observed_at_shutdown["run_status"] == "failed"
    assert observed_at_shutdown["fatal_error"]["subject"] == 1
    assert observed_at_shutdown["subjects"] == [
        {
            "subject": 1,
            "status": "failed",
            "error_type": "RuntimeError",
            "message": "first subject failed",
            "stage": "subject_parse",
        },
        {"subject": 2, "status": "cancelled"},
    ]
    assert observed_at_shutdown["shard_cleanup"] == {"status": "pending_worker_shutdown"}


def test_parallel_worker_result_must_match_submitted_subject(monkeypatch, tmp_path: Path) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)

    class FirstResultThenPendingExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, function, task):
            self.calls += 1
            future = Future()
            if self.calls == 1:
                future.set_result(function(task))
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True

    monkeypatch.setattr(
        adapter_module,
        "_create_subject_executor",
        lambda _workers: FirstResultThenPendingExecutor(),
    )

    def wrong_subject(task):
        return adapter_module._CausalSubjectResult(
            subject=2 if task.subject == 1 else 1,
            shard_path=task.shard_path,
            n_epochs=1,
            session_runs=(("0", "0"),),
            loader_path_resolutions=(),
            runtime_resolver={"schema": "wrong-subject"},
        )

    monkeypatch.setattr(adapter_module, "_process_causal_subject_task", wrong_subject)

    with pytest.raises(RuntimeError, match="result identity mismatch: task=1, result=2"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1, 2],
            preprocessing=profile,
            subject_workers=4,
        )


def test_parallel_worker_snapshot_provenance_must_match_task_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module, "_available_cpu_count", lambda: 25)
    monkeypatch.setattr(
        adapter_module,
        "_create_subject_executor",
        lambda _workers: _InlineSubjectExecutor(),
    )
    original_worker = adapter_module._process_causal_subject_task

    def tamper_snapshot_digest(task):
        result = original_worker(task)
        snapshot = json.loads(json.dumps(result.parser_snapshot))
        snapshot["files"][0]["sha256"] = "0" * 64
        return replace(result, parser_snapshot=snapshot)

    monkeypatch.setattr(
        adapter_module,
        "_process_causal_subject_task",
        tamper_snapshot_digest,
    )

    with pytest.raises(ValueError, match="file mapping disagrees with task contracts"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1, 2],
            preprocessing=profile,
            subject_workers=4,
        )


def test_failure_cleanup_issue_does_not_mask_primary_parser_error(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        adapter_module,
        "_process_causal_subject_task",
        lambda _task: (_ for _ in ()).throw(RuntimeError("primary parser error")),
    )
    monkeypatch.setattr(
        adapter_module,
        "_clean_subject_shard_files",
        lambda _workspace: [
            {
                "path": "1.npz",
                "error_type": "PermissionError",
                "message": "locked",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="primary parser error") as exc_info:
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
            subject_workers=1,
        )

    assert any("Shard cleanup issues" in note for note in exc_info.value.__notes__)
    roots = list((Path(attestation.cache_workspace_root) / "shards").glob("run-*"))
    journal = json.loads((roots[0] / "subject-workers.journal.json").read_text())
    assert journal["fatal_error"]["message"] == "primary parser error"
    assert journal["shard_cleanup"]["status"] == "incomplete"
    assert journal["shard_cleanup"]["issues"][0]["error_type"] == "PermissionError"


def test_success_cleanup_failure_leaves_audited_tombstone(monkeypatch, tmp_path: Path) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter_module.time, "sleep", lambda _seconds: None)
    original_rmtree = adapter_module.shutil.rmtree

    def fail_tombstone_cleanup(path):
        if Path(path).name.startswith(".run-"):
            raise PermissionError("scanner lock")
        return original_rmtree(path)

    monkeypatch.setattr(
        adapter_module.shutil,
        "rmtree",
        fail_tombstone_cleanup,
    )

    with pytest.raises(RuntimeError, match="shard cleanup failed"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
            subject_workers=1,
        )

    shard_root = Path(attestation.cache_workspace_root) / "shards"
    assert not list(shard_root.glob("run-*"))
    tombstones = list(shard_root.glob(".run-*.complete-*"))
    assert len(tombstones) == 1
    journal = json.loads((tombstones[0] / "subject-workers.journal.json").read_text())
    assert journal["run_status"] == "failed"
    assert journal["fatal_error"]["stage"] == "shard_cleanup"
    assert journal["shard_cleanup"]["status"] == "incomplete"
    assert journal["shard_cleanup"]["workspace"] == tombstones[0].name


def test_cleaning_journal_failure_still_cleans_and_keeps_typed_failure(
    monkeypatch, tmp_path: Path
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    original_write = adapter_module._atomic_write_subject_journal
    injected = False

    def fail_cleaning_once(path, payload):
        nonlocal injected
        if payload["run_status"] == "cleaning" and not injected:
            injected = True
            raise PermissionError("injected cleaning journal failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        adapter_module,
        "_atomic_write_subject_journal",
        fail_cleaning_once,
    )

    with pytest.raises(PermissionError, match="injected cleaning journal failure"):
        prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
            subject_workers=1,
        )

    shard_root = Path(attestation.cache_workspace_root) / "shards"
    assert not list(shard_root.glob("run-*"))
    audit_roots = list(shard_root.glob(".run-*.journal-failure-*"))
    assert len(audit_roots) == 1
    assert not list(audit_roots[0].glob("*.npz"))
    journal = json.loads((audit_roots[0] / "subject-workers.journal.json").read_text())
    assert journal["run_status"] == "failed"
    assert journal["fatal_error"]["stage"] == "cleaning_journal"
    assert journal["shard_cleanup"]["status"] == "completed"
    assert journal["shard_cleanup"]["failure_audit_workspace"] == audit_roots[0].name


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_moabb_adapter_rejects_invalid_subject_workers(
    monkeypatch, tmp_path: Path, workers: object
) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="subject_workers must be a positive integer"):
        adapter_module.prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
            subject_workers=workers,
        )


def test_moabb_adapter_rejects_parallel_workers_for_zero_phase(monkeypatch, tmp_path: Path) -> None:
    adapter_module, attestation, profile = _causal_parallel_fixture(monkeypatch, tmp_path)
    zero_phase = PreprocessingSpec(
        **{
            **profile.__dict__,
            "filter_phase": "zero",
            "causal_iir_initial_state": "not_applicable",
        }
    )
    with pytest.raises(ValueError, match="only for forward causal"):
        adapter_module.prepare_moabb_p300(
            "ParallelP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=zero_phase,
            subject_workers=2,
        )


def test_subject_executor_uses_spawn_context_and_worker_initializer(monkeypatch) -> None:
    from data import moabb as adapter_module

    captured: dict[str, object] = {}

    class CapturedExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(adapter_module, "ProcessPoolExecutor", CapturedExecutor)
    executor = adapter_module._create_subject_executor(4)

    assert isinstance(executor, CapturedExecutor)
    assert captured["max_workers"] == 4
    assert captured["mp_context"].get_start_method() == "spawn"
    assert captured["initializer"] is adapter_module._initialize_moabb_subject_worker


def test_subject_executor_spawn_initializer_bounds_native_thread_env() -> None:
    from data import moabb as adapter_module

    profile = PreprocessingSpec(
        name="spawn_pickle_contract",
        sfreq=100.0,
        l_freq=2.0,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        filter_phase="forward",
        causal_iir_initial_state=CAUSAL_IIR_INITIAL_STATE,
    )
    task = adapter_module._CausalSubjectTask(
        dataset_class="SpawnRoundTrip",
        subject=7,
        loader_contracts=(
            adapter_module._MaterializedLoaderContract(
                path="/machine/materialized/subject-7.mat",
                relative_path="subject-7.mat",
                size_bytes=123,
                sha256="a" * 64,
            ),
        ),
        parser_snapshot_root="/machine/shards/parser-snapshots/subject-7",
        shard_path="/machine/shards/7.npz",
        persist_shard=True,
        channels=("Fz", "Cz"),
        montage="standard_1005",
        preprocessing=profile,
        target_label="Target",
        source_reference="right earlobe",
        source_sample_rate_hz=100.0,
        raw_artifact_provenance={"nested": {"digests": ["b" * 64]}},
    )
    result = adapter_module._CausalSubjectResult(
        subject=7,
        shard_path=task.shard_path,
        n_epochs=4,
        session_runs=(("2", "10"),),
        loader_path_resolutions=({"subject": 7},),
        runtime_resolver={"schema": "roundtrip"},
    )
    executor = adapter_module._create_subject_executor(1)
    try:
        assert executor.submit(os.getenv, "OMP_NUM_THREADS").result(timeout=30) == "1"
        assert executor.submit(os.getenv, "MKL_NUM_THREADS").result(timeout=30) == "1"
        assert executor.submit(os.getenv, "OPENBLAS_NUM_THREADS").result(timeout=30) == "1"
        assert executor.submit(os.getenv, "NUMEXPR_NUM_THREADS").result(timeout=30) == "1"
        assert executor.submit(_spawn_roundtrip, task).result(timeout=30) == task
        assert executor.submit(_spawn_roundtrip, result).result(timeout=30) == result
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_moabb_session_and_run_keys_use_natural_stable_order() -> None:
    from data.moabb import _sorted_mapping_items

    values = {"session-10": 10, "session-2": 2, "session-01": 1, "session-1": 11}
    ordered = _sorted_mapping_items(values, label="session")

    assert [key for key, _ in ordered] == [
        "session-01",
        "session-1",
        "session-2",
        "session-10",
    ]


def test_private_parser_snapshot_cleanup_does_not_follow_symlinks(tmp_path: Path) -> None:
    from data.moabb import _remove_private_parser_snapshot

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must-survive")
    outside_mode = outside.stat().st_mode
    snapshot_root = tmp_path / "private-snapshot"
    snapshot_root.mkdir(mode=0o700)
    link = snapshot_root / "escape-link"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    _remove_private_parser_snapshot(snapshot_root)

    assert not snapshot_root.exists()
    assert outside.read_bytes() == b"must-survive"
    assert outside.stat().st_mode == outside_mode


def test_moabb_adapter_rejects_retired_fixed_artifact_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    from data import moabb as adapter_module

    data = np.zeros((3, 2, 10), dtype=np.float64)
    data[1, 0, 3] = 2.0
    events = np.column_stack(
        [np.array([100, 200, 300]), np.zeros(3, dtype=int), np.array([1, 2, 1])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = np.arange(10, dtype=float) / 100.0

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "Target"]),
                pd.DataFrame({"subject": [1, 1, 2]}),
            )

    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: _fake_dataset())
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=1.0,
    )

    with pytest.raises(ValueError, match="Fixed absolute-voltage artifact rejection is retired"):
        prepare_moabb_p300(
            "FakeP300",
            raw_artifact_attestation=_verified_artifact_attestation(tmp_path, "FakeP300"),
            preprocessing=profile,
        )


def test_moabb_adapter_executes_declared_mean_baseline(monkeypatch, tmp_path: Path) -> None:
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)

    data = np.zeros((4, 2, 30), dtype=np.float64)
    data[:, 0] += np.arange(4, dtype=float)[:, None] + 5.0
    data[:, 1] += np.arange(4, dtype=float)[:, None] - 3.0
    data[:, :, 20:] += 2.0
    events = np.column_stack(
        [np.arange(100, 500, 100), np.zeros(4, dtype=int), np.array([1, 2, 1, 2])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = -0.2 + np.arange(30, dtype=float) / 100.0

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "NonTarget", "Target"]),
                pd.DataFrame({"subject": [1, 1, 2, 2]}),
            )

    attestation = _verified_artifact_attestation(tmp_path, "FakeP300")
    _configure_attested_mne_root(monkeypatch, attestation)
    fake_dataset = _fake_dataset(
        loader_path=Path(attestation.expected_mne_data_root) / "source.bin"
    )
    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: fake_dataset)
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_executed_baseline",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=-200.0,
        tmax_ms=100.0,
        n_times=30,
        baseline_mode="mean_only",
        reject_threshold_v=None,
    )

    dataset = prepare_moabb_p300(
        "FakeP300",
        raw_artifact_attestation=attestation,
        preprocessing=profile,
    )

    np.testing.assert_allclose(dataset.X[:, :, :20].mean(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(dataset.X[:, :, 20:], 2.0, atol=1e-12)
    assert dataset.provenance["epoch_baseline"]["mode"] == "mean_only"


def test_moabb_adapter_rejects_retired_filter_before_subject_coverage(
    monkeypatch, tmp_path: Path
) -> None:
    from data import moabb as adapter_module

    data = np.zeros((3, 2, 10), dtype=np.float64)
    data[2, 0, 3] = 2.0
    events = np.column_stack(
        [np.array([100, 200, 300]), np.zeros(3, dtype=int), np.array([1, 2, 1])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = np.arange(10, dtype=float) / 100.0

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            pass

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "NonTarget"]),
                pd.DataFrame({"subject": [1, 1, 2]}),
            )

    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: _fake_dataset(),
    )
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_subject_loss",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=1.0,
    )

    with pytest.raises(ValueError, match="Fixed absolute-voltage artifact rejection is retired"):
        prepare_moabb_p300(
            "FakeP300",
            raw_artifact_attestation=_verified_artifact_attestation(tmp_path, "FakeP300"),
            preprocessing=profile,
        )


def test_moabb_adapter_uses_explicit_candidates_not_binary_event_codes(
    monkeypatch, tmp_path: Path
) -> None:
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)

    data = np.zeros((4, 2, 10), dtype=np.float64)
    events = np.column_stack(
        [np.arange(100, 500, 100), np.zeros(4, dtype=int), np.array([7, 8, 7, 8])]
    )

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = np.arange(10, dtype=float) / 100.0

        def __init__(self):
            self.events = events

        def get_data(self):
            return data.copy()

    class FakeP300:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_data(self, **kwargs):
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target", "NonTarget", "Target"]),
                pd.DataFrame(
                    {
                        "subject": [1] * 4,
                        "selection_id": ["choice-1"] * 4,
                        "candidate_id": ["left", "right", "left", "right"],
                        "target_candidate_id": ["right"] * 4,
                        "repetition_index": np.array([0, 0, 1, 1], dtype=np.int64),
                    }
                ),
            )

    attestation = _verified_artifact_attestation(tmp_path, "FakeCandidateP300")
    _configure_attested_mne_root(monkeypatch, attestation)
    fake_dataset = _fake_dataset(
        (1,), loader_path=Path(attestation.expected_mne_data_root) / "source.bin"
    )
    monkeypatch.setattr(adapter_module, "resolve_moabb_dataset", lambda name: fake_dataset)
    monkeypatch.setattr("moabb.paradigms.P300", FakeP300)
    profile = PreprocessingSpec(
        name="mock_candidates",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=None,
    )

    dataset = prepare_moabb_p300(
        "FakeCandidateP300",
        raw_artifact_attestation=attestation,
        preprocessing=profile,
    )

    timeline = dataset.event_timeline
    assert timeline.stimulus_ids.tolist() == [7, 8, 7, 8]
    assert timeline.candidate_ids.tolist() == ["left", "right", "left", "right"]
    assert timeline.has_repetition_structure is True
    assert timeline.supports_full_candidate_chain is False


def test_moabb_adapter_rejects_transient_loader_replacement_and_restore(
    monkeypatch, tmp_path: Path
) -> None:
    from data import moabb as adapter_module

    _allow_narrow_moabb_test_double(monkeypatch)

    events = np.column_stack([np.array([100, 200]), np.zeros(2, dtype=int), np.array([1, 2])])

    class FakeEpochs:
        ch_names = ["Fz", "Cz"]
        info = {"sfreq": 100.0}
        times = np.arange(10, dtype=float) / 100.0
        filenames: tuple[str, ...] = ()

        def __init__(self):
            self.events = events

        def get_data(self):
            return np.zeros((2, 2, 10), dtype=np.float64)

    forged = tmp_path / "forged-loader.bin"
    forged.write_bytes(b"forged-parser-input")

    class MutatingP300:
        def __init__(self, **kwargs):
            del kwargs

        def get_data(self, *, dataset, subjects, **kwargs):
            del kwargs
            loader = Path(dataset.data_path(subjects[0])[0])
            backup = loader.with_name(f".{loader.name}.canonical")
            try:
                loader.replace(backup)
                forged.replace(loader)
                loader.read_bytes()
            finally:
                if backup.exists():
                    loader.unlink(missing_ok=True)
                    backup.replace(loader)
            return (
                FakeEpochs(),
                np.array(["NonTarget", "Target"]),
                pd.DataFrame({"subject": [1, 1]}),
            )

    attestation = _verified_artifact_attestation(tmp_path, "MutatingP300")
    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: _fake_dataset((1,)),
    )
    monkeypatch.setattr("moabb.paradigms.P300", MutatingP300)
    profile = PreprocessingSpec(
        name="mutating_parser",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=None,
    )

    with pytest.raises(RuntimeError, match="Filesystem mutation 'os.rename' is forbidden"):
        prepare_moabb_p300(
            "MutatingP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
        )


def test_private_parser_snapshot_ignores_parallel_source_path_replace_restore(
    monkeypatch, tmp_path: Path
) -> None:
    (
        _adapter_module,
        attestation,
        source_path,
        canonical,
        observed,
        parser_action,
        profile,
    ) = _source_race_fixture(monkeypatch, tmp_path, "ThreadRaceP300")
    parser_started = threading.Event()
    source_replaced = threading.Event()
    parser_read = threading.Event()
    backup = source_path.with_name(f".{source_path.name}.canonical")
    forged = source_path.with_name(f".{source_path.name}.forged")
    forged.write_bytes(b"forged-source-path")
    attacker_errors: list[BaseException] = []

    def attack_source_path() -> None:
        try:
            assert parser_started.wait(timeout=10)
            source_path.chmod(0o600)
            source_path.replace(backup)
            forged.replace(source_path)
            source_replaced.set()
            assert parser_read.wait(timeout=10)
        except BaseException as error:
            attacker_errors.append(error)
            source_replaced.set()
        finally:
            if backup.exists():
                if source_path.exists():
                    source_path.chmod(0o600)
                    source_path.unlink()
                backup.replace(source_path)
                source_path.chmod(0o444)

    attacker = threading.Thread(target=attack_source_path)
    attacker.start()

    def wait_for_source_replacement() -> None:
        parser_started.set()
        assert source_replaced.wait(timeout=10)

    parser_action["before"] = wait_for_source_replacement
    parser_action["after"] = parser_read.set
    try:
        prepared = prepare_moabb_p300(
            "ThreadRaceP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
        )
    finally:
        parser_started.set()
        parser_read.set()
        attacker.join(timeout=10)

    assert not attacker.is_alive()
    assert attacker_errors == []
    assert observed == [canonical]
    assert source_path.read_bytes() == canonical
    snapshot = prepared.provenance["parser_loader_snapshots"][0]
    assert snapshot["schema"] == "n2p3_private_moabb_parser_snapshot/1"
    assert snapshot["removed_after_preload"] is True
    assert snapshot["files"][0]["sha256"] == hashlib.sha256(canonical).hexdigest()


def test_private_parser_snapshot_ignores_preopened_source_fd_write_restore(
    monkeypatch, tmp_path: Path
) -> None:
    (
        _adapter_module,
        attestation,
        source_path,
        canonical,
        observed,
        parser_action,
        profile,
    ) = _source_race_fixture(monkeypatch, tmp_path, "PreopenedFdP300")
    forged = b"F" * len(canonical)
    source_path.chmod(0o600)
    source_handle = source_path.open("r+b")

    def write_forged_source() -> None:
        source_handle.seek(0)
        source_handle.write(forged)
        source_handle.flush()
        os.fsync(source_handle.fileno())

    def restore_source() -> None:
        source_handle.seek(0)
        source_handle.write(canonical)
        source_handle.flush()
        os.fsync(source_handle.fileno())

    parser_action["before"] = write_forged_source
    parser_action["after"] = restore_source
    try:
        prepared = prepare_moabb_p300(
            "PreopenedFdP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
        )
    finally:
        restore_source()
        source_handle.close()
        source_path.chmod(0o444)

    assert observed == [canonical]
    assert source_path.read_bytes() == canonical
    snapshot = prepared.provenance["parser_loader_snapshots"][0]
    assert snapshot["byte_authority"] == ("single_pass_hash_and_copy_from_stable_source_descriptor")
    assert snapshot["removed_after_preload"] is True


def test_extensionless_moabb_path_is_materialized_from_exact_zip_member(
    tmp_path: Path,
) -> None:
    from data.moabb import _install_attested_data_path, _reported_moabb_loader_paths

    attestation, _, _ = _verified_zip_artifact_attestation(tmp_path, "ExtensionlessP300")
    materialization = attestation.materialize_moabb_loaders([1])
    dataset = _fake_dataset((1,))
    _install_attested_data_path(dataset, dict(materialization.paths_by_subject))

    resolutions = _reported_moabb_loader_paths(dataset, [1])
    materialization.verify_loader_paths([resolutions[0].resolved_loader_path])
    assert resolutions[0].appendmat_resolution is False
    assert resolutions[0].record(Path(materialization.expected_mne_data_root)) == {
        "subject": 1,
        "reported_path": "extracted/group_01_s1",
        "resolved_loader_path": "extracted/group_01_s1",
        "appendmat_resolution": False,
    }
    assert materialization.attestation.derived_loader_files[0].archive_member == (
        "official/group_01_s1.mat"
    )


def test_extensionless_moabb_path_rejects_existing_base_and_mat(
    tmp_path: Path,
) -> None:
    from data.moabb import _reported_moabb_loader_paths

    reported_loader = tmp_path / "ambiguous"
    reported_loader.write_bytes(b"ambiguous-base-file")
    reported_loader.with_suffix(".mat").write_bytes(b"ambiguous-mat-file")
    dataset = _fake_dataset((1,), loader_path=reported_loader)

    with pytest.raises(ValueError, match="ambiguous extensionless"):
        _reported_moabb_loader_paths(dataset, [1])


def test_extensionless_moabb_path_rejects_when_base_and_mat_are_missing(
    tmp_path: Path,
) -> None:
    from data.moabb import _reported_moabb_loader_paths

    reported_loader = tmp_path / "missing"
    dataset = _fake_dataset((1,), loader_path=reported_loader)

    with pytest.raises(FileNotFoundError, match="both missing"):
        _reported_moabb_loader_paths(dataset, [1])


def test_moabb_adapter_rejects_missing_loader_mapping_before_parsing(
    monkeypatch, tmp_path: Path
) -> None:
    from data import moabb as adapter_module

    attestation = _verified_artifact_attestation(tmp_path, "FakeP300")
    manifest_path = Path(attestation.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_root_contract"].pop("moabb_loader_mapping")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = tmp_path / "cache-no-mapping"
    cache.mkdir()
    no_mapping = verify_raw_artifact_manifest(
        manifest_path,
        attestation.artifact_root_path,
        snapshot_root=cache / "snapshots",
        cache_workspace_root=cache,
        expected_dataset_class="FakeP300",
    )
    monkeypatch.setattr(
        adapter_module,
        "resolve_moabb_dataset",
        lambda name: _fake_dataset(),
    )
    profile = PreprocessingSpec(
        name="config_preflight",
        sfreq=100.0,
        l_freq=0.1,
        h_freq=30.0,
        tmin_ms=0.0,
        tmax_ms=100.0,
        n_times=10,
        baseline_mode="none",
        reject_threshold_v=None,
    )

    with pytest.raises(ValueError, match="moabb_loader_mapping"):
        prepare_moabb_p300(
            "FakeP300",
            raw_artifact_attestation=no_mapping,
            preprocessing=profile,
        )


@pytest.mark.parametrize(
    ("provided", "missing_option"),
    [
        (
            ["--raw-artifact-manifest", "manifest.json", "--raw-artifact-root", "raw"],
            "--raw-artifact-snapshot-root",
        ),
        (
            [
                "--raw-artifact-manifest",
                "manifest.json",
                "--raw-artifact-snapshot-root",
                "cache/snapshots",
            ],
            "--raw-artifact-root",
        ),
        (
            [
                "--raw-artifact-root",
                "raw",
                "--raw-artifact-snapshot-root",
                "cache/snapshots",
            ],
            "--raw-artifact-manifest",
        ),
    ],
)
def test_forward_moabb_cli_requires_all_physical_artifact_arguments(
    monkeypatch,
    capsys,
    provided: list[str],
    missing_option: str,
) -> None:
    from experiments.prepare_eeg_dataset import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_eeg_dataset.py",
            "moabb",
            "--dataset-class",
            "FakeP300",
            "--filter-phase",
            "forward",
            "--output",
            "cache.npz",
            *provided,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert missing_option in capsys.readouterr().err


def test_forward_moabb_cli_rejects_snapshot_outside_cache_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from experiments import prepare_eeg_dataset as cli

    attestation = _verified_artifact_attestation(tmp_path, "FakeP300")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_eeg_dataset.py",
            "moabb",
            "--dataset-class",
            "FakeP300",
            "--filter-phase",
            "forward",
            "--raw-artifact-manifest",
            attestation.manifest_path,
            "--raw-artifact-root",
            attestation.artifact_root_path,
            "--raw-artifact-snapshot-root",
            str(tmp_path.parent / "outside-cache"),
            "--output",
            str(tmp_path / "cache.npz"),
        ],
    )

    with pytest.raises(ValueError, match="beneath cache_workspace_root"):
        cli.main()
