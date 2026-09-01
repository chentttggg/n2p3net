from __future__ import annotations

import hashlib
import json
import os
import zipfile
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
        "offline_guard": "python_socket_offline_guard/1",
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


def test_moabb_adapter_rejects_loader_mutation_during_third_party_parse(
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

    class MutatingP300:
        def __init__(self, **kwargs):
            del kwargs

        def get_data(self, *, dataset, subjects, **kwargs):
            del kwargs
            loader = Path(dataset.data_path(subjects[0])[0])
            loader.chmod(0o600)
            loader.write_bytes(b"mutated-during-parser")
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

    with pytest.raises(ValueError, match="Materialized MOABB loader.*mismatch"):
        prepare_moabb_p300(
            "MutatingP300",
            raw_artifact_attestation=attestation,
            subjects=[1],
            preprocessing=profile,
        )


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
