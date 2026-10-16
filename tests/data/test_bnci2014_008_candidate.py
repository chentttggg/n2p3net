from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from data.bnci2014_008_candidate import (
    BNCI2014_008_CANDIDATE_TASK_CONTRACT,
    BNCI2014_008_CHANNELS,
    BNCI2014_008_DATASET_ID,
    BNCI2014_008_FLASH_COUNT,
    BNCI2014_008_SAMPLE_COUNT,
    BNCI2014_008_SELECTION_COUNT,
    bnci2014_008_raw,
    build_bnci2014_008_subject_dataset,
    discover_bnci2014_008_files,
    load_bnci2014_008_candidate_record,
    read_bnci2014_008_candidate_record,
)
from data.candidate_task import validate_candidate_membership_metadata
from data.contract import (
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT,
)
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import RawSampleScope, prefix_raw_scopes, source_epoch_support
from data.raw_artifacts import (
    RAW_ARTIFACT_MANIFEST_SCHEMA,
    RAW_ARTIFACT_PATH_SEMANTICS,
    verify_raw_artifact_manifest,
)
from tests import PROJECT_ROOT
from tests.support.io import write_json
from tests.support.raw_artifacts import file_record

_FIRST_TRIAL_SAMPLE = 1800
_TRIAL_STRIDE = 9728
_FLASH_STRIDE = 64
_FLASH_DURATION = 32


@pytest.fixture
def bnci_fields() -> dict[str, object]:
    X = np.zeros((BNCI2014_008_SAMPLE_COUNT, 8), dtype=np.float32)
    y = np.zeros((BNCI2014_008_SAMPLE_COUNT, 1), dtype=np.float32)
    y_stim = np.zeros((BNCI2014_008_SAMPLE_COUNT, 1), dtype=np.int16)
    trial_starts = _FIRST_TRIAL_SAMPLE + np.arange(BNCI2014_008_SELECTION_COUNT) * _TRIAL_STRIDE
    trial = (trial_starts + 1).reshape(1, -1).astype(np.float64)
    for selection, start in enumerate(trial_starts):
        target_row = selection % 6
        target_col = (selection + 2) % 6
        for repetition in range(10):
            # Rotate acquisition order while retaining one complete candidate
            # vocabulary in each chronological repetition.
            order = np.roll(np.arange(12), repetition)
            for position, candidate in enumerate(order):
                onset = int(start + (repetition * 12 + position) * _FLASH_STRIDE)
                stop = onset + _FLASH_DURATION
                y_stim[onset:stop, 0] = int(candidate + 1)
                is_target = candidate == target_row or candidate == 6 + target_col
                y[onset:stop, 0] = 2.0 if is_target else 1.0
    return {
        "channels": np.asarray(BNCI2014_008_CHANNELS, dtype=object).reshape(1, -1),
        "X": X,
        "y": y,
        "y_stim": y_stim,
        "trial": trial,
        "classes": np.asarray(["NonTarget", "Target"], dtype=object).reshape(1, -1),
        "classes_stim": np.asarray(
            [*(f"Row{i}" for i in range(1, 7)), *(f"Col{i}" for i in range(1, 7))],
            dtype=object,
        ).reshape(1, -1),
        "gender": "synthetic",
        "age": "0",
        "ALSfrs": "0",
        "onsetALS": "synthetic",
    }


def _write_mat(path: Path, fields: dict[str, object], **extra) -> Path:
    savemat(path, {"data": fields, **extra}, do_compression=True, long_field_names=True)
    return path


def _attest_files(
    tmp_path: Path,
    paths: list[Path],
    *,
    artifact_root: Path | None = None,
):
    root = tmp_path if artifact_root is None else artifact_root
    (root / "mne_data").mkdir(parents=True, exist_ok=True)
    files = []
    for path in paths:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        files.append(file_record(path, relative, with_md5=False))
    manifest = {
        "schema": RAW_ARTIFACT_MANIFEST_SCHEMA,
        "dataset_class": BNCI2014_008_DATASET_ID,
        "official_source": {"kind": "unit_test_or_official_local_probe"},
        "official_record": {"record_id": "BNCI2014-008:test"},
        "artifact_root_contract": {
            "path_semantics": RAW_ARTIFACT_PATH_SEMANTICS,
            "expected_mne_data_root": "mne_data",
            "mne_dataset_path_key": "MNE_DATASETS_BNCI_PATH",
        },
        "files": files,
    }
    manifest_path = tmp_path / "bnci-raw-artifacts.json"
    write_json(manifest_path, manifest)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    attestation = verify_raw_artifact_manifest(
        manifest_path,
        root,
        snapshot_root=cache / "snapshots",
        cache_workspace_root=cache,
        expected_dataset_class=BNCI2014_008_DATASET_ID,
    )
    return attestation, tuple(item["relative_path"] for item in files)


def _load_attested(tmp_path: Path, path: Path):
    attestation, relative_paths = _attest_files(tmp_path, [path])
    return load_bnci2014_008_candidate_record(
        relative_paths[0], raw_artifact_attestation=attestation
    )


def _pulse_start(*, selection: int, repetition: int, candidate: int) -> int:
    order = np.roll(np.arange(12), repetition)
    position = int(np.flatnonzero(order == candidate)[0])
    return int(
        _FIRST_TRIAL_SAMPLE
        + selection * _TRIAL_STRIDE
        + (repetition * 12 + position) * _FLASH_STRIDE
    )


def test_load_and_build_synthetic_bnci_causal_candidate_dataset(
    tmp_path: Path, bnci_fields
) -> None:
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)

    record = _load_attested(tmp_path, path)

    assert record.eeg_uv.shape == (BNCI2014_008_SAMPLE_COUNT, 8)
    assert len(record.flash_sample) == BNCI2014_008_FLASH_COUNT
    assert record.flash_sample[0] == _FIRST_TRIAL_SAMPLE
    assert record.flash_sample_matlab_1based[0] == _FIRST_TRIAL_SAMPLE + 1
    assert len(np.unique(record.selection_id)) == BNCI2014_008_SELECTION_COUNT
    assert set(record.candidate_id.tolist()) == set(range(12))
    assert set(record.repetition_index.tolist()) == set(range(10))
    assert np.count_nonzero(record.raw_is_target) == 700
    assert record.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    for selection in np.unique(record.selection_id):
        rows = record.selection_id == selection
        assert len(np.unique(record.target_row[rows])) == 1
        assert len(np.unique(record.target_col[rows])) == 1

    preprocessing = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)
    dataset = build_bnci2014_008_subject_dataset(record, preprocessing=preprocessing)

    assert dataset.X.shape == (BNCI2014_008_FLASH_COUNT, 8, preprocessing.n_times)
    assert dataset.event_timeline.complete is True
    assert dataset.event_timeline.online_causal is True
    assert dataset.event_timeline.has_candidate_ids is True
    assert dataset.event_timeline.has_candidate_sets is False
    assert np.array_equal(
        dataset.metadata["flash_sample_matlab_1based"].to_numpy()[:3],
        np.asarray([1801, 1865, 1929]),
    )
    assert dataset.provenance["source_reference"] == "right earlobe"
    assert dataset.provenance["source_sample_rate_hz"] == 256.0
    receipt = dataset.provenance["preprocessing_execution"]
    assert receipt["declaration"] == preprocessing.record()
    assert receipt["output"]["scheduled_events"] == BNCI2014_008_FLASH_COUNT
    assert receipt["timing_source"] in dataset.event_timeline.timing_source
    assert dataset.provenance["candidate_task_contract"]["schema"] == (
        "n2p3_candidate_task_contract/1"
    )
    validate_candidate_membership_metadata(
        dataset.metadata,
        BNCI2014_008_CANDIDATE_TASK_CONTRACT,
        labels=dataset.y,
    )


def test_bnci_bidirectional_prefix_preserves_selection_and_ignores_future_eeg(tmp_path, bnci_fields):
    from transfer.calibration_candidates import calibration_decision_groups

    bnci_fields["X"][:] = np.sin(np.arange(BNCI2014_008_SAMPLE_COUNT)[:, None]
                                 / np.arange(7, 15)[None, :])
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    record = _load_attested(tmp_path, path)
    raw, events, labels = bnci2014_008_raw(record)
    np.testing.assert_array_equal(labels, record.raw_is_target)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    scope, _ = prefix_raw_scopes(raw, events, spec, calibration_events=144, maximum_calibration_events=240)
    dataset = build_bnci2014_008_subject_dataset(record, preprocessing=spec, raw_scope=scope, raw=raw)
    assert dataset.event_timeline.n_events == 144 and dataset.event_timeline.complete
    np.testing.assert_array_equal(dataset.event_timeline.event_ids,
                                  [f"BNCI2014-008:A01:{index}" for index in range(144)])
    assert dataset.preprocessing == spec and dataset.event_timeline.online_causal is False
    receipt = dataset.provenance["preprocessing_execution"]["source_context"]["raw_scope"]
    assert receipt["declaration"] == scope.record()
    assert receipt["operator_order"] == "crop_original_raw_before_filtering_and_epoching"
    groups, targets, _, ledger = calibration_decision_groups(dataset, np.ones(dataset.n_epochs, bool))
    assert len(groups) == 1 and groups[0].repetitions == 10 and targets == (2,)
    assert any(row["reason"] == "incomplete_requested_schedule" for row in ledger["excluded_groups"])
    changed = record.eeg_uv.copy()
    changed[scope.stop_sample:] += 1e6
    independent = build_bnci2014_008_subject_dataset(replace(record, eeg_uv=changed),
                                                     preprocessing=spec, raw_scope=scope)
    np.testing.assert_array_equal(dataset.X, independent.X)
    np.testing.assert_array_equal(dataset.y, independent.y)


def test_bnci_scoped_boundary_purge_keeps_original_physical_schedule(tmp_path, bnci_fields):
    from transfer.calibration_candidates import calibration_decision_groups

    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    record = _load_attested(tmp_path, path)
    raw, events, _ = bnci2014_008_raw(record)
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    _, ends = source_epoch_support(events, raw.info["sfreq"], spec)
    scope = RawSampleScope("selection2", int(events[240, 0] + 150), int(ends[359]), tuple(range(240, 360)))
    dataset = build_bnci2014_008_subject_dataset(record, preprocessing=spec, raw_scope=scope, raw=raw)
    timeline = dataset.event_timeline
    assert timeline.n_events == 120 and timeline.complete and dataset.n_epochs < 120
    assert timeline.event_ids[0] == "BNCI2014-008:A01:240"
    assert np.all(timeline.evidence_indices[:3] == -1)
    np.testing.assert_array_equal(timeline.candidate_ids, record.candidate_id[240:360].astype(str))
    np.testing.assert_array_equal(timeline.repetition_indices, record.repetition_index[240:360])
    groups, targets, _, _ = calibration_decision_groups(dataset, np.ones(dataset.n_epochs, bool))
    assert len(groups) == 1 and groups[0].schedule_complete and targets == (16,)
    assert np.any(groups[0].evidence_indices < 0)


def test_bnci_shared_parser_rejects_changed_declared_bytes(tmp_path, bnci_fields):
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    with pytest.raises(ValueError, match="declared source SHA256"):
        read_bnci2014_008_candidate_record(io.BytesIO(path.read_bytes()), source_relative_path="A01.mat",
                                          source_sha256="0" * 64)


def test_bnci_source_prefix_manifest_reaches_candidate_calibration(tmp_path, bnci_fields):
    from data.source_records import load_target_prefixes, prepare_source_record_prefixes
    from transfer.calibration_candidates import calibration_decision_groups

    bnci_fields["X"][:] = np.sin(np.arange(BNCI2014_008_SAMPLE_COUNT)[:, None]
                                 / np.arange(7, 15)[None, :])
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    item = {"study": "BNCI2014-008", "subject": "A01", "session": "0", "run": "0",
            "sample_rate_hz": 256., "reference": "right earlobe", "path": "A01.mat",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    prepared = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prefixes",
        budgets=(0, 200), minimum_test_trials=200, preprocessing=spec)
    manifest = tmp_path / "target_prefixes.json"
    manifest.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": spec.record(), "preparation_method": prepared["preparation_method"],
        "records": [prepared]}), encoding="utf-8")
    loaded = load_target_prefixes(manifest)["A01"]
    assert loaded["calibration"][0] is None
    calibration = loaded["calibration"][200]
    assert calibration.n_channels == 6 and calibration.event_timeline.n_events == 200
    assert calibration.provenance["candidate_task_contract"] == BNCI2014_008_CANDIDATE_TASK_CONTRACT.record()
    groups, targets, _, _ = calibration_decision_groups(calibration, np.ones(calibration.n_epochs, bool))
    assert len(groups) == 1 and targets == (2,)
    inner = loaded["early_stopping_inner"][200]
    assert inner["status"] == "ready"
    training = inner["training"]
    assert len(calibration_decision_groups(training, np.ones(training.n_epochs, bool))[0]) == 1
    assert loaded["test"].event_timeline.event_ids[0] == "BNCI2014-008:A01:200"
    assert not set(calibration.event_timeline.event_ids) & set(loaded["test"].event_timeline.event_ids)
    test = loaded["test"]
    test_groups, _, _, test_ledger = calibration_decision_groups(test, np.ones(test.n_epochs, bool))
    assert len(test_groups) == 33  # The initial tail of selection 01 is never a complete decision.
    assert any(row["reason"] == "incomplete_requested_schedule" for row in test_ledger["excluded_groups"])


def _off_by_one_trial(fields: dict[str, object]) -> None:
    trial = fields["trial"]
    trial[0, 0] -= 1


def _missing_candidate_coverage(fields: dict[str, object]) -> None:
    y_stim = fields["y_stim"]
    start = _pulse_start(selection=0, repetition=0, candidate=11)
    y_stim[start : start + _FLASH_DURATION, 0] = 11


def _drifting_target(fields: dict[str, object]) -> None:
    y = fields["y"]
    original_targets = (0, 8)
    replacement_targets = (3, 10)
    for candidate in original_targets:
        start = _pulse_start(selection=0, repetition=1, candidate=candidate)
        y[start : start + _FLASH_DURATION, 0] = 1
    for candidate in replacement_targets:
        start = _pulse_start(selection=0, repetition=1, candidate=candidate)
        y[start : start + _FLASH_DURATION, 0] = 2


def _short_flash(fields: dict[str, object]) -> None:
    y = fields["y"]
    y_stim = fields["y_stim"]
    stop = _FIRST_TRIAL_SAMPLE + _FLASH_DURATION
    y[stop - 1, 0] = 0
    y_stim[stop - 1, 0] = 0


def _direct_nonzero_transition(fields: dict[str, object]) -> None:
    y = fields["y"]
    y_stim = fields["y_stim"]
    stop = _FIRST_TRIAL_SAMPLE + _FLASH_DURATION
    y[stop, 0] = 1
    y_stim[stop, 0] = 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_off_by_one_trial, "MATLAB 1-based index"),
        (_missing_candidate_coverage, "cover each candidate 0..11 exactly once"),
        (_drifting_target, "target row/column drifts"),
        (_short_flash, "lasts 31 samples"),
        (_direct_nonzero_transition, "nonzero-to-nonzero transition"),
    ],
)
def test_rejects_schedule_counterexamples(
    tmp_path: Path,
    bnci_fields,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    mutate(bnci_fields)
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)

    with pytest.raises(ValueError, match=message):
        _load_attested(tmp_path, path)


def test_rejects_nonfinite_signal(tmp_path: Path, bnci_fields) -> None:
    bnci_fields["X"][0, 0] = np.nan
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    with pytest.raises(ValueError, match="X.*non-finite"):
        _load_attested(tmp_path, path)


def test_rejects_float_stimulus_trace_even_when_integer_valued(tmp_path: Path, bnci_fields) -> None:
    bnci_fields["y_stim"] = bnci_fields["y_stim"].astype(np.float64)
    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    with pytest.raises(ValueError, match="y_stim.*unsupported dtype"):
        _load_attested(tmp_path, path)


def test_rejects_unknown_top_level_mat_variable_before_loading_data(
    tmp_path: Path, bnci_fields
) -> None:
    path = _write_mat(tmp_path / "A01.mat", bnci_fields, unknown=np.asarray([1], dtype=np.int64))

    with pytest.raises(ValueError, match="exactly one 1x1 Level-5 struct"):
        _load_attested(tmp_path, path)


def test_discover_requires_exact_eight_file_inventory(tmp_path: Path) -> None:
    for index in range(1, 8):
        (tmp_path / f"A{index:02d}.mat").touch()
    attestation, _ = _attest_files(
        tmp_path, [tmp_path / f"A{index:02d}.mat" for index in range(1, 8)]
    )
    with pytest.raises(ValueError, match=r"missing=\['A08.mat'\]"):
        discover_bnci2014_008_files(attestation)

    (tmp_path / "A08.mat").touch()
    (tmp_path / "A09.mat").touch()
    attestation, _ = _attest_files(
        tmp_path, [tmp_path / f"A{index:02d}.mat" for index in range(1, 10)]
    )
    with pytest.raises(ValueError, match=r"unexpected=\['A09.mat'\]"):
        discover_bnci2014_008_files(attestation)

    (tmp_path / "A09.mat").unlink()
    attestation, _ = _attest_files(
        tmp_path, [tmp_path / f"A{index:02d}.mat" for index in range(1, 9)]
    )
    assert [Path(path).name for path in discover_bnci2014_008_files(attestation)] == [
        f"A{index:02d}.mat" for index in range(1, 9)
    ]


def test_bnci_parser_ignores_original_replacement_during_snapshot_parse(
    monkeypatch, tmp_path: Path, bnci_fields
) -> None:
    from data import bnci2014_008_candidate as module

    path = _write_mat(tmp_path / "A01.mat", bnci_fields)
    attestation, relative_paths = _attest_files(tmp_path, [path])
    original = module._read_level5_struct

    def replace_original(handle, *, source_name):
        path.write_bytes(b"replaced-during-snapshot-parse")
        return original(handle, source_name=source_name)

    monkeypatch.setattr(module, "_read_level5_struct", replace_original)

    record = load_bnci2014_008_candidate_record(
        relative_paths[0], raw_artifact_attestation=attestation
    )
    assert record.source_sha256 == attestation.verified_snapshots[0].sha256
    assert record.eeg_uv.shape == (BNCI2014_008_SAMPLE_COUNT, 8)


@pytest.mark.parametrize(
    "missing_option",
    [
        "--raw-artifact-manifest",
        "--raw-artifact-root",
        "--raw-artifact-snapshot-root",
    ],
)
def test_bnci_cli_requires_all_snapshot_arguments(
    monkeypatch,
    capsys,
    missing_option: str,
) -> None:
    from experiments.prepare_bnci2014_008_candidate import main

    required = {
        "--raw-artifact-manifest": "manifest.json",
        "--raw-artifact-root": "raw",
        "--raw-artifact-snapshot-root": "cache/snapshots",
    }
    del required[missing_option]
    provided = [token for pair in required.items() for token in pair]

    monkeypatch.setattr(
        "sys.argv",
        ["prepare_bnci2014_008_candidate.py", "--output", "cache.npz", *provided],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    assert missing_option in capsys.readouterr().err


_REAL_A01 = (
    PROJECT_ROOT
    / "mne_data"
    / "MNE-bnci-data"
    / "~bci"
    / "database"
    / "008-2014"
    / "A01.mat"
)


@pytest.mark.skipif(not _REAL_A01.is_file(), reason="official local A01.mat is unavailable")
def test_official_a01_read_only_contract_probe(tmp_path: Path) -> None:
    repository_root = PROJECT_ROOT
    attestation, relative_paths = _attest_files(
        tmp_path, [_REAL_A01], artifact_root=repository_root
    )
    record = load_bnci2014_008_candidate_record(
        relative_paths[0], raw_artifact_attestation=attestation
    )

    assert record.eeg_uv.shape == (347704, 8)
    assert len(record.flash_sample) == 4200
    assert record.flash_sample[0] == 1800
    assert record.flash_sample_matlab_1based[0] == 1801
    assert len(np.unique(record.selection_id)) == 35
    assert record.source_sha256 == (
        "cdf9bca4f48c61ee9c9ba998382a4e84392918c9dbc5a43f45ee0052614e25bd"
    )
