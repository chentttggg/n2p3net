from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
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
    build_bnci2014_008_subject_dataset,
    discover_bnci2014_008_files,
    load_bnci2014_008_candidate_record,
)
from data.candidate_task import validate_candidate_membership_metadata
from data.contract import SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT
from data.epochs import EPOCH_DATASET_SCHEMA, preprocessing_spec_from_contract
from data.raw_artifacts import (
    RAW_ARTIFACT_MANIFEST_SCHEMA,
    RAW_ARTIFACT_PATH_SEMANTICS,
    verify_raw_artifact_manifest,
)

_FIRST_TRIAL_SAMPLE = 1800
_TRIAL_STRIDE = 9728
_FLASH_STRIDE = 64
_FLASH_DURATION = 32


def _synthetic_fields() -> dict[str, object]:
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


def _write_mat(
    path: Path,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
    top_level_extra: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _synthetic_fields()
    if mutate is not None:
        mutate(fields)
    payload: dict[str, object] = {"data": fields}
    if top_level_extra:
        payload["unknown"] = np.asarray([1], dtype=np.int64)
    savemat(path, payload, do_compression=True, long_field_names=True)
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
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files.append(
            {
                "relative_path": relative,
                "size_bytes": len(payload),
                "official_md5": None,
                "local_sha256": digest,
                "remote_sha256": digest,
                "verified": True,
            }
        )
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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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


def test_load_synthetic_bnci_candidate_schedule(tmp_path: Path) -> None:
    path = _write_mat(tmp_path / "A01.mat")

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


def test_build_synthetic_bnci_causal_epoch_dataset(tmp_path: Path) -> None:
    path = _write_mat(tmp_path / "A01.mat")
    preprocessing = preprocessing_spec_from_contract(SINGLE_SUBJECT_CAUSAL_P300_DATA_CONTRACT)

    record = _load_attested(tmp_path, path)
    dataset = build_bnci2014_008_subject_dataset(record, preprocessing=preprocessing)

    assert EPOCH_DATASET_SCHEMA == "n2p3net_epoch_dataset/5"
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
    assert dataset.provenance["candidate_task_contract"]["schema"] == (
        "n2p3_candidate_task_contract/1"
    )
    validate_candidate_membership_metadata(
        dataset.metadata,
        BNCI2014_008_CANDIDATE_TASK_CONTRACT,
        labels=dataset.y,
    )


def _off_by_one_trial(fields: dict[str, object]) -> None:
    trial = np.asarray(fields["trial"])
    trial[0, 0] -= 1


def _missing_candidate_coverage(fields: dict[str, object]) -> None:
    y_stim = np.asarray(fields["y_stim"])
    start = _pulse_start(selection=0, repetition=0, candidate=11)
    y_stim[start : start + _FLASH_DURATION, 0] = 11


def _drifting_target(fields: dict[str, object]) -> None:
    y = np.asarray(fields["y"])
    original_targets = (0, 8)
    replacement_targets = (3, 10)
    for candidate in original_targets:
        start = _pulse_start(selection=0, repetition=1, candidate=candidate)
        y[start : start + _FLASH_DURATION, 0] = 1
    for candidate in replacement_targets:
        start = _pulse_start(selection=0, repetition=1, candidate=candidate)
        y[start : start + _FLASH_DURATION, 0] = 2


def _short_flash(fields: dict[str, object]) -> None:
    y = np.asarray(fields["y"])
    y_stim = np.asarray(fields["y_stim"])
    stop = _FIRST_TRIAL_SAMPLE + _FLASH_DURATION
    y[stop - 1, 0] = 0
    y_stim[stop - 1, 0] = 0


def _direct_nonzero_transition(fields: dict[str, object]) -> None:
    y = np.asarray(fields["y"])
    y_stim = np.asarray(fields["y_stim"])
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
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    path = _write_mat(tmp_path / "A01.mat", mutate=mutate)

    with pytest.raises(ValueError, match=message):
        _load_attested(tmp_path, path)


def test_rejects_nonfinite_signal(tmp_path: Path) -> None:
    def mutate(fields: dict[str, object]) -> None:
        np.asarray(fields["X"])[0, 0] = np.nan

    path = _write_mat(tmp_path / "A01.mat", mutate=mutate)

    with pytest.raises(ValueError, match="X.*non-finite"):
        _load_attested(tmp_path, path)


def test_rejects_float_stimulus_trace_even_when_integer_valued(tmp_path: Path) -> None:
    def mutate(fields: dict[str, object]) -> None:
        fields["y_stim"] = np.asarray(fields["y_stim"], dtype=np.float64)

    path = _write_mat(tmp_path / "A01.mat", mutate=mutate)

    with pytest.raises(ValueError, match="y_stim.*unsupported dtype"):
        _load_attested(tmp_path, path)


def test_rejects_unknown_top_level_mat_variable_before_loading_data(tmp_path: Path) -> None:
    path = _write_mat(tmp_path / "A01.mat", top_level_extra=True)

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
    monkeypatch, tmp_path: Path
) -> None:
    from data import bnci2014_008_candidate as module

    path = _write_mat(tmp_path / "A01.mat")
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
def test_bnci_cli_requires_all_snapshot_arguments(
    monkeypatch,
    capsys,
    provided: list[str],
    missing_option: str,
) -> None:
    from experiments.prepare_bnci2014_008_candidate import main

    monkeypatch.setattr(
        "sys.argv",
        ["prepare_bnci2014_008_candidate.py", "--output", "cache.npz", *provided],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    assert missing_option in capsys.readouterr().err


_REAL_A01 = (
    Path(__file__).resolve().parent.parent
    / "mne_data"
    / "MNE-bnci-data"
    / "~bci"
    / "database"
    / "008-2014"
    / "A01.mat"
)


@pytest.mark.skipif(not _REAL_A01.is_file(), reason="official local A01.mat is unavailable")
def test_official_a01_read_only_contract_probe(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parent.parent
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
