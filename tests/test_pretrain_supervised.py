from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.contract import SOURCE_COHORT_DATA_CONTRACTS
from data.epochs import preprocessing_spec_from_contract
from data.identity import (
    DatasetIdentityTable,
    training_identity_ledger_from_rows,
)
from experiments.run_pretrain_supervised import (
    build_subject_prefix_exposure,
    parse_subject_prefix_repeats,
)
from experiments.run_pretrain_supervised import main as run_pretrain_supervised_main
from models.n2p3net import N2P3Net
from transfer.checkpoint import checkpoint_training_contract


def _write_source_snapshot_manifest(
    directory: Path,
) -> tuple[Path, str, Path]:
    source = directory / "snapshot_source.py"
    source.write_text("SNAPSHOT_VALUE = 1\n", encoding="utf-8")
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
                "source_commit": "a" * 40,
                "member_count": 1,
                "byte_size": archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    return manifest, digest, archive


def _supervised_source_dataset() -> SimpleNamespace:
    contract = SOURCE_COHORT_DATA_CONTRACTS["causal"]
    subjects = np.repeat(np.asarray(["s1", "s2", "s3", "s4"]), 250)
    identity = DatasetIdentityTable.from_source_rows(
        subjects.tolist(),
        ["synthetic-source"] * len(subjects),
    )
    return SimpleNamespace(
        X=np.zeros((len(subjects), 3, contract.n_times), dtype=np.float32),
        y=np.tile(np.asarray([0, 1], dtype=np.int64), len(subjects) // 2),
        subject_ids=subjects,
        identity_table=identity,
        n_channels=3,
        n_times=contract.n_times,
        preprocessing=preprocessing_spec_from_contract(contract),
        channel_names=("Fz", "Cz", "Pz"),
        provenance={"source_reference": "synthetic-source"},
        name="synthetic_supervised_source",
    )


class _ExecutionRecord:
    def record(self) -> dict[str, object]:
        return {"optimizer": "test-double", "compiled": False}


class _FastSupervisedBaseline:
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        sfreq: float,
        *,
        tmin_s: float,
        pooling_mode: str,
        architecture,
        **_,
    ) -> None:
        self.model_ = N2P3Net(
            n_channels=n_channels,
            n_times=n_times,
            sfreq=sfreq,
            tmin_s=tmin_s,
            pooling_mode=pooling_mode,
            **architecture.model_kwargs(),
        )
        self.optimizer_execution = _ExecutionRecord()
        self.training_pos_weight_ = 1.0
        self.training_prior_ = 0.5
        self.calibration_logits_ = None
        self.calibration_labels_ = None
        self.calibration_source_ = None
        self.last_history: dict[str, object] = {}
        self._input_mean = np.zeros((1, n_channels, 1), dtype=np.float32)
        self._input_std = np.ones((1, n_channels, 1), dtype=np.float32)

    def fit(self, X, y, *, group_ids, input_stats_row_mask) -> None:
        del X, y, input_stats_row_mask
        self.last_history = {
            "best_epoch": 0 if group_ids is not None else None,
            "final_task_val_auc": 0.5 if group_ids is not None else None,
        }


def test_parse_subject_prefix_repeats_preserves_explicit_prefixes() -> None:
    assert parse_subject_prefix_repeats("BI::=3, BNCI::=1") == {
        "BI::": 3,
        "BNCI::": 1,
    }
    assert parse_subject_prefix_repeats("") == {}


@pytest.mark.parametrize("value", ["BI::", "=3", "BI::=0", "BI::=1.5", "BI::=2,BI::=3"])
def test_parse_subject_prefix_repeats_rejects_invalid_contracts(value: str) -> None:
    with pytest.raises(ValueError):
        parse_subject_prefix_repeats(value)


def test_subject_prefix_exposure_retains_rows_and_accounts_optimizer_rows() -> None:
    subjects = np.asarray(["BI::01", "BI::01", "BI::02", "BNCI::01", "other"])
    indices, report = build_subject_prefix_exposure(
        subjects,
        {"BI::": 3, "BNCI::": 1},
    )

    assert np.bincount(indices, minlength=len(subjects)).tolist() == [3, 3, 3, 1, 1]
    physical_values = np.arange(len(subjects), dtype=float)
    assert physical_values[indices].mean() == pytest.approx(
        (physical_values[:3].sum() * 3 + physical_values[3:].sum()) / 11
    )
    assert report["unique_physical_rows"] == 5
    assert report["optimizer_rows_per_epoch"] == 11
    assert report["all_unique_rows_retained"] is True
    assert report["prefixes"] == [
        {
            "prefix": "BI::",
            "repeat": 3,
            "unique_physical_rows": 3,
            "optimizer_rows": 9,
            "unique_subjects": 2,
            "optimizer_fraction": 9 / 11,
        },
        {
            "prefix": "BNCI::",
            "repeat": 1,
            "unique_physical_rows": 1,
            "optimizer_rows": 1,
            "unique_subjects": 1,
            "optimizer_fraction": 1 / 11,
        },
        {
            "prefix": None,
            "repeat": 1,
            "unique_physical_rows": 1,
            "optimizer_rows": 1,
            "unique_subjects": 1,
            "optimizer_fraction": 1 / 11,
        },
    ]


def test_subject_prefix_exposure_rejects_unknown_and_overlapping_prefixes() -> None:
    subjects = np.asarray(["BI::01", "BNCI::01"])
    with pytest.raises(ValueError, match="matches no retained source rows"):
        build_subject_prefix_exposure(subjects, {"GTN::": 2})
    with pytest.raises(ValueError, match="prefixes overlap"):
        build_subject_prefix_exposure(subjects, {"BI": 2, "BI::": 3})


def test_training_identity_ledger_uses_only_rows_retained_after_all_gates() -> None:
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s3"])
    table = DatasetIdentityTable.from_source_rows(
        subjects.tolist(),
        ["source"] * len(subjects),
    )
    post_holdout_and_qc = np.asarray([True, False, False, False, True])

    ledger = training_identity_ledger_from_rows(
        table,
        subjects,
        post_holdout_and_qc,
    )

    assert ledger.local_subject_ids == ("s1", "s3")
    assert "s2" not in ledger.local_subject_ids


def test_training_identity_ledger_rejects_index_list_as_ambiguous_mask() -> None:
    subjects = np.asarray(["s1", "s2"])
    table = DatasetIdentityTable.from_source_rows(subjects.tolist(), ["source"] * 2)

    with pytest.raises(ValueError, match="boolean mask"):
        training_identity_ledger_from_rows(table, subjects, [0, 1])


def test_supervised_runner_records_verified_physical_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _supervised_source_dataset()
    source_manifest, source_digest, _ = _write_source_snapshot_manifest(tmp_path)
    checkpoint = tmp_path / "supervised.pt"
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.load_epoch_dataset",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.read_epoch_cache_attestation",
        lambda *_args, **_kwargs: {"sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "experiments.run_pretrain_supervised.N2P3NetBaseline",
        _FastSupervisedBaseline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "synthetic.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )

    run_pretrain_supervised_main()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint_training_contract(payload)
    assert contract.source_snapshot_sha256 == source_digest
    assert payload["source_snapshot_sha256"] == source_digest
    assert payload["source_snapshot_manifest"] == str(source_manifest.resolve())


def test_supervised_runner_rejects_tampered_source_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, _, archive = _write_source_snapshot_manifest(tmp_path)
    tampered = bytearray(archive.read_bytes())
    tampered[4] ^= 1
    archive.write_bytes(tampered)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(tmp_path / "must-not-exist.pt"),
        ],
    )

    with pytest.raises(ValueError, match="archive hash disagrees"):
        run_pretrain_supervised_main()


def test_supervised_runner_rejects_legacy_naked_snapshot_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-sha256",
            "a" * 64,
            "--checkpoint",
            str(tmp_path / "unused.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_supervised_main()
