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
    main as run_pretrain_supervised_main,
)
from experiments.run_pretrain_supervised import (
    parse_source_domain_mass,
)
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
        self.last_source_risk = None
        self._input_mean = np.zeros((1, n_channels, 1), dtype=np.float32)
        self._input_std = np.ones((1, n_channels, 1), dtype=np.float32)

    def fit(self, X, y, *, group_ids, **_) -> None:
        del X, y
        self.last_history = {
            "best_epoch": 0 if group_ids is not None else None,
            "final_task_val_auc": 0.5 if group_ids is not None else None,
        }


def test_parse_source_domain_mass_preserves_exact_domains() -> None:
    assert parse_source_domain_mass(["BI=0.8", "BNCI=0.2"]) == {
        "BI": 0.8,
        "BNCI": 0.2,
    }
    assert parse_source_domain_mass([]) == {}


@pytest.mark.parametrize(
    "value",
    [
        ["BI"],
        ["=0.8", "BNCI=0.2"],
        ["BI=0", "BNCI=1"],
        ["BI=0.8", "BI=0.2"],
        ["BI=0.7", "BNCI=0.2"],
    ],
)
def test_parse_source_domain_mass_rejects_invalid_contracts(value: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_source_domain_mass(value)


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
        "experiments.run_pretrain_supervised.loaded_epoch_cache_attestation",
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
    assert contract.optimizer["selection_config"]["precision"] == "auto"
    assert contract.optimizer["refit_config"]["precision"] == "auto"


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


@pytest.mark.parametrize(
    "obsolete_option",
    ["--subject-prefix-repeat", "--input-stats-subject-prefix"],
)
def test_supervised_runner_rejects_obsolete_prefix_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    obsolete_option: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain_supervised.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(tmp_path / "unused.json"),
            "--checkpoint",
            str(tmp_path / "unused.pt"),
            obsolete_option,
            "BI=3",
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_supervised_main()
