from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from data.epochs import (
    save_epoch_dataset,
)
from data.identity import DatasetIdentityTable, origin_subject_key
from experiments.run_pretrain import _source_training_rows, _subject_probe_validation_mask
from experiments.run_pretrain import main as run_pretrain_main
from tests.support.candidate import causal_candidate_dataset
from tests.support.checkpoints import (
    small_trunk,
)
from tests.support.snapshots import write_source_snapshot
from transfer.checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_training_contract,
)
from transfer.pretraining import PretrainingConfig, PretrainingTask


def test_pretraining_task_loss_and_subject_probe() -> None:
    trunk = small_trunk()
    config = PretrainingConfig(subject_probe_subjects=4)
    task = PretrainingTask(trunk, config)
    x = torch.randn(8, 3, 128)
    ids = torch.randint(0, 4, (8,))
    components = task.loss_components(x, subject_ids=ids)
    assert set(components) == {
        "waveform",
        "spectral",
        "total",
        "subject_probe",
        "subject_probe_correct",
    }
    assert torch.isfinite(components["total"])
    task.zero_grad(set_to_none=True)
    components["subject_probe"].backward()
    assert all(parameter.grad is None for parameter in task.trunk.parameters())
    assert any(parameter.grad is not None for parameter in task.probe.parameters())
    task.discard_decoder()
    assert task.decoder is None and task.probe is None


def test_pretraining_holdout_must_exist_in_the_source_cache() -> None:
    rows, subjects = _source_training_rows(
        np.asarray(["s1", "s2", "s1"]),
        {"s2"},
    )
    assert subjects == {"s1", "s2"}
    assert rows.tolist() == [True, False, True]

    with pytest.raises(ValueError, match="absent from the source cache"):
        _source_training_rows(np.asarray(["s1", "s2"]), {"typo"})


def test_subject_probe_validation_split_holds_out_each_subject() -> None:
    subjects = np.repeat(np.asarray(["s1", "s2", "s3"]), 10)
    first = _subject_probe_validation_mask(subjects, seed=11)
    second = _subject_probe_validation_mask(subjects, seed=11)

    assert np.array_equal(first, second)
    assert [int(first[subjects == subject].sum()) for subject in np.unique(subjects)] == [2, 2, 2]


def test_pretraining_runner_records_trained_stop_gradient_subject_probe(
    tmp_path, monkeypatch
) -> None:
    dataset = causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "pretrain.npz", dataset)
    checkpoint = tmp_path / "pretrained.pt"
    source_manifest, source_digest, _ = write_source_snapshot(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(cache),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(checkpoint),
            "--cohort",
            "causal",
            "--holdout-subjects",
            "g3",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--device",
            "cpu",
        ],
    )

    run_pretrain_main()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["classifier_trained"] is False
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["subject_probe_audit"]["enabled"] is True
    assert payload["subject_probe_audit"]["stop_gradient"] is True
    assert payload["subject_probe_audit"]["n_subjects"] == 2
    assert payload["subject_probe_audit"]["final_validation_accuracy"] is not None
    assert "training_subject_keys" not in payload
    ledger = DatasetIdentityTable.from_payload(payload["training_identity_ledger"])
    assert ledger.local_subject_ids == ("g1", "g2")
    assert "g3" not in ledger.local_subject_ids
    assert payload["training_identity_ledger_digest"] == ledger.digest()
    contract = checkpoint_training_contract(payload)
    assert contract.source_snapshot_sha256 == source_digest
    assert payload["source_snapshot_sha256"] == source_digest
    assert payload["source_snapshot_manifest"] == str(source_manifest.resolve())
    assert set(contract.training_participant_keys) == set(ledger.authority_keys())
    assert origin_subject_key("synthetic", "g3") in contract.holdout_participant_keys
    changed = replace(
        contract,
        objective={**contract.objective, "waveform_weight": 99.0},
    )
    assert changed.digest() != contract.digest()
    mutated_payload = {
        **payload,
        "training_contract": changed.record(),
    }
    with pytest.raises(ValueError, match="training_contract_digest"):
        checkpoint_training_contract(mutated_payload)


def test_pretraining_runner_requires_source_snapshot_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "source.npz"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_main()


def test_pretraining_runner_rejects_legacy_naked_snapshot_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "source.npz"),
            "--source-snapshot-sha256",
            "a" * 64,
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
        ],
    )

    with pytest.raises(SystemExit):
        run_pretrain_main()


def test_pretraining_runner_rejects_tampered_source_archive(tmp_path, monkeypatch) -> None:
    source_manifest, _, archive = write_source_snapshot(tmp_path)
    tampered = bytearray(archive.read_bytes())
    tampered[4] ^= 1
    archive.write_bytes(tampered)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pretrain.py",
            "--source-cache",
            str(tmp_path / "unused.npz"),
            "--source-snapshot-manifest",
            str(source_manifest),
            "--checkpoint",
            str(tmp_path / "must-not-exist.pt"),
        ],
    )

    with pytest.raises(ValueError, match="archive hash disagrees"):
        run_pretrain_main()
