from dataclasses import replace

import pytest
import torch

import data.epochs as epochs_module
from data.bi2014a_candidate import build_bi2014a_subject_dataset
from data.exclusions import (
    assert_identity_participants_allowed,
    select_allowed_subjects,
)
from data.identity import DatasetIdentityTable
from tests.support.candidate import candidate_dataset
from tests.support.checkpoints import candidate_checkpoint_payload
from tests.support.datasets import epoch_dataset
from transfer.checkpoint import (
    checkpoint_training_contract,
    checkpoint_training_identity_ledger,
    load_checkpoint_payload,
)


@pytest.mark.parametrize("source", ["BI2014a", "BrainInvaders2014a", "BI2014a-candidate:62"])
def test_exclusion_survives_local_namespace_changes(source):
    table = DatasetIdentityTable.from_source_rows(["62"], [source])
    table = table.relabel_local_subjects({"62": "new_namespace:innocent_local_id"})
    with pytest.raises(ValueError, match="excluded BI2014a"):
        assert_identity_participants_allowed(table)


def test_default_selection_and_explicit_rejection_are_distinct():
    assert select_allowed_subjects("BI2014a", list(range(1, 65)), explicit=False) == list(range(1, 62))
    with pytest.raises(ValueError, match="explicitly requested"):
        select_allowed_subjects("BI2014a", [61, 62], explicit=True)
    assert select_allowed_subjects("BI2015a", [62, 63, 64], explicit=True) == [62, 63, 64]
    assert_identity_participants_allowed(DatasetIdentityTable.from_source_rows(["63"], ["BNCI2014-008"]))


@pytest.mark.parametrize("validation", ["full", "attested"])
def test_existing_cache_is_rejected_even_with_valid_attestation(tmp_path, monkeypatch, validation):
    dataset = epoch_dataset()
    dataset.identity_table = DatasetIdentityTable.from_source_rows(
        ["61", "62", "63"], ["BI2014a"] * 3
    ).relabel_local_subjects({"61": "s1", "62": "s2", "63": "s3"})
    path = tmp_path / "historical_cache.npz"
    # Simulate bytes created before the exclusion policy took effect.
    with monkeypatch.context() as context:
        context.setattr(epochs_module, "assert_identity_participants_allowed", lambda table: None)
        epochs_module.save_epoch_dataset(path, dataset)
    with pytest.raises(ValueError, match="excluded BI2014a"):
        epochs_module.load_epoch_dataset(path, validation=validation)


def test_checkpoint_with_excluded_source_cannot_be_reused(tmp_path):
    dataset = candidate_dataset()
    epochs_module.materialize_dataset_identity(dataset)
    payload = candidate_checkpoint_payload(
        dataset, source_snapshot_sha256="a" * 64
    )
    ledger = DatasetIdentityTable.from_source_rows(["63"], ["BI2014a-candidate:63"])
    contract = replace(
        checkpoint_training_contract(payload), source_identity_digest=ledger.digest(),
        training_participant_keys=ledger.authority_keys(),
    )
    payload.update(training_identity_ledger=ledger.payload(), training_identity_ledger_digest=ledger.digest(),
                   training_contract=contract.record(), training_contract_digest=contract.digest())
    path = tmp_path / "historical_source.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="excluded BI2014a"):
        load_checkpoint_payload(path)
    with pytest.raises(ValueError, match="excluded BI2014a"):
        checkpoint_training_identity_ledger(payload)


def test_direct_raw_builder_rejects_before_reading_deleted_files(tmp_path):
    with pytest.raises(ValueError, match="explicitly requested"):
        build_bi2014a_subject_dataset(tmp_path / "subject_64", preprocessing=epoch_dataset().preprocessing)
