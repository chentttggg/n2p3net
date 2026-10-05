"""Synthetic checkpoints builders for contract tests."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from data.epochs import EpochDataset, materialize_dataset_identity
from data.identity import DatasetIdentityTable, ParticipantIdentityRecord, origin_subject_key
from models.n2p3net import N2P3Net
from research.contracts import TrainingRunContract, semantic_sha256
from transfer.checkpoint import CHECKPOINT_SCHEMA


def _checkpoint_fields(dataset, table, contract):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "source_cache_sha256": contract.source_cache_sha256,
        "architecture": dict(contract.architecture),
        "input_preprocessing": asdict(dataset.preprocessing),
        "input_channel_names": list(dataset.channel_names),
        "input_source_reference": dataset.provenance["source_reference"],
        "classifier_trained": True,
        "input_mean": [0.0] * dataset.n_channels,
        "input_std": [1.0] * dataset.n_channels,
        "n_channels": dataset.n_channels,
        "n_times": dataset.n_times,
        "input_sample_rate_hz": dataset.preprocessing.sfreq,
        "input_tmin_s": dataset.preprocessing.tmin_ms / 1000.0,
        "training_identity_ledger": table.payload(),
        "training_identity_ledger_digest": table.digest(),
        "training_contract": contract.record(),
        "training_contract_digest": contract.digest(),
    }


def write_checkpoint(
    path: Path,
    dataset: EpochDataset,
    *,
    source_snapshot_sha256: str,
) -> Path:
    trunk = N2P3Net(
        dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="full_unfold",
        temporal_kernel_size=35,
    )
    training_identity_ledger = DatasetIdentityTable.from_source_rows(["other"], ["external-source"])
    architecture = trunk.architecture_record()
    training_contract = TrainingRunContract(
        source_cache_sha256="a" * 64,
        source_identity_digest=training_identity_ledger.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=architecture,
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": dataset.provenance["source_reference"],
        },
        optimizer={"kind": "synthetic_test_fixture"},
        validation={"kind": "synthetic_test_fixture"},
        objective={"kind": "supervised_binary_cross_entropy"},
        seed=0,
        training_participant_keys=training_identity_ledger.authority_keys("source"),
        holdout_participant_keys=(),
    )
    payload = _checkpoint_fields(dataset, training_identity_ledger, training_contract)
    payload.update(
        trunk_state_dict=trunk.state_dict(),
        source_dataset_name="external-source",
        source_calibration={
            "pos_weight": 8.0,
            "train_prior": 1.0 / 9.0,
            "temperature": 1.0,
            "source": "synthetic_test_fixture",
        },
    )
    torch.save(payload, path)
    return path


def candidate_checkpoint_payload(
    dataset: EpochDataset, *, source_snapshot_sha256: str
) -> dict[str, object]:
    target_identity = materialize_dataset_identity(dataset)
    trunk = N2P3Net(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
    )
    training_identity_ledger = DatasetIdentityTable.from_source_rows(["other"], [dataset.name])
    holdouts = tuple(
        sorted(
            {
                key
                for identity in target_identity.records
                for key in (*identity.origin_subject_keys, *identity.global_person_keys)
            }
        )
    )
    training_contract = TrainingRunContract(
        source_cache_sha256=semantic_sha256({"fixture": "source-cache"}),
        source_identity_digest=training_identity_ledger.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=trunk.architecture_record(),
        preprocessing={
            "epoch": asdict(dataset.preprocessing),
            "channel_names": list(dataset.channel_names),
            "source_reference": dataset.provenance["source_reference"],
        },
        optimizer={"name": "none"},
        validation={"mode": "none"},
        objective={"name": "none"},
        seed=0,
        training_participant_keys=training_identity_ledger.authority_keys("source"),
        holdout_participant_keys=holdouts,
    )
    return {
        **_checkpoint_fields(dataset, training_identity_ledger, training_contract),
        "trunk_state_dict": trunk.state_dict(),
        "source_calibration": {
            "pos_weight": 5.0,
            "train_prior": 1.0 / 6.0,
            "temperature": 1.0,
            "source": "unit_test_fixture",
        },
    }


def dataset_trunk(dataset: EpochDataset) -> N2P3Net:
    return N2P3Net(
        n_channels=dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="ms_flatten",
    )


def checkpoint_identity_fields(
    table: DatasetIdentityTable,
    dataset: EpochDataset,
    *,
    source_snapshot_sha256: str = "2" * 64,
) -> dict[str, object]:
    architecture = dataset_trunk(dataset).architecture_record()
    preprocessing = {
        "epoch": asdict(dataset.preprocessing),
        "channel_names": list(dataset.channel_names),
        "source_reference": dataset.provenance["source_reference"],
    }
    contract = TrainingRunContract(
        source_cache_sha256="1" * 64,
        source_identity_digest=table.digest(),
        source_snapshot_sha256=source_snapshot_sha256,
        architecture=architecture,
        preprocessing=preprocessing,
        optimizer={"fixture": "adam"},
        validation={"fixture": "target_excluded"},
        objective={"fixture": "binary_or_reconstruction"},
        seed=0,
        training_participant_keys=table.authority_keys(),
        holdout_participant_keys=(),
    )
    return {
        **_checkpoint_fields(dataset, table, contract),
        "source_snapshot_sha256": source_snapshot_sha256,
    }


def initialized_splitbn_core(target_buffer_source: str = "pooled_source") -> N2P3Net:
    torch.manual_seed(31)
    core = N2P3Net(
        n_channels=3,
        n_times=64,
        sfreq=128.0,
        tmin_s=-0.2,
        splitbn_domain_count=2,
        splitbn_target_buffer_source=target_buffer_source,
    )
    core.train()
    source = torch.randn(16, 3, 64)
    domains = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    with torch.no_grad():
        core(source, domain_indices=domains)
    core.eval()
    return core


def small_trunk() -> N2P3Net:
    return N2P3Net(
        n_channels=3,
        n_times=128,
        sfreq=128.0,
        tmin_s=-0.2,
        pooling_mode="ms_flatten",
    )


def participant_identity_table(
    source_dataset_id: str,
    *subjects: str,
    global_key: str | None = None,
) -> DatasetIdentityTable:
    return DatasetIdentityTable(
        records=tuple(
            ParticipantIdentityRecord(
                local_subject_id=subject,
                origin_subject_keys=(origin_subject_key(source_dataset_id, subject),),
                global_person_keys=(global_key,) if global_key is not None else (),
                identity_status=(
                    "global_verified" if global_key is not None else "source_verified"
                ),
            )
            for subject in subjects
        )
    )
