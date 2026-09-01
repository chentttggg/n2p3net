from __future__ import annotations

import pytest

from research.contracts import TrainingRunContract
from research.evaluation import (
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    source_snapshot_sha256_from_archive_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _training_contract(participant_key: str) -> TrainingRunContract:
    return TrainingRunContract(
        source_cache_sha256=SHA_A,
        source_identity_digest=SHA_B,
        source_snapshot_sha256=SHA_C,
        architecture={"name": "n2p3", "kernel": 35},
        preprocessing={"sfreq": 128.0},
        optimizer={"name": "adamw"},
        validation={"kind": "group_disjoint"},
        objective={"trial_ce": 1.0},
        seed=1,
        training_participant_keys=(participant_key,),
        holdout_participant_keys=(),
    )


def test_model_origin_union_never_invents_a_checkpoint() -> None:
    training = _training_contract("n2p3-origin://source/participant/train")
    checkpoint = checkpoint_model_origin(
        {
            "training_contract": training.record(),
            "training_contract_digest": training.digest(),
        },
        checkpoint_sha256=SHA_A,
    )
    scratch = scratch_model_origin({"architecture": "linear", "seed": 9})
    assert checkpoint["kind"] == "checkpoint"
    assert scratch["kind"] == "scratch"
    assert "checkpoint_sha256" not in scratch


def test_builder_uses_one_authority_key_per_requested_participant() -> None:
    from tests.test_epochs import _dataset

    dataset = _dataset()
    dataset.record()
    contract = build_evaluation_run_contract(
        arm_name="scratch",
        model_origin=scratch_model_origin({"architecture": "linear", "seed": 1}),
        dataset=dataset,
        target_cache_sha256=SHA_A,
        source_snapshot_sha256=SHA_B,
        requested_subjects=("s1", "s2", "s3"),
        identity_policy="source",
        target_protocol={"calibration_decisions": 5, "test_repetitions": 2},
        adaptation={"head": "linear", "normalization": "target_prefix"},
        decision={"aggregation": "mean", "tie_policy": "abstain"},
        evidence_scope={"stage": "development"},
    )
    assert len(contract.requested_participant_keys) == 3
    assert contract.target_identity_digest == dataset.identity_table.digest()  # type: ignore[union-attr]


def test_physical_snapshot_manifest_verifies_archive(tmp_path) -> None:
    import hashlib
    import json
    import tarfile

    archive = tmp_path / "source.tar.gz"
    source = tmp_path / "source.txt"
    source.write_text("physical source archive", encoding="utf-8")
    with tarfile.open(archive, mode="w:gz") as stream:
        stream.add(source, arcname="source.txt")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "source.manifest.json"
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
    assert source_snapshot_sha256_from_archive_manifest(manifest) == digest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["member_count"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="member_count"):
        source_snapshot_sha256_from_archive_manifest(manifest)
