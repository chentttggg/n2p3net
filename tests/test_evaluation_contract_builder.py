from __future__ import annotations

import pytest

from research.contracts import TrainingRunContract
from research.evaluation import (
    build_evaluation_run_contract,
    checkpoint_model_origin,
    scratch_model_origin,
    source_snapshot_sha256_from_archive_manifest,
    verified_source_snapshot,
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
    payload.update(working_tree_snapshot=True, source_commit_role="base_commit_only")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    verified = verified_source_snapshot(manifest)
    assert verified["working_tree_snapshot"] is True
    assert verified["source_commit_role"] == "base_commit_only"
    payload["source_commit_role"] = "exact_commit"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="base_commit_only"):
        verified_source_snapshot(manifest)
    payload["source_commit_role"] = "base_commit_only"
    payload["member_count"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="member_count"):
        source_snapshot_sha256_from_archive_manifest(manifest)


def test_executed_snapshot_rejects_missing_runtime_files_and_changed_bytes(tmp_path):
    import hashlib
    import json
    import tarfile

    from research.evaluation import verified_executed_snapshot

    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    files = {"src/model.py": "VALUE = 1\n", "pyproject.toml": "", "uv.lock": ""}
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in files:
            stream.add(root / name, arcname=name)
    payload = dict(schema="n2p3_source_freeze/1", archive=archive.name,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   source_commit="a" * 40, member_count=3, byte_size=archive.stat().st_size,
                   files={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert verified_executed_snapshot(manifest, root=root)["executed_files_verified"] == 3
    (root / "src/new.py").write_text("NEW = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing.*src/new.py"):
        verified_executed_snapshot(manifest, root=root)
    (root / "src/new.py").unlink()
    (root / "src/model.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs.*src/model.py"):
        verified_executed_snapshot(manifest, root=root)
    payload["files"]["src/model.py"] = hashlib.sha256((root / "src/model.py").read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="archive.*src/model.py"):
        verified_executed_snapshot(manifest, root=root)


def test_whole_library_checkpoint_gate_rejects_another_target_person_under_renamed_namespace():
    from data.identity import DatasetIdentityTable
    from transfer.checkpoint import assert_checkpoint_library_exclusion

    training = DatasetIdentityTable.from_source_rows(["renamed::q"], ["target_library"])
    target = DatasetIdentityTable.from_source_rows(["p"], ["target_library"])
    contract = _training_contract(training.authority_keys()[0])
    payload = {"training_identity_ledger": training.payload(),
               "training_identity_ledger_digest": training.digest(),
               "training_contract": contract.record(), "training_contract_digest": contract.digest()}
    with pytest.raises(ValueError, match="target_library"):
        assert_checkpoint_library_exclusion(payload, target)
