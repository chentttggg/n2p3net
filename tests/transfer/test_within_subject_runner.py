from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from data.epochs import (
    save_epoch_dataset,
)
from experiments.run_within_subject_transfer import (
    _target_fixed_budget_calibration_payload,
)
from experiments.run_within_subject_transfer import main as run_transfer_main
from tests.support.candidate import causal_candidate_dataset
from tests.support.checkpoints import (
    checkpoint_identity_fields,
    dataset_trunk,
    participant_identity_table,
)
from tests.support.snapshots import write_source_snapshot
from transfer.checkpoint import (
    checkpoint_scores_to_llr,
)


def test_target_fixed_budget_adapter_uses_current_calibration_contract() -> None:
    payload = _target_fixed_budget_calibration_payload(
        pos_weight=8.0,
        train_prior=1.0 / 9.0,
    )

    llr, record = checkpoint_scores_to_llr(payload, np.asarray([0.0, 1.0]))

    assert np.isfinite(llr).all()
    assert record["source"] == "target_fixed_budget_weighted_ce_analytic"
    assert "training_pos_weight" not in payload
    assert "training_prior" not in payload


def _transfer_argv(cache, manifest, output, *extra):
    return [
        "run_within_subject_transfer.py",
        "--dataset-cache",
        cache,
        "--source-snapshot-manifest",
        manifest,
        "--output",
        output,
        "--identity-exclusion-policy",
        "source",
        "--device",
        "cpu",
        *extra,
    ]


def test_within_subject_runner_declares_prefix_for_chronological_refit(
    tmp_path, monkeypatch
) -> None:
    dataset = causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "refit-cache.npz", dataset)
    manifest, _, _ = write_source_snapshot(tmp_path)
    output = tmp_path / "refit-result.json"
    monkeypatch.setattr(
        "sys.argv",
        _transfer_argv(
            str(cache),
            str(manifest),
            str(output),
            "--arm-name",
            "refit-scope-test",
            "--head",
            "linear",
            "--prefix-reps",
            "4",
            "--test-reps",
            "1",
            "--allow-oracle-same-selection-adaptation",
            "--epoch-selection",
            "target_time_split",
            "--epochs",
            "1",
        ),
    )
    run_transfer_main()
    result = json.loads(output.read_text())
    assert result["records"]
    for record in result["records"]:
        scope = record["prefix_refit"]["fit_scope"]
        assert record["prefix_refit"]["enabled"]
        assert scope["prefix_mask_explicit"]
        assert scope["refit_rows"] == record["n_prefix_after_qc"]
        assert scope["outside_prefix_rows"] == 0


@pytest.fixture
def zero_shot_source(tmp_path):
    dataset = causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "causal.npz", dataset)
    trunk = dataset_trunk(dataset)
    checkpoint = tmp_path / "supervised.pt"
    source_manifest, source_digest, _ = write_source_snapshot(tmp_path)
    torch.save(
        {
            "trunk_state_dict": trunk.state_dict(),
            **checkpoint_identity_fields(
                participant_identity_table("synthetic", "other"),
                dataset,
                source_snapshot_sha256=source_digest,
            ),
            "holdout_subjects": ["g1", "g2", "g3"],
            "source_dataset_name": dataset.name,
            "config": {"pooling_mode": "ms_flatten", "training": "supervised"},
            "source_calibration": {
                "pos_weight": 3.0,
                "train_prior": 0.25,
                "temperature": 1.0,
                "source": "fixture_weighted_ce_analytic",
            },
        },
        checkpoint,
    )
    output = tmp_path / "zero-shot.json"
    argv = _transfer_argv(
        str(cache), str(source_manifest), str(output), "--checkpoint", str(checkpoint)
    )
    return argv, output, source_manifest, source_digest


@pytest.mark.parametrize("separate_evaluation_snapshot", [False, True])
def test_session_start_zero_shot_runner_never_requires_prefix_training(
    tmp_path, monkeypatch, zero_shot_source, separate_evaluation_snapshot
) -> None:
    argv, output, source_manifest, source_digest = zero_shot_source
    argv = [*argv, "--arm-name", "zero_shot_source", "--head", "zero_shot", "--test-reps", "2"]
    evaluation_manifest, evaluation_digest = source_manifest, source_digest
    if separate_evaluation_snapshot:
        evaluation_directory = tmp_path / "evaluation"
        evaluation_directory.mkdir()
        evaluation_manifest, evaluation_digest, _ = write_source_snapshot(
            evaluation_directory, source_value=2
        )
        argv.extend(["--evaluation-snapshot-manifest", str(evaluation_manifest)])
    monkeypatch.setattr(sys, "argv", argv)
    run_transfer_main()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["prefix_reps"] == 0
    assert record["estimand"] == "target_excluded_session_start_zero_calibration"
    assert record["n_groups"] == 3
    assert record["source_snapshot_manifest"] == str(source_manifest.resolve())
    assert record["source_snapshot_sha256"] == source_digest
    assert record["evaluation_snapshot_manifest"] == str(evaluation_manifest.resolve())
    assert record["evaluation_snapshot_sha256"] == evaluation_digest
    assert record["evaluation_contract"]["source_snapshot_sha256"] == evaluation_digest


def test_zero_shot_all_repetitions_preserves_complete_suffix(monkeypatch, zero_shot_source):
    argv, output, _, _ = zero_shot_source
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--arm-name", "zero_shot_source_all", "--head", "zero_shot", "--test-reps", "all"],
    )
    run_transfer_main()
    all_record = json.loads(output.read_text(encoding="utf-8"))
    assert all_record["test_reps"] == "all"
    assert all_record["test_repetition_mode"] == "all_post_boundary"
    assert all(item["n_suffix"] == 24 for item in all_record["records"])
    assert set(all_record["hit_mean_by_repetition"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    }
    assert "raw_hit_at_all" in all_record
    assert "hit_at_all_balanced" in all_record


def test_zero_shot_runner_rejects_unknown_target_subject(tmp_path, monkeypatch, zero_shot_source):
    argv, _, _, _ = zero_shot_source
    target_file = tmp_path / "targets.json"
    target_file.write_text('["g1", "absent-subject"]', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--arm-name",
            "invalid_target_manifest",
            "--target-subjects-file",
            str(target_file),
        ],
    )
    with pytest.raises(SystemExit):
        run_transfer_main()


def test_transfer_runner_rejects_checkpoint_snapshot_mismatch(tmp_path, monkeypatch) -> None:
    dataset = causal_candidate_dataset()
    cache = save_epoch_dataset(tmp_path / "causal.npz", dataset)
    source_manifest, _, _ = write_source_snapshot(tmp_path)
    checkpoint = tmp_path / "mismatched.pt"
    torch.save(
        {
            "trunk_state_dict": dataset_trunk(dataset).state_dict(),
            **checkpoint_identity_fields(participant_identity_table("synthetic", "other"), dataset),
        },
        checkpoint,
    )
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        "sys.argv",
        _transfer_argv(
            str(cache),
            str(source_manifest),
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--arm-name",
            "snapshot_mismatch",
        ),
    )

    with pytest.raises(ValueError, match="verified physical source freeze"):
        run_transfer_main()
    assert not output.exists()


def test_transfer_runner_rejects_legacy_naked_snapshot_digest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(tmp_path / "unused.npz"),
            "--arm-name",
            "legacy_digest",
            "--source-snapshot-sha256",
            "a" * 64,
            "--identity-exclusion-policy",
            "source",
            "--output",
            str(tmp_path / "unused.json"),
        ],
    )

    with pytest.raises(SystemExit):
        run_transfer_main()


def test_target_subject_file_rejects_ambiguous_offset_or_limit(tmp_path, monkeypatch) -> None:
    source_manifest, _, _ = write_source_snapshot(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_within_subject_transfer.py",
            "--dataset-cache",
            str(tmp_path / "unused.npz"),
            "--arm-name",
            "ambiguous_scope",
            "--source-snapshot-manifest",
            str(source_manifest),
            "--identity-exclusion-policy",
            "source",
            "--target-subjects-file",
            str(tmp_path / "targets.json"),
            "--max-subjects",
            "1",
            "--output",
            str(tmp_path / "unused.json"),
        ],
    )

    with pytest.raises(SystemExit):
        run_transfer_main()
