import copy

import numpy as np
import pytest
import torch

from baselines.deep import DeepConfig
from research.source_selection import fit_source_trajectory
from tests.support.checkpoints import write_checkpoint
from tests.support.datasets import binary_dataset, source_dataset
from transfer.checkpoint import PreparedTransferCheckpoint, load_n2p3_trunk_checkpoint


def _source(tmp_path):
    dataset = source_dataset()
    _, payloads = fit_source_trajectory(dataset, np.arange(4),
        config=DeepConfig(fixed_updates=1, batch_size=4, sampling="uniform_rows", compile_mode=None),
        source_cache_sha256="a" * 64, source_snapshot_sha256="b" * 64,
        excluded_source_datasets=["outer"], device=torch.device("cpu"))
    path = tmp_path / "source.pt"
    torch.save(payloads[1], path)
    return dataset, path, payloads[1]


def test_prepared_checkpoint_keeps_identity_gate_and_independent_model_state(tmp_path):
    dataset, path, payload = _source(tmp_path)
    expected, _ = load_n2p3_trunk_checkpoint(path, dataset, target_subject="source::s3")
    prepared = PreparedTransferCheckpoint(path, dataset)
    first = prepared.trunk_for_target("source::s3")
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(first.state_dict()[name], value, rtol=0, atol=0)
    with torch.no_grad():
        next(first.parameters()).add_(10)
    second = prepared.trunk_for_target("source::s3")
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
    with pytest.raises(ValueError, match="includes target"):
        prepared.trunk_for_target("source::s1")
    corrupted = copy.deepcopy(payload)
    corrupted["training_contract"]["objective"]["name"] = "changed"
    with pytest.raises(ValueError, match="digest"):
        PreparedTransferCheckpoint(corrupted, dataset)


def test_e2_group_loads_checkpoint_once_and_preserves_failed_target(tmp_path, monkeypatch):
    from experiments.run_source_training_chain import _score_checkpoint
    from transfer import checkpoint
    from transfer.recipes import TargetRecipe

    dataset, path, _ = _source(tmp_path)
    calls = []
    original = checkpoint.load_checkpoint_payload
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(checkpoint, "load_checkpoint_payload", counted)
    common = {"domain": "source", "actual_K": 0, "train_rows": [], "test_rows": [4, 5], "purged_rows": []}
    cohort = [{**common, "subject": subject,
               "participant_key": dataset.identity_table.record_for(subject).authority_key()}
              for subject in ("source::s3", "source::s1", "source::s3")]
    rows = _score_checkpoint(path, dataset, cohort, cache_sha256="a" * 64,
        snapshot_sha256="b" * 64, recipe=TargetRecipe("fixed", batch_size=512),
        output=tmp_path, device="cpu", evidence_scope="software_correctness")
    assert len(calls) == 1
    assert [row["status"] for row in rows] == ["completed", "fit_failure", "completed"]
    assert "includes target" in rows[1]["failure"]
    assert rows[0]["auc"] == rows[2]["auc"]


def test_prepared_source_keeps_exact_cpu_calibration_state_and_predictions(tmp_path):
    from transfer.calibration_candidates import fit_calibration_candidate
    from transfer.checkpoint import checkpoint_input_stats
    from transfer.recipes import TargetRecipe

    dataset = binary_dataset()
    path = write_checkpoint(tmp_path / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    baseline, payload = load_n2p3_trunk_checkpoint(path, dataset, target_subject="p1")
    prepared = PreparedTransferCheckpoint(path, dataset)
    recipe = TargetRecipe("fixed", batch_size=512, epochs=2, seed=19)
    mask = np.arange(dataset.n_epochs) < 20
    original = fit_calibration_candidate(dataset, recipe, mask, source_trunk=baseline,
        source_input_stats=checkpoint_input_stats(payload, dataset.n_channels), device="cpu")
    optimized = fit_calibration_candidate(dataset, recipe, mask, source_trunk=prepared.trunk_for_target("p1"),
        source_input_stats=prepared.input_statistics, device="cpu")
    for key, value in original.state.items():
        torch.testing.assert_close(value, optimized.state[key], rtol=0, atol=0)
    np.testing.assert_array_equal(original.predict_logit(dataset.X), optimized.predict_logit(dataset.X))
