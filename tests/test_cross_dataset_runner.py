from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.run_cross_dataset_transfer import frozen_features, planned_splits
from models.n2p3net import N2P3Net
from transfer.checkpoint import predict_n2p3_checkpoint_with_features


def test_batched_frozen_features_preserve_scores_features_and_source_state():
    torch.manual_seed(7)
    model = N2P3Net(n_channels=3, n_times=179, sfreq=128)
    X = np.random.default_rng(7).normal(size=(7, 3, 179)).astype(np.float32)
    mask = np.ones((7, 3), dtype=bool)
    mask[:, 1] = False
    X[:, 1] = 0
    stats = (np.zeros((1, 3, 1), np.float32), np.ones((1, 3, 1), np.float32))
    device = torch.device("cpu")
    expected = predict_n2p3_checkpoint_with_features(model, X, input_stats=stats, device=device, trial_channel_mask=mask)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    actual = frozen_features(model, X, mask, stats, device, batch_size=2)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_whole_study_multirun_protocol_rejects_undeclared_order():
    from types import SimpleNamespace
    timeline = SimpleNamespace(
        evidence_indices=np.arange(600), onset_times_s=np.tile(np.arange(300), 2),
        evidence_available_times_s=np.tile(np.arange(300), 2)+0.8,
        session_ids=np.repeat(["0", "2"], 300), run_ids=np.repeat("run-10", 600),
    )
    dataset = SimpleNamespace(subject_ids=np.repeat("P1", 600), n_epochs=600, event_timeline=timeline)
    protocol = {"kind": "last_session_holdout", "session_order": ["0", "1", "2"],
                "run_order": ["run-2"], "test_session": "2"}
    with pytest.raises(ValueError, match="original protocol order"):
        planned_splits(dataset, "P1", protocol, (48, 96, 192))
