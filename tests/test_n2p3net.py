from __future__ import annotations

import numpy as np
import torch

from baselines.deep import DeepConfig
from baselines.n2p3net import N2P3NetBaseline
from models.n2p3net import N2P3Net
from train.runtime import GpuPerformanceScheduler


def test_n2p3net_returns_binary_logits_and_has_compact_capacity() -> None:
    model = N2P3Net(n_channels=3)
    logits = model(torch.randn(4, 3, 128))

    assert logits.shape == (4, 2)
    assert model.parameter_count() < 100_000


def test_n2p3net_baseline_fits_with_subject_disjoint_validation() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 3, 64)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 12)
    X[y == 1, 2, 28:42] += 1.5
    subjects = np.repeat(np.arange(6), 4)
    baseline = N2P3NetBaseline(
        3,
        64,
        128.0,
        config=DeepConfig(epochs=1, batch_size=8, val_subject_frac=0.25, val_subjects_min=1),
        device=torch.device("cpu"),
    ).fit(X, y, subject_ids=subjects)

    assert baseline.predict_logit(X).shape == (len(X),)
    assert baseline.calibration_source_ == "subject_disjoint_validation"


def test_n2p3net_baseline_accepts_the_shared_runtime() -> None:
    runtime = GpuPerformanceScheduler(torch.device("cpu"))
    baseline = N2P3NetBaseline(3, 64, 128.0, device=torch.device("cpu"), runtime=runtime)

    assert baseline.runtime is runtime
