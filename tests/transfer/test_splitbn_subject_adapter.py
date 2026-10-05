from __future__ import annotations

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net
from models.splitbn import SplitBN1d, SplitBN2d
from tests.support.checkpoints import initialized_splitbn_core
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig


def _target_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(37)
    values = rng.standard_normal((32, 3, 64)).astype(np.float32)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 16)
    values[labels == 1] += 0.25
    return values, labels


def _splitbn_buffers(core: N2P3Net) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in core.state_dict().items()
        if "running_" in key or "batches_tracked" in key or "buffer_initialized" in key
    }


def test_subject_adapter_uses_predeclared_splitbn_target_buffer() -> None:
    core = initialized_splitbn_core()
    before = _splitbn_buffers(core)
    values, labels = _target_data()
    adapter = SubjectAdapter(
        core,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=1,
            batch_size=16,
            val_group_fraction=None,
            refit_full_prefix=False,
            input_statistics="target_prefix",
        ),
        device=torch.device("cpu"),
    )

    adapter.fit(values, labels)
    scores = adapter.predict_logit(values[:4])

    assert scores.shape == (4,)
    after = _splitbn_buffers(core)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_subject_adapter_rejects_implicit_target_splitbn_updates() -> None:
    core = initialized_splitbn_core()
    adapter = SubjectAdapter(
        core,
        config=SubjectAdapterConfig(
            head_kind="full_fine",
            epochs=1,
            freeze_batchnorm_stats=False,
            val_group_fraction=None,
            input_statistics="target_prefix",
        ),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="cannot update SplitBN"):
        adapter._set_training_mode(trunk_trainable=True)


def test_subject_adapter_fails_when_target_calibration_buffer_is_absent() -> None:
    core = initialized_splitbn_core("target_calibration")
    assert all(
        not bool(module.target_buffer_initialized)
        for module in core.modules()
        if isinstance(module, (SplitBN1d, SplitBN2d))
    )
    values, labels = _target_data()
    adapter = SubjectAdapter(
        core,
        config=SubjectAdapterConfig(
            head_kind="linear",
            epochs=1,
            val_group_fraction=None,
            input_statistics="target_prefix",
        ),
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="not explicitly installed"):
        adapter.fit(values, labels)
