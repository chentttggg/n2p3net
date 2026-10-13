"""Strict warm starts initialize the complete declared model before fitting."""

import pytest
import torch

from baselines.n2p3net import N2P3NetBaseline


def make_baseline(state):
    baseline = N2P3NetBaseline(
        3, 64, 128.0, device=torch.device("cpu"), pretrained_state_dict=state,
    )
    baseline.model_ = baseline._make_model()
    return baseline


def test_strict_warm_start_rejects_missing_tensor_before_copy():
    reference = make_baseline(None)
    state = {key: value.clone() for key, value in reference.model_.state_dict().items()}
    missing = next(iter(state))
    del state[missing]
    baseline = make_baseline(state)
    before = {key: value.clone() for key, value in baseline.model_.state_dict().items()}
    with pytest.raises(ValueError, match="not_in_source"):
        baseline._apply_pretrained_state_dict()
    assert any(row.get("key") == missing and row["event"] == "not_in_source" for row in baseline.load_report)
    for key, value in baseline.model_.state_dict().items():
        assert torch.equal(value, before[key])


def test_strict_warm_start_rejects_empty_declared_checkpoint():
    with pytest.raises(ValueError, match="empty"):
        make_baseline({})._apply_pretrained_state_dict()


def test_complete_warm_start_copies_every_parameter_and_buffer():
    source = make_baseline(None)
    state = {key: value.clone() for key, value in source.model_.state_dict().items()}
    target = make_baseline(state)
    target._apply_pretrained_state_dict()
    assert {row["key"] for row in target.load_report if row["event"] == "loaded"} == set(state)
    for key, value in target.model_.state_dict().items():
        assert torch.equal(value, state[key])
